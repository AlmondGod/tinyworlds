import torch
import torch.nn as nn
from einops import rearrange
from models.positional_encoding import build_spatial_only_pe, sincos_time
from models.norms import AdaptiveNormalizer, AdaLNZeroNorm
from models.patch_embed import PatchEmbedding
import math
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, conditioning_dim=None, use_adaln_zero=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, f"embed dim must be divisible by num heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.use_adaln_zero = use_adaln_zero
        if use_adaln_zero:
            self.norm = AdaLNZeroNorm(embed_dim, conditioning_dim)
        else:
            self.norm = AdaptiveNormalizer(embed_dim, conditioning_dim)

    def forward(self, x, conditioning=None):
        B, T, P, E = x.shape

        if self.use_adaln_zero:
            x_in, gate = self.norm(x, conditioning)  # pre-norm + conditioning params
        else:
            x_in = x

        # project to Q, K, V and split into heads: [B, T, P, E] -> [(B*T), H, P, E/H]
        # (4 dims to work with torch compile attention)
        q = rearrange(self.q_proj(x_in), 'B T P (H D) -> (B T) H P D', H=self.num_heads)
        k = rearrange(self.k_proj(x_in), 'B T P (H D) -> (B T) H P D', H=self.num_heads)
        v = rearrange(self.v_proj(x_in), 'B T P (H D) -> (B T) H P D', H=self.num_heads)

        k_t = k.transpose(-2, -1) # [(B*T), H, P, D, P]

        # attention(q, k, v) = softmax(qk^T / sqrt(d)) v
        scores = torch.matmul(q, k_t) / math.sqrt(self.head_dim) # [(B*T), H, P, P]
        attn_weights = F.softmax(scores, dim=-1) # [(B*T), H, P, P]
        attn_output = torch.matmul(attn_weights, v) # [(B*T), H, P, D]
        attn_output = rearrange(attn_output, '(B T) H P D -> B T P (H D)', B=B, T=T) # [B, T, P, E]

        # out proj to mix head information
        attn_out = self.out_proj(attn_output)  # [B, T, P, E]

        if self.use_adaln_zero:
            # gated residual; gate is None when unconditioned (identity)
            return x + (gate * attn_out if gate is not None else attn_out)  # [B, T, P, E]
        else:
            # residual and optionally conditioned post-norm (FiLM)
            return self.norm(x + attn_out, conditioning)  # [B, T, P, E]

class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=True, conditioning_dim=None, use_adaln_zero=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.use_adaln_zero = use_adaln_zero
        if use_adaln_zero:
            self.norm = AdaLNZeroNorm(embed_dim, conditioning_dim)
        else:
            self.norm = AdaptiveNormalizer(embed_dim, conditioning_dim)
        self.causal = causal

    def forward(self, x, conditioning=None):
        B, T, P, E = x.shape

        if self.use_adaln_zero:
            x_in, gate = self.norm(x, conditioning)  # pre-norm + conditioning params
        else:
            x_in = x

        # project to Q, K, V and split into heads: [B, T, P, E] -> [(B*P), H, T, D]
        # (4 dims to work with torch compile attention)
        q = rearrange(self.q_proj(x_in), 'b t p (h d) -> (b p) h t d', h=self.num_heads)
        k = rearrange(self.k_proj(x_in), 'b t p (h d) -> (b p) h t d', h=self.num_heads)
        v = rearrange(self.v_proj(x_in), 'b t p (h d) -> (b p) h t d', h=self.num_heads)

        k_t = k.transpose(-2, -1) # [(B*P), H, T, D, T]

        # attention(q, k, v) = softmax(qk^T / sqrt(d)) v
        scores = torch.matmul(q, k_t) / math.sqrt(self.head_dim) # [(B*P), H, T, T]

        # causal mask for each token t in seq, mask out all tokens to the right of t (after t)
        if self.causal:
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
            scores = scores.masked_fill(mask, -torch.inf) # [(B*P), H, T, T]

        attn_weights = F.softmax(scores, dim=-1) # [(B*P), H, T, T]
        attn_output = torch.matmul(attn_weights, v) # [(B*P), H, T, D]
        attn_output = rearrange(attn_output, '(b p) h t d -> b t p (h d)', b=B, p=P) # [B, T, P, E]

        # out proj to mix head information
        attn_out = self.out_proj(attn_output)  # [B, T, P, E]

        if self.use_adaln_zero:
            return x + (gate * attn_out if gate is not None else attn_out)  # [B, T, P, E]
        else:
            # residual and optionally conditioned post-norm (FiLM)
            return self.norm(x + attn_out, conditioning)  # [B, T, P, E]

class SwiGLUFFN(nn.Module):
    # swiglu(x) = W3(sigmoid(W1(x) + b1) * (W2(x) + b2)) + b3
    def __init__(self, embed_dim, hidden_dim, conditioning_dim=None, use_adaln_zero=False):
        super().__init__()
        h = math.floor(2 * hidden_dim / 3)
        self.w_v = nn.Linear(embed_dim, h)
        self.w_g = nn.Linear(embed_dim, h)
        self.w_o = nn.Linear(h, embed_dim)
        self.use_adaln_zero = use_adaln_zero
        if use_adaln_zero:
            self.norm = AdaLNZeroNorm(embed_dim, conditioning_dim)
        else:
            self.norm = AdaptiveNormalizer(embed_dim, conditioning_dim)

    def forward(self, x, conditioning=None):
        if self.use_adaln_zero:
            x_in, gate = self.norm(x, conditioning)
        else:
            x_in = x
        v = F.silu(self.w_v(x_in))  # [B, T, P, h]
        g = self.w_g(x_in)  # [B, T, P, h]
        out = self.w_o(v * g)  # [B, T, P, E]
        if self.use_adaln_zero:
            return x + (gate * out if gate is not None else out)  # [B, T, P, E]
        return self.norm(x + out, conditioning)  # [B, T, P, E]


class SwiGLUExpert(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        h = math.floor(2 * hidden_dim / 3)
        self.w_v = nn.Linear(embed_dim, h)
        self.w_g = nn.Linear(embed_dim, h)
        self.w_o = nn.Linear(h, embed_dim)

    def forward(self, x):
        return self.w_o(F.silu(self.w_v(x)) * self.w_g(x))


class MoESwiGLUFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_experts=4, top_k=2,
                 aux_loss_coeff=0.01, conditioning_dim=None, use_adaln_zero=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff

        self.router = nn.Linear(embed_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(embed_dim, hidden_dim) for _ in range(num_experts)
        ])
        self.use_adaln_zero = use_adaln_zero
        if use_adaln_zero:
            self.norm = AdaLNZeroNorm(embed_dim, conditioning_dim)
        else:
            self.norm = AdaptiveNormalizer(embed_dim, conditioning_dim)

        self._aux_loss = None
        self._expert_counts = None  # per-expert token fractions from last forward

    @property
    def aux_loss(self):
        if self._aux_loss is None:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        return self._aux_loss

    @property
    def expert_utilization(self):
        return self._expert_counts

    def forward(self, x, conditioning=None):
        # x: [B, T, P, E]
        B, T, P, E = x.shape
        residual = x

        if self.use_adaln_zero:
            x, gate = self.norm(x, conditioning)  # pre-norm; gate applied after expert combine

        # flatten spatial dims for routing: [B*T*P, E]
        flat = x.reshape(-1, E)
        N = flat.shape[0]

        # route tokens to top-k experts
        logits = self.router(flat)  # [N, num_experts]
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)  # [N, top_k]
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [N, top_k]

        # load-balancing auxiliary loss
        if self.training:
            router_probs = F.softmax(logits, dim=-1)  # [N, num_experts]
            tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
            for k in range(self.top_k):
                tokens_per_expert.scatter_add_(
                    0, top_k_indices[:, k],
                    torch.ones(N, device=x.device),
                )
            fraction_dispatched = tokens_per_expert / (N * self.top_k)
            fraction_probs = router_probs.mean(dim=0)
            self._aux_loss = self.aux_loss_coeff * self.num_experts * (
                fraction_dispatched * fraction_probs
            ).sum()
            self._expert_counts = fraction_dispatched.detach()

        # compute expert outputs
        output = torch.zeros_like(flat)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # [N]
            weight = top_k_weights[:, k].unsqueeze(-1)  # [N, 1]
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if not mask.any():
                    continue
                expert_input = flat[mask]
                expert_output = self.experts[e](expert_input)
                output[mask] += weight[mask] * expert_output

        # reshape back and apply residual + norm
        out = output.reshape(B, T, P, E)
        if self.use_adaln_zero:
            return residual + (gate * out if gate is not None else out)  # [B, T, P, E]
        return self.norm(residual + out, conditioning)  # [B, T, P, E]

class STTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, causal=True, conditioning_dim=None,
                 use_moe=False, num_experts=4, top_k_experts=2, moe_aux_loss_coeff=0.01,
                 use_adaln_zero=False):
        super().__init__()
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, conditioning_dim, use_adaln_zero)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, causal, conditioning_dim, use_adaln_zero)
        if use_moe:
            self.ffn = MoESwiGLUFFN(
                embed_dim, hidden_dim,
                num_experts=num_experts, top_k=top_k_experts,
                aux_loss_coeff=moe_aux_loss_coeff,
                conditioning_dim=conditioning_dim,
                use_adaln_zero=use_adaln_zero,
            )
        else:
            self.ffn = SwiGLUFFN(embed_dim, hidden_dim, conditioning_dim, use_adaln_zero)

    def forward(self, x, conditioning=None):
        # x: [B, T, P, E]
        # out: [B, T, P, E]
        x = self.spatial_attn(x, conditioning)
        x = self.temporal_attn(x, conditioning)
        x = self.ffn(x, conditioning)
        return x

class STTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_blocks, causal=True, conditioning_dim=None,
                 use_moe=False, num_experts=4, top_k_experts=2, moe_aux_loss_coeff=0.01,
                 use_adaln_zero=False):
        super().__init__()
        # calculate temporal PE dim
        self.temporal_dim = (embed_dim // 3) & ~1  # round down to even number
        self.spatial_dims = embed_dim - self.temporal_dim  # rest goes to spatial

        self.blocks = nn.ModuleList([
            STTransformerBlock(
                embed_dim, num_heads, hidden_dim, causal, conditioning_dim,
                use_moe=use_moe, num_experts=num_experts,
                top_k_experts=top_k_experts, moe_aux_loss_coeff=moe_aux_loss_coeff,
                use_adaln_zero=use_adaln_zero,
            )
            for _ in range(num_blocks)
        ])
        
    def forward(self, x, conditioning=None):
        # x: [B, T, P, E]
        # conditioning: [B, T, E]
        B, T, P, E = x.shape
        tpe = sincos_time(T, self.temporal_dim, x.device, x.dtype)  # [T, E/3]

        # temporal PE (pad with 0s for first 2/3s spatial PE, last 1/3 temporal PE)
        tpe_padded = torch.cat([
            torch.zeros(T, self.spatial_dims, device=x.device, dtype=x.dtype),
            tpe
        ], dim=-1)  # [T, E]
        x = x + tpe_padded[None, :, None, :]  # [B,T,P,E]

        # apply transformer blocks
        for block in self.blocks:
            x = block(x, conditioning)
        return x

    def moe_aux_loss(self):
        device = next(self.parameters()).device
        total = torch.zeros((), device=device)
        for block in self.blocks:
            if isinstance(block.ffn, MoESwiGLUFFN):
                total = total + block.ffn.aux_loss
        return total

    def moe_expert_utilization(self):
        """Per-block expert token fractions. Returns dict of block_idx -> [num_experts] tensor."""
        util = {}
        for idx, block in enumerate(self.blocks):
            if isinstance(block.ffn, MoESwiGLUFFN) and block.ffn.expert_utilization is not None:
                util[idx] = block.ffn.expert_utilization
        return util
