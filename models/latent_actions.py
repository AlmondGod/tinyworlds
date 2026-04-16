from models.utils import ModelType
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from einops import rearrange, repeat, reduce
from models.st_transformer import STTransformer, PatchEmbedding
from models.fsq import FiniteScalarQuantizer

NUM_LATENT_ACTIONS_BINS = 2


class WindowedFrameAttention(nn.Module):
    """Self-attention over a length-2 window of consecutive frame patches.

    For each pair of frames (t, t+1), concatenates their P patch embeddings
    into a 2P sequence, applies self-attention, and mean-pools back to E.
    This lets patches in both frames interact before pooling, giving richer
    inter-frame signal than mean pool + concat.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, frame_a, frame_b):
        # frame_a, frame_b: [B, P, E]
        B, P, E = frame_a.shape
        x = torch.cat([frame_a, frame_b], dim=1)  # [B, 2P, E]
        x = self.norm(x)
        H, D = self.num_heads, self.head_dim
        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=H)  # [B, H, 2P, D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=H)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=H)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, 2P, 2P]
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, 2P, D]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, 2P, E]
        out = self.out_proj(out)  # [B, 2P, E]
        return out.mean(dim=1)  # [B, E] -- mean pool over all 2P attended patches


class LatentActionsEncoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, action_dim=3, use_windowed_attention=False):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        self.use_windowed_attention = use_windowed_attention

        if use_windowed_attention:
            self.window_attn = WindowedFrameAttention(embed_dim, num_heads)
            self.action_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 4 * action_dim),
                nn.GELU(),
                nn.Linear(4 * action_dim, action_dim),
            )
        else:
            self.action_head = nn.Sequential(
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(embed_dim * 2, 4 * action_dim),
                nn.GELU(),
                nn.Linear(4 * action_dim, action_dim),
            )

    def forward(self, frames):
        # frames: [B, T, C, H, W]
        batch_size, seq_len, C, H, W = frames.shape

        embeddings = self.patch_embed(frames)  # [B, T, P, E]
        transformed = self.transformer(embeddings)  # [B, T, P, E]

        actions = []
        for t in range(seq_len - 1):
            if self.use_windowed_attention:
                # length-2 windowed attention across all patches from both frames, then mean pool
                pooled = self.window_attn(transformed[:, t], transformed[:, t + 1])  # [B, E]
            else:
                # mean pool over patches then concatenate consecutive frame features
                pooled_t = transformed[:, t].mean(dim=1)      # [B, E]
                pooled_t1 = transformed[:, t + 1].mean(dim=1)  # [B, E]
                pooled = torch.cat([pooled_t, pooled_t1], dim=1)  # [B, E*2]
            actions.append(self.action_head(pooled))  # [B, A]

        return torch.stack(actions, dim=1)  # [B, T-1, A]

class LatentActionsDecoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, conditioning_dim=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True, conditioning_dim=conditioning_dim)

        # embeddings to mixed frame output patches
        self.frame_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3 * patch_size * patch_size),
            nn.Tanh()
        )

        self.frame_size = frame_size
        self.patch_size = patch_size
        self.num_patches = (frame_size[0] // patch_size) * (frame_size[1] // patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

    def forward(self, frames, actions, training=True):
        # frames: [B, T, C, H, W]
        # actions: [B, T - 1, A]
        B, T, C, H, W = frames.shape
        frames = frames[:, :-1] # [B, T-1, C, H, W]
        video_embeddings = self.patch_embed(frames)  # [B, T-1, P, E]
        _, _, P, E = video_embeddings.shape

        # mask certain tokens from all frames except first frame
        # this strongly forces actions to contain most useful info (I recommend to keep based on experiments)
        if training and self.training:
            keep_rate = 0.0
            keep = (torch.rand(B, T-1, P, 1, device=frames.device) < keep_rate)
            keep[:, 0] = 1  # never mask first frame tokens (anchor) TODO: try rid of ablation
            video_embeddings = torch.where(
                keep, video_embeddings,
                self.mask_token.to(video_embeddings.dtype).expand_as(video_embeddings)
            )

        transformed = self.transformer(video_embeddings, conditioning=actions)  # [B, T-1, P, E]
        patches = self.frame_head(transformed)  # [B, T-1, P, 3 * S * S]
        patches = rearrange(
            patches, 'b t p (c p1 p2) -> b t c p p1 p2', c=3, p1=self.patch_size, p2=self.patch_size
        ) # [B, T-1, C, P, S, S]
        pred_frames = rearrange(
            patches, 'b t c (h w) p1 p2 -> b t c (h p1) (w p2)', h=H//self.patch_size, w=W//self.patch_size
        ) # [B, T-1, C, H, W]
        return pred_frames  # [B, T-1, C, H, W]

class LatentActionModel(nn.Module):
    def __init__(self, frame_size=(128, 128), n_actions=8, patch_size=8, embed_dim=128,
                 num_heads=8, hidden_dim=256, num_blocks=4, use_windowed_attention=False):
        super().__init__()
        assert math.log(n_actions, NUM_LATENT_ACTIONS_BINS).is_integer(), f"n_actions must be a power of {NUM_LATENT_ACTIONS_BINS}"
        self.action_dim = int(math.log(n_actions, NUM_LATENT_ACTIONS_BINS))
        self.encoder = LatentActionsEncoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, action_dim=self.action_dim, use_windowed_attention=use_windowed_attention)
        self.quantizer = FiniteScalarQuantizer(latent_dim=self.action_dim, num_bins=NUM_LATENT_ACTIONS_BINS)
        self.decoder = LatentActionsDecoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, conditioning_dim=self.action_dim)
        self.var_target = 0.01
        self.var_lambda = 100.0

    def forward(self, frames):
        # frames: [B, T, C, H, W]

        # get quantized action latents
        action_latents = self.encoder(frames) # [B, T - 1, A]
        action_latents_quantized = self.quantizer(action_latents) # [B, T - 1, A]

        # decode to get predicted frames
        pred_frames = self.decoder(frames, action_latents_quantized, training=True)  # [B, T - 1, C, H, W]

        # reconstruction loss
        target_frames = frames[:, 1:]  # All frames except first [B, T - 1, C, H, W]
        recon_loss = F.smooth_l1_loss(pred_frames, target_frames)

        # variance loss across batch dim for pre-quant encoder outputs (helps prevent action collapse)
        z_var = action_latents.var(dim=0, unbiased=False).mean()
        var_penalty = F.relu(self.var_target - z_var)
        total_loss = recon_loss + self.var_lambda * var_penalty

        return total_loss, pred_frames

    def encode(self, frames):
        action_latents = self.encoder(frames)  # [B, T, A]
        action_latents_quantized = self.quantizer(action_latents) # [B, T, A]
        return action_latents_quantized
    
    @property
    def model_type(self) -> str:
        return ModelType.LatentActionModel