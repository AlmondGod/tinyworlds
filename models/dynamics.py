from models.utils import ModelType
import torch
import torch.nn as nn
import math
from models.positional_encoding import build_spatial_only_pe
from models.st_transformer import STTransformer
from einops import repeat

class DynamicsModel(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=4, embed_dim=128, num_heads=8,
                 hidden_dim=128, num_blocks=4, num_bins=4, n_actions=8, conditioning_dim=3, latent_dim=5,
                 use_moe=False, num_experts=4, top_k_experts=2, moe_aux_loss_coeff=0.01):
        super().__init__()
        H, W = frame_size
        codebook_size = num_bins**latent_dim

        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        self.transformer = STTransformer(
            embed_dim, num_heads, hidden_dim, num_blocks, causal=True,
            conditioning_dim=conditioning_dim,
            use_moe=use_moe, num_experts=num_experts,
            top_k_experts=top_k_experts, moe_aux_loss_coeff=moe_aux_loss_coeff,
        )
        self.output_mlp = nn.Linear(embed_dim, codebook_size)

        # shared spatial-only PE (zeros in temporal tail)
        pe_spatial = build_spatial_only_pe((H, W), patch_size, embed_dim, device='cpu', dtype=torch.float32)  # [1,P,E]
        self.register_buffer("pos_spatial_dec", pe_spatial, persistent=False)

        # learnable mask token latent
        # TODO; try leanable mask embedding in embed space instead of latent space
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, latent_dim) * 0.02)  # [1, 1, 1, L]

    def forward(self, discrete_latents, training=True, conditioning=None, targets=None):
        # discrete_latents: [B, T, P, L]
        # targets: [B, T, P] indices
        # conditioning: [B, T, A]
        B, T, P, L = discrete_latents.shape

        # convert latents to float for embedding
        discrete_latents = discrete_latents.to(dtype=torch.float32)

        # apply MaskGIT random masking during training
        if training and self.training:
            # per-batch mask ratio in [0.5, 1.0)
            mask_ratio = 0.5 + torch.rand((), device=discrete_latents.device) * 0.5 
            mask_positions = (torch.rand(B, T, P, device=discrete_latents.device) < mask_ratio) # [B, T, P]

            # guarantee at least one unmasked temporal anchor per (B, P)
            # pick a random timestep for each (B,P) and force it to unmask
            anchor_idx = torch.randint(0, T, (B, P), device=discrete_latents.device)  # [B, P]
            mask_positions[torch.arange(B)[:, None], anchor_idx, torch.arange(P)[None, :]] = False # [B, T, P]

            # replace selected latents with mask tokens
            mask_token = repeat(self.mask_token.to(discrete_latents.device, discrete_latents.dtype), '1 1 1 L -> B T P L', B=B, T=T, P=P) # [B, T, P, L]
            discrete_latents = torch.where(mask_positions.unsqueeze(-1), mask_token, discrete_latents) # [B, T, P, L]
        else:
            mask_positions = None

        embeddings = self.latent_embed(discrete_latents)  # [B, T, P, E]

        # add spatial PE (affects only first 2/3 of dimensions)
        # STTransformer adds temporal PE to last 1/3 of dimensions
        embeddings = embeddings + self.pos_spatial_dec.to(embeddings.device, embeddings.dtype)
        transformed = self.transformer(embeddings, conditioning=conditioning)  # [B, T, P, E]

        # transform to logits for each token in codebook
        predicted_logits = self.output_mlp(transformed)  # [B, T, P, L^D]

        # compute masked cross-entropy loss
        loss = None
        if training and self.training:
            assert targets is not None, "target indices are needed for training"
            Ld = predicted_logits.shape[-1] # L^D
            logits_flat = predicted_logits.reshape(-1, Ld) # [(B*T*P), L^D]
            targets_flat = targets.reshape(-1) # [(B*T*P)]
            mask_flat = mask_positions.reshape(-1).to(torch.float32) # [(B*T*P)]
            loss_per = nn.functional.cross_entropy(logits_flat, targets_flat, reduction='none')  # [(B*T*P)]
            denom = mask_flat.sum().clamp_min(1.0)
            loss = (loss_per * mask_flat).sum() / denom

        return predicted_logits, mask_positions, loss  # logits, mask, optional loss

    def exp_schedule_torch(self, t, T, P_total, k, device):
        # t: current step, T: total steps, P_total: total masked positions across the horizon window
        # exp schedule is P_total * (1 - exp(k * t / T)) / (1 - exp(k))
        x = t / max(T, 1)
        k_tensor = torch.tensor(k, device=device)
        result = P_total * torch.expm1(k_tensor * x) / torch.expm1(k_tensor)
        if t == T - 1:
            return torch.tensor(P_total, dtype=result.dtype, device=device)
        return result

    @torch.no_grad()
    def forward_inference(self, context_latents, prediction_horizon, num_steps, index_to_latents_fn, conditioning=None, schedule_k=5.0, temperature: float = 0.0):
        # MaskGIT-style iterative decoding across all prediction horizon steps
        # context_latents: [B, T_ctx, P, L]
        # T_ctx=context timesteps, H=prediction horizon, K=codebook size
        device = context_latents.device
        dtype = context_latents.dtype
        B, T_ctx, P, L = context_latents.shape  # B, T_ctx, P, L
        H = int(prediction_horizon)  # number of horizon steps to decode

        # append prediction_horizon masked frame latents to predict dynamics on
        mask_latents = self.mask_token.to(device, dtype).expand(B, H, P, -1)  # [B, H, P, L]
        input_latents = torch.cat([context_latents, mask_latents], dim=1)  # [B, T_ctx+H, P, L]
        mask = torch.ones(B, H, P, 1, dtype=torch.bool, device=device)  # [B, H, P, 1]

        P_total = H * P  # total masked positions across the horizon window
        for m in range(num_steps):
            n_tokens_raw = self.exp_schedule_torch(m, num_steps, P_total, schedule_k, device)

            # predict logits for current input
            logits, _, _ = self.forward(input_latents, training=False, conditioning=conditioning, targets=None)  # [B, T_ctx+H, P, L^D]
            # temperature scaling
            if temperature and temperature > 0:
                scaled_logits = logits / float(temperature)
            else:
                scaled_logits = logits
            probs = torch.softmax(scaled_logits, dim=-1)  # [B, T_ctx+H, P, L^D]
            # confidence for unmask selection always from max probability
            max_probs, _ = torch.max(probs, dim=-1)  # [B, T_ctx+H, P]
            # choose indices either via argmax (temperature==0) or sampling
            if temperature and temperature > 0:
                Bc, Tc, Pc, Ld = probs.shape  # Bc=B, Tc=T_ctx+H, Pc=P, Ld=L^D
                sampled = torch.distributions.Categorical(probs=probs.reshape(-1, Ld)).sample()
                predicted_indices = sampled.view(Bc, Tc, Pc)  # [B, T_ctx+H, P]
            else:
                _, predicted_indices = torch.max(probs, dim=-1)  # [B, T_ctx+H, P]

            horizon_probs = max_probs[:, -H:, :]  # [B, H, P]

            # for each batch element, select tokens to unmask from all masked positions
            for b in range(B):
                masked_mask_all = mask[b, :, :, 0]  # [H, P]
                masked_flat = masked_mask_all.view(-1)  # [H*P]
                masked_flat_idx = torch.where(masked_flat)[0]  # [num_masked]
                if masked_flat_idx.numel() == 0:
                    continue

                # TODO: try to clean this
                num_masked_b = int(masked_flat_idx.numel())
                prev_b = P_total - num_masked_b
                target_unmasked = int(torch.ceil(n_tokens_raw).item())
                k_floor = max(P_total // 16, 1)
                k_b = max(k_floor, min(max(target_unmasked - prev_b, 0), num_masked_b))

                pos_probs_flat = horizon_probs[b].contiguous().view(-1)[masked_flat_idx]  # [num_masked]
                if pos_probs_flat.numel() > k_b:
                    top_idx = torch.topk(pos_probs_flat, k_b, largest=True).indices
                    sel_flat = masked_flat_idx[top_idx]
                else:
                    sel_flat = masked_flat_idx

                # map back to (h, p)
                h_sel = torch.div(sel_flat, P, rounding_mode='floor')  # [k_b]
                p_sel = sel_flat % P  # [k_b]

                # group by unique h and write sampled tokens to input tensor
                if h_sel.numel() > 0:
                    unique_h = torch.unique(h_sel, sorted=True)
                    for uh in unique_h:
                        mask_h = (h_sel == uh)
                        p_list = p_sel[mask_h]
                        if p_list.numel() == 0:
                            continue
                        t_abs = T_ctx + int(uh.item())  # absolute time index in [T_ctx, T_ctx+H-1] (for current horizon step)
                        idx_sel = predicted_indices[b:b+1, t_abs:t_abs+1, p_list]  # [1,1,P_sel]
                        pred_latents_sel = index_to_latents_fn(idx_sel)  # [1,1,P_sel,L]
                        input_latents[b:b+1, t_abs:t_abs+1, p_list] = pred_latents_sel
                        mask[b, int(uh.item()), p_list, 0] = False

            # early exit if all horizon tokens are unmasked
            if not mask[:, :, :, 0].any():  # mask: [B,H,P,1]
                break

        # final completion: fill any remaining masked tokens across all horizon steps via argmax
        # TODO: try removing
        if mask[:, :, :, 0].any():
            logits, _, _ = self.forward(input_latents, training=False, conditioning=conditioning, targets=None)  # [B, T_ctx+H, P, L^D]
            if temperature and temperature > 0:
                scaled_logits = logits / float(temperature)
            else:
                scaled_logits = logits
            probs = torch.softmax(scaled_logits, dim=-1)  # [B, T_ctx+H, P, L^D]
            _, predicted_indices = torch.max(probs, dim=-1)  # [B, T_ctx+H, P]
            for b in range(B):
                h_idx, p_idx = torch.where(mask[b, :, :, 0])  # both [N_remaining]
                if h_idx.numel() == 0:
                    continue
                unique_h = torch.unique(h_idx, sorted=True)
                for uh in unique_h:
                    mask_h = (h_idx == uh)
                    p_list = p_idx[mask_h]
                    if p_list.numel() == 0:
                        continue
                    t_abs = T_ctx + int(uh.item())  # absolute time index
                    idx_sel = predicted_indices[b:b+1, t_abs:t_abs+1, p_list]  # [1,1,P_sel]
                    pred_latents_sel = index_to_latents_fn(idx_sel)  # [1,1,P_sel,L]
                    input_latents[b:b+1, t_abs:t_abs+1, p_list] = pred_latents_sel
                    mask[b, int(uh.item()), p_list, 0] = False

        return input_latents # [B, T_ctx + H, P, L]

    @property
    def model_type(self) -> str:
        return ModelType.DynamicsModel