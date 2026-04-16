from models.utils import ModelType
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from einops import rearrange, repeat, reduce
from models.st_transformer import STTransformer, PatchEmbedding, SpatialAttention
from models.fsq import FiniteScalarQuantizer

NUM_LATENT_ACTIONS_BINS = 2

class LatentActionsEncoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8, 
                 hidden_dim=256, num_blocks=4, action_dim=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        
        # windowed attention for action tokenization
        self.window_attn = SpatialAttention(embed_dim, num_heads)

        # embeddings to discrete latent bottleneck actions
        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * action_dim),
            nn.GELU(),
            nn.Linear(4 * action_dim, action_dim)
        )

    def forward(self, frames):
        # frames: [B, T, C, H, W]
        batch_size, seq_len, C, H, W = frames.shape

        embeddings = self.patch_embed(frames)  # [B, T, P, E]
        transformed = self.transformer(embeddings)

        # windowed attention over length-2 windows (current and next frame)
        # we want to combine frame t and t+1
        
        # 1. Create windows: [B, T-1, 2, P, E]
        # We slice to get T-1 windows
        current_frames = transformed[:, :-1] # [B, T-1, P, E]
        next_frames = transformed[:, 1:]     # [B, T-1, P, E]
        
        # 2. Concatenate along patch dimension to treat as one large spatial sequence
        # [B, T-1, 2*P, E]
        windows = torch.cat([current_frames, next_frames], dim=2)
        
        # 3. Apply spatial attention
        # SpatialAttention expects [B, T, P, E]
        attended = self.window_attn(windows) # [B, T-1, 2*P, E]
        
        # 4. Mean pool over the combined patches
        pooled = attended.mean(dim=2) # [B, T-1, E]

        # 5. Project to actions
        actions = self.action_head(pooled) # [B, T-1, A]

        return actions

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
                 num_heads=8, hidden_dim=256, num_blocks=4):
        super().__init__()
        assert math.log(n_actions, NUM_LATENT_ACTIONS_BINS).is_integer(), f"n_actions must be a power of {NUM_LATENT_ACTIONS_BINS}"
        self.action_dim=int(math.log(n_actions, NUM_LATENT_ACTIONS_BINS))
        self.encoder = LatentActionsEncoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, action_dim=self.action_dim)
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