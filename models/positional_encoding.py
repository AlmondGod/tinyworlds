import torch
from einops import rearrange, repeat


def sincos_1d(L, D, device, dtype):
    # 1d sinusoidal position encoding where element j of ith patch embedding is encoded as:
    # PE[i, 2j]   = sin(i / 10000^(2j/D))  # even indices
    # PE[i, 2j+1] = cos(i / 10000^(2j/D))  # odd indices

    assert D % 2 == 0, "Encoding dimension must be even"

    # position indices [L, 1] and dimension indices [1, D/2]
    pos = rearrange(torch.arange(L, device=device, dtype=dtype), 'l -> l 1')     # [L,1]
    i   = rearrange(torch.arange(D // 2, device=device, dtype=dtype), 'd -> 1 d')# [1,D/2]

    # angular frequencies: 1/10000^(2i/D) for each dimension
    div = torch.pow(torch.tensor(10000.0, device=device, dtype=dtype), (2*i)/D)

    # angles: pos * freq for each position-dimension pair
    angles = pos / div  # [L, D/2] (broadcasted together)
    pe = torch.zeros(L, D, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles)  # even indices
    pe[:, 1::2] = torch.cos(angles)  # odd indices
    return pe # [L, D]

def sincos_time(T, D, device, dtype):
    # temporal PE (1d sinusoidal PE across time)
    return sincos_1d(T, D, device, dtype)  # reuse the same 1D builder


def build_spatial_only_pe(frame_size, patch_size, embed_dim, device='cpu', dtype=torch.float32):
    # spatial positional encodings for a grid of patches in first 2/3 of embed dim (evenly into x and y axes)
    # last 1/3 for temporal PE padded with 0s
    H, W = frame_size
    Hp, Wp = H // patch_size, W // patch_size

    # split dimensions (ensure temporal even)
    temporal_dim = (embed_dim // 3) & ~1
    spatial_dims = embed_dim - temporal_dim

    # split spatial dims between x and y (ensure both even)
    spatial_x_dim = (spatial_dims // 2) & ~1
    spatial_y_dim = spatial_dims - spatial_x_dim

    assert spatial_x_dim % 2 == 0 and spatial_y_dim % 2 == 0 and temporal_dim % 2 == 0

    # 2d PE for x and y axes
    pe_x = sincos_1d(Wp, spatial_x_dim, device, dtype)  # [Wp, Dx]
    pe_y = sincos_1d(Hp, spatial_y_dim, device, dtype)  # [Hp, Dy]
    pe_x = repeat(pe_x, 'wp dx -> hp wp dx', hp=Hp) # [Hp, Wp, Dx]
    pe_y = repeat(pe_y, 'hp dy -> hp wp dy', wp=Wp) # [Hp, Wp, Dy]

    pe_spatial = torch.cat([
        pe_x,
        pe_y,
        torch.zeros(Hp, Wp, temporal_dim, device=device, dtype=dtype)  # zero temporal tail
    ], dim=-1)  # [Hp, Wp, E]

    pe_spatial = rearrange(pe_spatial, 'hp wp e -> 1 (hp wp) e')  # [1, P, E]
    return pe_spatial  # [1, P, E]


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def rope_1d_cos_sin(seq_len, head_dim, device, dtype):
    """1D RoPE frequencies using the chunked convention: pairs (i, i+D/2).
    Returns cos, sin each of shape [seq_len, head_dim].
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=dtype) / half))  # [D/2]
    pos = torch.arange(seq_len, device=device, dtype=dtype)  # [L]
    angles = torch.outer(pos, freqs)  # [L, D/2]
    angles = torch.cat([angles, angles], dim=-1)  # [L, D]
    return torch.cos(angles), torch.sin(angles)


def rope_2d_cos_sin(Hp, Wp, head_dim, device, dtype):
    """2D RoPE frequencies for a (Hp x Wp) patch grid.

    The first D/2 head dims encode the y-axis (row) and the last D/2 encode
    the x-axis (col).  Each half is treated as an independent 1D RoPE group
    so the two axes do not mix during rotation.

    Returns cos, sin each of shape [Hp*Wp, head_dim].
    """
    assert head_dim % 4 == 0, f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}"
    half = head_dim // 2
    cos_y, sin_y = rope_1d_cos_sin(Hp, half, device, dtype)  # [Hp, D/2]
    cos_x, sin_x = rope_1d_cos_sin(Wp, half, device, dtype)  # [Wp, D/2]
    # broadcast to patch grid
    cos_y = cos_y[:, None].expand(Hp, Wp, half).reshape(Hp * Wp, half)  # [P, D/2]
    sin_y = sin_y[:, None].expand(Hp, Wp, half).reshape(Hp * Wp, half)
    cos_x = cos_x[None, :].expand(Hp, Wp, half).reshape(Hp * Wp, half)  # [P, D/2]
    sin_x = sin_x[None, :].expand(Hp, Wp, half).reshape(Hp * Wp, half)
    return torch.cat([cos_y, cos_x], dim=-1), torch.cat([sin_y, sin_x], dim=-1)  # each [P, D]


def apply_rope_1d(x, cos, sin):
    """Apply 1D RoPE to Q or K. Chunked convention: pairs (i, i+D/2).

    x  : [N, H, L, D]
    cos: [L, D]
    sin: [L, D]
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)  # rotate_half for chunked pairs
    return x * cos[None, None] + rot * sin[None, None]


def apply_rope_2d(x, cos, sin):
    """Apply 2D RoPE to Q or K. First D/2 dims = y-axis, last D/2 dims = x-axis.

    Each axis is rotated independently within its own D/2 block using the
    chunked sub-pair convention: pairs (i, i+D/4) within each half.

    x  : [N, H, P, D]
    cos: [P, D]
    sin: [P, D]
    """
    half = x.shape[-1] // 2
    quarter = half // 2
    # y-axis block
    x1a, x1b = x[..., :quarter], x[..., quarter:half]
    rot_y = torch.cat([-x1b, x1a], dim=-1)  # [N, H, P, D/2]
    # x-axis block
    x2a, x2b = x[..., half:half + quarter], x[..., half + quarter:]
    rot_x = torch.cat([-x2b, x2a], dim=-1)  # [N, H, P, D/2]
    rot = torch.cat([rot_y, rot_x], dim=-1)  # [N, H, P, D]
    return x * cos[None, None] + rot * sin[None, None]