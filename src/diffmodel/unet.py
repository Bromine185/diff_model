"""A UNet noise predictor, written out rather than imported from `diffusers`.

The whole point of this project is that the internals are legible, so this is
built from primitives. It is the standard DDPM architecture (Ho et al. 2020):
a symmetric encoder/decoder with skip connections, residual blocks conditioned on
the diffusion timestep through a sinusoidal embedding, and self-attention at the
lowest resolution.

Shape note. Our images are 8 x 128 (levels x time), not square. With
`channel_mults = (1, 2, 4)` there are three resolution levels and therefore two
downsamples:

    level 0:  8 x 128   base
    level 1:  4 x  64   base * 2
    level 2:  2 x  32   base * 4      <- bottleneck, attention here

The paper's widths were 128-128-256-256-512 over four downsamples on a 16 x 256
image. Ours is one octave smaller in both dimensions, so it gets one fewer stage
while keeping the same channel progression (128 -> 256 -> 512 under FULL).

The rows axis is short and *ordered by scale*, not by space. Downsampling along
it mixes adjacent wavelet levels, which is intended: neighbouring scales are
correlated (a volatility burst shows up at several octaves at once) and letting
the network pool across them is the mechanism by which the wavelet
representation is supposed to help.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int, maximum: int = 32) -> int:
    """Largest divisor of `channels` not exceeding `maximum`, for GroupNorm."""
    g = min(maximum, channels)
    while channels % g != 0:
        g -= 1
    return max(g, 1)


class SinusoidalTimeEmbedding(nn.Module):
    """Transformer positional encoding, applied to the diffusion timestep.

    The network must behave differently at t = 900 (almost pure noise) than at
    t = 10 (almost clean), so t has to be fed in. A raw integer would be a poor
    input — the network would have to learn scale from scratch — so t is mapped
    to a bank of sinusoids at geometrically spaced frequencies. Nearby timesteps
    get nearby embeddings, which is what lets one network serve all T steps.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Two convolutions with a timestep-conditioned shift, plus a residual path.

    The time embedding enters as a per-channel additive bias between the two
    convolutions (Ho et al.'s formulation). Additive rather than concatenated
    keeps the spatial path unchanged in width and makes the conditioning cheap.
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # 1x1 projection so the residual can be added when widths differ
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Single-head self-attention over all H*W positions.

    Applied only at the bottleneck, where the map is 2 x 32 = 64 positions and
    the quadratic cost is irrelevant. Convolutions have a bounded receptive
    field; attention lets a coefficient at 09:35 in the finest level interact
    directly with one at 15:50 in the coarsest, which is how a whole-session
    volatility regime gets represented coherently.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        attn = torch.softmax(torch.einsum("bci,bcj->bij", q, k) * self.scale, dim=-1)
        out = torch.einsum("bij,bcj->bci", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Strided convolution. Learned, unlike pooling, and cheap."""

    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Nearest-neighbour interpolation then a 3x3 conv.

    Preferred over ConvTranspose2d, which produces checkerboard artifacts. In an
    image those are a cosmetic annoyance; here they would become periodic
    structure in the wavelet coefficients and decode into fake oscillations in
    the price path.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """Predicts the noise epsilon added to a wavelet-coefficient image.

    Input and output are both (B, C, H, W) with C = 3 — the network's job is to
    output an estimate of the noise, the same shape as the image itself.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_at_bottleneck: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        t_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- encoder ---
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = [base_channels]
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            stage = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage.append(ResBlock(ch, out_ch, t_dim, dropout))
                ch = out_ch
                skip_channels.append(ch)
            self.down_blocks.append(stage)
            if i < len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                skip_channels.append(ch)
            else:
                self.downsamples.append(nn.Identity())

        # --- bottleneck ---
        self.mid_block1 = ResBlock(ch, ch, t_dim, dropout)
        self.mid_attn = AttentionBlock(ch) if attention_at_bottleneck else nn.Identity()
        self.mid_block2 = ResBlock(ch, ch, t_dim, dropout)

        # --- decoder ---
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            stage = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                stage.append(ResBlock(ch + skip_channels.pop(), out_ch, t_dim, dropout))
                ch = out_ch
            self.up_blocks.append(stage)
            self.upsamples.append(Upsample(ch) if i > 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(_groups(ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        # Zero-init the final conv so the network starts by predicting zero noise.
        # Training then begins from a well-behaved identity-ish map rather than
        # from random garbage, which measurably stabilises the first epochs.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        h = self.stem(x)
        skips = [h]
        for stage, down in zip(self.down_blocks, self.downsamples):
            for block in stage:
                h = block(h, t_emb)
                skips.append(h)
            if not isinstance(down, nn.Identity):
                h = down(h)
                skips.append(h)

        h = self.mid_block2(self.mid_attn(self.mid_block1(h, t_emb)), t_emb)

        for stage, up in zip(self.up_blocks, self.upsamples):
            for block in stage:
                h = block(torch.cat([h, skips.pop()], dim=1), t_emb)
            if not isinstance(up, nn.Identity):
                h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
