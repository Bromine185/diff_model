"""The ablation that tests the paper's actual claim.

The paper's thesis is not "diffusion models can generate financial time series" —
it is "representing the series as a wavelet-coefficient IMAGE and then using an
image diffusion model beats operating on the raw series". Every other baseline
tests the first claim. Only this one tests the second, because it changes exactly
one thing: the representation.

Same diffusion process, same schedule, same training loop, same optimiser, same
loss. The only difference is a 1D UNet over the raw (3, 128) padded series
instead of a 2D UNet over the (3, 8, 128) coefficient planes. Parameter counts
are matched as closely as the dimensionality allows so the comparison is not
secretly about capacity.

If this scores as well as the wavelet model, the paper's central contribution
does not replicate on our data, and that is the finding — it would go in
RESEARCH.md as a null result, not get tuned away.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config import BARS_PER_DAY, PADDED_LEN
from ..unet import SinusoidalTimeEmbedding, _groups
from ..wavelet import pad_to_pow2, unpad


class ResBlock1d(nn.Module):
    """1D counterpart of `unet.ResBlock`, identical structure."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Attention1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, n).unbind(1)
        attn = torch.softmax(torch.einsum("bci,bcj->bij", q, k) * self.scale, dim=-1)
        return x + self.proj(torch.einsum("bij,bcj->bci", attn, v).reshape(b, c, n))


class UNet1D(nn.Module):
    """Noise predictor over raw series. Signature-compatible with `unet.UNet`.

    Four downsamples rather than the 2D model's two: the raw series is 128 long
    where the coefficient image is only 8 rows tall, so the 1D model needs more
    stages to reach a comparable receptive field. This is exactly the asymmetry
    the wavelet representation is supposed to remove — the transform hands the
    network multi-scale structure that the 1D model has to build for itself.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 2, 4),
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
        self.stem = nn.Conv1d(in_channels, base_channels, 3, padding=1)

        self.down_blocks, self.downsamples = nn.ModuleList(), nn.ModuleList()
        skips = [base_channels]
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            stage = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage.append(ResBlock1d(ch, out_ch, t_dim, dropout))
                ch = out_ch
                skips.append(ch)
            self.down_blocks.append(stage)
            if i < len(channel_mults) - 1:
                self.downsamples.append(nn.Conv1d(ch, ch, 3, stride=2, padding=1))
                skips.append(ch)
            else:
                self.downsamples.append(nn.Identity())

        self.mid1 = ResBlock1d(ch, ch, t_dim, dropout)
        self.mid_attn = Attention1d(ch) if attention_at_bottleneck else nn.Identity()
        self.mid2 = ResBlock1d(ch, ch, t_dim, dropout)

        self.up_blocks, self.upsamples = nn.ModuleList(), nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            stage = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                stage.append(ResBlock1d(ch + skips.pop(), out_ch, t_dim, dropout))
                ch = out_ch
            self.up_blocks.append(stage)
            self.upsamples.append(nn.Identity() if i == 0 else nn.Conv1d(ch, ch, 3, padding=1))
            self._needs_interp = True

        self.out_norm = nn.GroupNorm(_groups(ch), ch)
        self.out_conv = nn.Conv1d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.stem(x)
        acc = [h]
        for stage, down in zip(self.down_blocks, self.downsamples):
            for block in stage:
                h = block(h, t_emb)
                acc.append(h)
            if not isinstance(down, nn.Identity):
                h = down(h)
                acc.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)

        for stage, up in zip(self.up_blocks, self.upsamples):
            for block in stage:
                h = block(torch.cat([h, acc.pop()], dim=1), t_emb)
            if not isinstance(up, nn.Identity):
                h = up(F.interpolate(h, scale_factor=2, mode="nearest"))

        return self.out_conv(F.silu(self.out_norm(h)))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class RawSeriesCodec:
    """Encode/decode for the ablation: no wavelet, just pad and standardise.

    Deliberately mirrors `pipeline.Codec` so the two paths differ ONLY in the
    wavelet step. Same day-transform, same padding, same per-position
    standardisation — everything except the MODWT.
    """

    def __init__(self, day_transform, pad_mode: str = "mirror", scale: float = 3.0):
        self.day_transform = day_transform
        self.pad_mode = pad_mode
        self.scale = scale
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, series: np.ndarray) -> "RawSeriesCodec":
        """Fit per-channel mean/std on the TRAIN split only.

        Per channel, not per time-position: the 2D path standardises per
        (channel, wavelet level), and the positional analogue here would be per
        (channel, bar-of-day) — which would strip out the intraday U-shape, one
        of the very facts the generator is being scored on. Removing it from the
        ablation's inputs but not the wavelet model's would rig the comparison.
        """
        padded = pad_to_pow2(self.day_transform.forward(series), PADDED_LEN, self.pad_mode)
        self.mean = padded.mean(axis=(0, 2), keepdims=True)   # (1, 3, 1)
        self.std = np.maximum(padded.std(axis=(0, 2), keepdims=True), 1e-8)
        return self

    def encode(self, series: np.ndarray) -> np.ndarray:
        padded = pad_to_pow2(self.day_transform.forward(series), PADDED_LEN, self.pad_mode)
        return (padded - self.mean) / self.std / self.scale

    def decode(self, x: np.ndarray) -> np.ndarray:
        padded = x * self.scale * self.std + self.mean
        return self.day_transform.inverse(unpad(padded, BARS_PER_DAY))
