"""Haar MODWT, hand-rolled, and the coefficient-plane image packing.

Why undecimated (MODWT) rather than the textbook decimated DWT
--------------------------------------------------------------
The paper describes packing "the k-th order coefficients into the k-th row of
pixels". A decimated DWT cannot do that: it halves the coefficient count at every
level, so on a length-128 signal level 1 has 64 coefficients, level 2 has 32, and
so on down to 1. Those do not fill a rectangular image. The maximal-overlap
(a.k.a. undecimated, stationary, or a-trous) transform keeps N coefficients at
*every* level, which is exactly what makes a (levels x time) image well-defined.
It is also shift-equivariant, which matters here: a volatility burst that happens
five minutes later should move along a row, not scramble the representation.

The cost is redundancy — 8 planes x 128 samples = 1024 numbers describing 128 —
and that is the point. The DDPM gets a representation where scale is an axis it
can attend across, rather than something entangled in a flat sequence.

Definitions (Percival & Walden, *Wavelet Methods for Time Series Analysis*)
--------------------------------------------------------------------------
Haar filters rescaled for the MODWT are g~ = [1/2, 1/2] and h~ = [1/2, -1/2].
At level j the filters are dilated by s = 2^(j-1) (insert zeros between taps),
and convolution is circular. With V_0 = x:

    W_j[t] = 0.5 * (V_{j-1}[t] - V_{j-1}[t - s])      detail
    V_j[t] = 0.5 * (V_{j-1}[t] + V_{j-1}[t - s])      smooth

The inverse follows from two identities that both recover V_{j-1}[t]:

    V_j[t]     + W_j[t]     = V_{j-1}[t]
    V_j[t + s] - W_j[t + s] = V_{j-1}[t]

so averaging them gives the reconstruction actually used below. This is exact in
floating point to ~1e-16 relative, and `tests/test_wavelet.py` asserts it.

All functions operate on the last axis and broadcast over any leading axes, so
the same code handles a single series, a (channels, N) day, or a
(batch, channels, N) minibatch.
"""

from __future__ import annotations

import numpy as np

PadMode = str  # "mirror" | "zero" | "periodic"


# --------------------------------------------------------------------------
# Padding to a power of two
# --------------------------------------------------------------------------

def pad_to_pow2(x: np.ndarray, target: int, mode: PadMode = "mirror") -> np.ndarray:
    """Extend the last axis from n to `target` samples.

    The session is 78 five-minute bars; the transform wants 128 = 2^7. The paper
    calls this "mirror expansion" (it goes 390 -> 512).

    Modes, all ablatable — see RESEARCH.md:
      mirror   : reflect without repeating the edge sample, [a b c d] -> [a b c d c b].
                 Keeps the extension statistically like the signal, which matters
                 because circular convolution will wrap plane edges together.
      zero     : pad with zeros. Honest but injects a hard discontinuity that the
                 model will happily learn to reproduce.
      periodic : wrap the signal head onto its tail.

    The model WILL learn whatever symmetry the padding creates. That is a known
    artifact of the approach, not a bug in this function.
    """
    n = x.shape[-1]
    if n == target:
        return x.copy()
    if n > target:
        raise ValueError(f"signal length {n} exceeds target {target}")
    pad = target - n
    if mode == "mirror":
        if pad >= n:
            raise ValueError(f"mirror padding needs pad ({pad}) < n ({n})")
        widths = [(0, 0)] * (x.ndim - 1) + [(0, pad)]
        return np.pad(x, widths, mode="reflect")
    if mode == "zero":
        widths = [(0, 0)] * (x.ndim - 1) + [(0, pad)]
        return np.pad(x, widths, mode="constant", constant_values=0.0)
    if mode == "periodic":
        widths = [(0, 0)] * (x.ndim - 1) + [(0, pad)]
        return np.pad(x, widths, mode="wrap")
    raise ValueError(f"unknown pad mode {mode!r}")


def unpad(x: np.ndarray, n: int) -> np.ndarray:
    """Inverse of `pad_to_pow2`: keep the first n samples, drop the extension."""
    return x[..., :n]


# --------------------------------------------------------------------------
# Forward / inverse MODWT
# --------------------------------------------------------------------------

def modwt(x: np.ndarray, levels: int) -> np.ndarray:
    """Haar MODWT.

    Returns stacked coefficient planes with shape (..., levels + 1, N):
    rows 0..levels-1 are the details W_1..W_J (finest first), row `levels` is the
    final smooth V_J. Finest-first ordering means row 0 is 5-minute-scale
    variation and the last detail row is roughly half-session scale.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    n = x.shape[-1]
    if 2 ** levels > n:
        raise ValueError(f"levels={levels} needs N >= {2 ** levels}, got {n}")

    v = x.astype(np.float64, copy=True)
    planes = []
    for j in range(1, levels + 1):
        s = 1 << (j - 1)                     # 2^(j-1)
        v_lag = np.roll(v, s, axis=-1)       # V[t - s], circular
        planes.append(0.5 * (v - v_lag))     # W_j
        v = 0.5 * (v + v_lag)                # V_j
    planes.append(v)                          # V_J
    return np.stack(planes, axis=-2)


def imodwt(planes: np.ndarray) -> np.ndarray:
    """Exact inverse of `modwt`. Input (..., levels + 1, N) -> output (..., N)."""
    levels = planes.shape[-2] - 1
    v = planes[..., levels, :].astype(np.float64, copy=True)
    for j in range(levels, 0, -1):
        s = 1 << (j - 1)
        w = planes[..., j - 1, :]
        w_lead = np.roll(w, -s, axis=-1)     # W[t + s]
        v_lead = np.roll(v, -s, axis=-1)     # V[t + s]
        # average of the two identities in the module docstring
        v = 0.5 * (v + w) + 0.5 * (v_lead - w_lead)
    return v


# --------------------------------------------------------------------------
# Series <-> image
# --------------------------------------------------------------------------

def series_to_image(
    series: np.ndarray,
    levels: int,
    target: int,
    pad_mode: PadMode = "mirror",
) -> np.ndarray:
    """(..., C, n) time series -> (..., C, levels + 1, target) coefficient image.

    Channel order is the paper's: R = log returns, G = spread, B = volume. Each
    channel is transformed independently; the DDPM is what learns their joint
    structure, exactly as in the paper (this is why cross-correlation between
    returns, spreads and volumes is a fact the model can get *wrong*, and so is
    worth scoring).
    """
    padded = pad_to_pow2(series, target, mode=pad_mode)
    return modwt(padded, levels)


def image_to_series(image: np.ndarray, n: int) -> np.ndarray:
    """(..., C, levels + 1, target) -> (..., C, n). Inverse of `series_to_image`."""
    return unpad(imodwt(image), n)


# --------------------------------------------------------------------------
# Per-level standardisation
# --------------------------------------------------------------------------
# Detail planes at different scales have very different variances, and a DDPM
# wants inputs on a common scale near [-1, 1]. This is a fixed affine map fitted
# ONCE on the training split and applied unchanged at generation time, so it
# stays exactly invertible and leaks nothing from validation.

def fit_level_stats(images: np.ndarray, eps: float = 1e-8) -> dict[str, np.ndarray]:
    """Per (channel, level) mean and std over a training batch.

    `images` is (B, C, L, N). Returns arrays of shape (C, L, 1) so they broadcast.
    """
    mean = images.mean(axis=(0, 3), keepdims=False)[..., None]
    std = images.std(axis=(0, 3), keepdims=False)[..., None]
    return {"mean": mean, "std": np.maximum(std, eps)}


def apply_level_stats(images: np.ndarray, stats: dict[str, np.ndarray], scale: float = 3.0) -> np.ndarray:
    """Standardise per level, then divide by `scale` to land mostly in [-1, 1].

    No clipping: clipping would not be invertible, and the 10-sigma winsorisation
    already happened upstream on the raw series.
    """
    return (images - stats["mean"]) / stats["std"] / scale


def invert_level_stats(images: np.ndarray, stats: dict[str, np.ndarray], scale: float = 3.0) -> np.ndarray:
    """Exact inverse of `apply_level_stats`."""
    return images * scale * stats["std"] + stats["mean"]
