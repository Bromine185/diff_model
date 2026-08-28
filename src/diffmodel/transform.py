"""Preprocessing, with an exact inverse for every step.

The paper's normalisation is written as

    (X_t - mu(X_t))^(1/p) / sigma(X_t)

with p = 1.5 for log returns and p = 1.0 for trading volumes after an arsinh.
Taken literally that expression is **undefined for negative values**, and roughly
half of all log returns are negative. Raising -0.003 to the power 1/1.5 gives a
complex number.

The intent is clearly a variance-stabilising compression of the tails that keeps
the sign, so we implement the sign-preserving form

    sign(x) * |x|^(1/p)

which agrees with the paper's formula wherever the paper's formula is defined,
and is monotone and exactly invertible everywhere. This is a correction, not a
deviation, and it is flagged in the notebook at the cell where it happens.

Why compress at all: raw 5-minute log returns are extremely leptokurtic. Handed
to a diffusion model unmodified, the loss is dominated by a handful of jump bars
and the model spends its capacity on outliers. p = 1.5 pulls the tails in while
preserving ordering, so the inverse restores them exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Elementwise, invertible
# --------------------------------------------------------------------------

def signed_power(x: np.ndarray, p: float) -> np.ndarray:
    """sign(x) * |x|^(1/p). Monotone, odd, exactly invertible by `signed_power_inv`."""
    return np.sign(x) * np.abs(x) ** (1.0 / p)


def signed_power_inv(y: np.ndarray, p: float) -> np.ndarray:
    """Inverse of `signed_power`: sign(y) * |y|^p."""
    return np.sign(y) * np.abs(y) ** p


def arsinh(x: np.ndarray) -> np.ndarray:
    """log(x + sqrt(x^2 + 1)).

    Used on volume. Preferred over log1p because it is defined for zero and
    negative inputs (a 5-minute bar can legitimately have zero volume) and behaves
    like log for large x, so it compresses the volume distribution's long right
    tail without a special case.
    """
    return np.arcsinh(x)


def arsinh_inv(y: np.ndarray) -> np.ndarray:
    return np.sinh(y)


# --------------------------------------------------------------------------
# Fitted, per-channel
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelTransform:
    """A fitted, invertible pipeline for one channel.

    Order, applied in sequence:
        1. arsinh              (volume only)
        2. subtract mean
        3. signed power 1/p
        4. divide by std       (of the power-transformed values)
        5. winsorise at k sigma

    Step 5 is the only one that is not exactly invertible, and only for samples
    it actually clips. At k = 10 that is a vanishing fraction of real data; the
    inverse simply returns the clipped value. `roundtrip_exact` reports whether
    any clipping occurred, so tests can assert on unclipped data.
    """

    use_arsinh: bool
    power: float
    mean: float
    std: float
    winsor_sigma: float

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        *,
        power: float,
        use_arsinh: bool = False,
        winsor_sigma: float = 10.0,
        eps: float = 1e-12,
    ) -> "ChannelTransform":
        v = arsinh(x) if use_arsinh else x
        mean = float(np.mean(v))
        w = signed_power(v - mean, power)
        std = float(np.std(w))
        return cls(
            use_arsinh=use_arsinh,
            power=power,
            mean=mean,
            std=max(std, eps),
            winsor_sigma=winsor_sigma,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        v = arsinh(x) if self.use_arsinh else x
        w = signed_power(v - self.mean, self.power) / self.std
        return np.clip(w, -self.winsor_sigma, self.winsor_sigma)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Decode, clamping to the range the forward transform could have produced.

        The clamp is not cosmetic. A trained model's output is unbounded, and
        decoding is `sign(z)*|z|^1.5` for returns and `sinh` for volume — both
        explosive. `sinh` overflows float64 at around x = 710, so a single wild
        pixel turns generated volume into `inf`, then `nan` on the first mean, and
        every stylized-fact metric downstream silently becomes `nan`. Observed
        exactly that on an undertrained smoke model.

        Clamping at +/- winsor_sigma is the symmetric counterpart of the forward
        pass, which winsorises at the same bound: the encoder can never emit a
        value outside +/- k sigma, so the decoder should never be asked to invert
        one. Beyond that bound the model is extrapolating past anything it saw.

        This bounds generated values rather than silently dropping them, so a
        model producing many out-of-range pixels shows up as a saturated
        distribution in the scorecard instead of as `nan`.
        """
        y = np.clip(y, -self.winsor_sigma, self.winsor_sigma)
        v = signed_power_inv(y * self.std, self.power) + self.mean
        return arsinh_inv(v) if self.use_arsinh else v

    def saturated_fraction(self, y: np.ndarray) -> float:
        """Fraction of decoder inputs that hit the clamp — report on generated data."""
        return float(np.mean(np.abs(y) >= self.winsor_sigma))

    def clipped_fraction(self, x: np.ndarray) -> float:
        """Fraction of `x` that winsorisation would clip. Report this; don't hide it."""
        v = arsinh(x) if self.use_arsinh else x
        w = signed_power(v - self.mean, self.power) / self.std
        return float(np.mean(np.abs(w) > self.winsor_sigma))

    def to_dict(self) -> dict:
        return {
            "use_arsinh": self.use_arsinh,
            "power": self.power,
            "mean": self.mean,
            "std": self.std,
            "winsor_sigma": self.winsor_sigma,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelTransform":
        return cls(**d)


@dataclass(frozen=True)
class DayTransform:
    """The three channel transforms, in image order: returns, spread, volume."""

    ret: ChannelTransform
    spread: ChannelTransform
    volume: ChannelTransform

    @classmethod
    def fit(
        cls,
        series: np.ndarray,
        *,
        return_power: float = 1.5,
        spread_power: float = 1.0,
        volume_power: float = 1.0,
        winsor_sigma: float = 10.0,
    ) -> "DayTransform":
        """`series` is (B, 3, n) over the TRAINING split only."""
        if series.ndim != 3 or series.shape[1] != 3:
            raise ValueError(f"expected (B, 3, n), got {series.shape}")
        return cls(
            ret=ChannelTransform.fit(series[:, 0], power=return_power, winsor_sigma=winsor_sigma),
            spread=ChannelTransform.fit(series[:, 1], power=spread_power, winsor_sigma=winsor_sigma),
            volume=ChannelTransform.fit(
                series[:, 2], power=volume_power, use_arsinh=True, winsor_sigma=winsor_sigma
            ),
        )

    @property
    def channels(self) -> tuple[ChannelTransform, ChannelTransform, ChannelTransform]:
        return (self.ret, self.spread, self.volume)

    def forward(self, series: np.ndarray) -> np.ndarray:
        return np.stack([t.forward(series[:, i]) for i, t in enumerate(self.channels)], axis=1)

    def inverse(self, series: np.ndarray) -> np.ndarray:
        return np.stack([t.inverse(series[:, i]) for i, t in enumerate(self.channels)], axis=1)

    def clipped_report(self, series: np.ndarray) -> dict[str, float]:
        names = ("returns", "spread", "volume")
        return {n: t.clipped_fraction(series[:, i]) for i, (n, t) in enumerate(zip(names, self.channels))}

    def saturated_report(self, normed: np.ndarray) -> dict[str, float]:
        """Fraction of DECODER inputs hitting the clamp, per channel.

        Run this on generated data. A well-trained model should saturate at
        roughly the rate real data does (~0.03% on spread, ~0 elsewhere). A high
        rate means the model is emitting values it never saw in training, and the
        decoded series is being bounded rather than faithfully inverted — which
        would quietly flatter the tail metrics.
        """
        names = ("returns", "spread", "volume")
        return {n: t.saturated_fraction(normed[:, i]) for i, (n, t) in enumerate(zip(names, self.channels))}

    def to_dict(self) -> dict:
        return {n: t.to_dict() for n, t in zip(("ret", "spread", "volume"), self.channels)}

    @classmethod
    def from_dict(cls, d: dict) -> "DayTransform":
        return cls(**{k: ChannelTransform.from_dict(v) for k, v in d.items()})
