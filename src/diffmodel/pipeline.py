"""End-to-end encode/decode: raw bars -> model-ready images -> back to price series.

Keeping this in one place matters because the chain is long and every link must
invert. Encoding is

    bars -> (B, 3, 78) series
         -> per-channel power/arsinh normalisation      (fitted on TRAIN only)
         -> mirror-expand 78 -> 128
         -> Haar MODWT, 8 coefficient planes
         -> per-(channel, level) standardisation        (fitted on TRAIN only)
         -> (B, 3, 8, 128) image in roughly [-1, 1]

and decoding runs it backwards. Both normalisations are fitted on the training
split alone: fitting on everything would leak validation statistics into the
model's input scaling, which is a small leak but a real one, and exactly the kind
this project claims to care about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import BARS_PER_DAY, Config, N_LEVELS, PADDED_LEN
from .data import DayImages, build_day_images
from .seeding import fork
from .transform import DayTransform
from .wavelet import (
    apply_level_stats,
    fit_level_stats,
    image_to_series,
    invert_level_stats,
    series_to_image,
)


@dataclass
class Codec:
    """The fitted encode/decode pair. Everything needed to go both ways."""

    day_transform: DayTransform
    level_stats: dict[str, np.ndarray]
    pad_mode: str = "mirror"
    image_scale: float = 3.0

    def encode(self, series: np.ndarray) -> np.ndarray:
        """(B, 3, 78) raw series -> (B, 3, 8, 128) model-ready image."""
        normed = self.day_transform.forward(series)
        img = series_to_image(normed, N_LEVELS, PADDED_LEN, pad_mode=self.pad_mode)
        return apply_level_stats(img, self.level_stats, self.image_scale)

    def decode(self, image: np.ndarray) -> np.ndarray:
        """(B, 3, 8, 128) image -> (B, 3, 78) raw series. Exact inverse of `encode`."""
        img = invert_level_stats(image, self.level_stats, self.image_scale)
        normed = image_to_series(img, BARS_PER_DAY)
        return self.day_transform.inverse(normed)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "day_transform": self.day_transform.to_dict(),
                    "level_stats": {k: v.tolist() for k, v in self.level_stats.items()},
                    "pad_mode": self.pad_mode,
                    "image_scale": self.image_scale,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "Codec":
        d = json.loads(Path(path).read_text())
        return cls(
            day_transform=DayTransform.from_dict(d["day_transform"]),
            level_stats={k: np.asarray(v) for k, v in d["level_stats"].items()},
            pad_mode=d["pad_mode"],
            image_scale=d["image_scale"],
        )

    def roundtrip_report(self, series: np.ndarray, dtype=np.float32) -> dict[str, float]:
        """Encode/decode error per channel, measured in units of that channel's std.

        RELATIVE error is the wrong metric for this codec and using it will
        mislead you. The inverse transforms are `sign(z)*|z|^1.5` for returns and
        `sinh` for volume; both amplify relative error without bound as the value
        approaches zero. A 5-minute return of 1e-9 decoding as 2e-9 is a 100%
        relative error and completely irrelevant to every downstream metric.

        Absolute error scaled by the channel's own standard deviation is the
        meaningful quantity, and that is what this reports.

        Measured on the real fixture: ~4e-15 (float64) and ~5e-07 (float32) for
        returns. The spread channel is the exception — winsorisation clips ~0.03%
        of points at 10 sigma and those genuinely do not invert, which is what
        winsorisation IS, not a defect.
        """
        back = self.decode(self.encode(series).astype(dtype))
        err = np.abs(back - series)
        names = ("returns", "spread", "volume")
        return {
            n: float(err[:, i].max() / max(series[:, i].std(), 1e-30))
            for i, n in enumerate(names)
        }


@dataclass
class Prepared:
    train_images: np.ndarray     # (N, 3, 8, 128)
    val_images: np.ndarray
    train_series: np.ndarray     # (N, 3, 78) raw, for the scorecard's "real" reference
    val_series: np.ndarray
    codec: Codec
    meta_train: DayImages
    meta_val: DayImages

    def summary(self) -> str:
        return (
            f"train {self.train_images.shape}  val {self.val_images.shape}\n"
            f"  train sessions {np.unique(self.meta_train.sessions).size}, "
            f"val sessions {np.unique(self.meta_val.sessions).size}\n"
            f"  image range [{self.train_images.min():.2f}, {self.train_images.max():.2f}], "
            f"std {self.train_images.std():.3f}"
        )


def prepare(cfg: Config, *, spread_method: str = "highlow", verbose: bool = True) -> Prepared:
    """Load the fixture and produce everything training needs."""
    day = build_day_images(spread_method=spread_method)
    train_meta, val_meta = day.split_by_date(cfg.val_sessions)

    # SMOKE subsamples the TRAINING split only; validation stays whole so the
    # held-out comparison is still meaningful.
    if cfg.max_images is not None:
        train_meta = train_meta.take(cfg.max_images, rng=fork("smoke-subsample"))
        val_meta = val_meta.take(max(32, cfg.max_images // 4), rng=fork("smoke-subsample-val"))

    codec_transform = DayTransform.fit(
        train_meta.series,
        return_power=cfg.return_power,
        spread_power=cfg.spread_power,
        volume_power=cfg.volume_power,
        winsor_sigma=cfg.winsor_sigma,
    )
    if verbose:
        rep = codec_transform.clipped_report(train_meta.series)
        print("winsorised fraction at "
              f"{cfg.winsor_sigma:g} sigma: "
              + ", ".join(f"{k}={v:.2%}" for k, v in rep.items()))

    # Level statistics must be fitted on TRAIN images only, so build those first.
    tmp = Codec(codec_transform, {"mean": np.zeros((3, N_LEVELS + 1, 1)),
                                  "std": np.ones((3, N_LEVELS + 1, 1))})
    raw_train_img = series_to_image(
        codec_transform.forward(train_meta.series), N_LEVELS, PADDED_LEN, pad_mode=tmp.pad_mode
    )
    codec = Codec(codec_transform, fit_level_stats(raw_train_img), pad_mode=tmp.pad_mode)

    prepared = Prepared(
        train_images=codec.encode(train_meta.series).astype(np.float32),
        val_images=codec.encode(val_meta.series).astype(np.float32),
        train_series=train_meta.series,
        val_series=val_meta.series,
        codec=codec,
        meta_train=train_meta,
        meta_val=val_meta,
    )
    if verbose:
        print(prepared.summary())
    return prepared
