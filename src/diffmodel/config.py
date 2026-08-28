"""SMOKE and FULL presets, and device detection.

One dataclass, two presets. SMOKE is not a separate toy code path — it is the
same code with smaller numbers, so whatever runs locally is exactly what Colab
runs. Its job is to catch shape errors, alpha-bar indexing bugs, wrong inverse
transforms and NaN blowups. Sample quality under SMOKE will be garbage; that is
the expected result, not a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "data" / "fixtures"
CHECKPOINTS = REPO / "checkpoints"

# --- Geometry -------------------------------------------------------------
# Fixed by the 5-minute session and the wavelet, not tunable:
#     09:30-15:55 = 78 five-minute bars
#     mirror-expand 78 -> 128 = 2^7
#     Haar MODWT levels 1..7 + smooth V7 = 8 planes x 128 samples
BARS_PER_DAY = 78
PADDED_LEN = 128          # 2^7
N_LEVELS = 7              # log2(128); planes = N_LEVELS + 1 (details + smooth)
IMAGE_ROWS = N_LEVELS + 1  # 8
IMAGE_COLS = PADDED_LEN    # 128
CHANNELS = 3               # R = log return, G = spread, B = volume


def pick_device(prefer: str | None = None) -> torch.device:
    """cuda (Colab) > mps (local M4) > cpu.

    `prefer` overrides for tests that need CPU determinism.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class Config:
    name: str

    # data
    max_images: int | None      # None = use everything available
    val_sessions: int           # trailing sessions held out, split by DATE never randomly

    # diffusion
    timesteps: int
    beta_schedule: str          # "linear" | "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # unet
    base_channels: int = 128
    channel_mults: tuple[int, ...] = (1, 2, 4)   # 3 stages: 8 rows halve 3 times
    num_res_blocks: int = 2
    attention_at_bottleneck: bool = True
    dropout: float = 0.0

    # optimisation
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.999
    grad_clip: float = 1.0

    # checkpointing (Colab disconnects; a 100-epoch run that loses everything at
    # hour three is the default outcome without this)
    ckpt_every: int = 5
    ckpt_dir: Path = field(default=CHECKPOINTS)

    # preprocessing, per the paper
    return_power: float = 1.5   # sign-preserving: sign(x)*|x|^(1/p)
    volume_power: float = 1.0   # applied after arsinh
    spread_power: float = 1.0
    winsor_sigma: float = 10.0

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (CHANNELS, IMAGE_ROWS, IMAGE_COLS)


SMOKE = Config(
    name="SMOKE",
    # Enough images and epochs that the loop exercises a real optimisation
    # trajectory (~128 steps) rather than a dozen, while still finishing in
    # seconds. Sample QUALITY is still expected to be poor -- SMOKE proves the
    # code is correct, not that the model is good.
    max_images=512,
    val_sessions=3,
    timesteps=50,
    # Cosine, NOT linear, and this is not a style preference. The linear schedule's
    # endpoints (1e-4 .. 0.02) are tuned for T=1000; at T=50 they leave
    # alphabar_T = 0.61, so the forward chain never reaches pure noise and the
    # sampler would start from N(0, I) — a distribution the model was never
    # trained on. The cosine schedule defines alphabar directly and reaches ~0 at
    # any T. `test_short_schedule_warns_that_it_never_reaches_noise` pins this.
    beta_schedule="cosine",
    base_channels=32,
    channel_mults=(1, 2, 4),
    num_res_blocks=1,
    attention_at_bottleneck=False,
    epochs=4,
    batch_size=16,
    ckpt_every=1,
    # 0.9, not FULL's 0.999. SMOKE runs ~128 optimizer steps; at decay 0.999 the
    # EMA time constant is ~1000 steps, so the shadow would still be ~88% random
    # initialisation and we would be sampling from an untrained network while
    # believing we had tested the trained path. Decay must be matched to the step
    # count -- `train()` warns when the shadow would retain >20% of its init.
    ema_decay=0.9,
)

FULL = Config(
    name="FULL",
    max_images=None,
    val_sessions=10,
    timesteps=1000,
    beta_schedule="linear",
    base_channels=128,
    channel_mults=(1, 2, 4),   # 128 -> 256 -> 512, matching the paper's widths
    num_res_blocks=2,
    attention_at_bottleneck=True,
    epochs=100,
    batch_size=64,
)

PRESETS = {"SMOKE": SMOKE, "FULL": FULL}


def get_config(name: str = "SMOKE", **overrides) -> Config:
    """Fetch a preset, optionally tweaked. `get_config("FULL", epochs=20)`."""
    key = name.upper()
    if key not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; expected one of {sorted(PRESETS)}")
    cfg = PRESETS[key]
    return replace(cfg, **overrides) if overrides else cfg


def auto_config(**overrides) -> Config:
    """FULL on a CUDA machine (i.e. Colab), SMOKE otherwise.

    This is what the notebook calls, so the same file runs both places with no
    editing. MPS deliberately maps to SMOKE: the M4 can technically run FULL,
    but the project's rule is that real training happens on Colab.
    """
    name = "FULL" if torch.cuda.is_available() else "SMOKE"
    return get_config(name, **overrides)
