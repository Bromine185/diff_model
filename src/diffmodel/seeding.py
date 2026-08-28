"""Seeded, forkable randomness.

Ported from the `3d-night` convention. The point of forking by *label* rather
than by counter: re-running one component must not perturb another's draws. If
the training shuffle and the sampler share a single global generator, adding one
extra sampling call silently changes every subsequent training batch, and the
run stops being reproducible in the way that matters.

    rng = fork("train-shuffle")   # same stream every time, regardless of
    rng = fork("sampler")         # what else ran first
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch

#: Every stream in the project derives from this. Change it and every number moves.
ROOT_SEED = 20260828


def _label_seed(label: str, root: int = ROOT_SEED) -> int:
    """Derive a stable 63-bit seed from a text label.

    BLAKE2b rather than Python's `hash()` because the latter is salted per process
    (PYTHONHASHSEED) and would produce different streams on every run.
    """
    h = hashlib.blake2b(f"{root}:{label}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") >> 1


def fork(label: str, root: int = ROOT_SEED) -> np.random.Generator:
    """An independent numpy Generator for this label."""
    return np.random.default_rng(_label_seed(label, root))


def torch_generator(label: str, root: int = ROOT_SEED) -> torch.Generator:
    """An independent torch Generator for this label (CPU; use for DataLoader shuffles)."""
    g = torch.Generator()
    g.manual_seed(_label_seed(label, root) % (2**63 - 1))
    return g


def seed_everything(label: str = "global", root: int = ROOT_SEED) -> int:
    """Pin every global RNG. Returns the seed used, so it can be logged.

    Note this does NOT make CUDA/MPS matmuls bitwise deterministic on its own —
    see `deterministic_algorithms()` for that, and read its docstring before
    reaching for it.
    """
    seed = _label_seed(label, root) % (2**31 - 1)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def deterministic_algorithms(enabled: bool = True) -> None:
    """Force deterministic kernels where torch offers them.

    Costs throughput and, on some backends, raises for ops that have no
    deterministic implementation. Used in tests; left off for FULL training runs
    where the wall-clock matters more than bitwise reproducibility of a 100-epoch
    job that already checkpoints.
    """
    torch.use_deterministic_algorithms(enabled, warn_only=True)
    if enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
