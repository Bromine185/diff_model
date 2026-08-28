#!/usr/bin/env python
"""Score every generator against held-out real data and print one table.

    python scripts/run_scorecard.py                  # classical baselines + SMOKE DDPM (~2 min)
    python scripts/run_scorecard.py --n 1024         # more synthetic days per row
    python scripts/run_scorecard.py --no-ddpm        # classical only, ~3 seconds
    python scripts/run_scorecard.py --csv out.csv    # also write the table

Deterministic: two runs produce byte-identical output, with no filtering needed.
Verify with

    python scripts/run_scorecard.py > a.txt && \
    python scripts/run_scorecard.py > b.txt && diff a.txt b.txt

That is why wall-clock timings are behind `--timings` rather than printed by
default. A run that is "identical apart from three lines you have to remember to
strip" is a run whose reproducibility nobody actually checks, and the timings are
diagnostics rather than results.

WHICH DATA, AND WHY
-------------------
Three splits are in play and conflating them would silently rig the table.

  fit split       `prepare(get_config("SMOKE")).train_series` — the 512 day-images
                  the committed SMOKE checkpoint was actually trained on. Every
                  classical baseline is fitted on exactly this, so no generator
                  gets to see data the DDPM did not. Handing GARCH all 20,856
                  training days while the DDPM saw 512 would make the comparison
                  about sample size rather than about method.

  codec           the one `prepare(get_config("SMOKE"))` returns, fitted on those
                  same 512 images. It is not interchangeable with the FULL codec:
                  the day-transform's mean/std and the per-level statistics are
                  estimated from whatever split was passed, so decoding SMOKE
                  samples with a FULL-fitted codec would apply the inverse of a
                  transform the model never saw.

  real reference  the FULL held-out split — all 1,078 day-images from the last 3
                  sessions — NOT SMOKE's 128-image subsample of it. The reference
                  set's only job is to estimate the real stylized facts, and it is
                  not fitted to by anyone, so a bigger one is strictly better.
                  It matters: measuring the real-vs-real noise floor against the
                  128-image subsample gives cross_corr_frobenius = 0.414, against
                  the full split 0.107. Four-fifths of that residual was the
                  reference set being too small to pin down corr(|r|, volume), and
                  a noise floor that high would have swamped the differences
                  between generators.

So: SMOKE model, SMOKE codec, SMOKE training data, FULL validation reference.

THE ROW THAT MAKES THE TABLE READABLE
-------------------------------------
`real_train` scores the real TRAINING days as if they were a generator. It is not
a baseline; it is the noise floor: what "as good as it is possible to be" looks
like given finite samples and a genuine train/reference shift. Read every row
against this one, never against zero.

It is a NOISY floor, and that changes how a below-floor cell reads. The row is a
single fixed 512-day draw, so its distance from the reference is one realisation
of an estimator with its own spread. Bootstrap-resampling those same 512 days 20
times (RESEARCH.md Phase 5 carries the table) gives, for instance, kurtosis
0.059-0.420 around the shipped 0.258 and tail_within 0.169-0.417 around 0.268.
So a cell below the floor means ONE OF TWO things and the table cannot tell them
apart on its own:

  * the quantity was SUPPLIED rather than reproduced — `garch_seasonal` on
    `intraday_vol_mse`, where the real profile is multiplied back in at
    simulation time, or `shuffled` on any marginal, whose multiset is a real
    day's; or
  * the gap is smaller than the floor row's own estimator noise, which is the
    case for `block_bootstrap` on `kurtosis_rel_err` (0.118 against a floor of
    0.258 whose bootstrap band is 0.059-0.420) and for every other below-floor
    cell in the shipped table bar one.

Deciding between them takes an argument about how the generator is built, not a
comparison of two numbers, and the second case is common enough that the older
categorical reading — "below the floor means it was handed the answer" — is
wrong. RESEARCH.md Phase 5 works through the shipped table cell by cell.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from diffmodel import ddpm, evaluate, pipeline, seeding  # noqa: E402
from diffmodel.baselines import classical  # noqa: E402
from diffmodel.config import BARS_PER_DAY, get_config  # noqa: E402
from diffmodel.data import build_day_images  # noqa: E402
from diffmodel.unet import UNet  # noqa: E402
from diffmodel.wavelet import image_to_series, invert_level_stats  # noqa: E402

DEFAULT_CKPT = REPO / "checkpoints" / "smoke_latest.pt"


# --------------------------------------------------------------------------
# The diffusion row
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _global_rngs_restored():
    """Let the block pin the process-global RNGs, then hand the caller its streams back.

    The reverse chain no longer needs this: `ddpm.sample` now takes a `generator`
    argument and this script passes `seeding.torch_generator("scorecard-ddpm-sample")`,
    so sampling is reproducible without touching global state at all. The wrapper
    stays as containment for what is left. Constructing the UNet initialises its
    weights from the global torch RNG (they are overwritten by `load_state_dict`
    immediately after, but the stream has already advanced), and any future
    library call inside this block that reaches for a global would do the same.
    Non-negotiable #4 says streams fork by label so that re-running one component
    cannot move another's numbers; with the state saved and restored, `sample_ddpm`
    is side-effect-free for its callers no matter what it does internally. Saving
    numpy and python-random as well as torch because `seeding.seed_everything`
    pins all three.
    """
    saved_random = random.getstate()
    saved_numpy = np.random.get_state()
    saved_torch = torch.get_rng_state()
    saved_hashseed = os.environ.get("PYTHONHASHSEED")
    saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(saved_random)
        np.random.set_state(saved_numpy)
        torch.set_rng_state(saved_torch)
        if saved_cuda is not None:
            torch.cuda.set_rng_state_all(saved_cuda)
        if saved_hashseed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = saved_hashseed


def sample_ddpm(cfg, codec, ckpt: Path, n: int, device: torch.device) -> tuple[np.ndarray, dict]:
    """Load the EMA weights, run the reverse chain, decode to raw (n, 3, 78).

    Samples from the EMA shadow, not the live weights — that is what `train.py`
    checkpoints under the `"ema"` key and what the diffusion literature samples
    from. Under SMOKE that choice is load-bearing in an unusual direction:
    `config.SMOKE` sets ema_decay = 0.9 precisely so that a ~160-step run leaves
    the shadow tracking the trained weights rather than its own initialisation.

    Determinism comes from an explicit labelled generator handed to `ddpm.sample`,
    like every other stream in the project — no global seeding. The body still runs
    inside `_global_rngs_restored()` so that constructing the UNet cannot perturb a
    caller's torch draws; see that context manager for what is left to contain.

    Sampling passes `cfg.x0_clamp`, the data-derived bound on the implied clean
    image. It is not a cosmetic guard: unclamped, this checkpoint's chain reaches
    image-space std ~690 against a real 0.333 (RESEARCH.md Phase 4 correction).
    Clamped it lands at ~4.3 — still bad, because four epochs is four epochs, but
    bad in a way the scorecard can measure rather than a saturation artefact.

    Also returns a diagnostic dict. An undertrained diffusion model does not fail
    by producing slightly-wrong series; it fails by producing pixels far outside
    the range the codec was fitted on, which the decoder then clamps. Without
    these numbers the DDPM row looks like a model with unusual opinions about
    volume instead of a model whose output is being bounded on 99% of pixels.
    """
    with _global_rngs_restored():
        model = UNet(
            in_channels=3,
            base_channels=cfg.base_channels,
            channel_mults=cfg.channel_mults,
            num_res_blocks=cfg.num_res_blocks,
            attention_at_bottleneck=cfg.attention_at_bottleneck,
            dropout=cfg.dropout,
        )
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        if ck.get("config_name") != cfg.name:
            raise SystemExit(
                f"{ckpt} was written by preset {ck.get('config_name')!r}, not {cfg.name!r}. "
                "Refusing to score it against a mismatched codec."
            )
        # strict=True on purpose: a checkpoint whose architecture has drifted from
        # the current code must fail loudly here. Silently retraining over it would
        # destroy another run's work, and loading it non-strictly would score a
        # partially-random network without saying so.
        model.load_state_dict(ck["ema"], strict=True)
        model.eval().to(device)

        sched = ddpm.DiffusionSchedule.build(cfg.timesteps, kind=cfg.beta_schedule).to(device)

        t0 = time.time()
        images = (
            ddpm.sample(
                model, (n, *cfg.image_shape), sched, device,
                x0_clamp=cfg.x0_clamp,
                generator=seeding.torch_generator("scorecard-ddpm-sample"),
            )
            .cpu().numpy().astype(np.float64)
        )
        elapsed = time.time() - t0

    # The decoder's input, before the day-transform is inverted: this is where
    # `winsor_sigma` clamping happens and where an out-of-distribution sample
    # stops being faithfully invertible.
    normed = image_to_series(
        invert_level_stats(images, codec.level_stats, codec.image_scale), BARS_PER_DAY
    )
    diag = {
        "epoch": ck["epoch"],
        "train_loss": ck["history"]["train_loss"][-1] if ck["history"]["train_loss"] else float("nan"),
        "seconds": elapsed,
        "image_std": float(images.std()),
        "saturated": codec.day_transform.saturated_report(normed),
    }
    return codec.decode(images), diag


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=None,
                    help="synthetic day-images per generator. Defaults to the size of the fit "
                         "split, so every row — the real_train noise floor included — carries "
                         "the SAME estimator noise. Several columns (acf_decay_abs_err, "
                         "vol_dispersion_rel_err) are noisy at 512 samples, and matching the "
                         "sample size is what keeps that noise from favouring one row")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--no-ddpm", action="store_true", help="classical baselines only")
    ap.add_argument("--device", default="cpu",
                    help="cpu by default; the reverse chain must be bitwise reproducible and "
                         "MPS/CUDA reductions are not guaranteed to be")
    ap.add_argument("--block", type=int, default=6, help="block_bootstrap block length in bars")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--timings", action="store_true",
                    help="print wall-clock timings. Off by default so stdout is byte-identical "
                         "between runs")
    args = ap.parse_args()

    cfg = get_config("SMOKE")
    prep = pipeline.prepare(cfg, verbose=False)
    fit_series = prep.train_series
    n = args.n if args.n else len(fit_series)

    # The full held-out split, not prep.val_series (SMOKE subsamples that to 128).
    _, val_meta = build_day_images().split_by_date(cfg.val_sessions)
    real = val_meta.series

    print(f"preset            {cfg.name}")
    print(f"fit split         {fit_series.shape[0]:,} day-images "
          f"(what the checkpoint trained on)")
    print(f"real reference    {real.shape[0]:,} held-out day-images over "
          f"{np.unique(val_meta.sessions).size} sessions")
    print(f"synthetic per row {n:,} (matched to the fit split, so the noise floor row is comparable)")

    t0 = time.time()
    garch_fit = classical.SeasonalGarchFit.fit(fit_series)
    print(f"\n{garch_fit.summary()}")
    if args.timings:
        print(f"  fitted in {time.time() - t0:.1f}s")

    gens: dict[str, np.ndarray] = {
        "iid_gaussian": classical.iid_gaussian(fit_series, n),
        "shuffled": classical.shuffled(fit_series, n),
        "block_bootstrap": classical.block_bootstrap(fit_series, n, block=args.block),
        "garch_seasonal": classical.garch_seasonal(fit_series, n, fit=garch_fit),
    }

    if not args.no_ddpm:
        if not args.checkpoint.exists():
            raise SystemExit(
                f"{args.checkpoint} not found. Train one first (notebook, or "
                f"train.train(...) with ckpt_dir=checkpoints/scorecard), or pass --no-ddpm."
            )
        series, diag = sample_ddpm(cfg, prep.codec, args.checkpoint, n, torch.device(args.device))
        gens["ddpm_wavelet"] = series
        sat = ", ".join(f"{k}={v:.1%}" for k, v in diag["saturated"].items())
        print(f"\nddpm_wavelet      {args.checkpoint.name} @ epoch {diag['epoch']}, "
              f"final train loss {diag['train_loss']:.4f}")
        if args.timings:
            print(f"  sampled {n} images in {diag['seconds']:.1f}s on {args.device}")
        print(f"  image-space std {diag['image_std']:.3g} vs real "
              f"{prep.train_images.std():.3g}  <- the diagnostic that explains the row")
        print(f"  decoder inputs hitting the +/-{cfg.winsor_sigma:g} sigma clamp: {sat}")

    # The noise floor. Listed last so it reads as the reference line under the
    # generators rather than as another competitor.
    gens["real_train"] = fit_series

    table = evaluate.scorecard(real, gens)
    with pd.option_context("display.width", 250, "display.max_columns", 32):
        print("\n" + "=" * 118)
        print("stylized-fact distances vs held-out real data (lower is better; "
              "real_train is the noise floor, not a target)")
        print("=" * 118)
        print(table.round(4).to_string())

    print("\nfitted, not learned — do not read these as wins:")
    print("  garch_seasonal  vol_dispersion_rel_err, intraday_vol_mse, intraday_volume_mse")
    print("                  (per-day sigma resampled; both profiles multiplied back in)")
    print("                  (acf_abs_l2 / acf_decay_abs_err are NO LONGER on this list —")
    print("                  see the note below)")
    print("  shuffled        every marginal: tail_*, kurtosis_rel_err — each day is a")
    print("                  permutation of a real day, so its multiset IS real")
    print("  block_bootstrap all cross-channel correlations, copied verbatim by cutting")
    print("                  blocks at identical offsets in the three channels; also the")
    print("                  pooled marginal (kurtosis_rel_err, tail_pooled_abs_err)")
    print("\nacf_abs_l2 / acf_decay_abs_err are now DESEASONALISED — read them accordingly:")
    print("  every series, real and synthetic, has its |r| divided by the SAME real")
    print("  bar-of-day profile before the ACF. Previously acf_within_day de-meaned each")
    print("  day by a scalar and left the U-shape in, so 83% of the real |r| ACF was")
    print("  seasonality (lag-1 0.244 raw vs 0.042 deseasonalised) and garch_seasonal was")
    print("  handed the column by multiplying its fitted profile back in. Now the profile")
    print("  earns no reward — it only avoids a penalty: a generator with the WRONG")
    print("  U-shape has an inverse-U injected into its |r|, which reads as slow, strong")
    print("  spurious persistence. That is why iid_gaussian and block_bootstrap score")
    print("  ABOVE the real ACF here (block_bootstrap worst of all: it pastes opening")
    print("  blocks into midday slots) while shuffled, having no ordering at all, sits")
    print("  closer. The recursion's share of the shuffled-to-floor gap rises from ~5%")
    print("  to ~25% — the column now measures clustering rather than the session shape.")
    print("\ncolumns that do not rank anything — see RESEARCH.md Phase 5:")
    print("  acf_decay_abs_err  NaN whenever there is no positive ACF to fit a power law to")
    print("  acf_raw_l2         scores iid Gaussian noise the same as real data")
    print("  kurtosis_rel_err   spans 0.10-1.52 across seeds at this sample size")
    print("\nthe real_train floor is itself a single noisy draw. Bootstrapping the fit split")
    print("20 times gives e.g. kurtosis_rel_err 0.06-0.42 and tail_within_abs_err 0.17-0.42,")
    print("so a below-floor cell means EITHER the quantity was supplied OR the gap is inside")
    print("that band. Deciding which takes an argument about construction, not two numbers.")

    if args.csv:
        table.to_csv(args.csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
