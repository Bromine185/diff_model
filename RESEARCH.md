# Research log

Phase-by-phase findings. Null results and killed hypotheses are recorded here
too — a log that only contains wins is a marketing document.

---

## Phase 1 — Data acquisition

**Fixture built.** 380 tickers × 60 sessions = **21,934 day-images** of shape
(3, 78), spanning 2026-06-03 → 2026-08-27. 25.7 MB as zstd parquet, committed.

**20 tickers dropped** as delisted or acquired between the universe being written
and the fetch: BK, CHK, CMA, CTRA, DFS, EXAS, FI, HES, HOLX, IPG, K, MRO, PARA,
PXD, SEE, SWN, WMB, WRK, X, and one more. `fetch_bars.py` reports these rather
than failing, and the manifest records them.

**No half-days in the window.** Every one of the 60 sessions had exactly 78 bars
for surviving tickers, so the "drop incomplete sessions" branch never fired. It
stays in the code because a 60-day window that includes Thanksgiving or July 3rd
will need it.

**Confirmed limitation, not fixable within yfinance.** `interval="5m"` is capped
at `period="60d"`. There is no parameter that extends it. The fixture therefore
covers **one macro regime** (June–August 2026). Any claim about regime-robustness
is unsupported by this dataset.

---

## Phase 2 — Stylized facts on real data

Measured on all 21,934 day-images before any modelling, so we know what the
generators are being asked to reproduce.

| Fact | Value | Verdict |
|---|---|---|
| ACF of \|r\|, lag 1 | 0.208, decaying to 0.08 by lag 8 | textbook volatility clustering |
| ACF of raw r, lag 1 | −0.034, then ≈0 | correctly near-zero; the small negative is bid-ask bounce |
| Intraday \|r\| profile | 4.45× at open, 0.70× midday, 1.48× at close | textbook U |
| Intraday volume profile | 4.75× at open, 0.65× midday, **6.55× final bar** | the closing auction, clearly visible |
| Excess kurtosis | 34.4 | fat |
| Hill tail index α | 2.43 pooled | fat, but see below |

### Finding: a third of the "fat tails" is cross-sectional, not within-series

Pooling 380 tickers of differing volatility is itself a fat-tail generator — a
mixture of Gaussians with dispersed σ is leptokurtic even when every component is
thin-tailed.

| | Hill α | excess kurtosis |
|---|---|---|
| Pooled, raw | 2.43 | 34.4 |
| Per-ticker median | 2.73 (10th–90th: 2.47–3.10) | — |
| Pooled after per-ticker vol-standardisation | **2.73** | **19.9** |

Vol-standardising recovers the per-ticker median *exactly*. So "fat tails" is
two independent claims a generator can get right or wrong separately: the
**within-day tail shape**, and the **cross-sectional dispersion of volatility**.

**Action taken:** the scorecard now scores both — `tail_within_abs_err` (per-day
vol-standardised) and `vol_dispersion_rel_err` — instead of one pooled number
that would let a model hide a failure in one behind a success in the other.

α ≈ 2.7 remains below the 3–5 usually quoted for equities. That is expected: the
tail index falls as sampling frequency rises, and these are 5-minute bars.

### Null result: no leverage effect at this frequency

corr(r_t, |r_{t+k}|) for k = 1..5:

    raw               [ 0.0029  0.0019  0.0016 -0.0043 -0.0004]
    vol-standardised  [-0.0024  0.0006 -0.0002 -0.0030 -0.0035]

Essentially zero either way. The leverage effect is a daily-and-longer
phenomenon and is genuinely absent *within* a 5-minute session. This is a
property of the data, not a defect in the estimator.

**Action taken:** leverage is **excluded from the scorecard**. Every generator,
good or bad, would score ≈0 against a real value of ≈0, and the column would
reward nothing. It is still measured and plotted as a diagnostic so the finding
stays visible rather than being quietly dropped.

### Caveat: the |r|↔spread correlation is partly definitional

corr(|r|, spread proxy) = **0.815**, which looks like a triumphant "the model
learned that big moves come with wide spreads". It is not, entirely: our spread
proxy is the high-low range, and both it and |close-to-close return| are
within-bar dispersion measures. Some of that 0.815 is arithmetic, not
microstructure.

The honest reading: this correlation is worth *matching* but is weak evidence of
learned market structure. `corr(|r|, volume) = 0.197` and
`corr(spread, volume) = 0.278` are the more meaningful cross-channel targets,
since volume is measured independently of price.

### Open question: aggregational gaussianity is not monotone

Excess kurtosis by aggregation horizon: 34.4 → 34.7 → 42.7 → 34.1 → 19.7 (h = 1,
2, 4, 8, 16 bars). Theory says this should decay monotonically toward 0. It does
not, and the h=4 spike is unexplained. Candidate causes: cross-ticker pooling
again, or the within-day-only aggregation ceiling of 78 bars leaving too few
independent blocks at large h. **Not yet resolved.**

---

## Phase 3 — Wavelet layer

Haar MODWT implemented from scratch and verified:

- Round-trip `imodwt(modwt(x))` exact to **< 1e-10** at every level count tested.
- Additive decomposition: planes sum to the original signal.
- Shift-equivariance verified (the property a decimated DWT lacks, and the reason
  the coefficient-plane image is well-defined at all).
- Cross-checked against PyWavelets `swt` to **~2e-16**, after reconciling three
  convention differences that are documented in the test rather than absorbed
  into a loose tolerance: pywt's Haar `dec_hi` is sign-flipped relative to ours,
  its level-j coefficients carry a factor 2^(j/2) under `norm=False`, and its
  circular alignment lags ours by 2^j − 1 samples.

25/25 wavelet tests pass.

**Geometry confirmed:** 78 bars → mirror-expand to 128 = 2⁷ → 7 detail levels +
1 smooth = **8 × 128 × 3** image. One octave below the paper's 16 × 256 × 3.

---

## Phase 4 — Training

The `SMOKE` checkpoint at `checkpoints/smoke_latest.pt` was written elsewhere in
the project, not by this phase; it was scored, not retrained. What is recorded
here is what the checkpoint *contains* and what its samples actually *do*, because
"training completed without crashing" and "training produced a usable generator"
are different claims and only the first one holds. Everything below is read back
from the checkpoint file or measured from it directly — no training run was
launched to produce these numbers.

**Run recovered from the checkpoint.** `SMOKE`, 512 training day-images, batch 16,
`drop_last` → 32 optimiser steps per epoch, **160 steps total**.

| epoch | train loss | val loss (EMA) | seconds |
|---|---|---|---|
| 1 | 0.9548 | 0.9221 | 2.0 |
| 2 | 0.8223 | 0.7832 | 1.5 |
| 3 | 0.6854 | 0.6402 | 1.5 |
| 4 | 0.5581 | 0.5146 | 1.5 |
| 5 | 0.4494 | 0.4125 | 2.9 |

Monotone decrease, no NaN, no divergence. The objective is MSE against the true
noise, so **1.0 is the score for predicting zero noise** — the run has moved from
"knows nothing" to "explains ~55% of the noise variance" and stopped there.

**Do not read val < train as absence of overfitting.** They are not comparable:
`train.evaluate_loss` scores the *EMA shadow* at a fixed timestep seed, while the
train figure is a running average over the epoch taken on the *live* weights as
they were improving. Val below train is the expected artefact of that asymmetry
mid-run, not evidence about generalisation.

**EMA is correctly configured for a run this short.** `SMOKE` sets
`ema_decay = 0.9`, so after 160 steps the shadow retains 0.9^160 ≈ 4e-8 of its
initialisation. `train()`'s warning (which fires above 20% retained) correctly
does not trigger. At `FULL`'s 0.999 it would have retained 85% and every sample
would have come from a near-random network with no error raised.

### Null result: SMOKE samples do not decode to usable series, and that is expected

> **Superseded in part — read the correction two sections below before quoting any
> number here.** Everything in this section was measured with a sampler that had a
> genuine defect: no clamp on the implied clean image, and a cosine β clip of
> 0.9999 that allowed a 100× gain on the first reverse step. The *conclusion*
> (SMOKE samples are not usable, by design) survives; the *magnitude* was ~160×
> larger than the preset alone accounts for. The section is kept as written
> because the diagnostics in it are what located the bug.

| | image-space std |
|---|---|
| real training images | 0.333 |
| sampled from EMA weights | **1927.3** |
| sampled from live weights | 1853.4 |

Both measured the way `scripts/run_scorecard.py` measures them — 512 samples,
`seeding.seed_everything("scorecard-ddpm-sample")` immediately before the chain,
on CPU — because the figure moves with the draw and quoting it without the sample
size makes it unreproducible. n-dependence is mild at this scale (1926.1 at
n = 128, 1928.4 at n = 256, 1927.3 at n = 512), which is why the script prints only
three significant figures.

A factor of ~5800. Downstream, 99.8% of returns, 99.7% of spreads and 99.5% of
volumes hit the codec's ±10σ clamp, so decoded returns are pinned at ±0.0547 and
`tail_pooled_abs_err` comes out **NaN** — the top 5% of |r| are all the identical
clamp value, so the Hill estimator's mean log-ratio is exactly 0.

This is not a bug in the sampler, the codec, or the EMA. Three checks:

1. **The amplification is arithmetic.** A model predicting zero noise leaves
   `mu = x / sqrt(alpha_t)` at every step, so the chain's gain is
   `prod 1/sqrt(alpha_t) = 1/sqrt(alphabar_T)`. At T = 50 cosine,
   alphabar_T = 9.7e-08, giving **3209×** applied to a unit-variance start, plus
   the injected noise amplified by whatever is left of the chain. Observed 1927
   means the model *is* cancelling some of it — just nowhere near all.
2. **It is not the EMA.** Sampling from the live weights gives 1853, the same
   failure. The shadow is tracking, exactly as the decay setting intends.
3. **The model has not learned the easy end of the chain.** Noise-prediction MSE
   measured on held-out images, by timestep:

   | t | alphabar_t | eps-MSE |
   |---|---|---|
   | 0 | 0.9983 | 0.832 |
   | 10 | 0.8791 | 0.443 |
   | 20 | 0.6174 | 0.394 |
   | 30 | 0.3116 | 0.374 |
   | 40 | 0.0767 | 0.369 |
   | 49 | 9.7e-08 | 0.369 |

   At t = 49 the input is essentially pure noise, so the correct output is very
   nearly "return your input" — a converged model scores ≈0 there. This one
   scores 0.369. The residual it leaves at the top of the chain is precisely what
   the 3209× gain then multiplies.

**Verdict: expected, and deliberately not tuned away.** `SMOKE` exists to prove
the code path is correct — shapes, alphabar indexing, EMA, checkpoint layout,
codec inversion — not to produce good samples, and CLAUDE.md says so up front.
160 optimiser steps is **179× short of `FULL`**, which is 18,311 training images
at batch 64 for 100 epochs = 28,600 steps — and `FULL` is itself only the
project's target, not a proof of sufficiency. Real training is a Colab job. The useful output of this
phase is the *diagnostic*: image-space std against the real 0.333, and the
saturation fractions, both printed by `scripts/run_scorecard.py`. Without them
the DDPM scorecard row reads as a model with strange opinions about volume rather
than a model whose every pixel is being clamped.

**Gotcha worth knowing.** The checkpoint that produced the numbers above recorded
`epoch: 5` while `config.SMOKE` declares `epochs=4`, so it had been written under
an override. A later `train(..., resume=True)` against such a file resumes at
epoch 5, iterates `range(5, 4)` — which is empty — and returns an untouched model
without saying anything. Nothing is corrupted, but "training did nothing" and
"training succeeded" look identical from the outside. The current checkpoint was
retrained from scratch at `epochs=4` (below), so it records `epoch: 4`; a resume
against it is still a no-op, but now correctly so — the run is finished.

### Correction: the sampler was blowing up, and it was not only the preset

The section above attributes an image-space std of **1927** to an undertrained
`SMOKE` model amplified by the schedule. Two of the three claims hold. The third
— that this is entirely a property of the preset — was wrong, and it hid a real
defect in `ddpm.p_sample_step`.

**What was broken.** Two things, one of which is a bug in anyone's book:

1. **No clamp on the implied clean image.** The reverse step ran the raw
   ε-parameterised update. Every reference DDPM implementation carries a
   `clip_denoised` step — reconstruct
   `x0_hat = (x_t − sqrt(1−ᾱ_t)·ε̂) / sqrt(ᾱ_t)`, clamp it to the data range, and
   only then form the posterior mean — and this one did not. There is no
   principled bound to put on the ε-form mean directly (ε is legitimately
   *N*(0, I)); the bound exists on `x0_hat`, which is a statement about clean
   data. Without it, nothing stops a merely-inaccurate ε predictor's error from
   compounding through 50 steps.
2. **A cosine β clip of 0.9999.** Nichol & Dhariwal clip at 0.999. At *T* = 50
   the cosine schedule saturates the clip, so 0.9999 admitted `alpha_t = 1e-4` and
   a **100×** state gain on the very first reverse step — before the model's error
   had a chance to be anything. Traced: std 1.0 → 60.4 after the single step from
   *t* = 49, → 1922 at *t* = 0.

**Fixes tested, on the pre-fix checkpoint, before choosing.** 8×(3, 8, 128) samples:

| change | resulting image-space std |
|---|---|
| none (as shipped) | 1922 |
| β clip 0.9999 → 0.999 | 599 |
| β clip → 0.99 | 182 |
| β clip → 0.95 | 79 |
| **x0_hat clamp at ±4** | 2.50 |
| **x0_hat clamp at ±1** | 0.644 |

The clip alone is not a fix — three orders of magnitude too large is still three
orders. **The clamp is the fix**; the clip change is literature alignment and
caps the first-step gain at 31.6 rather than 100. Both shipped.

**Where the clamp bound comes from.** Not tuned: measured. The fit split's images
span ±6.35 (per-channel max |x| = 4.33, 5.99, 6.35; 99.99% of |x| within ±3.50),
so `config.x0_clamp = 6.5` admits every observed pixel with margin. It is set on
`FULL` too, where it cannot bind the same way — linear *T* = 1000 has a max
`1/sqrt(alpha_t)` of 1.010 — because a safety net that never fires costs nothing.

**Retrained, because the schedule changed.** The β clip is part of the *forward*
process, so the old checkpoint was trained against a schedule that no longer
exists. SMOKE retrained from scratch, 4 epochs, same preset:

| epoch | train loss | val loss (EMA) |
|---|---|---|
| 1 | 0.9536 | 0.9179 |
| 2 | 0.8221 | 0.7875 |
| 3 | 0.6864 | 0.6437 |
| 4 | 0.5569 | 0.5162 |

**After.** 512 samples on CPU, the same measurement as the table above:

| | image-space std |
|---|---|
| real training images | 0.333 |
| EMA weights, `x0_clamp = 6.5` | **4.32** |
| live weights, `x0_clamp = 6.5` | 4.24 |
| EMA weights, clamp disabled | 688 |
| live weights, clamp disabled | 662 |

A factor of ~13 too wide, against ~5800 before. Saturation at the codec's ±10σ
clamp falls from 99.8% / 99.7% / 99.5% (returns / spread / volume) to **19.3% /
1.5% / 0.0%**. The scorecard's `tail_within_abs_err` for `ddpm_wavelet` falls from
**250.2 to 19.9** against a floor of 0.27.

Note the two clamp-disabled rows: with the 0.999 clip in place and a model trained
under it, the unclamped chain still reaches 688. The blow-up was never one thing.

**The null result stands, and is now measurable.** The model is still bad — 160
optimiser steps against `FULL`'s 28,600 — and the per-timestep ε-MSE says why, on
held-out images:

| t | ᾱ_t | ε-MSE |
|---|---|---|
| 0 | 0.9983 | 0.883 |
| 10 | 0.8791 | 0.547 |
| 20 | 0.6174 | 0.501 |
| 30 | 0.3116 | 0.478 |
| 40 | 0.0767 | 0.478 |
| 49 | 9.7e-07 | 0.474 |

At *t* = 49 the input is essentially pure noise and the correct output is very
nearly "return your input"; a converged model scores ≈0 and this one scores 0.474.
The chain's total gain `1/sqrt(ᾱ_T)` is 1015 (3209 before, the ᾱ_T change being the
clip's doing). What the clamp buys is that the residual now fails **bounded** — the
final reverse step returns the clamped `x0_hat` exactly, so |pixel| ≤ 6.5 is a hard
ceiling — instead of running away. A saturated sample scores well on any column
that rewards a flat line, so this is not a cosmetic difference: it is the
difference between a scorecard row that measures the model and one that measures
the winsoriser.

**Regression tests**, in `tests/test_ddpm.py`: the x0-form step is asserted
identical to the ε-form when the clamp is disabled (so the rewrite is provably
algebra, not a change of method); the clamp is asserted to actually bind; and
sampling the checkpoint is asserted to stay under 30× the real std, a bar the
broken sampler missed by two orders of magnitude.

**Also fixed in passing.** `ddpm.sample` / `p_sample_step` now take a
`generator=` argument, so the reverse chain is reproducible from a labelled
stream (CLAUDE.md non-negotiable #4) instead of by seeding process-global RNG
state. `scripts/run_scorecard.py` and notebook 01 both routed around its absence;
neither has to now. A CPU `torch.Generator` cannot drive `randn` on an MPS/CUDA
tensor, so with a generator the draw happens on the generator's device and is
moved — determinism is the point of that path.

---

## Phase 5 — Classical baselines

Four non-neural generators, in `src/diffmodel/baselines/classical.py`. They exist
to make the scorecard readable: a table of distances means nothing until you know
what a bad number looks like, and it means less than nothing if a column happens
to rate white noise as competitive.

| generator | keeps | destroys |
|---|---|---|
| `iid_gaussian` | per-channel mean and variance | everything else |
| `shuffled` | each day's marginal, exactly | all ordering, all *within-day* cross-channel alignment |
| `block_bootstrap` | 30 minutes of local structure, cross-channel alignment | long-range structure, bar-of-day identity, exact per-day marginals |
| `garch_seasonal` | fitted GARCH(1,1)-t clustering, both intraday profiles, per-day vol distribution | — |

### Setup, and why these particular splits

Three splits are in play and conflating them would rig the table:

- **Fit split** — the **512** day-images the `SMOKE` checkpoint actually trained
  on. Every classical baseline is fitted on exactly these. Handing GARCH all
  20,856 training days while the DDPM saw 512 would make the comparison about
  sample size rather than method.
- **Codec** — the one `prepare(get_config("SMOKE"))` returns, fitted on those same
  512. It is not interchangeable with a `FULL`-fitted codec; decoding SMOKE
  samples with the wrong one applies the inverse of a transform the model never
  saw.
- **Real reference** — the **full** held-out split, all 1,078 day-images from the
  last 3 sessions, *not* SMOKE's 128-image subsample of it. Nobody fits to the
  reference, so a bigger one is strictly better, and it mattered: the real-vs-real
  noise floor for `cross_corr_frobenius` is 0.414 against the 128-image subsample
  and **0.107** against the full split. Four-fifths of that residual was the
  reference being too small to pin down corr(|r|, volume).

Every row draws 512 synthetic day-images, matched to the fit split so the noise
floor row carries the same estimator noise as the generators.

### The scorecard

`python scripts/run_scorecard.py`. Lower is better. **`real_train` is the noise
floor, not a target** — it is real training days scored as if they were a
generator, so it is what "as good as it is possible to be" looks like given
finite samples and a genuine train/reference shift. Read every row against it,
never against zero.

**Current table** — with the sampler fixed (Phase 4 correction) and `acf_abs_l2` /
`acf_decay` deseasonalised (resolution section below). The two ACF columns are not
comparable to any earlier draft of this file; every other column is.

| | tail_within | vol_disp | tail_pooled | kurtosis | acf_abs_l2 | acf_decay | intraday_vol | intraday_volume | cross_corr | acf_raw_l2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `iid_gaussian` | 3.4527 | 0.8143 | 3.7046 | 1.0005 | 0.0501 | 1.0646 | 0.4466 | 0.7103 | 1.2227 | 0.0151 |
| `shuffled` | 0.2338 | 0.1155 | 0.1299 | 0.0639 | 0.0253 | 1.1923 | 0.4571 | 0.7160 | 0.9229 | 0.0163 |
| `block_bootstrap` | 0.5482 | 0.2594 | 0.1325 | 0.1177 | 0.0486 | 0.4160 | 0.4467 | 0.7042 | **0.1238** | 0.0128 |
| `garch_seasonal` | 0.1474 | 0.1094 | 0.0788 | 0.4334 | **0.0076** | NaN | 0.0175 | 0.0343 | 0.4159 | 0.0145 |
| `ddpm_wavelet` (SMOKE) | 19.8972 | 0.8448 | NaN | 1.0147 | 0.0445 | 1.0773 | 0.4334 | 2.2473 | 1.1953 | 0.0465 |
| *`real_train` (floor)* | *0.2681* | *0.0523* | *0.1481* | *0.2582* | *0.0078* | *1.0275* | *0.0179* | *0.0252* | *0.1070* | *0.0140* |

Two rows moved for reasons that are findings rather than noise:

- **`ddpm_wavelet` on `tail_within`, 250.2 → 19.9.** The old row was the broken
  sampler: 99.8% of decoded returns pinned at the ±10σ clamp, i.e. a square wave.
  Still 74× the floor — four epochs is four epochs — but now a measurement of the
  model.
- **`ddpm_wavelet` on `intraday_volume`, 0.714 → 2.247, i.e. WORSE.** This is
  what fixing a metric artefact looks like from the wrong side. The old row was a
  fully-saturated constant, and a constant has *no* intraday profile to get wrong;
  it scored the same 0.71 that `shuffled` and `iid_gaussian` score for the flat
  profile they emit. The unsaturated model emits a volume profile that is actively
  wrong, which is a larger distance and a more honest one. `tail_pooled` stays NaN:
  19.3% of returns still clamp, so the Hill estimator's top 5% is still one
  repeated value.

**The pre-fix table**, kept as the record of what the broken sampler and the raw
ACF metric scored:

| | tail_within | acf_abs_l2 | acf_decay | intraday_volume |
|---|---|---|---|---|
| `iid_gaussian` | 3.4527 | 0.0898 | NaN | 0.7103 |
| `shuffled` | 0.2338 | 0.0907 | NaN | 0.7160 |
| `block_bootstrap` | 0.5482 | 0.0696 | NaN | 0.7042 |
| `garch_seasonal` | 0.1474 | **0.0110** | 0.4331 | 0.0343 |
| `ddpm_wavelet` (SMOKE) | **250.2089** | 0.1124 | NaN | 0.7143 |
| *`real_train` (floor)* | *0.2681* | *0.0093* | *0.1130* | *0.0252* |

Fitted GARCH parameters, for the record:

    GARCH(1,1)-t  omega=0.4158  alpha=0.0580  beta=0.5309  (persistence 0.5888)  nu=11.57
    spread  log(s/prof) = -4.262 + 0.258 log(|r|/prof) + N(0, 0.508^2)
    volume  log(v/prof) = level + 0.061 log|z|          + N(0, 0.494^2)
    fitted on 256 of 512 day-images (GARCH likelihood); 512 for the profiles,
    day-sigma pool and regressions

Two sample sizes, deliberately. The GARCH likelihood is the expensive step and
runs on a `max_fit_days = 256` subsample; everything else uses all 512. An earlier
version of `SeasonalGarchFit.summary()` printed a single "fitted on 512
day-images" directly under the parameter line, which attributed the larger sample
to the one number a reader is most likely to quote.

### What is FITTED and what is EARNED

The single most misreadable thing about this table. Several cells are handed to
the generator by construction, and a reader who forgets that will mistake a
lookup for a discovery:

| generator | handed to it | why |
|---|---|---|
| `garch_seasonal` | `vol_dispersion`, `intraday_vol`, `intraday_volume` | per-day sigma is resampled from the empirical distribution and each simulated day is renormalised to hit it (in expectation — see below); both intraday profiles are multiplied back in at simulation time |
| ~~`garch_seasonal`~~ | ~~**`acf_abs_l2`, `acf_decay`**~~ | ~~the same \|r\| profile, counted a second time: `acf_within_day` does not deseasonalise, so most of the ACF it measures is the U-shape~~ — **no longer true**: the metric now divides both series by the real profile first, so the fitted profile buys `garch_seasonal` no clustering score. See the resolution section below. This row sat in the EARNED column, was moved here, and has now moved back for a different reason than it started with |
| `garch_seasonal` | the \|r\|–spread corner of `cross_corr` | spread is a regression *on* \|r\| |
| `shuffled` | every marginal: `tail_*`, `kurtosis`, `vol_dispersion` | each day is a permutation of a real day, so its multiset is real |
| `block_bootstrap` | all cross-channel correlations, and the *pooled* marginal (`kurtosis`, `tail_pooled`) | blocks are cut at identical offsets in all three channels, so contemporaneous alignment is copied verbatim; the values themselves are real values resampled |

#### The floor is a noisy floor, and a below-floor cell has two possible causes

`garch_seasonal` landing at 0.0175 on `intraday_vol` against a floor of 0.0179 —
i.e. **beating real data** — is the signature of a handed column. But the earlier
draft of this section drew a categorical rule from it ("nothing can genuinely beat
the floor; a cell below it means the quantity was supplied") and that rule is
wrong. **Nine cells of the table it annotates sit below the floor**, in three
generators, and for most of them "supplied" is not the explanation.

`real_train` is one fixed 512-day draw, so its distance from the reference is a
single realisation of a noisy estimator. Bootstrap-resampling those same 512 days
20 times gives the floor's own band:

| column | floor as shipped | min | median | max |
|---|---|---|---|---|
| `tail_within_abs_err` | 0.2681 | 0.1687 | 0.2732 | 0.4168 |
| `vol_dispersion_rel_err` | 0.0523 | 0.0126 | 0.0570 | 0.1286 |
| `tail_pooled_abs_err` | 0.1481 | 0.0049 | 0.1441 | 0.2685 |
| `kurtosis_rel_err` | 0.2582 | 0.0590 | 0.2748 | 0.4201 |
| `acf_abs_l2` | 0.0093 | 0.0080 | 0.0103 | 0.0119 |
| `acf_decay_abs_err` | 0.1130 | 0.0010 | 0.1040 | 0.1911 |
| `intraday_vol_mse` | 0.0179 | 0.0163 | 0.0212 | 0.0278 |
| `intraday_volume_mse` | 0.0252 | 0.0129 | 0.0319 | 0.0510 |
| `cross_corr_frobenius` | 0.1070 | 0.0515 | 0.0923 | 0.1870 |
| `acf_raw_l2` | 0.0140 | 0.0124 | 0.0145 | 0.0167 |

So the rule is conditional, not categorical: **a cell below the floor means either
the quantity was supplied, or the gap is smaller than the floor row's own
estimator noise.** All nine below-floor cells, checked against that band:

| cell | value | inside the floor's band? | handed by construction? |
|---|---|---|---|
| `shuffled` / `tail_within` | 0.2338 | yes | yes |
| `shuffled` / `tail_pooled` | 0.1299 | yes | yes |
| `shuffled` / `kurtosis` | 0.0639 | yes (just) | yes |
| `block_bootstrap` / `tail_pooled` | 0.1325 | yes | yes |
| `block_bootstrap` / `kurtosis` | 0.1177 | yes | yes |
| `block_bootstrap` / `acf_raw_l2` | 0.0128 | yes | n/a — the column ranks nothing |
| `garch_seasonal` / `intraday_vol` | 0.0175 | yes | yes |
| `garch_seasonal` / `tail_pooled` | 0.0788 | yes | no |
| `garch_seasonal` / `tail_within` | 0.1474 | **no** | no |

Eight of the nine are explained by the floor's own noise alone, and for five of
those "handed" would also explain it — meaning the number by itself decides
nothing and the construction argument does all the work. `kurtosis` is the sharpest
case: it is named at RESEARCH.md's own summary as a column nobody is handed, yet
`block_bootstrap` beats the floor on it purely because a bootstrap resample of the
same 512 days lands on either side of a single noisy draw.

The one cell that genuinely sits below the band, `garch_seasonal` on `tail_within`
(0.1474 vs a band of 0.169–0.417), is *not* handed — the innovations are simulated
Student-t, not resampled — and it is inside the generator's own 12-seed spread on
that column (0.131–0.307). Read as a good draw, not as evidence.

The practical consequence: `acf_abs_l2` is the only column where the floor's band
(0.0080–0.0119) is narrow enough that a below-floor cell would mean something, and
nothing is below it. **This matters for the `ddpm_wavelet` row once a `FULL`
checkpoint exists** — the rule has to be right before it is applied to the model
the project is about.

Two further caveats on the "handed" column, because the realised numbers do not
read as cleanly as the argument:

- `vol_dispersion` is handed to `garch_seasonal` and `shuffled` *in expectation*,
  yet both score worse than the floor (0.109 and 0.116 vs 0.052). Both resample
  512 per-day sigmas out of 512 with replacement, and the bootstrap noise in
  std(log sigma) is larger than the train/reference shift the floor row measures.
  The column is not evidence about either model in either direction. *In
  expectation* is the load-bearing phrase, and `garch_seasonal`'s per-day
  renormalisation does not upgrade it to *exactly*: after that line the simulated
  day's **deseasonalised** std is exactly the drawn `day_sigma` (agreement to
  1e-18), but `evaluate.vol_dispersion` scores the **raw** per-day std and the
  profile is multiplied in afterwards, so the raw std is `day_sigma` × a
  day-varying factor measured at 0.89–1.48, mean 1.09.
- `block_bootstrap` does **not** preserve each day's marginal the way `shuffled`
  does. It draws 13 blocks with replacement from one day, so values repeat and
  drop out and the per-day multiset is a bootstrap of the real one, not a copy.
  That is why its `tail_within` (0.548) is more than twice `shuffled`'s (0.234)
  and its `vol_dispersion` (0.259) is the worst of the resampling baselines.

The columns `garch_seasonal` is not handed are **`tail_within`**, **`kurtosis`**,
and corr(\|r\|, volume) / corr(spread, volume). That list is shorter than the
earlier draft claimed, and the correction is the next section.

**The statement has to be scoped to one generator, because there is no column
nobody is handed.** An earlier version of this sentence read "the columns nobody
is handed are `acf_abs_l2`, `acf_decay`, `kurtosis`, and corr(\|r\|, volume) /
corr(spread, volume)", two rows below a table that hands `kurtosis` to `shuffled`
and `block_bootstrap` (whose pooled marginals are respectively a copy and a
bootstrap of the real one) and hands every cross-channel correlation to
`block_bootstrap` (blocks cut at identical offsets in all three channels). That
was a direct contradiction with the table it summarised.

### Correction: `acf_abs_l2` is mostly the intraday profile, not clustering

The previous draft of this phase called `acf_abs_l2` and `acf_decay` the columns
`garch_seasonal` **earns** from its GARCH recursion, and "the bar the diffusion
model has to clear". Measured, that was wrong, and wrong in the direction that
flatters the baseline. About 90% of `garch_seasonal`'s score on those two columns
comes from the fitted intraday |r| profile — the same quantity already flagged as
handed for `intraday_vol_mse`.

**Why.** `evaluate.acf_within_day` de-means each 78-bar day by that day's *scalar*
mean and never removes the bar-of-day profile, so the deterministic U-shape sits
inside every |r| series it measures. On the real reference split:

| \|r\| ACF, lag | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| as measured | 0.235 | 0.181 | 0.153 | 0.128 | 0.130 | 0.120 | 0.088 | 0.069 |
| after dividing \|r\| by the profile | 0.038 | 0.010 | 0.007 | 0.002 | 0.002 | −0.001 | −0.012 | −0.004 |

Sixteen percent of the real lag-1 ACF survives deseasonalisation. The RMS over 30
lags — which is exactly what `acf_abs_l2` compares — falls from 0.0822 to 0.0126,
so 84% of the quantity is seasonal. The profile's own ACF, read as a one-day
series, is 0.67 at lag 1 and still 0.38 at lag 6.

**The ablation.** `garch_seasonal`'s return channel, toggling the GARCH recursion
(replaced by iid Student-t with the same ν) and the fitted profile independently,
6 seeds each, scored against the same held-out reference (real `acf_decay` =
1.1530):

| variant | `acf_abs_l2`, mean [min, max] | `acf_decay_abs_err` |
|---|---|---|
| GARCH + fitted profile (shipped) | 0.0122 [0.0105, 0.0135] | finite, 6/6 |
| iid Student-t + fitted profile | 0.0162 [0.0142, 0.0183] | finite, 6/6 |
| GARCH, profile removed | 0.0829 [0.0823, 0.0834] | **NaN, 6/6** |
| iid Student-t, neither | 0.0897 [0.0884, 0.0909] | **NaN, 6/6** |

Against that bottom row as the zero point and the floor (0.0093) as one: the
profile alone covers **91%** of the distance, and the recursion adds **5 points**
on top of it. The recursion alone covers **9%** and cannot produce a finite
`acf_decay` at all, because a deseasonalised GARCH path has no positive ACF
surviving `acf_decay_exponent`'s filter.

**Two readings elsewhere in this document are confounded by the same thing.**
"The block bootstrap severs the ACF at exactly lag 5 = block − 1, as designed" is
part of the picture but not all of it: `block_bootstrap` also destroys bar-of-day
identity, so it loses the seasonal component that is 84% of the measured ACF. Its
0.0696 against `shuffled`'s 0.0907 is *both* effects at once and the table cannot
separate them. And "GARCH is a real bar, not a straw man" is still true — it is
the only generator here that models volatility persistence — but `acf_abs_l2` is
not the evidence for it.

**What would be evidence.** `acf_within_day(|r| / intraday_profile)` isolates
clustering: real lag-1 falls 0.235 → 0.038, and *that* 0.038 is a number a
generator would have to earn. That is now what the metric does — next section.

### Resolution: the metric now removes the profile

`evaluate.acf_abs_returns` gained a `deseasonalize` path (default on): each day's
|r| is divided by the bar-of-day |r| profile before `acf_within_day`. The raw
variant stays reachable as `deseasonalize=False`, so the ablation above is still
reproducible.

**The profile is the REAL one, for every row.** `StylizedFacts.measure` takes
`deseasonal_profile` and `scorecard` passes the reference's profile to real and
synthetic alike. Letting each generator divide out its *own* profile would hand it
its own seasonality — the same leak class as fitting a baseline on the data it is
scored against — and a generator with a badly wrong U-shape would have that
wrongness cancelled out of the column instead of scored by it.

That choice is what makes the new column behave, and it inverts two readings:

| \|r\| ACF by lag, deseasonalised | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| real (val) | 0.038 | 0.010 | 0.007 | 0.002 | 0.002 | −0.001 | −0.012 | −0.005 |
| `garch_seasonal` | 0.041 | 0.016 | 0.001 | 0.004 | −0.012 | −0.007 | −0.005 | −0.014 |
| `real_train` (floor) | 0.045 | 0.018 | 0.011 | 0.019 | 0.010 | 0.013 | 0.004 | −0.009 |
| `shuffled` | 0.057 | 0.052 | 0.036 | 0.041 | 0.044 | 0.043 | 0.033 | 0.022 |
| `iid_gaussian` | 0.108 | 0.103 | 0.094 | 0.081 | 0.078 | 0.080 | 0.070 | 0.050 |
| `block_bootstrap` | 0.206 | 0.140 | 0.106 | 0.079 | 0.053 | 0.027 | 0.026 | 0.022 |
| `profile_only` (ablation) | −0.004 | −0.009 | −0.013 | −0.004 | −0.014 | −0.007 | −0.004 | −0.012 |

1. **A wrong U-shape now costs you.** `iid_gaussian` and `shuffled` emit a flat
   profile, so dividing by the real one injects an *inverse* U into their |r| —
   which reads as strong, slowly-decaying persistence. They score above the real
   ACF at every lag, and `acf_decay_abs_err` is finite for them where it used to
   be NaN. The blind spot moved: what returns NaN now is `profile_only`, the
   generator with the right profile and no clustering, which is the correct
   answer for it.
2. **`block_bootstrap` is now the worst row, not the middle one.** Blocks start at
   uniform random bars, so an opening block lands in a midday slot and the real
   profile scales that whole run up together: manufactured persistence, 0.206 at
   lag 1 against a real 0.038. On the raw metric this read as partial credit
   (0.0696, better than `shuffled`'s 0.0907) because the same U-shape sat in both
   series and cancelled. The column now penalises *fake* clustering as well as
   absent clustering, which is what a distance is for.

**The ablation, redone.** Same paired control (`alpha = beta = 0`), scorecard
split, `n` = 512:

| | `acf_abs_l2` | share of the `shuffled`→floor gap |
|---|---|---|
| `shuffled` (zero point) | 0.0253 | — |
| `profile_only` | 0.0117 | profile: **78%** |
| `garch_seasonal` (shipped) | 0.0076 | recursion: **24%** |
| *`real_train` (floor)* | *0.0078* | — |

The recursion's share rises from ~5 points to ~24. The profile term has not
vanished but its meaning has changed: it no longer *rewards* `garch_seasonal` with
clustering it did not model, it *spares* it the inverse-U penalty that a wrong
seasonal now incurs. Getting the U-shape right buys the absence of a penalty, not
a score — which is the right price for a quantity that was fitted.

`garch_seasonal` at 0.0076 against a floor of 0.0078 is now at the floor, and its
`acf_decay` goes NaN at this sample size: its deseasonalised ACF is real but
short-lived, and `acf_decay_exponent` needs five positive lags above 1e-6 to fit.
That column remains the unreliable one — see the null result below — and
deseasonalisation did not rescue it.

`tests/test_classical.py` pins all of this: the rewritten
`test_garch_recursion_earns_a_real_share_of_acf_abs_l2` asserts the bound in the
opposite direction to the version that flagged this change, and
`test_block_bootstrap_fakes_clustering_by_pasting_wrong_time_of_day_blocks` pins
the reversal with the mechanism written down.

### Verdicts

**The metrics discriminate.** `iid_gaussian` loses to the noise floor on every
column that works, most starkly on `tail_within` (3.45 vs 0.27, a 13× margin) and
`cross_corr` (1.22 vs 0.11). This is the whole reason the null model is in the
table — without it, no column can be shown to measure anything.

**`acf_abs_l2` sees past one lag.** RAW (pre-deseasonalisation) ACF of |r| by lag
— the picture that motivated the correction two sections up; the deseasonalised
version of this same table is in the resolution section, and it is the one the
current scorecard measures:

| lag | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| real (val) | 0.235 | 0.181 | 0.153 | 0.128 | 0.130 | 0.120 | 0.088 | 0.069 |
| `garch_seasonal` | 0.203 | 0.167 | 0.135 | 0.132 | 0.102 | 0.108 | 0.089 | 0.066 |
| `block_bootstrap` (block 6) | 0.149 | 0.083 | 0.054 | 0.027 | −0.000 | −0.022 | −0.015 | −0.011 |
| `shuffled` | −0.015 | −0.011 | −0.021 | −0.014 | −0.010 | −0.009 | −0.012 | −0.015 |

The block bootstrap severs the ACF at **exactly lag 5 = block − 1**, as designed,
and also *attenuates* what it keeps (0.149 at lag 1 against a real 0.235) because
averaging over random block alignments dilutes the within-block correlation. So
the middle case behaves like a middle case, which is what makes it a usable
control on the metric's range — with the caveat from the correction above: the
block bootstrap destroys bar-of-day identity too, so its ACF profile is the block
structure *and* the loss of the seasonal, and this picture does not separate them.

**GARCH is a real bar, not a straw man** — but not because of `acf_abs_l2`. It is
the only generator here that models volatility persistence at all, and it is
competitive on `tail_within` and `kurtosis` without being handed either. What it
was not, on the raw metric, the earner of `acf_abs_l2`: ~90% of its 0.0110 there
came from the fitted intraday profile. On the deseasonalised metric it earns a
real if smaller share — 24% of the `shuffled`→floor gap against ~5 points before —
and lands **at** the floor (0.0076 vs 0.0078). Its `acf_decay` is not recursion
evidence either way: the exponent was fittable because of the profile, and goes
NaN once the profile is divided out. The bar the diffusion model has to clear on
clustering is the deseasonalised lag-1 ACF, real 0.038 — which the scorecard now
does measure.

**The SMOKE diffusion model is the worst of all five rows on 5 of the 9
comparable columns.** Stated as counted, because "loses to every baseline on most
columns" is easy to inflate. One of the ten is NaN for the DDPM (`tail_pooled` —
19.3% of returns still clamp, so the Hill estimator's top 5% is one repeated
value) and so is not comparable; of the remaining nine it is strictly worst on
`tail_within` (19.90), `vol_disp` (0.8448), `kurtosis` (1.0147),
`intraday_volume` (2.2473) and `acf_raw_l2` (0.0465).

It is *not* worst on the other four, and each of those is a fact about the column
rather than a sign of life in the model. `intraday_vol_mse` 0.4334 sits just under
the three flat-profile baselines (iid 0.4466, block 0.4467, shuffled 0.4571):
emitting no U-shape at all is worth about that much, whatever produces it.
`cross_corr_frobenius` 1.1953 undercuts iid's 1.2227 by a hair — inside a percent,
which is the honest reading of "these columns stop resolving at the bad end".
`acf_decay_abs_err` 1.0773 is in a column that ranks nothing. And on the
deseasonalised `acf_abs_l2`, 0.0445 against `block_bootstrap`'s 0.0486 says only
that the block bootstrap manufactures more fake persistence than an undertrained
diffusion model does, not that the diffusion model has learned any.

All of this is with the fixed sampler. Under the broken one the same row read
`tail_within` 250.2 and `intraday_volume` 0.7143 — and that second number was the
*better-looking* one precisely because the output was a saturated square wave with
no profile to get wrong. Fixing the sampler made a column go up. See the notes
under the current table.

This is the expected outcome and it is not being framed as anything else. Phase 4
explains it completely: 160 optimiser steps
leaves the reverse chain's 3209× gain uncancelled, every decoded pixel hits the
codec clamp, and the resulting series is a square wave. `tail_within = 250` and
`tail_pooled = NaN` are what a fully saturated distribution looks like through a
Hill estimator, not a subtle modelling failure. **The scorecard is not evidence
about the paper's method until a `FULL` run exists.** The row is here to prove the
plumbing — checkpoint → EMA weights → sampler → codec → raw series → scorecard —
runs end to end and is deterministic, which it does and is.

### Null result: the prescribed spread regression does *not* reproduce corr(|r|, spread)

Phase 2 flagged corr(|r|, spread) = 0.815 as *partly definitional* and warned
against reading a match as learned microstructure. Fitting the obvious model —
`log s = a + b log|r|` on deseasonalised inputs — and simulating from it gives:

| | corr(\|r\|, spread) | corr(\|r\|, volume) | corr(spread, volume) |
|---|---|---|---|
| real (val) | 0.792 | 0.188 | 0.265 |
| real (train) | 0.825 | 0.157 | 0.205 |
| `garch_seasonal` | **0.560** | 0.112 | 0.102 |
| `block_bootstrap` | 0.838 | 0.150 | 0.201 |
| `shuffled` | 0.173 | 0.068 | 0.097 |

The regression **undershoots** the correlation it was fitted to. The fit is
genuinely weak in log space — slope 0.258, R² 0.30 — and OLS is further attenuated
by errors-in-variables, since log|r| is the *realised* move and carries
Var(log|N(0,1)|) ≈ 1.23 of pure noise around the latent bar volatility.

The interesting part is *why* the correlation is nonetheless 0.8 in the data. The
high-low range is a near-lower bound on the close-to-close move: **spread < |r| on
only 5.8% of real bars, with a median ratio of 2.0.** So the definitional link is
a *bound in raw space*, not a log-linear conditional mean, and a conditional-mean
model cannot express it. Phase 2's caveat stands and is now sharper: the 0.815 is
mostly a mechanical inequality, and any generator that reproduces it has
reproduced an inequality.

Correspondingly, `block_bootstrap` — which copies real bars and therefore inherits
the inequality for free — **wins `cross_corr_frobenius` outright (0.124), beating
GARCH (0.416) and sitting essentially at the noise floor (0.107)**. That column
rewards copying, not modelling. It must be read next to `acf_abs_l2`, where
`block_bootstrap` is 6× worse than GARCH.

### Null result: three scorecard columns do not work as columns

Found while building the baselines. Reported, not fixed — `evaluate.py` is owned
elsewhere.

1. **`acf_decay_abs_err` blanks out, and deseasonalising only moved which rows.**
   `acf_decay_exponent` fits over lags where the ACF is positive and above 1e-6,
   and needs five of them. The mean within-day sample ACF of an *uncorrelated*
   series is biased to about −1/(T−1) = −0.013 at every lag, because each 78-bar
   window is de-meaned by its own mean. Under the raw metric that hit exactly the
   generators the column should punish: `iid_gaussian` and `shuffled` had **0 of
   30 lags positive** and both returned NaN instead of a large distance. Under the
   deseasonalised metric those two score finite and badly (the injected inverse-U
   gives them plenty of positive lags) — and `garch_seasonal` now returns NaN in
   **12 of 12 draws**, because a real but short-lived ACF does not clear the
   five-lag filter either. The estimator's fragility is the constant; which rows
   it eats is not. Near the boundary it is worse than noisy, it is
   *discontinuous*: a generator sitting on four-or-five surviving lags lands on
   either side of the threshold depending on the draw.
2. **`acf_raw_l2` does not discriminate at all.** Across all five generators plus
   the real training data it spans 0.0128–0.0163. `evaluate.py` already says why
   in a comment — a generator can fake uncorrelated returns by emitting pure
   noise — and the measurement confirms it. Worth reporting (a generator with a
   strong *positive* return ACF has invented a trading strategy and should be
   caught) but it ranks nothing and must never be summed into a total.
3. **`kurtosis_rel_err` is too noisy to rank at n = 512.** Twelve draws of
   `garch_seasonal` under different seeds span **0.0995 to 1.5232**, median 0.501.
   Pooled kurtosis is a fourth moment dominated by a handful of extreme bars. The
   headline table's 0.4334 is one draw from that distribution, not a measurement
   of the model.

Both `test_classical.py::test_acf_decay_is_nan_when_there_is_no_positive_acf_to_fit`
and `::test_acf_raw_l2_does_not_discriminate` pin these so they stay visible.

### Sampling noise: which columns can be read finely

Twelve `garch_seasonal` draws at n = 512, differing only in seed label (the
default draw plus labels `noise-probe-1` … `noise-probe-11`), **re-measured under
the deseasonalised ACF**:

| column | default draw | min | median | max | finite |
|---|---|---|---|---|---|
| `acf_abs_l2` | 0.0076 | 0.0060 | 0.0067 | 0.0085 | 12/12 |
| `tail_within_abs_err` | 0.1474 | 0.0609 | 0.1949 | 0.3211 | 12/12 |
| `vol_dispersion_rel_err` | 0.1094 | 0.0014 | 0.0669 | 0.1422 | 12/12 |
| `acf_decay_abs_err` | NaN | — | — | — | **0/12** |
| `kurtosis_rel_err` | 0.4334 | 0.2565 | 0.4298 | 0.5736 | 12/12 |

Only `acf_abs_l2` is tight enough to read to two significant figures, and it stays
tight after deseasonalisation (a ±19% band around the median, against ±20% before)
while measuring a different and smaller quantity.

`acf_decay_abs_err` is not merely noisy for `garch_seasonal` now — it is **absent
in all twelve draws**. A deseasonalised GARCH path's ACF is real but short, and
`acf_decay_exponent` needs five positive lags above 1e-6 before it will fit
anything. Under the raw metric this column produced a finite number for
`garch_seasonal` in every draw and the default one happened to be the worst of the
twelve (0.4331); read either way, it is not a column that ranks generators.

The `kurtosis_rel_err` band above is narrower than the 0.0995–1.5232 recorded in
the null-result list below, which came from a different twelve labels. Both are
measurements of the same thing — a fourth moment at n = 512 — and the wider one is
the safer quote; neither supports reading that column to two figures.

### Deliberate choices worth challenging

- **Circular block bootstrap, not moving-block.** The day-image is a fixed
  78-column rectangle and everything wavelet-side depends on that. Non-circular
  blocks either leave ragged tails needing padding, or under-sample the last
  `block−1` bars because no block may start there — which would systematically
  thin out the closing auction, the most distinctive feature of the volume
  channel. The cost is real: wrapping splices 15:55 onto 09:30 inside a block,
  manufacturing exactly the fake session edge that `evaluate.acf_within_day`
  refuses to create.
- **Concatenating 78-bar days to fit GARCH is a fudge.** It manufactures a fake
  overnight transition every 78 observations, carrying one ticker's closing
  variance into another's open. At alpha = 0.058 the contamination is one
  observation in 78 at 6% weight — small, not zero. The honest alternative is a
  panel GARCH re-initialised per day, which is a different model; the point of
  this baseline is to be the standard one.
- **Each simulated GARCH day is renormalised to unit variance before the drawn
  per-day sigma is applied** — and the measured effect is much smaller than the
  reasoning predicted, which is worth recording. Measured **paired** (the same
  `fork(label)` stream feeding both variants, so the only difference is that one
  line), 6 seeds: std(log per-day sigma) is **0.443 with the line and 0.455
  without**, a mean inflation of **+2.8%**, range +0.8% to +4.1%, against a real
  0.445. Not the large effect expected from "two dispersions compounding",
  because the fitted persistence is only 0.589 and a 78-step path accumulates
  little day-level variance to add in quadrature.

  The pairing is not a nicety. An earlier unpaired 5-draw measurement reported
  here gave 0.477 vs 0.461 — a "7%" inflation — and that figure does not
  reproduce: across those 6 seeds the estimate has an sd of 0.017 against an
  effect of 0.012 (range 0.418–0.470), so an unpaired design of that size cannot
  resolve it and can return the wrong sign. The direction and the conclusion survive; the magnitude and its
  two significant figures did not, and the corrected cost is one extra call.

  Kept anyway, for consistency rather than size — but what it makes exact is
  narrower than the earlier draft claimed. After the line, the simulated day's
  **deseasonalised** std is exactly the drawn `day_sigma` (to 1e-18);
  `vol_dispersion_rel_err` scores the **raw** per-day std, and the profile goes in
  afterwards, so the raw std is `day_sigma` × a day-varying factor (0.89–1.48,
  mean 1.09). "Handed in expectation" is the correct phrase, with the line or
  without it.
- **`sample_ddpm` has to seed a global RNG, and now puts it back.** `ddpm.sample`
  draws with bare `torch.randn` / `torch.randn_like` and takes no `generator`
  argument, so pinning the global torch RNG is the only way to make the reverse
  chain reproducible — and seeding a global is precisely the thing non-negotiable
  #4 rules out ("re-running one component does not perturb another's draws").
  Demonstrated: `seed_everything` moves the process numpy stream too, so a caller
  that put `sample_ddpm` mid-flow would silently reseed everything downstream of
  it. The delivered script was never actually exposed — all four classical
  generators are built before the DDPM row — but the function is importable, so
  it now saves and restores numpy / python-random / torch (and CUDA) state around
  its whole body. The UNet construction has to be inside that scope as well as the
  sampling: instantiating the module initialises weights from the global torch RNG
  before `load_state_dict` overwrites them, and that alone perturbs a caller.
  **The real fix is a `generator=` parameter on `ddpm.sample`** —
  `seeding.torch_generator` exists for exactly this — but `ddpm.py` is owned
  elsewhere, so this contains the blast radius rather than removing the cause.
- **`iid_gaussian` emits negative volume on 37.9% of bars and negative spread on
  15.8%, and is not clipped.** Clipping would hand the null model a free
  improvement on the volume marginal and hide the most basic defect a generator
  can have. `test_iid_gaussian_does_emit_negative_volume` exists so nobody helpfully
  fixes it.

### Not done

- **`notebooks/02_baselines_and_scorecard.ipynb` does not exist.** CLAUDE.md's
  Layout names it, and non-negotiable #1 is "the notebook is the deliverable".
  What was shipped instead is `scripts/run_scorecard.py`, a path the Layout does
  not contain. This is a **deviation, recorded here rather than quietly
  substituted**: the script's `main()` is already the cell sequence a notebook
  would run — `pipeline.prepare` → `SeasonalGarchFit.fit` → `classical.GENERATORS`
  → `sample_ddpm` → `evaluate.scorecard` — so promoting it is mechanical, but it
  has not been done and nobody should go looking for a notebook that was never
  written. The script form was chosen because byte-identical reproducibility is
  the phase's main claim and a notebook carries execution counts and output
  metadata that make `diff` useless; that is a reason, not a licence, and the
  notebook is still owed. (`notebooks/` is owned by another lane, so this phase
  could not add it.)
- **No memorisation check on the DDPM row.** `evaluate.nearest_neighbour_distances`
  is written and unused here: with samples this far out of distribution the answer
  is a foregone "no", and the check only becomes informative against a `FULL`
  model that actually scores well.
- **`shuffled` leaves a residual corr(|r|, spread) of 0.173, not 0.** Permuting
  within a day cannot touch the *between-day* component — a volatile day has large
  |r| and wide spreads in all 78 of its bars. So part of `cross_corr_frobenius` is
  a statement about which day you are looking at rather than which bar, and the
  scorecard does not currently separate the two.
- **The three-session validation window is thin.** 1,078 day-images over 3 sessions
  is one market mood. `real_train`'s 0.268 on `tail_within` is a genuine
  train/reference shift, not estimator noise, and it sets the resolution of every
  comparison above.

**33 new tests in `tests/test_classical.py`; 102/102 pass** (69 pre-existing plus
33). Twenty-one of them are the ordinary contracts — bit-identical output per label,
shapes, finiteness, `shuffled` preserving each day's multiset exactly under
`np.sort`, `block_bootstrap` at block = 78 reducing to a rotation of a single real
day. The other twelve assert that each baseline **loses** to the real-vs-real floor on the
metric it was designed to fail. If one of those starts passing, either a metric
has stopped discriminating or a generator has stopped being broken in the way the
scorecard's interpretation assumes, and either is a finding.
