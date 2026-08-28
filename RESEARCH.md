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
