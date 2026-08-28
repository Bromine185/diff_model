# wavelet-ddpm-equities — build spec

Reproduction of Takahashi & Mizuno, *Generation of synthetic financial time series by diffusion models*
([arXiv:2410.18897](https://arxiv.org/abs/2410.18897), *Quantitative Finance* 25:1507–1516), at 5-minute
resolution across a cross-section of US equities.

## Non-negotiables

1. **The notebook is the deliverable.** Mechanism code (wavelet, DDPM math, UNet, training loop) is
   written *inline* in the notebook and mirrored into `src/` for testing. Supporting code (data loading,
   metrics, plotting) is imported. Reading `notebooks/01_wavelet_ddpm.ipynb` top to bottom must show the
   math and the code that implements it, adjacent.
2. **Every transform is exactly invertible, and there is a test that proves it.** Wavelet round-trip,
   normalization round-trip, image pack/unpack round-trip. `max |x - inv(fwd(x))| < 1e-10`.
3. **No live network calls at runtime.** Real data is fetched *once* by `scripts/fetch_bars.py` and
   committed as parquet under `data/fixtures/`. Nothing downstream touches yfinance. (Convention carried
   over from the `3d-night` project.)
4. **Deterministic.** All randomness flows from a seeded generator. A seeded run twice produces
   bit-identical output. Fold/stream seeds are derived by hashing a label, so re-running one component
   does not perturb another's draws.
5. **No full training runs locally.** Local = `SMOKE` preset, which proves correctness. `FULL` runs on
   Colab. Both are the same code path with different numbers.
6. **Null results get written down.** `RESEARCH.md` records what failed and what was killed, not just
   what worked. If TimeGAN doesn't converge, that is the finding.

## Geometry

The paper works at 1-minute resolution; we work at 5-minute. This is the whole derivation:

```
US regular session 09:30–16:00      = 390 minutes
                                    = 78 five-minute bars
mirror-expand 78 → 128 = 2^7        (paper: 390 → 512 = 2^9)
Haar MODWT, levels 1..7 + smooth V7 = 8 coefficient planes × 128 samples
                                    = 8 × 128 image  (paper: 16 × 256)
3 channels: R = log returns, G = spread, B = volume
```

One octave smaller than the paper in both dimensions. The UNet therefore gets **three** downsample stages,
not four, because 8 rows only halve three times.

## Deviations from the paper, and why

| Paper | Here | Reason |
|---|---|---|
| AAPL 1-min, 2005–2014, paid vendor | ~300 tickers, 5-min, 60 days, yfinance | that vendor data isn't free; we buy sample count with breadth instead of history |
| True bid-ask spread | normalized high-low range `(h-l)/c` | yfinance 5-min bars are OHLCV; there is no quote data. Corwin–Schultz is implemented alongside and switchable |
| Decimated DWT (as written) | undecimated Haar MODWT | a decimated DWT halves coefficients per level, so levels cannot share a rectangular image row. The redundant transform is what makes "k-th order coefficients in the k-th row" work at all |
| `(X-μ)^(1/p)/σ` | `sign(x)·|x|^(1/p)` then `/σ` | the paper's form is undefined for negative returns |

Deviation 2 is the only one that changes the science. It is flagged in the notebook at the cell where it
happens, not buried here.

## Known limitation, stated up front

60 days of 5-minute data is **one macro regime**. The model sees cross-sectional variety across ~300
names but no 2008, no COVID, no rate cycle. Any claim about regime-robustness is unsupported by this
dataset and must not be made without a longer sample.

## Layout

```
scripts/fetch_bars.py       run once → data/fixtures/*.parquet, committed
src/diffmodel/
  config.py                 SMOKE / FULL presets, device detection
  seeding.py                seeded forkable RNG
  data.py                   fixture loader, session calendar, day-image assembly
  transform.py              sign-preserving power transform, arsinh, winsorize + inverses
  wavelet.py                Haar MODWT fwd/inv + image packing/unpacking
  unet.py                   UNet from scratch
  ddpm.py                   schedules, q(x_t|x_0), reverse posterior, sampler, EMA
  baselines/                nowavelet.py, classical.py, quantgan.py, timegan.py
  evaluate.py               stylized-fact metrics → scorecard
notebooks/
  01_wavelet_ddpm.ipynb     primary deliverable
  02_baselines_and_scorecard.ipynb
```

## Data policy

`data/fixtures/*.parquet` **is committed** (~20 MB compressed). This is deliberate: Colab gets the
training data by `git clone`, with no Drive mount, no re-fetch, and a guarantee that the cloud run trains
on byte-identical data to the local smoke run. Checkpoints are *not* committed.
