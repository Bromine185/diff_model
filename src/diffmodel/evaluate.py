"""Stylized facts as numbers, not plots.

The paper evaluates its generator by showing that synthetic data reproduces fat
tails, volatility clustering, intraday seasonality and cross-correlations — and
demonstrates this with figures. Figures are persuasive and unfalsifiable. Every
fact here is reduced to a scalar distance between real and synthetic, so the
comparison table can be read rather than squinted at.

Each `sf_*` function takes (B, 3, n) day-series and returns a summary. Each
`d_*` function turns a (real, synthetic) pair into a single non-negative
distance, zero being perfect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RET, SPREAD, VOL = 0, 1, 2


# --------------------------------------------------------------------------
# Fat tails
# --------------------------------------------------------------------------

def hill_tail_index(x: np.ndarray, tail_frac: float = 0.05) -> float:
    """Hill estimator of the tail index alpha for |x|.

    For a distribution with P(|X| > u) ~ u^-alpha, the Hill estimator on the top
    k order statistics is

        alpha_hat = 1 / mean( log(x_(i) / x_(k)) )      i = 1..k

    Smaller alpha means fatter tails. Equity returns typically land around 3-5;
    a Gaussian has no finite tail index and the estimator drifts upward with k.
    This is the single most diagnostic number for whether a generator has learned
    that markets jump: GANs and Gaussian-noise models routinely produce alpha > 6.
    """
    v = np.sort(np.abs(np.asarray(x).ravel()))
    v = v[v > 0]
    if v.size < 50:
        return np.nan
    k = max(10, int(tail_frac * v.size))
    tail = v[-k:]
    u = tail[0]
    logs = np.log(tail / u)
    m = logs.mean()
    return float(1.0 / m) if m > 0 else np.nan


def excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (0 for a Gaussian)."""
    v = np.asarray(x).ravel()
    v = v[np.isfinite(v)]
    c = v - v.mean()
    s = c.std()
    return float(np.mean(c**4) / s**4 - 3.0) if s > 0 else np.nan


def standardize_per_day(series: np.ndarray) -> np.ndarray:
    """Divide each day-image's returns by that day's own standard deviation.

    Measured on the real fixture, pooling 380 tickers of differing volatility is
    ITSELF a fat-tail generator: a mixture of Gaussians with dispersed sigma is
    leptokurtic even when every component is thin-tailed. Pooled Hill alpha is
    2.43, but the per-ticker median is 2.73 and vol-standardising recovers exactly
    2.73 with excess kurtosis falling 34.4 -> 19.9.

    So "fat tails" splits into two claims a generator can get right or wrong
    independently — the WITHIN-day tail shape, and the cross-sectional dispersion
    of volatility across names and days. Both are scored, separately, because a
    model can nail one while failing the other and a single pooled number would
    hide it. This function isolates the first.
    """
    r = series[:, RET]
    s = r.std(axis=-1, keepdims=True)
    out = series.copy()
    out[:, RET] = np.divide(r, s, out=np.zeros_like(r), where=s > 0)
    return out


def vol_dispersion(series: np.ndarray) -> float:
    """Std of log per-day return volatility — the cross-sectional half of fat tails.

    A generator producing every day at identical volatility scores ~0 here while
    potentially matching the within-day tail index perfectly.
    """
    s = series[:, RET].std(axis=-1)
    s = s[s > 0]
    return float(np.std(np.log(s))) if s.size > 2 else np.nan


def aggregational_gaussianity(series: np.ndarray, horizons=(1, 2, 4, 8, 16)) -> dict[int, float]:
    """Excess kurtosis of returns summed over increasing horizons.

    A real stylized fact: as you aggregate, returns become more Gaussian, so
    kurtosis should decay monotonically toward zero. A generator that produces
    iid fat-tailed noise gets the level right at horizon 1 and the decay wrong,
    which is why measuring the whole curve beats measuring one number.
    """
    r = series[:, RET]
    out = {}
    for h in horizons:
        if h > r.shape[-1]:
            continue
        usable = (r.shape[-1] // h) * h
        agg = r[:, :usable].reshape(r.shape[0], -1, h).sum(axis=-1)
        out[h] = excess_kurtosis(agg)
    return out


# --------------------------------------------------------------------------
# Volatility clustering
# --------------------------------------------------------------------------

def acf_within_day(x: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """Mean within-day autocorrelation of `x` at lags 1..max_lag.

    Computed per day and averaged, never across the day boundary — concatenating
    days would splice 15:55 onto the next 09:30 and manufacture a spurious jump
    at every session edge.
    """
    x = np.asarray(x, dtype=np.float64)
    c = x - x.mean(axis=-1, keepdims=True)
    denom = (c * c).sum(axis=-1)
    out = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        num = (c[..., lag:] * c[..., :-lag]).sum(axis=-1)
        ok = denom > 0
        out[lag - 1] = np.mean(num[ok] / denom[ok]) if ok.any() else np.nan
    return out


def acf_abs_returns(series: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """ACF of |returns| — the volatility-clustering signature."""
    return acf_within_day(np.abs(series[:, RET]), max_lag)


def acf_returns(series: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """ACF of raw returns. Should be ~0 at all lags; the market is not that easy."""
    return acf_within_day(series[:, RET], max_lag)


def acf_decay_exponent(acf: np.ndarray) -> float:
    """Fit acf(lag) ~ lag^-beta over positive values. Larger beta = faster decay."""
    lags = np.arange(1, len(acf) + 1)
    ok = np.isfinite(acf) & (acf > 1e-6)
    if ok.sum() < 5:
        return np.nan
    slope, _ = np.polyfit(np.log(lags[ok]), np.log(acf[ok]), 1)
    return float(-slope)


# --------------------------------------------------------------------------
# Intraday seasonality
# --------------------------------------------------------------------------

def intraday_profile(series: np.ndarray, channel: int = RET, absolute: bool = True) -> np.ndarray:
    """Mean value per bar-of-day, length 78. The U-shape lives here.

    Normalised to mean 1 so the profile's SHAPE is compared, not its level —
    a generator that gets the U right but the overall volatility wrong should
    fail on the tail-index metric, not twice on this one.
    """
    v = series[:, channel]
    if absolute:
        v = np.abs(v)
    prof = v.mean(axis=0)
    m = prof.mean()
    return prof / m if m != 0 else prof


# --------------------------------------------------------------------------
# Cross-channel structure
# --------------------------------------------------------------------------

def cross_correlation_matrix(series: np.ndarray) -> np.ndarray:
    """3x3 correlation of (|return|, spread, volume), pooled over all bars.

    Uses |return| rather than the signed return: the real relationship is that
    big moves come with wide spreads and heavy volume, and that is a magnitude
    effect. Signed returns would correlate with nothing and the matrix would be
    uninformative for every generator equally.
    """
    feats = np.stack(
        [np.abs(series[:, RET]).ravel(), series[:, SPREAD].ravel(), series[:, VOL].ravel()]
    )
    return np.corrcoef(feats)


def leverage_effect(series: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """corr(r_t, |r_{t+k}|) for k = 1..max_lag. Should be negative for equities.

    Down moves are followed by higher volatility than up moves of the same size.
    This is the hardest of the stylized facts for a generator to get, because it
    requires a sign-dependent relationship rather than a symmetric one.
    """
    r = series[:, RET]
    out = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        a = r[:, :-k].ravel()
        b = np.abs(r[:, k:]).ravel()
        out[k - 1] = np.corrcoef(a, b)[0, 1] if a.size > 2 else np.nan
    return out


# --------------------------------------------------------------------------
# Memorisation
# --------------------------------------------------------------------------

def nearest_neighbour_distances(
    query: np.ndarray, reference: np.ndarray, batch: int = 256
) -> np.ndarray:
    """Min L2 distance from each `query` day-image to any `reference` day-image.

    The check that usually gets skipped. A diffusion model trained on 22k images
    from a single 60-day regime can score perfectly on every stylized fact by
    memorising the training set and replaying it. Compare synthetic-to-train
    distances against held-out-val-to-train distances: if synthetic samples sit
    systematically CLOSER to training data than real unseen data does, the model
    is copying, and the scorecard above is meaningless.
    """
    q = query.reshape(len(query), -1).astype(np.float32)
    r = reference.reshape(len(reference), -1).astype(np.float32)
    r_sq = (r * r).sum(1)
    out = np.empty(len(q), dtype=np.float32)
    for i in range(0, len(q), batch):
        chunk = q[i : i + batch]
        d2 = (chunk * chunk).sum(1)[:, None] - 2.0 * chunk @ r.T + r_sq[None, :]
        out[i : i + batch] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


# --------------------------------------------------------------------------
# Scorecard
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StylizedFacts:
    """Everything measured on one dataset, real or synthetic."""

    tail_index: float           # pooled — conflates within-day tails and cross-sectional vol spread
    tail_index_within: float    # per-day vol-standardised — within-day tail shape alone
    vol_dispersion: float       # cross-sectional half of the same story
    kurtosis: float
    acf_abs: np.ndarray
    acf_raw: np.ndarray
    acf_decay: float
    intraday_vol: np.ndarray
    intraday_volume: np.ndarray
    cross_corr: np.ndarray
    leverage: np.ndarray        # DIAGNOSTIC ONLY — see note in `compare`
    agg_kurtosis: dict[int, float]

    @classmethod
    def measure(cls, series: np.ndarray, max_lag: int = 30) -> "StylizedFacts":
        acf_a = acf_abs_returns(series, max_lag)
        within = standardize_per_day(series)
        return cls(
            tail_index=hill_tail_index(series[:, RET]),
            tail_index_within=hill_tail_index(within[:, RET]),
            vol_dispersion=vol_dispersion(series),
            kurtosis=excess_kurtosis(series[:, RET]),
            acf_abs=acf_a,
            acf_raw=acf_returns(series, max_lag),
            acf_decay=acf_decay_exponent(acf_a),
            intraday_vol=intraday_profile(series, RET, absolute=True),
            intraday_volume=intraday_profile(series, VOL, absolute=False),
            cross_corr=cross_correlation_matrix(series),
            leverage=leverage_effect(series),
            agg_kurtosis=aggregational_gaussianity(series),
        )


def compare(real: StylizedFacts, synth: StylizedFacts) -> dict[str, float]:
    """Distances, all non-negative, all zero at perfect agreement."""

    def rel(a: float, b: float) -> float:
        if not np.isfinite(a) or not np.isfinite(b) or a == 0:
            return np.nan
        return abs(b - a) / abs(a)

    return {
        # fat tails, split into its two independent claims
        "tail_within_abs_err": abs(synth.tail_index_within - real.tail_index_within),
        "vol_dispersion_rel_err": rel(real.vol_dispersion, synth.vol_dispersion),
        "tail_pooled_abs_err": abs(synth.tail_index - real.tail_index),
        "kurtosis_rel_err": rel(real.kurtosis, synth.kurtosis),
        # volatility clustering
        "acf_abs_l2": float(np.sqrt(np.nanmean((synth.acf_abs - real.acf_abs) ** 2))),
        "acf_decay_abs_err": abs(synth.acf_decay - real.acf_decay),
        # intraday seasonality
        "intraday_vol_mse": float(np.nanmean((synth.intraday_vol - real.intraday_vol) ** 2)),
        "intraday_volume_mse": float(np.nanmean((synth.intraday_volume - real.intraday_volume) ** 2)),
        # cross-channel structure
        "cross_corr_frobenius": float(np.linalg.norm(synth.cross_corr - real.cross_corr)),
        # A generator can fake uncorrelated returns by emitting pure noise, so this
        # is reported alongside rather than summed into any total.
        "acf_raw_l2": float(np.sqrt(np.nanmean((synth.acf_raw - real.acf_raw) ** 2))),
    }


# Measured on the real fixture: corr(r_t, |r_{t+k}|) is ~0.00 at every lag, both
# raw and vol-standardised. The leverage effect is a daily-and-longer phenomenon
# and is genuinely absent within a 5-minute session — so it is NOT scored. Every
# generator, good or bad, would score ~0 against a real value of ~0, and the
# column would reward nothing. `StylizedFacts.leverage` is still measured and
# plotted, as a diagnostic and to keep the finding visible.
LEVERAGE_IS_DIAGNOSTIC_ONLY = True


def scorecard(real: np.ndarray, generators: dict[str, np.ndarray]) -> "object":
    """One row per generator. Returns a pandas DataFrame."""
    import pandas as pd

    rf = StylizedFacts.measure(real)
    rows = {name: compare(rf, StylizedFacts.measure(s)) for name, s in generators.items()}
    return pd.DataFrame(rows).T
