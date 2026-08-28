"""Non-neural generators: the floor and the ceiling of the scorecard.

A table of stylized-fact distances is only interpretable if you know what a bad
number and a good number look like. Four generators bracket the range, and none
of them involves a neural network:

    iid_gaussian     the null. Fails everything. Its job is to prove the metrics
                     discriminate at all — a column where iid noise scores as well
                     as the diffusion model is a column that measures nothing.
    shuffled         real marginals, destroyed time. Isolates "did the generator
                     learn temporal structure" from "did it learn the marginal",
                     which a single distance per fact otherwise conflates.
    block_bootstrap  real short-range structure, destroyed long-range. The middle
                     case, and the one that shows whether a metric can see past
                     30 minutes.
    garch_seasonal   the strong classical baseline. GARCH(1,1)-t on deseasonalised,
                     per-day vol-standardised returns, with the real intraday
                     profile and the real per-day volatility distribution put back.
                     This is the model that can legitimately beat a badly-trained
                     diffusion model, and if the paper's method cannot beat it,
                     the paper's method has not been demonstrated on our data.

Every generator takes real day-series (B, 3, 78) in RAW units — channel 0 log
returns, 1 high-low spread proxy, 2 volume — and returns synthetic (n, 3, 78) in
the same raw units, so `evaluate.scorecard(real, {...})` consumes them directly
with no codec in the loop. That is deliberate: the codec is part of the diffusion
pipeline under test, and a baseline that had to pass through it would be scored
on the codec's fidelity as well as its own.

WHAT IS FITTED VERSUS WHAT IS LEARNED
-------------------------------------
These baselines are *given* several of the quantities the scorecard measures, and
a reader who forgets that will mistake a lookup for a discovery:

  * `garch_seasonal` draws its per-day sigma from the empirical per-day sigma
    distribution and multiplies the real intraday |r| profile back in. So
    `vol_dispersion_rel_err` and `intraday_vol_mse` are fitted, not learned. They
    are not evidence for GARCH over diffusion. `acf_abs_l2` and
    `acf_decay_abs_err` belong on that list too, for a reason that is not obvious
    and is set out below.
  * `shuffled` resamples whole real days and permutes them, so each day's marginal
    is a real day's marginal EXACTLY. `block_bootstrap` draws blocks with
    replacement, so its per-day marginal is a bootstrap of a real one rather than
    a copy — close, but not the same claim, and it shows up as a worse tail index.
  * `garch_seasonal`'s spread channel is a regression on |r|. RESEARCH.md already
    flags corr(|r|, spread) = 0.815 as PARTLY DEFINITIONAL, since the high-low
    range and |close-to-close| are both within-bar dispersion measures. Fitting a
    line between them and then reporting that the line reproduces the correlation
    would be circular — though in fact it UNDERSHOOTS, 0.56 against 0.79, for the
    reason set out in `garch_seasonal`. corr(|r|, volume) and corr(spread, volume)
    are the meaningful cross-channel targets, because volume is measured
    independently of price.

THE |r| ACF IS MOSTLY THE INTRADAY PROFILE, NOT CLUSTERING
----------------------------------------------------------
`acf_abs_l2` and `acf_decay_abs_err` read like the two clean columns and they are
not. `evaluate.acf_within_day` de-means each 78-bar day by that day's *scalar*
mean and never removes the bar-of-day profile, so the deterministic U-shape sits
inside every |r| series it measures and dominates the result. On the real
reference split, dividing |r| by the profile drops the lag-1 ACF from 0.235 to
0.038 and the RMS over 30 lags — which is exactly what `acf_abs_l2` averages —
from 0.0822 to 0.0126. The profile's own ACF, treated as a one-day series, is
0.67 at lag 1 and still 0.38 at lag 6.

So multiplying the fitted profile back in at step 4 hands `garch_seasonal` most of
these two columns as well. Ablated over 6 seeds, holding everything else fixed
(`acf_abs_l2`, floor 0.0093):

    GARCH + fitted profile (shipped)  0.0122     iid Student-t + profile  0.0162
    GARCH, profile removed            0.0829     iid-t, neither           0.0897

Measured against that no-profile, no-recursion control, the profile covers 91% of
the distance to the floor and the recursion adds 5 points on top of it; the
recursion alone covers 9%, and leaves `acf_decay_abs_err` NaN in every run because
a deseasonalised GARCH path has no positive ACF surviving the estimator's filter.
The column a generator would have to earn without the seasonal is
`acf_within_day(|r| / profile)`, which `evaluate.py` does not currently expose.

Which means the statement has to be scoped per generator; there is no column that
nobody is handed. `garch_seasonal` is not handed `tail_within_abs_err` or
`kurtosis_rel_err` — its innovations are simulated Student-t, not resampled — nor
the volume half of `cross_corr_frobenius`, corr(|r|, volume) and
corr(spread, volume), which it reaches through a fitted loading on log|z| rather
than by copying, and which mean something because volume is measured
independently of price. But `block_bootstrap` IS handed those same cross-channel
numbers, because it cuts blocks at identical offsets in all three channels, and
`shuffled` is handed every marginal including `kurtosis_rel_err`, because each
day's multiset is a real day's. RESEARCH.md separately records that
`kurtosis_rel_err` is too noisy to rank anything at n = 512 in any case.

Fitting always happens on the series passed in — callers pass the TRAIN split —
and never on the validation data the generator will be scored against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evaluate import RET, SPREAD, VOL, intraday_profile
from ..seeding import fork


def _check(real: np.ndarray) -> np.ndarray:
    """Every generator's front door. Raw (B, 3, n) float64, finite."""
    real = np.asarray(real, dtype=np.float64)
    if real.ndim != 3 or real.shape[1] != 3:
        raise ValueError(f"expected (B, 3, n) raw day-series, got {real.shape}")
    if not np.isfinite(real).all():
        raise ValueError("real series contains non-finite values")
    return real


# --------------------------------------------------------------------------
# 1. The null model
# --------------------------------------------------------------------------

def iid_gaussian(real: np.ndarray, n: int, *, label: str = "iid-gaussian") -> np.ndarray:
    """Per-channel Gaussian matched to the pooled mean and standard deviation.

    The floor of the scorecard. It gets exactly two things right — the mean and
    the variance of each channel — and everything else wrong: no fat tails, no
    volatility clustering, no intraday U-shape, no cross-channel structure. If any
    column of the scorecard rates this competitive with a trained model, that
    column is not measuring what it claims to.

    Volume and spread are NOT clipped to be positive, and roughly a third of the
    volume draws come out negative because the volume distribution is strongly
    right-skewed with a mean well under one standard deviation from zero. That is
    left in on purpose. Clipping would hand the null model a free improvement on
    the volume marginal and hide the most basic failure mode there is — a
    generator emitting quantities that cannot physically exist. The scorecard
    should expose that, not launder it.
    """
    real = _check(real)
    rng = fork(label)
    mu = real.mean(axis=(0, 2))          # (3,)
    sd = real.std(axis=(0, 2))
    out = rng.normal(size=(n, 3, real.shape[-1]))
    return out * sd[None, :, None] + mu[None, :, None]


# --------------------------------------------------------------------------
# 2. Real marginals, destroyed time
# --------------------------------------------------------------------------

def shuffled(real: np.ndarray, n: int, *, label: str = "shuffled") -> np.ndarray:
    """Resample a real day, then permute time within it, independently per channel.

    The marginal distribution of every channel is EXACTLY real — not approximately,
    not in distribution: each synthetic day is a rearrangement of one real day's
    values, so its per-channel multiset is identical to a real day's. Tail index,
    kurtosis and per-day volatility therefore come out right by construction.

    What is destroyed is every ordering fact: the ACF of |r| goes to zero, the
    intraday U-shape flattens (a bar's position no longer predicts its size), and
    the three channels are permuted with *different* permutations so even the
    contemporaneous |r|-spread-volume alignment is gone.

    That separation is the whole point. A scorecard reports one distance per fact,
    and a generator can match the tail index because it learned the shape of the
    return distribution or because it memorised and reshuffled the training set.
    This baseline is the second thing, so any column where it scores well is a
    column that cannot tell the two apart. Read the diffusion model's row against
    this row, not against zero.
    """
    real = _check(real)
    b, _, t = real.shape
    rng = fork(label)
    src = rng.integers(0, b, size=n)
    out = real[src].copy()
    # A fresh permutation per (day, channel): shared permutations would leave the
    # cross-channel correlations intact, which is the opposite of the point.
    for i in range(n):
        for c in range(3):
            out[i, c] = out[i, c][rng.permutation(t)]
    return out


# --------------------------------------------------------------------------
# 3. Real short-range structure, destroyed long-range
# --------------------------------------------------------------------------

def block_bootstrap(
    real: np.ndarray, n: int, block: int = 6, *, label: str = "block-bootstrap"
) -> np.ndarray:
    """Circular block bootstrap within a single day. Default block = 6 bars = 30 min.

    Each synthetic day picks one real day, then tiles itself from ceil(78/block)
    blocks cut from that day at uniformly random start positions, wrapping at the
    end. Volatility clustering survives *inside* a block and is severed at every
    block boundary, so the ACF of |r| should track the real one out to lag
    `block-1` and collapse beyond it. That makes this the diagnostic for whether
    `acf_abs_l2` can see past half an hour, which a lag-1 number could not.

    Blocks are cut at the same offsets in all three channels, so contemporaneous
    cross-channel structure is preserved — that is what distinguishes this from
    `shuffled`, which destroys it. The intraday profile is still destroyed, since
    a block landing at bar 40 is equally likely to have come from the open.

    At the limit `block = 78` there is exactly one block per day and nothing to
    reassemble: the output is one real day read from a random circular start, i.e.
    a rotation, with no splicing between days or between channels. That degenerate
    case is what `test_block_bootstrap_with_full_length_block_keeps_the_day_intact`
    pins, and it is the cheapest available check that the block machinery is
    cutting and pasting the thing it claims to.

    CIRCULAR, not the more common moving-block scheme, for a mechanical reason:
    the day-image is a fixed 3 x 78 rectangle and every wavelet-side assumption in
    this project depends on that. A non-circular bootstrap either produces ragged
    tails that need padding, or under-samples the last `block-1` bars because no
    block may start there — which would systematically thin out the closing
    auction, the single most distinctive feature of the volume channel.

    The cost is real and worth stating: wrapping splices 15:55 onto 09:30 inside a
    block. That manufactures exactly the kind of fake session-edge transition that
    `evaluate.acf_within_day` refuses to create when it computes ACFs per day
    rather than on a concatenated series. The arithmetic at block = 6: each of the
    13 blocks wraps with probability 5/78, so a synthetic day carries about 0.8
    spliced transitions among its 77. Small, and a defect of the estimator rather
    than a feature of the data.
    """
    real = _check(real)
    b, _, t = real.shape
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    rng = fork(label)

    n_blocks = int(np.ceil(t / block))
    src = rng.integers(0, b, size=n)
    starts = rng.integers(0, t, size=(n, n_blocks))

    # index[i, k*block + j] = (starts[i, k] + j) mod t, truncated to t columns
    offsets = np.arange(block)[None, None, :]
    idx = (starts[:, :, None] + offsets) % t
    idx = idx.reshape(n, n_blocks * block)[:, :t]

    return np.take_along_axis(real[src], idx[:, None, :], axis=-1)


# --------------------------------------------------------------------------
# 4. The strong classical baseline
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SeasonalGarchFit:
    """Everything `garch_seasonal` estimates from the training split.

    Kept as an inspectable object rather than closed over inside the generator so
    the fitted parameters can be printed in the scorecard script and quoted in
    RESEARCH.md. A baseline whose numbers cannot be reported is a baseline nobody
    can check.
    """

    # returns
    profile_ret: np.ndarray     # (78,) mean |r| per bar-of-day, mean 1
    omega: float
    alpha: float
    beta: float
    nu: float                   # Student-t degrees of freedom
    n_garch_days: int           # days in the GARCH subsample — NOT len(day_sigma)
    day_sigma: np.ndarray       # (B,) empirical per-day sigma of deseasonalised r
    # spread:  log(s / profile) = a + b log(|r| / profile) + N(0, sd^2)
    profile_spread: np.ndarray  # (78,) mean spread per bar-of-day, mean 1
    spread_a: float
    spread_b: float
    spread_sd: float
    ret_floor: float
    # volume:  log(v / profile) = level + c * (log|z| - mean) + N(0, sd^2)
    profile_vol: np.ndarray     # (78,) mean volume per bar-of-day, mean 1
    day_vol_level: np.ndarray   # (B,) per-day mean of deseasonalised log volume
    vol_c: float
    vol_sd: float
    logabsz_mean: float
    z_floor: float
    vol_floor: float

    @property
    def persistence(self) -> float:
        """alpha + beta. Approaches 1 for a market with long-memory volatility."""
        return self.alpha + self.beta

    def summary(self) -> str:
        """One printable block. The two sample sizes are NOT the same number.

        The GARCH line is estimated on a `max_fit_days` subsample (256 by default)
        because the likelihood is the expensive step; every other quantity here —
        both profiles, the day-sigma pool, the two regressions — uses all of the
        day-images passed in. Printing one count under both would misattribute the
        larger sample to the parameter line, which is the one a reader is most
        likely to quote.
        """
        return (
            f"GARCH(1,1)-t  omega={self.omega:.4f} alpha={self.alpha:.4f} "
            f"beta={self.beta:.4f} (persistence {self.persistence:.4f}) nu={self.nu:.2f}\n"
            f"  spread   log(s/prof) = {self.spread_a:.3f} + {self.spread_b:.3f} log(|r|/prof) "
            f"+ N(0, {self.spread_sd:.3f}^2)\n"
            f"  volume   log(v/profile) = level + {self.vol_c:.3f} log|z| "
            f"+ N(0, {self.vol_sd:.3f}^2)\n"
            f"  fitted on {self.n_garch_days:,} of {len(self.day_sigma):,} day-images (GARCH "
            f"likelihood); {len(self.day_sigma):,} for the profiles, day-sigma pool "
            f"and regressions"
        )

    @classmethod
    def fit(cls, real: np.ndarray, *, max_fit_days: int = 256,
            label: str = "garch-fit-subsample") -> "SeasonalGarchFit":
        """Estimate every parameter from `real`, which must be the TRAIN split.

        The GARCH fit is the expensive step and the one with a methodological
        wart, so both are handled here rather than inside the generator.
        """
        real = _check(real)
        rng = fork(label)

        # --- returns: strip the seasonal, then the day level --------------
        # REUSE the scorecard's own profile estimator. Re-deriving it here would
        # let the baseline be deseasonalised by a slightly different U-shape than
        # the one it is scored against, and it would score better than it should.
        profile_ret = intraday_profile(real, RET, absolute=True)
        r_ds = real[:, RET] / profile_ret[None, :]

        day_sigma = r_ds.std(axis=-1)
        ok = day_sigma > 0
        z = r_ds[ok] / day_sigma[ok][:, None]

        # Concatenating 78-bar days into one long series to fit GARCH is a FUDGE
        # and it is worth naming: it manufactures a fake overnight transition at
        # every 78th observation, where the recursion carries yesterday's closing
        # variance into this morning's opening bar for a different ticker
        # entirely. With alpha ~ 0.06 the contamination is one observation in 78
        # at a weight of 6%, which is small; it is not zero. The honest
        # alternative — a panel GARCH re-initialised per day — is a different
        # model, and the point of this baseline is to be the standard one.
        take = min(max_fit_days, len(z))
        idx = np.sort(rng.choice(len(z), size=take, replace=False))
        omega, alpha, beta, nu = _fit_garch_t(z[idx].ravel())

        # --- spread: regress log s on log |r| -----------------------------
        # Literally the relationship RESEARCH.md warns is partly definitional.
        # Both sides are deseasonalised by their own bar-of-day profile first, so
        # the spread channel inherits the real intraday U rather than a copy of
        # the return U raised to the power b — which, at the fitted b of ~0.26,
        # would be almost flat. That makes the spread profile one more FITTED
        # quantity; it is not scored directly, but it feeds
        # `cross_corr_frobenius` through corr(spread, volume).
        #
        # Both series are floored at their own 0.1st percentile of positive
        # values before the log: the real fixture contains exact zeros (a bar that
        # never traded outside one price), and log(0) would take the whole fit
        # with it.
        profile_spread = intraday_profile(real, SPREAD, absolute=False)
        abs_r_ds = (np.abs(real[:, RET]) / profile_ret[None, :]).ravel()
        s_ds = (real[:, SPREAD] / profile_spread[None, :]).ravel()
        ret_floor = _positive_floor(abs_r_ds)
        x = np.log(np.maximum(abs_r_ds, ret_floor))
        y = np.log(np.maximum(s_ds, _positive_floor(s_ds)))
        spread_b, spread_a = np.polyfit(x, y, 1)
        spread_sd = float(np.std(y - (spread_a + spread_b * x)))

        # --- volume: seasonal profile x lognormal driven by |z| -----------
        # The profile is applied multiplicatively in RAW space, not in log space,
        # so that E[v | bar t] is proportional to `profile_vol[t]` exactly and the
        # closing-auction spike survives Jensen's inequality intact. Doing it in
        # logs would reproduce the profile of the median, not of the mean, and
        # `intraday_volume_mse` scores the mean.
        profile_vol = intraday_profile(real, VOL, absolute=False)
        vol = real[:, VOL]
        vol_floor = _positive_floor(vol.ravel())
        w = np.log(np.maximum(vol, vol_floor)) - np.log(profile_vol)[None, :]
        day_vol_level = w.mean(axis=-1)
        w_resid = (w - day_vol_level[:, None])[ok]

        # The regressor is the standardised innovation |z|, not |r|: the day's own
        # volatility level is already carried by `day_vol_level`, and regressing on
        # raw |r| would double-count it and inflate the loading.
        z_floor = _positive_floor(np.abs(z).ravel())
        logabsz = np.log(np.maximum(np.abs(z), z_floor))
        logabsz_mean = float(logabsz.mean())
        xz = (logabsz - logabsz_mean).ravel()
        vol_c, vol_intercept = np.polyfit(xz, w_resid.ravel(), 1)
        vol_sd = float(np.std(w_resid.ravel() - (vol_intercept + vol_c * xz)))

        return cls(
            profile_ret=profile_ret,
            omega=omega, alpha=alpha, beta=beta, nu=nu, n_garch_days=int(take),
            day_sigma=day_sigma[ok],
            profile_spread=profile_spread,
            spread_a=float(spread_a), spread_b=float(spread_b), spread_sd=spread_sd,
            ret_floor=float(ret_floor),
            profile_vol=profile_vol,
            day_vol_level=day_vol_level[ok],
            vol_c=float(vol_c), vol_sd=vol_sd, logabsz_mean=logabsz_mean,
            z_floor=float(z_floor), vol_floor=float(vol_floor),
        )

    def simulate(self, n: int, rng: np.random.Generator, burn_in: int = 200) -> np.ndarray:
        """Draw `n` synthetic day-series of shape (n, 3, 78) in raw units."""
        t = len(self.profile_ret)

        # --- day-level draws, jointly ------------------------------------
        # One index draw feeds BOTH the day's volatility and its volume level, so
        # the empirical dependence between the two (busy days are volatile days)
        # comes along for free. Drawing them independently would produce quiet
        # days with closing-auction volume and reduce corr(|r|, volume), one of the
        # two cross-channel numbers this baseline is not simply handed.
        di = rng.integers(0, len(self.day_sigma), size=n)
        sigma_day = self.day_sigma[di]
        level_day = self.day_vol_level[di]

        # --- GARCH(1,1)-t innovations ------------------------------------
        z = self._simulate_garch(n, t, rng, burn_in)

        # Renormalise each simulated day to unit variance. `day_sigma` was DEFINED
        # as the per-day standard deviation of deseasonalised returns, so without
        # this the drawn dispersion and the dispersion the GARCH recursion
        # generates on its own compound.
        #
        # Measured PAIRED — same `fork(label)` stream feeding both variants, so the
        # only difference is this line — the compounding is SMALL: over 6 seeds,
        # std(log per-day sigma) is 0.443 with the line and 0.455 without, a mean
        # inflation of +2.8% (range +0.8% to +4.1%) against a real 0.445. That is
        # well inside the 0.418-0.470 seed-to-seed spread of the estimate itself,
        # which is why the measurement has to be paired: an unpaired comparison of
        # a handful of draws cannot resolve an effect this size and can come out
        # with the wrong sign. The fitted persistence is only 0.589, so a 78-step
        # path just does not accumulate much day-level variance to add in
        # quadrature.
        #
        # The line is kept for consistency rather than for the size of the
        # correction, and what it makes exact is narrower than it looks: after it,
        # the simulated day's DESEASONALISED std is exactly the drawn `day_sigma`
        # (to 1e-18). But `evaluate.vol_dispersion` scores the RAW per-day std, and
        # the profile is multiplied back in AFTER this line, so the raw std is
        # `day_sigma` times std(z_norm * profile) — a day-varying factor measured
        # at 0.89-1.48 across days, mean 1.09. So `vol_dispersion_rel_err` is
        # handed in EXPECTATION, not exactly, with or without this line.
        z = z / np.maximum(z.std(axis=-1, keepdims=True), 1e-12)

        ret = z * sigma_day[:, None] * self.profile_ret[None, :]

        # --- spread from |r| ---------------------------------------------
        abs_r_ds = np.maximum(np.abs(z * sigma_day[:, None]), self.ret_floor)
        log_s = (
            self.spread_a
            + self.spread_b * np.log(abs_r_ds)
            + self.spread_sd * rng.standard_normal(size=(n, t))
        )
        spread = np.exp(log_s) * self.profile_spread[None, :]

        # --- volume: seasonal profile x lognormal in |z| ------------------
        log_absz = np.log(np.maximum(np.abs(z), self.z_floor)) - self.logabsz_mean
        w = level_day[:, None] + self.vol_c * log_absz + self.vol_sd * rng.standard_normal((n, t))
        volume = np.exp(w) * self.profile_vol[None, :]

        return np.stack([ret, spread, volume], axis=1)

    def _simulate_garch(self, n: int, t: int, rng: np.random.Generator, burn_in: int) -> np.ndarray:
        """n independent GARCH(1,1)-t paths of length t, after `burn_in` discarded steps.

        Simulated here rather than through `arch`'s own simulator so that every
        draw comes from the project's seeded generator. `arch` uses its own
        RandomState, and a baseline whose output depends on a second, separately
        seeded RNG would break the "same label twice, bit-identical output" rule
        that the rest of the project holds to.

        The burn-in exists because starting every path at the unconditional
        variance would make bar 0 of every synthetic day the one bar with no
        clustering — a systematic artefact landing exactly on the open, which is
        where the intraday profile is largest and the scorecard is most sensitive.
        """
        total = burn_in + t
        # Student-t scaled to unit variance: Var[t_nu] = nu / (nu - 2).
        e = rng.standard_t(self.nu, size=(n, total)) * np.sqrt((self.nu - 2.0) / self.nu)

        uncond = self.omega / max(1.0 - self.alpha - self.beta, 1e-6)
        h = np.full(n, uncond)
        y_prev = np.zeros(n)
        out = np.empty((n, total))
        for k in range(total):
            h = self.omega + self.alpha * y_prev**2 + self.beta * h
            y = np.sqrt(h) * e[:, k]
            out[:, k] = y
            y_prev = y
        return out[:, burn_in:]


def _positive_floor(x: np.ndarray, q: float = 0.001) -> float:
    """A small strictly-positive value to floor logs at.

    The 0.1st percentile of the strictly positive values, not an arbitrary
    epsilon: the real fixture genuinely contains exact zeros (31 zero spreads and
    3 zero volumes in the SMOKE train split), and an epsilon like 1e-12 would map
    them to log = -27 and drag every regression it touches.
    """
    v = np.asarray(x)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.quantile(v, q)) if v.size else 1e-12


def _fit_garch_t(y: np.ndarray) -> tuple[float, float, float, float]:
    """Fit GARCH(1,1) with Student-t innovations via `arch`. Returns (omega, a, b, nu)."""
    try:
        from arch import arch_model
    except ImportError as exc:   # pragma: no cover - environment problem, not logic
        raise ImportError(
            "garch_seasonal needs the `arch` package (pip install arch); it is in "
            "requirements.txt"
        ) from exc

    res = arch_model(
        y, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False
    ).fit(disp="off", show_warning=False)
    p = res.params
    return (
        float(p["omega"]),
        float(p["alpha[1]"]),
        float(p["beta[1]"]),
        float(max(p["nu"], 2.1)),   # nu <= 2 has infinite variance and no unit-variance scaling
    )


def garch_seasonal(
    real: np.ndarray,
    n: int,
    *,
    label: str = "garch-seasonal",
    max_fit_days: int = 256,
    fit: SeasonalGarchFit | None = None,
) -> np.ndarray:
    """Seasonal GARCH(1,1)-t with a fitted spread and volume overlay. (n, 3, 78) raw.

    The construction, in order:

        1. Deseasonalise returns by the real intraday |r| profile — reusing
           `evaluate.intraday_profile`, the same estimator the scorecard uses.
        2. Standardise each day by its own sigma, leaving pooled innovations z.
        3. Fit GARCH(1,1) with Student-t innovations to z.
        4. Simulate 78 steps after a burn-in, renormalise each day to unit
           variance, multiply by a sigma drawn from the empirical per-day
           distribution, multiply the profile back in.
        5. Spread from a fitted log s = a + b log|r| regression.
        6. Volume from the real bar-of-day profile times a lognormal with a fitted
           loading on log|z|.

    This is the baseline that can legitimately beat an undertrained diffusion
    model on most of the scorecard, and it should be read as the bar to clear.

    THE COLUMNS IT IS HANDED RATHER THAN EARNS, which the reader must not count
    as evidence:

      `vol_dispersion_rel_err`  the per-day sigma is drawn from the empirical
          per-day sigma distribution and each simulated day is renormalised so
          that its deseasonalised std hits it exactly (the RAW std, which is what
          this column scores, then picks up a day-varying factor from the
          profile — so: handed in expectation). This number is a resampling
          artefact.
      `intraday_vol_mse`        the real |r| profile is multiplied back in at
          step 4, and `intraday_volume_mse` likewise at step 6.
      `acf_abs_l2`, `acf_decay_abs_err`   the same profile, counted a second
          time. `evaluate.acf_within_day` does not deseasonalise, so ~84% of the
          real |r| ACF it measures is the U-shape rather than clustering. Ablated:
          replacing the GARCH recursion with iid Student-t moves `acf_abs_l2` only
          0.0122 -> 0.0162 (floor 0.0093), while removing the profile and keeping
          the recursion gives 0.0829 and NaN decay. See the module docstring for
          the full table. The recursion is doing something — it is worth 5 of the
          96 percentage points of distance-to-floor the shipped generator covers —
          but calling these columns earned overstates it by an order of magnitude.
      the |r|-spread corner of `cross_corr_frobenius`   step 5 fits a line
          between |r| and the high-low range. RESEARCH.md flags corr(|r|, spread)
          = 0.815 as partly DEFINITIONAL — both are within-bar dispersion measures
          — so reproducing it would demonstrate that OLS works, not that the model
          understands microstructure.

          Measured, it does not even reproduce it: the simulated correlation is
          0.56 against a real 0.79. The log-log fit is weak (slope 0.258, R^2 0.30)
          and OLS is further attenuated by errors-in-variables, since log|r| is the
          REALISED move and carries Var(log|N(0,1)|) ~ 1.23 of noise around the
          latent bar volatility. The definitional link turns out to be a bound in
          raw space rather than a conditional mean — the high-low range is below
          |close-to-close| on only 5.8% of real bars, median ratio 2.0 — and a
          log-linear conditional mean cannot express a bound. Kept as the model the
          task specifies rather than patched with a floor, because the shortfall is
          the more informative result.

    What it does have to earn: `tail_within_abs_err` and `kurtosis_rel_err` (from
    the Student-t innovations plus the recursion, once the day sigma is divided
    out), and corr(|r|, volume) / corr(spread, volume), which are the cross-channel
    targets that mean something because volume is measured independently of price.
    That is a shorter list than the construction suggests, and it is the honest
    one.

    Pass a pre-computed `fit` to reuse one estimation across several draws; the
    default refits, which costs about a second.
    """
    real = _check(real)
    f = fit if fit is not None else SeasonalGarchFit.fit(real, max_fit_days=max_fit_days)
    return f.simulate(n, fork(label))


#: Name -> callable, for scripts and notebooks that want the whole set.
#: Every entry is callable as `fn(real, n)` — `block_bootstrap`'s block length and
#: `garch_seasonal`'s pre-computed fit both default — so a scorecard can be built
#: with `{k: f(real, n) for k, f in GENERATORS.items()}`. Note that iterating this
#: way refits GARCH on every call; pass `fit=` explicitly to reuse one estimation.
GENERATORS = {
    "iid_gaussian": iid_gaussian,
    "shuffled": shuffled,
    "block_bootstrap": block_bootstrap,
    "garch_seasonal": garch_seasonal,
}
