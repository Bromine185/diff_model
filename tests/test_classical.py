"""The classical baselines, and the proof that the scorecard has teeth.

Two contracts are tested here and only one of them is about `classical.py`.

The first is the ordinary one: shapes, dtypes, determinism, and the structural
invariants each generator claims (`shuffled` really does preserve the marginal;
`block_bootstrap` really does keep a day intact when the block is the whole day).

The second is the reason this file matters. `evaluate.compare` returns ten
non-negative numbers per generator, and nothing in `evaluate.py` establishes that
any of them can tell a good generator from a bad one. Two of the ten turn out not
to: `acf_raw_l2` scores iid Gaussian noise the same as real data, and
`acf_decay_abs_err` returns NaN for any generator with no clustering to fit. Both
are pinned below, because a column that ranks nothing needs to be *known* to rank
nothing rather than quietly averaged into a verdict.

A third measures something other than its name. `acf_abs_l2` is 84% intraday
seasonality rather than volatility clustering, because `evaluate.acf_within_day`
de-means each day by a scalar and leaves the bar-of-day profile in. That is pinned
too, by `test_garch_recursion_earns_only_a_slice_of_acf_abs_l2`, which is the
correction to an earlier version of this file that asserted the opposite.

The rest of the tests at the bottom pin each baseline against a real-vs-real
reference on the metric it was DESIGNED to fail. If one of them starts passing,
either a metric has stopped discriminating or a generator has stopped being
broken in the way the scorecard's interpretation assumes, and either is a finding.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from diffmodel.baselines.classical import (
    SeasonalGarchFit,
    block_bootstrap,
    garch_seasonal,
    iid_gaussian,
    shuffled,
)
from diffmodel.config import BARS_PER_DAY, get_config
from diffmodel.data import build_day_images
from diffmodel.evaluate import (
    RET,
    StylizedFacts,
    compare,
    cross_correlation_matrix,
    intraday_profile,
)
from diffmodel.seeding import fork

N_FIT = 384      # real day-images the generators are fitted on
N_REF = 384      # held-out day-images they are scored against
N_GEN = 384      # synthetic day-images per generator — matched, so the noise is too


@pytest.fixture(scope="module")
def splits():
    """A small fit/reference pair cut from the real fixture, by DATE.

    Subsampled hard so the whole module runs in seconds. Split by date rather
    than by row for the same reason `data.split_by_date` insists on it: two
    tickers from the same session share a market-wide volatility shock, so a row
    split would put a contaminated reference on the other side and the
    discrimination tests below would be measuring nothing.
    """
    cfg = get_config("SMOKE")
    train, val = build_day_images().split_by_date(cfg.val_sessions)
    rng = fork("test-classical-subsample")
    return (
        train.take(N_FIT, rng=rng).series,
        val.take(N_REF, rng=rng).series,
    )


@pytest.fixture(scope="module")
def fit_series(splits):
    return splits[0]


@pytest.fixture(scope="module")
def garch_fit(fit_series):
    """One GARCH estimation shared across the module; refitting costs ~1s each."""
    return SeasonalGarchFit.fit(fit_series)


def _all_generators(real, n, garch_fit):
    return {
        "iid_gaussian": iid_gaussian(real, n),
        "shuffled": shuffled(real, n),
        "block_bootstrap": block_bootstrap(real, n),
        "garch_seasonal": garch_seasonal(real, n, fit=garch_fit),
    }


# --- determinism ----------------------------------------------------------

@pytest.mark.parametrize("gen", [iid_gaussian, shuffled, block_bootstrap])
def test_same_label_is_bit_identical(fit_series, gen):
    """Not 'close'. Identical. The project's reproducibility claim is bitwise."""
    a = gen(fit_series, 64, label="repeat-me")
    b = gen(fit_series, 64, label="repeat-me")
    assert np.array_equal(a, b)


def test_garch_same_label_is_bit_identical(fit_series, garch_fit):
    a = garch_seasonal(fit_series, 64, label="repeat-me", fit=garch_fit)
    b = garch_seasonal(fit_series, 64, label="repeat-me", fit=garch_fit)
    assert np.array_equal(a, b)


def test_garch_fit_is_itself_deterministic(fit_series):
    """The fit subsamples days, so it has its own RNG and its own way to drift."""
    a, b = SeasonalGarchFit.fit(fit_series), SeasonalGarchFit.fit(fit_series)
    assert (a.omega, a.alpha, a.beta, a.nu) == (b.omega, b.alpha, b.beta, b.nu)
    assert np.array_equal(a.day_sigma, b.day_sigma)


@pytest.mark.parametrize("gen", [iid_gaussian, shuffled, block_bootstrap])
def test_different_labels_give_different_draws(fit_series, gen):
    """Labels must actually fork the stream, not decorate a shared one."""
    a = gen(fit_series, 64, label="stream-a")
    b = gen(fit_series, 64, label="stream-b")
    assert not np.array_equal(a, b)


def test_garch_different_labels_give_different_draws(fit_series, garch_fit):
    a = garch_seasonal(fit_series, 64, label="stream-a", fit=garch_fit)
    b = garch_seasonal(fit_series, 64, label="stream-b", fit=garch_fit)
    assert not np.array_equal(a, b)


# --- shape, dtype, finiteness --------------------------------------------

def test_every_generator_returns_well_formed_raw_series(fit_series, garch_fit):
    for name, out in _all_generators(fit_series, 40, garch_fit).items():
        assert out.shape == (40, 3, BARS_PER_DAY), name
        assert out.dtype == np.float64, name
        assert np.isfinite(out).all(), name


def test_generators_registry_is_uniformly_callable(fit_series):
    """`GENERATORS` promises `fn(real, n)` works for all four. Notebooks rely on it.

    Cheap to assert and easy to break: adding a required keyword to any generator
    would silently make the registry unusable for the loop it exists to serve.
    """
    from diffmodel.baselines.classical import GENERATORS

    assert set(GENERATORS) == {"iid_gaussian", "shuffled", "block_bootstrap", "garch_seasonal"}
    for name, fn in GENERATORS.items():
        assert fn(fit_series, 8).shape == (8, 3, BARS_PER_DAY), name


def test_generators_reject_wrong_shape():
    with pytest.raises(ValueError, match=r"expected \(B, 3, n\)"):
        iid_gaussian(np.zeros((4, 2, BARS_PER_DAY)), 8)


def test_generators_reject_non_finite_input():
    bad = np.zeros((4, 3, BARS_PER_DAY))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        shuffled(bad, 8)


# --- structural invariants ------------------------------------------------

def test_shuffled_preserves_each_days_multiset_exactly(fit_series):
    """Every synthetic day is a permutation of SOME real day, per channel.

    This is what makes `shuffled` the right control for "did it learn the
    marginal or the dynamics": the marginal is not approximately real, it is the
    same multiset of numbers. `np.sort` equality is the strongest form of that
    claim available, and it is exact — no tolerance.
    """
    out = shuffled(fit_series, 48)
    # Channels are permuted independently, so match them independently.
    per_channel = [
        {np.sort(fit_series[j, c]).tobytes() for j in range(len(fit_series))} for c in range(3)
    ]
    for i in range(len(out)):
        for c in range(3):
            assert np.sort(out[i, c]).tobytes() in per_channel[c], f"day {i} channel {c}"


def test_shuffled_destroys_cross_channel_alignment(fit_series):
    """The independent-permutation claim, measured rather than asserted.

    Each channel is permuted separately, so the contemporaneous corr(|r|, spread)
    — 0.8 in real data, the largest entry off the diagonal — must collapse.

    It collapses to about 0.18, NOT to zero, and the residue is not a bug. The
    pooled correlation has a between-day component: a volatile day has larger |r|
    AND wider spreads in every one of its 78 bars, and permuting time within the
    day cannot touch that. So `shuffled` destroys the within-day alignment and
    leaves the cross-sectional alignment intact, which is worth knowing when
    reading `cross_corr_frobenius` — part of that column is a statement about
    which day you are looking at, not about which bar.
    """
    real_c = cross_correlation_matrix(fit_series)[0, 1]
    shuf_c = cross_correlation_matrix(shuffled(fit_series, N_GEN))[0, 1]
    assert real_c > 0.5
    assert shuf_c < real_c / 3, f"shuffled kept corr(|r|, spread) = {shuf_c:.3f}"


def test_block_bootstrap_with_full_length_block_keeps_the_day_intact(fit_series):
    """block = 78 leaves exactly one block per day, so no splicing can happen.

    With the block as long as the session there is nothing to reassemble: each
    synthetic day is one real day read from a random circular start, i.e. a
    rotation. Nothing crosses between days and nothing crosses between channels,
    which is the invariant that says the block machinery is cutting and pasting
    the intended thing.
    """
    real = fit_series[:40]
    out = block_bootstrap(real, 24, block=BARS_PER_DAY)
    rotations = {
        np.ascontiguousarray(np.roll(real[j], k, axis=-1)).tobytes()
        for j in range(len(real))
        for k in range(BARS_PER_DAY)
    }
    for i in range(len(out)):
        day = np.ascontiguousarray(out[i]).tobytes()
        assert day in rotations, f"day {i} is not a rotation of any real day"


def test_block_bootstrap_preserves_cross_channel_alignment(fit_series):
    """Blocks are cut at the same offsets in all three channels.

    This is the whole difference between `block_bootstrap` and `shuffled`: one
    breaks time while keeping the channels in step, the other breaks both. If
    this ever fails, the two baselines stop being independent controls.
    """
    real_c = cross_correlation_matrix(fit_series)[0, 1]
    boot_c = cross_correlation_matrix(block_bootstrap(fit_series, N_GEN))[0, 1]
    assert abs(boot_c - real_c) < 0.1, f"real {real_c:.3f} vs block bootstrap {boot_c:.3f}"


def test_block_bootstrap_rejects_nonsense_block():
    with pytest.raises(ValueError, match="block must be"):
        block_bootstrap(np.zeros((4, 3, BARS_PER_DAY)), 4, block=0)


def test_garch_emits_strictly_positive_spread_and_volume(fit_series, garch_fit):
    """Both channels are exponentials of a Gaussian, so this is structural.

    Worth pinning anyway: it is the one property `iid_gaussian` deliberately
    violates, and a scorecard that could not tell the two apart would be
    reporting on a generator emitting negative traded volume.
    """
    out = garch_seasonal(fit_series, 128, fit=garch_fit)
    assert (out[:, 1] > 0).all()
    assert (out[:, 2] > 0).all()


def test_iid_gaussian_does_emit_negative_volume(fit_series):
    """The null model's documented failure, asserted so nobody 'fixes' it.

    Clipping volume at zero would hand the null model a free improvement on the
    volume marginal and hide the most basic defect a generator can have. The
    scorecard is supposed to expose that, so it has to survive.
    """
    out = iid_gaussian(fit_series, 128)
    assert (out[:, 2] < 0).any()


def test_garch_fit_uses_only_the_series_it_is_given(splits):
    """No leakage from the reference split into the fitted parameters.

    Cheap to check and worth checking: a baseline accidentally fitted on the data
    it is scored against would beat the diffusion model for a reason that has
    nothing to do with GARCH.
    """
    fit_series, reference = splits
    same = SeasonalGarchFit.fit(fit_series)
    other = SeasonalGarchFit.fit(reference)
    assert (same.omega, same.alpha, same.beta) != (other.omega, other.alpha, other.beta)
    # every empirical quantity carried into simulation comes from `fit_series`
    assert len(same.day_sigma) <= len(fit_series)
    np.testing.assert_allclose(
        np.sort(same.day_sigma),
        np.sort((fit_series[:, 0] / same.profile_ret).std(axis=-1)),
    )


# --- does the scorecard discriminate? ------------------------------------
#
# Every test below compares a baseline's distance against `real_train`'s — real
# training days scored as if they were a generator. That row is the noise floor:
# it is what "as good as it is possible to be" looks like given finite samples
# and a genuine train/reference shift. Comparing to zero instead would make every
# assertion a statement about sample size.

@pytest.fixture(scope="module")
def distances(splits, garch_fit):
    fit_series, reference = splits
    # One deseasonalising profile — the REFERENCE one — for every row, exactly as
    # `evaluate.scorecard` does it. Letting each generator divide out its own
    # bar-of-day |r| profile would cancel a wrong U-shape out of acf_abs_l2
    # instead of scoring it.
    profile = intraday_profile(reference, RET, absolute=True)
    facts = StylizedFacts.measure(reference, deseasonal_profile=profile)
    gens = _all_generators(fit_series, N_GEN, garch_fit)
    gens["real_train"] = fit_series
    return {
        k: compare(facts, StylizedFacts.measure(v, deseasonal_profile=profile))
        for k, v in gens.items()
    }


@pytest.fixture(scope="module")
def profile_only(splits, garch_fit):
    """`garch_seasonal` with the volatility clustering switched off at source.

    Setting alpha = beta = 0 makes the conditional variance recursion collapse to
    a constant h = omega, so the innovations become iid Student-t with the same
    nu — and since each simulated day is renormalised to unit variance afterwards,
    the level omega drops out entirely. Everything else is held fixed: the same
    fitted intraday profile, the same day-sigma pool, the same spread and volume
    overlays, and (because the label is unchanged) the same RNG stream, so the
    comparison is paired rather than two independent draws.

    Built through `dataclasses.replace` on the public fit object rather than by
    re-implementing `simulate`, so the control cannot drift away from the thing it
    is controlling for.
    """
    fit_series, reference = splits
    flat = dataclasses.replace(garch_fit, alpha=0.0, beta=0.0)
    synth = garch_seasonal(fit_series, N_GEN, fit=flat)
    profile = intraday_profile(reference, RET, absolute=True)
    return compare(
        StylizedFacts.measure(reference, deseasonal_profile=profile),
        StylizedFacts.measure(synth, deseasonal_profile=profile),
    )


def test_iid_gaussian_loses_badly_on_the_tail_index(distances):
    """The column that proves the tail metric measures tails.

    A Gaussian has no finite tail index, so the Hill estimator on iid Gaussian
    draws runs away — it lands an order of magnitude from the real value while
    every resampling baseline lands within a fraction of it. If this margin ever
    narrows, `tail_within_abs_err` has stopped separating fat from thin.
    """
    assert distances["iid_gaussian"]["tail_within_abs_err"] > \
        10 * distances["real_train"]["tail_within_abs_err"]


def test_iid_gaussian_loses_on_every_scored_column(distances):
    """The null model should be beaten by the noise floor on every column that works.

    Stated as a sweep rather than one column on purpose: a scorecard where iid
    noise ties the floor anywhere has a column that is not measuring a stylized
    fact. Two columns are excluded and both exclusions are findings in their own
    right, pinned by the two tests below: `acf_decay_abs_err` returns NaN for a
    generator with no decay to fit, and `acf_raw_l2` does not discriminate at all.
    """
    floor = distances["real_train"]
    null = distances["iid_gaussian"]
    for column, value in null.items():
        if column in ("acf_decay_abs_err", "acf_raw_l2"):
            continue
        assert value > floor[column], f"{column}: iid Gaussian ties or beats the noise floor"


def test_acf_raw_l2_does_not_discriminate(distances):
    """`acf_raw_l2` cannot separate a real resample from Gaussian noise, by design.

    `evaluate.py` already says so in a comment — "a generator can fake
    uncorrelated returns by emitting pure noise" — and this makes the statement
    executable. The real ACF of signed returns is ~-0.03 at lag 1 and ~0
    thereafter, so a generator that emits pure noise scores about as well as real
    data does, and the column ranks nothing. It is worth reporting (a generator
    with a strong positive return ACF has invented a trading strategy and should
    be caught) but it must never be summed into a total or read as a win.
    """
    spread = [distances[k]["acf_raw_l2"] for k in
              ("iid_gaussian", "shuffled", "block_bootstrap", "garch_seasonal", "real_train")]
    assert max(spread) < 3 * min(spread), (
        "acf_raw_l2 has started discriminating between generators; re-read the "
        f"assumption behind the scorecard. values={spread}"
    )


def test_acf_decay_is_nan_when_there_is_no_positive_acf_to_fit(distances, profile_only):
    """The blind spot in `evaluate.acf_decay_exponent`, and where deseasonalising moved it.

    The exponent is fitted over lags where the ACF is positive and above 1e-6,
    and needs at least five of them. When a generator has no positive ACF at all
    the filter keeps nothing — and not because the ACF hovers around zero and
    lands either side by chance. The mean within-day sample ACF of an
    uncorrelated series is biased to about -1/(T-1) = -0.013 at EVERY lag, since
    each 78-bar window is de-meaned by its own mean. All 30 lags come out
    negative, none survive, and the estimator returns NaN. A NaN in this column
    means "no power law to fit", not "not measured".

    Before `acf_abs_returns` deseasonalised, that NaN hit `iid_gaussian` and
    `shuffled` — the generators the column should have penalised hardest. It no
    longer does, and the reason is worth stating because it is the whole point of
    the change: dividing by the REAL bar-of-day profile injects an inverse-U
    seasonal into any generator that does not have the U-shape, which shows up as
    a large, slowly-decaying positive ACF (iid lands at ~0.11 flat across 30
    lags, against a real ~0.04 decaying). Those two now score finite and badly,
    which is correct.

    What returns NaN now is `profile_only` — GARCH with the recursion switched
    off but the profile kept. It has the right U-shape, so nothing is injected,
    and no clustering, so nothing is left: a flat, slightly negative ACF. That is
    the honest reading of this column post-change — NaN means "this generator has
    genuinely no within-day clustering", which under the old raw metric was
    unsayable.
    """
    assert np.isnan(profile_only["acf_decay_abs_err"])
    for name in ("iid_gaussian", "shuffled", "garch_seasonal"):
        assert np.isfinite(distances[name]["acf_decay_abs_err"]), name


def test_shuffled_matches_the_marginal(distances):
    """Permuting time cannot change the distribution of values. Sanity first.

    `shuffled` must land at or below the noise floor on the marginal columns,
    because its marginal is a real day's marginal exactly. A margin of 1.5x the
    floor allows for the reference split being different days than the fit split.
    """
    floor, shuf = distances["real_train"], distances["shuffled"]
    assert shuf["tail_within_abs_err"] < 1.5 * floor["tail_within_abs_err"]
    assert shuf["kurtosis_rel_err"] < 1.5 * floor["kurtosis_rel_err"]


def test_shuffled_loses_on_volatility_clustering(distances):
    """...and must lose badly on the ordering columns, having destroyed the order.

    Together with the test above, this is what makes `shuffled` a control rather
    than a baseline: it separates "the generator learned the marginal" from "the
    generator learned the dynamics", which one distance per stylized fact
    otherwise conflates. A model that beats `shuffled` on `acf_abs_l2` has
    learned something about time.

    Post-deseasonalisation the margin is thinner in absolute terms (~4x the floor
    rather than the ~10x the raw metric showed) and that is expected: the real
    deseasonalised ACF is only ~0.04 at lag 1, so there is less total signal for
    anyone to miss. It is also earned differently — shuffling destroys the
    U-shape as well as the ordering, so part of what it loses here is the
    inverse-U artefact that dividing by the real profile injects into a flat
    generator. Both failures are real; neither is the intraday profile being
    handed back as a clustering score.
    """
    floor, shuf = distances["real_train"], distances["shuffled"]
    assert shuf["acf_abs_l2"] > 3 * floor["acf_abs_l2"]


def test_shuffled_loses_on_the_intraday_profile(distances):
    """The U-shape is a statement about WHERE in the day a bar sits.

    Permuting within the day makes bar position uninformative, so the profile
    flattens to 1.0 everywhere and `intraday_vol_mse` should blow out by an order
    of magnitude relative to the floor.
    """
    floor, shuf = distances["real_train"], distances["shuffled"]
    assert shuf["intraday_vol_mse"] > 10 * floor["intraday_vol_mse"]
    assert shuf["intraday_volume_mse"] > 10 * floor["intraday_volume_mse"]


def test_block_bootstrap_fakes_clustering_by_pasting_wrong_time_of_day_blocks(distances):
    """The ordering this column reports REVERSED when the metric was deseasonalised.

    On the raw metric `block_bootstrap` beat `shuffled`: 30-minute blocks keep the
    ACF of |r| intact out to lag 5, and that read as partial credit. Against the
    deseasonalised metric it is the worst of the four — a lag-1 ACF of ~0.20
    against a real ~0.04, five times too much clustering rather than too little.

    The mechanism is the reason the ordering is not a bug. Blocks start at
    uniformly random bars, so a high-volatility opening block gets pasted into a
    quiet midday slot. Dividing by the real bar-of-day profile then scales that
    whole block up together, producing a run of large deseasonalised values —
    persistence manufactured out of misplaced seasonality. The raw metric could
    not see it because the same U-shape sat in both series and cancelled.

    So the column now penalises fake clustering as well as absent clustering,
    which is what a distance is supposed to do. `shuffled` remains the control for
    "no ordering at all"; this row is the control for "ordering pasted in the
    wrong place", and it is a harder failure, not an easier one.
    """
    floor, boot, shuf = (distances[k] for k in ("real_train", "block_bootstrap", "shuffled"))
    assert boot["acf_abs_l2"] > shuf["acf_abs_l2"]
    assert boot["acf_abs_l2"] > 4 * floor["acf_abs_l2"]


def test_block_bootstrap_still_loses_on_the_intraday_profile(distances):
    """Blocks start at uniformly random bars, so bar-of-day information is gone.

    This is the column that distinguishes it from a generator that models the
    session: preserving half an hour of local structure buys nothing at all on
    the U-shape.
    """
    floor, boot = distances["real_train"], distances["block_bootstrap"]
    assert boot["intraday_vol_mse"] > 10 * floor["intraday_vol_mse"]


def test_garch_beats_the_resampling_baselines_on_clustering(distances):
    """The strong classical baseline has to actually be strong, or it is not a bar.

    GARCH is the only one of the four that models volatility persistence, so it
    must beat both resampling baselines on `acf_abs_l2`. What this does NOT show
    is that the recursion is what wins it — most of the margin is the fitted
    intraday profile, and the test below is the one that measures the split. The
    original version of this test claimed the stronger thing in its docstring and
    passed identically with the recursion deleted, which is the reason the pair
    exists.
    """
    g = distances["garch_seasonal"]
    assert g["acf_abs_l2"] < distances["shuffled"]["acf_abs_l2"]
    assert g["acf_abs_l2"] < distances["block_bootstrap"]["acf_abs_l2"]


def test_garch_recursion_earns_a_real_share_of_acf_abs_l2(distances, profile_only):
    """The decomposition redone against the deseasonalised column.

    This test's previous form asserted the OPPOSITE bound and said so in its own
    failure message: on the raw metric the profile covered ~89% of the distance
    from `shuffled` to the noise floor and the recursion added ~5 points, because
    `acf_within_day` de-meaned each day by a scalar and left the bar-of-day
    U-shape inside the |r| series whose ACF it measured. On the real reference
    split, dividing |r| by the profile drops lag-1 from 0.244 to 0.042 — 83% of
    the raw quantity was seasonal. The message said the 4x bound was there "to
    fail if somebody ever deseasonalises the ACF"; somebody did, so here is the
    bound flipped.

    `profile_only` is the same generator with the recursion switched off at source
    (alpha = beta = 0 makes h constant, so the innovations are iid Student-t with
    the same nu, the same profile, the same day-sigma pool). Measured now:

      * `profile_only`'s deseasonalised ACF is flat and slightly negative — with
        the U-shape divided out, a constant-variance generator has no clustering
        left to show, which is the correct answer for it;
      * the recursion closes about a quarter of the whole `shuffled`-to-floor gap
        on its own, against ~5 points before.

    The profile term has not vanished, but its meaning has changed: it no longer
    hands `garch_seasonal` clustering it did not model, it spares it the
    inverse-U artefact that dividing by the real profile injects into a generator
    with the WRONG U-shape (see the block-bootstrap test). Getting the seasonal
    right is now worth avoiding a penalty, not collecting a reward.
    """
    g, floor, shuf = (distances[k] for k in ("garch_seasonal", "real_train", "shuffled"))
    recursion_gain = profile_only["acf_abs_l2"] - g["acf_abs_l2"]
    profile_gain = shuf["acf_abs_l2"] - profile_only["acf_abs_l2"]
    assert recursion_gain > 0, "the GARCH recursion no longer improves acf_abs_l2 at all"
    assert profile_gain < 4 * recursion_gain, (
        "acf_abs_l2 has gone back to being dominated by the intraday profile — "
        "check that `acf_abs_returns` is still deseasonalising and that the "
        "profile threaded into `StylizedFacts.measure` is the REFERENCE one. "
        f"profile_gain={profile_gain:.4f} recursion_gain={recursion_gain:.4f}"
    )
    assert recursion_gain > 0.15 * (shuf["acf_abs_l2"] - floor["acf_abs_l2"]), (
        "the recursion's share of the shuffled-to-floor gap has collapsed; the "
        "column is no longer crediting volatility persistence"
    )
    # And the ablation now has nothing to fit a decay exponent to — see
    # `test_acf_decay_is_nan_when_there_is_no_positive_acf_to_fit`.
    assert np.isnan(profile_only["acf_decay_abs_err"])
    assert g["acf_abs_l2"] > floor["acf_abs_l2"]


def test_garch_is_handed_the_intraday_profile(distances):
    """Asserted so the scorecard is never read as GARCH having discovered the U.

    The real |r| profile is multiplied back in at simulation time, so this column
    lands AT or BELOW the noise floor. That is a resampling artefact, not a
    result, and the test exists to make the fact executable rather than a comment
    somebody skips.
    """
    floor, g = distances["real_train"], distances["garch_seasonal"]
    assert g["intraday_vol_mse"] < 3 * floor["intraday_vol_mse"]
