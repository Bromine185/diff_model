"""Preprocessing must be exactly invertible, or generated samples decode to nonsense."""

from __future__ import annotations

import numpy as np
import pytest

from diffmodel.seeding import fork
from diffmodel.transform import (
    ChannelTransform,
    DayTransform,
    arsinh,
    arsinh_inv,
    signed_power,
    signed_power_inv,
)


@pytest.fixture
def rng():
    return fork("test-transform")


# --- elementwise ----------------------------------------------------------

@pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0])
def test_signed_power_roundtrip(rng, p):
    x = rng.normal(size=5000) * 0.01
    assert np.max(np.abs(signed_power_inv(signed_power(x, p), p) - x)) < 1e-12


def test_signed_power_handles_negatives(rng):
    """The paper's literal (X-mu)^(1/p) is undefined here; the signed form is not.

    Half of all log returns are negative, so this is not an edge case.
    """
    x = np.array([-0.05, -1e-9, 0.0, 1e-9, 0.05])
    y = signed_power(x, 1.5)
    assert np.all(np.isfinite(y))
    assert np.all(np.sign(y) == np.sign(x)), "sign must be preserved"
    assert np.max(np.abs(signed_power_inv(y, 1.5) - x)) < 1e-15


def test_signed_power_is_monotone(rng):
    """Order must be preserved, or the transform reshuffles which day was volatile."""
    x = np.sort(rng.normal(size=1000))
    y = signed_power(x, 1.5)
    assert np.all(np.diff(y) >= -1e-15)


def test_signed_power_compresses_tails():
    """p > 1 should pull outliers in relative to the bulk — the reason it exists."""
    bulk, tail = 0.001, 0.05
    ratio_before = tail / bulk
    ratio_after = signed_power(np.array(tail), 1.5) / signed_power(np.array(bulk), 1.5)
    assert ratio_after < ratio_before


def test_arsinh_roundtrip(rng):
    x = np.abs(rng.normal(size=5000)) * 1e5
    assert np.max(np.abs(arsinh_inv(arsinh(x)) - x) / np.maximum(x, 1)) < 1e-10


def test_arsinh_defined_at_zero():
    """A 5-minute bar can legitimately have zero volume; log would blow up."""
    assert arsinh(np.array(0.0)) == 0.0
    assert np.isfinite(arsinh(np.array([0.0, 1e-30])).all())


# --- fitted channel -------------------------------------------------------

def test_channel_transform_roundtrip_unclipped(rng):
    x = rng.normal(size=(200, 78)) * 0.002
    t = ChannelTransform.fit(x, power=1.5)
    assert t.clipped_fraction(x) == 0.0
    assert np.max(np.abs(t.inverse(t.forward(x)) - x)) < 1e-12


def test_channel_transform_standardises(rng):
    x = rng.normal(size=(500, 78)) * 0.002 + 0.01
    t = ChannelTransform.fit(x, power=1.5)
    z = t.forward(x)
    assert abs(z.mean()) < 0.05
    assert z.std() == pytest.approx(1.0, abs=0.05)


def test_volume_channel_uses_arsinh(rng):
    v = np.abs(rng.normal(size=(200, 78))) * 1e5
    t = ChannelTransform.fit(v, power=1.0, use_arsinh=True)
    assert t.use_arsinh
    assert np.max(np.abs(t.inverse(t.forward(v)) - v) / np.maximum(v, 1)) < 1e-8


def test_winsorisation_clips_and_is_reported(rng):
    x = rng.normal(size=(500, 78))
    x[0, 0] = 500.0                       # a deliberate monster
    t = ChannelTransform.fit(x, power=1.0, winsor_sigma=3.0)
    z = t.forward(x)
    assert np.abs(z).max() <= 3.0 + 1e-9
    assert t.clipped_fraction(x) > 0, "clipping must be reported, not silent"


def test_inverse_clamps_wild_decoder_input(rng):
    """A generated image can contain anything; decoding must never produce inf/nan.

    Regression test for a real failure: an undertrained model emitted large
    values in the arsinh domain, `sinh` overflowed float64 (it does so around
    x = 710), generated volume became `inf`, and every stylized-fact metric
    downstream silently evaluated to `nan`.
    """
    v = np.abs(rng.normal(size=(100, 78))) * 1e5 + 10
    t = ChannelTransform.fit(v, power=1.0, use_arsinh=True, winsor_sigma=10.0)
    wild = np.array([[-1e6, -50.0, 0.0, 50.0, 1e6] * 15])[:, :78]
    out = t.inverse(wild)
    assert np.isfinite(out).all(), "decode must be finite for any input"


def test_inverse_clamp_is_symmetric_with_forward_winsorisation(rng):
    """Forward can never emit beyond +/-k sigma, so inverse should never accept beyond it."""
    x = rng.normal(size=(200, 78))
    t = ChannelTransform.fit(x, power=1.0, winsor_sigma=4.0)
    at_bound = t.inverse(np.full((1, 1), 4.0))
    beyond = t.inverse(np.full((1, 1), 400.0))
    assert np.allclose(at_bound, beyond), "values past the bound must decode to the bound"


def test_saturated_fraction_counts_clamped_inputs(rng):
    x = rng.normal(size=(100, 78))
    t = ChannelTransform.fit(x, power=1.0, winsor_sigma=3.0)
    y = np.concatenate([np.zeros((1, 50)), np.full((1, 50), 99.0)], axis=1)
    assert t.saturated_fraction(y) == pytest.approx(0.5)


def test_clamping_does_not_perturb_in_range_data(rng):
    """The guard must be inert on real data, or it would distort the round trip."""
    x = rng.normal(size=(500, 78)) * 0.002
    t = ChannelTransform.fit(x, power=1.5, winsor_sigma=10.0)
    assert t.saturated_fraction(t.forward(x)) == 0.0
    assert np.max(np.abs(t.inverse(t.forward(x)) - x)) < 1e-12


def test_saturated_report_names_all_channels(rng):
    s = _fake_day_series(rng)
    t = DayTransform.fit(s)
    rep = t.saturated_report(t.forward(s))
    assert set(rep) == {"returns", "spread", "volume"}
    assert all(v == 0.0 for v in rep.values())


def test_channel_transform_serialises(rng):
    x = rng.normal(size=(50, 78))
    t = ChannelTransform.fit(x, power=1.5)
    back = ChannelTransform.from_dict(t.to_dict())
    assert np.array_equal(back.forward(x), t.forward(x))


# --- the three-channel day transform --------------------------------------

def _fake_day_series(rng, n=300):
    ret = rng.normal(size=(n, 78)) * 0.002
    spread = np.abs(rng.normal(size=(n, 78))) * 0.001 + 1e-5
    vol = np.abs(rng.normal(size=(n, 78))) * 5e4 + 100
    return np.stack([ret, spread, vol], axis=1)


def test_day_transform_roundtrip(rng):
    s = _fake_day_series(rng)
    t = DayTransform.fit(s)
    back = t.inverse(t.forward(s))
    rel = np.abs(back - s) / np.maximum(np.abs(s), 1e-12)
    assert np.max(rel) < 1e-6


def test_day_transform_brings_channels_to_common_scale(rng):
    """Volume is ~1e5 and returns ~1e-3; the DDPM needs them comparable."""
    s = _fake_day_series(rng)
    z = DayTransform.fit(s).forward(s)
    stds = [z[:, i].std() for i in range(3)]
    assert all(0.8 < sd < 1.2 for sd in stds), stds


def test_day_transform_rejects_wrong_shape(rng):
    with pytest.raises(ValueError, match=r"expected \(B, 3, n\)"):
        DayTransform.fit(rng.normal(size=(10, 2, 78)))


def test_day_transform_serialises(rng):
    s = _fake_day_series(rng)
    t = DayTransform.fit(s)
    back = DayTransform.from_dict(t.to_dict())
    assert np.allclose(back.forward(s), t.forward(s))


def test_clipped_report_names_all_three_channels(rng):
    s = _fake_day_series(rng)
    rep = DayTransform.fit(s).clipped_report(s)
    assert set(rep) == {"returns", "spread", "volume"}
    assert all(v == 0.0 for v in rep.values())


def test_fit_uses_only_the_data_it_is_given(rng):
    """Fitting on train and applying to val must not peek at val statistics."""
    train = _fake_day_series(rng, 200)
    val = _fake_day_series(rng, 50) * 10.0     # deliberately different scale
    t = DayTransform.fit(train)
    assert t.ret.mean == pytest.approx(float(np.mean(train[:, 0])))
    # val is transformed by the TRAIN statistics, so it should not standardise to 1
    assert DayTransform.fit(train).forward(val)[:, 0].std() > 2.0
