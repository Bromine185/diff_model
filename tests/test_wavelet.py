"""The wavelet contract: every forward has an exact inverse.

If these fail, nothing downstream means anything — a generated image would decode
to the wrong time series and the stylized-fact scorecard would be measuring a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from diffmodel.config import BARS_PER_DAY, IMAGE_ROWS, N_LEVELS, PADDED_LEN
from diffmodel.seeding import fork
from diffmodel.wavelet import (
    apply_level_stats,
    fit_level_stats,
    image_to_series,
    imodwt,
    invert_level_stats,
    modwt,
    pad_to_pow2,
    series_to_image,
    unpad,
)

TOL = 1e-10


@pytest.fixture
def rng():
    return fork("test-wavelet")


# --- padding --------------------------------------------------------------

@pytest.mark.parametrize("mode", ["mirror", "zero", "periodic"])
def test_pad_then_unpad_is_identity(rng, mode):
    x = rng.normal(size=(4, 3, BARS_PER_DAY))
    padded = pad_to_pow2(x, PADDED_LEN, mode=mode)
    assert padded.shape[-1] == PADDED_LEN
    assert np.array_equal(unpad(padded, BARS_PER_DAY), x)


def test_mirror_padding_reflects_without_repeating_edge(rng):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = pad_to_pow2(x, 8, mode="mirror")
    # [a b c d e] -> [a b c d e | d c b]  (edge sample not duplicated)
    np.testing.assert_array_equal(out, [1, 2, 3, 4, 5, 4, 3, 2])


def test_mirror_padding_rejects_impossible_request(rng):
    with pytest.raises(ValueError, match="mirror padding needs"):
        pad_to_pow2(rng.normal(size=4), 16, mode="mirror")  # pad 12 >= n 4


# --- MODWT round trip -----------------------------------------------------

@pytest.mark.parametrize("levels", [1, 3, 7])
def test_modwt_roundtrip_1d(rng, levels):
    x = rng.normal(size=PADDED_LEN)
    err = np.max(np.abs(imodwt(modwt(x, levels)) - x))
    assert err < TOL, f"levels={levels} roundtrip error {err:.3e}"


def test_modwt_roundtrip_batched(rng):
    x = rng.normal(size=(9, 3, PADDED_LEN))
    err = np.max(np.abs(imodwt(modwt(x, N_LEVELS)) - x))
    assert err < TOL, f"batched roundtrip error {err:.3e}"


def test_modwt_plane_shape(rng):
    x = rng.normal(size=(5, 3, PADDED_LEN))
    planes = modwt(x, N_LEVELS)
    assert planes.shape == (5, 3, IMAGE_ROWS, PADDED_LEN)


def test_modwt_is_shift_equivariant(rng):
    """Shifting the input circularly shifts every coefficient plane identically.

    This is the property a decimated DWT lacks and the reason we use MODWT: a
    volatility burst arriving five minutes later should slide along a row, not
    land in a different set of coefficients.
    """
    x = rng.normal(size=PADDED_LEN)
    shift = 7
    a = np.roll(modwt(x, N_LEVELS), shift, axis=-1)
    b = modwt(np.roll(x, shift), N_LEVELS)
    assert np.max(np.abs(a - b)) < TOL


def test_details_sum_with_smooth_to_original(rng):
    """MODWT is an additive decomposition: sum of all planes == the signal.

    A direct consequence of V_j + W_j = V_{j-1} applied down the cascade.
    """
    x = rng.normal(size=PADDED_LEN)
    planes = modwt(x, N_LEVELS)
    assert np.max(np.abs(planes.sum(axis=-2) - x)) < TOL


def test_constant_signal_has_zero_detail(rng):
    """A flat series has no variation at any scale; all energy sits in the smooth."""
    x = np.full(PADDED_LEN, 3.7)
    planes = modwt(x, N_LEVELS)
    assert np.max(np.abs(planes[:-1])) < TOL          # details
    assert np.max(np.abs(planes[-1] - 3.7)) < TOL     # smooth


def test_modwt_rejects_too_many_levels(rng):
    with pytest.raises(ValueError, match="levels="):
        modwt(rng.normal(size=8), levels=5)  # 2^5 = 32 > 8


@pytest.mark.parametrize("levels", [4, 6, N_LEVELS])
def test_matches_pywavelets_swt(rng, levels):
    """Cross-check the hand-rolled cascade against PyWavelets once.

    A validation, never a runtime dependency. Three conventions differ and all
    three are reconciled here rather than papered over with a loose tolerance:

      scale  pywt's `norm=False` Haar keeps filters at 1/sqrt(2) per level, so its
             level-j coefficients carry a factor 2^(j/2) relative to the MODWT
             convention (filters at 1/2).
      sign   pywt's Haar `dec_hi` is [-1/sqrt2, +1/sqrt2]; ours is [+1/2, -1/2],
             so every detail plane is negated. The smooth plane is not.
      phase  pywt's circular alignment lags ours by 2^j - 1 samples.

    Agreement is then exact to floating point (~2e-16), not merely close.
    """
    pywt = pytest.importorskip("pywt")
    x = rng.normal(size=PADDED_LEN)

    ours = modwt(x, levels)
    theirs = pywt.swt(x, "haar", level=levels, norm=False, trim_approx=False)

    # pywt returns [(cA_J, cD_J), ..., (cA_1, cD_1)] — coarsest first.
    for j in range(1, levels + 1):
        _, cd = theirs[levels - j]
        expected = -np.roll(cd / (2.0 ** (j / 2)), (1 << j) - 1)
        err = np.max(np.abs(expected - ours[j - 1]))
        assert err < 1e-12, f"detail level {j} mismatch: {err:.3e}"

    ca, _ = theirs[0]
    expected_smooth = np.roll(ca / (2.0 ** (levels / 2)), (1 << levels) - 1)
    err = np.max(np.abs(expected_smooth - ours[levels]))
    assert err < 1e-12, f"smooth plane mismatch: {err:.3e}"


# --- series <-> image -----------------------------------------------------

def test_series_to_image_roundtrip(rng):
    series = rng.normal(size=(6, 3, BARS_PER_DAY)) * 0.01
    img = series_to_image(series, N_LEVELS, PADDED_LEN, pad_mode="mirror")
    assert img.shape == (6, 3, IMAGE_ROWS, PADDED_LEN)
    err = np.max(np.abs(image_to_series(img, BARS_PER_DAY) - series))
    assert err < TOL, f"series/image roundtrip error {err:.3e}"


@pytest.mark.parametrize("mode", ["mirror", "zero", "periodic"])
def test_series_to_image_roundtrip_all_pad_modes(rng, mode):
    series = rng.normal(size=(3, 3, BARS_PER_DAY))
    img = series_to_image(series, N_LEVELS, PADDED_LEN, pad_mode=mode)
    assert np.max(np.abs(image_to_series(img, BARS_PER_DAY) - series)) < TOL


# --- level standardisation ------------------------------------------------

def test_level_stats_roundtrip(rng):
    imgs = rng.normal(size=(32, 3, IMAGE_ROWS, PADDED_LEN)) * np.array([1.0, 5.0, 20.0])[:, None, None]
    stats = fit_level_stats(imgs)
    back = invert_level_stats(apply_level_stats(imgs, stats), stats)
    assert np.max(np.abs(back - imgs)) < 1e-8


def test_level_stats_normalises_scale(rng):
    """After standardisation every (channel, level) plane should sit near unit scale/3."""
    imgs = rng.normal(size=(64, 3, IMAGE_ROWS, PADDED_LEN)) * 7.0 + 2.0
    stats = fit_level_stats(imgs)
    z = apply_level_stats(imgs, stats, scale=3.0)
    assert abs(z.mean()) < 0.05
    assert 0.25 < z.std() < 0.42          # 1/3 by construction


def test_level_stats_shapes(rng):
    imgs = rng.normal(size=(10, 3, IMAGE_ROWS, PADDED_LEN))
    stats = fit_level_stats(imgs)
    assert stats["mean"].shape == (3, IMAGE_ROWS, 1)
    assert stats["std"].shape == (3, IMAGE_ROWS, 1)


# --- the full chain -------------------------------------------------------

def test_full_chain_roundtrip(rng):
    """series -> image -> standardise -> invert -> series, end to end."""
    series = rng.normal(size=(20, 3, BARS_PER_DAY)) * 0.005
    img = series_to_image(series, N_LEVELS, PADDED_LEN)
    stats = fit_level_stats(img)
    z = apply_level_stats(img, stats)
    back = image_to_series(invert_level_stats(z, stats), BARS_PER_DAY)
    assert np.max(np.abs(back - series)) < 1e-9
