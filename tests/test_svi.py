"""svi: the derivatives, g(k) against the density, the arbitrage checks and the recovery of
known params"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from ope.svi import (
    K_WIDE,
    SVIParams,
    calendar_checks,
    fit_svi,
    fit_svi_surface,
)
from ope.synthetic import synthetic_chain

SPX_SLICE = SVIParams(0.012, 0.08, -0.7, 0.02, 0.12)
# axel vogt's example in gatheral and jacquier 2014: looks fine, has butterfly arbitrage
VOGT = SVIParams(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)


def test_derivatives_match_finite_differences() -> None:
    k = np.linspace(-1.0, 0.6, 9)
    h = 1e-4
    w = SVI = SPX_SLICE
    fd1 = (np.asarray(w.w(k + h)) - np.asarray(w.w(k - h))) / (2 * h)
    fd2 = (np.asarray(w.w(k + h)) - 2 * np.asarray(w.w(k)) + np.asarray(w.w(k - h))) / h**2
    assert np.allclose(fd1, SVI.w1(k), atol=1e-7)
    assert np.allclose(fd2, SVI.w2(k), atol=1e-5)


def test_g_is_the_density_of_k() -> None:
    # breeden litzenberger: d2C/dK2 is the density of S_T, times K it is the density of k
    p = SPX_SLICE
    k = np.linspace(-0.8, 0.5, 7)
    K = np.exp(k)

    def call(strike: np.ndarray) -> np.ndarray:  # undiscounted, F = 1
        w = np.asarray(p.w(np.log(strike)))
        d1 = -np.log(strike) / np.sqrt(w) + np.sqrt(w) / 2
        return norm.cdf(d1) - strike * norm.cdf(d1 - np.sqrt(w))

    h = 1e-3
    density = K * (call(K + h) - 2 * call(K) + call(K - h)) / h**2
    w = np.asarray(p.w(k))
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2
    from_g = np.asarray(p.g(k)) / np.sqrt(2 * math.pi * w) * np.exp(-0.5 * d2**2)
    assert np.allclose(density / from_g, 1.0, atol=1e-4)


def test_vogts_example_has_butterfly_arbitrage() -> None:
    g = np.asarray(VOGT.g(K_WIDE))
    assert g.min() < -0.03
    neg = K_WIDE[g < 0]
    assert 0.6 < neg.min() < 0.7 and 1.2 < neg.max() < 1.3
    assert np.asarray(SPX_SLICE.g(K_WIDE)).min() > 0


def test_params_are_checked() -> None:
    with pytest.raises(ValueError, match="b >= 0"):
        SVIParams(0.01, -0.1, 0.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="below 0"):
        SVIParams(-0.1, 0.1, 0.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="finite"):
        SVIParams(math.nan, 0.1, 0.0, 0.0, 0.1)
    assert SPX_SLICE.min_variance == pytest.approx(0.012 + 0.08 * 0.12 * math.sqrt(1 - 0.49))
    assert SPX_SLICE.lee_ok and not SVIParams(0.01, 1.5, -0.5, 0.0, 0.1).lee_ok
    assert isinstance(SPX_SLICE.vol(0.0, 0.5), float)


def test_exact_recovery_without_noise() -> None:
    k = np.linspace(-0.45, 0.2, 25)
    fit = fit_svi(k, SPX_SLICE.vol(k, 0.5), 0.5)
    got = np.array([fit.params.a, fit.params.b, fit.params.rho, fit.params.m, fit.params.sigma])
    want = np.array([0.012, 0.08, -0.7, 0.02, 0.12])
    assert np.allclose(got, want, atol=1e-6)
    assert fit.rms < 1e-6 and not fit.penalized


@pytest.mark.parametrize("seed", range(10))
def test_the_curve_comes_back_through_noise(seed: int) -> None:
    # 0.3 vol points of noise: the params trade off (rho moves by up to 0.2 on this range), the
    # curve doesnt. 5 params on 25 points average the noise down to about 0.3 sqrt(5 / 25) =
    # 0.13 rms (0.06 to 0.20 measured over these seeds). the ends are the least pinned down:
    # up to 0.47 at k = -0.45, so the max is held to 2 noise sd
    k = np.linspace(-0.45, 0.2, 25)
    true = np.asarray(SPX_SLICE.vol(k, 0.5))
    noisy = true + np.random.default_rng(seed).normal(0, 0.003, k.size)
    fit = fit_svi(k, noisy, 0.5)
    err = (np.asarray(fit.params.vol(k, 0.5)) - true) * 100
    assert np.sqrt(np.mean(err**2)) < 0.3
    assert np.max(np.abs(err)) < 0.6
    assert fit.rms < 0.4


def test_arbitrage_triggers_the_penalty_and_removes_it() -> None:
    # vols from vogt's curve over the range where its g < 0: the plain fit copies the arbitrage
    k = np.linspace(0.4, 1.4, 21)
    fit = fit_svi(k, VOGT.vol(k, 1.0), 1.0)
    assert fit.penalized
    assert fit.min_g > -1e-3
    assert fit.k_lo < 0.4 and fit.k_hi > 1.4


def test_a_floor_keeps_the_next_slice_above() -> None:
    k = np.linspace(-0.3, 0.2, 15)
    first = fit_svi(k, SPX_SLICE.vol(k, 0.25), 0.25)
    # a later slice quoted w/ less total variance than the 1st: calendar arbitrage
    later = SVIParams(0.010, 0.08, -0.7, 0.02, 0.12)
    fit = fit_svi(k, later.vol(k, 0.3), 0.3, floor=first)
    assert fit.penalized
    gap = np.asarray(fit.params.w(fit.k_check())) - np.asarray(first.params.w(fit.k_check()))
    assert gap.min() > -1e-4


def test_inputs_are_checked() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        fit_svi([0.0, 0.1], [0.2, 0.2], 1.0)
    with pytest.raises(ValueError, match="T must be"):
        fit_svi(np.zeros(6), np.full(6, 0.2), 0.0)


def test_the_synthetic_surface_fits_clean() -> None:
    from ope.surface import iv_surface

    surface = iv_surface(synthetic_chain())
    fits = fit_svi_surface(surface)
    assert len(fits) == 7
    for f in fits:
        assert f.butterfly_ok and not f.penalized
        assert f.rms < 0.25 and f.within_spread == 1.0
    checks = calendar_checks(fits)
    assert len(checks) == 6 and all(c.ok for c in checks)
    assert all(math.isfinite(c.min_gap_wide) for c in checks)
