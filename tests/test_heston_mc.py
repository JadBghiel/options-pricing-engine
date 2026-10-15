"""tier 5 validation: heston by monte carlo

the interval's coverage over many seeds, and the discretization bias measured as a bias,
separately from the noise
"""

import math

import numpy as np
import pytest

from ope.heston import HestonParams, heston_price
from ope.heston_mc import heston_mc, heston_paths
from ope.monte_carlo import MCResult

# feller violated (2 kappa theta = 0.12 < xi^2 = 0.36): the euler variance goes negative often
P = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
S, K, T, R = 100.0, 100.0, 1.0, 0.05


# monte carlo


def test_mc_interval_holds_the_semi_analytical_price_about_95_percent_of_the_time() -> None:
    # 64 steps: the bias (about +0.03) is a seventh of the std error at 2,000 paths -> this
    # tests the noise, the next test measures the bias on its own
    exact = heston_price("call", S, K, T, R, P)
    inside = 0
    for seed in range(200):
        lo, hi = heston_mc("call", S, K, T, R, P, n_steps=64, n_paths=2_000, rng=seed).ci95
        inside += lo <= exact <= hi
    assert 0.90 <= inside / 200 <= 0.99


def test_discretization_bias_is_real_and_shrinks_with_steps() -> None:
    exact = heston_price("call", S, K, T, R, P)
    few = heston_mc("call", S, K, T, R, P, n_steps=4, n_paths=200_000, rng=1_000)
    many = heston_mc("call", S, K, T, R, P, n_steps=64, n_paths=200_000, rng=2_000)
    # 4 steps: about +0.71, far outside the noise -> a bias, bad luck cant explain it
    assert few.estimate - exact > 10 * few.std_error
    assert few.estimate - exact > 0.5
    assert abs(many.estimate - exact) < 4 * many.std_error


@pytest.mark.parametrize("option", ["call", "put"])
def test_mc_agrees_with_the_semi_analytical_price_for_calls_and_puts(option: str) -> None:
    p = HestonParams(0.06, 2.0, 0.05, 0.4, -0.5)
    exact = heston_price(option, S, 95.0, 0.5, R, p, 0.01)  # type: ignore[arg-type]
    res = heston_mc(
        option, S, 95.0, 0.5, R, p, 0.01, n_steps=100, n_paths=200_000,  # type: ignore[arg-type]
        antithetic=True, control_variate=True, rng=3,
    )  # fmt: skip
    assert abs(res.estimate - exact) < 4 * res.std_error


def test_discounted_stock_stays_a_martingale_with_truncation() -> None:
    # E[e^{-(r-q)T} S_T] = S even though v goes below 0 on many steps
    s, v = heston_paths(S, T, R, P, 0.02, n_steps=16, n_paths=200_000, rng=4)
    assert (v < 0).mean() > 0.1
    discounted = s[:, -1] * math.exp(-(R - 0.02) * T)
    se = discounted.std(ddof=1) / math.sqrt(discounted.size)
    assert abs(discounted.mean() - S) < 4 * se


def test_variance_reduction_cuts_the_standard_error() -> None:
    plain = heston_mc("call", S, 90.0, T, R, P, n_paths=50_000, rng=5)
    reduced = heston_mc(
        "call", S, 90.0, T, R, P, n_paths=50_000, antithetic=True, control_variate=True, rng=5
    )
    assert plain.std_error / reduced.std_error > 1.5


def test_paths_shape_and_reproducibility() -> None:
    s, v = heston_paths(S, T, R, P, n_steps=10, n_paths=3, rng=7)
    s2, _ = heston_paths(S, T, R, P, n_steps=10, n_paths=3, rng=np.random.default_rng(7))
    assert s.shape == v.shape == (3, 11)
    np.testing.assert_array_equal(s[:, 0], S)
    np.testing.assert_array_equal(v[:, 0], P.v0)
    np.testing.assert_array_equal(s, s2)
    assert np.all(s > 0)


def test_mc_same_seed_same_result() -> None:
    a = heston_mc("put", S, K, T, R, P, n_steps=10, n_paths=1_000, rng=3)
    b = heston_mc("put", S, K, T, R, P, n_steps=10, n_paths=1_000, rng=np.random.default_rng(3))
    assert a == b
    assert isinstance(a, MCResult)


def test_mc_inputs_are_checked() -> None:
    with pytest.raises(ValueError, match="even"):
        heston_mc("call", S, K, T, R, P, n_paths=1_001, antithetic=True)
    with pytest.raises(ValueError, match="pairs"):
        heston_mc("call", S, K, T, R, P, n_paths=2, antithetic=True)
    with pytest.raises(ValueError, match="n_steps"):
        heston_mc("call", S, K, T, R, P, n_steps=0)
    with pytest.raises(ValueError, match="n_paths"):
        heston_mc("call", S, K, T, R, P, n_paths=1)
    with pytest.raises(ValueError, match="strike"):
        heston_mc("call", S, 0.0, T, R, P)
    with pytest.raises(ValueError, match=">= 1"):
        heston_paths(S, T, R, P, n_steps=0, n_paths=1)
