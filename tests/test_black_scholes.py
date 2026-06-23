"""tier 1 validation: black scholes prices and greeks vs references and consistency checks

reference values come from an independent script (plain math + scipy.stats), not the book
"""

import math
from typing import Literal

import numpy as np
import pytest

from ope.black_scholes import (
    call_price,
    d1_d2,
    greeks,
    norm_cdf,
    norm_pdf,
    price,
    price_bounds,
    put_price,
)

Option = Literal["call", "put"]


def _norm_cdf_erf(x: float) -> float:
    # independent of scipy. fine in absolute terms, but in the far left tail 1 + erf(x)
    # cancels to 0 and loses every significant digit -> the library uses scipy ndtr
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _random_params(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "S": rng.uniform(50, 150, n),
        "K": rng.uniform(50, 150, n),
        "T": rng.uniform(0.1, 2.0, n),
        "r": rng.uniform(-0.01, 0.10, n),
        "sigma": rng.uniform(0.1, 0.6, n),
        "q": rng.uniform(0.0, 0.05, n),
    }


# normal distribution


def test_norm_cdf_matches_erf_version_on_minus_8_to_8() -> None:
    xs = np.linspace(-8, 8, 1601)
    ref = np.array([_norm_cdf_erf(x) for x in xs])
    np.testing.assert_allclose(norm_cdf(xs), ref, rtol=0, atol=1e-12)


def test_norm_pdf_is_the_slope_of_the_cdf() -> None:
    xs = np.linspace(-5, 5, 101)
    h = 1e-5
    slope = (norm_cdf(xs + h) - norm_cdf(xs - h)) / (2 * h)
    # rounding in the difference is ~eps/h = 1e-11, so the absolute floor sits above it
    np.testing.assert_allclose(norm_pdf(xs), slope, rtol=1e-8, atol=1e-10)
    assert float(norm_pdf(0.0)) == pytest.approx(1 / math.sqrt(2 * math.pi), abs=1e-15)


# reference values


def test_hull_call_and_put() -> None:
    # Hull, the Black-Scholes-Merton model: S=42, K=40, T=0.5, r=10%, sigma=20%
    assert call_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(4.7594223929, abs=1e-9)
    assert put_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(0.8085993729, abs=1e-9)
    assert round(float(call_price(42, 40, 0.5, 0.10, 0.20)), 2) == 4.76
    assert round(float(put_price(42, 40, 0.5, 0.10, 0.20)), 2) == 0.81


def test_hull_d1_d2() -> None:
    d1, d2 = d1_d2(42, 40, 0.5, 0.10, 0.20)
    assert d1 == pytest.approx(0.7692626281, abs=1e-9)
    assert d2 == pytest.approx(0.6278412719, abs=1e-9)


def test_hull_greeks_example() -> None:
    # Hull, the Greek letters: S=49, K=50, T=20/52, r=5%, sigma=20%
    args = (49, 50, 20 / 52, 0.05, 0.20)
    g = greeks("call", *args)
    assert call_price(*args) == pytest.approx(2.4005273233, abs=1e-9)
    assert g.delta == pytest.approx(0.5216046611, abs=1e-9)
    assert g.gamma == pytest.approx(0.0655440393, abs=1e-9)
    assert g.vega == pytest.approx(12.1054798826, abs=1e-8)
    assert g.theta == pytest.approx(-4.3053298229, abs=1e-8)
    assert g.rho == pytest.approx(8.9069619496, abs=1e-8)


def test_dividend_yield_reference() -> None:
    args = (100, 95, 0.75, 0.03, 0.25, 0.02)
    g = greeks("call", *args)
    assert call_price(*args) == pytest.approx(11.3631718658, abs=1e-9)
    assert put_price(*args) == pytest.approx(5.7383454389, abs=1e-9)
    assert g.delta == pytest.approx(0.6383091358, abs=1e-9)
    assert g.theta == pytest.approx(-5.5751867040, abs=1e-8)
    assert g.rho == pytest.approx(39.3508062834, abs=1e-8)


# consistency checks


def test_put_call_parity_on_500_random_sets() -> None:
    p = _random_params(500, seed=1)
    lhs = call_price(**p) - put_price(**p)
    rhs = p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"])
    np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-10)


def test_gamma_and_vega_are_the_same_for_calls_and_puts() -> None:
    p = _random_params(200, seed=2)
    c, pu = greeks("call", **p), greeks("put", **p)
    np.testing.assert_allclose(c.gamma, pu.gamma, rtol=1e-12)
    np.testing.assert_allclose(c.vega, pu.vega, rtol=1e-12)


@pytest.mark.parametrize("option", ["call", "put"])
def test_greeks_match_central_finite_differences(option: Option) -> None:
    p = _random_params(200, seed=3)
    S, K, T, r, sigma, q = (p[k] for k in ("S", "K", "T", "r", "sigma", "q"))
    g = greeks(option, S, K, T, r, sigma, q)

    def v(**bump: np.ndarray) -> np.ndarray:
        args = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q} | bump
        return np.asarray(price(option, **args))

    # bumps scaled to each input. gamma is a 2nd difference, the most sensitive to h
    hS, hT, hs, hr = 1e-4 * S, 1e-4 * T, 1e-4 * sigma, 1e-5
    fd = {
        "delta": (v(S=S + hS) - v(S=S - hS)) / (2 * hS),
        "gamma": (v(S=S + hS) - 2 * v() + v(S=S - hS)) / hS**2,
        "vega": (v(sigma=sigma + hs) - v(sigma=sigma - hs)) / (2 * hs),
        "theta": -(v(T=T + hT) - v(T=T - hT)) / (2 * hT),
        "rho": (v(r=r + hr) - v(r=r - hr)) / (2 * hr),
    }
    for name, approx in fd.items():
        np.testing.assert_allclose(getattr(g, name), approx, rtol=1e-4, atol=1e-8, err_msg=name)


def test_prices_stay_inside_no_arbitrage_bounds() -> None:
    p = _random_params(500, seed=4)
    for option in ("call", "put"):
        value = np.asarray(price(option, **p))
        lower, upper = price_bounds(option, p["S"], p["K"], p["T"], p["r"], p["q"])
        assert np.all(value >= np.asarray(lower) - 1e-12)
        assert np.all(value <= np.asarray(upper) + 1e-12)


def test_monotonic_in_spot_and_vol() -> None:
    spots = np.linspace(50, 150, 201)
    vols = np.linspace(0.05, 1.0, 201)
    assert np.all(np.diff(np.asarray(call_price(spots, 100, 1, 0.03, 0.2))) > 0)
    assert np.all(np.diff(np.asarray(put_price(spots, 100, 1, 0.03, 0.2))) < 0)
    assert np.all(np.diff(np.asarray(call_price(100, 100, 1, 0.03, vols))) > 0)
    assert np.all(np.diff(np.asarray(put_price(100, 100, 1, 0.03, vols))) > 0)


# degenerate inputs


def test_expiry_gives_intrinsic_value() -> None:
    spots = np.array([30.0, 40.0, 50.0])
    np.testing.assert_allclose(call_price(spots, 40, 0, 0.05, 0.2), [0, 0, 10])
    np.testing.assert_allclose(put_price(spots, 40, 0, 0.05, 0.2), [10, 0, 0])


def test_zero_vol_gives_discounted_intrinsic_on_the_forward() -> None:
    S, K, T, r, q = np.array([90.0, 100.0, 110.0]), 100.0, 1.0, 0.05, 0.02
    fwd_intrinsic = S * math.exp(-q * T) - K * math.exp(-r * T)
    np.testing.assert_allclose(call_price(S, K, T, r, 0, q), np.maximum(fwd_intrinsic, 0))
    np.testing.assert_allclose(put_price(S, K, T, r, 0, q), np.maximum(-fwd_intrinsic, 0))


def test_zero_spot() -> None:
    assert call_price(0, 100, 1, 0.05, 0.2) == 0.0
    assert put_price(0, 100, 1, 0.05, 0.2) == pytest.approx(100 * math.exp(-0.05))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("option", ["call", "put"])
def test_degenerate_greeks_are_finite_and_deterministic(option: Option) -> None:
    S = np.array([0.0, 90.0, 110.0, 90.0, 110.0])
    T = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
    sigma = np.array([0.2, 0.2, 0.2, 0.0, 0.0])
    g = greeks(option, S, 100, T, 0.05, sigma)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert np.all(np.isfinite(getattr(g, name))), name
    np.testing.assert_array_equal(g.gamma, 0.0)
    np.testing.assert_array_equal(g.vega, 0.0)
    itm = (
        [False, False, True, False, True] if option == "call" else [True, True, False, True, False]
    )
    np.testing.assert_allclose(np.abs(g.delta), np.where(itm, 1.0, 0.0))


# api


def test_vectorized_over_strikes_matches_scalar_calls() -> None:
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vec = np.asarray(call_price(100, strikes, 0.5, 0.03, 0.25))
    assert vec.shape == (5,)
    np.testing.assert_allclose(vec, [call_price(100, k, 0.5, 0.03, 0.25) for k in strikes])


def test_scalar_inputs_return_python_floats() -> None:
    assert isinstance(call_price(100, 100, 1, 0.05, 0.2), float)
    assert isinstance(greeks("put", 100, 100, 1, 0.05, 0.2).delta, float)
    assert price("put", 42, 40, 0.5, 0.10, 0.20) == put_price(42, 40, 0.5, 0.10, 0.20)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((-1, 100, 1, 0.05, 0.2), "spot"),
        ((100, 0, 1, 0.05, 0.2), "strike"),
        ((100, 100, -1, 0.05, 0.2), "maturity"),
        ((100, 100, 1, 0.05, -0.2), "volatility"),
        ((100, 100, 1, math.nan, 0.2), "finite"),
    ],
)
def test_invalid_inputs_raise(args: tuple[float, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call_price(*args)


def test_invalid_option_type_raises() -> None:
    with pytest.raises(ValueError, match="call' or 'put"):
        price("straddle", 100, 100, 1, 0.05, 0.2)  # type: ignore[arg-type]
