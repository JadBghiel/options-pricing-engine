"""tier 5 validation: gbm paths and asian options

the geometric asian's closed form vs monte carlo over many fixed seeds (coverage of the 95%
interval), the arithmetic one w/ its control variate vs a naive simulation written here
"""

import math
from typing import Literal

import numpy as np
import pytest

from helpers import coverage
from ope.black_scholes import price
from ope.exotics import asian_geometric_price, asian_mc, gbm_paths
from ope.monte_carlo import MCResult

Option = Literal["call", "put"]
BASE = (100.0, 100.0, 1.0, 0.05, 0.2)  # S, K, T, r, sigma
Q = 0.02


# paths


def test_gbm_paths_start_at_spot_and_are_reproducible() -> None:
    a = gbm_paths(100, 1, 0.05, 0.2, n_dates=12, n_paths=5, rng=1)
    b = gbm_paths(100, 1, 0.05, 0.2, n_dates=12, n_paths=5, rng=np.random.default_rng(1))
    assert a.shape == (5, 13)
    np.testing.assert_array_equal(a[:, 0], 100.0)
    np.testing.assert_array_equal(a, b)


def test_gbm_paths_are_exact_at_every_date() -> None:
    # log S_t has mean log S + (r - q - sigma^2/2) t and variance sigma^2 t at each date
    S, T, r, sigma = 100.0, 2.0, 0.05, 0.3
    paths = gbm_paths(S, T, r, sigma, Q, n_dates=4, n_paths=400_000, rng=2)
    for i, t in enumerate(np.linspace(0.5, 2.0, 4), start=1):
        logs = np.log(paths[:, i])
        assert logs.mean() == pytest.approx(math.log(S) + (r - Q - 0.5 * sigma**2) * t, abs=4e-3)
        assert logs.var() == pytest.approx(sigma**2 * t, rel=0.01)


# asian


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("n_dates", [1, 12, 52])
def test_geometric_asian_mc_matches_its_closed_form(option: Option, n_dates: int) -> None:
    exact = asian_geometric_price(option, *BASE, Q, n_dates=n_dates)

    def run(seed: int) -> MCResult:
        return asian_mc(
            option, *BASE, Q, n_dates=n_dates, n_paths=4_000, average="geometric", rng=seed
        )

    assert 0.90 <= coverage(run, exact) <= 0.99


def test_geometric_asian_on_1_date_is_the_vanilla() -> None:
    for option in ("call", "put"):
        assert asian_geometric_price(option, *BASE, Q, n_dates=1) == pytest.approx(
            float(price(option, *BASE, Q)), rel=1e-12
        )


def test_geometric_asian_many_dates_tends_to_the_continuous_formula() -> None:
    # continuous average (kemna vorst): black scholes w/ vol sigma/sqrt(3) and yield
    # q' = (r + q + sigma^2/6) / 2
    S, K, T, r, sigma = BASE
    q_adj = 0.5 * (r + Q + sigma**2 / 6)
    continuous = float(price("call", S, K, T, r, sigma / math.sqrt(3), q_adj))
    assert asian_geometric_price("call", *BASE, Q, n_dates=100_000) == pytest.approx(
        continuous, rel=1e-4
    )


def test_geometric_asian_zero_vol_is_the_discounted_payoff_of_the_forward_average() -> None:
    S, K, T, r = 100.0, 95.0, 1.0, 0.05
    dates = np.arange(1, 13) / 12
    geo_avg = S * math.exp((r - Q) * dates.mean())
    expected = math.exp(-r * T) * (geo_avg - K)
    assert asian_geometric_price("call", S, K, T, r, 0.0, Q, n_dates=12) == pytest.approx(expected)


def naive_arithmetic_asian(option: Option, n_dates: int, n_paths: int, seed: int) -> MCResult:
    """plain monte carlo written independently: cumulative product of gross returns"""
    S, K, T, r, sigma = BASE
    dt = T / n_dates
    rng = np.random.Generator(np.random.PCG64(seed))  # a different stream from the library's
    growth = np.exp(
        (r - Q - sigma**2 / 2) * dt + sigma * math.sqrt(dt) * rng.normal(size=(n_paths, n_dates))
    )
    avg = (S * np.cumprod(growth, axis=1)).mean(axis=1)
    pay = np.maximum(avg - K, 0) if option == "call" else np.maximum(K - avg, 0)
    x = math.exp(-r * T) * pay
    return MCResult(float(x.mean()), float(x.std(ddof=1) / math.sqrt(n_paths)), n_paths)


@pytest.mark.parametrize("option", ["call", "put"])
def test_arithmetic_asian_with_control_variate_matches_a_naive_simulation(option: Option) -> None:
    naive = naive_arithmetic_asian(option, 12, 400_000, seed=12345)
    cv = asian_mc(option, *BASE, Q, n_dates=12, n_paths=100_000, control_variate=True, rng=8)
    assert abs(cv.estimate - naive.estimate) < 4 * math.hypot(cv.std_error, naive.std_error)


def test_arithmetic_asian_control_variate_interval_coverage() -> None:
    # reference: 1 big run on seeds the coverage runs dont use, its std error is 7% of theirs
    ref = asian_mc("call", *BASE, Q, n_dates=12, n_paths=1_000_000, control_variate=True, rng=10**6)

    def run(seed: int) -> MCResult:
        return asian_mc("call", *BASE, Q, n_dates=12, n_paths=5_000, control_variate=True, rng=seed)

    assert run(0).std_error > 10 * ref.std_error
    assert 0.90 <= coverage(run, ref.estimate) <= 0.99


def test_control_variate_cuts_the_standard_error_more_than_20x() -> None:
    gains = []
    for seed in range(10):
        plain = asian_mc("call", *BASE, n_dates=12, n_paths=50_000, rng=seed)
        cv = asian_mc("call", *BASE, n_dates=12, n_paths=50_000, control_variate=True, rng=seed)
        gains.append(plain.std_error / cv.std_error)
    assert min(gains) > 20  # about 36x on this option


def test_arithmetic_average_beats_geometric_path_by_path() -> None:
    # am >= gm on every path -> on the same draws the call is worth more and the put less
    kw = {"n_dates": 12, "n_paths": 10_000, "rng": 3}
    for option, sign in (("call", 1), ("put", -1)):
        arith = asian_mc(option, *BASE, **kw).estimate
        geo = asian_mc(option, *BASE, average="geometric", **kw).estimate
        assert sign * (arith - geo) > 0


def test_asian_is_cheaper_than_the_vanilla() -> None:
    # averaging cuts the volatility of what the payoff sees
    asian = asian_mc("call", *BASE, n_dates=12, n_paths=100_000, control_variate=True, rng=1)
    assert asian.estimate < 0.7 * float(price("call", *BASE))


# shared input checks


def test_asian_inputs_are_checked() -> None:
    with pytest.raises(ValueError, match="average"):
        asian_mc("call", *BASE, average="harmonic")
    with pytest.raises(ValueError, match="control variate"):
        asian_mc("call", *BASE, average="geometric", control_variate=True)
    with pytest.raises(ValueError, match="n_dates"):
        asian_mc("call", *BASE, n_dates=0)
    with pytest.raises(ValueError, match="n_paths"):
        asian_mc("call", *BASE, n_paths=1)
    with pytest.raises(ValueError, match="option"):
        asian_geometric_price("straddle", *BASE, n_dates=12)
