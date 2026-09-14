"""tier 5 validation: asian, barrier and lookback options

closed forms are checked against monte carlo over many fixed seeds (coverage of the 95%
interval), discrete monitoring only against discrete prices, and the continuous closed forms only
against the brownian bridge simulation, which watches the path between dates too
"""

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
from scipy.special import ndtr, zeta

from ope.black_scholes import price
from ope.exotics import (
    BGK_BETA,
    BarrierKind,
    asian_geometric_price,
    asian_mc,
    barrier_mc,
    barrier_price,
    gbm_paths,
    lookback_mc,
    lookback_price,
)
from ope.monte_carlo import MCResult

Option = Literal["call", "put"]
BASE = (100.0, 100.0, 1.0, 0.05, 0.2)  # S, K, T, r, sigma
Q = 0.02


def coverage(run: Callable[[int], MCResult], exact: float, seeds: int = 200) -> float:
    """share of seeds whose 95% interval holds exact"""
    inside = 0
    for seed in range(seeds):
        lo, hi = run(seed).ci95
        inside += lo <= exact <= hi
    return inside / seeds


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
    rng = np.random.Generator(np.random.PCG64(seed))  # not the library's stream
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
    # am >= gm on every path, so on the same draws the call is worth more and the put less
    kw = {"n_dates": 12, "n_paths": 10_000, "rng": 3}
    for option, sign in (("call", 1), ("put", -1)):
        arith = asian_mc(option, *BASE, **kw).estimate
        geo = asian_mc(option, *BASE, average="geometric", **kw).estimate
        assert sign * (arith - geo) > 0


def test_asian_is_cheaper_than_the_vanilla() -> None:
    # averaging cuts the volatility of what the payoff sees
    asian = asian_mc("call", *BASE, n_dates=12, n_paths=100_000, control_variate=True, rng=1)
    assert asian.estimate < 0.7 * float(price("call", *BASE))


# barrier


def test_bgk_constant() -> None:
    assert pytest.approx(-zeta(0.5) / math.sqrt(2 * math.pi), rel=1e-15) == BGK_BETA


BARRIER_CASES = [
    # option, kind, K, H: every branch of the closed form
    ("call", "down-and-out", 100.0, 90.0),
    ("call", "down-and-out", 90.0, 95.0),
    ("call", "up-and-out", 100.0, 120.0),
    ("put", "up-and-out", 100.0, 110.0),
    ("put", "up-and-out", 110.0, 105.0),
    ("put", "down-and-out", 110.0, 85.0),
]


@pytest.mark.parametrize(("option", "kind", "K", "H"), BARRIER_CASES)
def test_barrier_closed_form_matches_the_bridge_simulation(
    option: Option, kind: BarrierKind, K: float, H: float
) -> None:
    exact = barrier_price(option, 100, K, H, 1, 0.05, 0.2, Q, kind=kind)
    res = barrier_mc(
        option, 100, K, H, 1, 0.05, 0.2, Q, kind=kind,
        n_dates=20, n_paths=200_000, monitoring="continuous", rng=21,
    )  # fmt: skip
    assert abs(res.estimate - exact) < 4 * res.std_error


def test_bridge_simulation_interval_coverage() -> None:
    # the bridge is exact whatever n_dates is, so 10 dates are enough
    exact = barrier_price("call", 100, 100, 120, 1, 0.05, 0.2, kind="up-and-out")

    def run(seed: int) -> MCResult:
        return barrier_mc(
            "call", 100, 100, 120, 1, 0.05, 0.2, kind="up-and-out",
            n_dates=10, n_paths=4_000, monitoring="continuous", rng=seed,
        )  # fmt: skip

    assert 0.90 <= coverage(run, exact) <= 0.99


def test_barrier_checked_once_at_expiry_is_a_call_spread_minus_a_digital() -> None:
    # up-and-out call watched only at T pays (S_T - K) when K < S_T < H:
    # call(K) - call(H) - (H - K) e^{-rT} N(d2(H)), exact
    S, K, H, T, r, sigma = 100.0, 100.0, 120.0, 1.0, 0.05, 0.2
    d2 = (math.log(S / H) + (r - Q - sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    exact = (
        float(price("call", S, K, T, r, sigma, Q))
        - float(price("call", S, H, T, r, sigma, Q))
        - (H - K) * math.exp(-r * T) * float(ndtr(d2))
    )

    def run(seed: int) -> MCResult:
        return barrier_mc(
            "call", S, K, H, T, r, sigma, Q, kind="up-and-out", n_dates=1, n_paths=4_000, rng=seed
        )

    assert 0.90 <= coverage(run, exact) <= 0.99


@pytest.mark.parametrize(("kind", "H"), [("up-and-out", 120.0), ("down-and-out", 90.0)])
def test_discrete_barrier_sits_between_the_vanilla_and_the_continuous_price(
    kind: BarrierKind, H: float
) -> None:
    # same seed -> same paths: a path alive under continuous watch is alive under daily checks
    kw = {"kind": kind, "n_dates": 252, "n_paths": 50_000, "rng": 4}
    discrete = barrier_mc("call", 100, 100, H, 1, 0.05, 0.2, **kw)
    cont = barrier_mc("call", 100, 100, H, 1, 0.05, 0.2, monitoring="continuous", **kw)
    assert cont.estimate < discrete.estimate < float(price("call", *BASE))


@pytest.mark.parametrize(("kind", "H"), [("up-and-out", 120.0), ("down-and-out", 90.0)])
def test_bgk_shift_approximates_the_daily_barrier_to_1_percent(kind: BarrierKind, H: float) -> None:
    approx = barrier_price("call", 100, 100, H, 1, 0.05, 0.2, kind=kind, n_dates=252)
    continuous = barrier_price("call", 100, 100, H, 1, 0.05, 0.2, kind=kind)
    daily = barrier_mc(
        "call", 100, 100, H, 1, 0.05, 0.2, kind=kind,
        n_dates=252, n_paths=200_000, rng=5,
    )  # fmt: skip
    assert abs(approx - daily.estimate) < 0.01 * daily.estimate + 4 * daily.std_error
    # and the gap it corrects is far bigger than that (12% for the up-and-out)
    assert abs(continuous - daily.estimate) > 0.03 * daily.estimate


def test_far_barrier_is_the_vanilla_and_impossible_payoffs_are_zero() -> None:
    S, K, T, r, sigma = BASE
    far_barriers: list[tuple[BarrierKind, float]] = [("up-and-out", 1e6), ("down-and-out", 1e-6)]
    for option in ("call", "put"):
        vanilla = float(price(option, *BASE, Q))
        for kind, H in far_barriers:
            far = barrier_price(option, S, K, H, T, r, sigma, Q, kind=kind)
            assert far == pytest.approx(vanilla, rel=1e-9)
    # up-and-out call w/ H <= K, down-and-out put w/ H >= K: the payoff zone is past the barrier
    assert barrier_price("call", S, 130, 120, T, r, sigma, kind="up-and-out") == 0.0
    assert barrier_price("put", S, 90, 95, T, r, sigma, kind="down-and-out") == 0.0


def test_barrier_inputs_are_checked() -> None:
    with pytest.raises(ValueError, match="already at or past"):
        barrier_price("call", 100, 100, 100, 1, 0.05, 0.2, kind="down-and-out")
    with pytest.raises(ValueError, match="already at or past"):
        barrier_mc("call", 100, 100, 95, 1, 0.05, 0.2, kind="up-and-out")
    with pytest.raises(ValueError, match="kind"):
        barrier_price("call", 100, 100, 90, 1, 0.05, 0.2, kind="down-and-in")
    with pytest.raises(ValueError, match="barrier H"):
        barrier_price("call", 100, 100, -1, 1, 0.05, 0.2, kind="down-and-out")
    with pytest.raises(ValueError, match="sigma > 0"):
        barrier_price("call", 100, 100, 90, 1, 0.05, 0.0, kind="down-and-out")
    with pytest.raises(ValueError, match="monitoring"):
        barrier_mc("call", 100, 100, 90, 1, 0.05, 0.2, kind="down-and-out", monitoring="daily")
    with pytest.raises(ValueError, match="n_dates"):
        barrier_price("call", 100, 100, 90, 1, 0.05, 0.2, kind="down-and-out", n_dates=0)


def test_barrier_zero_vol_continuous_mc_is_deterministic() -> None:
    # the forward path 100 e^{0.05 t} never reaches 120 in 1 year
    res = barrier_mc(
        "call", 100, 100, 120, 1, 0.05, 0.0, kind="up-and-out", n_paths=10, monitoring="continuous"
    )
    # 1e-10, not 1e-12: the log drift is added 252 times, and S_T - K is 20x smaller than S_T
    assert res.estimate == pytest.approx(100 - 100 * math.exp(-0.05), rel=1e-10)


# lookback


@pytest.mark.parametrize("option", ["call", "put"])
def test_lookback_closed_form_matches_the_exact_bridge_extremes(option: Option) -> None:
    exact = lookback_price(option, 100, 1, 0.05, 0.2, Q)

    def run(seed: int) -> MCResult:
        return lookback_mc(
            option, 100, 1, 0.05, 0.2, Q, n_dates=10, n_paths=4_000, monitoring="continuous",
            rng=seed,
        )  # fmt: skip

    assert 0.90 <= coverage(run, exact) <= 0.99


@pytest.mark.parametrize("option", ["call", "put"])
def test_lookback_checked_once_is_the_atm_vanilla(option: Option) -> None:
    # 1 date: min(S, S_T) or max(S, S_T), so the payoff is the atm call or put
    exact = float(price(option, 100, 100, 1, 0.05, 0.2, Q))

    def run(seed: int) -> MCResult:
        return lookback_mc(option, 100, 1, 0.05, 0.2, Q, n_dates=1, n_paths=4_000, rng=seed)

    assert 0.90 <= coverage(run, exact) <= 0.99


@pytest.mark.parametrize("option", ["call", "put"])
def test_discrete_lookback_is_below_the_continuous_one_on_the_same_paths(option: Option) -> None:
    kw = {"n_dates": 52, "n_paths": 20_000, "rng": 6}
    discrete = lookback_mc(option, 100, 1, 0.05, 0.2, **kw).estimate
    cont = lookback_mc(option, 100, 1, 0.05, 0.2, monitoring="continuous", **kw).estimate
    exact = lookback_price(option, 100, 1, 0.05, 0.2)
    assert discrete < cont
    assert discrete < exact


def test_lookback_inputs_are_checked() -> None:
    with pytest.raises(ValueError, match="r != q"):
        lookback_price("call", 100, 1, 0.03, 0.2, 0.03)
    with pytest.raises(ValueError, match="sigma > 0"):
        lookback_price("call", 100, 1, 0.05, 0.0)
    with pytest.raises(ValueError, match="monitoring"):
        lookback_mc("call", 100, 1, 0.05, 0.2, monitoring="weekly")


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
