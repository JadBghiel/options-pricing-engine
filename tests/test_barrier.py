"""tier 5 validation: barrier and lookback options

discrete monitoring is compared w/ discrete prices only, and the continuous closed forms w/ the
brownian bridge simulation, which watches the path between dates too
"""

import math
from typing import Literal

import pytest
from scipy.special import ndtr, zeta

from helpers import coverage
from ope.barrier import (
    BGK_BETA,
    BarrierKind,
    barrier_mc,
    barrier_price,
    lookback_mc,
    lookback_price,
)
from ope.black_scholes import price
from ope.monte_carlo import MCResult

Option = Literal["call", "put"]
BASE = (100.0, 100.0, 1.0, 0.05, 0.2)  # S, K, T, r, sigma
Q = 0.02


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
    # the bridge is exact whatever n_dates is -> 10 dates are enough
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
    # the forward path 100 e^{0.05 t} stays below 120 for 1 year
    res = barrier_mc(
        "call", 100, 100, 120, 1, 0.05, 0.0, kind="up-and-out", n_paths=10, monitoring="continuous"
    )
    # 1e-10 tolerance: the log drift is added 252 times, and S_T - K is 20x smaller than S_T
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
    # 1 date: min(S, S_T) or max(S, S_T) -> the payoff is the atm call or put
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
