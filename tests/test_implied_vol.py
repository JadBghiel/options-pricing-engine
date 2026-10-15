"""tier 1 validation: implied vol round trips, bounds and the bisection fallback"""

import numpy as np
import pytest

import ope.implied_volatility as iv_module
from ope.black_scholes import _price_and_vega, price
from ope.implied_volatility import implied_vol, implied_vol_result


def test_hull_round_trip() -> None:
    assert implied_vol("call", 4.7594223929, 42, 40, 0.5, 0.10) == pytest.approx(0.2, abs=1e-10)
    assert implied_vol("put", 0.8085993729, 42, 40, 0.5, 0.10) == pytest.approx(0.2, abs=1e-10)


def test_round_trip_on_random_sets() -> None:
    rng = np.random.default_rng(5)
    recovered_sigma = 0
    for i in range(400):
        option = "call" if i % 2 else "put"
        S, K = 100.0, float(rng.uniform(40, 250))
        T, r, q = float(rng.uniform(0.02, 3)), float(rng.uniform(-0.01, 0.1)), 0.02
        sigma = float(rng.uniform(0.05, 1.5))
        target, vega = _price_and_vega(option, S, K, T, r, sigma, q)
        res = implied_vol_result(option, target, S, K, T, r, q)

        # price -> vol -> price holds everywhere
        back = float(price(option, S, K, T, r, res.sigma, q))
        assert back == pytest.approx(target, abs=1e-11 * max(1.0, target))
        # vol -> price -> vol only where vega is big enough to move the price
        if vega > 1e-3:
            assert res.sigma == pytest.approx(sigma, abs=1e-6)
            recovered_sigma += 1
    assert recovered_sigma > 300


def test_newton_needs_few_steps_near_the_money() -> None:
    target = float(price("call", 100, 100, 1, 0.05, 0.3))
    res = implied_vol_result("call", target, 100, 100, 1, 0.05)
    assert res.bisection_steps == 0
    assert res.iterations <= 5


def test_bisection_kicks_in_when_newton_overshoots() -> None:
    # found by a random search: the inflection start is above the first bracket [0, 1] ->
    # newton starts at 0.5 in the convex region and its 1st steps overshoot the bracket
    args = (100.0, 225.65540010084422, 0.6991518512855576, 0.0918471623936054)
    q, sigma = 0.00574663166404526, 0.9296956867874604
    target = float(price("call", *args, sigma, q))
    res = implied_vol_result("call", target, *args, q)
    assert res.bisection_steps > 0
    assert res.sigma == pytest.approx(sigma, abs=1e-10)


def test_bisection_alone_still_converges(monkeypatch: pytest.MonkeyPatch) -> None:
    # vega forced to 0 -> every step is a bisection step
    def no_vega(*args: float) -> tuple[float, float]:
        return _price_and_vega(*args)[0], 0.0  # type: ignore[arg-type]

    monkeypatch.setattr(iv_module, "_price_and_vega", no_vega)
    res = implied_vol_result("call", 4.7594223929, 42, 40, 0.5, 0.10)
    assert res.newton_steps == 0
    assert res.sigma == pytest.approx(0.2, abs=1e-9)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("call", 42.5),  # above S
        ("call", 1.0),  # below S - K e^{-rT} = 3.95
        ("put", 39.0),  # above K e^{-rT} = 38.05
    ],
)
def test_price_outside_bounds_raises(option: str, value: float) -> None:
    with pytest.raises(ValueError, match="no arbitrage bounds"):
        implied_vol(option, value, 42, 40, 0.5, 0.10)  # type: ignore[arg-type]


def test_price_at_lower_bound_gives_zero_vol() -> None:
    lower = 42 - 40 * np.exp(-0.10 * 0.5)
    assert implied_vol("call", float(lower), 42, 40, 0.5, 0.10) == 0.0


def test_deep_itm_price_rounding_under_the_bound_gives_zero_vol() -> None:
    # time value below float resolution, the model price lands 1 ulp under the bound
    args = (100.0, 286.4665458733566, 1.6216467961910275, 0.037832405906065625)
    q = 0.028723648819691444
    target = float(price("put", *args, 0.10379033647137718, q))
    assert implied_vol("put", target, *args, q) == 0.0


def test_expiry_raises() -> None:
    with pytest.raises(ValueError, match="T > 0"):
        implied_vol("call", 2.0, 42, 40, 0.0, 0.10)


def test_price_needing_absurd_vol_raises() -> None:
    # T = 1e-8 -> even sigma = 1024 gives sigma*sqrt(T) = 0.1, the call is worth ~2.85, far
    # under its bound of 42. w/ a normal T the price hits the bound long before
    with pytest.raises(ValueError, match="above"):
        implied_vol("call", 41.0, 42, 40, 1e-8, 0.10)


def test_max_iter_reached_raises() -> None:
    with pytest.raises(RuntimeError, match="did not converge"):
        implied_vol_result("call", 4.7594223929, 42, 40, 0.5, 0.10, max_iter=1)
