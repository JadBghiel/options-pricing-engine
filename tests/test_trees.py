"""tier 3 validation: cox ross rubinstein tree vs tier 1 and vs a naive tree written here"""

import math
from typing import Literal

import numpy as np
import pytest

from ope.black_scholes import price
from ope.trees import backward_step, crr_params, crr_price

Option = Literal["call", "put"]


def _naive_tree(
    option: Option, S: float, K: float, T: float, r: float, sigma: float, n: int
) -> float:
    # independent reference: plain loops, node by node, american exercise checked everywhere
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    def pay(s: float) -> float:
        return max(s - K, 0.0) if option == "call" else max(K - s, 0.0)

    values = [pay(S * u**j * d ** (n - j)) for j in range(n + 1)]
    for i in range(n - 1, -1, -1):
        values = [
            max(disc * (p * values[j + 1] + (1 - p) * values[j]), pay(S * u**j * d ** (i - j)))
            for j in range(i + 1)
        ]
    return values[0]


def test_crr_params_are_risk_neutral() -> None:
    c = crr_params(T=1.0, r=0.05, sigma=0.2, steps=100, q=0.01)
    assert c.u * c.d == pytest.approx(1.0)
    assert 0 < c.p < 1
    # on average, 1 step grows the stock at r - q: the tree is in the risk neutral world
    assert c.p * c.u + (1 - c.p) * c.d == pytest.approx(math.exp((0.05 - 0.01) * c.dt), rel=1e-14)


def test_backward_step_by_hand() -> None:
    np.testing.assert_allclose(backward_step([0.0, 20.0], p=0.5, disc=1.0), [10.0])
    np.testing.assert_allclose(backward_step([0.0, 20.0], p=0.5, disc=1.0, exercise=[12.0]), [12.0])


def test_hull_five_step_american_put() -> None:
    # Hull, binomial trees: S=50, K=50, T=5 months, r=10%, sigma=40%, 5 steps -> about 4.49
    args = (50.0, 50.0, 5 / 12, 0.10, 0.40)
    tree = crr_price("put", *args, steps=5, american=True)
    assert tree == pytest.approx(_naive_tree("put", *args, n=5), abs=1e-12)
    assert round(tree, 2) == 4.49


@pytest.mark.parametrize("option", ["call", "put"])
def test_matches_a_naive_american_tree(option: Option) -> None:
    args = (100.0, 95.0, 1.0, 0.05, 0.3)
    assert crr_price(option, *args, steps=60, american=True) == pytest.approx(
        _naive_tree(option, *args, n=60), abs=1e-10
    )


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("args", [(42, 40, 0.5, 0.10, 0.20, 0.0), (100, 110, 1, 0.05, 0.25, 0.02)])
def test_european_tree_converges_to_black_scholes(option: Option, args: tuple[float, ...]) -> None:
    exact = float(price(option, *args))
    # averaging n and n + 1 steps cancels most of the odd/even zigzag
    tree = 0.5 * (crr_price(option, *args, steps=2000) + crr_price(option, *args, steps=2001))
    assert tree == pytest.approx(exact, abs=1e-4)


def test_american_call_without_dividends_is_the_european_call() -> None:
    args = (42.0, 40.0, 0.5, 0.10, 0.20)
    assert crr_price("call", *args, steps=500, american=True) == crr_price("call", *args, steps=500)


def test_dividends_make_early_exercise_of_a_call_worth_something() -> None:
    args = (100.0, 80.0, 1.0, 0.03, 0.25, 0.08)
    premium = crr_price("call", *args, steps=1000, american=True) - float(price("call", *args))
    assert premium > 2.0


def test_american_put_is_worth_at_least_european_and_intrinsic() -> None:
    for S in (30.0, 40.0, 50.0, 60.0):
        american = crr_price("put", S, 50, 1, 0.08, 0.3, steps=400, american=True)
        assert american >= float(price("put", S, 50, 1, 0.08, 0.3)) - 1e-3
        assert american >= max(50 - S, 0) - 1e-12


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="outside"):
        crr_params(T=1.0, r=0.5, sigma=0.01, steps=1)  # e^{0.5} is above u: p > 1
    with pytest.raises(ValueError, match="steps"):
        crr_params(T=1.0, r=0.05, sigma=0.2, steps=0)
    with pytest.raises(ValueError, match="sigma > 0"):
        crr_params(T=1.0, r=0.05, sigma=0.0, steps=10)
    with pytest.raises(ValueError, match="> 0"):
        crr_price("call", -1, 40, 1, 0.05, 0.2)
