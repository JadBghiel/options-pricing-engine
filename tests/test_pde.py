"""tier 3 validation: crank nicolson pde vs tier 1 (european) and vs the tree (american)"""

from typing import Literal

import numpy as np
import pytest

from ope.black_scholes import greeks, price
from ope.pde import pde_price, pde_solve
from ope.trees import crr_price

Option = Literal["call", "put"]


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("args", [(42, 40, 0.5, 0.10, 0.20, 0.0), (100, 110, 1, 0.05, 0.25, 0.02)])
def test_european_pde_converges_to_black_scholes(option: Option, args: tuple[float, ...]) -> None:
    exact = float(price(option, *args))
    # at 400 x 400 the error is ~1e-4 absolute on small prices, ~3e-5 relative on big ones, and
    # shrinks ~4x per grid doubling -> the larger of an absolute floor and a relative bound
    assert pde_price(option, *args, n_space=400, n_time=400) == pytest.approx(
        exact, rel=5e-5, abs=2e-4
    )


def test_strike_sits_on_a_grid_node() -> None:
    sol = pde_solve("put", 47.3, 50, 0.7, 0.04, 0.3, n_space=100, n_time=10)
    assert np.min(np.abs(sol.spots - 50)) < 1e-9


@pytest.mark.parametrize(
    "args",
    [
        (50, 50, 5 / 12, 0.10, 0.40, 0.0),
        (100, 110, 1, 0.05, 0.25, 0.0),
        (100, 90, 2, 0.03, 0.30, 0.02),
    ],
)
def test_american_put_pde_and_tree_agree(args: tuple[float, ...]) -> None:
    tree = 0.5 * (
        crr_price("put", *args, steps=2000, american=True)
        + crr_price("put", *args, steps=2001, american=True)
    )
    pde = pde_price("put", *args, american=True, n_space=300, n_time=300)
    # the stated tolerance: 0.05% of the price, tree 2000/2001 steps vs a 300 x 300 grid
    assert pde == pytest.approx(tree, rel=5e-4)


def test_american_call_without_dividends_is_the_european_call() -> None:
    args = (42, 40, 0.5, 0.10, 0.20)
    american = pde_price("call", *args, american=True, n_space=200, n_time=200)
    assert american == pytest.approx(pde_price("call", *args, n_space=200, n_time=200), abs=1e-6)


def test_put_exercise_boundary_rises_to_the_strike() -> None:
    sol = pde_solve("put", 50, 50, 5 / 12, 0.10, 0.40, american=True, n_space=300, n_time=300)
    b = sol.boundary[~np.isnan(sol.boundary)]
    assert np.all(b < 50)
    assert np.all(np.diff(b) >= -1e-9)  # doesnt fall as expiry gets closer
    assert b[-1] > 0.9 * 50  # right before expiry, exercise starts close to the strike
    assert np.isnan(
        pde_solve("put", 50, 50, 5 / 12, 0.10, 0.40, n_space=50, n_time=5).boundary
    ).all()


def test_rannacher_start_fixes_gamma_at_the_strike() -> None:
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
    exact = float(greeks("call", S, K, T, r, sigma).gamma)

    def grid_gamma(rannacher: int) -> float:
        sol = pde_solve("call", S, K, T, r, sigma, n_space=800, n_time=25, rannacher=rannacher)
        i = int(np.argmin(np.abs(sol.spots - K)))
        s, v = sol.spots, sol.values
        h1, h2 = s[i] - s[i - 1], s[i + 1] - s[i]
        return float(2 * (v[i + 1] * h1 - v[i] * (h1 + h2) + v[i - 1] * h2) / (h1 * h2 * (h1 + h2)))

    # plain crank nicolson lets the payoff's kink ring: gamma at the strike is garbage
    assert abs(grid_gamma(0) - exact) > 1.0
    assert grid_gamma(2) == pytest.approx(exact, rel=1e-3)


def test_psor_that_cannot_converge_raises() -> None:
    with pytest.raises(RuntimeError, match="psor"):
        pde_solve("put", 50, 50, 1, 0.1, 0.4, american=True, n_space=50, n_time=5, max_iter=1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"sigma": 0.0}, "> 0"), ({"n_space": 5}, "n_space"), ({"omega": 2.0}, "omega")],
)
def test_bad_inputs_raise(kwargs: dict[str, float], match: str) -> None:
    args = {"S": 50.0, "K": 50.0, "T": 1.0, "r": 0.05, "sigma": 0.3} | kwargs
    with pytest.raises(ValueError, match=match):
        pde_solve("put", **args)  # type: ignore[arg-type]
