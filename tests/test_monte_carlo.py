"""tier 2 validation: monte carlo prices and deltas vs the tier 1 closed form

every test uses fixed seeds, so it gives the same answer on every run. the statistical claims
(the 95% interval holds the true price about 95% of the time, the standard error matches the
real spread) are checked over many seeds, not on 1 lucky seed
"""

import math
from typing import Literal

import numpy as np
import pytest

from ope.black_scholes import greeks, price
from ope.monte_carlo import MCResult, gbm_terminal, mc_delta, mc_price, payoff

Option = Literal["call", "put"]
HULL = (42.0, 40.0, 0.5, 0.10, 0.20)  # S, K, T, r, sigma


# simulation


def test_gbm_terminal_with_zero_draw_is_the_median() -> None:
    S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.2, 0.01
    expected = S * math.exp((r - q - 0.5 * sigma**2) * T)
    assert float(gbm_terminal(S, T, r, sigma, 0.0, q)[()]) == pytest.approx(expected, rel=1e-14)


def test_gbm_terminal_grows_at_r_minus_q_on_average() -> None:
    S, T, r, sigma, q = 100.0, 2.0, 0.05, 0.3, 0.02
    z = np.random.default_rng(0).standard_normal(1_000_000)
    s_t = gbm_terminal(S, T, r, sigma, z, q)
    discounted = s_t * math.exp(-(r - q) * T)
    se = discounted.std(ddof=1) / math.sqrt(z.size)
    assert abs(discounted.mean() - S) < 4 * se
    # log S_T has variance sigma^2 T
    assert np.log(s_t).var() == pytest.approx(sigma**2 * T, rel=0.01)


def test_payoff() -> None:
    s_t = np.array([30.0, 40.0, 50.0])
    np.testing.assert_array_equal(payoff("call", s_t, 40), [0, 0, 10])
    np.testing.assert_array_equal(payoff("put", s_t, 40), [10, 0, 0])


# prices


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize(
    ("antithetic", "control"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_95_interval_holds_the_closed_form_about_95_percent_of_the_time(
    option: Option, antithetic: bool, control: bool
) -> None:
    exact = float(price(option, *HULL))
    inside = 0
    for seed in range(200):
        lo, hi = mc_price(
            option, *HULL, n_paths=4_000, antithetic=antithetic, control_variate=control, rng=seed
        ).ci95
        inside += lo <= exact <= hi
    assert 0.90 <= inside / 200 <= 0.99


@pytest.mark.parametrize(("antithetic", "control"), [(False, False), (True, False), (False, True)])
def test_standard_error_matches_the_real_spread(antithetic: bool, control: bool) -> None:
    runs = [
        mc_price(
            "call", *HULL, n_paths=4_000, antithetic=antithetic, control_variate=control, rng=s
        )
        for s in range(300)
    ]
    real_spread = np.std([r.estimate for r in runs], ddof=1)
    reported = np.mean([r.std_error for r in runs])
    assert reported == pytest.approx(real_spread, rel=0.15)


def test_variance_reduction_cuts_the_standard_error_at_equal_paths() -> None:
    def se(antithetic: bool, control: bool) -> float:
        return mc_price(
            "call", *HULL, n_paths=200_000, antithetic=antithetic, control_variate=control, rng=7
        ).std_error

    plain = se(False, False)
    assert plain / se(True, False) > 1.7  # antithetic, about 2x on this option
    assert plain / se(False, True) > 3.0  # control variate, about 3.6x
    assert plain / se(True, True) > 9.0  # both, about 12x


def test_zero_vol_is_exact() -> None:
    res = mc_price("call", 100, 90, 1, 0.05, 0.0, n_paths=10, rng=0)
    assert res.estimate == pytest.approx(100 - 90 * math.exp(-0.05), rel=1e-12)
    assert res.std_error < 1e-12  # only rounding noise is left, every path is the same


def test_same_seed_same_result_and_generator_accepted() -> None:
    a = mc_price("put", *HULL, n_paths=1_000, rng=3)
    b = mc_price("put", *HULL, n_paths=1_000, rng=np.random.default_rng(3))
    c = mc_price("put", *HULL, n_paths=1_000, rng=4)
    assert a == b
    assert a != c


def test_ci95_is_symmetric_around_the_estimate() -> None:
    res = MCResult(estimate=5.0, std_error=0.1, n_paths=100)
    assert res.ci95 == pytest.approx((5.0 - 0.196, 5.0 + 0.196))


# deltas


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("method", ["bump", "pathwise", "likelihood_ratio"])
def test_delta_matches_the_closed_form_within_its_standard_error(
    option: Option, method: str
) -> None:
    exact = float(greeks(option, *HULL).delta)
    res = mc_delta(option, *HULL, method=method, n_paths=200_000, rng=11)  # type: ignore[arg-type]
    assert abs(res.estimate - exact) < 4 * res.std_error


def test_likelihood_ratio_is_noisier_than_pathwise() -> None:
    se = {
        m: mc_delta("call", *HULL, method=m, n_paths=200_000, rng=5).std_error  # type: ignore[arg-type]
        for m in ("bump", "pathwise", "likelihood_ratio")
    }
    assert se["likelihood_ratio"] > 2.5 * se["pathwise"]
    # bumping w/ common random numbers is about as precise as pathwise
    assert se["bump"] == pytest.approx(se["pathwise"], rel=0.2)


# bad inputs


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"S": -1.0}, "> 0"),
        ({"T": 0.0}, "T must be > 0"),
        ({"sigma": -0.1}, ">= 0"),
        ({"S": math.nan}, "finite"),
        ({"n_paths": 1}, ">= 2"),
        ({"n_paths": 101, "antithetic": True}, "even"),
    ],
)
def test_bad_inputs_raise(kwargs: dict[str, float], match: str) -> None:
    args = {"S": 42.0, "K": 40.0, "T": 0.5, "r": 0.1, "sigma": 0.2, "n_paths": 100} | kwargs
    with pytest.raises(ValueError, match=match):
        mc_price("call", **args)  # type: ignore[arg-type]


def test_bad_delta_inputs_raise() -> None:
    with pytest.raises(ValueError, match="sigma > 0"):
        mc_delta("call", 42, 40, 0.5, 0.1, 0.0, method="likelihood_ratio", n_paths=10)
    with pytest.raises(ValueError, match="bump"):
        mc_delta("call", 42, 40, 0.5, 0.1, 0.2, method="bump", bump=50.0, n_paths=10)
    with pytest.raises(ValueError, match="method"):
        mc_delta("call", 42, 40, 0.5, 0.1, 0.2, method="finite", n_paths=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="call' or 'put"):
        mc_price("straddle", 42, 40, 0.5, 0.1, 0.2, n_paths=10)  # type: ignore[arg-type]
