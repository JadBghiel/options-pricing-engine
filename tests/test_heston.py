"""tier 5 validation: heston by its characteristic function

the semi analytical price compared w/ an independent integral (lewis 2000), and w/ black scholes
as the vol of vol goes to 0
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from ope.black_scholes import price
from ope.heston import (
    HestonParams,
    heston_cf,
    heston_implied_vol,
    heston_price,
)
from ope.implied_volatility import implied_vol

# feller violated (2 kappa theta = 0.12 < xi^2 = 0.36): the euler variance goes negative often
P = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
S, K, T, R = 100.0, 100.0, 1.0, 0.05


def lewis_call(S: float, K: float, T: float, r: float, p: HestonParams, q: float = 0.0) -> float:
    """lewis 2000, 1 integral along Im u = -1/2: a different route from P1 and P2"""
    F = S * math.exp((r - q) * T)
    f = math.log(F / K)

    def integrand(u: float) -> float:
        psi = heston_cf(u - 0.5j, S, T, r, p, q)[()] * np.exp(-1j * (u - 0.5j) * math.log(F))
        return float((np.exp(1j * u * f) * psi).real / (u * u + 0.25))

    integral = quad(integrand, 0, math.inf, limit=500, epsabs=1e-13)[0]
    return math.exp(-r * T) * (F - math.sqrt(F * K) / math.pi * integral)


# params


def test_params_are_checked_and_feller() -> None:
    assert not P.feller
    assert HestonParams(0.04, 3.0, 0.04, 0.4, -0.5).feller  # 0.24 >= 0.16
    for bad in (
        (-0.01, 1, 0.04, 0.5, 0),
        (0.04, 0, 0.04, 0.5, 0),
        (0.04, 1, 0.04, 0, 0),
        (0.04, 1, 0.04, 0.5, -1.5),
        (0.04, 1, math.nan, 0.5, 0),
    ):
        with pytest.raises(ValueError):
            HestonParams(*bad)


def test_effective_vol_is_the_root_mean_expected_variance() -> None:
    p = HestonParams(0.09, 1.5, 0.04, 0.3, 0.0)
    t = np.linspace(0, 2.0, 200_001)
    mean_var = np.mean(0.04 + 0.05 * np.exp(-1.5 * t))
    assert p.effective_vol(2.0) == pytest.approx(math.sqrt(mean_var), rel=1e-6)
    assert P.effective_vol(5.0) == pytest.approx(0.2)  # v0 = theta: flat


# characteristic function


def test_cf_at_0_and_at_minus_i() -> None:
    assert heston_cf(0.0, S, T, R, P)[()] == pytest.approx(1.0)
    # E[S_T] = S e^{(r-q)T}: the discounted stock is a martingale
    assert heston_cf(-1j, S, T, R, P, 0.02)[()] == pytest.approx(S * math.exp(0.03 * T))


def test_cf_tends_to_the_lognormal_one_as_xi_goes_to_0() -> None:
    u = np.linspace(0.1, 10, 50)
    errors = []
    for xi in (1e-3, 1e-4):
        p = HestonParams(0.09, 1.5, 0.04, xi, -0.7)
        sigma = p.effective_vol(T)
        drift = math.log(S) + (R - 0.5 * sigma**2) * T
        lognormal = np.exp(1j * u * drift - 0.5 * sigma**2 * T * u**2)
        errors.append(np.abs(heston_cf(u, S, T, R, p) - lognormal).max())
    # linear in xi (5.5e-4, then 5.5e-5), below xi = 1e-5 the formula loses digits instead:
    # kappa theta / xi^2 multiplies a bracket that is almost 0
    assert errors[1] < 1e-4
    assert errors[0] / errors[1] > 8


def test_original_form_agrees_at_1_year_and_jumps_at_5() -> None:
    u = np.linspace(0.01, 20, 4_000)
    np.testing.assert_allclose(
        heston_cf(u, S, 1.0, R, P, form="original"), heston_cf(u, S, 1.0, R, P), atol=1e-12
    )
    # 5 years: the complex log crosses its branch cut near u = 1.34 and the 2 forms split
    gap = np.abs(heston_cf(u, S, 5.0, R, P, form="original") - heston_cf(u, S, 5.0, R, P))
    assert gap[u < 1.3].max() < 1e-12
    assert gap[(u > 1.4) & (u < 3)].max() > 0.1
    # the trap form moves by at most 0.024 between neighbouring points (the phase e^{iu log F}
    # turning), the original jumps by 1.44 at the cut
    assert np.abs(np.diff(heston_cf(u, S, 5.0, R, P))).max() < 0.05
    assert np.abs(np.diff(heston_cf(u, S, 5.0, R, P, form="original"))).max() > 1
    with pytest.raises(ValueError, match="form"):
        heston_cf(1.0, S, T, R, P, form="lewis")  # type: ignore[arg-type]


# semi analytical price


@pytest.mark.parametrize("T_", [0.1, 1.0, 5.0, 10.0])
def test_price_matches_the_lewis_integral(T_: float) -> None:
    strikes = np.array([60.0, 80.0, 100.0, 120.0, 150.0])
    ours = heston_price("call", S, strikes, T_, R, P, 0.01)
    lewis = [lewis_call(S, k, T_, R, P, 0.01) for k in strikes]
    # atol 1e-8: a 1 month call 50% out of the money is worth about 1e-10, below both
    # integrals' noise
    np.testing.assert_allclose(ours, lewis, rtol=1e-7, atol=1e-8)


def test_put_call_parity() -> None:
    strikes = np.linspace(50, 200, 16)
    calls = heston_price("call", S, strikes, 2.0, R, P, 0.02)
    puts = heston_price("put", S, strikes, 2.0, R, P, 0.02)
    forward_value = S * math.exp(-0.02 * 2.0) - strikes * math.exp(-R * 2.0)
    np.testing.assert_allclose(np.asarray(calls) - puts, forward_value, atol=1e-9)


def test_scalar_strike_gives_a_float() -> None:
    assert isinstance(heston_price("call", S, K, T, R, P), float)
    assert isinstance(heston_implied_vol(S, K, T, R, P), float)


def test_heston_tends_to_black_scholes_as_xi_goes_to_0() -> None:
    gaps = []
    for xi in (1e-2, 1e-3, 1e-4):
        p = HestonParams(0.09, 1.5, 0.04, xi, -0.7)  # v0 != theta: the vol moves, deterministically
        bs = float(price("call", S, K, T, R, p.effective_vol(T)))
        gaps.append(abs(heston_price("call", S, K, T, R, p) - bs))
    # 2.8e-3, 3.0e-4, 3.0e-5: linear in xi once xi is small (the rho xi term leads), at xi = 0.1
    # the 1st and 2nd order terms nearly cancel -> the gap isnt monotone there
    assert gaps[2] < 1e-4
    assert gaps[0] / gaps[1] > 8
    assert gaps[1] / gaps[2] > 8


def test_price_inputs_are_checked() -> None:
    with pytest.raises(ValueError, match="strike"):
        heston_price("call", S, [100, -1], T, R, P)
    with pytest.raises(ValueError, match="maturity"):
        heston_price("call", S, K, 0.0, R, P)
    with pytest.raises(ValueError, match="spot"):
        heston_price("call", -1.0, K, T, R, P)
    with pytest.raises(ValueError, match="finite"):
        heston_price("call", math.inf, K, T, R, P)


# smile


def test_negative_rho_gives_a_downward_skew() -> None:
    vols = np.asarray(heston_implied_vol(S, np.linspace(70, 130, 13), T, R, P))
    assert np.all(np.diff(vols) < 0)
    assert vols[0] - vols[-1] > 0.10  # about 28% at K = 70 to 13% at K = 130


def test_zero_rho_gives_a_smile_and_tiny_xi_a_flat_line() -> None:
    p = HestonParams(0.04, 1.5, 0.04, 0.6, 0.0)
    forward = S * math.exp(R * T)
    strikes = forward * np.exp(np.linspace(-0.4, 0.4, 9))
    vols = np.asarray(heston_implied_vol(S, strikes, T, R, p))
    assert np.argmin(vols) in (4, 5)  # lowest near the forward
    assert vols[0] > vols[4] + 0.03 and vols[-1] > vols[4] + 0.01
    flat = heston_implied_vol(S, strikes, T, R, HestonParams(0.04, 1.5, 0.04, 1e-4, 0.0))
    np.testing.assert_allclose(flat, 0.2, atol=1e-4)


def test_implied_vol_reprices_the_heston_price() -> None:
    vol = heston_implied_vol(S, 90.0, T, R, P)
    assert float(price("put", S, 90, T, R, vol)) == pytest.approx(
        heston_price("put", S, 90.0, T, R, P), rel=1e-9
    )
    assert vol == pytest.approx(
        implied_vol("put", heston_price("put", S, 90.0, T, R, P), S, 90, T, R)
    )
