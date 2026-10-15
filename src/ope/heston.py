"""heston stochastic volatility: european prices by the characteristic function (the monte carlo
is in ope.heston_mc)

dS = (r - q) S dt + sqrt(v) S dW1
dv = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho

the variance v has its own noise (xi, the vol of vol) and is pulled back to theta at speed
kappa, rho < 0 (stock down -> vol up) gives the downward skew seen on equity indices
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad

from ope.black_scholes import FloatOrArray, OptionType, _check_option
from ope.implied_volatility import implied_vol

CFForm = Literal["trap", "original"]
ComplexArray = npt.NDArray[np.complex128]


@dataclass(frozen=True)
class HestonParams:
    """v0 variance today, kappa speed back to theta, theta long run variance, xi vol of vol,
    rho correlation between the stock and its variance, in variance terms: 0.04 = 20% vol"""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.v0, self.kappa, self.theta, self.xi, self.rho)):
            raise ValueError("heston params must be finite")
        if self.v0 < 0 or self.theta < 0:
            raise ValueError("variances v0 and theta must be >= 0")
        if self.kappa <= 0:
            raise ValueError("kappa must be > 0")
        if self.xi <= 0:
            raise ValueError("xi must be > 0, at xi = 0 heston is black scholes w/ a moving vol")
        if not -1 <= self.rho <= 1:
            raise ValueError("rho must be in [-1, 1]")

    @property
    def feller(self) -> bool:
        """2 kappa theta >= xi^2: the pull to theta beats the noise, the true v stays above 0"""
        return 2 * self.kappa * self.theta >= self.xi**2

    def effective_vol(self, T: float) -> float:
        """the black scholes vol heston tends to as xi -> 0: the root of the average of
        E[v_t] = theta + (v0 - theta) e^{-kappa t} over [0, T]"""
        decay = (1 - math.exp(-self.kappa * T)) / (self.kappa * T)
        return math.sqrt(self.theta + (self.v0 - self.theta) * decay)


def _check_market(S: float, T: float) -> None:
    if not (math.isfinite(S) and math.isfinite(T)):
        raise ValueError("inputs must be finite")
    if S <= 0:
        raise ValueError("spot S must be > 0")
    if T <= 0:
        raise ValueError("maturity T must be > 0")


# characteristic function


def heston_cf(
    u: npt.ArrayLike,
    S: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float = 0.0,
    *,
    form: CFForm = "trap",
) -> ComplexArray:
    """E[exp(i u log S_T)], the characteristic function of log S_T, for real or complex u

    form "trap": albrecher et al. 2007, "the little heston trap", continuous in u
    "original": heston 1993, same function on paper but its complex log jumps by 2 pi for long
    maturities, kept to show the problem, both lose digits below xi = 1e-5: kappa theta / xi^2
    then multiplies a bracket that is almost 0
    """
    if form not in ("trap", "original"):
        raise ValueError(f"form must be 'trap' or 'original', got {form!r}")
    p = params
    uu = np.asarray(u, dtype=np.complex128)
    iu = 1j * uu
    b = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(b * b + p.xi**2 * (iu + uu * uu))
    # LEARNING NOTE: d and -d both solve the riccati equations, the 2 forms pick a different
    # sign, heston's choice has e^{+dT}, and log((1 - g e^{dT}) / (1 - g)) crosses the negative
    # real axis as u grows, where numpy's log jumps by 2 pi -> a wrong price (16.39 instead of
    # 29.09 for our 5 year atm call, fine at 1 year), w/ -d (the trap form) the argument stays
    # on 1 side, and e^{-dT} -> 0 where e^{dT} overflows to nan
    if form == "trap":
        bd = b - d
        g = bd / (b + d)
        e = np.exp(-d * T)
    else:
        bd = b + d
        g = bd / (b - d)
        e = np.exp(d * T)
    C = p.kappa * p.theta / p.xi**2 * (bd * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = bd / p.xi**2 * (1.0 - e) / (1.0 - g * e)
    out = np.exp(iu * (math.log(S) + (r - q) * T) + C + D * p.v0)
    return np.asarray(out, dtype=np.complex128)


def _probabilities(
    S: float, K: float, T: float, r: float, params: HestonParams, q: float
) -> tuple[float, float]:
    """P1 and P2, heston's versions of N(d1) and N(d2), by gil pelaez inversion

    P2 = risk neutral probability of S_T > K, P1 = the same under the stock measure
    P = 1/2 + 1/pi int_0^inf Re[e^{-iu log K} phi(u) / (iu)] du, w/ phi(u - i) / phi(-i) for P1
    """
    log_k = math.log(K)
    forward = S * math.exp((r - q) * T)  # phi(-i) = E[S_T]

    def integrand(u: float, shift: complex, scale: float) -> float:
        phi = heston_cf(u + shift, S, T, r, params, q)[()]
        return float((np.exp(-1j * u * log_k) * phi / (1j * u * scale)).real)

    probs = []
    for shift, scale in ((-1j, forward), (0j, 1.0)):
        integral = quad(integrand, 0.0, math.inf, args=(shift, scale), limit=500, epsabs=1e-13)[0]
        probs.append(0.5 + integral / math.pi)
    return probs[0], probs[1]


def heston_price(
    option: OptionType,
    S: float,
    K: npt.ArrayLike,
    T: float,
    r: float,
    params: HestonParams,
    q: float = 0.0,
) -> FloatOrArray:
    """european price under heston, semi analytical (1 numerical integral per probability)

    call = S e^{-qT} P1 - K e^{-rT} P2, the black scholes formula w/ N(d1), N(d2) replaced by
    heston's probabilities, K can be an array (a chain), S and T are scalars
    """
    _check_option(option)
    _check_market(S, T)
    strikes = np.asarray(K, dtype=np.float64)
    if not (np.all(np.isfinite(strikes)) and np.all(strikes > 0)):
        raise ValueError("strike K must be finite and > 0")
    s_disc = S * math.exp(-q * T)
    out = np.empty(strikes.shape)
    for idx, k in np.ndenumerate(strikes):
        p1, p2 = _probabilities(S, float(k), T, r, params, q)
        k_disc = float(k) * math.exp(-r * T)
        if option == "call":
            out[idx] = s_disc * p1 - k_disc * p2
        else:  # from 1 - P instead of parity -> a deep itm call doesnt eat the put's digits
            out[idx] = k_disc * (1 - p2) - s_disc * (1 - p1)
    return float(out) if out.ndim == 0 else out


def heston_implied_vol(
    S: float,
    K: npt.ArrayLike,
    T: float,
    r: float,
    params: HestonParams,
    q: float = 0.0,
) -> FloatOrArray:
    """black scholes implied vol of the heston price for each strike: the smile

    prices the out of the money side (puts below the forward, calls above), where all of the
    price is time value, then inverts it w/ the tier 1 solver
    """
    strikes = np.asarray(K, dtype=np.float64)
    forward = S * math.exp((r - q) * T)
    out = np.empty(strikes.shape)
    for idx, k in np.ndenumerate(strikes):
        option: OptionType = "put" if k < forward else "call"
        value = float(heston_price(option, S, float(k), T, r, params, q))
        out[idx] = implied_vol(option, value, S, float(k), T, r, q)
    return float(out) if out.ndim == 0 else out
