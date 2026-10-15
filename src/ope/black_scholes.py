"""black scholes merton closed form for european options: d1/d2, prices, greeks & bounds

units: r, q and sigma annualized and continuously compounded, T in years
every func takes floats or numpy arrays (broadcast together) and returns a float for
scalar inputs, an array otherwise
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

OptionType = Literal["call", "put"]
FloatArray = npt.NDArray[np.float64]
FloatOrArray = float | FloatArray

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def norm_cdf(x: npt.ArrayLike) -> FloatArray:
    """standard normal cdf N(x), the area under the bell curve left of x"""
    return np.asarray(ndtr(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def norm_pdf(x: npt.ArrayLike) -> FloatArray:
    """standard normal density n(x), the height of the bell curve at x"""
    a = np.asarray(x, dtype=np.float64)
    return np.asarray(_INV_SQRT_2PI * np.exp(-0.5 * a * a), dtype=np.float64)


@dataclass(frozen=True)
class Greeks:
    """sensitivities of the price, in hull's units

    delta: dV/dS
    gamma: d2V/dS2
    vega: dV/dsigma per 1.00 of vol, divide by 100 for 1 vol point
    theta: dV/dt per year as calendar time passes (= -dV/dT), divide by 365 for 1 day
    rho: dV/dr per 1.00 of rate, divide by 100 for 1%
    """

    delta: FloatOrArray
    gamma: FloatOrArray
    vega: FloatOrArray
    theta: FloatOrArray
    rho: FloatOrArray


@dataclass(frozen=True)
class _Market:
    """broadcast inputs plus everything the formulas share, computed once"""

    S: FloatArray
    K: FloatArray
    T: FloatArray
    r: FloatArray
    sigma: FloatArray
    q: FloatArray
    df_r: FloatArray  # e^{-rT}, discounts the strike
    df_q: FloatArray  # e^{-qT}, strips the dividends paid before T
    sqrt_T: FloatArray
    degenerate: npt.NDArray[np.bool_]  # sigma*sqrt(T) = 0 or S = 0, no randomness left
    d1: FloatArray
    d2: FloatArray


def _check_option(option: str) -> None:
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def _out(x: FloatArray) -> FloatOrArray:
    return float(x) if x.ndim == 0 else x


def _prepare(
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    q: npt.ArrayLike,
) -> _Market:
    S_, K_, T_, r_, sigma_, q_ = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (S, K, T, r, sigma, q))
    )
    if not all(np.all(np.isfinite(a)) for a in (S_, K_, T_, r_, sigma_, q_)):
        raise ValueError("inputs must be finite")
    if np.any(S_ < 0):
        raise ValueError("spot S must be >= 0")
    if np.any(K_ <= 0):
        raise ValueError("strike K must be > 0")
    if np.any(T_ < 0):
        raise ValueError("maturity T must be >= 0")
    if np.any(sigma_ < 0):
        raise ValueError("volatility sigma must be >= 0")

    df_r = np.exp(-r_ * T_)
    df_q = np.exp(-q_ * T_)
    sqrt_T = np.sqrt(T_)
    vol_sqrt_T = sigma_ * sqrt_T
    degenerate = (vol_sqrt_T == 0.0) | (S_ == 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S_ / K_) + (r_ - q_ + 0.5 * sigma_**2) * T_) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T

    # LEARNING NOTE: when sigma*sqrt(T) -> 0 the stock ends at its forward for sure, and d1, d2
    # go to +inf if the forward is above K, -inf otherwise, setting them to that limit makes
    # N(d) a 0/1 indicator -> the usual formulas return the discounted intrinsic value on
    # the forward w/o any special case nor NaN
    limit = np.where(S_ * df_q > K_ * df_r, np.inf, -np.inf)
    d1 = np.where(degenerate, limit, d1)
    d2 = np.where(degenerate, limit, d2)

    return _Market(S_, K_, T_, r_, sigma_, q_, df_r, df_q, sqrt_T, degenerate, d1, d2)


def _call(m: _Market) -> FloatArray:
    return m.S * m.df_q * norm_cdf(m.d1) - m.K * m.df_r * norm_cdf(m.d2)


def _put(m: _Market) -> FloatArray:
    # LEARNING NOTE: N(-d2) and not 1 - N(d2), for a deep OTM put N(d2) rounds to 1.0 in
    # floating point, so 1 - N(d2) gives 0 while N(-d2) keeps every digit
    return m.K * m.df_r * norm_cdf(-m.d2) - m.S * m.df_q * norm_cdf(-m.d1)


def d1_d2(
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    q: npt.ArrayLike = 0.0,
) -> tuple[FloatOrArray, FloatOrArray]:
    """d1 and d2 of the black scholes formula

    N(d2) is the risk neutral probability that the option ends in the money (S_T > K)
    N(d1) is the same probability under the stock measure, where the stock is the unit of
    account, if sigma*sqrt(T) = 0 or S = 0 both are +inf or -inf, the sign of F - K
    """
    m = _prepare(S, K, T, r, sigma, q)
    return _out(m.d1), _out(m.d2)


def call_price(
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    q: npt.ArrayLike = 0.0,
) -> FloatOrArray:
    """european call: S e^{-qT} N(d1) - K e^{-rT} N(d2)

    T = 0 gives max(S - K, 0), sigma = 0 gives max(S e^{-qT} - K e^{-rT}, 0), S = 0 gives 0
    """
    return _out(_call(_prepare(S, K, T, r, sigma, q)))


def put_price(
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    q: npt.ArrayLike = 0.0,
) -> FloatOrArray:
    """european put: K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

    T = 0 gives max(K - S, 0), sigma = 0 gives max(K e^{-rT} - S e^{-qT}, 0),
    S = 0 gives K e^{-rT}
    """
    return _out(_put(_prepare(S, K, T, r, sigma, q)))


def price(
    option: OptionType,
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    q: npt.ArrayLike = 0.0,
) -> FloatOrArray:
    """call_price or put_price, picked by option"""
    _check_option(option)
    m = _prepare(S, K, T, r, sigma, q)
    return _out(_call(m) if option == "call" else _put(m))


def greeks(
    option: OptionType,
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    q: npt.ArrayLike = 0.0,
) -> Greeks:
    """delta, gamma, vega, theta and rho from 1 shared d1/d2 computation

    units are in the greeks docstring, when sigma*sqrt(T) = 0 or S = 0 they are the greeks
    of the deterministic value max(+-(S e^{-qT} - K e^{-rT}), 0): gamma and vega are 0
    """
    _check_option(option)
    m = _prepare(S, K, T, r, sigma, q)
    n_d1 = norm_pdf(m.d1)
    zeros = np.zeros_like(n_d1)
    live = ~m.degenerate

    # LEARNING NOTE: gamma and vega have no call/put branch, put call parity says
    # C - P = S e^{-qT} - K e^{-rT}, linear in S and w/o sigma: its 2nd derivative in S and
    # its derivative in sigma are 0 -> call and put share them
    gamma = np.divide(m.df_q * n_d1, m.S * m.sigma * m.sqrt_T, out=zeros.copy(), where=live)
    vega = m.S * m.df_q * n_d1 * m.sqrt_T
    decay = -np.divide(m.S * m.df_q * n_d1 * m.sigma, 2.0 * m.sqrt_T, out=zeros.copy(), where=live)

    if option == "call":
        delta = m.df_q * norm_cdf(m.d1)
        theta = decay + m.q * m.S * m.df_q * norm_cdf(m.d1) - m.r * m.K * m.df_r * norm_cdf(m.d2)
        rho = m.K * m.T * m.df_r * norm_cdf(m.d2)
    else:
        delta = -m.df_q * norm_cdf(-m.d1)
        theta = decay - m.q * m.S * m.df_q * norm_cdf(-m.d1) + m.r * m.K * m.df_r * norm_cdf(-m.d2)
        rho = -m.K * m.T * m.df_r * norm_cdf(-m.d2)

    return Greeks(_out(delta), _out(gamma), _out(vega), _out(theta), _out(rho))


def price_bounds(
    option: OptionType,
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    q: npt.ArrayLike = 0.0,
) -> tuple[FloatOrArray, FloatOrArray]:
    """no arbitrage bounds (lower, upper) that hold in any model

    call: [max(S e^{-qT} - K e^{-rT}, 0), S e^{-qT}]
    put: [max(K e^{-rT} - S e^{-qT}, 0), K e^{-rT}]
    """
    _check_option(option)
    m = _prepare(S, K, T, r, 0.0, q)
    stock = m.S * m.df_q
    strike = m.K * m.df_r
    if option == "call":
        return _out(np.maximum(stock - strike, 0.0)), _out(stock)
    return _out(np.maximum(strike - stock, 0.0)), _out(strike)


def _price_and_vega(
    option: OptionType, S: float, K: float, T: float, r: float, sigma: float, q: float
) -> tuple[float, float]:
    """scalar price and vega in 1 pass, for the implied vol solver"""
    m = _prepare(S, K, T, r, sigma, q)
    value = _call(m) if option == "call" else _put(m)
    vega = m.S * m.df_q * norm_pdf(m.d1) * m.sqrt_T
    return float(value), float(vega)
