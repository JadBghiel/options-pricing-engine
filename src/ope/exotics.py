"""exotic options on gbm paths: the walk from date to date, and asian options by monte carlo and
closed form (barrier and lookback options are in ope.barrier)

the payoff depends on the path, not only on S_T, the contract says on which dates the path is
looked at (the monitoring dates, equally spaced: T/n, 2T/n, ..., T), inputs are scalars, 1 option
per call
"""

import math
from collections.abc import Iterator
from typing import Literal

import numpy as np
from scipy.special import ndtr

from ope.black_scholes import FloatArray, OptionType, _check_option
from ope.monte_carlo import MCResult, RngLike, _check_inputs, _rng, payoff

Average = Literal["arithmetic", "geometric"]


def _check_dates(n_dates: int) -> None:
    if n_dates < 1:
        raise ValueError("n_dates must be >= 1")


def _log_walk(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_dates: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Iterator[FloatArray]:
    """log S at each monitoring date in turn -> memory stays at 1 array of n_paths"""
    # LEARNING NOTE: each step is the tier 2 exact gbm solution over dt -> the path is exact at
    # every monitoring date whatever n_dates is, the dates are a term of the contract here: a
    # barrier checked daily and one checked monthly are 2 different options
    dt = T / n_dates
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    x = np.full(n_paths, math.log(S))
    for _ in range(n_dates):
        x = x + drift + vol * rng.standard_normal(n_paths)
        yield x


def gbm_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    n_dates: int,
    n_paths: int,
    rng: RngLike = None,
) -> FloatArray:
    """stock paths, shape (n_paths, n_dates + 1): column 0 is S, column i is the price at i T/n

    same draws in the same order as the exotic pricers -> a seed gives the paths they price
    """
    _check_dates(n_dates)
    out = np.empty((n_paths, n_dates + 1))
    out[:, 0] = S
    for i, x in enumerate(_log_walk(S, T, r, sigma, q, n_dates, n_paths, _rng(rng)), start=1):
        out[:, i] = np.exp(x)
    return out


def _control_variate(x: FloatArray, y: FloatArray, y_mean: float) -> FloatArray:
    """x corrected by beta times y's error on its known mean, beta = regression slope of x on y"""
    cov = np.cov(x, y)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0
    return np.asarray(x - beta * (y - y_mean), dtype=np.float64)


def _result(x: FloatArray, n_paths: int) -> MCResult:
    return MCResult(float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size)), n_paths)


# asian


def asian_geometric_price(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    n_dates: int,
) -> float:
    """closed form of the geometric average asian over the n_dates dates T/n, ..., T

    log G = the average of the log prices, an average of normals -> a normal w/ mean
    log S + (r - q - sigma^2/2) T (n+1)/(2n) and variance sigma^2 T (n+1)(2n+1)/(6n^2),
    then black scholes on that normal, discounted at r
    """
    _check_option(option)
    _check_inputs(S, K, T, sigma, 2)
    _check_dates(n_dates)
    n = n_dates
    mean = math.log(S) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    var = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n * n)
    disc = math.exp(-r * T)
    forward = math.exp(mean + 0.5 * var)  # E[G]
    if var == 0.0:  # sigma = 0, G is known for sure
        return disc * float(payoff(option, forward, K))
    sd = math.sqrt(var)
    d2 = (mean - math.log(K)) / sd
    d1 = d2 + sd
    if option == "call":
        return disc * (forward * float(ndtr(d1)) - K * float(ndtr(d2)))
    return disc * (K * float(ndtr(-d2)) - forward * float(ndtr(-d1)))


def asian_mc(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    n_dates: int = 12,
    n_paths: int = 100_000,
    average: Average = "arithmetic",
    control_variate: bool = False,
    rng: RngLike = None,
) -> MCResult:
    """asian option by monte carlo: the payoff uses the average price over the n_dates dates

    average: "arithmetic" (the traded one, no closed form) or "geometric", control_variate:
    corrects the arithmetic estimate w/ the geometric asian on the same paths, whose price has a
    closed form (asian_geometric_price)
    """
    _check_option(option)
    _check_inputs(S, K, T, sigma, n_paths)
    _check_dates(n_dates)
    if average not in ("arithmetic", "geometric"):
        raise ValueError(f"average must be 'arithmetic' or 'geometric', got {average!r}")
    if control_variate and average == "geometric":
        raise ValueError("the control variate is the geometric asian, use it on the arithmetic one")

    total = np.zeros(n_paths)
    log_total = np.zeros(n_paths)
    for x in _log_walk(S, T, r, sigma, q, n_dates, n_paths, _rng(rng)):
        total += np.exp(x)
        log_total += x
    disc = math.exp(-r * T)
    arith = disc * payoff(option, total / n_dates, K)
    geo = disc * payoff(option, np.exp(log_total / n_dates), K)

    if average == "geometric":
        return _result(geo, n_paths)
    if control_variate:
        # LEARNING NOTE: the arithmetic and geometric averages of 1 path are almost the same number
        # (their correlation is above 0.99), the payoffs move together, we know the geometric
        # price in closed form -> its error on the sample tells us the arithmetic one's error, and
        # we remove it, this cuts the standard error by 10 to 100x, far more than the tier 2
        # control variate on the stock
        exact = asian_geometric_price(option, S, K, T, r, sigma, q, n_dates=n_dates)
        arith = _control_variate(arith, geo, exact)
    return _result(arith, n_paths)
