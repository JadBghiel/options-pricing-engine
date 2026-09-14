"""exotic options on gbm paths: asian, barrier and lookback, by monte carlo and closed forms

the payoff depends on the path, not only on S_T. the contract says on which dates the path is
looked at (the monitoring dates, equally spaced: T/n, 2T/n, ..., T). continuous monitoring is
the limit, simulated w/ the brownian bridge between dates. inputs are scalars, 1 option per call
"""

import math
from collections.abc import Iterator
from typing import Literal

import numpy as np
from scipy.special import ndtr

from ope.black_scholes import FloatArray, OptionType, _check_option, price
from ope.monte_carlo import MCResult, RngLike, _check_inputs, _rng, payoff

Average = Literal["arithmetic", "geometric"]
BarrierKind = Literal["down-and-out", "up-and-out"]
Monitoring = Literal["discrete", "continuous"]

# broadie glasserman kou 1997: -zeta(1/2) / sqrt(2 pi). a barrier checked every dt behaves like a
# continuous barrier moved away from spot by exp(0.5826 sigma sqrt(dt))
BGK_BETA = 0.5825971579390106


def _check_dates(n_dates: int) -> None:
    if n_dates < 1:
        raise ValueError("n_dates must be >= 1")


def _check_monitoring(monitoring: str) -> None:
    if monitoring not in ("discrete", "continuous"):
        raise ValueError(f"monitoring must be 'discrete' or 'continuous', got {monitoring!r}")


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
    """log S at each monitoring date in turn, so memory stays at 1 array of n_paths"""
    # LEARNING NOTE: each step is the tier 2 exact gbm solution over dt, so the path is exact at
    # every monitoring date whatever n_dates is. the dates are part of the contract here, not a
    # numerical choice: a barrier checked daily and one checked monthly are 2 different options
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

    same draws in the same order as the exotic pricers, so a seed gives the paths they price
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

    log G = the average of the log prices, an average of normals, so normal w/ mean
    log S + (r - q - sigma^2/2) T (n+1)/(2n) and variance sigma^2 T (n+1)(2n+1)/(6n^2).
    black scholes on that normal, discounted at r
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

    average: "arithmetic" (the traded one, no closed form) or "geometric". control_variate:
    corrects the arithmetic estimate w/ the geometric asian on the same paths, whose price is
    known exactly (asian_geometric_price)
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
        # (their correlation is above 0.99), so the payoffs move together. we know the geometric
        # price exactly, so its error on the sample tells us the arithmetic one's error, and we
        # remove it. this cuts the standard error by 10 to 100x, far more than the tier 2 control
        # variate on the stock
        exact = asian_geometric_price(option, S, K, T, r, sigma, q, n_dates=n_dates)
        arith = _control_variate(arith, geo, exact)
    return _result(arith, n_paths)


# barrier


def _check_barrier(S: float, H: float, kind: str) -> None:
    if kind not in ("down-and-out", "up-and-out"):
        raise ValueError(f"kind must be 'down-and-out' or 'up-and-out', got {kind!r}")
    if not (math.isfinite(H) and H > 0):
        raise ValueError("barrier H must be finite and > 0")
    if (kind == "down-and-out" and S <= H) or (kind == "up-and-out" and S >= H):
        raise ValueError(f"spot {S} is already at or past the barrier {H}, the option is dead")


def barrier_price(
    option: OptionType,
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    kind: BarrierKind,
    n_dates: int | None = None,
) -> float:
    """knock out barrier price, barrier H watched continuously (closed form, as in hull)

    down-and-out dies the first time S <= H, up-and-out the first time S >= H. n_dates: instead,
    approximate a barrier checked on n_dates equal dates, by the continuous price w/ the barrier
    moved away from spot by exp(0.5826 sigma sqrt(T/n)) (broadie glasserman kou 1997)
    """
    _check_option(option)
    _check_inputs(S, K, T, sigma, 2)
    _check_barrier(S, H, kind)
    if sigma == 0:
        raise ValueError("the barrier closed form needs sigma > 0")
    down = kind == "down-and-out"
    if n_dates is not None:
        _check_dates(n_dates)
        shift = math.exp(BGK_BETA * sigma * math.sqrt(T / n_dates))
        H = H / shift if down else H * shift

    sT = sigma * math.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    x1 = math.log(S / H) / sT + lam * sT
    y = math.log(H * H / (S * K)) / sT + lam * sT
    y1 = math.log(H / S) / sT + lam * sT
    s_disc, k_disc = S * math.exp(-q * T), K * math.exp(-r * T)
    a, b = float((H / S) ** (2 * lam)), float((H / S) ** (2 * lam - 2))  # reflection weights

    def n(x: float) -> float:
        return float(ndtr(x))

    vanilla = float(price(option, S, K, T, r, sigma, q))
    if option == "call" and down:
        if H <= K:  # vanilla minus the down-and-in call
            return vanilla - (s_disc * a * n(y) - k_disc * b * n(y - sT))
        return s_disc * n(x1) - k_disc * n(x1 - sT) - s_disc * a * n(y1) + k_disc * b * n(y1 - sT)
    if option == "call":  # up-and-out
        if H <= K:  # pays only above K >= H, but must stay below H
            return 0.0
        up_in = (
            s_disc * n(x1)
            - k_disc * n(x1 - sT)
            - s_disc * a * (n(-y) - n(-y1))
            + k_disc * b * (n(-y + sT) - n(-y1 + sT))
        )
        return vanilla - up_in
    if not down:  # up-and-out put
        if H >= K:  # vanilla minus the up-and-in put
            return vanilla - (-s_disc * a * n(-y) + k_disc * b * n(-y + sT))
        return (
            -s_disc * n(-x1) + k_disc * n(-x1 + sT) + s_disc * a * n(-y1) - k_disc * b * n(-y1 + sT)
        )
    if H >= K:  # down-and-out put: pays only below K <= H, but must stay above H
        return 0.0
    down_in = (
        -s_disc * n(-x1)
        + k_disc * n(-x1 + sT)
        + s_disc * a * (n(y) - n(y1))
        - k_disc * b * (n(y - sT) - n(y1 - sT))
    )
    return vanilla - down_in


def barrier_mc(
    option: OptionType,
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    kind: BarrierKind,
    n_dates: int = 252,
    n_paths: int = 100_000,
    monitoring: Monitoring = "discrete",
    rng: RngLike = None,
) -> MCResult:
    """knock out barrier by monte carlo, checked on n_dates equal dates (252 = daily for 1 year)

    monitoring "discrete": the barrier is only looked at on the dates, like the real contract.
    "continuous": also counts the crossings between dates, w/ the brownian bridge, so it
    converges to barrier_price(n_dates=None) whatever n_dates is
    """
    _check_option(option)
    _check_inputs(S, K, T, sigma, n_paths)
    _check_barrier(S, H, kind)
    _check_dates(n_dates)
    _check_monitoring(monitoring)
    down = kind == "down-and-out"
    h = math.log(H)
    var_dt = sigma**2 * T / n_dates
    alive = np.ones(n_paths)  # probability that the path is still alive
    prev = np.full(n_paths, math.log(S))
    x = prev
    for x in _log_walk(S, T, r, sigma, q, n_dates, n_paths, _rng(rng)):
        if monitoring == "continuous" and var_dt > 0:
            # LEARNING NOTE: between 2 dates the log path is a brownian bridge, and given both ends
            # the chance it touched H in between is exp(-2 a b / (sigma^2 dt)), a and b the
            # distances of the ends to log H. multiplying by the survival probability instead of
            # drawing a coin is exact and adds no noise
            ab = np.maximum((prev - h) * (x - h), 0.0)  # < 0 when the dates are on 2 sides of H
            alive *= 1.0 - np.exp(-2.0 * ab / var_dt)
        alive[(x <= h) if down else (x >= h)] = 0.0
        prev = x
    disc = math.exp(-r * T)
    return _result(disc * payoff(option, np.exp(x), K) * alive, n_paths)


# lookback


def lookback_price(
    option: OptionType, S: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """floating strike lookback, watched continuously, at inception (the running min/max is S)

    call pays S_T - min S, put pays max S - S_T: buy at the lowest, sell at the highest price.
    goldman sosin gatto 1979, as in hull. needs r != q, the formula divides by r - q
    """
    _check_option(option)
    _check_inputs(S, S, T, sigma, 2)
    if sigma == 0:
        raise ValueError("the lookback closed form needs sigma > 0")
    if abs(r - q) < 1e-12:
        raise ValueError("the lookback closed form needs r != q, use lookback_mc")
    sT = sigma * math.sqrt(T)
    k = sigma**2 / (2 * (r - q))
    s_disc, cash_disc = S * math.exp(-q * T), S * math.exp(-r * T)
    if option == "call":
        a1 = (r - q + 0.5 * sigma**2) * T / sT
        a3 = (-r + q + 0.5 * sigma**2) * T / sT
        return float(
            s_disc * ndtr(a1) - s_disc * k * ndtr(-a1) - cash_disc * (ndtr(a1 - sT) - k * ndtr(-a3))
        )
    b1 = (-r + q + 0.5 * sigma**2) * T / sT
    b2 = b1 - sT
    b3 = (r - q - 0.5 * sigma**2) * T / sT
    return float(
        cash_disc * (ndtr(b1) - k * ndtr(-b3)) + s_disc * k * ndtr(-b2) - s_disc * ndtr(b2)
    )


def lookback_mc(
    option: OptionType,
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    n_dates: int = 252,
    n_paths: int = 100_000,
    monitoring: Monitoring = "discrete",
    rng: RngLike = None,
) -> MCResult:
    """floating strike lookback by monte carlo, min/max over S and the n_dates dates

    monitoring "continuous": draws the exact min (or max) of the brownian bridge between dates,
    so it converges to lookback_price whatever n_dates is
    """
    _check_option(option)
    _check_inputs(S, S, T, sigma, n_paths)
    _check_dates(n_dates)
    _check_monitoring(monitoring)
    gen = _rng(rng)
    # the uniforms come from a child stream, so a seed gives the same paths in both modes
    uniforms = gen.spawn(1)[0]
    call = option == "call"
    var_dt = sigma**2 * T / n_dates
    prev = np.full(n_paths, math.log(S))
    extreme = prev.copy()  # running min of log S for a call, max for a put
    x = prev
    for x in _log_walk(S, T, r, sigma, q, n_dates, n_paths, gen):
        if monitoring == "continuous":
            # the bridge's min between a and b is (a + b - sqrt((b - a)^2 - 2 sigma^2 dt log u)) / 2
            # for a uniform u, its max the same w/ + (inverse of its known distribution)
            spread = np.sqrt((x - prev) ** 2 - 2.0 * var_dt * np.log(uniforms.random(n_paths)))
            between = 0.5 * (prev + x - spread) if call else 0.5 * (prev + x + spread)
        else:
            between = x
        extreme = np.minimum(extreme, between) if call else np.maximum(extreme, between)
        prev = x
    s_t = np.exp(x)
    disc = math.exp(-r * T)
    pay = s_t - np.exp(extreme) if call else np.exp(extreme) - s_t
    return _result(disc * pay, n_paths)
