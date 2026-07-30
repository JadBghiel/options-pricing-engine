"""cox ross rubinstein binomial tree for european and american options

the stock moves up by u or down by d = 1/u at each of n steps. we price backwards from expiry:
at each node the option is worth its discounted risk neutral average one step later, and an
american option is also worth at least what exercising right now pays
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ope.black_scholes import FloatArray, OptionType, _check_option
from ope.monte_carlo import payoff


@dataclass(frozen=True)
class CRRParams:
    """1 step of the tree: up and down factors, risk neutral up probability, 1 step discount"""

    dt: float
    u: float
    d: float
    p: float
    disc: float


def crr_params(T: float, r: float, sigma: float, steps: int, q: float = 0.0) -> CRRParams:
    """u = e^{sigma sqrt(dt)}, d = 1/u, p = (e^{(r-q)dt} - d) / (u - d), disc = e^{-r dt}"""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if T <= 0 or sigma <= 0:
        raise ValueError("the tree needs T > 0 and sigma > 0")
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk neutral probability p = {p:.4f} is outside (0, 1): the steps are too long for "
            "this rate and volatility, use more steps"
        )
    return CRRParams(dt, u, d, p, math.exp(-r * dt))


def backward_step(
    values: npt.ArrayLike,
    p: float,
    disc: float,
    exercise: npt.ArrayLike | None = None,
) -> FloatArray:
    """option values 1 time step earlier: n+1 values at step i+1 -> n values at step i

    each node is worth disc * (p * up child + (1 - p) * down child). for an american option,
    pass the exercise values at step i: the node is then worth the larger of the two
    """
    v = np.asarray(values, dtype=np.float64)
    cont = disc * (p * v[1:] + (1.0 - p) * v[:-1])
    if exercise is None:
        return np.asarray(cont, dtype=np.float64)
    # LEARNING NOTE: the early exercise check is just a max: the holder of an american option
    # compares keeping it (the discounted average of the next step) with exercising now, and
    # takes the better one. that single max is the whole difference w/ a european option
    return np.asarray(np.maximum(cont, np.asarray(exercise, dtype=np.float64)), dtype=np.float64)


def crr_price(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    steps: int = 1000,
    american: bool = False,
) -> float:
    """price on a cox ross rubinstein tree w/ `steps` time steps, european or american"""
    _check_option(option)
    if S <= 0 or K <= 0:
        raise ValueError("spot S and strike K must be > 0")
    c = crr_params(T, r, sigma, steps, q)
    # node j at step i has the stock at S u^j d^(i-j) = S u^(2j - i)
    j = np.arange(steps + 1)
    values = payoff(option, S * c.u ** (2 * j - steps), K)
    for i in range(steps - 1, -1, -1):
        exercise = payoff(option, S * c.u ** (2 * j[: i + 1] - i), K) if american else None
        values = backward_step(values, c.p, c.disc, exercise)
    return float(values[0])
