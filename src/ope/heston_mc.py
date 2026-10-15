"""heston by monte carlo: full truncation euler for the variance, log euler for the stock

the bias from the time step is measured w/ heston_price (the semi analytical price) as the
reference, more paths shrink the noise and leave the bias as it is
"""

import math
from collections.abc import Iterator

import numpy as np

from ope.black_scholes import FloatArray, OptionType, _check_option
from ope.exotics import _control_variate, _result
from ope.heston import HestonParams, _check_market
from ope.monte_carlo import MCResult, RngLike, _rng, payoff


def _walk(
    S: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool,
    rng: np.random.Generator,
) -> Iterator[tuple[FloatArray, FloatArray]]:
    """(log S, v) after each of the n_steps euler steps, w/ antithetic path i + n/2 uses -z"""
    p = params
    dt = T / n_steps
    sq_dt = math.sqrt(dt)
    rho_bar = math.sqrt(1.0 - p.rho**2)
    x = np.full(n_paths, math.log(S))
    v = np.full(n_paths, p.v0)
    for _ in range(n_steps):
        z = rng.standard_normal((2, n_paths // 2 if antithetic else n_paths))
        if antithetic:
            z = np.concatenate([z, -z], axis=1)
        z_s = z[0]
        z_v = p.rho * z_s + rho_bar * z[1]  # correlated w/ the stock's draw
        # LEARNING NOTE: full truncation (lord et al. 2010), euler lets v go below 0, and sqrt(v)
        # then fails, v itself is left negative, but every term that uses it takes max(v, 0)
        # w/ the feller condition violated it is common: 37% of steps at 2 steps a year, 8% at
        # 256 on our example, lord et al. found it the least biased of the simple fixes
        # (reflecting |v|, or truncating v itself)
        v_pos = np.maximum(v, 0.0)
        vol = np.sqrt(v_pos) * sq_dt
        x = x + (r - q - 0.5 * v_pos) * dt + vol * z_s  # log euler: E[S] stays exact
        v = v + p.kappa * (p.theta - v_pos) * dt + p.xi * vol * z_v
        yield x, v


def _check_mc(n_steps: int, n_paths: int, antithetic: bool) -> None:
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if n_paths < 2:
        raise ValueError("n_paths must be >= 2 to estimate a standard error")
    if antithetic and n_paths % 2:
        raise ValueError("antithetic sampling needs an even n_paths (paths come in pairs)")
    if antithetic and n_paths < 4:
        raise ValueError("antithetic sampling needs n_paths >= 4 (2 independent pairs)")


def heston_paths(
    S: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float = 0.0,
    *,
    n_steps: int,
    n_paths: int,
    rng: RngLike = None,
) -> tuple[FloatArray, FloatArray]:
    """stock and variance paths, each shape (n_paths, n_steps + 1), column 0 = today

    the variance is the raw euler one: it can dip below 0 where the pricer uses max(v, 0)
    """
    _check_market(S, T)
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be >= 1")
    s_out = np.empty((n_paths, n_steps + 1))
    v_out = np.empty((n_paths, n_steps + 1))
    s_out[:, 0], v_out[:, 0] = S, params.v0
    walk = _walk(S, T, r, params, q, n_steps, n_paths, False, _rng(rng))
    for i, (x, v) in enumerate(walk, start=1):
        s_out[:, i], v_out[:, i] = np.exp(x), v
    return s_out, v_out


def heston_mc(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float = 0.0,
    *,
    n_steps: int = 100,
    n_paths: int = 100_000,
    antithetic: bool = False,
    control_variate: bool = False,
    rng: RngLike = None,
) -> MCResult:
    """european price under heston by monte carlo, full truncation euler w/ n_steps time steps

    the standard error measures the noise, the euler scheme also has a bias that shrinks
    w/ n_steps (+1.14 at 2 steps, about +0.03 at 64 on our 1 year atm call), compare to
    heston_price to see it, control_variate: the discounted stock, its mean S e^{-qT} is exact
    in this scheme too
    """
    _check_option(option)
    _check_market(S, T)
    if not (math.isfinite(K) and K > 0):
        raise ValueError("strike K must be finite and > 0")
    _check_mc(n_steps, n_paths, antithetic)
    x = np.empty(0)
    for log_s, _ in _walk(S, T, r, params, q, n_steps, n_paths, antithetic, _rng(rng)):
        x = log_s  # a european payoff needs S_T alone
    s_t = np.exp(x)
    disc = math.exp(-r * T)
    pay = disc * payoff(option, s_t, K)
    stock = disc * s_t
    if antithetic:  # standard error from the pair averages, as in tier 2
        half = n_paths // 2
        pay = 0.5 * (pay[:half] + pay[half:])
        stock = 0.5 * (stock[:half] + stock[half:])
    if control_variate:
        pay = _control_variate(pay, stock, S * math.exp(-q * T))
    return _result(pay, n_paths)
