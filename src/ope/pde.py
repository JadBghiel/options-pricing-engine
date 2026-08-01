"""black scholes pde solved by finite differences: crank nicolson in time, psor for american

we work in x = ln S, where the pde has constant coefficients:
  dV/dtau = 1/2 sigma^2 V_xx + (r - q - sigma^2/2) V_x - r V,   tau = time to expiry
starting from the payoff at tau = 0 and stepping to tau = T. the strike sits exactly on a grid
node, the spot is read off the grid by interpolation
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_banded

from ope.black_scholes import FloatArray, OptionType, _check_option
from ope.monte_carlo import payoff


@dataclass(frozen=True)
class PDESolution:
    """option values today on the grid, plus the early exercise boundary over time

    times are calendar times from today (0) to expiry (T); boundary[i] is the stock price where
    exercise starts at times[i] (nan when no node is exercised, e.g. a european option)
    """

    spots: FloatArray
    values: FloatArray
    times: FloatArray
    boundary: FloatArray

    def price_at(self, S: float) -> float:
        """value today at spot S, interpolated in log price between grid nodes"""
        return float(np.interp(math.log(S), np.log(self.spots), self.values))


def _grid(S: float, K: float, T: float, sigma: float, n_space: int, width: float) -> FloatArray:
    """log price grid covering width standard deviations around S and K, w/ ln K on a node"""
    spread = width * sigma * math.sqrt(T)
    lo = min(math.log(S), math.log(K)) - spread
    hi = max(math.log(S), math.log(K)) + spread
    dx = (hi - lo) / n_space
    # LEARNING NOTE: the payoff has a kink at the strike. if ln K falls between 2 nodes the kink
    # gets smeared and the error stops shrinking smoothly as the grid gets finer, so the grid is
    # shifted to put ln K exactly on a node
    k = round((math.log(K) - lo) / dx)
    start = math.log(K) - k * dx
    return np.asarray(start + dx * np.arange(n_space + 1), dtype=np.float64)


def _edges(
    option: OptionType,
    s_lo: float,
    s_hi: float,
    K: float,
    tau: float,
    r: float,
    q: float,
    american: bool,
) -> tuple[float, float]:
    """option value at the 2 grid ends: far out of the money -> 0, deep in -> forward value"""
    stock_lo, stock_hi = s_lo * math.exp(-q * tau), s_hi * math.exp(-q * tau)
    strike = K * math.exp(-r * tau)
    if option == "call":
        lo, hi = 0.0, max(stock_hi - strike, 0.0)
        if american:
            hi = max(hi, s_hi - K)
    else:
        lo, hi = max(strike - stock_lo, 0.0), 0.0
        if american:
            lo = max(lo, K - s_lo)
    return lo, hi


def _psor(
    lower: float,
    diag: float,
    upper: float,
    rhs: list[float],
    floor: list[float],
    start: list[float],
    omega: float,
    tol: float,
    max_iter: int,
) -> list[float]:
    """solve the tridiagonal system w/ the constraint y >= floor, by projected SOR

    gauss seidel sweeps over the nodes, over-relaxed by omega, and each node is pushed back up to
    the exercise value whenever it falls below it
    """
    y = list(start)
    n = len(y)
    for _ in range(max_iter):
        change = 0.0
        for j in range(n):
            left = y[j - 1] if j > 0 else 0.0
            right = y[j + 1] if j < n - 1 else 0.0
            gs = (rhs[j] - lower * left - upper * right) / diag
            new = y[j] + omega * (gs - y[j])
            # LEARNING NOTE: the "projected" in psor is this max: an american option can never be
            # worth less than exercising now, so each sweep pushes the value back up to the
            # payoff where it dips below. where it does, that node is in the exercise region
            if new < floor[j]:
                new = floor[j]
            change = max(change, abs(new - y[j]))
            y[j] = new
        if change < tol:
            return y
    raise RuntimeError(f"psor did not converge in {max_iter} sweeps")


def pde_solve(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    american: bool = False,
    n_space: int = 400,
    n_time: int = 400,
    width: float = 6.0,
    rannacher: int = 2,
    omega: float = 1.2,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> PDESolution:
    """solve the black scholes pde on a log price grid from expiry back to today

    n_space intervals in ln S, n_time crank nicolson steps. the first `rannacher` steps are each
    replaced by 2 fully implicit half steps, which damps the wobble the payoff's kink causes
    """
    _check_option(option)
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("the pde solver needs S, K, T and sigma > 0")
    if n_space < 10 or n_time < 1:
        raise ValueError("need n_space >= 10 and n_time >= 1")
    if not 1.0 <= omega < 2.0:
        raise ValueError("omega must be in [1, 2) for psor to converge")

    x = _grid(S, K, T, sigma, n_space, width)
    spots = np.exp(x)
    dx = x[1] - x[0]
    a = 0.5 * sigma**2 / dx**2
    b = (r - q - 0.5 * sigma**2) / (2.0 * dx)
    lo_c, mid_c, up_c = a - b, -(2.0 * a + r), a + b  # L V_j = lo V_{j-1} + mid V_j + up V_{j+1}

    v = payoff(option, spots, K)
    exercise = v[1:-1].tolist()
    # time steps: `rannacher` crank nicolson steps become 2 implicit half steps each
    dtau = T / n_time
    steps = [(0.5 * dtau, 1.0)] * (2 * min(rannacher, n_time)) + [(dtau, 0.5)] * (
        n_time - min(rannacher, n_time)
    )

    taus, bounds = [0.0], [float("nan")]
    tau = 0.0
    for h, theta in steps:
        tau += h
        e_lo, e_hi = _edges(option, spots[0], spots[-1], K, tau, r, q, american)
        lv = lo_c * v[:-2] + mid_c * v[1:-1] + up_c * v[2:]
        rhs = v[1:-1] + (1.0 - theta) * h * lv
        rhs[0] += theta * h * lo_c * e_lo
        rhs[-1] += theta * h * up_c * e_hi
        lower, diag, upper = -theta * h * lo_c, 1.0 - theta * h * mid_c, -theta * h * up_c
        if american:
            start = np.maximum(v[1:-1], exercise).tolist()
            inner = np.asarray(
                _psor(lower, diag, upper, rhs.tolist(), exercise, start, omega, tol, max_iter)
            )
        else:
            m = rhs.size
            band = np.zeros((3, m))
            band[0, 1:] = upper
            band[1, :] = diag
            band[2, :-1] = lower
            inner = solve_banded((1, 1), band, rhs)
        v = np.concatenate([[e_lo], inner, [e_hi]])
        taus.append(tau)
        bounds.append(_boundary(option, spots, v, K) if american else float("nan"))

    times = T - np.asarray(taus[::-1])
    return PDESolution(spots, np.asarray(v, dtype=np.float64), times, np.asarray(bounds[::-1]))


def _boundary(option: OptionType, spots: FloatArray, v: npt.NDArray[np.float64], K: float) -> float:
    """stock price where exercise starts: highest exercised node for a put, lowest for a call"""
    intrinsic = payoff(option, spots, K)
    exercised = (intrinsic > 0) & (v - intrinsic <= 1e-9 * K)
    if not exercised.any():
        return float("nan")
    return float(spots[exercised].max() if option == "put" else spots[exercised].min())


def pde_price(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    american: bool = False,
    n_space: int = 400,
    n_time: int = 400,
) -> float:
    """price from the pde solver, see pde_solve for the method"""
    sol = pde_solve(option, S, K, T, r, sigma, q, american=american, n_space=n_space, n_time=n_time)
    return sol.price_at(S)
