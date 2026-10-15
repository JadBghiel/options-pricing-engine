"""heston calibrated to a whole implied vol surface at once

1 set of 5 numbers (v0, kappa, theta, xi, rho) for every expiry together, found by least squares
on the vol errors, from several starting points, heston_price is 4 to 9 ms per strike (2
adaptive integrals each), too slow for thousands of evaluations -> the prices here come from 1
shared quadrature grid per maturity, compared w/ heston_price in learning/verify/tier6/v3b
"""

import functools
import math
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.stats import qmc

from ope.black_scholes import FloatArray, OptionType, norm_pdf
from ope.heston import HestonParams, heston_cf
from ope.implied_volatility import implied_vol
from ope.surface import Surface

# box the optimizer stays in: v0, kappa, theta, xi, rho
BOUNDS = ((1e-4, 1e-2, 1e-4, 1e-2, -0.999), (1.0, 20.0, 1.0, 5.0, 0.999))
# where the starting points are drawn, the usual range for an equity index
START_BOX = ((0.005, 0.5, 0.005, 0.2, -0.95), (0.15, 6.0, 0.15, 1.5, -0.1))
# |phi(u)| / u under this -> the rest of the integral is negligible
CF_TOL = 1e-12


@functools.lru_cache(maxsize=64)
def _nodes(n: int) -> tuple[FloatArray, FloatArray]:
    """gauss legendre nodes and weights on [-1, 1], computed once per n (80% of the time
    before caching)"""
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def _cutoff(T: float, params: HestonParams) -> float:
    """the u where |phi(u)| / u drops under CF_TOL, from 1 scan on a log grid"""
    u = np.geomspace(1.0, 1e5, 200)
    a = np.abs(heston_cf(u, 1.0, T, 0.0, params)) / u
    below = np.flatnonzero(a < CF_TOL)
    return float(u[below[0]]) if below.size else 1e5


def heston_probabilities(
    kf: npt.ArrayLike, T: float, params: HestonParams
) -> tuple[FloatArray, FloatArray]:
    """P1, P2 for strikes kf = K / F of 1 maturity, on 1 shared grid

    same gil pelaez integrals as heston_price, in forward units (F = 1), phi(u) and phi(u - i)
    are computed once on the nodes, then each strike is 1 row of a matrix product
    """
    x = np.atleast_1d(np.asarray(kf, dtype=np.float64))
    log_k = np.log(x)
    U = _cutoff(T, params)
    # LEARNING NOTE: the nodes follow the oscillation of e^{-iu log(K/F)}: it turns
    # U |log(K/F)| / 2 pi times over [0, U], w/ a fixed 64 to 256 nodes the grid was off by up to
    # 0.09 F at 1 to 2 weeks and k = -0.6 (U = 2327, 220 turns), 32 + 12 nodes per turn gave
    # 1e-9 F, 1e-4 vol points vs heston_price on every case we tried
    turns = U * float(np.max(np.abs(log_k))) / (2 * math.pi)
    n = int(32 * math.ceil((32 + 12 * turns) / 32))
    xs, ws = _nodes(n)
    u = 0.5 * U * (xs + 1)
    w = 0.5 * U * ws
    phi2 = heston_cf(u, 1.0, T, 0.0, params)
    phi1 = heston_cf(u - 1j, 1.0, T, 0.0, params)  # phi(-i) = F = 1, no division
    e = np.exp(-1j * np.outer(log_k, u)) / (1j * u)
    p1 = 0.5 + (e * phi1).real @ w / math.pi
    p2 = 0.5 + (e * phi2).real @ w / math.pi
    return p1, p2


def heston_grid_prices(
    option: OptionType | npt.ArrayLike,
    F: float,
    D: float,
    T: float,
    K: npt.ArrayLike,
    params: HestonParams,
) -> FloatArray:
    """european heston prices for the strikes K of 1 maturity, w/ forward F and discount D,
    option is "call", "put" or an array of them (1 per strike)"""
    strikes = np.atleast_1d(np.asarray(K, dtype=np.float64))
    p1, p2 = heston_probabilities(strikes / F, T, params)
    calls = D * (F * p1 - strikes * p2)
    puts = D * (strikes * (1 - p2) - F * (1 - p1))  # from 1 - P, as in heston_price
    is_call = np.broadcast_to(np.asarray(option) == "call", strikes.shape)
    return np.where(is_call, calls, puts)


@dataclass(frozen=True)
class SliceError:
    """1 expiry's fit error in vol points (1 = 1% of vol)"""

    expiry: str
    T: float
    n: int
    rms: float
    max_abs: float
    within_spread: float  # share of quotes where the model vol is between the bid and ask vols


@dataclass(frozen=True)
class StartResult:
    """where 1 run of the optimizer started and ended, rms over every quote, weighted"""

    start: HestonParams
    params: HestonParams
    cost: float
    nfev: int
    seconds: float


@dataclass(frozen=True)
class HestonCalibration:
    params: HestonParams
    starts: tuple[StartResult, ...]
    errors: tuple[SliceError, ...]
    rms: float  # every quote, vol points, unweighted
    seconds: float

    @property
    def feller(self) -> bool:
        return self.params.feller


def _params(x: npt.ArrayLike) -> HestonParams:
    v0, kappa, theta, xi, rho = (float(v) for v in np.asarray(x, dtype=np.float64))
    return HestonParams(v0, kappa, theta, xi, rho)


class _Problem:
    """the surface regrouped by expiry once -> each evaluation is 1 grid per maturity"""

    def __init__(self, surface: Surface) -> None:
        self.slices = []
        weights = surface.weights / np.mean(surface.weights)
        for e in surface.expiries():
            m = surface.rows(e)
            fwd = surface.forwards[e]
            K = surface.K[m]
            iv = surface.iv[m]
            # black scholes vega in forward units at the market vol: price error / vega ~ the
            # vol error, w/o inverting a model price inside the loop
            d1 = (-np.log(K / fwd.F) + 0.5 * iv**2 * fwd.T) / (iv * math.sqrt(fwd.T))
            vega = fwd.D * fwd.F * norm_pdf(d1) * math.sqrt(fwd.T)
            option = np.where(surface.right[m] == "C", "call", "put")
            self.slices.append((e, fwd, K, option, surface.mid[m], vega, weights[m], m))

    def residuals(self, x: npt.ArrayLike) -> FloatArray:
        p = _params(x)
        out = []
        for _, fwd, K, option, mid, vega, w, _ in self.slices:
            model = heston_grid_prices(option, fwd.F, fwd.D, fwd.T, K, p)
            out.append((model - mid) / vega * 100 * w)  # ~ vol points x weight
        return np.concatenate(out)


def slice_errors(surface: Surface, params: HestonParams) -> tuple[SliceError, ...]:
    """model prices -> implied vols (tier 1 solver) -> error vs the market vols, per expiry"""
    out = []
    for e in surface.expiries():
        m = surface.rows(e)
        fwd = surface.forwards[e]
        option = np.where(surface.right[m] == "C", "call", "put")
        model = heston_grid_prices(option, fwd.F, fwd.D, fwd.T, surface.K[m], params)
        vols = np.array(
            [
                implied_vol(o, float(v), fwd.D * fwd.F, float(k), fwd.T, fwd.r, 0.0)
                for o, v, k in zip(option, model, surface.K[m], strict=True)
            ]
        )
        err = (vols - surface.iv[m]) * 100
        lo = np.nan_to_num(surface.iv_bid[m], nan=0.0)
        out.append(
            SliceError(
                e,
                fwd.T,
                int(m.sum()),
                float(np.sqrt(np.mean(err**2))),
                float(np.max(np.abs(err))),
                float(np.mean((vols >= lo) & (vols <= surface.iv_ask[m]))),
            )
        )
    return tuple(out)


def calibrate_heston(
    surface: Surface, *, n_starts: int = 8, seed: int = 0, max_nfev: int = 400
) -> HestonCalibration:
    """the heston params that best fit every quote of the surface at once

    n_starts points from a scrambled sobol sequence (seeded: the same surface and seed give the
    same numbers), each one run to convergence, the lowest cost kept, every start is reported
    """
    t0 = time.perf_counter()
    problem = _Problem(surface)
    # LEARNING NOTE: several starts, all reported, bc the heston fit has flat valleys: kappa and
    # xi trade off (a fast pull back w/ more noise looks like a slow one w/ less), 1 start
    # can stop in a local minimum and look converged, but it might not be the global minimum
    # sobol points cover the box evenly, and the seed makes the whole calibration rerun to the
    # same numbers
    sobol = qmc.Sobol(5, scramble=True, seed=seed)
    starts = qmc.scale(sobol.random(n_starts), *START_BOX)
    runs = []
    for x0 in starts:
        t1 = time.perf_counter()
        r = least_squares(problem.residuals, x0, bounds=BOUNDS, x_scale="jac", max_nfev=max_nfev)
        runs.append(
            StartResult(
                _params(x0), _params(r.x), float(r.cost), int(r.nfev), time.perf_counter() - t1
            )
        )
    best = min(runs, key=lambda s: s.cost).params
    errors = slice_errors(surface, best)
    n = sum(e.n for e in errors)
    rms = math.sqrt(sum(e.rms**2 * e.n for e in errors) / n)
    return HestonCalibration(best, tuple(runs), errors, rms, time.perf_counter() - t0)
