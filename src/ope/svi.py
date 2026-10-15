"""svi: a 5 parameter curve per expiry for the total implied variance w(k) = sigma_iv^2 T

raw form (gatheral 2004), k = log(K / F):
w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))
a level, b slope of the wings, rho tilt (rho < 0 = the equity skew), m shift, sigma how round
the bottom is, the wings are straight lines in k, as roger lee's moment formula says they must be

2 kinds of static arbitrage are checked:
- butterfly (1 expiry): the density of the price at expiry must be >= 0, for a w(k) curve that is
  g(k) >= 0 (gatheral and jacquier 2014)
- calendar (2 expiries): w must not decrease w/ T at any k, a later option cant be worth less
both on the checked range: the quoted strikes plus 10% of their width on each side, where the
data pins the fit down, far outside it a slice fitted to a narrow range can bend anywhere (g =
-86 at k = -0.35 on the real 1 week slice) -> reported as a flag, lee's wing bound is part of
every fit: on the real slices forcing it changed the error by 0.001 vol points at most
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from ope.black_scholes import FloatArray, FloatOrArray
from ope.surface import Surface

# the wide range, reported as a flag: what someone extrapolating the curve would see
K_WIDE = np.linspace(-1.5, 1.5, 601)
# checked range = quoted range +- this share of its width
MARGIN = 0.1
# a calendar gap above -this (total variance) is solver tolerance
CAL_TOL = 1e-6
# how hard a penalty pushes back on arbitrage, against residuals of about 1 per bid ask spread
PENALTY = 1e3


@dataclass(frozen=True)
class SVIParams:
    """raw svi: b >= 0, |rho| < 1, sigma > 0 and a + b sigma sqrt(1 - rho^2) >= 0 (the lowest
    total variance can't be negative)"""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (self.a, self.b, self.rho, self.m, self.sigma)):
            raise ValueError("svi params must be finite")
        if self.b < 0 or self.sigma <= 0 or not -1 < self.rho < 1:
            raise ValueError("svi needs b >= 0, sigma > 0 and -1 < rho < 1")
        if self.min_variance < -1e-12:
            raise ValueError(f"svi total variance goes below 0 (min {self.min_variance})")

    @property
    def min_variance(self) -> float:
        """the lowest total variance, at the bottom of the curve"""
        return self.a + self.b * self.sigma * math.sqrt(1 - self.rho**2)

    @property
    def lee_ok(self) -> bool:
        """roger lee's bound on the wings, b (1 + |rho|) <= 2, above it the wings grow too fast
        for any price distribution w/ a finite mean"""
        return self.b * (1 + abs(self.rho)) <= 2 + 1e-6

    def w(self, k: npt.ArrayLike) -> FloatOrArray:
        x = np.asarray(k, dtype=np.float64) - self.m
        out = self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma**2))
        return float(out) if out.ndim == 0 else out

    def w1(self, k: npt.ArrayLike) -> FloatOrArray:
        """dw/dk"""
        x = np.asarray(k, dtype=np.float64) - self.m
        out = self.b * (self.rho + x / np.sqrt(x * x + self.sigma**2))
        return float(out) if out.ndim == 0 else out

    def w2(self, k: npt.ArrayLike) -> FloatOrArray:
        """d2w/dk2"""
        x = np.asarray(k, dtype=np.float64) - self.m
        out = self.b * self.sigma**2 / (x * x + self.sigma**2) ** 1.5
        return float(out) if out.ndim == 0 else out

    def g(self, k: npt.ArrayLike) -> FloatOrArray:
        """gatheral and jacquier's g(k), the density of k is g / sqrt(2 pi w) e^{-d2^2 / 2} ->
        g < 0 somewhere = a negative probability = a butterfly that costs less than nothing"""
        kk = np.asarray(k, dtype=np.float64)
        w, w1, w2 = (np.asarray(f(kk)) for f in (self.w, self.w1, self.w2))
        out = (1 - kk * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2
        return float(out) if out.ndim == 0 else out

    def vol(self, k: npt.ArrayLike, T: float) -> FloatOrArray:
        """implied vol sqrt(w / T)"""
        out = np.sqrt(np.maximum(np.asarray(self.w(k)), 0.0) / T)
        return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class SVIFit:
    """1 expiry's fit, errors in vol points (1 = 1% of vol), unweighted"""

    expiry: str
    T: float
    params: SVIParams
    n: int
    rms: float
    max_abs: float
    within_spread: float  # share of quotes where the svi vol is between the bid and ask vols
    k_lo: float  # the checked range
    k_hi: float
    min_g: float  # min of g(k) on the checked range, < 0 = butterfly arbitrage
    min_g_wide: float  # the same on K_WIDE, a flag for extrapolation
    penalized: bool  # True if the plain fit had arbitrage and was refit w/ a penalty

    @property
    def butterfly_ok(self) -> bool:
        return self.min_g >= -1e-9

    def k_check(self, n: int = 201) -> FloatArray:
        return np.linspace(self.k_lo, self.k_hi, n)


def _from_x(x: npt.ArrayLike) -> SVIParams:
    """optimizer vector (vmin, b, rho, m, sigma) -> params, fitting vmin = the lowest total
    variance instead of a turns a + b sigma sqrt(1 - rho^2) >= 0 into a simple bound vmin >= 0"""
    vmin, b, rho, m, s = (float(v) for v in np.asarray(x, dtype=np.float64))
    a = vmin - b * s * math.sqrt(1 - rho * rho)
    return SVIParams(a, b, rho, m, s)


def fit_svi(
    k: npt.ArrayLike,
    iv: npt.ArrayLike,
    T: float,
    weights: npt.ArrayLike | None = None,
    *,
    expiry: str = "",
    iv_bid: npt.ArrayLike | None = None,
    iv_ask: npt.ArrayLike | None = None,
    floor: "SVIFit | None" = None,
) -> SVIFit:
    """fit raw svi to 1 expiry's vols, least squares on the vol error times the weights

    several starts (a grid over m, sigma and rho), the best one kept, if the result has butterfly
    arbitrage on the checked range, or goes under floor (the previous expiry's fit) where both
    checked ranges overlap, it is refit from every start (and from the plain fit) w/ a penalty
    on the violations
    """
    kk = np.asarray(k, dtype=np.float64)
    vols = np.asarray(iv, dtype=np.float64)
    if kk.shape != vols.shape or kk.size < 5:
        raise ValueError("fit_svi needs k and iv of the same shape, at least 5 quotes")
    if T <= 0:
        raise ValueError("T must be > 0")
    wts = np.ones_like(kk) if weights is None else np.asarray(weights, dtype=np.float64)
    wts = wts / np.mean(wts)
    margin = MARGIN * (float(kk.max()) - float(kk.min()))
    k_lo, k_hi = float(kk.min()) - margin, float(kk.max()) + margin
    k_chk = np.linspace(k_lo, k_hi, 201)
    # the floor is compared where both fits are meant to be used: outside its own range
    # the previous fit can be anything (a 1 week slice extrapolated to k = -0.5)
    k_cal = k_chk
    w_floor = None
    if floor is not None:
        k_cal = np.linspace(max(k_lo, floor.k_lo), min(k_hi, floor.k_hi), 201)
        w_floor = np.asarray(floor.params.w(k_cal))

    def resid(x: npt.NDArray[np.float64], penalty: bool) -> FloatArray:
        p = _from_x(x)
        out = (np.asarray(p.vol(kk, T)) - vols) * 100 * wts  # vol points x weight
        # always on: w/o it the real 1 week slice took b = 9.6, rho = -0.999 (wings 10x too steep)
        out = np.append(out, PENALTY * min(2 - p.b * (1 + abs(p.rho)), 0.0))
        if not penalty:
            return out
        extra = [PENALTY * np.minimum(np.asarray(p.g(k_chk)), 0.0)]
        if w_floor is not None:
            extra.append(PENALTY * 100 * np.minimum(np.asarray(p.w(k_cal)) - w_floor, 0.0))
        return np.concatenate([out, *extra])

    atm_w = float(np.interp(0.0, kk[np.argsort(kk)], vols[np.argsort(kk)])) ** 2 * T
    lo = [0.0, 0.0, -0.999, -1.0, 1e-4]
    hi = [max(4 * atm_w, 1e-4), 10.0, 0.999, 1.0, 2.0]
    starts = [
        [0.5 * atm_w, 0.1, rho0, m0, s0]
        for rho0 in (-0.7, -0.2)
        for m0 in (-0.1, 0.0, 0.1)
        for s0 in (0.05, 0.2)
    ]

    def solve(x0: npt.ArrayLike, penalty: bool) -> tuple[npt.NDArray[np.float64], float]:
        r = least_squares(
            resid, np.clip(x0, lo, hi), args=(penalty,), bounds=(lo, hi), x_scale="jac"
        )
        return r.x, float(r.cost)

    x, _ = min((solve(x0, False) for x0 in starts), key=lambda xc: xc[1])
    p = _from_x(x)
    min_g = float(np.min(np.asarray(p.g(k_chk))))
    crosses = w_floor is not None and bool(np.any(np.asarray(p.w(k_cal)) < w_floor - CAL_TOL))
    penalized = min_g < -1e-9 or crosses
    # arbitrage is fixed when it shows up, a penalty bends the curve away from the
    # quotes -> the plain fit is kept whenever it is already clean, and a penalized
    # fit says it (SVIFit.penalized) and doesnt pass as a normal one
    if penalized:
        tries = [np.asarray(x0, dtype=np.float64) for x0 in starts] + [x]
        p = _from_x(min((solve(x0, True) for x0 in tries), key=lambda xc: xc[1])[0])
        min_g = float(np.min(np.asarray(p.g(k_chk))))

    err = (np.asarray(p.vol(kk, T)) - vols) * 100
    within = float("nan")
    if iv_bid is not None and iv_ask is not None:
        model = np.asarray(p.vol(kk, T))
        lo_iv = np.nan_to_num(np.asarray(iv_bid, dtype=np.float64), nan=0.0)
        within = float(np.mean((model >= lo_iv) & (model <= np.asarray(iv_ask))))
    return SVIFit(
        expiry,
        T,
        p,
        int(kk.size),
        float(np.sqrt(np.mean(err**2))),
        float(np.max(np.abs(err))),
        within,
        k_lo,
        k_hi,
        min_g,
        float(np.min(np.asarray(p.g(K_WIDE)))),
        penalized,
    )


def fit_svi_surface(surface: Surface, *, calendar: bool = True) -> list[SVIFit]:
    """1 svi per expiry, shortest first, calendar=True: each fit is kept above the previous one
    on its checked range (w(k) not decreasing w/ T), w/ a penalty if the plain fit goes
    under it"""
    fits: list[SVIFit] = []
    for e in surface.expiries():
        m = surface.rows(e)
        prev = fits[-1] if (calendar and fits) else None
        fits.append(
            fit_svi(
                surface.k[m],
                surface.iv[m],
                float(surface.T[m][0]),
                surface.weights[m],
                expiry=e,
                iv_bid=surface.iv_bid[m],
                iv_ask=surface.iv_ask[m],
                floor=prev,
            )
        )
    return fits


@dataclass(frozen=True)
class CalendarCheck:
    """w(T2) - w(T1) for 2 consecutive expiries, on the overlap of their checked ranges,
    min_gap < 0 = calendar arbitrage, min_gap_wide: the same on K_WIDE as a flag"""

    expiry_1: str
    expiry_2: str
    min_gap: float
    at_k: float
    min_gap_wide: float

    @property
    def ok(self) -> bool:
        return self.min_gap >= -CAL_TOL


def calendar_checks(fits: Sequence[SVIFit]) -> list[CalendarCheck]:
    """every consecutive pair of fits, sorted by T"""
    ordered = sorted(fits, key=lambda f: f.T)
    out = []
    for f1, f2 in itertools.pairwise(ordered):
        k = np.linspace(max(f1.k_lo, f2.k_lo), min(f1.k_hi, f2.k_hi), 201)
        gap = np.asarray(f2.params.w(k)) - np.asarray(f1.params.w(k))
        wide = np.asarray(f2.params.w(K_WIDE)) - np.asarray(f1.params.w(K_WIDE))
        i = int(np.argmin(gap))
        out.append(
            CalendarCheck(f1.expiry, f2.expiry, float(gap[i]), float(k[i]), float(wide.min()))
        )
    return out
