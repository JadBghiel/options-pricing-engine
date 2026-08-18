"""the c++ core behind the same signatures as the python modules, so 1 test suite runs on both

from ope import cpp
cpp.price("call", 42, 40, 0.5, 0.1, 0.2)   # same as ope.price, computed in c++
cpp.mc_price("call", 42, 40, 0.5, 0.1, 0.2, n_paths=1_000_000, rng=0, threads=8)

extra keyword: threads (None = every core) on the chain and monte carlo functions. results
dont depend on it. random draws come from philox in c++, not from numpy, so a seed gives a
different (equally valid) estimate than ope.mc_price. same normals in -> same price out,
see mc_price_from_normals
"""

import numpy as np
import numpy.typing as npt

from ope import _core
from ope.black_scholes import FloatArray, FloatOrArray, Greeks, OptionType, _check_option
from ope.monte_carlo import MCResult, RngLike

__all__ = [
    "build_type",
    "call_price",
    "crr_price",
    "greeks",
    "hardware_threads",
    "mc_price",
    "mc_price_from_normals",
    "normals",
    "price",
    "put_price",
    "seed_of",
]

build_type = _core.build_type


def hardware_threads() -> int:
    """threads the machine runs at once, what threads=None uses"""
    return _core.hardware_threads()


_Flat = tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]


def _threads(threads: int | None) -> int:
    return 0 if threads is None else threads  # 0 = all cores on the c++ side


def _chain(
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    q: npt.ArrayLike,
) -> tuple[tuple[int, ...], _Flat]:
    # numpy broadcasting, then 1 flat contiguous float64 array per input: the c++ side only
    # sees 6 arrays of the same length
    arrays = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (S, K, T, r, sigma, q)))
    S_, K_, T_, r_, sigma_, q_ = (np.ascontiguousarray(a).ravel() for a in arrays)
    return arrays[0].shape, (S_, K_, T_, r_, sigma_, q_)


def _out(x: FloatArray, shape: tuple[int, ...]) -> FloatOrArray:
    return float(x[0]) if shape == () else x.reshape(shape)


def price(
    option: OptionType,
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    q: npt.ArrayLike = 0.0,
    *,
    threads: int | None = None,
) -> FloatOrArray:
    """black scholes price in c++, same as ope.price"""
    _check_option(option)
    shape, flat = _chain(S, K, T, r, sigma, q)
    return _out(_core.bs_price(option, *flat, _threads(threads)), shape)


def call_price(
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    q: npt.ArrayLike = 0.0,
    *,
    threads: int | None = None,
) -> FloatOrArray:
    """european call in c++, same as ope.call_price"""
    return price("call", S, K, T, r, sigma, q, threads=threads)


def put_price(
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    q: npt.ArrayLike = 0.0,
    *,
    threads: int | None = None,
) -> FloatOrArray:
    """european put in c++, same as ope.put_price"""
    return price("put", S, K, T, r, sigma, q, threads=threads)


def greeks(
    option: OptionType,
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    q: npt.ArrayLike = 0.0,
    *,
    threads: int | None = None,
) -> Greeks:
    """delta, gamma, vega, theta and rho in c++, same units as ope.greeks"""
    _check_option(option)
    shape, flat = _chain(S, K, T, r, sigma, q)
    out = _core.bs_greeks(option, *flat, _threads(threads))
    return Greeks(*(_out(g, shape) for g in out))


def seed_of(rng: RngLike) -> int:
    """the 64 bit philox key for a seed, a Generator or None (fresh entropy)

    goes through numpy's SeedSequence like the python side, so rng=3 and
    rng=np.random.default_rng(3) give the same c++ stream, and seeds 0, 1, 2 give unrelated keys
    """
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    return int(gen.integers(0, 2**64, dtype=np.uint64))


def normals(n: int, rng: RngLike = None, *, first: int = 0) -> FloatArray:
    """standard normals number first .. first + n - 1 of the c++ stream for rng

    exactly the draws mc_price(rng=rng) uses, so python can rerun the c++ simulation
    """
    return _core.normals(n, seed_of(rng), first)


def mc_price(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    n_paths: int = 100_000,
    antithetic: bool = False,
    control_variate: bool = False,
    rng: RngLike = None,
    threads: int | None = None,
) -> MCResult:
    """monte carlo price in c++ on `threads` threads, same estimator as ope.mc_price"""
    _check_option(option)
    est, se, n = _core.mc_price(
        option,
        S,
        K,
        T,
        r,
        sigma,
        q,
        n_paths,
        antithetic,
        control_variate,
        seed_of(rng),
        _threads(threads),
    )
    return MCResult(est, se, n)


def mc_price_from_normals(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    z: npt.ArrayLike,
    antithetic: bool = False,
    control_variate: bool = False,
) -> MCResult:
    """same as ope.mc_price_from_normals, in c++ (1 thread)"""
    _check_option(option)
    zz = np.ascontiguousarray(z, dtype=np.float64).ravel()
    est, se, n = _core.mc_price_from_normals(
        option, S, K, T, r, sigma, q, zz, antithetic, control_variate
    )
    return MCResult(est, se, n)


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
    """cox ross rubinstein tree in c++, same as ope.crr_price"""
    _check_option(option)
    return _core.crr_price(option, S, K, T, r, sigma, q, steps, american)
