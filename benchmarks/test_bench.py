"""python side of the benchmarks (pytest-benchmark), run by benchmarks/run.sh

3 workloads x 4 implementations, timed from python like a user would call them:
- pure python: plain loops and the math module, the floor
- numpy: the library itself (ope.price, ope.mc_price, ope.crr_price), vectorized
- c++ 1 thread and c++ on every core: ope.cpp

the fair comparison for c++ is numpy, not the loops (STEPS.md, tier 4 traps)
"""

import math
import random
from collections.abc import Callable

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import ope
from ope import cpp

N_OPTIONS = 100_000
N_PATHS = 1_000_000
STEPS = 2_000
HULL = (42.0, 40.0, 0.5, 0.10, 0.20)  # S, K, T, r, sigma
AMERICAN_PUT = (50.0, 50.0, 5 / 12, 0.10, 0.40)  # Hull's american put

_rng = np.random.default_rng(0)
CHAIN = {
    "S": np.full(N_OPTIONS, 100.0),
    "K": _rng.uniform(50, 150, N_OPTIONS),
    "T": _rng.uniform(0.1, 2.0, N_OPTIONS),
    "r": np.full(N_OPTIONS, 0.03),
    "sigma": _rng.uniform(0.1, 0.6, N_OPTIONS),
    "q": np.zeros(N_OPTIONS),
}


# pure python versions, written for speed within plain python (no numpy)


def _bs_call_loop(
    S: list[float], K: list[float], T: list[float], sigma: list[float]
) -> list[float]:
    out = []
    r = 0.03
    for s, k, t, v in zip(S, K, T, sigma, strict=True):
        vs = v * math.sqrt(t)
        d1 = (math.log(s / k) + (r + 0.5 * v * v) * t) / vs
        d2 = d1 - vs
        n1 = 0.5 * math.erfc(-d1 / math.sqrt(2))
        n2 = 0.5 * math.erfc(-d2 / math.sqrt(2))
        out.append(s * n1 - k * math.exp(-r * t) * n2)
    return out


def _mc_call_loop(n: int) -> float:
    S, K, T, r, sigma = HULL
    gen = random.Random(0)
    drift = (r - 0.5 * sigma * sigma) * T
    vol = sigma * math.sqrt(T)
    total = 0.0
    for _ in range(n):
        total += max(S * math.exp(drift + vol * gen.gauss(0.0, 1.0)) - K, 0.0)
    return math.exp(-r * T) * total / n


def _tree_put_loop(steps: int) -> float:
    S, K, T, r, sigma = AMERICAN_PUT
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    values = [max(K - S * u**j * d ** (steps - j), 0.0) for j in range(steps + 1)]
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * values[j + 1] + (1 - p) * values[j])
            values[j] = max(cont, K - S * u**j * d ** (i - j))
    return values[0]


def _chain_lists() -> tuple[list[float], ...]:
    return tuple(CHAIN[k].tolist() for k in ("S", "K", "T", "sigma"))


WORKLOADS: dict[str, dict[str, Callable[[], object]]] = {
    "black_scholes": {
        "pure_python": lambda: _bs_call_loop(*_LISTS),
        "numpy": lambda: ope.price("call", **CHAIN),
        "cpp_1_thread": lambda: cpp.price("call", **CHAIN, threads=1),
        "cpp_all_threads": lambda: cpp.price("call", **CHAIN, threads=None),
    },
    "monte_carlo": {
        "pure_python": lambda: _mc_call_loop(N_PATHS),
        "numpy": lambda: ope.mc_price("call", *HULL, n_paths=N_PATHS, rng=0),
        "cpp_1_thread": lambda: cpp.mc_price("call", *HULL, n_paths=N_PATHS, rng=0, threads=1),
        "cpp_all_threads": lambda: cpp.mc_price("call", *HULL, n_paths=N_PATHS, rng=0),
    },
    "american_tree": {
        "pure_python": lambda: _tree_put_loop(STEPS),
        "numpy": lambda: ope.crr_price("put", *AMERICAN_PUT, steps=STEPS, american=True),
        "cpp_1_thread": lambda: cpp.crr_price("put", *AMERICAN_PUT, steps=STEPS, american=True),
    },
}
_LISTS = _chain_lists()

CASES = [(w, impl) for w, impls in WORKLOADS.items() for impl in impls]


def test_benchmarks_measure_a_release_build() -> None:
    assert cpp.build_type == "Release", "rebuild w/ cmake.build-type = Release before timing"


def test_the_four_versions_agree() -> None:
    # a fast wrong answer is worth nothing: every version prices the same thing
    bs = WORKLOADS["black_scholes"]
    ref = np.asarray(bs["numpy"]())
    for impl in ("pure_python", "cpp_1_thread", "cpp_all_threads"):
        np.testing.assert_allclose(np.asarray(bs[impl]()), ref, rtol=1e-10, atol=1e-12)
    exact = float(ope.price("call", *HULL))
    for impl, fn in WORKLOADS["monte_carlo"].items():
        res = fn()
        estimate = res if isinstance(res, float) else res.estimate  # type: ignore[attr-defined]
        assert abs(estimate - exact) < 0.03, impl  # 6 standard errors at 1e6 paths
    tree = WORKLOADS["american_tree"]
    assert tree["pure_python"]() == pytest.approx(tree["numpy"](), rel=1e-12)
    assert tree["cpp_1_thread"]() == pytest.approx(tree["numpy"](), rel=1e-12)


@pytest.mark.parametrize(("workload", "impl"), CASES, ids=[f"{w}-{i}" for w, i in CASES])
def test_speed(benchmark: BenchmarkFixture, workload: str, impl: str) -> None:
    benchmark.group = workload
    fn = WORKLOADS[workload][impl]
    if impl == "pure_python":
        # around 1 s per call, 5 rounds are plenty and keep the whole run short
        benchmark.pedantic(fn, rounds=5, iterations=1, warmup_rounds=1)
    else:
        benchmark(fn)
