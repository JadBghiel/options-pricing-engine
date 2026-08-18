"""tier 4 validation: the c++ core against the python modules on the same inputs

the tier 1 to 3 tests already run on both sides (the impl fixture). this file checks the 2
against each other directly: same inputs -> same numbers within floating point tolerance.
the tolerances sit about 10x above the largest gap measured on 100,000 random options
"""

import itertools
import math
from typing import Literal

import numpy as np
import pytest
from scipy import stats

import ope
from ope import _core, cpp

Option = Literal["call", "put"]
HULL = (42.0, 40.0, 0.5, 0.10, 0.20)  # S, K, T, r, sigma


def _random_params(n: int, seed: int) -> dict[str, np.ndarray]:
    # wider than the tier 1 sets on purpose: tiny and huge prices, T = 0 and sigma = 0 included
    rng = np.random.default_rng(seed)
    p = {
        "S": rng.uniform(1, 300, n),
        "K": rng.uniform(1, 300, n),
        "T": rng.uniform(0, 5, n),
        "r": rng.uniform(-0.02, 0.15, n),
        "sigma": rng.uniform(0, 1.5, n),
        "q": rng.uniform(0, 0.1, n),
    }
    p["T"][:50] = 0.0
    p["sigma"][50:100] = 0.0
    return p


def test_the_extension_is_a_release_build() -> None:
    # benchmarking a debug build is a classic trap, the module says how it was compiled
    assert cpp.build_type == "Release"
    assert cpp.hardware_threads() >= 1


# black scholes


@pytest.mark.parametrize("option", ["call", "put"])
def test_prices_match_python_on_100k_random_options(option: Option) -> None:
    p = _random_params(100_000, seed=0)
    np.testing.assert_allclose(
        cpp.price(option, **p), ope.price(option, **p), rtol=1e-11, atol=1e-12
    )


@pytest.mark.parametrize("option", ["call", "put"])
def test_greeks_match_python_on_100k_random_options(option: Option) -> None:
    p = _random_params(100_000, seed=1)
    a, b = ope.greeks(option, **p), cpp.greeks(option, **p)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        np.testing.assert_allclose(
            getattr(b, name), getattr(a, name), rtol=1e-10, atol=1e-12, err_msg=name
        )


def test_broadcasting_and_shapes_follow_numpy() -> None:
    strikes = np.linspace(80, 120, 12).reshape(3, 4)
    out = np.asarray(cpp.call_price(100, strikes, 0.5, 0.03, [0.2, 0.25, 0.3, 0.35]))
    assert out.shape == (3, 4)
    np.testing.assert_allclose(
        out, ope.call_price(100, strikes, 0.5, 0.03, [0.2, 0.25, 0.3, 0.35]), rtol=1e-12
    )
    assert np.asarray(cpp.put_price(100, np.array([]), 1, 0.05, 0.2)).shape == (0,)


def test_thread_count_does_not_change_a_chain() -> None:
    p = _random_params(50_000, seed=2)
    one = cpp.price("call", **p, threads=1)
    for threads in (2, 3, 8):
        np.testing.assert_array_equal(cpp.price("call", **p, threads=threads), one)


def test_bad_chain_inputs_raise() -> None:
    with pytest.raises(ValueError, match="spot"):
        cpp.call_price([100, -1], 100, 1, 0.05, 0.2)
    with pytest.raises(ValueError, match="threads"):
        cpp.call_price(100, 100, 1, 0.05, 0.2, threads=-1)
    with pytest.raises(ValueError, match="1d"):
        _core.bs_price("call", *(np.ones((2, 2)),) * 6, 1)


# monte carlo


def test_philox_matches_numpy_bit_for_bit() -> None:
    # numpy's Philox is the same philox 4x64-10, it only adds 1 to the counter before each block
    for key, counter in itertools.product([(0, 0), (1, 2), (2**64 - 1, 12345)], [1, 2, 99, 2**40]):
        gen = np.random.Philox(key=np.array(key, dtype=np.uint64), counter=[counter - 1, 0, 0, 0])
        expected = [int(v) for v in gen.random_raw(4)]
        assert _core.philox4x64((counter, 0, 0, 0), key) == expected


def test_normals_are_standard_normal() -> None:
    z = cpp.normals(1_000_000, rng=0)
    assert abs(z.mean()) < 5 * 1e-3  # standard error of the mean at 1e6 draws: 0.001
    assert z.var() == pytest.approx(1.0, abs=0.007)
    # kolmogorov smirnov: largest gap between the sample cdf and N(x), p value under 1e-3 fails
    assert stats.kstest(z, "norm").pvalue > 1e-3


def test_seed_and_generator_give_the_same_cpp_stream() -> None:
    np.testing.assert_array_equal(cpp.normals(10, 3), cpp.normals(10, np.random.default_rng(3)))
    np.testing.assert_array_equal(cpp.normals(10, 3, first=5), cpp.normals(15, 3)[5:])
    assert not np.array_equal(cpp.normals(10, 3), cpp.normals(10, 4))


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize(
    ("antithetic", "control"), list(itertools.product([False, True], repeat=2))
)
def test_same_normals_same_price_in_python_and_cpp(
    option: Option, antithetic: bool, control: bool
) -> None:
    # c++ cant reproduce numpy's random stream, so both sides get the same pre generated normals
    for seed in range(10):
        z = np.random.default_rng(seed).standard_normal(50_001)
        args = (option, *HULL, 0.01)
        kw = {"z": z, "antithetic": antithetic, "control_variate": control}
        a, b = ope.mc_price_from_normals(*args, **kw), cpp.mc_price_from_normals(*args, **kw)
        assert b.estimate == pytest.approx(a.estimate, rel=1e-11)
        assert b.std_error == pytest.approx(a.std_error, rel=1e-11)
        assert b.n_paths == a.n_paths


@pytest.mark.parametrize("threads", [1, 2, 3, 8])
def test_threaded_cpp_run_equals_python_on_the_cpp_normals(threads: int) -> None:
    # the whole multithreaded simulation, replayed in python from the exact draws c++ used
    n = 3 * _core.mc_chunk + 123  # several chunks, the last 1 partial
    res = cpp.mc_price("call", *HULL, n_paths=n, control_variate=True, rng=7, threads=threads)
    z = cpp.normals(n, rng=7)
    ref = ope.mc_price_from_normals("call", *HULL, z=z, control_variate=True)
    assert res.estimate == pytest.approx(ref.estimate, rel=1e-11)
    assert res.std_error == pytest.approx(ref.std_error, rel=1e-11)


def test_same_seed_same_bits_for_any_thread_count() -> None:
    runs = {
        t: cpp.mc_price("put", *HULL, n_paths=200_000, antithetic=True, rng=1, threads=t)
        for t in (1, 2, 4, 8)
    }
    assert len(set(runs.values())) == 1


def test_antithetic_cpp_uses_the_mirror_of_each_draw() -> None:
    z = cpp.normals(1_000, rng=2)
    pairs = cpp.mc_price("call", *HULL, n_paths=2_000, antithetic=True, rng=2, threads=1)
    ref = ope.mc_price_from_normals("call", *HULL, z=z, antithetic=True)
    assert pairs.estimate == pytest.approx(ref.estimate, rel=1e-12)


def test_mc_price_from_normals_needs_two_independent_draws() -> None:
    for impl in (ope, cpp):
        with pytest.raises(ValueError, match="n_paths must be >= 2"):
            impl.mc_price_from_normals("call", *HULL, z=[0.1])
        # 1 antithetic pair is 2 paths but only 1 independent sample, no spread to measure
        with pytest.raises(ValueError, match="2 independent"):
            impl.mc_price_from_normals("call", *HULL, z=[0.1], antithetic=True)
        assert math.isfinite(impl.mc_price_from_normals("call", *HULL, z=[0.1, -0.3]).std_error)


# tree


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("american", [False, True])
def test_tree_matches_python(option: Option, american: bool) -> None:
    for steps, q in itertools.product([1, 2, 7, 100, 1000], [0.0, 0.04]):
        a = ope.crr_price(option, 100, 95, 1, 0.05, 0.3, q, steps=steps, american=american)
        b = cpp.crr_price(option, 100, 95, 1, 0.05, 0.3, q, steps=steps, american=american)
        assert b == pytest.approx(a, rel=1e-13, abs=1e-14)


def test_cpp_tree_errors_match_python() -> None:
    with pytest.raises(ValueError, match="outside"):
        cpp.crr_price("call", 42, 40, 1.0, 0.5, 0.01, steps=1)
    with pytest.raises(ValueError, match="steps"):
        cpp.crr_price("call", 42, 40, 1.0, 0.05, 0.2, steps=0)
    with pytest.raises(ValueError, match="sigma > 0"):
        cpp.crr_price("call", 42, 40, 1.0, 0.05, 0.0)
    with pytest.raises(ValueError, match="> 0"):
        cpp.crr_price("call", -1, 40, 1, 0.05, 0.2)
