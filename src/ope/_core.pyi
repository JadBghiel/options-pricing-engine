"""types of the compiled module ope._core (cpp/python/module.cpp), for mypy. use ope.cpp"""

import numpy as np
import numpy.typing as npt

_Array = npt.NDArray[np.float64]

build_type: str
compiler: str
mc_chunk: int

def hardware_threads() -> int: ...
def bs_price(
    option: str,
    S: _Array,
    K: _Array,
    T: _Array,
    r: _Array,
    sigma: _Array,
    q: _Array,
    threads: int,
) -> _Array: ...
def bs_greeks(
    option: str,
    S: _Array,
    K: _Array,
    T: _Array,
    r: _Array,
    sigma: _Array,
    q: _Array,
    threads: int,
) -> tuple[_Array, _Array, _Array, _Array, _Array]: ...
def mc_price(
    option: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_paths: int,
    antithetic: bool,
    control_variate: bool,
    seed: int,
    threads: int,
) -> tuple[float, float, int]: ...
def mc_price_from_normals(
    option: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    z: _Array,
    antithetic: bool,
    control_variate: bool,
) -> tuple[float, float, int]: ...
def normals(n: int, seed: int, first: int) -> _Array: ...
def philox4x64(
    counter: tuple[int, int, int, int], key: tuple[int, int]
) -> list[int]: ...  # pybind11 turns std::array into a list
def crr_price(
    option: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    steps: int,
    american: bool,
) -> float: ...
