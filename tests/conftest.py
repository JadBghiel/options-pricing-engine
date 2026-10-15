"""shared fixtures, impl runs a test twice: on the python modules and on the c++ core

ope and ope.cpp expose the same names (price, greeks, mc_price, crr_price...) w/ the same
signatures -> a test written on impl covers both implementations
"""

from types import ModuleType

import pytest

import ope
from ope import cpp


@pytest.fixture(params=[ope, cpp], ids=["python", "cpp"])
def impl(request: pytest.FixtureRequest) -> ModuleType:
    module: ModuleType = request.param
    return module
