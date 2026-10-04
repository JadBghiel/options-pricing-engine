"""the synthetic chain: the committed fixture is what synthetic_chain() makes, quotes bracket
the true price, the junk is what meta says"""

import math
from pathlib import Path

import numpy as np

from ope.chain import load_snapshot
from ope.synthetic import JUNK, synthetic_chain

FIXTURE = Path(__file__).parent / "fixtures" / "spx_synthetic.csv"


def test_the_fixture_is_the_default_synthetic_chain() -> None:
    saved, made = load_snapshot(FIXTURE), synthetic_chain()
    assert np.array_equal(saved.expiry, made.expiry)
    assert np.array_equal(saved.right, made.right)
    assert np.array_equal(saved.strike, made.strike)
    # bid and ask are rounded to the tick: a price on a tick boundary could round the other
    # way on another machine, so 1 tick of slack
    for name in ("bid", "ask"):
        a, b = getattr(saved, name), getattr(made, name)
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.nanmax(np.abs(a - b)) <= 0.1 + 1e-9
    assert dict(saved.meta) == dict(made.meta)
    assert saved.meta["source"] == "synthetic"


def test_quotes_bracket_the_true_price() -> None:
    chain = synthetic_chain(junk=False)
    two = np.isfinite(chain.bid)
    # last holds the true price rounded to the cent
    assert np.all(chain.bid[two] <= chain.last[two] + 0.005)
    assert np.all(chain.ask >= chain.last - 0.005)
    assert np.all(chain.open_interest >= 1)
    assert chain.meta["junk"] == [f"{k}: 0" for k in JUNK]


def test_junk_counts_in_meta() -> None:
    chain = synthetic_chain(seed=3)
    assert chain.meta["junk"] == [f"{k}: {n}" for k, n in JUNK.items()]
    assert int(np.sum(chain.mid > 9000)) == JUNK["out of bounds"]
    assert math.isfinite(float(chain.meta["spot_hint"]))
