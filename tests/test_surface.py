"""cleaning, the forward from parity and the iv surface, on chains whose truth is known"""

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from ope.chain import Chain, from_rows, load_snapshot
from ope.heston import heston_implied_vol
from ope.surface import (
    MIN_VOL_SPREAD,
    Forward,
    clean_chain,
    implied_forwards,
    iv_surface,
)
from ope.synthetic import AS_OF, JUNK, SPX_LIKE, synthetic_chain

FIXTURE = Path(__file__).parent / "fixtures" / "spx_synthetic.csv"
S, R, Q = 5000.0, 0.045, 0.013


@pytest.fixture(scope="module")
def pair() -> tuple[Chain, Chain]:
    """the same chain w/o and w/ the planted junk (same seed: only the broken rows differ)"""
    return synthetic_chain(junk=False), synthetic_chain(junk=True)


def _planted(clean: Chain, dirty: Chain) -> dict[str, list[int]]:
    """which rows were broken, and how, read from what changed"""
    kinds: dict[str, list[int]] = {k: [] for k in JUNK}
    for i in range(len(clean)):
        if np.isnan(dirty.bid[i]) and not np.isnan(clean.bid[i]):
            kinds["one sided"].append(i)
        elif dirty.bid[i] >= dirty.ask[i]:
            kinds["crossed"].append(i)
        elif dirty.last_trade_time[i] != clean.last_trade_time[i]:
            kinds["stale"].append(i)
        elif dirty.volume[i] == 0 and dirty.open_interest[i] == 0 and clean.open_interest[i] > 0:
            kinds["no interest"].append(i)
        elif dirty.mid[i] > 9000:
            kinds["out of bounds"].append(i)
        elif dirty.ask[i] != clean.ask[i]:
            kinds["wide"].append(i)
    return kinds


def test_each_planted_quote_is_caught_by_its_own_filter(pair: tuple[Chain, Chain]) -> None:
    clean, dirty = pair
    kinds = _planted(clean, dirty)
    assert {k: len(v) for k, v in kinds.items()} == JUNK
    by_filter = {
        "one sided": "no two sided quote",
        "crossed": "crossed or locked",
        "stale": "stale",
        "wide": "wide spread",
        "no interest": "no volume, no open interest",
    }
    for kind, name in by_filter.items():
        rows = np.zeros(len(dirty), dtype=bool)
        rows[kinds[kind]] = True
        _, report = clean_chain(dirty.select(rows))
        assert report.removed[name] == JUNK[kind], kind
        assert report.n_kept == 0, kind
    # out of bounds needs the forward: the broken rows must be gone from the surface
    surface = iv_surface(dirty)
    kept = set(
        zip(surface.expiry.tolist(), surface.K.tolist(), surface.right.tolist(), strict=True)
    )
    for i in kinds["out of bounds"]:
        assert (str(dirty.expiry[i]), float(dirty.strike[i]), str(dirty.right[i])) not in kept


def test_the_report_adds_up(pair: tuple[Chain, Chain]) -> None:
    surface = iv_surface(pair[1])
    r = surface.report
    assert r.n_raw == len(pair[1])
    assert r.n_raw - sum(r.removed.values()) == r.n_kept == len(surface)
    assert any("kept" in line for line in r.lines())


@pytest.mark.parametrize("seed", range(10))
def test_forward_and_discount_match_the_chain_over_seeds(seed: int) -> None:
    chain = synthetic_chain(seed=seed)
    # the out of bounds junk that reaches parity: its partner (same strike, other right) must
    # have survived the price filters too, else there is no pair to drop
    priced, _ = clean_chain(chain, liquidity=False)
    have = set(
        zip(priced.expiry.tolist(), priced.strike.tolist(), priced.right.tolist(), strict=True)
    )
    planted: Counter[str] = Counter()
    for i in np.flatnonzero(chain.mid > 9000):
        other = "P" if chain.right[i] == "C" else "C"
        if (str(chain.expiry[i]), float(chain.strike[i]), other) in have:
            planted[str(chain.expiry[i])] += 1
    surface = iv_surface(chain)
    assert len(surface.forwards) == 7
    for f in surface.forwards.values():
        F_true = S * math.exp((R - Q) * f.T)
        assert abs(f.F / F_true - 1) < 2e-4  # 2 bp
        # D off by 2.8e-4 at most over these seeds (r off by 0.06% at 6 months)
        assert abs(f.D - math.exp(-R * f.T)) < 5e-4
        assert f.n_outliers == planted[f.expiry]  # the garbage, and no good pair
        if f.T > 0.1:  # at 1 week D is within 1e-3 of 1 and r is noise
            assert abs(f.r - R) < 1e-3
            assert abs(f.q(S) - Q) < 1e-3


def test_one_bad_pair_doesnt_move_the_forward(pair: tuple[Chain, Chain]) -> None:
    clean = pair[0]
    clean, _ = clean_chain(clean, liquidity=False)
    before = implied_forwards(clean)
    e = clean.expiries()[3]
    i = int(np.flatnonzero((clean.expiry == e) & (clean.right == "C"))[5])
    bid, ask = clean.bid.copy(), clean.ask.copy()
    bid[i] += 300.0
    ask[i] += 300.0
    fat_finger = Chain(**{**_columns(clean), "bid": bid, "ask": ask}, meta=clean.meta)
    after = implied_forwards(fat_finger)
    assert after[e].n_outliers == 1
    assert abs(after[e].F / before[e].F - 1) < 1e-5


def _columns(chain: Chain) -> dict[str, np.ndarray]:
    from ope.chain import COLUMNS

    return {name: getattr(chain, name) for name in COLUMNS}


def test_an_expiry_w_too_few_pairs_gets_no_forward() -> None:
    meta = {"as_of": AS_OF.isoformat()}
    rows: list[dict[str, str | float]] = [
        {"expiry": "20260116", "right": r, "strike": k, "bid": b, "ask": b + 0.2,
         "volume": 1.0, "open_interest": 1.0}
        for k, r, b in ((5000.0, "C", 40.0), (5000.0, "P", 37.0), (5100.0, "C", 10.0))
    ]  # fmt: skip
    surface = iv_surface(from_rows(rows, meta))
    assert surface.forwards == {}
    assert surface.report.removed["expiry w/o a forward"] == 3
    assert surface.report.notes["stale"] == "the source gives no trade times"


def test_forward_properties() -> None:
    f = Forward("20270108", 1.0, 5100.0, math.exp(-0.05), 10, 0.1)
    assert f.r == pytest.approx(0.05)
    assert f.q(5000.0) == pytest.approx(0.05 - math.log(5100 / 5000))


def test_surface_vols_sit_on_the_true_heston_smile() -> None:
    surface = load_surface_no_junk()
    for e in surface.expiries():
        m = surface.rows(e)
        f = surface.forwards[e]
        true = np.asarray(heston_implied_vol(S, surface.K[m], f.T, R, SPX_LIKE, Q))
        # the mid is inside the spread by construction, so its vol is within the vol spread
        assert np.all(np.abs(surface.iv[m] - true) <= surface.vol_spread[m] + 1e-4), e
    assert np.all(surface.weights <= 1 / MIN_VOL_SPREAD)
    assert np.all(np.isin(surface.right[surface.k < 0], ["P"]))


def load_surface_no_junk():  # type: ignore[no-untyped-def]
    return iv_surface(synthetic_chain(junk=False))


def test_the_fixture_runs_through() -> None:
    surface = iv_surface(load_snapshot(FIXTURE))
    assert len(surface) > 100
    assert len(surface.expiries()) == 7
