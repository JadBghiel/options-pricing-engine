"""chain snapshots: same bits after a round trip, the time to expiry, the session close"""

import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from fakes import SATURDAY, TUESDAY_OPEN
from ope.chain import (
    COLUMNS,
    from_rows,
    last_session_close,
    load_snapshot,
    pick_expiries,
    pick_strikes,
    save_snapshot,
    snapshot_as_of,
    strike_band,
    year_fraction,
)

FIXTURE = Path(__file__).parent / "fixtures" / "spx_synthetic.csv"


def test_round_trip_gives_the_same_bits(tmp_path: Path) -> None:
    chain = load_snapshot(FIXTURE)
    again = load_snapshot(save_snapshot(chain, tmp_path / "copy"))
    for name in COLUMNS:
        a, b = getattr(chain, name), getattr(again, name)
        if a.dtype.kind == "f":
            assert np.array_equal(a, b, equal_nan=True), name
        else:
            assert np.array_equal(a, b), name
    assert dict(again.meta) == dict(chain.meta)


def test_load_from_the_stem_too() -> None:
    assert len(load_snapshot(FIXTURE.with_suffix(""))) == len(load_snapshot(FIXTURE))


def test_year_fraction_runs_to_4pm_new_york() -> None:
    # friday 16:15 new york -> the next friday 16:00: 7 days minus 15 minutes
    as_of = datetime(2026, 1, 9, 21, 15, tzinfo=UTC)
    expected = (7 * 24 - 0.25) / 24 / 365
    assert year_fraction(as_of, "20260116") == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("now", "close"),
    [
        (datetime(2026, 10, 24, 14, 44, tzinfo=UTC), datetime(2026, 10, 23, 20, 15, tzinfo=UTC)),
        (datetime(2026, 10, 26, 12, 0, tzinfo=UTC), datetime(2026, 10, 23, 20, 15, tzinfo=UTC)),
        (datetime(2026, 10, 27, 16, 0, tzinfo=UTC), None),  # tuesday noon in new york
        (datetime(2026, 10, 27, 21, 0, tzinfo=UTC), datetime(2026, 10, 27, 20, 15, tzinfo=UTC)),
    ],
    ids=["saturday", "monday before the open", "tuesday open", "tuesday after the close"],
)
def test_last_session_close(now: datetime, close: datetime | None) -> None:
    assert last_session_close(now) == close


def test_select_mid_and_expiries() -> None:
    chain = load_snapshot(FIXTURE)
    calls = chain.select(chain.right == "C")
    assert set(calls.right.tolist()) == {"C"}
    assert calls.meta == chain.meta
    i = int(np.flatnonzero(np.isfinite(chain.bid))[0])
    assert chain.mid[i] == 0.5 * (chain.bid[i] + chain.ask[i])
    assert chain.expiries() == sorted(set(chain.expiry.tolist()))
    assert np.all(np.diff([chain.T[chain.expiry == e][0] for e in chain.expiries()]) > 0)


def test_from_rows_fills_missing_numbers_with_nan() -> None:
    chain = from_rows([{"expiry": "20260116", "right": "C", "strike": 100.0, "bid": ""}], {})
    assert chain.strike[0] == 100.0
    assert math.isnan(chain.bid[0]) and math.isnan(chain.volume[0])


def test_bad_chains_raise() -> None:
    good = {"expiry": "20260116", "right": "C", "strike": 100.0}
    with pytest.raises(ValueError, match="right must be"):
        from_rows([{**good, "right": "X"}], {})
    chain = from_rows([good], {})
    with pytest.raises(ValueError, match="rows"):
        type(chain)(**{**{n: getattr(chain, n) for n in COLUMNS}, "bid": np.zeros(2)})


# what goes in a snapshot


def test_pick_expiries_nearest_once_and_not_too_short() -> None:
    as_of = datetime(2026, 10, 23, 20, 15, tzinfo=UTC)
    # sunday 16:00 is 1.99 days after friday 16:15 -> skipped, monday is 2.99 -> kept
    listed = ["20261023", "20261025", "20261026", "20261030", "20261106", "20270115"]
    expected = ["20261026", "20261030", "20261106", "20270115"]
    assert pick_expiries(listed, as_of, (1, 7, 8, 14, 90)) == expected
    assert pick_expiries(["20261023", "20261025"], as_of) == []


def test_pick_strikes_stays_in_the_band_and_prefers_round_ones() -> None:
    ks = [float(k) for k in range(1000, 10000, 5)]
    for T in (7 / 365, 0.25, 1.0):
        lo, hi = strike_band(T)
        picks = pick_strikes(ks, 6500.0, T, n=30)
        assert 0 < len(picks) <= 30
        assert all(lo <= math.log(k / 6500) <= hi for k in picks)
    assert all(k % 50 == 0 for k in pick_strikes(ks, 6500.0, 1.0))
    assert len(pick_strikes([6500.0, 6505.0], 6500.0, 1.0)) == 2  # too few round ones: keep all


def test_snapshot_as_of() -> None:
    assert snapshot_as_of(SATURDAY) == datetime(2026, 10, 23, 20, 15, tzinfo=UTC)
    assert snapshot_as_of(TUESDAY_OPEN) == TUESDAY_OPEN
    assert snapshot_as_of(TUESDAY_OPEN, 3) == datetime(2026, 10, 27, 18, 45, tzinfo=UTC)
