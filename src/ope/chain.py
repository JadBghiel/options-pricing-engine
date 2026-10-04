"""option chain snapshots: 1 chain at 1 moment, saved as csv + json so a calibration reruns from
disk w/ the same numbers

a snapshot is plain numpy columns, 1 row per quote (expiry, strike, call or put). the fetchers in
ope.market write them, everything after (cleaning, the iv surface, svi, heston) only reads them.
no pandas: the stdlib csv module and numpy are enough for a few thousand rows
"""

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

from ope.black_scholes import FloatArray

StrArray = npt.NDArray[np.str_]
MetaValue = str | float | int | list[str]

NEW_YORK = ZoneInfo("America/New_York")
# spxw settles on the 4pm close of the index, so a quote's T runs to 16:00 new york
SETTLEMENT = time(16, 0)
# spx options trade until 16:15, the frozen quotes after the close are the 16:15 ones
OPTIONS_CLOSE = time(16, 15)
OPTIONS_OPEN = time(9, 30)

FLOAT_COLUMNS = (
    "strike",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "last",
    "volume",
    "open_interest",
    "last_trade_time",
)
COLUMNS = ("expiry", "right", *FLOAT_COLUMNS)


@dataclass(frozen=True)
class Chain:
    """1 row per quote. expiry 'YYYYMMDD', right 'C' or 'P', missing numbers are nan.
    last_trade_time is epoch seconds. meta holds where and when (never any account data)"""

    expiry: StrArray
    right: StrArray
    strike: FloatArray
    bid: FloatArray
    ask: FloatArray
    bid_size: FloatArray
    ask_size: FloatArray
    last: FloatArray
    volume: FloatArray
    open_interest: FloatArray
    last_trade_time: FloatArray
    meta: Mapping[str, MetaValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.expiry)
        for name in COLUMNS:
            if len(getattr(self, name)) != n:
                raise ValueError(f"column {name} has {len(getattr(self, name))} rows, expected {n}")
        if not np.all(np.isin(self.right, ["C", "P"])):
            raise ValueError("right must be 'C' or 'P'")

    def __len__(self) -> int:
        return len(self.expiry)

    @property
    def as_of(self) -> datetime:
        """the moment the quotes belong to (utc)"""
        return datetime.fromisoformat(str(self.meta["as_of"]))

    @property
    def T(self) -> FloatArray:  # noqa: N802, T as in every pricing formula
        """years from as_of to each expiry's settlement, act/365"""
        return np.array([year_fraction(self.as_of, str(e)) for e in self.expiry])

    @property
    def mid(self) -> FloatArray:
        return 0.5 * (self.bid + self.ask)

    def expiries(self) -> list[str]:
        return sorted(set(self.expiry.tolist()))

    def select(self, mask: npt.NDArray[np.bool_]) -> "Chain":
        """the rows where mask is true, same meta"""
        cols = {name: getattr(self, name)[mask] for name in COLUMNS}
        return replace(self, **cols)


def year_fraction(as_of: datetime, expiry: str) -> float:
    """act/365 from as_of to 16:00 new york on the expiry date"""
    day = datetime.strptime(expiry, "%Y%m%d").date()
    settle = datetime.combine(day, SETTLEMENT, tzinfo=NEW_YORK)
    return (settle - as_of).total_seconds() / (365.0 * 86400.0)


def last_session_close(now: datetime) -> datetime | None:
    """the options close (16:15 new york) the quotes belong to when the market is shut, None
    while it's open. weekends handled, exchange holidays not (a holiday reads as a normal day)"""
    local = now.astimezone(NEW_YORK)
    weekday = local.weekday() < 5
    if weekday and OPTIONS_OPEN <= local.time() < OPTIONS_CLOSE:
        return None
    day = local.date()
    if not weekday or local.time() < OPTIONS_OPEN:
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return datetime.combine(day, OPTIONS_CLOSE, tzinfo=NEW_YORK).astimezone(UTC)


def from_rows(rows: list[dict[str, str | float]], meta: Mapping[str, MetaValue]) -> Chain:
    """a chain from dicts keyed by COLUMNS, missing or empty numbers -> nan"""

    def num(v: str | float | None) -> float:
        if v is None or v == "":
            return math.nan
        return float(v)

    cols: dict[str, npt.NDArray[np.str_] | FloatArray] = {
        "expiry": np.array([str(r["expiry"]) for r in rows], dtype=np.str_),
        "right": np.array([str(r["right"]) for r in rows], dtype=np.str_),
    }
    for name in FLOAT_COLUMNS:
        cols[name] = np.array([num(r.get(name)) for r in rows], dtype=np.float64)
    return Chain(**cols, meta=dict(meta))  # type: ignore[arg-type]


def save_snapshot(chain: Chain, stem: Path) -> Path:
    """writes stem.csv (the quotes) and stem.json (meta), returns the csv path. floats are
    written w/ repr, so reading back gives the same bits"""
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i in range(len(chain)):
            row: list[str] = [str(chain.expiry[i]), str(chain.right[i])]
            for name in FLOAT_COLUMNS:
                v = float(getattr(chain, name)[i])
                row.append("" if math.isnan(v) else repr(v))
            w.writerow(row)
    stem.with_suffix(".json").write_text(json.dumps(dict(chain.meta), indent=2) + "\n")
    return csv_path


def load_snapshot(path: Path | str) -> Chain:
    """reads a snapshot from its .csv (or the stem), meta from the .json next to it"""
    p = Path(path)
    csv_path = p if p.suffix == ".csv" else p.with_suffix(".csv")
    meta = json.loads(csv_path.with_suffix(".json").read_text())
    with csv_path.open(newline="") as f:
        rows: list[dict[str, str | float]] = list(csv.DictReader(f))
    return from_rows(rows, meta)
