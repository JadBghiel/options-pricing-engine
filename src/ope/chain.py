"""option chain snapshots: 1 chain at 1 moment, saved on disk as csv + json, plus what goes in a
snapshot (which expiries, which strikes, at what time)

a snapshot is plain numpy columns, 1 row per quote (expiry, strike, call or put), the fetchers in
ope.ibkr and ope.market write them, everything after (cleaning, the iv surface, svi, heston) reads
them, no pandas: the stdlib csv module and numpy are enough for a few thousand rows
"""

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
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
# spxw settles on the 4pm close of the index -> a quote's T goes to 16:00 new york
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
    """1 row per quote: expiry 'YYYYMMDD', right 'C' or 'P', missing numbers are nan,
    last_trade_time is epoch seconds, meta holds where and when (no account data)"""

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
    while it's open, weekends handled, exchange holidays not (a holiday reads as a normal day)"""
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
    """writes stem.csv (the quotes) and stem.json (meta), returns the csv path, floats are
    written w/ repr -> reading back gives the same bits"""
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


# batches on disk, missing numbers


def _num(x: float | None) -> float:
    """ib_async uses nan for empty, but -1 and None show up too"""
    if x is None:
        return math.nan
    v = float(x)
    return math.nan if v < 0 or not math.isfinite(v) else v


def _write_part(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            )


def _read_part(path: Path) -> list[dict[str, str | float]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# what goes in a snapshot

DATA_CHAINS = Path("data/chains")
# about 1 week to 1 year, the nearest listed expiry to each
TARGET_DAYS = (7, 14, 30, 60, 91, 182, 273, 365)


def pick_expiries(
    available: Iterable[str], as_of: datetime, target_days: Sequence[int] = TARGET_DAYS
) -> list[str]:
    """the listed expiry nearest to each target (in days), each one at most once, sorted,
    expiries under 2 days are skipped, their smile is all noise"""
    listed = sorted({e for e in available if year_fraction(as_of, e) * 365 >= 2})
    if not listed:
        return []
    days = np.array([year_fraction(as_of, e) * 365 for e in listed])
    return sorted({listed[int(np.argmin(np.abs(days - t)))] for t in target_days})


def strike_band(T: float) -> tuple[float, float]:
    """log moneyness range to request: 4 sd below the spot, 2 above, at a 20% reference vol,
    the downside goes further bc index puts are where the liquidity and the skew are"""
    sd = 0.2 * math.sqrt(T)
    return -min(max(4 * sd, 0.08), 0.5), min(max(2 * sd, 0.04), 0.25)


def pick_strikes(available: Iterable[float], spot: float, T: float, n: int = 30) -> list[float]:
    """up to n listed strikes spread evenly in log moneyness over strike_band(T)

    the strike list IB gives is the union over every expiry, and long dated spxw lists
    round strikes -> multiples of 25 past 1 month, 50 past 6 months, when there are enough"""
    lo, hi = strike_band(T)
    ks = np.array(sorted(set(available)), dtype=np.float64)
    step = 50.0 if T > 0.5 else 25.0 if T > 1 / 12 else 0.0
    if step:
        rounded = ks[np.isclose(np.mod(ks, step), 0.0)]
        if len(rounded) >= n:
            ks = rounded
    k = np.log(ks / spot)
    inside = ks[(k >= lo) & (k <= hi)]
    if len(inside) <= n:
        return [float(x) for x in inside]
    k_in = np.log(inside / spot)
    picks = {int(np.argmin(np.abs(k_in - t))) for t in np.linspace(lo, hi, n)}
    return sorted(float(inside[i]) for i in picks)


def snapshot_as_of(now: datetime, market_data_type: int | None = None) -> datetime:
    """the moment the quotes belong to: the last close when the market is shut, now (minus
    15 min for delayed data) while it's open"""
    close = last_session_close(now)
    if close is not None:
        return close
    lag = timedelta(minutes=15) if market_data_type in (3, 4) else timedelta(0)
    return (now - lag).replace(second=0, microsecond=0)


def snapshot_stem(symbol: str, source: str, as_of: datetime, out_dir: Path = DATA_CHAINS) -> Path:
    return out_dir / f"{symbol.lower()}_{source}_{as_of:%Y%m%d_%H%M}"
