"""pull a real option chain: ibkr read only through ib_async, yfinance as fallback

same approach as the sivers client (cached, paced, read only), adapted to a chain:
- every batch of quotes lands on disk under data/ before anything reads it, and a snapshot
  already on disk is never requested again. a rerun after a disconnect only asks for the
  missing batches, and only while the quotes are the same ones (market shut)
- no historical requests here, so IB's 60 per 10 min limit doesnt apply. the limits that do:
  45 messages a second (ib_async throttles itself) and the market data lines open at once
  (100 on most accounts) -> batches of 50 quotes, cancelled before the next batch
- only the socket handshake on connect, see connect_ibkr. no order, account or position call
  anywhere in this module, and the gateway's read only api setting blocks orders anyway

ib_async and yfinance come w/ the `market` extra and are imported inside the functions, so
`import ope` works w/o them. real quotes stay in the gitignored data/ (ibkr data agreement)
"""

import csv
import logging
import math
import os
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ope.chain import (
    COLUMNS,
    Chain,
    MetaValue,
    from_rows,
    last_session_close,
    load_snapshot,
    save_snapshot,
    year_fraction,
)

if TYPE_CHECKING:
    from ib_async import IB, Contract, Ticker

log = logging.getLogger(__name__)

DATA_CHAINS = Path("data/chains")
# about 1 week to 1 year, the nearest listed expiry to each
TARGET_DAYS = (7, 14, 30, 60, 91, 182, 273, 365)
# defaults for ib gateway live. each one can be overridden by an env var (OPE_IB_HOST,
# OPE_IB_PORT, OPE_IB_CLIENT_ID) or an argument: 4002 gateway paper, 7496/7497 tws live/paper
IB_HOST, IB_PORT, IB_CLIENT_ID = "127.0.0.1", 4001, 23
BATCH = 50
MARKET_DATA_TYPES = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed frozen"}
# "market data farm connection is OK" and friends, not errors
IB_NOISE = {2104, 2106, 2107, 2108, 2119, 2158}

Log = Callable[[str], None]


# what to ask for


def pick_expiries(
    available: Iterable[str], as_of: datetime, target_days: Sequence[int] = TARGET_DAYS
) -> list[str]:
    """the listed expiry nearest to each target (in days), each one at most once, sorted.
    expiries under 2 days are skipped, their smile is all noise"""
    listed = sorted({e for e in available if year_fraction(as_of, e) * 365 >= 2})
    if not listed:
        return []
    days = np.array([year_fraction(as_of, e) * 365 for e in listed])
    return sorted({listed[int(np.argmin(np.abs(days - t)))] for t in target_days})


def strike_band(T: float) -> tuple[float, float]:
    """log moneyness range to request: 4 sd below the spot, 2 above, at a 20% reference vol.
    the downside goes further bc index puts are where the liquidity and the skew are"""
    sd = 0.2 * math.sqrt(T)
    return -min(max(4 * sd, 0.08), 0.5), min(max(2 * sd, 0.04), 0.25)


def pick_strikes(available: Iterable[float], spot: float, T: float, n: int = 30) -> list[float]:
    """up to n listed strikes spread evenly in log moneyness over strike_band(T)

    the strike list IB gives is the union over every expiry, and long dated spxw only lists
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


# ibkr


def ib_settings(
    host: str | None = None, port: int | None = None, client_id: int | None = None
) -> tuple[str, int, int]:
    """argument, else env var, else default"""
    return (
        host or os.environ.get("OPE_IB_HOST", IB_HOST),
        port or int(os.environ.get("OPE_IB_PORT", IB_PORT)),
        client_id or int(os.environ.get("OPE_IB_CLIENT_ID", IB_CLIENT_ID)),
    )


def connect_ibkr(
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    timeout: float = 15,
) -> "IB":
    """connection that only does the api handshake, nothing about the account is requested"""
    from ib_async import IB

    host, port, client_id = ib_settings(host, port, client_id)
    # IB's per contract errors are collected by IBErrors and summed up in the snapshot meta
    logging.getLogger("ib_async").setLevel(logging.CRITICAL)
    ib = IB()
    # LEARNING NOTE: not ib.connect(). in ib_async 2.1 it asks for the positions on every connect,
    # even w/ fetchFields=StartupFetchNONE (the request isnt behind the flag), and open orders
    # and account values by default. client.connect is the bare handshake, and qualifyContracts
    # and reqMktData work on it the same. the gateway still pushes the account list on its own
    # during the handshake, it stays in ib.wrapper.accounts and nothing here reads it
    ib.wrapper.clientId = client_id
    ib.client.connect(host, port, client_id, timeout)
    return ib


class IBErrors:
    """keeps IB's error messages: an empty quote looks the same whether the contract doesnt
    exist, the data isnt subscribed or the market is shut, only the error text tells"""

    def __init__(self) -> None:
        self.entries: list[tuple[int, str]] = []

    def __call__(self, req_id: int, code: int, msg: str, contract: object) -> None:
        if code not in IB_NOISE:
            self.entries.append((code, msg))

    def summary(self) -> list[str]:
        counts: dict[tuple[int, str], int] = {}
        for e in self.entries:
            counts[e] = counts.get(e, 0) + 1
        return [f"{code}: {msg} (x{n})" for (code, msg), n in sorted(counts.items())]


def _num(x: float | None) -> float:
    """ib_async uses nan for empty, but -1 and None show up too"""
    if x is None:
        return math.nan
    v = float(x)
    return math.nan if v < 0 or not math.isfinite(v) else v


def _ticker_row(t: "Ticker", expiry: str, strike: float, right: str) -> dict[str, str | float]:
    oi = t.callOpenInterest if right == "C" else t.putOpenInterest
    if not math.isfinite(oi):
        oi = t.futuresOpenInterest
    return {
        "expiry": expiry,
        "right": right,
        "strike": strike,
        "bid": _num(t.bid),
        "ask": _num(t.ask),
        "bid_size": _num(t.bidSize),
        "ask_size": _num(t.askSize),
        "last": _num(t.last),
        "volume": _num(t.volume),
        "open_interest": _num(oi),
        "last_trade_time": t.lastTimestamp.timestamp() if t.lastTimestamp else math.nan,
    }


def _has_quote(t: "Ticker") -> bool:
    return math.isfinite(t.bid) or math.isfinite(t.ask)


def _has_open_interest(t: "Ticker") -> bool:
    return any(math.isfinite(x) for x in (t.callOpenInterest, t.putOpenInterest))


def _wait_for_quotes(
    ib: "IB", tickers: Sequence["Ticker"], max_wait: float, open_interest: bool = False
) -> None:
    """until every ticker has a bid or an ask (and an open interest if asked), or max_wait
    seconds. the open interest comes after the quote: waiting for the quote alone left it
    missing on 18 of 38 quotes 9 months out in the 1st real pull"""
    waited = 0.0
    while waited < max_wait:
        ib.sleep(0.25)
        waited += 0.25
        if all(_has_quote(t) and (not open_interest or _has_open_interest(t)) for t in tickers):
            return


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


def spot_hint_ibkr(
    ib: "IB", symbol: str = "SPX", max_wait: float = 6.0
) -> tuple["Contract", float]:
    """the index contract and a rough level, only used to choose the strikes. the forward the
    calibration uses comes from put call parity, never from this number"""
    from ib_async import Index

    index = Index(symbol, "CBOE", "USD")
    if not ib.qualifyContracts(index):
        raise RuntimeError(f"IB doesnt know the index {symbol} on CBOE")
    t = ib.reqMktData(index, "", snapshot=False, regulatorySnapshot=False)
    _wait_for_quotes(ib, [t], max_wait)
    level = next(
        (x for x in (t.last, t.close, t.marketPrice()) if math.isfinite(x) and x > 0), math.nan
    )
    ib.cancelMktData(index)
    return index, level


def fetch_ibkr(
    ib: "IB",
    *,
    symbol: str = "SPX",
    trading_class: str = "SPXW",
    target_days: Sequence[int] = TARGET_DAYS,
    strikes_per_expiry: int = 30,
    spot_hint: float | None = None,
    out_dir: Path = DATA_CHAINS,
    now: datetime | None = None,
    batch: int = BATCH,
    max_wait: float = 10.0,
    say: Log = print,
) -> Path:
    """1 chain snapshot from IB -> out_dir/<symbol>_ibkr_<as_of>.csv, returns its path

    market data type 4 (delayed frozen): IB gives the best the account is entitled to, live or
    frozen w/ a subscription, delayed or delayed frozen w/o. the type received is recorded
    """
    from ib_async import Option

    now = now or datetime.now(UTC)
    errors = IBErrors()
    ib.errorEvent += errors
    ib.reqMarketDataType(4)

    index, level = spot_hint_ibkr(ib, symbol)
    spot = spot_hint if spot_hint is not None else level
    if not (math.isfinite(spot) and spot > 0):
        raise RuntimeError(
            f"no level for {symbol} from IB ({'; '.join(errors.summary()) or 'no error text'}), "
            "pass spot_hint= to choose the strikes"
        )
    say(f"{symbol} level for the strike range: {spot:.2f}")

    params = ib.reqSecDefOptParams(symbol, "", "IND", index.conId)
    chain_def = next(
        (c for c in params if c.tradingClass == trading_class and c.exchange == "SMART"), None
    )
    if chain_def is None:
        raise RuntimeError(f"no {trading_class} chain on SMART for {symbol}")

    as_of = snapshot_as_of(now)
    expiries = pick_expiries(chain_def.expirations, as_of, target_days)
    wanted: list[tuple[str, float, str]] = []
    for e in expiries:
        for k in pick_strikes(chain_def.strikes, spot, year_fraction(as_of, e), strikes_per_expiry):
            wanted += [(e, k, "C"), (e, k, "P")]
    say(f"{len(expiries)} expiries ({', '.join(expiries)}), {len(wanted)} contracts")

    stem = snapshot_stem(symbol, "ibkr", as_of, out_dir)
    parts = stem.with_suffix(".parts")
    types_seen: set[int] = set()
    n_missing = 0
    for b, start in enumerate(range(0, len(wanted), batch)):
        part = parts / f"batch_{b:03d}.csv"
        if part.exists():  # already on disk from an interrupted run, never asked twice
            continue
        group = wanted[start : start + batch]
        contracts = [
            Option(symbol, e, k, r, "SMART", tradingClass=trading_class, currency="USD")
            for e, k, r in group
        ]
        qualified = ib.qualifyContracts(*contracts)
        live = [(g, c) for g, c in zip(group, qualified, strict=True) if c is not None]
        n_missing += len(group) - len(live)
        # streaming requests only. a snapshot request (snapshot=True) or a regulatory
        # snapshot (regulatorySnapshot=True) is billed per request ($0.01 each for regulatory
        # ones), streaming delayed data is free. tests fail if either flag is ever set
        tickers = [
            ib.reqMktData(c, "100,101", snapshot=False, regulatorySnapshot=False) for _, c in live
        ]
        _wait_for_quotes(ib, tickers, max_wait, open_interest=True)
        rows = [_ticker_row(t, e, k, r) for ((e, k, r), _), t in zip(live, tickers, strict=True)]
        types_seen |= {int(t.marketDataType) for t in tickers}
        for _, c in live:
            ib.cancelMktData(c)
        _write_part(part, rows)
        say(f"batch {b + 1}/{math.ceil(len(wanted) / batch)}: {len(rows)} quotes")

    rows_all = [r for p in sorted(parts.glob("batch_*.csv")) for r in _read_part(p)]
    if types_seen and not (types_seen & {1, 2}):  # delayed while open: quotes are 15 min old
        as_of = snapshot_as_of(now, min(types_seen))
    meta: dict[str, MetaValue] = {
        "source": "ibkr",
        "symbol": symbol,
        "trading_class": trading_class,
        "as_of": as_of.isoformat(),
        "pulled_at": now.isoformat(),
        "market_data_type": sorted(MARKET_DATA_TYPES[t] for t in types_seen),
        "spot_hint": spot,
        "contracts_requested": len(wanted),
        "contracts_unknown_to_ib": n_missing,
        "ib_errors": errors.summary(),
    }
    ib.errorEvent -= errors
    return save_snapshot(from_rows(rows_all, meta), stem)


# yfinance


def fetch_yfinance(
    *,
    symbol: str = "^SPX",
    trading_class: str = "SPXW",
    target_days: Sequence[int] = TARGET_DAYS,
    strikes_per_expiry: int = 30,
    out_dir: Path = DATA_CHAINS,
    now: datetime | None = None,
    yf_module: Any = None,
    say: Log = print,
) -> Path:
    """the same snapshot from yahoo (15 min delayed), when the gateway is down. yahoo mixes
    spx and spxw contracts in 1 list, the contract symbol's prefix tells them apart"""
    if yf_module is None:
        import yfinance

        yf_module = yfinance

    now = now or datetime.now(UTC)
    as_of = snapshot_as_of(now, 3)
    ticker = yf_module.Ticker(symbol)
    spot = float(ticker.fast_info["last_price"])
    available = [e.replace("-", "") for e in ticker.options]
    expiries = pick_expiries(available, as_of, target_days)
    say(f"{symbol} at {spot:.2f}, {len(expiries)} expiries ({', '.join(expiries)})")

    rows: list[dict[str, str | float]] = []
    for e in expiries:
        oc = ticker.option_chain(f"{e[:4]}-{e[4:6]}-{e[6:]}")
        strikes = set(
            pick_strikes(oc.calls["strike"], spot, year_fraction(as_of, e), strikes_per_expiry)
        )
        for right, frame in (("C", oc.calls), ("P", oc.puts)):
            for rec in frame.to_dict("records"):
                if float(rec["strike"]) not in strikes:
                    continue
                if not str(rec["contractSymbol"]).startswith(trading_class):
                    continue
                traded = rec.get("lastTradeDate")
                rows.append(
                    {
                        "expiry": e,
                        "right": right,
                        "strike": float(rec["strike"]),
                        "bid": _num(rec.get("bid")),
                        "ask": _num(rec.get("ask")),
                        "bid_size": math.nan,
                        "ask_size": math.nan,
                        "last": _num(rec.get("lastPrice")),
                        "volume": _num(rec.get("volume")),
                        "open_interest": _num(rec.get("openInterest")),
                        # a missing date is NaT in a dataframe, and NaT != NaT like nan
                        "last_trade_time": (
                            traded.timestamp()
                            if traded is not None and traded == traded
                            else math.nan
                        ),
                    }
                )
    meta: dict[str, MetaValue] = {
        "source": "yfinance",
        "symbol": "SPX" if symbol == "^SPX" else symbol,
        "trading_class": trading_class,
        "as_of": as_of.isoformat(),
        "pulled_at": now.isoformat(),
        "market_data_type": ["delayed"],
        "spot_hint": spot,
    }
    stem = snapshot_stem(str(meta["symbol"]), "yfinance", as_of, out_dir)
    return save_snapshot(from_rows(rows, meta), stem)


# entry point


def pull_chain(
    source: str = "auto",
    *,
    port: int | None = None,
    client_id: int | None = None,
    out_dir: Path = DATA_CHAINS,
    now: datetime | None = None,
    say: Log = print,
    **kwargs: Any,
) -> Chain:
    """a snapshot from disk if this session's is already there, else from IB, else yahoo.
    source: "ibkr", "yfinance" or "auto" (ibkr, yahoo if the gateway doesnt answer)"""
    if source not in ("auto", "ibkr", "yfinance"):
        raise ValueError(f"source must be 'auto', 'ibkr' or 'yfinance', got {source!r}")
    now = now or datetime.now(UTC)
    close = last_session_close(now)
    if close is not None:  # market shut: the session's snapshot never changes, reuse it
        for src in ("ibkr", "yfinance") if source == "auto" else (source,):
            cached = snapshot_stem("SPX", src, close, out_dir).with_suffix(".csv")
            if cached.exists():
                say(f"cached snapshot {cached}")
                return load_snapshot(cached)
    if source in ("auto", "ibkr"):
        try:
            ib = connect_ibkr(port=port, client_id=client_id)
        except (ConnectionError, OSError, TimeoutError) as exc:
            if source == "ibkr":
                raise
            say(f"no IB gateway on port {ib_settings(port=port)[1]} ({exc}), falling back to yahoo")
        else:
            try:
                return load_snapshot(fetch_ibkr(ib, out_dir=out_dir, now=now, say=say, **kwargs))
            finally:
                ib.disconnect()
    return load_snapshot(fetch_yfinance(out_dir=out_dir, now=now, say=say, **kwargs))
