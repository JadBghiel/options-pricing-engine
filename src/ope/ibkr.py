"""the ibkr side of fetching a chain: a read only connection through ib_async, quotes streamed in
batches and written to disk before anything reads them

same approach as the sivers client (cached, paced, read only), adapted to a chain:
- every batch lands under data/ before anything reads it, and a batch already on disk isnt asked
  for again: a rerun after a disconnect asks for the missing batches, while the quotes are the
  same ones (market shut)
- no historical requests here -> IB's 60 per 10 min limit doesnt apply, the limits that do: 45
  messages a second (ib_async throttles itself) and the market data lines open at once (100 on
  most accounts) -> batches of 50 quotes, cancelled before the next batch
- the socket handshake on connect and nothing else, see connect_ibkr, no order, account or
  position call anywhere in this module, and the gateway's read only api setting blocks orders
"""

import logging
import math
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ope.chain import (
    DATA_CHAINS,
    TARGET_DAYS,
    MetaValue,
    _num,
    _read_part,
    _write_part,
    from_rows,
    pick_expiries,
    pick_strikes,
    save_snapshot,
    snapshot_as_of,
    snapshot_stem,
    year_fraction,
)

if TYPE_CHECKING:
    from ib_async import IB, Contract, Ticker

log = logging.getLogger(__name__)

# defaults for ib gateway live, each one can be overridden by an env var (OPE_IB_HOST,
# OPE_IB_PORT, OPE_IB_CLIENT_ID) or an argument: 4002 gateway paper, 7496/7497 tws live/paper
IB_HOST, IB_PORT, IB_CLIENT_ID = "127.0.0.1", 4001, 23
BATCH = 50
MARKET_DATA_TYPES = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed frozen"}
# "market data farm connection is OK" and friends: info messages
IB_NOISE = {2104, 2106, 2107, 2108, 2119, 2158}

Log = Callable[[str], None]


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
    """connection that does the api handshake and nothing else, no request about the account"""
    from ib_async import IB

    host, port, client_id = ib_settings(host, port, client_id)
    # IB's per contract errors are collected by IBErrors and summed up in the snapshot meta
    logging.getLogger("ib_async").setLevel(logging.CRITICAL)
    ib = IB()
    # LEARNING NOTE: not ib.connect(), in ib_async 2.1 it asks for the positions on every connect,
    # even w/ fetchFields=StartupFetchNONE (the request isnt behind the flag), and open orders
    # and account values by default, client.connect is the bare handshake, and qualifyContracts
    # and reqMktData work on it the same, the gateway still pushes the account list on its own
    # during the handshake, it stays in ib.wrapper.accounts and nothing here reads it
    ib.wrapper.clientId = client_id
    ib.client.connect(host, port, client_id, timeout)
    return ib


class IBErrors:
    """keeps IB's error messages: an empty quote looks the same whether the contract doesnt
    exist, the data isnt subscribed or the market is shut, the error text tells them apart"""

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
    seconds, the open interest comes after the quote: waiting for the quote alone left it
    missing on 18 of 38 quotes 9 months out in the 1st real pull"""
    waited = 0.0
    while waited < max_wait:
        ib.sleep(0.25)
        waited += 0.25
        if all(_has_quote(t) and (not open_interest or _has_open_interest(t)) for t in tickers):
            return


def spot_hint_ibkr(
    ib: "IB", symbol: str = "SPX", max_wait: float = 6.0
) -> tuple["Contract", float]:
    """the index contract and a rough level, used to choose the strikes and nothing else, the
    forward the calibration uses comes from put call parity"""
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
    frozen w/ a subscription, delayed or delayed frozen w/o, the type received is recorded
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
        if part.exists():  # already on disk from an interrupted run, not asked again
            continue
        group = wanted[start : start + batch]
        contracts = [
            Option(symbol, e, k, r, "SMART", tradingClass=trading_class, currency="USD")
            for e, k, r in group
        ]
        qualified = ib.qualifyContracts(*contracts)
        live = [(g, c) for g, c in zip(group, qualified, strict=True) if c is not None]
        n_missing += len(group) - len(live)
        # streaming requests: a snapshot request (snapshot=True) or a regulatory snapshot
        # (regulatorySnapshot=True) is billed per request ($0.01 each for regulatory ones),
        # streaming delayed data is free, tests fail if either flag is set
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
