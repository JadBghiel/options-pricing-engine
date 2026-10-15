"""pulling a real option chain: ibkr read only (ope.ibkr), yahoo as fallback, both saved as a
snapshot in data/chains/ that every later step reads

ib_async and yfinance come w/ the `market` extra and are imported inside the funcs -> `import ope`
works w/o them, real quotes stay in the gitignored data/, the ibkr market data agreement forbids
passing them on
"""

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ope import ibkr
from ope.chain import (
    DATA_CHAINS,
    TARGET_DAYS,
    Chain,
    MetaValue,
    _num,
    from_rows,
    last_session_close,
    load_snapshot,
    pick_expiries,
    pick_strikes,
    save_snapshot,
    snapshot_as_of,
    snapshot_stem,
    year_fraction,
)
from ope.ibkr import Log


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
    """the same snapshot from yahoo (15 min delayed), when the gateway is down, yahoo mixes
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


def pull_chain(
    source: str = "auto",
    *,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    out_dir: Path = DATA_CHAINS,
    now: datetime | None = None,
    say: Log = print,
    **kwargs: Any,
) -> Chain:
    """a snapshot from disk if this session's is already there, else from IB, else yahoo,
    source: "ibkr", "yfinance" or "auto" (ibkr, yahoo if the gateway doesnt answer)"""
    if source not in ("auto", "ibkr", "yfinance"):
        raise ValueError(f"source must be 'auto', 'ibkr' or 'yfinance', got {source!r}")
    now = now or datetime.now(UTC)
    close = last_session_close(now)
    if close is not None:  # market shut: the session's snapshot stays the same, reuse it
        for src in ("ibkr", "yfinance") if source == "auto" else (source,):
            cached = snapshot_stem("SPX", src, close, out_dir).with_suffix(".csv")
            if cached.exists():
                say(f"cached snapshot {cached}")
                return load_snapshot(cached)
    if source in ("auto", "ibkr"):
        try:
            ib = ibkr.connect_ibkr(host=host, port=port, client_id=client_id)
        except (ConnectionError, OSError, TimeoutError) as exc:
            if source == "ibkr":
                raise
            port_used = ibkr.ib_settings(host, port, client_id)[1]
            say(f"no IB gateway on port {port_used} ({exc}), falling back to yahoo")
        else:
            try:
                return load_snapshot(
                    ibkr.fetch_ibkr(ib, out_dir=out_dir, now=now, say=say, **kwargs)
                )
            finally:
                ib.disconnect()
    return load_snapshot(fetch_yfinance(out_dir=out_dir, now=now, say=say, **kwargs))
