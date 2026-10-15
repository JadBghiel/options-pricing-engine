"""fake ib gateway and fake yahoo for the fetcher tests: no network in tests or ci

the fake IB has the methods a market data pull needs and nothing else, any other call
(positions, account values, orders...) raises -> the tests also prove the fetcher doesnt ask
for them
"""

import math
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd

SATURDAY = datetime(2026, 10, 24, 14, 44, tzinfo=UTC)  # quotes = friday's 16:15 close
TUESDAY_OPEN = datetime(2026, 10, 27, 19, 0, tzinfo=UTC)  # 15:00 in new york


class Event:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def __iadd__(self, f: Any) -> "Event":
        self.handlers.append(f)
        return self

    def __isub__(self, f: Any) -> "Event":
        self.handlers.remove(f)
        return self


def ticker(level: float, data_type: int = 2, **kw: Any) -> SimpleNamespace:
    fields = {
        "bid": level - 0.5,
        "ask": level + 0.5,
        "bidSize": 10.0,
        "askSize": 12.0,
        "last": level,
        "volume": 5.0,
        "callOpenInterest": 100.0,
        "putOpenInterest": 200.0,
        "futuresOpenInterest": math.nan,
        "lastTimestamp": None,
        "close": level,
        "marketDataType": data_type,
    }
    fields.update(kw)
    t = SimpleNamespace(**fields)
    t.marketPrice = lambda: t.last
    return t


class FakeIB:
    """a gateway w/ spx at 6500 and spxw listed every weekday, strikes every 5 points, and
    multiples of 10 are the ones that exist (the rest is "unknown", like IB's error 200)"""

    ALLOWED = frozenset(
        {"errorEvent", "reqMarketDataType", "qualifyContracts", "reqMktData", "cancelMktData",
         "sleep", "reqSecDefOptParams", "calls", "open_lines", "max_lines", "mkt_kwargs",
         "level", "data_type", "fail_after", "quote_sleeps"}
    )  # fmt: skip

    def __init__(self, level: float = 6500.0, data_type: int = 2, fail_after: int = 0) -> None:
        self.errorEvent = Event()
        self.calls: list[str] = []
        self.open_lines = 0
        self.max_lines = 0
        self.mkt_kwargs: list[dict[str, Any]] = []
        self.level = level
        self.data_type = data_type
        self.fail_after = fail_after
        self.quote_sleeps = 0

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the fetcher called {name}, which a market data pull never needs")

    def reqMarketDataType(self, t: int) -> None:  # noqa: N802, ib_async's names
        self.calls.append(f"type {t}")

    def qualifyContracts(self, *cs: Any) -> list[Any]:  # noqa: N802
        self.calls.append("qualify")
        if self.fail_after and self.calls.count("qualify") > self.fail_after:
            raise ConnectionError("gateway went away")
        out = []
        for c in cs:
            c.conId = 1
            ok = c.secType == "IND" or c.strike % 10 == 0
            out.append(c if ok else None)
        return out

    def reqMktData(self, c: Any, ticks: str, **kw: Any) -> SimpleNamespace:  # noqa: N802
        self.mkt_kwargs.append(kw)
        self.open_lines += 1
        self.max_lines = max(self.max_lines, self.open_lines)
        if c.secType == "IND":
            return ticker(self.level, self.data_type)
        return ticker(1.0 + c.strike / 1000, self.data_type)

    def cancelMktData(self, c: Any) -> None:  # noqa: N802
        self.open_lines -= 1

    def sleep(self, s: float) -> None:
        self.quote_sleeps += 1

    def reqSecDefOptParams(self, *a: Any) -> list[SimpleNamespace]:  # noqa: N802
        self.calls.append("secdef")
        days = pd.bdate_range("2026-10-26", periods=300)
        return [
            SimpleNamespace(exchange="CBOE", tradingClass="SPX", expirations=[], strikes=[]),
            SimpleNamespace(
                exchange="SMART",
                tradingClass="SPXW",
                expirations=[d.strftime("%Y%m%d") for d in days],
                strikes=[float(k) for k in range(2000, 12000, 5)],
            ),
        ]


class FakeYahoo:
    """yfinance's surface: Ticker(sym).options, .fast_info, .option_chain(date)"""

    def __init__(self) -> None:
        self.Ticker = lambda sym: self
        self.options = ("2026-10-30", "2026-11-23", "2027-01-15")
        self.fast_info = {"last_price": 6500.0}

    def option_chain(self, date: str) -> SimpleNamespace:
        strikes = [6000.0, 6400.0, 6500.0, 6600.0]
        frame = pd.DataFrame(
            {
                "contractSymbol": ["SPXW2610", "SPXW2610", "SPX2610", "SPXW2610"],
                "strike": strikes,
                "bid": [500.0, 100.0, 50.0, 10.0],
                "ask": [501.0, 101.0, 51.0, 11.0],
                "lastPrice": [500.5, 100.5, 50.5, 10.5],
                "volume": [1.0, None, 3.0, 4.0],
                "openInterest": [10, 20, 30, 40],
                "lastTradeDate": [pd.Timestamp("2026-10-23 19:00", tz="UTC"), None, None, None],
            }
        )
        return SimpleNamespace(calls=frame, puts=frame)
