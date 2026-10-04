"""the chain fetchers, on fake gateways and a fake yahoo: no network in tests or ci

the fake IB only has the methods a market data pull needs. any other call (positions, account
values, orders...) raises, so these tests also prove the fetcher never asks for them
"""

import math
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import ope.market as market
from ope.chain import load_snapshot, year_fraction
from ope.market import (
    IBErrors,
    fetch_ibkr,
    fetch_yfinance,
    ib_settings,
    pick_expiries,
    pick_strikes,
    pull_chain,
    snapshot_as_of,
    strike_band,
)

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
    """a gateway w/ spx at 6500 and spxw listed every weekday, strikes every 5 points, but only
    multiples of 10 exist (the rest is "unknown", like IB's error 200)"""

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


def test_fetch_ibkr_pulls_market_data_only(tmp_path: Path) -> None:
    ib = FakeIB()
    path = fetch_ibkr(ib, out_dir=tmp_path, now=SATURDAY, say=lambda s: None)  # type: ignore[arg-type]
    chain = load_snapshot(path)
    assert path.name == "spx_ibkr_20261023_2015.csv"  # friday 16:15 new york
    assert ib.calls[0] == "type 4"
    assert ib.max_lines <= market.BATCH
    assert ib.open_lines == 0  # every line cancelled
    assert len(chain.expiries()) == 8
    assert chain.meta["market_data_type"] == ["frozen"]
    assert chain.meta["contracts_unknown_to_ib"] == chain.meta["contracts_requested"] - len(chain)
    assert not any("account" in k for k in chain.meta)


def test_every_quote_is_a_free_streaming_request(tmp_path: Path) -> None:
    # snapshot and regulatory snapshot requests are billed per request, streaming is not
    ib = FakeIB()
    fetch_ibkr(ib, out_dir=tmp_path, now=SATURDAY, say=lambda s: None)  # type: ignore[arg-type]
    assert ib.mkt_kwargs
    for kw in ib.mkt_kwargs:
        assert kw == {"snapshot": False, "regulatorySnapshot": False}


def test_a_rerun_only_asks_for_the_missing_batches(tmp_path: Path) -> None:
    broken = FakeIB(fail_after=4)  # the index lookup, then 3 batches, then the gateway drops
    with pytest.raises(ConnectionError):
        fetch_ibkr(broken, out_dir=tmp_path, now=SATURDAY, say=lambda s: None)  # type: ignore[arg-type]
    ib = FakeIB()
    path = fetch_ibkr(ib, out_dir=tmp_path, now=SATURDAY, say=lambda s: None)  # type: ignore[arg-type]
    full = FakeIB()
    fetch_ibkr(full, out_dir=tmp_path / "fresh", now=SATURDAY, say=lambda s: None)  # type: ignore[arg-type]
    assert ib.calls.count("qualify") == full.calls.count("qualify") - 3
    assert len(load_snapshot(path)) == len(load_snapshot(next((tmp_path / "fresh").glob("*.csv"))))


def test_delayed_data_while_open_is_15_minutes_old(tmp_path: Path) -> None:
    ib = FakeIB(data_type=3)
    chain = load_snapshot(fetch_ibkr(ib, out_dir=tmp_path, now=TUESDAY_OPEN, say=lambda s: None))  # type: ignore[arg-type]
    assert chain.as_of == datetime(2026, 10, 27, 18, 45, tzinfo=UTC)
    assert chain.meta["market_data_type"] == ["delayed"]


def test_no_level_needs_a_spot_hint(tmp_path: Path) -> None:
    ib = FakeIB(level=math.nan)
    with pytest.raises(RuntimeError, match="spot_hint"):
        fetch_ibkr(ib, out_dir=tmp_path, now=SATURDAY, say=lambda s: None)  # type: ignore[arg-type]
    chain = load_snapshot(
        fetch_ibkr(FakeIB(level=math.nan), out_dir=tmp_path, now=SATURDAY, spot_hint=6500.0,
                   say=lambda s: None)  # type: ignore[arg-type]
    )  # fmt: skip
    assert chain.meta["spot_hint"] == 6500.0


def test_missing_index_or_chain_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ib = FakeIB()
    monkeypatch.setattr(ib, "qualifyContracts", lambda *cs: [], raising=False)
    with pytest.raises(RuntimeError, match="doesnt know the index"):
        fetch_ibkr(ib, out_dir=tmp_path, now=SATURDAY, say=lambda s: None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no SPXQ chain"):
        fetch_ibkr(
            FakeIB(), trading_class="SPXQ", out_dir=tmp_path, now=SATURDAY, say=lambda s: None
        )  # type: ignore[arg-type]


def test_quotes_that_never_come_stop_at_max_wait() -> None:
    ib = FakeIB()
    silent = [ticker(1.0, bid=math.nan, ask=math.nan)]
    market._wait_for_quotes(ib, silent, max_wait=2.0)  # type: ignore[arg-type]
    assert ib.quote_sleeps == 8  # 2 s in steps of 0.25


def test_ticker_row_open_interest_and_trade_time() -> None:
    t = ticker(1.0, callOpenInterest=math.nan, futuresOpenInterest=7.0,
               lastTimestamp=datetime(2026, 1, 2, tzinfo=UTC))  # fmt: skip
    row = market._ticker_row(t, "20260116", 100.0, "C")  # type: ignore[arg-type]
    assert row["open_interest"] == 7.0
    assert row["last_trade_time"] == datetime(2026, 1, 2, tzinfo=UTC).timestamp()
    assert market._num(None) != market._num(None)  # nan
    assert math.isnan(market._num(-1.0)) and market._num(3.0) == 3.0


def test_ib_errors_skip_the_noise_and_count() -> None:
    errors = IBErrors()
    errors(1, 2104, "market data farm connection is OK", None)
    errors(2, 200, "No security definition", None)
    errors(3, 200, "No security definition", None)
    assert errors.summary() == ["200: No security definition (x2)"]


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


def test_ib_settings_argument_then_env_then_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPE_IB_PORT", raising=False)
    monkeypatch.delenv("OPE_IB_HOST", raising=False)
    monkeypatch.delenv("OPE_IB_CLIENT_ID", raising=False)
    assert ib_settings() == ("127.0.0.1", 4001, 23)
    monkeypatch.setenv("OPE_IB_PORT", "7497")
    monkeypatch.setenv("OPE_IB_CLIENT_ID", "5")
    assert ib_settings() == ("127.0.0.1", 7497, 5)
    assert ib_settings(port=4002) == ("127.0.0.1", 4002, 5)


def test_connect_is_the_bare_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    import ib_async

    seen: dict[str, Any] = {}

    class FakeClient:
        def connect(self, host: str, port: int, client_id: int, timeout: float) -> None:
            seen.update(host=host, port=port, client_id=client_id)

    class StrictIB:
        def __init__(self) -> None:
            self.client = FakeClient()
            self.wrapper = SimpleNamespace(clientId=None)

        def connect(self, *a: Any, **kw: Any) -> None:
            raise AssertionError("IB.connect asks for the positions, never use it")

    monkeypatch.setattr(ib_async, "IB", StrictIB)
    ib = market.connect_ibkr(port=4002, client_id=9)
    assert seen == {"host": "127.0.0.1", "port": 4002, "client_id": 9}
    assert ib.wrapper.clientId == 9


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


def test_fetch_yfinance_keeps_spxw_only(tmp_path: Path) -> None:
    path = fetch_yfinance(out_dir=tmp_path, now=SATURDAY, yf_module=FakeYahoo(), say=lambda s: None)
    chain = load_snapshot(path)
    assert path.name == "spx_yfinance_20261023_2015.csv"
    assert 6500.0 not in chain.strike.tolist()  # the SPX (am settled) row is dropped
    assert chain.meta["market_data_type"] == ["delayed"]
    assert math.isnan(chain.volume[chain.strike == 6400.0][0])
    assert year_fraction(chain.as_of, chain.expiries()[0]) > 0


def test_pull_chain_reuses_a_closed_session_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fetch_ibkr(FakeIB(), out_dir=tmp_path, now=SATURDAY, say=lambda s: None)  # type: ignore[arg-type]

    def no_network(**kw: Any) -> None:
        raise AssertionError("a cached snapshot must not touch the gateway")

    monkeypatch.setattr(market, "connect_ibkr", no_network)
    assert len(pull_chain(out_dir=tmp_path, now=SATURDAY, say=lambda s: None)) == len(
        load_snapshot(path)
    )


def test_pull_chain_falls_back_to_yahoo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(**kw: Any) -> None:
        raise ConnectionRefusedError("nothing on 4001")

    monkeypatch.setattr(market, "connect_ibkr", refused)
    yahoo = FakeYahoo()
    chain = pull_chain(out_dir=tmp_path, now=SATURDAY, say=lambda s: None, yf_module=yahoo)
    assert chain.meta["source"] == "yfinance"
    with pytest.raises(ConnectionRefusedError):
        pull_chain("ibkr", out_dir=tmp_path / "x", now=SATURDAY, say=lambda s: None)
    with pytest.raises(ValueError, match="source"):
        pull_chain("bloomberg")


def test_pull_chain_from_the_gateway_disconnects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    done: list[bool] = []

    class Closable(FakeIB):
        def disconnect(self) -> None:
            done.append(True)

    ib = Closable()
    monkeypatch.setattr(market, "connect_ibkr", lambda **kw: ib)
    chain = pull_chain("ibkr", out_dir=tmp_path, now=TUESDAY_OPEN, say=lambda s: None)
    assert chain.meta["source"] == "ibkr" and done == [True]
