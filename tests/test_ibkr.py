"""the ibkr fetcher on a fake gateway (tests/fakes.py)"""

import math
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ope.ibkr as ibkr
from fakes import SATURDAY, TUESDAY_OPEN, FakeIB, ticker
from ope.chain import _num, load_snapshot
from ope.ibkr import IBErrors, fetch_ibkr, ib_settings


def test_fetch_ibkr_pulls_market_data_only(tmp_path: Path) -> None:
    ib = FakeIB()
    path = fetch_ibkr(ib, out_dir=tmp_path, now=SATURDAY, say=lambda s: None)  # type: ignore[arg-type]
    chain = load_snapshot(path)
    assert path.name == "spx_ibkr_20261023_2015.csv"  # friday 16:15 new york
    assert ib.calls[0] == "type 4"
    assert ib.max_lines <= ibkr.BATCH
    assert ib.open_lines == 0  # every line cancelled
    assert len(chain.expiries()) == 8
    assert chain.meta["market_data_type"] == ["frozen"]
    assert chain.meta["contracts_unknown_to_ib"] == chain.meta["contracts_requested"] - len(chain)
    assert not any("account" in k for k in chain.meta)


def test_every_quote_is_a_free_streaming_request(tmp_path: Path) -> None:
    # snapshot and regulatory snapshot requests are billed per request, streaming is free
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
    ibkr._wait_for_quotes(ib, silent, max_wait=2.0)  # type: ignore[arg-type]
    assert ib.quote_sleeps == 8  # 2 s in steps of 0.25


def test_ticker_row_open_interest_and_trade_time() -> None:
    t = ticker(1.0, callOpenInterest=math.nan, futuresOpenInterest=7.0,
               lastTimestamp=datetime(2026, 1, 2, tzinfo=UTC))  # fmt: skip
    row = ibkr._ticker_row(t, "20260116", 100.0, "C")  # type: ignore[arg-type]
    assert row["open_interest"] == 7.0
    assert row["last_trade_time"] == datetime(2026, 1, 2, tzinfo=UTC).timestamp()
    assert _num(None) != _num(None)  # nan
    assert math.isnan(_num(-1.0)) and _num(3.0) == 3.0


def test_ib_errors_skip_the_noise_and_count() -> None:
    errors = IBErrors()
    errors(1, 2104, "market data farm connection is OK", None)
    errors(2, 200, "No security definition", None)
    errors(3, 200, "No security definition", None)
    assert errors.summary() == ["200: No security definition (x2)"]


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
    ib = ibkr.connect_ibkr(port=4002, client_id=9)
    assert seen == {"host": "127.0.0.1", "port": 4002, "client_id": 9}
    assert ib.wrapper.clientId == 9
