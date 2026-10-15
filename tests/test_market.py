"""yahoo fetching and pull_chain (cache, gateway, fallback) on fakes (tests/fakes.py)"""

import math
from pathlib import Path
from typing import Any

import pytest

import ope.ibkr as ibkr
from fakes import SATURDAY, TUESDAY_OPEN, FakeIB, FakeYahoo
from ope.chain import load_snapshot, year_fraction
from ope.ibkr import fetch_ibkr
from ope.market import fetch_yfinance, pull_chain


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

    monkeypatch.setattr(ibkr, "connect_ibkr", no_network)
    assert len(pull_chain(out_dir=tmp_path, now=SATURDAY, say=lambda s: None)) == len(
        load_snapshot(path)
    )


def test_pull_chain_falls_back_to_yahoo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(**kw: Any) -> None:
        raise ConnectionRefusedError("nothing on 4001")

    monkeypatch.setattr(ibkr, "connect_ibkr", refused)
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
    monkeypatch.setattr(ibkr, "connect_ibkr", lambda **kw: ib)
    chain = pull_chain("ibkr", out_dir=tmp_path, now=TUESDAY_OPEN, say=lambda s: None)
    assert chain.meta["source"] == "ibkr" and done == [True]


def test_pull_chain_passes_host_port_and_client_id_to_the_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the cli passes host=..., which used to fall into **kwargs and crash fetch_ibkr
    seen: dict[str, Any] = {}

    def fake_connect(**kw: Any) -> FakeIB:
        seen.update(kw)

        class Closable(FakeIB):
            def disconnect(self) -> None:
                pass

        return Closable()

    monkeypatch.setattr(ibkr, "connect_ibkr", fake_connect)
    chain = pull_chain(
        "ibkr", host="10.0.0.2", port=7497, client_id=5, out_dir=tmp_path, now=TUESDAY_OPEN,
        say=lambda s: None,
    )  # fmt: skip
    assert seen == {"host": "10.0.0.2", "port": 7497, "client_id": 5}
    assert chain.meta["source"] == "ibkr"
