"""the connection settings and the read only lock (ope.ib_gateway), no gateway needed"""

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ope
from ope import ib_gateway
from ope.ib_gateway import BLOCKED_CLIENT, BLOCKED_IB, NoPortError, ib_settings, lock_read_only

ib_async = pytest.importorskip("ib_async")


@pytest.fixture
def no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPE_IB_HOST", "OPE_IB_PORT", "OPE_IB_CLIENT_ID"):
        monkeypatch.delenv(name, raising=False)


def test_ib_settings_argument_then_env_then_default(
    no_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert ib_settings(port=4002) == ("127.0.0.1", 4002, 23)
    monkeypatch.setenv("OPE_IB_PORT", "7497")
    monkeypatch.setenv("OPE_IB_CLIENT_ID", "5")
    assert ib_settings() == ("127.0.0.1", 7497, 5)
    assert ib_settings(port=4002) == ("127.0.0.1", 4002, 5)
    assert ib_settings(port=4002, client_id=0) == ("127.0.0.1", 4002, 0)  # 0 is a valid id


def test_no_port_no_connection(no_env: None) -> None:
    # no default port -> a clone never reaches a live account by accident
    with pytest.raises(NoPortError, match="OPE_IB_PORT"):
        ib_settings()
    with pytest.raises(NoPortError):
        ib_gateway.connect_ibkr()


def test_every_blocked_name_exists_in_ib_async() -> None:
    # a typo or a rename in ib_async would leave a method open w/o this
    for name in BLOCKED_IB:
        assert callable(getattr(ib_async.IB, name)), name
    for name in BLOCKED_CLIENT:
        assert callable(getattr(ib_async.client.Client, name)), name


def test_the_lock_refuses_orders_and_account_data() -> None:
    ib = lock_read_only(ib_async.IB())
    for name in BLOCKED_IB:
        with pytest.raises(PermissionError, match=name):
            getattr(ib, name)()
    for name in BLOCKED_CLIENT:
        with pytest.raises(PermissionError, match=name):
            getattr(ib.client, name)()
    assert ib.wrapper.accounts == [] and ib.client._accounts == []


def test_the_lock_refuses_billed_snapshots_and_lets_streaming_through() -> None:
    ib = lock_read_only(ib_async.IB())
    ib.client.isReady = lambda: True  # a request id w/o a gateway, the send still fails
    index = ib_async.Index("SPX", "CBOE", "USD", conId=1)  # a qualified contract has a conId
    for flags in ({"snapshot": True}, {"regulatorySnapshot": True}):
        with pytest.raises(PermissionError, match="billed"):
            ib.reqMktData(index, "", **flags)
    # streaming gets past the lock and reaches the socket, which isnt there
    with pytest.raises(ConnectionError, match="Not connected"):
        ib.reqMktData(index, "", snapshot=False, regulatorySnapshot=False)


def test_the_library_never_calls_a_blocked_method() -> None:
    # the lock covers a connection from connect_ibkr, this covers any other IB object
    names = "|".join(sorted(set(BLOCKED_IB) | set(BLOCKED_CLIENT)))
    call = re.compile(rf"\.({names})\b")
    for path in Path(ope.__file__).parent.rglob("*.py"):
        hits = call.findall(path.read_text())
        assert not hits, f"{path.name} calls {hits}"


def test_connect_is_the_bare_handshake_then_locked(
    no_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    class FakeClient:
        def __init__(self) -> None:
            self._accounts = ["DU0000000"]

        def connect(self, host: str, port: int, client_id: int, timeout: float) -> None:
            seen.update(host=host, port=port, client_id=client_id)

        def reqMktData(self, *a: Any) -> None:  # noqa: N802
            pass

    class StrictIB:
        def __init__(self) -> None:
            self.client = FakeClient()
            self.wrapper = SimpleNamespace(clientId=None, accounts=["DU0000000"])

        def connect(self, *a: Any, **kw: Any) -> None:
            raise AssertionError("IB.connect asks for the positions, never use it")

    monkeypatch.setattr(ib_async, "IB", StrictIB)
    ib = ib_gateway.connect_ibkr(port=4002, client_id=9)
    assert seen == {"host": "127.0.0.1", "port": 4002, "client_id": 9}
    assert ib.wrapper.clientId == 9
    assert ib.wrapper.accounts == [] and ib.client._accounts == []
    with pytest.raises(PermissionError):
        ib.reqPositions()
