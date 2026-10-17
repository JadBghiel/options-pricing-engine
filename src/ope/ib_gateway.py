"""the connection to an ib gateway or tws: settings, the bare handshake and the read only lock

the gateway's Read-Only API setting blocks orders but still answers positions, account values
and executions -> connect_ibkr locks the connection on our side too, every order or account
method raises, and the snapshot flags of reqMktData (billed per request) are refused, a guard
against mistakes in this repo: a raw socket message would still go through
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ib_async import IB

# ib's default ports: 4001 gateway live, 4002 gateway paper, 7496 tws live, 7497 tws paper,
# no default here -> nobody connects to a live account by accident
IB_HOST, IB_CLIENT_ID = "127.0.0.1", 23
PORTS_HELP = "4001 gateway live, 4002 gateway paper, 7496 TWS live, 7497 TWS paper"

# orders, positions, account values, p&l, executions: the IB methods first, then the
# Client ones underneath (IB's methods call them, other code could too)
BLOCKED_IB = (
    "placeOrder", "cancelOrder", "reqGlobalCancel", "exerciseOptions", "whatIfOrder",
    "whatIfOrderAsync", "managedAccounts", "accountValues", "accountSummary",
    "accountSummaryAsync", "portfolio", "positions", "pnl", "pnlSingle", "trades", "openTrades",
    "orders", "openOrders", "fills", "executions", "reqAccountSummary", "reqAccountSummaryAsync",
    "reqAccountUpdates", "reqAccountUpdatesAsync", "reqAccountUpdatesMulti",
    "reqAccountUpdatesMultiAsync", "reqPositions", "reqPositionsAsync", "reqPnL", "reqPnLSingle",
    "reqExecutions", "reqExecutionsAsync", "reqOpenOrders", "reqOpenOrdersAsync",
    "reqAllOpenOrders", "reqAllOpenOrdersAsync", "reqAutoOpenOrders", "reqCompletedOrders",
    "reqCompletedOrdersAsync",
)  # fmt: skip
BLOCKED_CLIENT = (
    "placeOrder", "cancelOrder", "reqGlobalCancel", "exerciseOptions", "getAccounts",
    "reqAccountSummary", "reqAccountUpdates", "reqAccountUpdatesMulti", "reqPositions",
    "reqPositionsMulti", "reqPnL", "reqPnLSingle", "reqExecutions", "reqOpenOrders",
    "reqAllOpenOrders", "reqAutoOpenOrders", "reqCompletedOrders", "reqFamilyCodes",
)  # fmt: skip


class NoPortError(ValueError):
    """no port given and no OPE_IB_PORT"""


def ib_settings(
    host: str | None = None, port: int | None = None, client_id: int | None = None
) -> tuple[str, int, int]:
    """argument, else env var (OPE_IB_HOST, OPE_IB_PORT, OPE_IB_CLIENT_ID), else default,
    the port has no default"""
    env_port = os.environ.get("OPE_IB_PORT", "")
    env_id = os.environ.get("OPE_IB_CLIENT_ID", "")
    if port is None and not env_port:
        raise NoPortError(f"no IB port, pass --port or set OPE_IB_PORT ({PORTS_HELP})")
    return (
        host or os.environ.get("OPE_IB_HOST") or IB_HOST,
        port if port is not None else int(env_port),
        client_id if client_id is not None else int(env_id) if env_id else IB_CLIENT_ID,
    )


def _blocked(name: str) -> Callable[..., Any]:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError(f"{name} is blocked: this connection is read only market data")

    return refuse


def _streaming_only(req: Callable[..., Any]) -> Callable[..., Any]:
    """Client.reqMktData w/ the 2 billed flags refused (IB.reqMktData calls it positionally)"""

    def call(
        req_id: int, contract: object, ticks: str, snapshot: bool, regulatory: bool, *rest: Any
    ) -> Any:
        if snapshot or regulatory:
            raise PermissionError("snapshot market data requests are billed, streaming only")
        return req(req_id, contract, ticks, snapshot, regulatory, *rest)

    return call


def lock_read_only(ib: "IB") -> "IB":
    """every order and account method of ib and its client raises, reqMktData streams only,
    and the account list the gateway sends during the handshake is dropped"""
    for name in BLOCKED_IB:
        setattr(ib, name, _blocked(name))
    for name in BLOCKED_CLIENT:
        setattr(ib.client, name, _blocked(name))
    ib.client.reqMktData = _streaming_only(ib.client.reqMktData)  # type: ignore[method-assign]
    ib.wrapper.accounts = []
    ib.client._accounts = []
    return ib


def connect_ibkr(
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    timeout: float = 15,
) -> "IB":
    """connection that does the api handshake and nothing else, locked read only"""
    from ib_async import IB

    host, port, client_id = ib_settings(host, port, client_id)
    ib = IB()
    # LEARNING NOTE: not ib.connect(), in ib_async 2.1 it asks for the positions on every connect,
    # even w/ fetchFields=StartupFetchNONE (the request isnt behind the flag), and open orders
    # and account values by default, client.connect is the bare handshake, and qualifyContracts
    # and reqMktData work on it the same, the gateway still pushes the account list on its own
    # during the handshake, lock_read_only drops it
    ib.wrapper.clientId = client_id
    ib.client.connect(host, port, client_id, timeout)
    return lock_read_only(ib)
