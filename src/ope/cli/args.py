"""arguments and output helpers the cli groups share"""

import argparse
import sys
from collections.abc import Callable
from typing import Any

from ope.monte_carlo import MCResult

Command = Callable[[argparse.Namespace], int]


def add_option_args(p: argparse.ArgumentParser, *, spot: bool = True, strike: bool = True) -> None:
    p.add_argument("--type", dest="option", choices=["call", "put"], default="call")
    if spot:
        p.add_argument("-S", "--spot", type=float, required=True)
    if strike:
        p.add_argument("-K", "--strike", type=float, required=True)
    p.add_argument("-T", "--maturity", type=float, required=True, help="in years")
    p.add_argument("-r", "--rate", type=float, required=True, help="continuous, 0.05 = 5%%")
    p.add_argument("-q", "--div-yield", type=float, default=0.0, help="continuous, default 0")


def add_sigma(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")


def add_cpp_arg(p: argparse.ArgumentParser, what: str = "it") -> None:
    p.add_argument("--cpp", action="store_true", help=f"compute {what} with the C++ core")


def add_mc_args(p: argparse.ArgumentParser, dates: int) -> None:
    p.add_argument("--dates", type=int, default=dates, help=f"monitoring dates, default {dates}")
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0, help="same seed, same result")


def plots_missing() -> int:
    print(
        "error: plotting needs matplotlib, install it with "
        "pip install 'options-pricing-engine[plots]'",
        file=sys.stderr,
    )
    return 1


def mc_line(label: str, res: MCResult, width: int = 24) -> str:
    lo, hi = res.ci95
    ci = f"95% CI [{lo:.4f}, {hi:.4f}]"
    return f"{label:<{width}}{res.estimate:.6f}  (std error {res.std_error:.6f}, {ci})"


def saved(fig: Any, out: str) -> int:
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    return 0
