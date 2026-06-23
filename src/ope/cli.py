"""cmd line: price an option and its greeks, solve implied vol, plot the greeks

ope price -S 42 -K 40 -T 0.5 -r 0.1 --sigma 0.2
ope iv --type put -S 42 -K 40 -T 0.5 -r 0.1 --price 0.81
ope plot-greeks -K 100 -T 0.5 -r 0.05 --sigma 0.2 --out greeks.png
"""

import argparse
import sys
from collections.abc import Callable, Sequence

from ope.black_scholes import greeks, price
from ope.implied_volatility import implied_vol_result


def _add_option_args(p: argparse.ArgumentParser, *, spot: bool = True) -> None:
    p.add_argument("--type", dest="option", choices=["call", "put"], default="call")
    if spot:
        p.add_argument("-S", "--spot", type=float, required=True)
    p.add_argument("-K", "--strike", type=float, required=True)
    p.add_argument("-T", "--maturity", type=float, required=True, help="in years")
    p.add_argument("-r", "--rate", type=float, required=True, help="continuous, 0.05 = 5%%")
    p.add_argument("-q", "--div-yield", type=float, default=0.0, help="continuous, default 0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ope", description="Options pricing engine.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("price", help="Black-Scholes price and Greeks")
    _add_option_args(p)
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")

    p = sub.add_parser("iv", help="implied volatility from a price")
    _add_option_args(p)
    p.add_argument("--price", type=float, required=True)

    p = sub.add_parser("plot-greeks", help="plot price and Greeks vs spot to a PNG")
    _add_option_args(p, spot=False)
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")
    p.add_argument("--out", default="greeks_vs_spot.png")
    return parser


def _cmd_price(a: argparse.Namespace) -> int:
    args = (a.spot, a.strike, a.maturity, a.rate, a.sigma, a.div_yield)
    value = price(a.option, *args)
    g = greeks(a.option, *args)
    print(
        f"{a.option}  S={a.spot:g} K={a.strike:g} T={a.maturity:g} r={a.rate:g} "
        f"q={a.div_yield:g} sigma={a.sigma:g}"
    )
    print(f"price  {value:12.6f}")
    print(f"delta  {g.delta:12.6f}")
    print(f"gamma  {g.gamma:12.6f}")
    print(f"vega   {g.vega:12.6f}  per 1.00 vol  ({g.vega / 100:.6f} per vol point)")
    print(f"theta  {g.theta:12.6f}  per year      ({g.theta / 365:.6f} per day)")
    print(f"rho    {g.rho:12.6f}  per 1.00 rate ({g.rho / 100:.6f} per 1%)")
    return 0


def _cmd_iv(a: argparse.Namespace) -> int:
    res = implied_vol_result(a.option, a.price, a.spot, a.strike, a.maturity, a.rate, a.div_yield)
    print(f"implied vol  {res.sigma:.6f}")
    print(
        f"iterations   {res.iterations} ({res.newton_steps} Newton, "
        f"{res.bisection_steps} bisection)"
    )
    return 0


def _cmd_plot_greeks(a: argparse.Namespace) -> int:
    try:
        from ope.plots import plot_greeks_vs_spot
    except ImportError:
        print(
            "error: plotting needs matplotlib, install it with "
            "pip install 'options-pricing-engine[plots]'",
            file=sys.stderr,
        )
        return 1
    fig = plot_greeks_vs_spot(a.strike, a.maturity, a.rate, a.sigma, a.div_yield)
    fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}")
    return 0


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "price": _cmd_price,
    "iv": _cmd_iv,
    "plot-greeks": _cmd_plot_greeks,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except ValueError as e:  # bad inputs are a msg for the user, not a traceback
        print(f"error: {e}", file=sys.stderr)
        return 2
