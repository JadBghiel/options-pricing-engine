"""cmd line: price an option and its greeks, solve implied vol, plot the greeks

ope price -S 42 -K 40 -T 0.5 -r 0.1 --sigma 0.2
ope iv --type put -S 42 -K 40 -T 0.5 -r 0.1 --price 0.81
ope plot-greeks -K 100 -T 0.5 -r 0.05 --sigma 0.2 --out greeks.png
ope mc -S 42 -K 40 -T 0.5 -r 0.1 --sigma 0.2 --paths 100000 --antithetic --control
ope plot-mc -S 42 -K 40 -T 0.5 -r 0.1 --sigma 0.2 --out convergence.png
ope american --type put -S 50 -K 50 -T 0.4167 -r 0.1 --sigma 0.4
ope plot-boundary -S 50 -K 50 -T 0.4167 -r 0.1 --sigma 0.4 --out boundary.png
ope mc -S 42 -K 40 -T 0.5 -r 0.1 --sigma 0.2 --paths 10000000 --cpp --threads 8
ope asian -S 100 -K 100 -T 1 -r 0.05 --sigma 0.2 --dates 12
ope barrier -S 100 -K 100 -T 1 -r 0.05 --sigma 0.2 --barrier 120 --kind up-and-out
ope lookback -S 100 -T 1 -r 0.05 --sigma 0.2
ope heston -S 100 -K 100 -T 1 -r 0.05 --v0 0.04 --kappa 1.5 --theta 0.04 --xi 0.6 --rho -0.7
ope plot-smile -S 100 -T 1 -r 0.05 --out smile.png
"""

import argparse
import sys
from collections.abc import Callable, Sequence

from ope import cpp
from ope.black_scholes import greeks, price
from ope.exotics import (
    asian_geometric_price,
    asian_mc,
    barrier_mc,
    barrier_price,
    lookback_mc,
    lookback_price,
)
from ope.heston import HestonParams, heston_implied_vol, heston_mc, heston_price
from ope.implied_volatility import implied_vol_result
from ope.monte_carlo import MCResult, mc_price
from ope.pde import pde_price
from ope.trees import crr_price


def _add_option_args(p: argparse.ArgumentParser, *, spot: bool = True, strike: bool = True) -> None:
    p.add_argument("--type", dest="option", choices=["call", "put"], default="call")
    if spot:
        p.add_argument("-S", "--spot", type=float, required=True)
    if strike:
        p.add_argument("-K", "--strike", type=float, required=True)
    p.add_argument("-T", "--maturity", type=float, required=True, help="in years")
    p.add_argument("-r", "--rate", type=float, required=True, help="continuous, 0.05 = 5%%")
    p.add_argument("-q", "--div-yield", type=float, default=0.0, help="continuous, default 0")


def _add_cpp_arg(p: argparse.ArgumentParser, what: str = "it") -> None:
    p.add_argument("--cpp", action="store_true", help=f"compute {what} with the C++ core")


def _add_mc_args(p: argparse.ArgumentParser, dates: int) -> None:
    p.add_argument("--dates", type=int, default=dates, help=f"monitoring dates, default {dates}")
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0, help="same seed, same result")


def _add_heston_args(p: argparse.ArgumentParser) -> None:
    # defaults: the readme example, feller violated so the variance often hits 0
    p.add_argument("--v0", type=float, default=0.04, help="variance today, 0.04 = 20%% vol")
    p.add_argument("--kappa", type=float, default=1.5, help="speed back to theta")
    p.add_argument("--theta", type=float, default=0.04, help="long run variance")
    p.add_argument("--xi", type=float, default=0.6, help="vol of vol")
    p.add_argument("--rho", type=float, default=-0.7, help="stock vs variance correlation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ope", description="Options pricing engine.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("price", help="Black-Scholes price and Greeks")
    _add_option_args(p)
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")
    _add_cpp_arg(p)

    p = sub.add_parser("iv", help="implied volatility from a price")
    _add_option_args(p)
    p.add_argument("--price", type=float, required=True)

    p = sub.add_parser("plot-greeks", help="plot price and Greeks vs spot to a PNG")
    _add_option_args(p, spot=False)
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")
    p.add_argument("--out", default="greeks_vs_spot.png")

    p = sub.add_parser("mc", help="Monte Carlo price with a 95% confidence interval")
    _add_option_args(p)
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0, help="same seed, same result")
    p.add_argument("--antithetic", action="store_true", help="pair each draw z with -z")
    p.add_argument("--control", action="store_true", help="control variate on the stock")
    _add_cpp_arg(p)
    p.add_argument("--threads", type=int, default=None, help="with --cpp, default all cores")

    p = sub.add_parser("plot-mc", help="plot Monte Carlo convergence vs paths to a PNG")
    _add_option_args(p)
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="mc_convergence.png")

    p = sub.add_parser("american", help="American price by tree and PDE, vs European")
    _add_option_args(p)
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")
    p.add_argument("--steps", type=int, default=2000, help="tree time steps")
    p.add_argument("--grid", type=int, default=300, help="pde nodes and time steps")
    _add_cpp_arg(p, what="the tree")

    p = sub.add_parser("plot-boundary", help="plot the early exercise boundary to a PNG")
    _add_option_args(p)
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")
    p.add_argument("--out", default="exercise_boundary.png")

    p = sub.add_parser("asian", help="arithmetic Asian by Monte Carlo, geometric control variate")
    _add_option_args(p)
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")
    _add_mc_args(p, dates=12)

    p = sub.add_parser("barrier", help="knock-out barrier, discrete vs continuous monitoring")
    _add_option_args(p)
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")
    p.add_argument("-H", "--barrier", type=float, required=True)
    p.add_argument("--kind", choices=["down-and-out", "up-and-out"], required=True)
    _add_mc_args(p, dates=252)

    p = sub.add_parser("lookback", help="floating strike lookback, discrete vs continuous")
    _add_option_args(p, strike=False)
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")
    _add_mc_args(p, dates=252)

    p = sub.add_parser("heston", help="Heston price: semi-analytical, Monte Carlo, implied vol")
    _add_option_args(p)
    _add_heston_args(p)
    p.add_argument("--steps", type=int, default=100, help="Monte Carlo time steps")
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("plot-smile", help="plot the Heston implied vol smile to a PNG")
    _add_option_args(p, strike=False)
    _add_heston_args(p)
    p.add_argument("--out", default="heston_smile.png")

    p = sub.add_parser("plot-barrier", help="plot a barrier price vs monitoring dates to a PNG")
    _add_option_args(p)
    p.add_argument("--sigma", type=float, required=True, help="volatility, 0.2 = 20%%")
    p.add_argument("-H", "--barrier", type=float, required=True)
    p.add_argument("--kind", choices=["down-and-out", "up-and-out"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="barrier_monitoring.png")
    return parser


def _cmd_price(a: argparse.Namespace) -> int:
    args = (a.spot, a.strike, a.maturity, a.rate, a.sigma, a.div_yield)
    value = (cpp.price if a.cpp else price)(a.option, *args)
    g = (cpp.greeks if a.cpp else greeks)(a.option, *args)
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


def _cmd_mc(a: argparse.Namespace) -> int:
    args = (a.spot, a.strike, a.maturity, a.rate, a.sigma, a.div_yield)
    kw = {
        "n_paths": a.paths,
        "antithetic": a.antithetic,
        "control_variate": a.control,
        "rng": a.seed,
    }
    if a.cpp:  # c++ draws from its own generator, so a seed gives a different (valid) estimate
        res = cpp.mc_price(a.option, *args, **kw, threads=a.threads)
    else:
        res = mc_price(a.option, *args, **kw)
    exact = float(price(a.option, *args))
    lo, hi = res.ci95
    where = "inside" if lo <= exact <= hi else "outside"
    engine = "c++" if a.cpp else "numpy"
    print(
        f"monte carlo  {res.estimate:.6f}  (std error {res.std_error:.6f}, {res.n_paths} paths, "
        f"{engine})"
    )
    print(f"95% CI       [{lo:.6f}, {hi:.6f}]")
    print(f"closed form  {exact:.6f}  ({where} the interval)")
    return 0


def _plots_missing() -> int:
    print(
        "error: plotting needs matplotlib, install it with "
        "pip install 'options-pricing-engine[plots]'",
        file=sys.stderr,
    )
    return 1


def _cmd_plot_greeks(a: argparse.Namespace) -> int:
    try:
        from ope.plots import plot_greeks_vs_spot
    except ImportError:
        return _plots_missing()
    fig = plot_greeks_vs_spot(a.strike, a.maturity, a.rate, a.sigma, a.div_yield)
    fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}")
    return 0


def _cmd_plot_mc(a: argparse.Namespace) -> int:
    try:
        from ope.plots import plot_mc_convergence
    except ImportError:
        return _plots_missing()
    fig = plot_mc_convergence(
        a.option, a.spot, a.strike, a.maturity, a.rate, a.sigma, a.div_yield, seed=a.seed
    )
    fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}")
    return 0


def _cmd_american(a: argparse.Namespace) -> int:
    args = (a.spot, a.strike, a.maturity, a.rate, a.sigma, a.div_yield)
    tree = (cpp.crr_price if a.cpp else crr_price)(a.option, *args, steps=a.steps, american=True)
    pde = pde_price(a.option, *args, american=True, n_space=a.grid, n_time=a.grid)
    european = float(price(a.option, *args))
    print(
        f"american {a.option}  tree {tree:.6f} ({a.steps} steps)  pde {pde:.6f} ({a.grid}x{a.grid})"
    )
    print(f"european {a.option}  {european:.6f}  (closed form)")
    print(f"early exercise premium  {tree - european:.6f}")
    return 0


def _cmd_plot_boundary(a: argparse.Namespace) -> int:
    try:
        from ope.plots import plot_exercise_boundary
    except ImportError:
        return _plots_missing()
    fig = plot_exercise_boundary(
        a.option, a.spot, a.strike, a.maturity, a.rate, a.sigma, a.div_yield
    )
    fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}")
    return 0


def _mc_line(label: str, res: MCResult, width: int = 24) -> str:
    lo, hi = res.ci95
    ci = f"95% CI [{lo:.4f}, {hi:.4f}]"
    return f"{label:<{width}}{res.estimate:.6f}  (std error {res.std_error:.6f}, {ci})"


def _cmd_asian(a: argparse.Namespace) -> int:
    args = (a.spot, a.strike, a.maturity, a.rate, a.sigma, a.div_yield)
    kw = {"n_dates": a.dates, "n_paths": a.paths, "rng": a.seed}
    cv = asian_mc(a.option, *args, **kw, control_variate=True)
    plain = asian_mc(a.option, *args, **kw)
    geo = asian_geometric_price(a.option, *args, n_dates=a.dates)
    print(f"arithmetic asian {a.option}, average of {a.dates} dates, {a.paths} paths")
    print(_mc_line("control variate", cv, 19))
    print(_mc_line("plain", plain, 19))
    print(f"variance reduction {plain.std_error / cv.std_error:.1f}x smaller std error")
    print(f"geometric asian    {geo:.6f}  (closed form, the control)")
    print(f"vanilla            {float(price(a.option, *args)):.6f}")
    return 0


def _cmd_barrier(a: argparse.Namespace) -> int:
    args = (a.spot, a.strike, a.barrier, a.maturity, a.rate, a.sigma, a.div_yield)
    kw = {"kind": a.kind, "n_dates": a.dates, "n_paths": a.paths, "rng": a.seed}
    discrete = barrier_mc(a.option, *args, **kw)
    bridge = barrier_mc(a.option, *args, **kw, monitoring="continuous")
    cont = barrier_price(a.option, *args, kind=a.kind)
    bgk = barrier_price(a.option, *args, kind=a.kind, n_dates=a.dates)
    print(f"{a.kind} {a.option}, barrier {a.barrier:g}")
    print(_mc_line(f"checked on {a.dates} dates", discrete))
    print(f"BGK approximation       {bgk:.6f}  (continuous formula, shifted barrier)")
    print(_mc_line("checked continuously", bridge))
    print(f"continuous closed form  {cont:.6f}")
    vanilla = float(price(a.option, a.spot, a.strike, a.maturity, a.rate, a.sigma, a.div_yield))
    print(f"vanilla                 {vanilla:.6f}")
    return 0


def _cmd_lookback(a: argparse.Namespace) -> int:
    args = (a.spot, a.maturity, a.rate, a.sigma, a.div_yield)
    kw = {"n_dates": a.dates, "n_paths": a.paths, "rng": a.seed}
    discrete = lookback_mc(a.option, *args, **kw)
    cont = lookback_mc(a.option, *args, **kw, monitoring="continuous")
    print(f"floating strike lookback {a.option}")
    print(_mc_line(f"checked on {a.dates} dates", discrete))
    print(_mc_line("checked continuously", cont))
    print(f"continuous closed form  {lookback_price(a.option, *args):.6f}")
    return 0


def _heston_params(a: argparse.Namespace) -> HestonParams:
    return HestonParams(a.v0, a.kappa, a.theta, a.xi, a.rho)


def _cmd_heston(a: argparse.Namespace) -> int:
    p = _heston_params(a)
    args = (a.spot, a.strike, a.maturity, a.rate, p, a.div_yield)
    exact = float(heston_price(a.option, *args))
    mc = heston_mc(a.option, *args, n_steps=a.steps, n_paths=a.paths, rng=a.seed)
    vol = float(heston_implied_vol(a.spot, a.strike, a.maturity, a.rate, p, a.div_yield))
    flat = p.effective_vol(a.maturity)
    bs = float(price(a.option, a.spot, a.strike, a.maturity, a.rate, flat, a.div_yield))
    print(f"heston {a.option}  feller condition {'holds' if p.feller else 'violated'}")
    print(f"semi-analytical  {exact:.6f}")
    print(_mc_line("monte carlo", mc, 17) + f"  {a.steps} steps")
    print(f"implied vol      {vol:.6f}")
    print(f"black-scholes    {bs:.6f}  (at the average vol {flat:.4f})")
    return 0


def _cmd_plot_smile(a: argparse.Namespace) -> int:
    try:
        from ope.plots import plot_heston_smile
    except ImportError:
        return _plots_missing()
    fig = plot_heston_smile(a.spot, a.maturity, a.rate, _heston_params(a), a.div_yield)
    fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}")
    return 0


def _cmd_plot_barrier(a: argparse.Namespace) -> int:
    try:
        from ope.plots import plot_barrier_monitoring
    except ImportError:
        return _plots_missing()
    fig = plot_barrier_monitoring(
        a.option, a.spot, a.strike, a.barrier, a.maturity, a.rate, a.sigma, a.div_yield,
        kind=a.kind, seed=a.seed,
    )  # fmt: skip
    fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}")
    return 0


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "price": _cmd_price,
    "iv": _cmd_iv,
    "plot-greeks": _cmd_plot_greeks,
    "mc": _cmd_mc,
    "plot-mc": _cmd_plot_mc,
    "american": _cmd_american,
    "plot-boundary": _cmd_plot_boundary,
    "asian": _cmd_asian,
    "barrier": _cmd_barrier,
    "lookback": _cmd_lookback,
    "heston": _cmd_heston,
    "plot-smile": _cmd_plot_smile,
    "plot-barrier": _cmd_plot_barrier,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except ValueError as e:  # bad inputs are a msg for the user, not a traceback
        print(f"error: {e}", file=sys.stderr)
        return 2
