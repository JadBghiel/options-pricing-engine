"""cli cmds for tier 5: asian, barrier and lookback options, heston"""

import argparse

from ope.barrier import barrier_mc, barrier_price, lookback_mc, lookback_price
from ope.black_scholes import price
from ope.cli.args import (
    Command,
    add_mc_args,
    add_option_args,
    add_sigma,
    mc_line,
    plots_missing,
    saved,
)
from ope.exotics import asian_geometric_price, asian_mc
from ope.heston import HestonParams, heston_implied_vol, heston_price
from ope.heston_mc import heston_mc


def _add_heston_args(p: argparse.ArgumentParser) -> None:
    # the readme example's defaults, feller violated -> the variance often hits 0
    p.add_argument("--v0", type=float, default=0.04, help="variance today, 0.04 = 20%% vol")
    p.add_argument("--kappa", type=float, default=1.5, help="speed back to theta")
    p.add_argument("--theta", type=float, default=0.04, help="long run variance")
    p.add_argument("--xi", type=float, default=0.6, help="vol of vol")
    p.add_argument("--rho", type=float, default=-0.7, help="stock vs variance correlation")


def _add_barrier_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-H", "--barrier", type=float, required=True)
    p.add_argument("--kind", choices=["down-and-out", "up-and-out"], required=True)


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> dict[str, Command]:
    p = sub.add_parser("asian", help="arithmetic Asian by Monte Carlo, geometric control variate")
    add_option_args(p)
    add_sigma(p)
    add_mc_args(p, dates=12)

    p = sub.add_parser("barrier", help="knock-out barrier, discrete vs continuous monitoring")
    add_option_args(p)
    add_sigma(p)
    _add_barrier_args(p)
    add_mc_args(p, dates=252)

    p = sub.add_parser("lookback", help="floating strike lookback, discrete vs continuous")
    add_option_args(p, strike=False)
    add_sigma(p)
    add_mc_args(p, dates=252)

    p = sub.add_parser("heston", help="Heston price: semi-analytical, Monte Carlo, implied vol")
    add_option_args(p)
    _add_heston_args(p)
    p.add_argument("--steps", type=int, default=100, help="Monte Carlo time steps")
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("plot-smile", help="plot the Heston implied vol smile to a PNG")
    add_option_args(p, strike=False)
    _add_heston_args(p)
    p.add_argument("--out", default="heston_smile.png")

    p = sub.add_parser("plot-barrier", help="plot a barrier price vs monitoring dates to a PNG")
    add_option_args(p)
    add_sigma(p)
    _add_barrier_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="barrier_monitoring.png")

    return {
        "asian": _cmd_asian,
        "barrier": _cmd_barrier,
        "lookback": _cmd_lookback,
        "heston": _cmd_heston,
        "plot-smile": _cmd_plot_smile,
        "plot-barrier": _cmd_plot_barrier,
    }


def _cmd_asian(a: argparse.Namespace) -> int:
    args = (a.spot, a.strike, a.maturity, a.rate, a.sigma, a.div_yield)
    kw = {"n_dates": a.dates, "n_paths": a.paths, "rng": a.seed}
    cv = asian_mc(a.option, *args, **kw, control_variate=True)
    plain = asian_mc(a.option, *args, **kw)
    geo = asian_geometric_price(a.option, *args, n_dates=a.dates)
    print(f"arithmetic asian {a.option}, average of {a.dates} dates, {a.paths} paths")
    print(mc_line("control variate", cv, 19))
    print(mc_line("plain", plain, 19))
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
    print(mc_line(f"checked on {a.dates} dates", discrete))
    print(f"BGK approximation       {bgk:.6f}  (continuous formula, shifted barrier)")
    print(mc_line("checked continuously", bridge))
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
    print(mc_line(f"checked on {a.dates} dates", discrete))
    print(mc_line("checked continuously", cont))
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
    print(mc_line("monte carlo", mc, 17) + f"  {a.steps} steps")
    print(f"implied vol      {vol:.6f}")
    print(f"black-scholes    {bs:.6f}  (at the average vol {flat:.4f})")
    return 0


def _cmd_plot_smile(a: argparse.Namespace) -> int:
    try:
        from ope.plots import plot_heston_smile
    except ImportError:
        return plots_missing()
    fig = plot_heston_smile(a.spot, a.maturity, a.rate, _heston_params(a), a.div_yield)
    return saved(fig, a.out)


def _cmd_plot_barrier(a: argparse.Namespace) -> int:
    try:
        from ope.plots import plot_barrier_monitoring
    except ImportError:
        return plots_missing()
    fig = plot_barrier_monitoring(
        a.option, a.spot, a.strike, a.barrier, a.maturity, a.rate, a.sigma, a.div_yield,
        kind=a.kind, seed=a.seed,
    )  # fmt: skip
    return saved(fig, a.out)
