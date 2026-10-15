"""cli cmds for tiers 1 to 4: black scholes, implied vol, monte carlo, american options"""

import argparse

from ope import cpp
from ope.black_scholes import greeks, price
from ope.cli.args import (
    Command,
    add_cpp_arg,
    add_option_args,
    add_sigma,
    plots_missing,
    saved,
)
from ope.implied_volatility import implied_vol_result
from ope.monte_carlo import mc_price
from ope.pde import pde_price
from ope.trees import crr_price


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> dict[str, Command]:
    p = sub.add_parser("price", help="Black-Scholes price and Greeks")
    add_option_args(p)
    add_sigma(p)
    add_cpp_arg(p)

    p = sub.add_parser("iv", help="implied volatility from a price")
    add_option_args(p)
    p.add_argument("--price", type=float, required=True)

    p = sub.add_parser("plot-greeks", help="plot price and Greeks vs spot to a PNG")
    add_option_args(p, spot=False)
    add_sigma(p)
    p.add_argument("--out", default="greeks_vs_spot.png")

    p = sub.add_parser("mc", help="Monte Carlo price with a 95% confidence interval")
    add_option_args(p)
    add_sigma(p)
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0, help="same seed, same result")
    p.add_argument("--antithetic", action="store_true", help="pair each draw z with -z")
    p.add_argument("--control", action="store_true", help="control variate on the stock")
    add_cpp_arg(p)
    p.add_argument("--threads", type=int, default=None, help="with --cpp, default all cores")

    p = sub.add_parser("plot-mc", help="plot Monte Carlo convergence vs paths to a PNG")
    add_option_args(p)
    add_sigma(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="mc_convergence.png")

    p = sub.add_parser("american", help="American price by tree and PDE, vs European")
    add_option_args(p)
    add_sigma(p)
    p.add_argument("--steps", type=int, default=2000, help="tree time steps")
    p.add_argument("--grid", type=int, default=300, help="pde nodes and time steps")
    add_cpp_arg(p, what="the tree")

    p = sub.add_parser("plot-boundary", help="plot the early exercise boundary to a PNG")
    add_option_args(p)
    add_sigma(p)
    p.add_argument("--out", default="exercise_boundary.png")

    return {
        "price": _cmd_price,
        "iv": _cmd_iv,
        "plot-greeks": _cmd_plot_greeks,
        "mc": _cmd_mc,
        "plot-mc": _cmd_plot_mc,
        "american": _cmd_american,
        "plot-boundary": _cmd_plot_boundary,
    }


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
    if a.cpp:  # c++ draws from its own generator -> a seed gives a different (valid) estimate
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


def _cmd_plot_greeks(a: argparse.Namespace) -> int:
    try:
        from ope.plots import plot_greeks_vs_spot
    except ImportError:
        return plots_missing()
    return saved(plot_greeks_vs_spot(a.strike, a.maturity, a.rate, a.sigma, a.div_yield), a.out)


def _cmd_plot_mc(a: argparse.Namespace) -> int:
    try:
        from ope.plots import plot_mc_convergence
    except ImportError:
        return plots_missing()
    fig = plot_mc_convergence(
        a.option, a.spot, a.strike, a.maturity, a.rate, a.sigma, a.div_yield, seed=a.seed
    )
    return saved(fig, a.out)


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
        return plots_missing()
    fig = plot_exercise_boundary(
        a.option, a.spot, a.strike, a.maturity, a.rate, a.sigma, a.div_yield
    )
    return saved(fig, a.out)
