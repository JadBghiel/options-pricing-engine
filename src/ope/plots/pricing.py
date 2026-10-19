"""plots for tiers 1 to 3: the greeks, monte carlo convergence, the tree and the pde boundary"""

import math

import numpy as np
from matplotlib.figure import Figure

from ope.black_scholes import OptionType, greeks, price
from ope.monte_carlo import mc_price
from ope.pde import pde_solve
from ope.trees import crr_price


def plot_greeks_vs_spot(
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    spot_range: tuple[float, float] = (0.5, 1.5),
    points: int = 301,
) -> Figure:
    """price and the 5 greeks of a call and a put vs spot, spot_range is in multiples of K"""
    spots = np.linspace(spot_range[0] * K, spot_range[1] * K, points)
    call, put = greeks("call", spots, K, T, r, sigma, q), greeks("put", spots, K, T, r, sigma, q)
    panels = [
        ("price", price("call", spots, K, T, r, sigma, q), price("put", spots, K, T, r, sigma, q)),
        ("delta", call.delta, put.delta),
        ("gamma", call.gamma, put.gamma),
        ("vega (per 1.00 vol)", call.vega, put.vega),
        ("theta (per year)", call.theta, put.theta),
        ("rho (per 1.00 rate)", call.rho, put.rho),
    ]

    # figure directly instead of pyplot: no global state and no GUI backend needed
    fig = Figure(figsize=(12, 7), layout="constrained")
    axes = fig.subplots(2, 3)
    for ax, (title, call_values, put_values) in zip(axes.flat, panels, strict=True):
        ax.plot(spots, call_values, label="call")
        ax.plot(spots, put_values, label="put", linestyle="--")
        ax.axvline(K, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("spot S")
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle(
        f"Black-Scholes greeks vs spot (K={K:g}, T={T:g}, r={r:g}, sigma={sigma:g}, q={q:g})"
    )
    return fig


def plot_mc_convergence(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    n_min: int = 100,
    n_max: int = 1_000_000,
    points: int = 13,
    seed: int = 0,
) -> Figure:
    """monte carlo price vs number of paths, plain and w/ variance reduction, vs the closed form

    left: each estimate w/ its 95% interval, right: the standard error on log scales, which
    falls along a line of slope -1/2 (error ~ 1/sqrt(N))
    """
    ns = np.unique((np.logspace(math.log10(n_min), math.log10(n_max), points) // 2 * 2).astype(int))
    exact = float(price(option, S, K, T, r, sigma, q))
    fig = Figure(figsize=(12, 4.5), layout="constrained")
    ax_est, ax_se = fig.subplots(1, 2)
    runs = [("plain", False, False, 0.97), ("antithetic + control variate", True, True, 1.03)]
    se_plain = np.empty(0)
    for label, anti, cv, shift in runs:
        res = [
            mc_price(
                option,
                S,
                K,
                T,
                r,
                sigma,
                q,
                n_paths=int(n),
                antithetic=anti,
                control_variate=cv,
                rng=seed + k,
            )
            for k, n in enumerate(ns)
        ]
        est = np.array([x.estimate for x in res])
        se = np.array([x.std_error for x in res])
        if not anti:
            se_plain = se
        # shift on the log x axis so the 2 sets of error bars don't sit on top of each other
        ax_est.errorbar(ns * shift, est, yerr=1.96 * se, fmt="o", capsize=3, label=label)
        ax_se.loglog(ns, se, "o-", label=label)
    ax_est.axhline(exact, color="gray", linestyle="--", label=f"closed form {exact:.4f}")
    ax_est.set_xscale("log")
    ax_est.set_xlabel("number of paths N")
    ax_est.set_ylabel(f"{option} price")
    ax_est.set_title("estimate and 95% interval")
    ax_est.legend()
    ax_se.loglog(ns, se_plain[0] * np.sqrt(ns[0] / ns), ":", color="gray", label="slope -1/2")
    ax_se.set_xlabel("number of paths N")
    ax_se.set_ylabel("standard error")
    ax_se.set_title("standard error shrinks like 1/sqrt(N)")
    ax_se.legend()
    for ax in (ax_est, ax_se):
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(
        f"Monte Carlo convergence ({option}, S={S:g}, K={K:g}, T={T:g}, r={r:g}, sigma={sigma:g})"
    )
    return fig


def plot_tree_convergence(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    max_steps: int = 200,
) -> Figure:
    """european crr price vs number of steps, next to the closed form: the odd/even zigzag"""
    steps = np.arange(10, max_steps + 1)
    tree = np.array([crr_price(option, S, K, T, r, sigma, q, steps=int(n)) for n in steps])
    exact = float(price(option, S, K, T, r, sigma, q))
    fig = Figure(figsize=(10, 4.2), layout="constrained")
    ax = fig.subplots()
    odd = steps % 2 == 1
    ax.plot(steps[~odd], tree[~odd], ".", ms=4, label="even number of steps")
    ax.plot(steps[odd], tree[odd], ".", ms=4, label="odd number of steps")
    ax.axhline(exact, color="gray", linestyle="--", label=f"closed form {exact:.4f}")
    ax.set_xlabel("number of time steps n")
    ax.set_ylabel(f"european {option} price")
    ax.set_title("CRR tree: converges to Black-Scholes, zigzagging between odd and even n")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_exercise_boundary(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    n_space: int = 300,
    n_time: int = 300,
) -> Figure:
    """early exercise boundary of an american option over its life, from the pde solver"""
    sol = pde_solve(option, S, K, T, r, sigma, q, american=True, n_space=n_space, n_time=n_time)
    keep = ~np.isnan(sol.boundary)
    fig = Figure(figsize=(10, 4.5), layout="constrained")
    ax = fig.subplots()
    ax.plot(sol.times[keep], sol.boundary[keep], color="crimson", label="exercise boundary S*(t)")
    region = "below" if option == "put" else "above"
    edge = sol.spots.min() if option == "put" else sol.spots.max()
    ax.fill_between(
        sol.times[keep],
        sol.boundary[keep],
        max(edge, 0.5 * K) if option == "put" else min(edge, 1.5 * K),
        alpha=0.15,
        color="crimson",
        label=f"exercise now ({region} the line)",
    )
    ax.axhline(K, color="gray", linestyle="--", label=f"strike K = {K:g}")
    ax.set_xlim(0, T)
    ax.set_xlabel("time from today (years)")
    ax.set_ylabel("stock price")
    ax.set_title(
        f"American {option}: when to exercise early "
        f"(S={S:g}, K={K:g}, r={r:g}, sigma={sigma:g}, q={q:g})"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left" if option == "put" else "upper left")
    return fig
