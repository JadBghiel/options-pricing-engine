"""plots for black scholes, monte carlo, trees, the pde, exotics and heston (needs matplotlib)"""

import math

import numpy as np
from matplotlib.figure import Figure

from ope.black_scholes import OptionType, greeks, price
from ope.exotics import BarrierKind, barrier_mc, barrier_price
from ope.heston import HestonParams, heston_implied_vol
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

    left: each estimate w/ its 95% interval. right: the standard error on log scales, which
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
    """european crr price vs number of steps, against the closed form: the odd/even zigzag"""
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


def plot_barrier_monitoring(
    option: OptionType,
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    kind: BarrierKind,
    dates: tuple[int, ...] = (4, 12, 26, 52, 126, 252),
    n_paths: int = 200_000,
    seed: int = 0,
) -> Figure:
    """knock out price vs how often the barrier is checked, against the continuous closed form

    left: discretely monitored monte carlo (95% intervals), the bgk approximation and the
    continuous price. right: the gap to the continuous price on log scales, slope -1/2
    """
    ns = np.array(dates)
    res = [
        barrier_mc(
            option, S, K, H, T, r, sigma, q, kind=kind, n_dates=int(n), n_paths=n_paths,
            rng=seed + k,
        )
        for k, n in enumerate(ns)
    ]  # fmt: skip
    est = np.array([x.estimate for x in res])
    se = np.array([x.std_error for x in res])
    bgk = [barrier_price(option, S, K, H, T, r, sigma, q, kind=kind, n_dates=int(n)) for n in ns]
    cont = barrier_price(option, S, K, H, T, r, sigma, q, kind=kind)
    vanilla = float(price(option, S, K, T, r, sigma, q))

    fig = Figure(figsize=(12, 4.5), layout="constrained")
    ax_p, ax_gap = fig.subplots(1, 2)
    ax_p.errorbar(ns, est, yerr=1.96 * se, fmt="o", capsize=3, label="checked on n dates (MC)")
    ax_p.plot(ns, bgk, "--", color="tab:orange", label="continuous formula, shifted barrier (BGK)")
    ax_p.axhline(cont, color="crimson", label=f"checked continuously {cont:.4f}")
    ax_p.set_xscale("log")
    ax_p.set_xlabel("number of monitoring dates n")
    ax_p.set_ylabel(f"{kind} {option} price")
    # the vanilla goes in the title: as a line it would squash the curve at the bottom
    ax_p.set_title(f"fewer checks, fewer knock outs, higher price (no barrier: {vanilla:.2f})")
    ax_p.legend(fontsize=8)
    ax_gap.loglog(ns, est - cont, "o-", label="MC minus continuous")
    ax_gap.loglog(ns, (est[0] - cont) * np.sqrt(ns[0] / ns), ":", color="gray", label="slope -1/2")
    ax_gap.set_xlabel("number of monitoring dates n")
    ax_gap.set_ylabel("gap to the continuous price")
    ax_gap.set_title("the gap shrinks like 1/sqrt(n)")
    ax_gap.legend()
    for ax in (ax_p, ax_gap):
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(
        f"Discrete vs continuous barrier ({kind} {option}, S={S:g}, K={K:g}, H={H:g}, T={T:g}, "
        f"r={r:g}, sigma={sigma:g})"
    )
    return fig


def plot_heston_smile(
    S: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float = 0.0,
    *,
    moneyness: tuple[float, float] = (0.6, 1.4),
    points: int = 25,
    rhos: tuple[float, ...] = (-0.7, 0.0, 0.7),
) -> Figure:
    """implied vol of heston prices vs strike, against the flat line black scholes assumes

    left: the smile of params. right: the same params w/ each correlation in rhos, rho tilts it
    """
    strikes = S * np.linspace(moneyness[0], moneyness[1], points)
    flat = params.effective_vol(T)
    fig = Figure(figsize=(12, 4.5), layout="constrained")
    ax_s, ax_rho = fig.subplots(1, 2, sharey=True)
    vols = np.asarray(heston_implied_vol(S, strikes, T, r, params, q))
    ax_s.plot(strikes, 100 * vols, "o-", ms=3, label="Heston")
    ax_s.axhline(
        100 * flat,
        color="gray",
        linestyle="--",
        label=f"Black-Scholes, {100 * flat:.1f}% everywhere",
    )
    ax_s.set_title(
        f"Heston smile (v0={params.v0:g}, kappa={params.kappa:g}, theta={params.theta:g}, "
        f"xi={params.xi:g}, rho={params.rho:g})",
        fontsize=10,
    )
    for rho in rhos:
        p = HestonParams(params.v0, params.kappa, params.theta, params.xi, rho)
        ax_rho.plot(strikes, 100 * np.asarray(heston_implied_vol(S, strikes, T, r, p, q)),
                    label=f"rho = {rho:g}")  # fmt: skip
    ax_rho.axhline(100 * flat, color="gray", linestyle="--")
    ax_rho.set_title("rho tilts the smile: < 0 skew down, > 0 skew up", fontsize=10)
    for ax in (ax_s, ax_rho):
        ax.axvline(S * math.exp((r - q) * T), color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("strike K (dotted: the forward)")
        ax.grid(alpha=0.3)
        ax.legend()
    ax_s.set_ylabel("implied volatility (%)")
    fig.suptitle(f"Implied volatility of Heston prices vs flat Black-Scholes (S={S:g}, T={T:g})")
    return fig
