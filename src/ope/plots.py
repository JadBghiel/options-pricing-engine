"""plots for the closed form model and monte carlo, needs the plots extra (matplotlib)"""

import math

import numpy as np
from matplotlib.figure import Figure

from ope.black_scholes import OptionType, greeks, price
from ope.monte_carlo import mc_price


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
