"""plots for the closed form model, needs the plots extra (matplotlib)"""

import numpy as np
from matplotlib.figure import Figure

from ope.black_scholes import greeks, price


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
