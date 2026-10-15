"""plots, they need matplotlib (the `plots` extra), each returns a matplotlib Figure and draws on
no global state -> they work w/o a gui backend"""

from ope.plots.pricing import (
    plot_exercise_boundary,
    plot_greeks_vs_spot,
    plot_mc_convergence,
    plot_tree_convergence,
)
from ope.plots.vol import (
    plot_barrier_monitoring,
    plot_heston_smile,
    plot_smiles,
    plot_term_structure,
)

__all__ = [
    "plot_barrier_monitoring",
    "plot_exercise_boundary",
    "plot_greeks_vs_spot",
    "plot_heston_smile",
    "plot_mc_convergence",
    "plot_smiles",
    "plot_term_structure",
    "plot_tree_convergence",
]
