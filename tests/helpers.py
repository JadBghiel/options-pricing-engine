"""helpers several test files share"""

from collections.abc import Callable

from ope.monte_carlo import MCResult


def coverage(run: Callable[[int], MCResult], exact: float, seeds: int = 200) -> float:
    """share of seeds whose 95% interval holds exact"""
    inside = 0
    for seed in range(seeds):
        lo, hi = run(seed).ci95
        inside += lo <= exact <= hi
    return inside / seeds
