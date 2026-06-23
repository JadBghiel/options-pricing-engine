"""implied volatility: the sigma that makes the black scholes price equal a quoted price

1 option at a time (scalars). safeguarded newton: newton steps on vega, kept inside a
bracket [lo, hi] that always holds the root, w/ a bisection step whenever newton would
jump out of it
"""

import math
from dataclasses import dataclass

from ope.black_scholes import OptionType, _check_option, _price_and_vega, price_bounds

_SIGMA_MAX = 1024.0  # 102400% vol, any price that needs more is the upper bound in practice


@dataclass(frozen=True)
class ImpliedVolResult:
    """solved sigma plus how the solver got there"""

    sigma: float
    iterations: int
    newton_steps: int
    bisection_steps: int


def implied_vol_result(
    option: OptionType,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> ImpliedVolResult:
    """sigma such that the black scholes price equals price, w/ iteration counts

    stops when |model price - price| <= tol * max(1, price), or when the step in sigma is
    below 1e-15 relative. raises ValueError if the price breaks the no arbitrage bounds,
    since no sigma can produce it
    """
    _check_option(option)
    if T <= 0:
        raise ValueError("implied vol needs T > 0, at expiry the price does not depend on sigma")
    lo_bound, hi_bound = (float(b) for b in price_bounds(option, S, K, T, r, q))
    price_tol = tol * max(1.0, price)
    if not lo_bound - price_tol <= price < hi_bound:
        raise ValueError(
            f"{option} price {price} is outside the no arbitrage bounds "
            f"[{lo_bound}, {hi_bound}), no volatility can produce it"
        )
    # deep ITM w/ little time left, the time value can be below float resolution and the
    # model's own price can round 1 ulp under the bound -> indistinguishable from sigma = 0
    if price <= lo_bound + price_tol:
        return ImpliedVolResult(0.0, 0, 0, 0)

    # price is increasing in sigma, so double hi until the root is inside [lo, hi]
    lo, hi = 0.0, 1.0
    while _price_and_vega(option, S, K, T, r, hi, q)[0] < price:
        lo, hi = hi, 2.0 * hi
        if hi > _SIGMA_MAX:
            raise ValueError(f"price {price} implies a volatility above {_SIGMA_MAX}")

    # LEARNING NOTE: start at the inflection point sigma* = sqrt(2|ln(F/K)|/T), where vega
    # peaks in sigma. the price is convex in sigma below it and concave above, so newton
    # from there walks to the root monotonically (Manaster and Koehler, 1982). outside the
    # bracket we start in the middle and let the safeguard catch overshoots, starting at the
    # nearest end is also monotone but ~5 iterations slower on deep OTM short dated options
    forward = S * math.exp((r - q) * T)
    guess = math.sqrt(2.0 * abs(math.log(forward / K)) / T) if forward > 0 else 0.0
    sigma = guess if lo < guess < hi else 0.5 * (lo + hi)

    newton = bisection = 0
    for iteration in range(1, max_iter + 1):
        model, vega = _price_and_vega(option, S, K, T, r, sigma, q)
        diff = model - price
        if abs(diff) <= price_tol:
            return ImpliedVolResult(sigma, iteration, newton, bisection)
        if diff > 0:
            hi = sigma
        else:
            lo = sigma

        # LEARNING NOTE: deep ITM/OTM or short dated, vega is ~0 and the newton step
        # diff/vega is huge (or vega underflows to 0). instead of a vega threshold to tune,
        # any step that leaves the bracket becomes a bisection step, which always halves it
        candidate = sigma - diff / vega if vega > 0 else math.nan
        if lo < candidate < hi:
            newton += 1
        else:
            candidate = 0.5 * (lo + hi)
            bisection += 1
        if abs(candidate - sigma) <= 1e-15 * max(1.0, sigma):
            return ImpliedVolResult(candidate, iteration, newton, bisection)
        sigma = candidate

    raise RuntimeError(f"implied vol did not converge in {max_iter} iterations")


def implied_vol(
    option: OptionType,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
) -> float:
    """sigma such that the black scholes price equals price, see implied_vol_result"""
    return implied_vol_result(option, price, S, K, T, r, q).sigma
