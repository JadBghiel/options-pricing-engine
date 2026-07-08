"""monte carlo pricer for european options under black scholes (geometric brownian motion)

the stock price at expiry is simulated exactly in 1 step, so there is no time stepping error.
every function takes the random generator (or a seed) explicitly, so runs are reproducible.
inputs are scalars: 1 option per call, n_paths random scenarios for it
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from ope.black_scholes import FloatArray, OptionType, _check_option

GreekMethod = Literal["bump", "pathwise", "likelihood_ratio"]
RngLike = np.random.Generator | int | None


@dataclass(frozen=True)
class MCResult:
    """a monte carlo estimate, its standard error and the number of paths behind it"""

    estimate: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        """95% confidence interval, estimate +- 1.96 standard errors (N(1.96) = 0.975)"""
        half = 1.96 * self.std_error
        return self.estimate - half, self.estimate + half


def _rng(rng: RngLike) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _check_inputs(S: float, K: float, T: float, sigma: float, n_paths: int) -> None:
    if not all(math.isfinite(v) for v in (S, K, T, sigma)):
        raise ValueError("inputs must be finite")
    if S <= 0 or K <= 0:
        raise ValueError("spot S and strike K must be > 0")
    if T <= 0:
        raise ValueError("maturity T must be > 0, at expiry there is nothing to simulate")
    if sigma < 0:
        raise ValueError("volatility sigma must be >= 0")
    if n_paths < 2:
        raise ValueError("n_paths must be >= 2 to estimate a standard error")


def gbm_terminal(
    S: float, T: float, r: float, sigma: float, z: npt.ArrayLike, q: float = 0.0
) -> FloatArray:
    """stock price at T for each standard normal draw in z, under the risk neutral measure

    S_T = S * exp((r - q - sigma^2/2) T + sigma sqrt(T) z)
    """
    # LEARNING NOTE: the gbm solution is exact in log space: log S_T is normal w/ mean
    # log S + (r - q - sigma^2/2) T and variance sigma^2 T. so 1 normal draw per path gives S_T
    # w/ no discretization error, whatever T is. the -sigma^2/2 keeps E[S_T] = S e^{(r-q)T}
    zz = np.asarray(z, dtype=np.float64)
    drift = (r - q - 0.5 * sigma**2) * T
    return np.asarray(S * np.exp(drift + sigma * math.sqrt(T) * zz), dtype=np.float64)


def payoff(option: OptionType, s_t: npt.ArrayLike, K: float) -> FloatArray:
    """value at expiry: max(S_T - K, 0) for a call, max(K - S_T, 0) for a put"""
    _check_option(option)
    s = np.asarray(s_t, dtype=np.float64)
    out = np.maximum(s - K, 0.0) if option == "call" else np.maximum(K - s, 0.0)
    return np.asarray(out, dtype=np.float64)


def _normals(gen: np.random.Generator, n_paths: int, antithetic: bool) -> FloatArray:
    if not antithetic:
        return gen.standard_normal(n_paths)
    if n_paths % 2:
        raise ValueError("antithetic sampling needs an even n_paths (paths come in pairs)")
    half = gen.standard_normal(n_paths // 2)
    return np.concatenate([half, -half])


def mc_price(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    n_paths: int = 100_000,
    antithetic: bool = False,
    control_variate: bool = False,
    rng: RngLike = None,
) -> MCResult:
    """price = average of the discounted payoffs over n_paths simulated expiries

    antithetic: each draw z is paired w/ -z. control_variate: corrects the estimate w/ the
    discounted stock price, whose true average S e^{-qT} is known. both reduce the standard
    error at the same n_paths. pass rng (a seed or a Generator) to make the run reproducible
    """
    _check_option(option)
    _check_inputs(S, K, T, sigma, n_paths)
    z = _normals(_rng(rng), n_paths, antithetic)
    s_t = gbm_terminal(S, T, r, sigma, z, q)
    disc = math.exp(-r * T)
    x = disc * payoff(option, s_t, K)  # what we average
    y = disc * s_t  # control, its true mean is known

    if antithetic:
        # LEARNING NOTE: a path and its mirror are not independent, so the standard error has
        # to come from the n/2 pair averages. treating the n payoffs as independent draws
        # would report a standard error that is too small
        half = n_paths // 2
        x = 0.5 * (x[:half] + x[half:])
        y = 0.5 * (y[:half] + y[half:])

    if control_variate:
        # LEARNING NOTE: when the simulated stocks come out high on average, so do the call
        # payoffs. we know the true average of the discounted stock (S e^{-qT}), so we remove
        # beta times the sample's error on it. beta is the regression slope of payoff on
        # stock, the choice that cuts the variance the most
        cov = np.cov(x, y)
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0
        x = x - beta * (y - S * math.exp(-q * T))

    m = x.size
    return MCResult(float(x.mean()), float(x.std(ddof=1) / math.sqrt(m)), n_paths)


def mc_delta(
    option: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    method: GreekMethod = "pathwise",
    n_paths: int = 100_000,
    bump: float | None = None,
    rng: RngLike = None,
) -> MCResult:
    """delta by monte carlo, 3 ways, each w/ its own standard error

    bump: reprice w/ S +- h on the same random draws (common random numbers), h = 1% of S
    by default. pathwise: differentiate each path's payoff w/ respect to S. likelihood_ratio:
    weight each payoff by the derivative of the log density of S_T w/ respect to S
    """
    _check_option(option)
    _check_inputs(S, K, T, sigma, n_paths)
    z = _rng(rng).standard_normal(n_paths)
    disc = math.exp(-r * T)

    if method == "bump":
        h = 0.01 * S if bump is None else bump
        if not 0 < h < S:
            raise ValueError("bump must be in (0, S)")
        up = payoff(option, gbm_terminal(S + h, T, r, sigma, z, q), K)
        down = payoff(option, gbm_terminal(S - h, T, r, sigma, z, q), K)
        samples = disc * (up - down) / (2.0 * h)
    elif method == "pathwise":
        # dS_T/dS = S_T/S, and the payoff's slope in S_T is 1 above the strike for a call,
        # -1 below it for a put, 0 elsewhere
        s_t = gbm_terminal(S, T, r, sigma, z, q)
        slope = (s_t > K).astype(np.float64) if option == "call" else -(s_t < K).astype(np.float64)
        samples = disc * slope * s_t / S
    elif method == "likelihood_ratio":
        if sigma == 0:
            raise ValueError("the likelihood ratio method needs sigma > 0")
        s_t = gbm_terminal(S, T, r, sigma, z, q)
        samples = disc * payoff(option, s_t, K) * z / (S * sigma * math.sqrt(T))
    else:
        raise ValueError(f"method must be 'bump', 'pathwise' or 'likelihood_ratio', got {method!r}")

    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / math.sqrt(n_paths)), n_paths)
