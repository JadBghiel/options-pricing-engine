"""a synthetic spx like chain, priced by the library itself: the committed fixture and the test bed

real quotes stay out of git (the ibkr market data agreement doesnt allow passing them on) -> the
notebook, the tests and ci run on this, it looks like a real snapshot on purpose: heston prices
w/ known params, a forward from known r and q, bid ask spreads that widen in the wings, prices
rounded to the tick, and a few broken quotes of each kind the cleaning must catch, a calibration
can then be compared w/ the params that made the chain
"""

import math
from datetime import UTC, datetime, timedelta

import numpy as np

from ope.black_scholes import OptionType, _price_and_vega
from ope.calibration import heston_grid_prices
from ope.chain import Chain, MetaValue, from_rows, pick_strikes, year_fraction
from ope.heston import HestonParams
from ope.implied_volatility import implied_vol

# a typical index surface: 14% vol now, 19% long run, a steep skew (rho -0.75)
SPX_LIKE = HestonParams(v0=0.02, kappa=2.5, theta=0.035, xi=0.8, rho=-0.75)
# a friday close (16:15 new york), fixed -> the fixture stays the same
AS_OF = datetime(2026, 1, 9, 21, 15, tzinfo=UTC)
EXPIRY_DAYS = (7, 14, 28, 63, 91, 182, 364)
# broken quotes added on top, per kind (the cleaning must catch each one)
JUNK = {"one sided": 6, "crossed": 4, "stale": 5, "wide": 5, "no interest": 6, "out of bounds": 3}


def _tick(price: float) -> float:
    """cboe spx ticks: 0.05 under 3, 0.10 above"""
    return 0.05 if price < 3 else 0.10


def synthetic_chain(
    params: HestonParams = SPX_LIKE,
    *,
    S: float = 5000.0,
    r: float = 0.045,
    q: float = 0.013,
    expiry_days: tuple[int, ...] = EXPIRY_DAYS,
    strikes_per_expiry: int = 30,
    vol_spread: float = 0.004,
    junk: bool = True,
    seed: int = 0,
    as_of: datetime = AS_OF,
) -> Chain:
    """calls and puts at every strike, priced w/ params, quoted around the true price

    vol_spread: the bid ask width in vol at the money (0.004 = 0.4 vol points), wider in the
    wings, the true price is inside every bid ask, the mid is off by up to 30% of the half
    spread, junk=True adds the broken quotes of JUNK, counted in meta["junk"]
    """
    rng = np.random.default_rng(seed)
    grid = np.arange(0.5 * S, 1.5 * S, 5.0)
    rows: list[dict[str, str | float]] = []
    for days in expiry_days:
        expiry = (as_of + timedelta(days=days)).strftime("%Y%m%d")
        T = year_fraction(as_of, expiry)
        D = math.exp(-r * T)
        F = S * math.exp((r - q) * T)
        strikes = np.array(pick_strikes(grid, F, T, strikes_per_expiry))
        sides: tuple[tuple[str, OptionType], ...] = (("C", "call"), ("P", "put"))
        for right, option in sides:
            true = heston_grid_prices(option, F, D, T, strikes, params)
            for K, price in zip(strikes, true, strict=True):
                k = math.log(K / F)
                # quote the out of the money side's vol spread, wider in the wings
                otm: OptionType = "call" if K >= F else "put"
                otm_price = (
                    price
                    if option == otm
                    else price + (D * (K - F) if otm == "put" else D * (F - K))
                )
                iv = (
                    implied_vol(otm, float(max(otm_price, 1e-12)), D * F, K, T, r, 0.0)
                    if otm_price > 1e-10
                    else 0.0
                )
                vega = _price_and_vega(otm, D * F, K, T, r, max(iv, 1e-4), 0.0)[1]
                half = max(0.5 * vol_spread * (1 + 8 * abs(k)) * vega, 0.5 * _tick(price))
                center = price + rng.uniform(-0.3, 0.3) * half
                tick = _tick(price)
                bid = math.floor((center - half) / tick) * tick
                ask = math.ceil((center + half) / tick) * tick
                rows.append(
                    {
                        "expiry": expiry,
                        "right": right,
                        "strike": float(K),
                        "bid": round(bid, 2) if bid > 0 else math.nan,
                        "ask": round(ask, 2),
                        "bid_size": float(rng.integers(1, 200)),
                        "ask_size": float(rng.integers(1, 200)),
                        "last": round(price, 2),
                        "volume": float(rng.poisson(200 * math.exp(-8 * abs(k)))),
                        "open_interest": float(rng.poisson(3000 * math.exp(-4 * abs(k))) + 1),
                        "last_trade_time": (
                            as_of - timedelta(hours=float(rng.exponential(6)))
                        ).timestamp(),
                    }
                )
    counts = dict.fromkeys(JUNK, 0)
    if junk:
        counts = _break_quotes(rows, rng, as_of)
    meta: dict[str, MetaValue] = {
        "source": "synthetic",
        "symbol": "SPX",
        "trading_class": "SPXW",
        "as_of": as_of.isoformat(),
        "spot_hint": S,
        "heston": [f"{f}={getattr(params, f)}" for f in ("v0", "kappa", "theta", "xi", "rho")],
        "r": r,
        "q": q,
        "seed": seed,
        "junk": [f"{name}: {n}" for name, n in counts.items()],
    }
    return from_rows(rows, meta)


def _break_quotes(
    rows: list[dict[str, str | float]], rng: np.random.Generator, as_of: datetime
) -> dict[str, int]:
    """breaks JUNK quotes in place, each picked once, on 2 sided quotes w/ a bid of at least 1"""
    usable = [i for i, r in enumerate(rows) if isinstance(r["bid"], float) and r["bid"] >= 1.0]
    picks = rng.choice(usable, size=sum(JUNK.values()), replace=False)
    counts: dict[str, int] = {}
    pos = 0
    for name, n in JUNK.items():
        for i in picks[pos : pos + n]:
            row = rows[int(i)]
            bid, ask = float(row["bid"]), float(row["ask"])
            if name == "one sided":
                row["bid"] = math.nan
            elif name == "crossed":
                row["bid"], row["ask"] = ask + 0.1, bid
            elif name == "stale":  # last traded 10 days ago, and the price moved since
                row["last_trade_time"] = (as_of - timedelta(days=10)).timestamp()
                row["bid"], row["ask"] = round(bid * 1.2, 2), round(ask * 1.2, 2)
            elif name == "wide":
                row["ask"] = round(bid * 4 + 1, 2)
            elif name == "no interest":
                row["volume"], row["open_interest"] = 0.0, 0.0
            else:  # out of bounds: a call above D F or a put above D K, scaled way up
                row["bid"], row["ask"] = round(bid * 50 + 9000, 2), round(ask * 50 + 9000.5, 2)
        counts[name] = n
        pos += n
    return counts
