"""from a raw chain to an implied vol surface: clean the quotes, imply the forward, invert mids

1. clean the prices: no 2 sided quote, crossed, stale, wide spread, each filter counted
2. forward: per expiry, put call parity C - P = D (F - K) is a line in K -> a regression gives
   the discount factor D and the forward F, no r or q assumed, it comes before step 3: parity
   needs good prices, traded or not, and itm quotes rarely trade
3. liquidity: drop the quotes nobody traded and nobody holds
4. surface: the implied vol of each out of the money mid inside the no arbitrage bounds (tier 1
   solver), plus the vols of the bid and the ask, whose gap says how much the market itself
   knows about that point
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import numpy.typing as npt

from ope.black_scholes import FloatArray, OptionType, price_bounds
from ope.chain import Chain, StrArray
from ope.implied_volatility import implied_vol

# (ask - bid) / mid above this -> the quote says little about the price (see EXPLANATION 19)
MAX_REL_SPREAD = 0.5
# a quote whose last trade is older than this, when the source records trade times
MAX_TRADE_AGE_DAYS = 5.0
# parity needs this many strikes w/ both a call and a put
MIN_PAIRS = 3
# floor on the bid ask spread in vol -> 1 very tight quote cant take all the weight
MIN_VOL_SPREAD = 0.001


@dataclass(frozen=True)
class CleanReport:
    """how many quotes each filter removed, in the order they ran"""

    n_raw: int
    removed: Mapping[str, int]
    n_kept: int
    notes: Mapping[str, str] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [f"{'raw quotes':<28}{self.n_raw:>6}"]
        for name, n in self.removed.items():
            note = f"  ({self.notes[name]})" if name in self.notes else ""
            out.append(f"  - {name:<24}{n:>6}{note}")
        out.append(f"{'kept':<28}{self.n_kept:>6}")
        return out


def clean_chain(
    chain: Chain,
    *,
    max_rel_spread: float = MAX_REL_SPREAD,
    max_trade_age_days: float = MAX_TRADE_AGE_DAYS,
    liquidity: bool = True,
    report: CleanReport | None = None,
) -> tuple[Chain, CleanReport]:
    """the quote level filters, each one counted, liquidity=False stops before the volume and
    open interest one (iv_surface implies the forward in between), report: counts so far, the
    new ones are added after them"""
    keep = np.ones(len(chain), dtype=bool)
    removed: dict[str, int] = dict(report.removed) if report else {}
    notes: dict[str, str] = dict(report.notes) if report else {}
    n_raw = report.n_raw if report else len(chain)

    def apply(name: str, bad: npt.NDArray[np.bool_]) -> None:
        nonlocal keep
        removed[name] = int(np.sum(keep & bad))
        keep &= ~bad

    bid, ask = chain.bid, chain.ask
    with np.errstate(invalid="ignore"):
        if report is None:
            apply("no two sided quote", ~(np.isfinite(bid) & np.isfinite(ask) & (ask > 0)))
            apply("crossed or locked", bid >= ask)
            traded = chain.last_trade_time
            if np.any(np.isfinite(traded)):
                cutoff = (chain.as_of - timedelta(days=max_trade_age_days)).timestamp()
                apply("stale", np.isfinite(traded) & (traded < cutoff))
            else:
                removed["stale"] = 0
                notes["stale"] = "the source gives no trade times"
            mid = 0.5 * (bid + ask)
            apply("wide spread", (ask - bid) / mid > max_rel_spread)
        if liquidity:
            # LEARNING NOTE: no trade today AND open interest known to be 0, volume = 0 alone
            # isnt enough: on the real spxw snapshot 57% of the 2 sided quotes didnt trade that
            # day, mostly long dated quotes market makers keep tight anyway (the 1 year atm put
            # at 366.3 / 369.1), and a missing open interest is unknown: it arrives after
            # the quote and was missing on 18 of 38 quotes 9 months out
            apply("no volume, no open interest", ~(chain.volume > 0) & (chain.open_interest == 0))
    return chain.select(keep), CleanReport(n_raw, removed, int(np.sum(keep)), notes)


@dataclass(frozen=True)
class Forward:
    """1 expiry's forward F and discount factor D from put call parity"""

    expiry: str
    T: float
    F: float
    D: float
    n_pairs: int
    max_resid: float  # largest |C - P - D (F - K)| over the pairs kept, in price units
    n_outliers: int = 0  # pairs dropped by the robust pass

    @property
    def r(self) -> float:
        """the rate D implies, -log(D) / T"""
        return -math.log(self.D) / self.T

    def q(self, S: float) -> float:
        """the dividend yield that would make S grow to F: r - log(F / S) / T"""
        return self.r - math.log(self.F / S) / self.T


# LEARNING NOTE: C - P = D F - D K, a straight line in K: the slope is -D, the intercept D F,
# on the real snapshot the index level from IB was out of sync w/ the 16:15 option
# quotes, and w/ it the 1 week dividend yield came out at -10%, the options carry their own
# forward and we read it from them
def _parity_line(ks: FloatArray, y: FloatArray, w: FloatArray) -> tuple[float, float, FloatArray]:
    """weighted least squares of y = D F - D K -> (D F, D, residuals)"""
    A = np.column_stack([np.ones_like(ks), -ks])
    coef = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    DF, D = float(coef[0]), float(coef[1])
    return DF, D, y - (DF - D * ks)


def implied_forwards(chain: Chain, min_pairs: int = MIN_PAIRS) -> dict[str, Forward]:
    """F and D per expiry from every strike w/ both a call and a put, weighted by 1 / (sum of
    the 2 spreads): wide quotes in the wings move the line less, pairs further from the line
    than twice their spread are dropped, the worst first, and the line refit each time"""
    out: dict[str, Forward] = {}
    T_all = chain.T
    for e in chain.expiries():
        m = chain.expiry == e
        calls = {float(chain.strike[i]): int(i) for i in np.flatnonzero(m & (chain.right == "C"))}
        puts = {float(chain.strike[i]): int(i) for i in np.flatnonzero(m & (chain.right == "P"))}
        ks = np.array(sorted(set(calls) & set(puts)))
        if len(ks) < min_pairs:
            continue
        ic = np.array([calls[k] for k in ks])
        ip = np.array([puts[k] for k in ks])
        y = chain.mid[ic] - chain.mid[ip]
        spread = (chain.ask[ic] - chain.bid[ic]) + (chain.ask[ip] - chain.bid[ip])
        w = 1.0 / np.maximum(spread, 1e-8)
        # 1 bad price (a fat finger, a quote from another moment) drags the whole line, weights
        # alone dont stop a residual of thousands -> the worst pair goes 1 at a time, then a
        # refit: dropping every pair far from the 1st line also dropped good ones at the ends
        # (the bad pair tilts it), 4 good pairs for 1 bad quote in the tests
        ok = np.ones(len(ks), dtype=bool)
        DF, D, resid = _parity_line(ks, y, w)
        while True:
            excess = np.abs(resid) - 2 * spread[ok]  # mids of a good pair: within 1/2 spread
            worst = int(np.argmax(excess))
            if excess[worst] <= 0 or np.sum(ok) <= min_pairs:
                break
            ok[np.flatnonzero(ok)[worst]] = False
            DF, D, resid = _parity_line(ks[ok], y[ok], w[ok])
        n_out = int(np.sum(~ok))
        if not (D > 0 and DF > 0):
            continue
        T = float(T_all[m][0])
        out[e] = Forward(e, T, DF / D, D, len(resid), float(np.max(np.abs(resid))), n_out)
    return out


@dataclass(frozen=True)
class Surface:
    """1 row per out of the money quote kept, k = log(K / F), iv_bid is nan when the bid is at
    or below the no arbitrage floor (no vol produces it)"""

    expiry: StrArray
    T: FloatArray
    K: FloatArray
    k: FloatArray
    right: StrArray
    mid: FloatArray
    iv: FloatArray
    iv_bid: FloatArray
    iv_ask: FloatArray
    forwards: Mapping[str, Forward]
    report: CleanReport

    def __len__(self) -> int:
        return len(self.expiry)

    def expiries(self) -> list[str]:
        return sorted(set(self.expiry.tolist()))

    @property
    def vol_spread(self) -> FloatArray:
        """ask vol - bid vol, the bid side taken as 0 vol when it has none"""
        return self.iv_ask - np.nan_to_num(self.iv_bid, nan=0.0)

    @property
    def weights(self) -> FloatArray:
        """1 / vol spread, floored: a quote the market itself is unsure about counts less"""
        return 1.0 / np.maximum(self.vol_spread, MIN_VOL_SPREAD)

    def rows(self, expiry: str) -> npt.NDArray[np.bool_]:
        return np.asarray(self.expiry == expiry, dtype=np.bool_)


def _iv(option: OptionType, price: float, fwd: Forward, K: float) -> float:
    """black scholes vol w/ forward F and discount D: S = D F, r = -log(D) / T, q = 0"""
    try:
        return implied_vol(option, price, fwd.D * fwd.F, K, fwd.T, fwd.r, 0.0)
    except ValueError:
        return math.nan


def iv_surface(chain: Chain, *, max_rel_spread: float = MAX_REL_SPREAD) -> Surface:
    """raw chain -> implied vols of the out of the money mids (puts below F, calls at or above),
    w/ every step counted in surface.report"""
    chain, report = clean_chain(chain, max_rel_spread=max_rel_spread, liquidity=False)
    forwards = implied_forwards(chain)
    chain, report = clean_chain(chain, report=report)
    removed = dict(report.removed)
    has_fwd = np.isin(chain.expiry, list(forwards))
    removed["expiry w/o a forward"] = int(np.sum(~has_fwd))

    rows: list[tuple[int, float, float, float, float]] = []
    n_bounds = n_itm = 0
    for i in np.flatnonzero(has_fwd):
        fwd = forwards[str(chain.expiry[i])]
        K = float(chain.strike[i])
        option: OptionType = "call" if chain.right[i] == "C" else "put"
        lo, hi = (float(b) for b in price_bounds(option, fwd.D * fwd.F, K, fwd.T, fwd.r, 0.0))
        mid = float(chain.mid[i])
        if not lo < mid < hi:
            n_bounds += 1
            continue
        if (option == "put") != (K < fwd.F):
            n_itm += 1  # the itm side is the otm one plus parity, it adds no information
            continue
        iv = _iv(option, mid, fwd, K)
        if math.isnan(iv):  # pragma: no cover, inside the bounds a vol always exists
            n_bounds += 1
            continue
        bid_iv = _iv(option, float(chain.bid[i]), fwd, K) if chain.bid[i] > lo else math.nan
        rows.append((int(i), K / fwd.F, iv, bid_iv, _iv(option, float(chain.ask[i]), fwd, K)))
    removed["outside no arbitrage bounds"] = n_bounds
    removed["in the money side (parity)"] = n_itm
    notes = dict(report.notes)
    notes["in the money side (parity)"] = "not junk, the otm side carries the same vol"

    idx = np.array([r[0] for r in rows], dtype=np.int64)
    return Surface(
        expiry=chain.expiry[idx],
        T=chain.T[idx],
        K=chain.strike[idx],
        k=np.log(np.array([r[1] for r in rows])),
        right=chain.right[idx],
        mid=chain.mid[idx],
        iv=np.array([r[2] for r in rows]),
        iv_bid=np.array([r[3] for r in rows]),
        iv_ask=np.array([r[4] for r in rows]),
        forwards=forwards,
        report=CleanReport(report.n_raw, removed, len(rows), notes),
    )
