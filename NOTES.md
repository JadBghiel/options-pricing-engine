# NOTES

Condensed notes for each tier: the key formulas, the gotchas we hit, and the takeaways. The full
version of each tier (intuition, derivations, validation plots, interview Q&A) lives in the
project's Notion pages; this file keeps the learning visible on GitHub.

Each tier gets the same 3 parts:

- **Key formulas:** the few equations the tier rests on, with every symbol defined.
- **Gotchas:** mistakes, numerical traps and convention mismatches, with how they were fixed.
- **Takeaways:** 3 to 5 lines I should be able to say from memory in an interview.

---

## Tier 1: Black-Scholes closed form

### Key formulas

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

S spot, K strike, T years, r rate, q dividend yield, σ volatility (all continuous, annualized).
N is the standard normal CDF and n its density.

- **Put-call parity:** $C - P = S e^{-qT} - K e^{-rT}$, in any model.
- **Greeks:** call delta $e^{-qT}N(d_1)$; gamma $e^{-qT}n(d_1)/(S\sigma\sqrt{T})$ and vega
  $S e^{-qT} n(d_1)\sqrt{T}$, the same for calls and puts; call rho $KTe^{-rT}N(d_2)$.
- **Implied vol (Newton):** $\sigma_{n+1} = \sigma_n - (C(\sigma_n) - C_{mkt})/\text{vega}(\sigma_n)$,
  started at $\sigma^* = \sqrt{2|\ln(F/K)|/T}$, where vega peaks.
- **Hull check:** S=42, K=40, T=0.5, r=10%, σ=20% gives a call of 4.7594 and a put of 0.8086.

### Gotchas

- Deep ITM with little time left, the model's own price rounded 1 ulp under the intrinsic bound,
  and the implied vol solver rejected it. A price within tolerance of the bound now gives σ = 0.
- Re-exporting the function `implied_vol` from the package shadowed the module
  `ope.implied_vol`. The module became `implied_volatility.py`.
- Finite differences have a noise floor of about eps/h. A tolerance below it fails on noise.
- `1 - N(d2)` cancels to 0 for deep OTM puts; `N(-d2)` keeps the digits.
- A 100× gap in vega or rho, or 365× in theta, is a unit convention, not a formula error.

### Takeaways

- The price comes from replication, so real-world drift and probabilities drop out. Only σ is
  unobservable, which is why traders quote vol.
- N(d2) is the risk-neutral probability of ending in the money. N(d1) is the same probability
  under the stock measure, and the call's delta when q = 0.
- Gamma and vega match for calls and puts because parity is linear in S and has no σ.
- Newton is fast near the money (3 to 6 iterations against about 40 for bisection) but stalls
  where vega vanishes, so it needs a bracket and a bisection fallback.
