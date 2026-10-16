# NOTES

Condensed notes for each tier: the main formulas, the gotchas we hit, and the takeaways. The full
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
- A 100× gap in vega or rho, or 365× in theta, comes from a unit convention: the formula is fine.

### Takeaways

- The price comes from replication, so real-world drift and probabilities drop out. Only σ is
  unobservable, which is why traders quote vol.
- N(d2) is the risk-neutral probability of ending in the money. N(d1) is the same probability
  under the stock measure, and the call's delta when q = 0.
- Gamma and vega match for calls and puts because parity is linear in S and has no σ.
- Newton is fast near the money (3 to 6 iterations against about 40 for bisection) but stalls
  where vega vanishes, so it needs a bracket and a bisection fallback.

## Tier 2: Monte Carlo

### Key formulas

$$S_T = S\,e^{(r - q - \sigma^2/2)T + \sigma\sqrt{T}\,Z}, \qquad Z \sim N(0, 1)$$

The stock price at expiry, simulated in one exact step. The −σ²/2 makes the average S_T equal
the forward S·e^((r−q)T).

$$\hat{C} = \frac{1}{N}\sum_{i=1}^{N} e^{-rT}\,\text{payoff}(S_T^{(i)}), \qquad \text{SE} = \frac{s}{\sqrt{N}}, \qquad \text{CI}_{95\%} = \hat{C} \pm 1.96\,\text{SE}$$

N paths, s the sample standard deviation of the discounted payoffs.

- **Antithetic:** use Z and −Z; the standard error comes from the N/2 pair averages.
- **Control variate:** $\hat{C}_{cv} = \bar{X} - \beta(\bar{Y} - S e^{-qT})$, with X the discounted
  payoff, Y the discounted stock (known average) and β the regression slope of X on Y.
- **Delta:** pathwise $e^{-rT}\,1_{S_T>K}\,S_T/S$ (call); likelihood ratio
  $e^{-rT}\,\text{payoff}\cdot Z/(S\sigma\sqrt{T})$; bump $(V(S+h) - V(S-h))/2h$ on the same draws.

### Gotchas

- A 95% interval misses 1 run in 20 by design: judge the method on coverage over many seeds,
  never on a single run. Seed 1 happened to miss with two methods at once.
- Antithetic pairs are not independent. Computing the standard error from the N payoffs instead
  of the N/2 pair averages reports an error that is too small.
- With σ = 0 every path is identical, but the computed standard error is 6e-16, not 0: averaging
  identical numbers in floating point leaves rounding noise.
- Estimating β from the same sample makes the control variate very slightly biased (order 1/N);
  invisible next to the standard error at any useful N.

### Takeaways

- Monte Carlo prices any payoff you can simulate; its error shrinks like 1/√N, so 100 times more
  paths buys 10 times less error.
- Variance reduction is free accuracy: on Hull's call, antithetic plus a control variate cut the
  error 12 times, worth about 150 times more paths.
- Pathwise and bump-with-common-random-numbers deltas are precise; likelihood ratio is about 3.5
  times noisier but works even for payoffs that jump.
- Sobol points cover the bell curve evenly and beat 1/√N, but then the usual standard error
  formula no longer applies.

## Tier 3: American options

### Key formulas

**Binomial tree (Cox-Ross-Rubinstein):**

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac{1}{u}, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}, \qquad V_{i,j} = \max\left(e^{-r\Delta t}\left[p\,V_{i+1,j+1} + (1-p)\,V_{i+1,j}\right],\ \text{payoff}(S_{i,j})\right)$$

The max applies to American options: at each node, the better of waiting and exercising.

**PDE in log price** (x = ln S, τ = time to expiry), solved backwards from the payoff:

$$\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 \frac{\partial^2 V}{\partial x^2} + \left(r - q - \frac{1}{2}\sigma^2\right)\frac{\partial V}{\partial x} - rV, \qquad V \geq \text{payoff for an American option}$$

Crank-Nicolson in time (average of explicit and implicit), the first steps fully implicit
(Rannacher), and PSOR to enforce V ≥ payoff.

### Gotchas

- The tree price zigzags between odd and even step counts; average n and n + 1.
- Put the strike on a grid node, or the kink gets smeared and convergence turns erratic.
- Plain Crank-Nicolson rings at the payoff's kink: gamma at the strike came out 5.1 instead of
  0.039. Two Rannacher steps fix it.
- One absolute tolerance doesn't fit every price: small prices need an absolute floor, big prices
  a relative bound.
- The American minus European "premium" must use the same tree for both, or the tree's own error
  shows up as a fake (even negative) premium.

### Takeaways

- An American option is priced backwards from expiry: at every node or grid point, the better of
  waiting and exercising now.
- Without dividends, an American call is never exercised early, so it equals the European call;
  with a dividend yield it can be worth more.
- An American put is exercised early when the stock falls below a boundary that rises toward the
  strike as expiry approaches.
- Tree and PDE are two independent methods; agreeing to 0.05% is the evidence both are right.

## Tier 4: C++ core

### Key formulas

**Philox random numbers:** a keyed scrambler, no state between draws.

$$\text{block}_k = \text{philox}(\text{counter} = k,\ \text{key} = \text{seed}), \qquad z_i = \text{BoxMuller}(\text{block}_{\lfloor i/4 \rfloor})_{\,i \bmod 4}$$

$$z_1 = \sqrt{-2\ln u_1}\,\cos(2\pi u_2), \qquad z_2 = \sqrt{-2\ln u_1}\,\sin(2\pi u_2)$$

**Merging two chunks** (counts n_a, n_b, means, centered sums of squares M2), exact in algebra:

$$\delta = \bar{x}_b - \bar{x}_a, \quad \bar{x} = \bar{x}_a + \delta\,\frac{n_b}{n}, \quad M_2 = M_{2,a} + M_{2,b} + \delta^2\,\frac{n_a n_b}{n}$$

**Amdahl's law:** with a serial fraction s, n threads give at most

$$\text{speedup}(n) = \frac{1}{s + (1 - s)/n}$$

### Gotchas

- Compare C++ with NumPy, not with Python loops: a loop is 10 to 100x slower than anything
  compiled, and nobody prices a chain that way.
- Benchmark a Release build. The module records its build type and the benchmarks refuse a
  Debug one.
- Never `-ffast-math`: it assumes no infinities, and the degenerate Black-Scholes cases set d1 to
  ±∞ on purpose. `option.hpp` refuses to compile under it.
- Don't expect identical bits: `erfc` vs. SciPy's `ndtr`, the C library's `exp` vs. NumPy's, and the
  summation order all differ in the last place. Compare within a tolerance.
- C++ can't reproduce NumPy's random stream: feed both sides the same normals for an exact check.
- Time threaded code with wall-clock time (`UseRealTime`): the main thread's CPU time hardly moves
  while the workers run.
- Thread scaling on a laptop is noisy (2.9x to 5.1x at 8 threads across runs): turbo, hyperthreads
  sharing a core, and other programs.

### Takeaways

- C++ wins outright where Python pays its interpreter in a loop (the tree: 30x). Where NumPy
  already runs compiled loops, one C++ thread ties it; threads bring the rest.
- Monte Carlo on 1 thread is a tie for a precise reason: C++ prices about 2x faster from the same
  normals, but its Box-Muller normals cost 2x more than NumPy's ziggurat ones.
- Reproducible parallel random numbers come from a counter-based generator plus a fixed reduction
  order.
- One test suite for two implementations: same signatures, a fixture that swaps the module.


## Tier 5: Exotics and Heston

### Key formulas

**Geometric Asian** over n dates T/n, ..., T: log G is normal, so Black-Scholes applies to it.

$$\mathbb{E}[\log G] = \log S + \left(r - q - \tfrac{1}{2}\sigma^2\right)T\,\frac{n+1}{2n}, \qquad \operatorname{Var}[\log G] = \sigma^2 T\,\frac{(n+1)(2n+1)}{6n^2}$$

**Brownian bridge crossing** between two dates, a and b the distances of log S to log H at both
ends: the path touched H in between with probability

$$p = \exp\left(-\frac{2ab}{\sigma^2 \Delta t}\right)$$

**BGK shift:** a barrier checked every Δt ≈ a continuous barrier moved away from spot,
H → H·exp(±0.5826·σ·√Δt).

**Heston:**

$$dS = (r-q)S\,dt + \sqrt{v}\,S\,dW_1, \qquad dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_2, \qquad dW_1\,dW_2 = \rho\,dt$$

Full truncation Euler: v⁺ = max(v, 0) in every term, the stored v may stay negative.

$$v_{t+\Delta t} = v_t + \kappa(\theta - v_t^+)\Delta t + \xi\sqrt{v_t^+\Delta t}\,Z_v$$

**Semi-analytical price:** the Black-Scholes shape with Heston's probabilities, each one integral
of the characteristic function φ (Gil-Pelaez).

$$C = S e^{-qT} P_1 - K e^{-rT} P_2, \qquad P_2 = \frac{1}{2} + \frac{1}{\pi}\int_0^\infty \operatorname{Re}\left[\frac{e^{-iu\log K}\,\varphi(u)}{iu}\right] du$$

### Gotchas

- A barrier checked daily is a different contract from one watched continuously: the up-and-out
  call at H = 120 is worth 1.32 against 1.18 (12.5% and 31 standard errors apart). Check each
  price against a reference for the same contract.
- Euler bias isn't noise: at 2 steps a year the Heston price is 83 standard errors off. More
  paths shrink the error bar, only more steps shrink the bias.
- With the Feller condition violated the Euler variance goes negative on 8% to 37% of steps:
  full truncation, never `sqrt` of a negative.
- Heston's 1993 characteristic function has a complex log that jumps by 2π at long maturities
  (16.39 instead of 29.09 at 5 years, with no error message). Use the "Little Heston Trap" form.
- Below ξ ≈ 1e-5 the characteristic function loses digits (κθ/ξ² times a bracket near 0).
- Tolerances after many steps: 252 additions of the log drift round to about 1e-12 relative.

### Takeaways

- For GBM, simulating date by date is exact: the monitoring dates are part of the contract, not
  a numerical choice. For Heston the step is a numerical choice with a bias.
- A control variate is as good as its correlation ρ, the gain is 1/√(1 − ρ²): 0.9996 for the
  geometric Asian gives 36x, the stock's 0.96 for Hull's call gave Tier 2's 3.6x.
- Discrete monitoring misses crossings between dates, so a discrete knock-out is worth more and a
  discrete lookback less; the gap shrinks like 1/√n.
- Heston's smile comes from random volatility; ρ < 0 tilts it into the equity skew, and the
  at-the-money vol (18.5%) falls below the 20% average vol.

## Tier 6: Calibration to market data

### Key formulas

**Forward and discount factor from put-call parity**, per expiry, with no r or q assumed: C − P
is a straight line in K, slope −D and intercept D·F.

$$C - P = D\,(F - K) = DF - D\,K$$

**Raw SVI**, total implied variance w = σ²T of one expiry against log moneyness k = log(K/F):

$$w(k) = a + b\left(\rho\,(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right)$$

**Butterfly arbitrage** (Gatheral and Jacquier): the density of k is
g(k)/√(2πw)·exp(−d₂²/2), so the curve is arbitrage-free only if

$$g(k) = \left(1 - \frac{k\,w'}{2w}\right)^2 - \frac{w'^2}{4}\left(\frac{1}{w} + \frac{1}{4}\right) + \frac{w''}{2} \ge 0$$

**Calendar arbitrage:** w(k, T) must not decrease in T at any k. **Lee's wing bound:**
b(1 + |ρ|) ≤ 2.

**Calibration residual** per quote: price error over vega ≈ vol error, weighted by the quote's
bid-ask spread in vol.

$$r_i = \frac{C^{\text{Heston}}_i - C^{\text{mid}}_i}{\text{vega}_i}\cdot\frac{1}{\sigma^{\text{ask}}_i - \sigma^{\text{bid}}_i}$$

### Gotchas

- The index level a broker returns after the close isn't synchronous with the option quotes
  (16:00 vs. 16:15 for SPX): with it the 1-week dividend yield came out at −10%. Read the forward
  from parity.
- `ib_async`'s `connect()` requests positions on every connection, whatever `fetchFields` says.
  For market data only, use the bare handshake (`ib.client.connect`).
- Snapshot and regulatory snapshot market data requests are billed per request; streaming
  delayed data is free.
- "Zero volume" is a bad liquidity filter for index options: 57% of good real quotes. A missing
  open interest is unknown, not zero.
- A one-pass outlier rule on a regression drops good points with the bad one, because the bad
  point tilts the first fit. Remove the worst, refit, repeat.
- Check SVI arbitrage where the data is. Far outside it a slice can bend anywhere (g = −86 at
  k = −0.35 on a 1-week slice), and penalties there distort every fit.
- SVI's parameters trade off on a narrow strike range (ρ from −0.4 to −0.9 for the same curve
  within noise): trust the curve, not ρ alone.
- A fixed quadrature grid fails when e^(−iu·log(K/F)) turns many times before φ decays (short
  maturities, far strikes): let the node count follow U·|log(K/F)|.

### Takeaways

- Calibration runs the pricing problem backwards: from prices to the parameters that explain
  them, and it is only as good as the quotes that go in.
- Put-call parity gives the forward and the discount factor for free, model-free, from the
  options themselves.
- SVI fits each expiry almost perfectly (0.003 to 0.39 vol points rms on the real SPX snapshot)
  because each expiry gets its own 5 numbers; the arbitrage checks are what keep it honest.
- Heston, with one set of 5 numbers for all expiries, fits the long end to 0.07 vol points and
  misses the 1-week skew by up to 7: a diffusion's skew levels off at short maturities, the
  market's doesn't. That is the case for jumps.
- Validate on synthetic data with known answers first: the chain with planted bad quotes exposed
  the parity regression dropping good pairs with the bad one, which the real snapshot (0
  outliers) could not show.
