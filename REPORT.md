# Report: pricing European and American options four ways

A technical write-up of the core engine (Tiers 1 to 3): what each method does, how it was
validated, how it converges, and where it stops being trustworthy. Updated after Tiers 4 and 6.

All numbers come from the code in this repository and can be reproduced with the commands and
notebooks referenced below. The reference case throughout is Hull's call, S = 42, K = 40,
T = 0.5, r = 10%, σ = 20%, worth 4.7594 in closed form.

## 1. Methods

| Method | Module | Prices | Idea |
|---|---|---|---|
| Closed form | `black_scholes.py` | European | Black-Scholes-Merton with a dividend yield, vectorized over NumPy arrays |
| Monte Carlo | `monte_carlo.py` | European (any simulable payoff later) | Exact one-step simulation of S_T, discounted average payoff |
| Binomial tree | `trees.py` | European and American | Cox-Ross-Rubinstein tree, backward induction with an early-exercise max |
| Finite differences | `pde.py` | European and American | Black-Scholes PDE in log price, Crank-Nicolson with a Rannacher start, PSOR for early exercise |

**Closed form.** Prices and the five Greeks come from one shared computation of d1, d2 and the
discount factors. Degenerate inputs (T = 0, σ = 0, S = 0) send d1 and d2 to ±∞, so the same
formulas return the intrinsic value on the forward with no special case. Implied volatility is a
safeguarded Newton method: Newton steps on vega inside a bracket that always contains the root,
with a bisection step whenever Newton would leave it, started at the point where vega peaks.

**Monte Carlo.** Under Black-Scholes, log S_T is exactly normal, so each path is one draw:
S_T = S·exp((r − q − σ²/2)T + σ√T·Z). The estimator reports a standard error and a 95% interval.
Two variance-reduction techniques are available together: antithetic pairs (Z, −Z), with the
standard error computed over pair averages, and a control variate on the discounted stock, whose
mean S·e^(−qT) is known, with the regression slope β. Delta is estimated three ways: bump and
reprice with common random numbers, pathwise, and likelihood ratio.

**Binomial tree.** u = e^(σ√Δt), d = 1/u, risk-neutral probability p = (e^((r−q)Δt) − d)/(u − d).
The backward step is vectorized per time slice; an American option takes the maximum of the
continuation value and the exercise value at every node.

**Finite differences.** In x = ln S the PDE has constant coefficients. The grid spans six standard
deviations around the spot and the strike, and is shifted so the strike sits exactly on a node.
Time steps are Crank-Nicolson, except the first two, each replaced by two fully implicit half
steps (Rannacher) to damp the oscillations caused by the payoff's kink. European steps solve a
tridiagonal system directly; American steps solve the linear complementarity problem
V ≥ payoff by projected successive over-relaxation (PSOR, ω = 1.2). The early-exercise boundary
is read off the grid at every step.

## 2. Validation

Every reference value is recomputed by an independent script, never copied from a book.

| Check | Result |
|---|---|
| Hull's call and put (closed form) | 4.7594 and 0.8086, to 1e-9 |
| Hull's Greeks example (S=49, K=50, T=20/52) | Δ 0.5216, Γ 0.0655, vega 12.105, Θ −4.305/yr, ρ 8.907, to 1e-8 |
| Put-call parity | 1e-10 on 500 random parameter sets |
| Greeks vs. central finite differences | 1e-4 relative on 200 random sets, calls and puts |
| Implied vol round trip | σ recovered to 1e-6 wherever vega > 1e-3; 20,000 random options, no failure |
| Monte Carlo 95% interval coverage | 93% to 96% over 400 seeds, for every variance-reduction setting |
| Monte Carlo standard error calibration | reported SE within 15% of the spread of estimates over 300 seeds |
| Monte Carlo deltas | every method within 4 standard errors of the analytical delta |
| European tree (2000/2001 steps averaged) | within 1e-4 of the closed form |
| European PDE (400 × 400) | within 5e-5 relative (2e-4 absolute floor) of the closed form |
| American put, tree vs. PDE (300 × 300) | agree to 0.05% of the price on four cases (largest gap 0.034%) |
| American call without dividends | equals the European call exactly on the tree, to 1e-6 on the PDE |
| Hull's 5-step American put | 4.4885, Hull's 4.49 |

112 tests, 99% branch coverage, run with lint, strict type checks and the three notebooks on every
push.

## 3. Convergence

**Monte Carlo** error falls like 1/√N: the standard error on a log-log plot follows a slope of
−1/2. Variance reduction lowers the line without changing its slope. At 200,000 paths on Hull's
call:

| Monte Carlo variant | Standard error | Cut vs. plain | Equivalent paths |
|---|---|---|---|
| Plain | 0.0111 | 1× | 1× |
| Antithetic | 0.0055 | 2.0× | 4× |
| Control variate | 0.0031 | 3.6× | 13× |
| Both | 0.0009 | 12.3× | 150× |

Delta: bump with common random numbers and pathwise both reach a standard error of 0.0011;
likelihood ratio 0.0037. Scrambled Sobol points (notebook only) reached an error of 0.00004 at
65,536 points against 0.019 for pseudo-random numbers.

![Monte Carlo convergence](docs/img/tier2_mc_convergence.png)

**The binomial tree** converges like 1/n but zigzags between odd and even step counts, because
the strike falls at a different position between nodes. Averaging n and n + 1 steps removes most
of it: at 1000/1001 steps the European call is 4.759453 against 4.759422.

![CRR odd/even convergence](docs/img/tier3_tree_convergence.png)

**The PDE** converges at second order in practice: on the European put, the error is 3.2e-3 at
50 × 50 and 1.4e-5 at 800 × 800. The drops per doubling are irregular because the spot falls at a
different place between grid nodes; on average they are close to 4×. Without the Rannacher start,
gamma read off a 25-step grid at the strike is 5.1 instead of 0.039; with it, 0.0393.

![Early exercise boundary](docs/img/tier3_exercise_boundary.png)

## 4. Limits

- **The model.** Everything assumes constant volatility, no jumps and frictionless continuous
  hedging. Real option prices show a volatility smile; Tier 5 (Heston) and Tier 6 (calibration)
  address it.
- **Monte Carlo is scalar and European.** One option per call, and no early exercise: pricing
  American options by simulation needs regression methods (Longstaff-Schwartz), not implemented.
- **Implied volatility is scalar** and, far out of the money and close to expiry, Newton barely
  beats bisection; Newton on log price would fix it if chain inversion gets slow (Tier 6).
- **PSOR is pure Python.** An American PDE price takes about 1.3 to 1.5 s on a 400 × 400 grid. Tier 4
  ports the hot paths to C++.
- **The exercise boundary is grid-resolution limited:** it moves in steps of one grid node, so
  it looks like a staircase unless the grid is fine.
- **Discrete dividends** are not modelled, only a continuous yield q.

## 5. Reproduce

```bash
uv sync --all-extras
uv run pytest                                   # 112 tests
uv run pytest --nbmake notebooks --no-cov       # the three walkthrough notebooks
uv run ope mc -S 42 -K 40 -T 0.5 -r 0.1 --sigma 0.2 --antithetic --control
uv run ope american --type put -S 50 -K 50 -T 0.4167 -r 0.1 --sigma 0.4
```
