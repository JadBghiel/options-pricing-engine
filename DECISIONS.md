# Decisions

Why each significant choice was made, what else was considered, and what was ruled out. One
entry per tier, added when the tier closes. The long version for each entry is in
[EXPLANATION.md](EXPLANATION.md).

---

## T0. Engineering setup

**Chosen:**
- Python 3.12, managed with `uv`, with `uv.lock` committed
- `src/ope/` layout, built with `hatchling`, version stored once in `ope.__version__`
- `ruff` for lint and format, `mypy` in strict mode on the library, `pytest` with coverage
- GitHub Actions on every push (lint, format check, types, tests) and Codecov for the coverage badge

**Why:**
- **A recruiter reads the setup first.** Strict types and a green CI badge say the code is checked
  before any pricing maths is read.
- **`uv` makes a fresh clone reproducible.** The lockfile pins every version, so CI and my laptop
  run the same environment.
- **`src/` layout stops accidental imports.** Tests run against the installed package, not the
  folder next to them, so a packaging mistake fails in CI and not in someone else's clone.
- **`hatchling` is easy to swap.** Tier 4 replaces the `[build-system]` block with
  scikit-build-core and the rest of `pyproject.toml` stays.

**Considered and ruled out:**
- *`poetry`*: its own lockfile and backend make the Tier 4 switch to scikit-build-core harder.
- *`pip` + `venv`*: works, but no lockfile, so CI could drift from my machine.
- *Python 3.11 or 3.13*: 3.12 is current and stable with wheels for NumPy and SciPy. 3.13 can lag
  on scientific wheels, and 3.11 adds nothing this project needs.
- *A coverage badge generated in CI*: more moving parts to explain than Codecov, for the same
  result.

**Known limits:**
- Codecov needs a `CODECOV_TOKEN` secret in the repo. Until it is set, the upload step doesn't
  fail the build, and the badge shows no value.
- `ruff` ignores the naming rules for `S`, `K`, `T` and `N(x)`, since they are the standard
  notation in every pricing text.

## T1. Black-Scholes closed form

**Chosen:**
- one module of vectorized functions (`price`, `greeks`, `d1_d2`, `price_bounds`) that take
  floats or NumPy arrays and return a float for scalar inputs
- a dividend yield `q` in every formula from the start
- Hull's units: vega and rho per 1.00, theta per year
- degenerate inputs (T = 0, σ = 0, S = 0) handled by sending d1 and d2 to ±∞, so the normal
  formulas return the deterministic value with no special branch
- implied vol by a safeguarded Newton: a bracket always holds the root, and a Newton step that
  would leave it becomes a bisection step

**Why:**
- **Arrays from day one.** Tier 6 prices whole option chains and Tier 7 whole portfolios; one
  call on an array replaces a Python loop over strikes. Tier 4 benchmarks the gap.
- **`q` now, not later.** Adding it after the Greeks exist would touch every formula and test.
- **Hull's units** let the tests use Hull's numbers directly, so any gap is a real bug.
- **The bracket replaces a vega threshold.** A threshold needs tuning per market; leaving the
  bracket is the exact signal that Newton is about to go wrong.

**Considered and ruled out:**
- *Classes (`EuropeanOption(...).price()`)*: more code for no gain, and functions map 1:1 to the
  formulas.
- *Pure Newton*: diverges on deep OTM short-dated options (vega underflows to 0 at step 2).
- *Starting Newton at the bracket end*: monotone, but about 5 iterations slower on deep OTM.
- *SciPy's `brentq` for implied vol*: robust, but it hides the method this tier is meant to
  teach, and it does not use vega, which we already compute.

**Notebooks (added 2026-10-24, for every tier):** the library stays in `.py` files, and each tier
gets a notebook in `notebooks/` that imports it: formulas, plots, experiments and an exercise.
It is committed with its outputs so GitHub shows the plots, and CI runs it top to bottom.
Ruled out: library code in notebooks (not importable or type-checked) and jupytext pairing (no
plots on GitHub).

**Validated:** Hull's price and Greeks examples, put-call parity on 500 random sets, all Greeks
against finite differences, implied vol round trips, bounds, monotonicity and edge cases. 45
tests, 99% branch coverage.

## T2. Monte Carlo pricer

**Chosen:**
- exact one-step simulation of the stock at expiry (`gbm_terminal`), no time stepping
- a random generator or seed passed in explicitly to every function, so runs are reproducible
- an `MCResult` (estimate, standard error, paths, 95% interval) for prices and deltas alike
- antithetic variates and a control variate on the discounted stock, usable together
- delta three ways: bump with common random numbers, pathwise, likelihood ratio
- validation by coverage over many seeds, not by a single run

**Why:**
- **Exact simulation.** Under Black-Scholes the price at expiry has a known distribution, so one
  draw per path has no discretization error. Time stepping only matters for path-dependent
  payoffs (Tier 5).
- **Explicit randomness.** A hidden global seed makes tests flaky and results irreproducible;
  passing the generator makes every number in the docs reproducible.
- **Coverage, not one seed.** A correct 95% interval misses 1 run in 20. Testing one seed either
  passes by luck or fails by luck; testing 200 seeds checks the claim itself.
- **Three deltas.** Pathwise is precise but needs a payoff with a slope; likelihood ratio works
  for any payoff, at a cost in noise; bumping is the generic fallback. Tier 5 needs all three.

**Considered and ruled out:**
- *Time-stepped (Euler) simulation for vanillas*: adds error and cost for nothing here.
- *NumPy's global `np.random.seed`*: shared hidden state, breaks reproducibility across calls.
- *Sobol points in the library*: kept to the notebook for now, since they need randomized
  replicas to give an error bar; worth adding if Tier 5 needs the speed.

**Validated:** coverage 93 to 96% over 400 seeds for every method, reported standard errors
within 15% of the real spread, variance reduction 2.0x / 3.6x / 12.3x at equal paths, every
delta method within its standard error. 80 tests, 99% branch coverage.

## T3. American options: tree and PDE

**Chosen:**
- a Cox-Ross-Rubinstein tree, backward induction vectorized per time slice, with the backward
  step exposed as its own function
- a Crank-Nicolson PDE solver in log price, with the strike on a grid node, two Rannacher steps,
  a tridiagonal solve for European options and PSOR for American ones
- the early-exercise boundary read off the PDE grid
- validation of American prices by agreement between the two methods (0.05%)

**Why:**
- **Two independent methods.** There is no closed form for an American put, so the best evidence
  is two unrelated methods landing on the same number.
- **Log price.** Constant coefficients and a grid that covers the lognormal spread evenly.
- **Strike on a node and Rannacher steps.** Both deal with the payoff's kink: without them the
  error converges erratically and gamma near the strike is meaningless.
- **PSOR.** The standard way to impose V ≥ payoff on an implicit scheme, and it's what PLAN.md
  asks for.

**Considered and ruled out:**
- *Fully implicit time stepping everywhere*: stable but only first order in time.
- *Brennan-Schwartz* for the American step: exact and faster for a put, but it relies on the
  exercise region being on one side, and PSOR is the general method the tier is about.
- *Longstaff-Schwartz Monte Carlo for American options*: a different family (regression), kept for
  later if needed.

**Validated:** Hull's 5-step put 4.4885; European tree within 1e-4 and PDE within 5e-5 relative of
Tier 1; American puts tree vs. PDE within 0.05%; American call = European call without dividends.
112 tests, 99% branch coverage. Write-up in REPORT.md.

## T4. C++ core

**Chosen:**
- Black-Scholes, Monte Carlo and the CRR tree in C++20, as a static library behind a thin
  pybind11 module (`ope._core`), wrapped by `ope.cpp` with the same signatures as the Python side
- scikit-build-core instead of hatchling, so one `pyproject.toml` builds the Python package and the
  C++ extension (Release, no `-ffast-math`, no fused multiply-add)
- Philox 4x64-10 with Box-Muller for the C++ normals: normal number i is a pure function of the
  seed and i
- Monte Carlo split into fixed chunks of 16,384 draws, summed on their own and merged in order
  with the Chan-Golub-LeVeque formula, on `std::thread`
- one pytest suite run on both implementations (an `impl` fixture), plus direct comparisons on the
  same inputs and the same pre-generated normals
- GoogleTest, Google Benchmark and pytest-benchmark; ASan, UBSan and TSan in CI

**Why:**
- **Same signatures.** The existing Tier 1 to 3 tests become the C++ tests for free, and a user
  switches by changing `ope.` to `ope.cpp.`.
- **Counter-based random numbers.** No generator state to share or split between threads: a
  thread starting at path 1,000,000 computes its block directly. Python can ask for the exact draws
  the C++ run used and replay it.
- **Fixed chunks, merged in order.** Floating-point addition isn't associative, so this is what
  makes a seed give the same bits on 1 thread or 8.
- **Same normals for both sides.** C++ can't reproduce NumPy's stream, so the exact check feeds
  both the same array; the statistical checks (coverage, standard errors) run on each side's own
  generator.

**Considered and ruled out:**
- *OpenMP*: shorter, but `std::thread` needs no extra runtime and keeps the work split explicit.
- *mt19937 seeded per thread*: streams that are probably independent, not provably, and results
  that change with the thread count.
- *Ziggurat for the normals*: about 2x faster than Box-Muller (it's why NumPy wins on 1 thread),
  but 100+ lines of tables. Kept as the next speed-up.
- *`-ffast-math` and `-march=native`*: faster, but they break the infinities the degenerate
  Black-Scholes cases rely on, and make results machine-dependent.
- *Bit-identical results to Python*: impossible with different `exp` and `N(x)` implementations;
  compared within 1e-11 relative instead.

**Validated:** prices and Greeks within 1e-11 relative of Python on 100,000 random options (largest
gap 1.1e-13), the tree to 1e-13, Monte Carlo on the same normals to 1e-11; Philox bit for bit
against NumPy and against the Random123 known answers; same bits on 1 to 8 threads. 196 Python
tests (99% coverage) and 20 GoogleTest tests, clean under ASan, UBSan and TSan. Benchmarks: tree
30x faster than NumPy, Black-Scholes and Monte Carlo about 3x with 8 threads (README).

## T5. Exotics and Heston

**Chosen:**
- exotics on GBM simulated date by date with the exact log step, so the only "steps" are the
  contract's monitoring dates
- the geometric Asian (closed form) as control variate for the arithmetic Asian
- barrier and lookback priced both ways: checked on dates (the real contract) and watched
  continuously (closed forms, and Monte Carlo through the Brownian bridge); the BGK shift links
  the two
- Heston by full-truncation Euler (log-Euler for the stock), and semi-analytically through the
  characteristic function in the "Little Heston Trap" form, P1 and P2 integrated to infinity
- the smile from Heston prices inverted with the Tier 1 implied vol solver, on the
  out-of-the-money side
- Python and NumPy only, no C++ port this tier

**Why:**
- **Exact steps for GBM.** The gap between a daily and a continuous barrier is then a property of
  the contract, never a discretization error. Heston has no exact step, so there the step count is
  a numerical choice and its bias is measured as such.
- **The geometric control variate.** Its payoffs have a correlation of 0.9996 with the arithmetic
  ones: 36x smaller standard error at equal paths, against 3.6x for the Tier 2 stock control.
- **Bridge instead of fine steps.** Multiplying by the exact survival probability gives the
  continuously watched price from any number of dates, with no extra noise, so the continuous
  closed forms can be tested against a simulation of the same contract.
- **Full truncation.** Among the simple fixes for a negative Euler variance it has the smallest
  bias (Lord, Koekkoek and van Dijk), and it keeps the discounted stock an exact martingale.
- **The trap form.** Heston's original formula gives a wrong price at long maturities (16.39
  instead of 29.09 at 5 years) because its complex log crosses a branch cut; the 2007 form is the
  same function without the jump, and it can be integrated to infinity without overflow.
- **P1 and P2.** The price keeps the Black-Scholes shape S·P1 − K·e^(−rT)·P2, which makes it
  explainable; Lewis's single integral is kept in the tests as the independent check.
- **No C++.** Tier 6 calibrates with the semi-analytical price, not the Monte Carlo, so Heston paths
  aren't a hot path yet; a 1,000,000-path, 64-step run takes 4 to 6 s in NumPy on this laptop.

**Considered and ruled out:**
- *Reflection (|v|) or partial truncation* for the variance: more bias than full truncation.
- *The QE scheme (Andersen 2008)*: less bias per step, but far more code to explain; the bias is
  measured and controlled with steps instead.
- *Heston's original characteristic function*: kept only to show the jump.
- *Carr-Madan FFT* for the semi-analytical price: prices a whole strike grid at once, worth it for
  calibration speed in Tier 6, not needed to validate one price.
- *Checking discretely monitored prices against the continuous formulas*: wrong reference, off by
  12.5% (31 standard errors) for the daily up-and-out call.
- *Fixed-strike lookbacks and knock-in barriers*: knock-ins follow from in-out parity (vanilla =
  in + out), fixed strikes add nothing new to the lesson.

**Validated:** every closed form against a simulation of the same contract, with 95% intervals
holding it 92 to 96% of the time over 200 seeds; Heston's price against Lewis's integral to 1e-7
from 1 month to 10 years; Heston → Black-Scholes linearly as ξ → 0 (3e-5 at ξ = 1e-4); the Monte
Carlo's 4-step bias over 10 standard errors and its 64-step coverage at 93%. 269 tests, 99%
branch coverage.

## T6. Calibration to market data

**Chosen:**
- SPXW (S&P 500 index options, European, cash-settled at the 4pm close on every expiry): put-call
  parity and the European Heston formula apply exactly
- a read-only IBKR fetcher on `ib_async` that only does the API handshake, streams delayed data
  in batches of 50 and writes every batch to `data/` before reading it; yfinance as fallback;
  both in an optional `market` extra
- no real quotes in git: a synthetic SPX-like chain, priced by the library with known parameters
  and planted bad quotes, is the committed fixture for the tests, the notebook and CI
- the forward and discount factor per expiry from a regression of C − P on K, weighted by
  1 / spread, with bad pairs removed one at a time
- SVI per expiry fitted in vol space (weights 1 / bid-ask spread in vol), Lee's wing bound in
  every fit, butterfly (g ≥ 0) and calendar checks on the quoted range, a penalty only when the
  plain fit breaks one
- Heston calibrated to all expiries at once, on one shared quadrature grid per expiry, residuals
  = price error / vega, 8 seeded Sobol starting points, all reported

**Why:**
- **SPXW, not SPY.** SPY options are American, so inverting them with a European pricer is wrong
  near early exercise. Using SPXW only (not the morning-settled monthly SPX) gives every expiry the
  same settlement time.
- **The forward from the options.** On the real snapshot the index level IBKR returned was out of
  sync with the 16:15 option quotes: with it, the 1-week dividend yield came out at −10%. Parity
  needs no spot, no r and no q.
- **The bare handshake.** `ib_async`'s `connect()` requests the account's positions on every
  connection, even with its "fetch nothing" flag (read in its source). `client.connect` does only
  the handshake, and market data works on it the same.
- **Vol space and 1 / spread weights.** A price error means very different things at the money
  and in the wings; an error in vol points divided by the bid-ask width in vol says how far
  outside what the market itself knows the fit is.
- **Checks on the quoted range.** Inside the quoted strikes every plain real fit was already
  arbitrage-free. Penalizing far outside them (where a 1-week fit reached g = −86) bent the fits
  for nothing and pushed later expiries above junk extrapolations.
- **The shared grid.** `heston_price` costs 4 to 9 ms per strike; the grid prices an expiry's
  whole strike set in 0.7 to 1.6 ms (110 to 210 times faster), within 1e-8 × F of `heston_price` (1e-4 vol points), with
  nodes that follow the oscillation of the integrand. No C++ needed: the hot loop is already a
  NumPy matrix product.

**Considered and ruled out:**
- *SPY or single stocks*: American exercise.
- *Assuming r and q* (or a rate curve) with the index level: wrong when the level and the quotes
  are not simultaneous, which is the normal case after the close.
- *Committing a small trimmed real snapshot*: not allowed by the data agreement, whatever the size.
- *Dropping every zero-volume quote*: removed 57% of the real two-sided quotes, many of them tight.
- *Treating a missing open interest as zero*: it arrives after the quote and was missing on half
  the 9-month quotes.
- *One robust pass on the parity line*: a bad pair tilts the first line, and 5 good pairs were
  dropped for 1 bad one; one at a time drops only the bad ones.
- *Checking SVI arbitrage on a wide fixed range*: see above. The wide-range result is still
  reported as a flag.
- *Carr-Madan FFT or COS* for Heston: both fine and fast, but new machinery (a damping parameter,
  a truncation range); the shared grid is the Tier 5 formula with a different integral.
- *A C++ port of the calibration*: the whole real calibration takes 4 to 7 seconds.

**Validated:** on the synthetic chain, every planted bad quote caught by its own filter (checked
by identity), the forward within 1 bp over 10 seeds, SVI exact without noise and within 0.3 vol
points of the true curve with noise, Heston back to its parameters from all 8 starts (κ 2.48 for
2.5, ξ 0.797 for 0.8), the grid pricer within 1e-8 × F of `heston_price`. On the real snapshot:
the fit error per expiry, and the same parameters bit for bit when rerun. 369 tests, 99% branch
coverage.
