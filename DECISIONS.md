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
