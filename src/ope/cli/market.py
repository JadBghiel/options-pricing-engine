"""cli cmds for tier 6: fetching a chain, the surface, svi, the heston calibration"""

import argparse
from pathlib import Path

from ope.calibration import calibrate_heston
from ope.chain import load_snapshot
from ope.cli.args import Command, plots_missing, saved
from ope.surface import Surface, iv_surface
from ope.svi import calendar_checks, fit_svi_surface


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> dict[str, Command]:
    p = sub.add_parser("fetch-chain", help="save an SPXW chain snapshot (IBKR read-only, or Yahoo)")
    p.add_argument("--source", choices=["auto", "ibkr", "yfinance"], default="auto")
    p.add_argument("--host", default=None, help="default $OPE_IB_HOST or 127.0.0.1")
    p.add_argument("--port", type=int, default=None, help="default $OPE_IB_PORT or 4001")
    p.add_argument("--client-id", type=int, default=None, help="default $OPE_IB_CLIENT_ID or 23")
    p.add_argument("--out-dir", default="data/chains")

    for name, help_ in (
        ("surface", "clean a chain snapshot, imply the forwards, count what each filter removed"),
        ("svi", "fit SVI per expiry and check butterfly and calendar arbitrage"),
        ("calibrate", "calibrate Heston to the whole surface, fit error per expiry"),
        ("plot-surface", "plot the smile per expiry with the SVI and Heston fits to a PNG"),
        ("plot-term", "plot the at-the-money vol and skew per expiry to a PNG"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("snapshot", help="a chain snapshot .csv (its .json must sit next to it)")
        if name in ("calibrate", "plot-surface"):
            p.add_argument("--starts", type=int, default=8, help="Heston starting points")
            p.add_argument("--seed", type=int, default=0, help="same seed, same result")
        if name.startswith("plot"):
            p.add_argument("--out", default=f"{name.removeprefix('plot-')}.png")

    return {
        "fetch-chain": _cmd_fetch_chain,
        "surface": _cmd_surface,
        "svi": _cmd_svi,
        "calibrate": _cmd_calibrate,
        "plot-surface": _cmd_plot_surface,
        "plot-term": _cmd_plot_term,
    }


def _cmd_fetch_chain(a: argparse.Namespace) -> int:
    from ope.market import pull_chain

    chain = pull_chain(
        a.source, host=a.host, port=a.port, client_id=a.client_id, out_dir=Path(a.out_dir)
    )
    print(f"{len(chain)} quotes, {len(chain.expiries())} expiries, as of {chain.meta['as_of']}")
    return 0


def _surface(path: str) -> Surface:
    chain = load_snapshot(path)
    meta = chain.meta
    print(f"{meta['source']} snapshot of {meta['symbol']} as of {meta['as_of']}")
    return iv_surface(chain)


def _cmd_surface(a: argparse.Namespace) -> int:
    s = _surface(a.snapshot)
    print("\n".join(s.report.lines()))
    print("\nforward and discount factor per expiry, from put-call parity")
    print(f"{'expiry':>9} {'days':>6} {'F':>10} {'D':>9} {'r':>7} {'pairs':>6}")
    for e, f in s.forwards.items():
        print(f"{e:>9} {f.T * 365:6.1f} {f.F:10.2f} {f.D:9.6f} {f.r * 100:6.2f}% {f.n_pairs:6d}")
    return 0


def _cmd_svi(a: argparse.Namespace) -> int:
    s = _surface(a.snapshot)
    fits = fit_svi_surface(s)
    print(
        f"{'expiry':>9} {'days':>6} {'n':>3} {'rms':>6} {'max':>6} {'in b/a':>7} "
        f"{'min g':>7}  penalty"
    )
    for f in fits:
        print(
            f"{f.expiry:>9} {f.T * 365:6.1f} {f.n:3d} {f.rms:6.3f} {f.max_abs:6.3f} "
            f"{f.within_spread * 100:6.0f}% {f.min_g:7.3f}  {'yes' if f.penalized else 'no'}"
        )
    print("errors in vol points; min g >= 0: no butterfly arbitrage on the quoted range")
    for c in calendar_checks(fits):
        status = "ok" if c.ok else "CALENDAR ARBITRAGE"
        print(f"calendar {c.expiry_1} -> {c.expiry_2}: min gap {c.min_gap:+.6f}  {status}")
    return 0


def _cmd_calibrate(a: argparse.Namespace) -> int:
    s = _surface(a.snapshot)
    cal = calibrate_heston(s, n_starts=a.starts, seed=a.seed)
    p = cal.params
    print(f"{a.starts} starts, {cal.seconds:.1f} s")
    for st in cal.starts:
        q = st.params
        print(
            f"  -> v0 {q.v0:.4f} kappa {q.kappa:.3f} theta {q.theta:.4f} xi {q.xi:.3f} "
            f"rho {q.rho:+.3f}  cost {st.cost:.6g}"
        )
    print(
        f"best: v0 {p.v0:.4f} kappa {p.kappa:.3f} theta {p.theta:.4f} xi {p.xi:.3f} "
        f"rho {p.rho:+.3f}"
    )
    print(
        f"feller 2 kappa theta = {2 * p.kappa * p.theta:.4f} vs xi^2 = {p.xi**2:.4f}: "
        f"{'holds' if cal.feller else 'violated'}"
    )
    print(f"{'expiry':>9} {'days':>6} {'n':>3} {'rms':>6} {'max':>6} {'in b/a':>7}")
    for e in cal.errors:
        print(
            f"{e.expiry:>9} {e.T * 365:6.1f} {e.n:3d} {e.rms:6.3f} {e.max_abs:6.3f} "
            f"{e.within_spread * 100:6.0f}%"
        )
    print(f"all quotes: rms {cal.rms:.3f} vol points")
    return 0


def _cmd_plot_surface(a: argparse.Namespace) -> int:
    try:
        from ope.plots import plot_smiles
    except ImportError:
        return plots_missing()
    s = _surface(a.snapshot)
    cal = calibrate_heston(s, n_starts=a.starts, seed=a.seed)
    return saved(plot_smiles(s, fit_svi_surface(s), cal), a.out)


def _cmd_plot_term(a: argparse.Namespace) -> int:
    try:
        from ope.plots import plot_term_structure
    except ImportError:
        return plots_missing()
    s = _surface(a.snapshot)
    return saved(plot_term_structure(s, fit_svi_surface(s)), a.out)
