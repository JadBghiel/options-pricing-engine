"""cli and plot smoke tests"""

import sys
from pathlib import Path

import pytest

from ope.cli import main


def test_price_prints_hull_values(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["price", "-S", "42", "-K", "40", "-T", "0.5", "-r", "0.1", "--sigma", "0.2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "4.759422" in out
    assert "0.779131" in out


def test_iv_round_trips_hull(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["iv", "--type", "put", "-S", "42", "-K", "40", "-T", "0.5", "-r", "0.1"]
    code = main([*argv, "--price", "0.8085993729"])
    assert code == 0
    assert "0.200000" in capsys.readouterr().out


def test_bad_input_is_a_message_not_a_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["iv", "-S", "42", "-K", "40", "-T", "0.5", "-r", "0.1", "--price", "100"])
    assert code == 2
    assert "no arbitrage bounds" in capsys.readouterr().err


def test_plot_greeks_writes_a_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "greeks.png"
    argv = ["plot-greeks", "-K", "100", "-T", "0.5", "-r", "0.05", "--sigma", "0.2"]
    assert main([*argv, "--out", str(out)]) == 0
    assert out.stat().st_size > 10_000


def test_mc_prints_the_interval_and_the_closed_form(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["mc", "-S", "42", "-K", "40", "-T", "0.5", "-r", "0.1", "--sigma", "0.2"]
    code = main([*argv, "--paths", "20000", "--antithetic", "--control"])
    out = capsys.readouterr().out
    assert code == 0
    assert "95% CI" in out
    assert "4.759422" in out


def test_plot_mc_writes_a_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "mc.png"
    argv = ["plot-mc", "-S", "42", "-K", "40", "-T", "0.5", "-r", "0.1", "--sigma", "0.2"]
    assert main([*argv, "--out", str(out)]) == 0
    assert out.stat().st_size > 10_000


def test_plot_mc_without_matplotlib(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "ope.plots", None)
    argv = ["plot-mc", "-S", "42", "-K", "40", "-T", "0.5", "-r", "0.1", "--sigma", "0.2"]
    assert main(argv) == 1
    assert "matplotlib" in capsys.readouterr().err


def test_american_prints_tree_pde_and_premium(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["american", "--type", "put", "-S", "50", "-K", "50", "-T", "0.4167", "-r", "0.1"]
    assert main([*argv, "--sigma", "0.4", "--steps", "200", "--grid", "100"]) == 0
    out = capsys.readouterr().out
    assert "early exercise premium" in out
    assert "4.076" in out  # the european put from tier 1


def test_plot_boundary_writes_a_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "boundary.png"
    argv = ["plot-boundary", "--type", "put", "-S", "50", "-K", "50", "-T", "0.42", "-r", "0.1"]
    assert main([*argv, "--sigma", "0.4", "--out", str(out)]) == 0
    assert out.stat().st_size > 10_000


def test_plot_boundary_without_matplotlib(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "ope.plots", None)
    argv = ["plot-boundary", "-S", "50", "-K", "50", "-T", "0.42", "-r", "0.1", "--sigma", "0.4"]
    assert main(argv) == 1
    assert "matplotlib" in capsys.readouterr().err


def test_plot_greeks_without_matplotlib(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "ope.plots", None)  # makes the import fail
    argv = ["plot-greeks", "-K", "100", "-T", "0.5", "-r", "0.05", "--sigma", "0.2"]
    assert main(argv) == 1
    assert "matplotlib" in capsys.readouterr().err


def test_tree_convergence_plot() -> None:
    pytest.importorskip("matplotlib")
    from ope.plots import plot_tree_convergence

    fig = plot_tree_convergence("call", 42, 40, 0.5, 0.10, 0.20, max_steps=40)
    assert len(fig.axes) == 1


def test_cpp_flag_gives_the_same_prices(capsys: pytest.CaptureFixture[str]) -> None:
    hull = ["-S", "42", "-K", "40", "-T", "0.5", "-r", "0.1", "--sigma", "0.2"]
    assert main(["price", *hull, "--cpp"]) == 0
    assert "4.759422" in capsys.readouterr().out
    assert main(["mc", *hull, "--paths", "20000", "--cpp", "--threads", "2"]) == 0
    out = capsys.readouterr().out
    assert "c++" in out
    assert "4.759422" in out
    put = ["--type", "put", "-S", "50", "-K", "50", "-T", "0.4167", "-r", "0.1", "--sigma", "0.4"]
    assert main(["american", *put, "--steps", "200", "--grid", "50"]) == 0
    numpy_line = capsys.readouterr().out.splitlines()[0]
    assert main(["american", *put, "--steps", "200", "--grid", "50", "--cpp"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == numpy_line  # same tree price to 6 digits


ATM = ["-S", "100", "-K", "100", "-T", "1", "-r", "0.05"]


def test_asian_prints_the_cv_estimate_and_the_geometric_price(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["asian", *ATM, "--sigma", "0.2", "--paths", "20000"]) == 0
    out = capsys.readouterr().out
    assert "5.940200" in out  # geometric closed form, 12 dates
    assert "smaller std error" in out


def test_barrier_prints_discrete_bgk_and_continuous(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["barrier", *ATM, "--sigma", "0.2", "-H", "120", "--kind", "up-and-out"]
    assert main([*argv, "--paths", "20000"]) == 0
    out = capsys.readouterr().out
    assert "1.176065" in out  # continuous closed form
    assert "1.332070" in out  # bgk shift for 252 dates
    # a barrier already crossed is a message, not a traceback
    assert main(["barrier", *ATM, "--sigma", "0.2", "-H", "90", "--kind", "up-and-out"]) == 2
    assert "already at or past" in capsys.readouterr().err


def test_lookback_prints_the_closed_form(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["lookback", "-S", "100", "-T", "1", "-r", "0.05", "--sigma", "0.2"]
    assert main([*argv, "--paths", "5000", "--dates", "52"]) == 0
    assert "17.216802" in capsys.readouterr().out


def test_heston_prints_semi_analytical_mc_and_implied_vol(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["heston", *ATM, "--paths", "5000", "--steps", "20"]) == 0
    out = capsys.readouterr().out
    assert "9.871330" in out
    assert "0.184528" in out  # below the 20% average vol: the skew's atm dip
    assert "violated" in out


def test_tier5_plots_write_pngs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    smile = tmp_path / "smile.png"
    argv = ["plot-smile", "-S", "100", "-T", "1", "-r", "0.05", "--out", str(smile)]
    assert main(argv) == 0
    assert smile.stat().st_size > 10_000
    barrier = tmp_path / "barrier.png"
    argv = ["plot-barrier", *ATM, "--sigma", "0.2", "-H", "120", "--kind", "up-and-out"]
    assert main([*argv, "--out", str(barrier)]) == 0
    assert barrier.stat().st_size > 10_000


@pytest.mark.parametrize(
    "argv",
    [
        ["plot-smile", "-S", "100", "-T", "1", "-r", "0.05"],
        ["plot-barrier", *ATM, "--sigma", "0.2", "-H", "120", "--kind", "up-and-out"],
    ],
)
def test_tier5_plots_without_matplotlib(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    monkeypatch.setitem(sys.modules, "ope.plots", None)
    assert main(argv) == 1
    assert "matplotlib" in capsys.readouterr().err


FIXTURE = str(Path(__file__).parent / "fixtures" / "spx_synthetic.csv")


def test_surface_prints_the_report_and_the_forwards(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["surface", FIXTURE]) == 0
    out = capsys.readouterr().out
    assert "synthetic snapshot of SPX" in out
    assert "crossed or locked" in out and "kept" in out
    assert "20270108" in out  # the 1 year forward row


def test_svi_prints_the_fits_and_the_calendar(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["svi", FIXTURE]) == 0
    out = capsys.readouterr().out
    assert out.count("calendar") == 6 and "ARBITRAGE" not in out


def test_calibrate_prints_params_feller_and_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["calibrate", FIXTURE, "--starts", "2"]) == 0
    out = capsys.readouterr().out
    assert "best: v0 0.0200" in out
    assert "violated" in out and "all quotes: rms" in out


def test_tier6_plots_write_pngs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    for cmd in ("plot-surface", "plot-term"):
        out = tmp_path / f"{cmd}.png"
        extra = ["--starts", "1"] if cmd == "plot-surface" else []
        assert main([cmd, FIXTURE, *extra, "--out", str(out)]) == 0
        assert out.stat().st_size > 10_000


@pytest.mark.parametrize("cmd", ["plot-surface", "plot-term"])
def test_tier6_plots_without_matplotlib(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cmd: str
) -> None:
    monkeypatch.setitem(sys.modules, "ope.plots", None)
    assert main([cmd, FIXTURE]) == 1
    assert "matplotlib" in capsys.readouterr().err


def test_fetch_chain_passes_the_connection_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import ope.market
    from ope.chain import load_snapshot

    seen: dict[str, object] = {}

    def fake_pull(source: str, **kw: object) -> object:
        seen.update(source=source, **kw)
        return load_snapshot(FIXTURE)

    monkeypatch.setattr(ope.market, "pull_chain", fake_pull)
    argv = ["fetch-chain", "--source", "ibkr", "--port", "7497", "--out-dir", str(tmp_path)]
    assert main(argv) == 0
    assert seen["source"] == "ibkr" and seen["port"] == 7497 and seen["host"] is None
    assert "420 quotes, 7 expiries" in capsys.readouterr().out
