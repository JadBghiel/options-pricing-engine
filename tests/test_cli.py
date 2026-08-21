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
