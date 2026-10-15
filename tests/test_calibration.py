"""the shared grid heston pricer vs heston_price, and the calibration recovering known params"""

import math
from pathlib import Path

import numpy as np
import pytest

import ope.calibration as calibration
from ope.calibration import (
    calibrate_heston,
    heston_grid_prices,
    heston_probabilities,
    slice_errors,
)
from ope.chain import load_snapshot
from ope.heston import HestonParams, heston_price
from ope.surface import iv_surface
from ope.synthetic import SPX_LIKE, synthetic_chain

FIXTURE = Path(__file__).parent / "fixtures" / "spx_synthetic.csv"
CASES = [
    HestonParams(0.04, 1.5, 0.04, 0.6, -0.7),  # tier 5's example
    SPX_LIKE,
    HestonParams(0.01, 5.0, 0.06, 0.9, -0.74),  # low v0, fast pull: slow cf decay
    HestonParams(0.09, 0.5, 0.02, 0.3, -0.2),
]


@pytest.mark.parametrize("params", CASES)
@pytest.mark.parametrize("T", [7 / 365, 30 / 365, 0.5, 2.0])
def test_grid_matches_heston_price(params: HestonParams, T: float) -> None:
    F, D = 100.0 * math.exp(0.03 * T), math.exp(-0.05 * T)
    K = F * np.exp(np.linspace(-0.6, 0.3, 13))
    for option in ("call", "put"):
        got = heston_grid_prices(option, F, D, T, K, params)
        # heston_price w/ S = D F, r from D, q = 0 has the same forward and discount
        ref = heston_price(option, D * F, K, T, -math.log(D) / T, params, 0.0)
        assert np.max(np.abs(got - ref)) < 1e-8 * F


def test_grid_prices_satisfy_parity() -> None:
    F, D, T = 5000.0, 0.98, 0.5
    K = F * np.exp(np.linspace(-0.5, 0.3, 9))
    calls = heston_grid_prices("call", F, D, T, K, SPX_LIKE)
    puts = heston_grid_prices("put", F, D, T, K, SPX_LIKE)
    assert np.allclose(calls - puts, D * (F - K), atol=1e-9 * F)
    mixed = heston_grid_prices(np.array(["call", "put"]), F, D, T, K[:2], SPX_LIKE)
    assert mixed[0] == calls[0] and mixed[1] == puts[1]


def test_probabilities_take_a_scalar() -> None:
    p1, p2 = heston_probabilities(1.0, 0.5, SPX_LIKE)
    assert p1.shape == (1,) and 0 < p2[0] < p1[0] < 1


def test_cutoff_falls_back_when_phi_never_decays(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(calibration, "CF_TOL", 0.0)
    assert calibration._cutoff(0.5, SPX_LIKE) == 1e5


@pytest.fixture(scope="module")
def fixture_fit() -> calibration.HestonCalibration:
    return calibrate_heston(iv_surface(load_snapshot(FIXTURE)))


def test_the_fixture_gives_back_its_params(fixture_fit: calibration.HestonCalibration) -> None:
    p = fixture_fit.params
    assert p.v0 == pytest.approx(SPX_LIKE.v0, rel=0.02)
    assert p.kappa == pytest.approx(SPX_LIKE.kappa, rel=0.05)
    assert p.theta == pytest.approx(SPX_LIKE.theta, rel=0.03)
    assert p.xi == pytest.approx(SPX_LIKE.xi, rel=0.03)
    assert p.rho == pytest.approx(SPX_LIKE.rho, abs=0.01)
    assert not fixture_fit.feller  # 2 kappa theta = 0.175 < xi^2 = 0.64, like real index fits
    assert fixture_fit.rms < 0.15
    assert all(e.within_spread == 1.0 for e in fixture_fit.errors)


def test_every_start_is_reported_and_the_best_kept(
    fixture_fit: calibration.HestonCalibration,
) -> None:
    assert len(fixture_fit.starts) == 8
    best = min(fixture_fit.starts, key=lambda s: s.cost)
    assert best.params == fixture_fit.params
    assert len({s.start for s in fixture_fit.starts}) == 8  # sobol points, all different


def test_same_snapshot_same_numbers(fixture_fit: calibration.HestonCalibration) -> None:
    again = calibrate_heston(iv_surface(load_snapshot(FIXTURE)))
    assert again.params == fixture_fit.params
    assert [e.rms for e in again.errors] == [e.rms for e in fixture_fit.errors]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_recovery_over_noise_seeds(seed: int) -> None:
    cal = calibrate_heston(iv_surface(synthetic_chain(seed=seed)), n_starts=2)
    p = cal.params
    assert p.v0 == pytest.approx(SPX_LIKE.v0, rel=0.03)
    assert p.kappa == pytest.approx(SPX_LIKE.kappa, rel=0.1)
    assert p.theta == pytest.approx(SPX_LIKE.theta, rel=0.05)
    assert p.xi == pytest.approx(SPX_LIKE.xi, rel=0.05)
    assert p.rho == pytest.approx(SPX_LIKE.rho, abs=0.02)


def test_the_true_params_sit_inside_every_quote() -> None:
    # the chain quotes bid and ask around the true price -> at the true params every model vol
    # is between the bid and ask vols ("error ~ 0" would fail: a 0.05 tick on a cheap 1 week
    # wing option is worth 0.16 vol points)
    errors = slice_errors(iv_surface(synthetic_chain(junk=False, seed=5)), SPX_LIKE)
    assert all(e.within_spread == 1.0 for e in errors)
