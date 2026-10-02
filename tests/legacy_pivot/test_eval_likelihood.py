"""Tests for ``nll_per_shot`` / ``base_measure_nll_per_shot`` against a real ShotCloudProcess.

These tests exercise the full evaluation stack end-to-end: fit a KDE, build
a process, sample observations, compute spatial NLL.
"""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud import CourtGrid, HierarchicalKDE
from shotcloud.legacy import KDEProduct
from shotcloud.legacy_pivot.eval_metrics import (
    base_measure_nll_per_shot,
    kde_gain,
    nll_per_shot,
)
from shotcloud.legacy_pivot.marked_process import ShotCloudProcess
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder
from shotcloud.legacy_pivot.timing import ConstantRateTimingModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_shots(n_per_player: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    for pos, mu_x, mu_y in [("G", 0.0, 24.0), ("F", 0.0, 4.0)]:
        for player_idx in range(2):
            pid = f"{pos}{player_idx}"
            x = rng.normal(mu_x + 2.0 * (player_idx - 0.5), 2.0, n_per_player)
            y = rng.normal(mu_y, 2.0, n_per_player)
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "player_id": np.array([pid] * n_per_player),
                    "position": np.array([pos] * n_per_player),
                }
            )
    out: dict[str, np.ndarray] = {}
    for key in ("x", "y", "player_id", "position"):
        out[key] = np.concatenate([r[key] for r in rows])
    return out


@pytest.fixture
def fitted_kde() -> HierarchicalKDE:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=40, ny=42)
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=200.0, recency_half_life_days=None)
    shots = _synthetic_shots(n_per_player=200)
    kde.fit(
        x=shots["x"],
        y=shots["y"],
        player_id=shots["player_id"],
        position=shots["position"],
    )
    return kde


@pytest.fixture
def process(fitted_kde: HierarchicalKDE) -> ShotCloudProcess:
    base = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.0, "a_0": 0.0})
    decoder = LowRankTiltDecoder(n_cells=fitted_kde.grid.n_cells, rank=4, zero_init=True)
    timing = ConstantRateTimingModel(mean_shots=20.0, game_length=48.0)
    return ShotCloudProcess(timing_model=timing, spatial_decoder=decoder, base_measure=base)


def _sample_observations(
    process: ShotCloudProcess, players: list[str], R: int = 5, seed: int = 0
) -> list[tuple[str, np.ndarray]]:
    """Sample R sequences per player; flatten to a list of (player_id, cells)."""
    rng = np.random.default_rng(seed)
    obs = []
    for pid in players:
        for _ in range(R):
            seq = process.sample(pid, rng=rng)
            if seq.n_shots > 0:
                obs.append((pid, seq.cells))
    return obs


# ---------------------------------------------------------------------------
# NLL invariants
# ---------------------------------------------------------------------------


def test_nll_is_finite_and_non_negative_on_sampled_obs(process: ShotCloudProcess) -> None:
    obs = _sample_observations(process, ["G0", "F0"], R=5)
    n = nll_per_shot(process, obs)
    assert np.isfinite(n)
    assert n >= 0.0  # NLL = -log p, and log p <= 0 since p ∈ [0, 1].


def test_zero_init_decoder_nll_equals_base_measure_nll(process: ShotCloudProcess) -> None:
    """At zero-init, p_θ = q_0, so the two NLLs must agree."""
    obs = _sample_observations(process, ["G0", "F0", "G1", "F1"], R=10)
    nll_model = nll_per_shot(process, obs)
    nll_base = base_measure_nll_per_shot(process.base_measure, obs)
    np.testing.assert_allclose(nll_model, nll_base, atol=1e-5)


def test_kde_gain_is_zero_at_zero_init(process: ShotCloudProcess) -> None:
    """Δ_KDE = base − model; at zero-init they are equal so gain = 0."""
    obs = _sample_observations(process, ["G0", "F0"], R=10)
    nll_model = nll_per_shot(process, obs)
    nll_base = base_measure_nll_per_shot(process.base_measure, obs)
    assert abs(kde_gain(nll_base, nll_model)) < 1e-5


def test_random_init_decoder_changes_nll(fitted_kde: HierarchicalKDE) -> None:
    """A non-zero V should produce a different NLL than the base measure."""
    base = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.0, "a_0": 0.0})
    decoder = LowRankTiltDecoder(n_cells=fitted_kde.grid.n_cells, rank=4, zero_init=False)
    timing = ConstantRateTimingModel()
    process = ShotCloudProcess(timing_model=timing, spatial_decoder=decoder, base_measure=base)

    # Sample with the zero-init process so observations are drawn from q_0;
    # then compare NLL under the random-init decoder.
    zero_proc = ShotCloudProcess(
        timing_model=timing,
        spatial_decoder=LowRankTiltDecoder(n_cells=fitted_kde.grid.n_cells, rank=4, zero_init=True),
        base_measure=base,
    )
    obs = _sample_observations(zero_proc, ["G0", "F0"], R=10)

    nll_model = nll_per_shot(process, obs)
    nll_base = base_measure_nll_per_shot(base, obs)
    # With random V (untrained), the random-init decoder typically does
    # *worse* than the prior on prior-distributed data.
    assert nll_model != nll_base


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_observations_returns_nan(process: ShotCloudProcess) -> None:
    assert np.isnan(nll_per_shot(process, []))
    assert np.isnan(base_measure_nll_per_shot(process.base_measure, []))


def test_observations_with_zero_shot_games_skip_them(process: ShotCloudProcess) -> None:
    """A (player_id, []) entry contributes nothing — NLL still computed on others."""
    real_obs = _sample_observations(process, ["G0"], R=3)
    obs_with_empty = [*real_obs, ("G0", np.array([], dtype=np.int64))]
    n_real = nll_per_shot(process, real_obs)
    n_with_empty = nll_per_shot(process, obs_with_empty)
    np.testing.assert_allclose(n_real, n_with_empty)
