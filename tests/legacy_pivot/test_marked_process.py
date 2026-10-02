"""End-to-end tests for :class:`shotcloud.models.ShotCloudProcess`.

The headline test is :func:`test_zero_init_sampling_recovers_q0_empirically`,
which confirms the entire stack — KDE → product → tilt decoder → softmax
→ categorical sample → dequantize — composes correctly under the
zero-init invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud import CourtGrid, HierarchicalKDE
from shotcloud.legacy import KDEProduct
from shotcloud.legacy_pivot.marked_process import ShotCloudProcess, ShotSequence
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
    base = KDEProduct(
        hierarchical_kde=fitted_kde,
        weights={"a_p": 1.0, "a_g": 0.0, "a_0": 0.0},  # pure player density
    )
    decoder = LowRankTiltDecoder(n_cells=fitted_kde.grid.n_cells, rank=4, zero_init=True)
    timing = ConstantRateTimingModel(mean_shots=20.0, game_length=48.0)
    return ShotCloudProcess(timing_model=timing, spatial_decoder=decoder, base_measure=base)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_decoder_n_cells_must_match_grid(fitted_kde: HierarchicalKDE) -> None:
    base = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.0, "a_0": 0.0})
    wrong_decoder = LowRankTiltDecoder(n_cells=99, rank=4)
    timing = ConstantRateTimingModel()
    with pytest.raises(ValueError, match="n_cells"):
        ShotCloudProcess(timing_model=timing, spatial_decoder=wrong_decoder, base_measure=base)


def test_grid_property_exposes_base_measure_grid(process: ShotCloudProcess) -> None:
    assert process.grid is process.base_measure.grid


# ---------------------------------------------------------------------------
# sample(): single-game sampling
# ---------------------------------------------------------------------------


def test_sample_returns_consistent_shapes(process: ShotCloudProcess) -> None:
    rng = np.random.default_rng(0)
    seq = process.sample("G0", rng=rng)
    assert isinstance(seq, ShotSequence)
    n = seq.n_shots
    assert seq.taus.shape == (n,)
    assert seq.cells.shape == (n,)
    assert seq.x.shape == (n,)
    assert seq.y.shape == (n,)


def test_sampled_taus_lie_in_game_window(process: ShotCloudProcess) -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        seq = process.sample("G0", rng=rng)
        if seq.n_shots == 0:
            continue
        assert seq.taus.min() >= 0.0
        assert seq.taus.max() <= 48.0


def test_sampled_xy_lie_inside_court_bounds(process: ShotCloudProcess) -> None:
    """Critical invariant from plan §5.2: dequantized samples stay in court."""
    g = process.grid
    rng = np.random.default_rng(0)
    for _ in range(20):
        seq = process.sample("G0", rng=rng)
        if seq.n_shots == 0:
            continue
        assert seq.x.min() >= g.xlim[0]
        assert seq.x.max() <= g.xlim[1]
        assert seq.y.min() >= g.ylim[0]
        assert seq.y.max() <= g.ylim[1]


def test_sampled_cells_round_trip_through_coord_to_cell(process: ShotCloudProcess) -> None:
    """Each (x, y) must map back to its source cell."""
    rng = np.random.default_rng(0)
    seq = process.sample("G0", rng=rng)
    if seq.n_shots == 0:
        pytest.skip("zero-shot sample")
    recovered = process.grid.coord_to_cell(seq.x, seq.y)
    np.testing.assert_array_equal(recovered, seq.cells)


def test_sample_is_deterministic_under_fixed_seed(process: ShotCloudProcess) -> None:
    s1 = process.sample("G0", rng=np.random.default_rng(7))
    s2 = process.sample("G0", rng=np.random.default_rng(7))
    np.testing.assert_array_equal(s1.cells, s2.cells)
    np.testing.assert_array_equal(s1.taus, s2.taus)
    np.testing.assert_array_equal(s1.x, s2.x)
    np.testing.assert_array_equal(s1.y, s2.y)


def test_zero_shots_sample_returns_empty_arrays() -> None:
    """K=0 timing returns an empty sequence cleanly."""
    grid = CourtGrid(nx=10, ny=10)
    base = HierarchicalKDE(grid=grid, recency_half_life_days=None)
    rng = np.random.default_rng(0)
    base.fit(
        x=rng.uniform(-10, 10, 50),
        y=rng.uniform(-2, 30, 50),
        player_id=np.array(["P"] * 50),
        position=np.array(["G"] * 50),
    )
    product = KDEProduct(hierarchical_kde=base, weights={"a_p": 1.0, "a_g": 0.0, "a_0": 0.0})
    decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4, zero_init=True)
    timing = ConstantRateTimingModel(mean_shots=0.0)  # always K=0
    process = ShotCloudProcess(timing_model=timing, spatial_decoder=decoder, base_measure=product)

    seq = process.sample("P", rng=np.random.default_rng(0))
    assert seq.n_shots == 0
    assert seq.taus.shape == (0,)
    assert seq.cells.shape == (0,)


# ---------------------------------------------------------------------------
# sample_shot_cloud(): aggregated R-game cloud
# ---------------------------------------------------------------------------


def test_sample_shot_cloud_aggregates_R_games(process: ShotCloudProcess) -> None:
    cloud = process.sample_shot_cloud("G0", R=50, rng=np.random.default_rng(0))
    assert cloud.n_games == 50
    # ~mean_shots * R = 1000; allow generous range.
    assert 600 < cloud.total_shots < 1400
    assert cloud.shots_per_game > 0.0


def test_sample_shot_cloud_R_zero_or_negative_raises(process: ShotCloudProcess) -> None:
    with pytest.raises(ValueError, match="R must be positive"):
        process.sample_shot_cloud("G0", R=0)
    with pytest.raises(ValueError, match="R must be positive"):
        process.sample_shot_cloud("G0", R=-5)


# ---------------------------------------------------------------------------
# THE end-to-end correctness test
# ---------------------------------------------------------------------------


def test_zero_init_sampling_recovers_q0_empirically(process: ShotCloudProcess) -> None:
    """End-to-end: with zero-init decoder, sampled cell freq ≈ q_0(player).

    This exercises the entire stack — KDE → product → tilt → softmax →
    categorical sample → dequantize. Total-variation tolerance is set
    generously to absorb finite-sample noise (~10K shots, ~1700 cells).
    """
    grid = process.grid
    rng = np.random.default_rng(0)
    cloud = process.sample_shot_cloud("G0", R=500, rng=rng)

    # Empirical cell distribution from sampled cells.
    empirical = np.bincount(cloud.cells, minlength=grid.n_cells) / cloud.total_shots
    expected = process.base_measure.density("G0").ravel()

    tv = 0.5 * float(np.abs(empirical - expected).sum())
    assert tv < 0.07, f"empirical TV from q_0 was {tv:.4f}, expected < 0.07"


def test_zero_init_sampling_recovers_q0_via_dequantized_xy(
    process: ShotCloudProcess,
) -> None:
    """Same check, but via dequantized (x, y) re-binned through the grid."""
    grid = process.grid
    rng = np.random.default_rng(1)
    cloud = process.sample_shot_cloud("G0", R=500, rng=rng)

    cells_from_xy = grid.coord_to_cell(cloud.x, cloud.y)
    empirical = np.bincount(cells_from_xy, minlength=grid.n_cells) / cloud.total_shots
    expected = process.base_measure.density("G0").ravel()

    tv = 0.5 * float(np.abs(empirical - expected).sum())
    assert tv < 0.07


# ---------------------------------------------------------------------------
# log_prob
# ---------------------------------------------------------------------------


def test_log_prob_is_finite_and_non_positive(process: ShotCloudProcess) -> None:
    rng = np.random.default_rng(0)
    seq = process.sample("G0", rng=rng)
    if seq.n_shots == 0:
        pytest.skip("zero-shot sample")
    lp = process.log_prob("G0", seq.cells, seq.taus)
    assert np.isfinite(lp)
    assert lp <= 0.0


def test_log_prob_decomposes_into_timing_plus_spatial(
    process: ShotCloudProcess,
) -> None:
    """Manually verify log p(S) = log p(K, τ) + Σ log q_0(c_i) at zero-init."""
    rng = np.random.default_rng(0)
    seq = process.sample("G0", rng=rng)
    if seq.n_shots == 0:
        pytest.skip("zero-shot sample")

    log_p_total = process.log_prob("G0", seq.cells, seq.taus)
    log_p_timing = process.timing_model.log_prob(seq.n_shots, seq.taus, {})
    # At zero-init, decoder log-prob = log q_0; sum over observed cells.
    log_q0 = process.base_measure.log_density("G0").ravel()
    log_p_spatial = float(log_q0[seq.cells].sum())

    np.testing.assert_allclose(log_p_total, log_p_timing + log_p_spatial, atol=1e-5)


def test_log_prob_for_zero_shots(process: ShotCloudProcess) -> None:
    """K=0 → only the timing term contributes."""
    lp = process.log_prob(
        "G0",
        cells=np.array([], dtype=np.int64),
        taus=np.array([], dtype=np.float64),
    )
    expected = process.timing_model.log_prob(0, np.array([], dtype=np.float64), {})
    np.testing.assert_allclose(lp, expected, atol=1e-12)


def test_log_prob_raises_on_length_mismatch(process: ShotCloudProcess) -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        process.log_prob("G0", cells=np.array([0, 1]), taus=np.array([1.0]))


def test_log_prob_raises_for_unknown_player(process: ShotCloudProcess) -> None:
    with pytest.raises(KeyError, match="unknown player"):
        process.log_prob("nobody", cells=np.array([0]), taus=np.array([1.0]))
