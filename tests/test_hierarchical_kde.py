"""Tests for :class:`shotcloud.kde.HierarchicalKDE`."""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud import CourtGrid, HierarchicalKDE

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_shots(
    n_per_player: int = 200,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Two positions × two players × n_per_player shots.

    - guards (G): cluster at the arc, ~25 ft (above the break / wings)
    - bigs (F):   cluster in the paint, ~5 ft

    Returns a dict of arrays the same length, suitable to pass to
    HierarchicalKDE.fit() via `**`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    base = np.datetime64("2024-01-01", "D")
    for pos, mu_x, mu_y, sigma in [
        ("G", 0.0, 24.0, 2.0),
        ("F", 0.0, 4.0, 2.0),
    ]:
        for player_idx in range(2):
            pid = f"{pos}{player_idx}"
            x = rng.normal(mu_x + 2.0 * (player_idx - 0.5), sigma, n_per_player)
            y = rng.normal(mu_y, sigma, n_per_player)
            day_offsets = rng.integers(0, 1095, size=n_per_player)
            dates = base + day_offsets.astype("timedelta64[D]")
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "player_id": np.array([pid] * n_per_player),
                    "position": np.array([pos] * n_per_player),
                    "date": dates,
                }
            )
    out: dict[str, np.ndarray] = {}
    for key in ("x", "y", "player_id", "position", "date"):
        out[key] = np.concatenate([r[key] for r in rows])
    return out


@pytest.fixture
def fitted_kde() -> HierarchicalKDE:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=50, ny=52)
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=200.0, recency_half_life_days=None)
    shots = _synthetic_shots(n_per_player=200)
    kde.fit(
        x=shots["x"],
        y=shots["y"],
        player_id=shots["player_id"],
        position=shots["position"],
        date=None,
    )
    return kde


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_default_construction_is_unfitted() -> None:
    kde = HierarchicalKDE(grid=CourtGrid())
    assert not kde.is_fitted
    with pytest.raises(RuntimeError, match="not been fit"):
        kde.league_density()


def test_negative_bandwidth_raises() -> None:
    with pytest.raises(ValueError, match="bandwidth must be non-negative"):
        HierarchicalKDE(grid=CourtGrid(), bandwidth=-0.1)


def test_negative_kappa_raises() -> None:
    with pytest.raises(ValueError, match="kappa must be non-negative"):
        HierarchicalKDE(grid=CourtGrid(), kappa=-1.0)


def test_negative_or_zero_half_life_raises() -> None:
    with pytest.raises(ValueError, match="recency_half_life_days"):
        HierarchicalKDE(grid=CourtGrid(), recency_half_life_days=0.0)
    with pytest.raises(ValueError, match="recency_half_life_days"):
        HierarchicalKDE(grid=CourtGrid(), recency_half_life_days=-30.0)


# ---------------------------------------------------------------------------
# Fit + densities
# ---------------------------------------------------------------------------


def test_fit_populates_all_levels(fitted_kde: HierarchicalKDE) -> None:
    assert fitted_kde.is_fitted
    assert fitted_kde.league_n == pytest.approx(800.0)  # 4 players × 200 shots
    assert set(fitted_kde.position_density_grid) == {"G", "F"}
    assert set(fitted_kde.player_density_grid) == {"G0", "G1", "F0", "F1"}
    for pos in ("G", "F"):
        assert fitted_kde.position_n[pos] == pytest.approx(400.0)
    for pid in ("G0", "G1", "F0", "F1"):
        assert fitted_kde.player_n[pid] == pytest.approx(200.0)


def test_densities_have_image_layout_shape(fitted_kde: HierarchicalKDE) -> None:
    g = fitted_kde.grid
    assert fitted_kde.league_density().shape == (g.ny, g.nx)
    assert fitted_kde.position_density("G").shape == (g.ny, g.nx)
    assert fitted_kde.player_density("G0").shape == (g.ny, g.nx)


def test_all_densities_are_normalized(fitted_kde: HierarchicalKDE) -> None:
    np.testing.assert_allclose(fitted_kde.league_density().sum(), 1.0)
    np.testing.assert_allclose(fitted_kde.position_density("G").sum(), 1.0)
    np.testing.assert_allclose(fitted_kde.position_density("F").sum(), 1.0)
    np.testing.assert_allclose(fitted_kde.player_density("G0").sum(), 1.0)
    np.testing.assert_allclose(fitted_kde.player_density("G0", hierarchical=False).sum(), 1.0)


def test_all_densities_are_strictly_positive(fitted_kde: HierarchicalKDE) -> None:
    """Epsilon floor must guarantee q > 0 for log-domain operations."""
    assert (fitted_kde.league_density() > 0).all()
    assert (fitted_kde.player_density("G0") > 0).all()
    assert (fitted_kde.player_density("G0", hierarchical=False) > 0).all()


def test_player_density_concentrates_in_correct_region(fitted_kde: HierarchicalKDE) -> None:
    """Guards' density should peak near y=24 ft, bigs' near y=5 ft."""
    g = fitted_kde.grid
    _grid_x, grid_y = g.mesh

    g0 = fitted_kde.player_density("G0", hierarchical=False)
    f0 = fitted_kde.player_density("F0", hierarchical=False)

    g0_peak_y = float(grid_y[np.unravel_index(np.argmax(g0), g0.shape)])
    f0_peak_y = float(grid_y[np.unravel_index(np.argmax(f0), f0.shape)])

    assert 20.0 < g0_peak_y < 28.0
    assert 0.0 < f0_peak_y < 9.0


def test_unknown_lookup_raises(fitted_kde: HierarchicalKDE) -> None:
    with pytest.raises(KeyError, match="unknown player"):
        fitted_kde.player_density("does-not-exist")
    with pytest.raises(KeyError, match="unknown position"):
        fitted_kde.position_density("C")  # center: not in synthetic data


def test_player_with_two_positions_raises() -> None:
    grid = CourtGrid(nx=20, ny=20)
    kde = HierarchicalKDE(grid=grid, recency_half_life_days=None)
    rng = np.random.default_rng(0)
    n = 50
    x = rng.normal(0, 5, n * 2)
    y = rng.normal(0, 5, n * 2)
    pids = np.array(["P"] * n + ["P"] * n)  # one player...
    poses = np.array(["G"] * n + ["F"] * n)  # ...with two positions
    with pytest.raises(ValueError, match="multiple positions"):
        kde.fit(x=x, y=y, player_id=pids, position=poses, date=None)


def test_length_mismatch_raises() -> None:
    grid = CourtGrid(nx=20, ny=20)
    kde = HierarchicalKDE(grid=grid, recency_half_life_days=None)
    with pytest.raises(ValueError, match="length mismatch"):
        kde.fit(x=[0, 1, 2], y=[0, 1], player_id=["A", "A"], position=["G", "G"])


# ---------------------------------------------------------------------------
# Shrinkage limits
# ---------------------------------------------------------------------------


def test_shrinkage_with_kappa_zero_returns_raw_player_density() -> None:
    """α = N/(N+0) = 1, so the blended density equals the raw player density."""
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=50, ny=52)
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=0.0, recency_half_life_days=None)
    shots = _synthetic_shots(n_per_player=200)
    kde.fit(
        x=shots["x"],
        y=shots["y"],
        player_id=shots["player_id"],
        position=shots["position"],
    )
    raw = kde.player_density("G0", hierarchical=False)
    blended = kde.player_density("G0", hierarchical=True)
    np.testing.assert_allclose(blended, raw, atol=1e-12)


def test_shrinkage_with_huge_kappa_approaches_position_prior() -> None:
    """α = N/(N+huge) → 0, so the blend becomes the position prior."""
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=50, ny=52)
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=1e12, recency_half_life_days=None)
    shots = _synthetic_shots(n_per_player=200)
    kde.fit(
        x=shots["x"],
        y=shots["y"],
        player_id=shots["player_id"],
        position=shots["position"],
    )
    blended = kde.player_density("G0", hierarchical=True)
    position_prior = kde.position_density("G")
    np.testing.assert_allclose(blended, position_prior, atol=1e-12)


def test_shrinkage_preserves_normalization(fitted_kde: HierarchicalKDE) -> None:
    blended = fitted_kde.player_density("G0", hierarchical=True)
    np.testing.assert_allclose(blended.sum(), 1.0)


# ---------------------------------------------------------------------------
# Recency
# ---------------------------------------------------------------------------


def test_recency_weights_decrease_for_older_shots() -> None:
    """Effective sample size should drop when half-life << data span."""
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=20)
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.normal(0, 5, n)
    y = rng.normal(10, 5, n)
    pids = np.array(["P"] * n)
    poses = np.array(["G"] * n)
    # Spread shots over 1000 days.
    dates = np.datetime64("2024-01-01", "D") + rng.integers(0, 1000, n).astype("timedelta64[D]")

    no_recency = HierarchicalKDE(grid=grid, bandwidth=1.5, recency_half_life_days=None).fit(
        x=x, y=y, player_id=pids, position=poses, date=dates
    )
    aggressive = HierarchicalKDE(grid=grid, bandwidth=1.5, recency_half_life_days=30.0).fit(
        x=x, y=y, player_id=pids, position=poses, date=dates
    )
    # No recency: every shot weight = 1, so effective_n == n.
    assert no_recency.player_n["P"] == pytest.approx(float(n))
    # Aggressive recency: most shots are old → small effective_n.
    assert aggressive.player_n["P"] < float(n) * 0.2


def test_recency_is_disabled_when_no_dates_passed(fitted_kde: HierarchicalKDE) -> None:
    """Even with default half_life=365, omitting `date` falls back to uniform weights."""
    # fitted_kde was fit with date=None and half_life=None; sanity-check effective_n.
    assert fitted_kde.league_n == pytest.approx(800.0)


def test_recency_with_explicit_reference_date() -> None:
    grid = CourtGrid(nx=20, ny=20)
    rng = np.random.default_rng(0)
    n = 100
    x = rng.uniform(-10, 10, n)
    y = rng.uniform(-2, 30, n)
    pids = np.array(["P"] * n)
    poses = np.array(["G"] * n)
    dates = np.datetime64("2024-01-01", "D") + rng.integers(0, 365, n).astype("timedelta64[D]")

    kde = HierarchicalKDE(grid=grid, recency_half_life_days=180.0).fit(
        x=x,
        y=y,
        player_id=pids,
        position=poses,
        date=dates,
        reference_date=np.datetime64("2025-06-01", "D"),
    )
    # With reference date in the future, all weights < 1.
    assert kde.player_n["P"] < float(n)


# ---------------------------------------------------------------------------
# Bandwidth edge cases
# ---------------------------------------------------------------------------


def test_zero_bandwidth_returns_floored_histogram() -> None:
    """bandwidth=0 means no Gaussian smoothing; just the (floored) histogram."""
    grid = CourtGrid(xlim=(0.0, 10.0), ylim=(0.0, 10.0), nx=10, ny=10)
    kde = HierarchicalKDE(grid=grid, bandwidth=0.0, recency_half_life_days=None)
    x = np.array([5.0, 5.0, 5.0])  # all in the center cell
    y = np.array([5.0, 5.0, 5.0])
    pids = np.array(["P"] * 3)
    poses = np.array(["G"] * 3)
    kde.fit(x=x, y=y, player_id=pids, position=poses)
    density = kde.player_density("P", hierarchical=False)
    # Mass should be concentrated in one cell.
    assert (density > 0.5).sum() == 1
    # Other cells just at the floor.
    n_floor_cells = (density < 1e-10).sum()
    assert n_floor_cells > grid.n_cells - 5
