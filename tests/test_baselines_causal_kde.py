"""Tests for :class:`~shotcloud.baselines.causal_kde.CausalGridKDEBaseline`.

Covers the causal fit cutoff, per-ft² density evaluation (nearest-cell and
bilinear), the shrinkage limits in κ, the cold-start fallback chain, cloud
sampling, and error handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shotcloud.baselines.causal_kde import CausalGridKDEBaseline
from shotcloud.grids import CourtGrid


def _make_shots(
    *,
    n_players: int = 6,
    shots_per_player: int = 200,
    train_end: str = "2023-06-30",
    val_window_days: int = 90,
    seed: int = 0,
) -> pd.DataFrame:
    """Build synthetic shots for ``n_players`` players, half guards and half centers.

    Each player's ``shots_per_player`` shots are split evenly between the
    year before ``train_end`` and the ``val_window_days`` after it.
    """
    rng = np.random.default_rng(seed)
    train_end_ts = pd.Timestamp(train_end)
    rows = []
    for p in range(n_players):
        pos = "G" if p < n_players // 2 else "C"
        # Guards skew perimeter, centers skew rim.
        center = np.array([0.0, 22.0]) if pos == "G" else np.array([0.0, 5.0])
        std = 4.0
        train_xy = rng.normal(center, std, size=(shots_per_player // 2, 2))
        val_xy = rng.normal(center, std, size=(shots_per_player // 2, 2))
        # Train dates: random within 1 year before train_end.
        train_dates = train_end_ts - pd.to_timedelta(
            rng.integers(1, 365, size=shots_per_player // 2), unit="D"
        )
        val_dates = train_end_ts + pd.to_timedelta(
            rng.integers(1, val_window_days, size=shots_per_player // 2), unit="D"
        )
        for (x, y), d in zip(train_xy, train_dates, strict=True):
            rows.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "player_id": str(p),
                    "position_group": pos,
                    "date": d,
                }
            )
        for (x, y), d in zip(val_xy, val_dates, strict=True):
            rows.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "player_id": str(p),
                    "position_group": pos,
                    "date": d,
                }
            )
    return pd.DataFrame(rows)


def _make_grid() -> CourtGrid:
    return CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=50, ny=52)


# ---------------------------------------------------------------------------
# Causal fit + train-end-date boundary
# ---------------------------------------------------------------------------


def test_fit_excludes_post_cutoff_shots() -> None:
    """Shots dated after ``train_end_date`` do not inform any density.

    Player ``"X"`` has shots only after the cutoff, all at one corner-3
    location, so X must fall back to the position prior with no spike there.
    """
    shots = _make_shots(n_players=4)
    # Inject player "X" with shots ONLY after train_end_date, placed at
    # a distinctive corner location.
    train_end = pd.Timestamp("2023-06-30")
    post_only_x = pd.DataFrame(
        {
            "x": np.full(30, 23.0),
            "y": np.full(30, 4.0),  # corner-3 region
            "player_id": ["X"] * 30,
            "position_group": ["G"] * 30,
            "date": [train_end + pd.Timedelta(days=10 + i) for i in range(30)],
        }
    )
    shots = pd.concat([shots, post_only_x], ignore_index=True)

    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid, bandwidth=1.5, kappa=500.0)
    baseline.fit(shots, train_end_date=train_end)

    # Player X had no pre-cutoff shots → cold-start → position fallback.
    # Density at the corner-3 location should be ordinary, not a sharp spike.
    corner_query = np.array([[23.0, 4.0]])
    far_query = np.array([[-23.0, 4.0]])  # mirror — symmetric under position-G prior
    d_corner = baseline.density_at_xy("X", corner_query, position_hint="G")[0]
    d_far = baseline.density_at_xy("X", far_query, position_hint="G")[0]
    # Position-G prior is approximately symmetric → corner densities should
    # be within a factor of 2. A leaked post-cutoff fit would spike d_corner
    # by 10x or more.
    assert d_corner > 0
    assert d_far > 0
    assert 0.5 < d_corner / d_far < 2.0, (
        f"corner/far ratio = {d_corner / d_far:.3f}; >2 suggests post-cutoff leak"
    )


def test_fit_requires_required_columns() -> None:
    """``fit`` raises ``KeyError`` when a required column is missing."""
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid)
    bad = pd.DataFrame({"x": [0.0], "y": [0.0], "player_id": ["a"]})  # no position_group, no date
    with pytest.raises(KeyError, match=r"position_group|date"):
        baseline.fit(bad, train_end_date=pd.Timestamp("2024-01-01"))


def test_fit_raises_on_empty_train_set() -> None:
    """``fit`` raises when every shot is after the cutoff."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid)
    impossible_cutoff = pd.Timestamp("1990-01-01")
    with pytest.raises(ValueError, match="no training shots"):
        baseline.fit(shots, train_end_date=impossible_cutoff)


# ---------------------------------------------------------------------------
# density_at_xy
# ---------------------------------------------------------------------------


def test_density_at_xy_shape_and_dtype() -> None:
    """``density_at_xy`` returns a finite, non-negative float64 array of shape ``(N,)``."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    xy = np.array([[0.0, 5.0], [10.0, 20.0], [-15.0, 24.0]])
    d = baseline.density_at_xy("0", xy, position_hint="G")
    assert d.shape == (3,)
    assert d.dtype == np.float64
    assert np.all(d >= 0)
    assert np.all(np.isfinite(d))


def test_density_is_per_ft_squared() -> None:
    """``density_at_xy`` returns density per ft², integrating to about 1 over the grid."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    # Sample at every cell center (nearest-cell mode = exact density grid).
    xs, ys = np.meshgrid(grid.xcenters, grid.ycenters)
    xy = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    d = baseline.density_at_xy("0", xy, position_hint="G", bilinear=False)
    total_mass = float(d.sum() * baseline.cell_area)
    assert 0.95 < total_mass < 1.05, f"total mass = {total_mass:.4f}, expected ≈ 1"


def test_off_grid_query_falls_to_floor_not_nan() -> None:
    """Queries outside the grid return a finite, non-negative floor value."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    # Far beyond the grid extent.
    off_grid = np.array([[100.0, 100.0], [-100.0, -100.0], [0.0, 200.0]])
    d_bilinear = baseline.density_at_xy("0", off_grid, position_hint="G", bilinear=True)
    d_nearest = baseline.density_at_xy("0", off_grid, position_hint="G", bilinear=False)
    assert np.all(np.isfinite(d_bilinear))
    assert np.all(np.isfinite(d_nearest))
    assert np.all(d_bilinear >= 0)
    assert np.all(d_nearest >= 0)


def test_density_at_xy_rejects_wrong_shape() -> None:
    """Queries not of shape ``(N, 2)`` raise ``ValueError``."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        baseline.density_at_xy("0", np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        baseline.density_at_xy("0", np.array([[1.0, 2.0, 3.0]]))


def test_bilinear_at_cell_center_matches_nearest() -> None:
    """At cell centers, bilinear interpolation approximately matches the nearest-cell density."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    # Pick interior cell centers (avoid grid boundary).
    xy = np.array(
        [
            [float(grid.xcenters[10]), float(grid.ycenters[10])],
            [float(grid.xcenters[25]), float(grid.ycenters[25])],
            [float(grid.xcenters[40]), float(grid.ycenters[40])],
        ]
    )
    d_bilinear = baseline.density_at_xy("0", xy, position_hint="G", bilinear=True)
    d_nearest = baseline.density_at_xy("0", xy, position_hint="G", bilinear=False)
    # Modest tolerance: the bilinear stencil is chosen by floor() and can
    # blend in a neighboring cell.
    np.testing.assert_allclose(d_bilinear, d_nearest, rtol=0.2)


# ---------------------------------------------------------------------------
# Shrinkage limits
# ---------------------------------------------------------------------------


def test_shrinkage_kappa_infinity_approaches_position_prior() -> None:
    """At very large κ, a player's density matches the position prior."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid, bandwidth=1.5, kappa=1e9).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    # Query at a small representative point set.
    xy = np.array([[float(grid.xcenters[i]), float(grid.ycenters[i])] for i in range(5, 45, 5)])
    # Pull player density and position density via the public surface.
    d_player = baseline.density_at_xy("0", xy, position_hint="G", bilinear=False)
    pos_density_grid = baseline.kde.position_density("G") / baseline.cell_area
    # Index into the position grid at the same cells.
    ix, iy = grid.coord_to_ij(xy[:, 0], xy[:, 1])
    d_pos = pos_density_grid[iy, ix]
    np.testing.assert_allclose(d_player, d_pos, rtol=1e-3)


def test_shrinkage_kappa_zero_approaches_pure_self() -> None:
    """At κ = 0, a player's density is unshrunk and differs from the position prior."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid, bandwidth=1.5, kappa=0.0).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    # At κ=0 the shrinkage weight α = N_p / (N_p + 0) = 1, so density should
    # NOT match the position prior (unless the player happens to be exactly
    # average for their position, which won't be true on this synthetic
    # data because positions are coarse).
    xy = np.array([[float(grid.xcenters[i]), float(grid.ycenters[i])] for i in range(5, 45, 5)])
    d_player_kappa0 = baseline.density_at_xy("0", xy, position_hint="G", bilinear=False)
    pos_density_grid = baseline.kde.position_density("G") / baseline.cell_area
    ix, iy = grid.coord_to_ij(xy[:, 0], xy[:, 1])
    d_pos = pos_density_grid[iy, ix]
    # Player density at κ=0 should NOT track position density tightly.
    rel_diff = float(np.abs(d_player_kappa0 - d_pos).mean() / (d_pos.mean() + 1e-9))
    assert rel_diff > 0.05, (
        f"at κ=0, player density should differ from position prior; got rel_diff={rel_diff:.3f}"
    )


# ---------------------------------------------------------------------------
# Cold-start fallback chain
# ---------------------------------------------------------------------------


def test_cold_start_falls_back_to_position_when_hint_given() -> None:
    """An unknown player with a position hint falls back to the position prior."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid, fallback_to_position=True).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    xy = np.array([[float(grid.xcenters[i]), float(grid.ycenters[i])] for i in range(5, 45, 5)])
    d_unknown_with_hint = baseline.density_at_xy(
        "UNKNOWN-PLAYER-999", xy, position_hint="G", bilinear=False
    )
    pos_density_grid = baseline.kde.position_density("G") / baseline.cell_area
    ix, iy = grid.coord_to_ij(xy[:, 0], xy[:, 1])
    d_pos = pos_density_grid[iy, ix]
    np.testing.assert_allclose(d_unknown_with_hint, d_pos, rtol=1e-6)


def test_cold_start_falls_back_to_league_when_no_hint() -> None:
    """An unknown player without a position hint falls back to the league density."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    xy = np.array([[float(grid.xcenters[i]), float(grid.ycenters[i])] for i in range(5, 45, 5)])
    d_unknown_no_hint = baseline.density_at_xy("UNKNOWN-XYZ", xy, bilinear=False)
    league_density_grid = baseline.kde.league_density() / baseline.cell_area
    ix, iy = grid.coord_to_ij(xy[:, 0], xy[:, 1])
    d_league = league_density_grid[iy, ix]
    np.testing.assert_allclose(d_unknown_no_hint, d_league, rtol=1e-6)


def test_fallback_to_position_disabled_uses_league() -> None:
    """With ``fallback_to_position=False``, an unknown player uses the league density."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid, fallback_to_position=False).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    xy = np.array([[float(grid.xcenters[i]), float(grid.ycenters[i])] for i in range(5, 45, 5)])
    d = baseline.density_at_xy("UNKNOWN-X", xy, position_hint="G", bilinear=False)
    league_density_grid = baseline.kde.league_density() / baseline.cell_area
    ix, iy = grid.coord_to_ij(xy[:, 0], xy[:, 1])
    d_league = league_density_grid[iy, ix]
    np.testing.assert_allclose(d, d_league, rtol=1e-6)


# ---------------------------------------------------------------------------
# sample_cloud
# ---------------------------------------------------------------------------


def test_sample_cloud_shape_and_in_court() -> None:
    """``sample_cloud`` returns ``(n_shots, 2)`` samples inside the grid extent."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    rng = np.random.default_rng(0)
    samples = baseline.sample_cloud("0", n_shots=200, rng=rng, position_hint="G")
    assert samples.shape == (200, 2)
    assert np.all(samples[:, 0] >= grid.xlim[0])
    assert np.all(samples[:, 0] <= grid.xlim[1])
    assert np.all(samples[:, 1] >= grid.ylim[0])
    assert np.all(samples[:, 1] <= grid.ylim[1])


def test_sample_cloud_zero_shots_returns_empty() -> None:
    """``sample_cloud`` with ``n_shots=0`` returns an empty ``(0, 2)`` array."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    rng = np.random.default_rng(0)
    samples = baseline.sample_cloud("0", n_shots=0, rng=rng)
    assert samples.shape == (0, 2)


def test_sample_cloud_is_reproducible_with_seed() -> None:
    """Identically seeded RNGs produce identical clouds."""
    shots = _make_shots()
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid).fit(
        shots, train_end_date=pd.Timestamp("2023-06-30")
    )
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    s_a = baseline.sample_cloud("0", n_shots=100, rng=rng_a, position_hint="G")
    s_b = baseline.sample_cloud("0", n_shots=100, rng=rng_b, position_hint="G")
    np.testing.assert_array_equal(s_a, s_b)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unfitted_density_raises() -> None:
    """``density_at_xy`` raises before ``fit``."""
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid)
    with pytest.raises(RuntimeError, match="not fitted"):
        baseline.density_at_xy("0", np.array([[0.0, 5.0]]))


def test_unfitted_sample_raises() -> None:
    """``sample_cloud`` raises before ``fit``."""
    grid = _make_grid()
    baseline = CausalGridKDEBaseline(grid=grid)
    with pytest.raises(RuntimeError, match="not fitted"):
        baseline.sample_cloud("0", n_shots=1, rng=np.random.default_rng(0))
