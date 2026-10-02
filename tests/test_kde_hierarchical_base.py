"""Tests for :class:`shotcloud.kde.HierarchicalKDEBase`."""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud import CourtGrid, HierarchicalKDE, HierarchicalKDEBase


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


def test_unfitted_kde_raises() -> None:
    unfitted = HierarchicalKDE(grid=CourtGrid())
    with pytest.raises(ValueError, match="must be fit"):
        HierarchicalKDEBase(hierarchical_kde=unfitted)


def test_negative_epsilon_raises(fitted_kde: HierarchicalKDE) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        HierarchicalKDEBase(hierarchical_kde=fitted_kde, epsilon=-1e-9)


def test_density_sums_to_one(fitted_kde: HierarchicalKDE) -> None:
    base = HierarchicalKDEBase(hierarchical_kde=fitted_kde)
    q0 = base.density("G0")
    np.testing.assert_allclose(q0.sum(), 1.0, atol=1e-12)


def test_density_strictly_positive(fitted_kde: HierarchicalKDE) -> None:
    base = HierarchicalKDEBase(hierarchical_kde=fitted_kde)
    q0 = base.density("G0")
    assert (q0 > 0).all()


def test_density_image_layout_shape(fitted_kde: HierarchicalKDE) -> None:
    base = HierarchicalKDEBase(hierarchical_kde=fitted_kde)
    g = base.grid
    assert base.density("G0").shape == (g.ny, g.nx)
    assert base.log_density("G0").shape == (g.ny, g.nx)


def test_density_matches_hierarchical_player_density(fitted_kde: HierarchicalKDE) -> None:
    """The base measure should equal what HierarchicalKDE produces directly."""
    base = HierarchicalKDEBase(hierarchical_kde=fitted_kde)
    direct = fitted_kde.player_density("G0", hierarchical=True)
    np.testing.assert_allclose(base.density("G0"), direct, atol=1e-10)


def test_log_density_consistent_with_density(fitted_kde: HierarchicalKDE) -> None:
    base = HierarchicalKDEBase(hierarchical_kde=fitted_kde)
    q0 = base.density("G0")
    log_q0 = base.log_density("G0")
    np.testing.assert_allclose(q0, np.exp(log_q0), atol=1e-10)


def test_log_density_normalizes(fitted_kde: HierarchicalKDE) -> None:
    base = HierarchicalKDEBase(hierarchical_kde=fitted_kde)
    log_q0 = base.log_density("G0")
    np.testing.assert_allclose(np.exp(log_q0).sum(), 1.0, atol=1e-12)


def test_grid_property_passes_through(fitted_kde: HierarchicalKDE) -> None:
    base = HierarchicalKDEBase(hierarchical_kde=fitted_kde)
    assert base.grid is fitted_kde.grid


def test_unknown_player_raises(fitted_kde: HierarchicalKDE) -> None:
    base = HierarchicalKDEBase(hierarchical_kde=fitted_kde)
    with pytest.raises(KeyError, match="unknown player"):
        base.density("does-not-exist")
    with pytest.raises(KeyError, match="unknown player"):
        base.log_density("does-not-exist")
