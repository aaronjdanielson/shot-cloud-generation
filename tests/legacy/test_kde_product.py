"""Tests for :class:`shotcloud.legacy.KDEProduct`."""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud import CourtGrid, HierarchicalKDE
from shotcloud.legacy import KDEProduct

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


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_missing_weight_key_raises(fitted_kde: HierarchicalKDE) -> None:
    with pytest.raises(ValueError, match="missing weight keys"):
        KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.3})


def test_unknown_weight_key_raises(fitted_kde: HierarchicalKDE) -> None:
    with pytest.raises(ValueError, match="unknown weight keys"):
        KDEProduct(
            hierarchical_kde=fitted_kde,
            weights={"a_p": 1.0, "a_g": 0.3, "a_0": 0.2, "a_d": 0.5},
        )


def test_negative_weight_raises(fitted_kde: HierarchicalKDE) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        KDEProduct(
            hierarchical_kde=fitted_kde,
            weights={"a_p": -0.1, "a_g": 0.3, "a_0": 0.2},
        )


def test_negative_epsilon_raises(fitted_kde: HierarchicalKDE) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        KDEProduct(
            hierarchical_kde=fitted_kde,
            weights={"a_p": 1.0, "a_g": 0.3, "a_0": 0.2},
            epsilon=-1e-9,
        )


def test_unfitted_hierarchical_kde_raises() -> None:
    unfitted = HierarchicalKDE(grid=CourtGrid())
    with pytest.raises(ValueError, match="must be fit"):
        KDEProduct(hierarchical_kde=unfitted, weights={"a_p": 1.0, "a_g": 0.3, "a_0": 0.2})


# ---------------------------------------------------------------------------
# Density invariants — q0.sum() == 1, q0 > 0, correct shape
# ---------------------------------------------------------------------------


def test_density_sums_to_one(fitted_kde: HierarchicalKDE) -> None:
    """Critical invariant: q0.sum() == 1."""
    product = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.3, "a_0": 0.2})
    q0 = product.density("G0")
    np.testing.assert_allclose(q0.sum(), 1.0, atol=1e-12)


def test_density_strictly_positive(fitted_kde: HierarchicalKDE) -> None:
    """Critical invariant: q0 > 0 everywhere — required for log q0."""
    product = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.3, "a_0": 0.2})
    q0 = product.density("G0")
    assert (q0 > 0).all()


def test_density_has_image_layout_shape(fitted_kde: HierarchicalKDE) -> None:
    product = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.3, "a_0": 0.2})
    g = product.grid
    assert product.density("G0").shape == (g.ny, g.nx)
    assert product.log_density("G0").shape == (g.ny, g.nx)


# ---------------------------------------------------------------------------
# Weight limits
# ---------------------------------------------------------------------------


def test_only_player_weight_returns_player_density(fitted_kde: HierarchicalKDE) -> None:
    """With a_p=1, a_g=0, a_0=0, the product equals the raw player density."""
    product = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.0, "a_0": 0.0})
    q0 = product.density("G0")
    raw = fitted_kde.player_density("G0", hierarchical=False)
    np.testing.assert_allclose(q0, raw, atol=1e-10)


def test_only_league_weight_returns_league_density(fitted_kde: HierarchicalKDE) -> None:
    product = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 0.0, "a_g": 0.0, "a_0": 1.0})
    q0 = product.density("G0")
    league = fitted_kde.league_density()
    np.testing.assert_allclose(q0, league, atol=1e-10)


def test_only_position_weight_returns_position_density(
    fitted_kde: HierarchicalKDE,
) -> None:
    product = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 0.0, "a_g": 1.0, "a_0": 0.0})
    q0_g0 = product.density("G0")
    q0_g1 = product.density("G1")  # same position group as G0
    pos_density = fitted_kde.position_density("G")
    np.testing.assert_allclose(q0_g0, pos_density, atol=1e-10)
    # Two players in the same position get the same density.
    np.testing.assert_allclose(q0_g0, q0_g1, atol=1e-10)


def test_zero_all_weights_returns_uniform(fitted_kde: HierarchicalKDE) -> None:
    """When all weights are zero, the geometric product is constant → uniform."""
    product = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 0.0, "a_g": 0.0, "a_0": 0.0})
    q0 = product.density("G0")
    g = product.grid
    np.testing.assert_allclose(q0, np.full((g.ny, g.nx), 1.0 / g.n_cells), atol=1e-12)


# ---------------------------------------------------------------------------
# log_density / density consistency
# ---------------------------------------------------------------------------


def test_density_equals_exp_log_density(fitted_kde: HierarchicalKDE) -> None:
    """density(p) and exp(log_density(p)) must agree."""
    product = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.3, "a_0": 0.2})
    q0 = product.density("G0")
    log_q0 = product.log_density("G0")
    np.testing.assert_allclose(q0, np.exp(log_q0), atol=1e-10)


def test_log_density_normalizes_in_probability_space(
    fitted_kde: HierarchicalKDE,
) -> None:
    """exp(log_density).sum() must equal 1."""
    product = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.3, "a_0": 0.2})
    log_q0 = product.log_density("G0")
    np.testing.assert_allclose(np.exp(log_q0).sum(), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Manual product check
# ---------------------------------------------------------------------------


def test_product_matches_manual_geometric_combine(fitted_kde: HierarchicalKDE) -> None:
    """Verify the formula against a direct elementwise geometric product."""
    weights = {"a_p": 0.7, "a_g": 0.4, "a_0": 0.2}
    product = KDEProduct(hierarchical_kde=fitted_kde, weights=weights)

    raw_p = fitted_kde.player_density("G0", hierarchical=False)
    raw_g = fitted_kde.position_density("G")
    raw_l = fitted_kde.league_density()

    manual = raw_p ** weights["a_p"] * raw_g ** weights["a_g"] * raw_l ** weights["a_0"]
    manual = manual / manual.sum()

    np.testing.assert_allclose(product.density("G0"), manual, atol=1e-10)


# ---------------------------------------------------------------------------
# Lookup errors
# ---------------------------------------------------------------------------


def test_unknown_player_raises(fitted_kde: HierarchicalKDE) -> None:
    product = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.3, "a_0": 0.2})
    with pytest.raises(KeyError, match="unknown player"):
        product.density("does-not-exist")
    with pytest.raises(KeyError, match="unknown player"):
        product.log_density("does-not-exist")


def test_grid_property_passes_through(fitted_kde: HierarchicalKDE) -> None:
    product = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.3, "a_0": 0.2})
    assert product.grid is fitted_kde.grid
