"""Tests for :class:`shotcloud.legacy_pivot.defensive_kde.DefensiveKDE`."""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud import CourtGrid
from shotcloud.legacy_pivot.defensive_kde import DefensiveKDE

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _two_opponent_shots(n_per_opp: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    """Synthetic shots-allowed: opponent A allows shots near (0, 4),
    opponent B allows shots near (0, 24). Different defensive geometries."""
    rng = np.random.default_rng(seed)
    rows = []
    for opp, mu_y in [("A", 4.0), ("B", 24.0)]:
        x = rng.normal(0, 2, n_per_opp)
        y = rng.normal(mu_y, 2, n_per_opp)
        rows.append({"x": x, "y": y, "opponent": np.array([opp] * n_per_opp)})
    out: dict[str, np.ndarray] = {}
    for key in ("x", "y", "opponent"):
        out[key] = np.concatenate([r[key] for r in rows])
    return out


@pytest.fixture
def fitted_def_kde() -> DefensiveKDE:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    kde = DefensiveKDE(grid=grid, bandwidth=1.5, recency_half_life_days=None)
    shots = _two_opponent_shots()
    kde.fit(x=shots["x"], y=shots["y"], opponent=shots["opponent"])
    return kde


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_negative_bandwidth_raises() -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        DefensiveKDE(grid=CourtGrid(), bandwidth=-1.0)


def test_negative_epsilon_raises() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        DefensiveKDE(grid=CourtGrid(), epsilon=-1e-9)


def test_zero_or_negative_half_life_raises() -> None:
    with pytest.raises(ValueError, match="recency_half_life_days"):
        DefensiveKDE(grid=CourtGrid(), recency_half_life_days=0.0)
    with pytest.raises(ValueError, match="recency_half_life_days"):
        DefensiveKDE(grid=CourtGrid(), recency_half_life_days=-30.0)


def test_unfitted_kde_is_not_fitted() -> None:
    kde = DefensiveKDE(grid=CourtGrid())
    assert not kde.is_fitted
    with pytest.raises(KeyError, match="unknown opponent"):
        kde.density("A")


def test_length_mismatch_raises() -> None:
    kde = DefensiveKDE(grid=CourtGrid())
    with pytest.raises(ValueError, match="length mismatch"):
        kde.fit(x=np.zeros(10), y=np.zeros(5), opponent=np.array(["A"] * 10))


# ---------------------------------------------------------------------------
# Fit + lookup invariants
# ---------------------------------------------------------------------------


def test_density_sums_to_one(fitted_def_kde: DefensiveKDE) -> None:
    for opp in fitted_def_kde.opponents:
        q = fitted_def_kde.density(opp)
        np.testing.assert_allclose(q.sum(), 1.0, atol=1e-12)


def test_density_strictly_positive(fitted_def_kde: DefensiveKDE) -> None:
    for opp in fitted_def_kde.opponents:
        assert (fitted_def_kde.density(opp) > 0).all()


def test_density_image_layout(fitted_def_kde: DefensiveKDE) -> None:
    g = fitted_def_kde.grid
    for opp in fitted_def_kde.opponents:
        assert fitted_def_kde.density(opp).shape == (g.ny, g.nx)


def test_log_density_consistent(fitted_def_kde: DefensiveKDE) -> None:
    for opp in fitted_def_kde.opponents:
        q = fitted_def_kde.density(opp)
        log_q = fitted_def_kde.log_density(opp)
        np.testing.assert_allclose(q, np.exp(log_q), atol=1e-12)


def test_per_opponent_fits_differ(fitted_def_kde: DefensiveKDE) -> None:
    """Two opponents with shifted shot-allowed clouds must produce
    different defensive densities."""
    q_a = fitted_def_kde.density("A")
    q_b = fitted_def_kde.density("B")
    # Sup norm should be substantial since the clouds are 20 ft apart.
    assert np.abs(q_a - q_b).max() > 0.001


def test_unknown_opponent_raises(fitted_def_kde: DefensiveKDE) -> None:
    with pytest.raises(KeyError, match="unknown opponent"):
        fitted_def_kde.density("does-not-exist")


def test_opponents_property_sorted(fitted_def_kde: DefensiveKDE) -> None:
    assert fitted_def_kde.opponents == ("A", "B")


def test_fit_stringifies_opponent_keys() -> None:
    """fit accepts non-string opponent identifiers; lookups string-normalize."""
    kde = DefensiveKDE(grid=CourtGrid(), bandwidth=1.5, recency_half_life_days=None)
    rng = np.random.default_rng(0)
    kde.fit(
        x=rng.normal(0, 2, 100),
        y=rng.normal(10, 2, 100),
        opponent=np.array([7, 7, 7, 11, 11] * 20),
    )
    # Integer keys should be retrievable by their string form.
    assert kde.opponents == ("11", "7")
    np.testing.assert_allclose(kde.density(7), kde.density("7"))


# ---------------------------------------------------------------------------
# Recency
# ---------------------------------------------------------------------------


def test_recency_disabled_by_none(fitted_def_kde: DefensiveKDE) -> None:
    """Default recency_half_life_days=None gives uniform weights."""
    assert fitted_def_kde.recency_half_life_days is None


def test_recency_weighting_changes_density() -> None:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 2, n)
    y = rng.normal(10, 2, n)
    opp = np.array(["A"] * n)
    # Half the shots are "old" (way in the past), half are "recent" but at a
    # different y center — so recency weighting should pull the density toward
    # the recent center.
    dates = np.array(
        ["2020-01-01"] * (n // 2) + ["2024-01-01"] * (n - n // 2),
        dtype="datetime64[D]",
    )
    y[: n // 2] = rng.normal(4, 2, n // 2)
    y[n // 2 :] = rng.normal(20, 2, n - n // 2)

    kde_no_recency = DefensiveKDE(grid=grid, bandwidth=1.5, recency_half_life_days=None)
    kde_no_recency.fit(x=x, y=y, opponent=opp, date=dates)

    kde_with_recency = DefensiveKDE(grid=grid, bandwidth=1.5, recency_half_life_days=180.0)
    kde_with_recency.fit(x=x, y=y, opponent=opp, date=dates)

    q_no = kde_no_recency.density("A")
    q_yes = kde_with_recency.density("A")
    # Recency-weighted density should differ noticeably from uniform-weighted.
    assert np.abs(q_no - q_yes).max() > 1e-3
