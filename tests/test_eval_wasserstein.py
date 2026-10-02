"""Tests for :mod:`shotcloud.evaluation.wasserstein`."""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud.evaluation.wasserstein import (
    sliced_wasserstein,
    sliced_wasserstein_grid,
    wasserstein_1d,
)

# ---------------------------------------------------------------------------
# wasserstein_1d
# ---------------------------------------------------------------------------


def test_w1d_self_is_zero() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(wasserstein_1d(a, a), 0.0)


def test_w1d_shifted_uniform() -> None:
    """W1 between identical samples shifted by Δ is exactly Δ."""
    a = np.linspace(0, 10, 1000)
    b = a + 3.0
    np.testing.assert_allclose(wasserstein_1d(a, b), 3.0, atol=1e-9)


def test_w1d_handles_unequal_sample_sizes_via_quantiles() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 100)
    b = rng.normal(0, 1, 50)  # different size
    out = wasserstein_1d(a, b)
    assert np.isfinite(out)
    assert out >= 0


def test_w1d_empty_returns_nan() -> None:
    assert np.isnan(wasserstein_1d(np.array([]), np.array([1.0, 2.0])))
    assert np.isnan(wasserstein_1d(np.array([1.0, 2.0]), np.array([])))


# ---------------------------------------------------------------------------
# sliced_wasserstein 2D
# ---------------------------------------------------------------------------


def test_sw_self_is_zero() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 1, (200, 2))
    np.testing.assert_allclose(
        sliced_wasserstein(pts, pts, n_projections=50, seed=0), 0.0, atol=1e-12
    )


def test_sw_shifted_cloud_recovers_translation() -> None:
    """A pure translation by t in 2D should give SW ≈ |t|."""
    rng = np.random.default_rng(0)
    p = rng.normal(0, 1, (1000, 2))
    q = p + np.array([3.0, 0.0])
    sw = sliced_wasserstein(p, q, n_projections=200, seed=0)
    # Mean abs cosine over uniform unit vectors is 2/π ≈ 0.637, so SW ≈ 3 * 2/π ≈ 1.91.
    np.testing.assert_allclose(sw, 3.0 * 2.0 / np.pi, atol=0.1)


def test_sw_is_deterministic_under_seed() -> None:
    rng = np.random.default_rng(0)
    p = rng.normal(0, 1, (200, 2))
    q = rng.normal(0, 1, (200, 2))
    sw1 = sliced_wasserstein(p, q, n_projections=100, seed=42)
    sw2 = sliced_wasserstein(p, q, n_projections=100, seed=42)
    np.testing.assert_allclose(sw1, sw2)


def test_sw_different_seeds_give_close_but_not_identical_values() -> None:
    rng = np.random.default_rng(0)
    p = rng.normal(0, 1, (200, 2))
    q = rng.normal(0, 1, (200, 2))
    sw1 = sliced_wasserstein(p, q, n_projections=20, seed=1)
    sw2 = sliced_wasserstein(p, q, n_projections=20, seed=2)
    # MC estimates with same n_projections — should be close.
    assert abs(sw1 - sw2) < 0.5
    # ...but not identical (unless we got really unlucky).
    assert sw1 != sw2


def test_sw_distinguishes_different_distributions() -> None:
    """SW should be substantially larger between very different clouds."""
    rng = np.random.default_rng(0)
    p_basket = rng.normal([0, 5], 1, (500, 2))
    p_arc = rng.normal([0, 25], 1, (500, 2))
    sw_close = sliced_wasserstein(p_basket, p_basket + np.array([0.0, 0.5]))
    sw_far = sliced_wasserstein(p_basket, p_arc)
    assert sw_far > 5 * sw_close


def test_sw_empty_input_returns_nan() -> None:
    assert np.isnan(sliced_wasserstein(np.zeros((0, 2)), np.ones((10, 2))))
    assert np.isnan(sliced_wasserstein(np.ones((10, 2)), np.zeros((0, 2))))


def test_sw_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="must have shape"):
        sliced_wasserstein(np.array([1.0, 2.0, 3.0]), np.zeros((10, 2)))
    with pytest.raises(ValueError, match="must have shape"):
        sliced_wasserstein(np.zeros((10, 2)), np.zeros((10, 3)))


# ---------------------------------------------------------------------------
# sliced_wasserstein_grid (simplex-over-cells variant)
# ---------------------------------------------------------------------------


def _line_centers(C: int, span: float = 50.0) -> np.ndarray:
    """1D line of cell centers along x, y=0 — simplest case to reason about."""
    return np.stack([np.linspace(-span / 2, span / 2, C), np.zeros(C)], axis=1)


def test_sw_grid_self_is_zero() -> None:
    """``SW(a, a) == 0`` for any simplex distribution over cells."""
    rng = np.random.default_rng(0)
    a = rng.dirichlet(np.ones(40)).astype(np.float64)
    centers = _line_centers(40)
    np.testing.assert_allclose(
        sliced_wasserstein_grid(a, a, centers, n_projections=20), 0.0, atol=1e-12
    )


def test_sw_grid_recovers_translation_on_line() -> None:
    """Two delta-like masses placed at known x-positions on a line of
    cell centers: SW grid should recover roughly the spatial gap."""
    C = 40
    centers = _line_centers(C, span=50.0)
    # Cell spacing is 50 / (C - 1) ~ 1.28 ft
    a = np.zeros(C, dtype=np.float64)
    b = np.zeros(C, dtype=np.float64)
    a[5] = 1.0  # at x ~ -43*50/39/2 ish
    b[25] = 1.0
    expected_dx = float(centers[25, 0] - centers[5, 0])
    sw = sliced_wasserstein_grid(a, b, centers, n_projections=200, seed=0)
    # SW averages |projection_distance| over directions — for a y=0 line
    # of centers and uniform random directions, the expected scaling is
    # E[|cos theta|] * dx = (2/pi) * dx for a strict 1D support.
    expected_sw = 2.0 / np.pi * expected_dx
    assert abs(sw - expected_sw) < 0.5


def test_sw_grid_shape_validation() -> None:
    """Mismatched a/b or wrong cell_centers shape raise."""
    centers = _line_centers(20)
    a = np.full(20, 1.0 / 20)
    with pytest.raises(ValueError, match="shapes must match"):
        sliced_wasserstein_grid(a, np.full(15, 1.0 / 15), centers)
    with pytest.raises(ValueError, match="cell_centers"):
        sliced_wasserstein_grid(a, a, np.zeros((25, 2)))
    with pytest.raises(ValueError, match="1-D"):
        sliced_wasserstein_grid(np.zeros((4, 5)), np.zeros((4, 5)), centers)
