"""Tests for ``shotcloud.evaluation.energy_distance``."""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud.evaluation.energy_distance import energy_distance


def test_identical_clouds_u_stat_is_near_zero() -> None:
    """The U-statistic estimate for identical clouds is O(1/m) and near zero.

    When ``x is y`` the estimate is ``-2 * within_mean / m``; for a 200-point
    cloud that is the mean pairwise distance divided by 100.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 2)) * 5.0
    e = energy_distance(x, x)
    # Negative under U-stat when same array passed twice; bounded by
    # ~2 * mean_pairwise / m. For m=200 with ~10 ft mean distances,
    # |e| < 0.2.
    assert abs(e) < 0.2, f"expected |E(X,X)| < 0.2 at m=200; got {e}"


def test_u_stat_x_equals_x_exact_formula() -> None:
    """When ``x is y``, the energy distance equals ``-2 * U_within / m`` exactly."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((50, 2))
    m = x.shape[0]
    diff = x[:, None, :] - x[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    u_within = float(d.sum() / (m * (m - 1)))
    e = energy_distance(x, x)
    expected = -2.0 * u_within / m
    assert abs(e - expected) < 1e-10, (
        f"U-stat E(X, X) should equal -2 U_within / m = {expected}; got {e}"
    )


def test_v_stat_would_inflate_relative_to_u() -> None:
    """The estimator is the unbiased U-statistic, not the upward-biased V-statistic.

    The V-statistic underestimates the within-sample mean distances by a
    factor ``(m-1)/m``, inflating the energy distance by exactly
    ``V_within_x / (m-1) + V_within_y / (n-1)``.
    """
    rng = np.random.default_rng(3)
    m, n = 20, 25
    x = rng.standard_normal((m, 2))
    y = rng.standard_normal((n, 2)) + np.array([2.0, 0.0])
    e_u = energy_distance(x, y, clamp_nonneg=False)
    # Synthesize the V-stat estimator manually.
    cross = float(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).mean())
    within_x_v = float(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1).mean())
    within_y_v = float(np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1).mean())
    e_v = 2.0 * cross - within_x_v - within_y_v
    # V-stat should be UPWARD-biased relative to U-stat.
    assert e_v > e_u, f"V-stat = {e_v:.5f} should exceed U-stat = {e_u:.5f}"
    # The exact bias is U_within_x / m + U_within_y / n.
    # Equivalently V_within_x / (m-1) + V_within_y / (n-1).
    expected_bias = within_x_v / (m - 1) + within_y_v / (n - 1)
    np.testing.assert_allclose(e_v - e_u, expected_bias, rtol=1e-6)


def test_symmetric() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((30, 2)) * 4.0
    y = rng.standard_normal((25, 2)) * 4.0 + np.array([3.0, 0.0])
    assert energy_distance(x, y) == pytest.approx(energy_distance(y, x), abs=1e-9)


def test_increases_with_mean_separation() -> None:
    rng = np.random.default_rng(2)
    base = rng.standard_normal((50, 2))
    x = base.copy()
    y_close = base + np.array([0.5, 0.0])
    y_far = base + np.array([10.0, 0.0])
    e_close = energy_distance(x, y_close)
    e_far = energy_distance(x, y_far)
    assert e_far > e_close
    assert e_close >= 0.0
    assert e_far >= 0.0


def test_negative_unclamped_returns_raw_value() -> None:
    """Negative estimates pass through by default and clamp to 0 with ``clamp_nonneg=True``."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((10, 2))
    y = x.copy()
    e_clamped = energy_distance(x, y, clamp_nonneg=True)
    e_raw = energy_distance(x, y, clamp_nonneg=False)
    # x is content-identical to y → U-stat E ≈ -2 U_within / m < 0.
    assert e_raw < 0.0
    assert e_clamped == 0.0


def test_single_point_returns_nan() -> None:
    """U-statistic within-sample term is undefined for m < 2."""
    x = np.array([[0.0, 0.0]])
    y = np.array([[1.0, 1.0], [2.0, 2.0]])
    assert np.isnan(energy_distance(x, y))
    assert np.isnan(energy_distance(y, x))


def test_empty_inputs_return_nan() -> None:
    x = np.zeros((0, 2))
    y = np.array([[1.0, 2.0]])
    assert np.isnan(energy_distance(x, y))
    assert np.isnan(energy_distance(y, x))


def test_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"x must have shape \(m, 2\)"):
        energy_distance(np.zeros((5, 3)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match=r"y must have shape \(n, 2\)"):
        energy_distance(np.zeros((5, 2)), np.zeros((4,)))


def test_units_in_feet() -> None:
    """Energy distance scales linearly with the cloud coordinates."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((20, 2))
    y = rng.standard_normal((20, 2)) + 2.0
    e_unit = energy_distance(x, y)
    e_scaled = energy_distance(x * 10.0, y * 10.0)
    assert e_scaled == pytest.approx(10.0 * e_unit, rel=1e-9)
