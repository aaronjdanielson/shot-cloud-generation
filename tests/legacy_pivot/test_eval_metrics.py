"""Tests for :mod:`shotcloud.legacy_pivot.eval_metrics` (zone distributions + KL)."""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud.legacy_pivot.eval_metrics import (
    N_ZONES_5,
    ZONE_NAMES_5,
    aggregate_metric,
    kde_gain,
    zone_distribution_5,
    zone_distribution_8,
    zone_kl_divergence,
)

# ---------------------------------------------------------------------------
# Zone distributions
# ---------------------------------------------------------------------------


def test_zone_distribution_8_sums_to_one_for_in_court_shots() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-15, 15, 200)
    y = rng.uniform(0, 25, 200)
    freqs = zone_distribution_8(x, y)
    assert freqs.shape == (8,)
    np.testing.assert_allclose(freqs.sum(), 1.0)


def test_zone_distribution_5_sums_to_one() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-15, 15, 200)
    y = rng.uniform(0, 25, 200)
    freqs = zone_distribution_5(x, y)
    assert freqs.shape == (N_ZONES_5,)
    np.testing.assert_allclose(freqs.sum(), 1.0)


def test_empty_input_returns_zero_vector() -> None:
    out_8 = zone_distribution_8(np.array([]), np.array([]))
    out_5 = zone_distribution_5(np.array([]), np.array([]))
    np.testing.assert_array_equal(out_8, np.zeros(8))
    np.testing.assert_array_equal(out_5, np.zeros(N_ZONES_5))


def test_only_basket_shots_concentrate_in_RA() -> None:
    """All shots at the basket → 100% Restricted Area (zone 0 in 8-zone, "RA" in 5-zone)."""
    x = np.zeros(50)
    y = np.zeros(50)
    eight = zone_distribution_8(x, y)
    five = zone_distribution_5(x, y)
    assert eight[0] == 1.0
    assert five[0] == 1.0  # "RA" is index 0 in ZONE_NAMES_5
    np.testing.assert_allclose(eight[1:].sum(), 0.0)


def test_5_zone_aggregates_8_zone_correctly() -> None:
    """The 5-zone collapse is a deterministic remap of the 8-zone."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-25, 25, 500)
    y = rng.uniform(-5, 30, 500)
    eight = zone_distribution_8(x, y)
    five = zone_distribution_5(x, y)
    # Above-Break 3 (5-zone idx 3) = zones 5+6+7 from 8-zone.
    np.testing.assert_allclose(five[3], eight[5] + eight[6] + eight[7])
    # Corner 3 (5-zone idx 4) = zones 3+4 from 8-zone.
    np.testing.assert_allclose(five[4], eight[3] + eight[4])


def test_zone_names_5_are_paper_canonical() -> None:
    assert ZONE_NAMES_5 == ("RA", "Paint", "Mid-Range", "Above-Break 3", "Corner 3")


def test_backcourt_shots_excluded_from_distribution() -> None:
    """Out-of-bounds shots (zone -1) shouldn't inflate or shift the distribution."""
    x = np.array([0.0, 0.0, 0.0])  # 1 RA, 1 Above-Break 3, 1 backcourt heave
    y = np.array([0.0, 25.0, 50.0])  # last is y > 47 → backcourt
    eight = zone_distribution_8(x, y)
    # Two valid shots (RA and Above-Break 3 zone 7), excluding the backcourt heave.
    np.testing.assert_allclose(eight[0], 0.5)  # RA
    np.testing.assert_allclose(eight[7], 0.5)  # Top of Key 3
    np.testing.assert_allclose(eight.sum(), 1.0)


# ---------------------------------------------------------------------------
# Zone KL divergence
# ---------------------------------------------------------------------------


def test_zone_kl_self_is_zero() -> None:
    p = np.array([0.4, 0.2, 0.1, 0.2, 0.1])
    np.testing.assert_allclose(zone_kl_divergence(p, p), 0.0, atol=1e-9)


def test_zone_kl_is_non_negative() -> None:
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5))
    q = rng.dirichlet(np.ones(5))
    assert zone_kl_divergence(p, q) >= 0.0


def test_zone_kl_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        zone_kl_divergence(np.zeros(5), np.zeros(8))


def test_zone_kl_smoothing_finite_for_zero_components() -> None:
    """A zone present in gen but not real shouldn't produce inf."""
    gen = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    real = np.array([0.0, 1.0, 0.0, 0.0, 0.0])  # all in Paint
    kl = zone_kl_divergence(gen, real)
    assert np.isfinite(kl)
    assert kl > 0.0


def test_zone_kl_matches_manual_calculation() -> None:
    """Manual KL on a tiny example to lock the convention."""
    gen = np.array([0.5, 0.5])
    real = np.array([0.25, 0.75])
    kl_computed = zone_kl_divergence(gen, real, eps=0.0)  # disable smoothing for exactness
    expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
    np.testing.assert_allclose(kl_computed, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# kde_gain
# ---------------------------------------------------------------------------


def test_kde_gain_positive_when_model_improves() -> None:
    assert kde_gain(0.5, 0.3) == pytest.approx(0.2)


def test_kde_gain_negative_when_model_worse() -> None:
    assert kde_gain(0.3, 0.5) == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# aggregate_metric
# ---------------------------------------------------------------------------


def test_aggregate_metric_basic() -> None:
    out = aggregate_metric([1.0, 2.0, 3.0, 4.0, 5.0])
    assert out["mean"] == pytest.approx(3.0)
    assert out["median"] == pytest.approx(3.0)
    assert out["n"] == 5


def test_aggregate_metric_drops_nans() -> None:
    out = aggregate_metric([1.0, float("nan"), 3.0, float("inf")])
    assert out["n"] == 2  # only 1.0 and 3.0 are finite
    assert out["mean"] == pytest.approx(2.0)


def test_aggregate_metric_empty() -> None:
    out = aggregate_metric([])
    assert out["n"] == 0
    assert np.isnan(out["mean"])
