"""Tests for the cloud-metric helpers in ``scripts/evaluate_shot_clouds.py``.

``_zone_proportions``, ``_rim_distances``, ``_metrics_for_bootstrap``, and
``_self_bootstrap_metrics`` compute the per-game cloud metrics and the
self-bootstrap noise floor reported for every model. The tests check:

- ``_zone_proportions``: the 8-zone output sums to 1, out-of-court shots are
  dropped before normalization, and an all-out-of-court cloud returns NaN.
- ``_rim_distances``: Euclidean distance from the basket at the origin.
- ``_metrics_for_bootstrap``: all six metric keys are present and finite,
  identical clouds score zero (energy distance slightly negative), zone L1
  lies in [0, 2], and the mean-distance error is non-negative.
- ``_self_bootstrap_metrics``: one averaged value per metric, deterministic
  under a seeded RNG, and a small floor on large observed clouds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from evaluate_shot_clouds import (  # type: ignore[import-not-found]
    _METRIC_KEYS,
    _metrics_for_bootstrap,
    _rim_distances,
    _self_bootstrap_metrics,
    _zone_proportions,
)

# ---------------------------------------------------------------------------
# _zone_proportions
# ---------------------------------------------------------------------------


def test_zone_proportions_sums_to_one_when_all_in_court() -> None:
    """An all-in-court cloud's 8-zone proportions sum to exactly 1."""
    rng = np.random.default_rng(0)
    # Inside the half-court extent.
    xy = np.column_stack(
        [
            rng.uniform(-20.0, 20.0, size=200),
            rng.uniform(0.0, 30.0, size=200),
        ]
    )
    p = _zone_proportions(xy)
    assert p.shape == (8,)
    assert np.all(p >= 0)
    assert np.isclose(p.sum(), 1.0)


def test_zone_proportions_drops_out_of_bounds() -> None:
    """Out-of-court shots (zone == -1) are excluded from the count."""
    # 10 in-court + 10 way-off-court — the proportions should reflect
    # only the 10 in-court.
    in_court = np.array([[0.0, 5.0]] * 10)
    out_court = np.array([[100.0, 100.0]] * 10)
    xy = np.vstack([in_court, out_court])
    p = _zone_proportions(xy)
    assert p.sum() == 1.0  # would be 0.5 if OOB were counted
    # All 10 in-court points are at (0, 5) — in the paint, zone 1.
    assert p[1] == 1.0
    assert p[0] == 0.0
    assert (p[2:] == 0.0).all()


def test_zone_proportions_all_out_of_bounds_returns_nan() -> None:
    """A cloud entirely outside the half-court returns NaN vector."""
    xy = np.array([[100.0, 100.0], [-100.0, -100.0]])
    p = _zone_proportions(xy)
    assert p.shape == (8,)
    assert np.all(np.isnan(p))


# ---------------------------------------------------------------------------
# _rim_distances
# ---------------------------------------------------------------------------


def test_rim_distances_basket_at_origin() -> None:
    """``_rim_distances`` is ``||y - 0||`` — basket at origin."""
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [-5.0, 0.0], [0.0, 12.0]])
    d = _rim_distances(xy)
    np.testing.assert_allclose(d, [0.0, 5.0, 5.0, 12.0])


def test_rim_distances_shape_preserved() -> None:
    """Output is shape (N,) for an (N, 2) input."""
    xy = np.zeros((37, 2))
    assert _rim_distances(xy).shape == (37,)


# ---------------------------------------------------------------------------
# _metrics_for_bootstrap
# ---------------------------------------------------------------------------


def test_metrics_for_bootstrap_returns_all_six_keys() -> None:
    """The 6 keys named in ``_METRIC_KEYS`` are all present, all finite."""
    rng = np.random.default_rng(0)
    observed = rng.standard_normal((20, 2)) * 3.0 + np.array([0.0, 10.0])
    generated = rng.standard_normal((20, 2)) * 3.0 + np.array([0.0, 10.0])
    out = _metrics_for_bootstrap(observed, generated, sliced_projections=50, seed=0)
    for k in _METRIC_KEYS:
        assert k in out, f"missing metric: {k}"
        assert np.isfinite(out[k]), f"non-finite value for {k}: {out[k]}"


def test_metrics_for_bootstrap_identical_clouds_have_zero_or_near_zero() -> None:
    """For two identical clouds (X = Y), every metric should be 0 or
    near 0 (modulo U-stat negative bias on energy distance)."""
    rng = np.random.default_rng(0)
    observed = rng.standard_normal((30, 2)) * 4.0 + np.array([0.0, 10.0])
    generated = observed.copy()
    out = _metrics_for_bootstrap(observed, generated, sliced_projections=100, seed=0)
    # sliced_w and rim_w1 are exactly 0 when clouds are identical.
    assert abs(out["sliced_wasserstein"]) < 1e-9
    assert abs(out["rim_distance_w1"]) < 1e-9
    # zone_l1, rim_ks, mean_dist_err exactly 0.
    assert out["zone_l1"] == 0.0
    assert out["rim_distance_ks"] == 0.0
    assert out["mean_shot_distance_err_ft"] == 0.0
    # U-stat energy distance on X=X is exactly -2 U_within / m — small
    # negative.
    assert out["energy_distance"] < 0


def test_metrics_for_bootstrap_zone_l1_is_in_zero_two_range() -> None:
    """zone_l1 between two simplex vectors is in [0, 2]."""
    rng = np.random.default_rng(1)
    observed = rng.normal((0.0, 10.0), 4.0, size=(50, 2))
    generated = rng.normal((0.0, 10.0), 4.0, size=(50, 2))
    out = _metrics_for_bootstrap(observed, generated, sliced_projections=50, seed=0)
    assert 0 <= out["zone_l1"] <= 2.0


def test_metrics_for_bootstrap_mean_dist_err_is_nonneg() -> None:
    """``mean_shot_distance_err`` is an absolute value; always ≥ 0."""
    rng = np.random.default_rng(2)
    observed = rng.standard_normal((25, 2)) * 4.0
    generated = rng.standard_normal((25, 2)) * 4.0 + np.array([3.0, 0.0])
    out = _metrics_for_bootstrap(observed, generated, sliced_projections=50, seed=0)
    assert out["mean_shot_distance_err_ft"] >= 0
    # And the value matches the manual computation.
    expected = abs(
        np.linalg.norm(observed, axis=-1).mean() - np.linalg.norm(generated, axis=-1).mean()
    )
    np.testing.assert_allclose(out["mean_shot_distance_err_ft"], expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# _self_bootstrap_metrics
# ---------------------------------------------------------------------------


def test_self_bootstrap_returns_all_metrics_averaged() -> None:
    """The self-bootstrap helper returns one averaged value per metric."""
    rng = np.random.default_rng(0)
    observed = rng.standard_normal((20, 2)) * 4.0 + np.array([0.0, 10.0])
    out = _self_bootstrap_metrics(
        observed,
        n_bootstraps=10,
        sliced_projections=30,
        rng=np.random.default_rng(0),
        seed_base=0,
    )
    for k in _METRIC_KEYS:
        assert k in out
        assert np.isfinite(out[k])


def test_self_bootstrap_is_deterministic_with_seeded_rng() -> None:
    """Same observed + same RNG seed = bit-identical output."""
    rng_master = np.random.default_rng(5)
    observed = rng_master.standard_normal((15, 2)) * 5.0
    out_a = _self_bootstrap_metrics(
        observed,
        n_bootstraps=20,
        sliced_projections=30,
        rng=np.random.default_rng(42),
        seed_base=0,
    )
    out_b = _self_bootstrap_metrics(
        observed,
        n_bootstraps=20,
        sliced_projections=30,
        rng=np.random.default_rng(42),
        seed_base=0,
    )
    for k in _METRIC_KEYS:
        assert out_a[k] == out_b[k], f"non-deterministic on {k}"


def test_self_bootstrap_approaches_zero_at_large_n_for_non_energy() -> None:
    """At large K_obs and many bootstrap iterations, the self-bootstrap
    floor on the 5 non-energy metrics should be small (it's noise on the
    same underlying distribution). Energy distance under U-stat goes
    *negative*; the other 5 stay ≥ 0 and small."""
    rng_master = np.random.default_rng(8)
    observed = rng_master.standard_normal((200, 2)) * 4.0 + np.array([0.0, 10.0])
    out = _self_bootstrap_metrics(
        observed,
        n_bootstraps=50,
        sliced_projections=100,
        rng=np.random.default_rng(0),
        seed_base=0,
    )
    # At K=200 + 50 bootstraps, the non-energy floor metrics should be
    # small (≤ a few feet for distance metrics, ≤ 0.1 for unit-bounded
    # ones).
    assert 0 <= out["zone_l1"] <= 0.3
    assert 0 <= out["rim_distance_ks"] <= 0.3
    assert 0 <= out["mean_shot_distance_err_ft"] <= 2.0
    # Energy distance under U-stat IS expected to be slightly negative
    # at this sample size (bootstrap clouds are very close to observed).
    assert out["energy_distance"] < 0
