"""Tests for :class:`shotcloud.legacy_pivot.timing.ConstantRateTimingModel`."""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud.legacy_pivot.timing import ConstantRateTimingModel


def test_construction_validates_args() -> None:
    with pytest.raises(ValueError, match="mean_shots"):
        ConstantRateTimingModel(mean_shots=-1.0)
    with pytest.raises(ValueError, match="game_length"):
        ConstantRateTimingModel(game_length=0.0)
    with pytest.raises(ValueError, match="game_length"):
        ConstantRateTimingModel(game_length=-5.0)


def test_sample_returns_sane_K_and_taus() -> None:
    timing = ConstantRateTimingModel(mean_shots=20.0, game_length=48.0)
    rng = np.random.default_rng(0)
    K, taus = timing.sample({}, rng)

    assert isinstance(K, int)
    assert K >= 0
    assert taus.shape == (K,)
    assert taus.dtype == np.float64
    if K > 0:
        assert taus.min() >= 0.0
        assert taus.max() <= 48.0


def test_sample_returns_sorted_taus() -> None:
    timing = ConstantRateTimingModel(mean_shots=30.0, game_length=48.0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        _, taus = timing.sample({}, rng)
        assert np.all(np.diff(taus) >= 0)


def test_empirical_mean_K_matches_mean_shots() -> None:
    timing = ConstantRateTimingModel(mean_shots=18.0, game_length=48.0)
    rng = np.random.default_rng(0)
    Ks = np.array([timing.sample({}, rng)[0] for _ in range(2000)])
    # Poisson SE with n=2000 trials: sqrt(λ/n) ≈ sqrt(18/2000) ≈ 0.095.
    np.testing.assert_allclose(Ks.mean(), 18.0, atol=0.5)


def test_sample_is_deterministic_under_seeded_rng() -> None:
    timing = ConstantRateTimingModel(mean_shots=20.0, game_length=48.0)
    K1, t1 = timing.sample({}, np.random.default_rng(42))
    K2, t2 = timing.sample({}, np.random.default_rng(42))
    assert K1 == K2
    np.testing.assert_array_equal(t1, t2)


# ---------------------------------------------------------------------------
# log_prob
# ---------------------------------------------------------------------------


def test_log_prob_is_finite_and_non_positive() -> None:
    timing = ConstantRateTimingModel(mean_shots=20.0, game_length=48.0)
    rng = np.random.default_rng(0)
    K, taus = timing.sample({}, rng)
    lp = timing.log_prob(K, taus, {})
    assert np.isfinite(lp)
    assert lp <= 0.0


def test_log_prob_for_K_zero() -> None:
    """K=0 should give log P(K=0) = -mean_shots, with no tau contribution."""
    timing = ConstantRateTimingModel(mean_shots=10.0, game_length=48.0)
    lp = timing.log_prob(0, np.array([], dtype=np.float64), {})
    np.testing.assert_allclose(lp, -10.0, atol=1e-10)


def test_log_prob_decomposes_into_count_plus_uniform_density() -> None:
    """Manual check: log p(K, τ) = log Poisson(K; μ) − K log T."""
    from scipy.stats import poisson

    timing = ConstantRateTimingModel(mean_shots=20.0, game_length=48.0)
    K = 5
    taus = np.array([2.0, 8.0, 19.0, 31.0, 44.0])
    lp = timing.log_prob(K, taus, {})

    expected = float(poisson.logpmf(K, 20.0)) + (-5.0 * np.log(48.0))
    np.testing.assert_allclose(lp, expected, atol=1e-10)


def test_log_prob_raises_on_length_mismatch() -> None:
    timing = ConstantRateTimingModel()
    with pytest.raises(ValueError, match="length mismatch"):
        timing.log_prob(3, np.array([1.0, 2.0]), {})


def test_log_prob_raises_on_out_of_bounds_taus() -> None:
    timing = ConstantRateTimingModel(game_length=48.0)
    with pytest.raises(ValueError, match="taus must lie"):
        timing.log_prob(2, np.array([-1.0, 24.0]), {})
    with pytest.raises(ValueError, match="taus must lie"):
        timing.log_prob(2, np.array([24.0, 50.0]), {})


def test_log_prob_raises_on_negative_K() -> None:
    timing = ConstantRateTimingModel()
    with pytest.raises(ValueError, match="non-negative"):
        timing.log_prob(-1, np.array([], dtype=np.float64), {})
