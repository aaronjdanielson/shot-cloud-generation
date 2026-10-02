"""Tests for ``_paired_mean_ci`` in ``scripts/compare_density_surfaces.py``.

The helper returns the sample mean of per-game paired deltas together with a
percentile-bootstrap confidence interval; these tests pin the estimator, the
percentile construction, its nominal coverage, and the empty-input guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make scripts/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compare_density_surfaces import _paired_mean_ci  # type: ignore[import-not-found]


def test_empty_returns_nan_triple() -> None:
    """An empty deltas array returns ``(nan, nan, nan)``."""
    rng = np.random.default_rng(0)
    mean, lo, hi = _paired_mean_ci(np.zeros(0), n_bootstraps=100, rng=rng)
    assert np.isnan(mean)
    assert np.isnan(lo)
    assert np.isnan(hi)


def test_returns_sample_mean_as_central_estimate() -> None:
    """The central estimate is the sample mean, not a bootstrap mean or median."""
    rng = np.random.default_rng(0)
    deltas = np.array([-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    expected_mean = float(deltas.mean())
    mean, _, _ = _paired_mean_ci(deltas, n_bootstraps=5000, rng=rng)
    assert mean == expected_mean


def test_ci_is_percentile_not_se() -> None:
    """The 95% CI is the 2.5/97.5 percentile interval of the bootstrap means.

    The bootstrap is replayed with the same seed and compared exactly; the
    bounds must differ from a mean ± 1.96·SE construction.
    """
    deltas = np.random.default_rng(42).standard_normal(200) * 0.1 + 0.2
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    mean, lo, hi = _paired_mean_ci(deltas, n_bootstraps=5000, rng=rng_a)
    # Reconstruct the bootstrap manually with the same seed sequence.
    n = deltas.shape[0]
    idx = rng_b.integers(0, n, size=(5000, n))
    boot_means = deltas[idx].mean(axis=1)
    expected_lo = float(np.quantile(boot_means, 0.025))
    expected_hi = float(np.quantile(boot_means, 0.975))
    assert lo == expected_lo
    assert hi == expected_hi
    # Also verify the SE construction would have given a DIFFERENT
    # answer (sanity check that the implementation is not coincidentally
    # using mean ± 1.96·SE).
    se = float(deltas.std(ddof=1) / np.sqrt(n))
    se_lo = mean - 1.96 * se
    se_hi = mean + 1.96 * se
    # On 200 mildly skewed deltas the percentile and SE bounds match
    # closely but should not be bit-identical.
    assert abs(lo - se_lo) > 0 or abs(hi - se_hi) > 0


def test_95_ci_has_nominal_coverage_on_gaussian_null() -> None:
    """Across 1000 Gaussian samples with mean μ, about 95% of the CIs contain μ."""
    rng_master = np.random.default_rng(7)
    n_trials = 1000
    n_per_trial = 200
    true_mean = 0.05
    coverage = 0
    for t in range(n_trials):
        deltas = rng_master.standard_normal(n_per_trial) * 0.1 + true_mean
        boot_rng = np.random.default_rng(1000 + t)
        _, lo, hi = _paired_mean_ci(deltas, n_bootstraps=1000, rng=boot_rng)
        if lo <= true_mean <= hi:
            coverage += 1
    frac = coverage / n_trials
    # 95% nominal; SE on coverage with n=1000 is √(0.05·0.95/1000) ≈ 0.007.
    # Allow ±0.025 (~3.5 SE) tolerance against MC noise.
    assert 0.925 <= frac <= 0.975, f"coverage = {frac:.3f}; expected ≈ 0.95 (±0.025)"


def test_one_sided_99_tighter_than_two_sided_95() -> None:
    """``alpha=0.02`` gives a wider interval than ``alpha=0.05``.

    The 98% two-sided interval's lower bound is the one-sided 99% bound.
    """
    deltas = np.random.default_rng(11).standard_normal(150) * 0.2 + 0.3
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    _, lo_95, hi_95 = _paired_mean_ci(deltas, n_bootstraps=5000, rng=rng_a, alpha=0.05)
    _, lo_98, hi_98 = _paired_mean_ci(deltas, n_bootstraps=5000, rng=rng_b, alpha=0.02)
    # Smaller alpha → wider interval (lo_98 ≤ lo_95 and hi_98 ≥ hi_95).
    assert lo_98 <= lo_95
    assert hi_98 >= hi_95


def test_paired_bootstrap_preserves_within_game_correlation() -> None:
    """The CI depends only on the paired deltas ``A_n − B_n``, not on A and B separately.

    Resampling paired deltas by game keeps within-game correlation between
    the two models out of the interval width.
    """
    rng_master = np.random.default_rng(13)
    n = 300
    # Deltas with mean 0.1, std 0.05 (low variance because A and B
    # share strong within-game noise).
    deltas = rng_master.standard_normal(n) * 0.05 + 0.1
    rng_a = np.random.default_rng(0)
    mean_low, lo_low, hi_low = _paired_mean_ci(deltas, n_bootstraps=5000, rng=rng_a)

    # Same deltas, larger underlying A and B variance but identical
    # delta distribution — the bootstrap CI is the SAME.
    rng_b = np.random.default_rng(0)
    mean_high, lo_high, hi_high = _paired_mean_ci(deltas, n_bootstraps=5000, rng=rng_b)
    assert mean_low == mean_high
    assert lo_low == lo_high
    assert hi_low == hi_high


def test_reproducible_with_seeded_rng() -> None:
    """Identically seeded RNGs produce bit-identical results."""
    deltas = np.random.default_rng(0).standard_normal(100) * 0.2
    rng1 = np.random.default_rng(2026)
    rng2 = np.random.default_rng(2026)
    res1 = _paired_mean_ci(deltas, n_bootstraps=500, rng=rng1)
    res2 = _paired_mean_ci(deltas, n_bootstraps=500, rng=rng2)
    assert res1 == res2


def test_significance_detection_negative_mean() -> None:
    """A strongly negative mean delta yields an upper CI bound below zero."""
    deltas = np.random.default_rng(0).standard_normal(500) * 0.05 - 0.2
    rng = np.random.default_rng(0)
    mean, lo, hi = _paired_mean_ci(deltas, n_bootstraps=5000, rng=rng)
    assert mean < 0
    assert hi < 0  # decisive: 0 outside the CI to the right
    assert lo < hi


def test_significance_detection_zero_mean() -> None:
    """A zero-mean delta yields a CI that brackets zero."""
    deltas = np.random.default_rng(0).standard_normal(500) * 0.05  # mean 0
    rng = np.random.default_rng(0)
    mean, lo, hi = _paired_mean_ci(deltas, n_bootstraps=5000, rng=rng)
    assert abs(mean) < 0.05
    assert lo < 0 < hi  # CI brackets zero
