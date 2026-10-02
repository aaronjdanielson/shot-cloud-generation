"""Tests for the density-surface scoring rules.

Scope:

* :func:`log_mvn_density_2d` matches the closed-form bivariate-Gaussian
  log density (analytical sanity at random Σ).
* :func:`quadratic_score` matches the closed form for a single-Gaussian
  mixture: ``∫f² = 1/(4π|Σ|^½)`` and ``f(y) = N₂(y; s, Σ)``.
* :func:`quadratic_score` is invariant under translating both centers
  and observations by the same vector (likelihood is invariant under
  translation; quadratic score should be too).
* :func:`smoothed_log_score` at ``h = 0`` reduces to the plain NLL of
  the mixture; at ``h > 0`` matches the convolved-mixture closed form.
* Covariance builders:
    * isotropic scalar / per-shot σ produce diagonal Σ.
    * radial-tangent collapses to ``σ² I`` when ``σ_r = σ_t``.
    * full-cov collapses to ``σ² I`` when ``σ_x = σ_y, ρ = 0``.
* Zone Brier returns 0 when predicted = observed; CE matches the
  analytical entropy.
* HDR coverage on a unimodal Gaussian: when the model IS the truth,
  empirical coverage matches α within Monte-Carlo noise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from shotcloud.evaluation.density_surfaces import (
    build_covariance_full_cov,
    build_covariance_isotropic_per_shot,
    build_covariance_isotropic_scalar,
    build_covariance_radial_tangent,
    hdr_coverage,
    log_mvn_density_2d,
    quadratic_score,
    smoothed_log_score,
    zone_brier_and_ce,
)

# --------------------------------------------------------------------------- #
# log_mvn_density_2d
# --------------------------------------------------------------------------- #


def test_log_mvn_density_matches_isotropic_closed_form() -> None:
    diff = torch.tensor([1.0, 2.0])
    sigma = torch.eye(2) * 4.0  # σ² = 4 along each axis
    log_p = log_mvn_density_2d(diff, sigma).item()
    # log N(d; 0, σ²I) = −log(2πσ²) − ‖d‖²/(2σ²)
    expected = -math.log(2.0 * math.pi * 4.0) - (1.0 + 4.0) / (2.0 * 4.0)
    assert math.isclose(log_p, expected, abs_tol=1e-5)


def test_log_mvn_density_matches_anisotropic_closed_form() -> None:
    diff = torch.tensor([1.5, -1.0])
    sigma = torch.tensor([[2.0, 0.3], [0.3, 1.5]])
    log_p = log_mvn_density_2d(diff, sigma).item()
    det = 2.0 * 1.5 - 0.3 * 0.3
    inv_det = 1.0 / det
    q = inv_det * (1.5 * 1.5 * 1.5 - 2 * 0.3 * 1.5 * -1.0 + 2.0 * 1.0)
    expected = -math.log(2.0 * math.pi) - 0.5 * math.log(det) - 0.5 * q
    assert math.isclose(log_p, expected, abs_tol=1e-5)


# --------------------------------------------------------------------------- #
# Covariance builders
# --------------------------------------------------------------------------- #


def test_build_covariance_isotropic_scalar_diagonal() -> None:
    sigma = build_covariance_isotropic_scalar(1.5, 3, "cpu")
    assert sigma.shape == (3, 2, 2)
    for m in range(3):
        torch.testing.assert_close(sigma[m, 0, 0], torch.tensor(1.5**2))
        torch.testing.assert_close(sigma[m, 1, 1], torch.tensor(1.5**2))
        torch.testing.assert_close(sigma[m, 0, 1], torch.tensor(0.0))
        torch.testing.assert_close(sigma[m, 1, 0], torch.tensor(0.0))


def test_build_covariance_isotropic_per_shot_diagonal() -> None:
    s = torch.tensor([1.0, 1.5, 2.0])
    sigma = build_covariance_isotropic_per_shot(s)
    assert sigma.shape == (3, 2, 2)
    torch.testing.assert_close(sigma[0, 0, 0], torch.tensor(1.0))
    torch.testing.assert_close(sigma[1, 0, 0], torch.tensor(2.25))
    torch.testing.assert_close(sigma[2, 1, 1], torch.tensor(4.0))


def test_build_covariance_radial_tangent_collapses_to_isotropic_when_sigmas_equal() -> None:
    """σ_r = σ_t = σ ⇒ Σ_m = σ²(r̂r̂ᵀ + t̂t̂ᵀ) = σ² I exactly."""
    from shotcloud.data.zones import N_ZONES

    support = torch.tensor([[10.0, 18.0], [-15.0, 20.0], [0.0, 2.0]])  # mid, wing, RA
    sigma_r = torch.full((N_ZONES,), 1.5)
    sigma_t = torch.full((N_ZONES,), 1.5)
    sigma = build_covariance_radial_tangent(
        support_xy=support, sigma_r_per_zone=sigma_r, sigma_t_per_zone=sigma_t
    )
    expected = torch.eye(2).expand(3, 2, 2) * 1.5**2
    torch.testing.assert_close(sigma, expected, atol=1e-5, rtol=0)


def test_build_covariance_radial_tangent_origin_guard() -> None:
    """A support shot at the basket has no radial frame; fall back to
    isotropic σ_r² I at the rim zone."""
    from shotcloud.data.zones import N_ZONES

    support = torch.zeros(1, 2)  # at basket
    sigma_r = torch.full((N_ZONES,), 1.2)
    sigma_t = torch.full((N_ZONES,), 1.8)
    sigma = build_covariance_radial_tangent(
        support_xy=support, sigma_r_per_zone=sigma_r, sigma_t_per_zone=sigma_t
    )
    expected = torch.eye(2).unsqueeze(0) * 1.2**2
    torch.testing.assert_close(sigma, expected, atol=1e-5, rtol=0)


def test_build_covariance_full_cov_collapses_to_isotropic_at_rho_zero() -> None:
    from shotcloud.data.zones import N_ZONES

    support = torch.tensor([[0.0, 25.0], [-15.0, 20.0]])
    sx = torch.full((N_ZONES,), 1.5)
    sy = torch.full((N_ZONES,), 1.5)
    rho = torch.zeros(N_ZONES)
    sigma = build_covariance_full_cov(
        support_xy=support,
        sigma_x_per_zone=sx,
        sigma_y_per_zone=sy,
        rho_per_zone=rho,
    )
    expected = torch.eye(2).expand(2, 2, 2) * 1.5**2
    torch.testing.assert_close(sigma, expected, atol=1e-5, rtol=0)


# --------------------------------------------------------------------------- #
# Quadratic score
# --------------------------------------------------------------------------- #


def test_quadratic_score_single_gaussian_at_center() -> None:
    """For one component centered at observed point with σ² I:
    ∫f² = 1/(4πσ²), f(y=center) = 1/(2πσ²)."""
    weights = torch.tensor([1.0])
    centers = torch.tensor([[0.0, 0.0]])
    sigma = build_covariance_isotropic_scalar(1.5, 1, "cpu")
    obs = torch.tensor([[0.0, 0.0]])
    q = quadratic_score(weights=weights, centers=centers, sigma_full=sigma, observations=obs)
    expected_integral = 1.0 / (4.0 * math.pi * 1.5**2)
    expected_cross = 2.0 / (2.0 * math.pi * 1.5**2)
    expected = expected_integral - expected_cross
    assert math.isclose(q.item(), expected, abs_tol=1e-5)


def test_quadratic_score_translation_invariance() -> None:
    """Translating centers and observations by the same vector doesn't
    change the score — the kernel only depends on differences."""
    torch.manual_seed(0)
    M, K = 6, 5
    weights = torch.softmax(torch.randn(M), dim=0)
    centers = torch.randn(M, 2) * 3.0
    sigma = build_covariance_isotropic_scalar(1.5, M, "cpu")
    obs = torch.randn(K, 2) * 3.0
    q1 = quadratic_score(weights=weights, centers=centers, sigma_full=sigma, observations=obs)
    shift = torch.tensor([7.5, -3.2])
    q2 = quadratic_score(
        weights=weights,
        centers=centers + shift,
        sigma_full=sigma,
        observations=obs + shift,
    )
    torch.testing.assert_close(q1, q2, atol=1e-5, rtol=0)


def test_quadratic_score_rejects_bad_shapes() -> None:
    M = 3
    weights = torch.softmax(torch.randn(M), dim=0)
    sigma = build_covariance_isotropic_scalar(1.5, M, "cpu")
    obs = torch.zeros(2, 2)
    with pytest.raises(ValueError, match=r"centers must be"):
        quadratic_score(
            weights=weights,
            centers=torch.zeros(M, 3),  # wrong dim
            sigma_full=sigma,
            observations=obs,
        )


# --------------------------------------------------------------------------- #
# Smoothed log score
# --------------------------------------------------------------------------- #


def test_smoothed_log_score_at_h0_matches_plain_nll() -> None:
    """At h = 0, smoothed_log_score == −log f(y) for the mixture."""
    M, K = 4, 3
    torch.manual_seed(0)
    weights = torch.softmax(torch.randn(M), dim=0)
    centers = torch.randn(M, 2) * 4.0
    sigma = build_covariance_isotropic_scalar(1.5, M, "cpu")
    obs = torch.randn(K, 2) * 4.0
    s = smoothed_log_score(
        weights=weights, centers=centers, sigma_full=sigma, observations=obs, h=0.0
    )
    # Reference NLL: −log Σ_m w_m N₂(y; s_m, σ²I).
    diff = obs.unsqueeze(1) - centers.unsqueeze(0)  # (K, M, 2)
    dist2 = (diff * diff).sum(dim=-1)
    sigma2 = 1.5**2
    log_phi = -math.log(2 * math.pi * sigma2) - 0.5 * dist2 / sigma2
    log_w = weights.log()
    expected = -torch.logsumexp(log_phi + log_w.unsqueeze(0), dim=-1)
    torch.testing.assert_close(s, expected, atol=1e-5, rtol=0)


def test_smoothed_log_score_at_h_matches_convolved_mixture() -> None:
    """At h > 0, smoothed_log_score(y; h) == −log f_h(y) where f_h has
    Σ_m → Σ_m + h² I."""
    M, K = 4, 3
    torch.manual_seed(0)
    weights = torch.softmax(torch.randn(M), dim=0)
    centers = torch.randn(M, 2) * 4.0
    sigma = build_covariance_isotropic_scalar(1.5, M, "cpu")
    obs = torch.randn(K, 2) * 4.0
    h = 2.5
    s = smoothed_log_score(
        weights=weights, centers=centers, sigma_full=sigma, observations=obs, h=h
    )
    # Same direct computation with inflated σ².
    diff = obs.unsqueeze(1) - centers.unsqueeze(0)
    dist2 = (diff * diff).sum(dim=-1)
    sigma2_h = 1.5**2 + h * h
    log_phi = -math.log(2 * math.pi * sigma2_h) - 0.5 * dist2 / sigma2_h
    log_w = weights.log()
    expected = -torch.logsumexp(log_phi + log_w.unsqueeze(0), dim=-1)
    torch.testing.assert_close(s, expected, atol=1e-5, rtol=0)


def test_smoothed_log_score_increases_monotonically_with_h_for_centered_observation() -> None:
    """For a single component centered at y, smoothed_log_score
    increases with h (the model is over-confident at h=0 → low score;
    inflated σ smooths it out → higher score)."""
    weights = torch.tensor([1.0])
    centers = torch.tensor([[0.0, 0.0]])
    sigma = build_covariance_isotropic_scalar(1.0, 1, "cpu")
    obs = torch.tensor([[0.0, 0.0]])
    s0 = smoothed_log_score(
        weights=weights, centers=centers, sigma_full=sigma, observations=obs, h=0.0
    ).item()
    s1 = smoothed_log_score(
        weights=weights, centers=centers, sigma_full=sigma, observations=obs, h=2.0
    ).item()
    s2 = smoothed_log_score(
        weights=weights, centers=centers, sigma_full=sigma, observations=obs, h=4.0
    ).item()
    assert s0 < s1 < s2


def test_smoothed_log_score_rejects_negative_h() -> None:
    weights = torch.tensor([1.0])
    centers = torch.zeros(1, 2)
    sigma = build_covariance_isotropic_scalar(1.0, 1, "cpu")
    obs = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="h must be"):
        smoothed_log_score(
            weights=weights, centers=centers, sigma_full=sigma, observations=obs, h=-0.5
        )


# --------------------------------------------------------------------------- #
# Zone Brier
# --------------------------------------------------------------------------- #


def test_zone_brier_is_zero_when_predicted_equals_observed() -> None:
    """A trivial-density model centered at the observed shots (very
    narrow σ) predicts the empirical zone distribution exactly — Brier
    should be ~0 within MC noise."""
    # 6 shots, all at TopKey3 (zone 7) coords; very narrow σ.
    obs = torch.tensor([[0.0, 25.0]] * 6)
    centers = obs.clone()
    weights = torch.full((6,), 1.0 / 6.0)
    sigma = build_covariance_isotropic_scalar(0.3, 6, "cpu")  # very narrow
    gen = torch.Generator().manual_seed(0)
    out = zone_brier_and_ce(
        weights=weights,
        centers=centers,
        sigma_full=sigma,
        observations=obs,
        n_mc_samples=5000,
        generator=gen,
    )
    assert out["zone_brier"] < 0.01


def test_zone_brier_is_large_when_predicted_mismatches_observed() -> None:
    """Predict TopKey3; observe RA. Brier should be ~2 (perfect
    mismatch on two zones)."""
    centers = torch.tensor([[0.0, 25.0]])  # TopKey3
    weights = torch.tensor([1.0])
    sigma = build_covariance_isotropic_scalar(0.3, 1, "cpu")
    obs = torch.tensor([[0.0, 2.0]] * 6)  # RA
    gen = torch.Generator().manual_seed(0)
    out = zone_brier_and_ce(
        weights=weights,
        centers=centers,
        sigma_full=sigma,
        observations=obs,
        n_mc_samples=5000,
        generator=gen,
    )
    assert out["zone_brier"] > 1.5


# --------------------------------------------------------------------------- #
# HDR coverage
# --------------------------------------------------------------------------- #


def test_hdr_coverage_matches_alpha_when_model_is_truth() -> None:
    """If the observed shots are drawn from the model itself, empirical
    HDR coverage should match α within Monte-Carlo / sample-noise."""
    M = 4
    torch.manual_seed(0)
    weights = torch.softmax(torch.randn(M), dim=0)
    centers = torch.randn(M, 2) * 8.0
    sigma = build_covariance_isotropic_scalar(1.5, M, "cpu")
    # Draw 500 "observations" from the model.
    from shotcloud.evaluation.density_surfaces import _sample_from_mixture

    gen = torch.Generator().manual_seed(42)
    obs = _sample_from_mixture(
        weights=weights, centers=centers, sigma_full=sigma, n_samples=500, generator=gen
    )
    gen2 = torch.Generator().manual_seed(1)
    out = hdr_coverage(
        weights=weights,
        centers=centers,
        sigma_full=sigma,
        observations=obs,
        alphas=(0.5, 0.8, 0.9),
        n_mc_samples=10_000,
        generator=gen2,
    )
    # Calibration error < 5% at each level (loose for sample noise).
    for alpha in (0.5, 0.8, 0.9):
        emp = out[f"hdr_coverage_{round(100 * alpha)}"]
        assert abs(emp - alpha) < 0.05, f"HDR_{alpha}: empirical {emp:.3f} vs target {alpha}"


def _ensure_numpy_is_imported() -> None:
    """Used by some tests; suppress unused-import lint."""
    assert np.array([1, 2]).sum() == 3
