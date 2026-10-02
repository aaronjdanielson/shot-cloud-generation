"""Density-surface evaluation scores for the continuous spatial mixture.

Cloud metrics (energy distance, sliced Wasserstein, zone L₁, rim-distance
W₁/KS, mean shot-distance error) evaluate **finite sampled shot clouds**:
they ask whether a generated finite cloud looks like the observed finite
cloud. They can be insensitive to improvements in the underlying
**predictive density surface** when (a) the model's density at observed
shots increases without the finite-sample geometry shifting visibly, or
(b) the noise floor of finite-K Wasserstein-type distances is comparable
to the structural signal.

This module provides **proper density-surface scoring rules** that
operate directly on the analytical Gaussian-mixture aggregate

.. math::

    \\bar f_g(y) = \\sum_m \\bar w_{g,m}\\,
    \\mathcal N_2(y; s_m, \\Sigma_m),

where the per-game support weights ``w̄_m = (1/K_g) Σ_r w_{r,m}`` are
the shot-context-averaged attention weights, ``s_m`` are the per-game
support-shot locations (fixed per ``(player_idx, snapshot_idx)``), and
``Σ_m`` is the per-component covariance dictated by the kernel kind:

* fixed isotropic: ``Σ_m = σ² I``;
* per-source/zone bandwidth
  (:class:`~shotcloud.models.zone_source_bandwidth.ZoneSourceBandwidth`):
  ``Σ_m = σ_m² I`` (still isotropic, but σ varies per support shot);
* radial-tangential
  (:class:`~shotcloud.models.anisotropic_kernel.RadialTangentZoneKernel`):
  ``Σ_m = σ_r²_z r̂ r̂ᵀ + σ_t²_z t̂ t̂ᵀ``;
* bounded-correlation full covariance
  (:class:`~shotcloud.models.anisotropic_kernel.FullCovarianceZoneKernel`):
  a generic per-zone 2×2 positive-definite matrix.

Scores (lower is better unless noted):

* :func:`quadratic_score` — ``S_quad = ∫ \\bar f² du − (2/K) Σ_r \\bar f(y_r)``
  — proper scoring rule; closed form for Gaussian mixtures.
* :func:`smoothed_log_score` — ``S_smooth(f, y; h) = −log Σ_m w_m
  φ(y; s_m, Σ_m + h² I)`` — log-density evaluated against a smoothed
  observation kernel, parameterized by the neighborhood scale ``h``.
  Tests whether mass is placed in the local neighborhood of the shot
  even if not at the exact point.
* :func:`zone_brier_and_ce` — predicted zone probabilities π_z
  computed by Monte Carlo against observed zone fractions. Scores the
  zone distribution without the finite-sample noise of the
  generated-cloud zone L₁.
* :func:`hdr_coverage` — highest-density-region calibration. For
  several α ∈ {0.5, 0.8, 0.9}, what fraction of observed shots fall
  inside the model's α-HDR? Calibration test for the density surface.

All scores accept a per-component covariance tensor ``Sigma`` of shape
``(M, 2, 2)`` so the same code path works for every kernel kind.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from shotcloud.data.zones import N_ZONES, zone_from_xy_vectorized

__all__ = [
    "build_covariance_full_cov",
    "build_covariance_isotropic_per_shot",
    "build_covariance_isotropic_scalar",
    "build_covariance_radial_tangent",
    "hdr_coverage",
    "log_mvn_density_2d",
    "quadratic_score",
    "smoothed_log_score",
    "zone_brier_and_ce",
]


# --------------------------------------------------------------------------- #
# Core building blocks
# --------------------------------------------------------------------------- #


def log_mvn_density_2d(diff: Tensor, sigma_full: Tensor) -> Tensor:
    """Log of the 2-D multivariate-normal density ``N(diff; 0, Σ)``.

    Closed-form for 2-D:

    .. math::

        \\log\\mathcal N_2(\\delta; 0, \\Sigma)
        = -\\log(2\\pi) - \\tfrac{1}{2}\\log|\\Sigma|
          - \\tfrac{1}{2}\\delta^{\\top}\\Sigma^{-1}\\delta.

    Inputs broadcast on the leading dims: ``diff`` is ``(..., 2)`` and
    ``sigma_full`` is ``(..., 2, 2)``. Returns ``(...,)``.

    Uses the closed-form 2×2 inverse: for ``Σ = [[a, b], [c, d]]``,
    ``|Σ| = ad − bc`` and ``Σ⁻¹ = (1/|Σ|)[[d, −b], [−c, a]]``. For
    symmetric Σ (``b = c``), the quadratic form is
    ``(1/|Σ|)(d δ_x² − 2b δ_x δ_y + a δ_y²)``.
    """
    a = sigma_full[..., 0, 0]
    b = sigma_full[..., 0, 1]
    d = sigma_full[..., 1, 1]
    # |Σ| = ad − b² (assumes symmetric Σ).
    det = (a * d - b * b).clamp_min(1e-12)
    inv_det = 1.0 / det
    log_det = torch.log(det)
    dx = diff[..., 0]
    dy = diff[..., 1]
    q = inv_det * (d * dx * dx - 2.0 * b * dx * dy + a * dy * dy)
    return -math.log(2.0 * math.pi) - 0.5 * log_det - 0.5 * q


# --------------------------------------------------------------------------- #
# Per-component covariance builders (one per kernel kind)
# --------------------------------------------------------------------------- #


def build_covariance_isotropic_scalar(
    sigma: float, n_components: int, device: torch.device | str
) -> Tensor:
    """``(M, 2, 2)`` Σ = σ² I (same σ for every component)."""
    sigma2 = sigma * sigma
    out = torch.zeros((n_components, 2, 2), device=device)
    out[:, 0, 0] = sigma2
    out[:, 1, 1] = sigma2
    return out


def build_covariance_isotropic_per_shot(sigma_per_shot: Tensor) -> Tensor:
    """``(M, 2, 2)`` Σ_m = σ_m² I from a per-shot bandwidth of shape ``(M,)``."""
    if sigma_per_shot.dim() != 1:
        raise ValueError(f"sigma_per_shot must be (M,); got {tuple(sigma_per_shot.shape)}")
    s2 = sigma_per_shot.pow(2)
    out = torch.zeros(
        (sigma_per_shot.shape[0], 2, 2),
        device=sigma_per_shot.device,
        dtype=sigma_per_shot.dtype,
    )
    out[:, 0, 0] = s2
    out[:, 1, 1] = s2
    return out


def build_covariance_radial_tangent(
    *,
    support_xy: Tensor,
    sigma_r_per_zone: Tensor,
    sigma_t_per_zone: Tensor,
    origin_eps: float = 1e-4,
) -> Tensor:
    """``(M, 2, 2)`` Σ_m = σ_r² r̂ r̂ᵀ + σ_t² t̂ t̂ᵀ in the rim-radial frame.

    Parameters
    ----------
    support_xy : Tensor of shape ``(M, 2)``
        Per-component support-shot coordinates in court feet.
    sigma_r_per_zone, sigma_t_per_zone : Tensor of shape ``(N_ZONES,)``
        Bounded σ values from a
        :class:`~shotcloud.models.anisotropic_kernel.RadialTangentZoneKernel`.
    origin_eps : float
        Below this radius, the radial frame is undefined; the support
        shot falls back to isotropic ``Σ = σ_r² I`` (the same fallback
        the kernel's forward uses).

    Returns
    -------
    Tensor of shape ``(M, 2, 2)`` symmetric positive-definite.
    """
    from shotcloud.models.zone_source_bandwidth import zone_from_xy_torch_bandwidth

    if support_xy.dim() != 2 or support_xy.shape[-1] != 2:
        raise ValueError(f"support_xy must be (M, 2); got {tuple(support_xy.shape)}")
    if sigma_r_per_zone.shape != (N_ZONES,):
        raise ValueError(
            f"sigma_r_per_zone must be ({N_ZONES},); got {tuple(sigma_r_per_zone.shape)}"
        )
    if sigma_t_per_zone.shape != (N_ZONES,):
        raise ValueError(
            f"sigma_t_per_zone must be ({N_ZONES},); got {tuple(sigma_t_per_zone.shape)}"
        )

    zone = zone_from_xy_torch_bandwidth(support_xy.unsqueeze(0)).squeeze(0).clamp_min(0)
    sigma_r = sigma_r_per_zone[zone]  # (M,)
    sigma_t = sigma_t_per_zone[zone]
    sx = support_xy[..., 0]
    sy = support_xy[..., 1]
    norm2 = sx * sx + sy * sy
    near_origin = norm2 < (origin_eps * origin_eps)
    norm = norm2.clamp_min(origin_eps * origin_eps).sqrt()
    r_hat_x = sx / norm
    r_hat_y = sy / norm
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x

    # Σ = σ_r² r̂ r̂ᵀ + σ_t² t̂ t̂ᵀ.
    sr2 = sigma_r.pow(2)
    st2 = sigma_t.pow(2)
    s00 = sr2 * r_hat_x * r_hat_x + st2 * t_hat_x * t_hat_x
    s11 = sr2 * r_hat_y * r_hat_y + st2 * t_hat_y * t_hat_y
    s01 = sr2 * r_hat_x * r_hat_y + st2 * t_hat_x * t_hat_y

    out = torch.zeros((support_xy.shape[0], 2, 2), device=support_xy.device)
    out[:, 0, 0] = s00
    out[:, 1, 1] = s11
    out[:, 0, 1] = s01
    out[:, 1, 0] = s01
    # Origin guard: fall back to isotropic σ_r² I — matches the
    # RadialTangentZoneKernel forward's origin-eps branch.
    if near_origin.any():
        iso = torch.zeros_like(out[near_origin])
        iso[:, 0, 0] = sr2[near_origin]
        iso[:, 1, 1] = sr2[near_origin]
        out[near_origin] = iso
    return out


def build_covariance_full_cov(
    *,
    support_xy: Tensor,
    sigma_x_per_zone: Tensor,
    sigma_y_per_zone: Tensor,
    rho_per_zone: Tensor,
) -> Tensor:
    """``(M, 2, 2)`` covariances from per-zone bounded-correlation ``(σ_x, σ_y, ρ)``.

    Each support shot takes the parameters of its court zone, as in
    :class:`~shotcloud.models.anisotropic_kernel.FullCovarianceZoneKernel`.
    """
    from shotcloud.models.zone_source_bandwidth import zone_from_xy_torch_bandwidth

    if support_xy.dim() != 2 or support_xy.shape[-1] != 2:
        raise ValueError(f"support_xy must be (M, 2); got {tuple(support_xy.shape)}")
    for name, t in (
        ("sigma_x", sigma_x_per_zone),
        ("sigma_y", sigma_y_per_zone),
        ("rho", rho_per_zone),
    ):
        if t.shape != (N_ZONES,):
            raise ValueError(f"{name}_per_zone must be ({N_ZONES},); got {tuple(t.shape)}")

    zone = zone_from_xy_torch_bandwidth(support_xy.unsqueeze(0)).squeeze(0).clamp_min(0)
    sx = sigma_x_per_zone[zone]
    sy = sigma_y_per_zone[zone]
    rho = rho_per_zone[zone]
    out = torch.zeros((support_xy.shape[0], 2, 2), device=support_xy.device)
    out[:, 0, 0] = sx.pow(2)
    out[:, 1, 1] = sy.pow(2)
    out[:, 0, 1] = rho * sx * sy
    out[:, 1, 0] = rho * sx * sy
    return out


# --------------------------------------------------------------------------- #
# Scoring rules
# --------------------------------------------------------------------------- #


def quadratic_score(
    *,
    weights: Tensor,
    centers: Tensor,
    sigma_full: Tensor,
    observations: Tensor,
) -> Tensor:
    """Quadratic density score (proper scoring rule).

    .. math::

        S_{\\text{quad}}(f, y_{1:K})
        = \\int f(u)^2\\,du - \\frac{2}{K}\\sum_{r=1}^{K} f(y_r).

    Lower is better. Closed-form for the Gaussian-mixture density
    ``f(y) = Σ_m w_m N₂(y; s_m, Σ_m)``:

    * ``∫ f² du = Σ_{m,ℓ} w_m w_ℓ N₂(s_m; s_ℓ, Σ_m + Σ_ℓ)`` — exact.
    * ``f(y_r) = Σ_m w_m N₂(y_r; s_m, Σ_m)`` — direct evaluation.

    Parameters
    ----------
    weights : Tensor of shape ``(M,)``
        Mixture weights, must sum to 1.
    centers : Tensor of shape ``(M, 2)``
        Per-component centers (support-shot coordinates).
    sigma_full : Tensor of shape ``(M, 2, 2)``
        Per-component covariance matrices.
    observations : Tensor of shape ``(K, 2)``
        Observed shots y_1..y_K.

    Returns
    -------
    Tensor scalar (the quadratic score).
    """
    if weights.dim() != 1:
        raise ValueError(f"weights must be (M,); got {tuple(weights.shape)}")
    M = weights.shape[0]
    if centers.shape != (M, 2):
        raise ValueError(f"centers must be (M={M}, 2); got {tuple(centers.shape)}")
    if sigma_full.shape != (M, 2, 2):
        raise ValueError(f"sigma_full must be (M={M}, 2, 2); got {tuple(sigma_full.shape)}")
    if observations.dim() != 2 or observations.shape[-1] != 2:
        raise ValueError(f"observations must be (K, 2); got {tuple(observations.shape)}")

    # ∫ f² du via the pairwise Gaussian-density identity.
    # log φ(s_m; s_ℓ, Σ_m + Σ_ℓ) for all m, ℓ.
    diff_centers = centers.unsqueeze(0) - centers.unsqueeze(1)  # (M, M, 2)
    sigma_sum = sigma_full.unsqueeze(0) + sigma_full.unsqueeze(1)  # (M, M, 2, 2)
    log_phi_mm = log_mvn_density_2d(diff_centers, sigma_sum)  # (M, M)
    integral = (weights.unsqueeze(0) * weights.unsqueeze(1) * log_phi_mm.exp()).sum()

    # (2/K) Σ_r f(y_r).
    K = observations.shape[0]
    diff_obs = observations.unsqueeze(1) - centers.unsqueeze(0)  # (K, M, 2)
    sigma_expanded = sigma_full.unsqueeze(0).expand(K, M, 2, 2)
    log_phi_obs = log_mvn_density_2d(diff_obs, sigma_expanded)  # (K, M)
    # f(y_r) = Σ_m w_m φ(y_r; s_m, Σ_m) — use logsumexp for stability.
    log_w = weights.clamp_min(1e-30).log()
    log_f_y = torch.logsumexp(log_phi_obs + log_w.unsqueeze(0), dim=-1)  # (K,)
    cross = (2.0 / K) * log_f_y.exp().sum()
    return integral - cross


def smoothed_log_score(
    *,
    weights: Tensor,
    centers: Tensor,
    sigma_full: Tensor,
    observations: Tensor,
    h: float,
) -> Tensor:
    """Smoothed log score: ``−log Σ_m w_m N₂(y; s_m, Σ_m + h² I)``.

    Evaluates the model at the observation convolved with a Gaussian
    observation kernel of bandwidth ``h``. At ``h = 0`` this reduces
    to the plain NLL (modulo numerical clamping); at ``h > 0`` it
    tests whether the model places mass in the local neighborhood of
    the shot, even if not exactly at the point. Run at multiple ``h``
    to get a scale-space picture of the density surface.

    Parameters
    ----------
    weights, centers, sigma_full
        Mixture parameters; same convention as :func:`quadratic_score`.
    observations : Tensor of shape ``(K, 2)``
        Observed shots ``y_1..y_K``.
    h : float
        Neighborhood bandwidth in feet.

    Returns
    -------
    Tensor of shape ``(K,)``
        Per-observation score. Lower is better; the caller typically
        takes the mean.
    """
    if h < 0.0:
        raise ValueError(f"h must be ≥ 0; got {h}")
    M = weights.shape[0]
    K = observations.shape[0]
    sigma_inflated = sigma_full.clone()
    sigma_inflated[:, 0, 0] = sigma_inflated[:, 0, 0] + h * h
    sigma_inflated[:, 1, 1] = sigma_inflated[:, 1, 1] + h * h
    diff = observations.unsqueeze(1) - centers.unsqueeze(0)  # (K, M, 2)
    sigma_expanded = sigma_inflated.unsqueeze(0).expand(K, M, 2, 2)
    log_phi = log_mvn_density_2d(diff, sigma_expanded)  # (K, M)
    log_w = weights.clamp_min(1e-30).log()
    log_f = torch.logsumexp(log_phi + log_w.unsqueeze(0), dim=-1)  # (K,)
    return -log_f


# --------------------------------------------------------------------------- #
# Zone Brier / cross-entropy + HDR calibration (Monte Carlo)
# --------------------------------------------------------------------------- #


def _sample_from_mixture(
    *,
    weights: Tensor,
    centers: Tensor,
    sigma_full: Tensor,
    n_samples: int,
    generator: torch.Generator,
) -> Tensor:
    """Sample ``n_samples`` points from the Gaussian mixture.

    Uses the standard two-step procedure: sample a component index from
    the multinomial defined by ``weights``, then sample a 2-D Gaussian
    around that component's center with its covariance.

    Returns ``(n_samples, 2)``.
    """
    component_idx = torch.multinomial(
        weights.clamp_min(1e-30), n_samples, replacement=True, generator=generator
    )  # (n_samples,)
    chosen_centers = centers[component_idx]  # (n_samples, 2)
    chosen_sigma = sigma_full[component_idx]  # (n_samples, 2, 2)
    # Cholesky of each 2×2 covariance. For symmetric Σ = [[a, b], [b, d]]:
    # L = [[√a, 0], [b/√a, √(d − b²/a)]].
    a = chosen_sigma[..., 0, 0].clamp_min(1e-12)
    b = chosen_sigma[..., 0, 1]
    d = chosen_sigma[..., 1, 1]
    l11 = a.sqrt()
    l21 = b / l11
    l22 = (d - l21 * l21).clamp_min(1e-12).sqrt()
    z = torch.randn(n_samples, 2, generator=generator, device=weights.device)
    out_x = l11 * z[:, 0]
    out_y = l21 * z[:, 0] + l22 * z[:, 1]
    return torch.stack([chosen_centers[:, 0] + out_x, chosen_centers[:, 1] + out_y], dim=-1)


def zone_brier_and_ce(
    *,
    weights: Tensor,
    centers: Tensor,
    sigma_full: Tensor,
    observations: Tensor,
    n_mc_samples: int = 10_000,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Zone Brier score and cross-entropy of the predicted zone distribution.

    For each zone ``z ∈ [0, N_ZONES)`` the predicted probability is

    .. math::

        \\pi_z = \\int_{Z_z} \\bar f_g(y)\\,dy,

    estimated via Monte Carlo with ``n_mc_samples`` draws from
    :math:`\\bar f_g`. The observed probability is the empirical zone
    fraction over the game's K observed shots. Both are normalized to
    valid distributions over the on-court zones (zone-(-1) out-of-court
    samples / observations are dropped before normalization).

    Returns ``{"zone_brier": ..., "zone_ce": ..., "n_obs_in_court": ...}``.
    Both Brier and CE: lower is better.
    """
    if generator is None:
        generator = torch.Generator(device=weights.device)
        generator.manual_seed(0)
    samples = _sample_from_mixture(
        weights=weights,
        centers=centers,
        sigma_full=sigma_full,
        n_samples=n_mc_samples,
        generator=generator,
    )
    # Zone histograms over in-court samples / observations.
    sample_zones = zone_from_xy_vectorized(
        samples[:, 0].detach().cpu().numpy(), samples[:, 1].detach().cpu().numpy()
    )
    obs_zones = zone_from_xy_vectorized(
        observations[:, 0].detach().cpu().numpy(), observations[:, 1].detach().cpu().numpy()
    )
    pi_pred = _zone_histogram(sample_zones)
    pi_obs = _zone_histogram(obs_zones)
    # Brier: Σ_z (π_pred − π_obs)².
    diff = pi_pred - pi_obs
    brier = float((diff * diff).sum())
    # Cross-entropy: −Σ_z π_obs log(π_pred + eps).
    eps = 1e-12
    ce = float(-(pi_obs * np.log(pi_pred + eps)).sum())
    n_in_court = int((obs_zones >= 0).sum())
    return {"zone_brier": brier, "zone_ce": ce, "n_obs_in_court": float(n_in_court)}


def _zone_histogram(zones: NDArray[np.int64]) -> NDArray[np.float64]:
    """8-bin probability over in-court zones (drop ``zone == -1``).

    Returns a length-``N_ZONES`` numpy array summing to 1 (or all-zeros
    if every input was out of court).
    """
    in_court = zones[zones >= 0]
    out = np.zeros(N_ZONES, dtype=np.float64)
    if in_court.size == 0:
        return out
    counts = np.bincount(in_court, minlength=N_ZONES).astype(np.float64)
    from typing import cast

    return cast(NDArray[np.float64], counts / counts.sum())


def hdr_coverage(
    *,
    weights: Tensor,
    centers: Tensor,
    sigma_full: Tensor,
    observations: Tensor,
    alphas: tuple[float, ...] = (0.5, 0.8, 0.9),
    n_mc_samples: int = 10_000,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Highest-density-region calibration of the predictive surface.

    For each ``α ∈ alphas`` the model's α-HDR is the smallest set
    ``H_α = {y : f(y) ≥ c_α}`` containing ``α`` of the model's
    probability mass. A well-calibrated density places α of the
    observed shots inside ``H_α`` for each α.

    The threshold ``c_α`` is estimated by drawing ``n_mc_samples`` from
    the mixture, computing their densities, and using the ``(1-α)``-
    quantile (so ``α`` of the samples — and ideally α of the
    observations — sit above the cutoff).

    Returns ``{"hdr_coverage_{int(100α)}": empirical_fraction, ...}``.
    Calibration error is ``|empirical − α|``.
    """
    if generator is None:
        generator = torch.Generator(device=weights.device)
        generator.manual_seed(0)
    samples = _sample_from_mixture(
        weights=weights,
        centers=centers,
        sigma_full=sigma_full,
        n_samples=n_mc_samples,
        generator=generator,
    )
    # Density at samples and at observations.
    f_samples = _mixture_log_density(weights, centers, sigma_full, samples).exp()
    f_obs = _mixture_log_density(weights, centers, sigma_full, observations).exp()
    out: dict[str, float] = {}
    for alpha in alphas:
        c_alpha = torch.quantile(f_samples, 1.0 - alpha)
        in_hdr = (f_obs >= c_alpha).float().mean().item()
        out[f"hdr_coverage_{round(100 * alpha)}"] = float(in_hdr)
    return out


def _mixture_log_density(
    weights: Tensor, centers: Tensor, sigma_full: Tensor, points: Tensor
) -> Tensor:
    """Log-density of the Gaussian mixture at ``points`` of shape ``(N, 2)``."""
    if points.dim() != 2 or points.shape[-1] != 2:
        raise ValueError(f"points must be (N, 2); got {tuple(points.shape)}")
    N = points.shape[0]
    M = weights.shape[0]
    diff = points.unsqueeze(1) - centers.unsqueeze(0)  # (N, M, 2)
    sigma_expanded = sigma_full.unsqueeze(0).expand(N, M, 2, 2)
    log_phi = log_mvn_density_2d(diff, sigma_expanded)  # (N, M)
    log_w = weights.clamp_min(1e-30).log()
    return torch.logsumexp(log_phi + log_w.unsqueeze(0), dim=-1)  # (N,)
