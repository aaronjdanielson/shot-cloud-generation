"""Tests for :class:`shotcloud.models.soft_kmeans_extractor.SoftKMeansModeExtractor`.

Covers weighted farthest-point seeding, mode centers inside the support convex hull,
recovery of two well-separated clusters without query tuning, mode mass tracking the
support attention ``ω``, normalized responsibilities, masked and cold-start rows, and
gradient flow back to ``ω``.
"""

from __future__ import annotations

import pytest
import torch

from shotcloud.data.context import CONTEXT_DIM
from shotcloud.models.mode_extractor import ModeExtractorOutputs
from shotcloud.models.soft_kmeans_extractor import (
    DEFAULT_MODE_KERNEL_BANDWIDTH_FT,
    DEFAULT_N_ITERATIONS,
    SoftKMeansModeExtractor,
    _batched_weighted_fps,
)


def _normalized_log_weights(b: int, m: int, *, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mask = torch.ones(b, m, dtype=torch.bool)
    logits = torch.zeros(b, m)
    logits = logits.masked_fill(~mask, float("-inf"))
    log_w = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    cold = ~mask.any(dim=-1)
    if cold.any():
        log_w = torch.where(cold.unsqueeze(-1), torch.full_like(log_w, float("-inf")), log_w)
    return log_w


def test_default_constants_match_spec() -> None:
    assert DEFAULT_N_ITERATIONS == 2
    assert DEFAULT_MODE_KERNEL_BANDWIDTH_FT == 5.0


# ---------------------------------------------------------------------------
# Weighted FPS
# ---------------------------------------------------------------------------


def test_weighted_fps_returns_k_seeds_with_correct_shape() -> None:
    """Seed 0 is the weighted mean; seeds 1..K-1 are distinct support points chosen by
    greedy FPS (the random support has no duplicate points)."""
    torch.manual_seed(0)
    b, m, k = 2, 12, 4
    support_xy = torch.randn(b, m, 2) * 10
    omega = torch.softmax(torch.randn(b, m), dim=-1)
    mask = torch.ones(b, m, dtype=torch.bool)
    seeds = _batched_weighted_fps(support_xy, omega, mask, k)
    assert seeds.shape == (b, k, 2)
    # Seeds 1..K-1 must be in the row's support.
    for b_idx in range(b):
        for k_idx in range(1, k):
            matches = (support_xy[b_idx] == seeds[b_idx, k_idx]).all(dim=-1)
            assert matches.any(), f"seed {k_idx}={seeds[b_idx, k_idx]} not in row {b_idx}'s support"
    # All K seeds distinct.
    for b_idx in range(b):
        pairwise = torch.cdist(seeds[b_idx : b_idx + 1], seeds[b_idx : b_idx + 1]).squeeze(0)
        off_diag = pairwise + torch.eye(k) * 1e9
        assert off_diag.min().item() > 0


def test_weighted_fps_first_seed_is_omega_weighted_mean() -> None:
    """Seed 0 is ``(Σ_j ω_j s_j) / (Σ_j ω_j)``, the support's center of mass, rather
    than a single high-weight shot."""
    k = 3
    support_xy = torch.tensor(
        [[[0, 0], [10, 0], [0, 10], [10, 10], [-10, -10]]], dtype=torch.float32
    )
    omega = torch.tensor([[0.05, 0.05, 0.7, 0.1, 0.1]], dtype=torch.float32)
    mask = torch.ones(1, 5, dtype=torch.bool)
    seeds = _batched_weighted_fps(support_xy, omega, mask, k)
    # Weighted mean: 0.05·(0,0) + 0.05·(10,0) + 0.7·(0,10) + 0.1·(10,10) + 0.1·(-10,-10)
    #              = (0 + 0.5 + 0 + 1 - 1, 0 + 0 + 7 + 1 - 1) = (0.5, 7.0)
    expected = torch.tensor([0.5, 7.0])
    torch.testing.assert_close(seeds[0, 0], expected, atol=1e-5, rtol=1e-5)


def test_weighted_fps_skips_masked_supports_for_seeds_one_onward() -> None:
    """Masked supports are never chosen as seeds 1..K-1, even when they are far from
    the other seeds."""
    k = 3
    support_xy = torch.tensor(
        [[[0, 0], [10, 0], [0, 10], [10, 10], [-10, -10]]], dtype=torch.float32
    )
    omega = torch.tensor([[0.25, 0.25, 0.25, 0.25, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True, True, False]])  # index 4 invalid
    seeds = _batched_weighted_fps(support_xy, omega, mask, k)
    for k_idx in range(1, k):
        assert tuple(seeds[0, k_idx].tolist()) != (-10.0, -10.0), (
            f"FPS seed {k_idx} landed on a masked support"
        )


# ---------------------------------------------------------------------------
# Extractor forward
# ---------------------------------------------------------------------------


def test_forward_output_shapes() -> None:
    b, m, k = 3, 20, 6
    ext = SoftKMeansModeExtractor(n_modes=k, n_iterations=2)
    support_xy = torch.randn(b, m, 2) * 5
    log_w = _normalized_log_weights(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    context = torch.randn(b, CONTEXT_DIM)
    out = ext(support_xy, log_w, mask, context)
    assert isinstance(out, ModeExtractorOutputs)
    assert out.mode_logits.shape == (b, k)
    assert out.mode_mu.shape == (b, k, 2)
    assert out.mode_attention.shape == (b, k, m)
    assert out.mode_mass.shape == (b, k)
    assert torch.isfinite(out.mode_logits).all()
    assert torch.isfinite(out.mode_mu).all()


def test_mode_centers_in_convex_hull() -> None:
    """Mean-shift centers are weighted averages of support coordinates, so they lie in
    the support's bounding box (a consequence of convex-hull membership)."""
    b, m, k = 2, 30, 4
    ext = SoftKMeansModeExtractor(n_modes=k, n_iterations=2)
    support_xy = torch.randn(b, m, 2) * 10
    log_w = _normalized_log_weights(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    context = torch.randn(b, CONTEXT_DIM)
    with torch.no_grad():
        out = ext(support_xy, log_w, mask, context)
    for b_idx in range(b):
        x_lo, x_hi = support_xy[b_idx, :, 0].min(), support_xy[b_idx, :, 0].max()
        y_lo, y_hi = support_xy[b_idx, :, 1].min(), support_xy[b_idx, :, 1].max()
        for k_idx in range(k):
            mu = out.mode_mu[b_idx, k_idx]
            assert x_lo - 1e-3 <= mu[0] <= x_hi + 1e-3
            assert y_lo - 1e-3 <= mu[1] <= y_hi + 1e-3


def test_cluster_recovery_with_two_well_separated_clusters() -> None:
    """With K = 2 and no query tuning, two well-separated support clusters each get
    one mode (FPS seeding plus mean-shift refinement)."""
    b, k = 1, 2
    cluster_a = torch.tensor([[-10.0, 5.0], [-9.5, 4.7], [-10.3, 5.4], [-9.8, 5.1]])
    cluster_b = torch.tensor([[10.0, 25.0], [10.2, 24.6], [9.7, 25.3], [10.4, 25.1]])
    support_xy = torch.cat([cluster_a, cluster_b], dim=0).unsqueeze(0)
    m = support_xy.shape[1]
    mask = torch.ones(b, m, dtype=torch.bool)
    log_w = _normalized_log_weights(b, m)
    context = torch.zeros(b, CONTEXT_DIM)
    ext = SoftKMeansModeExtractor(n_modes=k, n_iterations=3, use_context_correction=False)
    with torch.no_grad():
        out = ext(support_xy, log_w, mask, context)
    centers = torch.tensor([[-10.0, 5.0], [10.0, 25.0]])
    # Each mode lies within 2 ft of its nearest cluster center.
    dists = torch.cdist(out.mode_mu[0], centers)
    nearest = dists.argmin(dim=-1)  # (K,)
    nearest_dists = dists.min(dim=-1).values
    assert (nearest_dists < 2.0).all(), (
        f"mode centers {out.mode_mu[0]} too far from clusters; dists {nearest_dists}"
    )
    # Each cluster should be claimed by a distinct mode.
    assert set(nearest.tolist()) == {0, 1}


def test_concentrating_omega_on_a_cluster_raises_its_mode_mass() -> None:
    """Shifting ``ω`` toward a cluster raises the mass of the mode nearest it."""
    b, k = 1, 2
    cluster_a = torch.tensor([[-10.0, 5.0]] * 4)
    cluster_b = torch.tensor([[10.0, 25.0]] * 4)
    support_xy = torch.cat([cluster_a, cluster_b], dim=0).unsqueeze(0)
    m = support_xy.shape[1]
    mask = torch.ones(b, m, dtype=torch.bool)
    ext = SoftKMeansModeExtractor(n_modes=k, n_iterations=3, use_context_correction=False)

    def _masses(weight_on_a_per_shot: float) -> tuple[float, float]:
        weights = torch.tensor(
            [[weight_on_a_per_shot] * 4 + [(1.0 - 4 * weight_on_a_per_shot) / 4] * 4],
            dtype=torch.float32,
        )
        log_w = (weights + 1e-12).log()
        log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
        with torch.no_grad():
            out = ext(support_xy, log_w, mask, torch.zeros(1, CONTEXT_DIM))
        # Assign modes to clusters by distance.
        centers = torch.tensor([[-10.0, 5.0], [10.0, 25.0]])
        dists = torch.cdist(out.mode_mu[0], centers)
        mode_a = int(dists[:, 0].argmin())
        mode_b = 1 - mode_a
        return float(out.mode_mass[0, mode_a]), float(out.mode_mass[0, mode_b])

    m_a_hi, m_b_lo = _masses(0.9 / 4)  # 0.9 mass on A
    m_a_lo, m_b_hi = _masses(0.1 / 4)
    assert m_a_hi > m_a_lo, f"A-mass should rise when ω shifts to A; got {m_a_hi} vs {m_a_lo}"
    assert m_b_hi > m_b_lo, f"B-mass should rise when ω shifts to B; got {m_b_hi} vs {m_b_lo}"


def test_gradient_flow_to_omega_and_context_bias() -> None:
    from shotcloud.training.spatial_losses import mode_mixture_loglik

    b, m, k = 2, 20, 4
    ext = SoftKMeansModeExtractor(n_modes=k, n_iterations=2)
    support_xy = torch.randn(b, m, 2) * 5
    mask = torch.ones(b, m, dtype=torch.bool)
    log_w = _normalized_log_weights(b, m).requires_grad_(True)
    context = torch.randn(b, CONTEXT_DIM)
    shot_xy = torch.randn(b, 2)
    out = ext(support_xy, log_w, mask, context)
    loss = -mode_mixture_loglik(out.mode_logits, out.mode_mu, shot_xy, out.mode_sigma).mean()
    loss.backward()
    # ω (upstream support attention) receives gradient.
    assert log_w.grad is not None and log_w.grad.abs().sum() > 0
    # The zero-initialized mode-bias output layer still receives gradient.
    bias_out = ext.mode_bias[-1]  # type: ignore[index]
    assert bias_out.weight.grad is not None
    assert bias_out.weight.grad.abs().sum() > 0


def test_all_masked_row_does_not_nan() -> None:
    b, m, k = 3, 10, 4
    ext = SoftKMeansModeExtractor(n_modes=k, n_iterations=2)
    support_xy = torch.randn(b, m, 2)
    mask = torch.ones(b, m, dtype=torch.bool)
    mask[1] = False
    log_w = _normalized_log_weights(b, m, mask=mask)
    context = torch.zeros(b, CONTEXT_DIM)
    out = ext(support_xy, log_w, mask, context)
    assert torch.isfinite(out.mode_logits).all()
    assert torch.isfinite(out.mode_mu).all()
    assert torch.isfinite(out.mode_attention).all()
    assert torch.isfinite(out.mode_mass).all()
    assert out.cold_start[1] and not out.cold_start[0] and not out.cold_start[2]


def test_use_context_correction_false_yields_logits_equal_log_mass() -> None:
    b, m, k = 2, 10, 3
    ext = SoftKMeansModeExtractor(n_modes=k, n_iterations=2, use_context_correction=False)
    support_xy = torch.randn(b, m, 2)
    log_w = _normalized_log_weights(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    context = torch.randn(b, CONTEXT_DIM)
    with torch.no_grad():
        out = ext(support_xy, log_w, mask, context)
    expected = out.mode_mass.clamp_min(1e-12).log()
    torch.testing.assert_close(out.mode_logits, expected, atol=1e-6, rtol=1e-6)


def test_n_iterations_zero_returns_fps_init_centers_unchanged() -> None:
    """With ``n_iterations=0`` the centers are the FPS seeds: the weighted mean followed
    by support points."""
    b, k = 1, 3
    support_xy = torch.tensor(
        [[[-10.0, 5.0], [10.0, 25.0], [0.0, 0.0], [5.0, 15.0]]], dtype=torch.float32
    )
    m = support_xy.shape[1]
    mask = torch.ones(b, m, dtype=torch.bool)
    omega = torch.tensor([[0.3, 0.3, 0.2, 0.2]], dtype=torch.float32)
    log_w = (omega + 1e-12).log()
    log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
    ext = SoftKMeansModeExtractor(n_modes=k, n_iterations=0, use_context_correction=False)
    with torch.no_grad():
        out = ext(support_xy, log_w, mask, torch.zeros(1, CONTEXT_DIM))
    # Seed 0 = weighted mean = 0.3·(-10,5) + 0.3·(10,25) + 0.2·(0,0) + 0.2·(5,15)
    #                       = (1.0, 12.0).
    expected_mean = torch.tensor([1.0, 12.0])
    torch.testing.assert_close(out.mode_mu[0, 0], expected_mean, atol=1e-5, rtol=1e-5)
    # Seeds 1, 2 must be in the support set.
    for k_idx in range(1, k):
        matches = (support_xy[0] == out.mode_mu[0, k_idx]).all(dim=-1)
        assert matches.any(), f"mode {k_idx} center {out.mode_mu[0, k_idx]} not in support"


def test_responsibilities_normalize_over_modes() -> None:
    """``Σ_k r_{k,j} = 1`` for every valid support point, so responsibilities are a soft
    cluster assignment independent of ``ω``."""
    torch.manual_seed(0)
    b, m, k = 3, 16, 4
    ext = SoftKMeansModeExtractor(n_modes=k, n_iterations=2, use_context_correction=False)
    support_xy = torch.randn(b, m, 2) * 5
    log_w = _normalized_log_weights(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    with torch.no_grad():
        out = ext(support_xy, log_w, mask, torch.zeros(b, CONTEXT_DIM))
    # Sum over modes (dim=1) — must be 1 for valid support points.
    sums = out.mode_attention.sum(dim=1)  # (B, M)
    torch.testing.assert_close(sums, torch.ones(b, m), atol=1e-5, rtol=1e-5)


def test_default_mode_sigma_is_three_feet() -> None:
    """The default per-mode density bandwidth is 3 ft."""
    from shotcloud.models.mode_extractor import DEFAULT_MODE_SIGMA_FT

    assert DEFAULT_MODE_SIGMA_FT == 3.0
    ext = SoftKMeansModeExtractor(n_modes=4)
    assert float(ext.mode_sigma_buffer[0]) == 3.0


def test_constructor_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="n_modes"):
        SoftKMeansModeExtractor(n_modes=0)
    with pytest.raises(ValueError, match="n_iterations"):
        SoftKMeansModeExtractor(n_iterations=-1)
    with pytest.raises(ValueError, match="kernel_bandwidth"):
        SoftKMeansModeExtractor(kernel_bandwidth_ft=0.0)
    with pytest.raises(ValueError, match="mode_sigma"):
        SoftKMeansModeExtractor(mode_sigma_ft=0.0)
