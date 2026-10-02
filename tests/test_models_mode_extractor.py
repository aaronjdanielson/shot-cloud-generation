"""Tests for :class:`shotcloud.models.mode_extractor.SupportModeExtractor`.

Covers output shapes, mode centers inside the convex hull of valid support, masking,
recovery of two well-separated clusters, mode mass tracking the support attention ``ω``,
gradient flow through the mode-mixture NLL, finite outputs on cold-start rows, and the
``lambda_omega`` attention bias.
"""

from __future__ import annotations

import math

import pytest
import torch

from shotcloud.data.context import CONTEXT_DIM
from shotcloud.models.mode_extractor import (
    DEFAULT_MODE_QUERY_DIM,
    DEFAULT_N_COURT_MODES,
    ModeExtractorOutputs,
    SupportModeExtractor,
)
from shotcloud.training.spatial_losses import mode_mixture_loglik


def _normalized_log_weights(b: int, m: int, *, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Build (B, M) log-probability weights uniform over valid supports.

    Masked entries get -inf. This supplies ``log_support_weights`` without an
    upstream collaborative scorer.
    """
    if mask is None:
        mask = torch.ones(b, m, dtype=torch.bool)
    logits = torch.zeros(b, m)
    logits = logits.masked_fill(~mask, float("-inf"))
    log_w = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    # Cold-start rows give NaN from (-inf) - logsumexp(-inf); set them to -inf so
    # ω.exp() = 0. The extractor patches cold-start rows with its dummy slot 0.
    cold = ~mask.any(dim=-1)
    if cold.any():
        log_w = torch.where(cold.unsqueeze(-1), torch.full_like(log_w, float("-inf")), log_w)
    return log_w


def test_default_constants_match_plan() -> None:
    """Default ``K = 6`` modes and query dimension ``d = 32``."""
    assert DEFAULT_N_COURT_MODES == 6
    assert DEFAULT_MODE_QUERY_DIM == 32


def test_forward_output_shapes() -> None:
    b, m, k = 3, 20, 4
    ext = SupportModeExtractor(n_modes=k, mode_query_dim=8)
    support_xy = torch.randn(b, m, 2)
    log_w = _normalized_log_weights(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    context = torch.randn(b, CONTEXT_DIM)
    out = ext(support_xy, log_w, mask, context)
    assert isinstance(out, ModeExtractorOutputs)
    assert out.mode_logits.shape == (b, k)
    assert out.mode_mu.shape == (b, k, 2)
    assert out.mode_attention.shape == (b, k, m)
    assert out.mode_mass.shape == (b, k)
    assert out.cold_start.shape == (b,)
    assert torch.isfinite(out.mode_logits).all()
    assert torch.isfinite(out.mode_mu).all()


def test_mode_centers_lie_in_convex_hull_of_valid_support() -> None:
    """``μ_k = Σ_j α_{k,j} s_j`` with α a probability vector → each
    coordinate of ``μ_k`` is within [min, max] of the row's valid
    support coordinates."""
    b, m, k = 2, 30, 4
    ext = SupportModeExtractor(n_modes=k, mode_query_dim=8)
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


def test_masked_support_does_not_affect_mode_center_or_mass() -> None:
    """A masked support point at an extreme coordinate does not pull ``μ_k``."""
    b, m, k = 1, 5, 2
    ext = SupportModeExtractor(n_modes=k, mode_query_dim=8)
    # Four normal supports; one masked-out at a wildly distant point.
    support_xy = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [9999.0, 9999.0]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, True, True, False]])
    log_w = _normalized_log_weights(b, m, mask=mask)
    context = torch.zeros(b, CONTEXT_DIM)
    with torch.no_grad():
        out = ext(support_xy, log_w, mask, context)
    # The valid supports span [0, 1]^2; any leakage from the masked point at
    # (9999, 9999) would push the centers far outside it.
    assert (out.mode_mu.abs() < 100).all()


def _inject_cluster_aligned_queries(
    ext: SupportModeExtractor, cluster_centers: torch.Tensor, scale: float = 100.0
) -> None:
    """Set each mode query to a scaled embedding of its cluster center.

    With a random projection, an order-1 Q-K product does not separate the clusters
    sharply; scaling by ``scale`` makes the Q-K logits dominate the softmax, so each mode
    attends to its own cluster deterministically.
    """
    with torch.no_grad():
        psi_centers = ext.support_embedding(cluster_centers.unsqueeze(0)).squeeze(0)
        ext.mode_queries.copy_(psi_centers * scale)


def test_cluster_recovery_with_two_well_separated_clusters() -> None:
    """With K = 2 and queries aligned to two well-separated support clusters, each
    mode center lands near its cluster center."""
    b, k = 1, 2
    cluster_a = torch.tensor([[-10.0, 5.0], [-9.5, 4.7], [-10.3, 5.4], [-9.8, 5.1]])
    cluster_b = torch.tensor([[10.0, 25.0], [10.2, 24.6], [9.7, 25.3], [10.4, 25.1]])
    support_xy = torch.cat([cluster_a, cluster_b], dim=0).unsqueeze(0)
    m = support_xy.shape[1]
    mask = torch.ones(b, m, dtype=torch.bool)
    log_w = _normalized_log_weights(b, m)
    context = torch.zeros(b, CONTEXT_DIM)

    ext = SupportModeExtractor(n_modes=k, mode_query_dim=8, use_context_correction=False)
    cluster_centers = torch.tensor([[-10.0, 5.0], [10.0, 25.0]])
    _inject_cluster_aligned_queries(ext, cluster_centers)

    with torch.no_grad():
        out = ext(support_xy, log_w, mask, context)
    # Mode 0's center near cluster A; mode 1's near cluster B.
    for k_idx in range(k):
        dist = (out.mode_mu[0, k_idx] - cluster_centers[k_idx]).norm().item()
        assert dist < 3.0, (
            f"mode {k_idx}'s center {out.mode_mu[0, k_idx].tolist()} too far from "
            f"target cluster {cluster_centers[k_idx].tolist()}; dist={dist:.2f} ft"
        )


def test_concentrating_omega_on_a_cluster_raises_its_mode_mass() -> None:
    """Shifting ``ω`` toward a cluster raises the mass of the mode aligned with it.

    Cluster-aligned queries make the mode-to-cluster mapping deterministic.
    """
    b, k = 1, 2
    cluster_a = torch.tensor([[-10.0, 5.0]] * 4)
    cluster_b = torch.tensor([[10.0, 25.0]] * 4)
    support_xy = torch.cat([cluster_a, cluster_b], dim=0).unsqueeze(0)
    m = support_xy.shape[1]
    mask = torch.ones(b, m, dtype=torch.bool)

    ext = SupportModeExtractor(n_modes=k, mode_query_dim=8, use_context_correction=False)
    cluster_centers = torch.tensor([[-10.0, 5.0], [10.0, 25.0]])
    _inject_cluster_aligned_queries(ext, cluster_centers)

    def _masses(weight_on_a_per_shot: float) -> tuple[float, float]:
        weights = torch.tensor(
            [[weight_on_a_per_shot] * 4 + [(1.0 - 4 * weight_on_a_per_shot) / 4] * 4],
            dtype=torch.float32,
        )
        log_w = (weights + 1e-12).log()
        log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
        with torch.no_grad():
            out = ext(support_xy, log_w, mask, torch.zeros(1, CONTEXT_DIM))
        # Mode 0 is the A-mode by construction; mode 1 is the B-mode.
        return float(out.mode_mass[0, 0]), float(out.mode_mass[0, 1])

    # ω concentrated on A (0.9 / 4 per A shot, 0.025 per B shot).
    m_a_hi, m_b_lo = _masses(0.9 / 4)
    # ω concentrated on B (mirror).
    m_a_lo, m_b_hi = _masses(0.1 / 4)
    assert m_a_hi > m_a_lo, f"A-mass should rise when ω shifts to A; got {m_a_hi} vs {m_a_lo}"
    assert m_b_hi > m_b_lo, f"B-mass should rise when ω shifts to B; got {m_b_hi} vs {m_b_lo}"


def test_gradient_flow_to_logits_and_centers_via_mode_mixture_nll() -> None:
    """The mode-mixture NLL sends gradient to the extractor parameters and to the
    upstream ``log_support_weights``."""
    b, m, k = 2, 20, 4
    ext = SupportModeExtractor(n_modes=k, mode_query_dim=8)
    support_xy = torch.randn(b, m, 2) * 5
    mask = torch.ones(b, m, dtype=torch.bool)
    log_w = _normalized_log_weights(b, m).requires_grad_(True)
    context = torch.randn(b, CONTEXT_DIM)
    shot_xy = torch.randn(b, 2)
    out = ext(support_xy, log_w, mask, context)
    loss = -mode_mixture_loglik(out.mode_logits, out.mode_mu, shot_xy, out.mode_sigma).mean()
    loss.backward()
    assert ext.support_embedding.proj.weight.grad is not None
    assert ext.support_embedding.proj.weight.grad.abs().sum() > 0
    assert ext.mode_queries.grad is not None
    assert ext.mode_queries.grad.abs().sum() > 0
    # The zero-initialized bias output layer still receives a nonzero gradient.
    bias_out = ext.mode_bias[-1]  # type: ignore[index]
    assert bias_out.weight.grad is not None
    assert bias_out.weight.grad.abs().sum() > 0
    # log_w (upstream support attention) receives grad too.
    assert log_w.grad is not None and log_w.grad.abs().sum() > 0


def test_all_masked_row_does_not_nan_outputs() -> None:
    b, m, k = 3, 10, 4
    ext = SupportModeExtractor(n_modes=k, mode_query_dim=8)
    support_xy = torch.randn(b, m, 2)
    # Row 1 is entirely cold-start.
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


def test_constructor_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="n_modes"):
        SupportModeExtractor(n_modes=0)
    with pytest.raises(ValueError, match="mode_query_dim"):
        SupportModeExtractor(mode_query_dim=0)
    with pytest.raises(ValueError, match="mode_sigma"):
        SupportModeExtractor(mode_sigma_ft=0.0)
    with pytest.raises(ValueError, match="history_dim"):
        SupportModeExtractor(history_dim=-1)


def test_use_context_correction_false_yields_logits_equal_log_mass() -> None:
    """With ``use_context_correction=False`` the mode logits equal ``log m_k``."""
    b, m, k = 2, 10, 3
    ext = SupportModeExtractor(n_modes=k, mode_query_dim=8, use_context_correction=False)
    support_xy = torch.randn(b, m, 2)
    log_w = _normalized_log_weights(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    context = torch.randn(b, CONTEXT_DIM)
    with torch.no_grad():
        out = ext(support_xy, log_w, mask, context)
    expected = out.mode_mass.clamp_min(1e-12).log()
    torch.testing.assert_close(out.mode_logits, expected, atol=1e-6, rtol=1e-6)


def test_history_dim_zero_rejects_history_tensor_passed_in() -> None:
    """With ``history_dim=0`` a supplied history tensor is ignored without error."""
    b, m, k = 2, 8, 3
    ext = SupportModeExtractor(n_modes=k, mode_query_dim=8, history_dim=0)
    support_xy = torch.randn(b, m, 2)
    log_w = _normalized_log_weights(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    context = torch.randn(b, CONTEXT_DIM)
    with torch.no_grad():
        out_a = ext(support_xy, log_w, mask, context, history=None)
        out_b = ext(support_xy, log_w, mask, context, history=torch.randn(b, 5))
    torch.testing.assert_close(out_a.mode_logits, out_b.mode_logits)


def test_history_dim_positive_requires_history() -> None:
    b, m, k = 2, 8, 3
    ext = SupportModeExtractor(n_modes=k, mode_query_dim=8, history_dim=10)
    support_xy = torch.randn(b, m, 2)
    log_w = _normalized_log_weights(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    context = torch.randn(b, CONTEXT_DIM)
    with pytest.raises(ValueError, match="history is required"):
        ext(support_xy, log_w, mask, context, history=None)


def test_mode_queries_distinct_at_init_to_avoid_collapse() -> None:
    ext = SupportModeExtractor(n_modes=8, mode_query_dim=32)
    q = ext.mode_queries
    dists = torch.cdist(q.unsqueeze(0), q.unsqueeze(0)).squeeze(0)
    off_diag = dists + torch.eye(q.shape[0]) * 1e9
    assert off_diag.min().item() > 0


def test_recovers_uniform_mode_when_query_orthogonal_to_supports() -> None:
    """With zero queries and embeddings, every mode center is the ``ω``-weighted
    support mean.

    The Q-K term vanishes, so ``α`` reduces to ``softmax_j(λ_ω log ω_j)``; at the
    default ``λ_ω = 0`` that is uniform over the support, and with uniform ``ω`` the
    uniform and ``ω``-weighted means coincide.
    """
    b, m, k = 1, 8, 3
    ext = SupportModeExtractor(n_modes=k, mode_query_dim=8, use_context_correction=False)
    with torch.no_grad():
        ext.mode_queries.zero_()
        ext.support_embedding.proj.weight.zero_()
        ext.support_embedding.proj.bias.zero_()
    support_xy = torch.randn(b, m, 2)
    log_w = _normalized_log_weights(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    context = torch.zeros(b, CONTEXT_DIM)
    with torch.no_grad():
        out = ext(support_xy, log_w, mask, context)
    # All K mode centers should equal the support attention mean.
    omega = log_w.exp()  # (1, M)
    sup_mean = (omega.unsqueeze(-1) * support_xy).sum(dim=1)  # (1, 2)
    for k_idx in range(k):
        torch.testing.assert_close(out.mode_mu[0, k_idx], sup_mean[0], atol=1e-5, rtol=1e-5)


def test_lambda_omega_zero_yields_attention_independent_of_omega() -> None:
    """With ``lambda_omega = 0`` (the default) the attention ``α`` depends only on the
    Q-K product, so changing ``ω`` leaves ``α`` unchanged."""
    b, m, k = 1, 6, 3
    ext = SupportModeExtractor(
        n_modes=k, mode_query_dim=8, use_context_correction=False, lambda_omega=0.0
    )
    support_xy = torch.randn(b, m, 2)
    mask = torch.ones(b, m, dtype=torch.bool)
    context = torch.zeros(b, CONTEXT_DIM)
    log_w_uniform = _normalized_log_weights(b, m)
    # Skewed ω: concentrate on shot 0.
    skewed = torch.full((b, m), -10.0)
    skewed[:, 0] = 0.0
    log_w_skewed = skewed - torch.logsumexp(skewed, dim=-1, keepdim=True)
    with torch.no_grad():
        out_uniform = ext(support_xy, log_w_uniform, mask, context)
        out_skewed = ext(support_xy, log_w_skewed, mask, context)
    torch.testing.assert_close(
        out_uniform.mode_attention, out_skewed.mode_attention, atol=1e-6, rtol=1e-6
    )
    # Mass m_k = Σ_j ω_j α_{k,j} DOES change (ω only enters mass).
    assert not torch.allclose(out_uniform.mode_mass, out_skewed.mode_mass)


def test_lambda_omega_one_yields_attention_dependent_on_omega() -> None:
    """With ``lambda_omega = 1`` the attention is biased by ``log ω_j``, so ``α``
    changes when ``ω`` shifts."""
    b, m, k = 1, 6, 3
    ext = SupportModeExtractor(
        n_modes=k, mode_query_dim=8, use_context_correction=False, lambda_omega=1.0
    )
    support_xy = torch.randn(b, m, 2)
    mask = torch.ones(b, m, dtype=torch.bool)
    context = torch.zeros(b, CONTEXT_DIM)
    log_w_uniform = _normalized_log_weights(b, m)
    skewed = torch.full((b, m), -10.0)
    skewed[:, 0] = 0.0
    log_w_skewed = skewed - torch.logsumexp(skewed, dim=-1, keepdim=True)
    with torch.no_grad():
        out_uniform = ext(support_xy, log_w_uniform, mask, context)
        out_skewed = ext(support_xy, log_w_skewed, mask, context)
    assert not torch.allclose(out_uniform.mode_attention, out_skewed.mode_attention)


def test_lambda_omega_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="lambda_omega"):
        SupportModeExtractor(lambda_omega=-0.1)
    with pytest.raises(ValueError, match="lambda_omega"):
        SupportModeExtractor(lambda_omega=1.5)


def test_n_query_dim_used_in_sqrt_scaling() -> None:
    """The attention stays a normalized distribution over support for small and
    large query dimension ``d`` (Q-K logits are scaled by ``sqrt(d)``)."""
    b, m = 1, 8
    support_xy = torch.randn(b, m, 2)
    log_w = _normalized_log_weights(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    context = torch.zeros(b, CONTEXT_DIM)
    torch.manual_seed(0)
    ext_small = SupportModeExtractor(n_modes=2, mode_query_dim=8, use_context_correction=False)
    out_small = ext_small(support_xy, log_w, mask, context)
    torch.manual_seed(0)
    ext_large = SupportModeExtractor(n_modes=2, mode_query_dim=64, use_context_correction=False)
    out_large = ext_large(support_xy, log_w, mask, context)
    # Both should produce α values in [0, 1] that sum to 1 over j.
    np_sum_a = float(out_small.mode_attention.sum(dim=-1).flatten().max().item())
    np_sum_b = float(out_large.mode_attention.sum(dim=-1).flatten().max().item())
    assert math.isclose(np_sum_a, 1.0, abs_tol=1e-5)
    assert math.isclose(np_sum_b, 1.0, abs_tol=1e-5)
