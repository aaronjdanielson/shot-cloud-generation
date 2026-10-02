"""Support-driven soft k-means / mean-shift mode extractor.

Replaces the learned-query mode extractor
(:class:`~shotcloud.models.mode_extractor.SupportModeExtractor`) with
a math-first clustering operator: mode centers are obtained by
running a few **differentiable mean-shift iterations** over the
attended support set, initialized via deterministic weighted
farthest-point sampling.

The key advantage over learned queries is that the centers MUST come
from local support-density structure (the kernel-weighted average of
nearby support shots, pulled by ω). The learned-query version
empirically behaved as "implicit global anchors" — the queries
learned to attend to broad league-typical regions regardless of
player. With clustering, the modes are clusters of *this row's*
support by construction.

Per the 2026-05-18 mode-collapse diagnosis:

> If the learned queries are global parameters, then even though
> centers are convex combinations of support shots, the same query
> may always attend to the same broad region. That creates de facto
> global modes.

Mathematical specification (v2, 2026-05-18 normalized form):

1. Initialize K seeds via weighted FPS on ``support_xy``:
   * Seed 0 = the **ω-weighted mean** of the attended support,
     ``μ_1^{(0)} = (Σ_j ω_j s_j) / (Σ_j ω_j)``.
   * Seeds 1..K-1 picked greedily as
     ``argmax_j ω_j · D_j²`` where ``D_j`` is min distance to
     already-selected seeds.
2. For ``n_iterations`` steps:
   * **Normalized responsibility** (soft cluster assignment of
     each support point to modes):
     ``r_{k,j} = exp(-||s_j - μ_k||² / (2ρ²)) /
                 Σ_ℓ exp(-||s_j - μ_ℓ||² / (2ρ²))``,
     so ``Σ_k r_{k,j} = 1`` for every support point.
   * Centroid update with explicit ω weighting:
     ``μ_k = (Σ_j ω_j r_{k,j} s_j) / (Σ_j ω_j r_{k,j})``.
3. Mode mass = share of attended support assigned to mode k:
   ``m_k = Σ_j ω_j r_{k,j}``.
4. Mode logits: ``log(m_k + ε) + b_k(x_n, h_n)`` with optional
   context-bias MLP (zero-init last layer).

The v2 normalization makes ``r_{k,j}`` a proper soft cluster
assignment (cleaner math, cleaner viz), and decouples the
"cluster assignment" step from the "evidence weight" — ω only
enters at the centroid + mass aggregation. The previous v1 form
multiplied ω into raw mean-shift kernel weights, which conflated
the two.

The only learned parameters are the support embedding (not used in
v1 of this extractor — kept fixed-zero or absent) and the
context-bias MLP. Centers are non-parametric (cluster-derived);
mode mass is non-parametric; only the per-mode mixture weights get
a small context-conditional MLP correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from shotcloud.data.context import CONTEXT_DIM
from shotcloud.models.mode_extractor import (
    DEFAULT_MODE_SIGMA_FT,
    DEFAULT_N_COURT_MODES,
    ModeExtractorOutputs,
)

#: Default Gaussian kernel bandwidth ρ for the mean-shift
#: responsibility. 5 ft is the same scale used by the legacy
#: mode-membership kernel — wide enough that a few iterations smooth
#: meaningfully, narrow enough that distinct shot clusters separate.
DEFAULT_MODE_KERNEL_BANDWIDTH_FT: float = 5.0

#: Default number of mean-shift iterations. 2 is enough to refine the
#: FPS seeds toward local-density modes without making the forward
#: too expensive.
DEFAULT_N_ITERATIONS: int = 2


def _batched_weighted_fps(
    support_xy: Tensor,
    omega: Tensor,
    support_mask: Tensor,
    k: int,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Deterministic weighted farthest-point sampling for K seeds per row.

    For each batch row independently:

    1. Seed 0 = the **ω-weighted mean** of the attended support,
       ``μ_1 = (Σ_j ω_j s_j) / (Σ_j ω_j)``. This is a synthesized
       point (typically not one of the support shots) — it starts one
       mode at the support cloud's center of mass, more stable than
       starting at a single highest-ω shot.
    2. For each subsequent seed, score each candidate support point
       by ``ω_j · D_j²`` where ``D_j`` is the min distance to any
       already-selected seed (mean for seed 1, then mean + earlier
       picks). Pick ``argmax_j`` of the score; ties broken by index
       and excluding already-picked points.

    Argmax is non-differentiable (the picked indices), but the
    output coords are just gathered from ``support_xy``; gradients
    flow through ``support_xy`` (themselves coords from the dataset,
    typically detached) and through ``ω`` (which carries upstream
    support-attention gradient — and is also used directly for the
    weighted-mean seed 0).

    Returns ``(B, K, 2)`` seed coordinates.
    """
    b, m, _ = support_xy.shape
    if k > m:
        raise ValueError(f"requested K={k} seeds but only M={m} support points available")
    device = support_xy.device

    # Seed 0: weighted mean of attended support. ω is already masked
    # to zero on invalid rows by the caller, but we still clamp the
    # denominator for cold-start rows (all ω=0) so the division is
    # finite. Cold-start seed 0 = (0, 0), which is meaningless but
    # finite; the trainer's log-lik floor overrides those rows.
    omega_sum = omega.sum(dim=-1, keepdim=True).clamp_min(eps)  # (B, 1)
    seed0 = (omega.unsqueeze(-1) * support_xy).sum(dim=-2) / omega_sum  # (B, 2)
    seed_xy = seed0.unsqueeze(1)  # (B, 1, 2)

    # Subsequent seeds: pick from the support set via weighted FPS.
    # ``selected_mask`` prevents re-picking the same support index
    # when ω is concentrated and the score saturates to 0.
    selected_mask = torch.zeros(b, m, dtype=torch.bool, device=device)
    seed_indices = torch.empty(b, k - 1, dtype=torch.long, device=device) if k > 1 else None

    for i in range(1, k):
        diff = support_xy.unsqueeze(2) - seed_xy.unsqueeze(1)  # (B, M, i, 2)
        dist_sq = (diff * diff).sum(dim=-1)  # (B, M, i)
        min_dist_sq = dist_sq.min(dim=-1).values  # (B, M)
        score = omega.masked_fill(~support_mask, 0.0) * min_dist_sq
        score = score.masked_fill(~support_mask | selected_mask, float("-inf"))
        picked = score.argmax(dim=-1)  # (B,)
        assert seed_indices is not None
        seed_indices[:, i - 1] = picked
        selected_mask.scatter_(1, picked.unsqueeze(-1), True)
        new_seed = torch.gather(support_xy, 1, picked[:, None, None].expand(-1, -1, 2))
        seed_xy = torch.cat([seed_xy, new_seed], dim=1)  # (B, i+1, 2)

    return seed_xy


@dataclass(frozen=True)
class _SoftKMeansHyperparams:
    n_modes: int
    n_iterations: int
    kernel_bandwidth_ft: float
    mode_sigma_ft: float


class SoftKMeansModeExtractor(nn.Module):
    """Mean-shift mode extractor over attended support.

    Drop-in replacement for
    :class:`~shotcloud.models.mode_extractor.SupportModeExtractor`:
    same forward signature, same :class:`ModeExtractorOutputs` return.

    Parameters
    ----------
    n_modes : int, default 6
    n_iterations : int, default 2
        Number of mean-shift refinement steps after the FPS init.
    kernel_bandwidth_ft : float, default 5.0
        ρ — Gaussian kernel bandwidth on support distances during
        the mean-shift iterations.
    mode_sigma_ft : float, default 2.0
        Per-mode density bandwidth σ_k for the downstream mode-mixture
        Gaussian (the same fixed σ as the learned-query path).
    context_dim, history_dim, bias_hidden_dim, use_context_correction :
        Same semantics as
        :class:`~shotcloud.models.mode_extractor.SupportModeExtractor`.
    """

    mode_sigma_buffer: Tensor

    def __init__(
        self,
        n_modes: int = DEFAULT_N_COURT_MODES,
        n_iterations: int = DEFAULT_N_ITERATIONS,
        kernel_bandwidth_ft: float = DEFAULT_MODE_KERNEL_BANDWIDTH_FT,
        mode_sigma_ft: float = DEFAULT_MODE_SIGMA_FT,
        context_dim: int = CONTEXT_DIM,
        history_dim: int = 0,
        bias_hidden_dim: int = 32,
        use_context_correction: bool = True,
    ) -> None:
        super().__init__()
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive; got {n_modes}")
        if n_iterations < 0:
            raise ValueError(f"n_iterations must be non-negative; got {n_iterations}")
        if kernel_bandwidth_ft <= 0:
            raise ValueError(f"kernel_bandwidth_ft must be positive; got {kernel_bandwidth_ft}")
        if mode_sigma_ft <= 0:
            raise ValueError(f"mode_sigma_ft must be positive; got {mode_sigma_ft}")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive; got {context_dim}")
        if history_dim < 0:
            raise ValueError(f"history_dim must be non-negative; got {history_dim}")

        self.n_modes = int(n_modes)
        self.n_iterations = int(n_iterations)
        self.kernel_bandwidth_ft = float(kernel_bandwidth_ft)
        self.use_context_correction = bool(use_context_correction)
        self.history_dim = int(history_dim)

        if use_context_correction:
            bias_out = nn.Linear(bias_hidden_dim, n_modes)
            nn.init.zeros_(bias_out.weight)
            nn.init.zeros_(bias_out.bias)
            self.mode_bias: nn.Module = nn.Sequential(
                nn.Linear(context_dim + history_dim, bias_hidden_dim),
                nn.GELU(),
                bias_out,
            )
        else:
            self.mode_bias = nn.Identity()

        self.register_buffer(
            "mode_sigma_buffer",
            torch.full((n_modes,), float(mode_sigma_ft)),
            persistent=False,
        )

    def forward(
        self,
        support_xy: Tensor,
        log_support_weights: Tensor,
        support_mask: Tensor,
        context: Tensor,
        history: Tensor | None = None,
    ) -> ModeExtractorOutputs:
        """Mean-shift extract K modes from the attended support set."""
        if support_xy.dim() != 3 or support_xy.shape[-1] != 2:
            raise ValueError(f"support_xy must be (B, M, 2); got {tuple(support_xy.shape)}")
        b, m, _ = support_xy.shape
        if log_support_weights.shape != (b, m):
            raise ValueError(
                f"log_support_weights must be (B, M)={b, m}; got {tuple(log_support_weights.shape)}"
            )
        if support_mask.shape != (b, m):
            raise ValueError(f"support_mask must be (B, M)={b, m}; got {tuple(support_mask.shape)}")
        if self.history_dim > 0:
            if history is None:
                raise ValueError(f"history_dim={self.history_dim} > 0 → history is required")
            if history.shape != (b, self.history_dim):
                raise ValueError(
                    f"history must be (B, {self.history_dim}); got {tuple(history.shape)}"
                )

        cold_start = ~support_mask.any(dim=-1)  # (B,)
        # ω in linear space; zero out invalid + cold-start rows so the
        # FPS / mean-shift never sees their values. Cold-start rows
        # still need finite seeds (we patch them after) so the forward
        # doesn't NaN; the caller applies a log-lik floor.
        omega = log_support_weights.exp()
        omega = omega.masked_fill(~support_mask, 0.0)

        # FPS init. For cold-start rows the function picks arbitrary
        # indices (all ω are zero on those rows); the resulting seeds
        # are well-defined coords from support_xy, just meaningless.
        mu = _batched_weighted_fps(
            support_xy=support_xy,
            omega=omega,
            support_mask=support_mask,
            k=self.n_modes,
        )  # (B, K, 2)

        rho_sq = self.kernel_bandwidth_ft * self.kernel_bandwidth_ft
        # Per the v2 normalized spec:
        # * responsibility ``r_{k,j} = softmax_k(-d²_{k,j} / (2ρ²))``
        #   so Σ_k r_{k,j} = 1 for every support point;
        # * centroid update ``μ_k = Σ_j ω_j r_{k,j} s_j / Σ_j ω_j r_{k,j}``;
        # * mode mass ``m_k = Σ_j ω_j r_{k,j}``.
        # We compute one extra responsibility pass after the final
        # centroid update so ``r_kj_final`` and ``mode_mass`` reflect
        # the final ``μ``.
        r_kj_final: Tensor | None = None
        for step in range(self.n_iterations + 1):
            diff = support_xy.unsqueeze(1) - mu.unsqueeze(2)  # (B, K, M, 2)
            dist_sq = (diff * diff).sum(dim=-1)  # (B, K, M)
            # log-softmax over modes for numerical stability.
            log_kernel = -0.5 * dist_sq / rho_sq  # (B, K, M)
            log_r = log_kernel - torch.logsumexp(log_kernel, dim=1, keepdim=True)  # (B, K, M)
            r_kj = log_r.exp()  # (B, K, M); Σ_k r_kj == 1 per support point.
            # Mask out invalid support contributions. We do this AFTER
            # the softmax so the normalization is over the full mode
            # set; invalid points then simply contribute 0 mass.
            r_kj = r_kj.masked_fill(~support_mask.unsqueeze(1), 0.0)
            r_kj_final = r_kj
            if step < self.n_iterations:
                weighted = omega.unsqueeze(1) * r_kj  # (B, K, M)
                denom = weighted.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # (B, K, 1)
                mu = (weighted.unsqueeze(-1) * support_xy.unsqueeze(1)).sum(dim=-2) / denom
        assert r_kj_final is not None

        # Mode mass is the ω-weighted assignment, NOT the raw r sum
        # (which under the new normalization would equal "# valid
        # support points / K" — a useless constant).
        mode_mass = (omega.unsqueeze(1) * r_kj_final).sum(dim=-1)  # (B, K)
        if cold_start.any():
            mode_mass = mode_mass.clone()
            mode_mass[cold_start] = 1e-12

        # Mode logits + optional context bias.
        mode_logits = mode_mass.clamp_min(1e-12).log()
        if self.use_context_correction:
            if self.history_dim > 0:
                assert history is not None
                bias_in = torch.cat([context, history], dim=-1)
            else:
                bias_in = context
            b_k = self.mode_bias(bias_in)  # (B, K)
            mode_logits = mode_logits + b_k

        return ModeExtractorOutputs(
            mode_logits=mode_logits,
            mode_mu=mu,
            mode_sigma=self.mode_sigma_buffer,
            mode_attention=r_kj_final,
            mode_mass=mode_mass,
            cold_start=cold_start,
        )


__all__ = [
    "DEFAULT_MODE_KERNEL_BANDWIDTH_FT",
    "DEFAULT_N_ITERATIONS",
    "SoftKMeansModeExtractor",
]
