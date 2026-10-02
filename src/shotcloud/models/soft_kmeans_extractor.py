"""Soft k-means / mean-shift mode extractor over attended support shots.

:class:`SoftKMeansModeExtractor` is an alternative to the learned-query
:class:`~shotcloud.models.mode_extractor.SupportModeExtractor`. Mode
centers are obtained by a few differentiable mean-shift iterations over
the attended support set, initialized by deterministic weighted
farthest-point sampling (FPS). Because each center is a kernel-weighted
average of nearby support shots, the modes are clusters of the row's own
support by construction; learned queries, being shared across rows, can
instead settle on broad league-typical regions that act as global modes.

Algorithm:

1. Initialize ``K`` seeds by weighted FPS on ``support_xy``:

   * seed 1 is the ω-weighted mean of the attended support,
     ``μ_1^{(0)} = (Σ_j ω_j s_j) / (Σ_j ω_j)``;
   * seeds 2..K are picked greedily as ``argmax_j ω_j · D_j²``, where
     ``D_j`` is the distance from ``s_j`` to the nearest selected seed.

2. For ``n_iterations`` steps:

   * normalized responsibility (soft assignment of each support point
     to modes),
     ``r_{k,j} = exp(-||s_j - μ_k||² / (2ρ²)) /
     Σ_ℓ exp(-||s_j - μ_ℓ||² / (2ρ²))``, so ``Σ_k r_{k,j} = 1``;
   * ω-weighted centroid update,
     ``μ_k = (Σ_j ω_j r_{k,j} s_j) / (Σ_j ω_j r_{k,j})``.

3. Mode mass, the share of attended support assigned to mode ``k``:
   ``m_k = Σ_j ω_j r_{k,j}``.
4. Mode logits ``log max(m_k, ε) + b_k(x_n, h_n)`` with an optional
   context-bias MLP whose last layer is zero-initialized.

The responsibility is a proper soft cluster assignment that depends on
geometry alone; the support attention ω enters only through the seeding,
the centroid update and the mode mass. Centers and masses are non-parametric; the only
learned parameters are those of the context-bias MLP.
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

#: Default Gaussian kernel bandwidth ρ (feet) for the mean-shift
#: responsibility: wide enough that a few iterations smooth
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

    1. The first seed is the ω-weighted mean of the attended support,
       ``μ_1 = (Σ_j ω_j s_j) / (Σ_j ω_j)``. It is a synthesized point,
       usually not one of the support shots; starting one mode at the
       support's center of mass is more stable than starting at the
       single highest-ω shot.
    2. Each subsequent seed scores every valid, not-yet-picked support
       point by ``ω_j · D_j²``, where ``D_j`` is the distance to the
       nearest already-selected seed, and takes the ``argmax`` (ties go
       to the lowest index).

    The picked indices are non-differentiable; the picked seeds are
    gathered from ``support_xy``. Gradient reaches ``ω`` through the
    weighted-mean seed.

    Returns ``(B, K, 2)`` seed coordinates.
    """
    b, m, _ = support_xy.shape
    if k > m:
        raise ValueError(f"requested K={k} seeds but only M={m} support points available")
    device = support_xy.device

    # First seed: weighted mean of attended support. ω is already masked
    # to zero on invalid entries by the caller; the denominator is still
    # clamped for cold-start rows (all ω = 0) so the division is finite.
    # Their seed is (0, 0), meaningless but finite; the caller's log-lik
    # floor overrides those rows.
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

    Interchangeable with
    :class:`~shotcloud.models.mode_extractor.SupportModeExtractor`: same
    forward signature, same
    :class:`~shotcloud.models.mode_extractor.ModeExtractorOutputs` return.

    Parameters
    ----------
    n_modes : int, default 6
        Number of modes ``K``.
    n_iterations : int, default 2
        Number of mean-shift refinement steps after the FPS init.
    kernel_bandwidth_ft : float, default 5.0
        ρ — Gaussian kernel bandwidth on support distances during
        the mean-shift iterations.
    mode_sigma_ft : float, default 3.0
        Per-mode density bandwidth σ_k (feet) for the downstream
        mode-mixture Gaussian, shared with the learned-query extractor.
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
        """Extract ``K`` modes from the attended support set by mean shift.

        Parameters and return value are as in
        :meth:`~shotcloud.models.mode_extractor.SupportModeExtractor.forward`;
        ``mode_attention`` holds the final responsibilities ``r_{k,j}``.
        """
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
        # ω in linear space, zeroed on invalid entries so FPS and mean
        # shift never see them. Cold-start rows still need finite seeds so
        # the forward pass doesn't produce NaN; the caller applies a
        # log-lik floor to them.
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
        # * responsibility ``r_{k,j} = softmax_k(-d²_{k,j} / (2ρ²))``
        #   so Σ_k r_{k,j} = 1 for every support point;
        # * centroid update ``μ_k = Σ_j ω_j r_{k,j} s_j / Σ_j ω_j r_{k,j}``;
        # * mode mass ``m_k = Σ_j ω_j r_{k,j}``.
        # One extra responsibility pass runs after the final
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
            # Mask out invalid support contributions. This happens AFTER
            # the softmax so the normalization is over the full mode
            # set; invalid points then simply contribute 0 mass.
            r_kj = r_kj.masked_fill(~support_mask.unsqueeze(1), 0.0)
            r_kj_final = r_kj
            if step < self.n_iterations:
                weighted = omega.unsqueeze(1) * r_kj  # (B, K, M)
                denom = weighted.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # (B, K, 1)
                mu = (weighted.unsqueeze(-1) * support_xy.unsqueeze(1)).sum(dim=-2) / denom
        assert r_kj_final is not None

        # Mode mass is the ω-weighted assignment, not the raw r sum: since
        # Σ_k r_{k,j} = 1, the raw sums total the number of valid support
        # points and carry no evidence weighting.
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
