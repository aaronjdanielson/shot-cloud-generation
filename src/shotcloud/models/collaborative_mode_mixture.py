"""Cell-free spatial decoder built on per-player-game mode extraction.

Wraps:

* a :class:`~shotcloud.models.CollaborativeKDE` for support attention,
* an optional residual encoder + location embedding (reweights the
  support attention by within-game context, same as the
  continuous-mixture path),
* a :class:`~shotcloud.models.mode_extractor.SupportModeExtractor`
  that compresses the attended support into K court modes,

into a single :class:`nn.Module` that returns per-row log-likelihood
under a small K-mode Gaussian mixture at the **exact observed shot
coordinate**.

Mode centers are inferred per row as convex combinations of that
row's support coordinates. There is no global learnable basis of
court modes — modes are *extracted* per (player, game, shot) from
that row's attended support set.

Parallel to :class:`~shotcloud.models.ContinuousMixtureSpatial`
(the support-shot mixture path). Both consume the same upstream
support attention; they differ only in how the per-shot density is
formed from the attention.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from shotcloud.models.collaborative_kde import CollaborativeKDE
from shotcloud.models.context_residual import ContextResidualEncoder
from shotcloud.models.continuous_mixture_spatial import ContinuousMixtureOutputs
from shotcloud.models.location_embedding import LocationEmbedding
from shotcloud.models.mode_extractor import (
    DEFAULT_MODE_QUERY_DIM,
    DEFAULT_MODE_SIGMA_FT,
    DEFAULT_N_COURT_MODES,
    SupportModeExtractor,
)
from shotcloud.models.soft_kmeans_extractor import (
    DEFAULT_MODE_KERNEL_BANDWIDTH_FT,
    DEFAULT_N_ITERATIONS,
    SoftKMeansModeExtractor,
)

#: Uniform-court-density floor on per-row log-likelihood when a row's
#: causal support is entirely empty.
DEFAULT_COLD_START_LOG_LIK_FLOOR: float = -7.86


class CollaborativeModeMixtureSpatial(nn.Module):
    """Per-player-game K-mode Gaussian mixture spatial decoder.

    Parameters
    ----------
    offensive_prior : CollaborativeKDE
        Source of support shots, ``A + B`` pre-softmax support
        scores, causal mask, σ.
    residual_encoder, residual_location_embedding : optional
        When provided together, the residual contributes
        ``R_θ(s_j) = u_θ^T ψ(s_j)`` to the support attention
        (identical to the continuous-mixture path).
    n_court_modes, mode_query_dim, mode_sigma_ft :
        Forwarded to :class:`SupportModeExtractor`. Defaults: K=6,
        d=32, σ=2.0 ft.
    use_context_correction : bool, default True
        Mode-logit context bias ``b_k(x_n, h_n)`` on/off.
    bias_hidden_dim : int, default 32
        Hidden width of the bias MLP.
    lambda_omega : float, default 0.0
        Mode-to-support attention's ω-bias weight; see
        :class:`SupportModeExtractor` for the full spec. ``0.0``
        (default, the strengthened-model spec) separates geometry
        (α from queries only) from mass (ω only enters m_k).
    tail_weight : float, default 0.0
        ``λ_tail`` — mixing weight on the raw support KDE component
        of the strengthened-model spec:
        ``f = (1 - λ_tail) f_mode + λ_tail f_support``. Defends
        undercoverage by giving observed shots a nonparametric
        safety valve outside the K-mode mixture. Spec recommends
        ``0.02–0.05``; default ``0.0`` preserves the pure mode-mixture
        behavior for back-compat with v1 mode_mixture experiments.
    tail_sigma_ft : float, default 1.0
        ``σ_sup`` — per-shot Gaussian bandwidth for the support-KDE
        tail component. Only consulted when ``tail_weight > 0``.
    cold_start_log_lik_floor : float
        Per-row log-likelihood for cold-start rows (all support
        masked out). Defaults to log of uniform density over a
        ~50×52 ft court.
    """

    def __init__(
        self,
        offensive_prior: CollaborativeKDE,
        residual_encoder: ContextResidualEncoder | None = None,
        residual_location_embedding: LocationEmbedding | None = None,
        n_court_modes: int = DEFAULT_N_COURT_MODES,
        mode_query_dim: int = DEFAULT_MODE_QUERY_DIM,
        mode_sigma_ft: float = DEFAULT_MODE_SIGMA_FT,
        use_context_correction: bool = True,
        bias_hidden_dim: int = 32,
        lambda_omega: float = 0.0,
        tail_weight: float = 0.0,
        tail_sigma_ft: float = 1.0,
        cold_start_log_lik_floor: float = DEFAULT_COLD_START_LOG_LIK_FLOOR,
        #: Which mode-extraction operator to use. ``"soft_kmeans"``
        #: (default, the 2026-05-18 pivot) runs deterministic weighted
        #: farthest-point sampling + a few mean-shift iterations over
        #: the attended support — modes are clusters of *this row's*
        #: support by construction. ``"learned_query"`` is the
        #: original learned-query extractor (kept for ablation; the
        #: empirical issue was that queries acted as global anchors).
        extractor_kind: str = "soft_kmeans",
        mode_kernel_bandwidth_ft: float = DEFAULT_MODE_KERNEL_BANDWIDTH_FT,
        mode_n_iterations: int = DEFAULT_N_ITERATIONS,
    ) -> None:
        super().__init__()
        if (residual_encoder is None) != (residual_location_embedding is None):
            raise ValueError(
                "residual_encoder and residual_location_embedding must both be "
                "provided or both None"
            )
        if (
            residual_encoder is not None
            and residual_location_embedding is not None
            and residual_encoder.rank != residual_location_embedding.rank
        ):
            raise ValueError(
                f"residual_encoder.rank ({residual_encoder.rank}) must equal "
                f"residual_location_embedding.rank "
                f"({residual_location_embedding.rank})"
            )

        if not (0.0 <= tail_weight <= 1.0):
            raise ValueError(f"tail_weight must be in [0, 1]; got {tail_weight}")
        if tail_sigma_ft <= 0.0:
            raise ValueError(f"tail_sigma_ft must be > 0; got {tail_sigma_ft}")

        self.offensive_prior = offensive_prior
        self.residual_encoder = residual_encoder
        self.residual_location_embedding = residual_location_embedding
        self.cold_start_log_lik_floor = float(cold_start_log_lik_floor)
        self.tail_weight = float(tail_weight)
        self.tail_sigma_ft = float(tail_sigma_ft)

        history_dim = residual_encoder.within_game_dim if residual_encoder is not None else 0
        self.mode_extractor: SupportModeExtractor | SoftKMeansModeExtractor
        if extractor_kind == "learned_query":
            self.mode_extractor = SupportModeExtractor(
                n_modes=n_court_modes,
                mode_query_dim=mode_query_dim,
                mode_sigma_ft=mode_sigma_ft,
                history_dim=history_dim,
                bias_hidden_dim=bias_hidden_dim,
                use_context_correction=use_context_correction,
                lambda_omega=lambda_omega,
            )
        elif extractor_kind == "soft_kmeans":
            self.mode_extractor = SoftKMeansModeExtractor(
                n_modes=n_court_modes,
                n_iterations=mode_n_iterations,
                kernel_bandwidth_ft=mode_kernel_bandwidth_ft,
                mode_sigma_ft=mode_sigma_ft,
                history_dim=history_dim,
                bias_hidden_dim=bias_hidden_dim,
                use_context_correction=use_context_correction,
            )
        else:
            raise ValueError(
                f"extractor_kind must be 'learned_query' or 'soft_kmeans'; got {extractor_kind!r}"
            )
        self.extractor_kind = extractor_kind

    @property
    def has_residual(self) -> bool:
        return self.residual_encoder is not None and self.residual_location_embedding is not None

    @property
    def n_court_modes(self) -> int:
        return self.mode_extractor.n_modes

    def _support_attention(
        self,
        collab_support_xy: Tensor,
        collab_support_logits: Tensor,
        collab_support_mask: Tensor,
        x_n: Tensor,
        h_n: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute support attention ``ω_j``. Returns
        ``(log_omega, residual_logits, cold_start)``.

        Mirrors the continuous-mixture path's outer softmax block so
        the two paths share the same support attention semantics.
        """
        logits = collab_support_logits
        residual_logits = torch.zeros_like(logits)
        if self.has_residual:
            assert self.residual_encoder is not None
            assert self.residual_location_embedding is not None
            if self.residual_encoder.within_game_dim > 0:
                if h_n is None:
                    raise ValueError("residual_encoder.within_game_dim > 0 → h_n is required")
                u = self.residual_encoder(x_n, h_n)
            else:
                u = self.residual_encoder(x_n)
            psi_r = self.residual_location_embedding(collab_support_xy)
            residual_logits = (u.unsqueeze(1) * psi_r).sum(dim=-1)
            logits = logits + residual_logits

        masked_logits = logits.masked_fill(~collab_support_mask, float("-inf"))
        cold_start = ~collab_support_mask.any(dim=-1)
        if cold_start.any():
            masked_logits = masked_logits.clone()
            masked_logits[cold_start, 0] = 0.0
        log_omega = masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)
        return log_omega, residual_logits, cold_start

    def forward(
        self,
        player_idx: Tensor,
        snapshot_idx: Tensor,
        x_n_raw: Tensor,
        x_n: Tensor,
        shot_xy: Tensor,
        h_n: Tensor | None = None,
    ) -> ContinuousMixtureOutputs:
        """Per-row log-likelihood of the observed shot under the
        per-player-game K-mode Gaussian mixture, optionally blended
        with the raw support-KDE tail."""
        from shotcloud.training.spatial_losses import (
            continuous_mixture_loglik,
            mode_mixture_loglik,
        )

        collab = self.offensive_prior.forward_continuous(player_idx, snapshot_idx, x_n_raw, x_n)
        b = player_idx.shape[0]

        log_omega, residual_logits, cold_start = self._support_attention(
            collab_support_xy=collab.support_xy,
            collab_support_logits=collab.support_logits,
            collab_support_mask=collab.support_mask,
            x_n=x_n,
            h_n=h_n,
        )

        mode_out = self.mode_extractor(
            support_xy=collab.support_xy,
            log_support_weights=log_omega,
            support_mask=collab.support_mask,
            context=x_n,
            history=h_n if self.mode_extractor.history_dim > 0 else None,
        )

        log_f_mode = mode_mixture_loglik(
            mode_logits=mode_out.mode_logits,
            mode_mu=mode_out.mode_mu,
            shot_xy=shot_xy,
            sigma=mode_out.mode_sigma,
        )

        # Strengthened-model support-tail mixture (paper §"strengthened"):
        # f(y) = (1 - λ_tail) f_mode(y) + λ_tail f_support(y).
        # f_support is the raw collab-support KDE evaluated at the
        # exact observed coord — the same component family used by
        # ContinuousMixtureSpatial but with its own fixed σ_sup. The
        # mixture defends against undercoverage: if all K modes drift
        # away from a particular shot, the support tail keeps the
        # density bounded below by λ_tail · f_support(y).
        tail_responsibility: Tensor | None = None
        if self.tail_weight > 0.0:
            sigma_sup = torch.full(
                (b,),
                self.tail_sigma_ft,
                dtype=log_f_mode.dtype,
                device=log_f_mode.device,
            )
            log_f_support = continuous_mixture_loglik(
                log_weights=log_omega,
                support_xy=collab.support_xy,
                shot_xy=shot_xy,
                sigma=sigma_sup,
                weights_are_log_probs=True,
                support_mask=collab.support_mask,
            )
            log_w_mode = math.log(1.0 - self.tail_weight)
            log_w_tail = math.log(self.tail_weight)
            log_a = log_w_mode + log_f_mode  # log of (1-λ)·f_mode
            log_b = log_w_tail + log_f_support  # log of λ·f_tail
            log_lik = torch.logsumexp(torch.stack([log_a, log_b], dim=-1), dim=-1)
            # γ_tail = exp(log_b - log_lik) = posterior responsibility
            # of the tail component for the observed shot. Stays in
            # [0, 1] by construction (logsumexp ≥ each summand).
            tail_responsibility = (log_b - log_lik).exp()
        else:
            log_lik = log_f_mode

        # Cold-start floor for rows whose support was entirely masked.
        if cold_start.any():
            log_lik = torch.where(
                cold_start,
                torch.full_like(log_lik, self.cold_start_log_lik_floor),
                log_lik,
            )
            if tail_responsibility is not None:
                # γ_tail is meaningless on cold-start rows (both f_mode
                # and f_tail are degenerate there); zero them so the
                # diagnostic mean doesn't get poisoned.
                tail_responsibility = torch.where(
                    cold_start, torch.zeros_like(tail_responsibility), tail_responsibility
                )

        # Normalize mode logits for the output's log_weights field.
        log_pi = mode_out.mode_logits - torch.logsumexp(mode_out.mode_logits, dim=-1, keepdim=True)

        device = log_lik.device
        return ContinuousMixtureOutputs(
            log_lik=log_lik,
            log_weights=log_pi,  # (B, K)
            support_xy=mode_out.mode_mu,  # (B, K, 2) — per-row mode centers
            sigma=mode_out.mode_sigma.mean().expand(b),
            support_mask=torch.ones(b, self.n_court_modes, dtype=torch.bool, device=device),
            residual_logits=residual_logits,
            collab=collab,
            cold_start=cold_start,
            support_log_weights=log_omega,
            tail_responsibility=tail_responsibility,
        )


__all__ = [
    "DEFAULT_COLD_START_LOG_LIK_FLOOR",
    "CollaborativeModeMixtureSpatial",
]
