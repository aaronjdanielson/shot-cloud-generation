"""Player-specific mode extractor from attended causal shot history.

Implements Phase 2 of the collaborative mode-extraction plan: takes
the support-attention output of an upstream collaborative scorer and
compresses it into a small per-row Gaussian mixture over learned
court modes.

**Mode centers are convex combinations of the row's support
coordinates.** ``μ_{b,k} = Σ_j α_{b,k,j} s_{b,j}`` with
``α_{b,k,j} ≥ 0`` summing to 1 over ``j``. This guarantees modes
live in regions the target/analogue players actually shoot from —
the modes are *inferred per (player, game, shot)*, not
fixed/learnable as a global basis.

Pipeline:

1. Embed each support coord: ``e_j = ψ(s_j)`` via a learned
   :class:`~shotcloud.models.LocationEmbedding`.
2. K learnable mode queries ``q_k ∈ R^d``.
3. Mode-to-support attention biased by support-attention log-prob:

       α_{k,j} = softmax_j(q_k^T e_j / √d + log ω_j).

4. Mode centers ``μ_k = Σ_j α_{k,j} s_j``.
5. Mode evidence ``m_k = Σ_j ω_j α_{k,j}``.
6. Mode logits ``ℓ_k = log m_k + b_k(x, h)`` where ``b_k`` is an
   optional small context MLP (zero-init last layer → driven by
   evidence at step 0).

Cold-start safety: rows whose support set is entirely masked
have their dummy-slot-0 patched (matching the upstream
support-attention path) so the per-mode softmax stays well-
defined; the trainer applies a uniform-court-density floor on
``log_lik`` for those rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from shotcloud.data.context import CONTEXT_DIM
from shotcloud.models.location_embedding import LocationEmbedding

#: Default ``K`` (number of court modes). Plan §7 v1 recommends K=6.
DEFAULT_N_COURT_MODES: int = 6

#: Default dimension of the support embedding ``e_j`` and mode
#: queries ``q_k`` for the Q-K product.
DEFAULT_MODE_QUERY_DIM: int = 32

#: Default fixed per-mode density bandwidth σ in feet. Bumped from
#: 2.0 → 3.0 on 2026-05-18 after the H2/H3 mode plots showed the
#: 2-ft σ circles undercovered the local cluster spread; 3 ft is
#: closer to the empirical mass radius and keeps log-lik smoother.
DEFAULT_MODE_SIGMA_FT: float = 3.0


@dataclass(frozen=True)
class ModeExtractorOutputs:
    """Output bundle of :class:`SupportModeExtractor.forward`.

    Carries everything the trainer needs for both the mode-mixture
    NLL and the per-epoch mode diagnostics.
    """

    mode_logits: Tensor  # (B, K) pre-softmax mode scores ℓ_k
    mode_mu: Tensor  # (B, K, 2) convex-combination mode centers
    mode_sigma: Tensor  # (K,) or (B, K) — per-mode density bandwidth
    mode_attention: Tensor  # (B, K, M) mode-to-support α_{k,j}
    mode_mass: Tensor  # (B, K) support mass captured by each mode m_k
    cold_start: Tensor  # (B,) bool — rows with no valid causal support


class SupportModeExtractor(nn.Module):
    """Mode-extraction head over attended support shots.

    Construction-only piece; no support scoring inside. The caller
    (typically :class:`CollaborativeModeMixtureSpatial`) supplies the
    support attention via ``log_support_weights`` and the support
    coordinates / mask. This keeps mode extraction independent of
    *which* support scorer produced the attention.

    Parameters
    ----------
    n_modes : int, default 6
        K — number of court modes the extractor produces.
    mode_query_dim : int, default 32
        d — dimension of the support embedding and mode queries.
    mode_sigma_ft : float, default 2.0
        Per-mode fixed isotropic density bandwidth in feet.
    context_dim : int, default :data:`CONTEXT_DIM` (27)
        Width of ``context`` input to ``b_k``.
    history_dim : int, default 0
        Width of ``history`` input concatenated into ``b_k``'s input.
        0 means no within-game-history channel.
    bias_hidden_dim : int, default 32
        Hidden width of the ``b_k`` context-bias MLP.
    use_context_correction : bool, default True
        When True, add the context bias ``b_k(x_n, h_n)`` to mode
        logits. When False, mode logits are driven purely by the
        support evidence ``log m_k`` (no context correction).
    lambda_omega : float, default 0.0
        Weight on the ``log ω_j`` bias inside the mode-to-support
        attention. ``0.0`` (default, the strengthened-model spec):
        α uses **pure geometry** — α_{k,j} = softmax_j(q_k^T e_j / √d)
        — and ω only enters the mode mass m_k = Σ_j ω_j α_{k,j}.
        ``1.0``: the original support-weighted extraction
        α_{k,j} = softmax_j(q_k^T e_j / √d + log ω_j). Intermediate
        values blend the two. Separating geometry from mass
        prevents high-ω shots from dominating both center formation
        and mode probability.
    """

    mode_queries: Tensor  # (K, d) — nn.Parameter
    mode_sigma_buffer: Tensor  # (K,) fixed buffer

    def __init__(
        self,
        n_modes: int = DEFAULT_N_COURT_MODES,
        mode_query_dim: int = DEFAULT_MODE_QUERY_DIM,
        mode_sigma_ft: float = DEFAULT_MODE_SIGMA_FT,
        context_dim: int = CONTEXT_DIM,
        history_dim: int = 0,
        bias_hidden_dim: int = 32,
        use_context_correction: bool = True,
        lambda_omega: float = 0.0,
    ) -> None:
        super().__init__()
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive; got {n_modes}")
        if mode_query_dim <= 0:
            raise ValueError(f"mode_query_dim must be positive; got {mode_query_dim}")
        if mode_sigma_ft <= 0:
            raise ValueError(f"mode_sigma_ft must be positive; got {mode_sigma_ft}")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive; got {context_dim}")
        if history_dim < 0:
            raise ValueError(f"history_dim must be non-negative; got {history_dim}")
        if not (0.0 <= lambda_omega <= 1.0):
            raise ValueError(f"lambda_omega must be in [0, 1]; got {lambda_omega}")

        self.n_modes = int(n_modes)
        self.mode_query_dim = int(mode_query_dim)
        self.use_context_correction = bool(use_context_correction)
        self.history_dim = int(history_dim)
        self.lambda_omega = float(lambda_omega)

        # Support embedding ψ(s_j) — must carry spatial signal at
        # step 0 (not paired with a V=0 partner), so non-zero init.
        self.support_embedding = LocationEmbedding(rank=mode_query_dim, zero_init=False)

        # K learnable mode queries; small random init so the K queries
        # are distinct at step 0 (otherwise every mode would extract
        # the same cluster).
        self.mode_queries = nn.Parameter(
            torch.randn(n_modes, mode_query_dim) * (1.0 / math.sqrt(mode_query_dim))
        )

        # Context bias MLP b_k(x_n, h_n) → K logits. Zero-init last
        # layer so π_k starts driven purely by support evidence m_k.
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

        # Fixed σ_k buffer.
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
        """Produce per-row mode logits, centers, and diagnostics.

        Parameters
        ----------
        support_xy : Tensor of shape ``(B, M, 2)``
            Per-row support coordinates in court feet.
        log_support_weights : Tensor of shape ``(B, M)``
            Normalized log support-attention weights ``log ω_j``;
            ``-inf`` on masked entries (the upstream scorer is
            responsible for that).
        support_mask : Tensor of shape ``(B, M)`` bool
            True where the support shot is real + causal.
        context : Tensor of shape ``(B, context_dim)``
            Per-shot context input to the mode-bias MLP.
        history : Tensor of shape ``(B, history_dim)`` or ``None``
            Within-game history features; required when
            ``history_dim > 0``.

        Returns
        -------
        ModeExtractorOutputs
            ``mode_logits``, ``mode_mu``, ``mode_sigma``,
            ``mode_attention``, ``mode_mass``, ``cold_start``.
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
        if context.shape != (b, self.support_embedding.proj.in_features + 0) and (
            context.shape[0] != b
        ):
            # We only enforce batch consistency; downstream MLP will
            # error on the per-feature shape if wrong.
            raise ValueError(f"context first dim must be B={b}; got {tuple(context.shape)}")
        if self.history_dim > 0:
            if history is None:
                raise ValueError(f"history_dim={self.history_dim} > 0 → history is required")
            if history.shape != (b, self.history_dim):
                raise ValueError(
                    f"history must be (B, {self.history_dim}); got {tuple(history.shape)}"
                )

        cold_start = ~support_mask.any(dim=-1)  # (B,)
        # ψ(s_j) — (B, M, d).
        e_j = self.support_embedding(support_xy)
        # Q-K product: (K, d) × (B, M, d) → (B, K, M); scale by sqrt(d).
        qk_logits = torch.einsum("kd,bmd->bkm", self.mode_queries, e_j) / math.sqrt(
            self.mode_query_dim
        )
        # Optional ω-bias (off by default per the strengthened-model
        # spec — keeps α purely geometric so high-ω shots don't
        # dominate both center formation and mode mass).
        if self.lambda_omega > 0.0:
            qk_logits = qk_logits + self.lambda_omega * log_support_weights.unsqueeze(1)
        # Mask invalid support shots from per-mode softmax.
        qk_logits = qk_logits.masked_fill(~support_mask.unsqueeze(1), float("-inf"))
        # Cold-start safety: all-True masking sends every entry to
        # -inf → softmax NaN. Patch slot 0 with finite 0 per mode.
        # The caller / trainer applies a log_lik floor for these rows.
        if cold_start.any():
            qk_logits = qk_logits.clone()
            qk_logits[cold_start, :, 0] = 0.0
        alpha = torch.softmax(qk_logits, dim=-1)  # (B, K, M)

        # Convex-combination mode centers.
        mu = torch.einsum("bkm,bmd->bkd", alpha, support_xy)  # (B, K, 2)

        # Mode evidence m_k = Σ_j ω_j α_{k,j}.
        omega = log_support_weights.exp()  # (B, M)
        if cold_start.any():
            # ω for cold-start rows is ~0 everywhere after the upstream
            # dummy patch; that's fine — m_k is small but finite.
            omega = omega.clone()
            omega[cold_start] = 0.0
        mode_mass = (alpha * omega.unsqueeze(1)).sum(dim=-1)  # (B, K)

        # Mode logits ℓ_k = log m_k + b_k(x, h).
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
            mode_attention=alpha,
            mode_mass=mode_mass,
            cold_start=cold_start,
        )


__all__ = [
    "DEFAULT_MODE_QUERY_DIM",
    "DEFAULT_MODE_SIGMA_FT",
    "DEFAULT_N_COURT_MODES",
    "ModeExtractorOutputs",
    "SupportModeExtractor",
]
