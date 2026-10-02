"""Collaborative adaptive KDE — the headline spatial decoder (v1.1).

Implements

.. math::

    \\hat q_p^{\\mathrm{collab}}(c \\mid x_n, t_n)
    = \\sum_{p' \\in \\mathcal N_p^{+}(t_n)} \\alpha_{p,p'}(x_n, t_n)
      \\sum_{j \\in \\mathcal H_{p'}^{<t_n}} \\beta_{p,p',j}(x_n, t_n)
      K_{\\sigma_p(t_m)}(x_c - s_j),

per the canonical spec in
[docs/model_spec.md](../../../docs/model_spec.md). Two-stage attention
over a retrieved analogue set, summed against a per-target isotropic
Gaussian kernel whose bandwidth depends on the target player's
recency-weighted evidence volume.

v1.1 refactor (vs the v1.0 dense forward)
-----------------------------------------

Three changes, jointly motivated by the v1.0 forward being ~50× slower
than the underlying memory-bandwidth ceiling on M-series GPUs:

1. **Separable kernel**. The dense per-shot Gaussian
   ``K(c - s_j) = exp(-||c-s_j||²/2σ²)`` factors exactly into
   ``K_x(c_x - s_{j,x}) · K_y(c_y - s_{j,y})``. Aggregating via a bmm
   over the ``(L*R)`` shot axis avoids materializing the dominant
   ``(B, L, R, n_cells)`` intermediate. See
   :func:`shotcloud.models._separable_kernel.separable_gaussian_density`.
2. **Global support pool + per-player int64 index**. Per-shot context,
   coords, and dates live in a single flat
   :class:`shotcloud.models._support_pool.GlobalSupportPool`; the
   per-player history is an ``(n_players, max_R)`` int64 index. The
   pool is the precondition for change 3.
3. **Bilinear shot attention with batch-unique ``h_θ`` compute**.
   ``g_θ(x, z_j) = f_θ(x)^T h_θ(z_j) - λ_age Δt`` factors the v1.0
   concat MLP so ``h_θ(z_j)`` depends only on the shot. We run
   ``h_θ`` on the unique shot ids in the batch
   (``torch.unique(..., return_inverse=True)``) and gather back, so
   duplicates across batch rows pay the MLP cost once. The legacy
   concat form is preserved as an ablation knob
   (``shot_attention_form="concat"``).

Mathematical equivalence at step 0 is preserved: with the bilinear
``h_θ``'s output layer zero-initialized, ``g_θ ≡ 0`` for every shot,
so β is uniform within each analogue's causal history — identical
step-0 distribution to the v1.0 concat form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import Tensor, nn

from shotcloud.data.context import CONTEXT_DIM
from shotcloud.data.player_traits import (
    M_PLAY_SLOT,
    TRAIT_DIM,
    PlayerTraitsTable,
)
from shotcloud.grids import CourtGrid
from shotcloud.kde.adaptive import AdaptiveKDE
from shotcloud.models._separable_kernel import separable_gaussian_density
from shotcloud.models._support_pool import build_support_pool_from_adaptive_kde
from shotcloud.models.analogue_retrieval import AnalogueRetrievalCache

if TYPE_CHECKING:
    from shotcloud.data.snapshots import SnapshotStore
    from shotcloud.training.dataset import PlayerVocab

# Trait slots from PlayerTraitsTable used at forward time.
# Block B slots (multiplied by missingness in the table itself):
_SLOT_LOG1P_M: int = 14
_SLOT_LOG1P_S: int = 15
_SLOT_LOG_DENSITY: int = 16

ShotAttentionForm = Literal["bilinear", "concat"]
HzInit = Literal["zero", "warm"]
AlphaPrior = Literal["none", "similarity"]

# Standard deviation of the warm h_z output-layer normal init, in score
# units. With proj_dim=32 and unit-scale f_x at init, this yields per-
# shot scores of order O(1e-3 * sqrt(32) * f_x_scale) ≈ O(1e-4), which
# is small enough that β remains near-uniform (KL to uniform < 1e-4 by
# construction) but large enough that the f_x side of the bilinear
# bootstrap receives nonzero gradient at step 0.
_H_Z_WARM_INIT_STD: float = 1e-3


def _inverse_sigmoid(p: float) -> float:
    """Inverse sigmoid (logit). Used to initialize ``a_0`` so that
    ``σ = sigma_init`` at training start when the other coefficients
    are at zero."""
    return float(np.log(p / (1.0 - p)))


def _cosine_similarity_bl(u_self: Tensor, u_other: Tensor, *, eps: float = 1e-12) -> Tensor:
    """``(B, L)`` cosine similarity between paired ``(B, L, D)`` vectors.

    Handles the all-zero-trait edge case (rookies with no biographical
    record AND no usable play traits) by ``clamp_min`` on the
    denominator — pairs with a zero side get similarity 0 instead of
    NaN, which is the right neutral value (no signal either way).
    """
    num = (u_self * u_other).sum(dim=-1)
    denom = (u_self.norm(dim=-1) * u_other.norm(dim=-1)).clamp_min(eps)
    out: Tensor = num / denom
    return out


@dataclass(frozen=True)
class CollaborativeOutputs:
    """Per-batch intermediates from one CollaborativeKDE forward pass.

    Returned when ``return_components=True``. Useful for the
    diagnostic script and for the future paper-ablation comparisons.
    """

    log_q_collab: Tensor  # (B, n_cells)
    alpha: Tensor  # (B, L)         — player-level attention, rows sum to 1
    beta: Tensor  # (B, L, R)       — shot-level attention; rows sum to 1 per
    #                                 (b, l) with valid causal history,
    #                                 else identically 0
    sigma: Tensor  # (B,)           — per-target bandwidth in ft
    analogues: Tensor  # (B, L)     — int64 analogue vocab indices
    has_analogue_history: Tensor  # (B, L) bool — N_p^+(t_n) membership


@dataclass(frozen=True)
class CollaborativeContinuousOutputs:
    """Cell-free continuous-mixture ingredients from one forward pass.

    Returned by :meth:`CollaborativeKDE.forward_continuous`. The
    consumer (``ContinuousMixtureSpatial``) adds residual + defense
    contributions to ``support_logits`` and then computes
    :func:`shotcloud.training.spatial_losses.continuous_mixture_nll`
    on the resulting joint scores.

    The mixture density at the observed shot is

    .. math::

        f_\\Theta(y) = \\sum_m w_m\\, K_{\\sigma}(y - s_m),
        \\quad
        w_m = \\operatorname{softmax}_m(A + B + R + D),

    where ``m = (l, r)`` flattens the analogue × support-shot axes.
    """

    support_xy: Tensor  # (B, M, 2) — support shot coordinates (M = L * R)
    support_logits: Tensor  # (B, M) — A_l + B_{l,r} (pre-softmax; R, D added by caller)
    sigma: Tensor  # (B,)         — per-target bandwidth in ft
    support_mask: Tensor  # (B, M) bool — True where the support shot is real + causal
    own_mask: Tensor  # (B, M) bool — subset of ``support_mask`` whose shooter
    #                                 is the target player. Together with
    #                                 ``support_mask``, partitions the valid
    #                                 support into own vs pooled subsets.
    #                                 Backend-agnostic contract for the
    #                                 pooling gate (PR3): both the L×R
    #                                 collaborative backend and the upcoming
    #                                 retrieval backend populate this field.
    support_shooter: Tensor  # (B, M) int64 — vocab index of each support
    #                                         slot's shooter. Backend-agnostic
    #                                         per-slot identity. For invalid
    #                                         (~support_mask) slots the value
    #                                         is unspecified — callers must
    #                                         apply ``support_mask`` first.
    analogue_idx: Tensor  # (B, L)  — int64 analogue vocab indices (diagnostic;
    #                                 fixed_lr backend only — retrieval emits
    #                                 ``(B, 0)``).
    alpha_scores: Tensor  # (B, L)  — A_l pre-softmax (diagnostic; fixed_lr).
    beta_scores: Tensor  # (B, L, R) — B_{l,r} pre-softmax (diagnostic; fixed_lr).


class CollaborativeKDE(nn.Module):
    """Per :doc:`docs/model_spec.md`, the q_collab spatial decoder.

    Parameters
    ----------
    adaptive_kde : AdaptiveKDE
        Fitted per-player history (cells, context, coords, dates).
        Should be fit with ``max_history=R`` and ``history_policy=random``
        per the v1 plan.
    snapshot_store : SnapshotStore
        Provides anchor dates used by the causal date mask.
    traits_table : PlayerTraitsTable
        26-d per-(player, snapshot) trait vector.
    analogue_cache : AnalogueRetrievalCache
        Precomputed top-L analogues per (player, snapshot).
    vocab : PlayerVocab
        Player-id ↔ idx mapping; the three tables above must all be
        keyed to ``vocab.ids`` in the same order.
    grid : CourtGrid
        Court grid; the module pulls ``xcenters`` and ``ycenters`` for
        the separable Gaussian kernel. The grid's image-layout
        convention (``c = iy*nx + ix``) is what the returned ``log_q``
        is ravelled to.
    sigma_min, sigma_max : float
        Bandwidth bounds (feet). Defaults match the anisotropic
        kernel: ``0.75`` and ``4.0``.
    sigma_init : float
        Target bandwidth at step 0 (when σ-MLP coefficients are zero).
        Default ``1.5`` matches the legacy fixed bandwidth so the
        model reduces to "uniform analogue average × isotropic 1.5 ft
        Gaussian" at training start.
    phi_hidden_dim : int
        Hidden width of the player-attention MLP φ_θ.
    shot_attention_form : {"bilinear", "concat"}
        ``"bilinear"`` (default) factors the shot-attention scoring as
        ``g_θ(x, z) = f_θ(x)^T h_θ(z)`` so the per-shot ``h_θ`` is
        computed on the batch's *unique* shot ids only. ``"concat"``
        recovers the v1.0 ``g_θ([x; z])`` ablation form.
    shot_hidden_dim : int
        Hidden width of the shot-attention MLP(s) (g_θ in concat form,
        f_θ and h_θ in bilinear form).
    shot_proj_dim : int
        Output dimension of the bilinear projections f_θ, h_θ. Ignored
        for ``shot_attention_form="concat"``.
    h_z_init : {"zero", "warm"}
        Initialization of the bilinear shot-attention ``h_θ`` output
        layer. ``"zero"`` (default) sets weights and bias to exactly
        zero — this preserves the exact step-0 invariant
        ``g_θ ≡ 0 → β uniform → log_q_bilinear == log_q_concat`` (see
        :func:`tests.test_models_collaborative_kde.test_step0_bilinear_and_concat_produce_identical_log_q`).
        ``"warm"`` initializes ``h_θ`` weights with small normal noise
        (σ = 1e-3) and bias zero — β remains near-uniform at step 0
        (KL to uniform < 1e-4) but the ``f_θ`` side of the bilinear
        score immediately receives nonzero gradient through the chain
        rule. Use ``"warm"`` if the strict-init dual-zero saddle slows
        early spatial learning. Ignored for
        ``shot_attention_form="concat"``.
    eps : float
        Additive smoothing constant for the per-cell density. Matches
        the 2026-05-16 Decision-1 convention.
    """

    # Type hints for registered buffers (mypy).
    pool_coords: Tensor
    pool_context: Tensor
    pool_dates: Tensor
    pool_index: Tensor
    anchor_dates: Tensor
    xcenters: Tensor
    ycenters: Tensor
    traits: Tensor
    analogues: Tensor

    def __init__(
        self,
        adaptive_kde: AdaptiveKDE,
        snapshot_store: SnapshotStore,
        traits_table: PlayerTraitsTable,
        analogue_cache: AnalogueRetrievalCache,
        vocab: PlayerVocab,
        grid: CourtGrid,
        *,
        sigma_min: float = 0.75,
        sigma_max: float = 4.0,
        sigma_init: float = 1.5,
        phi_hidden_dim: int = 64,
        shot_attention_form: ShotAttentionForm = "bilinear",
        shot_hidden_dim: int = 64,
        shot_proj_dim: int = 32,
        h_z_init: HzInit = "zero",
        alpha_prior: AlphaPrior = "none",
        alpha_prior_scale_init: float = 2.0,
        same_player_bias_init: float = 0.0,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        if not adaptive_kde.is_fitted:
            raise ValueError("adaptive_kde must be fit before being passed to CollaborativeKDE")
        if not adaptive_kde.dates:
            raise ValueError("adaptive_kde must be fit with date= so the causal mask works")
        if not adaptive_kde.coords:
            raise ValueError("adaptive_kde must be fit on a version that populates per-shot coords")
        # ``sigma_min == sigma_max`` is allowed: it locks σ at that
        # value (the bandwidth-sigmoid formula collapses to a constant
        # because ``sigma_range = 0``). Use this to ablate the
        # learnable-bandwidth machinery — the trainer's σ-mean
        # diagnostic will be exactly the fixed value.
        if not (0.0 < sigma_min <= sigma_max):
            raise ValueError(f"require 0 < sigma_min ({sigma_min}) <= sigma_max ({sigma_max})")
        if not (sigma_min <= sigma_init <= sigma_max):
            raise ValueError(f"sigma_init ({sigma_init}) must lie in [sigma_min, sigma_max]")
        if traits_table.trait_dim != TRAIT_DIM:
            raise ValueError(f"traits_table trait_dim={traits_table.trait_dim} != {TRAIT_DIM}")
        if shot_attention_form not in ("bilinear", "concat"):
            raise ValueError(
                f"shot_attention_form must be 'bilinear' or 'concat'; got {shot_attention_form!r}"
            )
        if h_z_init not in ("zero", "warm"):
            raise ValueError(f"h_z_init must be 'zero' or 'warm'; got {h_z_init!r}")
        if alpha_prior not in ("none", "similarity"):
            raise ValueError(f"alpha_prior must be 'none' or 'similarity'; got {alpha_prior!r}")

        n_players = len(vocab)
        if traits_table.n_players != n_players:
            raise ValueError(
                f"traits_table has {traits_table.n_players} players, vocab has {n_players}"
            )
        if analogue_cache.n_players != n_players:
            raise ValueError(
                f"analogue_cache has {analogue_cache.n_players} players, vocab has {n_players}"
            )
        if traits_table.n_snapshots != analogue_cache.n_snapshots:
            raise ValueError("traits_table and analogue_cache disagree on n_snapshots")

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_range = self.sigma_max - self.sigma_min
        self.sigma_init = float(sigma_init)
        self.eps = float(eps)
        self.shot_attention_form: ShotAttentionForm = shot_attention_form
        self._L = int(analogue_cache.L)

        # ---- Anchor dates buffer (n_snapshots,) int64 epoch days. ----
        anchors_np = np.array(
            [
                np.asarray(b.anchor_date, dtype="datetime64[D]").astype(np.int64)
                for b in snapshot_store.bundles
            ],
            dtype=np.int64,
        )
        self.register_buffer("anchor_dates", torch.from_numpy(anchors_np), persistent=False)

        # ---- Global support pool + per-player int64 index ----
        pool, index = build_support_pool_from_adaptive_kde(adaptive_kde, vocab)
        self.register_buffer("pool_coords", pool.coords, persistent=False)
        self.register_buffer("pool_context", pool.context, persistent=False)
        self.register_buffer("pool_dates", pool.dates, persistent=False)
        self.register_buffer("pool_index", index.index, persistent=False)
        self._R = index.max_R
        self._n_pool = pool.n_shots

        # ---- Per-axis cell centers (separable kernel) ----
        self.register_buffer(
            "xcenters",
            torch.from_numpy(np.ascontiguousarray(grid.xcenters)).to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "ycenters",
            torch.from_numpy(np.ascontiguousarray(grid.ycenters)).to(torch.float32),
            persistent=False,
        )
        self._nx = int(grid.nx)
        self._ny = int(grid.ny)
        self._n_cells = self._nx * self._ny

        # ---- Traits and analogue cache ----
        self.register_buffer(
            "traits",
            torch.from_numpy(traits_table.traits.copy()).to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "analogues",
            torch.from_numpy(analogue_cache.analogues.copy()).to(torch.int64),
            persistent=False,
        )

        # ---- Learnable parameters ----
        # Player-level attention φ_θ.
        # Input: [u_self, u_other, u_self - u_other, u_self ⊙ u_other, x_n]
        # = (4 × TRAIT_DIM + CONTEXT_DIM)
        phi_in = 4 * TRAIT_DIM + CONTEXT_DIM
        phi_out_layer = nn.Linear(phi_hidden_dim, 1)
        nn.init.zeros_(phi_out_layer.weight)
        nn.init.zeros_(phi_out_layer.bias)
        self.phi = nn.Sequential(
            nn.Linear(phi_in, phi_hidden_dim),
            nn.GELU(),
            phi_out_layer,
        )

        # Player-level scalar terms.
        # ``b_same`` boosts the analogue slot that equals the target
        # player. Initialized to ``same_player_bias_init`` (default 0
        # for back-compat). Per the 2026-05-17 α-prior intervention,
        # values like +1.0 prevent the target's own causal history from
        # being washed out by retrieval analogues.
        self.b_same = nn.Parameter(torch.tensor(float(same_player_bias_init), dtype=torch.float32))
        self.lambda_M = nn.Parameter(torch.zeros(()))
        self.lambda_S = nn.Parameter(torch.zeros(()))
        # Similarity-prior coefficient γ on cos(u_self, u_other). When
        # ``alpha_prior="similarity"`` it's a learnable scalar
        # initialized to ``alpha_prior_scale_init`` (typically 2.0).
        # When ``alpha_prior="none"`` it's a no-grad zero buffer so
        # downstream code path stays identical structurally.
        self.alpha_prior: AlphaPrior = alpha_prior
        if alpha_prior == "similarity":
            self.gamma_sim = nn.Parameter(
                torch.tensor(float(alpha_prior_scale_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "gamma_sim", torch.zeros((), dtype=torch.float32), persistent=False
            )

        # Shot-level attention.
        # Bilinear form (default):
        #   g_θ(x, z) = f_θ(x)^T h_θ(z) - λ_age Δt
        # The h_θ output layer is either:
        #   - "zero":  W=0, b=0 → g_θ ≡ 0 → β uniform at step 0 (exact
        #              equivalence to concat-form zero-init via
        #              softmax(0). f_θ side has zero gradient at step 0
        #              until h_θ lifts off — the "dual saddle" that
        #              motivates the warm option below.
        #   - "warm":  W ~ N(0, 1e-3), b=0 → β stays near-uniform
        #              (KL to uniform < 1e-4 at init) but g_θ has small
        #              nonzero score, so f_θ's chain-rule gradient is
        #              nonzero from step 1.
        # Concat form (ablation):
        #   g_θ([x; z]) MLP with output layer zero-init.
        if shot_attention_form == "bilinear":
            f_out = nn.Linear(shot_hidden_dim, shot_proj_dim)
            h_out = nn.Linear(shot_hidden_dim, shot_proj_dim)
            if h_z_init == "zero":
                nn.init.zeros_(h_out.weight)
            else:  # "warm"
                nn.init.normal_(h_out.weight, mean=0.0, std=_H_Z_WARM_INIT_STD)
            nn.init.zeros_(h_out.bias)
            self.f_x = nn.Sequential(
                nn.Linear(CONTEXT_DIM, shot_hidden_dim),
                nn.GELU(),
                f_out,
            )
            self.h_z = nn.Sequential(
                nn.Linear(CONTEXT_DIM, shot_hidden_dim),
                nn.GELU(),
                h_out,
            )
            self._shot_proj_dim = int(shot_proj_dim)
            self.h_z_init: HzInit = h_z_init
        else:  # concat
            g_in = 2 * CONTEXT_DIM
            g_out_layer = nn.Linear(shot_hidden_dim, 1)
            nn.init.zeros_(g_out_layer.weight)
            nn.init.zeros_(g_out_layer.bias)
            self.g = nn.Sequential(
                nn.Linear(g_in, shot_hidden_dim),
                nn.GELU(),
                g_out_layer,
            )
            self._shot_proj_dim = 0
            # h_z_init is bilinear-only but we record the requested value
            # for diagnostic round-trip (extra_repr, manifest reads).
            self.h_z_init = h_z_init

        # Shot-level recency-decay rate.
        self.lambda_age = nn.Parameter(torch.zeros(()))

        # Bandwidth scalars (a_0, a_M, a_S, a_R). At step 0 we want
        # σ_p(t_m) = sigma_init for every player regardless of evidence.
        # The formula is
        #   σ = σ_min + (σ_max − σ_min) · sigmoid[
        #       a_0 − softplus(a_M)·log1p_M − softplus(a_S)·log1p_S + a_R·log_density
        #   ]
        # so we need softplus(a_M) ≈ softplus(a_S) ≈ 0 at init (NOT
        # a_M = a_S = 0, because softplus(0) = ln(2) ≈ 0.693 already
        # pulls σ toward σ_min for any player with non-trivial evidence).
        # Init a_M = a_S = -10 → softplus ≈ 4.5e-5 → the evidence terms
        # contribute negligibly at step 0, leaving σ = σ_min +
        # (σ_max − σ_min)·sigmoid(a_0) = sigma_init.
        # When σ is locked (sigma_range == 0) the bandwidth formula
        # collapses to ``σ ≡ σ_min``; a_0 has no effect on the output,
        # so initialize it to 0 to skip the inverse-sigmoid divide.
        if self.sigma_range > 0:
            a0_init = _inverse_sigmoid((self.sigma_init - self.sigma_min) / self.sigma_range)
        else:
            a0_init = 0.0
        self.a_0 = nn.Parameter(torch.tensor(a0_init, dtype=torch.float32))
        self.a_M = nn.Parameter(torch.tensor(-10.0, dtype=torch.float32))
        self.a_S = nn.Parameter(torch.tensor(-10.0, dtype=torch.float32))
        self.a_R = nn.Parameter(torch.zeros(()))

    @property
    def L(self) -> int:  # noqa: N802 — matches the spec's notation `L = |N_p^(t_m)|`
        return self._L

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def max_history(self) -> int:
        """Per-player capacity ``R`` from the underlying ``AdaptiveKDE``."""
        return self._R

    @property
    def n_pool_shots(self) -> int:
        """Total unique shots across all per-player histories."""
        return self._n_pool

    # -----------------------------------------------------------------
    # Forward sub-steps
    # -----------------------------------------------------------------

    def _gather_analogues(self, player_idx: Tensor, snapshot_idx: Tensor) -> Tensor:
        """Return ``(B, L)`` int64 analogue vocab indices for the batch."""
        return self.analogues[player_idx, snapshot_idx]

    def _gather_analogue_traits(self, analogue_idx: Tensor, snapshot_idx: Tensor) -> Tensor:
        """Gather analogue trait vectors at the batch's snapshot.

        ``analogue_idx``: ``(B, L)`` int64. ``snapshot_idx``: ``(B,)`` int64.
        Returns ``(B, L, TRAIT_DIM)`` float32.
        """
        # Expand snapshot_idx so each row's analogues read the right snapshot column.
        snap_b_l = snapshot_idx.unsqueeze(-1).expand_as(analogue_idx)  # (B, L)
        return self.traits[analogue_idx, snap_b_l]  # (B, L, TRAIT_DIM)

    def _compute_alpha_scores(
        self,
        player_idx: Tensor,
        snapshot_idx: Tensor,
        analogue_idx: Tensor,
        x_n_phi: Tensor,
    ) -> Tensor:
        """Pre-softmax per-analogue scores ``A_l`` (paper §"continuous
        collab"). Exposed separately so the cell-free
        :meth:`forward_continuous` path can fold ``A_l`` into the
        joint softmax over support shots without first running the
        per-analogue softmax (the grid path).

        Returns ``(B, L)`` float scores. Invalid-analogue masking is
        the caller's responsibility — this method returns raw scores.
        """
        u_self = self.traits[player_idx, snapshot_idx]  # (B, TRAIT_DIM)
        u_other = self._gather_analogue_traits(analogue_idx, snapshot_idx)  # (B, L, TRAIT_DIM)

        _b, L = analogue_idx.shape
        u_self_bcast = u_self.unsqueeze(1).expand(-1, L, -1)  # (B, L, TRAIT_DIM)
        diff = u_self_bcast - u_other
        prod = u_self_bcast * u_other
        x_n_bcast = x_n_phi.unsqueeze(1).expand(-1, L, -1)  # (B, L, CONTEXT_DIM)

        phi_in = torch.cat([u_self_bcast, u_other, diff, prod, x_n_bcast], dim=-1)
        phi_out = self.phi(phi_in).squeeze(-1)  # (B, L)

        # b_same: +b_same on the slot where analogue_idx == player_idx.
        is_self = (analogue_idx == player_idx.unsqueeze(-1)).to(phi_out.dtype)
        scores = phi_out + self.b_same * is_self

        # Similarity prior (alpha_prior="similarity" only): +γ · cos(u_p, u_{p'}).
        if self.alpha_prior == "similarity":
            cos_sim = _cosine_similarity_bl(u_self_bcast, u_other)  # (B, L)
            scores = scores + self.gamma_sim * cos_sim

        # Evidence-volume prior: λ_M log(1+M_{p'}) and λ_S log(1+S_{p'})
        # come from the analogue's traits at slot 14, 15 (log1p values).
        log1p_M_other = u_other[..., _SLOT_LOG1P_M]
        log1p_S_other = u_other[..., _SLOT_LOG1P_S]
        scores = scores + self.lambda_M * log1p_M_other + self.lambda_S * log1p_S_other
        out: Tensor = scores
        return out

    def _compute_alpha(
        self,
        player_idx: Tensor,
        snapshot_idx: Tensor,
        analogue_idx: Tensor,
        x_n_phi: Tensor,
        valid_analogue_mask: Tensor,
    ) -> Tensor:
        """Player-level attention α (post-softmax). Calls
        :meth:`_compute_alpha_scores` and applies the per-analogue
        masked softmax + defensive uniform-fallback for cold-start
        rows. Used by the grid (cell-based) forward."""
        scores = self._compute_alpha_scores(player_idx, snapshot_idx, analogue_idx, x_n_phi)
        _b, L = analogue_idx.shape
        # Mask invalid analogues with -inf so softmax assigns them α = 0.
        scores = scores.masked_fill(~valid_analogue_mask, float("-inf"))
        alpha: Tensor = torch.softmax(scores, dim=-1)
        # Defensive: rows with NO valid analogue would produce NaN from
        # softmax(all-inf). Replace with uniform-over-L; valid mask
        # downstream zeros their kernel contribution anyway.
        any_valid = valid_analogue_mask.any(dim=-1, keepdim=True)
        alpha = torch.where(any_valid, alpha, torch.full_like(alpha, 1.0 / L))
        return alpha

    def _compute_sigma(self, player_idx: Tensor, snapshot_idx: Tensor) -> Tensor:
        """Per-target bandwidth σ_p(t_m). Returns ``(B,)``.

        ``σ = σ_min + (σ_max − σ_min) · sigmoid[a_0
                  − softplus(a_M)·log1p_M − softplus(a_S)·log1p_S
                  + a_R·log_density]``

        with ``log1p_M, log1p_S, log_density`` pulled from the target's
        trait slots. softplus on ``a_M, a_S`` enforces the directional
        prior "more evidence → sharper kernel."
        """
        u_self = self.traits[player_idx, snapshot_idx]  # (B, TRAIT_DIM)
        log1p_M = u_self[..., _SLOT_LOG1P_M]
        log1p_S = u_self[..., _SLOT_LOG1P_S]
        log_density = u_self[..., _SLOT_LOG_DENSITY]

        z = (
            self.a_0
            - torch.nn.functional.softplus(self.a_M) * log1p_M
            - torch.nn.functional.softplus(self.a_S) * log1p_S
            + self.a_R * log_density
        )
        sigma: Tensor = self.sigma_min + self.sigma_range * torch.sigmoid(z)
        return sigma

    def _shot_attention_scores(
        self,
        x_n: Tensor,
        shot_idx: Tensor,
        shot_dates: Tensor,
        anchor_dates_b: Tensor,
    ) -> Tensor:
        """Pre-mask shot-attention scores ``g_θ(x, z_j) - λ_age Δt``.

        ``shot_idx``: ``(B, L, R)`` int64 with ``-1`` for padding. Padded
        entries get the same context as id 0 by ``clamp_min(0)``; the
        downstream causal/real mask zeros them out before softmax.

        Returns ``(B, L, R)``.
        """
        b = shot_idx.shape[0]
        delta_days = (anchor_dates_b.view(b, 1, 1) - shot_dates).to(torch.float32)

        if self.shot_attention_form == "bilinear":
            score = self._bilinear_score(x_n, shot_idx)
        else:
            score = self._concat_score(x_n, shot_idx)

        return score - self.lambda_age * delta_days

    def _bilinear_score(self, x_n: Tensor, shot_idx: Tensor) -> Tensor:
        """``f_θ(x_n)^T h_θ(z_j)`` with batch-unique ``h_θ`` compute.

        ``shot_idx``: ``(B, L, R)`` int64 (already ``clamp_min(0)``-safe).
        Returns ``(B, L, R)``.
        """
        b, L, R = shot_idx.shape
        flat = shot_idx.reshape(-1).clamp_min(0)  # (B*L*R,)

        # Dedupe: h_θ runs on the unique shot ids only.
        unique_ids, inverse = torch.unique(flat, return_inverse=True)
        unique_ctx = self.pool_context.index_select(0, unique_ids)  # (U, D)
        unique_h = self.h_z(unique_ctx)  # (U, proj_dim)
        h_flat = unique_h.index_select(0, inverse)  # (B*L*R, proj_dim)
        h = h_flat.view(b, L, R, self._shot_proj_dim)

        f_x = self.f_x(x_n)  # (B, proj_dim)
        score: Tensor = (f_x.view(b, 1, 1, -1) * h).sum(dim=-1)  # (B, L, R)
        return score

    def _concat_score(self, x_n: Tensor, shot_idx: Tensor) -> Tensor:
        """Legacy concat-MLP scorer ``g_θ([x; z])``, ablation only.

        Materializes the full ``(B, L, R, D)`` shot-context tensor and
        runs an MLP on it. Slower but kept as the v1.0 baseline.
        """
        b, L, R = shot_idx.shape
        safe_idx = shot_idx.clamp_min(0)
        shot_ctx = self.pool_context.index_select(0, safe_idx.reshape(-1)).view(
            b, L, R, CONTEXT_DIM
        )
        x_n_bcast = x_n.view(b, 1, 1, -1).expand(b, L, R, CONTEXT_DIM)
        g_in = torch.cat([x_n_bcast, shot_ctx], dim=-1)
        score: Tensor = self.g(g_in).squeeze(-1)  # (B, L, R)
        return score

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        player_idx: Tensor,
        snapshot_idx: Tensor,
        x_n_raw: Tensor,
        x_n: Tensor,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, CollaborativeOutputs]:
        """Compute ``log q_collab(c | x_n, t_n)`` for the batch.

        Parameters
        ----------
        player_idx : LongTensor of shape ``(B,)``
            Per-row vocab index of the target player.
        snapshot_idx : LongTensor of shape ``(B,)``
            Per-row snapshot index (the bundle whose anchor contains
            the shot's date).
        x_n_raw : Tensor of shape ``(B, CONTEXT_DIM)``
            Raw context (the :class:`ContextEncoder` output). Threaded
            through for API parity with the legacy
            ``AdaptiveOffensivePrior``; the collaborative module's
            MLPs are unstructured and consume the learned ``x_n``.
        x_n : Tensor of shape ``(B, CONTEXT_DIM)``
            Learned context :math:`x_n = f_{ctx}(\\tilde x_n)`.
            Consumed by φ_θ (player attention) and f_θ / h_θ (shot
            attention).
        return_components : bool, default False
            When True, returns ``(log_q, CollaborativeOutputs)``.
        """
        del x_n_raw  # accepted for API parity with AdaptiveOffensivePrior
        b = player_idx.shape[0]

        # 1. Gather analogues for the batch.
        analogue_idx = self._gather_analogues(player_idx, snapshot_idx)  # (B, L)
        anchor_dates_b = self.anchor_dates[snapshot_idx]  # (B,)

        # 2. Pull per-(analogue, shot) ids from the global pool.
        shot_idx = self.pool_index[analogue_idx]  # (B, L, R)
        analogue_real_mask = (shot_idx >= 0).to(x_n.dtype)  # (B, L, R)
        safe_shot_idx = shot_idx.clamp_min(0)

        # Gather coords + dates by safe ids; real_mask zeros padded slots.
        flat_safe = safe_shot_idx.reshape(-1)
        shot_coords = self.pool_coords.index_select(0, flat_safe).view(b, self._L, self._R, 2)
        shot_dates = self.pool_dates.index_select(0, flat_safe).view(b, self._L, self._R)

        # Causal mask: shot date < anchor date.
        causal_mask = (shot_dates < anchor_dates_b.view(b, 1, 1)).to(x_n.dtype)
        eff_mask = analogue_real_mask * causal_mask  # (B, L, R)
        valid_analogue_mask = eff_mask.sum(dim=-1) > 0  # (B, L) bool

        # 3. α (player-level attention) and σ (per-target bandwidth).
        alpha = self._compute_alpha(
            player_idx, snapshot_idx, analogue_idx, x_n, valid_analogue_mask
        )  # (B, L)
        sigma = self._compute_sigma(player_idx, snapshot_idx)  # (B,)

        # 4. β (shot-level attention). Score pre-mask via the chosen form.
        score = self._shot_attention_scores(x_n, safe_shot_idx, shot_dates, anchor_dates_b)
        score = score.masked_fill(eff_mask < 0.5, float("-inf"))
        beta = torch.softmax(score, dim=-1)
        has_any = (eff_mask.sum(dim=-1) > 0).unsqueeze(-1)
        beta = torch.where(has_any, beta, torch.zeros_like(beta))

        # 5. Combined per-shot weight α·β, zero on invalid slots, flattened.
        w = alpha.unsqueeze(-1) * beta * eff_mask  # (B, L, R)
        coords_flat = shot_coords.reshape(b, self._L * self._R, 2)
        w_flat = w.reshape(b, self._L * self._R)

        # 6. Separable Gaussian kernel evaluator → (B, n_cells).
        q_collab = separable_gaussian_density(
            coords=coords_flat,
            weights=w_flat,
            sigma=sigma,
            xcenters=self.xcenters,
            ycenters=self.ycenters,
        )

        # 7. Additive smoothing + log-normalize (Decision-1 convention).
        smoothed = q_collab.clamp_min(0.0) + (self.eps / self._n_cells)
        log_q = torch.log(smoothed) - torch.log(smoothed.sum(dim=-1, keepdim=True))

        if not return_components:
            return log_q

        components = CollaborativeOutputs(
            log_q_collab=log_q,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            analogues=analogue_idx,
            has_analogue_history=valid_analogue_mask,
        )
        return log_q, components

    # -----------------------------------------------------------------
    # Cell-free continuous-mixture forward (paper §"continuous collab")
    # -----------------------------------------------------------------

    def forward_continuous(
        self,
        player_idx: Tensor,
        snapshot_idx: Tensor,
        x_n_raw: Tensor,
        x_n: Tensor,
    ) -> CollaborativeContinuousOutputs:
        """Produce the ingredients for the cell-free mixture
        likelihood. Unlike :meth:`forward` (which returns a grid
        ``(B, n_cells)`` log-density via separable kernel), this
        method returns the support shots' coordinates + pre-softmax
        joint scores ``A_l + B_{l,r}`` so the trainer can add
        residual/defense contributions and then ``logsumexp`` over
        the Gaussian kernel at the *exact* observed coordinate
        (see :func:`shotcloud.training.spatial_losses.continuous_mixture_nll`).

        The grid is bypassed entirely. The two-stage α · β softmax
        is replaced by a single joint softmax over ``(l, r)``,
        normalized externally by the consumer.
        """
        del x_n_raw  # accepted for API parity
        b = player_idx.shape[0]
        analogue_idx = self._gather_analogues(player_idx, snapshot_idx)  # (B, L)
        anchor_dates_b = self.anchor_dates[snapshot_idx]  # (B,)

        # Gather support shots + causal mask, identical to the grid forward.
        shot_idx = self.pool_index[analogue_idx]  # (B, L, R) int64, -1 for pad
        analogue_real_mask = (shot_idx >= 0).to(x_n.dtype)
        safe_shot_idx = shot_idx.clamp_min(0)
        flat_safe = safe_shot_idx.reshape(-1)
        shot_coords = self.pool_coords.index_select(0, flat_safe).view(b, self._L, self._R, 2)
        shot_dates = self.pool_dates.index_select(0, flat_safe).view(b, self._L, self._R)
        causal_mask = (shot_dates < anchor_dates_b.view(b, 1, 1)).to(x_n.dtype)
        eff_mask = analogue_real_mask * causal_mask  # (B, L, R)
        valid_analogue_mask = eff_mask.sum(dim=-1) > 0  # (B, L) bool — only needed for diagnostic
        support_mask = eff_mask.bool().reshape(b, self._L * self._R)  # (B, M)

        # Per-analogue + per-shot scores (pre-softmax).
        alpha_scores = self._compute_alpha_scores(
            player_idx, snapshot_idx, analogue_idx, x_n
        )  # (B, L)
        beta_scores = self._shot_attention_scores(
            x_n, safe_shot_idx, shot_dates, anchor_dates_b
        )  # (B, L, R)

        # Joint logits: A_l broadcast over r, plus B_{l,r}. Flatten to (B, M).
        joint_logits = alpha_scores.unsqueeze(-1) + beta_scores  # (B, L, R)
        support_logits = joint_logits.reshape(b, self._L * self._R)

        # Per-target bandwidth (same as grid path).
        sigma = self._compute_sigma(player_idx, snapshot_idx)  # (B,)

        # Flatten support coords for downstream mixture eval.
        support_xy = shot_coords.reshape(b, self._L * self._R, 2)
        del valid_analogue_mask  # available as support_mask.any(-1) downstream

        # Backend-agnostic own/pooled partition for the pooling gate
        # (PR3). For the L×R backend, "own" slots are those whose
        # L-slot's analogue equals the target player, intersected with
        # the causal/real support mask.
        is_self_l = analogue_idx == player_idx.unsqueeze(-1)  # (B, L)
        is_self_m = (
            is_self_l.unsqueeze(-1).expand(-1, -1, self._R).reshape(b, self._L * self._R)
        )  # (B, M)
        own_mask = is_self_m & support_mask
        # Backend-agnostic per-slot shooter identity (PR3.2). Broadcast
        # the per-L analogue id over R shots per analogue.
        support_shooter = (
            analogue_idx.unsqueeze(-1).expand(-1, -1, self._R).reshape(b, self._L * self._R)
        )

        return CollaborativeContinuousOutputs(
            support_xy=support_xy,
            support_logits=support_logits,
            sigma=sigma,
            support_mask=support_mask,
            own_mask=own_mask,
            support_shooter=support_shooter,
            analogue_idx=analogue_idx,
            alpha_scores=alpha_scores,
            beta_scores=beta_scores,
        )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def learned_scalars(self) -> dict[str, float]:
        """Detached float view of the scalar learnable params. For logging."""
        with torch.no_grad():
            return {
                "b_same": float(self.b_same),
                "lambda_M": float(self.lambda_M),
                "lambda_S": float(self.lambda_S),
                "lambda_age": float(self.lambda_age),
                "a_0": float(self.a_0),
                "a_M": float(self.a_M),
                "a_S": float(self.a_S),
                "a_R": float(self.a_R),
                "softplus_a_M": float(torch.nn.functional.softplus(self.a_M)),
                "softplus_a_S": float(torch.nn.functional.softplus(self.a_S)),
                "gamma_sim": float(self.gamma_sim),
            }

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"L={self._L}, R={self._R}, n_pool={self._n_pool}, "
            f"n_cells={self._n_cells} ({self._ny}x{self._nx}), "
            f"sigma=({self.sigma_min}, {self.sigma_init}, {self.sigma_max}), "
            f"shot_attention_form={self.shot_attention_form!r}, "
            f"h_z_init={self.h_z_init!r}, "
            f"alpha_prior={self.alpha_prior!r}, "
            f"n_params={n_params}"
        )


# Silence the M_PLAY_SLOT unused-import warning — the slot isn't used
# directly by CollaborativeKDE, but we re-export the constant so
# downstream diagnostics (Phase 5) have it available without an extra
# import.
_ = M_PLAY_SLOT
