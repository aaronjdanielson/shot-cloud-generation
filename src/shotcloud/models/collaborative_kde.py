"""Collaborative adaptive KDE over a fixed set of analogue players.

:class:`CollaborativeKDE` is the fixed-grid support backend of the AC-KDE
spatial factor (``--support-backend fixed_lr``). For a target player
:math:`p` at snapshot :math:`t_n` it reads the top-:math:`L` analogue
players :math:`\\mathcal N_p(t_n)` from an
:class:`~shotcloud.models.analogue_retrieval.AnalogueRetrievalCache` and up
to :math:`R` shots from each analogue's history, keeping only shots dated
strictly before the snapshot's anchor date. Two forward paths share these
supports:

* :meth:`CollaborativeKDE.forward_continuous` returns the support shots,
  their joint support logits :math:`A_l + B_{l,r}`, the bandwidth, and the
  own/pooled partition consumed by
  :class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`.
* :meth:`CollaborativeKDE.forward` evaluates the two-stage-attention
  density on the court grid,

  .. math::

      \\hat q_p^{\\mathrm{collab}}(c \\mid x_n, t_n)
      = \\sum_{p' \\in \\mathcal N_p^{+}(t_n)} \\alpha_{p,p'}(x_n, t_n)
        \\sum_{j \\in \\mathcal H_{p'}^{<t_n}} \\beta_{p,p',j}(x_n, t_n)
        K_{\\sigma_p(t_n)}(x_c - s_j),

  where :math:`\\mathcal N_p^{+}(t_n)` are the analogues with causal
  history, :math:`\\alpha` is player-level attention, :math:`\\beta` is
  shot-level attention within an analogue's causal history
  :math:`\\mathcal H_{p'}^{<t_n}`, and :math:`K_\\sigma` is an isotropic
  Gaussian whose bandwidth depends on the target player's causal
  evidence volume.

Implementation
--------------
* **Separable kernel.** The isotropic Gaussian factors as
  ``K_x(c_x - s_{j,x}) · K_y(c_y - s_{j,y})``, so the grid density is
  aggregated with a batched matrix product over the ``L * R`` shot axis
  without materializing a ``(B, L, R, n_cells)`` intermediate (see
  :func:`~shotcloud.models._separable_kernel.separable_gaussian_density`).
* **Global support pool.** Per-shot context, coordinates, and dates live
  in one flat :class:`~shotcloud.models._support_pool.GlobalSupportPool`;
  each player's history is a row of an ``(n_players, max_R)`` int64 index.
* **Bilinear shot attention.** ``g_θ(x, z_j) = f_θ(x)^T h_θ(z_j) - λ_age Δt``
  makes ``h_θ(z_j)`` depend only on the shot, so ``h_θ`` runs once per
  unique shot id in the batch. The unfactored MLP ``g_θ([x; z_j])`` is
  available as ``shot_attention_form="concat"``.

With the ``h_θ`` output layer zero-initialized, ``g_θ ≡ 0`` and β is
uniform within each analogue's causal history, so the bilinear and
concat forms define the same density at initialization.
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

# PlayerTraitsTable slots read at forward time (play-derived block, already
# multiplied by the missingness indicator in the table):
# log1p_minutes_M, log1p_fga_S, log_shot_density.
_SLOT_LOG1P_M: int = 14
_SLOT_LOG1P_S: int = 15
_SLOT_LOG_DENSITY: int = 16

ShotAttentionForm = Literal["bilinear", "concat"]
HzInit = Literal["zero", "warm"]
AlphaPrior = Literal["none", "similarity"]

# Standard deviation of the warm h_z output-layer normal init. Per-shot
# scores stay small enough that β is near-uniform at initialization, yet
# nonzero, so the f_x side of the bilinear score receives a gradient from
# the first step.
_H_Z_WARM_INIT_STD: float = 1e-3


def _inverse_sigmoid(p: float) -> float:
    """Return ``logit(p)``; sets ``a_0`` so that ``σ = sigma_init`` at initialization."""
    return float(np.log(p / (1.0 - p)))


def _cosine_similarity_bl(u_self: Tensor, u_other: Tensor, *, eps: float = 1e-12) -> Tensor:
    """``(B, L)`` cosine similarity between paired ``(B, L, D)`` vectors.

    The denominator is clamped, so a pair with an all-zero trait vector
    (a player with neither biographical record nor play history) gets
    the neutral similarity 0 instead of NaN.
    """
    num = (u_self * u_other).sum(dim=-1)
    denom = (u_self.norm(dim=-1) * u_other.norm(dim=-1)).clamp_min(eps)
    out: Tensor = num / denom
    return out


@dataclass(frozen=True)
class CollaborativeOutputs:
    """Per-batch intermediates of one grid :meth:`CollaborativeKDE.forward` pass.

    Returned when ``return_components=True``.

    Attributes
    ----------
    log_q_collab : Tensor of shape (B, n_cells)
        Normalized log-density over the court grid.
    alpha : Tensor of shape (B, L)
        Player-level attention; rows sum to 1.
    beta : Tensor of shape (B, L, R)
        Shot-level attention; sums to 1 over ``R`` for each analogue with
        causal history and is identically 0 otherwise.
    sigma : Tensor of shape (B,)
        Per-target bandwidth in feet.
    analogues : Tensor of shape (B, L), int64
        Analogue vocabulary indices.
    has_analogue_history : Tensor of shape (B, L), bool
        Whether each analogue has causal history, i.e. membership in
        :math:`\\mathcal N_p^{+}(t_n)`.
    """

    log_q_collab: Tensor
    alpha: Tensor
    beta: Tensor
    sigma: Tensor
    analogues: Tensor
    has_analogue_history: Tensor


@dataclass(frozen=True)
class CollaborativeContinuousOutputs:
    """Support-set ingredients for the cell-free continuous mixture.

    Returned by the ``forward_continuous`` method of both support backends
    (:class:`CollaborativeKDE` and
    :class:`~shotcloud.models.retrieval_collaborative_kde.RetrievalCollaborativeKDE`).
    :class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`
    adds its own support-logit terms to ``support_logits`` and evaluates
    the mixture density at the observed shot,

    .. math::

        f_\\Theta(y) = \\sum_m w_m\\, K_{\\sigma}(y - s_m),
        \\quad
        w_m = \\operatorname{softmax}_m(A + B + R + D),

    with :func:`~shotcloud.training.spatial_losses.continuous_mixture_loglik`.

    Attributes
    ----------
    support_xy : Tensor of shape (B, M, 2)
        Support-shot coordinates in court feet. For :class:`CollaborativeKDE`,
        ``M = L * R`` and ``m = (l, r)`` flattens the analogue and shot axes.
    support_logits : Tensor of shape (B, M)
        Pre-softmax support logits :math:`A + B`; further terms are added
        by the caller.
    sigma : Tensor of shape (B,)
        Per-target bandwidth in feet.
    support_mask : Tensor of shape (B, M), bool
        ``True`` where the support slot holds a real, causal shot.
    own_mask : Tensor of shape (B, M), bool
        Subset of ``support_mask`` whose shooter is the target player.
        Together with ``support_mask`` it partitions the valid support
        into the own and pooled subsets used by the pooling gate.
    support_shooter : Tensor of shape (B, M), int64
        Vocabulary index of each support slot's shooter. Unspecified on
        invalid slots; apply ``support_mask`` first.
    analogue_idx : Tensor of shape (B, L), int64
        Analogue vocabulary indices (diagnostic). Shape ``(B, 0)`` for the
        retrieval backend.
    alpha_scores : Tensor of shape (B, L)
        Pre-softmax player-level scores :math:`A_l` (diagnostic). Empty for
        the retrieval backend.
    beta_scores : Tensor of shape (B, L, R)
        Pre-softmax shot-level scores :math:`B_{l,r}` (diagnostic). Empty for
        the retrieval backend.
    """

    support_xy: Tensor
    support_logits: Tensor
    sigma: Tensor
    support_mask: Tensor
    own_mask: Tensor
    support_shooter: Tensor
    analogue_idx: Tensor
    alpha_scores: Tensor
    beta_scores: Tensor


class CollaborativeKDE(nn.Module):
    r"""Collaborative adaptive KDE support backend over a fixed analogue grid.

    For each batch row the support set is the ``L × R`` grid of the
    target's top-``L`` analogues (from ``analogue_cache``) times up to
    ``R`` stored shots per analogue, masked to shots dated strictly before
    the snapshot's anchor date. The support logits are

    .. math::

        A_l = \phi_\theta([u_p, u_{p'}, u_p - u_{p'}, u_p \odot u_{p'}, x_n])
              + b_{\mathrm{same}}\,[p' = p]
              + \gamma \cos(u_p, u_{p'})
              + \lambda_M \log(1 + M_{p'}) + \lambda_S \log(1 + S_{p'}),

        B_{l,r} = g_\theta(x_n, z_{l,r}) - \lambda_{\mathrm{age}}\,\Delta t_{l,r},

    where :math:`u` are causal player traits, :math:`z_{l,r}` is the
    stored context of the support shot, :math:`\Delta t_{l,r}` its age in
    days at the anchor date, and the similarity term is present only
    with ``alpha_prior="similarity"``. The bandwidth is

    .. math::

        \sigma_p = \sigma_{\min} + (\sigma_{\max} - \sigma_{\min})\,
        \mathrm{sigmoid}\bigl(a_0 - \mathrm{softplus}(a_M)\log(1 + M_p)
        - \mathrm{softplus}(a_S)\log(1 + S_p) + a_R \log\rho_p\bigr),

    with :math:`M_p` and :math:`S_p` the target's recency-weighted causal
    minutes and field-goal attempts and :math:`\rho_p = (1 + S_p) / (1 + M_p)`; the
    softplus terms make more evidence give a sharper kernel.

    Parameters
    ----------
    adaptive_kde : AdaptiveKDE
        Fitted per-player history (context, coordinates, dates). Must be
        fit with dates and per-shot coordinates. ``R`` is the longest
        stored per-player history.
    snapshot_store : SnapshotStore
        Provides the anchor dates used by the causal date mask.
    traits_table : PlayerTraitsTable
        Causal per-(player, snapshot) trait vectors of dimension
        :data:`~shotcloud.data.player_traits.TRAIT_DIM`.
    analogue_cache : AnalogueRetrievalCache
        Precomputed top-``L`` analogues per (player, snapshot).
    vocab : PlayerVocab
        Player-id ↔ index mapping; the three tables above must be keyed
        to ``vocab.ids`` in the same order.
    grid : CourtGrid
        Court grid whose cell centers are used by the grid forward. The
        returned grid log-density uses the image-layout flat index
        ``c = iy * nx + ix``.
    sigma_min, sigma_max : float, default 0.75, 4.0
        Bandwidth bounds in feet. ``sigma_min == sigma_max`` fixes the
        bandwidth at that value.
    sigma_init : float, default 1.5
        Bandwidth at initialization, for every player.
    phi_hidden_dim : int, default 64
        Hidden width of the player-attention MLP :math:`\phi_\theta`.
    shot_attention_form : {"bilinear", "concat"}, default "bilinear"
        ``"bilinear"`` scores shots as
        :math:`g_\theta(x, z) = f_\theta(x)^\top h_\theta(z)`, computing
        :math:`h_\theta` once per unique shot in the batch. ``"concat"``
        uses an MLP on ``[x; z]`` (an ablation of the factorization).
    shot_hidden_dim : int, default 64
        Hidden width of the shot-attention MLPs.
    shot_proj_dim : int, default 32
        Output dimension of :math:`f_\theta` and :math:`h_\theta`. Ignored
        for ``"concat"``.
    h_z_init : {"zero", "warm"}, default "zero"
        Initialization of the :math:`h_\theta` output layer. ``"zero"``
        makes :math:`g_\theta \equiv 0`, so β is uniform at initialization
        and the bilinear and concat forms coincide; :math:`f_\theta` then
        receives no gradient until :math:`h_\theta` moves off zero.
        ``"warm"`` draws the weights from a normal with standard deviation
        ``1e-3``, keeping β near-uniform while giving
        :math:`f_\theta` a nonzero gradient from the first step. Ignored
        for ``"concat"``.
    alpha_prior : {"none", "similarity"}, default "none"
        With ``"similarity"``, adds the learnable term
        :math:`\gamma \cos(u_p, u_{p'})` to the player-level scores.
    alpha_prior_scale_init : float, default 2.0
        Initial value of :math:`\gamma` when ``alpha_prior="similarity"``.
    same_player_bias_init : float, default 0.0
        Initial value of :math:`b_{\mathrm{same}}`, the bonus on the
        analogue slot occupied by the target player. Positive values keep
        the target's own history from being outweighed by analogues.
    eps : float, default 1e-9
        Total additive smoothing mass, spread uniformly over the grid
        cells before log-normalization, so the grid density is strictly
        positive.

    Raises
    ------
    ValueError
        If ``adaptive_kde`` is unfitted or lacks dates or coordinates, the
        bandwidth bounds are inconsistent, a categorical option is
        unknown, or the tables disagree in shape.
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
        # player; positive values (e.g. +1.0) keep the target's own
        # causal history from being washed out by retrieval analogues.
        self.b_same = nn.Parameter(torch.tensor(float(same_player_bias_init), dtype=torch.float32))
        self.lambda_M = nn.Parameter(torch.zeros(()))
        self.lambda_S = nn.Parameter(torch.zeros(()))
        # Similarity-prior coefficient γ on cos(u_self, u_other). When
        # ``alpha_prior="similarity"`` it's a learnable scalar
        # initialized to ``alpha_prior_scale_init`` (typically 2.0).
        # When ``alpha_prior="none"`` it is a zero buffer, so
        # ``learned_scalars`` reports it in both configurations.
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
        #   - "zero":  W=0, b=0 → g_θ ≡ 0 → β uniform at init (same
        #              density as the zero-initialized concat form). f_θ
        #              receives no gradient until h_θ moves off zero.
        #   - "warm":  W ~ N(0, std=1e-3), b=0 → β stays near-uniform
        #              (KL to uniform < 1e-4 at init) but g_θ is small
        #              and nonzero, so f_θ's gradient is nonzero from
        #              the first step.
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
            # h_z_init only affects the bilinear form; it is recorded
            # here so extra_repr reports the requested value.
            self.h_z_init = h_z_init

        # Shot-level recency-decay rate.
        self.lambda_age = nn.Parameter(torch.zeros(()))

        # Bandwidth scalars (a_0, a_M, a_S, a_R). At initialization
        # σ_p = sigma_init for every player regardless of evidence.
        # The formula is
        #   σ = σ_min + (σ_max − σ_min) · sigmoid[
        #       a_0 − softplus(a_M)·log1p_M − softplus(a_S)·log1p_S + a_R·log_density
        #   ]
        # so we need softplus(a_M) ≈ softplus(a_S) ≈ 0 at init (NOT
        # a_M = a_S = 0, because softplus(0) = ln(2) ≈ 0.693 already
        # pulls σ toward σ_min for any player with non-trivial evidence).
        # Init a_M = a_S = -10 → softplus ≈ 4.5e-5 → the evidence terms
        # contribute negligibly at init, leaving σ = σ_min +
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
    def L(self) -> int:  # noqa: N802 — matches the notation `L = |N_p(t_n)|`
        """Number of analogue slots per row."""
        return self._L

    @property
    def n_cells(self) -> int:
        """Number of cells in the court grid."""
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
        """Pre-softmax per-analogue scores ``A_l``, shape ``(B, L)``.

        Separate from :meth:`_compute_alpha` so :meth:`forward_continuous`
        can fold ``A_l`` into the joint softmax over support shots.
        Returns raw scores; masking invalid analogues is the caller's job.
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
        """Player-level attention α for the grid forward, shape ``(B, L)``.

        Masked softmax of :meth:`_compute_alpha_scores` over valid
        analogues; rows with no valid analogue fall back to uniform.
        """
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
        """Per-target bandwidth σ_p. Returns ``(B,)``.

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
        """Concat-MLP scorer ``g_θ([x; z])`` (ablation form).

        Materializes the full ``(B, L, R, D)`` shot-context tensor, so it
        is slower than the bilinear form.
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
        """Compute the grid log-density ``log q_collab(c | x_n, t_n)``.

        Parameters
        ----------
        player_idx : LongTensor of shape ``(B,)``
            Vocabulary index of the target player.
        snapshot_idx : LongTensor of shape ``(B,)``
            Causal snapshot index of the shot's game.
        x_n_raw : Tensor of shape ``(B, CONTEXT_DIM)``
            Raw context (the
            :class:`~shotcloud.data.context.ContextEncoder` output).
            Accepted for interface compatibility with the other spatial
            backends and unused: all attention MLPs consume the learned
            ``x_n``.
        x_n : Tensor of shape ``(B, CONTEXT_DIM)``
            Learned context :math:`x_n = f_{ctx}(\\tilde x_n)`, consumed by
            φ_θ (player attention) and f_θ / h_θ (shot attention).
        return_components : bool, default False
            When True, also return the :class:`CollaborativeOutputs`.

        Returns
        -------
        log_q : Tensor of shape ``(B, n_cells)``
            Normalized log-density over the court grid.
        components : CollaborativeOutputs
            Returned only when ``return_components=True``.
        """
        del x_n_raw  # accepted for interface compatibility; unused
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

        # 7. Additive ε-smoothing keeps every cell strictly positive, then
        #    log-normalize.
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
    # Cell-free continuous-mixture forward
    # -----------------------------------------------------------------

    def forward_continuous(
        self,
        player_idx: Tensor,
        snapshot_idx: Tensor,
        x_n_raw: Tensor,
        x_n: Tensor,
    ) -> CollaborativeContinuousOutputs:
        """Return the support set and logits for the cell-free mixture.

        Unlike :meth:`forward`, no grid is involved: the method returns
        the support-shot coordinates and the joint pre-softmax logits
        ``A_l + B_{l,r}`` over the flattened ``M = L * R`` support axis.
        The caller adds further support-logit terms, normalizes with a
        single softmax over ``(l, r)`` in place of the two-stage α · β
        softmax, and evaluates the kernel mixture at the observed
        coordinate.

        Parameters
        ----------
        player_idx, snapshot_idx, x_n_raw, x_n
            As in :meth:`forward`.

        Returns
        -------
        CollaborativeContinuousOutputs
        """
        del x_n_raw  # accepted for interface compatibility; unused
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

        # Own/pooled partition for the pooling gate. For the L×R
        # backend, "own" slots are those whose
        # L-slot's analogue equals the target player, intersected with
        # the causal/real support mask.
        is_self_l = analogue_idx == player_idx.unsqueeze(-1)  # (B, L)
        is_self_m = (
            is_self_l.unsqueeze(-1).expand(-1, -1, self._R).reshape(b, self._L * self._R)
        )  # (B, M)
        own_mask = is_self_m & support_mask
        # Per-slot shooter identity: broadcast the per-L analogue id
        # over the R shots of each analogue.
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


# M_PLAY_SLOT is not used by CollaborativeKDE directly; it is imported so
# diagnostics can read it from this module, and referenced here to keep
# the import from being flagged as unused.
_ = M_PLAY_SLOT
