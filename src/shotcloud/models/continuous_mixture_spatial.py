"""Cell-free continuous-mixture spatial decoder: the AC-KDE spatial factor.

:class:`ContinuousMixtureSpatial` evaluates the spatial density of each
shot at its exact observed coordinate; no court-cell discretization
enters the likelihood. A support backend
(:class:`~shotcloud.models.retrieval_collaborative_kde.RetrievalCollaborativeKDE`
or :class:`~shotcloud.models.collaborative_kde.CollaborativeKDE`) supplies
the causal support shots :math:`s_m` together with their
shooter-similarity and shot-attention scores, and the density is the
Gaussian kernel mixture

.. math::

    f_\\Theta(y \\mid x_n, h_n, d_n, t_n)
    = \\sum_{m \\in \\mathcal M_p(t_n)}
        w_m(x_n, h_n, d_n, t_n)
        \\;K_{\\sigma_m}(y - s_m),

with mixture weights

.. math::

    w_m = \\operatorname{softmax}_m\\!\\left[
        A_{p, p'}(x_n)
      + B_{p, p', j}(x_n)
      + R_\\theta(s_m \\mid x_n, h_n, \\tau)
      + D_\\delta(s_m \\mid d_n, x_n)
    \\right].

The backend supplies :math:`A + B` and the bandwidth; this module adds
the low-rank residual tilt :math:`R_\\theta(s_m) = u_\\theta^\\top \\psi(s_m)`
(a context encoder paired with a Fourier-feature location embedding) and
the opponent reweighting :math:`D`. With a
:class:`~shotcloud.models.pooling_gate.PoolingGate`, the single softmax
is replaced by separate softmaxes over the own and pooled support
subsets, mixed by the gate weight :math:`\\lambda`:

.. math::

    f_\\Theta(y) = \\lambda\\, f_{\\mathrm{own}}(y)
                 + (1 - \\lambda)\\, f_{\\mathrm{pooled}}(y).

Every support-logit term is added before the subset softmaxes, so it
reshapes mass within each subset and leaves :math:`\\lambda` unchanged.

Causality contract: every support logit, gate input, residual input,
and kernel parameter is computed from information available strictly
before the shot's game (snapshot-indexed features and causal support)
or from earlier shots in the same game. The observed coordinate
:math:`y` enters only through the evaluation of the normalized density
itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from shotcloud.features.defense_features import DefenseFeatures
from shotcloud.features.matchup_features import MatchupFeatures
from shotcloud.models.anisotropic_kernel import (
    FullCovarianceZoneKernel,
    RadialTangentZoneKernel,
)
from shotcloud.models.collaborative_kde import (
    _SLOT_LOG1P_M,
    CollaborativeContinuousOutputs,
    CollaborativeKDE,
)
from shotcloud.models.context_residual import ContextResidualEncoder
from shotcloud.models.continuous_adaptive_defensive import (
    ContinuousAdaptiveDefensiveField,
    gather_defense_inputs,
)
from shotcloud.models.defensive_retrieval_cache import DefensiveRetrievalCache
from shotcloud.models.location_embedding import LocationEmbedding
from shotcloud.models.pooling_gate import PoolingGate
from shotcloud.models.retrieval_collaborative_kde import RetrievalCollaborativeKDE
from shotcloud.models.zone_defense_reweighting import (
    MatchupReweightingDefense,
    ZoneReweightingDefense,
)
from shotcloud.models.zone_source_bandwidth import ZoneSourceBandwidth

#: Anisotropic kernel modules accepted by the wrapper. Both expose
#: ``forward(*, support_xy, shot_xy) -> (B, M)`` per-support log-kernel
#: values, which is the only contract the wrapper relies on.
AnisotropicKernel = RadialTangentZoneKernel | FullCovarianceZoneKernel

#: Support backends accepted by the wrapper. Both implement
#: ``forward_continuous`` returning
#: :class:`~shotcloud.models.collaborative_kde.CollaborativeContinuousOutputs`.
OffensivePriorBackend = CollaborativeKDE | RetrievalCollaborativeKDE

#: Per-row log-likelihood assigned when every support shot of a row is
#: masked out (no causal support). Equals ``log(1 / 2600)`` ≈ -7.86, the
#: log of the uniform density over the 50 ft × 52 ft court, so such rows
#: contribute a finite value to the batch loss.
_COLD_START_LOG_LIK_FLOOR: float = -7.86


class CausalZoneBias(nn.Module):
    r"""Zone-pair bias on support attention, computed from causal inputs only.

    A small head predicts a distribution over query zones from the
    learned context vector :math:`x_n`,

    .. math::
        \pi_q(x_n) = \mathrm{softmax}(g_\theta(x_n)) \in \Delta^{N_{\mathrm{ZONES}} - 1},

    and a learnable :math:`(N_{\mathrm{ZONES}}, N_{\mathrm{ZONES}})`
    affinity matrix :math:`B` is contracted against it to give a
    per-support-shot bias

    .. math::
        b_m = \sum_{a=1}^{N_{\mathrm{ZONES}}} \pi_q(a \mid x_n)\, B[a, z(s_m)],

    which is added to the support logits before the own/pooled subset
    softmaxes. This is an optional support-logit term, evaluated as an
    ablation of the AC-KDE spatial factor.

    Parameters
    ----------
    context_dim : int, default 27
        Dimension of the learned context vector ``x_n``
        (:data:`~shotcloud.data.context.CONTEXT_DIM`).
    hidden_dim : int, default 32
        Hidden width of the query-zone head.

    Notes
    -----
    The forward pass consumes only the causal context ``x_n`` and the
    support coordinates; the observed shot location is never an input,
    so the bias cannot leak :math:`y_n` into the support weights.

    ``B`` is zero-initialized, so the bias vanishes at initialization
    and the decoder starts from the unbiased model. At exactly zero
    ``B`` the query head receives no gradient (its contribution is
    multiplied by ``B``); ``B`` itself receives a nonzero gradient, and
    the head begins to train once ``B`` moves off zero.
    """

    def __init__(self, context_dim: int = 27, hidden_dim: int = 32) -> None:
        super().__init__()
        from shotcloud.data.zones import N_ZONES

        self.q_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, N_ZONES),
        )
        #: ``(N_ZONES, N_ZONES)`` zone-affinity matrix; zero-initialized
        #: so the bias vanishes at initialization.
        self.B = nn.Parameter(torch.zeros(N_ZONES, N_ZONES))

    def forward(self, x_n: Tensor, support_xy: Tensor) -> Tensor:
        """Compute the per-support-shot zone bias.

        Parameters
        ----------
        x_n : Tensor of shape ``(B, context_dim)``
            Learned causal context vector. Must not depend on the
            observed shot location.
        support_xy : Tensor of shape ``(B, M, 2)``
            Support-shot coordinates in court feet.

        Returns
        -------
        edge_bias : Tensor of shape ``(B, M)``
            Additive support-logit bias. Support shots outside every
            zone (``zone == -1``) receive zero.
        """
        from shotcloud.data.zones import zone_from_xy_torch

        pi_q = self.q_head(x_n).softmax(dim=-1)  # (B, N_ZONES)
        piB = pi_q @ self.B  # (B, N_ZONES) — bias vector per query
        z_s = zone_from_xy_torch(support_xy[..., 0], support_xy[..., 1])  # (B, M)
        valid = z_s >= 0
        z_s_idx = z_s.clamp_min(0).long()
        edge_bias = piB.gather(1, z_s_idx)  # (B, M)
        result: Tensor = edge_bias * valid.to(edge_bias.dtype)
        return result


@dataclass(frozen=True)
class ContinuousMixtureOutputs:
    """Per-batch intermediates of one cell-free spatial forward.

    Mirrors :class:`shotcloud.models.gibbs_decoder.GibbsDecoderOutputs`
    in spirit: rich-enough for downstream diagnostics (α/β entropy,
    support entropy, expected distance) without forcing the trainer
    to redo the forward pass.
    """

    log_lik: Tensor  # (B,) per-row log-likelihood at the observed shot
    log_weights: Tensor  # (B, n_components) log mixture weights (normalized;
    #                      -inf on invalid). ``n_components = M`` for the
    #                      continuous-mixture variant (one component per
    #                      support shot); ``n_components = K`` for the
    #                      mode-mixture variant (one per court mode).
    support_xy: Tensor  # (B, n_components, 2) — component coords (support
    #                     shots or mode anchors).
    sigma: Tensor  # (B,) — per-row bandwidth (mean across modes for
    #              mode-mixture; per-row σ_p(t_m) for continuous-mixture).
    support_mask: Tensor  # (B, n_components) bool — which components are
    #                       valid this row. For mode-mixture all K modes
    #                       are valid for every row (cold-start is per-row,
    #                       not per-mode), so this is all-True there.
    residual_logits: Tensor  # (B, M) R_θ contribution to support attention
    #                          (zero when residual off; M = L*R always).
    collab: CollaborativeContinuousOutputs  # raw support A + B and diagnostics.
    cold_start: Tensor  # (B,) bool — rows whose causal support set was
    #                     entirely empty after masking.
    support_log_weights: Tensor  # (B, M) log of the support attention ω.
    #                              Equal to ``log_weights`` for the
    #                              continuous-mixture variant; distinct from
    #                              the K-shaped mode weights for the
    #                              mode-mixture variant. Always M-shaped, so
    #                              α/β diagnostics that marginalize over
    #                              the (L, R) factorization work in both
    #                              variants uniformly.
    tail_responsibility: Tensor | None = None
    #: (B,) — per-row γ_tail = λ_tail·f_tail / [(1-λ_tail)·f_mode +
    #: λ_tail·f_tail], the posterior responsibility of the support-KDE
    #: tail component for the observed shot. Only populated by the
    #: mode-mixture wrapper when ``tail_weight > 0``; ``None`` otherwise.
    #: Used as a per-epoch diagnostic — large ``mean(γ_tail)`` means
    #: the tail is doing the explanatory work and the K-mode mixture
    #: is failing to cover the support.
    gate_lambda: Tensor | None = None
    #: (B,) — per-row own-history mixing weight λ from the
    #: history-dependent pooling gate: density = λ·f_own +
    #: (1-λ)·f_pooled. Only populated when the continuous-mixture
    #: wrapper is run with a pooling gate; ``None`` otherwise.
    defense_logits: Tensor | None = None
    #: (B, M) — additive support-logit contribution from the
    #: defensive feasibility field :math:`D_\Delta(s_m)`. Populated
    #: only when a defensive field is wired into the wrapper (PR-D2a).
    #: ``None`` otherwise. Zero on cold-start defensive rows.
    defense_cold_start: Tensor | None = None
    #: (B,) bool — per-row flag for "defensive cache had no causal
    #: allowed shots against this opponent at this snapshot". The
    #: defensive field's contribution for these rows is identically
    #: zero (the field's own cold-start safety net). Populated only
    #: when a defensive field is wired.
    sigma_per_shot: Tensor | None = None
    #: (B, M) — per-support-shot bandwidth from the Tier-1a
    #: ``ZoneSourceBandwidth`` field. ``None`` when the bandwidth field
    #: isn't wired (the wrapper falls back to a per-row σ on
    #: :attr:`sigma`). When populated, :attr:`sigma` carries the
    #: per-row mean as a diagnostic; the per-shot tensor is what the
    #: loglik actually consumed.
    matchup_logits: Tensor | None = None
    #: (B, M) — additive support-logit contribution from the Tier-2a
    #: D-matchup field :math:`\\beta_{\\mathrm{match}}\\widehat\\Delta_{p,d,z(s_m)}(t)`.
    #: Populated only when a matchup field is wired; cold-start rows
    #: (no causal peer-vs-opponent evidence) produce exactly zero
    #: because the underlying Δ̂ is exactly zero.
    matchup_n_eff: Tensor | None = None
    #: (B,) — per-row effective peer-vs-opponent sample size used to
    #: shrink Δ̂. Carried for ESS-bucket falsification ("does D-matchup
    #: only help where evidence is strong?"). Populated only when the
    #: matchup field is wired.
    mode_log_pi: Tensor | None = None
    #: (B, K) — per-row mode router log-probabilities, renormalized
    #: over modes available for that row (modes with no causal support
    #: are masked to ``-inf``). Populated only by the mode-routed
    #: decoder; ``None`` for the single-softmax CMS path. Used both
    #: for the within-batch log-density evaluation and for the
    #: diagnostics ``H(π)``, ``mean max_k π_k``, ``mode usage``.
    mode_available: Tensor | None = None
    #: (B, K) bool — per-row mask indicating which modes have at
    #: least one causal support shot. Carried alongside
    #: :attr:`mode_log_pi` so downstream diagnostics can compute the
    #: empty-mode rate and the "renormalization mass lost" (the
    #: pre-renormalization mass that landed on unavailable modes).
    mode_log_pi_raw: Tensor | None = None
    #: (B, K) — per-row mode router log-probabilities BEFORE the
    #: availability renormalization. Storing both lets the trainer
    #: report ``mass lost`` = ``1 - Σ_{k available} exp(mode_log_pi_raw[k])``,
    #: i.e. how much router probability landed on empty modes.


class ContinuousMixtureSpatial(nn.Module):
    """Cell-free continuous-mixture spatial decoder.

    Parameters
    ----------
    offensive_prior : CollaborativeKDE
        Source of support shots, A_l (player) + B_{l,r} (shot)
        scores, and σ.
    residual_encoder : ContextResidualEncoder or None
        When provided, must be paired with ``location_embedding``.
        Produces ``u = u_θ(x_n, h_n) ∈ R^{rank}`` per shot.
    location_embedding : LocationEmbedding or None
        Coordinate embedding ``ψ(s) ∈ R^{rank}`` matching the
        residual encoder's ``rank``. The residual contribution is
        ``R(s_m) = u^T ψ(s_m)``.
    defensive_field : ContinuousAdaptiveDefensiveField or None
        When provided, must be paired with both ``defensive_cache`` and
        ``defensive_features``. Produces an additive support-logit
        contribution :math:`D_\\Delta(s_m)` (PR-D1a) which the wrapper
        adds to the support logits *before* the own/pooled subset
        softmaxes — defense reshapes mass within each subset while
        leaving the pooling gate :math:`\\lambda` untouched. Wiring all
        three to ``None`` preserves the no-defense path bit-exactly.
    defensive_cache : DefensiveRetrievalCache or None
        Per-(opponent, snapshot) allowed-shot retrieval cache (PR-D0).
        Consumed via :func:`shotcloud.models.continuous_adaptive_defensive.gather_defense_inputs`
        inside the wrapper.
    defensive_features : DefenseFeatures or None
        Per-(opponent, snapshot) defensive feature tensor (PR-D0.5).
    cold_start_log_lik_floor : float
        Per-row log-likelihood assigned to batch rows whose
        ``support_mask`` is all-``False`` (no valid causal support).
        Defaults to a uniform-court-density floor so the batch loss
        stays finite.
    """

    def __init__(
        self,
        offensive_prior: OffensivePriorBackend,
        residual_encoder: ContextResidualEncoder | None = None,
        location_embedding: LocationEmbedding | None = None,
        defensive_field: ContinuousAdaptiveDefensiveField | ZoneReweightingDefense | None = None,
        defensive_cache: DefensiveRetrievalCache | None = None,
        defensive_features: DefenseFeatures | None = None,
        matchup_field: MatchupReweightingDefense | None = None,
        matchup_features: MatchupFeatures | None = None,
        pooling_gate: PoolingGate | None = None,
        bandwidth_field: ZoneSourceBandwidth | None = None,
        anisotropic_kernel: AnisotropicKernel | None = None,
        count_head: nn.Module | None = None,
        khat_log1p_mean: float | None = None,
        khat_log1p_std: float | None = None,
        within_game_gru: nn.Module | None = None,
        cold_start_log_lik_floor: float = _COLD_START_LOG_LIK_FLOOR,
        stratified_epsilon: float = 1.0,
        causal_zone_bias: CausalZoneBias | None = None,
        court_bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        super().__init__()
        if (residual_encoder is None) != (location_embedding is None):
            raise ValueError(
                "residual_encoder and location_embedding must both be provided or both None"
            )
        if (
            residual_encoder is not None
            and location_embedding is not None
            and residual_encoder.rank != location_embedding.rank
        ):
            raise ValueError(
                f"residual_encoder.rank ({residual_encoder.rank}) must equal "
                f"location_embedding.rank ({location_embedding.rank})"
            )
        # Defense wiring rules (kind-aware after PR-D-lite-0):
        # * ContinuousAdaptiveDefensiveField (D-field, KDE) requires the
        #   full triple ``field + cache + features``.
        # * ZoneReweightingDefense (D-lite) requires ``field + features``
        #   only; the cache is unused and must be None.
        # * All three None → no defense.
        is_zone_lite = isinstance(defensive_field, ZoneReweightingDefense)
        if defensive_field is None:
            if defensive_cache is not None or defensive_features is not None:
                raise ValueError(
                    "defensive_cache and defensive_features must be None when "
                    "defensive_field is None"
                )
        elif is_zone_lite:
            if defensive_cache is not None:
                raise ValueError(
                    "ZoneReweightingDefense does not consume a defensive cache; "
                    "pass defensive_cache=None"
                )
            if defensive_features is None:
                raise ValueError(
                    "ZoneReweightingDefense requires defensive_features (it consumes "
                    "the centered-zone block); got None"
                )
        else:
            if defensive_cache is None or defensive_features is None:
                raise ValueError(
                    "ContinuousAdaptiveDefensiveField requires both defensive_cache "
                    "and defensive_features; got "
                    f"cache={defensive_cache is not None}, "
                    f"features={defensive_features is not None}"
                )
        # D-matchup wiring rules: ``matchup_field`` and ``matchup_features``
        # must both be provided or both None. The matchup channel is
        # composable with the cell-free defense (so the A/B/C
        # ablations in docs/log.md can all run from the same
        # constructor) — when both ``defensive_field`` and
        # ``matchup_field`` are wired, their logit contributions are
        # summed before the per-subset softmax.
        if (matchup_field is None) != (matchup_features is None):
            raise ValueError(
                "matchup_field and matchup_features must both be provided or both None; "
                f"got field={matchup_field is not None}, "
                f"features={matchup_features is not None}"
            )
        self.offensive_prior = offensive_prior
        self.residual_encoder = residual_encoder
        self.location_embedding = location_embedding
        self.pooling_gate = pooling_gate
        self.defensive_field = defensive_field
        # Cache and features are plain attributes (not registered as
        # ``nn.Module``s or buffers): they hold large tensors but no
        # learnable parameters, and ``gather_defense_inputs`` handles
        # device transfer at forward time.
        self._defensive_cache = defensive_cache
        self._defensive_features = defensive_features
        self.matchup_field = matchup_field
        self._matchup_features = matchup_features
        # Kernel-shape wiring rules (Tier-1a / Tier-2):
        # * ``bandwidth_field`` (Tier-1a): per-(source, zone) scalar σ
        #   on the existing isotropic Gaussian kernel.
        # * ``anisotropic_kernel`` (Tier-2 Option 1 or Option 3): replaces
        #   the isotropic kernel entirely with a per-zone anisotropic
        #   covariance.
        # The two cannot be wired together — both modify the same
        # kernel-shape axis but in different ways, so allowing both
        # would silently make ``bandwidth_field`` a no-op (the
        # anisotropic kernel computes its own log-kernel from its own
        # per-zone σ and bypasses the bandwidth field).
        if bandwidth_field is not None and anisotropic_kernel is not None:
            raise ValueError(
                "bandwidth_field and anisotropic_kernel are mutually exclusive — "
                "both modify the kernel-shape axis. Wire one or the other."
            )
        self.bandwidth_field = bandwidth_field
        self.anisotropic_kernel = anisotropic_kernel
        # Count-head wiring (count-location coupling, B2). When
        # provided, the wrapper appends a detached predicted-count
        # column ``K̂ = NegBinCountHead(x_n).μ`` to the causal usage
        # vector before passing it to the residual encoder. The count
        # head is owned by the trainer at top level (so it receives
        # L_count gradient signal); ``self.count_head`` is the
        # same nn.Module instance, registered here as a child to
        # ensure ``cms.to(device)`` moves it correctly. The detach
        # in the forward prevents L_spatial from bending the count
        # head into a hidden variable.
        # Residual usage_dim must match the count_head wiring under
        # one of three valid configurations:
        #   * usage_dim == USAGE_DIM, count_head is None   — B1 (usage only).
        #   * usage_dim == USAGE_KHAT_DIM, count_head wired — B2 mainline (usage + K̂).
        #   * usage_dim == 1, count_head wired              — K̂-only diagnostic.
        # Any other combination is a wiring bug.
        if residual_encoder is not None:
            from shotcloud.features.usage_features import USAGE_DIM, USAGE_KHAT_DIM

            usage_dim = residual_encoder.usage_dim
            valid_with_count = (usage_dim == USAGE_KHAT_DIM) or (usage_dim == 1)
            valid_without_count = (usage_dim == 0) or (usage_dim == USAGE_DIM)
            if count_head is not None and not valid_with_count:
                raise ValueError(
                    f"count_head wired but residual_encoder.usage_dim={usage_dim}; "
                    f"expected {USAGE_KHAT_DIM} (= USAGE_DIM={USAGE_DIM} + 1) for the "
                    "B2 mainline (usage + K̂) or 1 for the K̂-only diagnostic."
                )
            if count_head is None and not valid_without_count:
                raise ValueError(
                    f"residual_encoder.usage_dim={usage_dim} expects a wired "
                    "count_head: 1 = K̂-only diagnostic, "
                    f"{USAGE_KHAT_DIM} = B2 mainline; "
                    f"the K̂ column comes from NegBinCountHead.forward."
                )
        elif count_head is not None:
            raise ValueError(
                "count_head wired but residual_encoder is None — the count residual "
                "channel requires the residual-tilt encoder to be active too."
            )
        self.count_head = count_head
        # K̂-standardization stats (calibrated-B2 path). When both are
        # provided, the K̂ → residual cat replaces raw ``K̂`` with
        # ``(log1p(K̂) − μ) / σ`` (still detached from the count head).
        # This keeps the residual numerically stable when the count
        # head moves onto the K-scale via ``init_mean``; without it,
        # the residual would suddenly see K̂ swing from O(1) at the
        # uncalibrated initialization to O(10) once calibrated, which
        # would invalidate the residual's pre-trained weights for
        # any warm-start workflow. ``None`` → raw K̂ (backward-compat).
        if (khat_log1p_mean is None) != (khat_log1p_std is None):
            raise ValueError(
                "khat_log1p_mean and khat_log1p_std must both be provided or both None; "
                f"got mean={khat_log1p_mean}, std={khat_log1p_std}"
            )
        if khat_log1p_std is not None and khat_log1p_std <= 0.0:
            raise ValueError(f"khat_log1p_std must be positive, got {khat_log1p_std}")
        if khat_log1p_mean is not None and count_head is None:
            raise ValueError("khat_log1p_mean/std are only meaningful when count_head is wired")
        if khat_log1p_mean is None:
            self.register_buffer("khat_log1p_mean", None)
            self.register_buffer("khat_log1p_std", None)
        else:
            assert khat_log1p_std is not None  # narrowed by the pair check above
            self.register_buffer(
                "khat_log1p_mean", torch.tensor(float(khat_log1p_mean), dtype=torch.float32)
            )
            self.register_buffer(
                "khat_log1p_std", torch.tensor(float(khat_log1p_std), dtype=torch.float32)
            )
        # G1 within-game shot GRU (paper §10, locked 2026-06-05). The
        # GRU consumes the player's prior in-game shot sequence and
        # emits a per-row vector that is added to the residual
        # encoder's output. Wired only when ``residual_encoder`` is
        # active. The module's output projection is zero-initialized,
        # so the G1 augmentation contributes 0 at step 0 and the
        # invariant ``G1 ≡ B2 at init`` holds bit-exactly.
        if within_game_gru is not None and residual_encoder is None:
            raise ValueError(
                "within_game_gru wired but residual_encoder is None — the within-game "
                "GRU contribution can only enter through the residual tilt."
            )
        self.within_game_gru = within_game_gru
        self.cold_start_log_lik_floor = float(cold_start_log_lik_floor)
        # Stratified-court kernel (Phase 1 C1, 2026-06-09). At
        # ``stratified_epsilon=1.0`` the spatial decoder is bit-
        # identical to the unstratified mainline. At any value in
        # (0, 1), the per-(query, support) log-kernel gets an
        # additive penalty ``log(epsilon)`` when the observed shot's
        # zone differs from the support shot's zone. This stops
        # Gaussian mass from leaking freely across the 3-point arc,
        # the paint boundary, the corners, etc. — the half-court is
        # a stratified space, not Euclidean.
        if not 0.0 < stratified_epsilon <= 1.0:
            raise ValueError(f"stratified_epsilon must be in (0, 1]; got {stratified_epsilon}")
        self.stratified_epsilon = float(stratified_epsilon)
        # Phase 2 α1 (causal redesign, 2026-06-09): Graphormer-style
        # zone-pair edge bias on support attention, derived ONLY from
        # ``x_n`` and ``support_xy`` (both causal). See
        # :class:`CausalZoneBias` for the architecture and the locked
        # leakage rule. The previous wrapper-direct ``zone_pair_bias``
        # API was removed: it conditioned on ``z(y_n)`` and was a
        # quiet predictive-density violation that drove a fake −1.6
        # nat/shot val NLL improvement (see
        # ``outputs/joint_b2_outcome_zone_pair_bias_v1/INVALID.md``).
        self.causal_zone_bias = causal_zone_bias
        # Half-court boundary correction (AOAS audit item A1, 2026-06-13).
        # When set, every call into
        # :func:`shotcloud.training.spatial_losses.continuous_mixture_loglik`
        # subtracts the analytic per-support-shot
        # :math:`\\log Z_m(\\mathcal C)` so the predictive density is a
        # proper density on the rectangular court ``court_bounds``,
        # not on :math:`\\mathbb R^2`. ``None`` (the default) preserves
        # the pre-A1 unconstrained-:math:`\\mathbb R^2` formulation and
        # every existing trained checkpoint's bit-identical loss.
        if court_bounds is not None:
            if len(court_bounds) != 4:
                raise ValueError(
                    f"court_bounds must be (x_min, x_max, y_min, y_max); got "
                    f"length {len(court_bounds)}"
                )
            x_min, x_max, y_min, y_max = court_bounds
            if x_min >= x_max or y_min >= y_max:
                raise ValueError(
                    f"court_bounds must satisfy x_min<x_max and y_min<y_max; got {court_bounds}"
                )
        self.court_bounds = court_bounds

    @property
    def has_within_game_gru(self) -> bool:
        return self.within_game_gru is not None

    # Class-level annotations for ``register_buffer`` slots — keeps
    # mypy happy about the Tensor arithmetic below (register_buffer's
    # type stubs widen to ``Module | None``).
    khat_log1p_mean: Tensor | None
    khat_log1p_std: Tensor | None

    @property
    def has_khat_standardization(self) -> bool:
        return self.khat_log1p_mean is not None and self.khat_log1p_std is not None

    def _transform_khat_for_residual(self, mu: Tensor) -> Tensor:
        """Apply the optional log1p + standardize transform to K̂.

        Always returns a detached tensor with no grad to the count head.
        When no standardization stats are wired, returns the raw detached
        K̂ — preserves bit-identical behavior for the B2 and K̂-only
        paths trained before this option existed.
        """
        k_hat = mu.detach()
        if self.khat_log1p_mean is not None and self.khat_log1p_std is not None:
            k_hat = (torch.log1p(k_hat) - self.khat_log1p_mean) / self.khat_log1p_std
        return k_hat

    @property
    def has_residual(self) -> bool:
        return self.residual_encoder is not None and self.location_embedding is not None

    @property
    def has_outcome_residual(self) -> bool:
        """``True`` iff the residual encoder consumes a per-shot
        prior-outcome summary :math:`o_{n,r}` (paper Phase 2). The
        trainer reads this flag to decide whether to forward the
        ``o_n`` tensor through the spatial wrapper.
        """
        return self.residual_encoder is not None and self.residual_encoder.outcome_dim > 0

    @property
    def has_pooling_gate(self) -> bool:
        return self.pooling_gate is not None

    @property
    def has_defense(self) -> bool:
        if self.defensive_field is None:
            return False
        if isinstance(self.defensive_field, ZoneReweightingDefense):
            return self._defensive_features is not None
        return self._defensive_cache is not None and self._defensive_features is not None

    @property
    def has_bandwidth_field(self) -> bool:
        return self.bandwidth_field is not None

    @property
    def has_anisotropic_kernel(self) -> bool:
        return self.anisotropic_kernel is not None

    @property
    def has_matchup(self) -> bool:
        return self.matchup_field is not None and self._matchup_features is not None

    def forward(
        self,
        player_idx: Tensor,
        snapshot_idx: Tensor,
        x_n_raw: Tensor,
        x_n: Tensor,
        shot_xy: Tensor,
        h_n: Tensor | None = None,
        opp_idx: Tensor | None = None,
        prior_seq: Tensor | None = None,
        prior_lengths: Tensor | None = None,
        o_n: Tensor | None = None,
    ) -> ContinuousMixtureOutputs:
        """Compute the per-row continuous-mixture log-likelihood.

        Parameters
        ----------
        player_idx, snapshot_idx, x_n_raw, x_n
            Standard collaborative-KDE inputs.
        shot_xy : Tensor of shape ``(B, 2)``
            Exact observed shot coordinate, in court feet — the data
            point the mixture is evaluated at.
        h_n : Tensor of shape ``(B, within_game_dim)`` or ``None``
            Within-game causal shot history, required when the
            residual encoder's ``within_game_dim > 0``.
        opp_idx : Tensor of shape ``(B,)`` int64 or ``None``
            Per-row defending-team vocab index. Required iff
            ``has_defense`` (see :attr:`has_defense`); ignored when
            the wrapper has no defensive field wired.
        prior_seq : Tensor of shape ``(B, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)`` or ``None``
            Per-row padded prior-shot sequence for the G1 within-game
            GRU. Required iff :attr:`has_within_game_gru`.
        prior_lengths : Tensor of shape ``(B,)`` int64 or ``None``
            Per-row valid prior-shot count. Required iff
            :attr:`has_within_game_gru`.

        Returns
        -------
        ContinuousMixtureOutputs
            ``log_lik`` is the per-row log-density at ``shot_xy``;
            the trainer typically takes ``-log_lik.mean()`` as the
            spatial loss.
        """
        collab = self.offensive_prior.forward_continuous(player_idx, snapshot_idx, x_n_raw, x_n)
        logits = collab.support_logits  # (B, M) — A + B

        # Phase 2 α1 (causal redesign, 2026-06-09): zone-pair edge bias
        # on support attention, derived from x_n + support_xy only.
        # Zero-init B → no-op at step 0 (decoder bit-identical to
        # the unbiased mainline). See :class:`CausalZoneBias` and the
        # locked leakage rule above. The observed shot location is
        # NOT passed in — the leakage-guardrail test enforces this.
        if self.causal_zone_bias is not None:
            edge_bias = self.causal_zone_bias(x_n=x_n, support_xy=collab.support_xy)
            logits = logits + edge_bias
        # Kernel-shape dispatch (Tier-2 / Tier-1a / fixed):
        # * ``anisotropic_kernel`` (Tier-2): precompute per-shot
        #   ``log K_m(δ)`` from per-zone covariance and pass directly to
        #   the loglik. Bypasses the σ machinery entirely. ``sigma_eff``
        #   becomes a diagnostic-only placeholder in this mode (the
        #   wrapper output's ``sigma`` reports collab's per-row σ as a
        #   no-op-comparable baseline; the actual kernel shape comes
        #   from ``log_kernel_eff``).
        # * ``bandwidth_field`` (Tier-1a): per-(source, zone) scalar σ
        #   → ``sigma_eff`` of shape (B, M).
        # * Neither: ``sigma_eff = collab.sigma`` of shape (B,).
        log_kernel_eff: Tensor | None = None
        if self.anisotropic_kernel is not None:
            log_kernel_eff = self.anisotropic_kernel(support_xy=collab.support_xy, shot_xy=shot_xy)
            sigma_eff: Tensor = collab.sigma  # diagnostic only when anisotropic is on
        elif self.bandwidth_field is not None:
            sigma_eff = self.bandwidth_field(support_xy=collab.support_xy, own_mask=collab.own_mask)
        else:
            sigma_eff = collab.sigma

        # Stratified-court kernel mask (Phase 1 C1, 2026-06-09). At
        # ``stratified_epsilon=1.0`` this is a no-op (zero tensor →
        # bit-identical to mainline). Otherwise it's an additive
        # log-multiplier on the per-(query, support) kernel value:
        # ``0`` when query and support live in the same court zone,
        # ``log(stratified_epsilon)`` when they differ.
        log_kernel_extra: Tensor | None = None
        if self.stratified_epsilon != 1.0:
            from shotcloud.data.zones import zone_from_xy_torch

            z_y = zone_from_xy_torch(shot_xy[:, 0], shot_xy[:, 1])  # (B,)
            sup_xy = collab.support_xy  # (B, M, 2)
            z_s = zone_from_xy_torch(sup_xy[..., 0], sup_xy[..., 1])  # (B, M)
            same_zone = z_y.unsqueeze(-1) == z_s
            log_eps = math.log(self.stratified_epsilon)
            log_kernel_extra = torch.where(
                same_zone,
                torch.zeros_like(logits),
                torch.full_like(logits, log_eps),
            )
        residual_logits = torch.zeros_like(logits)
        defense_logits: Tensor | None = None
        defense_cold_start: Tensor | None = None
        matchup_logits: Tensor | None = None
        matchup_n_eff: Tensor | None = None

        if self.has_defense:
            if opp_idx is None:
                raise ValueError(
                    "defensive_field is wired but opp_idx was not passed; "
                    "thread opp_idx through the trainer batch."
                )
            assert self.defensive_field is not None
            assert self._defensive_features is not None
            if isinstance(self.defensive_field, ZoneReweightingDefense):
                # D-lite path: only needs per-row defense features
                # (the centered-zone block). No cache, no pairwise KDE.
                feat_buf = self._defensive_features.features.to(opp_idx.device)
                def_features_per_row = feat_buf[opp_idx, snapshot_idx]  # (B, D_def)
                defense_logits = self.defensive_field(
                    query_xy=collab.support_xy,
                    def_features=def_features_per_row,
                )
                # Cold-start = opponent's centered-zone block is all
                # zero (D-lite contribution is identically zero).
                from shotcloud.features.defense_features import _ZONE_CENTERED_SLICE

                czeros = def_features_per_row[:, _ZONE_CENTERED_SLICE].abs().sum(dim=-1)
                defense_cold_start = czeros == 0
            else:
                assert self._defensive_cache is not None
                gathered = gather_defense_inputs(
                    opp_idx=opp_idx,
                    snapshot_idx=snapshot_idx,
                    cache=self._defensive_cache,
                    features=self._defensive_features,
                )
                defense_logits = self.defensive_field(
                    query_xy=collab.support_xy,
                    def_xy=gathered["def_xy"],
                    def_mask=gathered["def_mask"],
                    def_age_days=gathered["def_age_days"],
                    def_features=gathered["def_features"],
                    x_n=x_n,
                    h_n=h_n,
                    opp_idx=opp_idx,
                )
                defense_cold_start = ~gathered["def_mask"].any(dim=-1)
            # Defense enters BEFORE the residual and BEFORE the
            # own/pooled subset softmax — it reshapes mass within
            # each subset, leaving λ untouched.
            logits = logits + defense_logits

        if self.has_matchup:
            if opp_idx is None:
                raise ValueError(
                    "matchup_field is wired but opp_idx was not passed; "
                    "thread opp_idx through the trainer batch."
                )
            assert self.matchup_field is not None
            assert self._matchup_features is not None
            # Gather per-row Δ̂_{player_idx, snapshot_idx, opp_idx, :}
            # and N^eff (the latter for the ESS-bucket diagnostic). The
            # feature tensors live on CPU and are gathered onto the
            # batch's device on the fly — same pattern as the D-lite
            # defense features.
            delta_buf = self._matchup_features.delta_hat.to(opp_idx.device)
            n_eff_buf = self._matchup_features.n_eff.to(opp_idx.device)
            delta_per_row = delta_buf[player_idx, snapshot_idx, opp_idx]  # (B, N_ZONES)
            matchup_n_eff = n_eff_buf[player_idx, snapshot_idx, opp_idx]  # (B,)
            matchup_logits = self.matchup_field(
                query_xy=collab.support_xy,
                delta_hat=delta_per_row,
            )
            # Matchup enters at the same stage as D-lite — both are
            # zone-conditional opponent reweighting, summed into the
            # support logits before the residual and the own/pooled
            # subset softmax.
            logits = logits + matchup_logits

        if self.has_residual:
            assert self.residual_encoder is not None
            assert self.location_embedding is not None
            h_for_residual: Tensor | None = None
            if self.residual_encoder.within_game_dim > 0:
                if h_n is None:
                    raise ValueError(
                        "residual_encoder.within_game_dim > 0 → h_n is required; "
                        "pass the per-shot within-game-history tensor"
                    )
                h_for_residual = h_n
            usage_for_residual: Tensor | None = None
            if self.residual_encoder.usage_dim > 0:
                # Extract the causal usage-state vector from the
                # offensive prior's trait buffer. ``traits`` is
                # ``(n_players, n_snapshots, trait_dim)`` and shared
                # across both backbones; the slot indices are
                # canonical (see ``USAGE_SLOT_INDICES``). The extracted
                # vector is causal by construction (trait values are
                # computed only from pre-snapshot data).
                from shotcloud.features.usage_features import (
                    USAGE_DIM,
                    USAGE_KHAT_DIM,
                    extract_usage,
                )

                usage_dim = self.residual_encoder.usage_dim
                if usage_dim == 1 and self.count_head is not None:
                    # K̂-only diagnostic: residual sees ONLY the detached
                    # predicted count — no causal-usage extract. Used to
                    # decompose how much of the B2 cloud-metric gain is
                    # carried by K̂ alone vs the (usage × K̂) interaction.
                    mu, _ = self.count_head(x_n)
                    usage_for_residual = self._transform_khat_for_residual(mu).unsqueeze(-1)
                elif usage_dim == USAGE_DIM:
                    # B1: causal usage vector only (no K̂ column).
                    usage_for_residual = extract_usage(
                        self.offensive_prior.traits, player_idx, snapshot_idx
                    )
                elif usage_dim == USAGE_KHAT_DIM and self.count_head is not None:
                    # B2 mainline: causal usage + detached K̂.
                    base_usage = extract_usage(
                        self.offensive_prior.traits, player_idx, snapshot_idx
                    )
                    mu, _ = self.count_head(x_n)
                    k_hat = self._transform_khat_for_residual(mu).unsqueeze(-1)
                    usage_for_residual = torch.cat([base_usage, k_hat], dim=-1)
                else:
                    raise ValueError(
                        f"unsupported residual usage_dim={usage_dim} with "
                        f"count_head={'wired' if self.count_head else 'None'}; "
                        f"expected one of: 1 + count_head (K̂-only), "
                        f"{USAGE_DIM} (B1 usage-only), {USAGE_KHAT_DIM} + "
                        "count_head (B2 usage + K̂)"
                    )
                # Defensive shape check — the encoder rejects mismatched
                # usage shapes with a clearer error, but the explicit
                # assertion documents the contract for future maintainers.
                assert usage_for_residual.shape[-1] == self.residual_encoder.usage_dim
            # Phase 2: optional causal prior-outcome summary branch.
            # When the encoder's ``outcome_dim > 0`` we forward the
            # per-shot tensor; otherwise we leave it as ``None`` and
            # the encoder's validation enforces consistency.
            outcome_for_residual: Tensor | None = None
            if self.residual_encoder.outcome_dim > 0:
                if o_n is None:
                    raise ValueError(
                        "residual_encoder.outcome_dim > 0 but o_n was not provided to "
                        "forward(); the trainer must pass the per-shot prior-outcome "
                        "summary tensor."
                    )
                outcome_for_residual = o_n
            u = self.residual_encoder(
                x_n, h_for_residual, usage=usage_for_residual, outcome=outcome_for_residual
            )
            # G1: add the within-game GRU contribution to the residual
            # encoder output before the location embedding. The GRU's
            # output projection is zero-initialized, so at step 0 this
            # adds the zero vector and the wrapper is bit-identical to
            # the no-GRU B2 path. ``has_within_game_gru`` is the
            # checkpoint flag eval-side reconstruction uses to rebuild
            # the module with matching shape.
            if self.has_within_game_gru:
                if prior_seq is None or prior_lengths is None:
                    raise ValueError(
                        "within_game_gru is wired but prior_seq/prior_lengths were not "
                        "provided to forward(); the trainer must pass both tensors."
                    )
                assert self.within_game_gru is not None  # narrowed by has_within_game_gru
                g_r = self.within_game_gru(prior_seq, prior_lengths)
                if g_r.shape != u.shape:
                    raise ValueError(
                        f"within_game_gru output shape {tuple(g_r.shape)} does not match "
                        f"the residual encoder output shape {tuple(u.shape)}; check the "
                        "GRU's out_dim against the residual rank."
                    )
                u = u + g_r
            elif prior_seq is not None or prior_lengths is not None:
                raise ValueError(
                    "prior_seq/prior_lengths supplied but within_game_gru is not wired; "
                    "drop the tensors or wire the GRU."
                )
            psi = self.location_embedding(collab.support_xy)  # (B, M, rank)
            residual_logits = (u.unsqueeze(1) * psi).sum(dim=-1)  # (B, M)
            logits = logits + residual_logits

        cold_start = ~collab.support_mask.any(dim=-1)

        from shotcloud.training.spatial_losses import continuous_mixture_loglik

        if self.has_pooling_gate:
            log_lik, log_w, gate_lambda = self._gated_forward(
                collab=collab,
                logits=logits,
                player_idx=player_idx,
                snapshot_idx=snapshot_idx,
                x_n=x_n,
                h_n=h_n,
                shot_xy=shot_xy,
                sigma_eff=sigma_eff,
                log_kernel_eff=log_kernel_eff,
                log_kernel_extra=log_kernel_extra,
            )
        else:
            # Mask invalid support and joint-softmax. ``logsumexp(all
            # -inf)`` returns NaN in PyTorch; patch a single dummy
            # 0-logit per cold-start row so the softmax is
            # well-defined; the row's ``log_lik`` is overwritten by
            # the cold-start floor below.
            masked_logits = logits.masked_fill(~collab.support_mask, float("-inf"))
            if cold_start.any():
                masked_logits = masked_logits.clone()
                masked_logits[cold_start, 0] = 0.0
            log_w = masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)
            if log_kernel_eff is not None:
                # Anisotropic path: court_bounds (isotropic) is not the
                # right normalizer here; the anisotropic-kernel module
                # would have to supply its own log_court_normalizer.
                # Not threaded for the present submission.
                log_lik = continuous_mixture_loglik(
                    log_w,
                    collab.support_xy,
                    shot_xy,
                    log_kernel=log_kernel_eff,
                    log_kernel_extra=log_kernel_extra,
                    weights_are_log_probs=True,
                    support_mask=collab.support_mask,
                )
            else:
                log_lik = continuous_mixture_loglik(
                    log_w,
                    collab.support_xy,
                    shot_xy,
                    sigma_eff,
                    log_kernel_extra=log_kernel_extra,
                    weights_are_log_probs=True,
                    support_mask=collab.support_mask,
                    court_bounds=self.court_bounds,
                )
            gate_lambda = None

        # Cold-start floor: rows with no valid support get a uniform-court
        # density rather than the dummy-support value, so the batch mean
        # stays finite and the loss can't be gamed by a row that has no
        # real support to score against.
        if cold_start.any():
            log_lik = torch.where(
                cold_start,
                torch.full_like(log_lik, self.cold_start_log_lik_floor),
                log_lik,
            )

        # ``ContinuousMixtureOutputs.sigma`` semantics: shape (B,), a
        # per-row scalar diagnostic. When per-shot σ is active it's the
        # mean over the row's support shots; otherwise it's collab's σ
        # unchanged. Per-shot σ is exposed via ``sigma_per_shot`` for
        # callers that want the full tensor.
        sigma_row = sigma_eff.mean(dim=-1) if sigma_eff.dim() == 2 else sigma_eff

        return ContinuousMixtureOutputs(
            log_lik=log_lik,
            log_weights=log_w,
            support_xy=collab.support_xy,
            sigma=sigma_row,
            sigma_per_shot=(sigma_eff if sigma_eff.dim() == 2 else None),
            support_mask=collab.support_mask,
            residual_logits=residual_logits,
            collab=collab,
            cold_start=cold_start,
            support_log_weights=log_w,
            gate_lambda=gate_lambda,
            defense_logits=defense_logits,
            defense_cold_start=defense_cold_start,
            matchup_logits=matchup_logits,
            matchup_n_eff=matchup_n_eff,
        )

    @staticmethod
    def _subset_log_weights(logits: Tensor, subset_mask: Tensor) -> Tensor:
        """Per-subset softmax-normalized log-weights ``(B, M)``.

        ``logits`` masked to ``subset_mask`` then normalized over the
        subset. Rows with an all-``False`` subset get a dummy 0-logit
        at slot 0 so the ``logsumexp`` is finite (no NaN); those rows'
        component is discarded by the gate's ``λ``/``1-λ`` anyway.
        """
        masked = logits.masked_fill(~subset_mask, float("-inf"))
        empty = ~subset_mask.any(dim=-1)
        if empty.any():
            masked = masked.clone()
            masked[empty, 0] = 0.0
        return masked - torch.logsumexp(masked, dim=-1, keepdim=True)

    def _gated_forward(
        self,
        collab: CollaborativeContinuousOutputs,
        logits: Tensor,
        player_idx: Tensor,
        snapshot_idx: Tensor,
        x_n: Tensor,
        h_n: Tensor | None,
        shot_xy: Tensor,
        sigma_eff: Tensor,
        log_kernel_eff: Tensor | None = None,
        log_kernel_extra: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Structured two-component density: ``λ·f_own + (1-λ)·f_pooled``.

        Each component normalizes its support attention *within* its
        own subset (own-player shots vs pooled-player shots); the
        history-dependent gate ``λ`` decides the mix. Returns
        ``(log_lik, log_w_effective, λ)``.
        """
        from shotcloud.training.spatial_losses import continuous_mixture_loglik

        assert self.pooling_gate is not None
        # Backend-agnostic own/pooled partition: read ``own_mask``
        # directly from the collab output (PR3.0 contract). Both the
        # L×R backend and the retrieval backend populate it.
        own_mask = collab.own_mask
        pooled_mask = (~own_mask) & collab.support_mask
        own_available = own_mask.any(dim=-1)  # (B,)
        pooled_available = pooled_mask.any(dim=-1)  # (B,)
        own_support_count = own_mask.sum(dim=-1)  # (B,)

        # Trait-derived log(1 + own causal shot count) — the same
        # causal count the σ-head consumes (trait slot log1p_M).
        log1p_h_hat = self.offensive_prior.traits[player_idx, snapshot_idx][..., _SLOT_LOG1P_M]

        lam = self.pooling_gate(
            log1p_h_hat=log1p_h_hat,
            x_n=x_n,
            own_support_count=own_support_count,
            own_available=own_available,
            h_n=h_n,
            pooled_available=pooled_available,
        )  # (B,) in [0, 1]

        log_w_own = self._subset_log_weights(logits, own_mask)
        log_w_pooled = self._subset_log_weights(logits, pooled_mask)
        # Dispatch: anisotropic ⇒ pass precomputed log_kernel; else σ.
        if log_kernel_eff is not None:
            log_f_own = continuous_mixture_loglik(
                log_w_own,
                collab.support_xy,
                shot_xy,
                log_kernel=log_kernel_eff,
                log_kernel_extra=log_kernel_extra,
                weights_are_log_probs=True,
                support_mask=own_mask,
            )
            log_f_pooled = continuous_mixture_loglik(
                log_w_pooled,
                collab.support_xy,
                shot_xy,
                log_kernel=log_kernel_eff,
                log_kernel_extra=log_kernel_extra,
                weights_are_log_probs=True,
                support_mask=pooled_mask,
            )
        else:
            log_f_own = continuous_mixture_loglik(
                log_w_own,
                collab.support_xy,
                shot_xy,
                sigma_eff,
                log_kernel_extra=log_kernel_extra,
                weights_are_log_probs=True,
                support_mask=own_mask,
                court_bounds=self.court_bounds,
            )  # (B,) — -inf for rows with no own support
            log_f_pooled = continuous_mixture_loglik(
                log_w_pooled,
                collab.support_xy,
                shot_xy,
                sigma_eff,
                log_kernel_extra=log_kernel_extra,
                weights_are_log_probs=True,
                support_mask=pooled_mask,
                court_bounds=self.court_bounds,
            )

        # log λ / log(1-λ) — clamp λ off the {0,1} endpoints purely for
        # gradient hygiene. The forced-edge rows (λ=0 with no own
        # support, λ=1 with no pooled support) have the corresponding
        # component log-lik at -inf, so the term is -inf either way;
        # the clamp only keeps d/dλ log(λ) finite.
        lam_c = lam.clamp(1e-12, 1.0 - 1e-12)
        log_lam = torch.log(lam_c)
        log_1m = torch.log1p(-lam_c)
        log_lik = torch.logaddexp(log_lam + log_f_own, log_1m + log_f_pooled)

        # Effective joint per-support weight for diagnostics: own slots
        # carry λ·ω_own, pooled slots (1-λ)·ω_pooled. Sums to 1 over M,
        # so ``pooling_diagnostics`` reads pooled_mass = 1-λ directly.
        log_w_eff = torch.where(
            own_mask,
            log_lam.unsqueeze(-1) + log_w_own,
            log_1m.unsqueeze(-1) + log_w_pooled,
        )
        return log_lik, log_w_eff, lam


__all__ = [
    "ContinuousMixtureOutputs",
    "ContinuousMixtureSpatial",
]
