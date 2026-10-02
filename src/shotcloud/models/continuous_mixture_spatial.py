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
    r"""Per-batch intermediates of one cell-free spatial forward pass.

    Shared by :class:`ContinuousMixtureSpatial` and the mode-based
    alternatives
    (:class:`~shotcloud.models.collaborative_mode_mixture.CollaborativeModeMixtureSpatial`,
    :class:`~shotcloud.models.mode_routed_spatial.ModeRoutedContinuousMixtureSpatial`),
    so diagnostics can be computed without repeating the forward pass.
    ``n_components`` is the number of support shots ``M`` for the
    continuous mixture and the number of court modes ``K`` for the
    mode-mixture variant.

    Attributes
    ----------
    log_lik : Tensor of shape (B,)
        Log-density of the mixture at the observed shot coordinate.
    log_weights : Tensor of shape (B, n_components)
        Normalized log mixture weights; ``-inf`` on invalid components.
        With a pooling gate these are the effective weights
        :math:`\lambda\,\omega_{\mathrm{own}}` on own slots and
        :math:`(1 - \lambda)\,\omega_{\mathrm{pooled}}` on pooled slots.
    support_xy : Tensor of shape (B, n_components, 2)
        Component centers (support shots or mode anchors), in court feet.
    sigma : Tensor of shape (B,)
        Per-row bandwidth. When a per-shot bandwidth is active this is
        the mean over the row's support shots (a diagnostic); for the
        mode-mixture variant it is the mean across modes.
    support_mask : Tensor of shape (B, n_components), bool
        Valid components. All ``True`` for the mode-mixture variant,
        where cold start is handled per row rather than per mode.
    residual_logits : Tensor of shape (B, M)
        Residual-tilt contribution :math:`R_\theta(s_m)` to the support
        logits; zero when no residual is wired.
    collab : CollaborativeContinuousOutputs
        Raw outputs of the support backend.
    cold_start : Tensor of shape (B,), bool
        Rows whose causal support set is empty after masking.
    support_log_weights : Tensor of shape (B, M)
        Log support attention :math:`\log \omega_m`. Equal to
        ``log_weights`` for the continuous mixture; always indexed by
        support shot, so support-level diagnostics apply to every
        variant.
    tail_responsibility : Tensor of shape (B,) or None
        Posterior responsibility of the support-KDE tail component for
        the observed shot,
        :math:`\lambda_{\mathrm{tail}} f_{\mathrm{tail}} /
        [(1 - \lambda_{\mathrm{tail}}) f_{\mathrm{mode}}
        + \lambda_{\mathrm{tail}} f_{\mathrm{tail}}]`. Set only by the
        mode-mixture variant with ``tail_weight > 0``.
    gate_lambda : Tensor of shape (B,) or None
        Own-support mixing weight :math:`\lambda` of the pooling gate,
        so that :math:`f = \lambda f_{\mathrm{own}} + (1 - \lambda)
        f_{\mathrm{pooled}}`. Set only when a pooling gate is wired.
    defense_logits : Tensor of shape (B, M) or None
        Opponent-reweighting contribution :math:`D(s_m)` to the support
        logits; zero on defensive cold-start rows. Set only when a
        defensive field is wired.
    defense_cold_start : Tensor of shape (B,), bool, or None
        Rows with no causal defensive evidence for the opponent at the
        snapshot; their defensive contribution is identically zero.
    sigma_per_shot : Tensor of shape (B, M) or None
        Per-support-shot bandwidth from
        :class:`~shotcloud.models.zone_source_bandwidth.ZoneSourceBandwidth`;
        the tensor the log-likelihood actually uses when set.
    matchup_logits : Tensor of shape (B, M) or None
        Player-versus-opponent matchup contribution
        :math:`\beta_{\mathrm{match}}\,\widehat\Delta_{p,d,z(s_m)}(t)` to the
        support logits. Exactly zero on rows with no causal matchup
        evidence. Set only when a matchup field is wired.
    matchup_n_eff : Tensor of shape (B,) or None
        Effective peer-versus-opponent sample size used to shrink
        :math:`\widehat\Delta`. Set only when a matchup field is wired.
    mode_log_pi : Tensor of shape (B, K) or None
        Mode-router log-probabilities renormalized over the modes with
        causal support (unavailable modes are ``-inf``). Set only by the
        mode-routed decoder.
    mode_available : Tensor of shape (B, K), bool, or None
        Modes with at least one causal support shot.
    mode_log_pi_raw : Tensor of shape (B, K) or None
        Mode-router log-probabilities before availability
        renormalization, so the mass assigned to empty modes,
        :math:`1 - \sum_{k\ \mathrm{available}} \pi_k`, can be reported.
    """

    log_lik: Tensor
    log_weights: Tensor
    support_xy: Tensor
    sigma: Tensor
    support_mask: Tensor
    residual_logits: Tensor
    collab: CollaborativeContinuousOutputs
    cold_start: Tensor
    support_log_weights: Tensor
    tail_responsibility: Tensor | None = None
    gate_lambda: Tensor | None = None
    defense_logits: Tensor | None = None
    defense_cold_start: Tensor | None = None
    sigma_per_shot: Tensor | None = None
    matchup_logits: Tensor | None = None
    matchup_n_eff: Tensor | None = None
    mode_log_pi: Tensor | None = None
    mode_available: Tensor | None = None
    mode_log_pi_raw: Tensor | None = None


class ContinuousMixtureSpatial(nn.Module):
    r"""Cell-free continuous-mixture spatial decoder (the AC-KDE spatial factor).

    For each shot the decoder gathers causal support shots
    :math:`s_m` from ``offensive_prior``, forms support logits

    .. math::

        \ell_m = \underbrace{A + B}_{\text{backend}}
                + b^{\mathrm{zone}}_m + D(s_m) + M(s_m) + R_\theta(s_m),

    and evaluates the kernel mixture
    :math:`f(y) = \sum_m \mathrm{softmax}_m(\ell_m)\, K_m(y - s_m)` at the
    observed coordinate :math:`y`. Each optional term is present only when
    its module is wired: the causal zone bias :math:`b^{\mathrm{zone}}`,
    the opponent reweighting :math:`D`, the matchup reweighting :math:`M`,
    and the low-rank residual tilt :math:`R_\theta(s_m) = u^\top \psi(s_m)`.
    With a ``pooling_gate`` the softmax is taken separately over own and
    pooled support and the two densities are mixed by the gate weight
    :math:`\lambda`.

    The kernel :math:`K_m` is an isotropic Gaussian whose bandwidth is the
    backend's per-row :math:`\sigma`, or a per-support-shot
    :math:`\sigma_m` from ``bandwidth_field``; alternatively
    ``anisotropic_kernel`` supplies the log-kernel directly.

    Parameters
    ----------
    offensive_prior : CollaborativeKDE or RetrievalCollaborativeKDE
        Support backend. Supplies the support shots and mask, the own/pooled
        partition, the support logits :math:`A + B`, the bandwidth, and the
        causal trait buffer.
    residual_encoder : ContextResidualEncoder or None
        Context encoder producing :math:`u \in \mathbb R^{\mathrm{rank}}`
        for the residual tilt. Must be paired with ``location_embedding``.
    location_embedding : LocationEmbedding or None
        Coordinate embedding :math:`\psi(s) \in \mathbb R^{\mathrm{rank}}`;
        its ``rank`` must equal the residual encoder's.
    defensive_field : ContinuousAdaptiveDefensiveField or ZoneReweightingDefense or None
        Opponent reweighting :math:`D(s_m)`.
        :class:`~shotcloud.models.zone_defense_reweighting.ZoneReweightingDefense`
        (zone-level reweighting, the mainline choice) requires
        ``defensive_features`` and no cache;
        :class:`~shotcloud.models.continuous_adaptive_defensive.ContinuousAdaptiveDefensiveField`
        (a kernel field over allowed shots, evaluated as an ablation)
        requires both ``defensive_cache`` and ``defensive_features``.
    defensive_cache : DefensiveRetrievalCache or None
        Per-(opponent, snapshot) causal allowed-shot cache, consumed
        through
        :func:`~shotcloud.models.continuous_adaptive_defensive.gather_defense_inputs`.
    defensive_features : DefenseFeatures or None
        Per-(opponent, snapshot) causal defensive feature tensor.
    matchup_field : MatchupReweightingDefense or None
        Player-versus-opponent zone reweighting :math:`M(s_m)`. Must be
        paired with ``matchup_features``; composes additively with
        ``defensive_field``.
    matchup_features : MatchupFeatures or None
        Per-(player, snapshot, opponent) shrunken zone deltas and
        effective sample sizes.
    pooling_gate : PoolingGate or None
        Gate producing the own-support weight :math:`\lambda`. When
        ``None``, a single softmax runs over all support shots.
    bandwidth_field : ZoneSourceBandwidth or None
        Per-(source, zone) isotropic bandwidth :math:`\sigma_m`. Mutually
        exclusive with ``anisotropic_kernel``.
    anisotropic_kernel : RadialTangentZoneKernel or FullCovarianceZoneKernel or None
        Per-zone anisotropic Gaussian kernel that replaces the isotropic
        kernel. Mutually exclusive with ``bandwidth_field``.
    count_head : nn.Module or None
        Count head (e.g. :class:`~shotcloud.models.count_head.NegBinCountHead`)
        whose predicted mean :math:`\hat K` is appended, detached, to the
        residual encoder's usage input. Requires ``residual_encoder`` with
        ``usage_dim`` equal to ``USAGE_KHAT_DIM`` (usage plus
        :math:`\hat K`) or 1 (:math:`\hat K` only).
    khat_log1p_mean, khat_log1p_std : float or None
        Standardization statistics; when given, the residual receives
        :math:`(\log(1 + \hat K) - \mu) / \sigma` instead of the raw
        :math:`\hat K`. Both or neither; require ``count_head``.
    within_game_gru : nn.Module or None
        Within-game recurrent encoder over the player's earlier shots in
        the same game, added to :math:`u` before the residual tilt
        (evaluated as an ablation). Requires ``residual_encoder``.
    cold_start_log_lik_floor : float, default ``_COLD_START_LOG_LIK_FLOOR``
        Log-likelihood assigned to rows with no valid causal support.
    stratified_epsilon : float, default 1.0
        Cross-zone kernel attenuation in ``(0, 1]``. Values below 1 multiply
        the kernel by ``stratified_epsilon`` whenever the evaluation point
        and the support shot lie in different court zones; 1.0 disables it.
    causal_zone_bias : CausalZoneBias or None
        Optional zone-pair bias on the support logits.
    court_bounds : tuple of float or None
        ``(x_min, x_max, y_min, y_max)`` in court feet. When set, each
        isotropic kernel is renormalized to integrate to one over this
        rectangle, so the predictive density is a proper density on the
        court. When ``None``, kernels are normalized on
        :math:`\mathbb R^2`. Not applied to ``anisotropic_kernel``.

    Raises
    ------
    ValueError
        If paired arguments are wired inconsistently, the residual and
        location-embedding ranks differ, ``bandwidth_field`` and
        ``anisotropic_kernel`` are both set, the residual ``usage_dim``
        does not match the ``count_head`` wiring, or ``stratified_epsilon``
        or ``court_bounds`` is out of range.

    Notes
    -----
    Causality contract: support shots, traits, defensive and matchup
    features are snapshot-indexed and use only games strictly before the
    shot's game; the within-game inputs use only earlier shots of the same
    game. The observed coordinate ``shot_xy`` is used only to evaluate the
    kernels (and, with ``stratified_epsilon < 1``, the zone of the
    evaluation point inside the kernel); it never enters a support logit,
    the gate, or the residual.

    Rows with an empty support set receive ``cold_start_log_lik_floor``
    instead of a mixture density.
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
        # Defense wiring rules:
        # * ContinuousAdaptiveDefensiveField (kernel field) requires the
        #   full triple ``field + cache + features``.
        # * ZoneReweightingDefense (zone-level) requires ``field + features``
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
        # Matchup wiring rules: ``matchup_field`` and ``matchup_features``
        # must both be provided or both None. The matchup channel
        # composes with the defensive field: when both are wired, their
        # logit contributions are summed before the per-subset softmax.
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
        # Kernel-shape wiring rules:
        # * ``bandwidth_field``: per-(source, zone) scalar σ on the
        #   isotropic Gaussian kernel.
        # * ``anisotropic_kernel``: replaces the isotropic kernel entirely
        #   with a per-zone anisotropic covariance.
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
        # Count-head wiring. When provided, the wrapper appends a
        # detached predicted-count column ``K̂ = NegBinCountHead(x_n).μ``
        # to the causal usage vector before passing it to the residual
        # encoder. The count head is owned by the trainer at top level
        # (so it receives the count-loss gradient); ``self.count_head``
        # is the same nn.Module instance, registered here as a child so
        # ``.to(device)`` moves it. The detach in the forward keeps the
        # spatial loss from turning the count head into a hidden
        # variable.
        # Residual usage_dim must match the count_head wiring under
        # one of these configurations:
        #   * usage_dim == 0, count_head is None              — no usage input.
        #   * usage_dim == USAGE_DIM, count_head is None      — usage only.
        #   * usage_dim == USAGE_KHAT_DIM, count_head wired   — usage + K̂ (mainline).
        #   * usage_dim == 1, count_head wired                — K̂ only (diagnostic).
        # Any other combination is a wiring error.
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
        # K̂-standardization stats. When both are provided, the residual
        # receives ``(log1p(K̂) − μ) / σ`` instead of raw ``K̂`` (still
        # detached from the count head). This keeps the residual input
        # O(1) when the count head is calibrated onto the count scale via
        # ``init_mean``, where raw K̂ is O(10). ``None`` → raw K̂.
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
        # Within-game shot GRU. The GRU consumes the player's earlier
        # shots in the same game and emits a per-row vector that is
        # added to the residual encoder's output, so it requires
        # ``residual_encoder``. Its output projection is
        # zero-initialized, so at initialization the decoder equals the
        # one without the GRU.
        if within_game_gru is not None and residual_encoder is None:
            raise ValueError(
                "within_game_gru wired but residual_encoder is None — the within-game "
                "GRU contribution can only enter through the residual tilt."
            )
        self.within_game_gru = within_game_gru
        self.cold_start_log_lik_floor = float(cold_start_log_lik_floor)
        # Stratified-court kernel. At ``stratified_epsilon=1.0`` the
        # kernel is unchanged. At any value in (0, 1), the
        # per-(query, support) log-kernel gets an additive penalty
        # ``log(epsilon)`` when the evaluation point's zone differs from
        # the support shot's zone, limiting Gaussian mass that leaks
        # across the 3-point arc, the paint boundary, and the corners.
        if not 0.0 < stratified_epsilon <= 1.0:
            raise ValueError(f"stratified_epsilon must be in (0, 1]; got {stratified_epsilon}")
        self.stratified_epsilon = float(stratified_epsilon)
        # Zone-pair bias on support attention, derived only from ``x_n``
        # and ``support_xy``. It must never condition on the zone of the
        # observed shot: a bias on ``z(y_n)`` makes the support weights
        # depend on the evaluation point, so the result is no longer a
        # normalized predictive density.
        self.causal_zone_bias = causal_zone_bias
        # Court-boundary correction. When set, every call into
        # ``continuous_mixture_loglik`` subtracts the analytic
        # per-support-shot ``log Z_m(C)`` so the predictive density
        # integrates to one over the rectangle ``court_bounds`` rather
        # than over R^2. ``None`` keeps the unbounded R^2 normalization.
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
        """``True`` if a within-game GRU is wired into the residual."""
        return self.within_game_gru is not None

    # Class-level annotations for the ``register_buffer`` slots, which
    # the type stubs would otherwise widen to ``Module | None``.
    khat_log1p_mean: Tensor | None
    khat_log1p_std: Tensor | None

    @property
    def has_khat_standardization(self) -> bool:
        """``True`` if K̂ is log1p-standardized before entering the residual."""
        return self.khat_log1p_mean is not None and self.khat_log1p_std is not None

    def _transform_khat_for_residual(self, mu: Tensor) -> Tensor:
        """Apply the optional log1p + standardize transform to K̂.

        Always returns a detached tensor, so no gradient reaches the
        count head. Without standardization stats, returns the raw
        detached K̂.
        """
        k_hat = mu.detach()
        if self.khat_log1p_mean is not None and self.khat_log1p_std is not None:
            k_hat = (torch.log1p(k_hat) - self.khat_log1p_mean) / self.khat_log1p_std
        return k_hat

    @property
    def has_residual(self) -> bool:
        """``True`` if the low-rank residual tilt is wired."""
        return self.residual_encoder is not None and self.location_embedding is not None

    @property
    def has_outcome_residual(self) -> bool:
        """``True`` if the residual encoder consumes a prior-outcome summary.

        When set, :meth:`forward` requires the per-shot causal
        prior-outcome tensor ``o_n``.
        """
        return self.residual_encoder is not None and self.residual_encoder.outcome_dim > 0

    @property
    def has_pooling_gate(self) -> bool:
        """``True`` if the own/pooled pooling gate is wired."""
        return self.pooling_gate is not None

    @property
    def has_defense(self) -> bool:
        """``True`` if an opponent reweighting field and its inputs are wired."""
        if self.defensive_field is None:
            return False
        if isinstance(self.defensive_field, ZoneReweightingDefense):
            return self._defensive_features is not None
        return self._defensive_cache is not None and self._defensive_features is not None

    @property
    def has_bandwidth_field(self) -> bool:
        """``True`` if a per-support-shot bandwidth field is wired."""
        return self.bandwidth_field is not None

    @property
    def has_anisotropic_kernel(self) -> bool:
        """``True`` if an anisotropic per-zone kernel replaces the isotropic one."""
        return self.anisotropic_kernel is not None

    @property
    def has_matchup(self) -> bool:
        """``True`` if the matchup reweighting field and its features are wired."""
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
        """Evaluate the per-row log-density at the observed shot coordinate.

        Parameters
        ----------
        player_idx : Tensor of shape ``(B,)`` int64
            Shooter vocabulary index.
        snapshot_idx : Tensor of shape ``(B,)`` int64
            Causal snapshot index of the shot's game.
        x_n_raw : Tensor of shape ``(B, context_dim)``
            Raw context vector, consumed by the backend's structured
            relevance terms.
        x_n : Tensor of shape ``(B, context_dim)``
            Learned context vector ``f_ctx(x_n_raw)``, consumed by the
            learned heads.
        shot_xy : Tensor of shape ``(B, 2)``
            Observed shot coordinate in court feet; the point at which
            the mixture is evaluated.
        h_n : Tensor of shape ``(B, within_game_dim)`` or ``None``
            Within-game causal shot history. Required when the residual
            encoder's ``within_game_dim > 0``; also passed to the
            pooling gate and the kernel defensive field.
        opp_idx : Tensor of shape ``(B,)`` int64 or ``None``
            Defending-team vocabulary index. Required when
            :attr:`has_defense` or :attr:`has_matchup`.
        prior_seq : Tensor of shape ``(B, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)`` or ``None``
            Padded sequence of the player's earlier shots in the same
            game. Required if and only if :attr:`has_within_game_gru`.
        prior_lengths : Tensor of shape ``(B,)`` int64 or ``None``
            Number of valid entries in ``prior_seq``. Required if and
            only if :attr:`has_within_game_gru`.
        o_n : Tensor of shape ``(B, outcome_dim)`` or ``None``
            Causal prior-outcome summary. Required when
            :attr:`has_outcome_residual`.

        Returns
        -------
        ContinuousMixtureOutputs
            ``log_lik`` holds the per-row log-density at ``shot_xy``;
            the spatial loss is typically ``-log_lik.mean()``.

        Raises
        ------
        ValueError
            If an input required by the wired components is missing, or
            ``prior_seq``/``prior_lengths`` are given without a GRU.
        """
        collab = self.offensive_prior.forward_continuous(player_idx, snapshot_idx, x_n_raw, x_n)
        logits = collab.support_logits  # (B, M) — A + B

        # Zone-pair bias from x_n and support_xy only; the observed shot
        # location is deliberately not an input (see CausalZoneBias).
        if self.causal_zone_bias is not None:
            edge_bias = self.causal_zone_bias(x_n=x_n, support_xy=collab.support_xy)
            logits = logits + edge_bias
        # Kernel-shape dispatch:
        # * ``anisotropic_kernel``: precompute per-shot ``log K_m(δ)``
        #   from the per-zone covariance and pass it directly to the
        #   log-likelihood, bypassing σ. ``sigma_eff`` is then only a
        #   diagnostic (the backend's per-row σ).
        # * ``bandwidth_field``: per-(source, zone) scalar σ
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

        # Stratified-court kernel mask: an additive log-multiplier on the
        # per-(query, support) kernel value, ``0`` when query and support
        # lie in the same court zone and ``log(stratified_epsilon)`` when
        # they differ. Skipped entirely at ``stratified_epsilon=1.0``.
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
                # Zone-level reweighting: only needs per-row defense
                # features (the centered-zone block); no cache, no
                # pairwise kernel.
                feat_buf = self._defensive_features.features.to(opp_idx.device)
                def_features_per_row = feat_buf[opp_idx, snapshot_idx]  # (B, D_def)
                defense_logits = self.defensive_field(
                    query_xy=collab.support_xy,
                    def_features=def_features_per_row,
                )
                # Cold start: the opponent's centered-zone block is all
                # zero, so the reweighting is identically zero.
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
            # and N^eff (the latter for diagnostics). The feature tensors
            # live on CPU and are gathered onto the batch's device on the
            # fly, as for the defense features.
            delta_buf = self._matchup_features.delta_hat.to(opp_idx.device)
            n_eff_buf = self._matchup_features.n_eff.to(opp_idx.device)
            delta_per_row = delta_buf[player_idx, snapshot_idx, opp_idx]  # (B, N_ZONES)
            matchup_n_eff = n_eff_buf[player_idx, snapshot_idx, opp_idx]  # (B,)
            matchup_logits = self.matchup_field(
                query_xy=collab.support_xy,
                delta_hat=delta_per_row,
            )
            # Matchup enters at the same stage as the zone-level defense:
            # both are zone-conditional opponent reweighting, summed into the
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
                    # K̂-only diagnostic: the residual sees only the
                    # detached predicted count, without the causal usage
                    # vector.
                    mu, _ = self.count_head(x_n)
                    usage_for_residual = self._transform_khat_for_residual(mu).unsqueeze(-1)
                elif usage_dim == USAGE_DIM:
                    # Causal usage vector only (no K̂ column).
                    usage_for_residual = extract_usage(
                        self.offensive_prior.traits, player_idx, snapshot_idx
                    )
                elif usage_dim == USAGE_KHAT_DIM and self.count_head is not None:
                    # Mainline: causal usage + detached K̂.
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
                assert usage_for_residual.shape[-1] == self.residual_encoder.usage_dim
            # Optional causal prior-outcome summary branch.
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
            # Add the within-game GRU contribution to the residual encoder
            # output before the location embedding. The GRU's output
            # projection is zero-initialized, so at initialization this
            # adds the zero vector.
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
                # Anisotropic path: the isotropic court_bounds normalizer
                # does not apply, so these kernels are normalized on R^2.
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
        # Backend-agnostic own/pooled partition: both the L×R backend
        # and the retrieval backend populate ``own_mask``.
        own_mask = collab.own_mask
        pooled_mask = (~own_mask) & collab.support_mask
        own_available = own_mask.any(dim=-1)  # (B,)
        pooled_available = pooled_mask.any(dim=-1)  # (B,)
        own_support_count = own_mask.sum(dim=-1)  # (B,)

        # Own-history volume Ĥ for the gate: the trait slot
        # log1p_minutes_M, log(1 + recency-weighted causal minutes), the
        # same evidence measure the bandwidth head consumes.
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
