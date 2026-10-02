r"""Mode-routed AC-KDE spatial decoder.

:class:`ModeRoutedContinuousMixtureSpatial` replaces the single global
support softmax of
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`
with a query-side mode router :math:`\pi_k(x_n)` and ``K`` within-mode
softmaxes :math:`\omega_{m|k}`, where the modes are the eight fixed court
zones. The density is

.. math::
    f(y \mid x_n, h_n, S_{p,t})
        = \sum_{k=1}^{8} \pi_k(x_n) \cdot
          \sum_{m: z(s_m)=k} \omega_{m|k}(x_n)
            \cdot K(y; s_m, \sigma).

Unlike the additive zone-pair bias
:class:`~shotcloud.models.continuous_mixture_spatial.CausalZoneBias`,
routing stops support shots in different zones (for example rim shots
and corner threes) from competing for mass inside one softmax
normalizer, which targets support sets that mix several shooting
regimes. It is an alternative to the single-softmax decoder, evaluated
as an ablation.

Causality
---------
No attention logit, gate, residual, or kernel modifier depends on the
observed location :math:`y_n` except through the normalized density
evaluation itself. The mode router consumes ``x_n`` only, and the
within-mode attention uses the same support logits as the parent
decoder, none of which read ``shot_xy``.

Empty modes
-----------
For each row, modes with no causal support are masked out and the
router probability is renormalized over the available modes; there is
no per-mode uniform floor. Rows with no support at all receive the
parent decoder's cold-start log-likelihood floor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from shotcloud.data.zones import N_ZONES, zone_from_xy_torch
from shotcloud.features.defense_features import _ZONE_CENTERED_SLICE
from shotcloud.models.continuous_adaptive_defensive import gather_defense_inputs
from shotcloud.models.continuous_mixture_spatial import (
    ContinuousMixtureOutputs,
    ContinuousMixtureSpatial,
)
from shotcloud.models.zone_defense_reweighting import ZoneReweightingDefense

if TYPE_CHECKING:
    pass


class ModeRouter(nn.Module):
    r"""Predicts a query-mode distribution from causal context only.

    .. math::
        \pi_k(x_n) = \mathrm{softmax}\bigl(g_\theta(x_n)\bigr)_k,
        \quad k = 1, \ldots, 8.

    Architecture: small 2-layer MLP ``Linear(context_dim, hidden_dim)
    → GELU → Linear(hidden_dim, N_ZONES)``. The final layer is
    zero-initialized so :math:`\pi_k` starts uniform; the first
    layer keeps its default init.

    Parameters
    ----------
    context_dim : int
        Dimension of the learned context vector ``x_n``. Defaults to
        27 (the project-wide ``CONTEXT_DIM``).
    hidden_dim : int
        Hidden width of the router head MLP. Defaults to 32.
    """

    def __init__(self, context_dim: int = 27, hidden_dim: int = 32) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, N_ZONES),
        )
        # Zero-init the final layer so π_k starts uniform. The decoder still
        # differs from the single-softmax decoder at initialization (the
        # normalization is structurally different), but the router starts
        # from a neutral point.
        final_layer = self.head[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, x_n: Tensor) -> Tensor:
        """Return raw mode logits of shape ``(B, N_ZONES)``.

        The caller applies the per-row availability mask and softmax.
        """
        logits: Tensor = self.head(x_n)
        return logits


class ModeRoutedContinuousMixtureSpatial(ContinuousMixtureSpatial):
    r"""Mode-routed continuous-mixture spatial decoder.

    Accepts the constructor arguments of
    :class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`
    (offensive prior, defense, residual, bandwidth, kernel, count head,
    ...) plus a required keyword ``mode_router`` (:class:`ModeRouter`),
    and overrides ``forward`` to apply mode routing instead of the
    single-softmax / pooling-gate back end.

    Differences from the parent forward pass:

    * **No pooling gate.** Own and pooled support are combined into one
      set and partitioned by ``z(s_m)``, giving a single density with
      mode routing over the combined support. Passing ``pooling_gate``
      raises ``ValueError``.
    * **No causal zone bias.** Mode routing supersedes the additive
      zone-pair bias, and using both would duplicate the signal.
      Passing ``causal_zone_bias`` raises ``ValueError``.
    * **Per-mode subset softmax** replaces the global subset softmax:
      ``ω_{m|k} = softmax_{m : z(s_m)=k}(logits_m)``.
    * **Mode router** ``π_k(x_n)`` is renormalized over the modes that
      have support in the row.

    Raises
    ------
    ValueError
        If ``pooling_gate`` or ``causal_zone_bias`` is given, or
        ``mode_router`` is missing.
    TypeError
        If ``mode_router`` is not a :class:`ModeRouter`.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        # Reject components that mode routing replaces.
        if kwargs.get("pooling_gate") is not None:
            raise ValueError(
                "ModeRoutedContinuousMixtureSpatial does not use a pooling gate; "
                "mode routing replaces own/pooled mixing. Pass pooling_gate=None."
            )
        if kwargs.get("causal_zone_bias") is not None:
            raise ValueError(
                "ModeRoutedContinuousMixtureSpatial does not use causal_zone_bias; "
                "the mode router supersedes the additive zone-pair bias. "
                "Pass causal_zone_bias=None."
            )
        # Pull mode_router out of kwargs; required.
        mode_router = kwargs.pop("mode_router", None)
        if mode_router is None:
            raise ValueError(
                "ModeRoutedContinuousMixtureSpatial requires a `mode_router` instance."
            )
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        if not isinstance(mode_router, ModeRouter):
            raise TypeError(
                f"mode_router must be a ModeRouter instance; got {type(mode_router).__name__}"
            )
        self.mode_router: ModeRouter = mode_router

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
        r"""Per-row mode-routed log-density at ``shot_xy``.

        Parameters are as in the parent ``forward``. The computation is:

        1. Assemble support logits ``E_m`` from the same components as
           the parent forward (defense, matchup, residual).
        2. Compute the mode router :math:`\pi_k(x_n)`, then mask out
           modes with no causal support and renormalize.
        3. For each mode :math:`k`, compute within-mode attention
           :math:`\omega_{m|k}` via subset softmax over ``E_m``
           restricted to ``z(s_m) = k``.
        4. Compute per-mode log-density
           :math:`\log f_k(y) = \mathrm{logsumexp}_m\bigl[\log
           \omega_{m|k} + \log K(y, s_m, \sigma)\bigr]`.
        5. Mix: :math:`\log f(y) = \mathrm{logsumexp}_k\bigl[\log
           \pi_k + \log f_k(y)\bigr]`.

        Returns
        -------
        ContinuousMixtureOutputs
            ``log_lik`` is the per-row log-density. ``mode_log_pi``
            and ``mode_available`` carry the diagnostics. The
            ``support_log_weights`` field reports the marginal
            attention :math:`\pi_{z(s_m)} \omega_{m | z(s_m)}` for
            sampling-side compatibility.
        """
        # ---- 1. Assemble support logits, as in the parent. ----
        collab = self.offensive_prior.forward_continuous(player_idx, snapshot_idx, x_n_raw, x_n)
        logits = collab.support_logits  # (B, M)
        support_xy = collab.support_xy
        support_mask = collab.support_mask

        # σ (per-row or per-shot depending on bandwidth field).
        log_kernel_eff: Tensor | None = None
        if self.anisotropic_kernel is not None:
            log_kernel_eff = self.anisotropic_kernel(support_xy=support_xy, shot_xy=shot_xy)
            sigma_eff: Tensor = collab.sigma
        elif self.bandwidth_field is not None:
            sigma_eff = self.bandwidth_field(support_xy=support_xy, own_mask=collab.own_mask)
        else:
            sigma_eff = collab.sigma

        residual_logits = torch.zeros_like(logits)
        defense_logits: Tensor | None = None
        defense_cold_start: Tensor | None = None
        matchup_logits: Tensor | None = None
        matchup_n_eff: Tensor | None = None

        if self.has_defense:
            if opp_idx is None:
                raise ValueError("defensive_field is wired but opp_idx was not passed.")
            assert self.defensive_field is not None
            assert self._defensive_features is not None
            if isinstance(self.defensive_field, ZoneReweightingDefense):
                feat_buf = self._defensive_features.features.to(opp_idx.device)
                def_features_per_row = feat_buf[opp_idx, snapshot_idx]
                defense_logits = self.defensive_field(
                    query_xy=support_xy,
                    def_features=def_features_per_row,
                )
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
                    query_xy=support_xy,
                    def_xy=gathered["def_xy"],
                    def_mask=gathered["def_mask"],
                    def_age_days=gathered["def_age_days"],
                    def_features=gathered["def_features"],
                    x_n=x_n,
                    h_n=h_n,
                    opp_idx=opp_idx,
                )
                defense_cold_start = ~gathered["def_mask"].any(dim=-1)
            logits = logits + defense_logits

        if self.has_matchup:
            if opp_idx is None:
                raise ValueError("matchup_field is wired but opp_idx was not passed.")
            assert self.matchup_field is not None
            assert self._matchup_features is not None
            delta_buf = self._matchup_features.delta_hat.to(opp_idx.device)
            n_eff_buf = self._matchup_features.n_eff.to(opp_idx.device)
            delta_per_row = delta_buf[player_idx, snapshot_idx, opp_idx]
            matchup_n_eff = n_eff_buf[player_idx, snapshot_idx, opp_idx]
            matchup_logits = self.matchup_field(
                query_xy=support_xy,
                delta_hat=delta_per_row,
            )
            logits = logits + matchup_logits

        if self.has_residual:
            assert self.residual_encoder is not None
            assert self.location_embedding is not None
            h_for_residual: Tensor | None = None
            if self.residual_encoder.within_game_dim > 0:
                if h_n is None:
                    raise ValueError("residual_encoder.within_game_dim > 0 → h_n is required.")
                h_for_residual = h_n
            usage_for_residual: Tensor | None = None
            if self.residual_encoder.usage_dim > 0:
                from shotcloud.features.usage_features import (
                    USAGE_DIM,
                    USAGE_KHAT_DIM,
                    extract_usage,
                )

                usage_dim = self.residual_encoder.usage_dim
                if usage_dim == 1 and self.count_head is not None:
                    mu, _ = self.count_head(x_n)
                    usage_for_residual = self._transform_khat_for_residual(mu).unsqueeze(-1)
                elif usage_dim == USAGE_DIM:
                    usage_for_residual = extract_usage(
                        self.offensive_prior.traits, player_idx, snapshot_idx
                    )
                elif usage_dim == USAGE_KHAT_DIM and self.count_head is not None:
                    base_usage = extract_usage(
                        self.offensive_prior.traits, player_idx, snapshot_idx
                    )
                    mu, _ = self.count_head(x_n)
                    k_hat = self._transform_khat_for_residual(mu).unsqueeze(-1)
                    usage_for_residual = torch.cat([base_usage, k_hat], dim=-1)
                else:
                    raise ValueError(f"unsupported residual usage_dim={usage_dim}")
                assert usage_for_residual.shape[-1] == self.residual_encoder.usage_dim
            outcome_for_residual: Tensor | None = None
            if self.residual_encoder.outcome_dim > 0:
                if o_n is None:
                    raise ValueError("residual_encoder.outcome_dim > 0 but o_n was not provided.")
                outcome_for_residual = o_n
            u = self.residual_encoder(
                x_n, h_for_residual, usage=usage_for_residual, outcome=outcome_for_residual
            )
            if self.has_within_game_gru:
                if prior_seq is None or prior_lengths is None:
                    raise ValueError(
                        "within_game_gru is wired but prior_seq/prior_lengths were not provided."
                    )
                assert self.within_game_gru is not None
                g_r = self.within_game_gru(prior_seq, prior_lengths)
                u = u + g_r
            elif prior_seq is not None or prior_lengths is not None:
                raise ValueError(
                    "prior_seq/prior_lengths supplied but within_game_gru is not wired."
                )
            psi = self.location_embedding(support_xy)  # (B, M, rank)
            residual_logits = (u.unsqueeze(1) * psi).sum(dim=-1)
            logits = logits + residual_logits

        # ---- 2. Mode router + availability renormalization. ----
        cold_start = ~support_mask.any(dim=-1)  # (B,) — no support at all

        z_s = zone_from_xy_torch(support_xy[..., 0], support_xy[..., 1])  # (B, M)
        # Per-mode subset masks: which support shots are valid & in mode k.
        # Stack along last dim → (B, M, K).
        k_range = torch.arange(N_ZONES, device=z_s.device)
        mode_member = (z_s.unsqueeze(-1) == k_range.view(1, 1, -1)) & support_mask.unsqueeze(-1)
        mode_available = mode_member.any(dim=1)  # (B, K)

        # Mode router logits.
        mode_logits_raw = self.mode_router(x_n)  # (B, K)
        # Pre-renormalization log-prob (used to compute "mass lost").
        mode_log_pi_raw = mode_logits_raw - torch.logsumexp(mode_logits_raw, dim=-1, keepdim=True)
        # Mask unavailable modes and renormalize.
        mode_logits_masked = mode_logits_raw.masked_fill(~mode_available, float("-inf"))
        # Cold-start rows (no modes available) would produce -inf logsumexp;
        # patch with one dummy logit so the softmax stays finite. The
        # row's log_lik is overwritten by the cold-start floor below.
        cold_logits = mode_logits_masked.clone()
        if cold_start.any():
            cold_logits[cold_start, 0] = 0.0
        mode_log_pi = cold_logits - torch.logsumexp(cold_logits, dim=-1, keepdim=True)

        # ---- 3+4. Per-mode within-subset softmax + per-mode log-density. ----
        # For each k, compute log_w_k = subset softmax of `logits` over m
        # where mode_member[..., k] is True. Then log_f_k =
        # logsumexp_m(log_w_k + log_K(y, s_m)).
        from shotcloud.training.spatial_losses import continuous_mixture_loglik

        log_f_k_list: list[Tensor] = []
        for k in range(N_ZONES):
            mode_k_mask = mode_member[..., k]  # (B, M)
            # Subset softmax over mode k.
            masked_logits_k = logits.masked_fill(~mode_k_mask, float("-inf"))
            row_empty_k = ~mode_k_mask.any(dim=-1)  # (B,)
            if row_empty_k.any():
                masked_logits_k = masked_logits_k.clone()
                masked_logits_k[row_empty_k, 0] = 0.0  # avoid -inf - -inf = nan
            log_w_k = masked_logits_k - torch.logsumexp(masked_logits_k, dim=-1, keepdim=True)
            # Per-mode log-density. Rows where mode k is empty produce a
            # meaningless value here; it is replaced by -inf below so
            # logsumexp_k drops the contribution.
            if log_kernel_eff is not None:
                log_f_k_b = continuous_mixture_loglik(
                    log_w_k,
                    support_xy,
                    shot_xy,
                    log_kernel=log_kernel_eff,
                    weights_are_log_probs=True,
                    support_mask=mode_k_mask | row_empty_k.unsqueeze(-1),
                )
            else:
                log_f_k_b = continuous_mixture_loglik(
                    log_w_k,
                    support_xy,
                    shot_xy,
                    sigma_eff,
                    weights_are_log_probs=True,
                    support_mask=mode_k_mask | row_empty_k.unsqueeze(-1),
                )
            # Force empty-mode-rows to -inf log_f so logsumexp drops them.
            if row_empty_k.any():
                log_f_k_b = torch.where(
                    row_empty_k, torch.full_like(log_f_k_b, float("-inf")), log_f_k_b
                )
            log_f_k_list.append(log_f_k_b)

        log_f_k = torch.stack(log_f_k_list, dim=-1)  # (B, K)

        # ---- 5. Mode-weighted log-density. ----
        log_lik = torch.logsumexp(mode_log_pi + log_f_k, dim=-1)  # (B,)

        # Cold-start floor, as in the parent.
        if cold_start.any():
            log_lik = torch.where(
                cold_start,
                torch.full_like(log_lik, self.cold_start_log_lik_floor),
                log_lik,
            )

        # ---- 6. Build outputs. ----
        # support_log_weights: marginal attention, used by the samplers.
        # w_m = π_{z(s_m)} · ω_{m | z(s_m)}; for support shots in mode k,
        # log w_m = log π_k + log ω_{m|k}.
        z_s_clamp = z_s.clamp_min(0).long()
        # log π for each support shot's mode.
        log_pi_per_m = mode_log_pi.gather(1, z_s_clamp)  # (B, M)
        # log ω for each support shot: rebuild the per-mode subset softmaxes
        # as a (B, M, K) stack and gather each shot's own-mode entry.
        log_w_per_k = []
        for k in range(N_ZONES):
            mode_k_mask = mode_member[..., k]
            masked_logits_k = logits.masked_fill(~mode_k_mask, float("-inf"))
            row_empty_k = ~mode_k_mask.any(dim=-1)
            if row_empty_k.any():
                masked_logits_k = masked_logits_k.clone()
                masked_logits_k[row_empty_k, 0] = 0.0
            log_w_k = masked_logits_k - torch.logsumexp(masked_logits_k, dim=-1, keepdim=True)
            log_w_per_k.append(log_w_k)
        log_w_stack = torch.stack(log_w_per_k, dim=-1)  # (B, M, K)
        log_omega_per_m = log_w_stack.gather(2, z_s_clamp.unsqueeze(-1)).squeeze(-1)  # (B, M)
        support_log_weights = log_pi_per_m + log_omega_per_m
        # OOB and masked-out support shots get -inf.
        support_log_weights = torch.where(
            support_mask, support_log_weights, torch.full_like(support_log_weights, float("-inf"))
        )

        return ContinuousMixtureOutputs(
            log_lik=log_lik,
            log_weights=support_log_weights,
            support_xy=support_xy,
            sigma=sigma_eff if sigma_eff.dim() == 1 else sigma_eff.mean(dim=-1),
            support_mask=support_mask,
            residual_logits=residual_logits,
            collab=collab,
            cold_start=cold_start,
            support_log_weights=support_log_weights,
            tail_responsibility=None,
            gate_lambda=None,
            defense_logits=defense_logits,
            defense_cold_start=defense_cold_start,
            sigma_per_shot=sigma_eff if sigma_eff.dim() > 1 else None,
            matchup_logits=matchup_logits,
            matchup_n_eff=matchup_n_eff,
            mode_log_pi=mode_log_pi,
            mode_available=mode_available,
            mode_log_pi_raw=mode_log_pi_raw,
        )


__all__ = ["ModeRoutedContinuousMixtureSpatial", "ModeRouter"]
