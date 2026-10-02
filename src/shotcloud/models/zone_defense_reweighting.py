"""Zone-level opponent reweighting of the support logits.

:class:`ZoneReweightingDefense` implements the opponent reweighting term
``D`` of the AC-KDE support logits. Each support shot receives the
additive logit::

    D_m = β_D · γ_{z(s_m)} · q̃_d^{z(s_m)}(t)

where:

* ``z(s_m)``: 8-zone label of the offensive support shot ``s_m``;
* ``q̃_d^z(t)``: league-centered allowed-shot zone proportion of
  opponent ``d`` at snapshot ``t`` (block C of
  :data:`~shotcloud.features.defense_features.DEFENSE_FEATURE_NAMES`),
  computed only from shots dated strictly before the snapshot anchor;
* ``β_D``: a learned scalar, and ``γ_z`` optional learned per-zone
  multipliers (``γ_z ≡ 1`` when disabled).

The mainline configuration enables the per-zone multipliers
(``--defense-kind zone_lite_gamma`` in ``scripts/train_gibbs.py``).

Compared with the kernel field
:class:`~shotcloud.models.continuous_adaptive_defensive.ContinuousAdaptiveDefensiveField`,
this term needs no allowed-shot retrieval cache and no pairwise kernel
evaluation: it reads the
:class:`~shotcloud.features.defense_features.DefenseFeatures` artifact
directly, has at most ``1 + N_ZONES`` parameters, and its forward pass is
``O(B · M)``.

Opponents with no causal history at the snapshot have an all-zero
feature row, so ``D_m = 0`` without additional masking.

:class:`MatchupReweightingDefense` is a player-conditional alternative
built on peer-versus-opponent zone residuals, evaluated as an ablation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from shotcloud.data.zones import N_ZONES
from shotcloud.features.defense_features import _ZONE_CENTERED_SLICE

__all__ = ["MatchupReweightingDefense", "ZoneReweightingDefense", "zone_from_xy_torch"]


def zone_from_xy_torch(xy: Tensor) -> Tensor:
    """Assign 8-zone labels to court coordinates in torch.

    Vectorized equivalent of :func:`shotcloud.data.zones.zone_from_xy`.

    Parameters
    ----------
    xy : Tensor of shape ``(..., 2)``
        Court coordinates in feet, basket at origin.

    Returns
    -------
    Tensor of shape ``(...,)`` int64 with zone index in ``[0, 7]``
    or ``-1`` for out-of-court (backcourt, behind baseline, beyond
    sidelines).
    """
    x = xy[..., 0]
    y = xy[..., 1]
    out_of_court = (y > 47.0) | (y < -5.0) | (x.abs() > 25.0)
    dist = (x * x + y * y).sqrt()
    is_corner_l = (x <= -22.0) & (y <= 7.8)
    is_corner_r = (x >= 22.0) & (y <= 7.8)
    is_arc_3 = (dist >= 23.75) & ~is_corner_l & ~is_corner_r
    is_atb_l = is_arc_3 & (x < -7.5)
    is_atb_r = is_arc_3 & (x > 7.5)
    is_atb_c = is_arc_3 & ~is_atb_l & ~is_atb_r
    is_ra = (~is_corner_l & ~is_corner_r & ~is_arc_3) & (dist < 4.0)
    is_paint = (
        ~is_corner_l
        & ~is_corner_r
        & ~is_arc_3
        & ~is_ra
        & (x.abs() <= 8.0)
        & (y >= 0.0)
        & (y <= 15.0)
    )
    # Priority-order writes so each rule overrides the more general
    # midrange default, matching the NumPy implementation.
    zones = torch.full_like(x, 2, dtype=torch.int64)
    zones = torch.where(is_paint, torch.full_like(zones, 1), zones)
    zones = torch.where(is_ra, torch.full_like(zones, 0), zones)
    zones = torch.where(is_atb_c, torch.full_like(zones, 7), zones)
    zones = torch.where(is_atb_r, torch.full_like(zones, 6), zones)
    zones = torch.where(is_atb_l, torch.full_like(zones, 5), zones)
    zones = torch.where(is_corner_r, torch.full_like(zones, 4), zones)
    zones = torch.where(is_corner_l, torch.full_like(zones, 3), zones)
    zones = torch.where(out_of_court, torch.full_like(zones, -1), zones)
    return zones


class ZoneReweightingDefense(nn.Module):
    """Opponent zone-allowance reweighting of the support logits.

    The logit is ``β_D`` times the opponent's league-centered zone
    allowance at each support shot's zone, with an optional per-zone
    multiplier ``γ_z``:

    * ``per_zone_gamma=False`` (default; CLI ``zone_lite``):
      ``D_m = β_D · q̃_d^{z(s_m)}``, one learnable scalar.
    * ``per_zone_gamma=True`` (CLI ``zone_lite_gamma``, the mainline):
      ``D_m = β_D · γ_{z(s_m)} · q̃_d^{z(s_m)}``, one scalar plus
      ``N_ZONES`` per-zone multipliers. ``γ_z`` is initialized to 1, so
      at initialization this variant equals the scalar-only variant
      exactly.

    Parameters
    ----------
    n_opponents : int
        Size of the opponent vocabulary. Stored so the constructor
        signature matches the kernel-field defense; the forward pass
        does not use it because the caller gathers the per-row
        zone-allowance vector.
    beta_init : float, default 1e-3
        Initial value of ``β_D``. Near zero, so the term starts as an
        approximate no-op; the same default as the kernel-field defense
        keeps the two on comparable scales.
    per_zone_gamma : bool, default False
        If True, add the learnable per-zone multiplier ``γ_z`` (shape
        ``(N_ZONES,)``, initialized to ones). ``β_D`` and ``γ_z`` are
        identified only up to a common scale; no constraint is imposed
        to fix it.
    """

    def __init__(
        self,
        *,
        n_opponents: int,
        beta_init: float = 1e-3,
        per_zone_gamma: bool = False,
    ) -> None:
        super().__init__()
        self._n_opponents = int(n_opponents)
        self._per_zone_gamma = bool(per_zone_gamma)
        self.beta_D = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))
        if self._per_zone_gamma:
            # Initialize to 1 so the term equals the scalar-only variant
            # at initialization.
            self.gamma_z = nn.Parameter(torch.ones(N_ZONES, dtype=torch.float32))
        else:
            # No γ_z parameter — keep the module dict clean.
            self.register_parameter("gamma_z", None)

    @property
    def per_zone_gamma(self) -> bool:
        """Whether the per-zone multipliers ``γ_z`` are enabled."""
        return self._per_zone_gamma

    def forward(
        self,
        *,
        query_xy: Tensor,
        def_features: Tensor,
    ) -> Tensor:
        """Compute the per-support-shot reweighting logit.

        Parameters
        ----------
        query_xy : Tensor of shape ``(B, M, 2)``
            Coordinates of the offensive support shots.
        def_features : Tensor of shape ``(B, DEFENSE_FEATURE_DIM)``
            Per-row defense features (already gathered by
            ``(opp_idx, snapshot_idx)`` upstream). The centered-zone
            block (slice ``_ZONE_CENTERED_SLICE``) is the only block
            used.

        Returns
        -------
        Tensor of shape ``(B, M)``
            Additive support-logit contribution. Out-of-court
            ``query_xy`` rows (zone == -1) get exactly zero; cold-
            start opponent rows get zero implicitly via the
            all-zero ``DefenseFeatures`` cell.
        """
        if query_xy.dim() != 3 or query_xy.shape[-1] != 2:
            raise ValueError(f"query_xy must be (B, M, 2); got {tuple(query_xy.shape)}")
        if def_features.dim() != 2:
            raise ValueError(f"def_features must be (B, D); got {tuple(def_features.shape)}")
        if def_features.shape[0] != query_xy.shape[0]:
            raise ValueError(
                f"batch dim mismatch: query_xy[0]={query_xy.shape[0]}, "
                f"def_features[0]={def_features.shape[0]}"
            )

        centered_zones = def_features[:, _ZONE_CENTERED_SLICE]  # (B, 8)
        zone_idx = zone_from_xy_torch(query_xy)  # (B, M)
        safe_zone = zone_idx.clamp_min(0)
        gathered = (
            centered_zones.unsqueeze(1)
            .expand(-1, safe_zone.shape[-1], -1)
            .gather(2, safe_zone.unsqueeze(-1))
            .squeeze(-1)
        )  # (B, M)
        if self._per_zone_gamma:
            assert self.gamma_z is not None
            gamma_per_shot = self.gamma_z[safe_zone]  # (B, M)
            gathered = gathered * gamma_per_shot
        valid_mask = (zone_idx >= 0).to(gathered.dtype)
        return self.beta_D * gathered * valid_mask


class MatchupReweightingDefense(nn.Module):
    """Player-conditional matchup reweighting of the support logits.

    The logit is a learned scalar ``β_match`` times the per-row, per-zone
    residualized response ``Δ̂_{p,d,z}(t)`` of similar players against
    the opponent. An alternative to :class:`ZoneReweightingDefense`,
    evaluated as an ablation.

    Per-support-shot logit contribution::

        D_m = β_match · Δ̂_{p, d, z(s_m)}(t)

    where:

    * ``z(s_m)``: 8-zone label of the offensive support shot ``s_m``;
    * ``Δ̂_{p, d, z}(t)``: shrunk peer-against-opp zone-residual,
      pre-gathered for the row by
      :class:`shotcloud.features.matchup_features.MatchupFeatures`;
    * ``β_match``: single learned scalar.

    Compared with :class:`ZoneReweightingDefense`:

    * The computation has the same shape (gather by support zone,
      multiply by a learned scalar), but ``Δ̂`` is a player-conditional
      residualized aggregate rather than an opponent-only quantity: it
      describes how the defense affects players similar to this one,
      which the zone-allowance term cannot express.
    * There is no per-zone multiplier ``γ_z``. The per-zone shape comes
      from ``Δ̂`` itself; adding ``γ_z`` would double-count zone
      structure and weaken identification relative to the no-defense
      model.

    Cells with no causal peer-versus-opponent evidence have ``Δ̂ = 0``
    exactly (zeroed by the feature builder), so those rows contribute
    nothing.

    Parameters
    ----------
    beta_init : float, default 1e-3
        Initial value of ``β_match``, matching the ``β_D`` default of
        :class:`ZoneReweightingDefense` so the two start on comparable
        scales.
    """

    def __init__(self, *, beta_init: float = 1e-3) -> None:
        super().__init__()
        self.beta_match = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))

    def forward(
        self,
        *,
        query_xy: Tensor,
        delta_hat: Tensor,
    ) -> Tensor:
        """Compute the per-support-shot matchup logit.

        Parameters
        ----------
        query_xy : Tensor of shape ``(B, M, 2)``
            Coordinates of the offensive support shots.
        delta_hat : Tensor of shape ``(B, N_ZONES)``
            Pre-gathered Δ̂ vector per row — already indexed by
            ``(player_idx, snapshot_idx, opp_idx)`` upstream. Cold-
            start rows are all-zero by construction.

        Returns
        -------
        Tensor of shape ``(B, M)``
            Additive support-logit contribution. Out-of-court
            ``query_xy`` rows (zone == -1) get exactly zero.
        """
        if query_xy.dim() != 3 or query_xy.shape[-1] != 2:
            raise ValueError(f"query_xy must be (B, M, 2); got {tuple(query_xy.shape)}")
        if delta_hat.dim() != 2 or delta_hat.shape[-1] != N_ZONES:
            raise ValueError(
                f"delta_hat must be (B, N_ZONES={N_ZONES}); got {tuple(delta_hat.shape)}"
            )
        if delta_hat.shape[0] != query_xy.shape[0]:
            raise ValueError(
                f"batch dim mismatch: query_xy[0]={query_xy.shape[0]}, "
                f"delta_hat[0]={delta_hat.shape[0]}"
            )

        zone_idx = zone_from_xy_torch(query_xy)  # (B, M)
        safe_zone = zone_idx.clamp_min(0)
        gathered = (
            delta_hat.unsqueeze(1)
            .expand(-1, safe_zone.shape[-1], -1)
            .gather(2, safe_zone.unsqueeze(-1))
            .squeeze(-1)
        )  # (B, M)
        valid_mask = (zone_idx >= 0).to(gathered.dtype)
        return self.beta_match * gathered * valid_mask
