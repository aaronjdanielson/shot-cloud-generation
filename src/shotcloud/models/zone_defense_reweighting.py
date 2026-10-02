"""Zone-based opponent defense reweighting (D-lite) — fast PR after the
D-field allowed-shot KDE landed as empirically non-load-bearing.

Per-support-shot logit contribution::

    D_m = β_D · q̃_d^{z(s_m)}(t)

where:

* ``z(s_m)``: 8-zone label of the offensive support shot ``s_m``;
* ``q̃_d^z(t)``: league-centered allowed-shot zone proportion for
  opponent ``d`` at snapshot ``t`` (block C of
  :data:`~shotcloud.features.defense_features.DEFENSE_FEATURE_NAMES`);
* ``β_D``: single learned scalar.

Compared to
:class:`~shotcloud.models.continuous_adaptive_defensive.ContinuousAdaptiveDefensiveField`
(D-field):

* No allowed-shot retrieval cache, no per-support pairwise KDE —
  defense uses the already-built :class:`DefenseFeatures` artifact
  directly.
* One scalar parameter, not a small MLP.
* Forward is ``O(B · M)`` and orders of magnitude faster.

Cold-start (opponent has no causal history at snapshot) is handled
automatically: the ``DefenseFeatures`` cell for those rows is
all-zero, so ``β_D · 0 = 0``. No additional masking required.

The diagnostic value of D-lite is the same falsification test as
D-field, but with the architecture stripped to its bare minimum:
if ``β_D`` random-walks around zero and cloud metrics do not move,
opponent zone allowance is not load-bearing for the one-shot
retrieval-KDE objective.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from shotcloud.data.zones import N_ZONES
from shotcloud.features.defense_features import _ZONE_CENTERED_SLICE

__all__ = ["MatchupReweightingDefense", "ZoneReweightingDefense", "zone_from_xy_torch"]


def zone_from_xy_torch(xy: Tensor) -> Tensor:
    """Torch-native vectorized version of
    :func:`shotcloud.data.zones.zone_from_xy`.

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
    # midrange default exactly as in the numpy version.
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
    """D-lite: scalar β_D times the opponent's centered zone-allowance
    at each support shot's zone, with optional per-zone γ_z multiplier.

    Two variants:

    * **D-lite-0** (``per_zone_gamma=False``, default):
      ``D_m = β_D · q̃_d^{z(s_m)}``.
      One learnable scalar.
    * **D-lite-zone** (``per_zone_gamma=True``):
      ``D_m = β_D · γ_{z(s_m)} · q̃_d^{z(s_m)}``.
      One scalar + ``N_ZONES`` per-zone multipliers. ``γ_z`` is
      initialized to 1.0, so at init D-lite-zone reduces *exactly*
      to D-lite-0 (the upgrade is a clean superset).

    Parameters
    ----------
    n_opponents : int
        Size of the opponent vocabulary. Stored for save/load
        round-trip parity with the D-field module; the forward does
        not consult it because the per-row zone-allowance vector is
        gathered by the caller.
    beta_init : float, default 1e-3
        Warm-start for the single scalar ``β_D``. Matches the
        D-field convention so paired ablations carry comparable
        global scales.
    per_zone_gamma : bool, default False
        When True, adds the learnable per-zone multiplier ``γ_z``
        (shape ``(N_ZONES,)``, initialized to all-ones). Note that
        ``β_D`` and ``γ_z`` are jointly identified only up to a
        global scale; the trainer's gradient flow disentangles them
        in practice without an explicit constraint.
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
            # Initialize to 1.0 so D-lite-zone == D-lite-0 at step 0.
            self.gamma_z = nn.Parameter(torch.ones(N_ZONES, dtype=torch.float32))
        else:
            # No γ_z parameter — keep the module dict clean.
            self.register_parameter("gamma_z", None)

    @property
    def per_zone_gamma(self) -> bool:
        return self._per_zone_gamma

    def forward(
        self,
        *,
        query_xy: Tensor,
        def_features: Tensor,
    ) -> Tensor:
        """Compute per-support-shot D-lite logit.

        Parameters
        ----------
        query_xy : Tensor of shape ``(B, M, 2)``
            Coordinates of the offensive support shots.
        def_features : Tensor of shape ``(B, DEFENSE_FEATURE_DIM)``
            Per-row defense features (already gathered by
            ``(opp_idx, snapshot_idx)`` upstream). The centered-zone
            block (slice ``_ZONE_CENTERED_SLICE``) is the only block
            consumed by D-lite.

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
    """D-matchup (Tier 2a): scalar β_match times the per-row, per-zone
    residualized similar-player defensive response Δ̂_{p,d,z}(t).

    Per-support-shot logit contribution::

        D_m = β_match · Δ̂_{p, d, z(s_m)}(t)

    where:

    * ``z(s_m)``: 8-zone label of the offensive support shot ``s_m``;
    * ``Δ̂_{p, d, z}(t)``: shrunk peer-against-opp zone-residual,
      pre-gathered for the row by
      :class:`shotcloud.features.matchup_features.MatchupFeatures`;
    * ``β_match``: single learned scalar.

    Compared to :class:`ZoneReweightingDefense` (D-lite-zone):

    * Same compute shape — gather Δ̂ by support zone, multiply by a
      learned scalar — but the per-row Δ̂ vector is itself a
      player-conditional residualized aggregate, not just opponent-
      conditional. This is the "how does this defense affect players
      *like* this player" signal the D-lite zone-allowance term
      cannot express by construction.
    * No ``γ_z`` per-zone multiplier in this first pass. The per-zone
      shape comes from Δ̂ itself; adding γ_z would double-count zone
      structure and complicates identification with the no-defense
      baseline.

    Cold-start handling: cells with no causal peer-vs-opponent
    evidence have ``Δ̂ = 0`` exactly (zeroed in the feature builder),
    so ``β_match · 0 = 0`` and those rows contribute nothing — the
    cold-start-safe fallback.

    Parameters
    ----------
    beta_init : float, default 1e-3
        Warm-start for the single scalar ``β_match``. Matches the
        D-lite ``β_D`` convention so paired ablations carry
        comparable global scales at init.
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
        """Compute per-support-shot D-matchup logit.

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
