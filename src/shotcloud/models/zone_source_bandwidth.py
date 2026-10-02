"""Per-(source, zone) kernel bandwidth for the continuous spatial mixture.

:class:`ZoneSourceBandwidth` assigns each support shot a Gaussian kernel
bandwidth indexed by its **source** (the player's own history or pooled
analogue history) and its **zone** (8-zone court taxonomy). It has
``2 × N_ZONES`` learnable scalars, each sigmoid-bounded into
``[σ_min, σ_max]``::

    σ_m = σ_min + (σ_max − σ_min) · sigmoid(z_{source(m), zone(s_m)})

When attached to
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`
it replaces the single per-row bandwidth of the support backend (e.g.
:class:`~shotcloud.models.retrieval_collaborative_kde.RetrievalCollaborativeKDE`),
so own and pooled support can be smoothed differently, as can zones
with different spatial spread (rim, paint, midrange, corner three,
above-the-break three). The mainline configuration uses this
zone-source bandwidth.

Every raw parameter is initialized to the logit that maps to
``sigma_init``, so at initialization every support shot has bandwidth
``sigma_init`` (up to floating-point rounding) and the mixture matches
the fixed-σ mixture.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from shotcloud.data.zones import N_ZONES, zone_from_xy_vectorized

__all__ = ["ZoneSourceBandwidth", "zone_from_xy_torch_bandwidth"]


def zone_from_xy_torch_bandwidth(xy: Tensor) -> Tensor:
    """Assign 8-zone labels to court coordinates in torch.

    Produces the same labels as :func:`shotcloud.data.zones.zone_from_xy`
    and :func:`shotcloud.data.zones.zone_from_xy_vectorized`.

    Parameters
    ----------
    xy : Tensor of shape ``(..., 2)``
        Court coordinates in feet, basket at origin.

    Returns
    -------
    Tensor of shape ``(...,)``, int64
        Zone index in ``[0, 7]``, or ``-1`` for out-of-court points;
        callers clamp ``-1`` to a valid index before table lookups.
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
    zones = torch.full_like(x, 2, dtype=torch.int64)  # default = midrange
    zones = torch.where(is_paint, torch.full_like(zones, 1), zones)
    zones = torch.where(is_ra, torch.full_like(zones, 0), zones)
    zones = torch.where(is_atb_c, torch.full_like(zones, 7), zones)
    zones = torch.where(is_atb_r, torch.full_like(zones, 6), zones)
    zones = torch.where(is_atb_l, torch.full_like(zones, 5), zones)
    zones = torch.where(is_corner_r, torch.full_like(zones, 4), zones)
    zones = torch.where(is_corner_l, torch.full_like(zones, 3), zones)
    zones = torch.where(out_of_court, torch.full_like(zones, -1), zones)
    return zones


class ZoneSourceBandwidth(nn.Module):
    """Per-(source, zone) bounded scalar bandwidth.

    Parameters
    ----------
    sigma_min, sigma_max : float
        Bounds (in feet) applied via sigmoid, default ``(1.0, 2.5)``.
        The bounds are deliberately tight so that a learned bandwidth
        cannot widen without limit and wash out spatial structure.
    sigma_init : float, default 1.5
        Initial bandwidth in feet. Every (source, zone) entry is
        initialized to the raw logit that maps to ``sigma_init`` under
        the sigmoid, so the module matches a fixed-σ mixture at
        initialization. Must satisfy
        ``sigma_min ≤ sigma_init ≤ sigma_max``.

    Attributes
    ----------
    raw : nn.Parameter
        Shape ``(2, N_ZONES)``. Row 0 = own-source σ logits, row 1 =
        pooled-source σ logits. Read through :meth:`sigma_table` to
        get the bounded σ values.
    """

    OWN: int = 0
    POOLED: int = 1

    # Type annotations for the buffers registered in ``__init__`` —
    # required so mypy sees ``self.sigma_min`` etc. as ``Tensor`` rather
    # than as the ``Module`` attribute fallback.
    sigma_min: Tensor
    sigma_max: Tensor
    sigma_range: Tensor

    def __init__(
        self,
        *,
        sigma_min: float = 1.0,
        sigma_max: float = 2.5,
        sigma_init: float = 1.5,
    ) -> None:
        super().__init__()
        if not (sigma_min < sigma_max):
            raise ValueError(f"sigma_min ({sigma_min}) must be < sigma_max ({sigma_max})")
        if not (sigma_min <= sigma_init <= sigma_max):
            raise ValueError(
                f"sigma_min={sigma_min} <= sigma_init={sigma_init} <= "
                f"sigma_max={sigma_max} required"
            )
        self.register_buffer(
            "sigma_min", torch.tensor(float(sigma_min), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sigma_max", torch.tensor(float(sigma_max), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sigma_range",
            torch.tensor(float(sigma_max) - float(sigma_min), dtype=torch.float32),
            persistent=False,
        )
        # Initialize the raw logit so sigmoid(z) * range + min == sigma_init.
        target = (float(sigma_init) - float(sigma_min)) / (float(sigma_max) - float(sigma_min))
        # Clamp away from {0, 1} for the log to stay finite.
        target = min(max(target, 1e-6), 1.0 - 1e-6)
        init_logit = math.log(target / (1.0 - target))
        self.raw = nn.Parameter(torch.full((2, N_ZONES), float(init_logit), dtype=torch.float32))

    def sigma_table(self) -> Tensor:
        """Bounded σ values, shape ``(2, N_ZONES)``."""
        return self.sigma_min + self.sigma_range * torch.sigmoid(self.raw)

    def forward(
        self,
        *,
        support_xy: Tensor,
        own_mask: Tensor,
    ) -> Tensor:
        """Per-support-shot bandwidth ``(B, M)``.

        Parameters
        ----------
        support_xy : Tensor of shape ``(B, M, 2)``
            Coordinates of each support shot in court feet.
        own_mask : Tensor of shape ``(B, M)``
            Boolean: ``True`` for own-history support, ``False`` for
            pooled. Sourced from
            :attr:`shotcloud.models.collaborative_kde.CollaborativeContinuousOutputs.own_mask`.

        Returns
        -------
        Tensor of shape ``(B, M)``
            ``σ_{source(m), zone(s_m)}`` per support shot.
        """
        if support_xy.dim() != 3 or support_xy.shape[-1] != 2:
            raise ValueError(f"support_xy must be (B, M, 2); got {tuple(support_xy.shape)}")
        if own_mask.shape != support_xy.shape[:2]:
            raise ValueError(
                f"own_mask must match support_xy first two dims; got "
                f"{tuple(own_mask.shape)} vs {tuple(support_xy.shape[:2])}"
            )

        zone_idx = zone_from_xy_torch_bandwidth(support_xy)  # (B, M), int64
        # Out-of-court support shots get zone == -1; clamp to 0 so the
        # lookup is index-safe. Any finite positive σ is acceptable for
        # them.
        safe_zone = zone_idx.clamp_min(0)
        source_idx = torch.where(
            own_mask, torch.full_like(safe_zone, self.OWN), torch.full_like(safe_zone, self.POOLED)
        )

        table = self.sigma_table()  # (2, N_ZONES)
        # Flatten the (2, N_ZONES) lookup to a (2 * N_ZONES,) vector and
        # index by a flat per-shot integer = source * N_ZONES + zone.
        flat = table.view(-1)  # (2 * N_ZONES,)
        flat_idx = source_idx * N_ZONES + safe_zone  # (B, M)
        sigma_per_shot: Tensor = flat[flat_idx]
        return sigma_per_shot


# own_mask comes from CollaborativeContinuousOutputs.own_mask, which both
# the fixed and the retrieval support backends populate, so this module
# works with either.

# zone_from_xy_torch_bandwidth must agree with the NumPy classifier; the
# reference keeps the pairing visible to anyone editing either one.
_NUMPY_ZONE_REFERENCE = zone_from_xy_vectorized
