"""Tier-1a adaptive bandwidth: per-(source, zone) scalar σ_{src,z}.

The simplest identifiable variant of the bandwidth axis (bandwidth
analog of D-lite-0 in the defense work): ``2 × N_ZONES`` learnable
scalars, sigmoid-bounded into ``[σ_min, σ_max]``, indexed per support
shot by its **source** (own vs pooled history) and **zone** (8-zone
NBA taxonomy).

Per-shot bandwidth::

    σ_m = σ_min + (σ_max − σ_min) · sigmoid(z_{source(m), zone(s_m)})

Replaces the existing single per-row σ from
:class:`~shotcloud.models.retrieval_collaborative_kde.RetrievalCollaborativeKDE`
when the bandwidth field is wired into
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`.

Tests the predictive hypothesis: own-history support and pooled
support need different smoothing (own probably sharper, pooled
smoother), and rim/paint/midrange/corner/above-the-break zones have
different natural spatial uncertainty.

At ``sigma_init = 1.5`` with the canonical bounds, all 16 raw
parameters initialize to the same logit such that
``σ_{src,z} ≡ 1.5`` everywhere — so the wrapper is **bit-identical** to
the fixed-σ=1.5 path at step 0, and the choice of using a per-shot
bandwidth instead of the per-row σ only matters once the parameters
move during training.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from shotcloud.data.zones import N_ZONES, zone_from_xy_vectorized

__all__ = ["ZoneSourceBandwidth", "zone_from_xy_torch_bandwidth"]


def zone_from_xy_torch_bandwidth(xy: Tensor) -> Tensor:
    """Torch-native vectorized zone assignment.

    Mirrors :func:`shotcloud.viz.energy_body_overlay.zone_from_xy_torch`
    / the numpy :func:`shotcloud.data.zones.zone_from_xy` exactly, so
    a sweep of court coordinates produces the same labels under all
    three. Returns int64 of shape ``(...,)`` with ``-1`` for
    out-of-court points (clipped to a safe zone before indexing by the
    caller).
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
        Bounds (in feet) applied via sigmoid. Conservative defaults
        (1.0, 2.5) chosen so unconstrained learned bandwidth cannot
        widen pathologically the way shot_flow's learned σ did.
    sigma_init : float
        Initial bandwidth in feet. Every (source, zone) entry is
        initialized to the raw logit that maps to ``sigma_init`` under
        the sigmoid, so the module is **bit-identical** to a fixed-σ
        wrapper at step 0. ``sigma_init`` must satisfy
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
        # Identical inversion pattern to RetrievalCollaborativeKDE.
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
        # Out-of-court support shots get zone == -1 — index-safe clamp to 0.
        # Such shots are masked out of the mixture anyway, so any σ value
        # for them is acceptable as long as it stays finite and positive.
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


# Note for downstream readers: own_mask is sourced from
# CollaborativeContinuousOutputs.own_mask (populated by both the L×R
# fixed and retrieval support backends), so this module works with
# either offensive prior. Verified by zone_from_xy_torch_bandwidth
# parity against shotcloud.data.zones.zone_from_xy_vectorized.

# The numpy parity is asserted in tests; keep an explicit reference in
# the file so refactors that drift the implementations are caught.
_NUMPY_ZONE_REFERENCE = zone_from_xy_vectorized
