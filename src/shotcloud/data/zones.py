"""NBA shot-zone taxonomy and grid masks.

Eight-zone categorization used for evaluation (zone KL, zone retrieval) and
for visualization. Adapted from
[shot_flow/src/shot_flow/data/zones.py](/Users/aarondanielson/Dropbox/shot_flow/src/shot_flow/data/zones.py)
and rewired to consume :class:`~shotcloud.grids.CourtGrid` instead of the
hardcoded 64×46 grid.

Zone indices
------------
0  Restricted Area (RA)        — distance < 4 ft from basket
1  Paint (non-RA)              — key rectangle, outside RA
2  Midrange                    — inside 3PT arc, outside key
3  Corner 3 — Left             — x ≤ -22 ft, y ≤ 7.8 ft
4  Corner 3 — Right            — x ≥  22 ft, y ≤ 7.8 ft
5  Wing 3 — Left               — above-break arc 3, x < -7.5 ft
6  Wing 3 — Right              — above-break arc 3, x >  7.5 ft
7  Top of Key 3                — above-break arc 3, center
-1 Backcourt / unclassified    — excluded from training

Coordinate convention: feet, basket at origin.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray

from shotcloud.grids import CourtGrid

N_ZONES: Final[int] = 8
ZONE_NAMES: Final[tuple[str, ...]] = (
    "RA",
    "Paint",
    "Midrange",
    "Corner3-L",
    "Corner3-R",
    "Wing3-L",
    "Wing3-R",
    "TopKey3",
)

_BASIC_TO_ZONE: Final[dict[str, int]] = {
    "Restricted Area": 0,
    "In The Paint (Non-RA)": 1,
    "Mid-Range": 2,
    "Left Corner 3": 3,
    "Right Corner 3": 4,
    "Backcourt": -1,
}

_ABOVE_BREAK_AREA: Final[dict[str, int]] = {
    "Left Side(L)": 5,
    "Left Side Center(LC)": 5,
    "Center(C)": 7,
    "Right Side Center(RC)": 6,
    "Right Side(R)": 6,
}


def zone_from_strings(shot_zone_basic: str, shot_zone_area: str) -> int:
    """Map NBA zone-label strings to a zone index in ``[0, 7]`` or ``-1``.

    NBA Stats API columns are ``SHOT_ZONE_BASIC`` and ``SHOT_ZONE_AREA``.
    """
    if shot_zone_basic == "Above the Break 3":
        return _ABOVE_BREAK_AREA.get(shot_zone_area, 7)
    return _BASIC_TO_ZONE.get(shot_zone_basic, -1)


def zone_from_xy(x: float, y: float) -> int:
    """Map a continuous shot location ``(x, y)`` in feet to a zone index.

    Returns ``-1`` for points outside the offensive half-court area
    (e.g., backcourt heaves with ``y > 47``, behind the baseline with
    ``y < -5``, or beyond the sidelines with ``|x| > 25``).

    Note: shot_flow's original ``zone_from_xy`` had unreachable ``return -1``
    code; this version adds explicit out-of-bounds checks at the top so
    backcourt shots are correctly flagged.
    """
    # Out of the offensive half-court — backcourt or out-of-bounds.
    if y > 47.0 or y < -5.0 or abs(x) > 25.0:
        return -1

    dist = (x * x + y * y) ** 0.5

    is_corner_l = (x <= -22.0) and (y <= 7.8)
    is_corner_r = (x >= 22.0) and (y <= 7.8)
    is_arc_3 = (dist >= 23.75) and not is_corner_l and not is_corner_r

    if is_corner_l:
        return 3
    if is_corner_r:
        return 4
    if is_arc_3:
        if x < -7.5:
            return 5
        if x > 7.5:
            return 6
        return 7

    if dist < 4.0:
        return 0
    if abs(x) <= 8.0 and 0.0 <= y <= 15.0:
        return 1
    return 2  # midrange: everything inside the arc, not RA or paint


def zone_from_xy_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Torch-vectorized version of :func:`zone_from_xy` for batched ops.

    Same zone taxonomy as :func:`zone_from_xy`; ports the scalar
    inequalities to torch so the result can flow through a forward
    pass without numpy round-trip. ``x`` and ``y`` must have matching
    shape. Integer inputs are promoted to ``float32``; floating
    inputs keep their dtype (MPS does not support ``float64``).

    Returns an ``int64`` tensor of the same shape carrying zone
    labels in ``[0, N_ZONES)`` or ``-1`` for out-of-bounds (backcourt
    / out-of-court).
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must share shape; got {tuple(x.shape)} vs {tuple(y.shape)}")
    xf = x if x.is_floating_point() else x.float()
    yf = y if y.is_floating_point() else y.float()
    out = torch.full(xf.shape, -1, dtype=torch.int64, device=xf.device)

    # Out-of-bounds first.
    oob = (yf > 47.0) | (yf < -5.0) | (xf.abs() > 25.0)
    in_court = ~oob
    if not in_court.any():
        return out

    dist = torch.sqrt(xf * xf + yf * yf)
    is_corner_l = (xf <= -22.0) & (yf <= 7.8) & in_court
    is_corner_r = (xf >= 22.0) & (yf <= 7.8) & in_court
    is_arc_3 = (dist >= 23.75) & ~is_corner_l & ~is_corner_r & in_court

    out = torch.where(is_corner_l, torch.full_like(out, 3), out)
    out = torch.where(is_corner_r, torch.full_like(out, 4), out)
    # Above-the-break 3: left / right / center by x.
    wing_l = is_arc_3 & (xf < -7.5)
    wing_r = is_arc_3 & (xf > 7.5)
    top_key = is_arc_3 & ~wing_l & ~wing_r
    out = torch.where(wing_l, torch.full_like(out, 5), out)
    out = torch.where(wing_r, torch.full_like(out, 6), out)
    out = torch.where(top_key, torch.full_like(out, 7), out)

    # Inside-arc: RA / paint / midrange.
    inside_arc = in_court & ~is_corner_l & ~is_corner_r & ~is_arc_3
    is_ra = inside_arc & (dist < 4.0)
    is_paint = inside_arc & ~is_ra & (xf.abs() <= 8.0) & (yf >= 0.0) & (yf <= 15.0)
    is_midrange = inside_arc & ~is_ra & ~is_paint
    out = torch.where(is_ra, torch.full_like(out, 0), out)
    out = torch.where(is_paint, torch.full_like(out, 1), out)
    out = torch.where(is_midrange, torch.full_like(out, 2), out)
    return out


def zone_from_xy_vectorized(x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.int64]:
    """Vectorized version of :func:`zone_from_xy` for arrays of locations."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    out = np.full(x_arr.shape, -1, dtype=np.int64)
    for i in range(out.size):
        out.flat[i] = zone_from_xy(float(x_arr.flat[i]), float(y_arr.flat[i]))
    return out


def zone_cell_mask(grid: CourtGrid) -> NDArray[np.int64]:
    """Return a ``(grid.ny, grid.nx)`` int array assigning each cell to a zone.

    Each cell is labeled by the zone of its center. Cells whose center
    falls outside any defined zone are labeled ``-1``.
    """
    grid_x, grid_y = grid.mesh
    mask = np.full((grid.ny, grid.nx), -1, dtype=np.int64)
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            mask[iy, ix] = zone_from_xy(float(grid_x[iy, ix]), float(grid_y[iy, ix]))
    return mask


def zone_cell_masks_per_zone(grid: CourtGrid) -> NDArray[np.bool_]:
    """Return ``(N_ZONES, grid.ny, grid.nx)`` boolean masks, one per zone."""
    cell_zones = zone_cell_mask(grid)
    masks = np.zeros((N_ZONES, grid.ny, grid.nx), dtype=bool)
    for z in range(N_ZONES):
        masks[z] = cell_zones == z
    return masks
