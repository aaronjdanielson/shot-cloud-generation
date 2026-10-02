"""Tests for :mod:`shotcloud.data.zones`."""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud import CourtGrid
from shotcloud.data.zones import (
    N_ZONES,
    ZONE_NAMES,
    zone_cell_mask,
    zone_cell_masks_per_zone,
    zone_from_strings,
    zone_from_xy,
    zone_from_xy_vectorized,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_n_zones_and_names_consistent() -> None:
    assert N_ZONES == 8
    assert len(ZONE_NAMES) == N_ZONES


# ---------------------------------------------------------------------------
# zone_from_xy
# ---------------------------------------------------------------------------


def test_basket_is_restricted_area() -> None:
    assert zone_from_xy(0.0, 0.0) == 0


def test_short_paint_is_paint() -> None:
    assert zone_from_xy(2.0, 8.0) == 1


def test_midrange_at_18_feet() -> None:
    assert zone_from_xy(0.0, 18.0) == 2


def test_corners_are_corner_threes() -> None:
    assert zone_from_xy(-23.0, 5.0) == 3  # left corner
    assert zone_from_xy(23.0, 5.0) == 4  # right corner


def test_above_break_three_split_by_x() -> None:
    """Above-break arc 3 (dist >= 23.75 ft) splits into Wing-L / Wing-R / Top."""
    # Use the canonical 3PT distance with explicit angles.
    r = 24.0
    assert zone_from_xy(-r * np.cos(np.deg2rad(45)), r * np.sin(np.deg2rad(45))) == 5
    assert zone_from_xy(r * np.cos(np.deg2rad(45)), r * np.sin(np.deg2rad(45))) == 6
    assert zone_from_xy(0.0, r) == 7


def test_returns_int_not_array() -> None:
    z = zone_from_xy(0.0, 5.0)
    assert isinstance(z, int)


# ---------------------------------------------------------------------------
# zone_from_strings (NBA Stats labels)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "basic, area, expected",
    [
        ("Restricted Area", "", 0),
        ("In The Paint (Non-RA)", "Center(C)", 1),
        ("Mid-Range", "Right Side(R)", 2),
        ("Left Corner 3", "Left Side(L)", 3),
        ("Right Corner 3", "Right Side(R)", 4),
        ("Above the Break 3", "Left Side(L)", 5),
        ("Above the Break 3", "Right Side(R)", 6),
        ("Above the Break 3", "Center(C)", 7),
        ("Backcourt", "", -1),
        ("UnknownZone", "", -1),
    ],
)
def test_zone_from_strings(basic: str, area: str, expected: int) -> None:
    assert zone_from_strings(basic, area) == expected


# ---------------------------------------------------------------------------
# Vectorized
# ---------------------------------------------------------------------------


def test_zone_from_xy_vectorized_matches_scalar() -> None:
    xs = np.array([0.0, 2.0, 18.0, -23.0, 23.0, 0.0])
    ys = np.array([0.0, 8.0, 0.0, 5.0, 5.0, 26.0])
    expected = np.array([zone_from_xy(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])
    np.testing.assert_array_equal(zone_from_xy_vectorized(xs, ys), expected)


def test_zone_from_xy_vectorized_dtype_and_shape() -> None:
    out = zone_from_xy_vectorized(np.array([0.0, 5.0]), np.array([0.0, 18.0]))
    assert out.dtype == np.int64
    assert out.shape == (2,)


# ---------------------------------------------------------------------------
# CourtGrid integration
# ---------------------------------------------------------------------------


def test_zone_cell_mask_shape_and_values() -> None:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=20)
    mask = zone_cell_mask(grid)
    assert mask.shape == (grid.ny, grid.nx)
    assert mask.dtype == np.int64
    # All values must be in {-1, 0, 1, ..., 7}.
    unique = set(mask.flatten().tolist())
    assert unique.issubset(set(range(-1, N_ZONES)))


def test_zone_cell_masks_per_zone_partition_property() -> None:
    """Each cell belongs to at most one zone (or to none, marked -1)."""
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=15, ny=15)
    masks = zone_cell_masks_per_zone(grid)
    assert masks.shape == (N_ZONES, grid.ny, grid.nx)
    assert masks.dtype == bool
    # No cell appears in two zones.
    assert (masks.sum(axis=0) <= 1).all()


def test_zone_cell_mask_basket_cell_is_RA() -> None:
    """The cell containing (0, 0) should be Restricted Area (zone 0)."""
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=20)
    mask = zone_cell_mask(grid)
    ix_basket, iy_basket = grid.coord_to_ij(0.0, 0.0)
    assert mask[int(iy_basket), int(ix_basket)] == 0
