"""Tests for :class:`shotcloud.grids.court.CourtGrid`."""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud import CourtGrid

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction_matches_canonical_extent() -> None:
    g = CourtGrid()
    assert g.xlim == (-25.0, 25.0)
    assert g.ylim == (-5.0, 47.0)
    assert g.nx == 64
    assert g.ny == 56
    assert g.n_cells == 64 * 56
    assert g.dx == pytest.approx(50.0 / 64)
    assert g.dy == pytest.approx(52.0 / 56)


def test_zero_or_negative_dimensions_raise() -> None:
    with pytest.raises(ValueError, match="positive"):
        CourtGrid(nx=0)
    with pytest.raises(ValueError, match="positive"):
        CourtGrid(ny=-1)


def test_inverted_limits_raise() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CourtGrid(xlim=(25.0, -25.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        CourtGrid(ylim=(10.0, 10.0))


def test_grid_is_frozen() -> None:
    g = CourtGrid()
    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError is enough
        g.nx = 100  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Edges, centers, mesh, extent
# ---------------------------------------------------------------------------


def test_edges_and_centers_have_correct_shapes_and_values() -> None:
    g = CourtGrid(xlim=(0.0, 10.0), ylim=(0.0, 5.0), nx=5, ny=5)
    assert g.xedges.shape == (g.nx + 1,)
    assert g.yedges.shape == (g.ny + 1,)
    assert g.xcenters.shape == (g.nx,)
    assert g.ycenters.shape == (g.ny,)
    np.testing.assert_allclose(g.xedges, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    np.testing.assert_allclose(g.xcenters, [1.0, 3.0, 5.0, 7.0, 9.0])
    np.testing.assert_allclose(g.yedges, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(g.ycenters, [0.5, 1.5, 2.5, 3.5, 4.5])


def test_mesh_shape_is_image_layout() -> None:
    g = CourtGrid(nx=10, ny=8)
    grid_x, grid_y = g.mesh
    assert grid_x.shape == (g.ny, g.nx)
    assert grid_y.shape == (g.ny, g.nx)
    # First row of grid_x is xcenters; first col of grid_y is ycenters.
    np.testing.assert_allclose(grid_x[0, :], g.xcenters)
    np.testing.assert_allclose(grid_y[:, 0], g.ycenters)


def test_extent_is_matplotlib_convention() -> None:
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=10)
    assert g.extent == (-25.0, 25.0, -5.0, 47.0)


# ---------------------------------------------------------------------------
# Coordinate ↔ cell conversions
# ---------------------------------------------------------------------------


def test_coord_to_cell_returns_int64_array() -> None:
    g = CourtGrid(nx=10, ny=10, xlim=(0.0, 10.0), ylim=(0.0, 10.0))
    cells = g.coord_to_cell([1.0, 2.0], [3.0, 4.0])
    assert cells.dtype == np.int64
    assert cells.shape == (2,)


def test_coord_to_cell_round_trip_recovers_original_cell() -> None:
    """Sample inside cells, map back, recover same cell index."""
    g = CourtGrid(xlim=(-10.0, 10.0), ylim=(-10.0, 10.0), nx=20, ny=20)
    rng = np.random.default_rng(0)
    x = rng.uniform(-9.99, 9.99, 100)
    y = rng.uniform(-9.99, 9.99, 100)
    cells = g.coord_to_cell(x, y)
    assert (cells >= 0).all()
    cx, cy = g.cell_to_coord(cells)
    # Cell centers should be within half-cell of the original points.
    assert np.all(np.abs(x - cx) <= g.dx / 2 + 1e-9)
    assert np.all(np.abs(y - cy) <= g.dy / 2 + 1e-9)


def test_coord_to_cell_marks_out_of_bounds_as_minus_one() -> None:
    g = CourtGrid(xlim=(0.0, 10.0), ylim=(0.0, 10.0), nx=10, ny=10)
    cells = g.coord_to_cell([-1.0, 5.0, 15.0], [5.0, 5.0, 5.0])
    assert cells[0] == -1
    assert cells[1] >= 0
    assert cells[2] == -1
    cells_y = g.coord_to_cell([5.0, 5.0], [-1.0, 11.0])
    assert (cells_y == -1).all()


def test_coord_to_cell_includes_upper_boundary() -> None:
    """Points exactly on the upper edge map to the last cell, not -1."""
    g = CourtGrid(xlim=(0.0, 10.0), ylim=(0.0, 10.0), nx=10, ny=10)
    cells = g.coord_to_cell([10.0], [10.0])
    assert cells[0] == g.n_cells - 1


def test_coord_to_cell_lower_boundary_is_in_first_cell() -> None:
    """Points exactly on the lower edge map to cell 0."""
    g = CourtGrid(xlim=(0.0, 10.0), ylim=(0.0, 10.0), nx=10, ny=10)
    cells = g.coord_to_cell([0.0], [0.0])
    assert cells[0] == 0


def test_ij_to_cell_and_cell_to_ij_are_inverses() -> None:
    g = CourtGrid(nx=10, ny=8)
    for ix in range(g.nx):
        for iy in range(g.ny):
            cell = int(g.ij_to_cell(ix, iy))
            assert 0 <= cell < g.n_cells
            ix2, iy2 = g.cell_to_ij(cell)
            assert int(ix2) == ix
            assert int(iy2) == iy


def test_ij_to_cell_uses_image_layout() -> None:
    """Flat index = iy * nx + ix (image-layout C-order ravel)."""
    g = CourtGrid(nx=10, ny=8)
    assert int(g.ij_to_cell(0, 0)) == 0
    assert int(g.ij_to_cell(9, 0)) == 9  # last x in first row
    assert int(g.ij_to_cell(0, 1)) == g.nx  # first x in second row
    assert int(g.ij_to_cell(9, 7)) == g.n_cells - 1


# ---------------------------------------------------------------------------
# Dequantization
# ---------------------------------------------------------------------------


def test_dequantize_lands_inside_originating_cell() -> None:
    """Dequantized samples must round-trip back to the same cell index."""
    g = CourtGrid(xlim=(-10.0, 10.0), ylim=(-10.0, 10.0), nx=20, ny=20)
    rng = np.random.default_rng(0)
    cells = np.arange(g.n_cells)
    x, y = g.dequantize(cells, rng)
    cells2 = g.coord_to_cell(x, y)
    np.testing.assert_array_equal(cells2, cells)


def test_dequantize_is_deterministic_under_seeded_rng() -> None:
    g = CourtGrid(xlim=(-10.0, 10.0), ylim=(-10.0, 10.0), nx=20, ny=20)
    cells = np.arange(g.n_cells)
    x1, y1 = g.dequantize(cells, np.random.default_rng(7))
    x2, y2 = g.dequantize(cells, np.random.default_rng(7))
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------


def test_effective_mask_default_is_all_true() -> None:
    g = CourtGrid()
    mask = g.effective_mask
    assert mask.shape == (g.ny, g.nx)
    assert mask.all()


def test_explicit_valid_mask_round_trips() -> None:
    custom = np.zeros((8, 10), dtype=bool)
    custom[2:6, 3:7] = True
    g = CourtGrid(nx=10, ny=8, valid_mask=custom)
    np.testing.assert_array_equal(g.effective_mask, custom)


def test_valid_mask_with_wrong_shape_raises() -> None:
    bad = np.ones((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="valid_mask shape"):
        CourtGrid(nx=10, ny=10, valid_mask=bad)
