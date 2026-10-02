"""Court grid for shot-cloud spatial discretization.

Provides coordinate ↔ cell mapping, dequantization, and plotting extent
for an axis-aligned rectangular grid over the basketball half-court.

Coordinate convention
---------------------
Basket at origin ``(0, 0)``. ``x`` positive = right side of the court
(offensive view). ``y`` positive = away from basket toward midcourt. Units
are feet. Default extent ``x ∈ [-25, 25]``, ``y ∈ [-5, 47]`` covers the
full half-court.

Indexing
--------
Two-dimensional grids are stored in **image layout**, shape ``(ny, nx)``,
indexed as ``[iy, ix]`` to match matplotlib's row-major image convention.

Flat (linear) cell indices use C-order ravel of the image layout::

    c = iy * nx + ix

Per-cell arrays, such as densities evaluated on the grid, are indexed by
flat cell index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CourtGrid:
    """Axis-aligned rectangular grid over the basketball half-court.

    Parameters
    ----------
    xlim, ylim : (float, float)
        Lower and upper grid bounds in feet. Default: ``x ∈ [-25, 25]``,
        ``y ∈ [-5, 47]``.
    nx, ny : int
        Number of cells along x and y. Total cells = ``nx * ny``.
    valid_mask : np.ndarray, optional
        Boolean mask of shape ``(ny, nx)`` selecting in-court cells.
        When omitted, :attr:`effective_mask` is all-True.
    """

    xlim: tuple[float, float] = (-25.0, 25.0)
    ylim: tuple[float, float] = (-5.0, 47.0)
    nx: int = 64
    ny: int = 56
    valid_mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(f"nx and ny must be positive (got nx={self.nx}, ny={self.ny})")
        if self.xlim[1] <= self.xlim[0] or self.ylim[1] <= self.ylim[0]:
            raise ValueError(
                f"xlim and ylim must be strictly increasing "
                f"(got xlim={self.xlim}, ylim={self.ylim})"
            )
        if self.valid_mask is not None and self.valid_mask.shape != (self.ny, self.nx):
            raise ValueError(
                f"valid_mask shape {self.valid_mask.shape} != expected (ny, nx) = "
                f"({self.ny}, {self.nx})"
            )

    # -- Sizes -------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Total number of cells, ``nx * ny``."""
        return self.nx * self.ny

    @property
    def dx(self) -> float:
        """Cell width along x, in feet."""
        return (self.xlim[1] - self.xlim[0]) / self.nx

    @property
    def dy(self) -> float:
        """Cell height along y, in feet."""
        return (self.ylim[1] - self.ylim[0]) / self.ny

    # -- Bin edges and cell centers ----------------------------------------

    @property
    def xedges(self) -> NDArray[np.float64]:
        """Bin edges along x, shape ``(nx + 1,)``."""
        return np.linspace(self.xlim[0], self.xlim[1], self.nx + 1)

    @property
    def yedges(self) -> NDArray[np.float64]:
        """Bin edges along y, shape ``(ny + 1,)``."""
        return np.linspace(self.ylim[0], self.ylim[1], self.ny + 1)

    @property
    def xcenters(self) -> NDArray[np.float64]:
        """Cell centers along x, shape ``(nx,)``."""
        return 0.5 * (self.xedges[:-1] + self.xedges[1:])

    @property
    def ycenters(self) -> NDArray[np.float64]:
        """Cell centers along y, shape ``(ny,)``."""
        return 0.5 * (self.yedges[:-1] + self.yedges[1:])

    @property
    def mesh(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(grid_x, grid_y)`` of cell centers, shape ``(ny, nx)`` each."""
        return tuple(np.meshgrid(self.xcenters, self.ycenters))  # type: ignore[return-value]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """matplotlib imshow extent: ``(left, right, bottom, top)``."""
        return (self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1])

    # -- Coordinate ↔ cell conversions -------------------------------------

    def coord_to_ij(
        self,
        x: NDArray[np.floating] | float,
        y: NDArray[np.floating] | float,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Map continuous ``(x, y)`` to integer ``(ix, iy)``.

        Out-of-bounds points return ``(-1, -1)``. Points exactly on the upper
        boundary (``x == xlim[1]`` or ``y == ylim[1]``) map to the last cell
        (``nx - 1`` / ``ny - 1``) rather than being marked OOB.

        Inputs may be scalars or arrays; outputs are always arrays.
        """
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)

        ix = np.floor((xa - self.xlim[0]) / self.dx).astype(np.int64)
        iy = np.floor((ya - self.ylim[0]) / self.dy).astype(np.int64)

        # Clip exact upper-edge points back into the last cell.
        ix = np.where((ix == self.nx) & (xa == self.xlim[1]), self.nx - 1, ix)
        iy = np.where((iy == self.ny) & (ya == self.ylim[1]), self.ny - 1, iy)

        in_bounds = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        ix = np.where(in_bounds, ix, -1)
        iy = np.where(in_bounds, iy, -1)
        return ix, iy

    def coord_to_cell(
        self,
        x: NDArray[np.floating] | float,
        y: NDArray[np.floating] | float,
    ) -> NDArray[np.int64]:
        """Map continuous ``(x, y)`` to flat cell index. Out-of-bounds → ``-1``."""
        ix, iy = self.coord_to_ij(x, y)
        in_bounds = (ix >= 0) & (iy >= 0)
        cell = iy * self.nx + ix
        # numpy stubs lose the dtype through np.where + .astype; cast is explicit.
        return cast("NDArray[np.int64]", np.where(in_bounds, cell, -1).astype(np.int64))

    def ij_to_cell(
        self,
        ix: NDArray[np.integer] | int,
        iy: NDArray[np.integer] | int,
    ) -> NDArray[np.int64]:
        """Map ``(ix, iy)`` to flat cell index. No bounds checking."""
        return cast(
            "NDArray[np.int64]",
            (np.asarray(iy) * self.nx + np.asarray(ix)).astype(np.int64),
        )

    def cell_to_ij(
        self, cell: NDArray[np.integer] | int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Inverse of :meth:`ij_to_cell`. No bounds checking."""
        c = np.asarray(cell, dtype=np.int64)
        iy, ix = np.divmod(c, self.nx)
        return (
            cast("NDArray[np.int64]", ix.astype(np.int64)),
            cast("NDArray[np.int64]", iy.astype(np.int64)),
        )

    def cell_to_coord(
        self, cell: NDArray[np.integer] | int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Cell-center ``(x, y)`` for each flat cell index."""
        ix, iy = self.cell_to_ij(cell)
        x = self.xlim[0] + (ix.astype(np.float64) + 0.5) * self.dx
        y = self.ylim[0] + (iy.astype(np.float64) + 0.5) * self.dy
        return x, y

    def dequantize(
        self,
        cell: NDArray[np.integer] | int,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sample a uniform ``(x, y)`` inside each given cell."""
        ix, iy = self.cell_to_ij(cell)
        u = rng.uniform(0.0, 1.0, size=ix.shape)
        v = rng.uniform(0.0, 1.0, size=iy.shape)
        x = self.xlim[0] + (ix.astype(np.float64) + u) * self.dx
        y = self.ylim[0] + (iy.astype(np.float64) + v) * self.dy
        return x, y

    # -- Court mask --------------------------------------------------------

    @property
    def effective_mask(self) -> NDArray[np.bool_]:
        """``valid_mask`` if set, else an all-True array of shape ``(ny, nx)``."""
        if self.valid_mask is not None:
            return self.valid_mask
        return np.ones((self.ny, self.nx), dtype=bool)
