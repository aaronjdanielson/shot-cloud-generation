"""Hierarchical KDE with sample-size shrinkage and recency weighting.

Fits Gaussian KDEs over a :class:`~shotcloud.grids.CourtGrid` at three
levels — player, position group, league — and provides shrinkage of sparse
players toward their position prior:

.. math::

    \\alpha_p = \\frac{N_p}{N_p + \\kappa},\\qquad
    \\hat q_p^{\\text{hier}}(c)
        = \\alpha_p\\, \\hat q_p(c) + (1 - \\alpha_p)\\, \\hat q_{g(p)}(c).

Optional recency weighting downweights older shots via exponential decay
with a configurable half-life.

Densities are stored on the grid in **image layout** ``(ny, nx)``,
normalized so each grid sums to ``1``, with a small ``epsilon`` floor to
guarantee strict positivity for downstream log-domain operations.

KDE estimation uses the histogram + Gaussian-blur approximation of
:func:`~shotcloud.kde._kernel.fit_density_grid`: for bandwidths that span
multiple grid cells (the typical regime), this is functionally equivalent
to a per-shot Gaussian sum but runs in ``O(N + n_cells * kernel_size)``
instead of ``O(N * n_cells)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from shotcloud.grids import CourtGrid
from shotcloud.kde._kernel import build_recency_weights, fit_density_grid


@dataclass
class HierarchicalKDE:
    """Player / position / league KDE with shrinkage on a fixed court grid.

    Parameters
    ----------
    grid : CourtGrid
        Spatial discretization. Densities are returned on this grid.
    bandwidth : float, default 1.5
        Gaussian bandwidth in **feet**. Internally converted to grid cells.
    kappa : float, default 500.0
        Shrinkage strength. Player density gets weight
        ``α_p = N_p / (N_p + κ)`` and the position prior gets ``1 - α_p``.
    recency_half_life_days : float or None, default 365.0
        Exponential-decay half-life for shot recency in days. Set to
        ``None`` to disable recency weighting (all shots weighted equally).
    epsilon : float, default 1e-9
        Density floor (relative to grid sum). Guarantees ``q > 0`` for
        log-domain operations downstream.
    """

    grid: CourtGrid
    bandwidth: float = 1.5
    kappa: float = 500.0
    recency_half_life_days: float | None = 365.0
    epsilon: float = 1e-9

    league_density_grid: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    league_n: float = field(default=0.0, init=False, repr=False)

    position_density_grid: dict[str, NDArray[np.float64]] = field(
        default_factory=dict, init=False, repr=False
    )
    position_n: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    player_density_grid: dict[str, NDArray[np.float64]] = field(
        default_factory=dict, init=False, repr=False
    )
    player_n: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    player_position: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bandwidth < 0:
            raise ValueError(f"bandwidth must be non-negative (got {self.bandwidth})")
        if self.kappa < 0:
            raise ValueError(f"kappa must be non-negative (got {self.kappa})")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative (got {self.epsilon})")
        if self.recency_half_life_days is not None and self.recency_half_life_days <= 0:
            raise ValueError(
                "recency_half_life_days must be positive or None "
                f"(got {self.recency_half_life_days})"
            )

    # ----------------------------------------------------------------------
    # Fit
    # ----------------------------------------------------------------------

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        player_id: ArrayLike,
        position: ArrayLike,
        date: ArrayLike | None = None,
        reference_date: np.datetime64 | None = None,
    ) -> HierarchicalKDE:
        """Fit league, position, and per-player KDEs from a flat shot table.

        All input arrays must be the same length. Pandas Series work
        transparently.

        Parameters
        ----------
        x, y : array-like of float
            Shot coordinates in feet, basket at origin.
        player_id : array-like
            Player identifier (string or hashable). Used as dict key.
        position : array-like
            Position-group label (string or hashable). Each player must have
            a single, consistent position across their shots.
        date : array-like of datetime64, optional
            Shot date for recency weighting. When omitted, all shots are
            weighted equally regardless of ``recency_half_life_days``.
        reference_date : np.datetime64, optional
            Recency reference. Defaults to ``max(date)``.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the input lengths differ or a player appears under more
            than one position.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        pid_arr = np.asarray(player_id)
        pos_arr = np.asarray(position)

        n = x_arr.shape[0]
        for name, arr in (("y", y_arr), ("player_id", pid_arr), ("position", pos_arr)):
            if arr.shape[0] != n:
                raise ValueError(f"length mismatch: x has {n} elements, {name} has {arr.shape[0]}")

        weights = self._build_weights(date, reference_date, n)

        # Reset any prior fit.
        self.position_density_grid = {}
        self.position_n = {}
        self.player_density_grid = {}
        self.player_n = {}
        self.player_position = {}

        # League KDE — every shot.
        self.league_density_grid, self.league_n = self._fit_kde(x_arr, y_arr, weights)

        # Position KDEs.
        for pos in np.unique(pos_arr):
            mask = pos_arr == pos
            density, eff_n = self._fit_kde(x_arr[mask], y_arr[mask], weights[mask])
            self.position_density_grid[str(pos)] = density
            self.position_n[str(pos)] = eff_n

        # Player KDEs (and remember each player's position).
        for pid in np.unique(pid_arr):
            mask = pid_arr == pid
            player_positions = np.unique(pos_arr[mask])
            if player_positions.size != 1:
                raise ValueError(
                    f"player {pid!r} has multiple positions in the data: "
                    f"{player_positions.tolist()}"
                )
            density, eff_n = self._fit_kde(x_arr[mask], y_arr[mask], weights[mask])
            self.player_density_grid[str(pid)] = density
            self.player_n[str(pid)] = eff_n
            self.player_position[str(pid)] = str(player_positions[0])

        return self

    # ----------------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self.league_density_grid is not None

    def league_density(self) -> NDArray[np.float64]:
        """Normalized league-wide density grid, shape ``(ny, nx)``."""
        if self.league_density_grid is None:
            raise RuntimeError("HierarchicalKDE has not been fit yet")
        return self.league_density_grid

    def position_density(self, position: object) -> NDArray[np.float64]:
        """Normalized position-group density grid, shape ``(ny, nx)``.

        ``position`` is coerced to ``str`` to match the keys stored at fit
        time, so ``np.str_``, plain ``str``, or even integer codes all work.
        """
        key = str(position)
        if key not in self.position_density_grid:
            raise KeyError(
                f"unknown position {position!r}; known: {sorted(self.position_density_grid)}"
            )
        return self.position_density_grid[key]

    def player_density(self, player_id: object, hierarchical: bool = True) -> NDArray[np.float64]:
        """Normalized player density grid, shape ``(ny, nx)``.

        ``player_id`` is coerced to ``str`` so callers can pass whatever
        type the data table provides (``np.int64`` from pandas,
        ``np.str_``, plain ``str``, etc.) — fit and lookup use the same
        key normalization.

        If ``hierarchical`` (default), apply shrinkage toward the player's
        position prior:

            ``q_p^hier = α_p * q_p + (1 - α_p) * q_position``,
            ``α_p = N_p / (N_p + κ)``.

        If ``hierarchical=False``, return the raw player KDE.
        """
        pid = str(player_id)
        if pid not in self.player_density_grid:
            raise KeyError(
                f"unknown player {player_id!r}; {len(self.player_density_grid)} players fit"
            )

        raw = self.player_density_grid[pid]
        if not hierarchical:
            return raw

        position = self.player_position[pid]
        prior = self.position_density_grid[position]
        n = self.player_n[pid]
        alpha = n / (n + self.kappa) if (n + self.kappa) > 0 else 0.0

        blended = alpha * raw + (1.0 - alpha) * prior
        return cast("NDArray[np.float64]", blended / blended.sum())

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _build_weights(
        self,
        date: ArrayLike | None,
        reference_date: np.datetime64 | None,
        n: int,
    ) -> NDArray[np.float64]:
        return build_recency_weights(
            date, reference_date, n, half_life_days=self.recency_half_life_days
        )

    def _fit_kde(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        """Thin wrapper around :func:`fit_density_grid`."""
        return fit_density_grid(
            x, y, weights, grid=self.grid, bandwidth=self.bandwidth, epsilon=self.epsilon
        )
