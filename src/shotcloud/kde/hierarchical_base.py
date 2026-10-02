"""Hierarchical KDE used directly as a spatial base measure.

:class:`HierarchicalKDEBase` exposes the additive Bayesian shrinkage of
:class:`~shotcloud.kde.HierarchicalKDE` as a fixed per-player base
measure,

.. math::

    q_0(c \\mid p) = \\hat q_p^{\\text{hier}}(c)
    = \\alpha_p \\hat q_p(c) + (1 - \\alpha_p) \\hat q_{g(p)}(c),
    \\qquad \\alpha_p = \\frac{N_p}{N_p + \\kappa}.

It shares the public surface of the geometric product
:class:`~shotcloud.legacy.product.KDEProduct` (``grid``, ``density``,
``log_density``, ``hierarchical_kde``), so either can be passed to
:class:`~shotcloud.legacy_pivot.shot_cell_dataset.ShotCellDataset`.
Unlike the product, which combines player, position, and league
densities with fixed weights, shrinkage adapts to each player's sample
size, moving sparse players toward their position-group prior. There is
no separate league term: league-wide structure enters through the
position-group density.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from shotcloud.grids import CourtGrid
from shotcloud.kde.hierarchical import HierarchicalKDE


@dataclass(frozen=True)
class HierarchicalKDEBase:
    """Shrinkage-regularized player density used as a fixed base measure.

    Parameters
    ----------
    hierarchical_kde : HierarchicalKDE
        Source of player + position + league densities. Must already be
        fitted; the shrinkage strength ``κ`` is taken from the fit.
    epsilon : float, default 1e-9
        Density floor (relative to grid). The shrunk player density is
        already strictly positive (HierarchicalKDE floors each component
        at fit time), so this is a redundant safety net for the
        ``log_density`` call.

    Raises
    ------
    ValueError
        If ``epsilon`` is negative or ``hierarchical_kde`` is not fitted.

    Notes
    -----
    The public surface (``grid``, ``density(player_id)``,
    ``log_density(player_id)``, ``hierarchical_kde``) matches
    :class:`~shotcloud.legacy.product.KDEProduct`, so consumers need not
    dispatch on type.
    """

    hierarchical_kde: HierarchicalKDE
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")
        if not self.hierarchical_kde.is_fitted:
            raise ValueError(
                "hierarchical_kde must be fit before being passed to HierarchicalKDEBase"
            )

    @property
    def grid(self) -> CourtGrid:
        """Court grid of the underlying :class:`~shotcloud.kde.HierarchicalKDE`."""
        return self.hierarchical_kde.grid

    def density(self, player_id: object) -> NDArray[np.float64]:
        """Return ``q_0(c | player_id)`` on the grid, shape ``(ny, nx)``.

        Equivalent to :meth:`HierarchicalKDE.player_density` with
        ``hierarchical=True``, followed by an ``epsilon`` floor and
        renormalization for numerical safety.
        """
        q = self.hierarchical_kde.player_density(player_id, hierarchical=True)
        floor = self.epsilon / self.grid.n_cells
        q = np.maximum(q, floor)
        return cast("NDArray[np.float64]", (q / q.sum()).astype(np.float64))

    def log_density(self, player_id: object) -> NDArray[np.float64]:
        """Return ``log q_0(c | player_id)`` on the grid, shape ``(ny, nx)``."""
        return cast("NDArray[np.float64]", np.log(self.density(player_id)).astype(np.float64))
