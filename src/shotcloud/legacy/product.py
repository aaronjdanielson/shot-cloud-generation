"""Geometric weighted product of KDE grids.

Deprecated; retained to reproduce the KDE-product ablation. Superseded by
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`.

Implements the KDE-product base measure ``q_0(c | p, d)``:

.. math::

    q_0(c \\mid p, d) =
    \\frac{
        \\hat q_p(c)^{a_p}\\,
        \\hat q_{g(p)}(c)^{a_g}\\,
        \\hat q_d(c)^{a_d}\\,
        \\hat q_0(c)^{a_0}
    }{
        \\sum_{c'}
        \\hat q_p(c')^{a_p}
        \\hat q_{g(p)}(c')^{a_g}
        \\hat q_d(c')^{a_d}
        \\hat q_0(c')^{a_0}
    },\\qquad a_p, a_g, a_d, a_0 \\ge 0.

This is a generalized product-of-experts density. When the weights sum to
one, it is a geometric mixture; otherwise it remains a valid probability
mass over the grid after normalization.

The implementation covers the player, position, and league terms -- the
three densities exposed by :class:`~shotcloud.kde.hierarchical.HierarchicalKDE`.
The defensive term ``q_d`` is not implemented, so ``a_d`` is rejected as a
weight key.

The combiner runs in log-space and uses :func:`scipy.special.logsumexp`
for the partition function, so under/overflow is not a concern even with
large weight values or sparsely-fitted players.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from shotcloud.grids import CourtGrid
from shotcloud.kde.hierarchical import HierarchicalKDE

REQUIRED_WEIGHT_KEYS: frozenset[str] = frozenset({"a_p", "a_g", "a_0"})


@dataclass(frozen=True)
class KDEProduct:
    """KDE-product base measure on a :class:`~shotcloud.grids.CourtGrid`.

    Parameters
    ----------
    hierarchical_kde : HierarchicalKDE
        Source of player, position, and league density grids. Must be
        already fitted before any density lookup.
    weights : dict[str, float]
        Weights of the three KDE terms. Must contain exactly the keys
        ``"a_p"``, ``"a_g"``, ``"a_0"``. All values must be non-negative.
        They are **not** required to sum to one.
    epsilon : float, default 1e-9
        Density floor (relative to grid). The result is post-floored and
        renormalized so ``q0 > 0`` everywhere — required for log-domain
        operations downstream (the low-rank tilt decoder).
    """

    hierarchical_kde: HierarchicalKDE
    weights: dict[str, float]
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        keys = set(self.weights)
        missing = REQUIRED_WEIGHT_KEYS - keys
        extra = keys - REQUIRED_WEIGHT_KEYS
        if missing:
            raise ValueError(f"missing weight keys: {sorted(missing)}")
        if extra:
            raise ValueError(
                f"unknown weight keys: {sorted(extra)}. "
                "Only {a_p, a_g, a_0} are supported in v1; "
                "'a_d' arrives with DefensiveKDE."
            )
        for k, v in self.weights.items():
            if v < 0:
                raise ValueError(f"weights must be non-negative, got {k}={v}")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")
        if not self.hierarchical_kde.is_fitted:
            raise ValueError("hierarchical_kde must be fit before being passed to KDEProduct")

    # ----------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------

    @property
    def grid(self) -> CourtGrid:
        return self.hierarchical_kde.grid

    # ----------------------------------------------------------------------
    # Density lookup
    # ----------------------------------------------------------------------

    def log_density(self, player_id: object) -> NDArray[np.float64]:
        """Return ``log q_0(c | player_id)`` on the grid, shape ``(ny, nx)``.

        Computed in log-space with stable normalization::

            log_unnorm = a_p log q_p + a_g log q_g + a_0 log q_0
            log_z      = logsumexp(log_unnorm)
            log_q0     = log_unnorm - log_z

        The result is normalized so ``exp(log_q0).sum() == 1``.
        """
        q_p = self.hierarchical_kde.player_density(player_id, hierarchical=False)
        position = self.hierarchical_kde.player_position[str(player_id)]
        q_g = self.hierarchical_kde.position_density(position)
        q_l = self.hierarchical_kde.league_density()

        # Inputs are already ε-floored by HierarchicalKDE, so log is safe.
        log_q_p = np.log(q_p)
        log_q_g = np.log(q_g)
        log_q_l = np.log(q_l)

        log_unnorm = (
            self.weights["a_p"] * log_q_p
            + self.weights["a_g"] * log_q_g
            + self.weights["a_0"] * log_q_l
        )

        log_z = float(logsumexp(log_unnorm))
        return cast("NDArray[np.float64]", (log_unnorm - log_z).astype(np.float64))

    def density(self, player_id: object) -> NDArray[np.float64]:
        """Return ``q_0(c | player_id)`` on the grid, shape ``(ny, nx)``.

        Strictly positive (post-ε-floor) and sums to 1.
        """
        log_q0 = self.log_density(player_id)
        q0 = np.exp(log_q0)

        floor = self.epsilon / self.grid.n_cells
        q0 = np.maximum(q0, floor)
        return cast("NDArray[np.float64]", (q0 / q0.sum()).astype(np.float64))
