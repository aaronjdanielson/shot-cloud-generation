"""Per-opponent KDE of shots **allowed** by each defense.

A team's defensive density ``q̂_def(c | opp)`` is fit on the spatial
distribution of shots taken *against* that team — i.e., the locations
the defense permits attempts. This is a different signal from any
offensive prior:

* Offensive KDEs (player / position / league) describe where players
  *want* to shoot from, given their habits.
* Defensive KDE describes where defenses *let* players shoot from,
  given how they cover the floor.

The two are orthogonal in our model. They compose multiplicatively
inside the softmax via the Phase-2 scalar ``α_def``:

.. math::

    \\ell_{n,i,c}
    = \\tau \\log \\hat q_p^{\\mathrm{hier}}(c)
      + \\alpha_{\\text{def}} \\log \\hat q_{\\text{def}}(c \\mid \\text{opp}_n)
      + u_{n,i}^{\\top} v_c.

Implementation note. We use the same histogram + Gaussian-blur
estimator as :class:`~shotcloud.kde.HierarchicalKDE`, refactored into
``shotcloud.kde._kernel``. The fit interpretation is the only thing
that differs: **the input table is filtered/grouped by opponent, not
by shooter**. A row with ``opponent == "BOS"`` and shot location
``(x, y)`` is interpreted as "Boston's defense permitted a shot at
``(x, y)``" — so the fit accumulates over rows whose opponent column
equals the team being scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from shotcloud.grids import CourtGrid
from shotcloud.kde._kernel import build_recency_weights, fit_density_grid


@dataclass
class DefensiveKDE:
    """Per-opponent KDE of shots allowed.

    Parameters
    ----------
    grid : CourtGrid
    bandwidth : float, default 1.5
        Gaussian bandwidth in feet. Same convention as
        :class:`HierarchicalKDE`.
    recency_half_life_days : float or None, default 365.0
        Exponential-decay half-life for shot recency. ``None`` disables
        recency weighting.
    epsilon : float, default 1e-9
        Density floor. Guarantees ``q > 0`` for log-domain operations.

    Notes
    -----
    Defensive identifiers are stringified at fit and lookup, mirroring
    :class:`HierarchicalKDE`'s convention so callers can pass plain
    Python strings, ``np.str_``, or any hashable type.

    The class deliberately does **not** include shrinkage to a "league
    defense" prior in v1: with 30 teams in the NBA each accumulating
    thousands of shots-allowed per season, the per-opponent fits are
    not data-starved the way the per-player offensive fits are.
    Adding hierarchical shrinkage is a Phase 3+ option.
    """

    grid: CourtGrid
    bandwidth: float = 1.5
    recency_half_life_days: float | None = 365.0
    epsilon: float = 1e-9

    opponent_density_grid: dict[str, NDArray[np.float64]] = field(
        default_factory=dict, init=False, repr=False
    )
    opponent_n: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bandwidth < 0:
            raise ValueError(f"bandwidth must be non-negative (got {self.bandwidth})")
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
        opponent: ArrayLike,
        date: ArrayLike | None = None,
        reference_date: np.datetime64 | None = None,
    ) -> DefensiveKDE:
        """Fit per-opponent KDEs from a flat shot table.

        Each row ``(x_j, y_j, opponent_j)`` is interpreted as "the
        defense ``opponent_j`` allowed a shot at ``(x_j, y_j)``". The
        table is grouped by opponent and one density is fit per group.

        Parameters
        ----------
        x, y : array-like of float
            Shot coordinates in feet, basket at origin.
        opponent : array-like
            Defending-team identifier (string or hashable).
        date : array-like of datetime64, optional
            For recency weighting (required if
            ``recency_half_life_days`` is set).
        reference_date : np.datetime64, optional
            Recency reference. Defaults to ``max(date)``.

        Returns
        -------
        self
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        opp_arr = np.asarray(opponent)

        n = x_arr.shape[0]
        for name, arr in (("y", y_arr), ("opponent", opp_arr)):
            if arr.shape[0] != n:
                raise ValueError(f"length mismatch: x has {n} elements, {name} has {arr.shape[0]}")

        weights = build_recency_weights(
            date, reference_date, n, half_life_days=self.recency_half_life_days
        )

        # Reset any prior fit.
        self.opponent_density_grid = {}
        self.opponent_n = {}

        for opp in np.unique(opp_arr):
            mask = opp_arr == opp
            density, eff_n = fit_density_grid(
                x_arr[mask],
                y_arr[mask],
                weights[mask],
                grid=self.grid,
                bandwidth=self.bandwidth,
                epsilon=self.epsilon,
            )
            self.opponent_density_grid[str(opp)] = density
            self.opponent_n[str(opp)] = eff_n

        return self

    # ----------------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return len(self.opponent_density_grid) > 0

    def density(self, opponent: object) -> NDArray[np.float64]:
        """Normalized defensive density grid, shape ``(ny, nx)``.

        ``opponent`` is coerced to ``str`` to match the keys stored at
        fit time, so ``np.str_``, plain ``str``, or even integer codes
        all work.
        """
        key = str(opponent)
        if key not in self.opponent_density_grid:
            raise KeyError(
                f"unknown opponent {opponent!r}; known: {sorted(self.opponent_density_grid)}"
            )
        return self.opponent_density_grid[key]

    def log_density(self, opponent: object) -> NDArray[np.float64]:
        """``log q_def(c | opponent)`` on the grid, shape ``(ny, nx)``."""
        return np.log(self.density(opponent))

    @property
    def opponents(self) -> tuple[str, ...]:
        """Sorted tuple of opponent IDs the KDE has been fit for."""
        return tuple(sorted(self.opponent_density_grid))
