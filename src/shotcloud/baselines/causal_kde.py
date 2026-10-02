"""Causal grid KDE baseline (paper §5.2 / Figure~4).

Wraps :class:`~shotcloud.kde.HierarchicalKDE` with a thin, evaluation-only
interface so the baseline can be scored against the same metric pipeline
as the AC-KDE checkpoint:

* :meth:`fit` filters shots to ``date <= train_end_date`` (causal by
  construction for any val game whose ``game_date`` is strictly after)
  and fits the underlying player + position + league grids.
* :meth:`density_at_xy` evaluates the per-player hierarchically-shrunk
  density at a set of query coordinates by table lookup on the court
  grid (with bilinear interpolation between cell centers for smoother
  density-surface scoring at coarse smoothing scales).
* :meth:`sample_cloud` produces a synthetic shot cloud of size ``K``
  by drawing cells from the density and uniform coordinates within the
  chosen cell.

The baseline is intentionally minimal: a per-player KDE with shrinkage
to the position prior, no within-game context, no defense, no residual.
That is the point --- it is the simple non-neural reference the AC-KDE
needs to beat to claim improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shotcloud.grids import CourtGrid
from shotcloud.kde.hierarchical import HierarchicalKDE


@dataclass
class CausalGridKDEBaseline:
    """Per-player grid KDE with shrinkage, trained on pre-cutoff shots.

    Parameters
    ----------
    grid : CourtGrid
        Court discretization for the underlying KDE.
    bandwidth : float, default 1.5
        Gaussian bandwidth in feet, matching the AC-KDE mainline default.
    kappa : float, default 500.0
        Shrinkage strength toward the position prior. Players with
        ``N_p`` effective shots get density
        ``α q_p + (1-α) q_position`` with ``α = N_p / (N_p + κ)``.
    recency_half_life_days : float | None, default None
        Optional exponential recency weighting on training shots.
    fallback_to_position : bool, default True
        For player-games whose player_id is unknown at fit time
        (cold-start), substitute the position density when an inferred
        position is supplied at query time; otherwise the league
        density. Off-grid queries always evaluate to the
        league-density floor.
    """

    grid: CourtGrid
    bandwidth: float = 1.5
    kappa: float = 500.0
    recency_half_life_days: float | None = None
    fallback_to_position: bool = True

    kde: HierarchicalKDE = field(init=False)
    fitted: bool = field(init=False, default=False)
    cell_area: float = field(init=False)

    def __post_init__(self) -> None:
        self.kde = HierarchicalKDE(
            grid=self.grid,
            bandwidth=self.bandwidth,
            kappa=self.kappa,
            recency_half_life_days=self.recency_half_life_days,
        )
        self.cell_area = float(self.grid.dx * self.grid.dy)

    def fit(self, shots_df: pd.DataFrame, train_end_date: pd.Timestamp) -> CausalGridKDEBaseline:
        """Fit on shots with ``date <= train_end_date``.

        ``shots_df`` is the standard loader output and must carry
        ``date``, ``x``, ``y``, ``player_id``, and ``position_group``.
        """
        required = ("date", "x", "y", "player_id", "position_group")
        for col in required:
            if col not in shots_df.columns:
                raise KeyError(f"CausalGridKDEBaseline.fit needs column {col!r}")
        cutoff = pd.to_datetime(train_end_date)
        train_mask = pd.to_datetime(shots_df["date"]) <= cutoff
        train = shots_df.loc[train_mask].dropna(subset=["position_group"]).reset_index(drop=True)
        if len(train) == 0:
            raise ValueError("no training shots after applying causal cutoff")
        self.kde.fit(
            x=train["x"].to_numpy(dtype=np.float64),
            y=train["y"].to_numpy(dtype=np.float64),
            player_id=train["player_id"].astype(str).to_numpy(),
            position=train["position_group"].astype(str).to_numpy(),
            date=train["date"].to_numpy(dtype="datetime64[D]"),
            reference_date=np.datetime64(cutoff.to_numpy(), "D"),
        )
        self.fitted = True
        return self

    def _player_density_grid(
        self, player_id: object, position_hint: object | None
    ) -> NDArray[np.float64]:
        """Look up a player's hierarchical density, with cold-start fallback."""
        if not self.fitted:
            raise RuntimeError("CausalGridKDEBaseline is not fitted")
        pid = str(player_id)
        try:
            return self.kde.player_density(pid, hierarchical=True)
        except KeyError:
            # Cold-start: player has no training shots.
            if self.fallback_to_position and position_hint is not None:
                try:
                    return self.kde.position_density(position_hint)
                except KeyError:
                    return self.kde.league_density()
            return self.kde.league_density()

    def density_at_xy(
        self,
        player_id: object,
        xy: NDArray[np.float64],
        *,
        position_hint: object | None = None,
        bilinear: bool = True,
    ) -> NDArray[np.float64]:
        """Predicted density at query coordinates ``xy`` of shape ``(N, 2)``.

        Returns
        -------
        NDArray of shape ``(N,)`` --- density (probability per ft²) at each
        query point. Off-grid queries get the league-density-at-floor value.
        """
        density_grid = self._player_density_grid(player_id, position_hint)
        density = density_grid / self.cell_area  # mass-per-cell → density per ft²
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy must have shape (N, 2), got {tuple(xy.shape)}")
        if bilinear:
            return _bilinear_density(self.grid, density, xy)
        # Nearest-cell lookup.
        ix, iy = self.grid.coord_to_ij(xy[:, 0], xy[:, 1])
        valid = (ix >= 0) & (iy >= 0)
        floor = float(self.kde.league_density().min() / self.cell_area)
        out = np.full(xy.shape[0], floor, dtype=np.float64)
        ix_v = ix[valid]
        iy_v = iy[valid]
        out[valid] = density[iy_v, ix_v]
        return out

    def sample_cloud(
        self,
        player_id: object,
        n_shots: int,
        rng: np.random.Generator,
        *,
        position_hint: object | None = None,
    ) -> NDArray[np.float64]:
        """Draw ``n_shots`` synthetic locations from the player's density.

        The sampler does a categorical draw over cells weighted by the
        density-grid mass, then samples a uniform coordinate inside the
        chosen cell. Shape: ``(n_shots, 2)``.
        """
        if n_shots <= 0:
            return np.zeros((0, 2), dtype=np.float64)
        density_grid = self._player_density_grid(player_id, position_hint)
        nx = density_grid.shape[1]
        flat = density_grid.ravel()
        flat = flat / flat.sum()
        cell_idx = rng.choice(flat.shape[0], size=n_shots, replace=True, p=flat)
        iy = (cell_idx // nx).astype(np.int64)
        ix = (cell_idx % nx).astype(np.int64)
        xedges = self.grid.xedges
        yedges = self.grid.yedges
        u = rng.uniform(0.0, 1.0, size=n_shots)
        v = rng.uniform(0.0, 1.0, size=n_shots)
        x = xedges[ix] + u * (xedges[ix + 1] - xedges[ix])
        y = yedges[iy] + v * (yedges[iy + 1] - yedges[iy])
        return np.stack([x, y], axis=-1)


def _bilinear_density(
    grid: CourtGrid, density: NDArray[np.float64], xy: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Bilinear interpolation of a density grid at query coords.

    Off-grid points fall back to the density floor (the per-cell minimum
    so a query outside the court does not blow up a log-likelihood).
    """
    xc = grid.xcenters
    yc = grid.ycenters
    nx = xc.shape[0]
    ny = yc.shape[0]
    floor = float(density.min())
    out = np.full(xy.shape[0], floor, dtype=np.float64)

    fx = (xy[:, 0] - xc[0]) / (xc[-1] - xc[0]) * (nx - 1)
    fy = (xy[:, 1] - yc[0]) / (yc[-1] - yc[0]) * (ny - 1)
    valid = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
    if not valid.any():
        return out

    fxv = fx[valid]
    fyv = fy[valid]
    ix0 = np.floor(fxv).astype(np.int64).clip(0, nx - 2)
    iy0 = np.floor(fyv).astype(np.int64).clip(0, ny - 2)
    ax = fxv - ix0
    ay = fyv - iy0
    d00 = density[iy0, ix0]
    d10 = density[iy0, ix0 + 1]
    d01 = density[iy0 + 1, ix0]
    d11 = density[iy0 + 1, ix0 + 1]
    interp = (1 - ax) * (1 - ay) * d00 + ax * (1 - ay) * d10 + (1 - ax) * ay * d01 + ax * ay * d11
    out[valid] = cast(NDArray[np.float64], interp)
    return out


__all__ = ["CausalGridKDEBaseline"]
