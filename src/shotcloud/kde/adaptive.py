"""Per-player shot histories for context-adaptive KDE.

:class:`AdaptiveKDE` stores, for each player, a ragged history of past
shots -- flat cell index, court coordinates, canonical context vector,
and date -- together with a shared grid kernel matrix. These histories
are the own-player support from which
:class:`~shotcloud.models.collaborative_kde.CollaborativeKDE` builds its
support pool; consumers apply a causal date mask against snapshot
anchors so that only shots strictly before the target game contribute.

A context-adaptive density reweights a player's historical shots by
their relevance to the current context:

.. math::

    \\pi_{\\phi,j}(x_n) = \\mathrm{softmax}_j[f_\\phi(z_j, x_n)],
    \\qquad
    \\hat q_\\phi(c \\mid p, x_n) = \\sum_{j \\in \\mathcal H_p}
        \\pi_{\\phi,j}(x_n) \\cdot K_h(c - s_j),

where :math:`z_j` is the context of historical shot :math:`j` and
:math:`f_\\phi` is a learned relevance score such as
:class:`~shotcloud.models.RelevanceScore`. The kernel is fixed; the
relevance weights decide which shots contribute.

On the grid, each historical shot lies in a single cell :math:`k_j`, so
the density factors as one matrix product,

.. math::

    \\hat q_\\phi(\\cdot \\mid p, x_n) = M \\cdot \\pi_\\phi(x_n; Z[p]),

with ``M`` the ``(n_cells, n_cells)`` kernel matrix from
:func:`~shotcloud.kde._kernel.build_kernel_matrix`. No per-shot kernel
grid is stored.

Each player's stored history is capped at ``max_history`` shots at fit
time, which keeps batched relevance and support computation tractable;
a few hundred shots already capture a high-volume player's spatial
distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from shotcloud.data.context import CONTEXT_DIM, ContextEncoder
from shotcloud.data.zones import zone_from_xy_vectorized
from shotcloud.grids import CourtGrid
from shotcloud.kde._kernel import build_kernel_matrix


def _stratified_pick(
    zones: NDArray[np.int64], n_pick: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Pick ``n_pick`` local indices stratified by zone.

    Returns local indices into the input ``zones`` array. Allocations
    are roughly proportional to per-zone counts, with each zone that
    exists in the pool guaranteed at least one slot (to keep rare
    zones represented). Capped to per-zone available counts so we
    never request more than exists.

    Deterministic under ``rng``.
    """
    n_zones = len(zones)
    if n_pick >= n_zones:
        return np.arange(n_zones, dtype=np.int64)

    unique_zones, zone_counts = np.unique(zones, return_counts=True)
    total = int(zone_counts.sum())
    # Proportional allocation with min-1 floor (so any zone present in
    # the pool gets at least one shot represented), capped at zone size.
    raw_alloc = n_pick * zone_counts / total
    allocations = np.maximum(1, np.round(raw_alloc).astype(np.int64))
    allocations = np.minimum(allocations, zone_counts)

    # Reconcile to exactly n_pick. Add/remove one at a time from the
    # zone with most spare capacity / largest allocation respectively.
    diff = n_pick - int(allocations.sum())
    while diff > 0:
        spare = zone_counts - allocations
        if int(spare.max()) <= 0:
            break
        idx = int(np.argmax(spare))
        allocations[idx] += 1
        diff -= 1
    while diff < 0:
        removable = np.where(allocations > 1, allocations, -1)
        if int(removable.max()) <= 0:
            # Every allocation is at the floor of 1; can't reduce
            # further without dropping a zone entirely. Drop the zone
            # with the smallest count.
            drop_idx = int(np.argmin(np.where(allocations > 0, zone_counts, np.inf)))
            allocations[drop_idx] = 0
            diff += 1
        else:
            idx = int(np.argmax(removable))
            allocations[idx] -= 1
            diff += 1

    sampled_local: list[int] = []
    for z, alloc in zip(unique_zones, allocations, strict=False):
        if alloc <= 0:
            continue
        z_local = np.flatnonzero(zones == z)
        if alloc >= len(z_local):
            sampled_local.extend(int(i) for i in z_local)
        else:
            sampled_local.extend(int(i) for i in rng.choice(z_local, int(alloc), replace=False))

    return np.array(sampled_local, dtype=np.int64)


@dataclass
class AdaptiveKDE:
    """Per-player ragged shot histories plus a shared grid kernel matrix.

    Parameters
    ----------
    grid : CourtGrid
        Spatial discretization.
    bandwidth : float, default 1.5
        Gaussian bandwidth in feet used to build ``M`` via
        :func:`~shotcloud.kde._kernel.build_kernel_matrix`.
    epsilon : float, default 1e-12
        Floor on the kernel matrix entries, relative to the grid.
    max_history : int or None, default 500
        Cap on the number of stored shots per player, enforced at fit
        time according to ``history_policy``. ``None`` keeps every shot.
    seed : int, default 0
        Seed for the subsampling RNG.
    history_policy : {"random", "recent_stratified"}, default "random"
        How a player's history is subsampled when it exceeds
        ``max_history``. ``"random"`` draws a uniform random subsample.
        ``"recent_stratified"`` keeps the ``recent_history`` most recent
        shots and fills the remaining ``stratified_history`` slots with a
        court-zone-stratified sample of the older shots, preserving
        recency while keeping rare zones represented.
    recent_history : int, default 350
        Number of most-recent shots kept under ``"recent_stratified"``.
    stratified_history : int, default 150
        Number of zone-stratified older shots kept under
        ``"recent_stratified"``. ``recent_history + stratified_history``
        must equal ``max_history`` under that policy.

    Attributes
    ----------
    M : NDArray[float64], shape ``(n_cells, n_cells)``
        Kernel matrix; each column is the grid density of a point mass at
        one cell. Available after :meth:`fit`.
    cells : dict[str, NDArray[int64]]
        Per-player flat cell indices, shape ``(N_p,)``.
    context : dict[str, NDArray[float32]]
        Per-player canonical context features, shape
        ``(N_p, CONTEXT_DIM)``.
    coords : dict[str, NDArray[float32]]
        Per-player shot coordinates ``(x, y)`` in feet, shape
        ``(N_p, 2)``. Continuous-kernel consumers evaluate kernels at
        these locations rather than at cell centers.
    dates : dict[str, NDArray[int64]]
        Per-player shot dates as epoch days, shape ``(N_p,)``. Populated
        only when :meth:`fit` receives ``date``.
    n_history : dict[str, int]
        Stored history size ``N_p`` per player after capping.

    Notes
    -----
    Player IDs are stored as ``str(player_id)``; lookups stringify their
    argument the same way. All per-player arrays share one row order.
    """

    grid: CourtGrid
    bandwidth: float = 1.5
    epsilon: float = 1e-12
    max_history: int | None = 500
    seed: int = 0
    history_policy: str = "random"
    recent_history: int = 350
    stratified_history: int = 150

    M: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    cells: dict[str, NDArray[np.int64]] = field(default_factory=dict, init=False, repr=False)
    context: dict[str, NDArray[np.float32]] = field(default_factory=dict, init=False, repr=False)
    n_history: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    dates: dict[str, NDArray[np.int64]] = field(default_factory=dict, init=False, repr=False)
    coords: dict[str, NDArray[np.float32]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bandwidth < 0:
            raise ValueError(f"bandwidth must be non-negative, got {self.bandwidth}")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")
        if self.max_history is not None and self.max_history <= 0:
            raise ValueError(f"max_history must be positive or None, got {self.max_history}")
        if self.history_policy not in ("random", "recent_stratified"):
            raise ValueError(
                f"history_policy must be 'random' or 'recent_stratified', "
                f"got {self.history_policy!r}"
            )
        if self.history_policy == "recent_stratified":
            if self.recent_history < 0 or self.stratified_history < 0:
                raise ValueError("recent_history and stratified_history must be non-negative")
            if self.max_history is None:
                raise ValueError("recent_stratified policy requires max_history to be set")
            if self.recent_history + self.stratified_history != self.max_history:
                raise ValueError(
                    f"recent_history ({self.recent_history}) + "
                    f"stratified_history ({self.stratified_history}) must equal "
                    f"max_history ({self.max_history}) under recent_stratified"
                )

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has run and stored at least one player."""
        return self.M is not None and len(self.cells) > 0

    @property
    def context_dim(self) -> int:
        """Dimension of the stored context vectors (``CONTEXT_DIM``)."""
        return CONTEXT_DIM

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        player_id: ArrayLike,
        context_features: NDArray[np.float32],
        context_encoder: ContextEncoder | None = None,
        date: ArrayLike | None = None,
    ) -> AdaptiveKDE:
        """Build per-player histories and the shared kernel matrix.

        Parameters
        ----------
        x, y : array-like of float
            Shot coordinates in feet, basket at origin.
        player_id : array-like
            Per-shot player identifier (string or hashable).
        context_features : ndarray, shape ``(N, CONTEXT_DIM)``
            Canonical context vector for each shot, typically from
            :meth:`~shotcloud.data.ContextEncoder.transform` on the same
            shot frame.
        context_encoder : ContextEncoder, optional
            Not used by the fit; accepted so callers can pass the encoder
            that produced ``context_features``.
        date : array-like of datetime64 or convertible, optional
            Per-shot date. When provided, stored as int64 epoch days in
            :attr:`dates`, parallel to :attr:`cells`. Required by
            consumers that apply a causal date mask against snapshot
            anchors, such as
            :class:`~shotcloud.models.collaborative_kde.CollaborativeKDE`.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the input lengths differ or ``context_features`` does not
            have shape ``(N, CONTEXT_DIM)``.

        Notes
        -----
        Shots outside the grid (e.g. backcourt heaves) are dropped. Any
        previous fit is discarded.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        pid_arr = np.asarray(player_id)
        ctx_arr = np.asarray(context_features, dtype=np.float32)

        n = x_arr.shape[0]
        if y_arr.shape[0] != n:
            raise ValueError(f"length mismatch: x has {n}, y has {y_arr.shape[0]}")
        if pid_arr.shape[0] != n:
            raise ValueError(f"length mismatch: x has {n}, player_id has {pid_arr.shape[0]}")
        if ctx_arr.shape != (n, CONTEXT_DIM):
            raise ValueError(
                f"context_features shape {ctx_arr.shape} does not match ({n}, {CONTEXT_DIM})"
            )
        date_arr: NDArray[np.int64] | None = None
        if date is not None:
            date_raw = np.asarray(date)
            if date_raw.shape[0] != n:
                raise ValueError(f"length mismatch: x has {n}, date has {date_raw.shape[0]}")
            # Coerce to int64 epoch days; works for datetime64, pandas
            # Timestamp arrays, and ISO strings.
            date_arr = np.asarray(date_raw, dtype="datetime64[D]").astype(np.int64)

        # Compute flat cell indices once.
        cells_all = self.grid.coord_to_cell(x_arr, y_arr).astype(np.int64)
        valid = cells_all >= 0  # drop backcourt
        cells_all = cells_all[valid]
        pid_arr = pid_arr[valid]
        ctx_arr = ctx_arr[valid]
        # Per-shot (x, y) coordinates parallel to cells/context.
        coords_all = np.stack([x_arr[valid], y_arr[valid]], axis=1).astype(np.float32)
        if date_arr is not None:
            date_arr = date_arr[valid]

        # Reset.
        self.cells = {}
        self.context = {}
        self.n_history = {}
        self.dates = {}
        self.coords = {}

        rng = np.random.default_rng(self.seed)
        for pid in np.unique(pid_arr):
            mask = pid_arr == pid
            p_cells = cells_all[mask]
            p_ctx = ctx_arr[mask]
            p_coords = coords_all[mask]
            p_dates = date_arr[mask] if date_arr is not None else None
            n_p = p_cells.size

            if self.max_history is not None and n_p > self.max_history:
                idx = self._select_history_idx(n_p, p_dates, p_coords, rng)
                p_cells = p_cells[idx]
                p_ctx = p_ctx[idx]
                p_coords = p_coords[idx]
                if p_dates is not None:
                    p_dates = p_dates[idx]
                n_p = idx.size

            key = str(pid)
            self.cells[key] = p_cells.astype(np.int64)
            self.context[key] = p_ctx.astype(np.float32)
            self.coords[key] = p_coords.astype(np.float32)
            self.n_history[key] = int(n_p)
            if p_dates is not None:
                self.dates[key] = p_dates.astype(np.int64)

        # Build shared kernel matrix once.
        self.M = build_kernel_matrix(self.grid, self.bandwidth, self.epsilon)
        return self

    def _select_history_idx(
        self,
        n_p: int,
        p_dates: NDArray[np.int64] | None,
        p_coords: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        """Return the indices to keep from a player's history under ``history_policy``.

        Assumes ``n_p > self.max_history``; the caller handles the
        no-subsample case.
        """
        assert self.max_history is not None and n_p > self.max_history

        if self.history_policy == "random":
            return rng.choice(n_p, size=self.max_history, replace=False).astype(np.int64)

        # recent_stratified: keep the most-recent ``recent_history`` shots
        # by date, then stratified-sample ``stratified_history`` shots
        # from the older pool by court zone. Preserves recency while
        # maintaining geometric coverage of older shots.
        if p_dates is None:
            # Recency is undefined without dates; fall back to a uniform
            # random subsample.
            return rng.choice(n_p, size=self.max_history, replace=False).astype(np.int64)

        order = np.argsort(p_dates, kind="stable")
        n_recent = min(self.recent_history, n_p)
        recent_idx = order[-n_recent:]
        n_strat_target = self.max_history - n_recent
        if n_strat_target <= 0:
            return recent_idx.astype(np.int64)

        older_idx = order[:-n_recent]
        if older_idx.size <= n_strat_target:
            # Older pool is too small to stratified-sample from; take all.
            return np.concatenate([recent_idx, older_idx]).astype(np.int64)

        older_coords = p_coords[older_idx]
        older_zones = zone_from_xy_vectorized(older_coords[:, 0], older_coords[:, 1])
        strat_local_idx = _stratified_pick(older_zones, n_strat_target, rng)
        strat_idx = older_idx[strat_local_idx]
        return np.concatenate([recent_idx, strat_idx]).astype(np.int64)

    @property
    def players(self) -> tuple[str, ...]:
        """Sorted tuple of player IDs the KDE has been fit for."""
        return tuple(sorted(self.cells))

    def player_history(self, player_id: object) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Return ``(cells, context)`` arrays for the player.

        ``cells`` has shape ``(N_p,)`` and ``context`` has shape
        ``(N_p, CONTEXT_DIM)``. The stored arrays are returned without
        copying, so callers must not mutate them.
        """
        key = str(player_id)
        if key not in self.cells:
            raise KeyError(f"unknown player {player_id!r}")
        return self.cells[key], self.context[key]

    def density_with_uniform_relevance(self, player_id: object) -> NDArray[np.float64]:
        """Adaptive density with uniform relevance weights.

        With ``π_j = 1/N_p`` for all ``j``, the matrix product reduces to
        the mean column of ``M`` over the player's history cells, i.e. the
        fixed-bandwidth Gaussian KDE of the stored history. Useful as a
        reference against which learned relevance weights are compared.

        Returns
        -------
        NDArray[float64], shape ``(n_cells,)``
            Flat-cell density summing to 1.
        """
        if self.M is None:
            raise RuntimeError("AdaptiveKDE.fit must be called first")
        cells = self.cells[str(player_id)]
        # M[:, cells] has shape (n_cells, N_p); average over the columns.
        density = self.M[:, cells].mean(axis=1)
        # Should already be normalized (each column of M sums to 1), but
        # be defensive.
        return cast("NDArray[np.float64]", (density / density.sum()).astype(np.float64))
