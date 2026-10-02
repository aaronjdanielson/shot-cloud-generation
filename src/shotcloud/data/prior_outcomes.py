"""Causal prior-outcome summary ``o_{n,r}`` for the residual tilt.

The within-game history ``h_{n,r}``
(:mod:`shotcloud.data.within_game_history`) summarizes *where* the
player has shot earlier in the current game. This module's
counterpart summarizes *what happened* on those earlier shots (makes,
misses, outcome by zone, recent make rate, recent distance), so the
residual tilt can condition on outcome dynamics as well as location
dynamics. The summary is a fixed-dimensional, non-recurrent vector
consumed by the outcome branch of
:class:`~shotcloud.models.context_residual.ContextResidualEncoder`.

============= ================= =============================================
slot          name              description
============= ================= =============================================
0 ``prior_fga_log1p``           log1p of prior shots in this player-game
1 ``prior_makes_log1p``         log1p of prior shots that were made
2 ``prior_misses_log1p``        log1p of prior shots that were missed
3 ``prior_3pa_log1p``           log1p of prior 3-point attempts
4 ``prior_3pm_log1p``           log1p of prior 3-point makes
5 ``prior_rim_attempts_log1p``  log1p of prior rim attempts (RA + Paint)
6 ``prior_rim_makes_log1p``     log1p of prior rim makes
7 ``recent_make_rate``          fraction of last min(5, n_prior) shots made
8 ``recent_dist_mean``          mean of last min(5, n_prior) shot distances /
                                ``_DIST_NORM`` (35 ft, half-court typical max)
============= ================= =============================================

The seven count slots are ``log1p``-transformed so they share a scale
with the residual encoder's other inputs (``h_{n,r}`` applies log1p to
its count slot for the same reason). ``recent_make_rate`` lies in
[0, 1]; ``recent_dist_mean`` is divided by 35 ft, so it lies in roughly
[0, 1.3].

**Causality.** For shot ``i`` in a ``(player, game)`` group, only
shots strictly before ``i`` in the group's chronological order
contribute; the outcome of shot ``i`` itself never enters its own
features. The first shot of a player-game gets an all-zero vector.
The encoder's outcome branch is zero-initialized, so enabling it
leaves the model unchanged at initialization.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shotcloud.data.zones import zone_from_xy_vectorized

#: Number of prior-outcome summary slots; pass as ``outcome_dim`` to
#: :class:`shotcloud.models.context_residual.ContextResidualEncoder`.
PRIOR_OUTCOME_DIM: Final[int] = 9

#: Distance normalization (feet) for the ``recent_dist_mean`` slot.
#: 35 ft is comfortably above the typical half-court shot distance
#: while keeping the normalized value below ~1.3.
_DIST_NORM: Final[float] = 35.0

#: Recency window for ``recent_make_rate`` and ``recent_dist_mean``.
_RECENT_K: Final[int] = 5

#: Zone sets, indexed as in :data:`shotcloud.data.zones.ZONE_NAMES` and
#: matching :mod:`shotcloud.data.within_game_history`.
_RIM_ZONES: Final[set[int]] = {0, 1}  # RA, Paint
_THREE_PT_ZONES: Final[set[int]] = {3, 4, 5, 6, 7}  # Corner3/Wing3/TopKey3

#: Slot names, in output column order.
PRIOR_OUTCOME_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "prior_fga_log1p",
    "prior_makes_log1p",
    "prior_misses_log1p",
    "prior_3pa_log1p",
    "prior_3pm_log1p",
    "prior_rim_attempts_log1p",
    "prior_rim_makes_log1p",
    "recent_make_rate",
    "recent_dist_mean",
)


def compute_prior_outcome_features(
    shots_df: pd.DataFrame,
) -> NDArray[np.float32]:
    """Compute the per-shot causal prior-outcome summary.

    Shots are ordered within each ``(player_id, game_id)`` group by
    ``time_remaining_sec`` (elapsed game seconds), with ties broken by
    input row order.

    Parameters
    ----------
    shots_df : DataFrame
        Must carry ``x``, ``y``, ``made``, ``player_id``, ``game_id``,
        ``time_remaining_sec``. ``made`` is treated as a 0/1 integer.

    Returns
    -------
    NDArray of shape ``(n_shots, PRIOR_OUTCOME_DIM)`` float32
        Per-shot feature vectors in the input row order. First shots in
        a player-game group get an all-zero vector.

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If ``made`` takes a value other than 0 or 1.
    """
    required = ("x", "y", "made", "player_id", "game_id", "time_remaining_sec")
    for col in required:
        if col not in shots_df.columns:
            raise KeyError(f"prior-outcome featurizer needs column {col!r}")

    n = len(shots_df)
    if n == 0:
        return np.zeros((0, PRIOR_OUTCOME_DIM), dtype=np.float32)

    work = shots_df.reset_index(drop=True).copy()
    work["_row"] = np.arange(n, dtype=np.int64)
    work_sorted = work.sort_values(
        by=["player_id", "game_id", "time_remaining_sec", "_row"],
        kind="stable",
    )
    x_arr = work_sorted["x"].to_numpy(dtype=np.float64)
    y_arr = work_sorted["y"].to_numpy(dtype=np.float64)
    made = work_sorted["made"].to_numpy(dtype=np.float64)
    if not np.all((made == 0.0) | (made == 1.0)):
        raise ValueError("prior-outcome featurizer expects `made` ∈ {0, 1}")
    row = work_sorted["_row"].to_numpy(dtype=np.int64)
    zone = zone_from_xy_vectorized(x_arr, y_arr)
    distance = np.sqrt(x_arr * x_arr + y_arr * y_arr)

    is_rim = np.isin(zone, list(_RIM_ZONES)).astype(np.float64)
    is_three = np.isin(zone, list(_THREE_PT_ZONES)).astype(np.float64)

    out = np.zeros((n, PRIOR_OUTCOME_DIM), dtype=np.float32)

    # Group bounds: contiguous runs of (player_id, game_id) in the
    # sorted view. Same construction as
    # :func:`shotcloud.data.within_game_history.compute_within_game_features`.
    keys = (
        work_sorted["player_id"].astype(str).to_numpy()
        + "\x00"
        + work_sorted["game_id"].astype(str).to_numpy()
    )
    boundaries = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.where(keys[1:] != keys[:-1])[0] + 1,
            np.array([n], dtype=np.int64),
        ]
    )

    for grp_idx in range(len(boundaries) - 1):
        a = int(boundaries[grp_idx])
        b = int(boundaries[grp_idx + 1])
        size = b - a
        if size <= 1:
            continue  # first (and only) shot in this (player, game)

        made_g = made[a:b]
        rim_g = is_rim[a:b]
        three_g = is_three[a:b]
        dist_g = distance[a:b]

        # Exclusive prefix sums over the group's chronological order:
        # ``cs[k]`` is the sum over the ``k`` shots before position ``k``.
        cs_fga = np.arange(size, dtype=np.float64)  # 0, 1, ..., size-1
        cs_makes = np.concatenate([[0.0], np.cumsum(made_g)])[:-1]
        cs_misses = cs_fga - cs_makes
        cs_3pa = np.concatenate([[0.0], np.cumsum(three_g)])[:-1]
        cs_3pm = np.concatenate([[0.0], np.cumsum(three_g * made_g)])[:-1]
        cs_rim_att = np.concatenate([[0.0], np.cumsum(rim_g)])[:-1]
        cs_rim_makes = np.concatenate([[0.0], np.cumsum(rim_g * made_g)])[:-1]

        # Position 0 stays all-zero. ``log1p`` keeps the count slots
        # well-scaled when shot counts reach double digits.
        out_grp_rows = row[a + 1 : b]  # original row indices for positions 1..size-1
        out[out_grp_rows, 0] = np.log1p(cs_fga[1:])
        out[out_grp_rows, 1] = np.log1p(cs_makes[1:])
        out[out_grp_rows, 2] = np.log1p(cs_misses[1:])
        out[out_grp_rows, 3] = np.log1p(cs_3pa[1:])
        out[out_grp_rows, 4] = np.log1p(cs_3pm[1:])
        out[out_grp_rows, 5] = np.log1p(cs_rim_att[1:])
        out[out_grp_rows, 6] = np.log1p(cs_rim_makes[1:])

        # Recency window (last min(_RECENT_K, n_prior) prior shots).
        # For position ``k`` the window is ``[max(0, k-_RECENT_K), k)``.
        for k in range(1, size):
            lo = max(0, k - _RECENT_K)
            window_made = made_g[lo:k]
            window_dist = dist_g[lo:k]
            if window_made.shape[0] > 0:
                out[row[a + k], 7] = float(window_made.mean())
                out[row[a + k], 8] = float(window_dist.mean()) / _DIST_NORM

    return out


__all__ = [
    "PRIOR_OUTCOME_DIM",
    "PRIOR_OUTCOME_FEATURE_NAMES",
    "compute_prior_outcome_features",
]
