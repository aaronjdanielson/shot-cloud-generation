"""Causal prior-outcome summary ``o_{n,r}`` for the residual tilt.

The within-game-history module ``h_{n,r}``
(:mod:`shotcloud.data.within_game_history`) summarizes *where* the
player has shot earlier in the current game. This module's
counterpart vector summarizes *what happened* on those earlier shots
— makes, misses, zone-by-outcome breakdowns, recent make rate, recent
distance — so the residual decoder can condition on outcome dynamics
rather than only location dynamics.

Paper §5.7 reported that a one-layer causal GRU over prior shot
locations did not improve density-surface or finite-cloud metrics
beyond the existing ``h_{n,r}`` summary. The 2026-06-07 Phase 1
findings established the count factor's pretrain-and-freeze protocol;
the natural next conditioning is on prior outcomes rather than only
prior locations. Following the user's plan, this module starts
**non-recurrently**: a small fixed-dimensional vector that the
residual encoder consumes alongside ``x_n``, ``h_{n,r}``, the causal
usage state, and the count-supervised latent score. A GRU over the
outcome summaries is reserved for future work and only justified by
positive evidence from this simpler baseline.

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

All eight count slots are ``log1p``'d so they live on a similar scale
to the residual encoder's other inputs (``h_{n,r}`` uses log1p for
its first slot for the same reason). The two recency slots are
normalized: ``recent_make_rate`` already lives in [0, 1];
``recent_dist_mean`` is divided by 35 ft so it lives in roughly
[0, 1.3].

**First-shot causal edge case.** For a shot that is the first shot
of its (player, game), all features default to zero. The residual
encoder's outcome branch is zero-initialized (matching the
existing usage-branch pattern), so an all-zero ``o_{n,r}`` produces
zero outcome-contribution at step 0 — preserving the AC-KDE's
zero-init invariant.

The featurizer is causal by construction: for shot ``i`` in a
``(player, game)`` group, only shots strictly before ``i`` in the
group's chronological order contribute.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shotcloud.data.zones import zone_from_xy_vectorized

#: Number of prior-outcome summary feature slots. Keep this
#: synchronized with the constructor + tests for
#: :class:`shotcloud.models.context_residual.ContextResidualEncoder`'s
#: ``outcome_dim`` parameter.
PRIOR_OUTCOME_DIM: Final[int] = 9

#: Distance normalization (feet) for the ``recent_dist_mean`` slot.
#: 35 ft is comfortably above the typical half-court shot distance
#: while keeping the normalized value below ~1.3.
_DIST_NORM: Final[float] = 35.0

#: Recency window for ``recent_make_rate`` and ``recent_dist_mean``.
_RECENT_K: Final[int] = 5

#: Zone-set constants, mirroring
#: :mod:`shotcloud.data.within_game_history`. Kept in sync with
#: :data:`shotcloud.data.zones.ZONE_NAMES`.
_RIM_ZONES: Final[set[int]] = {0, 1}  # RA, Paint
_THREE_PT_ZONES: Final[set[int]] = {3, 4, 5, 6, 7}  # Corner3/Wing3/TopKey3

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
    """Per-shot causal prior-outcome summary in the input row order.

    Parameters
    ----------
    shots_df : DataFrame
        Must carry ``x``, ``y``, ``made``, ``player_id``, ``game_id``,
        ``time_remaining_sec``. ``made`` is treated as a 0/1 integer.

    Returns
    -------
    NDArray of shape ``(n_shots, PRIOR_OUTCOME_DIM)`` float32
        Per-shot feature vector aligned to ``shots_df.index`` order.
        First shots in a player-game group get an all-zero vector.
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

        # Cumulative sums (prefix sums) over the group's chronological
        # order. ``cs[k]`` = sum of the first ``k`` values. The number
        # of prior shots for position ``k`` (0-indexed) is ``k`` and
        # the prefix sum we want is ``cs[k]``.
        cs_fga = np.arange(size, dtype=np.float64)  # 0, 1, ..., size-1
        cs_makes = np.concatenate([[0.0], np.cumsum(made_g)])[:-1]
        cs_misses = cs_fga - cs_makes
        cs_3pa = np.concatenate([[0.0], np.cumsum(three_g)])[:-1]
        cs_3pm = np.concatenate([[0.0], np.cumsum(three_g * made_g)])[:-1]
        cs_rim_att = np.concatenate([[0.0], np.cumsum(rim_g)])[:-1]
        cs_rim_makes = np.concatenate([[0.0], np.cumsum(rim_g * made_g)])[:-1]

        # Position-local outputs (skip position 0 — leave all-zero).
        # Use ``log1p`` for the count slots so the residual sees a
        # well-scaled feature even when shot counts run into double
        # digits in a high-volume game.
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
