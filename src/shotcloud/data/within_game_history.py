"""Within-game causal shot-history features ``h_{n,r}``.

The collaborative support of the spatial factor captures the target
player's **long-run** geometry: their own past shots and those of
retrieved analogues. The within-game history :math:`h_{n,r}` instead
summarizes the shots the player has **already taken earlier in the
current game**, and can be fed to the residual-tilt encoder
:class:`~shotcloud.models.context_residual.ContextResidualEncoder`
alongside the context :math:`x_n`. This module builds that summary as a
fixed-dimensional per-shot vector:

============= ====== ============================================
slot          name   description
============= ====== ============================================
0 ``n_prior_log1p``  log1p of prior same-player same-game shots
1 ``frac_3pa``       fraction of prior shots in any 3PA zone
2 ``frac_rim``       fraction of prior shots in {RA, Paint}
3 ``frac_mid``       fraction in Midrange
4 ``frac_corner``    fraction in {Corner3-L, Corner3-R}
5 ``mean_x_recent3`` mean x of last min(3, n_prior) prior shots
6 ``mean_y_recent3`` mean y of last min(3, n_prior) prior shots
7 ``dt_prev_min``    minutes since previous shot (capped at 12)
8 ``shot_density``   ``n_prior / (elapsed_minutes + 1)``
9 ``mask_has_history`` 1 if ``n_prior > 0`` else 0
============= ====== ============================================

For the **first** shot of a (player, game) every slot is zero,
including ``mask_has_history``, so the encoder can tell an empty history
from one whose means and fractions happen to be zero.

:func:`compute_within_game_sequence` provides the same history as a
padded per-shot sequence for
:class:`~shotcloud.models.within_game_gru.WithinGameGRU`, an alternative
encoder of the within-game history evaluated as an ablation.

Both featurizers are causal by construction: for shot ``i`` in a
``(player, game)`` group, only shots strictly before ``i`` in the
group's chronological order contribute. Ties on ``time_remaining_sec``
break by the original DataFrame row order (stable sort).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shotcloud.data.zones import zone_from_xy_vectorized

#: Number of within-game-history feature slots; pass as ``within_game_dim``
#: to :class:`shotcloud.models.context_residual.ContextResidualEncoder`.
WITHIN_GAME_DIM: Final[int] = 10

#: Per-prior-shot feature dimension of :func:`compute_within_game_sequence`.
#: Slots: [x_norm, y_norm, zone_id_norm, distance_norm, dt_min_norm, period_norm].
WITHIN_GAME_SEQ_DIM: Final[int] = 6

#: Maximum number of prior shots retained per shot in the sequence form.
#: Older shots beyond the cap are dropped; player-games with more
#: attempts than this are rare.
MAX_PRIOR_SHOTS: Final[int] = 40

# Normalization constants for the per-prior-shot feature vector, matched
# to the operating range of each raw feature.
_X_NORM: Final[float] = 25.0  # half-court x extent (ft)
_Y_NORM: Final[float] = 47.0  # half-court y extent (ft)
_ZONE_NORM: Final[float] = 8.0  # number of zones — divides zone_id to [0, 1)
_DIST_NORM: Final[float] = 30.0  # rim-distance scale (ft)
_DT_NORM: Final[float] = 12.0  # ``_DT_CAP_MIN``; matches dt_prev_min cap
_PERIOD_NORM: Final[float] = 4.0  # 4 quarters

#: Named slots of the within-game-history feature vector, in order.
WITHIN_GAME_SLOT_NAMES: Final[tuple[str, ...]] = (
    "n_prior_log1p",
    "frac_3pa",
    "frac_rim",
    "frac_mid",
    "frac_corner",
    "mean_x_recent3",
    "mean_y_recent3",
    "dt_prev_min",
    "shot_density",
    "mask_has_history",
)

# Zone-id sets, indexed as in shotcloud.data.zones.ZONE_NAMES.
_RIM_ZONES: Final[set[int]] = {0, 1}  # RA, Paint
_MID_ZONES: Final[set[int]] = {2}  # Midrange
_CORNER_ZONES: Final[set[int]] = {3, 4}  # Corner3-L, Corner3-R
_THREE_PT_ZONES: Final[set[int]] = {3, 4, 5, 6, 7}  # Corner3-L/R, Wing3-L/R, TopKey3

_RECENT_K: Final[int] = 3  # number of "recent" prior shots in the mean.
_DT_CAP_MIN: Final[float] = 12.0  # cap for dt_prev_min so a halftime gap doesn't dominate.


def compute_within_game_features(
    shots_df: pd.DataFrame,
) -> NDArray[np.float32]:
    """Compute the per-shot within-game history features.

    Parameters
    ----------
    shots_df : DataFrame
        Must carry ``x``, ``y``, ``player_id``, ``game_id``,
        ``time_remaining_sec`` (despite the name, **elapsed** seconds
        in the game; see :func:`shotcloud.data.loaders.load_shots`).

    Returns
    -------
    NDArray of shape ``(n_shots, WITHIN_GAME_DIM)`` float32
        Per-shot feature vectors in the input row order. First shots
        in a player-game group get an all-zero vector (including the
        ``mask_has_history`` slot).

    Raises
    ------
    KeyError
        If a required column is missing.
    """
    required = ("x", "y", "player_id", "game_id", "time_remaining_sec")
    for col in required:
        if col not in shots_df.columns:
            raise KeyError(f"within-game featurizer needs column {col!r}")

    n = len(shots_df)
    if n == 0:
        return np.zeros((0, WITHIN_GAME_DIM), dtype=np.float32)

    # Stable sort by (player_id, game_id, time_remaining_sec) so each
    # group is in chronological order; the original row position is kept
    # to scatter the per-group output back at the end.
    work = shots_df.reset_index(drop=True).copy()
    work["_row"] = np.arange(n, dtype=np.int64)
    work_sorted = work.sort_values(
        by=["player_id", "game_id", "time_remaining_sec", "_row"],
        kind="stable",
    )
    x_arr = work_sorted["x"].to_numpy(dtype=np.float64)
    y_arr = work_sorted["y"].to_numpy(dtype=np.float64)
    t_sec = work_sorted["time_remaining_sec"].to_numpy(dtype=np.float64)
    zone = zone_from_xy_vectorized(x_arr, y_arr)
    row = work_sorted["_row"].to_numpy(dtype=np.int64)
    # Zone-membership indicators for the four prefix-summed masks.
    is_rim = np.isin(zone, list(_RIM_ZONES)).astype(np.float64)
    is_mid = np.isin(zone, list(_MID_ZONES)).astype(np.float64)
    is_corner = np.isin(zone, list(_CORNER_ZONES)).astype(np.float64)
    is_three = np.isin(zone, list(_THREE_PT_ZONES)).astype(np.float64)

    out = np.zeros((n, WITHIN_GAME_DIM), dtype=np.float32)

    # Group bounds in the sorted view: start/end (exclusive) indices of
    # each (player_id, game_id) run. Computed once and reused.
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
        if b - a <= 1:
            continue  # only shot in this (player, game) — leave all zero
        # Prefix sums over [a, b). For each within-group position k>0,
        # the "prior" stats are sums over the first k entries (indices
        # a .. a+k-1).
        seg_x = x_arr[a:b]
        seg_y = y_arr[a:b]
        seg_t = t_sec[a:b]
        seg_rim = is_rim[a:b]
        seg_mid = is_mid[a:b]
        seg_corner = is_corner[a:b]
        seg_three = is_three[a:b]
        seg_row = row[a:b]

        cum_rim = np.cumsum(seg_rim)
        cum_mid = np.cumsum(seg_mid)
        cum_corner = np.cumsum(seg_corner)
        cum_three = np.cumsum(seg_three)

        for k in range(1, b - a):
            n_prior = float(k)
            prev_t_sec = float(seg_t[k - 1])
            curr_t_sec = float(seg_t[k])
            dt_min = max(0.0, (curr_t_sec - prev_t_sec) / 60.0)
            dt_min = min(dt_min, _DT_CAP_MIN)
            recent_start = max(0, k - _RECENT_K)
            mean_x_recent = float(seg_x[recent_start:k].mean())
            mean_y_recent = float(seg_y[recent_start:k].mean())

            elapsed_min = curr_t_sec / 60.0
            density = n_prior / (elapsed_min + 1.0)

            r = int(seg_row[k])
            out[r, 0] = float(np.log1p(n_prior))
            out[r, 1] = float(cum_three[k - 1] / n_prior)
            out[r, 2] = float(cum_rim[k - 1] / n_prior)
            out[r, 3] = float(cum_mid[k - 1] / n_prior)
            out[r, 4] = float(cum_corner[k - 1] / n_prior)
            out[r, 5] = mean_x_recent
            out[r, 6] = mean_y_recent
            out[r, 7] = dt_min
            out[r, 8] = density
            out[r, 9] = 1.0

    return out


def compute_within_game_sequence(
    shots_df: pd.DataFrame,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Compute the per-shot causal prior-shot sequence.

    Parameters
    ----------
    shots_df : DataFrame
        Must carry ``x``, ``y``, ``player_id``, ``game_id``,
        ``time_remaining_sec`` (elapsed game seconds).

    Returns
    -------
    prior_seq : NDArray of shape ``(n_shots, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)``
        For each shot ``i``, ``prior_seq[i, :L_i, :]`` holds the feature
        vectors of its ``L_i`` most recent prior shots in the same
        player-game, in chronological order (oldest first). Slots:
        ``[x/_X_NORM, y/_Y_NORM, zone_id/_ZONE_NORM, distance/_DIST_NORM,
        dt_prev_min/_DT_NORM, period/_PERIOD_NORM]``, with distance
        clipped to ``[0, 2]`` after normalization and ``dt_prev_min``
        capped at 12 minutes. Padding rows past ``L_i`` are zero-filled.
    prior_lengths : NDArray of shape ``(n_shots,)`` int64
        Number of valid prior shots ``L_i`` per row, capped at
        :data:`MAX_PRIOR_SHOTS`; the first shot of each player-game gets
        length 0 and an all-zero ``prior_seq`` row.

    Raises
    ------
    KeyError
        If a required column is missing.

    Notes
    -----
    Only shots strictly before shot ``i`` within its ``(player, game)``
    group contribute. The within-group order matches
    :func:`compute_within_game_features`, and both outputs are in input
    row order.
    """
    required = ("x", "y", "player_id", "game_id", "time_remaining_sec")
    for col in required:
        if col not in shots_df.columns:
            raise KeyError(f"within-game sequence featurizer needs column {col!r}")

    n = len(shots_df)
    if n == 0:
        return (
            np.zeros((0, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    work = shots_df.reset_index(drop=True).copy()
    work["_row"] = np.arange(n, dtype=np.int64)
    work_sorted = work.sort_values(
        by=["player_id", "game_id", "time_remaining_sec", "_row"],
        kind="stable",
    )
    x_arr = work_sorted["x"].to_numpy(dtype=np.float64)
    y_arr = work_sorted["y"].to_numpy(dtype=np.float64)
    t_sec = work_sorted["time_remaining_sec"].to_numpy(dtype=np.float64)
    zone = zone_from_xy_vectorized(x_arr, y_arr).astype(np.float64)
    row = work_sorted["_row"].to_numpy(dtype=np.int64)
    # Distance to the rim, and period assuming 12-minute periods.
    dist = np.sqrt(x_arr**2 + y_arr**2)
    period = (t_sec // (12 * 60)).astype(np.float64) + 1.0  # 1..N

    out_seq = np.zeros((n, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM), dtype=np.float32)
    out_len = np.zeros((n,), dtype=np.int64)

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
        seg_n = b - a
        if seg_n <= 1:
            continue
        # Pre-compute the normalized per-shot feature vector for the group.
        # Δt is computed against the previous shot in the same group;
        # the first shot in the group has Δt = 0.
        seg_dt = np.zeros(seg_n, dtype=np.float64)
        seg_dt[1:] = np.clip((t_sec[a + 1 : b] - t_sec[a : b - 1]) / 60.0, 0.0, _DT_NORM)
        seg_feat = np.stack(
            [
                x_arr[a:b] / _X_NORM,
                y_arr[a:b] / _Y_NORM,
                zone[a:b] / _ZONE_NORM,
                np.clip(dist[a:b] / _DIST_NORM, 0.0, 2.0),
                seg_dt / _DT_NORM,
                period[a:b] / _PERIOD_NORM,
            ],
            axis=1,
        ).astype(np.float32)

        for k in range(1, seg_n):
            # Prior = first k entries (oldest first). Truncate to the
            # MAX_PRIOR_SHOTS most recent if k exceeds the cap.
            length = min(k, MAX_PRIOR_SHOTS)
            start = k - length
            r = int(row[a + k])
            out_seq[r, :length, :] = seg_feat[start:k, :]
            out_len[r] = length

    return out_seq, out_len


__all__ = [
    "MAX_PRIOR_SHOTS",
    "WITHIN_GAME_DIM",
    "WITHIN_GAME_SEQ_DIM",
    "WITHIN_GAME_SLOT_NAMES",
    "compute_within_game_features",
    "compute_within_game_sequence",
]
