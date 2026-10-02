"""Causal spatial-Hawkes feature — prior-shot KDE evaluated at zone centroids.

Phase 1 B1 of the Phase-0 follow-up (2026-06-09 log entry). The
diagnostic D1 showed that within-game shot-zone transitions have a
signed, lag-structured profile: at short lag the same-zone diagonal
is positively excited (Hawkes-style self-excitation), at long lag it
is negatively suppressed (cross-quarter adaptation). This module
exposes the causal *spatial* memory of within-game shots as an
8-dim feature evaluated at the eight zone centroids:

.. math::

    \\phi_{n,r}^{\\mathrm{self}}(c)
    = \\frac{1}{r-1} \\sum_{r'<r}
      \\mathcal N(c;\\, y_{n,r'},\\, \\sigma_h^2 I).

Slot ``c`` is the canonical centroid of zone ``c`` (one of the eight
shot-zones defined in :mod:`shotcloud.data.zones`):

============= ================ ===================
slot          name             centroid (x, y) ft
============= ================ ===================
0             RA (restricted)  (0.0, 1.5)
1             Paint            (0.0, 7.5)
2             Midrange         (0.0, 15.0)
3             Corner 3 — L     (-22.5, 5.0)
4             Corner 3 — R     ( 22.5, 5.0)
5             Wing 3 — L       (-17.0, 22.0)
6             Wing 3 — R       ( 17.0, 22.0)
7             Top of Key 3     (0.0, 25.0)
============= ================ ===================

The kernel bandwidth :math:`\\sigma_h` defaults to 4 ft — close to
the source/zone-σ field's typical own-bandwidth, large enough to
smooth across the shot-cluster scale of D1's positive same-zone
diagonal, small enough not to leak across the corner/wing or
rim/paint boundaries.

First-shot causal edge case: for the first shot of a (player, game),
all eight slots are zero by construction; the model's residual
encoder zero-init absorbs this without breaking the AC-KDE step-0
invariant.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PRIOR_SHOT_KDE_DIM: Final[int] = 8

#: Canonical zone centroids in feet (basket at origin, x positive = right,
#: y positive = away from basket). Eight rows aligned with the
#: :data:`shotcloud.data.zones.ZONE_NAMES` ordering.
ZONE_CENTROIDS_FT: Final[NDArray[np.float32]] = np.array(
    [
        (0.0, 1.5),  # 0 RA
        (0.0, 7.5),  # 1 Paint
        (0.0, 15.0),  # 2 Midrange
        (-22.5, 5.0),  # 3 Corner 3 — L
        (22.5, 5.0),  # 4 Corner 3 — R
        (-17.0, 22.0),  # 5 Wing 3 — L
        (17.0, 22.0),  # 6 Wing 3 — R
        (0.0, 25.0),  # 7 Top of Key 3
    ],
    dtype=np.float32,
)

#: Default kernel bandwidth in feet. Chosen to be close to the source/zone-σ
#: field's typical own-bandwidth — smooths across the shot-cluster scale
#: without leaking across major zone boundaries.
DEFAULT_SIGMA_H_FT: Final[float] = 4.0


def compute_prior_shot_kde_features(
    shots_df: pd.DataFrame,
    sigma_h_ft: float = DEFAULT_SIGMA_H_FT,
) -> NDArray[np.float32]:
    """Per-shot causal spatial-Hawkes summary in the input row order.

    For each shot, the 8-dim feature is the causal Gaussian-KDE
    summary of *prior* in-game shots evaluated at the eight zone
    centroids. The first shot of each (player, game) gets the
    all-zero feature.

    Parameters
    ----------
    shots_df : DataFrame
        Must carry ``x``, ``y``, ``player_id``, ``game_id``,
        ``time_remaining_sec`` (the loader's misnamed "time elapsed
        in game" field).
    sigma_h_ft : float, default 4 ft
        Kernel bandwidth at zone-centroid evaluation. Must be > 0.

    Returns
    -------
    NDArray of shape ``(n_shots, PRIOR_SHOT_KDE_DIM)`` float32
        Per-shot feature vector aligned to ``shots_df.index`` order.
    """
    if sigma_h_ft <= 0:
        raise ValueError(f"sigma_h_ft must be positive; got {sigma_h_ft}")
    required = ("x", "y", "player_id", "game_id", "time_remaining_sec")
    for col in required:
        if col not in shots_df.columns:
            raise KeyError(f"prior-shot-KDE featurizer needs column {col!r}")

    n = len(shots_df)
    if n == 0:
        return np.zeros((0, PRIOR_SHOT_KDE_DIM), dtype=np.float32)

    work = shots_df.reset_index(drop=True).copy()
    work["_row"] = np.arange(n, dtype=np.int64)
    work_sorted = work.sort_values(
        by=["player_id", "game_id", "time_remaining_sec", "_row"],
        kind="stable",
    )
    x_arr = work_sorted["x"].to_numpy(dtype=np.float32)
    y_arr = work_sorted["y"].to_numpy(dtype=np.float32)
    row = work_sorted["_row"].to_numpy(dtype=np.int64)

    out = np.zeros((n, PRIOR_SHOT_KDE_DIM), dtype=np.float32)

    # Group bounds — same pattern as compute_prior_outcome_features /
    # compute_within_game_features.
    keys = (
        work_sorted["player_id"].astype(str).to_numpy()
        + "|"
        + work_sorted["game_id"].astype(str).to_numpy()
    )
    boundaries = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.where(keys[1:] != keys[:-1])[0] + 1,
            np.array([n], dtype=np.int64),
        ]
    )

    centroids = ZONE_CENTROIDS_FT  # (8, 2)
    inv_two_sigma_sq = 1.0 / (2.0 * sigma_h_ft * sigma_h_ft)
    log_norm_const = -math.log(2.0 * math.pi * sigma_h_ft * sigma_h_ft)

    for grp_idx in range(len(boundaries) - 1):
        a = int(boundaries[grp_idx])
        b = int(boundaries[grp_idx + 1])
        size = b - a
        if size <= 1:
            continue

        xs = x_arr[a:b]  # (size,)
        ys = y_arr[a:b]
        # (size, 8) kernel evaluations of each shot at the 8 centroids.
        dx = xs[:, None] - centroids[None, :, 0]  # (size, 8)
        dy = ys[:, None] - centroids[None, :, 1]
        sq = dx * dx + dy * dy
        kernel = np.exp(log_norm_const - sq * inv_two_sigma_sq)  # (size, 8)

        # Causal cumulative mean: phi[r, c] = (1 / r) Σ_{r'<r} kernel[r', c].
        cum = np.cumsum(kernel, axis=0)  # cum[r] = sum of kernel[0..r] inclusive
        denom = np.arange(1, size, dtype=np.float32)  # 1, 2, ..., size-1
        # For position k (1..size-1), phi[k] = cum[k-1] / k
        phi = cum[:-1] / denom[:, None]

        # Write back to original row positions.
        out_rows = row[a + 1 : b]
        out[out_rows] = phi.astype(np.float32, copy=False)

    return out
