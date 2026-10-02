"""Causal spatial-Hawkes feature: a prior-shot KDE evaluated at zone centroids.

To let the residual tilt capture within-game spatial self-excitation
(Hawkes-style: a player's recent shot locations raising the likelihood
of nearby locations for later shots), this module exposes the
within-game spatial memory as an 8-dimensional
feature: a Gaussian KDE of the player's earlier shots in the current
game, evaluated at the eight zone centroids,

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

The kernel bandwidth :math:`\\sigma_h` defaults to 4 ft, comparable to
the typical own-support bandwidth of the spatial kernel: wide enough to
smooth within a shot cluster, narrow enough not to spread across the
corner/wing or rim/paint boundaries.

The feature is causal: only shots strictly before shot ``r`` in the
same player-game contribute, and the first shot of a player-game gets
an all-zero vector. When enabled, it is appended to the prior-outcome
summary (:mod:`shotcloud.data.prior_outcomes`) consumed by the residual
encoder's zero-initialized outcome branch.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: Number of feature slots, one per zone centroid.
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

#: Default kernel bandwidth in feet; see the module docstring for the
#: rationale.
DEFAULT_SIGMA_H_FT: Final[float] = 4.0


def compute_prior_shot_kde_features(
    shots_df: pd.DataFrame,
    sigma_h_ft: float = DEFAULT_SIGMA_H_FT,
) -> NDArray[np.float32]:
    """Compute the per-shot causal spatial-Hawkes feature.

    For each shot, the feature is the mean of isotropic Gaussian
    kernels centered on the player's *prior* shots in the same game,
    evaluated at the eight zone centroids. Shots are ordered within a
    player-game by ``time_remaining_sec``, ties broken by input row
    order. The first shot of each (player, game) gets the all-zero
    feature.

    Parameters
    ----------
    shots_df : DataFrame
        Must carry ``x``, ``y``, ``player_id``, ``game_id``,
        ``time_remaining_sec`` (which holds elapsed game seconds; see
        :mod:`shotcloud.data.schemas`).
    sigma_h_ft : float, default 4 ft
        Kernel bandwidth at zone-centroid evaluation. Must be > 0.

    Returns
    -------
    NDArray of shape ``(n_shots, PRIOR_SHOT_KDE_DIM)`` float32
        Per-shot feature vectors in the input row order.

    Raises
    ------
    ValueError
        If ``sigma_h_ft`` is not positive.
    KeyError
        If a required column is missing.
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

    # Contiguous (player_id, game_id) runs in the sorted view.
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
