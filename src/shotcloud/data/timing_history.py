"""Causal per-player historical shot-time distribution.

For each shot, builds a 48-bin normalized histogram of the times (one
bin per game minute) at which the player shot in **strictly prior**
games. The featurizer is causal at game granularity: a shot on date
``D`` uses only shots from dates strictly before ``D``, so neither the
shot's own game nor any same-day shot contributes.

The 27-dimensional context vector ``x_n`` carries starter status,
minutes, and role, but not *when* in a game the player tends to shoot.
Appended to ``x_n``, this feature gives the timing head that signal
directly.

The signature follows
:func:`shotcloud.data.within_game_history.compute_within_game_features`
and :func:`shotcloud.data.prior_outcomes.compute_prior_outcome_features`:
a shots DataFrame in, a ``(n_shots, TIMING_HISTORY_DIM)`` array in input
row order out.

With the default Laplace smoothing, players with no prior games get the
uniform distribution and players with few prior shots a near-uniform
one, which carries little information beyond the other context slots.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shotcloud.training.gibbs_dataset import N_TIMING_BINS, _compute_tau_bin

#: Number of historical-timing feature slots; equal to the timing
#: head's bin count.
TIMING_HISTORY_DIM: Final[int] = N_TIMING_BINS  # 48


def compute_player_timing_history_features(
    shots_df: pd.DataFrame,
    *,
    smoothing: float = 1.0,
) -> NDArray[np.float32]:
    """Compute the per-shot causal historical shot-time distribution.

    Parameters
    ----------
    shots_df : DataFrame
        Must carry ``player_id``, ``date``, ``period``,
        ``time_remaining_sec``. ``time_remaining_sec`` is total
        elapsed seconds in the game (per
        :func:`shotcloud.data.loaders.load_shots`).
    smoothing : float, default 1.0
        Laplace smoothing constant added to each bin's count before
        normalization. With a positive value, players without prior
        games get the uniform $1/48$ distribution; with 0, they get an
        all-zero vector.

    Returns
    -------
    NDArray of shape ``(n_shots, TIMING_HISTORY_DIM)`` float32
        Per-shot feature vectors in the input row order. Row $i$ is
        the player's normalized histogram of ``tau_bin`` over their
        shots from strictly earlier dates.

    Raises
    ------
    KeyError
        If a required column is missing.
    """
    required = ("player_id", "date", "period", "time_remaining_sec")
    for col in required:
        if col not in shots_df.columns:
            raise KeyError(f"timing-history featurizer needs column {col!r}")

    n = len(shots_df)
    if n == 0:
        return np.zeros((0, TIMING_HISTORY_DIM), dtype=np.float32)

    # Per-shot tau_bin via the same canonical mapping the dataset uses.
    tau_bin = _compute_tau_bin(
        shots_df["period"].to_numpy(dtype=np.int64),
        shots_df["time_remaining_sec"].to_numpy(dtype=np.float64),
    )

    work = shots_df.reset_index(drop=True).copy()
    work["_row"] = np.arange(n, dtype=np.int64)
    work["_tau_bin"] = tau_bin

    # Sort by player, then date so each player's groupby yields shots
    # in chronological order; ties on date break by original row index
    # (stable sort).
    work_sorted = work.sort_values(
        by=["player_id", "date", "_row"],
        kind="stable",
    )

    out = np.zeros((n, TIMING_HISTORY_DIM), dtype=np.float32)

    for _, g in work_sorted.groupby("player_id", sort=False):
        # Per-game-date histograms for this player.
        dates = g["date"].to_numpy()
        # Shots on the same day count as the same game for the
        # strict-prior cutoff, so within-game shots never enter each
        # other's histories.
        dates_day = dates.astype("datetime64[D]")
        taus = g["_tau_bin"].to_numpy(dtype=np.int64)
        rows = g["_row"].to_numpy(dtype=np.int64)

        unique_dates, date_idx = np.unique(dates_day, return_inverse=True)
        n_unique = len(unique_dates)

        # Per-game per-bin counts, in chronological order.
        per_game_hist = np.zeros((n_unique, TIMING_HISTORY_DIM), dtype=np.int64)
        np.add.at(per_game_hist, (date_idx, taus), 1)

        # Cumulative-from-the-past: history available for game $d$
        # is the sum over games $0..d-1$ (strictly prior dates).
        cumulative = np.cumsum(per_game_hist, axis=0)
        history_at_date = np.zeros_like(cumulative)
        history_at_date[1:] = cumulative[:-1]

        # Laplace smoothing + normalize per game.
        smoothed = history_at_date.astype(np.float64) + float(smoothing)
        norms = smoothed.sum(axis=-1, keepdims=True)
        # Cold-start rows with smoothing=0 have all-zero counts; treat them
        # as a deliberate zero feature rather than dividing by zero into NaN.
        safe_norms = np.where(norms > 0, norms, 1.0)
        normalized = (smoothed / safe_norms).astype(np.float32)
        normalized[norms.squeeze(-1) == 0] = 0.0

        # Scatter each shot's per-game-date row back into the
        # original DataFrame index space.
        out[rows] = normalized[date_idx]

    return out


__all__ = [
    "TIMING_HISTORY_DIM",
    "compute_player_timing_history_features",
]
