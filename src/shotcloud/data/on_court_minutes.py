"""Per-shot causal on-court presence feature.

The 27-dimensional context vector ``x_n`` carries starter status and
minutes played but not *when* in the game a player is typically on
court. This module supplies that feature for the timing factor: a
30-bin curve giving, for each 2-minute bin of elapsed game time, the
expected fraction of the bin the player spends on court.

* **Per-game source.** The per-(game, player) on-court table produced
  by ``scripts/build_oncourt_table.py``. Each row is
  ``(game_id, game_date, player_id, starter, bin_0..bin_{N-1})``, with
  ``bin_b`` the fraction of the bin's 120 seconds the player was on
  court, derived from play-by-play lineup tracking.
* **Causal lookup.** For a shot on date $D$ by player $p$ with starter
  status $s$, the prior set is the player's games strictly before $D$
  with the same starter status. Same-day games are excluded, since a
  same-day game is the shot's own game.
* **Smoothing.** A frozen :class:`~shotcloud.models.presence.PresenceModel`
  maps the prior set to the output curve, gating between the player's
  own history and a learned position-by-starter pool; rows with no
  prior games fall back to the pool.

The output, one ``ON_COURT_HISTORY_DIM``-vector per shot, is appended
to ``x_n_raw`` when training the timing head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: Number of 2-minute on-court bins of elapsed game time (matches ``N_BINS`` in
#: ``scripts/build_oncourt_table.py``).
ON_COURT_HISTORY_DIM: Final[int] = 30

#: Bin width in seconds.
ON_COURT_BIN_WIDTH_SEC: Final[int] = 120

_BIN_COLS: Final[tuple[str, ...]] = tuple(f"bin_{b}" for b in range(ON_COURT_HISTORY_DIM))


def load_on_court_table(path: str | Path) -> pd.DataFrame:
    """Load the per-(game, player) on-court table.

    Parameters
    ----------
    path : str or Path
        CSV produced by ``scripts/build_oncourt_table.py``.

    Returns
    -------
    DataFrame
        The table, with ``game_date`` parsed to datetime.

    Raises
    ------
    ValueError
        If any identifier column or any of the ``ON_COURT_HISTORY_DIM``
        bin columns is missing.
    """
    df = pd.read_csv(path, parse_dates=["game_date"])
    required = ("game_id", "game_date", "player_id", "starter", *_BIN_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"on-court table missing columns: {missing}")
    return df


def _global_mean_bins(table: pd.DataFrame) -> NDArray[np.float32]:
    """League-mean bin vector over all rows of ``table``."""
    arr: NDArray[np.float32] = table[list(_BIN_COLS)].mean(axis=0).to_numpy(dtype=np.float32)
    return arr


def compute_player_on_court_features(
    shots_df: pd.DataFrame,
    table: pd.DataFrame,
    presence_model: object,
    player_to_position: dict[int, int],
    *,
    chunk_size: int = 2048,
) -> NDArray[np.float32]:
    """Compute the per-shot on-court feature with a frozen presence model.

    For each shot $(p, d, s)$ this builds the strictly prior,
    matching-starter set of games from ``table``, then queries
    ``presence_model(prior_bins, prior_ages_days, prior_mask,
    position_idx, starter_idx, history_count)`` to get the 30-bin
    on-court fraction. Backoff is handled by the model's gate rather
    than an explicit chain: rows with no prior games
    (``history_count = 0``) fall back to the learned position-starter
    pool.

    Parameters
    ----------
    shots_df : DataFrame
        Must carry ``player_id``, ``date``, and ``starter``. When
        ``starter`` is missing, every shot is treated as a bench shot.
    table : DataFrame
        Output of :func:`load_on_court_table`. Provides per-(game,
        player) on-court vectors that feed the model's self-curve.
    presence_model : PresenceModel
        Trained model; it is put in ``eval()`` mode and run without
        gradients. Must expose the forward signature
        ``forward(prior_bins, prior_ages_days, prior_mask, position_idx,
        starter_idx, history_count)``.
    player_to_position : dict[int, int]
        ``{player_id: position_idx}`` where ``position_idx`` is in
        ``[0, n_positions)``. Unknown players default to position 0.
    chunk_size : int, default 2048
        Batch size for the model forward; tunes memory/throughput.

    Returns
    -------
    NDArray of shape ``(n_shots, ON_COURT_HISTORY_DIM)`` float32.
    """
    import torch

    if "player_id" not in shots_df.columns or "date" not in shots_df.columns:
        raise KeyError("on-court featurizer needs 'player_id' and 'date' columns")
    if "starter" not in shots_df.columns:
        shots_df = shots_df.copy()
        shots_df["starter"] = 0

    n = len(shots_df)
    out = np.zeros((n, ON_COURT_HISTORY_DIM), dtype=np.float32)
    if n == 0:
        return out

    by_player: dict[int, pd.DataFrame] = {
        int(pid): g.sort_values("game_date").reset_index(drop=True)
        for pid, g in table.groupby("player_id", sort=False)
    }
    bin_cols = list(_BIN_COLS)
    pids = shots_df["player_id"].astype(int).to_numpy()
    dates = pd.to_datetime(shots_df["date"]).to_numpy(dtype="datetime64[D]")
    starters = shots_df["starter"].astype(int).to_numpy()

    device = next(presence_model.parameters()).device  # type: ignore[attr-defined]

    # Process in chunks. Within a chunk, K_max is the max prior count
    # across the chunk's shots; pad shorter rows with zeros.
    presence_model.eval()  # type: ignore[attr-defined]
    n_cold = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_prior_bins: list[np.ndarray] = []
        chunk_prior_ages: list[np.ndarray] = []
        chunk_pos: list[int] = []
        chunk_starter: list[int] = []
        chunk_history: list[int] = []
        k_max = 0
        for i in range(start, end):
            pid = int(pids[i])
            d = dates[i]
            s = int(starters[i])
            g = by_player.get(pid)
            if g is None:
                chunk_prior_bins.append(np.zeros((0, ON_COURT_HISTORY_DIM), dtype=np.float32))
                chunk_prior_ages.append(np.zeros(0, dtype=np.float32))
                chunk_history.append(0)
                n_cold += 1
            else:
                prior = g[(g["game_date"].values < d) & (g["starter"].values == s)]
                if prior.empty:
                    chunk_prior_bins.append(np.zeros((0, ON_COURT_HISTORY_DIM), dtype=np.float32))
                    chunk_prior_ages.append(np.zeros(0, dtype=np.float32))
                    chunk_history.append(0)
                    n_cold += 1
                else:
                    prior_bins_arr = prior[bin_cols].to_numpy(dtype=np.float32)
                    prior_dates = prior["game_date"].values.astype("datetime64[D]")
                    prior_ages_arr = (d - prior_dates).astype("timedelta64[D]").astype(np.float32)
                    chunk_prior_bins.append(prior_bins_arr)
                    chunk_prior_ages.append(prior_ages_arr)
                    chunk_history.append(int(prior_bins_arr.shape[0]))
                    k_max = max(k_max, prior_bins_arr.shape[0])
            chunk_pos.append(int(player_to_position.get(pid, 0)))
            chunk_starter.append(s)
        if k_max == 0:
            k_max = 1
        b = end - start
        bins_pad = np.zeros((b, k_max, ON_COURT_HISTORY_DIM), dtype=np.float32)
        ages_pad = np.zeros((b, k_max), dtype=np.float32)
        mask_pad = np.zeros((b, k_max), dtype=np.float32)
        for j, (bins_j, ages_j) in enumerate(zip(chunk_prior_bins, chunk_prior_ages, strict=False)):
            k_j = bins_j.shape[0]
            if k_j > 0:
                bins_pad[j, :k_j] = bins_j
                ages_pad[j, :k_j] = ages_j
                mask_pad[j, :k_j] = 1.0
        with torch.no_grad():
            curve = presence_model(  # type: ignore[operator]
                prior_bins=torch.from_numpy(bins_pad).to(device),
                prior_ages_days=torch.from_numpy(ages_pad).to(device),
                prior_mask=torch.from_numpy(mask_pad).to(device),
                position_idx=torch.tensor(chunk_pos, dtype=torch.long, device=device),
                starter_idx=torch.tensor(chunk_starter, dtype=torch.long, device=device),
                history_count=torch.tensor(chunk_history, dtype=torch.float32, device=device),
            )
        out[start:end] = curve.detach().cpu().numpy().astype(np.float32)

    print(
        f"[on-court] {n} shots: cold_start_fraction={100.0 * n_cold / n:.1f}% "
        f"(cold rows route through the gate to the position-starter pool)",
        flush=True,
    )
    return out


def league_mean_curve(table: pd.DataFrame) -> NDArray[np.float32]:
    """Return the league-mean on-court curve over all rows of ``table``.

    A static fallback for callers that need an ``ON_COURT_HISTORY_DIM``
    feature without a presence model. The mean is taken over every row,
    so it is causal only when ``table`` is restricted to training games.
    """
    return _global_mean_bins(table)


__all__ = [
    "ON_COURT_BIN_WIDTH_SEC",
    "ON_COURT_HISTORY_DIM",
    "compute_player_on_court_features",
    "league_mean_curve",
    "load_on_court_table",
]
