"""NBA player game-log join.

Per-``(player_id, game_id)`` statistics: starter status and minutes
played, plus the recency-weighted ``recent_3pa_frac``,
``recent_usage``, and ``recent_fga`` features of the context vector
``x_n``.

**Upstream data.** The game-log CSV is produced by
``scripts/fetch_game_logs.py`` (one NBA Stats request per season) and
joined onto the shot table.

**Starter inference.** The game-log CSV does not carry the NBA Stats
``START_POSITION`` column, so :func:`load_game_logs` derives a
heuristic starter flag from minutes played:
``minutes >= STARTER_MINUTES_THRESHOLD`` (default 20). Starters
typically log 28-32 minutes and bench players 8-15, so the threshold
falls in the gap. The heuristic is a discretization of minutes and
carries no information beyond ``minutes_norm``, so it serves only as a
fallback. Pass ``starters_path`` to :func:`load_game_logs` to use
``START_POSITION``-derived values fetched by
``scripts/fetch_starters.py``; pairs present in the starters file get
the fetched value and the rest keep the heuristic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

#: Minutes-played threshold at or above which the heuristic classifies a
#: player as a starter. See the module docstring for the rationale.
STARTER_MINUTES_THRESHOLD: Final[int] = 20

#: Columns the canonical game-log frame is required to expose.
CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "player_id",
    "game_id",
    "minutes",
    "starter",
)

#: Default half-life (in days) for the exponential-recency weighting
#: used to compute ``recent_3pa_frac``, ``recent_usage``, ``recent_fga``.
#: At roughly 3-4 games per week, 30 days spans about the last 15
#: games, trading responsiveness against variance. Override via
#: :func:`load_game_logs`.
DEFAULT_RECENCY_HALFLIFE_DAYS: Final[float] = 30.0

#: Hollinger-style usage proxy weight on free-throw attempts.
#: Standard NBA convention: ``USG ~ (FGA + 0.44 * FTA + TOV) / MIN``.
USAGE_FT_WEIGHT: Final[float] = 0.44


def load_game_logs(
    path: str | Path,
    starters_path: str | Path | None = None,
    recency_halflife_days: float = DEFAULT_RECENCY_HALFLIFE_DAYS,
) -> pd.DataFrame:
    """Load ``player_game_logs.csv`` into a canonical frame.

    Parameters
    ----------
    path : str or Path
        Path to the CSV produced by ``scripts/fetch_game_logs.py``.
    starters_path : str or Path, optional
        Path to the CSV produced by ``scripts/fetch_starters.py`` —
        per-(player, game) ``position`` and ``starter`` derived from
        NBA Stats ``BoxScoreTraditionalV3``. When provided, the
        ``starter`` column is **overridden** with the real
        ``START_POSITION``-derived value for any (player_id, game_id)
        present in this file; any pair not in the file (the file may be
        partial) keeps the minutes-derived heuristic. When ``None``
        (default), starter is purely minutes-derived.
    recency_halflife_days : float, default 30
        Half-life of the exponential decay used to compute
        ``recent_3pa_frac``, ``recent_usage``, and ``recent_fga``.
        These columns capture each game's shot-mix and volume
        tendencies over the player's recent prior games (causal: only
        games strictly before the current one contribute). Set to a
        non-positive value to skip the computation (the columns are
        then absent from the output).

    Returns
    -------
    DataFrame with the canonical columns plus, when the source CSV
    carries the necessary stats, the three recency-weighted features:

    * ``player_id`` : int64
    * ``game_id`` : int64
    * ``minutes`` : int64 (NaN → 0)
    * ``starter`` : int64 (0/1)
    * ``starter_source`` : str — ``"position"`` if real start-position
      data was used for this row, ``"minutes"`` if the heuristic was
      used. Only present when ``starters_path`` is given.
    * ``recent_3pa_frac`` : float — recency-weighted ``fg3a / fga``
      over prior games. NaN for the player's first game and games
      where prior FGA is zero.
    * ``recent_usage`` : float — recency-weighted Hollinger-like
      usage proxy ``(fga + 0.44 * fta + tov) / minutes`` over prior
      games. NaN for the first game and games where prior minutes
      is zero.
    * ``recent_fga`` : float — recency-weighted mean FGA per prior
      game. NaN for the player's first game.

    The recency features need ``game_date``, ``fga``, ``fg3a``,
    ``fta``, ``tov`` from the source CSV; if any are missing, the
    corresponding column is omitted. Duplicate ``(player_id,
    game_id)`` rows are collapsed via ``drop_duplicates(keep="first")``.

    Raises
    ------
    ValueError
        If required columns ``{player_id, game_id, minutes}`` are
        missing from the source file.
    """
    df = pd.read_csv(path)
    required = {"player_id", "game_id", "minutes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"game logs missing required columns: {sorted(missing)}; got {list(df.columns)}"
        )

    df["player_id"] = df["player_id"].astype("int64")
    df["game_id"] = df["game_id"].astype("int64")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype("int64")
    df["starter"] = (df["minutes"] >= STARTER_MINUTES_THRESHOLD).astype("int64")

    keep = ["player_id", "game_id", "minutes", "starter"]
    for opt in ("season", "fga", "fg3a", "fta", "tov", "game_date"):
        if opt in df.columns:
            keep.append(opt)
    out: pd.DataFrame = df[keep].drop_duplicates(subset=["player_id", "game_id"], keep="first")
    out = out.reset_index(drop=True)

    if starters_path is not None:
        starters = load_starters(starters_path)
        out = _merge_real_starters(out, starters)

    if recency_halflife_days > 0:
        out = _add_recent_features(out, halflife_days=recency_halflife_days)

    return out


def load_starters(path: str | Path) -> pd.DataFrame:
    """Load the per-(player, game) real starter table.

    Parameters
    ----------
    path : str or Path
        Path to the CSV produced by ``scripts/fetch_starters.py``.

    Returns
    -------
    DataFrame with columns ``[game_id, player_id, position, starter]``,
    dtype-normalized to int64 ids and int64 starter (0/1). ``position``
    is the NBA Stats ``START_POSITION`` string (``"G"``, ``"F"``,
    ``"C"``) for starters and empty for bench. Duplicate
    ``(game_id, player_id)`` rows are collapsed via
    ``drop_duplicates(keep="last")`` so a later refetch overrides
    an earlier one.

    Raises
    ------
    ValueError
        If required columns ``{game_id, player_id, starter}`` are
        missing from the source file.
    """
    df = pd.read_csv(path)
    required = {"game_id", "player_id", "starter"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"starters file missing required columns: {sorted(missing)}; got {list(df.columns)}"
        )
    df["game_id"] = df["game_id"].astype("int64")
    df["player_id"] = df["player_id"].astype("int64")
    df["starter"] = pd.to_numeric(df["starter"], errors="coerce").fillna(0).astype("int64")
    if "position" in df.columns:
        df["position"] = df["position"].fillna("").astype(str)
    else:
        df["position"] = ""
    df = df[["game_id", "player_id", "position", "starter"]]
    df = df.drop_duplicates(subset=["game_id", "player_id"], keep="last")
    return df.reset_index(drop=True)


def _add_recent_features(
    gl: pd.DataFrame,
    halflife_days: float = DEFAULT_RECENCY_HALFLIFE_DAYS,
) -> pd.DataFrame:
    """Compute recency-weighted recent shot-mix features per (player, game).

    For each ``(player_id, game_id)`` row, the produced ``recent_*``
    columns aggregate the player's *prior* games with exponential
    decay, half-life ``halflife_days``. The aggregation is **strictly
    causal**: a row's recent values reflect games before its date, not
    including the row itself.

    Adds three columns to ``gl``:

    * ``recent_3pa_frac = sum_w * fg3a / sum_w * fga``
    * ``recent_usage   = sum_w * (fga + 0.44 * fta + tov) / sum_w * minutes``
    * ``recent_fga     = mean_w(fga)``

    Where ``sum_w`` is the recency-weighted sum over prior games and
    ``mean_w`` divides by the sum of weights (≠ count). Each is NaN
    when the prior pool is empty (first game of the player) or the
    relevant denominator is zero.

    Required columns in ``gl``: ``player_id``, ``game_date``, ``fga``,
    ``fg3a``, ``fta``, ``tov``, ``minutes``. When any are missing the
    function returns ``gl`` unchanged.
    """
    needed = {"player_id", "game_date", "fga", "fg3a", "fta", "tov", "minutes"}
    if not needed.issubset(gl.columns):
        return gl

    out = gl.sort_values(["player_id", "game_date"], kind="stable").reset_index(drop=True)
    n = len(out)

    rec_3pa = np.full(n, np.nan, dtype=np.float64)
    rec_usage = np.full(n, np.nan, dtype=np.float64)
    rec_fga = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        out["recent_3pa_frac"] = rec_3pa
        out["recent_usage"] = rec_usage
        out["recent_fga"] = rec_fga
        return out

    decay = float(np.log(2.0) / halflife_days)
    dates = (
        pd.to_datetime(out["game_date"], errors="coerce")
        .to_numpy(dtype="datetime64[D]")
        .astype(np.int64)
    )
    fga_arr = pd.to_numeric(out["fga"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    fg3a_arr = pd.to_numeric(out["fg3a"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    fta_arr = pd.to_numeric(out["fta"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    tov_arr = pd.to_numeric(out["tov"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    min_arr = pd.to_numeric(out["minutes"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)

    for _pid, idx in out.groupby("player_id", sort=False).indices.items():
        sum_w = 0.0
        sum_fga = 0.0
        sum_fg3a = 0.0
        sum_fta = 0.0
        sum_tov = 0.0
        sum_min = 0.0
        prev_date: int | None = None

        for i in idx:
            d = int(dates[i])
            if prev_date is not None:
                df = float(np.exp(-decay * (d - prev_date)))
                sum_w *= df
                sum_fga *= df
                sum_fg3a *= df
                sum_fta *= df
                sum_tov *= df
                sum_min *= df

            # Recent values reflect PRIOR games only (causal).
            if sum_fga > 0:
                rec_3pa[i] = sum_fg3a / sum_fga
            if sum_min > 0:
                rec_usage[i] = (sum_fga + USAGE_FT_WEIGHT * sum_fta + sum_tov) / sum_min
            if sum_w > 0:
                rec_fga[i] = sum_fga / sum_w

            # Add this row to the running sums for the next iteration.
            sum_w += 1.0
            sum_fga += fga_arr[i]
            sum_fg3a += fg3a_arr[i]
            sum_fta += fta_arr[i]
            sum_tov += tov_arr[i]
            sum_min += min_arr[i]
            prev_date = d

    out["recent_3pa_frac"] = rec_3pa
    out["recent_usage"] = rec_usage
    out["recent_fga"] = rec_fga
    return out


def _merge_real_starters(game_logs: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """Override the minutes-derived ``starter`` with real values where available.

    Adds ``starter_source`` ∈ {"position", "minutes"} for diagnostics.
    Rows present in ``starters`` get ``starter`` from there; rows absent
    keep the heuristic value already in ``game_logs``.
    """
    merged = game_logs.merge(
        starters.rename(columns={"starter": "_starter_real"})[
            ["game_id", "player_id", "_starter_real"]
        ],
        on=["game_id", "player_id"],
        how="left",
    )
    has_real = merged["_starter_real"].notna()
    merged["starter_source"] = "minutes"
    merged.loc[has_real, "starter_source"] = "position"
    merged.loc[has_real, "starter"] = merged.loc[has_real, "_starter_real"].astype("int64")
    merged["starter"] = merged["starter"].astype("int64")
    return merged.drop(columns=["_starter_real"])


def join_game_logs(
    shots: pd.DataFrame,
    game_logs: pd.DataFrame,
    impute_missing: bool = True,
) -> pd.DataFrame:
    """Left-join ``game_logs`` onto ``shots`` on ``(player_id, game_id)``.

    Parameters
    ----------
    shots : DataFrame
        Output of :func:`shotcloud.data.load_shots`. Must have lowercase
        ``player_id`` and ``game_id`` columns (the canonical post-rename
        names from ``schemas.NBA_STATS_RENAME``).
    game_logs : DataFrame
        Output of :func:`load_game_logs`.
    impute_missing : bool, default True
        When True, shots whose ``(player_id, game_id)`` is absent from
        ``game_logs`` get ``minutes=0`` and ``starter=0``. When False,
        those rows have NaN — useful for diagnostics. The default is
        True so downstream model code can safely assume non-null.

    Returns
    -------
    DataFrame with the original shots columns plus ``minutes`` and
    ``starter`` (both int64 when imputed) and any of
    ``recent_3pa_frac``, ``recent_usage``, ``recent_fga`` present in
    ``game_logs`` (NaN filled with 0 when imputed).

    Notes
    -----
    The join can have missing rows for (a) shots from seasons before
    the game-logs file covers, (b) games where the player logged 0 FGA
    *and* 0 minutes (these can be filtered out of game logs upstream),
    or (c) data-quality gaps. With ``impute_missing=True`` the model
    sees these as bench players with 0 minutes — a reasonable bench
    fallback that keeps the pipeline robust.
    """
    for col in ("player_id", "game_id"):
        if col not in shots.columns:
            raise ValueError(
                f"shots must have '{col}' column; got {list(shots.columns)}. "
                f"Did you forget to call load_shots() first?"
            )

    shots = shots.copy()
    shots["player_id"] = shots["player_id"].astype("int64")
    shots["game_id"] = shots["game_id"].astype("int64")

    # Only carry the join-relevant subset of game_logs into the merge to
    # avoid silently overwriting shots-side columns (e.g., 'season').
    keep = ["player_id", "game_id", "minutes", "starter"]
    for col in ("recent_3pa_frac", "recent_usage", "recent_fga"):
        if col in game_logs.columns:
            keep.append(col)
    gl_subset = game_logs[keep]
    merged = shots.merge(gl_subset, on=["player_id", "game_id"], how="left")

    if impute_missing:
        merged["minutes"] = merged["minutes"].fillna(0).astype("int64")
        merged["starter"] = merged["starter"].fillna(0).astype("int64")
        # Recency features: NaN → 0 (consistent with "no prior history").
        for col in ("recent_3pa_frac", "recent_usage", "recent_fga"):
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0).astype("float64")

    return merged
