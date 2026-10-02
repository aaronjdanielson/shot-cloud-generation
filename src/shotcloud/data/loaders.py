"""Load NBA shot data into shotcloud's canonical schema.

The public entry point :func:`load_shots` accepts CSV or Parquet,
auto-detects NBA Stats vs. canonical column conventions, and applies
standard cleaning. For NBA Stats tables it:

- divides ``LOC_X`` / ``LOC_Y`` by 10 (tenths of feet → feet);
- renames NBA Stats columns to canonical names (``x``, ``y``,
  ``player_id``, ``game_id``, ``team``, ``date``, ``period``, ``made``);
- parses ``date`` to a datetime column;
- derives ``zone``, ``time_remaining_sec`` (elapsed game seconds), and
  ``home_away`` when the source columns are present.

For every table it then drops rows with NaN in required columns, drops
backcourt shots (zone ``-1``) by default, and derives an ``opponent``
column from ``(game_id, team)``. The returned DataFrame follows the
schema documented in :mod:`shotcloud.data.schemas`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from shotcloud.data.schemas import NBA_STATS_RENAME, REQUIRED_COLUMNS
from shotcloud.data.zones import zone_from_strings, zone_from_xy_vectorized

Format = Literal["auto", "canonical", "nba_stats"]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _detect_format(columns: list[str]) -> Literal["canonical", "nba_stats"]:
    cols = set(columns)
    if "LOC_X" in cols and "LOC_Y" in cols:
        return "nba_stats"
    if {"x", "y"}.issubset(cols):
        return "canonical"
    raise ValueError(
        "Could not auto-detect shot table format. "
        f"Found columns: {sorted(columns)[:20]}{'...' if len(columns) > 20 else ''}. "
        "Pass `format='nba_stats'` or `format='canonical'` explicitly."
    )


# ---------------------------------------------------------------------------
# NBA Stats → canonical conversion
# ---------------------------------------------------------------------------


def _nba_stats_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Rename NBA Stats columns to canonical, divide LOC by 10, parse dates.

    Also derives a ``zone`` column from ``SHOT_ZONE_BASIC`` /
    ``SHOT_ZONE_AREA`` if those are present.
    """
    df = df.copy()

    # Date hygiene: NBA Stats CSVs sometimes carry both ``GAME_DATE`` (a packed
    # YYYYMMDD integer like ``20241115``) and ``game_date`` (a string
    # ``"2024-11-15"``). The integer form is hostile to ``pd.to_datetime`` —
    # ``format="mixed"`` interprets ints as nanoseconds-since-epoch, so a
    # value like ``20241115`` becomes ``1970-01-01 00:00:00.020241115``,
    # collapsing every date to the Unix epoch. Prefer the string form when
    # both are present.
    if "game_date" in df.columns and "GAME_DATE" in df.columns:
        df = df.drop(columns=["GAME_DATE"]).rename(columns={"game_date": "GAME_DATE"})

    # Some NBA Stats exports carry lowercase duplicates of uppercase columns
    # (e.g. both PLAYER_ID and player_id). Keep the uppercase originals and
    # let the rename map handle them.
    drop_dupes = [c for c in ("player_id", "game_date", "season") if c in df.columns]
    if drop_dupes:
        # Stash season under its canonical name before dropping, since the lowercase
        # variant tends to be the readable string ("2023-24") rather than just an int.
        if "season" in df.columns and "SEASON" not in df.columns:
            df = df.rename(columns={"season": "_season_str"})
            drop_dupes = [c for c in drop_dupes if c != "season"]
        df = df.drop(columns=drop_dupes, errors="ignore")
        if "_season_str" in df.columns:
            df = df.rename(columns={"_season_str": "season"})

    rename = {src: dst for src, dst in NBA_STATS_RENAME.items() if src in df.columns}
    df = df.rename(columns=rename)

    # NBA's LOC_X / LOC_Y are tenths of feet; convert to feet.
    if "x" in df.columns:
        df["x"] = pd.to_numeric(df["x"], errors="coerce").astype(np.float64) / 10.0
    if "y" in df.columns:
        df["y"] = pd.to_numeric(df["y"], errors="coerce").astype(np.float64) / 10.0

    # Parse date — NBA exports it either as "YYYYMMDD" int (legacy uppercase
    # column) or as an ISO-ish string. Defensive parser handles both.
    if "date" in df.columns:
        if pd.api.types.is_integer_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
        df["date"] = df["date"].astype("datetime64[s]")

    # Normalize types of common identifier columns.
    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    if "made" in df.columns:
        df["made"] = pd.to_numeric(df["made"], errors="coerce").astype("Int64")
    if "period" in df.columns:
        df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")

    # Derive zone from NBA's string columns if available.
    if "SHOT_ZONE_BASIC" in df.columns and "SHOT_ZONE_AREA" in df.columns:
        df["zone"] = [
            zone_from_strings(str(b), str(a))
            for b, a in zip(df["SHOT_ZONE_BASIC"], df["SHOT_ZONE_AREA"], strict=True)
        ]

    # Compute time_remaining_sec if PERIOD/MINUTES_REMAINING/SECONDS_REMAINING available.
    if {"period", "MINUTES_REMAINING", "SECONDS_REMAINING"}.issubset(df.columns):
        # Time elapsed in game (in minutes), assuming 12-min quarters.
        period_start = (df["period"].astype("Int64") - 1) * 12
        secs_remaining = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]
        # Time elapsed in current period:
        period_elapsed = 12 * 60 - secs_remaining
        df["time_remaining_sec"] = (period_start * 60 + period_elapsed).astype("Int64")

    # Derive home_away from (TEAM_ID, HTM): a shot is a home shot iff the
    # row's team's 3-letter abbreviation matches the game's home-team
    # abbreviation. Falls back to NaN on missing / unknown teams.
    if "team" in df.columns and "HTM" in df.columns:
        from shotcloud.data.teams import NBA_TEAM_ID_TO_ABBREV

        team_id_int = pd.to_numeric(df["team"], errors="coerce").astype("Int64")
        team_abbrev = team_id_int.map(NBA_TEAM_ID_TO_ABBREV)
        df["home_away"] = (team_abbrev == df["HTM"].astype(str)).astype("Int64")
        # Restore NaN where team_id was unparseable / out of map.
        df.loc[team_abbrev.isna(), "home_away"] = pd.NA

    return df


# ---------------------------------------------------------------------------
# Cleaning passes
# ---------------------------------------------------------------------------


def _drop_nan_required(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)


def _attach_opponent(df: pd.DataFrame) -> pd.DataFrame:
    """Derive an ``opponent`` column from ``game_id`` and ``team``.

    For each game, the two participating teams' identifiers are
    aggregated; each shot's opponent is the *other* team's identifier
    in that game. Mirrors the convention from ``shot_flow``'s data
    pipeline (``opponent_team_id``).

    No-op if ``opponent`` is already present, or if either ``game_id``
    or ``team`` is missing. Games with only one team's shots in the
    table — typical at the head of a truncated read where games span
    the truncation boundary — get ``opponent = NA`` (those shots are
    not useful for Phase-2 defensive density anyway).

    Preserves the dtype of ``team``: integer-valued teams produce a
    pandas nullable ``Int64`` opponent column.
    """
    if "opponent" in df.columns:
        return df
    if not {"game_id", "team"}.issubset(df.columns):
        return df

    # Per-game min/max of `team`. When the game has exactly two distinct
    # teams these are them in some order; when only one team's shots are
    # present, min == max (which we then mask to NA).
    pair = df.groupby("game_id")["team"].agg(["min", "max"])
    out = df.merge(pair, on="game_id", how="left", suffixes=("", "_pair"))

    # `pd.Series.where(cond, other)` keeps self where cond is True.
    # Want: opponent = max if team == min, else min.
    opponent = out["min"].where(out["team"] != out["min"], out["max"])
    same_only = out["min"] == out["max"]
    if pd.api.types.is_integer_dtype(out["team"]):
        # Use nullable Int64 so the missing-opponent rows survive as <NA>
        # rather than being coerced to float NaN.
        opponent = opponent.astype("Int64")
    opponent = opponent.mask(same_only)
    out["opponent"] = opponent
    return out.drop(columns=["min", "max"])


def _drop_backcourt(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose ``zone`` is ``-1``.

    If ``zone`` is missing, derive it on the fly from ``(x, y)``.
    """
    if "zone" not in df.columns:
        zones = zone_from_xy_vectorized(df["x"].to_numpy(), df["y"].to_numpy())
    else:
        zones = df["zone"].to_numpy()
    return df.loc[zones >= 0].reset_index(drop=True)


def _attach_position(df: pd.DataFrame, position_map: Mapping[Any, str]) -> pd.DataFrame:
    df = df.copy()
    df["position"] = df["player_id"].map(position_map)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_shots(
    path: str | Path,
    *,
    format: Format = "auto",
    position_map: Mapping[Any, str] | None = None,
    drop_backcourt: bool = True,
    drop_nan: bool = True,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a shot table from CSV (or Parquet, if ``pyarrow`` is installed).

    Parameters
    ----------
    path : str or Path
        File path. Format inferred from extension (``.csv`` vs ``.parquet``).
    format : {"auto", "canonical", "nba_stats"}, default "auto"
        Column convention. ``"auto"`` detects from columns.
    position_map : Mapping, optional
        ``{player_id: position}`` mapping. If provided, adds a ``position``
        column derived from ``player_id``.
    drop_backcourt : bool, default True
        Drop rows whose zone is -1.
    drop_nan : bool, default True
        Drop rows with NaN in required columns (x, y, player_id, date).
    nrows : int, optional
        Read only the first ``nrows`` rows (CSV only). Useful for tests.

    Returns
    -------
    pd.DataFrame
        With at least the columns in
        :data:`~shotcloud.data.schemas.REQUIRED_COLUMNS`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"shot data file not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if nrows is not None:
            df = df.head(nrows)
    elif path.suffix in {".csv", ".csv.gz"}:
        df = pd.read_csv(path, nrows=nrows, low_memory=False)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}. Use .csv or .parquet.")

    actual = _detect_format(list(df.columns)) if format == "auto" else format

    if actual == "nba_stats":
        df = _nba_stats_to_canonical(df)
    # canonical format passes through; we still parse types defensively.

    # Make sure required columns now exist.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"required columns missing after loading: {missing}. "
            f"Available: {sorted(df.columns)[:20]}"
        )

    if drop_nan:
        df = _drop_nan_required(df)
    if drop_backcourt:
        df = _drop_backcourt(df)
    if position_map is not None:
        df = _attach_position(df, position_map)

    df = _attach_opponent(df)

    return df
