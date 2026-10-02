"""Canonical column names for shotcloud shot tables.

Loaders normalize input data to this schema so downstream code can
reference columns without worrying about provider-specific names.

Required columns
----------------
- ``x``, ``y`` : float, feet (basket at origin).
- ``player_id`` : hashable.
- ``date`` : ``datetime64[D]``.

Optional columns
----------------
- ``position`` : string (e.g., ``"G"``, ``"F"``, ``"C"``). Position label
  used by :class:`~shotcloud.kde.HierarchicalKDE` for shrinkage; the
  loader can populate it via a ``position_map``.
- ``game_id``, ``team``, ``opponent``, ``season`` : grouping / context.
- ``period``, ``time_remaining_sec`` : timing. Despite its name,
  ``time_remaining_sec`` holds the seconds *elapsed* since the start of
  the game (see :func:`~shotcloud.data.load_shots`).
- ``made`` : 0/1 shot outcome.
- ``zone`` : 0–7 NBA zone index from :mod:`~shotcloud.data.zones`.
- ``home_away`` : 1 if the row's team is the game's home team, 0 if visitor.
"""

from __future__ import annotations

from typing import Final

#: Columns every loaded shot table must carry; :func:`~shotcloud.data.load_shots`
#: raises if any is missing.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("x", "y", "player_id", "date")

#: Columns preserved when present; missing optional columns are not synthesized.
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "position",
    "game_id",
    "team",
    "opponent",
    "season",
    "period",
    "time_remaining_sec",
    "made",
    "zone",
)

#: Mapping from NBA Stats API column names to shotcloud canonical names,
#: applied by the loader's NBA Stats adapter.
NBA_STATS_RENAME: Final[dict[str, str]] = {
    "LOC_X": "x",
    "LOC_Y": "y",
    "PLAYER_ID": "player_id",
    "GAME_ID": "game_id",
    "TEAM_ID": "team",
    "GAME_DATE": "date",
    "PERIOD": "period",
    "SHOT_MADE_FLAG": "made",
}
