"""Data-driven position-group inference.

Derives ``"big"`` / ``"wing"`` / ``"guard"`` position labels for each
player from their **restricted-area shot rate** — the fraction of their
shots taken within 4 ft of the basket. Adapted from
[shot_flow/scripts/precompute_kde_maps.py](/Users/aarondanielson/Dropbox/shot_flow/scripts/precompute_kde_maps.py)
where this approach was validated against the full neural ablation.

**Why this over a roster CSV.** Self-contained, reproducible from the
shot data alone, no external API dependency or staleness. NBA position
labels are themselves squishy — a 2025 wing might play "big" minutes
when small lineups are deployed. RA rate measures actual shot behavior,
which is exactly what the model's position-group KDE prior wants to
capture.

Default thresholds match shot_flow:

==========  ===============  ============================
Group       RA rate          Typical NBA positions
==========  ===============  ============================
``big``     ``≥ 0.35``       Centers, power forwards
``wing``    ``[0.20, 0.35)`` Small forwards, shooting guards
``guard``   ``< 0.20``       Point guards
==========  ===============  ============================

Players with fewer than ``min_shots`` (default 10) get the league-average
fallback label (``"wing"``).
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Final

import numpy as np
import pandas as pd

DEFAULT_BIG_THRESHOLD: Final[float] = 0.35
DEFAULT_WING_THRESHOLD: Final[float] = 0.20
DEFAULT_FALLBACK: Final[str] = "wing"
DEFAULT_MIN_SHOTS: Final[int] = 10
RA_RADIUS_FT: Final[float] = 4.0

POSITION_GROUPS: Final[tuple[str, ...]] = ("big", "wing", "guard")


def ra_rate_per_player(
    shots_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    player_col: str = "player_id",
) -> pd.Series:
    """Compute each player's restricted-area rate (fraction of shots within 4 ft).

    Returns a Series indexed by ``player_id``. Inputs missing ``x``, ``y``
    are skipped. A player with no in-data shots simply doesn't appear in
    the output — callers that want to handle missing players use
    :func:`derive_positions_from_ra_rate` instead.
    """
    for col in (x_col, y_col, player_col):
        if col not in shots_df.columns:
            raise KeyError(f"shots_df must contain column {col!r}")

    df = shots_df[[x_col, y_col, player_col]].dropna(subset=[x_col, y_col])
    if len(df) == 0:
        return pd.Series(dtype=np.float64, name="ra_rate")

    distances = np.sqrt(df[x_col].to_numpy() ** 2 + df[y_col].to_numpy() ** 2)
    in_ra = pd.Series(distances < RA_RADIUS_FT, index=df.index)
    rate = in_ra.groupby(df[player_col]).mean()
    rate.name = "ra_rate"
    return rate


def assign_position_group(
    ra_rate: float,
    *,
    big_threshold: float = DEFAULT_BIG_THRESHOLD,
    wing_threshold: float = DEFAULT_WING_THRESHOLD,
) -> str:
    """Map a single RA-rate value to a position group.

    >>> assign_position_group(0.40)
    'big'
    >>> assign_position_group(0.25)
    'wing'
    >>> assign_position_group(0.15)
    'guard'
    """
    if not (0.0 <= wing_threshold <= big_threshold <= 1.0):
        raise ValueError(
            f"thresholds must satisfy 0 ≤ wing ≤ big ≤ 1; "
            f"got wing={wing_threshold}, big={big_threshold}"
        )
    if ra_rate >= big_threshold:
        return "big"
    if ra_rate >= wing_threshold:
        return "wing"
    return "guard"


def derive_positions_from_ra_rate(
    shots_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    player_col: str = "player_id",
    big_threshold: float = DEFAULT_BIG_THRESHOLD,
    wing_threshold: float = DEFAULT_WING_THRESHOLD,
    min_shots: int = DEFAULT_MIN_SHOTS,
    fallback: str = DEFAULT_FALLBACK,
) -> dict[Hashable, str]:
    """Return ``{player_id: position_group}`` derived from RA rate.

    Players with fewer than ``min_shots`` shots get the ``fallback`` label
    (default ``"wing"``, the league-average bucket). Player IDs are
    preserved with their original type (``int``, ``str``, etc.).
    """
    if min_shots < 1:
        raise ValueError(f"min_shots must be ≥ 1; got {min_shots}")

    rates = ra_rate_per_player(shots_df, x_col=x_col, y_col=y_col, player_col=player_col)
    counts = shots_df.dropna(subset=[x_col, y_col]).groupby(player_col).size()

    out: dict[Hashable, str] = {}
    for pid, n in counts.items():
        if n < min_shots:
            out[pid] = fallback
            continue
        rate = rates.get(pid)
        if rate is None or pd.isna(rate):
            out[pid] = fallback
            continue
        out[pid] = assign_position_group(
            float(rate),
            big_threshold=big_threshold,
            wing_threshold=wing_threshold,
        )
    return out
