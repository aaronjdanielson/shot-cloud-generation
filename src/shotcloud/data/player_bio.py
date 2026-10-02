"""Player biographical data loader.

Consumes the CSV produced by :doc:`scripts/fetch_player_bio`. Provides
a typed dataframe + helpers for the trait-vector builder in
:mod:`shotcloud.data.player_traits` (Phase 2 of the collaborative-KDE
pivot).

The biographical data is the cold-start-tolerant block of the trait
vector — every NBA player has a height/weight/birthdate/position
regardless of whether they've taken any shots yet. See
:doc:`docs/model_spec.md` §"Causal player traits".
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: Five-way biographical position groups (best-effort mapping from the
#: NBA Stats POSITION field, which is 3-way + hyphenated combos).
POSITION_GROUPS: Final[tuple[str, ...]] = ("PG", "SG", "SF", "PF", "C")

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "player_id",
    "display_name",
    "birthdate",
    "height_inches",
    "weight_lbs",
    "position_raw",
    "position_group",
    "status",
)


def load_player_bio(path: Path | str) -> pd.DataFrame:
    """Load the player_bio CSV produced by ``scripts/fetch_player_bio.py``.

    Returns a dataframe with typed columns:

    * ``player_id``  int64
    * ``display_name``  string (may be NaN if status != "ok")
    * ``birthdate``  datetime64[ns] (NaT on missing)
    * ``height_inches``  Int64 (nullable; NaN on missing)
    * ``weight_lbs``  Int64 (nullable; NaN on missing)
    * ``position_raw``  string
    * ``position_group``  one of :data:`POSITION_GROUPS` or NaN
    * ``status``  ``"ok" | "failed" | "missing"``

    Other columns from the CSV (``source``, ``fetched_at``, ``error_message``)
    are kept as-is for provenance but aren't required by downstream code.

    Rows with ``status != "ok"`` are kept (the caller decides how to
    handle them — typically by treating their bio fields as NaN, which
    Phase 2 then represents via the missingness indicator).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"player_bio CSV not found: {p}")
    df = pd.read_csv(p)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{p} is missing required columns: {missing} (have: {list(df.columns)})")

    df = df.copy()
    df["player_id"] = df["player_id"].astype("int64")
    df["display_name"] = df["display_name"].astype("string")
    df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce")
    df["height_inches"] = df["height_inches"].astype("Int64")
    df["weight_lbs"] = df["weight_lbs"].astype("Int64")
    df["position_raw"] = df["position_raw"].astype("string")
    df["position_group"] = df["position_group"].astype("string")
    df["status"] = df["status"].astype("string")
    return df


def compute_age_years(
    birthdate: pd.Series, ref_date: np.datetime64 | pd.Timestamp | str
) -> NDArray[np.float64]:
    """Age in years at ``ref_date``, per row of ``birthdate``.

    Returns ``NaN`` for rows where birthdate is NaT.
    """
    ref = pd.Timestamp(ref_date)
    # Convert to seconds, divide by avg seconds/year. Using 365.25
    # absorbs leap-year drift; sub-day precision doesn't matter for
    # snapshot-level traits.
    deltas = (ref - birthdate).dt.total_seconds()
    years: NDArray[np.float64] = (deltas / (365.25 * 24 * 3600)).to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    return years


def position_group_to_onehot(group: str | None) -> NDArray[np.float64]:
    """Map a position group (one of :data:`POSITION_GROUPS`) to a 5-d one-hot.

    Returns an all-zero vector for unrecognized / missing values. The
    trait builder pairs this with the missingness indicator so the
    model can distinguish "we don't know the position" from "this is a
    different valid position."
    """
    out = np.zeros(len(POSITION_GROUPS), dtype=np.float64)
    if group is None or (isinstance(group, float) and np.isnan(group)):
        return out
    try:
        idx = POSITION_GROUPS.index(str(group))
    except ValueError:
        return out
    out[idx] = 1.0
    return out


def position_group_onehot_matrix(groups: pd.Series) -> NDArray[np.float64]:
    """Vectorized one-hot encoding of a pandas Series of position groups.

    Returns a ``(len(groups), 5)`` matrix. Unrecognized / NaN rows are
    all-zero.
    """
    n = len(groups)
    out = np.zeros((n, len(POSITION_GROUPS)), dtype=np.float64)
    for i, g in enumerate(POSITION_GROUPS):
        out[:, i] = (groups == g).fillna(False).astype(np.float64).to_numpy()
    return out
