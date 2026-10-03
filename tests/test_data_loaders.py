"""Tests for :func:`shotcloud.data.load_shots`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shotcloud import load_shots

# The collected shot data (scripts/fetch_shots.py). The integration test is
# skipped when this file is unavailable.
SHOT_FLOW_CSV = Path(__file__).resolve().parents[1] / "data/raw/shot_data.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, tmp_path: Path, name: str = "shots.csv") -> Path:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _canonical_synthetic(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x": rng.normal(0.0, 5.0, n),
            "y": rng.normal(10.0, 5.0, n),
            "player_id": rng.choice(["A", "B"], n),
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        }
    )


def _nba_stats_synthetic(n: int = 50) -> pd.DataFrame:
    """Build a synthetic NBA Stats frame with uppercase columns and tenths-of-feet coordinates."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "GRID_TYPE": ["Shot Chart Detail"] * n,
            "GAME_ID": rng.integers(20000000, 30000000, n).astype(str),
            "PLAYER_ID": rng.choice([1001, 1002], n),
            "PLAYER_NAME": rng.choice(["A", "B"], n),
            "TEAM_ID": rng.choice([100, 200], n),
            "PERIOD": rng.integers(1, 5, n),
            "MINUTES_REMAINING": rng.integers(0, 12, n),
            "SECONDS_REMAINING": rng.integers(0, 60, n),
            "EVENT_TYPE": rng.choice(["Made Shot", "Missed Shot"], n),
            "ACTION_TYPE": ["Jump Shot"] * n,
            "SHOT_TYPE": ["2PT Field Goal"] * n,
            "SHOT_ZONE_BASIC": rng.choice(["Restricted Area", "Mid-Range", "Above the Break 3"], n),
            "SHOT_ZONE_AREA": rng.choice(["Center(C)", "Right Side(R)", "Left Side(L)"], n),
            "SHOT_ZONE_RANGE": ["8-16 ft."] * n,
            "SHOT_DISTANCE": rng.uniform(0, 30, n),
            # tenths of feet
            "LOC_X": rng.integers(-250, 250, n),
            "LOC_Y": rng.integers(-50, 470, n),
            "SHOT_ATTEMPTED_FLAG": [1] * n,
            "SHOT_MADE_FLAG": rng.choice([0, 1], n),
            "GAME_DATE": ["20241115"] * n,
        }
    )


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def test_load_shots_canonical(tmp_path: Path) -> None:
    df_in = _canonical_synthetic(20)
    path = _write_csv(df_in, tmp_path)
    df_out = load_shots(path, drop_backcourt=False)

    assert {"x", "y", "player_id", "date"}.issubset(df_out.columns)
    assert len(df_out) == 20


def test_load_shots_nba_stats_renames_and_scales(tmp_path: Path) -> None:
    df_in = _nba_stats_synthetic(20)
    path = _write_csv(df_in, tmp_path)
    df_out = load_shots(path, drop_backcourt=False)

    # Canonical columns are present.
    assert {"x", "y", "player_id", "date", "made", "period", "game_id"}.issubset(df_out.columns)
    # LOC_X / LOC_Y / 10 → bounded by [-25, 47].
    assert df_out["x"].between(-25.0, 25.0).all()
    assert df_out["y"].between(-5.0, 47.0).all()
    # Dates are datetime.
    assert pd.api.types.is_datetime64_any_dtype(df_out["date"])
    # 'made' decoded to 0/1.
    assert set(df_out["made"].dropna().unique()).issubset({0, 1})


def test_unrecognized_format_raises(tmp_path: Path) -> None:
    df_in = pd.DataFrame({"foo": [1], "bar": [2]})
    path = _write_csv(df_in, tmp_path)
    with pytest.raises(ValueError, match="auto-detect"):
        load_shots(path)


def test_explicit_format_override(tmp_path: Path) -> None:
    """With ``format="canonical"``, columns must already be canonical."""
    df_in = _canonical_synthetic(10)
    path = _write_csv(df_in, tmp_path)
    df_out = load_shots(path, format="canonical", drop_backcourt=False)
    assert len(df_out) == 10


# ---------------------------------------------------------------------------
# Cleaning passes
# ---------------------------------------------------------------------------


def test_drop_nan_required(tmp_path: Path) -> None:
    df = _canonical_synthetic(10)
    df.loc[2, "x"] = np.nan
    df.loc[5, "player_id"] = np.nan
    path = _write_csv(df, tmp_path)
    df_out = load_shots(path, drop_backcourt=False, drop_nan=True)
    assert len(df_out) == 8


def test_drop_nan_can_be_disabled(tmp_path: Path) -> None:
    df = _canonical_synthetic(10)
    df.loc[2, "x"] = np.nan
    path = _write_csv(df, tmp_path)
    df_out = load_shots(path, drop_backcourt=False, drop_nan=False)
    assert len(df_out) == 10


def test_drop_backcourt_removes_far_y_shots(tmp_path: Path) -> None:
    """A shot beyond the half-court line (y=50) is backcourt (zone -1)."""
    df = pd.DataFrame(
        {
            "x": [0.0, 0.0],
            "y": [5.0, 50.0],  # near basket, beyond half-court line
            "player_id": ["A", "A"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    path = _write_csv(df, tmp_path)
    df_out = load_shots(path, drop_backcourt=True)
    assert len(df_out) == 1
    assert df_out["y"].iloc[0] == 5.0


# ---------------------------------------------------------------------------
# position_map
# ---------------------------------------------------------------------------


def test_position_map_attaches_position_column(tmp_path: Path) -> None:
    df_in = _canonical_synthetic(10)
    path = _write_csv(df_in, tmp_path)
    pos = {"A": "G", "B": "F"}
    df_out = load_shots(path, position_map=pos, drop_backcourt=False)
    assert "position" in df_out.columns
    assert set(df_out["position"].dropna().unique()).issubset({"G", "F"})


# ---------------------------------------------------------------------------
# Opponent derivation
# ---------------------------------------------------------------------------


def test_opponent_derived_from_game_pair(tmp_path: Path) -> None:
    """The ``opponent`` column is the other ``TEAM_ID`` in the same game."""
    rng = np.random.default_rng(0)
    n_per_team = 20
    rows = []
    for game in (101, 102):
        for team_a, team_b in [("T_AAA", "T_BBB")] if game == 101 else [("T_CCC", "T_DDD")]:
            for shooter in (team_a, team_b):
                for _ in range(n_per_team):
                    rows.append(
                        {
                            "x": float(rng.normal(0, 5)),
                            "y": float(rng.normal(15, 5)),
                            "player_id": "P",
                            "date": "2024-01-01",
                            "game_id": game,
                            "team": shooter,
                        }
                    )
    df_in = pd.DataFrame(rows)
    path = _write_csv(df_in, tmp_path)
    df_out = load_shots(path, drop_backcourt=False)

    assert "opponent" in df_out.columns
    # Every row's opponent must equal the *other* team in its game.
    for _, row in df_out.iterrows():
        same_game = df_out[df_out["game_id"] == row["game_id"]]
        teams = set(same_game["team"].unique())
        assert row["opponent"] != row["team"]
        assert row["opponent"] in teams - {row["team"]}


def test_opponent_is_na_for_singleton_team_game(tmp_path: Path) -> None:
    """When only one team's shots are present in a game, opponent is NA."""
    df_in = pd.DataFrame(
        [
            {
                "x": 0.0,
                "y": 5.0,
                "player_id": "P",
                "date": "2024-01-01",
                "game_id": 200,
                "team": "T_LONELY",
            }
            for _ in range(5)
        ]
    )
    path = _write_csv(df_in, tmp_path)
    df_out = load_shots(path, drop_backcourt=False)
    assert df_out["opponent"].isna().all()


def test_opponent_passthrough_when_already_present(tmp_path: Path) -> None:
    """An existing ``opponent`` column is passed through unchanged."""
    df_in = _canonical_synthetic(8)
    df_in["opponent"] = "PRESET"
    path = _write_csv(df_in, tmp_path)
    df_out = load_shots(path, drop_backcourt=False)
    assert (df_out["opponent"] == "PRESET").all()


# ---------------------------------------------------------------------------
# Date parsing: integer YYYYMMDD vs string, dual-column preference
# ---------------------------------------------------------------------------


def _nba_stats_synthetic_for_dates(
    n: int = 5,
    *,
    use_integer_game_date: bool = False,
    include_lowercase_game_date: bool = False,
) -> pd.DataFrame:
    """Build an NBA-Stats-format frame for the date-parsing tests.

    Unlike :func:`_nba_stats_synthetic`, this fixture exercises only the
    date columns and lets the caller choose the ``GAME_DATE`` format.
    """
    rows = []
    for i in range(n):
        rows.append(
            {
                "LOC_X": 0,
                "LOC_Y": 50,
                "PLAYER_ID": 100 + i,
                "GAME_ID": "001",
                "TEAM_ID": 1,
                "PERIOD": 1,
                "MINUTES_REMAINING": 11,
                "SECONDS_REMAINING": 0,
                "SHOT_MADE_FLAG": 0,
                "GAME_DATE": 20241115 if use_integer_game_date else "2024-11-15",
                "SHOT_ZONE_BASIC": "Mid-Range",
                "SHOT_ZONE_AREA": "Center(C)",
            }
        )
    df = pd.DataFrame(rows)
    if include_lowercase_game_date:
        df["game_date"] = "2024-11-15"
    return df


def test_date_integer_yyyymmdd_parses_correctly(tmp_path: Path) -> None:
    """GAME_DATE as int 20241115 must parse to 2024-11-15, not 1970-01-01."""
    df_in = _nba_stats_synthetic_for_dates(n=5, use_integer_game_date=True)
    path = _write_csv(df_in, tmp_path)
    df_out = load_shots(path, drop_backcourt=False)
    assert df_out["date"].dtype.kind == "M"
    assert df_out["date"].min().year == 2024
    assert df_out["date"].iloc[0] == pd.Timestamp("2024-11-15")


def test_date_string_iso_still_parses(tmp_path: Path) -> None:
    """``GAME_DATE`` as an ISO string parses correctly."""
    df_in = _nba_stats_synthetic_for_dates(n=5, use_integer_game_date=False)
    path = _write_csv(df_in, tmp_path)
    df_out = load_shots(path, drop_backcourt=False)
    assert df_out["date"].iloc[0] == pd.Timestamp("2024-11-15")


def test_loader_prefers_lowercase_game_date_over_integer_uppercase(tmp_path: Path) -> None:
    """When both integer ``GAME_DATE`` and string ``game_date`` are present, the string wins."""
    df_in = _nba_stats_synthetic_for_dates(
        n=5, use_integer_game_date=True, include_lowercase_game_date=True
    )
    path = _write_csv(df_in, tmp_path)
    df_out = load_shots(path, drop_backcourt=False)
    # If the loader had used the integer GAME_DATE, dates would all be 1970-01-01.
    # With the lowercase preference the parse uses the string form.
    assert df_out["date"].iloc[0] == pd.Timestamp("2024-11-15")


# ---------------------------------------------------------------------------
# File-format errors
# ---------------------------------------------------------------------------


def test_unknown_extension_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "shots.txt"
    bogus.write_text("ignore me")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_shots(bogus)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_shots(tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# Real-data smoke test (skipped when the shot data is not on disk)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SHOT_FLOW_CSV.exists(), reason="shot data not available")
def test_load_real_shot_flow_csv_sample() -> None:
    """The first 5,000 rows of the real NBA dataset load with the canonical schema."""
    df = load_shots(SHOT_FLOW_CSV, nrows=5_000)

    assert {"x", "y", "player_id", "date"}.issubset(df.columns)
    # After backcourt drop and NaN drop, we should have most rows.
    assert 4_000 <= len(df) <= 5_000
    # Coordinates are in feet, not tenths.
    assert df["x"].abs().max() <= 30.0
    assert df["y"].max() <= 50.0
    # Dates parsed.
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_home_away_derived_from_team_id_and_htm(tmp_path: Path) -> None:
    """``home_away`` is 1 when the row's ``TEAM_ID`` maps to ``HTM`` and 0 otherwise.

    The synthetic game is BOS (1610612738, visitor) at LAL (1610612747, home);
    half the rows are BOS shots and half LAL shots.
    """
    n = 20
    df_in = _nba_stats_synthetic(n)
    # Override team / HTM / VTM with a known pair.
    df_in["TEAM_ID"] = np.array([1610612738, 1610612747] * (n // 2))
    df_in["HTM"] = ["LAL"] * n
    df_in["VTM"] = ["BOS"] * n
    df_in["TEAM_NAME"] = np.where(
        df_in["TEAM_ID"] == 1610612738, "Boston Celtics", "Los Angeles Lakers"
    )

    path = _write_csv(df_in, tmp_path)
    df_out = load_shots(path, drop_backcourt=False)

    assert "home_away" in df_out.columns
    # BOS rows (TEAM_ID 1610612738) → home_away == 0; LAL rows → home_away == 1.
    bos_mask = df_out["team"] == 1610612738
    assert (df_out.loc[bos_mask, "home_away"] == 0).all()
    assert (df_out.loc[~bos_mask, "home_away"] == 1).all()
