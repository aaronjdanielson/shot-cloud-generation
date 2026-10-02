"""Tests for :mod:`shotcloud.data.game_logs`.

Covers game-log and starter-file loading, the minutes-based starter
heuristic, the left join onto shots, and the causal recency-weighted
features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shotcloud.data import (
    GAME_LOG_COLUMNS,
    STARTER_MINUTES_THRESHOLD,
    join_game_logs,
    load_game_logs,
    load_starters,
)


def _write_synthetic_game_logs(tmp_path: Path) -> Path:
    """Write a minimal CSV in the ``player_game_logs.csv`` schema."""
    df = pd.DataFrame(
        {
            "player_id": [201583, 201583, 1628983, 1629029],
            "season": ["2014-15", "2014-15", "2024-25", "2024-25"],
            "player_name": ["Ryan Anderson", "Ryan Anderson", "SGA", "Luka"],
            "team_name": ["NOP", "NOP", "OKC", "DAL"],
            "game_id": [21400001, 21400002, 22400018, 22400019],
            "game_date": ["2014-10-28", "2014-10-30", "2024-11-15", "2024-11-16"],
            "minutes": [22, 8, 35, 0],
            "fgm": [9, 2, 12, 0],
            "fga": [22, 5, 25, 0],
        }
    )
    csv_path = tmp_path / "player_game_logs.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_game_logs_returns_canonical_schema(tmp_path: Path) -> None:
    csv_path = _write_synthetic_game_logs(tmp_path)
    df = load_game_logs(csv_path)
    for col in GAME_LOG_COLUMNS:
        assert col in df.columns, f"missing canonical column {col}"
    assert df["player_id"].dtype == np.int64
    assert df["game_id"].dtype == np.int64
    assert df["minutes"].dtype == np.int64
    assert df["starter"].dtype == np.int64


def test_load_game_logs_derives_starter_from_minutes(tmp_path: Path) -> None:
    csv_path = _write_synthetic_game_logs(tmp_path)
    df = load_game_logs(csv_path)
    # Threshold is 20; minutes=[22, 8, 35, 0] -> starter=[1, 0, 1, 0].
    starter_by_minutes = dict(zip(df["minutes"], df["starter"], strict=False))
    assert starter_by_minutes[22] == 1
    assert starter_by_minutes[35] == 1
    assert starter_by_minutes[8] == 0
    assert starter_by_minutes[0] == 0


def test_load_game_logs_starter_threshold_boundary(tmp_path: Path) -> None:
    """Minutes exactly at the threshold count as a start."""
    csv_path = tmp_path / "boundary.csv"
    pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "game_id": [10, 11, 12],
            "minutes": [
                STARTER_MINUTES_THRESHOLD - 1,
                STARTER_MINUTES_THRESHOLD,
                STARTER_MINUTES_THRESHOLD + 1,
            ],
        }
    ).to_csv(csv_path, index=False)
    df = load_game_logs(csv_path)
    starter = df.set_index("player_id")["starter"]
    assert starter[1] == 0
    assert starter[2] == 1
    assert starter[3] == 1


def test_load_game_logs_drops_duplicate_player_game(tmp_path: Path) -> None:
    """Duplicate ``(player_id, game_id)`` rows collapse to the first occurrence."""
    csv_path = tmp_path / "dups.csv"
    pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "game_id": [10, 10, 11],
            "minutes": [25, 99, 30],  # second row should be dropped
        }
    ).to_csv(csv_path, index=False)
    df = load_game_logs(csv_path)
    assert len(df) == 2
    minutes_for_pid1_g10 = df[(df["player_id"] == 1) & (df["game_id"] == 10)]["minutes"].iloc[0]
    assert minutes_for_pid1_g10 == 25  # kept the first


def test_load_game_logs_handles_missing_minutes_as_zero(tmp_path: Path) -> None:
    """Missing minutes load as 0 minutes and a non-starter."""
    csv_path = tmp_path / "nan_minutes.csv"
    pd.DataFrame(
        {
            "player_id": [1, 2],
            "game_id": [10, 11],
            "minutes": [25.0, np.nan],
        }
    ).to_csv(csv_path, index=False)
    df = load_game_logs(csv_path)
    minutes_by_pid = dict(zip(df["player_id"], df["minutes"], strict=False))
    starter_by_pid = dict(zip(df["player_id"], df["starter"], strict=False))
    assert minutes_by_pid[2] == 0
    assert starter_by_pid[2] == 0


def test_load_game_logs_rejects_missing_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"player_id": [1], "minutes": [25]}).to_csv(csv_path, index=False)  # no game_id
    with pytest.raises(ValueError, match="missing required columns"):
        load_game_logs(csv_path)


def _synthetic_shots_frame() -> pd.DataFrame:
    """Minimal shots frame with the canonical ``load_shots`` columns."""
    return pd.DataFrame(
        {
            "x": [-3.0, 5.0, 0.0, 8.0],
            "y": [10.0, 23.0, 5.0, 18.0],
            "player_id": [1, 1, 2, 99],  # player 99 is absent from game logs
            "game_id": [100, 101, 100, 100],
            "date": pd.to_datetime(["2024-11-01", "2024-11-02", "2024-11-01", "2024-11-01"]),
        }
    )


def _synthetic_game_logs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "game_id": [100, 101, 100],
            "minutes": [32, 28, 12],
            "starter": [1, 1, 0],
        }
    )


def test_join_game_logs_attaches_minutes_and_starter() -> None:
    shots = _synthetic_shots_frame()
    gl = _synthetic_game_logs_frame()
    joined = join_game_logs(shots, gl)
    assert "minutes" in joined.columns
    assert "starter" in joined.columns
    # The first shot is player 1 in game 100 (32 min, starter=1).
    row = joined.iloc[0]
    assert row["minutes"] == 32
    assert row["starter"] == 1
    # The third shot is player 2 in game 100 (12 min, starter=0).
    row2 = joined.iloc[2]
    assert row2["minutes"] == 12
    assert row2["starter"] == 0


def test_join_game_logs_imputes_missing_with_zero() -> None:
    """Shots with no game-log row are imputed as 0 minutes and a non-starter."""
    shots = _synthetic_shots_frame()
    gl = _synthetic_game_logs_frame()
    joined = join_game_logs(shots, gl)
    missing_row = joined[joined["player_id"] == 99].iloc[0]
    assert missing_row["minutes"] == 0
    assert missing_row["starter"] == 0


def test_join_game_logs_can_disable_imputation() -> None:
    """With ``impute_missing=False``, unmatched shots get NaN."""
    shots = _synthetic_shots_frame()
    gl = _synthetic_game_logs_frame()
    joined = join_game_logs(shots, gl, impute_missing=False)
    missing_row = joined[joined["player_id"] == 99].iloc[0]
    assert pd.isna(missing_row["minutes"])
    assert pd.isna(missing_row["starter"])


def test_join_game_logs_preserves_shot_order_and_rowcount() -> None:
    """The left join neither drops nor reorders shot rows."""
    shots = _synthetic_shots_frame()
    gl = _synthetic_game_logs_frame()
    joined = join_game_logs(shots, gl)
    assert len(joined) == len(shots)
    # The original 'x' column ordering should be preserved.
    np.testing.assert_array_equal(joined["x"].values, shots["x"].values)


def test_join_game_logs_rejects_shots_without_game_id() -> None:
    shots = pd.DataFrame({"x": [1.0], "y": [2.0], "player_id": [1]})  # no game_id
    gl = _synthetic_game_logs_frame()
    with pytest.raises(ValueError, match="must have 'game_id'"):
        join_game_logs(shots, gl)


# ---------------------------------------------------------------------------
# Observed starters (scripts/fetch_starters.py output)
# ---------------------------------------------------------------------------


def _write_synthetic_starters(tmp_path: Path) -> Path:
    """Write a starters CSV in the ``scripts/fetch_starters.py`` output schema."""
    df = pd.DataFrame(
        {
            "game_id": [21400001, 21400001, 21400002, 22400018],
            "player_id": [201583, 999999, 201583, 1628983],
            "position": ["F", "", "", "G"],
            "starter": [1, 0, 0, 1],
        }
    )
    csv_path = tmp_path / "starters.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_starters_returns_canonical_schema(tmp_path: Path) -> None:
    csv_path = _write_synthetic_starters(tmp_path)
    df = load_starters(csv_path)
    assert list(df.columns) == ["game_id", "player_id", "position", "starter"]
    assert df["game_id"].dtype == np.int64
    assert df["player_id"].dtype == np.int64
    assert df["starter"].dtype == np.int64


def test_load_starters_drops_duplicates_keeping_last(tmp_path: Path) -> None:
    """Duplicate ``(game_id, player_id)`` rows keep the last occurrence."""
    csv_path = tmp_path / "dups.csv"
    pd.DataFrame(
        {
            "game_id": [10, 10],
            "player_id": [1, 1],
            "position": ["F", "G"],  # later row should win
            "starter": [1, 1],
        }
    ).to_csv(csv_path, index=False)
    df = load_starters(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["position"] == "G"


def test_load_starters_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"game_id": [1], "player_id": [2]}).to_csv(csv_path, index=False)  # no starter
    with pytest.raises(ValueError, match="missing required columns"):
        load_starters(csv_path)


def test_load_game_logs_overrides_starter_with_real_position(tmp_path: Path) -> None:
    """Observed starter status overrides the minutes heuristic when they disagree."""
    # Game logs in which the minutes heuristic misclassifies two players.
    gl_path = tmp_path / "gl.csv"
    pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "game_id": [10, 10, 10],
            # Player 1: minutes=18 → heuristic bench, observed starter (e.g. foul-out)
            # Player 2: minutes=22 → heuristic starter, observed bench (high-minute reserve)
            # Player 3: minutes=10 → bench under both (no override)
            "minutes": [18, 22, 10],
        }
    ).to_csv(gl_path, index=False)

    starters_path = tmp_path / "starters.csv"
    pd.DataFrame(
        {
            "game_id": [10, 10],
            "player_id": [1, 2],
            "position": ["G", ""],
            "starter": [1, 0],
        }
    ).to_csv(starters_path, index=False)

    df = load_game_logs(gl_path, starters_path=starters_path)
    by_pid = df.set_index("player_id")
    # Player 1: observed starter (overrides heuristic of 0 since minutes < 20)
    assert by_pid.loc[1, "starter"] == 1
    assert by_pid.loc[1, "starter_source"] == "position"
    # Player 2: observed bench (overrides heuristic of 1 since minutes >= 20)
    assert by_pid.loc[2, "starter"] == 0
    assert by_pid.loc[2, "starter_source"] == "position"
    # Player 3: not in the starters file; heuristic fallback (bench, since minutes < 20)
    assert by_pid.loc[3, "starter"] == 0
    assert by_pid.loc[3, "starter_source"] == "minutes"


def test_load_game_logs_without_starters_path_uses_minutes_heuristic(tmp_path: Path) -> None:
    """Without ``starters_path``, starter status comes from the minutes heuristic alone."""
    csv_path = _write_synthetic_game_logs(tmp_path)
    df = load_game_logs(csv_path)
    assert "starter_source" not in df.columns
    # minutes=[22, 8, 35, 0], threshold=20 → starter=[1, 0, 1, 0]
    by_minutes = dict(zip(df["minutes"], df["starter"], strict=False))
    assert by_minutes[STARTER_MINUTES_THRESHOLD + 2] == 1  # 22
    assert by_minutes[STARTER_MINUTES_THRESHOLD - 12] == 0  # 8


# ---------------------------------------------------------------------------
# Recency-weighted features (recent_3pa_frac, recent_usage, recent_fga)
# ---------------------------------------------------------------------------


def _write_recency_game_logs(tmp_path: Path) -> Path:
    """Write a three-game CSV for one player with the columns the recency pass needs."""
    df = pd.DataFrame(
        {
            "player_id": [1] * 3,
            "season": ["2024-25"] * 3,
            "player_name": ["A"] * 3,
            "team_name": ["BOS"] * 3,
            "game_id": [22400001, 22400002, 22400003],
            "game_date": ["2024-11-01", "2024-11-02", "2024-11-03"],
            "minutes": [30, 25, 35],
            "fgm": [5, 4, 6],
            "fga": [10, 8, 12],
            "fg3a": [2, 3, 5],
            "fta": [4, 2, 6],
            "tov": [1, 0, 2],
        }
    )
    csv_path = tmp_path / "recency_game_logs.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_game_logs_adds_recent_features(tmp_path: Path) -> None:
    """Recency features are NaN for a player's first game and use only prior games after."""
    csv_path = _write_recency_game_logs(tmp_path)
    df = load_game_logs(csv_path)
    assert {"recent_3pa_frac", "recent_usage", "recent_fga"}.issubset(df.columns)
    df = df.sort_values("game_id").reset_index(drop=True)

    # First game: no prior data → NaN.
    assert pd.isna(df.loc[0, "recent_3pa_frac"])
    assert pd.isna(df.loc[0, "recent_usage"])
    assert pd.isna(df.loc[0, "recent_fga"])

    # Second game's recent values reflect ONLY game 1 (with one day of decay).
    # The decay factor cancels in ratios:
    #   recent_3pa_frac = fg3a_1 / fga_1 = 2 / 10 = 0.2
    #   recent_fga      = fga_1 / 1     = 10.0
    # recent_usage = (fga_1 + 0.44*fta_1 + tov_1) / minutes_1
    #              = (10 + 1.76 + 1) / 30 = 0.4253...
    assert abs(df.loc[1, "recent_3pa_frac"] - 0.2) < 1e-9
    assert abs(df.loc[1, "recent_fga"] - 10.0) < 1e-9
    assert abs(df.loc[1, "recent_usage"] - (10 + 0.44 * 4 + 1) / 30) < 1e-9


def test_load_game_logs_recency_is_per_player_causal(tmp_path: Path) -> None:
    """Recency features for one player never draw on another player's games."""
    df = pd.DataFrame(
        {
            "player_id": [1, 2, 1, 2],
            "season": ["2024-25"] * 4,
            "player_name": ["A", "B", "A", "B"],
            "team_name": ["BOS", "LAL", "BOS", "LAL"],
            "game_id": [1001, 1002, 1003, 1004],
            "game_date": ["2024-11-01", "2024-11-01", "2024-11-02", "2024-11-02"],
            "minutes": [30, 30, 30, 30],
            "fgm": [5, 5, 5, 5],
            "fga": [10, 4, 12, 6],
            "fg3a": [2, 0, 5, 0],
            "fta": [4, 2, 4, 2],
            "tov": [1, 1, 1, 1],
        }
    )
    csv_path = tmp_path / "two_player.csv"
    df.to_csv(csv_path, index=False)
    out = load_game_logs(csv_path)
    out = out.sort_values(["player_id", "game_id"]).reset_index(drop=True)

    # First game per player → NaN.
    assert pd.isna(out.loc[0, "recent_3pa_frac"])  # player 1, game 1001
    assert pd.isna(out.loc[2, "recent_3pa_frac"])  # player 2, game 1002
    # Second game per player is computed from THAT player's prior only.
    # Player 1: 2/10 = 0.2; Player 2: 0/4 = 0.0.
    assert abs(out.loc[1, "recent_3pa_frac"] - 0.2) < 1e-9
    assert abs(out.loc[3, "recent_3pa_frac"] - 0.0) < 1e-9


def test_join_game_logs_propagates_recent_features() -> None:
    """``join_game_logs`` carries ``recent_*`` columns onto shots, imputing NaN as zero."""
    shots = pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "game_id": [1001, 1002, 1003],
            "x": [0.0, 1.0, 2.0],
            "y": [10.0, 11.0, 12.0],
        }
    )
    gl = pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "game_id": [1001, 1002, 1003],
            "minutes": [30, 30, 30],
            "starter": [1, 1, 1],
            "recent_3pa_frac": [np.nan, 0.2, 0.3],
            "recent_usage": [np.nan, 0.5, 0.6],
            "recent_fga": [np.nan, 10.0, 11.0],
        }
    )
    merged = join_game_logs(shots, gl)
    assert {"recent_3pa_frac", "recent_usage", "recent_fga"}.issubset(merged.columns)
    # First-game NaN imputed to 0.
    assert merged.loc[merged["game_id"] == 1001, "recent_3pa_frac"].iloc[0] == 0.0
    # Subsequent values pass through.
    assert merged.loc[merged["game_id"] == 1002, "recent_fga"].iloc[0] == 10.0


def test_load_game_logs_skips_recency_when_columns_missing(tmp_path: Path) -> None:
    """Without the ``fg3a``/``fta``/``tov`` columns, the recency pass is skipped."""
    csv_path = _write_synthetic_game_logs(tmp_path)
    df = load_game_logs(csv_path)
    # The synthetic CSV has fga and game_date but not fg3a/fta/tov.
    assert "recent_3pa_frac" not in df.columns
    assert "recent_usage" not in df.columns
    assert "recent_fga" not in df.columns


def test_load_game_logs_recency_disabled_with_negative_halflife(tmp_path: Path) -> None:
    """A non-positive ``recency_halflife_days`` disables the recency pass."""
    csv_path = _write_recency_game_logs(tmp_path)
    df = load_game_logs(csv_path, recency_halflife_days=0)
    assert "recent_3pa_frac" not in df.columns
