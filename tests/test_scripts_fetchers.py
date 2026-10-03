"""Tests for the offline parts of the NBA Stats fetchers in ``scripts/``.

Network calls are not exercised; these tests cover season enumeration, the
player-season target list, resume bookkeeping, clock arithmetic, and the
lineup carry-over of the play-by-play fetcher.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

pytest.importorskip("nba_api")
pytest.importorskip("pbpstats")

import fetch_game_logs  # noqa: E402
import fetch_play_by_play  # noqa: E402
import fetch_shots  # noqa: E402

# ---------------------------------------------------------------------------
# fetch_game_logs
# ---------------------------------------------------------------------------


def test_season_slugs_are_inclusive_and_ordered() -> None:
    assert fetch_game_logs.season_slugs("2014-15", "2016-17") == ["2014-15", "2015-16", "2016-17"]
    assert fetch_game_logs.season_slugs("2024-25", "2024-25") == ["2024-25"]


def test_season_slugs_reject_reversed_range() -> None:
    with pytest.raises(ValueError, match="precedes"):
        fetch_game_logs.season_slugs("2020-21", "2019-20")


def test_game_log_output_order_starts_with_keys() -> None:
    assert fetch_game_logs._OUTPUT_ORDER[:6] == [
        "player_id",
        "season",
        "player_name",
        "team_name",
        "game_id",
        "game_date",
    ]
    assert fetch_game_logs._OUTPUT_ORDER[-1] == "plusminus"


# ---------------------------------------------------------------------------
# fetch_shots
# ---------------------------------------------------------------------------


def _write_game_logs(path: Path) -> None:
    pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2, 3],
            "season": ["2023-24", "2023-24", "2023-24", "2024-25", "2024-25"],
            "fga": [10, 5, 0, 7, 1],
        }
    ).to_csv(path, index=False)


def test_player_seasons_sum_fga_and_apply_threshold(tmp_path: Path) -> None:
    """Targets are (player, season) pairs with summed FGA at or above the threshold."""
    path = tmp_path / "logs.csv"
    _write_game_logs(path)
    assert fetch_shots._player_seasons(path, min_fga=1) == [
        (1, "2023-24"),
        (2, "2024-25"),
        (3, "2024-25"),
    ]
    assert fetch_shots._player_seasons(path, min_fga=5) == [(1, "2023-24"), (2, "2024-25")]


def test_read_done_returns_pairs_already_written(tmp_path: Path) -> None:
    """Resume bookkeeping reads the distinct (player_id, season) pairs of a CSV."""
    path = tmp_path / "shots.csv"
    assert fetch_shots._read_done(path) == set()
    pd.DataFrame(
        {"player_id": [7, 7, 8], "season": ["2023-24", "2023-24", "2024-25"], "LOC_X": [0, 1, 2]}
    ).to_csv(path, index=False)
    assert fetch_shots._read_done(path) == {(7, "2023-24"), (8, "2024-25")}


def test_append_rows_writes_header_once(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    fetch_shots._append_rows(path, pd.DataFrame({"a": [1]}))
    fetch_shots._append_rows(path, pd.DataFrame({"a": [2]}))
    assert pd.read_csv(path)["a"].tolist() == [1, 2]


# ---------------------------------------------------------------------------
# fetch_play_by_play
# ---------------------------------------------------------------------------


def test_clock_parsing_accepts_both_formats() -> None:
    assert fetch_play_by_play.clock_to_seconds_remaining("PT11M58.00S") == pytest.approx(718.0)
    assert fetch_play_by_play.clock_to_seconds_remaining("11:58") == pytest.approx(718.0)


def test_seconds_into_game_handles_regulation_and_overtime() -> None:
    assert fetch_play_by_play.seconds_into_game(1, "12:00") == 0.0
    assert fetch_play_by_play.seconds_into_game(2, "12:00") == 720.0
    assert fetch_play_by_play.seconds_into_game(4, "0:00") == 2880.0
    assert fetch_play_by_play.seconds_into_game(5, "5:00") == 2880.0
    assert fetch_play_by_play.seconds_into_game(6, "4:00") == 2880.0 + 300.0 + 60.0


def test_parse_lineup() -> None:
    assert fetch_play_by_play._parse_lineup("1-2-3-4-5") == [1, 2, 3, 4, 5]
    assert fetch_play_by_play._parse_lineup("") == []


def test_fetch_game_carries_lineups_forward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Events without lineup data inherit the lineups of the previous event."""
    home, away = 100, 200
    events = [
        SimpleNamespace(
            period=1, clock="12:00", lineup_ids={home: "1-2-3-4-5", away: "6-7-8-9-10"}
        ),
        SimpleNamespace(period=1, clock="11:30", lineup_ids=None),
        SimpleNamespace(period=1, clock="10:00", lineup_ids={home: "1-2-3-4-11", away: ""}),
        SimpleNamespace(period=2, clock="12:00", lineup_ids={}),
    ]
    monkeypatch.setattr(
        fetch_play_by_play, "_game_summary", lambda game_id, *, timeout: ("2024-11-12", home, away)
    )
    monkeypatch.setattr(fetch_play_by_play, "_load_events", lambda game_id, cache_dir: events)

    df = fetch_play_by_play.fetch_game("0022400001", tmp_path)

    assert list(df.columns) == fetch_play_by_play._OUTPUT_COLUMNS
    assert df["game_date"].unique().tolist() == ["2024-11-12"]
    assert df["home_lineup"].tolist() == [
        "[1, 2, 3, 4, 5]",
        "[1, 2, 3, 4, 5]",
        "[1, 2, 3, 4, 11]",
        "[1, 2, 3, 4, 11]",
    ]
    assert df["away_lineup"].unique().tolist() == ["[6, 7, 8, 9, 10]"]
    assert df["seconds_into_game"].tolist() == [0.0, 30.0, 120.0, 720.0]


def test_fetch_game_rejects_empty_feed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        fetch_play_by_play, "_game_summary", lambda game_id, *, timeout: ("2024-11-12", 1, 2)
    )
    monkeypatch.setattr(fetch_play_by_play, "_load_events", lambda game_id, cache_dir: [])
    with pytest.raises(RuntimeError, match="no events"):
        fetch_play_by_play.fetch_game("0022400001", tmp_path)
