"""Tests for :mod:`shotcloud.data.splits`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shotcloud.data.splits import split_by_season, split_fractional


def _df_with_seasons(n_per_season: int = 10) -> pd.DataFrame:
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    rows = []
    for s in seasons:
        for i in range(n_per_season):
            rows.append({"x": float(i), "y": float(i), "season": s, "game_id": f"{s}-{i // 5}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# split_by_season
# ---------------------------------------------------------------------------


def test_split_by_season_partitions_correctly() -> None:
    df = _df_with_seasons(10)
    train, val, test = split_by_season(
        df,
        train_seasons=["2021-22", "2022-23"],
        val_seasons=["2023-24"],
        test_seasons=["2024-25"],
    )
    assert len(train) == 20
    assert len(val) == 10
    assert len(test) == 10
    assert set(train["season"]) == {"2021-22", "2022-23"}
    assert set(val["season"]) == {"2023-24"}
    assert set(test["season"]) == {"2024-25"}


def test_split_by_season_overlap_raises() -> None:
    df = _df_with_seasons(5)
    with pytest.raises(ValueError, match="overlap"):
        split_by_season(
            df,
            train_seasons=["2022-23"],
            val_seasons=["2022-23"],
            test_seasons=["2024-25"],
        )


def test_split_by_season_missing_column_raises() -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    with pytest.raises(KeyError, match="season"):
        split_by_season(df, train_seasons=[], val_seasons=[], test_seasons=[])


def test_split_by_season_drops_unlisted_seasons() -> None:
    df = _df_with_seasons(5)
    train, val, test = split_by_season(
        df,
        train_seasons=["2021-22"],
        val_seasons=["2022-23"],
        test_seasons=[],  # 2023-24 and 2024-25 dropped
    )
    assert len(train) == 5
    assert len(val) == 5
    assert len(test) == 0


# ---------------------------------------------------------------------------
# split_fractional — row-level
# ---------------------------------------------------------------------------


def test_split_fractional_partitions_to_correct_sizes() -> None:
    df = pd.DataFrame({"x": np.arange(100)})
    train, val, test = split_fractional(df, train_frac=0.7, val_frac=0.2, test_frac=0.1)
    assert len(train) == 70
    assert len(val) == 20
    assert len(test) == 10


def test_split_fractional_is_deterministic_under_seed() -> None:
    df = pd.DataFrame({"x": np.arange(100)})
    a1, a2, a3 = split_fractional(df, seed=7)
    b1, b2, b3 = split_fractional(df, seed=7)
    pd.testing.assert_frame_equal(a1, b1)
    pd.testing.assert_frame_equal(a2, b2)
    pd.testing.assert_frame_equal(a3, b3)


def test_split_fractional_different_seeds_give_different_splits() -> None:
    df = pd.DataFrame({"x": np.arange(100)})
    a, _, _ = split_fractional(df, seed=1)
    b, _, _ = split_fractional(df, seed=2)
    # Same length, but different content.
    assert not (a["x"].to_numpy() == b["x"].to_numpy()).all()


def test_split_fractional_no_overlap() -> None:
    df = pd.DataFrame({"x": np.arange(100)})
    train, val, test = split_fractional(df)
    assert set(train["x"]).isdisjoint(set(val["x"]))
    assert set(train["x"]).isdisjoint(set(test["x"]))
    assert set(val["x"]).isdisjoint(set(test["x"]))


def test_split_fractional_fraction_sum_must_be_one() -> None:
    df = pd.DataFrame({"x": np.arange(10)})
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        split_fractional(df, train_frac=0.5, val_frac=0.3, test_frac=0.3)


def test_split_fractional_negative_or_too_large_raises() -> None:
    df = pd.DataFrame({"x": np.arange(10)})
    with pytest.raises(ValueError, match="must be in"):
        split_fractional(df, train_frac=-0.1, val_frac=0.5, test_frac=0.6)
    with pytest.raises(ValueError, match="must be in"):
        split_fractional(df, train_frac=1.5, val_frac=-0.5, test_frac=0.0)


# ---------------------------------------------------------------------------
# split_fractional — group-level
# ---------------------------------------------------------------------------


def test_split_fractional_group_keeps_groups_intact() -> None:
    df = _df_with_seasons(10)
    train, val, test = split_fractional(
        df, train_frac=0.7, val_frac=0.2, test_frac=0.1, group_column="game_id"
    )
    # No game_id appears in more than one split.
    train_games = set(train["game_id"])
    val_games = set(val["game_id"])
    test_games = set(test["game_id"])
    assert train_games.isdisjoint(val_games)
    assert train_games.isdisjoint(test_games)
    assert val_games.isdisjoint(test_games)


def test_split_fractional_group_missing_column_raises() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(KeyError, match="game_id"):
        split_fractional(df, group_column="game_id")
