"""Tests for :func:`shotcloud.data.compute_player_timing_history_features`.

The featurizer is causal at game-date granularity: a shot on date ``D`` sees only
the same player's shots from games dated strictly before ``D``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shotcloud.data.timing_history import (
    TIMING_HISTORY_DIM,
    compute_player_timing_history_features,
)


def _hand_checkable_df() -> pd.DataFrame:
    """One player, three games on three distinct dates.

    G1 (2024-01-05): shots at game-minute 5 and 30.
    G2 (2024-01-10): shots at game-minute 12 and 20.
    G3 (2024-01-15): one shot at game-minute 0.
    """
    return pd.DataFrame(
        {
            "player_id": [101] * 5,
            "date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-05",
                    "2024-01-10",
                    "2024-01-10",
                    "2024-01-15",
                ]
            ),
            # The featurizer ignores period and reads time_remaining_sec as total
            # elapsed seconds.
            "period": [1, 3, 2, 2, 1],
            "time_remaining_sec": [5 * 60, 30 * 60, 12 * 60, 20 * 60, 0],
        }
    )


def test_dim_constant_is_48() -> None:
    assert TIMING_HISTORY_DIM == 48


def test_first_game_no_prior_zero_with_no_smoothing() -> None:
    df = _hand_checkable_df()
    out = compute_player_timing_history_features(df, smoothing=0.0)
    np.testing.assert_array_equal(out[0], np.zeros(TIMING_HISTORY_DIM, dtype=np.float32))
    np.testing.assert_array_equal(out[1], np.zeros(TIMING_HISTORY_DIM, dtype=np.float32))


def test_first_game_no_prior_uniform_with_smoothing() -> None:
    df = _hand_checkable_df()
    out = compute_player_timing_history_features(df, smoothing=1.0)
    expected = np.full(TIMING_HISTORY_DIM, 1.0 / TIMING_HISTORY_DIM, dtype=np.float32)
    np.testing.assert_allclose(out[0], expected, atol=1e-6)
    np.testing.assert_allclose(out[1], expected, atol=1e-6)


def test_second_game_history_only_from_first_game() -> None:
    df = _hand_checkable_df()
    out = compute_player_timing_history_features(df, smoothing=0.0)
    # Game 2 shots see only Game 1's shots (bins 5 and 30; 0.5 each).
    assert out[2, 5] == pytest.approx(0.5, abs=1e-5)
    assert out[2, 30] == pytest.approx(0.5, abs=1e-5)
    # All other bins are zero.
    mass_elsewhere = out[2].sum() - out[2, 5] - out[2, 30]
    assert mass_elsewhere == pytest.approx(0.0, abs=1e-5)


def test_third_game_history_aggregates_both_prior_games() -> None:
    df = _hand_checkable_df()
    out = compute_player_timing_history_features(df, smoothing=0.0)
    # Game 3 sees G1 (bins 5, 30) + G2 (bins 12, 20) = 4 bins each 0.25.
    for b in (5, 12, 20, 30):
        assert out[4, b] == pytest.approx(0.25, abs=1e-5), f"bin {b}"
    mass_elsewhere = out[4].sum() - sum(out[4, b] for b in (5, 12, 20, 30))
    assert mass_elsewhere == pytest.approx(0.0, abs=1e-5)


def test_same_date_shots_do_not_leak_into_each_other() -> None:
    """Shots on the same date (the same game) do not enter each other's history; the
    strict cutoff is at day granularity."""
    df = pd.DataFrame(
        {
            "player_id": [101, 101],
            "date": pd.to_datetime(["2024-01-05", "2024-01-05"]),
            "period": [1, 3],
            "time_remaining_sec": [5 * 60, 30 * 60],
        }
    )
    out = compute_player_timing_history_features(df, smoothing=0.0)
    # Both shots have no prior games; both rows must be all-zero.
    np.testing.assert_array_equal(out[0], np.zeros(TIMING_HISTORY_DIM, dtype=np.float32))
    np.testing.assert_array_equal(out[1], np.zeros(TIMING_HISTORY_DIM, dtype=np.float32))


def test_different_players_do_not_share_history() -> None:
    df = pd.DataFrame(
        {
            "player_id": [101, 101, 102, 102],
            "date": pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-05", "2024-01-10"]),
            "period": [1, 1, 1, 1],
            "time_remaining_sec": [5 * 60, 12 * 60, 30 * 60, 20 * 60],
        }
    )
    out = compute_player_timing_history_features(df, smoothing=0.0)
    # Player 101's second shot sees bin 5 from its own first game, not bin 30 from
    # player 102's.
    assert out[1, 5] == pytest.approx(1.0, abs=1e-5)
    assert out[1, 30] == pytest.approx(0.0, abs=1e-5)
    # Player 102's second shot sees bin 30 from its own first game, not bin 5.
    assert out[3, 30] == pytest.approx(1.0, abs=1e-5)
    assert out[3, 5] == pytest.approx(0.0, abs=1e-5)


def test_empty_df_returns_empty_array() -> None:
    df = pd.DataFrame({col: [] for col in ("player_id", "date", "period", "time_remaining_sec")})
    out = compute_player_timing_history_features(df)
    assert out.shape == (0, TIMING_HISTORY_DIM)


def test_validation_error_on_missing_column() -> None:
    df = _hand_checkable_df().drop(columns=["period"])
    with pytest.raises(KeyError, match="period"):
        compute_player_timing_history_features(df)
