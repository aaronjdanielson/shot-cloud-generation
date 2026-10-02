"""Tests for ``shotcloud.data.within_game_history.compute_within_game_features``.

Load-bearing invariants:

* **Causality.** A shot is only summarized over shots strictly before
  it within the same ``(player, game)`` group.
* **First-shot edge case.** The first shot of a player-game has an
  all-zero feature vector — including ``mask_has_history``.
* **Group isolation.** Different ``(player, game)`` pairs do not leak
  shots into each other's prefix-stats.
* **Row alignment.** Output is in the input row order, not the
  internal sorted order.
* **Zone arithmetic.** ``frac_rim``, ``frac_mid``, ``frac_corner``,
  ``frac_3pa`` agree with the underlying ``ZONE_NAMES`` definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shotcloud.data.within_game_history import (
    WITHIN_GAME_DIM,
    WITHIN_GAME_SLOT_NAMES,
    compute_within_game_features,
)


def _slot(name: str) -> int:
    return WITHIN_GAME_SLOT_NAMES.index(name)


# ---------------------------------------------------------------------------
# Layout sanity
# ---------------------------------------------------------------------------


def test_dim_and_slot_names_round_trip() -> None:
    assert WITHIN_GAME_DIM == 10
    assert len(WITHIN_GAME_SLOT_NAMES) == WITHIN_GAME_DIM
    assert WITHIN_GAME_SLOT_NAMES[0] == "n_prior_log1p"
    assert WITHIN_GAME_SLOT_NAMES[-1] == "mask_has_history"


def test_empty_dataframe_returns_zero_rows() -> None:
    df = pd.DataFrame({"x": [], "y": [], "player_id": [], "game_id": [], "time_remaining_sec": []})
    feats = compute_within_game_features(df)
    assert feats.shape == (0, WITHIN_GAME_DIM)
    assert feats.dtype == np.float32


def test_missing_column_raises() -> None:
    df = pd.DataFrame({"x": [0.0], "y": [0.0], "player_id": [1], "game_id": ["g"]})
    with pytest.raises(KeyError, match="time_remaining_sec"):
        compute_within_game_features(df)


# ---------------------------------------------------------------------------
# Causality + first-shot edge case
# ---------------------------------------------------------------------------


def test_first_shot_of_player_game_has_all_zeros_including_mask() -> None:
    """A player's first shot of a game has no prior history — every
    feature should be zero, including the ``mask_has_history`` slot
    so downstream gating works on a single flag."""
    df = pd.DataFrame(
        [
            {"x": 0.0, "y": 5.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 60},
            {"x": 1.0, "y": 5.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 120},
            {"x": -1.0, "y": 5.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 180},
        ]
    )
    feats = compute_within_game_features(df)
    # Row 0 is the first shot for player 1 in game g1.
    np.testing.assert_allclose(feats[0], np.zeros(WITHIN_GAME_DIM, dtype=np.float32))
    # Rows 1, 2 have priors → mask is 1.
    assert feats[1, _slot("mask_has_history")] == 1.0
    assert feats[2, _slot("mask_has_history")] == 1.0


def test_n_prior_log1p_grows_with_within_game_shot_index() -> None:
    """``n_prior`` is the count of strictly-earlier shots. For 5
    sequential shots, n_prior takes 0, 1, 2, 3, 4 → log1p of those.
    """
    df = pd.DataFrame(
        [
            {
                "x": 0.0,
                "y": 5.0,
                "player_id": 1,
                "game_id": "g1",
                "time_remaining_sec": 60 * (i + 1),
            }
            for i in range(5)
        ]
    )
    feats = compute_within_game_features(df)
    expected_log1p = np.log1p(np.arange(5, dtype=np.float64))
    np.testing.assert_allclose(
        feats[:, _slot("n_prior_log1p")], expected_log1p.astype(np.float32), rtol=1e-6
    )


# ---------------------------------------------------------------------------
# Group isolation
# ---------------------------------------------------------------------------


def test_two_player_games_do_not_leak_into_each_other() -> None:
    df = pd.DataFrame(
        [
            {"x": 0.0, "y": 5.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 60},
            {"x": 0.0, "y": 5.0, "player_id": 2, "game_id": "g1", "time_remaining_sec": 90},
            {"x": 0.0, "y": 5.0, "player_id": 1, "game_id": "g2", "time_remaining_sec": 120},
            {"x": 0.0, "y": 5.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 150},
        ]
    )
    feats = compute_within_game_features(df)
    # Row 1 (player 2, game g1, only shot) → all zero.
    np.testing.assert_allclose(feats[1], np.zeros(WITHIN_GAME_DIM, dtype=np.float32))
    # Row 2 (player 1, game g2, only shot in g2) → all zero.
    np.testing.assert_allclose(feats[2], np.zeros(WITHIN_GAME_DIM, dtype=np.float32))
    # Row 3 (player 1, game g1, second shot in g1) → mask = 1, n_prior = 1.
    assert feats[3, _slot("mask_has_history")] == 1.0
    assert feats[3, _slot("n_prior_log1p")] == pytest.approx(float(np.log1p(1)))


# ---------------------------------------------------------------------------
# Zone fractions
# ---------------------------------------------------------------------------


def test_zone_fractions_match_known_priors() -> None:
    """Build a 4-shot history with known zones: 2 rim shots and 2
    above-break 3s. The 5th shot's prior fractions must be exactly
    ``frac_rim = 0.5``, ``frac_3pa = 0.5``, ``frac_mid = 0``,
    ``frac_corner = 0``.

    Coordinates: (0, 2) is rim (RA, dist < 4), (0, 30) is above-break 3.
    """
    df = pd.DataFrame(
        [
            # 2 rim shots:
            {"x": 0.0, "y": 2.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 60},
            {"x": 0.0, "y": 2.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 120},
            # 2 above-break 3s:
            {"x": 0.0, "y": 30.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 180},
            {"x": 0.0, "y": 30.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 240},
            # The shot we're asking about:
            {"x": 5.0, "y": 15.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 300},
        ]
    )
    feats = compute_within_game_features(df)
    assert feats[4, _slot("frac_rim")] == pytest.approx(0.5)
    assert feats[4, _slot("frac_3pa")] == pytest.approx(0.5)
    assert feats[4, _slot("frac_mid")] == pytest.approx(0.0)
    assert feats[4, _slot("frac_corner")] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Recent-mean coords, dt_prev, density
# ---------------------------------------------------------------------------


def test_mean_xy_recent_uses_last_three_prior_shots() -> None:
    """With 5 prior shots at known coords, the 6th shot's recent-mean
    should use shots 3, 4, 5 (the last three priors)."""
    coords = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0)]
    df_rows = [
        {
            "x": x,
            "y": y,
            "player_id": 1,
            "game_id": "g1",
            "time_remaining_sec": 60 * (i + 1),
        }
        for i, (x, y) in enumerate(coords)
    ]
    df_rows.append(
        {"x": 99.0, "y": 99.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 60 * 10}
    )
    df = pd.DataFrame(df_rows)
    feats = compute_within_game_features(df)
    # Mean of last 3 priors = mean of shots 3, 4, 5 (1-indexed) →
    # (3, 4, 5) → mean 4; (4, 5, 6) → mean 5.
    assert feats[5, _slot("mean_x_recent3")] == pytest.approx((3.0 + 4.0 + 5.0) / 3)
    assert feats[5, _slot("mean_y_recent3")] == pytest.approx((4.0 + 5.0 + 6.0) / 3)


def test_dt_prev_min_is_capped_at_twelve() -> None:
    """A halftime gap (~15 min) between shots should be capped at 12
    min so it doesn't dominate the residual encoder."""
    df = pd.DataFrame(
        [
            {"x": 0.0, "y": 5.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 60},
            {"x": 0.0, "y": 5.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 60 + 900},
        ]
    )
    feats = compute_within_game_features(df)
    assert feats[1, _slot("dt_prev_min")] == pytest.approx(12.0)


def test_shot_density_is_n_prior_over_elapsed_minutes_plus_one() -> None:
    """5 prior shots, 6th shot at minute 6 elapsed → density = 5 / 7."""
    df = pd.DataFrame(
        [
            {
                "x": 0.0,
                "y": 5.0,
                "player_id": 1,
                "game_id": "g1",
                "time_remaining_sec": 60 * (i + 1),
            }
            for i in range(5)
        ]
        + [{"x": 0.0, "y": 5.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 60 * 6}]
    )
    feats = compute_within_game_features(df)
    # 5 prior shots, elapsed at row 5 = 6 minutes → density = 5 / 7.
    assert feats[5, _slot("shot_density")] == pytest.approx(5.0 / 7.0)


# ---------------------------------------------------------------------------
# Row alignment + non-monotone input order
# ---------------------------------------------------------------------------


def test_output_row_order_matches_input_dataframe_row_order() -> None:
    """The featurizer sorts internally by (player, game, time) but
    must return rows in the input DataFrame's original index order."""
    # Build out-of-order: shot 2 (later in time) comes before shot 1.
    df = pd.DataFrame(
        [
            {"x": 0.0, "y": 5.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 200},
            {"x": 0.0, "y": 5.0, "player_id": 1, "game_id": "g1", "time_remaining_sec": 100},
        ]
    )
    feats = compute_within_game_features(df)
    # Row 0 = the later shot (t=200); it has 1 prior.
    assert feats[0, _slot("mask_has_history")] == 1.0
    assert feats[0, _slot("n_prior_log1p")] == pytest.approx(float(np.log1p(1)))
    # Row 1 = the earlier shot (t=100); first in the game, all zero.
    np.testing.assert_allclose(feats[1], np.zeros(WITHIN_GAME_DIM, dtype=np.float32))
