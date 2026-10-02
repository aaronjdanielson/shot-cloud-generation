"""Tests for :mod:`shotcloud.data.player_traits`, the per-(player, snapshot) trait builder.

Covers causal time filtering (snapshot t_m sees only data dated before t_m),
the missingness mechanic (Block B is exactly zero when m_play = 0), the
``(n_players, n_snapshots, TRAIT_DIM)`` shape contract, determinism, and
z-scoring of the biographical block: statistics from the players active
before the anchor, with NaN imputed to 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shotcloud.data.player_traits import (
    BLOCK_A_END,
    BLOCK_B_END,
    M_PLAY_SLOT,
    SLOT_NAMES,
    TRAIT_DIM,
    PlayerTraitsTable,
    _recency_aggregate_per_snapshot,
    build_player_traits_table,
)
from shotcloud.data.role_profile import build_role_profiles
from shotcloud.data.snapshots import build_snapshot_store_from_shots


def _synth_shots(player_id_to_n: dict[int, int], seed: int = 0) -> pd.DataFrame:
    """Build a synthetic shots frame with controllable per-player counts."""
    rng = np.random.default_rng(seed)
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for pid, n in player_id_to_n.items():
        for i in range(n):
            rows.append(
                {
                    "x": float(rng.normal(0, 5)),
                    "y": float(rng.normal(10, 5)),
                    "player_id": pid,
                    "opponent": "BOS",
                    "made": int(rng.random() < 0.5),
                    "period": (i % 4) + 1,
                    "time_remaining_sec": int(60 * (i % 48)),
                    "date": base_date + pd.Timedelta(days=i),
                }
            )
    return pd.DataFrame(rows)


def _synth_bio(player_ids: list[int]) -> pd.DataFrame:
    """Build a synthetic bio frame with one row per player_id."""
    positions = ["PG", "SG", "SF", "PF", "C"]
    rows = []
    for i, pid in enumerate(player_ids):
        rows.append(
            {
                "player_id": pid,
                "display_name": f"Player {pid}",
                "birthdate": pd.Timestamp("1990-01-15") + pd.Timedelta(days=i * 30),
                "height_inches": 72 + (i % 10),
                "weight_lbs": 190 + (i * 5),
                "position_raw": "Guard" if i % 2 == 0 else "Forward",
                "position_group": positions[i % 5],
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def _synth_game_logs(player_id_to_n_games: dict[int, int]) -> pd.DataFrame:
    """Build a synthetic game-logs frame with the columns the trait builder reads."""
    rng = np.random.default_rng(1)
    base_date = pd.Timestamp("2024-01-01")
    rows = []
    for pid, n in player_id_to_n_games.items():
        for i in range(n):
            rows.append(
                {
                    "player_id": pid,
                    "game_date": base_date + pd.Timedelta(days=i),
                    "minutes": int(rng.integers(15, 40)),
                    "fga": int(rng.integers(5, 25)),
                    "fta": int(rng.integers(0, 10)),
                    "tov": int(rng.integers(0, 5)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shape + slot-name contract
# ---------------------------------------------------------------------------


def test_trait_dim_matches_slot_names() -> None:
    assert TRAIT_DIM == len(SLOT_NAMES) == 26
    assert BLOCK_A_END == 8
    assert BLOCK_B_END == 25
    assert M_PLAY_SLOT == 25


def test_output_shape_matches_inputs() -> None:
    player_id_to_n = {1: 100, 2: 100, 3: 100}
    shots = _synth_shots(player_id_to_n)
    bio = _synth_bio([1, 2, 3])
    gl = _synth_game_logs({1: 50, 2: 50, 3: 50})
    anchors = [
        np.datetime64("2024-02-15", "D"),
        np.datetime64("2024-03-15", "D"),
        np.datetime64("2024-04-15", "D"),
    ]
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )

    table = build_player_traits_table(
        snapshot_store=store,
        vocab_ids=[1, 2, 3],
        bio_df=bio,
        game_logs_df=gl,
    )
    assert isinstance(table, PlayerTraitsTable)
    assert table.traits.shape == (3, 3, TRAIT_DIM)
    assert table.player_ids.tolist() == [1, 2, 3]
    assert len(table.snapshot_anchors) == 3
    # All values finite (no NaN propagated through).
    assert np.isfinite(table.traits).all()


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_causality_only_uses_pre_anchor_data() -> None:
    """A player whose first game follows the anchor is cold-start: Block B zero, m_play = 0."""
    # Player 1: games all in Feb. Player 2: games all in May.
    base = pd.Timestamp("2024-01-01")
    gl = pd.DataFrame(
        [
            *[
                {
                    "player_id": 1,
                    "game_date": base + pd.Timedelta(days=30 + i),
                    "minutes": 30,
                    "fga": 15,
                    "fta": 4,
                    "tov": 2,
                }
                for i in range(10)
            ],
            *[
                {
                    "player_id": 2,
                    "game_date": base + pd.Timedelta(days=120 + i),
                    "minutes": 30,
                    "fga": 15,
                    "fta": 4,
                    "tov": 2,
                }
                for i in range(10)
            ],
        ]
    )
    # Use synthetic shots only so the snapshot store builds (player 1 in
    # early Feb shots, player 2 in late May shots).
    shots = pd.concat(
        [
            _synth_shots({1: 50}, seed=0),
            _synth_shots({2: 50}, seed=1).assign(date=lambda d: d["date"] + pd.Timedelta(days=120)),
        ],
        ignore_index=True,
    )
    bio = _synth_bio([1, 2])
    anchors = [np.datetime64("2024-03-15", "D")]  # before player 2's first game
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )

    table = build_player_traits_table(
        snapshot_store=store,
        vocab_ids=[1, 2],
        bio_df=bio,
        game_logs_df=gl,
    )

    # Player 1 had pre-anchor games → m_play = 1, Block B nonzero.
    assert table.traits[0, 0, M_PLAY_SLOT] == 1.0
    assert table.traits[0, 0, BLOCK_A_END:BLOCK_B_END].sum() != 0

    # Player 2's first game is post-anchor → m_play = 0, Block B all zero.
    assert table.traits[1, 0, M_PLAY_SLOT] == 0.0
    np.testing.assert_array_equal(
        table.traits[1, 0, BLOCK_A_END:BLOCK_B_END],
        np.zeros(BLOCK_B_END - BLOCK_A_END, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Missingness mechanic
# ---------------------------------------------------------------------------


def test_cold_start_player_has_block_b_zero_and_m_play_zero() -> None:
    """A player with no FGA before the anchor has Block B = 0 and m_play = 0."""
    # Player 99 has bio but no shots, no game logs.
    shots = _synth_shots({1: 50})
    gl = _synth_game_logs({1: 20})
    bio = _synth_bio([1, 99])  # bio for both
    anchors = [np.datetime64("2024-03-15", "D")]
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )

    table = build_player_traits_table(
        snapshot_store=store,
        vocab_ids=[1, 99],
        bio_df=bio,
        game_logs_df=gl,
    )

    # Player 99 (cold-start): Block B is exactly zero, m_play = 0.
    np.testing.assert_array_equal(
        table.traits[1, 0, BLOCK_A_END:BLOCK_B_END],
        np.zeros(BLOCK_B_END - BLOCK_A_END, dtype=np.float32),
    )
    assert table.traits[1, 0, M_PLAY_SLOT] == 0.0
    # Player 99 STILL has Block A populated from bio (z-scored).
    # At least the position one-hot should be set.
    assert table.traits[1, 0, 3:BLOCK_A_END].sum() > 0


def test_active_player_has_block_b_nonzero_and_m_play_one() -> None:
    shots = _synth_shots({1: 50})
    gl = _synth_game_logs({1: 20})
    bio = _synth_bio([1])
    anchors = [np.datetime64("2024-03-15", "D")]
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )
    table = build_player_traits_table(
        snapshot_store=store,
        vocab_ids=[1],
        bio_df=bio,
        game_logs_df=gl,
    )
    assert table.traits[0, 0, M_PLAY_SLOT] == 1.0
    assert table.traits[0, 0, BLOCK_A_END:BLOCK_B_END].sum() != 0


# ---------------------------------------------------------------------------
# Recency aggregate helper
# ---------------------------------------------------------------------------


def test_recency_aggregate_only_includes_pre_ref_games() -> None:
    """Games on or after ref_date must not contribute to the aggregate."""
    gl = pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_date": pd.Timestamp("2024-01-01"),
                "minutes": 20,
                "fga": 10,
                "fta": 4,
                "tov": 2,
            },
            {
                "player_id": 1,
                "game_date": pd.Timestamp("2024-03-15"),  # exactly at ref → excluded
                "minutes": 30,
                "fga": 99,
                "fta": 0,
                "tov": 0,
            },
            {
                "player_id": 1,
                "game_date": pd.Timestamp("2024-05-01"),  # after ref → excluded
                "minutes": 30,
                "fga": 99,
                "fta": 0,
                "tov": 0,
            },
        ]
    )
    agg = _recency_aggregate_per_snapshot(gl, np.datetime64("2024-03-15", "D"), half_life_days=30.0)
    assert len(agg) == 1  # only player 1
    # The post-ref FGA=99 games must NOT appear in S (only the Jan 1 game does).
    # delta = (2024-03-15 - 2024-01-01) = 74 days (2024 is a leap year),
    # weight = 2^(-74/30) ≈ 0.181, S = weight * 10 FGA ≈ 1.81.
    delta_days = (pd.Timestamp("2024-03-15") - pd.Timestamp("2024-01-01")).days
    expected_S = 10.0 * 2.0 ** (-delta_days / 30.0)
    np.testing.assert_allclose(float(agg["S"].iloc[0]), expected_S, atol=0.005)


def test_recency_aggregate_empty_when_all_games_post_ref() -> None:
    gl = pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_date": pd.Timestamp("2024-12-01"),
                "minutes": 20,
                "fga": 10,
                "fta": 4,
                "tov": 2,
            },
        ]
    )
    agg = _recency_aggregate_per_snapshot(gl, np.datetime64("2024-03-15", "D"), half_life_days=30.0)
    assert agg.empty


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_builder_is_deterministic() -> None:
    shots = _synth_shots({1: 100, 2: 100})
    bio = _synth_bio([1, 2])
    gl = _synth_game_logs({1: 30, 2: 30})
    anchors = [np.datetime64("2024-03-15", "D")]
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )

    t1 = build_player_traits_table(
        snapshot_store=store, vocab_ids=[1, 2], bio_df=bio, game_logs_df=gl
    )
    t2 = build_player_traits_table(
        snapshot_store=store, vocab_ids=[1, 2], bio_df=bio, game_logs_df=gl
    )
    np.testing.assert_array_equal(t1.traits, t2.traits)


# ---------------------------------------------------------------------------
# Z-score behavior
# ---------------------------------------------------------------------------


def test_z_score_imputes_missing_bio_to_zero() -> None:
    """NaN height, weight, and age z-score to 0."""
    shots = _synth_shots({1: 50, 2: 50})
    gl = _synth_game_logs({1: 30, 2: 30})
    # Player 2 has NaN height/weight/birthdate.
    bio = pd.DataFrame(
        [
            {
                "player_id": 1,
                "display_name": "Player 1",
                "birthdate": pd.Timestamp("1990-01-15"),
                "height_inches": 78,
                "weight_lbs": 220,
                "position_raw": "Guard",
                "position_group": "SG",
                "status": "ok",
            },
            {
                "player_id": 2,
                "display_name": None,
                "birthdate": pd.NaT,
                "height_inches": pd.NA,
                "weight_lbs": pd.NA,
                "position_raw": None,
                "position_group": pd.NA,
                "status": "failed",
            },
        ]
    )
    anchors = [np.datetime64("2024-03-15", "D")]
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )
    table = build_player_traits_table(
        snapshot_store=store,
        vocab_ids=[1, 2],
        bio_df=bio,
        game_logs_df=gl,
    )
    # Player 2's height/weight/age z-scores should be 0 (NaN imputed).
    assert table.traits[1, 0, 0] == 0.0  # height_z
    assert table.traits[1, 0, 1] == 0.0  # weight_z
    assert table.traits[1, 0, 2] == 0.0  # age_z
    # Position one-hot all zero (unknown).
    np.testing.assert_array_equal(
        table.traits[1, 0, 3:BLOCK_A_END],
        np.zeros(5, dtype=np.float32),
    )


def _bio_rows(heights: dict[int, float]) -> pd.DataFrame:
    """Bio frame with the given heights and a shared weight, birthdate, and position."""
    return pd.DataFrame(
        [
            {
                "player_id": pid,
                "display_name": f"Player {pid}",
                "birthdate": pd.Timestamp("1995-01-01"),
                "height_inches": height,
                "weight_lbs": 200,
                "position_raw": "Guard",
                "position_group": "SG",
                "status": "ok",
            }
            for pid, height in heights.items()
        ]
    )


def _games(player_id: int, first_day: str, n_games: int = 5) -> list[dict[str, object]]:
    """Game-log rows for one player on consecutive days from ``first_day``."""
    return [
        {
            "player_id": player_id,
            "game_date": pd.Timestamp(first_day) + pd.Timedelta(days=i),
            "minutes": 30,
            "fga": 12,
            "fta": 3,
            "tov": 2,
        }
        for i in range(n_games)
    ]


def _single_anchor_store(anchor: str) -> object:
    shots = _synth_shots({1: 50, 2: 50})
    return build_snapshot_store_from_shots(
        shots,
        [np.datetime64(anchor, "D")],
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )


def test_bio_z_scores_use_only_players_active_before_anchor() -> None:
    """Height is standardized by the mean and std of players with games before the anchor."""
    store = _single_anchor_store("2024-03-15")
    # Players 1 and 2 play before the anchor; player 3 debuts after it.
    gl = pd.DataFrame(
        [*_games(1, "2024-01-01"), *_games(2, "2024-01-01"), *_games(3, "2024-05-01")]
    )
    bio = _bio_rows({1: 72.0, 2: 76.0, 3: 80.0})

    table = build_player_traits_table(
        snapshot_store=store, vocab_ids=[1, 2, 3], bio_df=bio, game_logs_df=gl
    )
    # Reference population {72, 76}: mean 74, std 2. The later debut is
    # standardized with those statistics and does not enter them.
    np.testing.assert_allclose(table.traits[:, 0, 0], [-1.0, 1.0, 3.0], atol=1e-6)


def test_bio_z_scores_unchanged_by_adding_a_later_debut() -> None:
    """Adding a player who debuts after the anchor leaves the others' Block A unchanged."""
    store = _single_anchor_store("2024-03-15")
    gl_before = pd.DataFrame([*_games(1, "2024-01-01"), *_games(2, "2024-01-01")])
    gl_after = pd.concat([gl_before, pd.DataFrame(_games(3, "2024-05-01"))], ignore_index=True)

    without = build_player_traits_table(
        snapshot_store=store,
        vocab_ids=[1, 2],
        bio_df=_bio_rows({1: 72.0, 2: 76.0}),
        game_logs_df=gl_before,
    )
    with_debut = build_player_traits_table(
        snapshot_store=store,
        vocab_ids=[1, 2, 3],
        bio_df=_bio_rows({1: 72.0, 2: 76.0, 3: 90.0}),
        game_logs_df=gl_after,
    )
    np.testing.assert_array_equal(
        with_debut.traits[:2, 0, :BLOCK_A_END], without.traits[:, 0, :BLOCK_A_END]
    )


def test_bio_reference_population_is_limited_to_the_window() -> None:
    """A player whose games all precede the reference window is not in the statistics."""
    store = _single_anchor_store("2024-03-15")
    # Player 3's games end more than 30 days before the anchor.
    gl = pd.DataFrame(
        [*_games(1, "2024-03-01"), *_games(2, "2024-03-01"), *_games(3, "2024-01-01")]
    )
    bio = _bio_rows({1: 72.0, 2: 76.0, 3: 80.0})

    table = build_player_traits_table(
        snapshot_store=store,
        vocab_ids=[1, 2, 3],
        bio_df=bio,
        game_logs_df=gl,
        reference_window_days=30.0,
    )
    np.testing.assert_allclose(table.traits[:, 0, 0], [-1.0, 1.0, 3.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Vocab-bundle alignment
# ---------------------------------------------------------------------------


def test_vocab_player_not_in_bundle_has_zero_block_b() -> None:
    """A vocabulary player absent from ``bundle.player_ids`` gets Block B = 0 and m_play = 0."""
    # Player 5 has zero shots — not in bundle at all.
    shots = _synth_shots({1: 50, 2: 50})  # only players 1 and 2 have shots
    bio = _synth_bio([1, 2, 5])  # bio for 5 still
    gl = _synth_game_logs({1: 20, 2: 20})  # no game logs for player 5 either
    anchors = [np.datetime64("2024-03-15", "D")]
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )
    table = build_player_traits_table(
        snapshot_store=store,
        vocab_ids=[1, 2, 5],
        bio_df=bio,
        game_logs_df=gl,
    )
    # Player 5: Block B is zero, m_play = 0.
    np.testing.assert_array_equal(
        table.traits[2, 0, BLOCK_A_END:BLOCK_B_END],
        np.zeros(BLOCK_B_END - BLOCK_A_END, dtype=np.float32),
    )
    assert table.traits[2, 0, M_PLAY_SLOT] == 0.0
