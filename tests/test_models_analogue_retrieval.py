"""Tests for :mod:`shotcloud.models.analogue_retrieval`."""

from __future__ import annotations

import numpy as np

from shotcloud.data.player_traits import TRAIT_DIM, PlayerTraitsTable
from shotcloud.models.analogue_retrieval import (
    AnalogueRetrievalCache,
    build_analogue_cache,
)


def _make_synthetic_traits(
    n_players: int,
    n_snapshots: int = 1,
    seed: int = 0,
) -> PlayerTraitsTable:
    """Random unit-norm trait table for testing."""
    rng = np.random.default_rng(seed)
    traits = rng.standard_normal((n_players, n_snapshots, TRAIT_DIM)).astype(np.float32)
    return PlayerTraitsTable(
        traits=traits,
        player_ids=np.arange(n_players, dtype=np.int64),
        snapshot_anchors=np.array(
            [
                np.datetime64("2024-01-01", "D") + np.timedelta64(m * 30, "D")
                for m in range(n_snapshots)
            ],
            dtype="datetime64[D]",
        ),
    )


# ---------------------------------------------------------------------------
# Shape + structural contract
# ---------------------------------------------------------------------------


def test_cache_shape_matches_inputs() -> None:
    traits = _make_synthetic_traits(n_players=20, n_snapshots=3)
    cache = build_analogue_cache(traits, L=5)
    assert isinstance(cache, AnalogueRetrievalCache)
    assert cache.analogues.shape == (20, 3, 5)
    assert cache.analogues.dtype == np.int64
    assert cache.L == 5
    assert cache.n_players == 20
    assert cache.n_snapshots == 3


def test_all_indices_are_valid_player_indices() -> None:
    traits = _make_synthetic_traits(n_players=30, n_snapshots=2)
    cache = build_analogue_cache(traits, L=10)
    assert cache.analogues.min() >= 0
    assert cache.analogues.max() < 30


def test_self_always_first_when_ensure_self_true() -> None:
    traits = _make_synthetic_traits(n_players=15, n_snapshots=2)
    cache = build_analogue_cache(traits, L=8, ensure_self=True)
    # Diagonal: cache[p, m, 0] should equal p.
    for p in range(15):
        for m in range(2):
            assert int(cache.analogues[p, m, 0]) == p, (
                f"self not at position 0 for (p={p}, m={m}): got {cache.analogues[p, m, 0]}"
            )


def test_self_not_forced_when_ensure_self_false() -> None:
    """Without ensure_self, the target may or may not appear; cosine
    of u with itself is always 1.0 so self is often top, but it's not
    *guaranteed* to be first (random ties)."""
    traits = _make_synthetic_traits(n_players=20, n_snapshots=1, seed=1)
    cache = build_analogue_cache(traits, L=10, ensure_self=False)
    # Self's cosine with itself is 1.0 — it will be in the top-L,
    # but not necessarily at position 0 if ties exist. Just check
    # self appears somewhere in the L for each player.
    for p in range(20):
        assert p in cache.analogues[p, 0].tolist()


# ---------------------------------------------------------------------------
# Correctness vs brute-force baseline
# ---------------------------------------------------------------------------


def test_topL_matches_brute_force_cosine() -> None:
    """The top-L set (as a SET, ignoring within-L ordering) matches
    a naive cosine-and-sort baseline on a small case."""
    traits = _make_synthetic_traits(n_players=12, n_snapshots=1, seed=42)
    L = 6
    cache = build_analogue_cache(traits, L=L, ensure_self=True)

    u = traits.traits[:, 0, :].astype(np.float64)
    norms = np.linalg.norm(u, axis=-1, keepdims=True)
    u_norm = u / np.maximum(norms, 1e-12)
    sim = u_norm @ u_norm.T
    np.fill_diagonal(sim, np.inf)

    for p in range(12):
        expected_topL = set(np.argsort(-sim[p])[:L].tolist())
        actual_topL = set(cache.analogues[p, 0].tolist())
        assert expected_topL == actual_topL, (
            f"top-L mismatch for player {p}: expected {expected_topL}, got {actual_topL}"
        )


def test_within_L_ordering_descending_by_similarity() -> None:
    """The first entry must be the highest cosine, the last entry the
    lowest within the top-L."""
    traits = _make_synthetic_traits(n_players=20, n_snapshots=1, seed=7)
    L = 8
    cache = build_analogue_cache(traits, L=L, ensure_self=True)

    u = traits.traits[:, 0, :].astype(np.float64)
    norms = np.linalg.norm(u, axis=-1, keepdims=True)
    u_norm = u / np.maximum(norms, 1e-12)
    sim = u_norm @ u_norm.T
    np.fill_diagonal(sim, np.inf)

    for p in range(20):
        retrieved = cache.analogues[p, 0]
        retrieved_sims = sim[p, retrieved]
        # Monotonically non-increasing.
        assert np.all(np.diff(retrieved_sims) <= 1e-9), (
            f"player {p}: retrieved sims not sorted: {retrieved_sims}"
        )


# ---------------------------------------------------------------------------
# Cold-start behavior (bio-only retrieval)
# ---------------------------------------------------------------------------


def test_cold_start_retrieves_by_biographical_similarity() -> None:
    """A player with all-zero Block B (cold-start) should retrieve
    analogues that match on Block A (biographical traits)."""
    n_players = 6
    # Construct trait vectors by hand to control similarity:
    # Players 0..2: tall centers (height_z=+2, bio_pos_C=1).
    # Players 3..5: short guards (height_z=-2, bio_pos_PG=1).
    # Player 0 is "cold-start": Block B all zero, m_play=0.
    traits = np.zeros((n_players, 1, TRAIT_DIM), dtype=np.float32)
    for p in range(3):  # centers
        traits[p, 0, 0] = 2.0  # height_z
        traits[p, 0, 7] = 1.0  # bio_pos_C (slot 7)
    for p in range(3, 6):  # guards
        traits[p, 0, 0] = -2.0
        traits[p, 0, 3] = 1.0  # bio_pos_PG (slot 3)

    # Players 1, 2, 4, 5 have nonzero Block B (active players);
    # player 0 stays all-zero in Block B (cold-start).
    for p in (1, 2):
        traits[p, 0, 8:25] = 0.5  # play traits (Block B)
        traits[p, 0, 25] = 1.0  # m_play
    for p in (4, 5):
        traits[p, 0, 8:25] = 0.5
        traits[p, 0, 25] = 1.0

    table = PlayerTraitsTable(
        traits=traits,
        player_ids=np.arange(n_players, dtype=np.int64),
        snapshot_anchors=np.array([np.datetime64("2024-01-01", "D")], dtype="datetime64[D]"),
    )

    cache = build_analogue_cache(table, L=3, ensure_self=True)
    # Player 0 (cold-start center) should retrieve the other centers (1, 2)
    # ahead of any guard (3, 4, 5) via Block A similarity alone.
    p0_analogues = set(cache.analogues[0, 0].tolist())
    assert p0_analogues == {0, 1, 2}, (
        f"cold-start center should retrieve other centers; got {p0_analogues}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_l_too_large_raises() -> None:
    import pytest

    traits = _make_synthetic_traits(n_players=10)
    with pytest.raises(ValueError, match="exceeds n_players"):
        build_analogue_cache(traits, L=15)


def test_l_nonpositive_raises() -> None:
    import pytest

    traits = _make_synthetic_traits(n_players=10)
    with pytest.raises(ValueError, match="positive"):
        build_analogue_cache(traits, L=0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_repeated_builds() -> None:
    traits = _make_synthetic_traits(n_players=25, n_snapshots=4, seed=11)
    c1 = build_analogue_cache(traits, L=7)
    c2 = build_analogue_cache(traits, L=7)
    np.testing.assert_array_equal(c1.analogues, c2.analogues)
