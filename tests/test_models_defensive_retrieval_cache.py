"""Tests for ``shotcloud.models.defensive_retrieval_cache`` (PR-D0).

The ten acceptance criteria from the build approval (2026-05-25):

1. Cache is deterministic for synthetic data.
2. All retrieved shots are causal: ``shot_date < snapshot_anchor_date``.
3. Retrieved shots are exactly shots attempted against opponent ``d``.
4. No shots by team ``d`` are accidentally treated as allowed shots
   against ``d``.
5. Padding uses ``-1`` indices and ``False`` mask.
6. Cold-start (opponent, snapshot) cells produce all padding +
   empty masks.
7. Top-``M_def`` ordering respects recency: shots stored
   most-recent-first; ties resolve to lower original index reproducibly.
8. Disk round-trip preserves config and tensors exactly.
9. ``config_hash`` changes when any retrieval-defining setting changes.
10. Atomic save leaves no stale ``.tmp`` files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud.models.defensive_retrieval_cache import (
    DefensiveRetrievalCache,
    DefensiveRetrievalCacheConfig,
    build_defensive_retrieval_cache,
    defensive_shots_fingerprint,
)
from shotcloud.training.dataset import OpponentVocab

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _epoch_day(s: str) -> int:
    return int(np.datetime64(s, "D").astype(np.int64))


def _build_simple_fixture(*, seed: int = 0) -> tuple[pd.DataFrame, OpponentVocab, np.ndarray]:
    """4 teams (A/B/C/D), 1 snapshot, controlled allowed-shot histories.

    * 2 shots per (defending_team) pair before the anchor (causal).
    * 1 shot per (defending_team) on the anchor date (non-causal —
      tests the strict ``< anchor`` cutoff).
    * Teams D has *zero* causal allowed shots in the window — used as
      the cold-start cell.
    """
    rng = np.random.default_rng(seed)
    anchor = _epoch_day("2024-04-15")
    pre_dates = [anchor - 10, anchor - 5]
    non_causal_date = anchor  # equal-to-anchor counts as non-causal
    rows: list[dict[str, object]] = []
    teams = ["A", "B", "C"]  # D is intentionally absent — cold-start
    # For each defending team, build allowed shots by *some other* team
    # shooting against them.
    for defending in teams:
        # Pick an attacking team that's not the defending team.
        for d in pre_dates:
            attacker_pool = [t for t in teams if t != defending] + ["E"]
            attacker = attacker_pool[int(rng.integers(0, len(attacker_pool)))]
            rows.append(
                {
                    "player_id": f"player_{attacker}_{int(rng.integers(0, 1000))}",
                    "team": attacker,
                    "opponent": defending,
                    "date": np.datetime64(d, "D"),
                    "x": float(rng.normal(0, 1)),
                    "y": float(rng.normal(0, 1)),
                    "game_id": f"g_{defending}_{d}",
                }
            )
        # Non-causal: a shot on the anchor day; should be excluded.
        rows.append(
            {
                "player_id": "player_NC",
                "team": "E",
                "opponent": defending,
                "date": np.datetime64(non_causal_date, "D"),
                "x": 0.0,
                "y": 0.0,
                "game_id": f"g_NC_{defending}",
            }
        )
    shots = pd.DataFrame(rows)
    # Vocab covers all 4 teams (including the cold-start D).
    vocab = OpponentVocab.from_ids(["A", "B", "C", "D"])
    anchor_dates = np.array([anchor], dtype=np.int64)
    return shots, vocab, anchor_dates


def _default_config(anchor_dates: np.ndarray, **overrides: object) -> DefensiveRetrievalCacheConfig:
    kwargs: dict[str, object] = {
        "shots_fingerprint": "fixture_v0",
        "anchor_dates": tuple(int(d) for d in anchor_dates),
        "defensive_support_max": 4,
        "defensive_recency_window_days": 30,
    }
    kwargs.update(overrides)
    return DefensiveRetrievalCacheConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------


def test_1_build_is_deterministic() -> None:
    shots, vocab, anchors = _build_simple_fixture()
    cfg = _default_config(anchors)
    c1 = build_defensive_retrieval_cache(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    c2 = build_defensive_retrieval_cache(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    torch.testing.assert_close(c1.global_xy, c2.global_xy)
    torch.testing.assert_close(c1.global_dates, c2.global_dates)
    torch.testing.assert_close(c1.global_opponent_idx, c2.global_opponent_idx)
    torch.testing.assert_close(c1.def_idx, c2.def_idx)
    torch.testing.assert_close(c1.def_mask, c2.def_mask)


def test_2_causality_no_retrieved_shot_at_or_after_anchor() -> None:
    """Every valid ``def_idx`` points at a global shot whose date is
    strictly less than the snapshot anchor."""
    shots, vocab, anchors = _build_simple_fixture()
    cfg = _default_config(anchors)
    c = build_defensive_retrieval_cache(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    for s_idx, anchor in enumerate(anchors.tolist()):
        idxs = c.def_idx[:, s_idx, :]
        mask = c.def_mask[:, s_idx, :]
        if mask.any():
            dates = c.global_dates[idxs[mask]]
            assert (dates < anchor).all(), f"defensive shot at/after anchor {anchor}: {dates}"


def test_3_retrieved_shots_are_exactly_against_target_opponent() -> None:
    """For each (opp, snapshot), every valid retrieved row has
    ``opponent == opp`` in the global table."""
    shots, vocab, anchors = _build_simple_fixture()
    cfg = _default_config(anchors)
    c = build_defensive_retrieval_cache(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    for d_idx in range(len(vocab)):
        mask = c.def_mask[d_idx, 0, :]
        if not mask.any():
            continue
        opps = c.global_opponent_idx[c.def_idx[d_idx, 0, :][mask]]
        assert (opps == d_idx).all(), (
            f"opp d_idx={d_idx} pool contains shots against other opponents: {opps.tolist()}"
        )


def test_4_cache_raises_when_any_row_has_team_equal_to_opponent() -> None:
    """The cache enforces the load_shots() invariant that a team
    cannot play itself: a row with ``team == opponent`` is a real
    bug we want to surface loudly, not propagate silently. This is
    the load-bearing fence for criterion #4 (no shots by team d
    accidentally treated as allowed against d) — when ``team`` is
    present in the dataframe, the cache enforces it.

    Cross-check on the clean fixture (`test_4b`) verifies that the
    fence isn't over-eager: on legitimately attacker-vs-defender
    rows the build proceeds and the global opponent indices are
    correct."""
    shots, vocab, anchors = _build_simple_fixture()
    # Corrupt one row to have team == opponent, both in-vocab (so the
    # row survives the in-vocab filter and the team-eq-opponent fence
    # is the thing that fires).
    shots = shots.copy()
    bad_row = shots.index[0]
    shots.loc[bad_row, "team"] = "A"
    shots.loc[bad_row, "opponent"] = "A"
    cfg = _default_config(anchors)
    with pytest.raises(ValueError, match="team == opponent"):
        build_defensive_retrieval_cache(
            shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
        )


def test_4b_clean_fixture_passes_team_neq_opponent_check() -> None:
    """Sanity: on the unaltered fixture (which respects the
    load_shots() invariant), the build succeeds and every retrieved
    row's defending opponent matches the index it was placed under.
    This is what criterion #4 is *really* asking — that the
    bookkeeping doesn't swap opponent for team."""
    shots, vocab, anchors = _build_simple_fixture()
    cfg = _default_config(anchors)
    c = build_defensive_retrieval_cache(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    # For every valid slot, the global opponent index must equal the
    # row's d_idx. This is the same as test_3 but framed against
    # criterion #4's "no team-as-opponent swap" concern.
    for d_idx in range(len(vocab)):
        mask = c.def_mask[d_idx, 0, :]
        if not mask.any():
            continue
        opps = c.global_opponent_idx[c.def_idx[d_idx, 0, :][mask]]
        assert (opps == d_idx).all().item()


def test_5_padding_uses_negative_one_idx_and_false_mask() -> None:
    """With cap=4 and only 2 causal allowed shots per non-cold-start
    opp, slots 2,3 must be (-1, False)."""
    shots, vocab, anchors = _build_simple_fixture()
    cfg = _default_config(anchors, defensive_support_max=4)
    c = build_defensive_retrieval_cache(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    for d_idx, opp_label in enumerate(vocab.ids):
        if opp_label == "D":
            continue  # cold-start; covered by test_6
        idxs = c.def_idx[d_idx, 0, :]
        mask = c.def_mask[d_idx, 0, :]
        # Slots 0,1 should be valid; slots 2,3 padded.
        assert (idxs[mask] >= 0).all()
        assert (idxs[~mask] == -1).all()
        assert mask[:2].all() and not mask[2:].any()


def test_6_cold_start_opp_snapshot_cell_is_all_padding() -> None:
    """Opponent D has zero allowed-shot history before the anchor.
    Its full (cap-long) row must be -1 indices + False mask."""
    shots, vocab, anchors = _build_simple_fixture()
    cfg = _default_config(anchors)
    c = build_defensive_retrieval_cache(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    d_idx_cold = vocab.to_idx("D")
    idxs = c.def_idx[d_idx_cold, 0, :]
    mask = c.def_mask[d_idx_cold, 0, :]
    assert not mask.any().item()
    assert (idxs == -1).all().item()


def test_7_top_m_def_ordering_respects_recency() -> None:
    """Top-M_def ordering is by date descending: with cap=2 and four
    causal allowed shots against opp A on dates anchor-5, anchor-10,
    anchor-15, anchor-20, the cache should keep the two most-recent
    (anchor-5 and anchor-10) in that order."""
    anchor = _epoch_day("2024-04-15")
    dates = [anchor - 20, anchor - 15, anchor - 10, anchor - 5]
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "player_id": f"p{i}",
                "team": "Z",
                "opponent": "A",
                "date": np.datetime64(d, "D"),
                "x": float(i),  # use index so we can identify which shot was retained
                "y": 0.0,
                "game_id": f"g{i}",
            }
        )
    shots = pd.DataFrame(rows)
    vocab = OpponentVocab.from_ids(["A"])
    anchors = np.array([anchor], dtype=np.int64)
    cfg = _default_config(anchors, defensive_support_max=2)
    c = build_defensive_retrieval_cache(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    d_idx_A = vocab.to_idx("A")
    mask = c.def_mask[d_idx_A, 0, :]
    assert mask.all().item()  # both slots used
    selected = c.def_idx[d_idx_A, 0, :].tolist()
    # global table is sorted ascending by date: indices 0..3 correspond
    # to dates [anchor-20, anchor-15, anchor-10, anchor-5]. Most-recent
    # first → indices [3, 2].
    assert selected == [3, 2], f"expected [3, 2] (most-recent first); got {selected}"
    # And the retained coordinates correspond to the most-recent shots'
    # x values (3 and 2 in our fixture).
    chosen_x = c.global_xy[c.def_idx[d_idx_A, 0, :]][:, 0].tolist()
    assert chosen_x == [3.0, 2.0]


def test_7b_recency_tie_resolves_to_lower_original_index() -> None:
    """When two allowed shots share the same date, the stable argsort
    picks the lower original-index row first — deterministic and
    reproducible across NumPy versions."""
    anchor = _epoch_day("2024-04-15")
    # Three shots on the same date; cap=2.
    rows = []
    for i in range(3):
        rows.append(
            {
                "player_id": f"p{i}",
                "team": "Z",
                "opponent": "A",
                "date": np.datetime64(anchor - 5, "D"),
                "x": float(i),
                "y": 0.0,
                "game_id": f"g{i}",
            }
        )
    shots = pd.DataFrame(rows)
    vocab = OpponentVocab.from_ids(["A"])
    anchors = np.array([anchor], dtype=np.int64)
    cfg = _default_config(anchors, defensive_support_max=2)
    c = build_defensive_retrieval_cache(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    d_idx_A = vocab.to_idx("A")
    selected = c.def_idx[d_idx_A, 0, :].tolist()
    # Stable argsort by -date with all dates equal preserves original
    # order, so we get [0, 1].
    assert selected == [0, 1], f"expected stable [0, 1]; got {selected}"


def test_8_disk_round_trip_preserves_tensors_and_config(tmp_path: Path) -> None:
    shots, vocab, anchors = _build_simple_fixture()
    cfg = _default_config(anchors)
    c = build_defensive_retrieval_cache(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    save_path = tmp_path / "def_cache.pt"
    c.save(save_path)
    loaded = DefensiveRetrievalCache.load(save_path)
    assert loaded.config == c.config, "config mismatch on round trip"
    torch.testing.assert_close(loaded.global_xy, c.global_xy)
    torch.testing.assert_close(loaded.global_dates, c.global_dates)
    torch.testing.assert_close(loaded.global_opponent_idx, c.global_opponent_idx)
    torch.testing.assert_close(loaded.def_idx, c.def_idx)
    torch.testing.assert_close(loaded.def_mask, c.def_mask)


def test_8b_build_with_cache_dir_loads_on_second_call(tmp_path: Path) -> None:
    """First call writes the cache; second call loads it without
    re-running the build (verified by passing an empty DataFrame on
    the second call — if rebuild ran it would crash)."""
    shots, vocab, anchors = _build_simple_fixture()
    cfg = _default_config(anchors)
    c1 = build_defensive_retrieval_cache(
        shots_df=shots,
        opp_vocab=vocab,
        anchor_dates=anchors,
        config=cfg,
        cache_dir=tmp_path,
    )
    cache_file = tmp_path / f"defensive_retrieval_cache_{cfg.config_hash}.pt"
    assert cache_file.exists()
    c2 = build_defensive_retrieval_cache(
        shots_df=pd.DataFrame(),  # empty; would crash if rebuild ran
        opp_vocab=vocab,
        anchor_dates=anchors,
        config=cfg,
        cache_dir=tmp_path,
    )
    torch.testing.assert_close(c1.def_idx, c2.def_idx)
    torch.testing.assert_close(c1.def_mask, c2.def_mask)


def test_9_config_hash_changes_on_any_retrieval_defining_field() -> None:
    """Each retrieval-defining field, when changed, must produce a
    different ``config_hash``."""
    anchors = np.array([_epoch_day("2024-04-15")], dtype=np.int64)
    base = _default_config(anchors)
    base_h = base.config_hash
    variants: dict[str, DefensiveRetrievalCacheConfig] = {
        "shots_fingerprint": _default_config(anchors, shots_fingerprint="other_v0"),
        "anchor_dates": _default_config(np.array([_epoch_day("2024-04-16")], dtype=np.int64)),
        "defensive_support_max": _default_config(anchors, defensive_support_max=8),
        "defensive_recency_window_days": _default_config(anchors, defensive_recency_window_days=60),
        "seed": _default_config(anchors, seed=1),
    }
    for field, variant in variants.items():
        assert variant.config_hash != base_h, f"{field} change did not alter config_hash"
    # Same fields → same hash (sanity).
    assert _default_config(anchors).config_hash == base_h


def test_10_atomic_save_leaves_no_tmp_file(tmp_path: Path) -> None:
    """After a successful save, no leftover ``.tmp`` file remains."""
    shots, vocab, anchors = _build_simple_fixture()
    cfg = _default_config(anchors)
    c = build_defensive_retrieval_cache(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    save_path = tmp_path / "def_cache.pt"
    c.save(save_path)
    assert save_path.exists()
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == [], f"leftover .tmp files: {leftover}"


# ---------------------------------------------------------------------------
# Smaller helpers
# ---------------------------------------------------------------------------


def test_config_validates_positive_fields() -> None:
    anchors = (_epoch_day("2024-04-15"),)
    with pytest.raises(ValueError, match="defensive_support_max"):
        DefensiveRetrievalCacheConfig(
            shots_fingerprint="x", anchor_dates=anchors, defensive_support_max=0
        )
    with pytest.raises(ValueError, match="defensive_recency_window_days"):
        DefensiveRetrievalCacheConfig(
            shots_fingerprint="x", anchor_dates=anchors, defensive_recency_window_days=0
        )
    with pytest.raises(ValueError, match="ranking_kind"):
        DefensiveRetrievalCacheConfig(
            shots_fingerprint="x", anchor_dates=anchors, ranking_kind="cosine"
        )


def test_build_rejects_missing_opponent_column() -> None:
    """The cache requires a canonical ``opponent`` column (populated
    by ``shotcloud.data.loaders.load_shots``). Without it, the
    builder must raise a clear error rather than silently producing
    an empty cache."""
    shots = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "team": "A",
                "date": np.datetime64("2024-04-10", "D"),
                "x": 0.0,
                "y": 0.0,
            }
        ]
    )
    vocab = OpponentVocab.from_ids(["A"])
    anchors = np.array([_epoch_day("2024-04-15")], dtype=np.int64)
    cfg = _default_config(anchors)
    with pytest.raises(ValueError, match="opponent"):
        build_defensive_retrieval_cache(
            shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
        )


def test_build_drops_rows_with_missing_opponent() -> None:
    """Rows where ``opponent`` is NA (boundary games) are silently
    dropped — they're not valuable for defense and shouldn't crash
    the build."""
    shots = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "team": "Z",
                "opponent": "A",
                "date": np.datetime64("2024-04-10", "D"),
                "x": 0.0,
                "y": 0.0,
            },
            {
                "player_id": "p2",
                "team": "Z",
                "opponent": pd.NA,
                "date": np.datetime64("2024-04-10", "D"),
                "x": 1.0,
                "y": 1.0,
            },
        ]
    )
    vocab = OpponentVocab.from_ids(["A"])
    anchors = np.array([_epoch_day("2024-04-15")], dtype=np.int64)
    cfg = _default_config(anchors)
    c = build_defensive_retrieval_cache(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    # Only one in-vocab non-NA row → global table has length 1.
    assert c.global_xy.shape[0] == 1


def test_defensive_shots_fingerprint_changes_with_mtime(tmp_path: Path) -> None:
    p = tmp_path / "shots.csv"
    p.write_text("x,y,player_id,opponent,date\n0,0,1,2,2024-01-01\n")
    h0 = defensive_shots_fingerprint(p)
    import os
    import time

    time.sleep(0.01)
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    h1 = defensive_shots_fingerprint(p)
    assert h0 != h1
