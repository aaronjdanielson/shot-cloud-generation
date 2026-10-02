"""Tests for :mod:`shotcloud.models.retrieval_cache`.

The numbered tests check these properties of the own/pooled support cache:

1. Builds a deterministic cache for synthetic data.
2. Enforces causality: no retrieved shot has date >= anchor date.
3. Own pool contains only target-player shots.
4. Pooled pool excludes target-player shots.
5. Pooled retrieval respects cosine-sim + recency ranking on a
   controlled example.
6. Padding uses -1 indices and False masks.
7. Disk round-trip preserves tensors and config exactly.
8. Config hash changes when retrieval-defining settings change.
9. Atomic write path works (no leftover ``.tmp`` after save).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud.models.retrieval_cache import (
    RetrievalCache,
    RetrievalCacheConfig,
    build_retrieval_cache,
    shots_fingerprint,
    traits_fingerprint,
)
from shotcloud.training.dataset import PlayerVocab

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _epoch_day(s: str) -> int:
    return int(np.datetime64(s, "D").astype(np.int64))


def _build_simple_fixture(
    *, trait_dim: int = 3, seed: int = 0
) -> tuple[pd.DataFrame, PlayerVocab, np.ndarray, torch.Tensor]:
    """Four players, one snapshot, controlled traits.

    * The target is player 0, whose traits point along the x-axis.
    * Players 1, 2, 3 have trait cosines 0.9, 0.5, 0.1 with player 0.
    * Each player has two causal shots before the anchor and one shot on the anchor
      date, which the strict ``< anchor`` cutoff must exclude. All shots fall inside
      the recency window.
    """
    rng = np.random.default_rng(seed)
    anchor = _epoch_day("2024-04-15")
    pre_dates = [anchor - 10, anchor - 5]
    non_causal_date = anchor  # equal-to-anchor counts as non-causal
    rows: list[dict[str, object]] = []
    for pid in [0, 1, 2, 3]:
        for d in pre_dates:
            rows.append(
                {
                    "player_id": str(pid),
                    "date": np.datetime64(d, "D"),
                    "x": float(rng.normal(0, 1)),
                    "y": float(rng.normal(0, 1)),
                }
            )
        rows.append(
            {
                "player_id": str(pid),
                "date": np.datetime64(non_causal_date, "D"),
                "x": 0.0,
                "y": 0.0,
            }
        )
    shots = pd.DataFrame(rows)
    vocab = PlayerVocab.from_ids([0, 1, 2, 3])
    anchor_dates = np.array([anchor], dtype=np.int64)

    # Traits at the single snapshot: player 0 along x, the others at known cosines
    # with player 0.
    traits = torch.zeros(4, 1, trait_dim)
    traits[0, 0, 0] = 1.0
    # Player 1: cos = 0.9
    traits[1, 0, 0] = 0.9
    traits[1, 0, 1] = float(np.sqrt(1.0 - 0.81))
    # Player 2: cos = 0.5
    traits[2, 0, 0] = 0.5
    traits[2, 0, 1] = float(np.sqrt(1.0 - 0.25))
    # Player 3: cos = 0.1
    traits[3, 0, 0] = 0.1
    traits[3, 0, 1] = float(np.sqrt(1.0 - 0.01))
    return shots, vocab, anchor_dates, traits


def _default_config(anchor_dates: np.ndarray, **overrides: object) -> RetrievalCacheConfig:
    kwargs: dict[str, object] = {
        "shots_fingerprint": "fixture_v0",
        "anchor_dates": tuple(int(d) for d in anchor_dates),
        "own_support_max": 4,
        "pooled_support_max": 4,
        "pooled_recency_window_days": 30,
        "pooled_recency_half_life_days": 30.0,
    }
    kwargs.update(overrides)
    return RetrievalCacheConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------


def test_1_build_is_deterministic() -> None:
    shots, vocab, anchors, traits = _build_simple_fixture()
    cfg = _default_config(anchors)
    c1 = build_retrieval_cache(
        shots_df=shots, player_vocab=vocab, anchor_dates=anchors, traits=traits, config=cfg
    )
    c2 = build_retrieval_cache(
        shots_df=shots, player_vocab=vocab, anchor_dates=anchors, traits=traits, config=cfg
    )
    torch.testing.assert_close(c1.global_xy, c2.global_xy)
    torch.testing.assert_close(c1.global_dates, c2.global_dates)
    torch.testing.assert_close(c1.global_shooter_idx, c2.global_shooter_idx)
    torch.testing.assert_close(c1.own_idx, c2.own_idx)
    torch.testing.assert_close(c1.own_mask, c2.own_mask)
    torch.testing.assert_close(c1.pooled_idx, c2.pooled_idx)
    torch.testing.assert_close(c1.pooled_mask, c2.pooled_mask)


def test_2_causality_no_retrieved_shot_at_or_after_anchor() -> None:
    """Every valid ``own_idx`` / ``pooled_idx`` points at a global shot dated strictly
    before the snapshot anchor."""
    shots, vocab, anchors, traits = _build_simple_fixture()
    cfg = _default_config(anchors)
    c = build_retrieval_cache(
        shots_df=shots, player_vocab=vocab, anchor_dates=anchors, traits=traits, config=cfg
    )
    for s_idx, anchor in enumerate(anchors.tolist()):
        # Own
        idxs = c.own_idx[:, s_idx, :]
        mask = c.own_mask[:, s_idx, :]
        if mask.any():
            dates = c.global_dates[idxs[mask]]
            assert (dates < anchor).all(), f"own shot at/after anchor {anchor}: {dates}"
        # Pooled
        idxs = c.pooled_idx[:, s_idx, :]
        mask = c.pooled_mask[:, s_idx, :]
        if mask.any():
            dates = c.global_dates[idxs[mask]]
            assert (dates < anchor).all(), f"pooled shot at/after anchor {anchor}: {dates}"


def test_3_own_pool_only_target_player() -> None:
    shots, vocab, anchors, traits = _build_simple_fixture()
    cfg = _default_config(anchors)
    c = build_retrieval_cache(
        shots_df=shots, player_vocab=vocab, anchor_dates=anchors, traits=traits, config=cfg
    )
    for p_idx in range(len(vocab)):
        mask = c.own_mask[p_idx, 0, :]
        if not mask.any():
            continue
        shooters = c.global_shooter_idx[c.own_idx[p_idx, 0, :][mask]]
        assert (shooters == p_idx).all()


def test_4_pooled_pool_excludes_target_player() -> None:
    shots, vocab, anchors, traits = _build_simple_fixture()
    cfg = _default_config(anchors)
    c = build_retrieval_cache(
        shots_df=shots, player_vocab=vocab, anchor_dates=anchors, traits=traits, config=cfg
    )
    for p_idx in range(len(vocab)):
        mask = c.pooled_mask[p_idx, 0, :]
        if not mask.any():
            continue
        shooters = c.global_shooter_idx[c.pooled_idx[p_idx, 0, :][mask]]
        assert (shooters != p_idx).all()


def test_5_pooled_retrieval_respects_cosine_sim_ranking() -> None:
    """With equal shot dates (flat recency term), player 0's top-2 pooled shots come
    from player 1, the most similar shooter."""
    shots, vocab, anchors, traits = _build_simple_fixture()
    # Put every causal shot on the same date so the recency term is constant.
    a = int(anchors[0])
    shots = shots.copy()
    mask = shots["date"] < np.datetime64(a, "D")
    shots.loc[mask, "date"] = np.datetime64(a - 5, "D")
    cfg = _default_config(anchors, pooled_support_max=2)
    c = build_retrieval_cache(
        shots_df=shots, player_vocab=vocab, anchor_dates=anchors, traits=traits, config=cfg
    )
    pool_mask = c.pooled_mask[0, 0, :]
    pool_idx = c.pooled_idx[0, 0, :][pool_mask]
    shooters = c.global_shooter_idx[pool_idx]
    # Both top-2 shots come from player 1 (cosine 0.9).
    assert (shooters == 1).all(), f"expected top-2 from player 1; got {shooters.tolist()}"


def test_5b_pooled_retrieval_ranks_higher_cosine_above_lower() -> None:
    """With ``pooled_support_max=4`` and two equal-date shots from each of players 1,
    2, 3, the top 4 come from players 1 and 2 (cosine 0.9 and 0.5), not player 3."""
    shots, vocab, anchors, traits = _build_simple_fixture()
    a = int(anchors[0])
    shots = shots.copy()
    mask = shots["date"] < np.datetime64(a, "D")
    shots.loc[mask, "date"] = np.datetime64(a - 5, "D")
    cfg = _default_config(anchors, pooled_support_max=4)
    c = build_retrieval_cache(
        shots_df=shots, player_vocab=vocab, anchor_dates=anchors, traits=traits, config=cfg
    )
    pool_mask = c.pooled_mask[0, 0, :]
    pool_idx = c.pooled_idx[0, 0, :][pool_mask]
    shooters = c.global_shooter_idx[pool_idx]
    assert set(shooters.tolist()) == {1, 2}, (
        f"expected top-4 from players {{1, 2}}; got {sorted(set(shooters.tolist()))}"
    )


def test_6_padding_uses_negative_one_idx_and_false_mask() -> None:
    """With cap 4 and two causal own shots, slots 2 and 3 are ``(-1, False)``."""
    shots, vocab, anchors, traits = _build_simple_fixture()
    cfg = _default_config(anchors, own_support_max=4)
    c = build_retrieval_cache(
        shots_df=shots, player_vocab=vocab, anchor_dates=anchors, traits=traits, config=cfg
    )
    # Player 0 has 2 causal own shots.
    own_idx_p0 = c.own_idx[0, 0, :]
    own_mask_p0 = c.own_mask[0, 0, :]
    assert (own_idx_p0[own_mask_p0] >= 0).all()
    assert (own_idx_p0[~own_mask_p0] == -1).all()
    assert own_mask_p0[:2].all() and not own_mask_p0[2:].any()


def test_7_disk_round_trip_preserves_tensors_and_config(tmp_path: Path) -> None:
    shots, vocab, anchors, traits = _build_simple_fixture()
    cfg = _default_config(anchors)
    c = build_retrieval_cache(
        shots_df=shots, player_vocab=vocab, anchor_dates=anchors, traits=traits, config=cfg
    )
    save_path = tmp_path / "cache.pt"
    c.save(save_path)
    loaded = RetrievalCache.load(save_path)
    assert loaded.config == c.config, "config mismatch on round trip"
    torch.testing.assert_close(loaded.global_xy, c.global_xy)
    torch.testing.assert_close(loaded.global_dates, c.global_dates)
    torch.testing.assert_close(loaded.global_shooter_idx, c.global_shooter_idx)
    torch.testing.assert_close(loaded.own_idx, c.own_idx)
    torch.testing.assert_close(loaded.own_mask, c.own_mask)
    torch.testing.assert_close(loaded.pooled_idx, c.pooled_idx)
    torch.testing.assert_close(loaded.pooled_mask, c.pooled_mask)


def test_7b_build_with_cache_dir_loads_on_second_call(tmp_path: Path) -> None:
    """With ``cache_dir`` set, the first call writes the cache and the second loads it
    without rebuilding (the second call passes an empty DataFrame, on which a rebuild
    would fail)."""
    shots, vocab, anchors, traits = _build_simple_fixture()
    cfg = _default_config(anchors)
    c1 = build_retrieval_cache(
        shots_df=shots,
        player_vocab=vocab,
        anchor_dates=anchors,
        traits=traits,
        config=cfg,
        cache_dir=tmp_path,
    )
    cache_file = tmp_path / f"retrieval_cache_{cfg.config_hash}.pt"
    assert cache_file.exists()
    # shots_df is ignored when a cache file with a matching hash exists.
    c2 = build_retrieval_cache(
        shots_df=pd.DataFrame(),  # empty; if rebuild ran it would crash
        player_vocab=vocab,
        anchor_dates=anchors,
        traits=traits,
        config=cfg,
        cache_dir=tmp_path,
    )
    torch.testing.assert_close(c1.own_idx, c2.own_idx)
    torch.testing.assert_close(c1.pooled_idx, c2.pooled_idx)


def test_8_config_hash_changes_on_any_retrieval_defining_field() -> None:
    """Changing any retrieval-defining field changes ``config_hash``."""
    anchors = np.array([_epoch_day("2024-04-15")], dtype=np.int64)
    base = _default_config(anchors)
    base_h = base.config_hash
    variants: dict[str, RetrievalCacheConfig] = {
        "shots_fingerprint": _default_config(anchors, shots_fingerprint="other_v0"),
        "anchor_dates": _default_config(np.array([_epoch_day("2024-04-16")], dtype=np.int64)),
        "own_support_max": _default_config(anchors, own_support_max=8),
        "pooled_support_max": _default_config(anchors, pooled_support_max=8),
        "pooled_recency_window_days": _default_config(anchors, pooled_recency_window_days=60),
        "pooled_recency_half_life_days": _default_config(
            anchors, pooled_recency_half_life_days=60.0
        ),
        "seed": _default_config(anchors, seed=1),
        "traits_hash": _default_config(anchors, traits_hash="other_traits"),
    }
    for field, variant in variants.items():
        assert variant.config_hash != base_h, f"{field} change did not alter config_hash"
    # Same fields → same hash (sanity).
    assert _default_config(anchors).config_hash == base_h


def test_traits_fingerprint_tracks_trait_values() -> None:
    """Equal trait values give equal fingerprints; any changed value gives a new one."""
    traits = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    changed = traits.clone()
    changed[0, 0, 0] += 1.0
    assert traits_fingerprint(traits) == traits_fingerprint(traits.clone().numpy())
    assert traits_fingerprint(traits) != traits_fingerprint(changed)


def test_9_atomic_save_leaves_no_tmp_file(tmp_path: Path) -> None:
    """After a successful save, no leftover ``.tmp`` file remains."""
    shots, vocab, anchors, traits = _build_simple_fixture()
    cfg = _default_config(anchors)
    c = build_retrieval_cache(
        shots_df=shots, player_vocab=vocab, anchor_dates=anchors, traits=traits, config=cfg
    )
    save_path = tmp_path / "cache.pt"
    c.save(save_path)
    assert save_path.exists()
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == [], f"leftover .tmp files: {leftover}"


# ---------------------------------------------------------------------------
# Smaller helpers
# ---------------------------------------------------------------------------


def test_shots_fingerprint_changes_with_mtime(tmp_path: Path) -> None:
    p = tmp_path / "shots.csv"
    p.write_text("x,y,player_id,date\n0,0,1,2024-01-01\n")
    h0 = shots_fingerprint(p)
    # Touch with a strictly larger mtime.
    import os
    import time

    time.sleep(0.01)
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    h1 = shots_fingerprint(p)
    assert h0 != h1


def test_config_validates_positive_fields() -> None:
    anchors = (_epoch_day("2024-04-15"),)
    with pytest.raises(ValueError, match="own_support_max"):
        RetrievalCacheConfig(shots_fingerprint="x", anchor_dates=anchors, own_support_max=0)
    with pytest.raises(ValueError, match="pooled_support_max"):
        RetrievalCacheConfig(shots_fingerprint="x", anchor_dates=anchors, pooled_support_max=0)
    with pytest.raises(ValueError, match="pooled_recency_window_days"):
        RetrievalCacheConfig(
            shots_fingerprint="x", anchor_dates=anchors, pooled_recency_window_days=0
        )
    with pytest.raises(ValueError, match="pooled_recency_half_life_days"):
        RetrievalCacheConfig(
            shots_fingerprint="x",
            anchor_dates=anchors,
            pooled_recency_half_life_days=0.0,
        )
    with pytest.raises(ValueError, match="similarity_kind"):
        RetrievalCacheConfig(
            shots_fingerprint="x",
            anchor_dates=anchors,
            similarity_kind="dot_product",
        )


def test_build_rejects_traits_hash_mismatch() -> None:
    """A config keyed to one trait table cannot be used with another."""
    shots, vocab, anchors, traits = _build_simple_fixture()
    cfg = _default_config(anchors, traits_hash=traits_fingerprint(traits))
    build_retrieval_cache(
        shots_df=shots, player_vocab=vocab, anchor_dates=anchors, traits=traits, config=cfg
    )
    with pytest.raises(ValueError, match="traits_hash"):
        build_retrieval_cache(
            shots_df=shots,
            player_vocab=vocab,
            anchor_dates=anchors,
            traits=traits + 1.0,
            config=cfg,
        )


def test_build_rejects_traits_shape_mismatch() -> None:
    shots, vocab, anchors, _traits = _build_simple_fixture()
    cfg = _default_config(anchors)
    # Wrong number of snapshots in traits.
    bad_traits = torch.zeros(len(vocab), anchors.shape[0] + 1, 3)
    with pytest.raises(ValueError, match="traits must be"):
        build_retrieval_cache(
            shots_df=shots,
            player_vocab=vocab,
            anchor_dates=anchors,
            traits=bad_traits,
            config=cfg,
        )
    # Wrong P.
    bad_traits = torch.zeros(len(vocab) + 1, anchors.shape[0], 3)
    with pytest.raises(ValueError, match=r"traits\.shape"):
        build_retrieval_cache(
            shots_df=shots,
            player_vocab=vocab,
            anchor_dates=anchors,
            traits=bad_traits,
            config=cfg,
        )
