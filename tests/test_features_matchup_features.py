"""Tests for the Tier-2a v2 (group-level) D-matchup feature pipeline.

Scope:

* ``MatchupFeaturesConfig`` hash sensitivity to each tunable field
  (including the new ``grouping_K``, ``grouping_seed``, ``traits_hash``).
* ``assign_player_groups``: K-means assignment shape + determinism.
* ``build_matchup_features`` arithmetic on a small synthetic dataset
  where we can hand-compute the expected Δ^int / Δ̂ / N^eff at the
  group level.
* ``K = 1`` limit: matchup term collapses to exactly zero (the
  orthogonal-decomposition identity made explicit).
* Players in the same group share Δ̂ exactly (by construction).
* Shrinkage limits: at very-large τ, Δ̂ ≈ 0; at τ → 0 with N_eff > 0,
  Δ̂ ≈ Δ^int.
* Causal mask: shots at date ≥ anchor must not contribute.
* Cold-start cell (group has no shots vs opponent d at anchor t):
  N^eff = 0, Δ̂ = 0 exactly.
* ``save`` / ``load`` round-trip preserves every field.
* Cache reuse: builder reads from ``cache_dir`` on the second call
  with the same config hash without rerunning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud.data.player_traits import PlayerTraitsTable
from shotcloud.data.zones import N_ZONES, zone_from_xy_vectorized
from shotcloud.features.matchup_features import (
    DEFAULT_MATCHUP_HALF_LIFE_DAYS,
    DEFAULT_MATCHUP_WINDOW_DAYS,
    MatchupFeatures,
    MatchupFeaturesConfig,
    assign_player_groups,
    build_matchup_features,
    traits_table_hash,
)
from shotcloud.training.dataset import OpponentVocab, PlayerVocab

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_synthetic_shots(
    shots_spec: list[tuple[str, str, int, float, float]],
) -> pd.DataFrame:
    """Build a small canonical shots DataFrame from a list of tuples.

    Each tuple is ``(player_id, opponent, date_ord_epoch_days, x, y)``.
    """
    rows = [
        {
            "player_id": p,
            "opponent": o,
            "date": np.datetime64(int(d), "D"),
            "x": float(x),
            "y": float(y),
        }
        for p, o, d, x, y in shots_spec
    ]
    return pd.DataFrame(rows)


def _zone_coords() -> dict[int, tuple[float, float]]:
    """One canonical (x, y) per zone — basket at origin, ft."""
    coords = {
        0: (0.0, 2.0),  # RA
        1: (0.0, 8.0),  # Paint
        2: (10.0, 18.0),  # Midrange
        3: (-23.0, 5.0),  # Corner3-L
        4: (23.0, 5.0),  # Corner3-R
        5: (-15.0, 20.0),  # Wing3-L
        6: (15.0, 20.0),  # Wing3-R
        7: (0.0, 25.0),  # TopKey3
    }
    xs = np.array([c[0] for c in coords.values()])
    ys = np.array([c[1] for c in coords.values()])
    expected = np.array(list(coords.keys()))
    got = zone_from_xy_vectorized(xs, ys)
    assert (got == expected).all(), (
        f"zone coord helper drifted: {dict(zip(expected, got, strict=True))}"
    )
    return coords


def _make_traits_table(
    *,
    player_ids: np.ndarray,
    n_snapshots: int,
    trait_dim: int = 4,
    seed: int = 0,
) -> PlayerTraitsTable:
    """Build a synthetic PlayerTraitsTable with random trait values."""
    rng = np.random.default_rng(seed)
    traits = rng.standard_normal((len(player_ids), n_snapshots, trait_dim), dtype=np.float32)
    return PlayerTraitsTable(
        traits=traits,
        player_ids=player_ids.astype(np.int64),
        snapshot_anchors=np.array(
            [np.datetime64("2024-01-01", "D")] * n_snapshots, dtype="datetime64[D]"
        ),
    )


# --------------------------------------------------------------------------- #
# Config hashing
# --------------------------------------------------------------------------- #


def test_config_hash_is_stable_for_same_inputs() -> None:
    a = MatchupFeaturesConfig(
        shots_fingerprint="fp",
        traits_hash="th",
        opp_vocab_hash="ovh",
        anchor_dates=(100, 200),
    )
    b = MatchupFeaturesConfig(
        shots_fingerprint="fp",
        traits_hash="th",
        opp_vocab_hash="ovh",
        anchor_dates=(100, 200),
    )
    assert a.config_hash == b.config_hash
    assert len(a.config_hash) == 16


def test_config_hash_changes_with_each_field() -> None:
    base = MatchupFeaturesConfig(
        shots_fingerprint="fp",
        traits_hash="th",
        opp_vocab_hash="ovh",
        anchor_dates=(100,),
        grouping_K=5,
        grouping_seed=0,
        window_days=365,
        half_life_days=90.0,
        tau=20.0,
        seed=0,
    )
    base_h = base.config_hash
    perturbations: list[dict[str, object]] = [
        {"shots_fingerprint": "fp2"},
        {"traits_hash": "th2"},
        {"opp_vocab_hash": "ovh2"},
        {"anchor_dates": (100, 200)},
        {"grouping_K": 8},
        {"grouping_seed": 1},
        {"window_days": 180},
        {"half_life_days": 30.0},
        {"tau": 50.0},
        {"seed": 1},
    ]
    base_kwargs: dict[str, object] = {
        "shots_fingerprint": base.shots_fingerprint,
        "traits_hash": base.traits_hash,
        "opp_vocab_hash": base.opp_vocab_hash,
        "anchor_dates": base.anchor_dates,
        "grouping_K": base.grouping_K,
        "grouping_seed": base.grouping_seed,
        "window_days": base.window_days,
        "half_life_days": base.half_life_days,
        "tau": base.tau,
        "seed": base.seed,
    }
    for delta in perturbations:
        cfg = MatchupFeaturesConfig(**{**base_kwargs, **delta})  # type: ignore[arg-type]
        assert cfg.config_hash != base_h, f"hash unchanged under perturbation {delta}"


def test_config_validation() -> None:
    kwargs: dict[str, object] = {
        "shots_fingerprint": "fp",
        "traits_hash": "th",
        "opp_vocab_hash": "ovh",
        "anchor_dates": (100,),
    }
    with pytest.raises(ValueError, match="grouping_K must be"):
        MatchupFeaturesConfig(**kwargs, grouping_K=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="window_days must be positive"):
        MatchupFeaturesConfig(**kwargs, window_days=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="half_life_days must be positive"):
        MatchupFeaturesConfig(**kwargs, half_life_days=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tau must be positive"):
        MatchupFeaturesConfig(**kwargs, tau=0.0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# assign_player_groups
# --------------------------------------------------------------------------- #


def test_assign_player_groups_shape_and_range() -> None:
    traits = _make_traits_table(player_ids=np.arange(20), n_snapshots=3, trait_dim=4)
    groups = assign_player_groups(traits=traits, K=5, seed=0)
    assert groups.shape == (20, 3)
    assert groups.dtype == np.int64
    assert int(groups.min()) >= 0
    assert int(groups.max()) < 5


def test_assign_player_groups_is_deterministic_with_seed() -> None:
    traits = _make_traits_table(player_ids=np.arange(20), n_snapshots=2, trait_dim=4, seed=42)
    a = assign_player_groups(traits=traits, K=5, seed=7)
    b = assign_player_groups(traits=traits, K=5, seed=7)
    np.testing.assert_array_equal(a, b)


def test_assign_player_groups_K1_is_all_zero() -> None:
    """K=1 puts every player in group 0 at every snapshot."""
    traits = _make_traits_table(player_ids=np.arange(10), n_snapshots=2, trait_dim=4)
    groups = assign_player_groups(traits=traits, K=1, seed=0)
    np.testing.assert_array_equal(groups, np.zeros_like(groups))


def test_assign_player_groups_rejects_bad_K() -> None:
    traits = _make_traits_table(player_ids=np.arange(5), n_snapshots=1, trait_dim=3)
    with pytest.raises(ValueError, match="K must be"):
        assign_player_groups(traits=traits, K=0, seed=0)
    with pytest.raises(ValueError, match="exceeds n_players"):
        assign_player_groups(traits=traits, K=10, seed=0)


def test_traits_table_hash_changes_with_values() -> None:
    a = _make_traits_table(player_ids=np.arange(5), n_snapshots=2, trait_dim=3, seed=0)
    b = _make_traits_table(player_ids=np.arange(5), n_snapshots=2, trait_dim=3, seed=1)
    assert traits_table_hash(a) != traits_table_hash(b)
    # Same trait values → same hash.
    c = PlayerTraitsTable(
        traits=a.traits.copy(),
        player_ids=a.player_ids.copy(),
        snapshot_anchors=a.snapshot_anchors.copy(),
    )
    assert traits_table_hash(a) == traits_table_hash(c)


# --------------------------------------------------------------------------- #
# K=1 collapse limit
# --------------------------------------------------------------------------- #


def _simple_shots_two_groups() -> tuple[pd.DataFrame, OpponentVocab, PlayerVocab, np.ndarray]:
    """Two players (P0, P1), two opponents (TEAM_A, TEAM_B), one snapshot.

    Shots (date 50, anchor 100, window 365), all flat-weighted via a
    huge half-life:

    * P0 vs A: 1× RA, 1× Paint
    * P0 vs B: 2× TopKey3
    * P1 vs A: 1× Paint, 1× TopKey3
    * P1 vs B: 1× RA
    """
    z = _zone_coords()
    rim, paint, top = z[0], z[1], z[7]
    shots_spec = [
        ("P0", "TEAM_A", 50, *rim),
        ("P0", "TEAM_A", 50, *paint),
        ("P0", "TEAM_B", 50, *top),
        ("P0", "TEAM_B", 50, *top),
        ("P1", "TEAM_A", 50, *paint),
        ("P1", "TEAM_A", 50, *top),
        ("P1", "TEAM_B", 50, *rim),
    ]
    df = _make_synthetic_shots(shots_spec)
    player_vocab = PlayerVocab.from_ids(["P0", "P1"])
    opp_vocab = OpponentVocab.from_ids(["TEAM_A", "TEAM_B"])
    anchor_dates = np.array([100], dtype=np.int64)
    return df, opp_vocab, player_vocab, anchor_dates


def test_K1_collapses_delta_to_exact_zero() -> None:
    """The orthogonal-decomposition identity, made explicit.

    With K=1, P^vs_{1,d}(z) ≡ P^allow_d(z) by construction, so
    Δ^int = (P^allow − P^lg) − (P^allow − P^lg) = 0 identically.
    """
    df, opp_vocab, player_vocab, anchor_dates = _simple_shots_two_groups()
    cfg = MatchupFeaturesConfig(
        shots_fingerprint="syn",
        traits_hash="syn",
        opp_vocab_hash="syn",
        anchor_dates=(100,),
        grouping_K=1,
        window_days=365,
        half_life_days=10_000.0,
        tau=1e-6,
    )
    groups = np.zeros((2, 1), dtype=np.int64)  # K=1 → all-zero
    feats = build_matchup_features(
        shots_df=df,
        groups=groups,
        opp_vocab=opp_vocab,
        player_vocab=player_vocab,
        anchor_dates=anchor_dates,
        config=cfg,
    )
    np.testing.assert_allclose(feats.delta_int.numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(feats.delta_hat.numpy(), 0.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Group-level arithmetic
# --------------------------------------------------------------------------- #


def test_same_group_players_share_delta_hat_exactly() -> None:
    """Any two players assigned to the same group must have
    bit-identical Δ̂ rows at every (snapshot, opp). This is the
    structural payoff of group-level aggregation — within a group,
    matchup is shared by construction."""
    z = _zone_coords()
    rim, paint, top = z[0], z[1], z[7]
    shots_spec = [
        ("P0", "TEAM_A", 50, *rim),
        ("P1", "TEAM_A", 50, *paint),
        ("P2", "TEAM_A", 50, *top),
        ("P3", "TEAM_B", 50, *rim),
        ("P0", "TEAM_B", 50, *paint),
        ("P1", "TEAM_B", 50, *top),
        ("P2", "TEAM_B", 50, *rim),
        ("P3", "TEAM_A", 50, *paint),
    ]
    df = _make_synthetic_shots(shots_spec)
    player_vocab = PlayerVocab.from_ids(["P0", "P1", "P2", "P3"])
    opp_vocab = OpponentVocab.from_ids(["TEAM_A", "TEAM_B"])
    anchor_dates = np.array([100], dtype=np.int64)
    # Hand-built groups: P0 and P2 in group 0; P1 and P3 in group 1.
    groups = np.array([[0], [1], [0], [1]], dtype=np.int64)
    cfg = MatchupFeaturesConfig(
        shots_fingerprint="syn",
        traits_hash="syn",
        opp_vocab_hash="syn",
        anchor_dates=(100,),
        grouping_K=2,
        window_days=365,
        half_life_days=10_000.0,
        tau=1e-6,
    )
    feats = build_matchup_features(
        shots_df=df,
        groups=groups,
        opp_vocab=opp_vocab,
        player_vocab=player_vocab,
        anchor_dates=anchor_dates,
        config=cfg,
    )
    dh = feats.delta_hat.numpy()
    # Same-group players → identical Δ̂.
    np.testing.assert_allclose(dh[0], dh[2], atol=1e-6)  # both in group 0
    np.testing.assert_allclose(dh[1], dh[3], atol=1e-6)  # both in group 1
    # Cross-group: at least one cell differs (otherwise the test is
    # vacuous). Different shot patterns guarantee this on this fixture.
    assert not np.allclose(dh[0], dh[1], atol=1e-3)


def test_shrinkage_limits() -> None:
    df, opp_vocab, player_vocab, anchor_dates = _simple_shots_two_groups()
    # K=2: P0 alone in group 0, P1 alone in group 1. Each group has
    # one player, so the (group, opp, zone) cell is single-player —
    # but the group infrastructure is exercised.
    groups = np.array([[0], [1]], dtype=np.int64)

    # τ → ∞ → Δ̂ ≈ 0.
    cfg_big = MatchupFeaturesConfig(
        shots_fingerprint="syn",
        traits_hash="syn",
        opp_vocab_hash="syn",
        anchor_dates=(100,),
        grouping_K=2,
        window_days=365,
        half_life_days=10_000.0,
        tau=1e6,
    )
    feats_big = build_matchup_features(
        shots_df=df,
        groups=groups,
        opp_vocab=opp_vocab,
        player_vocab=player_vocab,
        anchor_dates=anchor_dates,
        config=cfg_big,
    )
    nonzero = np.abs(feats_big.delta_int.numpy()) > 1e-6
    if nonzero.any():
        r = float(
            np.abs(
                feats_big.delta_hat.numpy()[nonzero] / feats_big.delta_int.numpy()[nonzero]
            ).max()
        )
        assert r < 1e-3, f"big-τ shrinkage ineffective: ratio={r}"

    # τ → 0 → Δ̂ ≈ Δ^int.
    cfg_small = MatchupFeaturesConfig(
        shots_fingerprint="syn",
        traits_hash="syn",
        opp_vocab_hash="syn",
        anchor_dates=(100,),
        grouping_K=2,
        window_days=365,
        half_life_days=10_000.0,
        tau=1e-6,
    )
    feats_small = build_matchup_features(
        shots_df=df,
        groups=groups,
        opp_vocab=opp_vocab,
        player_vocab=player_vocab,
        anchor_dates=anchor_dates,
        config=cfg_small,
    )
    np.testing.assert_allclose(
        feats_small.delta_hat.numpy(), feats_small.delta_int.numpy(), atol=1e-5
    )


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #


def test_shots_at_or_after_anchor_do_not_contribute() -> None:
    z = _zone_coords()
    rim, paint = z[0], z[1]
    df_pre = _make_synthetic_shots([("P0", "TEAM_A", 50, *rim)])
    df_post = _make_synthetic_shots(
        [
            ("P0", "TEAM_A", 50, *rim),
            ("P0", "TEAM_A", 150, *paint),  # after anchor — must be ignored
        ]
    )
    player_vocab = PlayerVocab.from_ids(["P0"])
    opp_vocab = OpponentVocab.from_ids(["TEAM_A"])
    groups = np.zeros((1, 1), dtype=np.int64)
    cfg = MatchupFeaturesConfig(
        shots_fingerprint="caus",
        traits_hash="caus",
        opp_vocab_hash="caus",
        anchor_dates=(100,),
        grouping_K=1,
        window_days=365,
        half_life_days=10_000.0,
        tau=1e-6,
    )
    f_pre = build_matchup_features(
        shots_df=df_pre,
        groups=groups,
        opp_vocab=opp_vocab,
        player_vocab=player_vocab,
        anchor_dates=np.array([100], dtype=np.int64),
        config=cfg,
    )
    f_post = build_matchup_features(
        shots_df=df_post,
        groups=groups,
        opp_vocab=opp_vocab,
        player_vocab=player_vocab,
        anchor_dates=np.array([100], dtype=np.int64),
        config=cfg,
    )
    np.testing.assert_allclose(f_pre.n_eff.numpy(), f_post.n_eff.numpy(), atol=1e-6)
    np.testing.assert_allclose(f_pre.delta_int.numpy(), f_post.delta_int.numpy(), atol=1e-6)


# --------------------------------------------------------------------------- #
# Cold-start cell
# --------------------------------------------------------------------------- #


def test_cold_start_cell_is_zero() -> None:
    """A (group, opp) cell with no causal shots vs that opponent
    produces N_eff = 0 and Δ̂ = 0 exactly."""
    z = _zone_coords()
    rim = z[0]
    # P0 only plays TEAM_A. The (group 0, TEAM_B) cell is cold-start.
    shots = _make_synthetic_shots([("P0", "TEAM_A", 50, *rim)])
    player_vocab = PlayerVocab.from_ids(["P0"])
    opp_vocab = OpponentVocab.from_ids(["TEAM_A", "TEAM_B"])
    groups = np.zeros((1, 1), dtype=np.int64)
    cfg = MatchupFeaturesConfig(
        shots_fingerprint="cold",
        traits_hash="cold",
        opp_vocab_hash="cold",
        anchor_dates=(100,),
        grouping_K=1,
        window_days=365,
        half_life_days=10_000.0,
        tau=1.0,
    )
    feats = build_matchup_features(
        shots_df=shots,
        groups=groups,
        opp_vocab=opp_vocab,
        player_vocab=player_vocab,
        anchor_dates=np.array([100], dtype=np.int64),
        config=cfg,
    )
    # Cold-start: (player=0, snap=0, opp=B=1) → all zeros.
    assert float(feats.n_eff[0, 0, 1].item()) == 0.0
    assert torch.equal(feats.delta_int[0, 0, 1, :], torch.zeros(N_ZONES))
    assert torch.equal(feats.delta_hat[0, 0, 1, :], torch.zeros(N_ZONES))


# --------------------------------------------------------------------------- #
# Round-trip + cache
# --------------------------------------------------------------------------- #


def test_save_load_round_trip(tmp_path: Path) -> None:
    df, opp_vocab, player_vocab, anchor_dates = _simple_shots_two_groups()
    cfg = MatchupFeaturesConfig(
        shots_fingerprint="rt",
        traits_hash="rt",
        opp_vocab_hash="rt",
        anchor_dates=(100,),
        grouping_K=2,
        window_days=365,
        half_life_days=DEFAULT_MATCHUP_HALF_LIFE_DAYS,
    )
    groups = np.array([[0], [1]], dtype=np.int64)
    feats = build_matchup_features(
        shots_df=df,
        groups=groups,
        opp_vocab=opp_vocab,
        player_vocab=player_vocab,
        anchor_dates=anchor_dates,
        config=cfg,
    )
    path = tmp_path / "matchup.pt"
    feats.save(path)
    loaded = MatchupFeatures.load(path)
    torch.testing.assert_close(loaded.delta_hat, feats.delta_hat)
    torch.testing.assert_close(loaded.delta_int, feats.delta_int)
    torch.testing.assert_close(loaded.n_eff, feats.n_eff)
    assert loaded.config.config_hash == cfg.config_hash


def test_cache_reuse(tmp_path: Path) -> None:
    df, opp_vocab, player_vocab, anchor_dates = _simple_shots_two_groups()
    cfg = MatchupFeaturesConfig(
        shots_fingerprint="cache",
        traits_hash="cache",
        opp_vocab_hash="cache",
        anchor_dates=(100,),
        grouping_K=2,
        window_days=365,
        half_life_days=DEFAULT_MATCHUP_WINDOW_DAYS * 1.0,
    )
    groups = np.array([[0], [1]], dtype=np.int64)
    a = build_matchup_features(
        shots_df=df,
        groups=groups,
        opp_vocab=opp_vocab,
        player_vocab=player_vocab,
        anchor_dates=anchor_dates,
        config=cfg,
        cache_dir=tmp_path,
    )
    expected_path = tmp_path / f"matchup_features_{cfg.config_hash}.pt"
    assert expected_path.exists()
    mtime_before = expected_path.stat().st_mtime_ns
    b = build_matchup_features(
        shots_df=df,
        groups=groups,
        opp_vocab=opp_vocab,
        player_vocab=player_vocab,
        anchor_dates=anchor_dates,
        config=cfg,
        cache_dir=tmp_path,
    )
    mtime_after = expected_path.stat().st_mtime_ns
    assert mtime_before == mtime_after, "cache file was rewritten — not a hit"
    torch.testing.assert_close(a.delta_hat, b.delta_hat)


def test_anchor_dates_mismatch_rejects() -> None:
    df, opp_vocab, player_vocab, anchor_dates = _simple_shots_two_groups()
    cfg = MatchupFeaturesConfig(
        shots_fingerprint="x",
        traits_hash="x",
        opp_vocab_hash="x",
        anchor_dates=(200,),  # disagrees with anchor_dates arg below
        grouping_K=2,
    )
    groups = np.array([[0], [1]], dtype=np.int64)
    with pytest.raises(ValueError, match="anchor_dates"):
        build_matchup_features(
            shots_df=df,
            groups=groups,
            opp_vocab=opp_vocab,
            player_vocab=player_vocab,
            anchor_dates=anchor_dates,
            config=cfg,
        )


def test_groups_shape_validation() -> None:
    df, opp_vocab, player_vocab, anchor_dates = _simple_shots_two_groups()
    cfg = MatchupFeaturesConfig(
        shots_fingerprint="x",
        traits_hash="x",
        opp_vocab_hash="x",
        anchor_dates=(100,),
        grouping_K=2,
    )
    # Wrong shape — should mismatch (n_players=2, n_snapshots=1).
    with pytest.raises(ValueError, match="groups must have shape"):
        build_matchup_features(
            shots_df=df,
            groups=np.zeros((3, 1), dtype=np.int64),
            opp_vocab=opp_vocab,
            player_vocab=player_vocab,
            anchor_dates=anchor_dates,
            config=cfg,
        )
    # Group value ≥ grouping_K.
    with pytest.raises(ValueError, match=r"config\.grouping_K"):
        build_matchup_features(
            shots_df=df,
            groups=np.array([[0], [5]], dtype=np.int64),  # 5 ≥ K=2
            opp_vocab=opp_vocab,
            player_vocab=player_vocab,
            anchor_dates=anchor_dates,
            config=cfg,
        )
