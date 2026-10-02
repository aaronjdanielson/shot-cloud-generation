"""Tests for :mod:`shotcloud.data.role_profile`.

The role profile is a closed-form, deterministic 8-dim summary of a
player's shot geometry. These tests cover:

* simplex behavior of the five zone rates,
* normalization / unit-range invariants,
* player-archetype distinguishability (rim-big vs corner-three),
* deterministic reproducibility (no rng),
* causal-window invariance (function depends only on rows passed in),
* sparse-history stability (low-shot-count edge cases),
* shape contracts,
* DataFrame wrapper output.

The function is stateless; causality is the caller's responsibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shotcloud.data import (
    ROLE_FEATURE_NAMES,
    ROLE_PROFILE_DIM,
    build_role_profiles,
    build_role_profiles_dataframe,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _shot_at(x: float, y: float, pid: int = 1) -> dict[str, float | int]:
    return {"player_id": pid, "x": x, "y": y}


def _rim_big_shots(pid: int = 1, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Player whose shots cluster in the restricted area."""
    rng = np.random.default_rng(seed)
    rows = [
        _shot_at(float(rng.normal(0.0, 1.5)), float(rng.normal(2.5, 1.5)), pid) for _ in range(n)
    ]
    return pd.DataFrame(rows)


def _corner_three_shots(pid: int = 2, n: int = 200, seed: int = 1) -> pd.DataFrame:
    """Player whose shots cluster in the corner three zones."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        side = 1 if rng.random() < 0.5 else -1
        rows.append(
            _shot_at(
                float(side * rng.normal(23.0, 0.8)),
                float(rng.normal(4.0, 1.5)),
                pid,
            )
        )
    return pd.DataFrame(rows)


def _midrange_shots(pid: int = 3, n: int = 200, seed: int = 2) -> pd.DataFrame:
    """Player whose shots cluster in the midrange."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        # Midrange: r in [10, 20] ft, any angle in upper half
        r = float(rng.uniform(10.0, 20.0))
        theta = float(rng.uniform(0.2, np.pi - 0.2))
        rows.append(_shot_at(r * np.cos(theta), r * np.sin(theta), pid))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shape and basic structure
# ---------------------------------------------------------------------------


def test_role_profile_shape_matches_role_profile_dim() -> None:
    shots = _rim_big_shots()
    profiles = build_role_profiles(shots)
    assert 1 in profiles
    assert profiles[1].shape == (ROLE_PROFILE_DIM,)
    assert profiles[1].dtype == np.float32


def test_role_feature_names_length_matches_role_profile_dim() -> None:
    assert len(ROLE_FEATURE_NAMES) == ROLE_PROFILE_DIM


def test_empty_shots_returns_empty_dict() -> None:
    empty = pd.DataFrame(columns=["player_id", "x", "y"])
    profiles = build_role_profiles(empty)
    assert profiles == {}


def test_missing_required_column_raises() -> None:
    bad = pd.DataFrame({"player_id": [1], "x": [0.0]})  # missing y
    with pytest.raises(ValueError, match="missing required column 'y'"):
        build_role_profiles(bad)


# ---------------------------------------------------------------------------
# Simplex invariants
# ---------------------------------------------------------------------------


def test_zone_rates_sum_to_one_per_player() -> None:
    shots = pd.concat([_rim_big_shots(1), _corner_three_shots(2), _midrange_shots(3)])
    profiles = build_role_profiles(shots)
    for pid, vec in profiles.items():
        rate_sum = float(vec[:5].sum())  # rim + paint + midrange + corner3 + atb3
        assert abs(rate_sum - 1.0) < 1e-5, f"player {pid} zone rates sum to {rate_sum}, not 1"


def test_zone_rates_are_in_unit_interval() -> None:
    shots = pd.concat([_rim_big_shots(1), _corner_three_shots(2), _midrange_shots(3)])
    profiles = build_role_profiles(shots)
    for vec in profiles.values():
        rates = vec[:5]
        assert (rates >= 0.0).all() and (rates <= 1.0).all()


# ---------------------------------------------------------------------------
# Player-archetype distinguishability
# ---------------------------------------------------------------------------


def test_rim_big_has_high_rim_rate() -> None:
    shots = _rim_big_shots(1)
    profiles = build_role_profiles(shots)
    # ~76% rim shots given the synthetic cluster width; some leak into
    # the surrounding paint zone — both signal a rim-anchored shooter.
    assert profiles[1][0] > 0.7  # rim_rate
    assert profiles[1][0] + profiles[1][1] > 0.95  # rim + paint dominates
    assert profiles[1][3] < 0.05  # corner3_rate ≈ 0
    assert profiles[1][4] < 0.05  # atb3_rate ≈ 0


def test_corner_three_specialist_has_high_corner3_rate() -> None:
    shots = _corner_three_shots(2)
    profiles = build_role_profiles(shots)
    assert profiles[2][3] > 0.8  # corner3_rate
    assert profiles[2][0] < 0.05  # rim_rate ≈ 0


def test_midrange_specialist_has_high_midrange_rate() -> None:
    shots = _midrange_shots(3)
    profiles = build_role_profiles(shots)
    assert profiles[3][2] > 0.5  # midrange_rate; midrange is broad, not as concentrated
    assert profiles[3][0] < 0.1  # rim_rate ≈ 0


def test_distinct_archetypes_produce_distinguishable_profiles() -> None:
    shots = pd.concat([_rim_big_shots(1), _corner_three_shots(2), _midrange_shots(3)])
    profiles = build_role_profiles(shots)
    # L1 distance between any two distinct-archetype profiles is large
    big = profiles[1]
    corner = profiles[2]
    mid = profiles[3]
    assert np.abs(big - corner).sum() > 1.0
    assert np.abs(big - mid).sum() > 1.0
    assert np.abs(corner - mid).sum() > 1.0


# ---------------------------------------------------------------------------
# Distance and entropy features
# ---------------------------------------------------------------------------


def test_rim_big_has_smaller_mean_dist_than_corner_three() -> None:
    shots = pd.concat([_rim_big_shots(1), _corner_three_shots(2)])
    profiles = build_role_profiles(shots)
    big_mean_dist = profiles[1][5]
    corner_mean_dist = profiles[2][5]
    assert big_mean_dist < corner_mean_dist


def test_normalized_features_within_unit_range() -> None:
    shots = pd.concat([_rim_big_shots(1), _corner_three_shots(2), _midrange_shots(3)])
    profiles = build_role_profiles(shots, normalize=True)
    for vec in profiles.values():
        # All 8 coordinates in [0, 1.5] (allowing some headroom for outlier shots).
        assert (vec >= 0.0).all()
        assert (vec <= 1.5).all()


def test_unnormalized_distance_features_in_feet() -> None:
    shots = _rim_big_shots(1)
    profiles_raw = build_role_profiles(shots, normalize=False)
    profiles_norm = build_role_profiles(shots, normalize=True)
    raw_mean = profiles_raw[1][5]
    norm_mean = profiles_norm[1][5]
    # Normalized = raw / 30.0
    assert abs(raw_mean - norm_mean * 30.0) < 1e-3


def test_concentrated_player_has_lower_entropy_than_diverse_player() -> None:
    """A player who shoots from one tiny patch has lower entropy than
    a player who covers the full half-court."""
    rng = np.random.default_rng(7)
    concentrated = pd.DataFrame(
        [_shot_at(float(rng.normal(0, 0.3)), float(rng.normal(2.0, 0.3)), 1) for _ in range(200)]
    )
    diverse = pd.DataFrame(
        [_shot_at(float(rng.uniform(-22, 22)), float(rng.uniform(0, 35)), 2) for _ in range(200)]
    )
    profiles = build_role_profiles(pd.concat([concentrated, diverse]))
    entropy_concentrated = profiles[1][7]
    entropy_diverse = profiles[2][7]
    assert entropy_concentrated < entropy_diverse


# ---------------------------------------------------------------------------
# Determinism + causal-window invariance
# ---------------------------------------------------------------------------


def test_deterministic_output() -> None:
    """Identical input shots produce identical role profiles."""
    shots = _rim_big_shots(1, n=150, seed=42)
    p1 = build_role_profiles(shots)
    p2 = build_role_profiles(shots)
    np.testing.assert_array_equal(p1[1], p2[1])


def test_function_depends_only_on_passed_rows() -> None:
    """Filtering shots before passing them in is what gives causality.
    The function output for player p must be identical whether we pass
    the full frame or pre-filter to p's rows."""
    shots = pd.concat([_rim_big_shots(1), _corner_three_shots(2)])
    profile_full = build_role_profiles(shots)[1]

    only_player_1 = shots[shots["player_id"] == 1]
    profile_filtered = build_role_profiles(only_player_1)[1]

    np.testing.assert_array_equal(profile_full, profile_filtered)


def test_causal_window_changes_change_profile_predictably() -> None:
    """Profile fit on a strict prefix of shots is *different* from profile
    fit on the full pool — confirms the function honors the filter."""
    rng = np.random.default_rng(0)
    early = pd.DataFrame(
        [_shot_at(float(rng.normal(0, 1)), float(rng.normal(2, 1)), 1) for _ in range(100)]
    )  # rim big at first
    late = pd.DataFrame(
        [_shot_at(float(rng.normal(20, 1)), float(rng.normal(20, 1)), 1) for _ in range(100)]
    )  # then 3-point shooter

    profile_early = build_role_profiles(early)[1]
    profile_combined = build_role_profiles(pd.concat([early, late]))[1]

    assert not np.allclose(profile_early, profile_combined)
    # Specifically: rim rate is much higher in the early-only profile.
    assert profile_early[0] > profile_combined[0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_shot_player_produces_well_defined_profile() -> None:
    """A player with exactly 1 shot has std_dist=0 and entropy=0
    (no spread information), but other features are well-defined."""
    shots = pd.DataFrame([_shot_at(0.0, 2.5, 1)])  # single rim shot
    profiles = build_role_profiles(shots)
    assert 1 in profiles
    vec = profiles[1]
    assert vec[0] == 1.0  # rim_rate = 1
    assert vec[6] == 0.0  # std_dist = 0
    assert vec[7] == 0.0  # entropy = 0


def test_min_shots_filter_drops_low_count_players() -> None:
    shots = pd.concat(
        [
            pd.DataFrame([_shot_at(0.0, 2.5, 1) for _ in range(100)]),
            pd.DataFrame([_shot_at(20.0, 20.0, 2)]),  # only 1 shot
        ]
    )
    profiles_no_filter = build_role_profiles(shots, min_shots=1)
    profiles_with_filter = build_role_profiles(shots, min_shots=50)
    assert 1 in profiles_no_filter and 2 in profiles_no_filter
    assert 1 in profiles_with_filter and 2 not in profiles_with_filter


def test_negative_min_shots_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_role_profiles(_rim_big_shots(), min_shots=-1)


def test_backcourt_shots_excluded_from_zone_rates() -> None:
    """Shots beyond half-court (y > 47) get zone -1; they should be
    excluded from the zone-rate denominator. Mean/std/entropy still
    use them since those are positional."""
    shots = pd.concat(
        [
            _rim_big_shots(1, n=20),
            pd.DataFrame([_shot_at(0.0, 60.0, 1)]),  # backcourt heave
        ]
    )
    profiles = build_role_profiles(shots)
    rate_sum = float(profiles[1][:5].sum())
    assert abs(rate_sum - 1.0) < 1e-5  # rates still sum to 1


# ---------------------------------------------------------------------------
# DataFrame wrapper
# ---------------------------------------------------------------------------


def test_dataframe_wrapper_returns_named_columns() -> None:
    shots = pd.concat([_rim_big_shots(1), _corner_three_shots(2)])
    df = build_role_profiles_dataframe(shots)
    assert df.index.name == "player_id"
    assert sorted(df.index.tolist()) == [1, 2]
    for col in ROLE_FEATURE_NAMES:
        assert col in df.columns


def test_dataframe_wrapper_empty_input() -> None:
    df = build_role_profiles_dataframe(pd.DataFrame(columns=["player_id", "x", "y"]))
    assert len(df) == 0
    for col in ROLE_FEATURE_NAMES:
        assert col in df.columns


# ---------------------------------------------------------------------------
# Snapshot integration
# ---------------------------------------------------------------------------


def test_role_profile_works_as_snapshot_role_fn() -> None:
    """build_role_profiles is the canonical role_profile_fn for
    build_snapshot_store_from_shots: pass shots, get back a dict
    keyed by player_id with 8-dim float32 vectors."""
    from shotcloud.data import build_snapshot_store_from_shots

    shots = pd.concat([_rim_big_shots(1), _corner_three_shots(2)])
    shots["date"] = pd.date_range("2018-01-01", periods=len(shots), freq="D")
    shots["opponent"] = "BOS"
    shots["made"] = 1

    anchors = [np.datetime64("2018-09-01"), np.datetime64("2019-01-01")]
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=build_role_profiles,
    )

    for bundle in store.bundles:
        # Causality: every shot indexed in the bundle has date < anchor
        bundle.assert_causal(shots)
        # Role profiles are populated and shaped correctly
        assert bundle.role_profiles.shape == (len(bundle.player_ids), ROLE_PROFILE_DIM)
        # Rim big (player 1) should have higher rim_rate than corner specialist (player 2)
        idx_1 = bundle.player_idx(1)
        idx_2 = bundle.player_idx(2)
        if idx_1 is not None and idx_2 is not None:
            assert bundle.role_profiles[idx_1, 0] > bundle.role_profiles[idx_2, 0]
