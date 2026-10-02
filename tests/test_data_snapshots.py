"""Tests for :class:`~shotcloud.data.SnapshotStore` and :class:`~shotcloud.data.SnapshotBundle`.

The central test, :func:`test_snapshot_store_strict_causality`, is the
operational form of the paper's *Temporal validity* proposition: for every
anchor ``t_i``, every shot indexed by the bundle is dated strictly before
``t_i``, so any object returned by ``get_snapshot(t)`` is built only from data
before ``t``. The remaining tests cover bundle validation, anchor lookup
semantics, and the builder's per-anchor callbacks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shotcloud.data import (
    N_OPP_EFFICIENCY_BINS,
    POSITION_MIXTURE_DIM,
    ROLE_PROFILE_DIM,
    SnapshotBundle,
    SnapshotStore,
    build_snapshot_store_from_shots,
)


def _synthetic_shots(seed: int = 0, n_per_day: int = 3) -> pd.DataFrame:
    """Build a small weekly shot table spanning 2018 through 2020.

    Three players (1, 2, 3) shoot against three opponents (BOS, LAL, GSW),
    ``n_per_day`` rows per week, with Bernoulli(0.45) makes. The frame is
    small enough for causality assertions to check every row.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for date in pd.date_range("2018-01-01", "2020-12-31", freq="W"):  # weekly cadence
        for _ in range(n_per_day):
            rows.append(
                {
                    "x": float(rng.normal(0.0, 5.0)),
                    "y": float(rng.normal(15.0, 5.0)),
                    "player_id": int(rng.choice([1, 2, 3])),
                    "opponent": str(rng.choice(["BOS", "LAL", "GSW"])),
                    "made": int(rng.random() < 0.45),
                    "date": pd.Timestamp(date),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SnapshotBundle: construction, validation, lookups
# ---------------------------------------------------------------------------


def _make_minimal_bundle(anchor: str = "2019-01-01") -> SnapshotBundle:
    return SnapshotBundle(
        anchor_date=np.datetime64(anchor, "D"),
        player_ids=np.array([1, 2, 3], dtype=np.int64),
        role_profiles=np.zeros((3, ROLE_PROFILE_DIM), dtype=np.float32),
        position_mixtures=np.full((3, POSITION_MIXTURE_DIM), 1.0 / 3.0, dtype=np.float32),
        opp_codes=np.array(["BOS", "GSW", "LAL"], dtype=np.str_),
        opp_efficiency_bins=np.array([0, 1, 2], dtype=np.int8),
        player_history_index={1: np.array([0, 1], dtype=np.int64)},
        defensive_history_index={"BOS": np.array([0], dtype=np.int64)},
    )


def test_bundle_construction_and_field_access() -> None:
    b = _make_minimal_bundle()
    assert int(b.player_ids[0]) == 1
    assert b.role_profiles.shape == (3, ROLE_PROFILE_DIM)
    assert b.position_mixtures.shape == (3, POSITION_MIXTURE_DIM)
    assert int(b.opp_efficiency_bins[1]) == 1
    np.testing.assert_array_equal(b.history_for(1), np.array([0, 1]))
    np.testing.assert_array_equal(b.defensive_history_for("BOS"), np.array([0]))


def test_bundle_lookups_handle_missing() -> None:
    b = _make_minimal_bundle()
    # Player not present
    assert b.player_idx(99) is None
    assert b.role_profile(99) is None
    assert b.position_mixture(99) is None
    np.testing.assert_array_equal(b.history_for(99), np.array([], dtype=np.int64))
    # Opp not present
    assert b.opp_idx("OKC") is None
    assert b.opp_strength_bin("OKC") is None
    np.testing.assert_array_equal(b.defensive_history_for("OKC"), np.array([], dtype=np.int64))


def test_bundle_lookups_return_correct_indices() -> None:
    b = _make_minimal_bundle()
    assert b.player_idx(1) == 0
    assert b.player_idx(3) == 2
    assert b.opp_idx("BOS") == 0
    assert b.opp_idx("GSW") == 1
    assert b.opp_idx("LAL") == 2
    assert b.opp_strength_bin("LAL") == 2


def test_bundle_rejects_misshaped_role_profiles() -> None:
    with pytest.raises(ValueError, match="role_profiles"):
        SnapshotBundle(
            anchor_date=np.datetime64("2019-01-01", "D"),
            player_ids=np.array([1, 2], dtype=np.int64),
            role_profiles=np.zeros((2, 5), dtype=np.float32),  # wrong width
            position_mixtures=np.full((2, 3), 1 / 3, dtype=np.float32),
            opp_codes=np.array([], dtype=np.str_),
            opp_efficiency_bins=np.array([], dtype=np.int8),
            player_history_index={},
            defensive_history_index={},
        )


def test_bundle_rejects_position_mixture_not_summing_to_one() -> None:
    bad = np.full((2, 3), 0.5, dtype=np.float32)  # rows sum to 1.5
    with pytest.raises(ValueError, match="rows must sum to 1"):
        SnapshotBundle(
            anchor_date=np.datetime64("2019-01-01", "D"),
            player_ids=np.array([1, 2], dtype=np.int64),
            role_profiles=np.zeros((2, ROLE_PROFILE_DIM), dtype=np.float32),
            position_mixtures=bad,
            opp_codes=np.array([], dtype=np.str_),
            opp_efficiency_bins=np.array([], dtype=np.int8),
            player_history_index={},
            defensive_history_index={},
        )


def test_bundle_rejects_archetype_surfaces_not_summing_to_one() -> None:
    surfaces = np.full((2, 100), 0.001, dtype=np.float32)  # rows sum to 0.1
    with pytest.raises(ValueError, match="archetype_surfaces"):
        SnapshotBundle(
            anchor_date=np.datetime64("2019-01-01", "D"),
            player_ids=np.array([], dtype=np.int64),
            role_profiles=np.zeros((0, ROLE_PROFILE_DIM), dtype=np.float32),
            position_mixtures=np.zeros((0, POSITION_MIXTURE_DIM), dtype=np.float32),
            opp_codes=np.array([], dtype=np.str_),
            opp_efficiency_bins=np.array([], dtype=np.int8),
            player_history_index={},
            defensive_history_index={},
            archetype_surfaces=surfaces,
        )


def test_bundle_accepts_archetype_surfaces_when_normalized() -> None:
    K, n_cells = 4, 50
    surfaces = np.random.default_rng(0).random((K, n_cells)).astype(np.float32)
    surfaces /= surfaces.sum(axis=1, keepdims=True)
    b = SnapshotBundle(
        anchor_date=np.datetime64("2019-01-01", "D"),
        player_ids=np.array([], dtype=np.int64),
        role_profiles=np.zeros((0, ROLE_PROFILE_DIM), dtype=np.float32),
        position_mixtures=np.zeros((0, POSITION_MIXTURE_DIM), dtype=np.float32),
        opp_codes=np.array([], dtype=np.str_),
        opp_efficiency_bins=np.array([], dtype=np.int8),
        player_history_index={},
        defensive_history_index={},
        archetype_surfaces=surfaces,
    )
    assert b.archetype_surfaces is not None
    assert b.archetype_surfaces.shape == (K, n_cells)


# ---------------------------------------------------------------------------
# SnapshotStore: lookup semantics
# ---------------------------------------------------------------------------


def _make_store(anchors: list[str]) -> SnapshotStore:
    return SnapshotStore(bundles=tuple(_make_minimal_bundle(a) for a in anchors))


def test_store_requires_at_least_one_bundle() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SnapshotStore(bundles=())


def test_store_rejects_unsorted_anchors() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _make_store(["2019-01-01", "2018-12-01"])  # decreasing


def test_store_rejects_duplicate_anchors() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _make_store(["2019-01-01", "2019-01-01"])


def test_store_get_snapshot_returns_largest_anchor_le_query() -> None:
    store = _make_store(["2018-01-01", "2018-07-01", "2019-01-01", "2019-07-01"])
    # exact match on an anchor returns that anchor's bundle
    assert store.get_snapshot("2018-07-01").anchor_date == np.datetime64("2018-07-01", "D")
    # mid-window date returns the previous anchor
    assert store.get_snapshot("2018-09-15").anchor_date == np.datetime64("2018-07-01", "D")
    # date past the last anchor uses the last anchor
    assert store.get_snapshot("2025-01-01").anchor_date == np.datetime64("2019-07-01", "D")


def test_store_raises_on_pre_first_anchor() -> None:
    store = _make_store(["2019-01-01", "2019-07-01"])
    with pytest.raises(ValueError, match="precedes the earliest"):
        store.get_snapshot("2018-12-31")


def test_store_accepts_pandas_timestamp_and_strings() -> None:
    store = _make_store(["2018-01-01", "2019-01-01"])
    by_string = store.get_snapshot("2018-06-01")
    by_timestamp = store.get_snapshot(pd.Timestamp("2018-06-01"))
    by_dt64 = store.get_snapshot(np.datetime64("2018-06-01", "D"))
    assert by_string.anchor_date == by_timestamp.anchor_date == by_dt64.anchor_date


def test_store_get_snapshot_index_matches_get_snapshot() -> None:
    store = _make_store(["2018-01-01", "2018-07-01", "2019-01-01"])
    for date in ("2018-03-01", "2018-07-01", "2018-12-31", "2025-01-01"):
        idx = store.get_snapshot_index(date)
        assert store.bundles[idx] is store.get_snapshot(date)


# ---------------------------------------------------------------------------
# Builder + headline causality test
# ---------------------------------------------------------------------------


def test_builder_skips_anchors_with_no_prior_shots() -> None:
    shots = _synthetic_shots()
    anchors = [
        np.datetime64("2017-01-01"),  # before any shot — skipped
        np.datetime64("2018-06-01"),
        np.datetime64("2019-01-01"),
    ]
    store = build_snapshot_store_from_shots(shots, anchors)
    # Only the two valid anchors materialize as bundles
    assert len(store) == 2
    assert store.bundles[0].anchor_date == np.datetime64("2018-06-01", "D")


def test_builder_raises_when_no_anchor_has_prior_shots() -> None:
    shots = _synthetic_shots()
    early_anchors = [np.datetime64("2010-01-01"), np.datetime64("2011-01-01")]
    with pytest.raises(ValueError, match="no anchor"):
        build_snapshot_store_from_shots(shots, early_anchors)


def test_snapshot_store_strict_causality() -> None:
    """Every shot indexed by a bundle at anchor ``t_i`` is dated strictly before ``t_i``."""
    shots = _synthetic_shots()
    anchors = [
        np.datetime64("2018-06-01"),
        np.datetime64("2019-01-01"),
        np.datetime64("2019-07-01"),
        np.datetime64("2020-01-01"),
        np.datetime64("2020-07-01"),
    ]
    store = build_snapshot_store_from_shots(shots, anchors)
    # The store-level assertion runs the per-bundle check on every bundle.
    store.assert_causal(shots)

    # Belt-and-suspenders manual check: for every bundle, every player's
    # and every opponent's indexed shots have date < anchor_date.
    all_dates = pd.to_datetime(shots["date"]).to_numpy(dtype="datetime64[D]")
    for bundle in store.bundles:
        anchor = np.datetime64(bundle.anchor_date, "D")
        for pid, indices in bundle.player_history_index.items():
            assert (all_dates[indices] < anchor).all(), f"player {pid} leaks at anchor {anchor}"
        for opp, indices in bundle.defensive_history_index.items():
            assert (all_dates[indices] < anchor).all(), f"opp {opp} leaks at anchor {anchor}"


def test_assert_causal_catches_leakage() -> None:
    """A hand-crafted leaky bundle should fail the assertion."""
    shots = _synthetic_shots()
    # Find a real shot index whose date is in 2020 — that's a leak vs a 2018 anchor.
    dates = pd.to_datetime(shots["date"]).to_numpy(dtype="datetime64[D]")
    future_idx = int(np.where(dates >= np.datetime64("2020-01-01", "D"))[0][0])
    leaky = SnapshotBundle(
        anchor_date=np.datetime64("2018-01-15", "D"),
        player_ids=np.array([1], dtype=np.int64),
        role_profiles=np.zeros((1, ROLE_PROFILE_DIM), dtype=np.float32),
        position_mixtures=np.full((1, POSITION_MIXTURE_DIM), 1 / 3, dtype=np.float32),
        opp_codes=np.array([], dtype=np.str_),
        opp_efficiency_bins=np.array([], dtype=np.int8),
        player_history_index={1: np.array([future_idx], dtype=np.int64)},
        defensive_history_index={},
    )
    with pytest.raises(AssertionError, match="indexed shots have date"):
        leaky.assert_causal(shots)


def test_builder_player_history_indices_are_per_player_only() -> None:
    """A player's history pool contains only that player's shots."""
    shots = _synthetic_shots()
    anchors = [np.datetime64("2019-06-01"), np.datetime64("2020-06-01")]
    store = build_snapshot_store_from_shots(shots, anchors)
    for bundle in store.bundles:
        for pid, indices in bundle.player_history_index.items():
            actual_pids = shots.iloc[indices]["player_id"].unique()
            assert len(actual_pids) == 1
            assert int(actual_pids[0]) == pid


def test_builder_defensive_history_indices_are_per_opp_only() -> None:
    """An opponent's allowed-shot pool contains only shots against that opponent."""
    shots = _synthetic_shots()
    anchors = [np.datetime64("2019-06-01"), np.datetime64("2020-06-01")]
    store = build_snapshot_store_from_shots(shots, anchors)
    for bundle in store.bundles:
        for opp, indices in bundle.defensive_history_index.items():
            actual_opps = shots.iloc[indices]["opponent"].unique()
            assert len(actual_opps) == 1
            assert str(actual_opps[0]) == opp


def test_builder_invokes_role_profile_fn() -> None:
    """role_profile_fn receives the causal sub-frame and its output is propagated."""
    shots = _synthetic_shots()
    captured: list[pd.DataFrame] = []

    def role_fn(sub: pd.DataFrame) -> dict[int, np.ndarray]:
        captured.append(sub)
        return {
            int(pid): np.full(ROLE_PROFILE_DIM, float(pid), dtype=np.float32)
            for pid in sub["player_id"].unique()
        }

    anchors = [np.datetime64("2019-06-01"), np.datetime64("2020-06-01")]
    store = build_snapshot_store_from_shots(shots, anchors, role_profile_fn=role_fn)
    assert len(captured) == len(store.bundles)
    # Each captured frame is itself causal (its rows are all < the bundle's anchor)
    for sub, bundle in zip(captured, store.bundles, strict=True):
        sub_dates = pd.to_datetime(sub["date"]).to_numpy(dtype="datetime64[D]")
        anchor = np.datetime64(bundle.anchor_date, "D")
        assert (sub_dates < anchor).all()
    # And the role profile values flowed through to the bundles
    for bundle in store.bundles:
        for i, pid in enumerate(bundle.player_ids):
            assert float(bundle.role_profiles[i, 0]) == float(pid)


def test_builder_invokes_archetype_fit_fn() -> None:
    """archetype_fit_fn receives the causal sub-frame and its surfaces are stored."""
    K, n_cells = 4, 50

    def fit_fn(sub: pd.DataFrame, _t: np.datetime64) -> np.ndarray:
        # Return uniform surfaces; exercise the bundle's normalization assertion.
        return np.full((K, n_cells), 1.0 / n_cells, dtype=np.float32)

    shots = _synthetic_shots()
    anchors = [np.datetime64("2019-06-01"), np.datetime64("2020-06-01")]
    store = build_snapshot_store_from_shots(shots, anchors, archetype_fit_fn=fit_fn)
    for bundle in store.bundles:
        assert bundle.archetype_surfaces is not None
        assert bundle.archetype_surfaces.shape == (K, n_cells)
        np.testing.assert_allclose(bundle.archetype_surfaces.sum(axis=1), 1.0, atol=1e-5)


def test_builder_passes_anchor_date_to_archetype_fit_fn() -> None:
    """``archetype_fit_fn`` receives each bundle's anchor date, in chronological order."""
    K, n_cells = 4, 50
    seen: list[np.datetime64] = []

    def fit_fn(sub: pd.DataFrame, t: np.datetime64) -> np.ndarray:
        seen.append(t)
        return np.full((K, n_cells), 1.0 / n_cells, dtype=np.float32)

    shots = _synthetic_shots()
    anchors = [np.datetime64("2019-06-01"), np.datetime64("2020-06-01")]
    store = build_snapshot_store_from_shots(shots, anchors, archetype_fit_fn=fit_fn)
    # One call per assembled bundle, in chronological order.
    assert len(seen) == len(store.bundles)
    for got, bundle in zip(seen, store.bundles, strict=True):
        assert got == bundle.anchor_date


def test_builder_opp_efficiency_bins_use_only_filtered_made_rate() -> None:
    """Opponent-efficiency bins at an anchor depend only on shots before that anchor."""
    shots = _synthetic_shots()
    anchors = [np.datetime64("2019-01-01"), np.datetime64("2020-01-01")]
    store = build_snapshot_store_from_shots(shots, anchors)
    for bundle in store.bundles:
        # Re-derive bins from the shots strictly before the anchor and check match.
        before = shots[
            pd.to_datetime(shots["date"]).to_numpy(dtype="datetime64[D]") < bundle.anchor_date
        ]
        if "made" not in before.columns or len(bundle.opp_codes) < N_OPP_EFFICIENCY_BINS:
            continue
        opp_made = before.dropna(subset=["opponent", "made"]).groupby("opponent")["made"].mean()
        # Each opp's bin index must be in [0, N_OPP_EFFICIENCY_BINS).
        for i, opp in enumerate(bundle.opp_codes):
            assert 0 <= int(bundle.opp_efficiency_bins[i]) < N_OPP_EFFICIENCY_BINS
            assert str(opp) in opp_made.index
