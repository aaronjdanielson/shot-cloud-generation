"""Tests for :class:`shotcloud.data.context.ContextEncoder`, the encoder for the raw context x_n.

Every feature must derive from the row itself, the snapshot bundle in force
at the row's date, or a fixed pregame normalization. The tests cover the
``FEATURE_LAYOUT`` schema, snapshot lookups, deterministic fallbacks for
unknown players and opponents, and rejection of dates before the first
snapshot anchor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shotcloud.data import (
    POSITION_MIXTURE_DIM,
    ROLE_PROFILE_DIM,
    SnapshotStore,
    build_role_profiles,
    build_snapshot_store_from_shots,
)
from shotcloud.data.context import (
    CONTEXT_DIM,
    FEATURE_LAYOUT,
    N_PERIOD_ONEHOT,
    ContextEncoder,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_shots(seed: int = 0) -> pd.DataFrame:
    """Build weekly synthetic shots from 2018 through 2020 with every column the encoder reads."""
    rng = np.random.default_rng(seed)
    rows = []
    for date in pd.date_range("2018-01-01", "2020-12-31", freq="W"):
        for _ in range(3):
            rows.append(
                {
                    "x": float(rng.normal(0.0, 5.0)),
                    "y": float(rng.normal(15.0, 5.0)),
                    "player_id": int(rng.choice([1, 2, 3])),
                    "opponent": str(rng.choice(["BOS", "LAL", "GSW"])),
                    "made": int(rng.random() < 0.45),
                    "period": int(rng.choice([1, 2, 3, 4])),
                    "time_remaining_sec": int(rng.uniform(0, 12 * 60 * 4)),
                    "starter": int(rng.random() < 0.7),
                    "minutes": int(rng.uniform(8, 40)),
                    "date": pd.Timestamp(date),
                }
            )
    return pd.DataFrame(rows)


def _build_store(shots: pd.DataFrame) -> SnapshotStore:
    anchors = [
        np.datetime64("2018-06-01"),
        np.datetime64("2019-01-01"),
        np.datetime64("2019-07-01"),
        np.datetime64("2020-01-01"),
        np.datetime64("2020-07-01"),
    ]
    return build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=build_role_profiles,
    )


def _encoder(shots: pd.DataFrame) -> ContextEncoder:
    return ContextEncoder.fit(_build_store(shots), shots)


def _shots_after_first_anchor(shots: pd.DataFrame, store: SnapshotStore) -> pd.DataFrame:
    """Filter to shots whose date is on or after the first snapshot anchor."""
    first = store.bundles[0].anchor_date
    return shots[pd.to_datetime(shots["date"]).to_numpy(dtype="datetime64[D]") >= first].copy()


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


def test_context_dim_matches_feature_layout() -> None:
    """The slices in FEATURE_LAYOUT must be contiguous and cover [0, CONTEXT_DIM)."""
    keys = (
        "period_onehot",
        "time_in_period",
        "season_recency",
        "starter",
        "minutes_norm",
        "home_away",
        "recent_3pa_frac",
        "recent_usage",
        "recent_fga",
        "role_profile",
        "position_mixture",
        "opp_efficiency_onehot",
    )
    expected_widths = (
        N_PERIOD_ONEHOT,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        ROLE_PROFILE_DIM,
        POSITION_MIXTURE_DIM,
        4,
    )
    cursor = 0
    for key, width in zip(keys, expected_widths, strict=True):
        sl = FEATURE_LAYOUT[key]
        assert sl.start == cursor, f"{key}: expected start {cursor}, got {sl.start}"
        assert sl.stop == cursor + width, f"{key}: expected stop {cursor + width}, got {sl.stop}"
        cursor += width
    assert cursor == CONTEXT_DIM


def test_context_dim_is_27() -> None:
    assert CONTEXT_DIM == 27


def test_home_away_slice_populated_from_column() -> None:
    """A ``home_away`` column fills its slice; without the column the slice is zero."""
    shots = _synthetic_shots()
    enc = _encoder(shots)

    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(8).copy()
    sub["home_away"] = [1, 1, 1, 1, 0, 0, 0, 0]
    out = enc.transform(sub)
    home_idx = FEATURE_LAYOUT["home_away"].start
    np.testing.assert_array_equal(
        out[:, home_idx], np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    )

    # Without the column, the slice stays zero.
    sub2 = _shots_after_first_anchor(shots, enc.snapshot_store).head(8).copy()
    out2 = enc.transform(sub2)
    np.testing.assert_array_equal(out2[:, home_idx], np.zeros(8, dtype=np.float32))


def test_recent_features_slices_populated() -> None:
    """``recent_*`` columns fill their slices.

    ``recent_3pa_frac`` is clipped to [0, 1]; usage and FGA are z-scored with
    the encoder's stored statistics.
    """
    shots = _synthetic_shots()
    enc = ContextEncoder.fit(
        _build_store(shots),
        shots,
        recent_usage_mean=0.5,
        recent_usage_std=0.2,
        recent_fga_mean=8.0,
        recent_fga_std=4.0,
    )
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(4).copy()
    sub["recent_3pa_frac"] = [0.0, 0.4, 1.0, 1.5]  # last clips to 1.0
    sub["recent_usage"] = [0.5, 0.7, 0.3, 0.5]  # z = (x - 0.5) / 0.2
    sub["recent_fga"] = [8.0, 12.0, 4.0, 8.0]  # z = (x - 8) / 4

    out = enc.transform(sub)
    np.testing.assert_allclose(
        out[:, FEATURE_LAYOUT["recent_3pa_frac"].start],
        [0.0, 0.4, 1.0, 1.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out[:, FEATURE_LAYOUT["recent_usage"].start],
        [0.0, 1.0, -1.0, 0.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out[:, FEATURE_LAYOUT["recent_fga"].start],
        [0.0, 1.0, -1.0, 0.0],
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_fit_derives_normalization_stats_from_shots() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    assert enc.date_min <= np.datetime64("2018-01-08", "D")
    assert enc.date_max >= np.datetime64("2020-12-25", "D")
    # Minutes uniformly drawn from [8, 40] → mean ~24.
    assert 15.0 < enc.minutes_mean < 35.0
    assert enc.minutes_std > 1.0


def test_fit_overrides_normalization_stats_when_provided() -> None:
    shots = _synthetic_shots()
    store = _build_store(shots)
    enc = ContextEncoder.fit(
        store,
        shots,
        date_min=np.datetime64("2017-01-01"),
        date_max=np.datetime64("2025-01-01"),
        minutes_mean=20.0,
        minutes_std=8.0,
    )
    assert enc.date_min == np.datetime64("2017-01-01", "D")
    assert enc.date_max == np.datetime64("2025-01-01", "D")
    assert enc.minutes_mean == 20.0
    assert enc.minutes_std == 8.0


def test_fit_works_without_shots() -> None:
    shots = _synthetic_shots()
    store = _build_store(shots)
    enc = ContextEncoder.fit(store)
    assert enc.date_min == store.anchor_dates[0]
    assert enc.date_max == store.anchor_dates[-1]


# ---------------------------------------------------------------------------
# Transform: shape and basic invariants
# ---------------------------------------------------------------------------


def test_transform_returns_correct_shape() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(50)
    out = enc.transform(sub)
    assert out.shape == (len(sub), CONTEXT_DIM)
    assert out.dtype == np.float32


def test_transform_empty_input() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    empty = pd.DataFrame(columns=shots.columns)
    out = enc.transform(empty)
    assert out.shape == (0, CONTEXT_DIM)


def test_transform_period_onehot_is_correct() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(20)
    out = enc.transform(sub)
    period_slice = FEATURE_LAYOUT["period_onehot"]
    period_block = out[:, period_slice]
    np.testing.assert_array_equal(period_block.sum(axis=1), np.ones(len(sub), dtype=np.float32))
    for i, period in enumerate(sub["period"].to_numpy()):
        assert period_block[i, int(period) - 1] == 1.0


def test_transform_starter_routes_correctly() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(30)
    out = enc.transform(sub)
    starter_idx = FEATURE_LAYOUT["starter"].start
    np.testing.assert_array_equal(out[:, starter_idx], sub["starter"].astype(np.float32).to_numpy())


def test_transform_minutes_z_score() -> None:
    shots = _synthetic_shots()
    enc = ContextEncoder.fit(_build_store(shots), shots, minutes_mean=24.0, minutes_std=8.0)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(30)
    out = enc.transform(sub)
    minutes_idx = FEATURE_LAYOUT["minutes_norm"].start
    expected = (sub["minutes"].to_numpy() - 24.0) / 8.0
    np.testing.assert_allclose(out[:, minutes_idx], expected.astype(np.float32), atol=1e-6)


# ---------------------------------------------------------------------------
# Snapshot-derived features
# ---------------------------------------------------------------------------


def test_role_profile_pulled_from_correct_snapshot() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(50)
    out = enc.transform(sub)
    rp_slice = FEATURE_LAYOUT["role_profile"]
    for i, (date, pid) in enumerate(
        zip(sub["date"].to_numpy(), sub["player_id"].to_numpy(), strict=True)
    ):
        bundle = enc.snapshot_store.get_snapshot(date)
        rp = bundle.role_profile(int(pid))
        if rp is not None:
            np.testing.assert_allclose(out[i, rp_slice], rp, atol=1e-6)


def test_position_mixture_always_sums_to_one() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(50)
    out = enc.transform(sub)
    pm_slice = FEATURE_LAYOUT["position_mixture"]
    for i in range(len(sub)):
        np.testing.assert_allclose(out[i, pm_slice].sum(), 1.0, atol=1e-4)


def test_opp_strength_one_hot_pulled_from_correct_snapshot() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(50)
    out = enc.transform(sub)
    opp_slice = FEATURE_LAYOUT["opp_efficiency_onehot"]
    for i, (date, opp) in enumerate(
        zip(sub["date"].to_numpy(), sub["opponent"].to_numpy(), strict=True)
    ):
        bundle = enc.snapshot_store.get_snapshot(date)
        bin_idx = bundle.opp_strength_bin(str(opp))
        if bin_idx is None:
            assert out[i, opp_slice].sum() == 0.0
        else:
            assert int(out[i, opp_slice].argmax()) == bin_idx
            assert out[i, opp_slice].sum() == 1.0


# ---------------------------------------------------------------------------
# Missing-value policy
# ---------------------------------------------------------------------------


def test_unknown_player_falls_back_to_zero_role_uniform_position() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(5).copy()
    sub["player_id"] = 99999  # never appears in training shots
    out = enc.transform(sub)
    rp_slice = FEATURE_LAYOUT["role_profile"]
    pm_slice = FEATURE_LAYOUT["position_mixture"]
    np.testing.assert_array_equal(out[:, rp_slice], np.zeros((len(sub), ROLE_PROFILE_DIM)))
    np.testing.assert_allclose(
        out[:, pm_slice],
        np.full((len(sub), POSITION_MIXTURE_DIM), 1.0 / POSITION_MIXTURE_DIM),
        atol=1e-6,
    )


def test_unknown_opponent_falls_back_to_zero_one_hot() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(5).copy()
    sub["opponent"] = "ZZZ"
    out = enc.transform(sub)
    opp_slice = FEATURE_LAYOUT["opp_efficiency_onehot"]
    np.testing.assert_array_equal(out[:, opp_slice], np.zeros((len(sub), 4)))


def test_missing_starter_column_defaults_to_zero() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(5).drop(columns=["starter"])
    out = enc.transform(sub)
    assert (out[:, FEATURE_LAYOUT["starter"].start] == 0.0).all()


def test_missing_minutes_column_defaults_to_zero_z_score() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(5).drop(columns=["minutes"])
    out = enc.transform(sub)
    assert (out[:, FEATURE_LAYOUT["minutes_norm"].start] == 0.0).all()


# ---------------------------------------------------------------------------
# Causality enforcement
# ---------------------------------------------------------------------------


def test_transform_raises_for_pre_first_anchor_dates() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    leaky = shots.head(2).copy()
    leaky["date"] = pd.Timestamp("2017-12-01")  # before first anchor
    with pytest.raises(ValueError, match=r"precedes the earliest snapshot anchor|date before"):
        enc.transform(leaky)


def test_no_feature_uses_future_rows() -> None:
    """Encoding a row alone matches its encoding within a batch, so rows never see neighbors."""
    shots = _synthetic_shots()
    enc = _encoder(shots)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(20)
    full_out = enc.transform(sub)
    for i in range(len(sub)):
        single = sub.iloc[i : i + 1]
        single_out = enc.transform(single)
        np.testing.assert_allclose(
            single_out[0],
            full_out[i],
            atol=1e-6,
            err_msg=f"row {i} transform depends on neighbors",
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_transform_is_deterministic() -> None:
    shots = _synthetic_shots()
    enc = _encoder(shots)
    sub = _shots_after_first_anchor(shots, enc.snapshot_store).head(30)
    out1 = enc.transform(sub)
    out2 = enc.transform(sub)
    np.testing.assert_array_equal(out1, out2)
