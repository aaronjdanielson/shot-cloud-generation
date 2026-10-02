"""Tests for ``shotcloud.features.defense_features`` (PR-D0.5).

The nine acceptance criteria from the build approval (2026-05-25):

1. Causal: feature at snapshot ``t`` uses only rows with
   ``date < t``.
2. Correct opponent: features for ``d`` use only shots with
   ``opponent == d``.
3. Zone rates sum to 1 when count > 0.
4. Empty / cold-start opponent gets zeros plus reliability
   indicators showing no data.
5. Centered zone rates have league-weighted mean approximately 0
   at each snapshot.
6. Recency weights behave as expected on controlled synthetic
   dates.
7. Feature names length matches tensor dimension.
8. Deterministic build.
9. Disk round-trip preserves tensor + config exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud.features.defense_features import (
    DEFENSE_FEATURE_DIM,
    DEFENSE_FEATURE_NAMES,
    DefenseFeatures,
    DefenseFeaturesConfig,
    build_defense_features,
)
from shotcloud.training.dataset import OpponentVocab


def _epoch_day(s: str) -> int:
    return int(np.datetime64(s, "D").astype(np.int64))


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _build_fixture(
    *,
    seed: int = 0,
) -> tuple[pd.DataFrame, OpponentVocab, np.ndarray]:
    """4 teams (A/B/C/D), 1 snapshot, controlled allowed-shot
    histories.

    * Team A: 10 allowed shots before the anchor, half at the rim
      (x=0, y=2) and half at corner-3 (x=-22, y=1). Mixes 2pt + 3pt.
    * Team B: 5 allowed shots, all in midrange (x=10, y=15).
    * Team C: 3 allowed shots, all at the rim — sparse-history opp.
    * Team D: zero causal allowed shots — cold-start opp.
    * One non-causal shot per defending team on the anchor day
      (excluded by the strict ``< anchor`` cutoff).
    """
    rng = np.random.default_rng(seed)
    anchor = _epoch_day("2024-04-15")
    rows: list[dict[str, object]] = []
    # Team A: half rim, half corner-3.
    for i in range(10):
        is_three = i >= 5
        x = -22.0 if is_three else 0.0
        y = 1.0 if is_three else 2.0
        rows.append(
            {
                "team": "Z",
                "opponent": "A",
                "date": np.datetime64(anchor - 10 + i, "D"),
                "x": x,
                "y": y,
            }
        )
    # Team B: all midrange.
    for i in range(5):
        rows.append(
            {
                "team": "Z",
                "opponent": "B",
                "date": np.datetime64(anchor - 5 + i, "D"),
                "x": 10.0,
                "y": 15.0,
            }
        )
    # Team C: 3 rim shots.
    for i in range(3):
        rows.append(
            {
                "team": "Z",
                "opponent": "C",
                "date": np.datetime64(anchor - 3 + i, "D"),
                "x": 0.0,
                "y": 2.0,
            }
        )
    # Non-causal shots on the anchor day for A, B, C.
    for opp in ["A", "B", "C", "D"]:
        rows.append(
            {
                "team": "Z",
                "opponent": opp,
                "date": np.datetime64(anchor, "D"),
                "x": float(rng.normal(0, 1)),
                "y": float(rng.normal(0, 1)),
            }
        )
    shots = pd.DataFrame(rows)
    vocab = OpponentVocab.from_ids(["A", "B", "C", "D"])
    anchor_dates = np.array([anchor], dtype=np.int64)
    return shots, vocab, anchor_dates


def _default_config(anchor_dates: np.ndarray, **overrides: object) -> DefenseFeaturesConfig:
    kwargs: dict[str, object] = {
        "shots_fingerprint": "fixture_v0",
        "anchor_dates": tuple(int(d) for d in anchor_dates),
        "window_days": 30,
        "half_life_days": 30.0,
    }
    kwargs.update(overrides)
    return DefenseFeaturesConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------


def test_1_causal_features_use_only_pre_anchor_rows() -> None:
    """Add a future-dated row with extreme coordinates and confirm it
    doesn't perturb the feature vector — proving the causal cutoff is
    strict (``date < anchor``)."""
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f_base = build_defense_features(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    # Inject a future shot AGAINST team A at an extreme location.
    future_day = int(anchors[0]) + 30
    future = pd.DataFrame(
        [
            {
                "team": "Z",
                "opponent": "A",
                "date": np.datetime64(future_day, "D"),
                "x": 100.0,
                "y": 100.0,
            }
        ]
    )
    shots_future = pd.concat([shots, future], ignore_index=True)
    f_inject = build_defense_features(
        shots_df=shots_future, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    # The injected future shot must not change the feature tensor.
    torch.testing.assert_close(f_base.features, f_inject.features)


def test_1b_features_on_anchor_date_are_excluded() -> None:
    """The fixture's non-causal shots dated exactly on the anchor must
    be excluded — the team A non-causal shot has random coordinates
    that would noisily perturb features if included. Verify by
    removing them and confirming the feature tensor is unchanged."""
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f_with_nc = build_defense_features(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    pre_anchor = pd.to_datetime(shots["date"]).to_numpy(dtype="datetime64[D]").astype(
        np.int64
    ) < int(anchors[0])
    shots_clean = shots.iloc[pre_anchor].reset_index(drop=True)
    f_clean = build_defense_features(
        shots_df=shots_clean, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    torch.testing.assert_close(f_with_nc.features, f_clean.features)


def test_2_features_use_only_shots_against_target_opponent() -> None:
    """Build features for team A in isolation (only A's allowed
    shots) and confirm the row matches the full-fixture row. If the
    aggregator leaked other teams' shots into A's features, this
    would fail."""
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f_full = build_defense_features(
        shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg
    )
    a_only = shots[shots["opponent"] == "A"].reset_index(drop=True)
    # Build with the A-only fixture; centered values will differ
    # (league baseline shrinks), but the *raw* zone rates, team
    # scalars, and reliability features should match A's row.
    f_a = build_defense_features(shots_df=a_only, opp_vocab=vocab, anchor_dates=anchors, config=cfg)
    d_idx_a = vocab.to_idx("A")
    # Block A (team scalars) and Block B (raw zones) and Block D
    # (reliability) depend only on A's shots and should agree.
    for slot_name in (
        "allowed_mean_shot_distance",
        "allowed_std_shot_distance",
        "allowed_3pa_rate",
        "allowed_2pa_rate",
        "q_rim_raw",
        "q_LC3_raw",
        "log1p_allowed_count",
        "effective_sample_size",
    ):
        i = DEFENSE_FEATURE_NAMES.index(slot_name)
        assert f_full.features[d_idx_a, 0, i].item() == pytest.approx(
            f_a.features[d_idx_a, 0, i].item(), rel=1e-5, abs=1e-7
        ), f"feature {slot_name!r} differs between full-fixture and A-only"


def test_3_zone_rates_sum_to_one_when_count_positive() -> None:
    """For every (opp, snapshot) cell with a non-zero allowed-shot
    count, the 8 raw zone proportions sum to ≈ 1."""
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f = build_defense_features(shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg)
    raw_idx = [
        DEFENSE_FEATURE_NAMES.index(f"q_{z}_raw")
        for z in ("rim", "paint", "mid", "LC3", "RC3", "LW3", "RW3", "ATB3")
    ]
    count_idx = DEFENSE_FEATURE_NAMES.index("log1p_allowed_count")
    for d_idx in range(len(vocab)):
        has_data = f.features[d_idx, 0, count_idx].item() > 0
        if not has_data:
            continue
        zone_sum = float(f.features[d_idx, 0, raw_idx].sum().item())
        assert zone_sum == pytest.approx(1.0, abs=1e-5), (
            f"opp {vocab.ids[d_idx]} zone rates sum = {zone_sum} ≠ 1"
        )


def test_4_cold_start_opp_is_all_zeros_with_zero_reliability() -> None:
    """Team D has no causal allowed shots → its whole feature row is
    zero; the reliability block confirms no-data via
    ``log1p_allowed_count = 0`` and ``effective_sample_size = 0``."""
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f = build_defense_features(shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg)
    d_idx_cold = vocab.to_idx("D")
    row = f.features[d_idx_cold, 0]
    assert (row == 0).all().item(), f"cold-start row not all-zero: {row.tolist()}"
    log1p_count_i = DEFENSE_FEATURE_NAMES.index("log1p_allowed_count")
    ess_i = DEFENSE_FEATURE_NAMES.index("effective_sample_size")
    assert row[log1p_count_i].item() == 0.0
    assert row[ess_i].item() == 0.0


def test_5_centered_zone_rates_have_league_weighted_mean_zero() -> None:
    """The league-weighted mean of each centered-zone column is ≈ 0
    at each snapshot (the centering construction's defining
    property)."""
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f = build_defense_features(shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg)
    centered_idx = [
        DEFENSE_FEATURE_NAMES.index(f"q_{z}_centered")
        for z in ("rim", "paint", "mid", "LC3", "RC3", "LW3", "RW3", "ATB3")
    ]
    ess_i = DEFENSE_FEATURE_NAMES.index("effective_sample_size")
    # League weights are per-opp ESS (Σ w_j across that opp's shots).
    # Σ_d ESS_d · q_d^z_centered = 0 over the non-cold-start opps.
    weights = f.features[:, 0, ess_i].numpy()
    for col in centered_idx:
        centered = f.features[:, 0, col].numpy()
        weighted_mean = float((weights * centered).sum() / max(weights.sum(), 1e-12))
        assert weighted_mean == pytest.approx(0.0, abs=1e-5), (
            f"centered {DEFENSE_FEATURE_NAMES[col]!r} weighted mean = "
            f"{weighted_mean:.2e} (expected ≈ 0)"
        )


def test_6_recency_weights_decay_with_age() -> None:
    """A more-recent allowed shot must contribute more to the ESS
    than an older one. Verified on a controlled fixture where opp X
    has two shots: one 1 day before the anchor, one 90 days before.
    With ``half_life_days=30``:

        w_1   = exp(-1 · ln(2)/30)  ≈ 0.977
        w_90  = exp(-90 · ln(2)/30) ≈ 0.125

    so ESS_X ≈ 1.10 and the older shot should contribute
    ≈ 0.125 / 1.10 ≈ 11.4% of the weighted mean — far less than the
    50% it would contribute under uniform weighting.
    """
    anchor = _epoch_day("2024-04-15")
    shots = pd.DataFrame(
        [
            {
                "team": "Z",
                "opponent": "X",
                "date": np.datetime64(anchor - 1, "D"),
                "x": 0.0,
                "y": 2.0,  # rim
            },
            {
                "team": "Z",
                "opponent": "X",
                "date": np.datetime64(anchor - 90, "D"),
                "x": 0.0,
                "y": 30.0,  # mid-ish — distance 30 ft, weighted heavily
            },
        ]
    )
    vocab = OpponentVocab.from_ids(["X"])
    anchors = np.array([anchor], dtype=np.int64)
    cfg = _default_config(anchors, window_days=365, half_life_days=30.0)
    f = build_defense_features(shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg)
    ess_i = DEFENSE_FEATURE_NAMES.index("effective_sample_size")
    ess = float(f.features[0, 0, ess_i].item())
    expected = float(np.exp(-1 * np.log(2) / 30) + np.exp(-90 * np.log(2) / 30))
    assert ess == pytest.approx(expected, rel=1e-4), f"ESS = {ess} != expected {expected}"
    mean_dist_i = DEFENSE_FEATURE_NAMES.index("allowed_mean_shot_distance")
    weighted_mean = float(f.features[0, 0, mean_dist_i].item())
    # Recency-weighted mean must be much closer to the recent shot's
    # distance (2.0) than to the old shot's distance (30.0).
    uniform_mean = (2.0 + 30.0) / 2  # = 16.0
    assert weighted_mean < uniform_mean, (
        f"recency-weighted mean {weighted_mean} >= uniform {uniform_mean} "
        "— recency weighting isn't reducing the older shot's contribution"
    )
    # Closer to the recent shot than to the old shot.
    assert abs(weighted_mean - 2.0) < abs(weighted_mean - 30.0)


def test_7_feature_names_length_matches_tensor_dim() -> None:
    """The canonical name tuple length must equal the tensor's last
    dim — fence against feature-layout drift."""
    assert len(DEFENSE_FEATURE_NAMES) == DEFENSE_FEATURE_DIM
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f = build_defense_features(shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg)
    assert f.features.shape[-1] == DEFENSE_FEATURE_DIM
    assert len(f.feature_names) == DEFENSE_FEATURE_DIM
    assert tuple(f.feature_names) == DEFENSE_FEATURE_NAMES


def test_8_build_is_deterministic() -> None:
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f1 = build_defense_features(shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg)
    f2 = build_defense_features(shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg)
    torch.testing.assert_close(f1.features, f2.features)
    assert f1.config == f2.config


def test_9_disk_round_trip_preserves_tensor_and_config(tmp_path: Path) -> None:
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f = build_defense_features(shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg)
    save_path = tmp_path / "defense_features.pt"
    f.save(save_path)
    loaded = DefenseFeatures.load(save_path)
    torch.testing.assert_close(loaded.features, f.features)
    assert loaded.config == f.config
    assert loaded.feature_names == f.feature_names


# ---------------------------------------------------------------------------
# Smaller helpers + invariants
# ---------------------------------------------------------------------------


def test_config_validates_positive_fields() -> None:
    anchors = (_epoch_day("2024-04-15"),)
    with pytest.raises(ValueError, match="window_days"):
        DefenseFeaturesConfig(shots_fingerprint="x", anchor_dates=anchors, window_days=0)
    with pytest.raises(ValueError, match="half_life_days"):
        DefenseFeaturesConfig(shots_fingerprint="x", anchor_dates=anchors, half_life_days=0.0)


def test_config_hash_changes_on_any_field() -> None:
    """Each config-defining field, when changed, must produce a
    different ``config_hash``."""
    anchors = np.array([_epoch_day("2024-04-15")], dtype=np.int64)
    base = _default_config(anchors)
    base_h = base.config_hash
    variants: dict[str, DefenseFeaturesConfig] = {
        "shots_fingerprint": _default_config(anchors, shots_fingerprint="other"),
        "anchor_dates": _default_config(np.array([_epoch_day("2024-04-16")], dtype=np.int64)),
        "window_days": _default_config(anchors, window_days=60),
        "half_life_days": _default_config(anchors, half_life_days=60.0),
        "seed": _default_config(anchors, seed=1),
    }
    for field, variant in variants.items():
        assert variant.config_hash != base_h, f"{field} change did not alter config_hash"
    assert _default_config(anchors).config_hash == base_h


def test_build_with_cache_dir_loads_on_second_call(tmp_path: Path) -> None:
    """First call writes the feature file; second call loads it
    without re-running the build (verified by passing an empty
    DataFrame on the second call — if rebuild ran it would crash on
    the missing 'opponent' column)."""
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f1 = build_defense_features(
        shots_df=shots,
        opp_vocab=vocab,
        anchor_dates=anchors,
        config=cfg,
        cache_dir=tmp_path,
    )
    feature_file = tmp_path / f"defense_features_{cfg.config_hash}.pt"
    assert feature_file.exists()
    f2 = build_defense_features(
        shots_df=pd.DataFrame(),  # empty; would crash if rebuild ran
        opp_vocab=vocab,
        anchor_dates=anchors,
        config=cfg,
        cache_dir=tmp_path,
    )
    torch.testing.assert_close(f1.features, f2.features)


def test_atomic_save_leaves_no_tmp_file(tmp_path: Path) -> None:
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f = build_defense_features(shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg)
    save_path = tmp_path / "defense_features.pt"
    f.save(save_path)
    assert save_path.exists()
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == [], f"leftover .tmp files: {leftover}"


def test_zone_assignment_on_synthetic_fixture_is_correct() -> None:
    """Sanity: in the fixture, team B's shots are all at (10, 15) →
    midrange. Verify q_mid_raw = 1.0 for team B."""
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f = build_defense_features(shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg)
    d_idx_b = vocab.to_idx("B")
    q_mid_i = DEFENSE_FEATURE_NAMES.index("q_mid_raw")
    assert f.features[d_idx_b, 0, q_mid_i].item() == pytest.approx(1.0, abs=1e-5)


def test_team_3pa_plus_2pa_rate_sum_to_one_on_non_cold_start() -> None:
    """The 3PA + 2PA rates partition the allowed shots → sum is 1.0
    for every non-cold-start opp."""
    shots, vocab, anchors = _build_fixture()
    cfg = _default_config(anchors)
    f = build_defense_features(shots_df=shots, opp_vocab=vocab, anchor_dates=anchors, config=cfg)
    three_i = DEFENSE_FEATURE_NAMES.index("allowed_3pa_rate")
    two_i = DEFENSE_FEATURE_NAMES.index("allowed_2pa_rate")
    count_i = DEFENSE_FEATURE_NAMES.index("log1p_allowed_count")
    for d_idx in range(len(vocab)):
        if f.features[d_idx, 0, count_i].item() == 0:
            continue
        s = float(f.features[d_idx, 0, three_i].item() + f.features[d_idx, 0, two_i].item())
        assert s == pytest.approx(1.0, abs=1e-5)
