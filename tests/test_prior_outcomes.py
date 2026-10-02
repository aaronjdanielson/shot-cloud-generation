"""Tests for :func:`shotcloud.data.prior_outcomes.compute_prior_outcome_features`
and the outcome branch of :class:`shotcloud.models.ContextResidualEncoder`.

Phase 2 of the 2026-06-07 audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud.data.context import CONTEXT_DIM
from shotcloud.data.prior_outcomes import (
    PRIOR_OUTCOME_DIM,
    PRIOR_OUTCOME_FEATURE_NAMES,
    compute_prior_outcome_features,
)
from shotcloud.models.context_residual import ContextResidualEncoder


def _hand_checkable_df() -> pd.DataFrame:
    """Six shots in chronological order for one player-game.

    Shot 0: rim, made
    Shot 1: 3PA wing, missed
    Shot 2: 3PA wing, made
    Shot 3: rim, made
    Shot 4: mid, missed
    Shot 5: 3PA wing, made
    """
    return pd.DataFrame(
        {
            "x": [0.0, 22.0, 22.0, 0.0, 12.0, 22.0],
            "y": [4.0, 25.0, 25.0, 4.0, 15.0, 25.0],
            "made": [1, 0, 1, 1, 0, 1],
            "player_id": [101] * 6,
            "game_id": ["g1"] * 6,
            "time_remaining_sec": [60, 120, 180, 240, 300, 360],
        }
    )


def test_dim_constant_matches_feature_names() -> None:
    assert PRIOR_OUTCOME_DIM == len(PRIOR_OUTCOME_FEATURE_NAMES) == 9


def test_first_shot_zeros() -> None:
    df = _hand_checkable_df()
    out = compute_prior_outcome_features(df)
    assert out.shape == (6, PRIOR_OUTCOME_DIM)
    np.testing.assert_array_equal(out[0], np.zeros(PRIOR_OUTCOME_DIM, dtype=np.float32))


def test_counts_use_log1p_and_are_causal() -> None:
    df = _hand_checkable_df()
    out = compute_prior_outcome_features(df)
    # Shot 3: after 3 priors (rim+made, 3PA+miss, 3PA+made).
    # FGA=3, makes=2, misses=1, 3PA=2, 3PM=1, rim_att=1, rim_makes=1.
    name_to_idx = {n: i for i, n in enumerate(PRIOR_OUTCOME_FEATURE_NAMES)}
    expected = {
        "prior_fga_log1p": float(np.log1p(3)),
        "prior_makes_log1p": float(np.log1p(2)),
        "prior_misses_log1p": float(np.log1p(1)),
        "prior_3pa_log1p": float(np.log1p(2)),
        "prior_3pm_log1p": float(np.log1p(1)),
        "prior_rim_attempts_log1p": float(np.log1p(1)),
        "prior_rim_makes_log1p": float(np.log1p(1)),
    }
    for k, v in expected.items():
        assert out[3, name_to_idx[k]] == pytest.approx(v, abs=1e-5)


def test_recent_make_rate_window() -> None:
    df = _hand_checkable_df()
    out = compute_prior_outcome_features(df)
    rate_idx = PRIOR_OUTCOME_FEATURE_NAMES.index("recent_make_rate")
    # Shot 4: last 4 priors (rim+made, 3PA+miss, 3PA+made, rim+made) → 3/4.
    assert out[4, rate_idx] == pytest.approx(0.75, abs=1e-5)
    # Shot 5: last 5 priors (rim+made, 3PA+miss, 3PA+made, rim+made, mid+miss) → 3/5.
    assert out[5, rate_idx] == pytest.approx(0.60, abs=1e-5)


def test_recent_dist_mean_normalized() -> None:
    """Sanity-check the distance-mean slot stays in a reasonable range."""
    df = _hand_checkable_df()
    out = compute_prior_outcome_features(df)
    dist_idx = PRIOR_OUTCOME_FEATURE_NAMES.index("recent_dist_mean")
    # Shot 1 has one prior at (0, 4) → distance 4 ft, /35 ≈ 0.114.
    assert out[1, dist_idx] == pytest.approx(4.0 / 35.0, abs=1e-5)


def test_validation_errors() -> None:
    # Missing column.
    df = _hand_checkable_df().drop(columns=["made"])
    with pytest.raises(KeyError, match="made"):
        compute_prior_outcome_features(df)
    # Bad `made` values.
    df_bad = _hand_checkable_df()
    df_bad.loc[0, "made"] = 2
    with pytest.raises(ValueError, match=r"made"):
        compute_prior_outcome_features(df_bad)


def test_empty_df_returns_empty_array() -> None:
    df = pd.DataFrame(
        {col: [] for col in ("x", "y", "made", "player_id", "game_id", "time_remaining_sec")}
    )
    out = compute_prior_outcome_features(df)
    assert out.shape == (0, PRIOR_OUTCOME_DIM)


def test_multiple_player_games_independent() -> None:
    """Cross-group counts must NOT leak between (player, game) groups."""
    df = pd.DataFrame(
        {
            "x": [0.0, 0.0, 22.0, 22.0],
            "y": [4.0, 4.0, 25.0, 25.0],
            "made": [1, 0, 1, 0],
            "player_id": [101, 101, 102, 102],
            "game_id": ["g1", "g1", "g2", "g2"],
            "time_remaining_sec": [60, 120, 60, 120],
        }
    )
    out = compute_prior_outcome_features(df)
    # Both group's first shot must be all-zero (no prior shots in OWN group).
    np.testing.assert_array_equal(out[0], np.zeros(PRIOR_OUTCOME_DIM, dtype=np.float32))
    np.testing.assert_array_equal(out[2], np.zeros(PRIOR_OUTCOME_DIM, dtype=np.float32))
    # Group 2's second shot's FGA log1p = log1p(1) — not 2 (no leak from group 1).
    fga_idx = PRIOR_OUTCOME_FEATURE_NAMES.index("prior_fga_log1p")
    assert out[3, fga_idx] == pytest.approx(float(np.log1p(1)), abs=1e-5)


def test_outcome_branch_zero_init_invariance() -> None:
    """At init, the outcome branch contributes zero to the residual
    regardless of the outcome input. Mirrors the existing usage-branch
    contract — preserves AC-KDE's zero-init invariant.
    """
    torch.manual_seed(0)
    enc = ContextResidualEncoder(
        rank=8, context_dim=CONTEXT_DIM, within_game_dim=10, usage_dim=3, outcome_dim=9
    )
    x_n = torch.randn(4, CONTEXT_DIM)
    h_n = torch.randn(4, 10)
    usage = torch.randn(4, 3)
    outcome_nonzero = torch.randn(4, 9)
    outcome_zero = torch.zeros(4, 9)
    u_nonzero = enc(x_n, h_n=h_n, usage=usage, outcome=outcome_nonzero)
    u_zero = enc(x_n, h_n=h_n, usage=usage, outcome=outcome_zero)
    assert torch.allclose(u_nonzero, u_zero)


def test_outcome_branch_required_when_dim_positive() -> None:
    enc = ContextResidualEncoder(rank=8, within_game_dim=10, outcome_dim=9)
    x_n = torch.randn(2, CONTEXT_DIM)
    h_n = torch.randn(2, 10)
    with pytest.raises(ValueError, match="outcome is required"):
        enc(x_n, h_n=h_n, outcome=None)


def test_outcome_rejected_when_dim_zero() -> None:
    enc = ContextResidualEncoder(rank=8, within_game_dim=10, outcome_dim=0)
    x_n = torch.randn(2, CONTEXT_DIM)
    h_n = torch.randn(2, 10)
    bogus_outcome = torch.zeros(2, 9)
    with pytest.raises(ValueError, match="outcome must be None"):
        enc(x_n, h_n=h_n, outcome=bogus_outcome)
