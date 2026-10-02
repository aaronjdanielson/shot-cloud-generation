"""Tests for :mod:`shotcloud.data.prior_shot_kde`, the causal within-game prior-shot KDE feature."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from shotcloud.data.prior_shot_kde import (
    DEFAULT_SIGMA_H_FT,
    PRIOR_SHOT_KDE_DIM,
    ZONE_CENTROIDS_FT,
    compute_prior_shot_kde_features,
)


def _make_shots(
    rows: list[tuple[str, str, float, float, float]],
) -> pd.DataFrame:
    """Build a shot frame from ``(player_id, game_id, x, y, t_elapsed_sec)`` tuples.

    Elapsed seconds go in the ``time_remaining_sec`` column, which is how the
    featurizer orders shots within a game.
    """
    return pd.DataFrame(
        rows,
        columns=["player_id", "game_id", "x", "y", "time_remaining_sec"],
    )


def test_dim_constant_is_eight() -> None:
    assert PRIOR_SHOT_KDE_DIM == 8


def test_centroids_are_eight_2d_finite() -> None:
    assert ZONE_CENTROIDS_FT.shape == (8, 2)
    assert np.isfinite(ZONE_CENTROIDS_FT).all()


def test_empty_input_returns_zero_rows() -> None:
    df = _make_shots([])
    out = compute_prior_shot_kde_features(df)
    assert out.shape == (0, PRIOR_SHOT_KDE_DIM)
    assert out.dtype == np.float32


def test_first_shot_is_all_zero() -> None:
    df = _make_shots([("p1", "g1", 0.0, 1.5, 60.0)])
    out = compute_prior_shot_kde_features(df)
    assert out.shape == (1, PRIOR_SHOT_KDE_DIM)
    assert (out == 0.0).all()


def test_second_shot_matches_analytic_kernel_at_centroids() -> None:
    """A single prior shot at the rim should produce phi(c) = N(c; rim, σ^2 I)
    for every centroid c."""
    df = _make_shots(
        [
            ("p1", "g1", 0.0, 1.5, 60.0),  # at RA centroid
            ("p1", "g1", 22.5, 5.0, 120.0),  # at R corner centroid
        ]
    )
    out = compute_prior_shot_kde_features(df, sigma_h_ft=4.0)
    # Row 0 first shot: zero.
    np.testing.assert_array_equal(out[0], np.zeros(8))
    # Row 1 second shot: phi(c) = N(c; (0, 1.5), 4^2 I)
    sigma = 4.0
    norm = math.log(2.0 * math.pi * sigma * sigma)
    centroids = ZONE_CENTROIDS_FT
    diff = centroids - np.array([0.0, 1.5], dtype=np.float32)
    sq = (diff * diff).sum(axis=-1)
    expected = np.exp(-norm - 0.5 * sq / (sigma * sigma)).astype(np.float32)
    np.testing.assert_allclose(out[1], expected, atol=1e-6, rtol=1e-5)


def test_causal_running_mean() -> None:
    """Three shots all at the rim should yield phi[0]=0, phi[1]=N(c; rim),
    phi[2]=N(c; rim) (the running mean of two identical kernels)."""
    df = _make_shots(
        [
            ("p1", "g1", 0.0, 1.5, 60.0),
            ("p1", "g1", 0.0, 1.5, 120.0),
            ("p1", "g1", 0.0, 1.5, 180.0),
        ]
    )
    out = compute_prior_shot_kde_features(df, sigma_h_ft=4.0)
    np.testing.assert_array_equal(out[0], np.zeros(8))
    np.testing.assert_allclose(out[1], out[2], atol=1e-6)
    # The RA slot should be the largest (the kernel evaluated at the
    # shot location is the peak of the Gaussian).
    assert out[1].argmax() == 0


def test_two_players_independent_groups() -> None:
    """Shots from one player do NOT leak into another player's causal KDE,
    even when interleaved by time-elapsed."""
    df = _make_shots(
        [
            ("p1", "g1", 0.0, 1.5, 60.0),
            ("p2", "g1", 22.5, 5.0, 90.0),
            ("p1", "g1", 22.5, 5.0, 120.0),
            ("p2", "g1", 0.0, 1.5, 150.0),
        ]
    )
    out = compute_prior_shot_kde_features(df, sigma_h_ft=4.0)
    # p1's first shot (row 0) and p2's first shot (row 1) are both zero.
    np.testing.assert_array_equal(out[0], np.zeros(8))
    np.testing.assert_array_equal(out[1], np.zeros(8))
    # p1's second shot (row 2): N(c; (0, 1.5), σ^2 I) — only its OWN prior.
    sigma = 4.0
    norm = math.log(2.0 * math.pi * sigma * sigma)
    centroids = ZONE_CENTROIDS_FT
    expected_p1 = np.exp(
        -norm - 0.5 * ((centroids - np.array([0.0, 1.5])) ** 2).sum(-1) / (sigma * sigma)
    )
    np.testing.assert_allclose(out[2], expected_p1, atol=1e-6, rtol=1e-5)
    # p2's second shot (row 3): N(c; (22.5, 5.0), σ^2 I) — its own prior.
    expected_p2 = np.exp(
        -norm - 0.5 * ((centroids - np.array([22.5, 5.0])) ** 2).sum(-1) / (sigma * sigma)
    )
    np.testing.assert_allclose(out[3], expected_p2, atol=1e-6, rtol=1e-5)


def test_two_games_independent() -> None:
    """Same player across two different games — the second game's first
    shot should be zero (causal boundary at game boundary)."""
    df = _make_shots(
        [
            ("p1", "g1", 0.0, 1.5, 60.0),
            ("p1", "g1", 22.5, 5.0, 120.0),
            ("p1", "g2", 0.0, 25.0, 60.0),  # different game's first shot
        ]
    )
    out = compute_prior_shot_kde_features(df, sigma_h_ft=4.0)
    np.testing.assert_array_equal(out[0], np.zeros(8))
    np.testing.assert_array_equal(out[2], np.zeros(8))


def test_default_sigma_is_4ft() -> None:
    assert pytest.approx(4.0) == DEFAULT_SIGMA_H_FT


def test_rejects_bad_sigma() -> None:
    df = _make_shots([("p1", "g1", 0.0, 0.0, 60.0)])
    with pytest.raises(ValueError, match=r"sigma_h_ft must be positive"):
        compute_prior_shot_kde_features(df, sigma_h_ft=0.0)


def test_missing_columns_raise() -> None:
    df = pd.DataFrame({"player_id": ["p1"]})
    with pytest.raises(KeyError, match=r"needs column"):
        compute_prior_shot_kde_features(df)


def test_aligned_to_input_row_order() -> None:
    """Output rows align to the input DataFrame's row order, even when the
    internal causal sort scrambles them."""
    # Two shots out of chronological order in input (later one first).
    df = _make_shots(
        [
            ("p1", "g1", 22.5, 5.0, 200.0),  # input row 0 = chronologically second
            ("p1", "g1", 0.0, 1.5, 100.0),  # input row 1 = chronologically first
        ]
    )
    out = compute_prior_shot_kde_features(df, sigma_h_ft=4.0)
    # Chronologically first shot (input row 1) is the first shot → zero.
    np.testing.assert_array_equal(out[1], np.zeros(8))
    # Chronologically second shot (input row 0) has prior = chronological 1st.
    assert (out[0] > 0).any()
