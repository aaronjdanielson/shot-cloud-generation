"""Tests for the G1 within-game shot GRU (paper §10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud.data.within_game_history import (
    MAX_PRIOR_SHOTS,
    WITHIN_GAME_SEQ_DIM,
    compute_within_game_sequence,
)
from shotcloud.models.within_game_gru import WithinGameGRU


def _make_shots(rows: list[tuple]) -> pd.DataFrame:
    """Build a minimal shots DataFrame for the sequence featurizer."""
    return pd.DataFrame(
        rows,
        columns=("player_id", "game_id", "time_remaining_sec", "x", "y"),
    )


def test_compute_within_game_sequence_empty_input() -> None:
    df = _make_shots([])
    df = df.assign(time_remaining_sec=pd.Series([], dtype=np.int64))
    seq, lengths = compute_within_game_sequence(df)
    assert seq.shape == (0, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)
    assert lengths.shape == (0,)


def test_compute_within_game_sequence_first_shot_is_zero_length() -> None:
    df = _make_shots(
        [
            ("p1", "g1", 60, 0.0, 5.0),  # first shot of (p1, g1)
            ("p1", "g1", 120, 5.0, 10.0),  # second shot — has 1 prior
            ("p1", "g1", 180, -3.0, 8.0),  # third shot — has 2 priors
        ]
    )
    seq, lengths = compute_within_game_sequence(df)
    assert seq.shape == (3, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)
    assert lengths.tolist() == [0, 1, 2]
    # First-shot row is all-zero in the prior sequence.
    assert np.all(seq[0] == 0.0)
    # Second row's first prior matches the first shot's normalized features.
    # x=0/25=0, y=5/47≈0.106. The exact values depend on _X_NORM/_Y_NORM.
    assert seq[1, 0, 0] == pytest.approx(0.0)
    assert seq[1, 0, 1] == pytest.approx(5.0 / 47.0, rel=1e-4)


def test_compute_within_game_sequence_truncates_to_max_prior() -> None:
    n = MAX_PRIOR_SHOTS + 5
    rows = [("p1", "g1", 60 * i, float(i), float(i)) for i in range(n)]
    df = _make_shots(rows)
    seq, lengths = compute_within_game_sequence(df)
    # The shot at index n-1 has n-1 priors, which exceeds MAX_PRIOR_SHOTS;
    # length is capped and only the MOST RECENT MAX_PRIOR_SHOTS shots
    # appear in the sequence.
    assert lengths[n - 1] == MAX_PRIOR_SHOTS
    # The 0th column of the truncated sequence corresponds to the
    # oldest *retained* prior, which is the shot at index
    # n-1-MAX_PRIOR_SHOTS = 5-1 = 4. Its x is 4.0/_X_NORM.
    assert seq[n - 1, 0, 0] == pytest.approx(4.0 / 25.0, rel=1e-4)


def test_within_game_gru_zero_init_returns_zero_for_any_input() -> None:
    """The output projection is zero-init, so the GRU contributes 0 at
    step 0 regardless of inputs. Load-bearing invariant: G1 ≡ B2 at init."""
    gru = WithinGameGRU(hidden_dim=16, out_dim=8)
    torch.manual_seed(0)
    prior_seq = torch.randn(4, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)
    prior_lengths = torch.tensor([5, 0, 10, 3], dtype=torch.int64)
    out = gru(prior_seq, prior_lengths)
    assert out.shape == (4, 8)
    assert torch.all(out == 0.0)


def test_within_game_gru_zero_length_rows_bypass_the_gru() -> None:
    """Rows with prior_lengths==0 must not enter pack_padded_sequence
    (which rejects zero lengths). They produce the zero output directly."""
    gru = WithinGameGRU(hidden_dim=8, out_dim=4)
    # Manually set the projection weight nonzero so we'd detect anything
    # leaking through the GRU path. Zero-length rows must still emit 0.
    with torch.no_grad():
        gru.proj.weight.normal_(std=0.1)
    prior_seq = torch.randn(3, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)
    prior_lengths = torch.zeros(3, dtype=torch.int64)
    out = gru(prior_seq, prior_lengths)
    assert torch.all(out == 0.0)


def test_within_game_gru_nonzero_after_unzeroing_projection() -> None:
    """Confirm the GRU pipeline runs end-to-end when the projection is
    no longer zero (so we know the zero output above is genuine and not
    a misimplementation that silently drops everything)."""
    gru = WithinGameGRU(hidden_dim=8, out_dim=4)
    with torch.no_grad():
        gru.proj.weight.normal_(std=1.0)  # bigger than zero-init
    prior_seq = torch.randn(3, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)
    prior_lengths = torch.tensor([3, 1, 7], dtype=torch.int64)
    out = gru(prior_seq, prior_lengths)
    assert out.shape == (3, 4)
    # The unprojected GRU state for nonzero rows must be nonzero.
    assert torch.any(out != 0.0)


def test_within_game_gru_rejects_bad_shapes() -> None:
    gru = WithinGameGRU(hidden_dim=8, out_dim=4)
    bad_seq = torch.zeros(2, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM + 1)
    with pytest.raises(ValueError, match=r"prior_seq must have shape"):
        gru(bad_seq, torch.zeros(2, dtype=torch.int64))
    seq = torch.zeros(2, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)
    with pytest.raises(ValueError, match=r"prior_lengths must have shape"):
        gru(seq, torch.zeros(3, dtype=torch.int64))


def test_within_game_gru_constructor_rejects_nonpositive_dims() -> None:
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        WithinGameGRU(hidden_dim=0, out_dim=4)
    with pytest.raises(ValueError, match="out_dim must be positive"):
        WithinGameGRU(hidden_dim=8, out_dim=-1)


def test_within_game_gru_proj_weight_is_zero_at_init() -> None:
    """The load-bearing init: ``proj.weight == 0`` so G1 ≡ B2 at init."""
    gru = WithinGameGRU(hidden_dim=16, out_dim=8)
    assert torch.all(gru.proj.weight == 0.0)
    assert gru.proj.bias is None  # bias=False in the projection
