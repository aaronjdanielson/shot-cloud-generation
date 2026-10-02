"""Tests for ``shotcloud.evaluation.support_masks``."""

from __future__ import annotations

import pytest
import torch

from shotcloud.evaluation.support_masks import support_source_masks


def test_partition_basic() -> None:
    """own and pooled partition the valid support; the target player's
    pool slots are own, the rest pooled."""
    # B=2, L=3, R=2 → M=6.
    analogue_idx = torch.tensor([[5, 0, 9], [0, 1, 2]])
    player_idx = torch.tensor([0, 7])  # row 0 self is slot 1; row 1 no self
    out = support_source_masks(analogue_idx, player_idx, n_shots_per_analogue=2)
    # Row 0: pool player 0 at L-slot 1 == target 0 → M-slots 2,3 are own.
    assert out.own[0].tolist() == [False, False, True, True, False, False]
    assert out.pooled[0].tolist() == [True, True, False, False, True, True]
    # Row 1: no L-slot equals target 7 → all pooled.
    assert out.own[1].tolist() == [False] * 6
    assert out.pooled[1].tolist() == [True] * 6
    # own and pooled partition valid.
    for r in range(2):
        assert (out.own[r] | out.pooled[r]).tolist() == out.valid[r].tolist()
        assert not (out.own[r] & out.pooled[r]).any()


def test_valid_mask_excludes_invalid_slots() -> None:
    """Invalid support slots are in neither own nor pooled."""
    analogue_idx = torch.tensor([[3, 0]])  # L=2
    player_idx = torch.tensor([0])  # self at L-slot 1
    valid = torch.tensor([[True, False, True, False]])  # R=2 → M=4
    out = support_source_masks(analogue_idx, player_idx, 2, valid_mask=valid)
    # M-slots 2,3 are the self player's; only slot 2 is valid.
    assert out.own[0].tolist() == [False, False, True, False]
    # M-slots 0,1 pooled; only slot 0 valid.
    assert out.pooled[0].tolist() == [True, False, False, False]
    # Invalid slots 1,3 in neither.
    assert not (out.own[0, 1] or out.pooled[0, 1])
    assert not (out.own[0, 3] or out.pooled[0, 3])


def test_multiple_self_slots_all_count_as_own() -> None:
    """If the target player appears in more than one pool slot, all
    of them are own."""
    analogue_idx = torch.tensor([[0, 0, 4]])  # target 0 in two L-slots
    player_idx = torch.tensor([0])
    out = support_source_masks(analogue_idx, player_idx, n_shots_per_analogue=1)
    assert out.own[0].tolist() == [True, True, False]
    assert out.pooled[0].tolist() == [False, False, True]


def test_no_valid_mask_treats_all_slots_valid() -> None:
    analogue_idx = torch.tensor([[1, 2]])
    player_idx = torch.tensor([9])
    out = support_source_masks(analogue_idx, player_idx, n_shots_per_analogue=3)
    assert out.valid[0].all()
    assert out.pooled[0].all()
    assert not out.own[0].any()


def test_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="analogue_idx must be"):
        support_source_masks(torch.zeros(3), torch.zeros(3), 2)
    with pytest.raises(ValueError, match="player_idx must be"):
        support_source_masks(torch.zeros(2, 3), torch.zeros(5), 2)
    with pytest.raises(ValueError, match="n_shots_per_analogue must be positive"):
        support_source_masks(torch.zeros(2, 3), torch.zeros(2), 0)
    with pytest.raises(ValueError, match="valid_mask must be"):
        support_source_masks(
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            2,
            valid_mask=torch.ones(2, 5, dtype=torch.bool),
        )
