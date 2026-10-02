"""Tests for the causal usage-state feature helper.

Scope:

* The three slot names match the load-bearing trait-table slots
  (log1p_minutes_M, log1p_fga_S, log_shot_density); ``USAGE_DIM == 3``.
* ``USAGE_SLOT_INDICES`` resolves to (14, 15, 16) under the current
  trait layout. A future re-order of :data:`SLOT_NAMES` is caught
  here.
* :func:`extract_usage` shape / dtype, correct row selection, and
  input-shape validation.
* Causal-by-construction sanity: only the three named slots are
  selected; no other slots leak through.
"""

from __future__ import annotations

import pytest
import torch

from shotcloud.data.player_traits import SLOT_NAMES, TRAIT_DIM
from shotcloud.features.usage_features import (
    USAGE_DIM,
    USAGE_FEATURE_NAMES,
    USAGE_SLOT_INDICES,
    extract_usage,
)


def test_usage_feature_names_are_exactly_three_known_slots() -> None:
    assert USAGE_DIM == 3
    assert USAGE_FEATURE_NAMES == (
        "log1p_minutes_M",
        "log1p_fga_S",
        "log_shot_density",
    )


def test_usage_slot_indices_match_trait_layout() -> None:
    """Indices computed from SLOT_NAMES at import-time. A re-order of
    the trait table is caught here."""
    assert (
        SLOT_NAMES.index("log1p_minutes_M"),
        SLOT_NAMES.index("log1p_fga_S"),
        SLOT_NAMES.index("log_shot_density"),
    ) == USAGE_SLOT_INDICES
    # Under the current 26-slot layout, these resolve to (14, 15, 16).
    assert USAGE_SLOT_INDICES == (14, 15, 16)
    # And every slot index lies in [0, TRAIT_DIM).
    assert all(0 <= i < TRAIT_DIM for i in USAGE_SLOT_INDICES)


def test_extract_usage_shape_and_dtype() -> None:
    traits = torch.randn(5, 3, TRAIT_DIM)
    pid = torch.tensor([0, 2, 4], dtype=torch.long)
    sid = torch.tensor([0, 1, 2], dtype=torch.long)
    out = extract_usage(traits, pid, sid)
    assert out.shape == (3, USAGE_DIM)
    assert out.dtype == traits.dtype


def test_extract_usage_selects_correct_slots() -> None:
    """The output should equal the trait tensor's columns at the three
    canonical slot indices, sliced for the requested rows."""
    traits = torch.randn(4, 2, TRAIT_DIM)
    pid = torch.tensor([1, 3])
    sid = torch.tensor([0, 1])
    out = extract_usage(traits, pid, sid)
    for j, slot in enumerate(USAGE_SLOT_INDICES):
        torch.testing.assert_close(out[:, j], traits[pid, sid, slot])


def test_extract_usage_rejects_bad_shapes() -> None:
    traits = torch.randn(5, 3, TRAIT_DIM)
    with pytest.raises(ValueError, match=r"traits must be"):
        extract_usage(torch.randn(5, TRAIT_DIM), torch.tensor([0]), torch.tensor([0]))
    with pytest.raises(ValueError, match=r"player_idx must be"):
        extract_usage(traits, torch.tensor([[0]]), torch.tensor([0]))
    with pytest.raises(ValueError, match=r"snapshot_idx must match"):
        extract_usage(traits, torch.tensor([0, 1]), torch.tensor([0]))


def test_usage_slots_are_disjoint_from_each_other() -> None:
    """No two usage features map to the same slot."""
    assert len(set(USAGE_SLOT_INDICES)) == USAGE_DIM
