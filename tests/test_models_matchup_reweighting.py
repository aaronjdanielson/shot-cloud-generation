"""Tests for :class:`~shotcloud.models.zone_defense_reweighting.MatchupReweightingDefense`.

Covers the forward pass only (zone gather, ``β_match · Δ̂``, out-of-court masking,
gradient flow); construction of the matchup features ``Δ̂`` is tested in
``tests/test_features_matchup_features.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from shotcloud.data.zones import N_ZONES
from shotcloud.models.zone_defense_reweighting import (
    MatchupReweightingDefense,
    zone_from_xy_torch,
)


def _zone_corners() -> tuple[torch.Tensor, torch.Tensor]:
    """A small ``(B=8, M=1, 2)`` ``query_xy`` where row ``i`` is in zone
    ``i``, plus the matching expected zone-index tensor."""
    # One canonical (x, y) per zone, matching the coordinates used in the matchup
    # feature tests with ``zone_from_xy_vectorized``.
    coords = torch.tensor(
        [
            [0.0, 2.0],  # RA
            [0.0, 8.0],  # Paint
            [10.0, 18.0],  # Midrange
            [-23.0, 5.0],  # Corner3-L
            [23.0, 5.0],  # Corner3-R
            [-15.0, 20.0],  # Wing3-L
            [15.0, 20.0],  # Wing3-R
            [0.0, 25.0],  # TopKey3
        ],
        dtype=torch.float32,
    )
    xy = coords.unsqueeze(1)  # (8, 1, 2)
    expected_zone = torch.arange(N_ZONES, dtype=torch.int64).unsqueeze(-1)  # (8, 1)
    # Guards the fixture against a change in the zone ordering of zone_from_xy_torch.
    assert torch.equal(zone_from_xy_torch(xy), expected_zone)
    return xy, expected_zone


def test_init_beta_match_is_parameter() -> None:
    m = MatchupReweightingDefense(beta_init=0.5)
    assert isinstance(m.beta_match, torch.nn.Parameter)
    assert m.beta_match.requires_grad
    torch.testing.assert_close(m.beta_match, torch.tensor(0.5))


def test_forward_shape_and_dtype() -> None:
    m = MatchupReweightingDefense(beta_init=1.0)
    B, M = 4, 6
    xy = torch.randn(B, M, 2)
    delta = torch.randn(B, N_ZONES)
    out = m(query_xy=xy, delta_hat=delta)
    assert out.shape == (B, M)
    assert out.dtype == torch.float32


def test_beta_zero_collapses_to_exact_zero() -> None:
    """At β_match = 0, the module is a no-op regardless of Δ̂."""
    m = MatchupReweightingDefense(beta_init=0.0)
    B, M = 5, 4
    xy = torch.randn(B, M, 2)
    delta = torch.randn(B, N_ZONES) * 10.0
    out = m(query_xy=xy, delta_hat=delta)
    assert torch.equal(out, torch.zeros_like(out))


def test_score_equals_beta_times_delta_hat_at_zone() -> None:
    """A shot in zone ``z`` contributes ``β_match · Δ̂_{row, z}``."""
    xy, _expected_zone = _zone_corners()  # (8, 1, 2), (8, 1)
    B = xy.shape[0]
    # Diagonal Δ̂: row i has 1.0 at zone i, 0 elsewhere. So the gather
    # at (row=i, shot=0) → 1.0 exactly.
    delta = torch.eye(N_ZONES)  # (8, 8) == (B, N_ZONES)
    m = MatchupReweightingDefense(beta_init=2.5)
    out = m(query_xy=xy, delta_hat=delta)  # (8, 1)
    expected = torch.full((B, 1), 2.5)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)


def test_cold_start_delta_hat_yields_zero() -> None:
    """A row with ``Δ̂ = 0`` (cold-start cell) contributes zero at any ``β_match``."""
    m = MatchupReweightingDefense(beta_init=5.0)
    B, M = 3, 4
    xy = torch.tensor(
        [
            [[0.0, 2.0]] * M,
            [[0.0, 25.0]] * M,
            [[10.0, 18.0]] * M,
        ],
        dtype=torch.float32,
    )
    delta = torch.zeros(B, N_ZONES)
    out = m(query_xy=xy, delta_hat=delta)
    assert torch.equal(out, torch.zeros_like(out))


def test_out_of_court_xy_is_zeroed() -> None:
    """Out-of-court support shots (zone ``-1``) contribute zero even when ``Δ̂`` is
    large."""
    m = MatchupReweightingDefense(beta_init=1.0)
    # All three shots in row 0 are out-of-court (backcourt, behind
    # baseline, far sideline).
    xy = torch.tensor(
        [[[0.0, 60.0], [0.0, -10.0], [40.0, 5.0]]],
        dtype=torch.float32,
    )
    delta = torch.full((1, N_ZONES), 10.0)
    out = m(query_xy=xy, delta_hat=delta)
    assert torch.equal(out, torch.zeros_like(out))


def test_gradient_flows_to_beta_match() -> None:
    m = MatchupReweightingDefense(beta_init=1e-3)
    xy = torch.tensor([[[0.0, 25.0]]], dtype=torch.float32)  # TopKey3
    delta = torch.zeros(1, N_ZONES)
    delta[0, 7] = 1.0  # TopKey3 column
    out = m(query_xy=xy, delta_hat=delta)
    out.sum().backward()
    assert m.beta_match.grad is not None
    # ∂(β · Δ̂_7) / ∂β = Δ̂_7 = 1.0
    torch.testing.assert_close(m.beta_match.grad, torch.tensor(1.0))


def test_state_dict_round_trip() -> None:
    src = MatchupReweightingDefense(beta_init=0.7)
    with torch.no_grad():
        src.beta_match.fill_(1.234)
    sd = src.state_dict()
    dst = MatchupReweightingDefense(beta_init=0.0)
    dst.load_state_dict(sd)
    torch.testing.assert_close(dst.beta_match, src.beta_match)


def test_input_shape_validation() -> None:
    m = MatchupReweightingDefense()
    delta = torch.zeros(3, N_ZONES)

    import pytest

    with pytest.raises(ValueError, match=r"\(B, M, 2\)"):
        m(query_xy=torch.zeros(3, 4), delta_hat=delta)

    xy = torch.zeros(3, 4, 2)
    with pytest.raises(ValueError, match=f"N_ZONES={N_ZONES}"):
        m(query_xy=xy, delta_hat=torch.zeros(3, N_ZONES - 1))

    with pytest.raises(ValueError, match="batch dim mismatch"):
        m(query_xy=torch.zeros(3, 4, 2), delta_hat=torch.zeros(2, N_ZONES))


def test_per_shot_sigma_grids_dont_leak_across_rows() -> None:
    """Each row's output depends only on its own ``Δ̂`` row."""
    m = MatchupReweightingDefense(beta_init=1.0)
    # Two rows, both in RA (zone 0); Δ̂_0 differs.
    xy = torch.tensor([[[0.0, 2.0]], [[0.0, 2.0]]], dtype=torch.float32)  # (2, 1, 2)
    delta = torch.zeros(2, N_ZONES)
    delta[0, 0] = 0.3
    delta[1, 0] = -0.5
    out = m(query_xy=xy, delta_hat=delta)
    torch.testing.assert_close(out, torch.tensor([[0.3], [-0.5]]), atol=1e-6, rtol=0)


def test_only_beta_match_is_a_parameter() -> None:
    """``β_match`` is the module's only parameter, and it has no buffers."""
    m = MatchupReweightingDefense()
    params = list(m.parameters())
    buffers = list(m.buffers())
    assert len(params) == 1, [p.shape for p in params]
    assert params[0] is m.beta_match
    assert len(buffers) == 0


def test_state_dict_size_matches_d_lite_zero() -> None:
    """The state dict holds a single scalar ``beta_match``, the same footprint as
    ``ZoneReweightingDefense`` without per-zone ``γ``."""
    m = MatchupReweightingDefense()
    sd = m.state_dict()
    assert tuple(sd.keys()) == ("beta_match",)
    assert sd["beta_match"].numel() == 1


def _sanity_match_zone_helper_consistent() -> None:
    """Keep the ``numpy`` import in use."""
    np.testing.assert_array_equal(np.array([1, 2]), np.array([1, 2]))
