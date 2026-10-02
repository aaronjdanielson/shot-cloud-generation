"""Tests for the Tier-1a per-(source, zone) bandwidth module.

Scope: unit behavior of :class:`ZoneSourceBandwidth` (init = σ_init
everywhere, bounded outputs, source/zone indexing semantics, grad
flow, state-dict round-trip, numpy/torch zone parity). Wrapper-level
equivalence (zone_source at init ≡ fixed σ=1.5) is covered separately
in :mod:`tests.test_models_continuous_mixture_spatial` so the wrapper
test fixtures stay co-located with the rest of the wrapper checks.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shotcloud.data.zones import N_ZONES, zone_from_xy_vectorized
from shotcloud.models.zone_source_bandwidth import (
    ZoneSourceBandwidth,
    zone_from_xy_torch_bandwidth,
)


def test_zone_from_xy_torch_matches_numpy() -> None:
    """``zone_from_xy_torch_bandwidth`` must agree with the canonical
    numpy ``zone_from_xy_vectorized`` on a sweep of court coordinates."""
    xs = np.linspace(-26.0, 26.0, 41)
    ys = np.linspace(-6.0, 48.0, 41)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    expected = zone_from_xy_vectorized(xx.ravel(), yy.ravel()).reshape(xx.shape)
    xy = torch.from_numpy(np.stack([xx, yy], axis=-1).astype(np.float32))
    got = zone_from_xy_torch_bandwidth(xy).numpy()
    np.testing.assert_array_equal(got, expected)


def test_init_sigma_is_uniform_at_sigma_init() -> None:
    """At ``sigma_init = 1.5`` every (source, zone) entry initializes
    to σ = 1.5 exactly — the load-bearing invariant that makes the
    wrapper bit-equivalent to the fixed-σ path at step 0."""
    bw = ZoneSourceBandwidth(sigma_min=1.0, sigma_max=2.5, sigma_init=1.5)
    table = bw.sigma_table()
    assert table.shape == (2, N_ZONES)
    torch.testing.assert_close(table, torch.full((2, N_ZONES), 1.5), atol=1e-6, rtol=1e-6)


def test_forward_shape_and_per_shot_sigma() -> None:
    """``forward`` returns ``(B, M)`` with the correct source/zone
    bandwidth per support shot."""
    bw = ZoneSourceBandwidth(sigma_min=1.0, sigma_max=2.5, sigma_init=1.5)
    # Tweak the raw logits to give distinguishable σ per (source, zone).
    with torch.no_grad():
        # Make own/rim very sharp, pooled/topkey3 very smooth, leave
        # other entries at the default init.
        bw.raw[ZoneSourceBandwidth.OWN, 0] = -5.0  # σ ≈ σ_min for own/rim
        bw.raw[ZoneSourceBandwidth.POOLED, 7] = 5.0  # σ ≈ σ_max for pooled/ATB3
    # Two rows. Row 0: rim shot (zone 0), own; row 1: top-key-3 (zone 7), pooled.
    xy = torch.tensor([[[0.0, 2.0]], [[0.0, 24.0]]])
    own = torch.tensor([[True], [False]])
    out = bw(support_xy=xy, own_mask=own)
    assert out.shape == (2, 1)
    # own/rim collapsed near σ_min (1.0); pooled/ATB3 near σ_max (2.5).
    assert out[0, 0] < 1.05
    assert out[1, 0] > 2.45


def test_forward_in_bounds() -> None:
    """All outputs lie within [σ_min, σ_max] regardless of raw logits."""
    bw = ZoneSourceBandwidth(sigma_min=1.0, sigma_max=2.5, sigma_init=1.5)
    with torch.no_grad():
        bw.raw.fill_(100.0)  # extreme — sigmoid → 1 → σ → σ_max
    xy = torch.randn(3, 5, 2)
    own = torch.randint(0, 2, (3, 5), dtype=torch.bool)
    out = bw(support_xy=xy, own_mask=own)
    assert torch.all(out <= 2.5 + 1e-6)
    assert torch.all(out >= 1.0 - 1e-6)


def test_out_of_court_shots_clamp_to_zone_zero_not_crash() -> None:
    """Out-of-court support shots (zone == -1) must not crash the
    indexed gather. The σ value for such shots is unspecified but
    must be finite and positive (callers mask these shots out of the
    mixture downstream)."""
    bw = ZoneSourceBandwidth()
    # Backcourt + behind-baseline + far sideline = all zone -1.
    xy = torch.tensor([[[0.0, 60.0], [0.0, -10.0], [40.0, 5.0]]])
    own = torch.tensor([[True, False, True]])
    out = bw(support_xy=xy, own_mask=own)
    assert torch.isfinite(out).all()
    assert (out > 0).all()


def test_grad_flows_to_raw() -> None:
    """``sum(σ_m)`` produces nonzero gradient on the raw logits for
    every (source, zone) that any support shot maps to."""
    bw = ZoneSourceBandwidth()
    xy = torch.tensor([[[0.0, 2.0], [-23.5, 5.0], [0.0, 24.0]]])  # rim, LC3, ATB3
    own = torch.tensor([[True, False, True]])
    out = bw(support_xy=xy, own_mask=own)
    out.sum().backward()
    assert bw.raw.grad is not None
    # Hit cells: (OWN, 0), (POOLED, 3), (OWN, 7) — all should be nonzero.
    assert bw.raw.grad[ZoneSourceBandwidth.OWN, 0].abs() > 0
    assert bw.raw.grad[ZoneSourceBandwidth.POOLED, 3].abs() > 0
    assert bw.raw.grad[ZoneSourceBandwidth.OWN, 7].abs() > 0
    # Untouched cells must have exactly-zero grad (no leakage).
    assert bw.raw.grad[ZoneSourceBandwidth.OWN, 1] == 0


def test_state_dict_round_trip() -> None:
    """State-dict save/load preserves the learned σ table exactly."""
    src = ZoneSourceBandwidth(sigma_min=1.0, sigma_max=2.5, sigma_init=1.5)
    with torch.no_grad():
        src.raw.copy_(torch.randn_like(src.raw))
    sd = src.state_dict()
    dst = ZoneSourceBandwidth(sigma_min=1.0, sigma_max=2.5, sigma_init=1.5)
    dst.load_state_dict(sd)
    torch.testing.assert_close(dst.sigma_table(), src.sigma_table())


def test_bad_bounds_rejected() -> None:
    """Constructor rejects σ_min ≥ σ_max and sigma_init out of bounds."""
    with pytest.raises(ValueError, match="must be <"):
        ZoneSourceBandwidth(sigma_min=2.0, sigma_max=1.0)
    with pytest.raises(ValueError, match="required"):
        ZoneSourceBandwidth(sigma_min=1.0, sigma_max=2.0, sigma_init=2.5)
    with pytest.raises(ValueError, match="required"):
        ZoneSourceBandwidth(sigma_min=1.0, sigma_max=2.0, sigma_init=0.5)


def test_forward_shape_validation() -> None:
    bw = ZoneSourceBandwidth()
    with pytest.raises(ValueError, match=r"\(B, M, 2\)"):
        bw(support_xy=torch.zeros(4, 5), own_mask=torch.zeros(4, 5, dtype=torch.bool))
    with pytest.raises(ValueError, match="own_mask"):
        bw(
            support_xy=torch.zeros(4, 5, 2),
            own_mask=torch.zeros(4, 6, dtype=torch.bool),
        )
