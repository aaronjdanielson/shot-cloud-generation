"""Tests for the D-lite zone-reweighting defense module (PR-D-lite-0).

Scope: unit behavior of :class:`ZoneReweightingDefense` and the
torch-native :func:`zone_from_xy_torch`. Equivalence with the wrapper
``ContinuousMixtureSpatial`` is covered separately in
``tests/test_models_continuous_mixture_spatial.py`` (β_D=0 collapses
to the no-defense likelihood).
"""

from __future__ import annotations

import numpy as np
import torch

from shotcloud.data.zones import N_ZONES, zone_from_xy_vectorized
from shotcloud.features.defense_features import (
    _ZONE_CENTERED_SLICE,
    DEFENSE_FEATURE_DIM,
)
from shotcloud.models.zone_defense_reweighting import (
    ZoneReweightingDefense,
    zone_from_xy_torch,
)


def _random_features(B: int, *, seed: int = 0) -> torch.Tensor:
    """Build a (B, DEFENSE_FEATURE_DIM) tensor with nonzero centered-zone
    block so D-lite is exercised."""
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((B, DEFENSE_FEATURE_DIM)).astype(np.float32) * 0.1
    return torch.from_numpy(feats)


def test_zone_from_xy_torch_matches_numpy_implementation() -> None:
    """``zone_from_xy_torch`` must agree with the canonical numpy
    ``zone_from_xy_vectorized`` on a court-grid sweep."""
    xs = np.linspace(-26.0, 26.0, 41)
    ys = np.linspace(-6.0, 48.0, 41)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    expected = zone_from_xy_vectorized(xx.ravel(), yy.ravel()).reshape(xx.shape)
    xy = torch.from_numpy(np.stack([xx, yy], axis=-1).astype(np.float32))
    got = zone_from_xy_torch(xy).numpy()
    np.testing.assert_array_equal(got, expected)


def test_forward_shape_and_dtype() -> None:
    field = ZoneReweightingDefense(n_opponents=30, beta_init=0.5)
    B, M = 4, 7
    xy = torch.randn(B, M, 2)
    feats = _random_features(B)
    out = field(query_xy=xy, def_features=feats)
    assert out.shape == (B, M)
    assert out.dtype == feats.dtype


def test_beta_zero_collapses_to_exact_zero() -> None:
    """β_D = 0 → D-lite contribution is identically zero for any input."""
    field = ZoneReweightingDefense(n_opponents=30, beta_init=0.0)
    xy = torch.tensor([[[0.0, 5.0], [22.5, 4.0], [-23.0, 3.0]]])
    feats = _random_features(1)
    out = field(query_xy=xy, def_features=feats)
    assert torch.allclose(out, torch.zeros_like(out))


def test_support_shots_in_same_zone_get_same_score() -> None:
    """Two support shots in the same zone for the same opponent must
    receive identical D-lite contributions."""
    field = ZoneReweightingDefense(n_opponents=30, beta_init=1.0)
    # Two rim points (zone 0) and two TopKey3 points (zone 7).
    xy = torch.tensor([[[0.0, 2.0], [1.0, 3.0], [0.0, 24.0], [3.0, 24.0]]])
    feats = _random_features(1)
    out = field(query_xy=xy, def_features=feats)
    assert torch.isclose(out[0, 0], out[0, 1])
    assert torch.isclose(out[0, 2], out[0, 3])
    # Different zones with different feature values → different scores.
    assert not torch.isclose(out[0, 0], out[0, 2])


def test_changing_opp_features_changes_scores() -> None:
    """Same xy + β_D, different per-row features → different D-lite."""
    field = ZoneReweightingDefense(n_opponents=30, beta_init=1.0)
    xy = torch.tensor([[[15.0, 10.0]], [[15.0, 10.0]]])  # one midrange point
    feats_a = _random_features(1, seed=0)
    feats_b = _random_features(1, seed=1)
    feats = torch.cat([feats_a, feats_b], dim=0)
    out = field(query_xy=xy, def_features=feats)
    # Both rows share xy + β_D but their feature rows differ → scores differ.
    assert not torch.isclose(out[0, 0], out[1, 0])


def test_cold_start_features_yield_zero() -> None:
    """An all-zero ``def_features`` row (cold-start opponent at snapshot)
    must produce exactly zero D-lite scores regardless of β_D."""
    field = ZoneReweightingDefense(n_opponents=30, beta_init=2.5)
    xy = torch.tensor([[[10.0, 10.0], [-20.0, 5.0]]])
    feats = torch.zeros(1, DEFENSE_FEATURE_DIM)
    out = field(query_xy=xy, def_features=feats)
    assert torch.allclose(out, torch.zeros_like(out))


def test_out_of_court_xy_is_zeroed() -> None:
    """Out-of-court support shots (zone == -1) must contribute exactly
    zero, even if β_D ≠ 0 and the centered-zone vector is nonzero."""
    field = ZoneReweightingDefense(n_opponents=30, beta_init=1.0)
    # First point is in court (rim), second is behind the baseline.
    xy = torch.tensor([[[0.0, 2.0], [0.0, -10.0]]])
    feats = _random_features(1)
    out = field(query_xy=xy, def_features=feats)
    assert not torch.isclose(out[0, 0], torch.tensor(0.0))
    assert torch.isclose(out[0, 1], torch.tensor(0.0))


def test_gradient_flows_to_beta_d() -> None:
    """L = sum(D) must produce a nonzero gradient on ``beta_D``."""
    field = ZoneReweightingDefense(n_opponents=30, beta_init=0.1)
    xy = torch.tensor([[[0.0, 2.0], [22.5, 4.0]]])
    feats = _random_features(1)
    # Use same-sign values on the queried zones (0=rim, 4=corner3R) so
    # the sum-gradient on β_D doesn't cancel to zero.
    feats[0, _ZONE_CENTERED_SLICE] = torch.tensor([0.2, -0.05, 0.0, 0.1, 0.15, 0.05, -0.05, 0.0])
    out = field(query_xy=xy, def_features=feats)
    loss = out.sum()
    loss.backward()
    assert field.beta_D.grad is not None
    assert not torch.isclose(field.beta_D.grad, torch.tensor(0.0))


def test_n_opponents_stored() -> None:
    """``_n_opponents`` is preserved for downstream diagnostics
    (matches the D-field attribute name)."""
    field = ZoneReweightingDefense(n_opponents=42, beta_init=1e-3)
    assert field._n_opponents == 42


def test_score_equals_beta_times_centered_zone() -> None:
    """The closed-form check: D_m = β_D * centered_zones[zone(s_m)]."""
    field = ZoneReweightingDefense(n_opponents=30, beta_init=0.0)
    with torch.no_grad():
        field.beta_D.copy_(torch.tensor(2.0))
    # A rim point (zone 0) and a corner-3-left point (zone 3).
    xy = torch.tensor([[[0.0, 1.0], [-23.5, 5.0]]])
    centered = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])  # (1, 8)
    feats = torch.zeros(1, DEFENSE_FEATURE_DIM)
    feats[:, _ZONE_CENTERED_SLICE] = centered
    out = field(query_xy=xy, def_features=feats)
    # zone(0, 1) = 0 (rim) → 2.0 * 0.1 = 0.2
    # zone(-23.5, 5) = 3 (LC3) → 2.0 * 0.4 = 0.8
    assert torch.isclose(out[0, 0], torch.tensor(0.2))
    assert torch.isclose(out[0, 1], torch.tensor(0.8))


def test_zone_count_matches_features() -> None:
    """The 8-zone centered block in features must have N_ZONES entries —
    sanity check that D-lite's gather over the slice matches the
    feature layout exactly."""
    sl = _ZONE_CENTERED_SLICE
    assert (sl.stop - sl.start) == N_ZONES


# ---------------------------------------------------------------------------
# D-lite-zone (per-zone γ_z multiplier)
# ---------------------------------------------------------------------------


def test_per_zone_gamma_disabled_by_default() -> None:
    """``per_zone_gamma=False`` (default) → no ``gamma_z`` parameter."""
    field = ZoneReweightingDefense(n_opponents=30)
    assert field.gamma_z is None
    assert not field.per_zone_gamma


def test_per_zone_gamma_creates_unit_init_parameter() -> None:
    """``per_zone_gamma=True`` registers a learnable ``gamma_z`` of
    shape (N_ZONES,), initialized to all-ones so that at step 0
    D-lite-zone is bit-equal to D-lite-0."""
    field = ZoneReweightingDefense(n_opponents=30, per_zone_gamma=True)
    assert field.gamma_z is not None
    assert field.gamma_z.shape == (N_ZONES,)
    assert torch.allclose(field.gamma_z, torch.ones(N_ZONES))
    assert field.per_zone_gamma


def test_per_zone_gamma_at_init_equals_d_lite_zero() -> None:
    """At init (γ_z = 1.0), D-lite-zone produces *exactly* the same
    scores as D-lite-0 with the same β_D init."""
    beta = 0.7
    field_0 = ZoneReweightingDefense(n_opponents=30, beta_init=beta, per_zone_gamma=False)
    field_gamma = ZoneReweightingDefense(n_opponents=30, beta_init=beta, per_zone_gamma=True)
    xy = torch.tensor([[[0.0, 2.0], [22.5, 4.0], [15.0, 10.0]]])
    feats = _random_features(1, seed=42)
    with torch.no_grad():
        out_0 = field_0(query_xy=xy, def_features=feats)
        out_gamma = field_gamma(query_xy=xy, def_features=feats)
    torch.testing.assert_close(out_0, out_gamma)


def test_per_zone_gamma_changes_scores_when_gamma_diverges() -> None:
    """After γ_z deviates from 1.0, D-lite-zone scores differ from
    D-lite-0 zone-by-zone — confirms γ_z modulates per-zone
    independently."""
    field = ZoneReweightingDefense(n_opponents=30, beta_init=1.0, per_zone_gamma=True)
    assert field.gamma_z is not None
    with torch.no_grad():
        field.gamma_z.copy_(torch.tensor([2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]))
    # Two rim points (zone 0, γ=2) and two LC3 points (zone 3, γ=0.5).
    xy = torch.tensor([[[0.0, 2.0], [1.0, 3.0], [-23.0, 4.0], [-23.5, 6.0]]])
    centered = torch.tensor([[0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]])  # rim=LC3=0.1
    feats = torch.zeros(1, DEFENSE_FEATURE_DIM)
    feats[:, _ZONE_CENTERED_SLICE] = centered
    with torch.no_grad():
        out = field(query_xy=xy, def_features=feats)
    # Rim: β·γ·q = 1·2·0.1 = 0.2
    # LC3: β·γ·q = 1·0.5·0.1 = 0.05
    assert torch.isclose(out[0, 0], torch.tensor(0.2))
    assert torch.isclose(out[0, 1], torch.tensor(0.2))  # same zone, same score
    assert torch.isclose(out[0, 2], torch.tensor(0.05))
    assert torch.isclose(out[0, 3], torch.tensor(0.05))


def test_per_zone_gamma_gradient_flows_to_both_beta_and_gamma() -> None:
    """L = sum(D) produces nonzero gradients on both β_D and γ_z."""
    field = ZoneReweightingDefense(n_opponents=30, beta_init=0.5, per_zone_gamma=True)
    # Rim, LC3, ATB3.
    xy = torch.tensor([[[0.0, 2.0], [-23.5, 5.0], [0.0, 25.0]]])
    feats = torch.zeros(1, DEFENSE_FEATURE_DIM)
    feats[0, _ZONE_CENTERED_SLICE] = torch.tensor([0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.15])
    out = field(query_xy=xy, def_features=feats)
    loss = out.sum()
    loss.backward()
    assert field.beta_D.grad is not None and not torch.isclose(field.beta_D.grad, torch.tensor(0.0))
    assert field.gamma_z is not None
    assert field.gamma_z.grad is not None
    # Touched zones (0, 3, 7) must have nonzero γ_z grad; untouched
    # zones must have exactly zero γ_z grad.
    g = field.gamma_z.grad
    assert g[0] != 0 and g[3] != 0 and g[7] != 0
    assert g[1] == 0 and g[2] == 0 and g[4] == 0 and g[5] == 0 and g[6] == 0


def test_per_zone_gamma_state_dict_roundtrip() -> None:
    """``per_zone_gamma=True`` state_dict carries both β_D and γ_z; a
    fresh module with the same flag loads them correctly."""
    src = ZoneReweightingDefense(n_opponents=30, beta_init=0.1, per_zone_gamma=True)
    assert src.gamma_z is not None
    with torch.no_grad():
        src.beta_D.copy_(torch.tensor(0.75))
        src.gamma_z.copy_(torch.tensor([0.5, 1.5, 2.0, 0.25, 1.0, 1.0, 0.8, 1.2]))
    sd = src.state_dict()
    assert "beta_D" in sd and "gamma_z" in sd

    dst = ZoneReweightingDefense(n_opponents=30, per_zone_gamma=True)
    dst.load_state_dict(sd)
    assert torch.isclose(dst.beta_D, torch.tensor(0.75))
    assert dst.gamma_z is not None
    torch.testing.assert_close(dst.gamma_z, src.gamma_z)
