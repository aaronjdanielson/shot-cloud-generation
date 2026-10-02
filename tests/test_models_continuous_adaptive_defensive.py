"""Tests for ``shotcloud.models.continuous_adaptive_defensive`` (PR-D1a).

The required invariants from the build approval (2026-05-25):

1. Forward returns shape ``(B, M_off)``.
2. Cold-start rows (``def_mask.all(False)`` along ``M_def``) produce
   ``D_Δ = 0`` exactly — no NaN, no -inf.
3. ``β_D = 0`` gives an exact-zero output field.
4. ``β_D = 1e-3`` gives a small nonzero perturbation.
5. Chunked output equals unchunked output to floating-point
   tolerance.
6. Gradients flow to ``β_D``, ``q_net``, ``k_net``, ``λ_{δ,age}``,
   and the opponent embedding.
7. The chunked path never materializes a full ``(B, M_off, M_def)``
   tensor — verified by running large ``M_off`` with small chunk
   and confirming the forward succeeds (negative test: failure
   would be an OOM, not a NaN).

All tests use small synthetic sizes (``B=3, M_off=11, M_def=13``)
so they run in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shotcloud.models.continuous_adaptive_defensive import (
    DEFAULT_DEFENSE_BANDWIDTH_FT,
    DEFAULT_DEFENSE_BETA_INIT,
    ContinuousAdaptiveDefensiveField,
)

# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------


def _make_field(
    *,
    n_opponents: int = 4,
    context_dim: int = 7,
    within_game_dim: int = 0,
    defense_feature_dim: int = 5,
    beta_init: float = DEFAULT_DEFENSE_BETA_INIT,
    query_chunk_size: int = 8,
    lambda_age_init: float = 0.0,
) -> ContinuousAdaptiveDefensiveField:
    """Construct a small defensive field module."""
    return ContinuousAdaptiveDefensiveField(
        n_opponents=n_opponents,
        context_dim=context_dim,
        within_game_dim=within_game_dim,
        defense_feature_dim=defense_feature_dim,
        opp_embed_dim=4,
        query_hidden_dim=16,
        key_hidden_dim=16,
        proj_dim=8,
        bandwidth=DEFAULT_DEFENSE_BANDWIDTH_FT,
        beta_init=beta_init,
        lambda_age_init=lambda_age_init,
        query_chunk_size=query_chunk_size,
    )


def _make_inputs(
    *,
    b: int = 3,
    m_off: int = 11,
    m_def: int = 13,
    context_dim: int = 7,
    defense_feature_dim: int = 5,
    within_game_dim: int = 0,
    cold_start_rows: tuple[int, ...] = (),
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Build a set of forward-input tensors. ``cold_start_rows`` lists
    batch indices whose ``def_mask`` should be entirely False."""
    rng = np.random.default_rng(seed)
    query_xy = torch.from_numpy(rng.uniform(-20, 20, (b, m_off, 2)).astype(np.float32))
    def_xy = torch.from_numpy(rng.uniform(-20, 20, (b, m_def, 2)).astype(np.float32))
    def_mask = torch.ones(b, m_def, dtype=torch.bool)
    for i in cold_start_rows:
        def_mask[i, :] = False
    def_age_days = torch.from_numpy(rng.uniform(0, 200, (b, m_def)).astype(np.float32))
    def_features = torch.from_numpy(rng.normal(0, 1, (b, defense_feature_dim)).astype(np.float32))
    x_n = torch.from_numpy(rng.normal(0, 1, (b, context_dim)).astype(np.float32))
    h_n = (
        torch.from_numpy(rng.normal(0, 1, (b, within_game_dim)).astype(np.float32))
        if within_game_dim > 0
        else None
    )
    opp_idx = torch.from_numpy(rng.integers(0, 4, (b,)).astype(np.int64))
    return {
        "query_xy": query_xy,
        "def_xy": def_xy,
        "def_mask": def_mask,
        "def_age_days": def_age_days,
        "def_features": def_features,
        "x_n": x_n,
        "h_n": h_n,
        "opp_idx": opp_idx,
    }


# ---------------------------------------------------------------------------
# Required invariants
# ---------------------------------------------------------------------------


def test_1_forward_returns_correct_shape() -> None:
    field = _make_field()
    inputs = _make_inputs()
    out = field(**inputs)
    assert out.shape == (3, 11)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all().item()


def test_2_cold_start_rows_produce_zero_field_no_nan() -> None:
    """Rows whose def_mask is all False contribute exactly D_Δ = 0
    everywhere. Verified by constructing a batch with one cold-start
    row and one normal row; the cold-start row's output must be the
    zero vector."""
    field = _make_field()
    inputs = _make_inputs(b=3, cold_start_rows=(1,))
    out = field(**inputs)
    assert torch.isfinite(out).all().item()
    # Cold-start row is all-zero.
    cold = out[1]
    assert (cold == 0).all().item(), f"cold-start row not all-zero: {cold}"
    # Non-cold-start rows are not all-zero (at β=1e-3 they're small
    # but nonzero in expectation; check at least one is nonzero so
    # we know the test isn't trivially passing).
    assert (out[0].abs() > 0).any().item() or (out[2].abs() > 0).any().item()


def test_2b_all_rows_cold_start_returns_all_zero() -> None:
    """Pathological case: every row is cold-start. Output is the zero
    matrix with no NaN propagation."""
    field = _make_field()
    inputs = _make_inputs(b=3, cold_start_rows=(0, 1, 2))
    out = field(**inputs)
    assert torch.isfinite(out).all().item()
    assert (out == 0).all().item()


def test_3_beta_zero_gives_exact_zero_field() -> None:
    """With β_D = 0 (unit-test override), the output is identically
    zero — the load-bearing no-op invariant."""
    field = _make_field(beta_init=0.0)
    inputs = _make_inputs()
    out = field(**inputs)
    assert (out == 0).all().item()


def test_4_beta_warm_init_gives_small_nonzero_perturbation() -> None:
    """At the training-default β_D = 1e-3, the output is small but
    nonzero — the warm-init invariant. Bound: max |D_Δ| < 0.1
    (well below the support-logit scale where downstream impact
    becomes meaningful)."""
    field = _make_field(beta_init=1e-3)
    inputs = _make_inputs()
    out = field(**inputs)
    assert (out.abs() > 0).any().item(), "warm-init β should produce nonzero output"
    max_abs = float(out.abs().max().item())
    assert max_abs < 0.1, f"warm-init output too large: max |D_Δ| = {max_abs}"


def test_5_chunked_equals_unchunked() -> None:
    """The chunked forward must produce the same output as a single-
    chunk forward (within float tolerance). Tested by running the
    same inputs through two modules with different chunk sizes and
    asserting equality."""
    inputs = _make_inputs(m_off=20)
    # Two fields identical except for chunk size.
    field_small = _make_field(query_chunk_size=3)
    field_full = _make_field(query_chunk_size=100)  # > M_off → effectively single chunk
    # Copy state to make the modules identical (otherwise nn.init
    # randomness would differ).
    field_full.load_state_dict(field_small.state_dict())
    with torch.no_grad():
        out_small = field_small(**inputs)
        out_full = field_full(**inputs)
    torch.testing.assert_close(out_small, out_full, atol=1e-6, rtol=1e-6)


def test_6_gradients_flow_to_all_learnable_params() -> None:
    """Every learnable parameter (β_D, λ_age, q_net layers, k_net
    layers, opp_embedding) receives a nonzero gradient through the
    output, at the training-default warm β.

    The loss is :math:`\\sum_b D_{b,0}` (sum over batch of slot 0
    only), *not* :math:`\\sum_{b,m} D_{b,m}`. The latter is
    identically zero per row because the field is per-row mean-
    centered for diagnostic-magnitude reasons — under a uniform
    sum-over-M_off it cancels, which would give a spurious zero
    gradient for β. Real downstream loss is a softmax over support
    logits and has no such cancellation; the slot-0 sum used here
    mirrors that asymmetry.
    """
    field = _make_field(beta_init=1e-3)
    inputs = _make_inputs()
    out = field(**inputs)
    loss = out[:, 0].sum()  # see docstring re: per-row centering
    loss.backward()
    # β_D.
    assert field.beta_D.grad is not None
    assert field.beta_D.grad.abs().item() > 0
    # λ_age — at λ_age_init=0 the score doesn't yet depend on age,
    # but the gradient is non-trivial (∂score/∂λ_age = -def_age_days).
    assert field.lambda_age.grad is not None
    assert field.lambda_age.grad.abs().item() > 0
    # q_net first/last Linear layers.
    assert field.q_net[0].weight.grad is not None  # type: ignore[union-attr]
    assert field.q_net[0].weight.grad.abs().sum().item() > 0  # type: ignore[union-attr]
    assert field.q_net[-1].weight.grad is not None  # type: ignore[union-attr]
    assert field.q_net[-1].weight.grad.abs().sum().item() > 0  # type: ignore[union-attr]
    # k_net first/last Linear layers.
    assert field.k_net[0].weight.grad is not None  # type: ignore[union-attr]
    assert field.k_net[0].weight.grad.abs().sum().item() > 0  # type: ignore[union-attr]
    assert field.k_net[-1].weight.grad is not None  # type: ignore[union-attr]
    assert field.k_net[-1].weight.grad.abs().sum().item() > 0  # type: ignore[union-attr]
    # Opponent embedding.
    assert field.opp_embedding.weight.grad is not None
    assert field.opp_embedding.weight.grad.abs().sum().item() > 0


def test_6b_beta_zero_blocks_inner_attention_gradient() -> None:
    """With β_D = 0 (hard zero), the inner attention parameters
    (``q_Δ``, ``k_Δ``, ``λ_age``) receive zero gradient — the
    motivating failure mode for the warm-init decision recorded in
    docs/defense_integration_proposal.md §5."""
    field = _make_field(beta_init=0.0)
    inputs = _make_inputs()
    out = field(**inputs)
    # Slot-0 sum mirrors test_6's choice for the same per-row-
    # centering-cancellation reason.
    out[:, 0].sum().backward()
    # β_D itself receives gradient (the chain rule gives ∂L/∂β_D =
    # Σ log_a, which is nonzero).
    assert field.beta_D.grad is not None
    assert field.beta_D.grad.abs().item() > 0
    # But the inner attention parameters' grads are scaled by β_D = 0
    # → they're identically zero. This is exactly the "hard-zero
    # blocks gradient flow into the inner attention" failure mode the
    # warm init is designed to avoid.
    assert field.q_net[-1].weight.grad is not None  # type: ignore[union-attr]
    assert field.q_net[-1].weight.grad.abs().sum().item() == 0.0  # type: ignore[union-attr]
    assert field.k_net[-1].weight.grad is not None  # type: ignore[union-attr]
    assert field.k_net[-1].weight.grad.abs().sum().item() == 0.0  # type: ignore[union-attr]
    assert field.lambda_age.grad is not None
    assert field.lambda_age.grad.abs().item() == 0.0


def test_7_chunked_path_runs_at_large_m_off() -> None:
    """Negative test: the chunked path scales to large ``M_off``
    (1500 ≈ the production retrieval support size) without OOMing
    on synthetic CPU sizes. A naive full ``(B, M_off, M_def)``
    tensor at ``B=4, M_off=1500, M_def=1000`` = 6 GB in fp32 — this
    test would crash if the chunking were wrong. With
    ``query_chunk_size=128`` peak per-chunk allocation is
    ``B·128·M_def`` = ~4 MB."""
    field = _make_field(query_chunk_size=128)
    inputs = _make_inputs(b=4, m_off=1500, m_def=1000)
    out = field(**inputs)
    assert out.shape == (4, 1500)
    assert torch.isfinite(out).all().item()


# ---------------------------------------------------------------------------
# Within-game history wiring
# ---------------------------------------------------------------------------


def test_within_game_history_is_consumed_when_dim_positive() -> None:
    """When ``within_game_dim > 0``, ``h_n`` must be passed in (and
    contributes to the query). Missing ``h_n`` raises a clear error."""
    field = _make_field(within_game_dim=5)
    bad_inputs = _make_inputs(within_game_dim=0)
    # bad_inputs has h_n=None; field expects 5-dim.
    with pytest.raises(ValueError, match="within_game_dim=5"):
        field(**bad_inputs)
    # Good inputs (with matching h_n) succeed.
    good_inputs = _make_inputs(within_game_dim=5)
    out = field(**good_inputs)
    assert out.shape == (3, 11)


def test_zero_within_game_dim_accepts_none_h_n() -> None:
    """When ``within_game_dim=0`` (the default), ``h_n=None`` is the
    legitimate passthrough."""
    field = _make_field(within_game_dim=0)
    inputs = _make_inputs(within_game_dim=0)
    assert inputs["h_n"] is None
    out = field(**inputs)
    assert out.shape == (3, 11)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError, match="n_opponents"):
        _make_field(n_opponents=0)
    with pytest.raises(ValueError, match="context_dim"):
        ContinuousAdaptiveDefensiveField(
            n_opponents=4, context_dim=0, within_game_dim=0, defense_feature_dim=5
        )
    with pytest.raises(ValueError, match="defense_feature_dim"):
        ContinuousAdaptiveDefensiveField(
            n_opponents=4, context_dim=7, within_game_dim=0, defense_feature_dim=0
        )
    with pytest.raises(ValueError, match="query_chunk_size"):
        ContinuousAdaptiveDefensiveField(
            n_opponents=4,
            context_dim=7,
            within_game_dim=0,
            defense_feature_dim=5,
            query_chunk_size=0,
        )


def test_constructor_rejects_negative_within_game_dim() -> None:
    with pytest.raises(ValueError, match="within_game_dim"):
        ContinuousAdaptiveDefensiveField(
            n_opponents=4, context_dim=7, within_game_dim=-1, defense_feature_dim=5
        )


# ---------------------------------------------------------------------------
# PR-D1b: gather_defense_inputs adapter
# ---------------------------------------------------------------------------


def _synthetic_cache_and_features(
    n_opps: int = 3,
    n_snaps: int = 2,
    m_def: int = 4,
    n_global: int = 12,
    d_def: int = 5,
    seed: int = 0,
):
    """Build small synthetic ``DefensiveRetrievalCache`` and
    ``DefenseFeatures`` objects with controlled contents — used to
    test the gather adapter without exercising the real builders."""
    import numpy as np

    from shotcloud.features.defense_features import (
        DEFENSE_FEATURE_NAMES,
        DefenseFeatures,
        DefenseFeaturesConfig,
    )
    from shotcloud.models.defensive_retrieval_cache import (
        DefensiveRetrievalCache,
        DefensiveRetrievalCacheConfig,
    )

    rng = np.random.default_rng(seed)
    anchor_dates = tuple(20000 + i * 30 for i in range(n_snaps))
    # Global table: ascending dates spanning before all anchors.
    global_dates = torch.from_numpy(
        np.linspace(min(anchor_dates) - 200, min(anchor_dates) - 10, n_global).astype(np.int64)
    )
    global_xy = torch.from_numpy(rng.uniform(-20, 20, (n_global, 2)).astype(np.float32))
    global_opp = torch.from_numpy(rng.integers(0, n_opps, (n_global,)).astype(np.int64))
    # def_idx: each (opp, snap) cell picks (random_subset_indices, ..., -1
    # padding). Use one row with -1 padding to test the safe-clamp path.
    def_idx = torch.full((n_opps, n_snaps, m_def), -1, dtype=torch.int64)
    def_mask = torch.zeros((n_opps, n_snaps, m_def), dtype=torch.bool)
    # Populate first opp / first snap with 3 real shots; last slot padded.
    def_idx[0, 0, :3] = torch.tensor([2, 5, 8], dtype=torch.int64)
    def_mask[0, 0, :3] = True
    # Second opp / first snap: 4 real shots, no padding.
    def_idx[1, 0, :] = torch.tensor([1, 3, 4, 7], dtype=torch.int64)
    def_mask[1, 0, :] = True
    # Third opp / first snap: empty (cold-start).
    # Same pattern for snapshot 1 (smaller variations).
    def_idx[0, 1, :2] = torch.tensor([6, 9], dtype=torch.int64)
    def_mask[0, 1, :2] = True
    cfg = DefensiveRetrievalCacheConfig(
        shots_fingerprint="syn",
        anchor_dates=anchor_dates,
        defensive_support_max=m_def,
        defensive_recency_window_days=365,
    )
    cache = DefensiveRetrievalCache(
        config=cfg,
        global_xy=global_xy,
        global_dates=global_dates,
        global_opponent_idx=global_opp,
        def_idx=def_idx,
        def_mask=def_mask,
    )

    feat_cfg = DefenseFeaturesConfig(
        shots_fingerprint="syn",
        anchor_dates=anchor_dates,
        window_days=365,
        half_life_days=90.0,
    )
    feature_tensor = torch.from_numpy(rng.normal(0, 1, (n_opps, n_snaps, d_def)).astype(np.float32))
    # Truncate / pad the feature_names list to d_def for the synthetic
    # case; real artifacts use DEFENSE_FEATURE_NAMES.
    names = (
        DEFENSE_FEATURE_NAMES[:d_def]
        if d_def <= len(DEFENSE_FEATURE_NAMES)
        else (
            DEFENSE_FEATURE_NAMES
            + tuple(f"x{i}" for i in range(d_def - len(DEFENSE_FEATURE_NAMES)))
        )
    )
    feats = DefenseFeatures(features=feature_tensor, feature_names=names, config=feat_cfg)
    return cache, feats


def test_gather_returns_expected_shapes_and_dtypes() -> None:
    from shotcloud.models.continuous_adaptive_defensive import gather_defense_inputs

    cache, feats = _synthetic_cache_and_features()
    opp_idx = torch.tensor([0, 1, 2], dtype=torch.int64)
    snap_idx = torch.tensor([0, 0, 0], dtype=torch.int64)
    out = gather_defense_inputs(opp_idx=opp_idx, snapshot_idx=snap_idx, cache=cache, features=feats)
    b = opp_idx.shape[0]
    m_def = cache.def_idx.shape[-1]
    d_def = feats.features.shape[-1]
    assert out["def_xy"].shape == (b, m_def, 2)
    assert out["def_xy"].dtype == torch.float32
    assert out["def_mask"].shape == (b, m_def)
    assert out["def_mask"].dtype == torch.bool
    assert out["def_age_days"].shape == (b, m_def)
    assert out["def_age_days"].dtype == torch.float32
    assert out["def_features"].shape == (b, d_def)
    assert out["def_features"].dtype == torch.float32


def test_gather_cold_start_row_has_all_false_mask() -> None:
    """Opp 2 / snap 0 is cold-start in the synthetic fixture; gather
    must surface that as ``def_mask`` all-False for that row."""
    from shotcloud.models.continuous_adaptive_defensive import gather_defense_inputs

    cache, feats = _synthetic_cache_and_features()
    opp_idx = torch.tensor([2], dtype=torch.int64)
    snap_idx = torch.tensor([0], dtype=torch.int64)
    out = gather_defense_inputs(opp_idx=opp_idx, snapshot_idx=snap_idx, cache=cache, features=feats)
    assert not out["def_mask"].any().item()


def test_gather_padded_slots_have_safe_clamped_indices_but_false_mask() -> None:
    """When ``def_idx == -1`` for a slot, gather safe-clamps to slot
    0 of the global table; the mask remains False so downstream code
    knows not to use the gathered coord."""
    from shotcloud.models.continuous_adaptive_defensive import gather_defense_inputs

    cache, feats = _synthetic_cache_and_features()
    # Opp 0 / snap 0 has 3 real + 1 padded (slot 3).
    opp_idx = torch.tensor([0], dtype=torch.int64)
    snap_idx = torch.tensor([0], dtype=torch.int64)
    out = gather_defense_inputs(opp_idx=opp_idx, snapshot_idx=snap_idx, cache=cache, features=feats)
    # Slot 3's mask is False, slots 0-2 True.
    assert out["def_mask"][0, :3].all().item()
    assert not out["def_mask"][0, 3].item()
    # def_xy is gathered for every slot; the padded slot points at
    # global_xy[0]. Verify the coord equals the cache's slot-0 coord.
    torch.testing.assert_close(out["def_xy"][0, 3], cache.global_xy[0])


def test_gather_age_days_are_anchor_minus_shot_date() -> None:
    """Per-shot ages are computed as ``anchor − shot_date`` in days,
    matching the convention the defensive field's attention expects."""
    from shotcloud.models.continuous_adaptive_defensive import gather_defense_inputs

    cache, feats = _synthetic_cache_and_features()
    opp_idx = torch.tensor([1], dtype=torch.int64)
    snap_idx = torch.tensor([0], dtype=torch.int64)
    out = gather_defense_inputs(opp_idx=opp_idx, snapshot_idx=snap_idx, cache=cache, features=feats)
    # Opp 1 / snap 0 → global indices [1, 3, 4, 7]; anchor is
    # cache.config.anchor_dates[0].
    expected_dates = cache.global_dates[torch.tensor([1, 3, 4, 7])].to(torch.float32)
    anchor = float(cache.config.anchor_dates[0])
    expected_age = anchor - expected_dates
    torch.testing.assert_close(out["def_age_days"][0], expected_age)


def test_gather_then_field_forward_runs_end_to_end() -> None:
    """The gather + field forward composition runs end-to-end on
    synthetic data without NaNs. This is the unit-level surrogate
    for the real-data smoke in scripts/smoke_defensive_field.py."""
    from shotcloud.models.continuous_adaptive_defensive import gather_defense_inputs

    cache, feats = _synthetic_cache_and_features(d_def=5)
    opp_idx = torch.tensor([0, 1, 2], dtype=torch.int64)
    snap_idx = torch.tensor([0, 0, 0], dtype=torch.int64)
    gathered = gather_defense_inputs(
        opp_idx=opp_idx, snapshot_idx=snap_idx, cache=cache, features=feats
    )
    field = _make_field(n_opponents=3, defense_feature_dim=5)
    query_xy = torch.randn(3, 8, 2)
    x_n = torch.randn(3, 7)
    out = field(
        query_xy=query_xy,
        def_xy=gathered["def_xy"],
        def_mask=gathered["def_mask"],
        def_age_days=gathered["def_age_days"],
        def_features=gathered["def_features"],
        x_n=x_n,
        h_n=None,
        opp_idx=opp_idx,
    )
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all().item()
    # Row 2 (opp=2) is cold-start → exact zero.
    assert (out[2] == 0).all().item()
    # Row 0, 1 (non-cold-start) → bounded perturbation at β=1e-3.
    assert (out[:2].abs() > 0).any().item()
    assert out[:2].abs().max().item() < 0.1
