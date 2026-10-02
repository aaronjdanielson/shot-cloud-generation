"""Tests for :mod:`shotcloud.legacy_pivot.archetypes`.

The archetype dictionary holds the frozen archetype surfaces; the
mixture network produces ``rho_xi(p, x_n)``. Together they implement
``q^arch_xi(c | p, x_n) = sum_k rho_xi,k * A_k^{(t_i)}(c)``.

Tests cover:

* dictionary normalization invariants (rows sum to 1, non-negative),
* mixture-weight simplex output,
* batched forward pass shape and identity (q_arch sums to 1 per row),
* integration with :class:`SnapshotStore` (archetype basis frozen,
  matches ``bundle.archetype_surfaces``),
* deterministic forward (no rng),
* gradient flows through mixture parameters,
* fallback to uniform surfaces when no pretrain has run,
* role-vector signature ``a_k`` is recoverable from the trained
  parameter (not buried in an opaque MLP).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud.data import (
    ROLE_PROFILE_DIM,
    SnapshotStore,
    build_role_profiles,
    build_snapshot_store_from_shots,
)
from shotcloud.data.context import CONTEXT_DIM, FEATURE_LAYOUT
from shotcloud.legacy_pivot.archetypes import DEFAULT_K, ArchetypeDictionary, ArchetypeMixture

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _normalized_surfaces(
    n_anchors: int = 3, n_archetypes: int = 4, n_cells: int = 64, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.random((n_anchors, n_archetypes, n_cells)).astype(np.float32)
    raw /= raw.sum(axis=-1, keepdims=True)
    return raw


def _synthetic_shots(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for date in pd.date_range("2018-01-01", "2020-12-31", freq="W"):
        for _ in range(3):
            rows.append(
                {
                    "x": float(rng.normal(0.0, 5.0)),
                    "y": float(rng.normal(15.0, 5.0)),
                    "player_id": int(rng.choice([1, 2, 3])),
                    "opponent": str(rng.choice(["BOS", "LAL", "GSW"])),
                    "made": int(rng.random() < 0.45),
                    "period": int(rng.choice([1, 2, 3, 4])),
                    "time_remaining_sec": int(rng.uniform(0, 12 * 60 * 4)),
                    "starter": int(rng.random() < 0.7),
                    "minutes": int(rng.uniform(8, 40)),
                    "date": pd.Timestamp(date),
                }
            )
    return pd.DataFrame(rows)


def _build_store_with_archetypes(
    shots: pd.DataFrame, K: int = 4, n_cells: int = 50
) -> SnapshotStore:
    """Build a snapshot store with synthetic archetype surfaces in every bundle."""
    anchors = [
        np.datetime64("2018-06-01"),
        np.datetime64("2019-06-01"),
        np.datetime64("2020-06-01"),
    ]
    rng = np.random.default_rng(42)

    def stub_archetype_fit(_: pd.DataFrame, _t: np.datetime64) -> np.ndarray:
        surfaces = rng.random((K, n_cells)).astype(np.float32)
        surfaces /= surfaces.sum(axis=-1, keepdims=True)
        return surfaces

    return build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=build_role_profiles,
        archetype_fit_fn=stub_archetype_fit,
    )


# ---------------------------------------------------------------------------
# ArchetypeDictionary: construction and validation
# ---------------------------------------------------------------------------


def test_dictionary_construction_shape() -> None:
    surfaces = _normalized_surfaces(n_anchors=5, n_archetypes=3, n_cells=64)
    d = ArchetypeDictionary(surfaces)
    assert d.n_anchors == 5
    assert d.n_archetypes == 3
    assert d.n_cells == 64


def test_dictionary_buffers_not_parameters() -> None:
    """surfaces must be a buffer (frozen), not a parameter."""
    d = ArchetypeDictionary(_normalized_surfaces())
    assert "surfaces" in dict(d.named_buffers())
    assert "surfaces" not in dict(d.named_parameters())
    # No learnable parameters at all
    assert list(d.parameters()) == []


def test_dictionary_rejects_non_3d_input() -> None:
    with pytest.raises(ValueError, match="3-D"):
        ArchetypeDictionary(np.ones((4, 64), dtype=np.float32))


def test_dictionary_rejects_empty_dimension() -> None:
    with pytest.raises(ValueError, match="empty"):
        ArchetypeDictionary(np.zeros((0, 4, 64), dtype=np.float32))


def test_dictionary_rejects_negative_values() -> None:
    surfaces = _normalized_surfaces()
    surfaces[0, 0, 0] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        ArchetypeDictionary(surfaces)


def test_dictionary_rejects_non_simplex_rows() -> None:
    surfaces = np.full((2, 3, 64), 0.001, dtype=np.float32)  # rows sum to 0.064
    with pytest.raises(ValueError, match="rows must sum to 1"):
        ArchetypeDictionary(surfaces)


def test_dictionary_uniform_constructor() -> None:
    d = ArchetypeDictionary.uniform(n_anchors=3, n_archetypes=8, n_cells=100)
    assert d.surfaces.shape == (3, 8, 100)
    np.testing.assert_allclose(d.surfaces.sum(dim=-1).numpy(), 1.0, atol=1e-5)
    # Every cell at every (anchor, archetype) is identical
    assert torch.allclose(d.surfaces, torch.full_like(d.surfaces, 1.0 / 100), atol=1e-7)


def test_dictionary_uniform_rejects_zero_dim() -> None:
    with pytest.raises(ValueError, match="positive"):
        ArchetypeDictionary.uniform(n_anchors=0, n_archetypes=8, n_cells=100)


# ---------------------------------------------------------------------------
# ArchetypeDictionary: density lookups
# ---------------------------------------------------------------------------


def test_density_returns_full_snapshot_when_k_is_none() -> None:
    surfaces = _normalized_surfaces(n_anchors=3, n_archetypes=4, n_cells=50)
    d = ArchetypeDictionary(surfaces)
    snap = d.density(snapshot_idx=1)
    assert snap.shape == (4, 50)
    np.testing.assert_allclose(snap.numpy(), surfaces[1], atol=1e-6)


def test_density_returns_single_archetype_when_k_provided() -> None:
    surfaces = _normalized_surfaces()
    d = ArchetypeDictionary(surfaces)
    arch = d.density(snapshot_idx=2, k=1)
    assert arch.shape == (surfaces.shape[2],)
    np.testing.assert_allclose(arch.numpy(), surfaces[2, 1], atol=1e-6)


def test_density_raises_on_out_of_range() -> None:
    d = ArchetypeDictionary(_normalized_surfaces(n_anchors=3, n_archetypes=4))
    with pytest.raises(IndexError, match="snapshot_idx 99"):
        d.density(snapshot_idx=99)
    with pytest.raises(IndexError, match="k 99"):
        d.density(snapshot_idx=0, k=99)


# ---------------------------------------------------------------------------
# ArchetypeDictionary: forward (batched mixture)
# ---------------------------------------------------------------------------


def test_forward_q_arch_sums_to_one_per_row() -> None:
    surfaces = _normalized_surfaces(n_anchors=4, n_archetypes=8, n_cells=100)
    d = ArchetypeDictionary(surfaces)
    B = 16
    snapshot_idx = torch.randint(0, 4, (B,))
    # Random simplex mixture
    raw = torch.rand(B, 8)
    mixture = raw / raw.sum(dim=-1, keepdim=True)
    q_arch = d(snapshot_idx, mixture)
    assert q_arch.shape == (B, 100)
    np.testing.assert_allclose(q_arch.sum(dim=-1).numpy(), 1.0, atol=1e-5)


def test_forward_one_hot_mixture_recovers_archetype_surface() -> None:
    surfaces = _normalized_surfaces(n_anchors=3, n_archetypes=5, n_cells=80)
    d = ArchetypeDictionary(surfaces)
    B = 5
    snapshot_idx = torch.zeros(B, dtype=torch.long)  # all at snapshot 0
    # One-hot mixture: each row picks one archetype
    mixture = torch.eye(5)
    q_arch = d(snapshot_idx, mixture)
    # Row b should equal surfaces[0, b, :]
    expected = surfaces[0]
    np.testing.assert_allclose(q_arch.numpy(), expected, atol=1e-6)


def test_forward_validates_input_shapes() -> None:
    d = ArchetypeDictionary(_normalized_surfaces(n_anchors=3, n_archetypes=4))
    # snapshot_idx wrong dim
    with pytest.raises(ValueError, match="snapshot_idx must be 1-D"):
        d(torch.zeros(2, 3, dtype=torch.long), torch.ones(6, 4))
    # mixture wrong K
    with pytest.raises(ValueError, match="mixture must have shape"):
        d(torch.zeros(3, dtype=torch.long), torch.ones(3, 5))
    # batch size mismatch
    with pytest.raises(ValueError, match="batch dims must match"):
        d(torch.zeros(3, dtype=torch.long), torch.ones(4, 4))


def test_forward_is_deterministic() -> None:
    d = ArchetypeDictionary(_normalized_surfaces())
    snapshot_idx = torch.tensor([0, 1, 2, 0])
    raw = torch.rand(4, 4)
    mixture = raw / raw.sum(dim=-1, keepdim=True)
    out1 = d(snapshot_idx, mixture)
    out2 = d(snapshot_idx, mixture)
    np.testing.assert_array_equal(out1.numpy(), out2.numpy())


# ---------------------------------------------------------------------------
# ArchetypeDictionary: SnapshotStore integration
# ---------------------------------------------------------------------------


def test_dictionary_from_snapshot_store() -> None:
    shots = _synthetic_shots()
    store = _build_store_with_archetypes(shots, K=4, n_cells=50)
    d = ArchetypeDictionary.from_snapshot_store(store)
    assert d.n_anchors == len(store.bundles)
    assert d.n_archetypes == 4
    assert d.n_cells == 50
    # Each anchor's surfaces match the bundle
    for i, bundle in enumerate(store.bundles):
        np.testing.assert_allclose(d.surfaces[i].numpy(), bundle.archetype_surfaces, atol=1e-6)


def test_dictionary_from_snapshot_store_raises_on_missing_archetypes() -> None:
    shots = _synthetic_shots()
    anchors = [np.datetime64("2019-06-01"), np.datetime64("2020-06-01")]
    # Build store WITHOUT archetype_fit_fn -> bundles have None archetype_surfaces
    store = build_snapshot_store_from_shots(shots, anchors)
    with pytest.raises(ValueError, match=r"no .*archetype_surfaces|run scripts"):
        ArchetypeDictionary.from_snapshot_store(store)


# ---------------------------------------------------------------------------
# ArchetypeMixture: construction and forward
# ---------------------------------------------------------------------------


def test_mixture_construction_shapes() -> None:
    m = ArchetypeMixture(n_archetypes=8)
    assert m.bias.shape == (8,)
    assert m.role_weight.shape == (8, ROLE_PROFILE_DIM)
    assert m.context_weight.shape == (8, CONTEXT_DIM)


def test_mixture_default_k_is_8() -> None:
    assert DEFAULT_K == 8
    m = ArchetypeMixture()
    assert m.n_archetypes == DEFAULT_K


def test_mixture_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError, match="n_archetypes"):
        ArchetypeMixture(n_archetypes=0)
    with pytest.raises(ValueError, match="role_profile_dim"):
        ArchetypeMixture(n_archetypes=4, role_profile_dim=0)
    with pytest.raises(ValueError, match="context_dim"):
        ArchetypeMixture(n_archetypes=4, context_dim=-1)


def test_mixture_rejects_role_profile_dim_mismatch() -> None:
    """If role_profile_dim doesn't match FEATURE_LAYOUT['role_profile'], raise."""
    rp_width = FEATURE_LAYOUT["role_profile"].stop - FEATURE_LAYOUT["role_profile"].start
    with pytest.raises(ValueError, match=r"role_profile.*width"):
        ArchetypeMixture(role_profile_dim=rp_width + 3)


def test_mixture_forward_returns_simplex() -> None:
    m = ArchetypeMixture(n_archetypes=4)
    B = 16
    x_n = torch.randn(B, CONTEXT_DIM)
    rho = m(x_n)
    assert rho.shape == (B, 4)
    np.testing.assert_allclose(rho.sum(dim=-1).detach().numpy(), 1.0, atol=1e-5)
    assert (rho >= 0).all()


def test_mixture_at_zero_init_produces_uniform_distribution() -> None:
    """All-zero parameters → all logits 0 → uniform softmax."""
    m = ArchetypeMixture(n_archetypes=8)
    x_n = torch.randn(3, CONTEXT_DIM)
    rho = m(x_n)
    expected = torch.full_like(rho, 1.0 / 8)
    torch.testing.assert_close(rho, expected, atol=1e-6, rtol=0)


def test_mixture_forward_validates_x_n_shape() -> None:
    m = ArchetypeMixture()
    with pytest.raises(ValueError, match="x_n must be"):
        m(torch.zeros(CONTEXT_DIM))  # 1-D, not (B, D)
    with pytest.raises(ValueError, match="last dim"):
        m(torch.zeros(2, CONTEXT_DIM + 5))


def test_mixture_role_signature_is_separately_inspectable() -> None:
    """The semi-structured form's interpretability claim: a_k is a
    distinct learnable parameter, not buried inside d_k. This test
    locks in that structural property."""
    m = ArchetypeMixture(n_archetypes=4)
    # role_weight is an nn.Parameter of shape (K, ROLE_PROFILE_DIM)
    assert m.role_weight.requires_grad
    assert m.role_weight.shape == (4, ROLE_PROFILE_DIM)
    # We can read it directly without running forward.
    a_k_for_archetype_2 = m.role_weight[2]
    assert a_k_for_archetype_2.shape == (ROLE_PROFILE_DIM,)


def test_mixture_gradient_flows_through_all_parameters() -> None:
    m = ArchetypeMixture(n_archetypes=4)
    # Random init with non-zero parameters so gradients are non-trivial.
    with torch.no_grad():
        m.bias.fill_(0.1)
        m.role_weight.normal_(0.0, 0.1)
        m.context_weight.normal_(0.0, 0.1)
    x_n = torch.randn(8, CONTEXT_DIM)
    rho = m(x_n)
    # Loss = -log rho_0 (encourages archetype 0 mass).
    loss = -torch.log(rho[:, 0] + 1e-8).sum()
    loss.backward()
    assert m.bias.grad is not None and m.bias.grad.abs().sum() > 0
    assert m.role_weight.grad is not None and m.role_weight.grad.abs().sum() > 0
    assert m.context_weight.grad is not None and m.context_weight.grad.abs().sum() > 0


def test_mixture_explicit_r_p_overrides_slice() -> None:
    """Passing ``r_p`` explicitly bypasses the slice-from-x_n convention.

    Locks in the API used by the f_ctx pipeline: when the caller has
    transformed ``x_n`` via :class:`ContextMLP`, the named-slice layout
    is no longer guaranteed and ``r_p`` must be supplied separately.
    """
    torch.manual_seed(0)
    m = ArchetypeMixture(n_archetypes=4)
    with torch.no_grad():
        m.role_weight.normal_(0.0, 0.5)  # break r_p invariance
        m.context_weight.normal_(0.0, 0.5)

    B = 8
    x_n = torch.randn(B, CONTEXT_DIM)
    # Build an explicit r_p that does NOT match x_n's slice.
    r_p_explicit = torch.zeros(B, ROLE_PROFILE_DIM)

    rho_default = m(x_n)  # reads r_p from slice
    rho_explicit = m(x_n, r_p=r_p_explicit)  # uses zeros instead

    # When r_p is explicit and zero, the role-weight contribution
    # vanishes; logits = bias + x_n @ d^T. So rho_explicit differs
    # from rho_default unless the slice happens to be exactly zero.
    assert not torch.allclose(rho_default, rho_explicit, atol=1e-4)

    # Sanity: rho_explicit still sums to 1 per row.
    np.testing.assert_allclose(rho_explicit.sum(dim=-1).detach().numpy(), 1.0, atol=1e-5)


def test_mixture_explicit_r_p_matches_auto_extracted() -> None:
    """Passing the auto-extracted ``r_p`` explicitly is bit-identical to
    leaving it ``None``: confirms the default branch and the explicit
    branch share their math through a single ``r_p`` channel."""
    torch.manual_seed(0)
    m = ArchetypeMixture(n_archetypes=4)
    with torch.no_grad():
        m.role_weight.normal_(0.0, 0.5)
        m.context_weight.normal_(0.0, 0.5)

    x_n = torch.randn(8, CONTEXT_DIM)
    rp_slice = FEATURE_LAYOUT["role_profile"]
    r_p_extracted = x_n[:, rp_slice]

    rho_default = m(x_n)
    rho_explicit = m(x_n, r_p=r_p_extracted)

    torch.testing.assert_close(rho_default, rho_explicit, atol=0, rtol=0)


def test_mixture_validates_explicit_r_p_shape() -> None:
    m = ArchetypeMixture(n_archetypes=4)
    x_n = torch.randn(4, CONTEXT_DIM)
    with pytest.raises(ValueError, match=r"r_p must be"):
        m(x_n, r_p=torch.zeros(ROLE_PROFILE_DIM))  # 1-D
    with pytest.raises(ValueError, match=r"r_p has last dim"):
        m(x_n, r_p=torch.zeros(4, ROLE_PROFILE_DIM + 2))
    with pytest.raises(ValueError, match=r"r_p batch dim"):
        m(x_n, r_p=torch.zeros(7, ROLE_PROFILE_DIM))


def test_mixture_composes_with_context_mlp_residual_init() -> None:
    """End-to-end: ContextMLP at init is identity, so rho via the
    f_ctx pipeline equals rho via raw \\tilde x_n exactly."""
    from shotcloud import ContextMLP

    torch.manual_seed(0)
    m = ArchetypeMixture(n_archetypes=4)
    with torch.no_grad():
        m.role_weight.normal_(0.0, 0.3)
        m.context_weight.normal_(0.0, 0.3)
        m.bias.normal_(0.0, 0.1)

    f_ctx = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    f_ctx.eval()  # disable any potential dropout

    x_tilde = torch.randn(8, CONTEXT_DIM)
    r_p = x_tilde[:, FEATURE_LAYOUT["role_profile"]]

    # Path 1: raw \tilde x_n (no f_ctx).
    rho_raw = m(x_tilde)

    # Path 2: f_ctx(x_tilde) + explicit r_p.
    with torch.no_grad():
        x_n = f_ctx(x_tilde)
    rho_via_f_ctx = m(x_n, r_p=r_p)

    # f_ctx is identity at init → rho should match exactly.
    torch.testing.assert_close(rho_raw, rho_via_f_ctx, atol=1e-6, rtol=0)


def test_mixture_initialize_from_pretrained() -> None:
    K = 6
    m = ArchetypeMixture(n_archetypes=K)
    bias = np.random.default_rng(0).standard_normal(K).astype(np.float32)
    rw = np.random.default_rng(1).standard_normal((K, ROLE_PROFILE_DIM)).astype(np.float32)
    cw = np.random.default_rng(2).standard_normal((K, CONTEXT_DIM)).astype(np.float32)
    m.initialize_from_pretrained(bias, rw, cw)
    np.testing.assert_array_equal(m.bias.detach().numpy(), bias)
    np.testing.assert_array_equal(m.role_weight.detach().numpy(), rw)
    np.testing.assert_array_equal(m.context_weight.detach().numpy(), cw)


# ---------------------------------------------------------------------------
# End-to-end: dictionary + mixture composing q_arch
# ---------------------------------------------------------------------------


def test_end_to_end_q_arch_pipeline() -> None:
    """SnapshotStore → ArchetypeDictionary → ArchetypeMixture → q_arch.

    Validates the full archetype-prior inference path: get bundles from store,
    stack into a dictionary, compute mixture from x_n, combine.
    """
    shots = _synthetic_shots()
    store = _build_store_with_archetypes(shots, K=4, n_cells=50)
    arch_dict = ArchetypeDictionary.from_snapshot_store(store)
    mixture_net = ArchetypeMixture(n_archetypes=4)

    # Random batch of x_n vectors
    B = 12
    x_n = torch.randn(B, CONTEXT_DIM)
    snapshot_idx = torch.randint(0, len(store), (B,))

    mixture = mixture_net(x_n)
    q_arch = arch_dict(snapshot_idx, mixture)

    # q_arch is (B, n_cells), each row a probability distribution.
    assert q_arch.shape == (B, 50)
    np.testing.assert_allclose(q_arch.sum(dim=-1).detach().numpy(), 1.0, atol=1e-5)
    assert (q_arch >= 0).all()


def test_end_to_end_with_uniform_fallback() -> None:
    """When no pretraining is available, the uniform fallback still
    produces a valid q_arch."""
    arch_dict = ArchetypeDictionary.uniform(n_anchors=2, n_archetypes=4, n_cells=50)
    mixture_net = ArchetypeMixture(n_archetypes=4)
    x_n = torch.randn(8, CONTEXT_DIM)
    snapshot_idx = torch.randint(0, 2, (8,))
    mixture = mixture_net(x_n)
    q_arch = arch_dict(snapshot_idx, mixture)
    # q_arch is uniform (since all archetypes are uniform) → 1/n_cells everywhere.
    expected = torch.full_like(q_arch, 1.0 / 50)
    torch.testing.assert_close(q_arch, expected, atol=1e-6, rtol=0)


def test_dictionary_does_not_accumulate_gradients_in_joint_training() -> None:
    """The frozen-after-pretrain guarantee: even when the mixture
    network is trained against a loss involving q_arch, the
    dictionary surfaces never receive gradients."""
    shots = _synthetic_shots()
    store = _build_store_with_archetypes(shots, K=4, n_cells=50)
    arch_dict = ArchetypeDictionary.from_snapshot_store(store)
    mixture_net = ArchetypeMixture(n_archetypes=4)
    with torch.no_grad():
        mixture_net.role_weight.normal_(0.0, 0.1)
    x_n = torch.randn(4, CONTEXT_DIM)
    snapshot_idx = torch.zeros(4, dtype=torch.long)
    q_arch = arch_dict(snapshot_idx, mixture_net(x_n))
    loss = -torch.log(q_arch[:, 0] + 1e-8).sum()
    loss.backward()
    # No parameter on the dictionary, so nothing to check directly —
    # but mixture_net params should have grads.
    assert mixture_net.role_weight.grad is not None
    assert mixture_net.role_weight.grad.abs().sum() > 0
    # And the buffer is not in named_parameters().
    assert "surfaces" not in dict(arch_dict.named_parameters())
