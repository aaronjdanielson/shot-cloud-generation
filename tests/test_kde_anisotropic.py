"""Tests for :class:`shotcloud.kde.AnisotropicKernelEvaluator`.

Load-bearing invariants verified here:

1. **Zero-init equivalence to isotropic.** At step 0 (with output
   layers zero-init and ``a_0`` set so ``σ = init_sigma``), the
   anisotropic kernel reduces to a fixed isotropic Gaussian of
   bandwidth ``init_sigma`` for both ``σ_∥`` and ``σ_⊥``.
2. **Per-shot normalization.** Each per-shot kernel row sums to 1
   over cells (it's softmax-normalized in log-space).
3. **Mask handling.** Padded rows produce all-zero kernel outputs.
4. **Form ablation.** The three forms ``z_only / additive /
   factored`` differ only in which parameters are present; at
   step 0 all three produce the same kernel.
5. **Gradient flow.** Backward touches the bias, both marginal MLPs,
   and the bilinear (in factored mode).
6. **σ stays bounded.** Trained widths are always in [σ_min, σ_max]
   regardless of how the parameters drift.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shotcloud import AnisotropicKernelEvaluator, CourtGrid
from shotcloud.data import CONTEXT_DIM


def _make_grid() -> CourtGrid:
    return CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)


def _random_inputs(
    grid: CourtGrid, B: int = 3, max_N: int = 5, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper: random ``(x_n, z_j, s_j, u_j)`` quad for a B×max_N batch."""
    rng = np.random.default_rng(seed)
    x_n = torch.from_numpy(rng.standard_normal((B, CONTEXT_DIM)).astype(np.float32))
    z_j = torch.from_numpy(rng.standard_normal((B, max_N, CONTEXT_DIM)).astype(np.float32))
    # Shot coordinates inside the court extent, away from origin.
    s_j_np = rng.uniform(low=[-10, 0], high=[10, 25], size=(B, max_N, 2)).astype(np.float32)
    s_j = torch.from_numpy(s_j_np)
    # Rim-radial unit vectors from rim at (0, 0).
    norms = np.linalg.norm(s_j_np, axis=-1, keepdims=True)
    u_j_np = s_j_np / np.maximum(norms, 1e-6)
    u_j = torch.from_numpy(u_j_np.astype(np.float32))
    return x_n, z_j, s_j, u_j


# ---------------------------------------------------------------------------
# Construction / configuration validation
# ---------------------------------------------------------------------------


def test_construction_rejects_invalid_kernel_form() -> None:
    g = _make_grid()
    with pytest.raises(ValueError, match="kernel_form"):
        AnisotropicKernelEvaluator(g, kernel_form="bogus")


def test_construction_rejects_invalid_sigma_bounds() -> None:
    g = _make_grid()
    with pytest.raises(ValueError, match="sigma_min"):
        AnisotropicKernelEvaluator(g, sigma_min=2.0, sigma_max=1.0)
    with pytest.raises(ValueError, match="init_sigma"):
        AnisotropicKernelEvaluator(g, sigma_min=0.5, sigma_max=2.0, init_sigma=5.0)


def test_construction_rejects_negative_rank_in_factored() -> None:
    g = _make_grid()
    with pytest.raises(ValueError, match="rank"):
        AnisotropicKernelEvaluator(g, kernel_form="factored", rank=0)


# ---------------------------------------------------------------------------
# Step-0 equivalence to isotropic Gaussian
# ---------------------------------------------------------------------------


def test_step0_widths_equal_init_sigma_for_all_forms() -> None:
    """With zero-init output layers and ``a_0`` set from ``init_sigma``,
    every shot's ``σ_∥ = σ_⊥ = init_sigma`` exactly."""
    g = _make_grid()
    for form in ("z_only", "additive", "factored"):
        m = AnisotropicKernelEvaluator(
            g, kernel_form=form, sigma_min=0.5, sigma_max=4.0, init_sigma=1.5
        )
        x_n, z_j, _s, _u = _random_inputs(g)
        sigma_par, sigma_perp = m._compute_sigmas(x_n, z_j)
        torch.testing.assert_close(sigma_par, torch.full_like(sigma_par, 1.5), atol=1e-5, rtol=0)
        torch.testing.assert_close(sigma_perp, torch.full_like(sigma_perp, 1.5), atol=1e-5, rtol=0)


def test_step0_kernel_matches_isotropic_gaussian_on_grid() -> None:
    """At init, the anisotropic kernel for a single shot at known
    location should equal the (discretized + softmax-normalized)
    isotropic Gaussian centered there."""
    g = _make_grid()
    init_sigma = 1.5
    m = AnisotropicKernelEvaluator(
        g, kernel_form="factored", sigma_min=0.5, sigma_max=4.0, init_sigma=init_sigma
    )

    # Single shot at a clear, off-origin location.
    s = np.array([5.0, 10.0], dtype=np.float32)
    s_j = torch.from_numpy(s[None, None, :])  # (1, 1, 2)
    norm = float(np.linalg.norm(s))
    u_j = torch.from_numpy((s / norm)[None, None, :].astype(np.float32))
    x_n = torch.zeros(1, CONTEXT_DIM, dtype=torch.float32)
    z_j = torch.zeros(1, 1, CONTEXT_DIM, dtype=torch.float32)

    with torch.no_grad():
        K = m(x_n, z_j, s_j, u_j)
    K_row = K[0, 0]
    assert K_row.shape == (g.n_cells,)
    torch.testing.assert_close(K_row.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)

    # Compare to ground-truth isotropic Gaussian on the grid
    # (softmax-normalized, since K is normalized in log-space).
    centers_x = np.tile(g.xcenters, len(g.ycenters))
    centers_y = np.repeat(g.ycenters, len(g.xcenters))
    dx = centers_x - s[0]
    dy = centers_y - s[1]
    log_K_truth_np = -0.5 * (dx**2 + dy**2) / (init_sigma**2)
    log_K_truth = torch.from_numpy(log_K_truth_np).float()
    K_truth = torch.log_softmax(log_K_truth, dim=-1).exp()
    torch.testing.assert_close(K_row.double(), K_truth.double(), atol=1e-5, rtol=0)


def test_step0_kernel_is_form_independent() -> None:
    """All three kernel forms produce identical output at step 0 (no
    parameters are non-zero except the shared ``a_0``)."""
    g = _make_grid()
    x_n, z_j, s_j, u_j = _random_inputs(g)
    common = dict(sigma_min=0.5, sigma_max=4.0, init_sigma=1.5)
    Ks = {}
    with torch.no_grad():
        for form in ("z_only", "additive", "factored"):
            m = AnisotropicKernelEvaluator(g, kernel_form=form, **common)
            Ks[form] = m(x_n, z_j, s_j, u_j)
    torch.testing.assert_close(Ks["z_only"], Ks["additive"], atol=1e-6, rtol=0)
    torch.testing.assert_close(Ks["z_only"], Ks["factored"], atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Normalization + masking contracts
# ---------------------------------------------------------------------------


def test_per_shot_kernel_sums_to_one() -> None:
    g = _make_grid()
    m = AnisotropicKernelEvaluator(g, kernel_form="factored")
    x_n, z_j, s_j, u_j = _random_inputs(g, B=4, max_N=6)
    with torch.no_grad():
        K = m(x_n, z_j, s_j, u_j)
    row_sums = K.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=0)


def test_mask_zeros_padded_rows() -> None:
    g = _make_grid()
    m = AnisotropicKernelEvaluator(g, kernel_form="factored")
    x_n, z_j, s_j, u_j = _random_inputs(g, B=2, max_N=5)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        K = m(x_n, z_j, s_j, u_j, mask=mask)
    # Padded rows must be all-zero kernels.
    assert (K[0, 2:] == 0).all()
    assert (K[1, 1:] == 0).all()
    # Unmasked rows still sum to 1.
    torch.testing.assert_close(K[0, :2].sum(dim=-1), torch.ones(2), atol=1e-5, rtol=0)
    torch.testing.assert_close(K[1, :1].sum(dim=-1), torch.ones(1), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Form-specific structural checks
# ---------------------------------------------------------------------------


def test_z_only_form_has_no_x_or_bilinear_params() -> None:
    g = _make_grid()
    m = AnisotropicKernelEvaluator(g, kernel_form="z_only")
    assert m.mlp_x is None
    assert m.bilinear_W is None
    assert m.bilinear_V is None


def test_additive_form_has_x_marginal_but_no_bilinear() -> None:
    g = _make_grid()
    m = AnisotropicKernelEvaluator(g, kernel_form="additive")
    assert m.mlp_x is not None
    assert m.bilinear_W is None
    assert m.bilinear_V is None


def test_factored_form_has_all_components() -> None:
    g = _make_grid()
    m = AnisotropicKernelEvaluator(g, kernel_form="factored", rank=4)
    assert m.mlp_x is not None
    assert m.bilinear_W is not None and m.bilinear_W.shape == (2, 4, CONTEXT_DIM)
    assert m.bilinear_V is not None and m.bilinear_V.shape == (2, 4, CONTEXT_DIM)


# ---------------------------------------------------------------------------
# Adaptivity check: non-zero parameters → non-isotropic σ
# ---------------------------------------------------------------------------


def test_perturbing_mlp_z_breaks_isotropy() -> None:
    """If we manually push σ_∥ params away from σ_⊥ params, σ_∥ ≠ σ_⊥."""
    g = _make_grid()
    m = AnisotropicKernelEvaluator(
        g, kernel_form="z_only", sigma_min=0.5, sigma_max=4.0, init_sigma=1.5
    )
    # The final layer of mlp_z is (hidden, 2); column 0 → σ_∥, column 1 → σ_⊥.
    # Perturb only the σ_∥ output to break symmetry.
    with torch.no_grad():
        m.mlp_z[-1].weight[0].fill_(0.5)
    x_n, z_j, _s, _u = _random_inputs(g)
    sigma_par, sigma_perp = m._compute_sigmas(x_n, z_j)
    assert (sigma_par - sigma_perp).abs().max() > 0.05


def test_perturbing_x_marginal_changes_sigma_with_context() -> None:
    """In additive/factored, σ should respond to changes in x_n."""
    g = _make_grid()
    m = AnisotropicKernelEvaluator(
        g, kernel_form="additive", sigma_min=0.5, sigma_max=4.0, init_sigma=1.5
    )
    with torch.no_grad():
        m.mlp_x[-1].weight[0].fill_(0.5)  # type: ignore[index]
    _, z_j, _s, _u = _random_inputs(g)
    x_a = torch.zeros(z_j.shape[0], CONTEXT_DIM, dtype=torch.float32)
    x_b = torch.ones(z_j.shape[0], CONTEXT_DIM, dtype=torch.float32)
    sigma_a, _ = m._compute_sigmas(x_a, z_j)
    sigma_b, _ = m._compute_sigmas(x_b, z_j)
    assert (sigma_a - sigma_b).abs().mean() > 0.01


def test_perturbing_bilinear_creates_x_z_interaction() -> None:
    """Only the bilinear term can create x × z interaction. With the
    marginals zero-init, any change in σ across different x at fixed z
    must come from the bilinear."""
    g = _make_grid()
    m = AnisotropicKernelEvaluator(
        g, kernel_form="factored", rank=2, sigma_min=0.5, sigma_max=4.0, init_sigma=1.5
    )
    with torch.no_grad():
        m.bilinear_W[0, 0].fill_(0.5)  # type: ignore[index]
        m.bilinear_V[0, 0].fill_(0.5)  # type: ignore[index]
    _, z_j, _s, _u = _random_inputs(g)
    x_a = torch.zeros(z_j.shape[0], CONTEXT_DIM, dtype=torch.float32)
    x_b = torch.ones(z_j.shape[0], CONTEXT_DIM, dtype=torch.float32)
    sigma_a, _ = m._compute_sigmas(x_a, z_j)
    sigma_b, _ = m._compute_sigmas(x_b, z_j)
    # Different x → different σ_∥ at fixed z.
    assert (sigma_a - sigma_b).abs().mean() > 0.01


# ---------------------------------------------------------------------------
# Gradient flow + boundedness
# ---------------------------------------------------------------------------


def test_gradient_flows_to_all_factored_params() -> None:
    g = _make_grid()
    m = AnisotropicKernelEvaluator(g, kernel_form="factored", rank=2)
    x_n, z_j, s_j, u_j = _random_inputs(g, B=2, max_N=3)
    K = m(x_n, z_j, s_j, u_j)
    loss = K.pow(2).sum()
    loss.backward()
    assert m.a_0.grad is not None and m.a_0.grad.abs().sum() > 0
    assert m.mlp_z[-1].weight.grad is not None and m.mlp_z[-1].weight.grad.abs().sum() > 0
    assert m.mlp_x is not None
    assert m.mlp_x[-1].weight.grad is not None and m.mlp_x[-1].weight.grad.abs().sum() > 0
    # Bilinear: gradient depends on its current value being nonzero;
    # at zero-init the gradient through the product is zero. Verify that
    # gradient exists (grad tensor is created) — exact magnitude depends
    # on init.
    assert m.bilinear_W is not None and m.bilinear_W.grad is not None
    assert m.bilinear_V is not None and m.bilinear_V.grad is not None


def test_sigmas_remain_bounded_after_extreme_param_drift() -> None:
    g = _make_grid()
    sigma_min, sigma_max = 0.75, 4.0
    m = AnisotropicKernelEvaluator(
        g, kernel_form="factored", sigma_min=sigma_min, sigma_max=sigma_max
    )
    # Drive every parameter to large magnitudes.
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(10.0)
    x_n, z_j, _, _ = _random_inputs(g, B=3, max_N=5)
    sigma_par, sigma_perp = m._compute_sigmas(x_n, z_j)
    assert (sigma_par >= sigma_min - 1e-6).all() and (sigma_par <= sigma_max + 1e-6).all()
    assert (sigma_perp >= sigma_min - 1e-6).all() and (sigma_perp <= sigma_max + 1e-6).all()
    # Now drive to large negatives.
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(-10.0)
    sigma_par, sigma_perp = m._compute_sigmas(x_n, z_j)
    assert (sigma_par >= sigma_min - 1e-6).all() and (sigma_par <= sigma_max + 1e-6).all()
    assert (sigma_perp >= sigma_min - 1e-6).all() and (sigma_perp <= sigma_max + 1e-6).all()


# ---------------------------------------------------------------------------
# Chunking equivalence
# ---------------------------------------------------------------------------


def test_chunking_along_max_n_is_equivalent_to_full_eval() -> None:
    """Splitting max_N into chunks gives the same output as evaluating
    the full max_N in one shot. Memory-control mechanism must be a
    no-op on results."""
    g = _make_grid()
    x_n, z_j, s_j, u_j = _random_inputs(g, B=2, max_N=10)
    # Force varying widths so the comparison is non-trivial.
    m_full = AnisotropicKernelEvaluator(g, kernel_form="factored", max_n_chunk=100)
    m_chunked = AnisotropicKernelEvaluator(g, kernel_form="factored", max_n_chunk=3)
    # Use the same parameters in both.
    m_chunked.load_state_dict(m_full.state_dict())
    with torch.no_grad():
        # Perturb to make σ vary.
        m_full.mlp_z[-1].weight.fill_(0.3)
        m_chunked.mlp_z[-1].weight.fill_(0.3)
        K_full = m_full(x_n, z_j, s_j, u_j)
        K_chunked = m_chunked(x_n, z_j, s_j, u_j)
    torch.testing.assert_close(K_full, K_chunked, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------


def test_sigma_stats_returns_expected_keys() -> None:
    g = _make_grid()
    m = AnisotropicKernelEvaluator(g, kernel_form="factored")
    x_n, z_j, _, _ = _random_inputs(g)
    stats = m.sigma_stats(x_n, z_j)
    assert set(stats.keys()) == {
        "sigma_par_mean",
        "sigma_par_std",
        "sigma_perp_mean",
        "sigma_perp_std",
        "sigma_anisotropy",
    }
    # At step 0, σ_∥ = σ_⊥ = init_sigma → anisotropy = 0, both stds = 0.
    assert stats["sigma_anisotropy"] < 1e-5
    assert stats["sigma_par_std"] < 1e-5
    assert stats["sigma_perp_std"] < 1e-5


# ---------------------------------------------------------------------------
# Integration with the grid-cell AdaptiveOffensivePrior (shotcloud.legacy_pivot)
# ---------------------------------------------------------------------------


def test_anisotropic_prior_matches_isotropic_at_step_zero() -> None:
    """At step 0 the anisotropic prior is within small TV distance of the isotropic prior.

    With ``init_sigma`` equal to ``AdaptiveKDE.bandwidth``, the anisotropic
    path matches the precomputed isotropic kernel matrix ``M`` up to tail
    handling: ``M`` is built with ``scipy.ndimage.gaussian_filter`` and L1
    normalization, the anisotropic kernel with a softmax over cells, so the
    two are close but not bit-identical.
    """
    import pandas as pd

    from shotcloud import AdaptiveKDE, PlayerVocab, RelevanceScore
    from shotcloud.data import ContextEncoder
    from shotcloud.data.role_profile import build_role_profiles
    from shotcloud.data.snapshots import build_snapshot_store_from_shots
    from shotcloud.legacy_pivot.adaptive_prior import AdaptiveOffensivePrior
    from shotcloud.legacy_pivot.archetypes import ArchetypeDictionary, ArchetypeMixture

    g = _make_grid()
    rng = np.random.default_rng(0)
    rows = []
    for pid in (1, 2, 3):
        for i in range(30):
            rows.append(
                {
                    "x": float(rng.normal(0, 5)),
                    "y": float(rng.normal(15, 5)),
                    "player_id": pid,
                    "opponent": "BOS",
                    "made": 0,
                    "period": (i % 4) + 1,
                    "time_remaining_sec": int(60 * (i % 48)),
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                }
            )
    df = pd.DataFrame(rows)
    anchor = np.datetime64("2024-04-01", "D")
    store = build_snapshot_store_from_shots(
        df,
        [anchor],
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
        archetype_fit_fn=lambda sub, _t: np.full((4, g.n_cells), 1.0 / g.n_cells, dtype=np.float32),
    )
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    akde = AdaptiveKDE(grid=g, bandwidth=1.5, max_history=30)
    akde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    vocab = PlayerVocab.from_ids(akde.players)
    ad = ArchetypeDictionary.from_snapshot_store(store)
    am = ArchetypeMixture(n_archetypes=4)

    # Isotropic prior.
    iso = AdaptiveOffensivePrior(akde, store, ad, am, vocab, RelevanceScore(), kappa=20.0)
    # Anisotropic prior with init_sigma = bandwidth so step-0 kernels
    # match the precomputed isotropic M.
    aniso_kernel = AnisotropicKernelEvaluator(
        g, kernel_form="factored", sigma_min=0.5, sigma_max=4.0, init_sigma=1.5
    )
    aniso = AdaptiveOffensivePrior(
        akde,
        store,
        ad,
        am,
        vocab,
        RelevanceScore(),
        kappa=20.0,
        anisotropic_kernel=aniso_kernel,
    )

    B = 3
    x_n = torch.from_numpy(ctx[:B]).float()
    player_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        log_q_iso, _, _ = iso(player_idx, snapshot_idx, x_n, x_n)
        log_q_aniso, _, _ = aniso(player_idx, snapshot_idx, x_n, x_n)

    # Both should be normalized densities.
    np.testing.assert_allclose(torch.exp(log_q_iso).sum(dim=-1).numpy(), np.ones(B), atol=1e-4)
    np.testing.assert_allclose(torch.exp(log_q_aniso).sum(dim=-1).numpy(), np.ones(B), atol=1e-4)
    # Total-variation between the two paths should be small (the
    # discretized softmax-Gaussian and the gaussian_filter-precomputed
    # kernels differ in tail handling but agree to within a few percent).
    tv = 0.5 * (torch.exp(log_q_iso) - torch.exp(log_q_aniso)).abs().sum(dim=-1)
    assert (tv < 0.15).all(), f"TV between isotropic and step-0 anisotropic: {tv.tolist()}"


def test_anisotropic_prior_construction_requires_coords() -> None:
    """An ``AdaptiveKDE`` without stored shot coordinates is rejected at construction."""
    import pandas as pd

    from shotcloud import AdaptiveKDE, PlayerVocab, RelevanceScore
    from shotcloud.data import ContextEncoder
    from shotcloud.data.role_profile import build_role_profiles
    from shotcloud.data.snapshots import build_snapshot_store_from_shots
    from shotcloud.legacy_pivot.adaptive_prior import AdaptiveOffensivePrior
    from shotcloud.legacy_pivot.archetypes import ArchetypeDictionary, ArchetypeMixture

    g = _make_grid()
    rng = np.random.default_rng(0)
    rows = [
        {
            "x": float(rng.normal(0, 5)),
            "y": float(rng.normal(15, 5)),
            "player_id": 1,
            "opponent": "BOS",
            "made": 0,
            "period": 1,
            "time_remaining_sec": 60,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
        }
        for i in range(20)
    ]
    df = pd.DataFrame(rows)
    anchor = np.datetime64("2024-04-01", "D")
    store = build_snapshot_store_from_shots(
        df,
        [anchor],
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
        archetype_fit_fn=lambda sub, _t: np.full((4, g.n_cells), 1.0 / g.n_cells, dtype=np.float32),
    )
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    akde = AdaptiveKDE(grid=g, bandwidth=1.5, max_history=20)
    akde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    # Remove the stored coordinates the anisotropic kernel needs.
    akde.coords = {}
    vocab = PlayerVocab.from_ids(akde.players)
    ad = ArchetypeDictionary.from_snapshot_store(store)
    am = ArchetypeMixture(n_archetypes=4)
    kernel = AnisotropicKernelEvaluator(g, kernel_form="z_only")
    with pytest.raises(ValueError, match="coords"):
        AdaptiveOffensivePrior(
            akde, store, ad, am, vocab, RelevanceScore(), anisotropic_kernel=kernel
        )
