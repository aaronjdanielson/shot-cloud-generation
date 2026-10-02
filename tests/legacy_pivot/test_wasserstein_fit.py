"""Tests for :mod:`shotcloud.wasserstein_fit`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shotcloud.legacy_pivot.wasserstein_fit import (
    ArchetypeFitResult,
    apply_separable_log_kernel,
    build_separable_log_kernel,
    fit_archetypes_v1,
    fit_v2_rho_given_A,
    sinkhorn_distance_separable,
    sinkhorn_divergence_separable,
    wasserstein_barycenter_separable,
)

# ---------------------------------------------------------------------------
# Separable log kernel
# ---------------------------------------------------------------------------


def test_log_kernel_shapes_and_diagonal_zero() -> None:
    log_Ky, log_Kx = build_separable_log_kernel(grid_ny=8, grid_nx=10, epsilon=1.0)
    assert log_Ky.shape == (8, 8)
    assert log_Kx.shape == (10, 10)
    # Diagonal cost is zero → log kernel diagonal is zero.
    assert torch.allclose(log_Ky.diag(), torch.zeros(8))
    # Symmetric.
    assert torch.allclose(log_Ky, log_Ky.t())
    # Off-diagonal entries are negative (cost > 0).
    assert (log_Ky[0, 1:] < 0).all()


def test_log_kernel_invalid_dims_raise() -> None:
    with pytest.raises(ValueError, match="grid dims"):
        build_separable_log_kernel(grid_ny=0, grid_nx=10)
    with pytest.raises(ValueError, match="epsilon"):
        build_separable_log_kernel(grid_ny=8, grid_nx=10, epsilon=0)


def test_apply_separable_log_kernel_matches_dense() -> None:
    """log(K @ exp(log_v)) under the separable factorization equals
    log(dense_K @ exp(log_v)) up to float roundoff."""
    torch.manual_seed(0)
    ny, nx = 6, 7
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=1.5)
    Ky = log_Ky.exp()
    Kx = log_Kx.exp()

    # Dense K = Ky ⊗ Kx (image-major reshape).
    iy_grid = torch.arange(ny).unsqueeze(1).repeat(1, nx).reshape(-1)
    ix_grid = torch.arange(nx).unsqueeze(0).repeat(ny, 1).reshape(-1)
    K_dense = (
        Ky[iy_grid.unsqueeze(1), iy_grid.unsqueeze(0)]
        * Kx[ix_grid.unsqueeze(1), ix_grid.unsqueeze(0)]
    )

    log_v = torch.randn(3, ny, nx) * 0.5
    out_log = apply_separable_log_kernel(log_v, log_Ky, log_Kx)
    out_dense_log = (K_dense @ log_v.exp().view(3, -1).t()).t().view(3, ny, nx).log()
    torch.testing.assert_close(out_log, out_dense_log, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Sinkhorn distance / divergence
# ---------------------------------------------------------------------------


def _uniform_simplex(B: int, ny: int, nx: int, seed: int = 0) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    a = torch.rand(B, ny, nx, generator=rng) + 0.01
    a = a / a.sum(dim=(-2, -1), keepdim=True)
    return a


def test_sinkhorn_distance_shape_and_finiteness() -> None:
    ny, nx = 6, 8
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=1.0)
    a = _uniform_simplex(4, ny, nx)
    cost = sinkhorn_distance_separable(a, a, log_Ky, log_Kx, epsilon=1.0, n_iter=20)
    assert cost.shape == (4,)
    assert torch.isfinite(cost).all()


def test_sinkhorn_divergence_is_zero_for_self() -> None:
    """The debiased Sinkhorn divergence equals zero (up to numerics)
    when ``a == b``."""
    torch.manual_seed(0)
    ny, nx = 6, 8
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=1.0)
    a = _uniform_simplex(3, ny, nx)
    s = sinkhorn_divergence_separable(a, a, log_Ky, log_Kx, epsilon=1.0, n_iter=40)
    # Float32 + 40 Sinkhorn iters → tolerance ~1e-3.
    assert s.abs().max().item() < 1e-3


def test_sinkhorn_divergence_positive_for_separated_distributions() -> None:
    """``S_eps(a, b) > 0`` when a and b are spatially separated deltas."""
    ny, nx = 6, 8
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=0.5)

    a = torch.zeros(2, ny, nx)
    b = torch.zeros(2, ny, nx)
    a[0, 1, 1] = 1.0
    a[1, 4, 5] = 1.0
    b[0, 5, 7] = 1.0
    b[1, 0, 0] = 1.0
    s = sinkhorn_divergence_separable(a, b, log_Ky, log_Kx, epsilon=0.5, n_iter=80)
    assert (s > 0).all()


def test_sinkhorn_distance_rejects_wrong_shape() -> None:
    log_Ky, log_Kx = build_separable_log_kernel(grid_ny=4, grid_nx=4, epsilon=1.0)
    a = torch.ones(2, 5, 5) / 25
    b = torch.ones(2, 5, 5) / 25
    with pytest.raises(ValueError, match="must be"):
        sinkhorn_distance_separable(a, b, log_Ky, log_Kx, epsilon=1.0)


# ---------------------------------------------------------------------------
# Numerical validation: separable Sinkhorn vs brute-force dense Sinkhorn
# ---------------------------------------------------------------------------


def _dense_sinkhorn_distance_brute_force(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float,
    n_iter: int,
    eps_safe: float = 1e-30,
) -> torch.Tensor:
    """Reference Sinkhorn distance via the dense kernel + log-domain
    iteration. Matches the closed-form regularized OT cost
    ``W_eps(a, b) = epsilon * (<f, a> + <g, b>)`` after convergence,
    where ``f, g`` are the log-domain dual potentials.

    Inputs:
        a, b: (B, n) probability distributions on n cells.
        cost: (n, n) symmetric ground cost matrix.

    Used in tests only; not optimized for size.
    """
    log_K = -cost / epsilon  # (n, n)
    log_a = a.clamp_min(eps_safe).log()  # (B, n)
    log_b = b.clamp_min(eps_safe).log()
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(n_iter):
        # log(K @ exp(g)) = logsumexp_j(log_K[:, j] + g[..., j])
        # broadcast: log_K (n, n) + g (B, 1, n) -> (B, n, n) -> logsumexp dim=-1
        f = log_a - torch.logsumexp(log_K + g.unsqueeze(-2), dim=-1)
        g = log_b - torch.logsumexp(log_K + f.unsqueeze(-2), dim=-1)
    return epsilon * ((f * a).sum(-1) + (g * b).sum(-1))


def _build_dense_cost(
    ny: int, nx: int, cell_size_y: float = 1.0, cell_size_x: float = 1.0
) -> torch.Tensor:
    """Squared-Euclidean cost over a (ny, nx) grid, in image-major order."""
    iy = torch.arange(ny, dtype=torch.float32).unsqueeze(1).repeat(1, nx).reshape(-1)
    ix = torch.arange(nx, dtype=torch.float32).unsqueeze(0).repeat(ny, 1).reshape(-1)
    dy = (iy.unsqueeze(0) - iy.unsqueeze(1)) * cell_size_y
    dx = (ix.unsqueeze(0) - ix.unsqueeze(1)) * cell_size_x
    return dy.pow(2) + dx.pow(2)


@pytest.mark.parametrize("epsilon", [0.5, 1.0, 2.0])
def test_separable_sinkhorn_matches_dense_brute_force(epsilon: float) -> None:
    """Separable log-domain Sinkhorn produces the same cost as the
    dense brute-force reference, on random simplex pairs over a small
    grid where both implementations are tractable."""
    torch.manual_seed(0)
    ny, nx = 5, 6  # 30 cells: brute force is fine
    n_iter = 200  # plenty for convergence at these epsilons
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=epsilon)
    cost = _build_dense_cost(ny, nx)

    a = _uniform_simplex(4, ny, nx, seed=1)
    b = _uniform_simplex(4, ny, nx, seed=2)

    sep = sinkhorn_distance_separable(
        a, b, log_Ky, log_Kx, epsilon=epsilon, n_iter=n_iter, last_iter_with_grad=False
    )
    dense = _dense_sinkhorn_distance_brute_force(
        a.view(4, -1), b.view(4, -1), cost, epsilon=epsilon, n_iter=n_iter
    )
    # Both implementations should match to ~1e-3 in cost units.
    torch.testing.assert_close(sep, dense, atol=1e-3, rtol=1e-3)


def test_separable_sinkhorn_matches_dense_for_diagonal_constant_pair() -> None:
    """``W_eps(a, a)`` returned by both implementations must agree.

    This is the load-bearing case for the debiased divergence: a
    miscalibrated ``waa`` term would shift every loss value by a
    constant per row.
    """
    torch.manual_seed(0)
    ny, nx = 4, 5
    epsilon = 1.0
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=epsilon)
    cost = _build_dense_cost(ny, nx)
    a = _uniform_simplex(3, ny, nx, seed=42)
    sep = sinkhorn_distance_separable(
        a, a, log_Ky, log_Kx, epsilon=epsilon, n_iter=200, last_iter_with_grad=False
    )
    dense = _dense_sinkhorn_distance_brute_force(
        a.view(3, -1), a.view(3, -1), cost, epsilon=epsilon, n_iter=200
    )
    torch.testing.assert_close(sep, dense, atol=1e-3, rtol=1e-3)


def test_sinkhorn_divergence_matches_brute_force() -> None:
    """The debiased S_eps(a, b) = W(a,b) - 0.5 W(a,a) - 0.5 W(b,b)
    computed via the separable path equals the brute-force version
    using dense Sinkhorn for each W term."""
    torch.manual_seed(0)
    ny, nx = 4, 5
    epsilon = 1.0
    n_iter = 200
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=epsilon)
    cost = _build_dense_cost(ny, nx)

    a = _uniform_simplex(3, ny, nx, seed=11)
    b = _uniform_simplex(3, ny, nx, seed=22)

    sep_div = sinkhorn_divergence_separable(
        a, b, log_Ky, log_Kx, epsilon=epsilon, n_iter=n_iter, last_iter_with_grad=False
    )

    a_flat = a.view(3, -1)
    b_flat = b.view(3, -1)
    wab = _dense_sinkhorn_distance_brute_force(a_flat, b_flat, cost, epsilon, n_iter)
    waa = _dense_sinkhorn_distance_brute_force(a_flat, a_flat, cost, epsilon, n_iter)
    wbb = _dense_sinkhorn_distance_brute_force(b_flat, b_flat, cost, epsilon, n_iter)
    dense_div = wab - 0.5 * waa - 0.5 * wbb

    torch.testing.assert_close(sep_div, dense_div, atol=1e-3, rtol=1e-3)
    # And: divergence is non-negative (within numerical tolerance).
    assert (sep_div >= -1e-3).all()


# ---------------------------------------------------------------------------
# Fit archetypes — synthetic ground truth
# ---------------------------------------------------------------------------


def _gaussian_blob_2d(ny: int, nx: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    iy = np.arange(ny, dtype=np.float64)
    ix = np.arange(nx, dtype=np.float64)
    yy = np.exp(-((iy[:, None] - cy) ** 2) / (2 * sigma**2))
    xx = np.exp(-((ix[None, :] - cx) ** 2) / (2 * sigma**2))
    blob = (yy * xx).astype(np.float32)
    blob /= blob.sum()
    return blob.reshape(-1)


def _make_synthetic_Q(ny: int, nx: int, n_players: int, K_true: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(low=2, high=ny - 2, size=(K_true, 2))
    sigmas = rng.uniform(low=1.0, high=2.5, size=K_true)
    A_true = np.stack(
        [
            _gaussian_blob_2d(ny, nx, cy=cy, cx=cx, sigma=s)
            for (cy, cx), s in zip(centers, sigmas, strict=True)
        ]
    ).astype(np.float32)
    rho = rng.dirichlet(alpha=np.ones(K_true), size=n_players).astype(np.float32)
    Q = (rho @ A_true).astype(np.float32)
    return Q, A_true, rho


def test_fit_returns_simplex_archetypes_and_mixtures() -> None:
    ny, nx = 8, 10
    Q, _, _ = _make_synthetic_Q(ny, nx, n_players=12, K_true=3, seed=0)
    res = fit_archetypes_v1(
        Q,
        grid_ny=ny,
        grid_nx=nx,
        K=4,
        max_iter=10,
        sinkhorn_iter=10,
        seed=1,
    )
    assert isinstance(res, ArchetypeFitResult)
    assert res.archetypes.shape == (4, ny * nx)
    assert res.mixtures.shape == (12, 4)
    assert res.loss_history.shape == (10,)
    np.testing.assert_allclose(res.archetypes.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(res.mixtures.sum(axis=1), 1.0, atol=1e-5)
    assert (res.archetypes >= 0).all()
    assert (res.mixtures >= 0).all()
    assert np.isfinite(res.loss_history).all()


def test_fit_reduces_loss_over_iterations() -> None:
    """The reported loss history decreases on average from start to end."""
    ny, nx = 8, 10
    Q, _, _ = _make_synthetic_Q(ny, nx, n_players=20, K_true=3, seed=0)
    res = fit_archetypes_v1(
        Q,
        grid_ny=ny,
        grid_nx=nx,
        K=3,
        max_iter=60,
        sinkhorn_iter=20,
        seed=0,
    )
    n = len(res.loss_history)
    q = n // 4
    assert res.loss_history[:q].mean() > res.loss_history[-q:].mean(), (
        f"loss not decreasing: head={res.loss_history[:q].mean()} "
        f"tail={res.loss_history[-q:].mean()}"
    )


def test_fit_warm_start_reproduces_A_init_at_step_zero() -> None:
    """``max_iter=0`` returns the warm-start basis exactly (modulo the
    softmax-of-log roundtrip floor)."""
    ny, nx = 6, 8
    Q, A_true, _ = _make_synthetic_Q(ny, nx, n_players=8, K_true=3, seed=0)
    res = fit_archetypes_v1(
        Q,
        grid_ny=ny,
        grid_nx=nx,
        K=3,
        max_iter=0,
        sinkhorn_iter=5,
        A_init=A_true,
        seed=42,
    )
    np.testing.assert_allclose(res.archetypes, A_true, atol=1e-5)


def test_fit_supports_sample_weights() -> None:
    """Heavy weight on player 0 → that player's reconstruction error
    is smaller than under uniform weights."""
    ny, nx = 10, 10
    rng = np.random.default_rng(0)
    K_true = 3
    P = 6
    A_true = np.stack(
        [
            _gaussian_blob_2d(ny, nx, cy=2, cx=2, sigma=1.0),
            _gaussian_blob_2d(ny, nx, cy=8, cx=8, sigma=1.0),
            _gaussian_blob_2d(ny, nx, cy=2, cx=8, sigma=1.0),
        ]
    ).astype(np.float32)
    rho = rng.dirichlet(np.ones(K_true), size=P).astype(np.float32)
    Q = (rho @ A_true).astype(np.float32)

    weights = np.full(P, 0.01, dtype=np.float32)
    weights[0] = 1.0

    res_w = fit_archetypes_v1(
        Q,
        grid_ny=ny,
        grid_nx=nx,
        K=3,
        max_iter=40,
        sinkhorn_iter=20,
        sample_weights=weights,
        seed=0,
    )
    res_u = fit_archetypes_v1(
        Q,
        grid_ny=ny,
        grid_nx=nx,
        K=3,
        max_iter=40,
        sinkhorn_iter=20,
        seed=0,
    )

    Q_hat_w = res_w.mixtures[0] @ res_w.archetypes
    Q_hat_u = res_u.mixtures[0] @ res_u.archetypes
    err_w = float(np.abs(Q[0] - Q_hat_w).sum())
    err_u = float(np.abs(Q[0] - Q_hat_u).sum())
    assert err_w < err_u + 1e-3, f"weighted err={err_w}, uniform err={err_u}"


def test_fit_minibatching_does_not_break_optimization() -> None:
    """Minibatched fit converges to a similar archetype neighborhood
    as full-batch (Adam state differs across chunk-summing order so
    they aren't bit-identical)."""
    ny, nx = 6, 7
    Q, _, _ = _make_synthetic_Q(ny, nx, n_players=10, K_true=3, seed=0)
    res_full = fit_archetypes_v1(
        Q,
        grid_ny=ny,
        grid_nx=nx,
        K=3,
        max_iter=15,
        sinkhorn_iter=10,
        batch_size=None,
        seed=0,
    )
    res_chunks = fit_archetypes_v1(
        Q,
        grid_ny=ny,
        grid_nx=nx,
        K=3,
        max_iter=15,
        sinkhorn_iter=10,
        batch_size=4,
        seed=0,
    )
    diff = float(np.abs(res_full.archetypes - res_chunks.archetypes).max())
    assert diff < 0.05, f"max archetype diff: {diff}"


def test_fit_use_sinkhorn_divergence_flag() -> None:
    ny, nx = 6, 7
    Q, _, _ = _make_synthetic_Q(ny, nx, n_players=10, K_true=3, seed=0)
    res_div = fit_archetypes_v1(
        Q,
        grid_ny=ny,
        grid_nx=nx,
        K=3,
        max_iter=8,
        sinkhorn_iter=10,
        use_sinkhorn_divergence=True,
        seed=0,
    )
    res_raw = fit_archetypes_v1(
        Q,
        grid_ny=ny,
        grid_nx=nx,
        K=3,
        max_iter=8,
        sinkhorn_iter=10,
        use_sinkhorn_divergence=False,
        seed=0,
    )
    np.testing.assert_allclose(res_div.archetypes.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(res_raw.archetypes.sum(axis=1), 1.0, atol=1e-5)


def test_fit_rejects_bad_K() -> None:
    Q, _, _ = _make_synthetic_Q(6, 8, n_players=4, K_true=2, seed=0)
    with pytest.raises(ValueError, match="K must be positive"):
        fit_archetypes_v1(Q, grid_ny=6, grid_nx=8, K=0)


def test_fit_rejects_grid_size_mismatch() -> None:
    Q, _, _ = _make_synthetic_Q(6, 8, n_players=4, K_true=2, seed=0)
    with pytest.raises(ValueError, match=r"grid_ny \* grid_nx"):
        fit_archetypes_v1(Q, grid_ny=10, grid_nx=10, K=2)


def test_fit_rejects_bad_A_init_shape() -> None:
    Q, _, _ = _make_synthetic_Q(6, 8, n_players=4, K_true=2, seed=0)
    bad_init = np.ones((3, 6 * 8 + 5), dtype=np.float32)
    with pytest.raises(ValueError, match="A_init has shape"):
        fit_archetypes_v1(
            Q,
            grid_ny=6,
            grid_nx=8,
            K=3,
            A_init=bad_init,
            max_iter=1,
        )


def test_fit_rejects_bad_sample_weights_shape() -> None:
    Q, _, _ = _make_synthetic_Q(6, 8, n_players=4, K_true=2, seed=0)
    bad_weights = np.ones(99, dtype=np.float32)
    with pytest.raises(ValueError, match="sample_weights has shape"):
        fit_archetypes_v1(
            Q,
            grid_ny=6,
            grid_nx=8,
            K=2,
            sample_weights=bad_weights,
            max_iter=1,
        )


# ---------------------------------------------------------------------------
# Entropic Wasserstein barycenter (V2 reconstruction object)
# ---------------------------------------------------------------------------


def _gaussian_atom(ny: int, nx: int, cy: float, cx: float, sigma: float = 1.5) -> np.ndarray:
    """Concentrated 2D-Gaussian-like distribution on the grid, returned flat (C,)."""
    iy = np.arange(ny, dtype=np.float64)
    ix = np.arange(nx, dtype=np.float64)
    Y, X = np.meshgrid(iy, ix, indexing="ij")
    p = np.exp(-((Y - cy) ** 2 + (X - cx) ** 2) / (2 * sigma**2))
    return (p / p.sum()).flatten().astype(np.float32)


def _stack_atoms_2d(atoms_flat: list[np.ndarray], ny: int, nx: int) -> torch.Tensor:
    """Stack flat (C,) atoms into a (K, ny, nx) tensor for the barycenter helper."""
    A_flat = np.stack(atoms_flat)
    return torch.from_numpy(A_flat).clamp_min(1e-12).log().view(len(atoms_flat), ny, nx)


def test_barycenter_one_hot_recovers_atom() -> None:
    """ρ ≈ (1, 0) should produce a barycenter ≈ A_0 (entropic blur tolerance)."""
    ny, nx = 16, 16
    A_flat = [_gaussian_atom(ny, nx, 4, 4), _gaussian_atom(ny, nx, 12, 12)]
    log_A = _stack_atoms_2d(A_flat, ny, nx)
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=1.0)
    rho = torch.tensor([[1.0 - 1e-6, 1e-6]], dtype=torch.float32)
    log_bary = wasserstein_barycenter_separable(
        rho, log_A, log_Ky, log_Kx, n_iter=40, last_iter_with_grad=False
    )
    bary = torch.exp(log_bary).squeeze(0).numpy()
    # Argmax cell of the barycenter matches argmax cell of A_0.
    assert np.unravel_index(int(bary.argmax()), bary.shape) == (4, 4)
    # And the L1 distance is small (entropic regularization smooths slightly).
    A0_2d = A_flat[0].reshape(ny, nx)
    assert float(np.abs(bary - A0_2d).sum()) < 0.2


def test_barycenter_two_atom_midpoint_is_unimodal() -> None:
    """ρ = (0.5, 0.5) of two well-separated atoms gives a unimodal barycenter
    centered at the midpoint — distinct from the linear average, which is
    bimodal."""
    ny, nx = 16, 16
    A0_flat = _gaussian_atom(ny, nx, 4, 4)
    A1_flat = _gaussian_atom(ny, nx, 12, 12)
    log_A = _stack_atoms_2d([A0_flat, A1_flat], ny, nx)
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=1.0)
    rho = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    log_bary = wasserstein_barycenter_separable(
        rho, log_A, log_Ky, log_Kx, n_iter=40, last_iter_with_grad=False
    )
    bary = torch.exp(log_bary).squeeze(0).numpy()
    # Centroid lies near the midpoint (8, 8).
    iy = np.arange(ny, dtype=np.float64)
    ix = np.arange(nx, dtype=np.float64)
    cy = float((bary.sum(axis=1) * iy).sum() / bary.sum())
    cx = float((bary.sum(axis=0) * ix).sum() / bary.sum())
    assert abs(cy - 8.0) < 0.5
    assert abs(cx - 8.0) < 0.5
    # The barycenter must differ substantially from the linear average:
    # linear avg is bimodal (peaks at 4 and 12); barycenter is unimodal at 8.
    A0_2d = A0_flat.reshape(ny, nx)
    A1_2d = A1_flat.reshape(ny, nx)
    linear_avg = 0.5 * A0_2d + 0.5 * A1_2d
    l1_diff = float(np.abs(bary - linear_avg).sum())
    assert l1_diff > 0.5, f"barycenter and linear avg too similar: L1={l1_diff:.3f}"


def test_barycenter_identical_atoms_recovers_that_atom() -> None:
    """If all K atoms are identical to some distribution A_*, the
    barycenter must equal A_* for every choice of ρ.

    This is a load-bearing sanity check on the barycenter fixed-point
    iteration: a degenerate atom set (zero variation across the
    dictionary) collapses the optimization onto a single fixed point
    that doesn't depend on the weights. If this test fails, the
    fixed-point recursion is not implementing the barycenter
    correctly.
    """
    ny, nx = 16, 16
    A_star_flat = _gaussian_atom(ny, nx, 7, 8)  # arbitrary atom
    K = 4
    A = np.tile(A_star_flat[None, :], (K, 1))  # K copies of A_star
    log_A = torch.log(torch.from_numpy(A.reshape(K, ny, nx)).clamp_min(1e-12))
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=1.0)
    # Three different ρ rows — none should affect the result.
    rho = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # one-hot
            [0.25, 0.25, 0.25, 0.25],  # uniform
            [0.7, 0.1, 0.1, 0.1],  # skewed
        ],
        dtype=torch.float32,
    )
    log_bary = wasserstein_barycenter_separable(
        rho, log_A, log_Ky, log_Kx, n_iter=30, last_iter_with_grad=False
    )
    bary = torch.exp(log_bary).numpy()
    A_star_2d = A_star_flat.reshape(ny, nx)
    # All three rows must collapse onto A_star regardless of ρ.
    for i in range(rho.shape[0]):
        l1 = float(np.abs(bary[i] - A_star_2d).sum())
        # Allow small entropic blur (~0.15 L1 in this regime; same
        # tolerance band as the one-hot test).
        assert l1 < 0.2, f"rho row {i}: barycenter L1 from A_star = {l1:.4f}"
    # And all three rows must agree with each other (the optimization
    # is genuinely ρ-independent on a degenerate atom set).
    np.testing.assert_allclose(bary[0], bary[1], atol=1e-5)
    np.testing.assert_allclose(bary[0], bary[2], atol=1e-5)


def test_barycenter_simplex_normalization() -> None:
    """Each row of the barycenter sums to 1 in exp space."""
    ny, nx = 12, 12
    A_flat = [
        _gaussian_atom(ny, nx, 3, 3),
        _gaussian_atom(ny, nx, 9, 9),
        _gaussian_atom(ny, nx, 3, 9),
    ]
    log_A = _stack_atoms_2d(A_flat, ny, nx)
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=1.0)
    rho = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.5, 0.4], [0.33, 0.34, 0.33]])
    log_bary = wasserstein_barycenter_separable(
        rho, log_A, log_Ky, log_Kx, n_iter=20, last_iter_with_grad=False
    )
    bary = torch.exp(log_bary)
    sums = bary.reshape(3, -1).sum(dim=1)
    np.testing.assert_allclose(sums.numpy(), 1.0, atol=1e-5)


def test_barycenter_input_validation() -> None:
    """Wrong shape on log_A or kernel raises."""
    ny, nx = 12, 12
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=1.0)
    log_A = torch.zeros(3, ny, nx)
    rho = torch.tensor([[0.3, 0.3, 0.4]])
    # Wrong K in log_A.
    with pytest.raises(ValueError, match="log_A must have shape"):
        wasserstein_barycenter_separable(rho, torch.zeros(3, ny + 2), log_Ky, log_Kx, n_iter=5)
    # Mismatched grid in kernel.
    with pytest.raises(ValueError, match="kernel shapes"):
        bad_log_Ky, _ = build_separable_log_kernel(ny + 2, nx, epsilon=1.0)
        wasserstein_barycenter_separable(rho, log_A, bad_log_Ky, log_Kx, n_iter=5)


def test_barycenter_gradient_flows_to_rho() -> None:
    """When ``rho`` requires_grad, autograd produces a gradient on it."""
    ny, nx = 8, 8
    A = np.stack([_gaussian_atom(ny, nx, 2, 2), _gaussian_atom(ny, nx, 6, 6)])
    log_A = torch.log(torch.from_numpy(A).clamp_min(1e-12)).view(2, ny, nx)
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=1.0)
    rho = torch.tensor([[0.5, 0.5]], dtype=torch.float32, requires_grad=True)
    log_bary = wasserstein_barycenter_separable(
        rho, log_A, log_Ky, log_Kx, n_iter=10, last_iter_with_grad=True
    )
    loss = log_bary.sum()
    loss.backward()
    assert rho.grad is not None
    assert rho.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# fit_v2_rho_given_A: the headline V2 prototype
# ---------------------------------------------------------------------------


def test_v2_recovers_known_sparse_rho() -> None:
    """Generate Q via TRUE entropic barycenters with known sparse ρ; verify
    that fit_v2_rho_given_A recovers ρ to within ~0.05 mean L1 error."""
    np.random.seed(0)
    ny, nx = 16, 16
    K = 3
    A = np.stack(
        [
            _gaussian_atom(ny, nx, 3, 3),
            _gaussian_atom(ny, nx, 3, 12),
            _gaussian_atom(ny, nx, 12, 8),
        ]
    )

    true_rhos = np.array(
        [
            [0.95, 0.025, 0.025],
            [0.025, 0.95, 0.025],
            [0.025, 0.025, 0.95],
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.5, 0.4, 0.1],
        ],
        dtype=np.float32,
    )
    P = true_rhos.shape[0]

    # Generate Q as true entropic barycenters of A with these weights.
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=1.0)
    log_A = torch.log(torch.from_numpy(A).clamp_min(1e-12)).view(K, ny, nx)
    with torch.no_grad():
        log_bary = wasserstein_barycenter_separable(
            torch.from_numpy(true_rhos),
            log_A,
            log_Ky,
            log_Kx,
            n_iter=40,
            last_iter_with_grad=False,
        )
        Q = torch.exp(log_bary).view(P, ny * nx).numpy().astype(np.float32)
    Q = Q / Q.sum(axis=1, keepdims=True)  # defensive renormalization

    # Fit V2 with frozen A.
    result = fit_v2_rho_given_A(
        Q,
        A,
        grid_ny=ny,
        grid_nx=nx,
        cell_size_y=1.0,
        cell_size_x=1.0,
        epsilon=1.0,
        max_iter=200,
        sinkhorn_iter=15,
        barycenter_iter=20,
        lr=0.1,
        batch_size=P,
        seed=0,
        device="cpu",
        verbose=False,
    )
    recovered = result.mixtures
    mean_l1 = float(np.abs(true_rhos - recovered).mean(axis=1).mean())
    assert mean_l1 < 0.05, f"V2 failed to recover sparse ρ: mean L1 = {mean_l1:.4f}"


def test_v2_returns_input_archetypes_unchanged() -> None:
    """The frozen-A contract: A on the way out equals A on the way in."""
    np.random.seed(0)
    ny, nx = 8, 8
    A = np.stack(
        [
            _gaussian_atom(ny, nx, 2, 2),
            _gaussian_atom(ny, nx, 2, 6),
            _gaussian_atom(ny, nx, 6, 4),
        ]
    )
    Q = np.full((4, ny * nx), 1.0 / (ny * nx), dtype=np.float32)
    result = fit_v2_rho_given_A(
        Q,
        A,
        grid_ny=ny,
        grid_nx=nx,
        max_iter=2,
        sinkhorn_iter=4,
        barycenter_iter=4,
        batch_size=4,
        device="cpu",
    )
    np.testing.assert_array_equal(result.archetypes, A)


def test_v2_recovers_true_rho_better_than_v1_on_barycenter_data() -> None:
    """The headline V2 success criterion: when Q is generated as the
    *true* Wasserstein barycenter of A with known ρ_true, **V2
    recovers ρ_true closely while V1 lands on a different ρ** (because
    V1's linear ``ρ A`` is a different geometric object from the
    barycenter even when both use the same atoms).

    Compares mean per-row L1 between recovered and true ρ. The
    interesting empirical fact this test pins: V1's recovered ρ
    is *not* the diffuse-toward-uniform pattern we see on real NBA
    data — on synthetic-from-barycenter data V1 actually finds a
    *different* ρ that minimizes its (mismatched) reconstruction
    objective. V2 has the right reconstruction object and recovers
    ρ_true.
    """
    np.random.seed(0)
    ny, nx = 12, 12
    K = 3
    A = np.stack(
        [
            _gaussian_atom(ny, nx, 3, 3),
            _gaussian_atom(ny, nx, 3, 9),
            _gaussian_atom(ny, nx, 9, 6),
        ]
    )

    # Sample sparse true ρ from Dirichlet(0.3) (favors near-vertices of simplex).
    rng = np.random.default_rng(0)
    P = 24
    true_rhos = rng.dirichlet(0.3 * np.ones(K), size=P).astype(np.float32)

    # Generate Q as true entropic barycenters.
    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=1.0)
    log_A = torch.log(torch.from_numpy(A).clamp_min(1e-12)).view(K, ny, nx)
    with torch.no_grad():
        log_bary = wasserstein_barycenter_separable(
            torch.from_numpy(true_rhos),
            log_A,
            log_Ky,
            log_Kx,
            n_iter=40,
            last_iter_with_grad=False,
        )
        Q = torch.exp(log_bary).view(P, ny * nx).numpy().astype(np.float32)
    Q = Q / Q.sum(axis=1, keepdims=True)

    # V1: solve linear ρ A reconstruction (atoms warm-started at the truth).
    v1 = fit_archetypes_v1(
        Q,
        grid_ny=ny,
        grid_nx=nx,
        K=K,
        cell_size_y=1.0,
        cell_size_x=1.0,
        epsilon=1.0,
        max_iter=200,
        sinkhorn_iter=15,
        lr=0.1,
        A_init=A,  # warm-start at truth so V1 doesn't have to find atoms
        batch_size=P,
        seed=0,
        device="cpu",
        verbose=False,
    )
    # V2: solve barycentric reconstruction with same A.
    v2 = fit_v2_rho_given_A(
        Q,
        A,
        grid_ny=ny,
        grid_nx=nx,
        cell_size_y=1.0,
        cell_size_x=1.0,
        epsilon=1.0,
        max_iter=200,
        sinkhorn_iter=15,
        barycenter_iter=20,
        lr=0.1,
        batch_size=P,
        seed=0,
        device="cpu",
        verbose=False,
    )

    # V2 should recover ρ_true to high precision.
    v2_err = float(np.abs(v2.mixtures - true_rhos).mean(axis=1).mean())
    assert v2_err < 0.05, f"V2 failed to recover ρ_true: mean L1 = {v2_err:.4f}"

    # V1 cannot recover ρ_true on barycenter-generated data — its
    # reconstruction object (linear ρA) is different from the barycenter,
    # so the optimal ρ for V1's loss is different from ρ_true.
    v1_err = float(np.abs(v1.mixtures - true_rhos).mean(axis=1).mean())
    assert v2_err < v1_err, (
        f"V2 must recover ρ_true better than V1 on barycenter data: "
        f"V2 err = {v2_err:.4f}, V1 err = {v1_err:.4f}"
    )


def test_v2_input_validation() -> None:
    """Q rows must be simplices; A rows must be simplices; shape mismatch raises."""
    np.random.seed(0)
    ny, nx = 6, 6
    A = np.stack([_gaussian_atom(ny, nx, 1, 1), _gaussian_atom(ny, nx, 4, 4)])
    Q_good = np.full((3, ny * nx), 1.0 / (ny * nx), dtype=np.float32)

    # Grid mismatch.
    with pytest.raises(ValueError, match="grid_ny \\* grid_nx"):
        fit_v2_rho_given_A(Q_good, A, grid_ny=5, grid_nx=5)

    # Q has NaN.
    Q_nan = Q_good.copy()
    Q_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_v2_rho_given_A(Q_nan, A, grid_ny=ny, grid_nx=nx, max_iter=1)

    # Q rows don't sum to 1.
    Q_bad = Q_good * 2.0
    with pytest.raises(ValueError, match="rows must sum to 1"):
        fit_v2_rho_given_A(Q_bad, A, grid_ny=ny, grid_nx=nx, max_iter=1)

    # A rows don't sum to 1.
    A_bad = A * 0.5
    with pytest.raises(ValueError, match="A rows must sum to 1"):
        fit_v2_rho_given_A(Q_good, A_bad, grid_ny=ny, grid_nx=nx, max_iter=1)
