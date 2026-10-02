"""Tests for :mod:`shotcloud.training.spatial_losses`.

For the grid-based continuous-coordinate loss:

A. **Exact limit.** As ``τ → 0`` with shots on cell centers, the continuous NLL
   reduces to the exact-cell NLL.
B. **Distance monotonicity.** With all predicted mass on one cell, the loss strictly
   increases as the observed coordinate moves away from it.
C. **Kernel normalization.** With ``normalize_kernel=True`` each shot's observation
   kernel sums to 1 over cells.
D. **Gradient flow.** ``log_probs`` and ``shot_xy`` receive nonzero gradient.
E. **Shapes.** Realistic batch shapes run on CPU; malformed inputs raise.

Also covered: ``expected_distance_ft``, ``exact_cell_nll``, the continuous-mixture
and mode-mixture log-likelihoods (closed-form values, masking, per-shot σ,
precomputed kernels), and the half-court normalizer that makes the mixture a proper
density on the court.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from shotcloud.grids import CourtGrid
from shotcloud.training.spatial_losses import (
    continuous_coordinate_nll,
    exact_cell_nll,
    expected_distance_ft,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _grid_centers(grid: CourtGrid) -> torch.Tensor:
    """Image-layout (c = iy*nx + ix) cell centers as a (C, 2) float32 tensor."""
    cx = np.tile(grid.xcenters, grid.ny)
    cy = np.repeat(grid.ycenters, grid.nx)
    return torch.from_numpy(np.stack([cx, cy], axis=1).astype(np.float32))


def _onehot_log_probs(target_cell: int, n_cells: int) -> torch.Tensor:
    """Return ``(1, C)`` log-probs concentrated almost entirely on ``target_cell``.

    A literal one-hot would give -inf elsewhere; we use a small-mass
    floor so the log is finite while the result is numerically a delta.
    """
    eps = 1e-30
    p = torch.full((1, n_cells), eps, dtype=torch.float32)
    p[0, target_cell] = 1.0 - eps * (n_cells - 1)
    return p.log()


# ---------------------------------------------------------------------------
# A. Exact limit: τ → 0 collapses to exact-cell NLL
# ---------------------------------------------------------------------------


def test_continuous_nll_collapses_to_exact_cell_nll_as_tau_tends_to_zero() -> None:
    """For shots on cell centers, ``τ → 0`` turns the observation kernel into an
    indicator of the cell, and the continuous loss matches the exact-cell NLL.
    """
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=16, ny=14)
    centers = _grid_centers(grid)
    n_cells = grid.n_cells

    # Place four shots exactly on randomly chosen cell centers.
    rng = np.random.default_rng(0)
    target_cells = torch.from_numpy(rng.choice(n_cells, size=4, replace=False).astype(np.int64))
    shot_xy = centers[target_cells]

    # With uniform predicted log-probs the exact-cell NLL is log(n_cells).
    log_probs = torch.full((4, n_cells), -np.log(n_cells), dtype=torch.float32)
    exact = exact_cell_nll(log_probs, target_cells)
    # The unnormalized log K_τ is 0 at the matching cell and → -inf elsewhere, so the
    # logsumexp returns the predicted log-prob of the observed cell. The normalized
    # kernel gives the same value, since only one cell has non-vanishing mass.
    cont = continuous_coordinate_nll(log_probs, shot_xy, centers, tau=1e-4, normalize_kernel=False)
    torch.testing.assert_close(cont, exact, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# B. Distance monotonicity
# ---------------------------------------------------------------------------


def test_continuous_nll_increases_monotonically_with_distance_to_predicted_mass() -> None:
    """With a near-delta prediction on cell ``c*``, the continuous NLL strictly
    increases as ``shot_xy`` moves away from ``x_{c*}`` along a line.
    """
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=18)
    centers = _grid_centers(grid)
    n_cells = grid.n_cells

    target_cell = grid.coord_to_cell(0.0, 5.0).item()
    log_probs = _onehot_log_probs(int(target_cell), n_cells)
    x_center = centers[int(target_cell)]

    distances = [0.5, 1.5, 5.0, 12.0, 25.0]
    losses = []
    for d in distances:
        shot_xy = (x_center + torch.tensor([d, 0.0])).unsqueeze(0)
        loss = continuous_coordinate_nll(
            log_probs, shot_xy, centers, tau=1.0, normalize_kernel=True
        )
        losses.append(float(loss.item()))
    for prev, curr in itertools.pairwise(losses):
        assert prev < curr, f"loss should grow with distance; got {losses}"


# ---------------------------------------------------------------------------
# C. Kernel normalization
# ---------------------------------------------------------------------------


def test_normalized_observation_kernel_rows_sum_to_one() -> None:
    """The normalized observation kernel is a distribution over cells for any observed
    coordinate.

    Checked indirectly: with uniform ``log_probs`` the loss equals ``log n_cells``
    exactly when the kernel rows sum to 1.
    """
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=12, ny=10)
    centers = _grid_centers(grid)
    n_cells = grid.n_cells

    # Pick shots scattered around the court including a near-corner case.
    shot_xy = torch.tensor(
        [[0.0, 5.0], [10.0, 20.0], [-22.0, 45.0], [24.0, -4.0]], dtype=torch.float32
    )
    log_probs = torch.full((4, n_cells), -np.log(n_cells), dtype=torch.float32)
    cont = continuous_coordinate_nll(log_probs, shot_xy, centers, tau=2.0, normalize_kernel=True)
    # With both rows normalized to sum-to-1, the marginal log-likelihood is:
    #   log Σ_c p(c) K(y - c) = log Σ_c (1/C) k(c) = log(1/C) + logsumexp log k
    # where logsumexp log k = 0 (k normalized). So loss = -log(1/C) = log C.
    expected = torch.full((4,), float(np.log(n_cells)), dtype=torch.float32)
    torch.testing.assert_close(cont, expected, atol=1e-5, rtol=1e-5)


def test_unnormalized_kernel_loss_differs_from_normalized_loss_for_off_center_shots() -> None:
    """At a non-trivial ``τ``, the normalized and unnormalized losses differ for shots
    away from cell centers."""
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=16, ny=14)
    centers = _grid_centers(grid)
    n_cells = grid.n_cells
    shot_xy = torch.tensor([[0.3, 5.7]], dtype=torch.float32)  # off-center
    log_probs = torch.full((1, n_cells), -np.log(n_cells), dtype=torch.float32)
    loss_norm = continuous_coordinate_nll(
        log_probs, shot_xy, centers, tau=1.0, normalize_kernel=True
    )
    loss_raw = continuous_coordinate_nll(
        log_probs, shot_xy, centers, tau=1.0, normalize_kernel=False
    )
    assert float(loss_norm.item()) != pytest.approx(float(loss_raw.item()))


# ---------------------------------------------------------------------------
# D. Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_log_probs_and_shot_xy() -> None:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=12, ny=10)
    centers = _grid_centers(grid)
    n_cells = grid.n_cells

    log_probs = torch.full((3, n_cells), -np.log(n_cells), dtype=torch.float32).requires_grad_(True)
    shot_xy = torch.tensor(
        [[0.0, 5.0], [10.0, 20.0], [-3.0, 12.0]], dtype=torch.float32
    ).requires_grad_(True)

    loss = continuous_coordinate_nll(
        log_probs, shot_xy, centers, tau=1.0, normalize_kernel=True
    ).mean()
    loss.backward()
    assert log_probs.grad is not None and log_probs.grad.abs().sum() > 0
    assert shot_xy.grad is not None and shot_xy.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# E. Batch + device + shape sanity
# ---------------------------------------------------------------------------


def test_returns_per_shot_tensor_of_shape_b() -> None:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=8)
    centers = _grid_centers(grid)
    n_cells = grid.n_cells
    rng = np.random.default_rng(0)
    log_probs = torch.from_numpy(rng.normal(size=(7, n_cells)).astype(np.float32)).log_softmax(
        dim=-1
    )
    shot_xy = torch.from_numpy(
        np.stack([rng.uniform(-15, 15, 7), rng.uniform(0, 25, 7)], axis=1).astype(np.float32)
    )
    out = continuous_coordinate_nll(log_probs, shot_xy, centers, tau=1.0)
    assert out.shape == (7,)


def test_rejects_invalid_inputs() -> None:
    log_probs = torch.zeros(3, 100)
    shot_xy_good = torch.zeros(3, 2)
    centers = torch.zeros(100, 2)
    # Bad shot_xy rank.
    with pytest.raises(ValueError, match="shot_xy"):
        continuous_coordinate_nll(log_probs, torch.zeros(3), centers, tau=1.0)
    # Bad centers shape.
    with pytest.raises(ValueError, match="cell_centers"):
        continuous_coordinate_nll(log_probs, shot_xy_good, torch.zeros(100), tau=1.0)
    # Non-positive tau.
    with pytest.raises(ValueError, match="tau"):
        continuous_coordinate_nll(log_probs, shot_xy_good, centers, tau=0.0)


# ---------------------------------------------------------------------------
# expected_distance_ft
# ---------------------------------------------------------------------------


def test_expected_distance_zero_when_all_mass_at_observed_coordinate() -> None:
    """With (nearly) all predicted mass on the observed shot's cell center, the expected
    distance is approximately 0."""
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=16, ny=14)
    centers = _grid_centers(grid)
    n_cells = grid.n_cells

    target_cell = grid.coord_to_cell(0.0, 5.0).item()
    log_probs = _onehot_log_probs(int(target_cell), n_cells)
    shot_xy = centers[int(target_cell)].unsqueeze(0)
    out = expected_distance_ft(log_probs, shot_xy, centers)
    assert out.shape == (1,)
    assert float(out.item()) < 1e-3


def test_expected_distance_grows_with_observation_displacement() -> None:
    """With the prediction fixed at a cell, moving the observation away from it increases
    ``E[||x_c - y||]``."""
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=14, ny=12)
    centers = _grid_centers(grid)
    n_cells = grid.n_cells

    target_cell = grid.coord_to_cell(0.0, 5.0).item()
    log_probs = _onehot_log_probs(int(target_cell), n_cells)
    x_center = centers[int(target_cell)]
    distances = []
    for d in (0.0, 3.0, 10.0, 25.0):
        shot_xy = (x_center + torch.tensor([d, 0.0])).unsqueeze(0)
        distances.append(float(expected_distance_ft(log_probs, shot_xy, centers).item()))
    for prev, curr in itertools.pairwise(distances):
        assert curr > prev, f"expected_distance should grow with offset; got {distances}"


# ---------------------------------------------------------------------------
# exact_cell_nll
# ---------------------------------------------------------------------------


def test_exact_cell_nll_matches_gather() -> None:
    rng = np.random.default_rng(0)
    n_cells = 20
    log_probs = torch.from_numpy(rng.normal(size=(5, n_cells)).astype(np.float32)).log_softmax(
        dim=-1
    )
    cell_idx = torch.from_numpy(rng.integers(0, n_cells, size=5).astype(np.int64))
    nll = exact_cell_nll(log_probs, cell_idx)
    manual = -torch.stack([log_probs[i, cell_idx[i]] for i in range(5)])
    torch.testing.assert_close(nll, manual, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Continuous-mixture NLL
# ---------------------------------------------------------------------------


def _import_mixture():  # type: ignore[no-untyped-def]
    from shotcloud.training.spatial_losses import (
        continuous_mixture_loglik,
        continuous_mixture_nll,
    )

    return continuous_mixture_loglik, continuous_mixture_nll


def test_mixture_nll_finite_and_differentiable_on_random_input() -> None:
    continuous_mixture_loglik, continuous_mixture_nll = _import_mixture()
    rng = np.random.default_rng(0)
    b, m = 4, 25
    support_xy = torch.from_numpy(
        np.stack(
            [rng.uniform(-15, 15, size=(b, m)), rng.uniform(0, 30, size=(b, m))], axis=-1
        ).astype(np.float32)
    )
    log_weights = torch.from_numpy(rng.normal(size=(b, m)).astype(np.float32)).requires_grad_(True)
    shot_xy = torch.from_numpy(
        np.stack([rng.uniform(-15, 15, size=b), rng.uniform(0, 30, size=b)], axis=-1).astype(
            np.float32
        )
    )
    sigma = torch.from_numpy(rng.uniform(1.0, 3.0, size=b).astype(np.float32)).requires_grad_(True)
    loglik = continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma)
    assert loglik.shape == (b,)
    assert torch.isfinite(loglik).all()
    nll = continuous_mixture_nll(log_weights, support_xy, shot_xy, sigma)
    torch.testing.assert_close(nll, -loglik)
    nll.mean().backward()
    assert log_weights.grad is not None and log_weights.grad.abs().sum() > 0
    assert sigma.grad is not None and sigma.grad.abs().sum() > 0


def test_mixture_one_component_at_observed_yields_log_2pi_sigma2() -> None:
    """With one support point at the observed shot, the density is ``1 / (2π σ²)``, so
    the NLL is exactly ``log(2π σ²)`` (checks the kernel normalization)."""
    continuous_mixture_loglik, _ = _import_mixture()
    b = 3
    shot_xy = torch.tensor([[0.0, 5.0], [-3.0, 10.0], [12.0, 22.0]], dtype=torch.float32)
    support_xy = shot_xy.unsqueeze(1).clone()  # (B, M=1, 2)
    log_weights = torch.zeros(b, 1, dtype=torch.float32)  # singleton → softmax = [1]
    sigma = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)
    loglik = continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma)
    expected = -torch.log(2 * torch.pi * sigma.pow(2))
    torch.testing.assert_close(loglik, expected, atol=1e-5, rtol=1e-5)


def test_mixture_distance_monotonicity_single_support() -> None:
    """With one support point at the origin, the log-likelihood decreases as the
    observed coordinate moves away from it."""
    continuous_mixture_loglik, _ = _import_mixture()
    sigma = torch.tensor([1.5], dtype=torch.float32)
    support_xy = torch.zeros(1, 1, 2, dtype=torch.float32)
    log_weights = torch.zeros(1, 1, dtype=torch.float32)
    last = None
    for d in [0.0, 1.0, 3.0, 6.0, 10.0]:
        shot_xy = torch.tensor([[d, 0.0]], dtype=torch.float32)
        loglik = continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma).item()
        if last is not None:
            assert loglik < last, f"loglik should drop as shot moves away; got {loglik} >= {last}"
        last = loglik


def test_mixture_benefit_when_weight_concentrates_on_correct_component() -> None:
    """With the observed shot near support 1, shifting weight from support 0 to support
    1 increases the log-likelihood."""
    continuous_mixture_loglik, _ = _import_mixture()
    support_xy = torch.tensor([[[10.0, 10.0], [0.0, 0.0]]], dtype=torch.float32)  # (1, 2, 2)
    shot_xy = torch.tensor([[0.2, 0.1]], dtype=torch.float32)  # near support 1
    sigma = torch.tensor([1.0], dtype=torch.float32)
    last = None
    for logit_on_correct in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        log_weights = torch.tensor([[0.0, logit_on_correct]], dtype=torch.float32)
        loglik = continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma).item()
        if last is not None:
            assert loglik > last, (
                f"loglik should grow as weight shifts to the correct component; "
                f"got {loglik} <= {last}"
            )
        last = loglik


def test_mixture_mask_zeroes_out_invalid_support() -> None:
    """A masked support point does not contribute, even at the observed shot; padded
    slots and cold-start rows rely on this."""
    continuous_mixture_loglik, _ = _import_mixture()
    # support[0] = obs (would dominate if unmasked); support[1] far away.
    shot_xy = torch.tensor([[0.0, 5.0]], dtype=torch.float32)
    support_xy = torch.tensor([[[0.0, 5.0], [20.0, 30.0]]], dtype=torch.float32)
    log_weights = torch.zeros(1, 2, dtype=torch.float32)
    sigma = torch.tensor([1.0], dtype=torch.float32)
    mask_all = torch.tensor([[True, True]])
    mask_far = torch.tensor([[False, True]])
    loglik_all = continuous_mixture_loglik(
        log_weights, support_xy, shot_xy, sigma, support_mask=mask_all
    ).item()
    loglik_far = continuous_mixture_loglik(
        log_weights, support_xy, shot_xy, sigma, support_mask=mask_far
    ).item()
    # With the near support masked, only the far support remains and the likelihood
    # is much lower.
    assert loglik_far < loglik_all - 5.0


def test_mixture_all_masked_row_does_not_nan_poison_sigma_gradient() -> None:
    """Cold-start rows in a batch leave the σ gradient finite.

    ``logsumexp`` over an all ``-inf`` row has a NaN backward, and ``0 · NaN = NaN``
    under the ``torch.where`` mask would spread it to the shared σ and weight
    gradients. The loglik patches one dummy log-weight per cold-start row before the
    ``logsumexp`` to prevent this.
    """
    continuous_mixture_loglik, _ = _import_mixture()
    b, m = 4, 5
    support_xy = torch.zeros(b, m, 2, dtype=torch.float32)
    shot_xy = torch.zeros(b, 2, dtype=torch.float32)
    log_weights = torch.zeros(b, m, dtype=torch.float32)
    sigma = torch.tensor([1.0, 1.5, 2.0, 1.2], dtype=torch.float32, requires_grad=True)
    # Rows 0 and 2 are cold-start; rows 1 and 3 have valid support.
    mask = torch.tensor(
        [
            [False, False, False, False, False],  # cold start
            [True, True, False, False, False],
            [False, False, False, False, False],  # cold start
            [True, False, True, False, True],
        ]
    )
    loglik = continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma, support_mask=mask)
    # Valid rows are finite and cold-start rows are -inf (the caller applies the
    # floor). Backward through the valid rows must give a finite σ.grad, with zeros
    # rather than NaN on cold-start rows.
    valid_loglik = loglik[torch.isfinite(loglik)]
    assert valid_loglik.numel() == 2
    valid_loglik.sum().backward()
    assert sigma.grad is not None
    assert torch.isfinite(sigma.grad).all(), f"σ.grad has non-finite values: {sigma.grad}"


def test_mixture_all_masked_row_yields_neg_inf() -> None:
    """Rows with no valid support get ``log_lik = -inf``; applying a floor is left to
    the caller."""
    continuous_mixture_loglik, _ = _import_mixture()
    support_xy = torch.zeros(1, 3, 2, dtype=torch.float32)
    log_weights = torch.zeros(1, 3, dtype=torch.float32)
    shot_xy = torch.zeros(1, 2, dtype=torch.float32)
    sigma = torch.tensor([1.0], dtype=torch.float32)
    mask = torch.zeros(1, 3, dtype=torch.bool)  # all invalid
    loglik = continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma, support_mask=mask)
    assert torch.isinf(loglik).all() and (loglik < 0).all()


def test_mixture_weights_are_log_probs_passthrough() -> None:
    """With ``weights_are_log_probs=True`` normalized log-probs are used as given and
    match the default path on the corresponding unnormalized logits."""
    continuous_mixture_loglik, _ = _import_mixture()
    rng = np.random.default_rng(0)
    b, m = 4, 10
    support_xy = torch.from_numpy(
        np.stack(
            [rng.uniform(-15, 15, size=(b, m)), rng.uniform(0, 30, size=(b, m))], axis=-1
        ).astype(np.float32)
    )
    shot_xy = torch.zeros(b, 2, dtype=torch.float32)
    sigma = torch.ones(b, dtype=torch.float32)
    logits = torch.from_numpy(rng.normal(size=(b, m)).astype(np.float32))
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    a = continuous_mixture_loglik(logits, support_xy, shot_xy, sigma)
    b_ = continuous_mixture_loglik(
        log_probs, support_xy, shot_xy, sigma, weights_are_log_probs=True
    )
    torch.testing.assert_close(a, b_, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Mode-mixture NLL: per-row K-mode Gaussian mixture used by the mode-mixture
# spatial decoder.
# ---------------------------------------------------------------------------


def _import_mode_mix():  # type: ignore[no-untyped-def]
    from shotcloud.training.spatial_losses import (
        mode_mixture_loglik,
        mode_mixture_nll,
    )

    return mode_mixture_loglik, mode_mixture_nll


def test_mode_mixture_one_mode_at_observed_yields_log_2pi_sigma2() -> None:
    """K=1 with μ = y → log f(y) = log φ_2(y; y, σ² I) = -log(2π σ²)."""
    mode_mixture_loglik, _ = _import_mode_mix()
    b = 3
    shot_xy = torch.tensor([[0.0, 5.0], [-3.0, 10.0], [12.0, 22.0]], dtype=torch.float32)
    mode_mu = shot_xy.unsqueeze(1)  # (B, K=1, 2)
    mode_logits = torch.zeros(b, 1, dtype=torch.float32)
    sigma = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)  # per-row
    loglik = mode_mixture_loglik(mode_logits, mode_mu, shot_xy, sigma)
    expected = -torch.log(2 * torch.pi * sigma.pow(2))
    torch.testing.assert_close(loglik, expected, atol=1e-5, rtol=1e-5)


def test_mode_mixture_distance_monotonicity_single_mode() -> None:
    mode_mixture_loglik, _ = _import_mode_mix()
    mode_mu = torch.zeros(1, 1, 2)
    mode_logits = torch.zeros(1, 1)
    last = None
    for d in [0.0, 1.0, 3.0, 6.0, 10.0]:
        shot_xy = torch.tensor([[d, 0.0]], dtype=torch.float32)
        loglik = mode_mixture_loglik(mode_logits, mode_mu, shot_xy, 1.5).item()
        if last is not None:
            assert loglik < last
        last = loglik


def test_mode_mixture_weight_shift_toward_nearby_mode_increases_loglik() -> None:
    """With one mode near the observed shot and one far, raising the near mode's logit
    strictly increases the log-likelihood."""
    mode_mixture_loglik, _ = _import_mode_mix()
    mode_mu = torch.tensor([[[10.0, 10.0], [0.0, 0.0]]], dtype=torch.float32)
    shot_xy = torch.tensor([[0.2, 0.1]], dtype=torch.float32)
    last = None
    for logit_on_correct in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        mode_logits = torch.tensor([[0.0, logit_on_correct]], dtype=torch.float32)
        loglik = mode_mixture_loglik(mode_logits, mode_mu, shot_xy, 1.0).item()
        if last is not None:
            assert loglik > last
        last = loglik


def test_mode_mixture_gradient_flows_to_logits_and_centers() -> None:
    mode_mixture_loglik, mode_mixture_nll = _import_mode_mix()
    rng = np.random.default_rng(0)
    b, k = 4, 6
    mode_logits = torch.from_numpy(rng.normal(size=(b, k)).astype(np.float32)).requires_grad_(True)
    mode_mu = torch.from_numpy(
        np.stack(
            [rng.uniform(-15, 15, size=(b, k)), rng.uniform(0, 30, size=(b, k))], axis=-1
        ).astype(np.float32)
    ).requires_grad_(True)
    shot_xy = torch.zeros(b, 2, dtype=torch.float32)
    loglik = mode_mixture_loglik(mode_logits, mode_mu, shot_xy, 2.0)
    assert loglik.shape == (b,)
    nll = mode_mixture_nll(mode_logits, mode_mu, shot_xy, 2.0)
    torch.testing.assert_close(nll, -loglik)
    nll.mean().backward()
    assert mode_logits.grad is not None and mode_logits.grad.abs().sum() > 0
    assert mode_mu.grad is not None and mode_mu.grad.abs().sum() > 0


def test_mode_mixture_sigma_broadcasting_accepts_scalar_K_B_BK() -> None:
    """``sigma`` can be a scalar, ``(K,)``, ``(B,)``, or ``(B, K)``."""
    mode_mixture_loglik, _ = _import_mode_mix()
    b, k = 2, 3
    mode_logits = torch.zeros(b, k)
    mode_mu = torch.zeros(b, k, 2)
    shot_xy = torch.zeros(b, 2)
    # Scalar.
    a = mode_mixture_loglik(mode_logits, mode_mu, shot_xy, 1.5)
    # (K,) — matches mode dim.
    b_k = mode_mixture_loglik(mode_logits, mode_mu, shot_xy, torch.full((k,), 1.5))
    # (B,) — matches batch dim.
    b_b = mode_mixture_loglik(mode_logits, mode_mu, shot_xy, torch.full((b,), 1.5))
    # (B, K) — full per-row per-mode.
    b_bk = mode_mixture_loglik(mode_logits, mode_mu, shot_xy, torch.full((b, k), 1.5))
    torch.testing.assert_close(a, b_k, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a, b_b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a, b_bk, atol=1e-6, rtol=1e-6)


def test_mode_mixture_rejects_shape_mismatches() -> None:
    mode_mixture_loglik, _ = _import_mode_mix()
    with pytest.raises(ValueError, match="mode_logits"):
        mode_mixture_loglik(torch.zeros(3), torch.zeros(3, 2, 2), torch.zeros(3, 2), 1.0)
    with pytest.raises(ValueError, match="mode_mu"):
        mode_mixture_loglik(torch.zeros(3, 2), torch.zeros(3, 2, 3), torch.zeros(3, 2), 1.0)
    with pytest.raises(ValueError, match="shot_xy"):
        mode_mixture_loglik(torch.zeros(3, 2), torch.zeros(3, 2, 2), torch.zeros(2, 2), 1.0)
    with pytest.raises(ValueError, match="sigma"):
        mode_mixture_loglik(
            torch.zeros(3, 2), torch.zeros(3, 2, 2), torch.zeros(3, 2), torch.zeros(5)
        )


def test_mixture_rejects_shape_mismatches() -> None:
    continuous_mixture_loglik, _ = _import_mixture()
    support_xy = torch.zeros(2, 3, 2)
    log_weights = torch.zeros(2, 3)
    shot_xy = torch.zeros(2, 2)
    sigma = torch.ones(2)
    # Bad sigma shape.
    with pytest.raises(ValueError, match="sigma"):
        continuous_mixture_loglik(log_weights, support_xy, shot_xy, torch.ones(2, 1))
    # Bad shot_xy shape.
    with pytest.raises(ValueError, match="shot_xy"):
        continuous_mixture_loglik(log_weights, support_xy, torch.zeros(3, 2), sigma)
    # Bad support_xy shape.
    with pytest.raises(ValueError, match="support_xy"):
        continuous_mixture_loglik(log_weights, torch.zeros(2, 3, 3), shot_xy, sigma)
    # Bad mask shape.
    with pytest.raises(ValueError, match="support_mask"):
        continuous_mixture_loglik(
            log_weights, support_xy, shot_xy, sigma, support_mask=torch.ones(2, 5).bool()
        )


def test_mixture_loglik_accepts_per_shot_sigma_and_matches_broadcast() -> None:
    """A ``(B, M)`` per-shot σ that is constant within each row matches the ``(B,)``
    path; a non-uniform per-shot σ changes the result."""
    continuous_mixture_loglik, _ = _import_mixture()
    torch.manual_seed(0)
    log_weights = torch.randn(3, 4, requires_grad=True)
    support_xy = torch.randn(3, 4, 2)
    shot_xy = torch.randn(3, 2)
    sigma_row = torch.tensor([1.5, 1.2, 1.8])
    sigma_per_shot_uniform = sigma_row.unsqueeze(-1).expand(3, 4).contiguous()
    out_row = continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma_row)
    out_per_shot = continuous_mixture_loglik(
        log_weights, support_xy, shot_xy, sigma_per_shot_uniform
    )
    torch.testing.assert_close(out_row, out_per_shot, atol=1e-6, rtol=1e-6)
    # Perturbing a single (B, M) entry changes the result.
    sigma_perturbed = sigma_per_shot_uniform.clone()
    sigma_perturbed[0, 0] = 2.5
    out_perturbed = continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma_perturbed)
    assert not torch.allclose(out_perturbed, out_row)


def test_mixture_loglik_rejects_bad_per_shot_sigma_shape() -> None:
    """``sigma`` must be ``(B,)`` or ``(B, M)``; other shapes raise."""
    continuous_mixture_loglik, _ = _import_mixture()
    log_weights = torch.randn(3, 4)
    support_xy = torch.randn(3, 4, 2)
    shot_xy = torch.randn(3, 2)
    # (B, M) with wrong M.
    with pytest.raises(ValueError, match=r"sigma \(B, M\)"):
        continuous_mixture_loglik(log_weights, support_xy, shot_xy, torch.ones(3, 5))
    # 3-D σ.
    with pytest.raises(ValueError, match=r"\(B,\) or \(B, M\)"):
        continuous_mixture_loglik(log_weights, support_xy, shot_xy, torch.ones(3, 4, 1))


# --------------------------------------------------------------------------- #
# Precomputed log_kernel path (anisotropic kernels)
# --------------------------------------------------------------------------- #


def test_mixture_loglik_accepts_precomputed_log_kernel_and_matches_sigma_path() -> None:
    """A precomputed ``log_kernel`` equal to the isotropic Gaussian kernel reproduces
    the σ path."""
    continuous_mixture_loglik, _ = _import_mixture()
    import math

    torch.manual_seed(0)
    log_weights = torch.randn(4, 6)
    support_xy = torch.randn(4, 6, 2)
    shot_xy = torch.randn(4, 2)
    sigma = torch.tensor([1.5, 1.2, 1.8, 2.0]).unsqueeze(-1).expand(4, 6).contiguous()
    # The isotropic Gaussian log-kernel at the same σ.
    diff = shot_xy.unsqueeze(1) - support_xy
    dist2 = (diff * diff).sum(dim=-1)
    sigma2 = sigma.pow(2)
    log_kernel = -math.log(2.0 * math.pi) - torch.log(sigma2) - 0.5 * dist2 / sigma2
    out_sigma = continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma)
    out_lk = continuous_mixture_loglik(log_weights, support_xy, shot_xy, log_kernel=log_kernel)
    torch.testing.assert_close(out_sigma, out_lk, atol=1e-5, rtol=0)


def test_mixture_loglik_rejects_both_sigma_and_log_kernel() -> None:
    """Passing both ``sigma`` and ``log_kernel`` raises."""
    continuous_mixture_loglik, _ = _import_mixture()
    log_weights = torch.zeros(2, 3)
    support_xy = torch.zeros(2, 3, 2)
    shot_xy = torch.zeros(2, 2)
    sigma = torch.full((2,), 1.5)
    log_kernel = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="exactly one of"):
        continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma, log_kernel=log_kernel)


def test_mixture_loglik_rejects_neither_sigma_nor_log_kernel() -> None:
    continuous_mixture_loglik, _ = _import_mixture()
    log_weights = torch.zeros(2, 3)
    support_xy = torch.zeros(2, 3, 2)
    shot_xy = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="exactly one of"):
        continuous_mixture_loglik(log_weights, support_xy, shot_xy)


def test_mixture_loglik_rejects_bad_log_kernel_shape() -> None:
    continuous_mixture_loglik, _ = _import_mixture()
    log_weights = torch.zeros(2, 3)
    support_xy = torch.zeros(2, 3, 2)
    shot_xy = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="log_kernel must match"):
        continuous_mixture_loglik(log_weights, support_xy, shot_xy, log_kernel=torch.zeros(2, 5))


# ---------------------------------------------------------------------------
# F. Half-court boundary correction
#
# A Gaussian kernel on R^2 leaks mass off the court, so without a normalizer the
# mixture is not a proper density on the court and is not comparable across models
# that allocate σ differently. These tests cover the analytic erf-based normalizer
# Z_m and its use in the mixture log-likelihoods.
# ---------------------------------------------------------------------------


def _import_normalizer():  # type: ignore[no-untyped-def]
    from shotcloud.training.spatial_losses import (
        DEFAULT_COURT_BOUNDS,
        half_court_log_normalizer,
    )

    return DEFAULT_COURT_BOUNDS, half_court_log_normalizer


def test_half_court_normalizer_deep_interior_log_z_near_zero() -> None:
    """A modest-sigma Gaussian centered deep inside the court has log Z ≈ 0.

    The 17-ft midrange centroid (0, 17) sits roughly 22 ft from the
    nearest court edge on every side, so a sigma = 1.5 ft Gaussian
    deposits essentially all of its mass on-court.
    """
    bounds, half_court = _import_normalizer()
    support_xy = torch.tensor([[[0.0, 17.0]]])  # (B=1, M=1, 2)
    sigma = torch.tensor([1.5])  # (B,)
    log_z = half_court(support_xy, sigma, bounds)
    assert log_z.shape == (1, 1)
    assert log_z.item() == pytest.approx(0.0, abs=1e-6)


def test_half_court_normalizer_at_rim_loses_baseline_mass() -> None:
    """A Gaussian at the basket (y = 0), with the baseline at y = -5, loses measurable
    mass behind the baseline when σ = 3 ft.
    """
    bounds, half_court = _import_normalizer()
    support_xy = torch.tensor([[[0.0, 0.0]]])
    sigma = torch.tensor([3.0])
    log_z = half_court(support_xy, sigma, bounds)
    z = log_z.exp().item()
    assert 0.85 < z < 0.97, f"expected partial leakage off baseline, got Z={z}"


def test_half_court_normalizer_huge_sigma_log_z_very_negative() -> None:
    """As sigma → ∞ the kernel spreads over all R^2 and the on-court mass → 0.

    With sigma = 1000 ft the on-court rectangle is a vanishing fraction
    of the kernel's footprint; log Z should be strongly negative.
    """
    bounds, half_court = _import_normalizer()
    support_xy = torch.tensor([[[0.0, 20.0]]])
    sigma = torch.tensor([1000.0])
    log_z = half_court(support_xy, sigma, bounds)
    assert log_z.item() < -5.0


def test_half_court_normalizer_factorizes_into_axis_marginals() -> None:
    """``Z`` equals the product of the x and y erf marginals."""
    import math as _m

    bounds, half_court = _import_normalizer()
    s_x, s_y = -10.0, 4.0
    sig = 2.5
    support_xy = torch.tensor([[[s_x, s_y]]])
    sigma = torch.tensor([sig])
    log_z = half_court(support_xy, sigma, bounds)
    x_min, x_max, y_min, y_max = bounds
    inv = 1.0 / (sig * _m.sqrt(2.0))
    z_expected = 0.25 * (
        (_m.erf((x_max - s_x) * inv) - _m.erf((x_min - s_x) * inv))
        * (_m.erf((y_max - s_y) * inv) - _m.erf((y_min - s_y) * inv))
    )
    assert log_z.item() == pytest.approx(_m.log(z_expected), abs=1e-6)


def test_half_court_normalizer_per_support_sigma_shape() -> None:
    """The normalizer accepts both (B,) and (B, M) sigma."""
    bounds, half_court = _import_normalizer()
    support_xy = torch.tensor([[[0.0, 17.0], [-22.0, 3.0]], [[0.0, 0.0], [10.0, 35.0]]])
    sigma_b = torch.tensor([1.5, 2.0])  # (B,)
    sigma_bm = sigma_b.unsqueeze(-1).expand(2, 2).clone()  # (B, M)
    log_z_b = half_court(support_xy, sigma_b, bounds)
    log_z_bm = half_court(support_xy, sigma_bm, bounds)
    torch.testing.assert_close(log_z_b, log_z_bm, atol=1e-6, rtol=1e-6)


def test_half_court_normalizer_clamp_floor_avoids_neg_inf() -> None:
    """Extremely large sigma drives Z below the clamp; log Z stays finite."""
    bounds, half_court = _import_normalizer()
    support_xy = torch.tensor([[[0.0, 17.0]]])
    sigma = torch.tensor([1e12])  # ridiculous
    log_z = half_court(support_xy, sigma, bounds)
    assert torch.isfinite(log_z).all().item()


def test_mixture_loglik_court_bounds_none_is_bit_identical_to_v1() -> None:
    """``court_bounds=None`` (the default) gives the unnormalized mixture loglik bit for
    bit, so results computed without the normalizer are reproducible."""
    continuous_mixture_loglik, _ = _import_mixture()
    torch.manual_seed(0)
    log_weights = torch.randn(4, 6)
    support_xy = torch.randn(4, 6, 2) * 5.0
    shot_xy = torch.randn(4, 2) * 5.0
    sigma = torch.full((4,), 1.5)
    out_no_bounds = continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma)
    out_default = continuous_mixture_loglik(
        log_weights, support_xy, shot_xy, sigma, court_bounds=None
    )
    torch.testing.assert_close(out_no_bounds, out_default, atol=0.0, rtol=0.0)


def test_mixture_loglik_court_bounds_subtracts_log_z_pointwise() -> None:
    """The court-bounded loglik divides each kernel by its on-court mass ``Z_m``, so it
    exceeds the unbounded loglik by an amount between ``-max log Z`` and
    ``-min log Z``."""
    continuous_mixture_loglik, _ = _import_mixture()
    bounds, half_court = _import_normalizer()
    # One support shot deep interior, one near the rim.
    log_weights = torch.tensor([[0.0, 0.0]])  # equal weights → softmax(0,0) = (1/2,1/2)
    support_xy = torch.tensor([[[0.0, 17.0], [0.0, 0.0]]])
    shot_xy = torch.tensor([[1.0, 10.0]])
    sigma = torch.tensor([2.0])
    log_lik_v1 = continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma)
    log_lik_c = continuous_mixture_loglik(
        log_weights, support_xy, shot_xy, sigma, court_bounds=bounds
    )
    # Per-support log Z.
    log_z = half_court(support_xy, sigma, bounds).squeeze(0)  # (M,)
    # log_lik_c = logsumexp(log_w + log_K - log_Z) and log_lik_v1 =
    # logsumexp(log_w + log_K). A constant log_Z would give an offset of exactly
    # -log_Z; here it differs between the slots, so the difference lies between
    # -max(log_Z) and -min(log_Z).
    diff = (log_lik_c - log_lik_v1).item()
    log_z_min = log_z.min().item()
    log_z_max = log_z.max().item()
    assert -log_z_max - 1e-6 <= diff <= -log_z_min + 1e-6, (
        f"expected diff in [-log_z_max, -log_z_min] = [{-log_z_max}, {-log_z_min}], got {diff}"
    )
    # The interior kernel has log_z ≈ 0 and the rim kernel loses mass behind the
    # baseline (log_z < 0), so the court-bounded loglik is strictly larger.
    assert log_z_max == pytest.approx(0.0, abs=1e-5)
    assert log_z_min < 0.0
    assert diff > 0.0


def test_mixture_loglik_rejects_court_bounds_with_log_kernel() -> None:
    """``court_bounds`` with a precomputed ``log_kernel`` raises, pointing to
    ``log_court_normalizer``."""
    continuous_mixture_loglik, _ = _import_mixture()
    bounds, _ = _import_normalizer()
    log_weights = torch.zeros(2, 3)
    support_xy = torch.zeros(2, 3, 2)
    shot_xy = torch.zeros(2, 2)
    log_kernel = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="court_bounds requires the isotropic"):
        continuous_mixture_loglik(
            log_weights,
            support_xy,
            shot_xy,
            log_kernel=log_kernel,
            court_bounds=bounds,
        )


def test_mixture_loglik_rejects_court_bounds_with_log_court_normalizer() -> None:
    """The two normalizer paths are mutually exclusive."""
    continuous_mixture_loglik, _ = _import_mixture()
    bounds, _ = _import_normalizer()
    log_weights = torch.zeros(2, 3)
    support_xy = torch.zeros(2, 3, 2)
    shot_xy = torch.zeros(2, 2)
    sigma = torch.ones(2)
    log_court_normalizer = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="mutually exclusive"):
        continuous_mixture_loglik(
            log_weights,
            support_xy,
            shot_xy,
            sigma,
            court_bounds=bounds,
            log_court_normalizer=log_court_normalizer,
        )


def test_mixture_loglik_log_court_normalizer_subtracts_pointwise() -> None:
    """The precomputed ``log_court_normalizer`` path (anisotropic kernel)
    behaves identically to passing the same value as ``court_bounds`` would
    produce in the isotropic case."""
    continuous_mixture_loglik, _ = _import_mixture()
    bounds, half_court = _import_normalizer()
    log_weights = torch.tensor([[0.0, 0.0, 0.0]])
    support_xy = torch.tensor([[[0.0, 17.0], [-22.0, 3.0], [0.0, 0.0]]])
    shot_xy = torch.tensor([[1.0, 10.0]])
    sigma = torch.tensor([2.0])
    log_z = half_court(support_xy, sigma, bounds)  # (B, M)
    out_bounds = continuous_mixture_loglik(
        log_weights, support_xy, shot_xy, sigma, court_bounds=bounds
    )
    out_precomputed = continuous_mixture_loglik(
        log_weights, support_xy, shot_xy, sigma, log_court_normalizer=log_z
    )
    torch.testing.assert_close(out_bounds, out_precomputed, atol=1e-6, rtol=1e-6)


def test_mode_mixture_loglik_court_bounds_passthrough() -> None:
    """``mode_mixture_loglik`` accepts ``court_bounds``; for a single mode losing mass
    off court, the bounded loglik exceeds the unbounded one by ``-log Z``."""
    from shotcloud.training.spatial_losses import mode_mixture_loglik

    bounds, half_court = _import_normalizer()
    # A single 3-ft mode at the rim (baseline at y = -5) loses roughly 5-10% of its
    # mass behind the baseline. With one mode, only the -log Z term changes.
    mode_logits = torch.tensor([[0.0]])
    mode_mu = torch.tensor([[[0.0, 0.0]]])
    shot_xy = torch.tensor([[0.0, 0.0]])
    sigma = torch.tensor([3.0])
    out_v1 = mode_mixture_loglik(mode_logits, mode_mu, shot_xy, sigma)
    out_c = mode_mixture_loglik(mode_logits, mode_mu, shot_xy, sigma, court_bounds=bounds)
    log_z = half_court(mode_mu, sigma, bounds).squeeze().item()
    assert log_z < 0.0, f"expected off-baseline leakage at the rim, got log Z={log_z}"
    diff = (out_c - out_v1).item()
    # The single mode's kernel value is upweighted by exactly -log Z.
    assert diff == pytest.approx(-log_z, abs=1e-6)


def test_mixture_loglik_court_bounds_preserves_gradients() -> None:
    """The court-bounded path does not block gradients on log_weights or sigma."""
    continuous_mixture_loglik, _ = _import_mixture()
    bounds, _ = _import_normalizer()
    log_weights = torch.zeros(2, 3, requires_grad=True)
    support_xy = torch.tensor(
        [[[0.0, 17.0], [-22.0, 3.0], [0.0, 0.0]], [[5.0, 8.0], [-5.0, 8.0], [0.0, 30.0]]]
    )
    shot_xy = torch.tensor([[1.0, 10.0], [0.0, 8.0]])
    sigma = torch.tensor([2.0, 1.5], requires_grad=True)
    out = continuous_mixture_loglik(log_weights, support_xy, shot_xy, sigma, court_bounds=bounds)
    out.sum().backward()
    assert log_weights.grad is not None
    assert (log_weights.grad.abs() > 0).any().item()
    assert sigma.grad is not None
    assert (sigma.grad.abs() > 0).any().item()
