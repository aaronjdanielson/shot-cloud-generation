"""Tests for :func:`shotcloud.models._separable_kernel.separable_gaussian_density`.

The separable kernel lets :class:`~shotcloud.models.CollaborativeKDE` evaluate grid
densities without materializing a dense ``(B, J, n_cells)`` kernel tensor. The tests
check numerical equivalence with two references:

1. A direct dense rank-1 outer product ``K_x ⊗ K_y``.
2. The per-shot 2-D isotropic-Gaussian softmax over flattened cells, which is
   identical because an isotropic Gaussian factors exactly across axes.

They also cover the row-sum invariant, image-layout cell ordering, gradient flow, and
input validation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shotcloud.models._separable_kernel import separable_gaussian_density

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _dense_separable_reference(
    coords: torch.Tensor,
    weights: torch.Tensor,
    sigma: torch.Tensor,
    xcenters: torch.Tensor,
    ycenters: torch.Tensor,
) -> torch.Tensor:
    """Materialize ``K_x ⊗ K_y`` directly and aggregate.

    Returns ``(B, n_y * n_x)`` flattened in image-layout (``c = iy*nx + ix``).
    """
    b, j, _ = coords.shape
    nx = int(xcenters.shape[0])
    ny = int(ycenters.shape[0])
    inv_two_sigma_sq = (1.0 / (2.0 * sigma.pow(2))).view(b, 1, 1)
    dx = coords[..., 0].unsqueeze(-1) - xcenters.view(1, 1, nx)
    dy = coords[..., 1].unsqueeze(-1) - ycenters.view(1, 1, ny)
    log_kx = -(dx * dx) * inv_two_sigma_sq  # (B, J, n_x)
    log_ky = -(dy * dy) * inv_two_sigma_sq  # (B, J, n_y)
    kx = torch.softmax(log_kx, dim=-1)
    ky = torch.softmax(log_ky, dim=-1)
    # Full rank-1 outer product: (B, J, n_x, n_y).
    k_full = kx.unsqueeze(-1) * ky.unsqueeze(-2)
    q_xy = (weights.view(b, j, 1, 1) * k_full).sum(dim=1)  # (B, n_x, n_y)
    q_yx = q_xy.transpose(1, 2).contiguous()  # (B, n_y, n_x)
    return q_yx.reshape(b, ny * nx)


def _dense_full2d_reference(
    coords: torch.Tensor,
    weights: torch.Tensor,
    sigma: torch.Tensor,
    xcenters: torch.Tensor,
    ycenters: torch.Tensor,
) -> torch.Tensor:
    """Per-shot 2-D Gaussian softmax over flattened cells.

    Computes ``K = softmax_c(-||c - s||^2 / 2σ²)`` per shot, then the weighted sum over
    shots. For an isotropic Gaussian this equals the separable form by rank-1
    factorization. Returns image-layout ``(B, n_y * n_x)``.
    """
    b, j, _ = coords.shape
    nx = int(xcenters.shape[0])
    ny = int(ycenters.shape[0])
    # Cell centers in image layout (c = iy*nx + ix → (cx, cy)).
    cx_grid = xcenters.view(1, nx).expand(ny, nx).reshape(ny * nx)
    cy_grid = ycenters.view(ny, 1).expand(ny, nx).reshape(ny * nx)
    cells = torch.stack([cx_grid, cy_grid], dim=-1)  # (n_cells, 2)

    inv_two_sigma_sq = (1.0 / (2.0 * sigma.pow(2))).view(b, 1, 1)
    # ||c - s||² per (b, j, c).
    diff = cells.view(1, 1, ny * nx, 2) - coords.view(b, j, 1, 2)
    dist_sq = (diff * diff).sum(dim=-1)  # (B, J, n_cells)
    log_k = -dist_sq * inv_two_sigma_sq
    log_k = torch.log_softmax(log_k, dim=-1)  # per-shot normalize over cells
    k = log_k.exp()
    q = (weights.view(b, j, 1) * k).sum(dim=1)
    return q


# ---------------------------------------------------------------------------
# Equivalence tests at B=2, L=3, R=4, n_x=8, n_y=7
# ---------------------------------------------------------------------------


def _user_spec_inputs(
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (coords, weights, sigma, xcenters, ycenters) at shape
    ``B=2, L=3, R=4, n_x=8, n_y=7`` with ``J = L*R = 12``."""
    rng = np.random.default_rng(seed)
    b, j = 2, 12
    nx, ny = 8, 7
    coords = torch.from_numpy(
        np.stack(
            [
                rng.uniform(-15.0, 15.0, size=(b, j)),
                rng.uniform(-2.0, 30.0, size=(b, j)),
            ],
            axis=-1,
        ).astype(np.float32)
    )
    weights = torch.from_numpy(rng.uniform(0.05, 1.0, size=(b, j)).astype(np.float32))
    sigma = torch.from_numpy(rng.uniform(0.8, 3.0, size=(b,)).astype(np.float32))
    xcenters = torch.linspace(-20.0, 20.0, nx, dtype=torch.float32)
    ycenters = torch.linspace(-5.0, 35.0, ny, dtype=torch.float32)
    return coords, weights, sigma, xcenters, ycenters


def test_matches_dense_separable_reference_on_user_spec_shapes() -> None:
    """``separable_gaussian_density`` matches a direct ``K_x ⊗ K_y`` reference.

    The ``bmm``-based aggregation evaluates the outer product without materializing the
    ``(B, J, n_x, n_y)`` tensor.
    """
    coords, weights, sigma, xc, yc = _user_spec_inputs()
    got = separable_gaussian_density(coords, weights, sigma, xc, yc)
    expected = _dense_separable_reference(coords, weights, sigma, xc, yc)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_matches_full_2d_softmax_reference_on_user_spec_shapes() -> None:
    """``separable_gaussian_density`` matches the per-shot 2-D Gaussian softmax.

    The isotropic Gaussian factors exactly across (x, y), so per-axis normalization
    equals per-shot 2-D normalization.
    """
    coords, weights, sigma, xc, yc = _user_spec_inputs()
    got = separable_gaussian_density(coords, weights, sigma, xc, yc)
    expected = _dense_full2d_reference(coords, weights, sigma, xc, yc)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Behavior under varying shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("b", "j", "nx", "ny"),
    [(1, 1, 4, 5), (3, 10, 16, 12), (8, 50, 32, 28)],
)
def test_matches_dense_reference_across_shapes(b: int, j: int, nx: int, ny: int) -> None:
    rng = np.random.default_rng(b * 100 + j)
    coords = torch.from_numpy(
        np.stack(
            [
                rng.uniform(-15.0, 15.0, size=(b, j)),
                rng.uniform(-2.0, 30.0, size=(b, j)),
            ],
            axis=-1,
        ).astype(np.float32)
    )
    weights = torch.from_numpy(rng.uniform(0.0, 1.0, size=(b, j)).astype(np.float32))
    sigma = torch.from_numpy(rng.uniform(0.8, 3.0, size=(b,)).astype(np.float32))
    xcenters = torch.linspace(-20.0, 20.0, nx, dtype=torch.float32)
    ycenters = torch.linspace(-5.0, 35.0, ny, dtype=torch.float32)

    got = separable_gaussian_density(coords, weights, sigma, xcenters, ycenters)
    expected = _dense_separable_reference(coords, weights, sigma, xcenters, ycenters)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Row-sum invariant: per-batch sum equals sum of weights.
# ---------------------------------------------------------------------------


def test_per_batch_sum_equals_sum_of_weights() -> None:
    """Because each shot's K_x ⊗ K_y sums to 1 over the cells, the
    aggregated density's row sum equals the weight total per batch row.
    """
    coords, weights, sigma, xc, yc = _user_spec_inputs()
    q = separable_gaussian_density(coords, weights, sigma, xc, yc)
    np.testing.assert_allclose(
        q.sum(dim=-1).numpy(),
        weights.sum(dim=-1).numpy(),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Image-layout cell ordering: a single shot peaks at its nearest cell.
# ---------------------------------------------------------------------------


def test_single_shot_peaks_at_nearest_cell_in_image_layout() -> None:
    """A single shot at a cell center peaks at flat index ``c = iy*nx + ix``
    (image-layout C-order ravel)."""
    nx, ny = 8, 7
    xcenters = torch.linspace(-15.0, 15.0, nx, dtype=torch.float32)
    ycenters = torch.linspace(-3.0, 30.0, ny, dtype=torch.float32)
    # Pick (ix=5, iy=3) as the target peak.
    ix_peak, iy_peak = 5, 3
    coords = torch.tensor(
        [[[float(xcenters[ix_peak]), float(ycenters[iy_peak])]]], dtype=torch.float32
    )  # (B=1, J=1, 2)
    weights = torch.ones(1, 1, dtype=torch.float32)
    sigma = torch.tensor([0.5], dtype=torch.float32)
    q = separable_gaussian_density(coords, weights, sigma, xcenters, ycenters)  # (1, n_cells)
    cell_peak = int(q.argmax(dim=-1).item())
    expected_cell = iy_peak * nx + ix_peak
    assert cell_peak == expected_cell, (
        f"expected peak at c = iy*nx + ix = {iy_peak}*{nx} + {ix_peak} = "
        f"{expected_cell}, got c = {cell_peak}"
    )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_coords_weights_and_sigma() -> None:
    """``coords``, ``weights`` and ``sigma`` all receive gradients."""
    coords, weights, sigma, xc, yc = _user_spec_inputs()
    coords = coords.clone().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)
    sigma = sigma.clone().requires_grad_(True)
    q = separable_gaussian_density(coords, weights, sigma, xc, yc)
    loss = q.sum()
    loss.backward()
    assert coords.grad is not None and coords.grad.abs().sum() > 0
    assert weights.grad is not None and weights.grad.abs().sum() > 0
    assert sigma.grad is not None and sigma.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_wrong_coords_rank() -> None:
    weights = torch.ones(2, 3)
    sigma = torch.ones(2)
    xc = torch.zeros(4)
    yc = torch.zeros(5)
    with pytest.raises(ValueError, match="coords"):
        separable_gaussian_density(
            torch.zeros(2, 3),  # 2-D, not 3-D
            weights,
            sigma,
            xc,
            yc,
        )


def test_rejects_weights_shape_mismatch() -> None:
    coords = torch.zeros(2, 3, 2)
    sigma = torch.ones(2)
    xc = torch.zeros(4)
    yc = torch.zeros(5)
    with pytest.raises(ValueError, match="weights"):
        separable_gaussian_density(coords, torch.ones(2, 4), sigma, xc, yc)


def test_rejects_sigma_shape_mismatch() -> None:
    coords = torch.zeros(2, 3, 2)
    weights = torch.ones(2, 3)
    xc = torch.zeros(4)
    yc = torch.zeros(5)
    with pytest.raises(ValueError, match="sigma"):
        separable_gaussian_density(coords, weights, torch.ones(3), xc, yc)


def test_rejects_non_1d_xy_centers() -> None:
    coords = torch.zeros(2, 3, 2)
    weights = torch.ones(2, 3)
    sigma = torch.ones(2)
    with pytest.raises(ValueError, match="xcenters"):
        separable_gaussian_density(coords, weights, sigma, torch.zeros(4, 2), torch.zeros(5))
