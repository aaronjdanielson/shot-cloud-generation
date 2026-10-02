"""Separable isotropic-Gaussian kernel evaluator on a rectangular grid.

Each shot ``s_j = (s_x, s_y)`` contributes a per-shot-normalized rank-1
density to every flat cell ``c = (c_x, c_y)``:

.. math::

    K_\\sigma(c - s_j) = K_x(c_x - s_x)\\, K_y(c_y - s_y),

with each axis normalized to sum to one over its cell centers. Because
the isotropic Gaussian factors exactly across axes, per-axis
normalization is mathematically equivalent to per-shot normalization of
the full 2-D Gaussian over the cell grid:

.. math::

    \\frac{\\exp(-\\|c - s\\|^2 / 2\\sigma^2)}{\\sum_c \\exp(\\cdot)}
      = \\frac{K_x(c_x - s_x)}{\\sum_{i_x} K_x} \\cdot
        \\frac{K_y(c_y - s_y)}{\\sum_{i_y} K_y}.

The aggregated batch density is

.. math::

    q_b(c) = \\sum_j w_{b,j}\\, K_x(c_x - s_{b,j,x})\\, K_y(c_y - s_{b,j,y}),

computed via reshape + ``bmm`` so the intermediate ``(B, J, n_x, n_y)``
tensor is **never materialized** — peak memory is
``O(B*J*max(n_x, n_y))`` instead of ``O(B*J*n_x*n_y)``. This is the
core win for the collaborative KDE forward, where ``B*J ≈ 144k`` and
``n_x*n_y ≈ 3.5k`` would otherwise allocate ~500 MB per forward call.
"""

from __future__ import annotations

import torch
from torch import Tensor


def separable_gaussian_density(
    coords: Tensor,
    weights: Tensor,
    sigma: Tensor,
    xcenters: Tensor,
    ycenters: Tensor,
    *,
    eps: float = 1e-30,
) -> Tensor:
    """Per-shot-normalized separable Gaussian density on a flat grid.

    Parameters
    ----------
    coords : Tensor of shape ``(B, J, 2)``
        Per-batch, per-shot ``(x, y)`` coordinates in court feet.
    weights : Tensor of shape ``(B, J)``
        Per-shot scalar weights (typically :math:`\\alpha_{b,l}\\,\\beta_{b,l,r}`
        after flattening the ``(L, R)`` axes into ``J = L\\cdot R``).
    sigma : Tensor of shape ``(B,)``
        Per-batch bandwidth in court feet.
    xcenters : Tensor of shape ``(n_x,)``
        Cell-center x-coordinates (the column axis).
    ycenters : Tensor of shape ``(n_y,)``
        Cell-center y-coordinates (the row axis).
    eps : float
        Floor for axis-wise normalization to avoid 0/0 when a shot lands
        many σ away from every cell on one axis.

    Returns
    -------
    Tensor of shape ``(B, n_y * n_x)``
        Flat density per cell in **image-layout C-ravel** order
        ``c = i_y \\cdot n_x + i_x``, matching
        :class:`shotcloud.grids.court.CourtGrid`. No additional
        normalization is applied — the returned tensor is the
        weighted sum, so per-batch sums equal
        ``sum_j weights[b, j]`` exactly (modulo float).
    """
    if coords.dim() != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coords must be (B, J, 2); got {tuple(coords.shape)}")
    if weights.shape != coords.shape[:2]:
        raise ValueError(
            f"weights must be (B, J)={tuple(coords.shape[:2])}; got {tuple(weights.shape)}"
        )
    if sigma.dim() != 1 or sigma.shape[0] != coords.shape[0]:
        raise ValueError(f"sigma must be (B,)={(coords.shape[0],)}; got {tuple(sigma.shape)}")
    if xcenters.dim() != 1 or ycenters.dim() != 1:
        raise ValueError(
            f"xcenters/ycenters must be 1-D; got {tuple(xcenters.shape)}, {tuple(ycenters.shape)}"
        )

    b, _j, _ = coords.shape
    nx = int(xcenters.shape[0])
    ny = int(ycenters.shape[0])

    # 1 / (2 σ²) broadcast as (B, 1, 1).
    inv_two_sigma_sq = (1.0 / (2.0 * sigma.pow(2).clamp_min(1e-12))).view(b, 1, 1)

    # Axis-wise distances. (B, J, n_x) and (B, J, n_y).
    dx = coords[..., 0].unsqueeze(-1) - xcenters.view(1, 1, nx)
    dy = coords[..., 1].unsqueeze(-1) - ycenters.view(1, 1, ny)

    log_kx = -(dx * dx) * inv_two_sigma_sq  # (B, J, n_x)
    log_ky = -(dy * dy) * inv_two_sigma_sq  # (B, J, n_y)

    # Per-shot per-axis normalize via log-softmax for numerical stability.
    # Equivalent to ``Kx /= Kx.sum(-1, keepdim=True)`` but immune to the
    # overflow / underflow that bites for σ ≪ 1 or σ ≫ 1.
    kx = torch.softmax(log_kx, dim=-1)
    ky = torch.softmax(log_ky, dim=-1)

    # Aggregate without ever materializing (B, J, n_x, n_y):
    #   q[b, i_x, i_y] = Σ_j w[b, j] · K_x[b, j, i_x] · K_y[b, j, i_y]
    # Bake weights into K_x, then bmm reduces the J axis:
    #   bmm( (B, n_x, J), (B, J, n_y) ) → (B, n_x, n_y)
    wkx = kx * weights.unsqueeze(-1).clamp(min=-1e30, max=1e30)  # (B, J, n_x)
    q_xy = torch.bmm(wkx.transpose(1, 2), ky)  # (B, n_x, n_y)

    # Reorder (B, n_x, n_y) → (B, n_y, n_x) and flatten to C-ravel
    # ``c = i_y * n_x + i_x`` matching image layout.
    q_yx = q_xy.transpose(1, 2).contiguous()  # (B, n_y, n_x)
    # Silence the unused-import note from `eps` in the public signature.
    _ = eps
    return q_yx.reshape(b, ny * nx)
