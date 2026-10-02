"""Sample shot locations from the AC-KDE kernel mixture.

For a row whose support is :math:`\\{(s_j, \\omega_j)\\}_{j=1}^M`
under bandwidth :math:`\\sigma`, the per-shot generative density is

.. math::

    f_\\Theta(y) = \\sum_j \\omega_j\\,\\mathcal N_2(y;\\,s_j,\\sigma^2 I).

A single sample is the ancestral pair

.. math::

    J \\sim \\operatorname{Categorical}(\\omega_1,\\dots,\\omega_M),
    \\qquad
    Y \\mid J=j \\sim \\mathcal N_2(s_j,\\sigma^2 I).

:func:`sample_locations` draws these pairs in batch, with rejection
sampling against the court rectangle. Samples still off the court after
``max_attempts`` redraws are clipped to the court bounds, so every
returned location lies on the court.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Default court extents in feet, matching the default
#: :class:`~shotcloud.grids.CourtGrid` (basket at the origin,
#: x ∈ [-25, 25], y ∈ [-5, 47]).
DEFAULT_COURT_XLIM: tuple[float, float] = (-25.0, 25.0)
DEFAULT_COURT_YLIM: tuple[float, float] = (-5.0, 47.0)


def sample_locations(
    support_xy: Tensor,
    log_omega: Tensor,
    support_mask: Tensor,
    sigma: Tensor,
    n_samples: int,
    *,
    court_xlim: tuple[float, float] = DEFAULT_COURT_XLIM,
    court_ylim: tuple[float, float] = DEFAULT_COURT_YLIM,
    max_attempts: int = 8,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``n_samples`` shot locations per batch row.

    Parameters
    ----------
    support_xy : Tensor of shape ``(B, M, 2)``
        Support shot coordinates per row.
    log_omega : Tensor of shape ``(B, M)``
        Log mixture weights, with invalid slots at ``-inf`` (as produced
        by the decoder's softmax). Rows with no mass, and cold-start rows
        whose only weight is a placeholder at slot 0, sample around
        ``support_xy[:, 0, :]``; callers should discard such rows.
    support_mask : Tensor of shape ``(B, M)``
        Valid-support mask. Only its shape is checked; the mass is
        carried by ``log_omega``.
    sigma : Tensor of shape ``(B,)``
        Per-row Gaussian bandwidth in feet (same units as
        ``support_xy``).
    n_samples : int
        Number of locations to draw per batch row.
    court_xlim, court_ylim : tuple[float, float]
        Court bounds for rejection sampling. Defaults match
        :class:`~shotcloud.grids.CourtGrid`.
    max_attempts : int, default 8
        Maximum number of redraw rounds for off-court samples. Each
        redraw keeps the sampled support shot and draws new Gaussian
        noise. Samples still off the court afterwards are clipped to the
        court bounds.
    generator : torch.Generator, optional
        Random generator for reproducible sampling.

    Returns
    -------
    Tensor of shape ``(B, n_samples, 2)``
        Sampled locations, all inside the court rectangle.

    Raises
    ------
    ValueError
        On inconsistent shapes, non-positive ``n_samples`` or empty court
        bounds.
    """
    if support_xy.dim() != 3 or support_xy.shape[-1] != 2:
        raise ValueError(f"support_xy must be (B, M, 2); got {tuple(support_xy.shape)}")
    b, m, _ = support_xy.shape
    if log_omega.shape != (b, m):
        raise ValueError(f"log_omega must be (B, M); got {tuple(log_omega.shape)}")
    if support_mask.shape != (b, m):
        raise ValueError(f"support_mask must be (B, M); got {tuple(support_mask.shape)}")
    if sigma.shape != (b,):
        raise ValueError(f"sigma must be (B,); got {tuple(sigma.shape)}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive; got {n_samples}")

    device = support_xy.device
    x_lo, x_hi = court_xlim
    y_lo, y_hi = court_ylim
    if not (x_lo < x_hi and y_lo < y_hi):
        raise ValueError(f"court bounds invalid: x={court_xlim}, y={court_ylim}")

    # Categorical sample J from the nonnegative weights ω = exp(log ω);
    # -inf slots get zero weight.
    omega = log_omega.exp()
    # Sampling requires positive total weight per row. Rows with none get
    # unit weight on slot 0, so they return a finite (meaningless) value
    # that the caller discards.
    row_sum = omega.sum(dim=-1)
    if (row_sum <= 0).any():
        omega = omega.clone()
        bad = row_sum <= 0
        omega[bad, 0] = 1.0

    # (B, n_samples) component indices, drawn with replacement. With a
    # generator, use inverse-CDF sampling, which honors the generator on
    # every PyTorch version; otherwise torch.multinomial.
    if generator is not None:
        probs = omega / omega.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cdf = probs.cumsum(dim=-1)  # (B, M)
        u = torch.rand(b, n_samples, device=device, generator=generator)
        idx = torch.searchsorted(cdf, u)  # (B, n_samples)
        idx = idx.clamp_max(m - 1)
    else:
        idx = torch.multinomial(omega, num_samples=n_samples, replacement=True)
    # Gather selected support coordinates: (B, n_samples, 2).
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, 2)
    centers = torch.gather(support_xy, 1, gather_idx)

    # Initial Gaussian draws around centers.
    sigma_b = sigma.view(b, 1, 1)
    if generator is not None:
        noise = torch.randn(b, n_samples, 2, device=device, generator=generator)
    else:
        noise = torch.randn(b, n_samples, 2, device=device)
    y = centers + sigma_b * noise

    # Rejection sampling for off-court points.
    def _bad(samples: Tensor) -> Tensor:
        return (
            (samples[..., 0] < x_lo)
            | (samples[..., 0] > x_hi)
            | (samples[..., 1] < y_lo)
            | (samples[..., 1] > y_hi)
        )

    for _ in range(max_attempts):
        bad_mask = _bad(y)
        if not bad_mask.any():
            break
        # Redraw only the bad slots.
        if generator is not None:
            new_noise = torch.randn(b, n_samples, 2, device=device, generator=generator)
        else:
            new_noise = torch.randn(b, n_samples, 2, device=device)
        candidate = centers + sigma_b * new_noise
        y = torch.where(bad_mask.unsqueeze(-1), candidate, y)

    # Clip any samples still off the court.
    y = torch.stack(
        [
            y[..., 0].clamp(min=x_lo, max=x_hi),
            y[..., 1].clamp(min=y_lo, max=y_hi),
        ],
        dim=-1,
    )
    return y


__all__ = [
    "DEFAULT_COURT_XLIM",
    "DEFAULT_COURT_YLIM",
    "sample_locations",
]
