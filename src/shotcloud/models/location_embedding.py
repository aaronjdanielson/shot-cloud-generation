r"""Continuous-coordinate location embedding :math:`\psi(s)`.

The residual tilt of the continuous-mixture decoder,
:math:`R_\theta(s_m) = u_\theta^\top \psi(s_m)`, must be evaluated at
arbitrary court coordinates rather than at grid-cell centers.
:class:`LocationEmbedding` provides a smooth map
:math:`\psi : \mathbb R^2 \to \mathbb R^{r}`:

1. scale the coordinates, :math:`\tilde x = x / x_{\text{scale}}` and
   :math:`\tilde y = y / y_{\text{scale}}`;
2. concatenate the scaled coordinates with Fourier features
   :math:`\sin(f \tilde x), \cos(f \tilde x), \sin(f \tilde y),
   \cos(f \tilde y)` for each frequency :math:`f`;
3. project to rank :math:`r` with a single linear layer.

The projection is zero-initialized, so :math:`\psi \equiv 0` and the
residual tilt vanishes at initialization. Because the paired
:class:`~shotcloud.models.context_residual.ContextResidualEncoder` starts
from a small nonzero output, the projection still receives a nonzero
gradient and can move away from zero.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

#: Default frequencies for the Fourier feature bank: powers of 2. With
#: the default scales the wavelengths run from well beyond the court
#: width (f = 1) down to about 10 ft (f = 16), so the residual captures
#: zone-scale corrections rather than fine detail.
_DEFAULT_FREQUENCIES: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0, 16.0)


class LocationEmbedding(nn.Module):
    r"""Fourier-feature coordinate embedding :math:`\psi : \mathbb R^2 \to \mathbb R^{r}`.

    Parameters
    ----------
    rank : int
        Output dimension :math:`r`. Must match the rank of the paired
        :class:`~shotcloud.models.ContextResidualEncoder`.
    n_frequencies : int, default 5
        Number of Fourier frequencies. The first five are
        ``(1, 2, 4, 8, 16)``; larger values extend the sequence
        ``2**k``. Each frequency contributes a sine and a cosine for both
        coordinates, so the feature width is ``2 + 4 * n_frequencies``.
    x_scale, y_scale : float, default 25.0, 26.0
        Divisors applied to ``x`` and ``y`` (feet) before featurization.
        The defaults are the half-extents of the default court grid
        (``x ∈ [-25, 25]``, ``y ∈ [-5, 47]``); coordinates are scaled,
        not centered.
    zero_init : bool, default True
        Zero-initialize the output projection so that :math:`\psi \equiv 0`
        and the residual tilt vanishes at initialization.

    Raises
    ------
    ValueError
        If ``rank``, ``x_scale`` or ``y_scale`` is non-positive, or
        ``n_frequencies`` is negative.
    """

    freqs: Tensor

    def __init__(
        self,
        rank: int,
        n_frequencies: int = 5,
        x_scale: float = 25.0,
        y_scale: float = 26.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if n_frequencies < 0:
            raise ValueError(f"n_frequencies must be non-negative, got {n_frequencies}")
        if x_scale <= 0 or y_scale <= 0:
            raise ValueError(f"x_scale and y_scale must be positive; got {x_scale}, {y_scale}")

        self.rank = int(rank)
        self.n_frequencies = int(n_frequencies)
        self.x_scale = float(x_scale)
        self.y_scale = float(y_scale)

        # Frequencies as a non-learnable buffer.
        freqs_t = torch.tensor(
            _DEFAULT_FREQUENCIES[:n_frequencies]
            if n_frequencies <= len(_DEFAULT_FREQUENCIES)
            else [2.0**k for k in range(n_frequencies)],
            dtype=torch.float32,
        )
        self.register_buffer("freqs", freqs_t, persistent=False)

        # Raw feature dim: linear (x, y) + (sin, cos) × (x, y) × n_freqs.
        raw_dim = 2 + 4 * n_frequencies
        self.proj = nn.Linear(raw_dim, rank)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, xy: Tensor) -> Tensor:
        """Map ``(..., 2)`` court coordinates to ``(..., rank)`` embeddings.

        Parameters
        ----------
        xy : Tensor of shape ``(..., 2)``
            Court coordinates in feet. Any leading shape is supported;
            the typical input is a ``(B, M, 2)`` support set.

        Returns
        -------
        Tensor of shape ``(..., rank)``

        Raises
        ------
        ValueError
            If the last dimension of ``xy`` is not 2.
        """
        if xy.shape[-1] != 2:
            raise ValueError(f"xy last dim must be 2; got shape {tuple(xy.shape)}")
        x_norm = xy[..., 0:1] / self.x_scale
        y_norm = xy[..., 1:2] / self.y_scale
        if self.n_frequencies > 0:
            phases_x = self.freqs * x_norm  # (..., n_freqs) via broadcasting
            phases_y = self.freqs * y_norm
            features = torch.cat(
                [
                    x_norm,
                    y_norm,
                    torch.sin(phases_x),
                    torch.cos(phases_x),
                    torch.sin(phases_y),
                    torch.cos(phases_y),
                ],
                dim=-1,
            )
        else:
            features = torch.cat([x_norm, y_norm], dim=-1)
        out: Tensor = self.proj(features)
        return out

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, n_frequencies={self.n_frequencies}, "
            f"x_scale={self.x_scale}, y_scale={self.y_scale}"
        )


__all__ = ["LocationEmbedding"]
