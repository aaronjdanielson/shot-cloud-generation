"""Low-rank exponential-tilt decoder for the spatial mark model.

Deprecated; retained to reproduce the grid-cell decoder ablations. Superseded
by :class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`.

The decoder adds a low-rank tilt to a log base measure on the court grid:

.. math::

    \\ell_{n,i,c}
    = \\log q_0(c \\mid p_n, d_n) + u_{n,i}^\\top v_c,
    \\qquad
    p_\\theta(c_{n,i} = c \\mid \\cdot)
    = \\mathrm{softmax}_c(\\ell_{n,i,c}),

where ``u ∈ ℝ^r`` is the per-shot context vector (output of an upstream
encoder) and ``v_c ∈ ℝ^r`` is a learned spatial basis vector for cell
``c``, with rank ``r ≪ n_cells``. Training learns the basis matrix
``V ∈ ℝ^{n_cells × r}``.

**Critical invariant.** At zero-init (``V = 0``), the tilt term is
identically zero and the logits collapse to ``log q_0``. Because ``q_0``
is a probability that sums to 1, ``softmax(log q_0) = q_0`` exactly. The
decoder therefore reproduces the base measure at initialization,
which lets the model start training from a strong, statistically
principled prior — the neural correction only learns the contextual
deformation on top. ``zero_init=True`` (the default) selects this
initialization.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LowRankTiltDecoder(nn.Module):
    """Low-rank exponential tilt of a base measure on a discrete cell grid.

    Parameters
    ----------
    n_cells : int
        Number of court cells (output dimension). Typically the
        ``n_cells`` of the model's :class:`~shotcloud.grids.CourtGrid`.
    rank : int, default 8
        Latent rank ``r`` of the spatial basis ``V``. Smaller ``r`` →
        smoother / smaller corrections, more stable training.
    zero_init : bool, default True
        If ``True``, ``V`` is initialized to zeros. The decoder then
        reproduces the base measure exactly at initialization — the load-bearing
        invariant for stable training. If ``False``, ``V`` is randomly
        initialized with std ``1 / sqrt(rank)`` (Glorot-style).

    Forward
    -------
    ``forward(log_q0, u)`` returns logits ``ℓ`` of shape
    ``(batch, n_cells)``. Use :py:meth:`probs` or :py:meth:`log_probs`
    for the normalized output.
    """

    def __init__(self, n_cells: int, rank: int = 8, zero_init: bool = True) -> None:
        super().__init__()
        if n_cells <= 0:
            raise ValueError(f"n_cells must be positive, got {n_cells}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.n_cells = n_cells
        self.rank = rank
        self.zero_init = zero_init

        if zero_init:
            self.V = nn.Parameter(torch.zeros(n_cells, rank))
        else:
            init_std = 1.0 / math.sqrt(rank)
            self.V = nn.Parameter(torch.randn(n_cells, rank) * init_std)

    # ----------------------------------------------------------------------
    # Operations
    # ----------------------------------------------------------------------

    def tilt(self, u: Tensor) -> Tensor:
        """Compute the tilt term ``u^T v_c`` for every cell ``c``.

        Parameters
        ----------
        u : Tensor of shape ``(batch, rank)``

        Returns
        -------
        Tensor of shape ``(batch, n_cells)``.
        """
        if u.dim() != 2 or u.shape[1] != self.rank:
            raise ValueError(f"u must have shape (batch, rank={self.rank}); got {tuple(u.shape)}")
        return u @ self.V.T

    def forward(self, log_q0: Tensor, u: Tensor) -> Tensor:
        """Return un-normalized logits ``log q_0 + u @ V.T``.

        Parameters
        ----------
        log_q0 : Tensor of shape ``(batch, n_cells)``
            Log of the base measure for each example. Must be the log of
            a normalized probability vector (i.e., ``exp(log_q0).sum(-1) ≈ 1``)
            for the zero-init invariant to hold.
        u : Tensor of shape ``(batch, rank)``
            Per-shot context vectors from the upstream encoder.

        Returns
        -------
        Tensor of shape ``(batch, n_cells)``. **Not** softmax-normalized;
        use :py:meth:`probs` or :py:meth:`log_probs` for that.
        """
        if log_q0.dim() != 2 or log_q0.shape[1] != self.n_cells:
            raise ValueError(
                f"log_q0 must have shape (batch, n_cells={self.n_cells}); got {tuple(log_q0.shape)}"
            )
        if log_q0.shape[0] != u.shape[0]:
            raise ValueError(
                f"batch dim mismatch: log_q0 has {log_q0.shape[0]}, u has {u.shape[0]}"
            )
        return log_q0 + self.tilt(u)

    def log_probs(self, log_q0: Tensor, u: Tensor) -> Tensor:
        """``log_softmax(forward(log_q0, u))``. Numerically stable."""
        return torch.log_softmax(self.forward(log_q0, u), dim=-1)

    def probs(self, log_q0: Tensor, u: Tensor) -> Tensor:
        """``softmax(forward(log_q0, u))``."""
        return torch.softmax(self.forward(log_q0, u), dim=-1)

    def extra_repr(self) -> str:
        return f"n_cells={self.n_cells}, rank={self.rank}, zero_init={self.zero_init}"
