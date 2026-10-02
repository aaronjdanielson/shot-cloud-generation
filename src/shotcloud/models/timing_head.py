"""48-bin softmax timing head ``ρ_η(t | x_n)``.

Models per-shot timing as a categorical distribution over a fixed
discretization of game time

.. math::

    \\rho_\\eta(t \\mid x_n) = \\mathrm{softmax}_t\\bigl[a(t) + b_\\eta(t, x_n)\\bigr],

with two factors:

* **Baseline pacing profile** :math:`a(t) \\in \\mathbb R^{n_{\\mathrm{bins}}}` —
  a single learnable per-bin offset, independent of context. Captures
  the league-mean timing distribution (e.g. higher density right
  before the half / end of the game).
* **Context-dependent residual** :math:`b_\\eta(t, x_n) \\in
  \\mathbb R^{n_{\\mathrm{bins}}}` — a small MLP that lets the
  distribution shift with starter status, role, recent usage, etc.
  These signals enter through :math:`x_n` rather than through
  separately engineered modules.

**Zero-init invariant.** The residual MLP's final layer is
zero-initialized by default (``zero_init_residual=True``), so at
initialization the timing distribution is exactly
:math:`\\mathrm{softmax}(a(t))`, the baseline alone. Since
:math:`a(t) = \\mathbf 0` initially as well, the head starts as the
uniform distribution over bins.

The default 48 bins give one bin per minute of regulation
(4 quarters × 12 minutes). Override ``n_bins`` for other temporal
resolutions (e.g. quarter-only, half-minute).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from shotcloud.data.context import CONTEXT_DIM


class TimingSoftmaxHead(nn.Module):
    """Discrete-bin softmax timing head ``ρ_η(t | x_n)``.

    Parameters
    ----------
    n_bins : int, default 48
        Number of timing bins. Default = one bin per game minute
        (4 × 12).
    context_dim : int, default :data:`CONTEXT_DIM` (27)
        Input dimension of the learned context vector.
    hidden_dim : int, default 32
        Hidden width of the residual MLP, kept small like
        :class:`~shotcloud.models.ContextResidualEncoder` and
        :class:`~shotcloud.models.NegBinCountHead`.
    zero_init_residual : bool, default True
        If True, the residual MLP's final layer (weight and bias) is
        zero-initialized so timing starts as ``softmax(a(t))``. If
        False, the final weights are drawn from a normal distribution
        with standard deviation ``1 / sqrt(n_bins)``.
    """

    def __init__(
        self,
        n_bins: int = 48,
        context_dim: int = CONTEXT_DIM,
        hidden_dim: int = 32,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.n_bins = n_bins
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.zero_init_residual = zero_init_residual

        # a(t) — baseline pacing profile.
        self.bin_baseline = nn.Parameter(torch.zeros(n_bins))

        # b_η(t, x_n) — context-dependent residual.
        self.fc1 = nn.Linear(context_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_bins)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        if zero_init_residual:
            # Zero-init the residual so the timing distribution begins as
            # softmax(a(t)). There is no zero saddle: ∂L/∂a(t) comes
            # directly from the softmax cross-entropy, and fc2.weight
            # receives a gradient through the nonzero hidden activations.
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            nn.init.normal_(self.fc2.weight, std=1.0 / math.sqrt(n_bins))
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x_n: Tensor) -> Tensor:
        """Return per-row log-probabilities over ``n_bins`` bins.

        Parameters
        ----------
        x_n : Tensor of shape ``(B, context_dim)``
            Learned context vector.

        Returns
        -------
        Tensor of shape ``(B, n_bins)``
            Log-probabilities; each row exponentiates to a distribution
            summing to 1.
        """
        if x_n.dim() != 2 or x_n.shape[1] != self.context_dim:
            raise ValueError(
                f"x_n must have shape (B, context_dim={self.context_dim}); got {tuple(x_n.shape)}"
            )
        h = torch.nn.functional.gelu(self.fc1(x_n))
        residual = self.fc2(h)  # (B, n_bins)
        logits = self.bin_baseline.unsqueeze(0) + residual  # (B, n_bins)
        return torch.log_softmax(logits, dim=-1)

    def log_prob(self, t_bin: Tensor, x_n: Tensor) -> Tensor:
        """Differentiable log-likelihood ``log ρ_η(t_bin | x_n)``.

        Parameters
        ----------
        t_bin : Tensor of integer dtype, shape ``(B,)``
            Per-row timing-bin index in ``[0, n_bins)``.
        x_n : Tensor of float, shape ``(B, context_dim)``
            Learned context vector.

        Returns
        -------
        Tensor of float, shape ``(B,)``
        """
        if t_bin.dim() != 1 or t_bin.shape[0] != x_n.shape[0]:
            raise ValueError(
                f"t_bin must have shape (B,) matching x_n; "
                f"got t_bin {tuple(t_bin.shape)}, x_n {tuple(x_n.shape)}"
            )
        log_p = self.forward(x_n)  # (B, n_bins)
        return log_p.gather(-1, t_bin.unsqueeze(-1).long()).squeeze(-1)

    def extra_repr(self) -> str:
        return (
            f"n_bins={self.n_bins}, context_dim={self.context_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"zero_init_residual={self.zero_init_residual}"
        )
