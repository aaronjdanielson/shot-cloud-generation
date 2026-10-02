"""Negative-binomial count head ``p_η^count(K_n | x_n)``.

Models the per-game shot-attempt count :math:`K_n` as

.. math::

    K_n \\sim \\mathrm{NegBin}\\bigl(\\mu_\\eta(x_n),\\,\\kappa\\bigr),

where :math:`\\mu_\\eta(x_n) = \\mathrm{softplus}(g_\\eta(x_n))` is a small
context-dependent MLP and :math:`\\kappa` is a single global learnable
dispersion. The negative binomial is preferred over a Poisson because
shot counts are overdispersed; :math:`\\kappa` controls how much: the
variance is :math:`\\mu + \\mu^2/\\kappa`, recovering the Poisson as
:math:`\\kappa \\to \\infty`.

Internally :math:`(\\mu, \\kappa)` is converted to torch's
``(total_count, probs)`` parameterization
(``total_count = κ``, ``probs = μ / (μ + κ)``), and the likelihood is
evaluated with :class:`torch.distributions.NegativeBinomial`.

The parameters are the MLP weights (one hidden layer, ``hidden_dim=32``
by default, deliberately small like the residual encoder) and the
scalar ``log_kappa``, with :math:`\\kappa = \\mathrm{softplus}(\\text{log\\_kappa})`.
The head consumes only the context vector; it has no player-identity
input.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import NegativeBinomial

from shotcloud.data.context import CONTEXT_DIM


class NegBinCountHead(nn.Module):
    """Per-game negative-binomial count head ``p(K_n | x_n)``.

    Parameters
    ----------
    context_dim : int, default :data:`CONTEXT_DIM` (27)
        Input dimension of the learned context vector.
    hidden_dim : int, default 32
        Hidden width of the ``μ_η`` MLP, kept small like
        :class:`~shotcloud.models.ContextResidualEncoder`.
    init_log_kappa : float, default 0.0
        Initial value of the raw dispersion parameter ``log_kappa``;
        ``κ = softplus(log_kappa)``, so the default gives
        ``κ ≈ 0.693``, a strongly overdispersed starting point. Larger
        values move toward the Poisson.
    init_mean : float or None, default None
        If set, initialize ``fc2.bias`` to ``log(exp(init_mean) - 1)``
        so that ``μ_η(x_n) ≈ init_mean`` at initialization for typical
        ``x_n``. Without it the initial mean is near
        ``softplus(0) ≈ 0.69``, far below realistic shot counts, and in
        joint training the count head can stay under-fitted while the
        spatial loss dominates. The training-set mean count is the
        natural value.
    """

    def __init__(
        self,
        context_dim: int = CONTEXT_DIM,
        hidden_dim: int = 32,
        init_log_kappa: float = 0.0,
        init_mean: float | None = None,
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if init_mean is not None and init_mean <= 0.0:
            raise ValueError(f"init_mean must be positive, got {init_mean}")

        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # μ_η(x_n) = softplus(g_η(x_n)) — single hidden-layer MLP.
        self.fc1 = nn.Linear(context_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        if init_mean is None:
            nn.init.zeros_(self.fc2.bias)
        else:
            # inv-softplus(y) = log(exp(y) - 1), via expm1 for stability.
            # At init g(x) ≈ fc2.bias (fc2.weight is small and h moderate),
            # so μ = softplus(g) ≈ init_mean.
            inv_softplus = math.log(math.expm1(init_mean))
            nn.init.constant_(self.fc2.bias, inv_softplus)

        # Global learnable dispersion shared across rows: κ is a single
        # scalar, not a function of x_n.
        self.log_kappa = nn.Parameter(torch.tensor(init_log_kappa))

    def forward(self, x_n: Tensor) -> tuple[Tensor, Tensor]:
        """Return the negative-binomial parameters ``(μ, κ)`` per row.

        Parameters
        ----------
        x_n : Tensor of shape ``(B, context_dim)``
            Learned context vector.

        Returns
        -------
        mu, kappa : Tensor of shape ``(B,)``
            Positive mean and dispersion; ``kappa`` is the global scalar
            broadcast to ``(B,)``.
        """
        if x_n.dim() != 2 or x_n.shape[1] != self.context_dim:
            raise ValueError(
                f"x_n must have shape (B, context_dim={self.context_dim}); got {tuple(x_n.shape)}"
            )
        h = torch.nn.functional.gelu(self.fc1(x_n))
        log_mu = self.fc2(h).squeeze(-1)  # (B,)
        mu = torch.nn.functional.softplus(log_mu)
        # ``.contiguous()`` is required: without it kappa is a stride-0
        # broadcast view of a scalar, and on the MPS backend
        # ``torch.distributions.NegativeBinomial.log_prob`` returns ±Inf
        # for nearly all batch entries when ``total_count`` has stride 0
        # (a PyTorch MPS issue; CPU and CUDA are unaffected).
        kappa = torch.nn.functional.softplus(self.log_kappa).expand_as(mu).contiguous()
        return mu, kappa

    def log_prob(self, K: Tensor, x_n: Tensor) -> Tensor:
        """Differentiable log-likelihood ``log P(K_n | x_n)``.

        Parameters
        ----------
        K : Tensor of integer or float dtype, shape ``(B,)``
            Observed shot counts. Cast to float internally; must be
            non-negative integers.
        x_n : Tensor of float, shape ``(B, context_dim)``
            Learned context vector.

        Returns
        -------
        Tensor of float, shape ``(B,)``
        """
        if K.dim() != 1 or K.shape[0] != x_n.shape[0]:
            raise ValueError(
                f"K must have shape (B,) matching x_n's first dim; "
                f"got K {tuple(K.shape)}, x_n {tuple(x_n.shape)}"
            )
        mu, kappa = self.forward(x_n)
        # Convert (μ, κ) → torch's (total_count=κ, probs=μ/(μ+κ)).
        # Clamp probs strictly inside (0, 1) for backward stability:
        # the NegBin log_prob has a ``value * log(probs)`` term whose
        # gradient ``value / probs`` blows up to ``+inf`` as probs → 0,
        # which then multiplies by ``d(probs)/d(kappa) ≈ 0`` in the
        # small-μ limit to produce ``inf * 0 = NaN`` flowing back into
        # ``log_kappa``. Clamping at both ends prevents this without
        # changing the forward value at sensible operating points.
        probs = mu / (mu + kappa)
        probs = probs.clamp(min=1e-7, max=1.0 - 1e-7)
        dist = NegativeBinomial(total_count=kappa, probs=probs)
        result: Tensor = dist.log_prob(K.to(mu.dtype))  # type: ignore[no-untyped-call]
        return result

    def extra_repr(self) -> str:
        return (
            f"context_dim={self.context_dim}, hidden_dim={self.hidden_dim}, "
            f"log_kappa={self.log_kappa.item():.3f}"
        )
