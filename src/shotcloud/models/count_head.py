"""Negative-binomial count head ``p_η^count(K_n | x_n)`` (paper §5).

Models the per-game shot-attempt count :math:`K_n` as

.. math::

    K_n \\sim \\mathrm{NegBin}\\bigl(\\mu_\\eta(x_n),\\,\\kappa\\bigr),

where :math:`\\mu_\\eta(x_n) = \\mathrm{softplus}(g_\\eta(x_n))` is a small
context-dependent MLP and :math:`\\kappa` is a single global learnable
dispersion. The negative binomial is the natural choice over a
Poisson because shot-count data is overdispersed; the parameter
:math:`\\kappa` controls how much: variance is :math:`\\mu + \\mu^2/\\kappa`
in this parameterization, recovering Poisson as :math:`\\kappa \\to \\infty`.

Internally we convert :math:`(\\mu, \\kappa)` to torch's
``(total_count, probs)`` parameterization
(``total_count = κ``, ``probs = μ / (μ + κ)``) and call
:class:`torch.distributions.NegativeBinomial.log_prob` for a
numerically-stable gradient-friendly likelihood.

Module-owned parameters: the MLP weights (one hidden layer,
``hidden_dim=32`` by default — capacity-limited like the residual
encoder) and the scalar ``log_kappa``. Forward returns a per-row
``(μ, κ)`` pair; :meth:`log_prob` returns a per-row log-likelihood.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import NegativeBinomial

from shotcloud.data.context import CONTEXT_DIM


class NegBinCountHead(nn.Module):
    """Per-game NegBin count head ``p(K_n | x_n)`` (paper §5).

    Parameters
    ----------
    context_dim : int, default :data:`CONTEXT_DIM` (27)
        Input dimension of the learned context vector.
    hidden_dim : int, default 32
        Hidden width of the ``μ_η`` MLP. Same capacity discipline as
        :class:`ContextResidualEncoder`.
    init_log_kappa : float, default 0.0
        Initial value of the learnable ``log_kappa`` parameter.
        ``softplus(0) ≈ 0.693`` gives a moderately overdispersed
        starting point; larger values recover Poisson, smaller values
        produce heavier-tailed counts.
    init_mean : float | None, default ``None``
        If set, warm-start ``fc2.bias`` to ``log(exp(init_mean) - 1)``
        so that at initialization ``μ_η(x_n) ≈ init_mean`` for typical
        ``x_n``. Mitigates the under-trained-count-head failure mode
        observed in 20-epoch joint runs where the spatial gradient
        dominates and ``μ`` stays near ``softplus(0) ≈ 0.69``. Pass
        the training-set ``K̄`` (≈ 9.6 for the current NBA corpus).

    Forward
    -------
    ``forward(x_n) -> (μ, κ)`` — both shape ``(B,)``, both positive.
    ``κ`` is broadcast to ``(B,)`` from a single learnable scalar.

    log_prob
    --------
    ``log_prob(K, x_n) -> (B,)`` differentiable per-row log-likelihood.
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
            # inv-softplus(y) = log(exp(y) - 1). Use expm1 for stability.
            # At init `g(x) ≈ fc2.bias` (since fc2.weight is small kaiming
            # and h is moderate), so μ = softplus(g) ≈ init_mean.
            inv_softplus = math.log(math.expm1(init_mean))
            nn.init.constant_(self.fc2.bias, inv_softplus)

        # Global learnable log-dispersion. Shared across rows; the
        # paper writes κ as a single scalar parameter, not κ(x_n).
        # TODO(if validation shows it's needed): per-context κ via
        # a second MLP head.
        self.log_kappa = nn.Parameter(torch.tensor(init_log_kappa))

    def forward(self, x_n: Tensor) -> tuple[Tensor, Tensor]:
        if x_n.dim() != 2 or x_n.shape[1] != self.context_dim:
            raise ValueError(
                f"x_n must have shape (B, context_dim={self.context_dim}); got {tuple(x_n.shape)}"
            )
        h = torch.nn.functional.gelu(self.fc1(x_n))
        log_mu = self.fc2(h).squeeze(-1)  # (B,)
        mu = torch.nn.functional.softplus(log_mu)
        # ``.contiguous()`` is load-bearing here: without it, kappa is a
        # stride-0 broadcast view of a scalar, and on the MPS backend
        # ``torch.distributions.NegativeBinomial.log_prob`` produces ±Inf
        # for ~all batch entries when ``total_count`` has stride 0
        # (PyTorch MPS bug; CPU and CUDA are unaffected). Materializing a
        # contiguous (B,) tensor is the load-bearing workaround.
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

        Returns
        -------
        Tensor of float, shape ``(B,)``.
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
