"""Learnable scalar ``α_def`` for the Phase-2 defensive product factor.

Mirrors :class:`~shotcloud.models.LearnableTemperature` in shape: a
single softplus-positive scalar applied multiplicatively to the log of
the defensive density:

.. math::

    \\alpha_{\\text{def}} = \\mathrm{softplus}(\\theta_d), \\qquad
    \\ell \\mathrel{+}= \\alpha_{\\text{def}} \\cdot
        \\log \\hat q_{\\text{def}}(c \\mid \\text{opp}).

The decoder's softmax in ``c`` absorbs the partition function (just
like for the temperature on the offensive prior), so implementation
reduces to a scalar multiply per minibatch.

Initialization defaults to ``α_def = 0.5`` per
[docs/research_plan.md](../../../docs/research_plan.md) §5 — small
enough that the defensive contribution starts subdominant to the
offensive prior, large enough that the gradient signal on ``θ_d`` is
non-trivial. Setting ``init=ε`` close to zero recovers the Phase-1
model at step 0 and is useful for ablations that strictly generalize
the previous phase.

Stacks orthogonally with :class:`LearnableTemperature` and
:class:`LearnableKDEProductWeights` — defense is a different signal
from any offensive prior.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from shotcloud.legacy.learnable_weights import _invert_softplus
from shotcloud.legacy_pivot._context_mlp import _ContextMLP

DEFAULT_INIT_ALPHA_DEF: float = 0.5


class LearnableDefensiveScale(nn.Module):
    """Learnable defensive scale ``α_def`` applied to ``log q_def``.

    Two modes mirror :class:`LearnableTemperature`:

    * **Scalar mode** (``context_dim=0``, default, Phase-2 configuration).
      Owns one parameter ``θ_d``; ``α_def = softplus(θ_d)`` is global.
    * **Context-conditioned mode** (Phase 3, ``context_dim>0``). Owns a
      tiny MLP; ``α_def(x_n) = softplus(MLP(x_n))``.

    Parameters
    ----------
    init : float, default 0.5
        Initial value. Strictly positive. Step-0 puts defense
        subdominant to the offensive prior in scalar mode; in
        context-conditioned mode the MLP is initialized so its output
        ≈ ``init`` for typical inputs.
    context_dim : int, default 0
    mlp_hidden : int, default 8
    """

    context_dim: int

    def __init__(
        self,
        init: float = DEFAULT_INIT_ALPHA_DEF,
        context_dim: int = 0,
        mlp_hidden: int = 8,
    ) -> None:
        super().__init__()
        if init <= 0:
            raise ValueError(f"init must be strictly positive (softplus is positive); got {init}")
        if context_dim < 0:
            raise ValueError(f"context_dim must be non-negative, got {context_dim}")

        self.context_dim = context_dim
        self.init_value = init
        if context_dim == 0:
            self.theta = nn.Parameter(torch.tensor(_invert_softplus(init)))
            self.context_mlp: _ContextMLP | None = None
        else:
            self.theta = None  # type: ignore[assignment]
            self.context_mlp = _ContextMLP(
                context_dim=context_dim, hidden=mlp_hidden, init_value=init
            )

    def alpha(self, x_n: Tensor | None = None) -> Tensor:
        """Current ``α_def`` (graph-attached). Vector ``(B,)`` in context mode."""
        if self.context_dim == 0:
            assert self.theta is not None
            return F.softplus(self.theta)
        if x_n is None:
            raise ValueError("context-conditioned mode requires x_n")
        assert self.context_mlp is not None
        return F.softplus(self.context_mlp(x_n))

    def alpha_as_float(self, x_n: Tensor | None = None) -> float:
        with torch.no_grad():
            a = self.alpha(x_n)
            return float(a.mean()) if a.dim() > 0 else float(a)

    def forward(self, log_q_def: Tensor, x_n: Tensor | None = None) -> Tensor:
        """Return ``α_def · log q_def`` (or ``α_def(x_n) · log q_def``).

        ``log q_def`` is shape ``(B, n_cells)``. The decoder's softmax
        absorbs the joint partition function, so the trainer can add
        the returned tensor to the offensive ``log q_0`` in-place.
        """
        if self.context_dim == 0:
            assert self.theta is not None
            return F.softplus(self.theta) * log_q_def
        if x_n is None:
            raise ValueError(
                "LearnableDefensiveScale was constructed with context_dim>0 "
                "but forward() was not given an x_n tensor"
            )
        assert self.context_mlp is not None
        a = F.softplus(self.context_mlp(x_n))  # (B,)
        return a.unsqueeze(-1) * log_q_def

    def extra_repr(self) -> str:
        if self.context_dim == 0:
            return f"alpha_def={self.alpha_as_float():.3f}"
        return f"alpha_def(x), context_dim={self.context_dim}, init={self.init_value:.3f}"
