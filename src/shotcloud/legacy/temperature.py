"""Learnable temperature on the KDE base measure.

Deprecated; retained to reproduce the temperature ablation. Superseded by
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`.

A softplus-positive temperature ``τ = softplus(θ)`` multiplies the log
base measure before the low-rank tilt is added inside the softmax. It is
the one-parameter alternative to
:class:`~shotcloud.legacy.learnable_weights.LearnableKDEProductWeights`:
a single interpretable knob that controls how sharp the base measure is
(``τ > 1`` sharpens, ``τ < 1`` flattens).

Math
----
For player ``p``, the spatial decoder is

.. math::

    p_\\theta(c \\mid p, h, \\tau) =
    \\mathrm{softmax}_c\\!\\left[
        \\tau \\cdot \\log \\hat q_p^{\\mathrm{hier}}(c) + u^\\top v_c
    \\right].

The softmax in cell ``c`` absorbs the partition function
``log Z_τ(p) = logsumexp_c[τ log q_hier(c)]``, which does **not**
depend on ``c``. So implementation reduces to a scalar multiply on
``log_q0`` before the decoder consumes it — no separate
re-normalization step is needed, and the temperature pathway works
with **any** base measure (kde-product or hierarchical) without
requiring per-component caching.

Pairing
-------
The recommended pairing is :class:`~shotcloud.kde.HierarchicalKDEBase`.
Pairing with :class:`~shotcloud.legacy.product.KDEProduct` is also valid
and produces the same math (only ``log q_0`` is multiplied; the dataset's
cached ``log_q0_table`` is sufficient).

Mutually exclusive with :class:`LearnableKDEProductWeights` -- running
both at once produces an over-specified parameterization (two layers
of softplus scalars) that is hard to interpret.
:func:`~shotcloud.legacy_pivot.trainer.train_decoder` enforces the
exclusion.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from shotcloud.legacy.learnable_weights import _invert_softplus
from shotcloud.legacy_pivot._context_mlp import _ContextMLP

DEFAULT_INIT_TAU: float = 1.0


class LearnableTemperature(nn.Module):
    """Learnable temperature ``τ`` applied to ``log q_0``.

    Two modes:

    * **Scalar mode** (``context_dim=0``, the default). Owns one scalar
      parameter ``θ``; ``τ = softplus(θ)`` is global.
    * **Context-conditioned mode** (``context_dim>0``). Owns a tiny MLP
      whose input is a per-shot context vector ``x_n``;
      ``τ(x_n) = softplus(MLP(x_n))`` varies per shot.

    Parameters
    ----------
    init : float, default 1.0
        Initial temperature value. Must be strictly positive. Step-0
        reproduces the base measure exactly (``τ = init``) in scalar
        mode and approximately in context-conditioned mode (the MLP is
        initialized so its output ≈ ``init`` for typical inputs).
    context_dim : int, default 0
        Dimensionality of the per-shot context vector. ``0`` selects
        scalar mode; ``>0`` selects context-conditioned mode.
    mlp_hidden : int, default 8
        Width of the tiny MLP's hidden layer (only used when
        ``context_dim > 0``). Kept small so ``τ(x_n)`` stays
        interpretable.

    Notes
    -----
    The decoder's softmax absorbs the joint partition function, so
    forward simply returns ``τ · log_q0`` (or ``τ(x_n) · log_q0``);
    no separate normalization is needed. Mutually exclusive with
    :class:`~shotcloud.legacy.learnable_weights.LearnableKDEProductWeights`
    (the trainer enforces this).
    """

    context_dim: int

    def __init__(
        self,
        init: float = DEFAULT_INIT_TAU,
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

    def tau(self, x_n: Tensor | None = None) -> Tensor:
        """Current temperature ``τ`` (graph-attached).

        Returns a scalar in scalar mode and a per-row vector ``(B,)``
        in context-conditioned mode (requires ``x_n``).
        """
        if self.context_dim == 0:
            assert self.theta is not None
            return F.softplus(self.theta)
        if x_n is None:
            raise ValueError("context-conditioned mode requires x_n")
        assert self.context_mlp is not None
        return F.softplus(self.context_mlp(x_n))

    def tau_as_float(self, x_n: Tensor | None = None) -> float:
        """Detached float view of ``τ``.

        Scalar mode → the global temperature. Context-conditioned mode →
        the *mean* over ``x_n`` (shape ``(B, context_dim)``); useful as
        a single number to log per epoch.
        """
        with torch.no_grad():
            t = self.tau(x_n)
            return float(t.mean()) if t.dim() > 0 else float(t)

    def forward(self, log_q0: Tensor, x_n: Tensor | None = None) -> Tensor:
        """Return ``τ · log_q0`` (scalar mode) or ``τ(x_n) · log_q0`` (context).

        Parameters
        ----------
        log_q0 : Tensor, shape ``(B, n_cells)``
        x_n : Tensor, shape ``(B, context_dim)``, optional
            Required when ``context_dim > 0``; ignored otherwise.

        Returns
        -------
        Tensor, shape ``(B, n_cells)``.
        """
        if self.context_dim == 0:
            assert self.theta is not None
            return F.softplus(self.theta) * log_q0
        if x_n is None:
            raise ValueError(
                "LearnableTemperature was constructed with context_dim>0 "
                "but forward() was not given an x_n tensor"
            )
        assert self.context_mlp is not None
        tau = F.softplus(self.context_mlp(x_n))  # (B,)
        return tau.unsqueeze(-1) * log_q0

    def extra_repr(self) -> str:
        if self.context_dim == 0:
            return f"tau={self.tau_as_float():.3f}"
        return f"tau(x), context_dim={self.context_dim}, init={self.init_value:.3f}"
