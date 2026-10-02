"""History-dependent own-vs-pooled mixing gate.

The collaborative KDE assembles a support set of the target
player's own historical shots plus *pooled* shots from other
players. An unconstrained joint softmax over that support
empirically collapses to a near-constant pooled-mass fraction
(~0.80) regardless of how much own history a player has --- the
attention never learns to trust own history more as it accumulates.

The :class:`PoolingGate` breaks that pathology structurally. The
spatial density is re-parameterized as a two-component mixture

.. math::

    f_\\Theta(y) = \\lambda\\,f_{\\mathrm{own}}(y)
                 + (1-\\lambda)\\,f_{\\mathrm{pooled}}(y),

with ``f_own`` and ``f_pooled`` each normalized within their own
support subset, and the mixing weight ``λ`` produced by this gate
as a function of the player's causal own-history count --- *not*
of the support logits. Parameterizing ``λ`` from history forbids
the collapse: the gate cannot route own-mass through whatever the
logits happen to produce.

Parameterization (monotone in history by construction):

.. math::

    \\operatorname{logit}\\lambda
    = b_0 + \\operatorname{softplus}(b_H)\\,\\log(1+\\hat H_p(t_n))
      + g_\\theta(x_n, h_n, \\log(1 + \\text{own\\_support\\_count})).

``softplus(b_H) \\ge 0`` guarantees ``λ`` is non-decreasing in the
own-history count. ``g_θ`` is a small MLP with a zero-initialized
final layer, so at step 0 the gate is exactly the closed-form
history schedule set by ``(b_0, b_H)``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from shotcloud.data.context import CONTEXT_DIM

#: Default intercept ``b_0``. With the default slope it puts
#: ``λ ≈ 0.076`` at zero own history (log(1+H)=0 ⇒ logit=b_0).
DEFAULT_GATE_B0: float = -2.5

#: Default pre-softplus slope ``b_H``. ``softplus(-0.20) ≈ 0.60``,
#: giving the history schedule λ(H=25)≈0.37, λ(H=100)≈0.57,
#: λ(H=300)≈0.71, λ(H=1000)≈0.84 --- monotone, low at cold-start,
#: clearly past 0.70 by the dense-history regime.
DEFAULT_GATE_BH_INIT: float = -0.20

#: Default hidden width of the context MLP ``g_θ``.
DEFAULT_GATE_HIDDEN_DIM: int = 32


class PoolingGate(nn.Module):
    """Own-vs-pooled mixing weight ``λ ∈ (0, 1)``.

    Parameters
    ----------
    context_dim : int, default :data:`CONTEXT_DIM`
        Width of the learned context vector ``x_n``.
    history_dim : int, default 0
        Width of the within-game history vector ``h_n``. When 0,
        ``g_θ`` consumes only ``x_n`` and the own-support-count
        feature.
    hidden_dim : int, default 32
        Hidden width of the context MLP ``g_θ``.
    b0 : float, default :data:`DEFAULT_GATE_B0`
        Initial logit intercept.
    bh_init : float, default :data:`DEFAULT_GATE_BH_INIT`
        Initial value of the pre-softplus slope parameter. The
        effective slope is ``softplus(b_H) ≥ 0``, so ``λ`` is
        monotone non-decreasing in the own-history count for every
        value this parameter can take during training.
    """

    def __init__(
        self,
        context_dim: int = CONTEXT_DIM,
        history_dim: int = 0,
        hidden_dim: int = DEFAULT_GATE_HIDDEN_DIM,
        b0: float = DEFAULT_GATE_B0,
        bh_init: float = DEFAULT_GATE_BH_INIT,
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive; got {context_dim}")
        if history_dim < 0:
            raise ValueError(f"history_dim must be non-negative; got {history_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}")

        self.context_dim = int(context_dim)
        self.history_dim = int(history_dim)

        self.b0 = nn.Parameter(torch.tensor(float(b0)))
        self.b_h = nn.Parameter(torch.tensor(float(bh_init)))

        # g_θ consumes [x_n, h_n?, log(1 + own_support_count)].
        g_in = context_dim + history_dim + 1
        g_out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(g_out.weight)
        nn.init.zeros_(g_out.bias)
        self.g_theta = nn.Sequential(
            nn.Linear(g_in, hidden_dim),
            nn.GELU(),
            g_out,
        )

    def forward(
        self,
        log1p_h_hat: Tensor,
        x_n: Tensor,
        own_support_count: Tensor,
        own_available: Tensor,
        h_n: Tensor | None = None,
        pooled_available: Tensor | None = None,
    ) -> Tensor:
        """Mixing weight ``λ`` per row.

        Parameters
        ----------
        log1p_h_hat : Tensor of shape ``(B,)``
            ``log(1 + Ĥ_p(t_n))`` --- log of the (estimated) causal
            own-history shot count.
        x_n : Tensor of shape ``(B, context_dim)``
            Learned context vector.
        own_support_count : Tensor of shape ``(B,)``
            Number of own-player support shots in the support set.
        own_available : Tensor of shape ``(B,)`` bool
            Whether the row has any own-player support. Rows with
            none get ``λ = 0`` (density falls back to ``f_pooled``).
        h_n : Tensor of shape ``(B, history_dim)`` or None
            Within-game history; required when ``history_dim > 0``.
        pooled_available : Tensor of shape ``(B,)`` bool or None
            Whether the row has any pooled support. Rows with none
            get ``λ = 1``. When ``None``, all rows are assumed to
            have pooled support.

        Returns
        -------
        Tensor of shape ``(B,)``
            ``λ ∈ [0, 1]``.
        """
        b = x_n.shape[0]
        if log1p_h_hat.shape != (b,):
            raise ValueError(f"log1p_h_hat must be (B,)={(b,)}; got {tuple(log1p_h_hat.shape)}")
        if own_support_count.shape != (b,):
            raise ValueError(
                f"own_support_count must be (B,)={(b,)}; got {tuple(own_support_count.shape)}"
            )
        if own_available.shape != (b,):
            raise ValueError(f"own_available must be (B,)={(b,)}; got {tuple(own_available.shape)}")
        if self.history_dim > 0:
            if h_n is None:
                raise ValueError(f"history_dim={self.history_dim} > 0 → h_n is required")
            if h_n.shape != (b, self.history_dim):
                raise ValueError(f"h_n must be (B, {self.history_dim}); got {tuple(h_n.shape)}")

        log1p_count = torch.log1p(own_support_count.to(x_n.dtype)).unsqueeze(-1)  # (B, 1)
        if self.history_dim > 0:
            assert h_n is not None
            g_in = torch.cat([x_n, h_n, log1p_count], dim=-1)
        else:
            g_in = torch.cat([x_n, log1p_count], dim=-1)
        g_out = self.g_theta(g_in).squeeze(-1)  # (B,)

        slope = torch.nn.functional.softplus(self.b_h)
        logit = self.b0 + slope * log1p_h_hat + g_out  # (B,)
        lam = torch.sigmoid(logit)

        # Edge handling: no own support → λ = 0 (density = f_pooled);
        # no pooled support → λ = 1 (density = f_own).
        lam = torch.where(own_available, lam, torch.zeros_like(lam))
        if pooled_available is not None:
            lam = torch.where(pooled_available, lam, torch.ones_like(lam))
        return lam

    def history_schedule(self, h_hat: float) -> float:
        """Closed-form ``λ`` at own-history count ``h_hat`` with
        ``g_θ = 0`` (the step-0 schedule). Useful for tests and for
        documenting the initialization."""
        b_h = float(self.b_h.detach())
        b0 = float(self.b0.detach())
        slope = math.log1p(math.exp(b_h))  # softplus
        logit = b0 + slope * math.log1p(h_hat)
        return 1.0 / (1.0 + math.exp(-logit))


__all__ = [
    "DEFAULT_GATE_B0",
    "DEFAULT_GATE_BH_INIT",
    "DEFAULT_GATE_HIDDEN_DIM",
    "PoolingGate",
]
