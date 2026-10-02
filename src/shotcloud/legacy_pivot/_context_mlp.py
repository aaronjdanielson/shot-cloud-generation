"""Tiny MLP for context-conditioned positive scalars.

Deprecated; retained as a dependency of
:class:`~shotcloud.legacy.temperature.LearnableTemperature` and
:class:`~shotcloud.legacy_pivot.defensive_scale.LearnableDefensiveScale`
in context-conditioned mode (``context_dim > 0``).

A single hidden layer ``Linear(d, h) → tanh → Linear(h, 1)`` whose
output (pre-softplus) is interpreted as ``θ`` for a softplus-positive
scalar. Capacity is intentionally minimal (one hidden layer, width
8-16): the scalar is meant to stay interpretable, and modeling capacity
belongs to the offensive prior and the low-rank tilt. With
``context_dim=10`` and ``hidden=8``, each scalar has 97 parameters.

Initialization makes the pre-softplus output ≈ ``softplus⁻¹(init)`` for
typical inputs, so the initial forward pass reproduces scalar-mode
behavior at the chosen ``init`` value to within MLP noise.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from shotcloud.legacy.learnable_weights import _invert_softplus


class _ContextMLP(nn.Module):
    """Linear → tanh → Linear → scalar (pre-softplus)."""

    def __init__(
        self,
        context_dim: int,
        hidden: int = 8,
        init_value: float = 1.0,
        weight_std: float = 0.1,
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        if init_value <= 0:
            raise ValueError(f"init_value must be strictly positive, got {init_value}")

        self.fc1 = nn.Linear(context_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

        # For typical inputs the pre-softplus output starts close to
        # invert_softplus(init_value): small random hidden weights keep the
        # hidden activations near 0, so fc2's output ≈ fc2.bias, which is
        # set to invert_softplus(init_value).
        nn.init.normal_(self.fc1.weight, std=weight_std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.constant_(self.fc2.bias, _invert_softplus(init_value))

    def forward(self, x: Tensor) -> Tensor:
        """Return pre-softplus output, shape ``(B,)``.

        The caller passes ``x`` of shape ``(B, context_dim)`` and is
        expected to apply :func:`torch.nn.functional.softplus` to the
        output to get the positive scalar.
        """
        h = torch.tanh(self.fc1(x))
        out: Tensor = self.fc2(h).squeeze(-1)
        return out
