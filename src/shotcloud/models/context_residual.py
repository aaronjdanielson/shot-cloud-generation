r"""Residual-tilt encoder :math:`u_\theta`.

:class:`ContextResidualEncoder` produces the low-rank vector
:math:`u_\theta \in \mathbb R^{r}` behind the residual tilt of the
support logits. In
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`
it is paired with the coordinate embedding
:math:`\psi` of :class:`~shotcloud.models.location_embedding.LocationEmbedding`,
giving the additive logit for support shot :math:`s_m`

.. math::

    R_\theta(s_m \mid x_n, h_n) = u_\theta(x_n, h_n)^\top \psi(s_m).

The deprecated grid-cell decoder pairs the same vector with the per-cell
basis of :class:`~shotcloud.legacy_pivot.tilt_decoder.LowRankTiltDecoder`.

Player identity is deliberately excluded. The residual is a local,
context-driven refinement of the kernel-mixture geometry; player
identity already enters through the player's own support shots and the
analogue retrieval, and a player embedding here would duplicate that
capacity.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from shotcloud.data.context import CONTEXT_DIM


class ContextResidualEncoder(nn.Module):
    r"""Low-rank residual-tilt encoder :math:`u_\theta`.

    The output is the sum of up to three branches, each a one-hidden-layer
    GELU MLP of width ``hidden_dim``:

    * a base branch on :math:`[x_n, h_n]`, where :math:`x_n` is the learned
      pre-game context and the optional :math:`h_n` summarizes the player's
      earlier shots in the same game
      (:func:`shotcloud.data.within_game_history.compute_within_game_features`);
    * an optional usage branch on :math:`[x_n, u_{p,t}]`, where
      :math:`u_{p,t}` is the causal usage-state vector from
      :func:`shotcloud.features.usage_features.extract_usage` (the spatial
      wrapper may append a detached count-head prediction to it);
    * an optional outcome branch on :math:`[x_n, o_n]`, where :math:`o_n`
      summarizes the outcomes of the player's earlier shots in the same
      game (:func:`shotcloud.data.prior_outcomes.compute_prior_outcome_features`).

    The base branch's output layer has a small random initialization. With a
    zero-initialized location embedding the tilt is exactly zero at
    initialization regardless of :math:`u_\theta`, while the embedding still
    receives a gradient proportional to :math:`u_\theta`; a zero-initialized
    encoder would leave both factors stuck at the zero saddle. The usage and
    outcome branches have zero-initialized output layers, so enabling them
    leaves the initial output unchanged.

    Parameters
    ----------
    rank : int, default 8
        Output dimension; must equal the rank of the paired
        :class:`~shotcloud.models.location_embedding.LocationEmbedding`.
    context_dim : int, default :data:`~shotcloud.data.context.CONTEXT_DIM`
        Dimension of the learned context :math:`x_n`.
    within_game_dim : int, default 0
        Dimension of :math:`h_n`
        (:data:`~shotcloud.data.within_game_history.WITHIN_GAME_DIM` for the
        standard featurizer). ``0`` disables the within-game input.
    usage_dim : int, default 0
        Dimension of the usage vector. ``0`` disables the usage branch.
    outcome_dim : int, default 0
        Dimension of :math:`o_n`
        (:data:`~shotcloud.data.prior_outcomes.PRIOR_OUTCOME_DIM` for the
        standard featurizer). ``0`` disables the outcome branch.
    hidden_dim : int, default 32
        Hidden width of every branch. Kept narrow so the residual cannot
        dominate the kernel-mixture geometry: the tilt is kept small by
        capacity rather than by a penalty.
    init_std_scale : float, default 1.0
        The base branch's output weights are drawn with standard deviation
        ``init_std_scale / sqrt(rank)``. Smaller values keep the initial
        :math:`u_\theta` closer to zero.

    Raises
    ------
    ValueError
        If ``rank``, ``context_dim``, ``hidden_dim`` or ``init_std_scale``
        is non-positive, or an optional input dimension is negative.
    """

    def __init__(
        self,
        rank: int = 8,
        context_dim: int = CONTEXT_DIM,
        within_game_dim: int = 0,
        usage_dim: int = 0,
        outcome_dim: int = 0,
        hidden_dim: int = 32,
        init_std_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if within_game_dim < 0:
            raise ValueError(f"within_game_dim must be non-negative, got {within_game_dim}")
        if usage_dim < 0:
            raise ValueError(f"usage_dim must be non-negative, got {usage_dim}")
        if outcome_dim < 0:
            raise ValueError(f"outcome_dim must be non-negative, got {outcome_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if init_std_scale <= 0:
            raise ValueError(f"init_std_scale must be positive, got {init_std_scale}")

        self.rank = rank
        self.context_dim = context_dim
        self.within_game_dim = within_game_dim
        self.usage_dim = usage_dim
        self.outcome_dim = outcome_dim
        self.hidden_dim = hidden_dim
        self.init_std_scale = float(init_std_scale)

        input_dim = context_dim + within_game_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, rank)

        # Small but nonzero output init. The paired embedding ψ (or basis V)
        # starts at zero, so the tilt u^T ψ is zero and the encoder receives
        # no gradient at step 0; the embedding's gradient is proportional to
        # u, so u must be nonzero for training to leave the zero saddle.
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=init_std_scale / math.sqrt(rank))
        nn.init.zeros_(self.fc2.bias)

        # Usage branch. Zero-initialized output, so at initialization the
        # encoder equals the no-usage encoder exactly.
        if usage_dim > 0:
            self.usage_fc1 = nn.Linear(context_dim + usage_dim, hidden_dim)
            self.usage_fc2 = nn.Linear(hidden_dim, rank)
            nn.init.kaiming_uniform_(self.usage_fc1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.usage_fc1.bias)
            nn.init.zeros_(self.usage_fc2.weight)
            nn.init.zeros_(self.usage_fc2.bias)

        # Prior-outcome branch, zero-initialized like the usage branch.
        if outcome_dim > 0:
            self.outcome_fc1 = nn.Linear(context_dim + outcome_dim, hidden_dim)
            self.outcome_fc2 = nn.Linear(hidden_dim, rank)
            nn.init.kaiming_uniform_(self.outcome_fc1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.outcome_fc1.bias)
            nn.init.zeros_(self.outcome_fc2.weight)
            nn.init.zeros_(self.outcome_fc2.bias)

    def forward(
        self,
        x_n: Tensor,
        h_n: Tensor | None = None,
        usage: Tensor | None = None,
        outcome: Tensor | None = None,
    ) -> Tensor:
        r"""Compute :math:`u_\theta` for a batch of shots.

        Each optional input is required when its configured dimension is
        positive and must be ``None`` when that dimension is zero.

        Parameters
        ----------
        x_n : Tensor of shape ``(B, context_dim)``
            Learned context, typically the output of
            :class:`~shotcloud.models.ContextMLP`.
        h_n : Tensor of shape ``(B, within_game_dim)`` or None
            Within-game history features.
        usage : Tensor of shape ``(B, usage_dim)`` or None
            Causal usage-state vector :math:`u_{p,t}`.
        outcome : Tensor of shape ``(B, outcome_dim)`` or None
            Prior-outcome summary :math:`o_n`.

        Returns
        -------
        Tensor of shape ``(B, rank)``

        Raises
        ------
        ValueError
            If an input has the wrong shape or batch size, or an optional
            input is inconsistent with its configured dimension.
        """
        if x_n.dim() != 2 or x_n.shape[1] != self.context_dim:
            raise ValueError(
                f"x_n must have shape (B, context_dim={self.context_dim}); got {tuple(x_n.shape)}"
            )
        if self.within_game_dim == 0:
            if h_n is not None:
                raise ValueError(
                    f"within_game_dim=0 → h_n must be None; got tensor of shape {tuple(h_n.shape)}"
                )
            joined = x_n
        else:
            if h_n is None:
                raise ValueError(f"within_game_dim={self.within_game_dim} → h_n is required")
            if h_n.dim() != 2 or h_n.shape[1] != self.within_game_dim:
                raise ValueError(
                    f"h_n must have shape (B, within_game_dim={self.within_game_dim}); "
                    f"got {tuple(h_n.shape)}"
                )
            if h_n.shape[0] != x_n.shape[0]:
                raise ValueError(
                    f"x_n and h_n batch sizes must match; got {x_n.shape[0]} vs {h_n.shape[0]}"
                )
            joined = torch.cat([x_n, h_n], dim=-1)
        h_hidden = torch.nn.functional.gelu(self.fc1(joined))
        u: Tensor = self.fc2(h_hidden)

        if self.usage_dim == 0:
            if usage is not None:
                raise ValueError(
                    f"usage_dim=0 → usage must be None; got tensor of shape {tuple(usage.shape)}"
                )
        else:
            if usage is None:
                raise ValueError(f"usage_dim={self.usage_dim} → usage is required")
            if usage.dim() != 2 or usage.shape[1] != self.usage_dim:
                raise ValueError(
                    f"usage must have shape (B, usage_dim={self.usage_dim}); "
                    f"got {tuple(usage.shape)}"
                )
            if usage.shape[0] != x_n.shape[0]:
                raise ValueError(
                    f"x_n and usage batch sizes must match; got {x_n.shape[0]} vs {usage.shape[0]}"
                )
            usage_joined = torch.cat([x_n, usage], dim=-1)
            usage_hidden = torch.nn.functional.gelu(self.usage_fc1(usage_joined))
            u_delta: Tensor = self.usage_fc2(usage_hidden)
            u = u + u_delta

        if self.outcome_dim == 0:
            if outcome is not None:
                raise ValueError(
                    f"outcome_dim=0 → outcome must be None; got tensor of shape "
                    f"{tuple(outcome.shape)}"
                )
            return u
        if outcome is None:
            raise ValueError(f"outcome_dim={self.outcome_dim} → outcome is required")
        if outcome.dim() != 2 or outcome.shape[1] != self.outcome_dim:
            raise ValueError(
                f"outcome must have shape (B, outcome_dim={self.outcome_dim}); "
                f"got {tuple(outcome.shape)}"
            )
        if outcome.shape[0] != x_n.shape[0]:
            raise ValueError(
                f"x_n and outcome batch sizes must match; got {x_n.shape[0]} vs {outcome.shape[0]}"
            )
        outcome_joined = torch.cat([x_n, outcome], dim=-1)
        outcome_hidden = torch.nn.functional.gelu(self.outcome_fc1(outcome_joined))
        u_outcome_delta: Tensor = self.outcome_fc2(outcome_hidden)
        return u + u_outcome_delta

    def extra_repr(self) -> str:
        return (
            f"context_dim={self.context_dim}, within_game_dim={self.within_game_dim}, "
            f"usage_dim={self.usage_dim}, outcome_dim={self.outcome_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"rank={self.rank}, init_std_scale={self.init_std_scale}"
        )
