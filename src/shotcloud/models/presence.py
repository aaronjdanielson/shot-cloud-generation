"""Causal player-presence model :math:`q_\\mathrm{pres}(b\\mid p,t,s,\\mathrm{pos})`.

A small, interpretable model of a player's on-court fraction in each
2-minute bin of a game, given the player, game date, starter status and
position. It can supply an optional on-court presence feature to the
timing head. Its structure parallels the AC-KDE pooling gate, an
own/pooled mixture under a history-driven gate:

.. math::

    q_\\mathrm{pres}(b\\mid p,t,s,\\mathrm{pos})
    =
    \\lambda_{p,s}(t)\\,q_\\mathrm{self}(b\\mid p,t,s)
    + \\bigl(1-\\lambda_{p,s}(t)\\bigr)\\,q_\\mathrm{pool}(b\\mid \\mathrm{pos},s),

where:

* :math:`q_\\mathrm{self}(b\\mid p,t,s)` is the recency-weighted mean
  of the player's strictly-prior on-court vectors restricted to games
  with matching starter status :math:`s`. Weights
  :math:`w(\\Delta t)=\\exp(-\\rho\\,\\Delta t)` with learned
  :math:`\\rho=\\mathrm{softplus}(\\tilde\\rho)`.
* :math:`q_\\mathrm{pool}(b\\mid \\mathrm{pos},s)=
  \\sigma(\\theta_{\\mathrm{pos},s,b})` is a per-(position, starter)
  learned pool histogram with one sigmoid scalar per bin.
* :math:`\\lambda_{p,s}(t)=
  \\sigma(b_0+\\mathrm{softplus}(\\tilde\\beta_h)\\,
  \\log(1+N^{<t}_{p,s}))` is the history-driven gate with learned
  :math:`b_0,\\tilde\\beta_h`, the same functional form as the
  closed-form history schedule of
  :class:`~shotcloud.models.pooling_gate.PoolingGate`.

The prior-game inputs to :math:`q_\\mathrm{self}` and
:math:`N^{<t}_{p,s}` must come from games strictly before the query date,
which makes the model causal. Each target :math:`z_{p,g,b}\\in[0,1]` is an
observed fraction, and the model is trained with per-bin binary
cross-entropy between :math:`q_\\mathrm{pres}(b)` and
:math:`z_{p,g,b}`.

The module is deliberately small: ``3 + n_positions × n_starter ×
n_bins`` learnable scalars (183 with the defaults). Its capacity is set
by the data layout rather than by a hidden width; there is no large
neural component.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

#: Number of 2-minute bins (matches ``N_BINS`` in
#: ``scripts/build_oncourt_table.py``, which builds the on-court table).
PRESENCE_N_BINS: Final[int] = 30

#: Hard position classes for the pool lookup: guard, wing, big (see
#: :mod:`shotcloud.data.positions`). This constant sets the pool table's
#: first axis; unknown positions map to index 0 (guard).
PRESENCE_N_POSITIONS: Final[int] = 3

#: Starter axis is binary: 0 = bench, 1 = starter.
PRESENCE_N_STARTER: Final[int] = 2


class PresenceModel(nn.Module):
    """Recency-weighted own/pool mixture for per-bin on-court fraction.

    Parameters
    ----------
    n_bins : int, default :data:`PRESENCE_N_BINS`
        Number of time bins per game.
    n_positions : int, default :data:`PRESENCE_N_POSITIONS`
        Number of position classes in the pool table.
    n_starter : int, default :data:`PRESENCE_N_STARTER`
        Number of starter-status classes in the pool table.
    rho_init : float, default 1/45.0
        Initial value of :math:`\\rho` (per-day decay). 1/45 puts the
        recency half-life at roughly 30 days.
    b0_init : float, default -1.0
        Initial gate bias. With :math:`\\beta_h=0`, the gate starts at
        :math:`\\sigma(-1)\\approx 0.27`, favoring the pool so a
        zero-history player relies mostly on it.
    beta_h_init : float, default 0.5
        Pre-softplus initial value of :math:`\\tilde\\beta_h`.
        ``softplus(0.5) ≈ 0.97`` — a strong history slope so the gate
        opens up to the self curve once the player accumulates games.
    """

    def __init__(
        self,
        n_bins: int = PRESENCE_N_BINS,
        n_positions: int = PRESENCE_N_POSITIONS,
        n_starter: int = PRESENCE_N_STARTER,
        rho_init: float = 1.0 / 45.0,
        b0_init: float = -1.0,
        beta_h_init: float = 0.5,
    ) -> None:
        super().__init__()
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        if n_positions <= 0:
            raise ValueError(f"n_positions must be positive, got {n_positions}")
        if n_starter <= 0:
            raise ValueError(f"n_starter must be positive, got {n_starter}")
        if rho_init <= 0:
            raise ValueError(f"rho_init must be positive, got {rho_init}")

        self.n_bins = n_bins
        self.n_positions = n_positions
        self.n_starter = n_starter

        # Recency decay ρ = softplus(ρ̃).
        rho_inv_softplus = torch.log(torch.expm1(torch.tensor(rho_init)))
        self.rho_raw = nn.Parameter(rho_inv_softplus.clone())

        # History gate: λ = σ(b0 + softplus(β̃_h) · log1p(N^<t_{p,s})).
        self.b0 = nn.Parameter(torch.tensor(float(b0_init)))
        self.beta_h_raw = nn.Parameter(torch.tensor(float(beta_h_init)))

        # Pool histogram θ_{pos, s, b}, sigmoid'd to a per-bin fraction.
        # Initialized at 0 → σ(0) = 0.5, a neutral half-on-court prior.
        self.pool_theta = nn.Parameter(torch.zeros(n_positions, n_starter, n_bins))

    @property
    def rho(self) -> Tensor:
        """Effective per-day decay (>0)."""
        return torch.nn.functional.softplus(self.rho_raw)

    @property
    def beta_h(self) -> Tensor:
        """Effective gate slope (>0)."""
        return torch.nn.functional.softplus(self.beta_h_raw)

    def pool_curve(self, position_idx: Tensor, starter_idx: Tensor) -> Tensor:
        """Per-(position, starter) sigmoid pool histogram, shape ``(B, n_bins)``."""
        if position_idx.shape != starter_idx.shape:
            raise ValueError(
                f"position_idx and starter_idx must have matching shape; "
                f"got {tuple(position_idx.shape)} vs {tuple(starter_idx.shape)}"
            )
        theta = self.pool_theta[position_idx, starter_idx]  # (B, n_bins)
        return torch.sigmoid(theta)

    def history_gate(self, history_count: Tensor) -> Tensor:
        """:math:`\\lambda = \\sigma(b_0 + \\mathrm{softplus}(\\tilde\\beta_h)\\log(1+N))`."""
        n = history_count.clamp_min(0.0).to(self.b0.dtype)
        return torch.sigmoid(self.b0 + self.beta_h * torch.log1p(n))

    def self_curve(
        self,
        prior_bins: Tensor,
        prior_ages_days: Tensor,
        prior_mask: Tensor,
    ) -> Tensor:
        """Recency-weighted self curve.

        Parameters
        ----------
        prior_bins : Tensor of shape ``(B, K_max, n_bins)``
            Per-shot list of the player's prior-game on-court vectors
            restricted to matching starter status. Padded to ``K_max``
            with zeros for shots that have fewer than ``K_max`` priors.
        prior_ages_days : Tensor of shape ``(B, K_max)``
            Age (in days) of each prior game relative to the query
            date. Zero for padded slots (the mask suppresses them).
        prior_mask : Tensor of shape ``(B, K_max)``
            ``1.0`` for real prior games, ``0.0`` for padding.

        Returns
        -------
        Tensor of shape ``(B, n_bins)``
            Rows with ``prior_mask.sum() == 0`` (no real priors) are zero,
            so in :meth:`forward` such rows reduce to the gated pool term
            :math:`(1-\\lambda)\\,q_\\mathrm{pool}`.
        """
        if prior_bins.dim() != 3 or prior_bins.shape[-1] != self.n_bins:
            raise ValueError(
                f"prior_bins must have shape (B, K_max, n_bins={self.n_bins}); "
                f"got {tuple(prior_bins.shape)}"
            )
        b, k_max, _ = prior_bins.shape
        if prior_ages_days.shape != (b, k_max) or prior_mask.shape != (b, k_max):
            raise ValueError(
                f"prior_ages_days/prior_mask must have shape (B, K_max)=({b}, {k_max}); "
                f"got ages {tuple(prior_ages_days.shape)} / mask {tuple(prior_mask.shape)}"
            )
        # w(Δt) = exp(-ρ · Δt) · mask
        w = torch.exp(-self.rho * prior_ages_days.clamp_min(0.0)) * prior_mask  # (B, K_max)
        denom = w.sum(dim=-1, keepdim=True)  # (B, 1)
        # Cold-start row (no prior games): denom == 0; produce zeros so the
        # output reduces to the gated pool term.
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        weighted_bins = (prior_bins * w.unsqueeze(-1)).sum(dim=1)  # (B, n_bins)
        return weighted_bins / safe_denom

    def forward(
        self,
        prior_bins: Tensor,
        prior_ages_days: Tensor,
        prior_mask: Tensor,
        position_idx: Tensor,
        starter_idx: Tensor,
        history_count: Tensor,
    ) -> Tensor:
        """Compute :math:`q_\\mathrm{pres}(b)` for a batch of queries.

        Parameters
        ----------
        prior_bins, prior_ages_days, prior_mask : Tensor
            Prior-game on-court vectors, ages and mask; see
            :meth:`self_curve`.
        position_idx, starter_idx : Tensor of shape ``(B,)``, int64
            Position class and starter status indexing the pool table.
        history_count : Tensor of shape ``(B,)``
            Number of prior games :math:`N^{<t}_{p,s}` feeding the gate.

        Returns
        -------
        Tensor of shape ``(B, n_bins)``
            Predicted on-court fraction per bin, in :math:`[0, 1]`.
        """
        q_self = self.self_curve(prior_bins, prior_ages_days, prior_mask)
        q_pool = self.pool_curve(position_idx, starter_idx)
        lam = self.history_gate(history_count).unsqueeze(-1)  # (B, 1)
        return lam * q_self + (1.0 - lam) * q_pool

    def extra_repr(self) -> str:
        return (
            f"n_bins={self.n_bins}, n_positions={self.n_positions}, "
            f"n_starter={self.n_starter}, rho={self.rho.item():.5f}, "
            f"b0={self.b0.item():.3f}, beta_h={self.beta_h.item():.3f}"
        )


__all__ = [
    "PRESENCE_N_BINS",
    "PRESENCE_N_POSITIONS",
    "PRESENCE_N_STARTER",
    "PresenceModel",
]
