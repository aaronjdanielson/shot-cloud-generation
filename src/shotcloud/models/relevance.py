"""Structured relevance score for the context-adaptive KDE.

Computes per-historical-shot relevance scores ``f_φ(z_j, x_n)`` from the
27-dim context vector. Paper App.~B calls these "interpretable
similarity terms" — five named scalars, each acting on a specific
slice of the context representation:

============  =================================================  =====================
parameter      acts on                                            interpretation
============  =================================================  =====================
``β_q``        ``period_onehot`` slice (4 dims) — dot product     same-quarter bonus
``β_m``        ``time_in_period`` (1 dim) — negative absolute Δ    similar-time bonus
``β_t``        ``season_recency`` (1 dim) — negative absolute Δ    nearby-date bonus
``β_o``        ``opp_efficiency_onehot`` slice (4 dims) — dot product same-bucket bonus
``λ_g``        per-shot scalar (NOT in x_n) — exponential decay   recency baseline
============  =================================================  =====================

Notes:

* All five params are stored as **unconstrained reals**. Sign matters
  (``λ_g`` should be ≥ 0 for the recency interpretation to hold), but
  the optimizer decides — a positive ``λ_g`` for a player whose oldest
  shots predict best is a fine empirical signal.
* ``λ_g`` is the only term that doesn't have a symmetric ``f(z_j, x_n)``
  shape — it's an absolute "how old is shot j?" prior, not a similarity
  to ``x_n``. We pass it as a separate ``games_ago`` tensor at forward
  time. When unavailable, callers pass zeros and ``λ_g`` drifts to 0
  under weak gradient pressure.
* Paper App.~B enumerates seven similarity terms (the five above plus
  ``β_r`` starter-status and ``β_s`` minutes-played). The two extra
  terms are not implemented; resolving the paper/code mismatch is
  flagged in ``docs/audit_2026-05-15.md``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from shotcloud.data.context import FEATURE_LAYOUT


def _inverse_bounded(value: float, beta_max: float) -> Tensor:
    """Invert ``β = β_max · tanh(θ / β_max)`` so init reproduces ``value``."""
    return torch.tensor(float(beta_max) * float(torch.atanh(torch.tensor(value / beta_max))))


class RelevanceScore(nn.Module):
    """Structured ``f_φ(z_j, x_n)`` returning per-history-shot logits.

    Parameters
    ----------
    init_beta_q, init_beta_m, init_beta_t, init_beta_o : float, default 0.0
        Initial values for the four similarity-term scalars. ``0.0``
        means uniform relevance at step 0 (every shot is equally
        relevant), so the adaptive KDE starts equivalent to a fixed
        uniform-recency Gaussian-kernel density. Any deviation is
        data-driven.
    init_lambda_g : float, default 0.0
        Initial value of the recency-decay scalar ``λ_g``. The
        ``games_ago`` tensor is multiplied by ``-λ_g``, so a positive
        ``λ_g`` downweights older shots. Default 0.0 disables the
        decay until the optimizer engages it.
    beta_max : float or None, default None
        Anti-overfit bound. When set, the four similarity-scalars
        ``β_{q,m,t,o}`` are wrapped through ``β_max · tanh(θ / β_max)``
        so each effective β is bounded in ``(-β_max, β_max)``. The
        underlying ``nn.Parameter`` (``θ``) remains unconstrained — the
        wrapping is applied at every forward pass. ``λ_g`` is left
        unbounded; its semantics (recency decay) is non-symmetric and
        unbounded magnitude there isn't an overfit risk in the same way.
        ``None`` (default) preserves the unbounded behavior. Motivation:
        without the bound, ``β_m`` can run to +3.67 and dominate the
        relevance softmax. ``β_max = 2.0`` lets the model say "this
        shot is 7× more relevant" (e^2 ≈ 7.4) but prevents single-
        feature collapse.
    """

    def __init__(
        self,
        init_beta_q: float = 0.0,
        init_beta_m: float = 0.0,
        init_beta_t: float = 0.0,
        init_beta_o: float = 0.0,
        init_lambda_g: float = 0.0,
        beta_max: float | None = None,
    ) -> None:
        super().__init__()
        if beta_max is not None and beta_max <= 0:
            raise ValueError(f"beta_max must be positive or None, got {beta_max}")
        if beta_max is not None:
            for name, init in (
                ("init_beta_q", init_beta_q),
                ("init_beta_m", init_beta_m),
                ("init_beta_t", init_beta_t),
                ("init_beta_o", init_beta_o),
            ):
                if abs(init) >= beta_max:
                    raise ValueError(
                        f"{name}={init} must be strictly within (-beta_max, beta_max)="
                        f"(-{beta_max}, {beta_max}); the tanh wrap saturates at the edges"
                    )
        self.beta_max = beta_max
        # When bounded, store the *pre-image* under tanh so
        # ``β_max · tanh(θ / β_max)`` reproduces ``init`` exactly.
        if beta_max is None:
            self.beta_q = nn.Parameter(torch.tensor(float(init_beta_q)))
            self.beta_m = nn.Parameter(torch.tensor(float(init_beta_m)))
            self.beta_t = nn.Parameter(torch.tensor(float(init_beta_t)))
            self.beta_o = nn.Parameter(torch.tensor(float(init_beta_o)))
        else:
            self.beta_q = nn.Parameter(_inverse_bounded(init_beta_q, beta_max))
            self.beta_m = nn.Parameter(_inverse_bounded(init_beta_m, beta_max))
            self.beta_t = nn.Parameter(_inverse_bounded(init_beta_t, beta_max))
            self.beta_o = nn.Parameter(_inverse_bounded(init_beta_o, beta_max))
        self.lambda_g = nn.Parameter(torch.tensor(float(init_lambda_g)))

        # Cache slice ranges as (start, stop) ints — Tensor.index_select
        # is more autograd-friendly than slicing inside forward.
        period = FEATURE_LAYOUT["period_onehot"]
        time = FEATURE_LAYOUT["time_in_period"]
        recency = FEATURE_LAYOUT["season_recency"]
        opp = FEATURE_LAYOUT["opp_efficiency_onehot"]
        self._period_slice = (period.start, period.stop)
        self._time_idx = time.start
        self._recency_idx = recency.start
        self._opp_slice = (opp.start, opp.stop)

    def _bound(self, theta: Tensor) -> Tensor:
        """Apply ``β_max · tanh(θ / β_max)`` if bounded, else identity."""
        if self.beta_max is None:
            return theta
        return self.beta_max * torch.tanh(theta / self.beta_max)

    @property
    def effective_beta_q(self) -> Tensor:
        return self._bound(self.beta_q)

    @property
    def effective_beta_m(self) -> Tensor:
        return self._bound(self.beta_m)

    @property
    def effective_beta_t(self) -> Tensor:
        return self._bound(self.beta_t)

    @property
    def effective_beta_o(self) -> Tensor:
        return self._bound(self.beta_o)

    def forward(
        self,
        z_j: Tensor,
        x_n: Tensor,
        games_ago: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute per-shot logits, shape ``(B, max_N)``.

        Parameters
        ----------
        z_j : Tensor, shape ``(B, max_N, CONTEXT_DIM)``
            Per-historical-shot context, padded along the second axis.
        x_n : Tensor, shape ``(B, CONTEXT_DIM)``
            Per-target-shot context.
        games_ago : Tensor, shape ``(B, max_N)``, optional
            Per-historical-shot recency (in games or days; the unit is
            absorbed by ``λ_g``). When ``None``, the recency term is
            zero. Recommended: pre-normalize by the train set's max so
            ``λ_g`` lives near unit scale.
        mask : Tensor, shape ``(B, max_N)``, optional
            Boolean (or 0/1) mask: ``1`` for real shots, ``0`` for
            padding. The returned logits are set to ``-inf`` where
            ``mask == 0`` so a downstream softmax produces zero weight
            on padded positions. When ``None``, all positions are real.

        Returns
        -------
        Tensor of shape ``(B, max_N)`` — per-shot pre-softmax logits.
        """
        if z_j.dim() != 3 or x_n.dim() != 2:
            raise ValueError(
                f"expected z_j (B, N, D) and x_n (B, D); got {tuple(z_j.shape)} and "
                f"{tuple(x_n.shape)}"
            )
        if z_j.shape[0] != x_n.shape[0] or z_j.shape[2] != x_n.shape[1]:
            raise ValueError(f"shape mismatch: z_j={tuple(z_j.shape)}, x_n={tuple(x_n.shape)}")

        # Period match: ⟨period_onehot_j, period_onehot_n⟩ ∈ {0, 1}.
        ps, pe = self._period_slice
        period_match = (z_j[..., ps:pe] * x_n[:, ps:pe].unsqueeze(1)).sum(dim=-1)

        # Time-in-period similarity: -|t_j - t_n|, in [-1, 0].
        time_diff = (z_j[..., self._time_idx] - x_n[:, self._time_idx].unsqueeze(1)).abs()

        # Season-recency similarity: -|s_j - s_n|, in [-1, 0].
        recency_diff = (z_j[..., self._recency_idx] - x_n[:, self._recency_idx].unsqueeze(1)).abs()

        # Opp-strength match: dot product of one-hots in {0, 1}.
        os, oe = self._opp_slice
        opp_match = (z_j[..., os:oe] * x_n[:, os:oe].unsqueeze(1)).sum(dim=-1)

        logits = (
            self.effective_beta_q * period_match
            - self.effective_beta_m * time_diff
            - self.effective_beta_t * recency_diff
            + self.effective_beta_o * opp_match
        )

        if games_ago is not None:
            if games_ago.shape != logits.shape:
                raise ValueError(
                    f"games_ago shape {tuple(games_ago.shape)} != logits shape "
                    f"{tuple(logits.shape)}"
                )
            logits = logits - self.lambda_g * games_ago

        if mask is not None:
            if mask.shape != logits.shape:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} != logits shape {tuple(logits.shape)}"
                )
            # Set padded positions to a very negative number so softmax → 0.
            logits = logits.masked_fill(mask < 0.5, float("-inf"))

        return logits

    def softmax(
        self,
        z_j: Tensor,
        x_n: Tensor,
        games_ago: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Convenience: ``softmax_j[f_φ(z_j, x_n)]``."""
        logits = self.forward(z_j, x_n, games_ago, mask)
        return torch.softmax(logits, dim=-1)

    def params_as_floats(self) -> dict[str, float]:
        """Detached float view of the five effective params, useful for logging.

        With ``beta_max`` set, returns the *bounded* β values (the ones
        actually entering the logits), not the raw underlying θ. This is
        what's interpretable for the paper / per-epoch printout.
        """
        with torch.no_grad():
            return {
                "beta_q": float(self.effective_beta_q),
                "beta_m": float(self.effective_beta_m),
                "beta_t": float(self.effective_beta_t),
                "beta_o": float(self.effective_beta_o),
                "lambda_g": float(self.lambda_g),
            }

    def extra_repr(self) -> str:
        p = self.params_as_floats()
        return (
            f"beta_q={p['beta_q']:.3f}, beta_m={p['beta_m']:.3f}, "
            f"beta_t={p['beta_t']:.3f}, beta_o={p['beta_o']:.3f}, "
            f"lambda_g={p['lambda_g']:.3f}"
        )


class RelevanceMLP(nn.Module):
    """MLP-based ``f_φ(z_j, x_n)`` returning per-history-shot logits.

    Drop-in replacement for :class:`RelevanceScore`. Same forward and
    :meth:`softmax` shape contract — callers (notably
    :class:`AdaptiveOffensivePrior` and :class:`AdaptiveDefensiveField`)
    don't need to branch on type.

    Architectural motivation. The structured 5-scalar form is a sum
    of per-feature similarity terms; it cannot represent any cross-
    feature interaction (quarter × starter, period × position, etc.).
    Empirically that's a hard ceiling — the optimizer collapses onto
    whichever single feature has the strongest marginal signal (see
    the v1 G1-D diagnostic, 2026-05-15 working-log entry). The MLP
    can represent arbitrary smooth functions of the joint
    ``(z_j, x_n, games_ago)`` input.

    Parameters
    ----------
    context_dim : int, default 27
        Dimension of the per-shot context vector — must match
        :data:`shotcloud.data.context.CONTEXT_DIM`. Both query and
        history use this dimension.
    hidden_dim : int, default 64
        Hidden width of the single-hidden-layer MLP. ~3.6K params
        total at the default config; trivial relative to dataset
        size (1.86M shots).
    """

    def __init__(self, context_dim: int = 27, hidden_dim: int = 64) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        # +1 for the per-shot games_ago scalar (substituted with zeros
        # when games_ago is not supplied at forward time).
        in_dim = 2 * context_dim + 1
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        # Zero-init the output layer so step-0 logits are exactly zero
        # for every (z_j, x_n) pair → softmax is uniform → π matches
        # what RelevanceScore() produces at its all-zeros default. This
        # preserves the AdaptiveOffensivePrior warm-up contract: at
        # init, the relevance term is a uniform average over historical
        # shots, and q_self reduces to a fixed-kernel KDE.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        z_j: Tensor,
        x_n: Tensor,
        games_ago: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute per-shot logits, shape ``(B, max_N)``.

        Parameters
        ----------
        z_j : Tensor, shape ``(B, max_N, context_dim)``
        x_n : Tensor, shape ``(B, context_dim)``
            Raw context :math:`\\tilde x_n` — same convention as
            :class:`RelevanceScore` post the 2026-05-15 fix.
        games_ago : Tensor, shape ``(B, max_N)``, optional
            Per-shot recency feature. When ``None``, substituted with
            zeros so the MLP sees a consistent input dimension.
        mask : Tensor, shape ``(B, max_N)``, optional
            ``1`` for real shots, ``0`` for padding. Logits at padded
            positions are set to ``-inf`` so a downstream softmax
            produces zero weight there.

        Returns
        -------
        Tensor of shape ``(B, max_N)`` of pre-softmax logits.
        """
        if z_j.dim() != 3 or x_n.dim() != 2:
            raise ValueError(
                f"expected z_j (B, N, D) and x_n (B, D); got {tuple(z_j.shape)} and "
                f"{tuple(x_n.shape)}"
            )
        if z_j.shape[0] != x_n.shape[0] or z_j.shape[2] != x_n.shape[1]:
            raise ValueError(f"shape mismatch: z_j={tuple(z_j.shape)}, x_n={tuple(x_n.shape)}")
        if z_j.shape[2] != self.context_dim:
            raise ValueError(
                f"z_j has context_dim={z_j.shape[2]}, MLP was built with "
                f"context_dim={self.context_dim}"
            )

        b, max_n, _ = z_j.shape
        x_n_expanded = x_n.unsqueeze(1).expand(-1, max_n, -1)
        if games_ago is None:
            ga = torch.zeros(b, max_n, 1, device=z_j.device, dtype=z_j.dtype)
        else:
            if games_ago.shape != (b, max_n):
                raise ValueError(f"games_ago shape {tuple(games_ago.shape)} != ({b}, {max_n})")
            ga = games_ago.unsqueeze(-1)
        features = torch.cat([z_j, x_n_expanded, ga], dim=-1)
        logits: Tensor = self.fc2(torch.nn.functional.gelu(self.fc1(features))).squeeze(-1)

        if mask is not None:
            if mask.shape != logits.shape:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} != logits shape {tuple(logits.shape)}"
                )
            logits = logits.masked_fill(mask < 0.5, float("-inf"))

        return logits

    def softmax(
        self,
        z_j: Tensor,
        x_n: Tensor,
        games_ago: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Convenience: ``softmax_j[f_φ(z_j, x_n)]``."""
        logits = self.forward(z_j, x_n, games_ago, mask)
        return torch.softmax(logits, dim=-1)

    def params_as_floats(self) -> dict[str, float]:
        """Summary statistics over MLP parameters, for per-epoch logging.

        Returns mean and L2 norm of each layer's weights so the
        trainer's progress line and the diagnostic script can report
        meaningful numbers in place of the structured form's named
        scalars.
        """
        with torch.no_grad():
            return {
                "fc1_w_norm": float(self.fc1.weight.norm()),
                "fc1_b_norm": float(self.fc1.bias.norm()),
                "fc2_w_norm": float(self.fc2.weight.norm()),
                "fc2_b_norm": float(self.fc2.bias.norm()),
            }

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return f"context_dim={self.context_dim}, hidden_dim={self.hidden_dim}, n_params={n_params}"
