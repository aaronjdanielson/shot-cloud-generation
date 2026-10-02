"""Relevance scores for context-adaptive KDE over a player's shot history.

A relevance score ``f_φ(z_j, x_n)`` assigns each historical shot ``j``
(with context ``z_j``) a logit measuring its relevance to the target
context ``x_n``; a softmax over the history turns the logits into kernel
weights. Both arguments are the raw 27-dim context :math:`\\tilde x_n`
of :class:`shotcloud.data.ContextEncoder`, so the named feature slices
stay interpretable.

:class:`RelevanceScore` is the structured form: five named scalars, each
acting on a specific slice of the context vector.

* ``β_q``: dot product of the ``period_onehot`` slices (4 dims), a
  same-quarter bonus.
* ``β_m``: negative absolute difference of ``time_in_period`` (1 dim), a
  similar-time bonus.
* ``β_t``: negative absolute difference of ``season_recency`` (1 dim), a
  nearby-date bonus.
* ``β_o``: dot product of the ``opp_efficiency_onehot`` slices (4 dims),
  a same-opponent-bucket bonus.
* ``λ_g``: exponential decay in a per-shot age ``games_ago`` that is not
  part of the context vector, a recency baseline.

:class:`RelevanceMLP` is a small MLP with the same interface that can
represent cross-feature interactions.

Notes
-----
* All five scalars are unconstrained reals. ``λ_g ≥ 0`` gives the
  recency interpretation (older shots down-weighted), but a negative
  value, which up-weights older shots, is allowed.
* ``λ_g`` is not a similarity between ``z_j`` and ``x_n`` but an
  absolute prior on how old shot ``j`` is, so its input is passed
  separately as ``games_ago``. When ``games_ago`` is omitted the recency
  term is zero and ``λ_g`` receives no gradient.

Both scorers are used by the deprecated grid-cell priors
(:class:`~shotcloud.legacy_pivot.adaptive_prior.AdaptiveOffensivePrior`,
:class:`~shotcloud.legacy_pivot.adaptive_defensive.AdaptiveDefensiveField`).
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
        means uniform relevance at initialization (every shot is equally
        relevant), so the adaptive KDE starts equivalent to a fixed
        uniform-recency Gaussian-kernel density. Any deviation is
        data-driven.
    init_lambda_g : float, default 0.0
        Initial value of the recency-decay scalar ``λ_g``. The
        ``games_ago`` tensor is multiplied by ``-λ_g``, so a positive
        ``λ_g`` downweights older shots. Default 0.0 disables the
        decay until the optimizer engages it.
    beta_max : float or None, default None
        Bound that guards against overfitting. When set, the four
        similarity scalars ``β_{q,m,t,o}`` are passed through
        ``β_max · tanh(θ / β_max)``, so each effective β lies in
        ``(-β_max, β_max)``; the underlying ``nn.Parameter`` ``θ`` stays
        unconstrained. ``λ_g`` is not bounded: as a recency decay it does
        not carry the same overfitting risk. ``None`` (default) leaves
        all scalars unbounded. Without a bound a single similarity term
        can grow large and dominate the relevance softmax; for example,
        ``β_max = 2.0`` caps each term's effect on a shot's relative
        weight at a factor of ``e^2 ≈ 7.4``.
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

        # Cache the bounds of the named context slices as plain ints.
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
        """Effective (bounded, if ``beta_max`` is set) ``β_q``."""
        return self._bound(self.beta_q)

    @property
    def effective_beta_m(self) -> Tensor:
        """Effective (bounded, if ``beta_max`` is set) ``β_m``."""
        return self._bound(self.beta_m)

    @property
    def effective_beta_t(self) -> Tensor:
        """Effective (bounded, if ``beta_max`` is set) ``β_t``."""
        return self._bound(self.beta_t)

    @property
    def effective_beta_o(self) -> Tensor:
        """Effective (bounded, if ``beta_max`` is set) ``β_o``."""
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
            Per-target-shot raw context :math:`\\tilde x_n`.
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
        Tensor of shape ``(B, max_N)``
            Per-shot pre-softmax logits.
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
            # -inf on padded positions so the softmax gives them zero weight.
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
        """Detached float view of the five effective parameters, for logging.

        With ``beta_max`` set, the bounded β values that enter the logits
        are returned, not the raw underlying θ.
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

    Interchangeable with :class:`RelevanceScore`: the ``forward`` and
    :meth:`softmax` signatures and shapes are the same, so callers need
    not branch on type.

    The structured five-scalar form is a sum of per-feature similarity
    terms and cannot represent cross-feature interactions (quarter ×
    starter, period × position, ...), which leaves the optimizer to
    concentrate on whichever single feature has the strongest marginal
    signal. The MLP can represent smooth functions of the joint
    ``(z_j, x_n, games_ago)`` input.

    Parameters
    ----------
    context_dim : int, default 27
        Dimension of the per-shot context vector — must match
        :data:`shotcloud.data.context.CONTEXT_DIM`. Both query and
        history use this dimension.
    hidden_dim : int, default 64
        Hidden width of the single-hidden-layer MLP (about 3.6K
        parameters at the defaults).
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
        # Zero-init the output layer so the initial logits are exactly zero
        # for every (z_j, x_n) pair and the softmax is uniform, matching
        # RelevanceScore at its all-zero default: at initialization the
        # self-KDE is a uniform average over historical shots, i.e. a
        # fixed-kernel KDE.
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
            Per-historical-shot context, padded along the second axis.
        x_n : Tensor, shape ``(B, context_dim)``
            Raw context :math:`\\tilde x_n`, as for :class:`RelevanceScore`.
        games_ago : Tensor, shape ``(B, max_N)``, optional
            Per-shot recency feature. When ``None``, substituted with
            zeros so the MLP sees a consistent input dimension.
        mask : Tensor, shape ``(B, max_N)``, optional
            ``1`` for real shots, ``0`` for padding. Logits at padded
            positions are set to ``-inf`` so a downstream softmax
            produces zero weight there.

        Returns
        -------
        Tensor of shape ``(B, max_N)``
            Per-shot pre-softmax logits.
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

        Returns the L2 norm of each layer's weight and bias, reported in
        place of the structured form's named scalars.
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
