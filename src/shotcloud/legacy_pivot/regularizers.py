"""Phase 4 — regularizers for the context-adaptive KDE relevance weights.

Three loss-augmenting terms, all functions of the per-shot relevance
softmax ``π_φ,j(x_n)`` produced by
:class:`~shotcloud.models.RelevanceScore`. Each is a pure function with
no learnable state; the trainer applies ``λ * R(π, ...)`` and adds the
result to the spatial NLL.

Default scaling factors (locked in :doc:`docs/research_plan.md` §7):

==================  ===========  ===============================
term                default ``λ`` purpose
==================  ===========  ===============================
``entropy``         ``0.005``    floor on attention entropy;
                                 prevents collapse to one shot
``prior_anchor``    ``0.05``     keep ``π`` near a fixed-recency
                                 baseline ``π^(0)``
``ess``             ``0`` (off)  penalize tiny ``N_eff``;
                                 typically redundant with
                                 ``entropy``
==================  ===========  ===============================

All three reduce by mean over the batch; the trainer multiplies by
``λ`` and adds. The module accepts the masking tensor from the
``RelevanceScore`` forward so padded positions don't contribute.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    """Mean of ``values`` over the batch axis 0, optionally masked."""
    if mask is None:
        return values.mean()
    weight = mask.sum(dim=-1).clamp_min(1.0)  # rows can have variable real-shot counts
    return (values * weight).sum() / weight.sum()


def entropy_regularizer(pi: Tensor, mask: Tensor | None = None) -> Tensor:
    """Negative mean entropy ``-H(π) = Σ π log π`` per row.

    Adding this to the loss with positive λ pushes the model toward
    *higher* entropy (more uniform attention). Returns a scalar.

    Parameters
    ----------
    pi : Tensor of shape ``(B, max_N)``
        Softmax-normalized relevance weights.
    mask : Tensor of shape ``(B, max_N)``, optional
        ``1`` for real shots, ``0`` for padding. Padded positions have
        ``π = 0`` already, so they contribute ``0 · log 0 = 0`` (we
        guard against the actual computation).
    """
    safe_pi = pi.clamp_min(1e-12)
    log_pi = torch.log(safe_pi)
    # Per-row entropy.
    entropy_per_row = -(pi * log_pi).sum(dim=-1)  # (B,)
    # Negative-entropy is what we add to the loss.
    return -entropy_per_row.mean()


def prior_anchor_regularizer(pi: Tensor, pi_prior: Tensor, mask: Tensor | None = None) -> Tensor:
    """Squared deviation of ``log π`` from a fixed prior ``π^(0)``.

    Regularizes against drifting too far from a fixed-recency baseline
    (typically uniform or exponential-decay-on-games-ago). Returns a
    scalar.

    Parameters
    ----------
    pi : Tensor of shape ``(B, max_N)``
    pi_prior : Tensor of shape ``(B, max_N)``
        Same shape as ``pi`` — the fixed reference distribution. Must
        be strictly positive on real (unmasked) positions.
    mask : Tensor of shape ``(B, max_N)``, optional
    """
    if pi.shape != pi_prior.shape:
        raise ValueError(f"pi {tuple(pi.shape)} != pi_prior {tuple(pi_prior.shape)}")
    safe_pi = pi.clamp_min(1e-12)
    safe_prior = pi_prior.clamp_min(1e-12)
    diff = (torch.log(safe_pi) - torch.log(safe_prior)).pow(2)
    if mask is not None:
        diff = diff * mask
    # Mean over (B, max_N); use mask to count real positions.
    if mask is None:
        return diff.mean()
    n_real = mask.sum().clamp_min(1.0)
    return diff.sum() / n_real


def ess_regularizer(pi: Tensor, mask: Tensor | None = None) -> Tensor:
    """``1 / N_eff`` per row, mean over batch.

    ``N_eff = 1 / Σ π²``; small ``N_eff`` (concentrated attention)
    yields a large penalty. Setting ``λ_ess > 0`` tends to push toward
    higher entropy, similar to :func:`entropy_regularizer` but on a
    different scale (more sensitive to extreme concentration).

    Off by default; turn on if entropy regularization alone is not
    enough to prevent collapse to one or two shots.
    """
    inv_n_eff = pi.pow(2).sum(dim=-1)  # = 1 / N_eff per row, but as Σ π² ∈ (0, 1]
    return inv_n_eff.mean()


def fixed_recency_prior(games_ago: Tensor, mask: Tensor | None = None, lam: float = 1.0) -> Tensor:
    """Build the fixed-recency prior ``π^(0) ∝ exp(-λ · gamesAgo_j)``.

    Convenience for :func:`prior_anchor_regularizer`. ``games_ago`` is
    typically pre-normalized so ``λ ≈ 1`` is a reasonable scale.

    Returns the same shape as ``games_ago``, normalized so each row
    sums to 1 (over unmasked positions).
    """
    logits = -lam * games_ago
    if mask is not None:
        logits = logits.masked_fill(mask < 0.5, float("-inf"))
    return torch.softmax(logits, dim=-1)
