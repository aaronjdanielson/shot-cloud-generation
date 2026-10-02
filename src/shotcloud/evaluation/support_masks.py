"""Decompose a collaborative support set into own vs pooled sources.

The collaborative KDE's support set for a shooting context is a
flattened ``M = L * R`` set of (pool-player, shot) pairs: ``L``
pool players, ``R`` historical shots each. Some of those ``L``
players are the target player itself (own history); the rest are
other players selected by similarity/availability (pooled support).

Several post-training analyses --- the self-only support ablation
and the pooling diagnostics --- need the same partition of support
slots into *own* and *pooled*. This module provides the single
shared helper :func:`support_source_masks` so the partition logic
lives in one tested place.

Terminology: the estimator interpolates between player-specific
empirical support and *pooled* support from other players. We call
the other-player contribution "pooled support" (not "borrowing" or
"analogues").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SupportSourceMasks:
    """Boolean partition of a ``(B, M)`` collaborative support set.

    ``own`` and ``pooled`` are disjoint and their union is ``valid``
    (an invalid support slot is in neither).
    """

    own: Tensor  # (B, M) bool — valid support slots shot by the target player
    pooled: Tensor  # (B, M) bool — valid support slots shot by other players
    valid: Tensor  # (B, M) bool — valid (causal + real) support slots


def support_source_masks(
    analogue_idx: Tensor,
    player_idx: Tensor,
    n_shots_per_analogue: int,
    valid_mask: Tensor | None = None,
) -> SupportSourceMasks:
    """Partition the flattened support set into own / pooled / valid.

    Parameters
    ----------
    analogue_idx : Tensor of shape ``(B, L)``
        Vocab indices of the ``L`` pool players per row. The target
        player typically appears as one of these slots (the
        collaborative cache ensures own history is included when
        available).
    player_idx : Tensor of shape ``(B,)``
        Target-player vocab index per row.
    n_shots_per_analogue : int
        ``R`` --- the number of support shots per pool player. The
        flattened support index is ``m = l * R + r``.
    valid_mask : Tensor of shape ``(B, M)`` or None
        Optional causal + real support mask (``M = L * R``). When
        ``None``, every slot is treated as valid.

    Returns
    -------
    SupportSourceMasks
        ``own``, ``pooled``, ``valid`` each ``(B, M)`` bool.
        ``own`` and ``pooled`` partition ``valid``.
    """
    if analogue_idx.dim() != 2:
        raise ValueError(f"analogue_idx must be (B, L); got {tuple(analogue_idx.shape)}")
    b, ell = analogue_idx.shape
    if player_idx.shape != (b,):
        raise ValueError(f"player_idx must be (B,)={(b,)}; got {tuple(player_idx.shape)}")
    if n_shots_per_analogue <= 0:
        raise ValueError(f"n_shots_per_analogue must be positive; got {n_shots_per_analogue}")
    m = ell * n_shots_per_analogue

    if valid_mask is None:
        valid = torch.ones(b, m, dtype=torch.bool, device=analogue_idx.device)
    else:
        if valid_mask.shape != (b, m):
            raise ValueError(f"valid_mask must be (B, M)={(b, m)}; got {tuple(valid_mask.shape)}")
        valid = valid_mask.bool()

    # (B, L) self indicator → broadcast over the R shots per pool player.
    is_self_l = analogue_idx == player_idx.unsqueeze(-1)  # (B, L)
    is_self_m = is_self_l.unsqueeze(-1).expand(-1, -1, n_shots_per_analogue).reshape(b, m)  # (B, M)

    own = is_self_m & valid
    pooled = (~is_self_m) & valid
    return SupportSourceMasks(own=own, pooled=pooled, valid=valid)


__all__ = ["SupportSourceMasks", "support_source_masks"]
