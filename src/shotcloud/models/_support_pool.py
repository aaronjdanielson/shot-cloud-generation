"""Global shot-support pool and per-player index for the collaborative KDE.

Per-shot data used by
:class:`~shotcloud.models.collaborative_kde.CollaborativeKDE` is stored
as a single flat **global pool** plus a per-player **int64 index** into
it:

* :class:`GlobalSupportPool` holds ``(coords, context, dates)`` for
  every shot exactly once.
* :class:`PerPlayerSupportIndex` holds an ``(n_players, max_R)`` int64
  table whose entries point into the pool, with ``-1`` marking empty
  slots.

Global shot ids let the bilinear shot attention run ``h_θ(z_j)`` once
per unique shot in a batch (``torch.unique`` over the gathered ids, then
gather back via the inverse), and a batch gather of coordinates or dates
is a single ``index_select``. The occupancy mask is ``index >= 0``;
padded indices are ``clamp_min(0)``-ed before the gather and the pulled
values are masked out downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from shotcloud.kde.adaptive import AdaptiveKDE
    from shotcloud.training.dataset import PlayerVocab


_PAD_INDEX: int = -1


@dataclass(frozen=True)
class GlobalSupportPool:
    """Read-only flat per-shot bank backing the collaborative KDE.

    Each shot appears exactly once. Lookup is by global ``int64`` id;
    the :class:`PerPlayerSupportIndex` provides per-player pointer
    lists into this pool.

    Attributes
    ----------
    coords : Tensor of shape ``(N_global, 2)``
        Per-shot ``(x, y)`` in court feet.
    context : Tensor of shape ``(N_global, context_dim)``
        Per-shot context features (the ``ContextEncoder`` output for
        that shot).
    dates : Tensor of shape ``(N_global,)``
        Per-shot date as int64 epoch days. Used by the causal mask.
    """

    coords: Tensor
    context: Tensor
    dates: Tensor

    def __post_init__(self) -> None:
        n = int(self.coords.shape[0])
        if self.coords.dim() != 2 or self.coords.shape[1] != 2:
            raise ValueError(f"coords must be (N, 2); got {tuple(self.coords.shape)}")
        if self.context.dim() != 2 or self.context.shape[0] != n:
            raise ValueError(f"context must be (N={n}, D); got {tuple(self.context.shape)}")
        if self.dates.dim() != 1 or self.dates.shape[0] != n:
            raise ValueError(f"dates must be (N={n},); got {tuple(self.dates.shape)}")
        if self.dates.dtype != torch.int64:
            raise ValueError(f"dates must be int64; got {self.dates.dtype}")

    @property
    def n_shots(self) -> int:
        """Number of shots in the pool."""
        return int(self.coords.shape[0])

    @property
    def context_dim(self) -> int:
        """Width of the per-shot context vectors."""
        return int(self.context.shape[1])


@dataclass(frozen=True)
class PerPlayerSupportIndex:
    """Per-player int64 pointers into a :class:`GlobalSupportPool`.

    Padded with ``-1`` so every player row has the same width
    ``max_R`` regardless of how many shots they actually have. The
    pad sentinel is exposed as :attr:`PAD_INDEX`.

    Attributes
    ----------
    index : Tensor of shape ``(n_players, max_R)``, int64
        Pool ids of each player's shots; ``-1`` marks padding.
    """

    index: Tensor

    PAD_INDEX: int = _PAD_INDEX

    def __post_init__(self) -> None:
        if self.index.dim() != 2:
            raise ValueError(f"index must be 2-D (n_players, max_R); got {self.index.dim()}-D")
        if self.index.dtype != torch.int64:
            raise ValueError(f"index must be int64; got {self.index.dtype}")

    @property
    def n_players(self) -> int:
        """Number of player rows."""
        return int(self.index.shape[0])

    @property
    def max_R(self) -> int:  # noqa: N802 — `R` is the model's per-player history cap
        """Row width: the longest per-player history."""
        return int(self.index.shape[1])

    def real_mask(self) -> Tensor:
        """Boolean ``(n_players, max_R)`` mask: True where a real shot lives."""
        return self.index >= 0


def build_support_pool_from_adaptive_kde(
    adaptive_kde: AdaptiveKDE,
    vocab: PlayerVocab,
) -> tuple[GlobalSupportPool, PerPlayerSupportIndex]:
    """Construct ``(GlobalSupportPool, PerPlayerSupportIndex)`` from a fitted AdaptiveKDE.

    Pool order: shots concatenated in ``vocab.ids`` order, each player's
    block in the order ``AdaptiveKDE`` stored them. The per-player index
    points into this concatenated layout. ``vocab`` is the source of
    truth for player ordering: the per-player-index row ``vocab.to_idx(pid)``
    holds ``pid``'s shot ids (or all ``-1`` if the player has no
    history in ``adaptive_kde``).

    Returned tensors live on CPU; the consumer (typically
    :class:`~shotcloud.models.collaborative_kde.CollaborativeKDE`)
    registers them as module buffers, which move with the module.

    Raises
    ------
    ValueError
        If ``adaptive_kde`` is unfitted, lacks per-shot coordinates,
        context, or dates, or has no history for any vocabulary player,
        or if ``vocab`` is empty.
    """
    if not adaptive_kde.is_fitted:
        raise ValueError("adaptive_kde must be fitted")
    if not adaptive_kde.coords:
        raise ValueError("adaptive_kde must store per-shot coords")
    if not adaptive_kde.context:
        raise ValueError("adaptive_kde must store per-shot context")
    if not adaptive_kde.dates:
        raise ValueError("adaptive_kde must store per-shot dates")

    n_players = len(vocab)
    if n_players == 0:
        raise ValueError("vocab is empty")

    # Pre-compute the max_R from any player that does have history.
    max_r = 0
    for pid in vocab.ids:
        if pid in adaptive_kde.cells:
            max_r = max(max_r, int(adaptive_kde.context[pid].shape[0]))
    if max_r == 0:
        raise ValueError("AdaptiveKDE has no history for any vocab player")

    coords_blocks: list[np.ndarray] = []
    context_blocks: list[np.ndarray] = []
    date_blocks: list[np.ndarray] = []
    index = np.full((n_players, max_r), _PAD_INDEX, dtype=np.int64)
    cursor = 0
    for pid in vocab.ids:
        p_idx = vocab.to_idx(pid)
        if pid not in adaptive_kde.cells:
            continue
        p_ctx = adaptive_kde.context[pid]
        p_coords = adaptive_kde.coords[pid]
        p_dates = adaptive_kde.dates[pid]
        n_p = int(p_ctx.shape[0])
        if n_p == 0:
            continue
        coords_blocks.append(np.asarray(p_coords, dtype=np.float32))
        context_blocks.append(np.asarray(p_ctx, dtype=np.float32))
        date_blocks.append(np.asarray(p_dates, dtype=np.int64))
        index[p_idx, :n_p] = np.arange(cursor, cursor + n_p, dtype=np.int64)
        cursor += n_p

    if cursor == 0:  # pragma: no cover — guarded above via max_r > 0
        raise ValueError("AdaptiveKDE has empty per-player histories despite max_r > 0")

    pool = GlobalSupportPool(
        coords=torch.from_numpy(np.concatenate(coords_blocks, axis=0)),
        context=torch.from_numpy(np.concatenate(context_blocks, axis=0)),
        dates=torch.from_numpy(np.concatenate(date_blocks, axis=0)),
    )
    index_t = PerPlayerSupportIndex(index=torch.from_numpy(index))
    return pool, index_t


__all__ = [
    "GlobalSupportPool",
    "PerPlayerSupportIndex",
    "build_support_pool_from_adaptive_kde",
]
