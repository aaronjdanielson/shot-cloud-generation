"""Global shot-support pool + per-player index for collaborative KDE.

The v1.0 CollaborativeKDE stored four padded ``(n_players, max_R, *)``
buffers — one of ``(2,)`` coords, ``(CONTEXT_DIM,)`` context,
``(,)`` int64 date, and a ``(,)`` float32 occupancy mask. Every batch
gather read a different slice of those buffers, and a shot appearing
in two players' histories was duplicated in storage. This module
replaces that layout with a single flat **global pool** plus a per-player
**int64 index** into the pool:

* :class:`GlobalSupportPool` holds ``(coords, context, dates)`` for
  every shot exactly once.
* :class:`PerPlayerSupportIndex` holds an ``(n_players, max_R)`` int64
  table whose entries point into the pool, with ``-1`` marking empty
  slots.

Why bother? Three reasons:

1. **Deduplicated shot-attention compute.** The bilinear shot-attention
   form computes ``h_θ(z_j)`` on each shot context once per batch.
   With a global pool we can take ``torch.unique`` over the batch's
   shot ids, run the MLP on the unique-only context tensor, and
   gather back via ``inverse``. The v1.0 padded layout reuses
   ``shot_idx == 0`` for "this slot is empty" and provides no way to
   distinguish duplicates.
2. **Incremental updates.** Future causal pre-fitting can append new
   shots to the pool and grow the per-player index without rebuilding
   the entire ``(n_players, max_R, D)`` array — important when the
   snapshot calendar walks forward across seasons.
3. **Cheaper gather.** The gather of ``coords`` and ``dates`` for a
   batch is one ``index_select`` per buffer instead of three padded
   slices.

The v1.0 occupancy mask is now derived on the fly as
``index >= 0``; padded indices are ``clamp_min(0)``-ed before the
gather and the pulled values are masked out downstream.
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
        return int(self.coords.shape[0])

    @property
    def context_dim(self) -> int:
        return int(self.context.shape[1])


@dataclass(frozen=True)
class PerPlayerSupportIndex:
    """Per-player int64 pointers into a :class:`GlobalSupportPool`.

    Padded with ``-1`` so every player row has the same width
    ``max_R`` regardless of how many shots they actually have. The
    pad sentinel is exposed as :pyattr:`PAD_INDEX` for downstream code.
    """

    index: Tensor  # (n_players, max_R) int64, -1 = pad

    PAD_INDEX: int = _PAD_INDEX

    def __post_init__(self) -> None:
        if self.index.dim() != 2:
            raise ValueError(f"index must be 2-D (n_players, max_R); got {self.index.dim()}-D")
        if self.index.dtype != torch.int64:
            raise ValueError(f"index must be int64; got {self.index.dtype}")

    @property
    def n_players(self) -> int:
        return int(self.index.shape[0])

    @property
    def max_R(self) -> int:  # noqa: N802 — `R` is the model's per-player history cap
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
    truth for player ordering — the per-player-index row ``vocab.to_idx(pid)``
    holds ``pid``'s shot ids (or all ``-1`` if the player has no
    history in ``adaptive_kde``).

    Returned tensors live on CPU; the consumer (typically
    :class:`CollaborativeKDE`) registers them as module buffers and
    PyTorch handles the device transfer.
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
