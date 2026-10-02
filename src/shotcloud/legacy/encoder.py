"""Player-identity encoders for the low-rank tilt decoder.

Deprecated; retained to reproduce the player-embedding tilt baselines and
to load checkpoints written by
:func:`~shotcloud.legacy_pivot.trainer.train_decoder`. Superseded by
:class:`~shotcloud.models.ContextResidualEncoder`.

These encoders produce the vector ``u`` for
:class:`~shotcloud.legacy_pivot.tilt_decoder.LowRankTiltDecoder` by
looking up a player-id embedding. The current residual encoder conditions
on context features and has **no player-id input**: player identity is
carried by the player's own support shots, and a player embedding in the
residual would duplicate that capacity and let the residual define, rather
than refine, the spatial geometry.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn


class PlayerEmbeddingEncoder(nn.Module):
    """Lookup-table encoder: ``u_p = embedding[player_idx]``.

    Parameters
    ----------
    n_players : int
        Vocabulary size. Must cover every player ID the trainer will see.
    rank : int
        Output dimension; **must equal the rank of the paired**
        :class:`~shotcloud.legacy_pivot.tilt_decoder.LowRankTiltDecoder`.
    zero_init : bool, default False
        If ``True`` the embedding table starts at zero so ``u`` is
        identically zero. **For training this is usually wrong**: when
        both ``V = 0`` (decoder zero-init) and ``u = 0`` (encoder
        zero-init), ``∂(u^T V)/∂V = u = 0`` and ``∂(u^T V)/∂u = V = 0``,
        so neither factor receives a gradient signal — training is
        trapped at the dead zero. The standard low-rank factorization
        pattern is to zero-init **one** factor; the decoder is
        zero-initialized (preserving the prior-equal-to-``q_0``
        invariant) and the encoder is randomly initialized.

        Set ``zero_init=True`` only when you want both factors at zero,
        e.g., to verify the invariant before any training step.
    """

    def __init__(self, n_players: int, rank: int, zero_init: bool = False) -> None:
        super().__init__()
        if n_players <= 0:
            raise ValueError(f"n_players must be positive, got {n_players}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.n_players = n_players
        self.rank = rank
        self.zero_init = zero_init

        self.embedding = nn.Embedding(n_players, rank)
        if zero_init:
            nn.init.zeros_(self.embedding.weight)
        else:
            init_std = 1.0 / (rank**0.5)
            nn.init.normal_(self.embedding.weight, std=init_std)

    def forward(self, player_idx: Tensor) -> Tensor:
        """Look up the per-player context vector.

        Parameters
        ----------
        player_idx : LongTensor of shape ``(batch,)``

        Returns
        -------
        Tensor of shape ``(batch, rank)``.
        """
        if player_idx.dim() != 1:
            raise ValueError(
                f"player_idx must be 1-D LongTensor; got shape {tuple(player_idx.shape)}"
            )
        if player_idx.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"player_idx must be integer dtype; got {player_idx.dtype}")
        return cast(Tensor, self.embedding(player_idx))

    def extra_repr(self) -> str:
        return f"n_players={self.n_players}, rank={self.rank}, zero_init={self.zero_init}"


class PlayerPositionEncoder(nn.Module):
    """Position-aware encoder: ``u_p = player_emb[p] + position_emb[g(p)]``.

    Adds a position-group prior to the per-player embedding. For sparse
    players the position embedding gives them a meaningful starting
    point even when the player-specific embedding is poorly trained;
    for dense players the player embedding can override the position
    contribution. The mapping ``g(p)`` is fixed at construction (each
    player belongs to exactly one position group).

    Position also enters :class:`~shotcloud.kde.HierarchicalKDE` (as a
    shrinkage target) and :class:`~shotcloud.legacy.product.KDEProduct`
    (as the geometric term ``q̂_g(p)``); this encoder surfaces it in the
    neural correction as well.

    Parameters
    ----------
    n_players : int
        Player vocabulary size.
    n_positions : int
        Position vocabulary size (typically 3: big / wing / guard).
    rank : int
        Output dimension; **must equal the rank of the paired**
        :class:`~shotcloud.legacy_pivot.tilt_decoder.LowRankTiltDecoder`.
    player_to_position : sequence of int, length ``n_players``
        ``player_to_position[player_idx]`` gives the position index for
        that player. Stored as a non-trainable buffer so it persists
        across save/load.
    zero_init_player : bool, default False
        Mirrors :class:`PlayerEmbeddingEncoder` — random by default to
        avoid the dead-zero saddle when the decoder is also zero-init.
    zero_init_position : bool, default False
        Random by default. Setting ``True`` makes the encoder identical
        to :class:`PlayerEmbeddingEncoder` at step 0.
    """

    # Class-level annotation: `register_buffer` types attributes as
    # `Tensor | Module`; declaring it here narrows it to `Tensor`.
    player_to_position: Tensor

    def __init__(
        self,
        n_players: int,
        n_positions: int,
        rank: int,
        player_to_position: Sequence[int],
        zero_init_player: bool = False,
        zero_init_position: bool = False,
    ) -> None:
        super().__init__()
        if n_players <= 0:
            raise ValueError(f"n_players must be positive, got {n_players}")
        if n_positions <= 0:
            raise ValueError(f"n_positions must be positive, got {n_positions}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if len(player_to_position) != n_players:
            raise ValueError(
                f"player_to_position has length {len(player_to_position)}; "
                f"expected n_players={n_players}"
            )
        for i, p in enumerate(player_to_position):
            if not (0 <= int(p) < n_positions):
                raise ValueError(f"player_to_position[{i}]={p} out of range [0, {n_positions})")

        self.n_players = n_players
        self.n_positions = n_positions
        self.rank = rank
        self.zero_init_player = zero_init_player
        self.zero_init_position = zero_init_position

        self.player_emb = nn.Embedding(n_players, rank)
        self.position_emb = nn.Embedding(n_positions, rank)

        if zero_init_player:
            nn.init.zeros_(self.player_emb.weight)
        else:
            nn.init.normal_(self.player_emb.weight, std=1.0 / (rank**0.5))

        if zero_init_position:
            nn.init.zeros_(self.position_emb.weight)
        else:
            nn.init.normal_(self.position_emb.weight, std=1.0 / (rank**0.5))

        self.register_buffer(
            "player_to_position",
            torch.tensor(list(player_to_position), dtype=torch.long),
        )

    def forward(self, player_idx: Tensor) -> Tensor:
        """Look up ``u_p = player_emb[p] + position_emb[g(p)]``."""
        if player_idx.dim() != 1:
            raise ValueError(
                f"player_idx must be 1-D LongTensor; got shape {tuple(player_idx.shape)}"
            )
        if player_idx.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"player_idx must be integer dtype; got {player_idx.dtype}")
        u_player = self.player_emb(player_idx)
        position_idx = self.player_to_position[player_idx]
        u_position = self.position_emb(position_idx)
        return cast(Tensor, u_player + u_position)

    def extra_repr(self) -> str:
        return f"n_players={self.n_players}, n_positions={self.n_positions}, rank={self.rank}"
