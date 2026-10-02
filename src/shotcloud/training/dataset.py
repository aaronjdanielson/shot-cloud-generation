"""Player / opponent vocabularies for the joint Gibbs trainer.

The pre-pivot per-shot dataset (``ShotCellDataset``) moved to
:mod:`shotcloud.legacy_pivot.shot_cell_dataset` on 2026-05-15.
The current production dataset is
:class:`~shotcloud.training.GibbsShotDataset`.

This module retains only the two vocab classes, which are
essential to both the legacy pre-pivot dataset and the current
``GibbsShotDataset``. Both are bidirectional string-normalized
maps that match the convention used by :class:`HierarchicalKDE`
and :class:`AdaptiveKDE` — callers can construct a vocab from
KDE keys (strings) and look up by raw pandas values
(``np.int64``, ``np.str_``, plain ``str``) without typing friction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerVocab:
    """Bidirectional ``player_id ↔ index`` mapping.

    Player IDs are normalized via ``str()`` at construction *and* at
    lookup, matching the convention used by :class:`HierarchicalKDE`
    and :class:`AdaptiveKDE`. Indices are contiguous ``[0, n_players)``
    so they can drive an ``nn.Embedding`` or index a per-player table.
    """

    id_to_idx: dict[str, int]
    ids: tuple[str, ...]

    @classmethod
    def from_ids(cls, player_ids: Sequence[object]) -> PlayerVocab:
        unique = sorted({str(p) for p in player_ids})
        id_to_idx = {pid: i for i, pid in enumerate(unique)}
        return cls(id_to_idx=id_to_idx, ids=tuple(unique))

    def __len__(self) -> int:
        return len(self.ids)

    def to_idx(self, player_id: object) -> int:
        key = str(player_id)
        if key not in self.id_to_idx:
            raise KeyError(f"player {player_id!r} not in vocab")
        return self.id_to_idx[key]

    def to_id(self, idx: int) -> str:
        return self.ids[idx]


@dataclass(frozen=True)
class OpponentVocab:
    """Bidirectional ``opponent_id ↔ index`` mapping.

    Same string-normalization convention as :class:`PlayerVocab`.
    Indices are contiguous ``[0, n_opponents)``.
    """

    id_to_idx: dict[str, int]
    ids: tuple[str, ...]

    @classmethod
    def from_ids(cls, opponent_ids: Sequence[object]) -> OpponentVocab:
        unique = sorted({str(o) for o in opponent_ids})
        id_to_idx = {oid: i for i, oid in enumerate(unique)}
        return cls(id_to_idx=id_to_idx, ids=tuple(unique))

    def __len__(self) -> int:
        return len(self.ids)

    def to_idx(self, opponent_id: object) -> int:
        key = str(opponent_id)
        if key not in self.id_to_idx:
            raise KeyError(f"opponent {opponent_id!r} not in vocab")
        return self.id_to_idx[key]

    def to_id(self, idx: int) -> str:
        return self.ids[idx]
