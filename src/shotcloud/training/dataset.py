"""Player and opponent vocabularies.

Both vocabularies are bidirectional, string-normalized maps between raw
IDs and contiguous integer indices, matching the key convention of
:class:`~shotcloud.kde.HierarchicalKDE` and :class:`~shotcloud.kde.AdaptiveKDE`.
A vocabulary can therefore be built from KDE keys (strings) and queried
with raw pandas values (``np.int64``, ``np.str_``, ``str``). They are used
by :class:`~shotcloud.training.GibbsShotDataset` and by the per-shot
dataset in :mod:`shotcloud.legacy_pivot.shot_cell_dataset`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerVocab:
    """Bidirectional ``player_id ↔ index`` mapping.

    Player IDs are normalized via ``str()`` at construction and at
    lookup. Indices are contiguous in ``[0, n_players)`` and follow the
    sorted order of the string IDs, so they can index an
    ``nn.Embedding`` or a per-player table.
    """

    id_to_idx: dict[str, int]
    ids: tuple[str, ...]

    @classmethod
    def from_ids(cls, player_ids: Sequence[object]) -> PlayerVocab:
        """Build a vocabulary from raw IDs; duplicates are collapsed."""
        unique = sorted({str(p) for p in player_ids})
        id_to_idx = {pid: i for i, pid in enumerate(unique)}
        return cls(id_to_idx=id_to_idx, ids=tuple(unique))

    def __len__(self) -> int:
        return len(self.ids)

    def to_idx(self, player_id: object) -> int:
        """Return the index of ``player_id``; raise ``KeyError`` if absent."""
        key = str(player_id)
        if key not in self.id_to_idx:
            raise KeyError(f"player {player_id!r} not in vocab")
        return self.id_to_idx[key]

    def to_id(self, idx: int) -> str:
        """Return the string ID at index ``idx``."""
        return self.ids[idx]


@dataclass(frozen=True)
class OpponentVocab:
    """Bidirectional ``opponent_id ↔ index`` mapping.

    Same string-normalization convention as :class:`PlayerVocab`.
    Indices are contiguous in ``[0, n_opponents)``.
    """

    id_to_idx: dict[str, int]
    ids: tuple[str, ...]

    @classmethod
    def from_ids(cls, opponent_ids: Sequence[object]) -> OpponentVocab:
        """Build a vocabulary from raw IDs; duplicates are collapsed."""
        unique = sorted({str(o) for o in opponent_ids})
        id_to_idx = {oid: i for i, oid in enumerate(unique)}
        return cls(id_to_idx=id_to_idx, ids=tuple(unique))

    def __len__(self) -> int:
        return len(self.ids)

    def to_idx(self, opponent_id: object) -> int:
        """Return the index of ``opponent_id``; raise ``KeyError`` if absent."""
        key = str(opponent_id)
        if key not in self.id_to_idx:
            raise KeyError(f"opponent {opponent_id!r} not in vocab")
        return self.id_to_idx[key]

    def to_id(self, idx: int) -> str:
        """Return the string ID at index ``idx``."""
        return self.ids[idx]
