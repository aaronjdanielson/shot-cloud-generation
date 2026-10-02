"""Causal usage state vector for the residual-tilt encoder.

High-volume player-games have systematically different shot
distributions than low-volume ones (more pull-up threes for volume
scorers, more rim and paint attempts for high-usage bigs, more corner
catch-and-shoot attempts for role players). The usage branch of the
residual tilt captures this count-location coupling by feeding a small
causal usage-state vector ``u_{p,t}`` into the residual-tilt encoder
``R_θ(s_m, x, h, u_{p,t})``.

The usage vector slices three slots of the causal
:class:`~shotcloud.data.player_traits.PlayerTraitsTable`:

* ``log1p_minutes_M`` — ``log(1 + M_p^{<t_m})`` — log historical minutes.
* ``log1p_fga_S``     — ``log(1 + S_p^{<t_m})`` — log historical FGA.
* ``log_shot_density`` — ``log((S+1) / (M+1))`` — historical shot rate.

The trait table z-scores every slot per snapshot, so the encoder
receives a unit-scaled vector without further normalization.
:data:`USAGE_SLOT_INDICES` is used both at training time and when
reconstructing a model from a checkpoint, so the same slots are sliced
in both places.

Causality: ``u_{p,t}`` is a function of the snapshot (player and time)
only, never of the realized game's shot count ``K`` or shot locations.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

from shotcloud.data.player_traits import SLOT_NAMES

#: Names of the three causal usage features, in vector order. The slot
#: indices are resolved by name against the trait table's
#: :data:`~shotcloud.data.player_traits.SLOT_NAMES`, so a reordering of
#: trait slots cannot silently change which features are sliced.
USAGE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "log1p_minutes_M",
    "log1p_fga_S",
    "log_shot_density",
)

#: Trait-slot indices of :data:`USAGE_FEATURE_NAMES`.
USAGE_SLOT_INDICES: Final[tuple[int, ...]] = tuple(
    SLOT_NAMES.index(name) for name in USAGE_FEATURE_NAMES
)

#: Dimension of the usage vector consumed by the residual encoder.
USAGE_DIM: Final[int] = len(USAGE_FEATURE_NAMES)

#: Dimension of the augmented usage vector when the count head's
#: score ``K̂`` is appended. Order is
#: ``[u_{p,t} (USAGE_DIM,), detach(K̂) (1,)]``; the ``K̂`` column is
#: always last so the slot indices for ``u_{p,t}`` stay stable.
USAGE_KHAT_DIM: Final[int] = USAGE_DIM + 1

__all__ = [
    "USAGE_DIM",
    "USAGE_FEATURE_NAMES",
    "USAGE_KHAT_DIM",
    "USAGE_SLOT_INDICES",
    "extract_usage",
]


def extract_usage(
    traits: Tensor,
    player_idx: Tensor,
    snapshot_idx: Tensor,
) -> Tensor:
    """Slice the causal usage vector for a batch of (player, snapshot) rows.

    Parameters
    ----------
    traits : Tensor of shape ``(n_players, n_snapshots, trait_dim)``
        The collaborative KDE's causal trait buffer
        (``offensive_prior.traits`` on either support backend), z-scored
        per snapshot, so the sliced columns are unit-scale.
    player_idx : Tensor of shape ``(B,)`` int64
        Player vocabulary index per row.
    snapshot_idx : Tensor of shape ``(B,)`` int64
        Snapshot index per row.

    Returns
    -------
    Tensor of shape ``(B, USAGE_DIM)`` float
        Per-row usage vector. Order matches
        :data:`USAGE_FEATURE_NAMES`.

    Raises
    ------
    ValueError
        If ``traits`` is not 3-D or the index tensors are not matching
        1-D tensors.
    """
    if traits.dim() != 3:
        raise ValueError(
            f"traits must be (n_players, n_snapshots, trait_dim); got {tuple(traits.shape)}"
        )
    if player_idx.dim() != 1:
        raise ValueError(f"player_idx must be (B,); got {tuple(player_idx.shape)}")
    if snapshot_idx.shape != player_idx.shape:
        raise ValueError(
            f"snapshot_idx must match player_idx shape; got "
            f"{tuple(snapshot_idx.shape)} vs {tuple(player_idx.shape)}"
        )
    rows = traits[player_idx, snapshot_idx]  # (B, trait_dim)
    slot_idx = torch.tensor(USAGE_SLOT_INDICES, device=traits.device, dtype=torch.long)
    return rows.index_select(dim=-1, index=slot_idx)  # (B, USAGE_DIM)
