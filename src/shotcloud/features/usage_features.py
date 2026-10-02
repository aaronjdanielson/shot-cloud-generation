"""Causal usage state vector for the residual-tilt encoder.

After the density-surface evaluation closed the kernel-shape axis
(Tier-1a real win, RT artifact), the next predictive-capacity item
is **count↔location coupling** — capturing the basketball fact that
high-volume player-games have qualitatively different shot
distributions than low-volume ones (more pull-up threes for volume
scorers, more rim/paint for high-usage bigs, more corner / catch-and-
shoot for role players). The first-cut implementation routes a small
causal usage-state vector ``u_{p,t}`` into the residual-tilt encoder
``R_θ(s_m, x, h, u_{p,t})``; the rest of the model (retrieval,
pooling gate, D-lite-zone, source/zone σ) is unchanged. This isolates
"does causal usage information improve spatial support reweighting?"
as a single-axis ablation.

The usage vector slices three slots from the existing causal
:class:`shotcloud.data.player_traits.PlayerTraitsTable` (built at
training start from snapshot store + bio + game logs, z-scored per
snapshot, causal-by-construction). The slots correspond to:

* ``log1p_minutes_M`` — ``log(1 + M_p^{<t_m})`` — log historical minutes.
* ``log1p_fga_S``     — ``log(1 + S_p^{<t_m})`` — log historical FGA.
* ``log_shot_density`` — ``log((S+1) / (M+1))`` — historical shot rate.

All three are already z-scored when the trait table is built, so
the residual encoder consumes a unit-scaled vector with no further
normalization. The recorded ``USAGE_SLOT_INDICES`` are used both at
training time (to slice the trait tensor) and at eval-reconstruction
time (to extract the same features from the loaded checkpoint).

The constraint that makes this safe: ``u_{p,t}`` is a function of
**snapshot** (player + time), never a function of the realized
game's shot count ``K`` or the realized shot locations. So the
usage signal is causal-by-construction at every batch row, and the
ablation result generalizes to held-out games.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

from shotcloud.data.player_traits import SLOT_NAMES

#: Trait-slot indices for the three causal usage features. Resolved
#: at module import against the trait table's stable
#: :data:`SLOT_NAMES` layout so a future re-ordering of slots is
#: caught here rather than silently shifting which slots get sliced.
USAGE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "log1p_minutes_M",
    "log1p_fga_S",
    "log_shot_density",
)

USAGE_SLOT_INDICES: Final[tuple[int, ...]] = tuple(
    SLOT_NAMES.index(name) for name in USAGE_FEATURE_NAMES
)

#: Dimension of the usage vector consumed by the residual encoder.
USAGE_DIM: Final[int] = len(USAGE_FEATURE_NAMES)

#: Dimension of the augmented usage vector when the predicted count
#: ``K̂`` from the count head is appended (Tier-2 count-location
#: coupling, ablation B2). Order is
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
        The collaborative KDE's causal trait buffer (``offensive_prior
        .traits`` on either backbone). z-scored per snapshot upstream,
        so the sliced columns are unit-scale.
    player_idx : Tensor of shape ``(B,)`` int64
    snapshot_idx : Tensor of shape ``(B,)`` int64

    Returns
    -------
    Tensor of shape ``(B, USAGE_DIM)`` float
        Per-row usage vector. Order matches
        :data:`USAGE_FEATURE_NAMES`.
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
