"""Top-L analogue retrieval cache for the collaborative KDE (Phase 3).

For each (target player, snapshot anchor), precompute the top-L most
similar players in trait space. The cache is consumed by
:class:`CollaborativeKDE` (Phase 4) — each forward pass simply gathers
``cache[player_idx, snapshot_idx]`` to get the analogue indices for
the batch's player-level attention.

Per :doc:`docs/model_spec.md`, the retrieval is:

.. math::

    \\mathcal N_p^{(t_m)}
    = \\mathrm{TopL}_{p'} \\mathrm{cosine}(u_p(t_m), u_{p'}(t_m))

with the target player ``p`` always included in :math:`\\mathcal N_p^{(t_m)}`.
Cold-start retrieval works automatically: when a player's play-derived
traits are all zero, the cosine reduces to similarity on the
biographical block (height/weight/age/position), and analogues with
similar physical profile surface naturally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from shotcloud.data.player_traits import PlayerTraitsTable


@dataclass(frozen=True)
class AnalogueRetrievalCache:
    """(n_players, n_snapshots, L) int64 cache of analogue player indices.

    Each row ``cache[p, m, :]`` is the top-L analogue *indices* (into
    ``player_ids``) for target player ``player_ids[p]`` at snapshot
    ``snapshot_anchors[m]``. Indices are sorted by descending
    cosine similarity. When ``ensure_self`` was True (default),
    ``cache[p, m, 0] == p`` for every (p, m).
    """

    analogues: NDArray[np.int64]
    player_ids: NDArray[np.int64]
    snapshot_anchors: NDArray[np.datetime64]
    L: int

    @property
    def n_players(self) -> int:
        return int(self.analogues.shape[0])

    @property
    def n_snapshots(self) -> int:
        return int(self.analogues.shape[1])


def _l2_normalize_rows(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise L2 normalize. Zero-norm rows are left as zero vectors.

    Cosine similarity with a zero vector evaluates to 0 (undefined),
    so zero-norm rows simply don't contribute to anyone's top-L. The
    target player's ``ensure_self`` slot guarantees they still get an
    analogue list.
    """
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    # Avoid div-by-zero: where norm is 0, leave the vector as zeros
    # (so the result row is 0, cosine == 0 everywhere → topk arbitrary).
    safe = np.where(norms > 0, norms, 1.0)
    return (x / safe).astype(np.float64)


def build_analogue_cache(
    traits: PlayerTraitsTable,
    L: int = 15,
    *,
    ensure_self: bool = True,
) -> AnalogueRetrievalCache:
    """Build the per-(player, snapshot) top-L analogue cache.

    Parameters
    ----------
    traits : PlayerTraitsTable
        Output of :func:`shotcloud.data.player_traits.build_player_traits_table`.
    L : int, default 15
        Number of analogues per target. Must be ≤ ``n_players``.
    ensure_self : bool, default True
        When True, the target player is guaranteed to be the first
        entry of every analogue list. This matches the spec's
        ``s_{p,p}(x,t) = b_same · 1{p=p'} + ...`` self-bias term —
        the model can prefer self via ``b_same`` only if self is in
        the analogue set.

    Returns
    -------
    AnalogueRetrievalCache
        Frozen wrapper around the int64 (n_players, n_snapshots, L)
        analogue indices.
    """
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    n_players = traits.n_players
    if n_players < L:
        raise ValueError(f"L={L} exceeds n_players={n_players}")

    analogues = np.zeros((n_players, traits.n_snapshots, L), dtype=np.int64)

    for m in range(traits.n_snapshots):
        u = traits.traits[:, m, :].astype(np.float64)
        u_norm = _l2_normalize_rows(u)
        # (n_players, n_players) cosine similarity matrix.
        sim = u_norm @ u_norm.T

        if ensure_self:
            # Force self to rank first by setting the diagonal to +inf.
            # This both guarantees inclusion and stable position 0.
            np.fill_diagonal(sim, np.inf)

        # Top-L per row, sorted by descending similarity. argpartition
        # gives the top-L unordered in O(n_players); we then sort
        # within the top-L for stable ranking.
        partition_idx = np.argpartition(-sim, kth=L - 1, axis=-1)[:, :L]
        rows = np.arange(n_players)[:, None]
        top_sim = sim[rows, partition_idx]
        order_within = np.argsort(-top_sim, axis=-1)
        analogues[:, m, :] = partition_idx[rows, order_within]

    return AnalogueRetrievalCache(
        analogues=analogues,
        player_ids=traits.player_ids.copy(),
        snapshot_anchors=traits.snapshot_anchors.copy(),
        L=L,
    )
