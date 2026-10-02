"""Player retrieval accuracy: top-k retrieval of the correct player.

For each player ``i``, we compute the distance from real cloud ``i`` to
*every* generated cloud, then check whether the correct player ``i`` is
in the top-k closest. A model that has collapsed to a league-average
generation will score near chance (1/N for top-1); a well-conditioned
model will retrieve the correct player with high probability.

Adapted from shot_flow's ``player_retrieval_accuracy``
(``shot_flow/src/shot_flow/evaluation/metrics.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shotcloud.evaluation.wasserstein import sliced_wasserstein

DistanceFn = Callable[[NDArray[np.floating], NDArray[np.floating]], float]


@dataclass(frozen=True)
class RetrievalResult:
    """Output of :func:`top_k_retrieval`."""

    top1_accuracy: float
    topk_accuracy: float
    mean_rank: float
    n_players: int
    k: int
    ranks: dict[Any, int]  # player_id → 1-indexed rank of self in retrieval


def top_k_retrieval(
    real_clouds: Mapping[Any, NDArray[np.floating]],
    gen_clouds: Mapping[Any, NDArray[np.floating]],
    *,
    k: int = 5,
    distance: DistanceFn | None = None,
    n_projections: int = 200,
    seed: int = 0,
) -> RetrievalResult:
    """Top-1 / top-k retrieval accuracy via pairwise distance.

    Parameters
    ----------
    real_clouds : mapping of ``player_id → (N_i, 2)`` real shot points
    gen_clouds : mapping of ``player_id → (M_i, 2)`` generated shot points
        Must have the same key set as ``real_clouds``.
    k : int, default 5
        Top-k threshold.
    distance : callable, optional
        ``(p, q) → float`` distance between two clouds. Default uses
        :func:`~shotcloud.evaluation.wasserstein.sliced_wasserstein`
        with the given ``n_projections`` and ``seed``.
    n_projections, seed : passed to the default sliced-Wasserstein.

    Returns
    -------
    RetrievalResult
    """
    if set(real_clouds) != set(gen_clouds):
        only_real = set(real_clouds) - set(gen_clouds)
        only_gen = set(gen_clouds) - set(real_clouds)
        raise ValueError(
            "real_clouds and gen_clouds must have the same key set; "
            f"only in real: {sorted(only_real)[:5]}; "
            f"only in gen: {sorted(only_gen)[:5]}"
        )
    players: Sequence[Any] = list(real_clouds.keys())
    n = len(players)
    if n < 2:
        raise ValueError(f"need at least 2 players for retrieval; got {n}")
    if k < 1 or k > n:
        raise ValueError(f"k={k} must satisfy 1 <= k <= n_players={n}")

    if distance is None:

        def distance(p: NDArray[np.floating], q: NDArray[np.floating]) -> float:
            return sliced_wasserstein(p, q, n_projections=n_projections, seed=seed)

    # dist_matrix[i, j] = distance(real[i], gen[j]).
    dist_matrix = np.full((n, n), np.nan)
    for i, pi in enumerate(players):
        for j, pj in enumerate(players):
            dist_matrix[i, j] = distance(real_clouds[pi], gen_clouds[pj])

    # For each row i, the rank of column i (the correct player) when sorting ascending.
    ranks: dict[Any, int] = {}
    rank_values: list[int] = []
    for i, pi in enumerate(players):
        order = np.argsort(dist_matrix[i])
        rank = int(np.where(order == i)[0][0]) + 1  # 1-indexed
        ranks[pi] = rank
        rank_values.append(rank)

    rank_arr = np.array(rank_values)
    return RetrievalResult(
        top1_accuracy=float((rank_arr == 1).mean()),
        topk_accuracy=float((rank_arr <= k).mean()),
        mean_rank=float(rank_arr.mean()),
        n_players=n,
        k=k,
        ranks=ranks,
    )
