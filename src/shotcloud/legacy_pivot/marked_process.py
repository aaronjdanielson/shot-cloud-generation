"""End-to-end marked point process: timing × spatial decoder × base measure.

Deprecated; retained as a minimal sampler for the grid-cell decoder.
Superseded by the count, timing, and spatial factors trained jointly by
:mod:`shotcloud.training.train_gibbs`.

Implements the player-game generative model

.. math::

    p(S_n \\mid x_n)
    = p_\\eta(K_n, \\tau_{1:K_n} \\mid x_n)
        \\prod_{i=1}^{K_n}
        p_\\theta(c_{n,i} \\mid x_n, h_{n,i}, \\tau_{n,i}).

Notes
-----
No context encoder is attached: the tilt context ``u`` is fixed at
zero, so the tilt term ``u^T v_c`` vanishes and the spatial decoder
reproduces the base measure ``q_0`` exactly. The optional outcome term
``p_ψ(o | c, ...)`` is not modeled.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from shotcloud.grids import CourtGrid
from shotcloud.legacy import KDEProduct
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder
from shotcloud.legacy_pivot.timing import TimingModel


@dataclass(frozen=True)
class ShotSequence:
    """One game's worth of shots: timing + cells + dequantized locations."""

    taus: NDArray[np.float64]
    cells: NDArray[np.int64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]

    def __post_init__(self) -> None:
        n = len(self.taus)
        for name, arr in (("cells", self.cells), ("x", self.x), ("y", self.y)):
            if len(arr) != n:
                raise ValueError(f"length mismatch: taus has {n}, {name} has {len(arr)}")

    @property
    def n_shots(self) -> int:
        return len(self.taus)


@dataclass(frozen=True)
class ShotCloud:
    """Empirical spatial measure aggregated from R independent player-games.

    The mathematical shot cloud
    :math:`\\mu_\\Theta(\\cdot \\mid x)` is the limit of this empirical
    measure as :math:`R \\to \\infty`.
    """

    taus: NDArray[np.float64]
    cells: NDArray[np.int64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    n_games: int

    @property
    def total_shots(self) -> int:
        return len(self.taus)

    @property
    def shots_per_game(self) -> float:
        return self.total_shots / self.n_games if self.n_games > 0 else 0.0


@dataclass
class ShotCloudProcess:
    """Marked point process over a discrete court grid.

    Parameters
    ----------
    timing_model : TimingModel
        Samples ``(K, τ_{1:K})`` and evaluates ``log p(K, τ | x)``.
    spatial_decoder : LowRankTiltDecoder
        Computes ``log p_θ(c | x, h, τ)`` from ``log q_0`` and a context
        vector ``u``. The process always feeds ``u = 0``.
    base_measure : KDEProduct
        Provides ``log q_0(c | player_id)``.

    The decoder's ``n_cells`` must match the base measure's grid.
    """

    timing_model: TimingModel
    spatial_decoder: LowRankTiltDecoder
    base_measure: KDEProduct

    def __post_init__(self) -> None:
        if self.spatial_decoder.n_cells != self.grid.n_cells:
            raise ValueError(
                f"spatial_decoder.n_cells ({self.spatial_decoder.n_cells}) "
                f"!= grid.n_cells ({self.grid.n_cells})"
            )

    @property
    def grid(self) -> CourtGrid:
        return self.base_measure.grid

    # ----------------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------------

    def sample(
        self,
        player_id: str,
        context: Mapping[str, object] | None = None,
        rng: np.random.Generator | None = None,
    ) -> ShotSequence:
        """Sample a single player-game shot sequence."""
        rng = rng if rng is not None else np.random.default_rng()
        ctx: Mapping[str, object] = context if context is not None else {}

        K, taus = self.timing_model.sample(ctx, rng)
        if K == 0:
            return ShotSequence(
                taus=np.array([], dtype=np.float64),
                cells=np.array([], dtype=np.int64),
                x=np.array([], dtype=np.float64),
                y=np.array([], dtype=np.float64),
            )

        cells = self._sample_cells(player_id, K, rng)
        x, y = self.grid.dequantize(cells, rng)
        return ShotSequence(taus=taus, cells=cells, x=x, y=y)

    def sample_shot_cloud(
        self,
        player_id: str,
        R: int = 500,
        context: Mapping[str, object] | None = None,
        rng: np.random.Generator | None = None,
    ) -> ShotCloud:
        """Sample R independent player-games and aggregate into a shot cloud."""
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        rng = rng if rng is not None else np.random.default_rng()
        seqs = [self.sample(player_id, context, rng) for _ in range(R)]
        return ShotCloud(
            taus=np.concatenate([s.taus for s in seqs]),
            cells=np.concatenate([s.cells for s in seqs]).astype(np.int64),
            x=np.concatenate([s.x for s in seqs]),
            y=np.concatenate([s.y for s in seqs]),
            n_games=R,
        )

    # ----------------------------------------------------------------------
    # Likelihood
    # ----------------------------------------------------------------------

    def spatial_log_probs(
        self,
        player_id: str,
        cells: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Per-shot ``log p_θ(c_i | x)`` (no timing term).

        Used directly by the NLL metric and by :py:meth:`log_prob`.

        Parameters
        ----------
        player_id : str
            Key into the base measure.
        cells : array of int, shape ``(K,)``
            Observed flat cell indices for the shots.

        Returns
        -------
        np.ndarray of shape ``(K,)``, dtype ``float64``.
        """
        cells_arr = np.asarray(cells, dtype=np.int64)
        K = int(cells_arr.shape[0])
        if K == 0:
            return np.array([], dtype=np.float64)

        log_q0 = self.base_measure.log_density(player_id).ravel()
        log_q0_t = (
            torch.as_tensor(log_q0, dtype=self.spatial_decoder.V.dtype).unsqueeze(0).expand(K, -1)
        )
        u = torch.zeros(K, self.spatial_decoder.rank, dtype=self.spatial_decoder.V.dtype)
        with torch.no_grad():
            log_probs_full = self.spatial_decoder.log_probs(log_q0_t, u).numpy()

        out: NDArray[np.float64] = log_probs_full[np.arange(K), cells_arr].astype(np.float64)
        return out

    def log_prob(
        self,
        player_id: str,
        cells: NDArray[np.int64],
        taus: NDArray[np.float64],
        context: Mapping[str, object] | None = None,
    ) -> float:
        """Return ``log p(S_n | x_n)`` for an observed player-game.

        Parameters
        ----------
        player_id : str
            Key into the base measure.
        cells : array of int, shape ``(K,)``
            Observed flat cell indices.
        taus : array of float, shape ``(K,)``
            Observed shot times.
        context : Mapping, optional
            Passed through to the timing model.
        """
        cells_arr = np.asarray(cells, dtype=np.int64)
        taus_arr = np.asarray(taus, dtype=np.float64)
        K = int(cells_arr.shape[0])
        if taus_arr.shape[0] != K:
            raise ValueError(f"length mismatch: cells has {K}, taus has {taus_arr.shape[0]}")
        ctx: Mapping[str, object] = context if context is not None else {}

        log_p_timing = self.timing_model.log_prob(K, taus_arr, ctx)
        if K == 0:
            return float(log_p_timing)

        log_p_spatial = float(self.spatial_log_probs(player_id, cells_arr).sum())
        return float(log_p_timing + log_p_spatial)

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _sample_cells(
        self,
        player_id: str,
        K: int,
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        log_q0 = self.base_measure.log_density(player_id).ravel()
        log_q0_t = (
            torch.as_tensor(log_q0, dtype=self.spatial_decoder.V.dtype).unsqueeze(0).expand(K, -1)
        )
        u = torch.zeros(K, self.spatial_decoder.rank, dtype=self.spatial_decoder.V.dtype)
        with torch.no_grad():
            probs = self.spatial_decoder.probs(log_q0_t, u).numpy()
        return _sample_categorical_batched(probs, rng)


def _sample_categorical_batched(
    probs: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Vectorized categorical sampling: ``probs`` ``(K, n_cells)`` → cells ``(K,)``."""
    cumsum = np.cumsum(probs, axis=-1)
    u: NDArray[np.float64] = np.asarray(rng.uniform(0.0, 1.0, size=probs.shape[0]))
    cells = (cumsum >= u[:, None]).argmax(axis=-1)
    return cast("NDArray[np.int64]", cells.astype(np.int64))
