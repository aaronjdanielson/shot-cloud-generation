"""Timing/count models for the marked point process.

Deprecated; retained for :class:`~shotcloud.legacy_pivot.marked_process.ShotCloudProcess`.
Superseded by :class:`~shotcloud.models.NegBinCountHead` and
:class:`~shotcloud.models.TimingSoftmaxHead`.

Defines a structural protocol :class:`TimingModel` and a constant-rate
implementation, :class:`ConstantRateTimingModel`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson


class TimingModel(Protocol):
    """Structural protocol for shot timing/count models.

    A timing model samples ``(K, τ_{1:K})`` given player-game context, and
    evaluates ``log p(K, τ | context)`` for a given observation. The
    ordering / sortedness convention is left to the implementation;
    callers must use a single timing model consistently.
    """

    def sample(
        self,
        context: Mapping[str, object],
        rng: np.random.Generator,
    ) -> tuple[int, NDArray[np.float64]]: ...

    def log_prob(
        self,
        K: int,
        taus: NDArray[np.float64],
        context: Mapping[str, object],
    ) -> float: ...


@dataclass
class ConstantRateTimingModel:
    """Constant-rate timing model: ``K ~ Poisson(mean_shots)``, ``τ_i iid Uniform[0, T]``.

    Ignores ``context``; a minimal timing model for
    :class:`~shotcloud.legacy_pivot.marked_process.ShotCloudProcess`.

    Parameters
    ----------
    mean_shots : float, default 21.0
        Expected number of shots per game (≈ NBA per-player average for a
        starter).
    game_length : float, default 48.0
        Game length, in the same units as ``taus`` (default minutes).
    """

    mean_shots: float = 21.0
    game_length: float = 48.0

    def __post_init__(self) -> None:
        if self.mean_shots < 0:
            raise ValueError(f"mean_shots must be non-negative, got {self.mean_shots}")
        if self.game_length <= 0:
            raise ValueError(f"game_length must be positive, got {self.game_length}")

    def sample(
        self,
        context: Mapping[str, object],
        rng: np.random.Generator,
    ) -> tuple[int, NDArray[np.float64]]:
        K = int(rng.poisson(self.mean_shots))
        taus = np.sort(rng.uniform(0.0, self.game_length, size=K)).astype(np.float64)
        return K, taus

    def log_prob(
        self,
        K: int,
        taus: NDArray[np.float64],
        context: Mapping[str, object],
    ) -> float:
        if K < 0:
            raise ValueError(f"K must be non-negative, got {K}")
        taus_arr = np.asarray(taus, dtype=np.float64)
        if taus_arr.shape[0] != K:
            raise ValueError(f"length mismatch: K={K} but taus has length {taus_arr.shape[0]}")
        if K > 0 and (taus_arr.min() < 0 or taus_arr.max() > self.game_length):
            raise ValueError(
                f"taus must lie in [0, {self.game_length}]; got "
                f"[{taus_arr.min()}, {taus_arr.max()}]"
            )

        # log p(K) under Poisson(mean_shots).
        log_p_K = float(poisson.logpmf(K, self.mean_shots))
        # iid uniform: log ρ(τ) = -log(T) per shot, so K terms sum to -K log T.
        log_p_taus = -float(K) * float(np.log(self.game_length))
        return log_p_K + log_p_taus
