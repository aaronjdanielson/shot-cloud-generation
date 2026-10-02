"""Static KDE base measures, temperatures, and player-identity encoders.

Deprecated; retained to reproduce the KDE-product and temperature ablations.
Superseded by the continuous-mixture AC-KDE spatial factor,
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`.

* :class:`KDEProduct` -- geometric product of player, position, and league
  KDE grids.
* :class:`LearnableTemperature` -- global or context-dependent temperature
  applied to a KDE log-density.
* :class:`LearnableKDEProductWeights` -- learnable non-negative exponents for
  :class:`KDEProduct`.
* :class:`PlayerEmbeddingEncoder`, :class:`PlayerPositionEncoder` --
  player-identity encoders for the low-rank tilt decoder.

The ablation runner in :mod:`shotcloud.legacy_pivot.eval_ablation` builds
these as baseline rows.
"""

from __future__ import annotations

from shotcloud.legacy.encoder import PlayerEmbeddingEncoder, PlayerPositionEncoder
from shotcloud.legacy.learnable_weights import (
    LearnableKDEProductWeights,
    _invert_softplus,
)
from shotcloud.legacy.product import KDEProduct
from shotcloud.legacy.temperature import LearnableTemperature

__all__ = [
    "KDEProduct",
    "LearnableKDEProductWeights",
    "LearnableTemperature",
    "PlayerEmbeddingEncoder",
    "PlayerPositionEncoder",
    "_invert_softplus",
]
