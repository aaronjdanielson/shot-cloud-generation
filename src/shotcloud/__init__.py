"""shotcloud — Adaptive Archetypal KDE for marked point processes on basketball shot clouds.

Public API for the consolidated AA-KDE model. See
:doc:`paper/shot_cloud.tex` for the model description and
:doc:`docs/code_inventory.md` for the inventory of essential vs.
legacy modules.

Two legacy buckets:

* :mod:`shotcloud.legacy` — pre-AA-KDE-pivot classes
  (``KDEProduct``, ``LearnableTemperature``,
  ``LearnableKDEProductWeights``, ``PlayerEmbeddingEncoder``,
  ``PlayerPositionEncoder``).
* :mod:`shotcloud.legacy_pivot` — pre-Gibbs-trainer classes
  (``DefensiveKDE``, ``LearnableDefensiveScale``,
  ``ConstantRateTimingModel``, ``TimingModel``, ``ShotCloud``,
  ``ShotCloudProcess``, ``ShotSequence``, ``ShotCellDataset``,
  ``train_decoder``, ``TrainHistory``, ``DecoderCheckpoint``).

Neither is surfaced at the top level. Import from the legacy
package directly when needed.
"""

from __future__ import annotations

from shotcloud.data import load_shots
from shotcloud.grids import CourtGrid
from shotcloud.kde import (
    AdaptiveKDE,
    AnisotropicKernelEvaluator,
    HierarchicalKDE,
    HierarchicalKDEBase,
)
from shotcloud.models import (
    ContextMLP,
    ContextResidualEncoder,
    NegBinCountHead,
    RelevanceMLP,
    RelevanceScore,
    TimingSoftmaxHead,
)
from shotcloud.training import (
    GibbsShotDataset,
    GibbsTrainHistory,
    OpponentVocab,
    PlayerVocab,
    train_gibbs,
)

__version__ = "0.1.0"

__all__ = [
    "AdaptiveKDE",
    "AnisotropicKernelEvaluator",
    "ContextMLP",
    "ContextResidualEncoder",
    "CourtGrid",
    "GibbsShotDataset",
    "GibbsTrainHistory",
    "HierarchicalKDE",
    "HierarchicalKDEBase",
    "NegBinCountHead",
    "OpponentVocab",
    "PlayerVocab",
    "RelevanceMLP",
    "RelevanceScore",
    "TimingSoftmaxHead",
    "__version__",
    "load_shots",
    "train_gibbs",
]
