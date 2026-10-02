"""shotcloud: Adaptive Collaborative KDE for forecasting NBA player-game shot clouds.

Each player-game is modeled as a marked point process factored into a shot
count (:class:`NegBinCountHead`), shot timing (:class:`TimingSoftmaxHead`),
and spatial location. The spatial factor is the Adaptive Collaborative KDE
(AC-KDE): a continuous kernel mixture over causal support shots, the
player's own past shots plus pooled shots from analogue players,
implemented by
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`.
The factors are trained by :func:`train_gibbs` on a
:class:`GibbsShotDataset`, optionally with a pretrained, frozen count
head.

The top-level namespace re-exports the data loader, court grid, KDE
engines, learned heads, and trainer. Deprecated components are kept for
reproducibility and are not re-exported here; import them from
:mod:`shotcloud.legacy` (static KDE product, learnable temperature and
weights, player-identity encoders) or :mod:`shotcloud.legacy_pivot`
(grid-cell Gibbs decoder, Wasserstein archetypes, and their training and
evaluation harness).
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
