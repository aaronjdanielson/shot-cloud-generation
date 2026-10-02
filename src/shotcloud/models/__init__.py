"""Learned components of the shot-cloud marked point-process model.

The package namespace exports the context MLP ``f_ctx``
(:class:`ContextMLP`), the residual-tilt encoder
(:class:`ContextResidualEncoder`), the negative-binomial count head
(:class:`NegBinCountHead`), the 48-bin timing head
(:class:`TimingSoftmaxHead`), the per-shot relevance scorers
(:class:`RelevanceScore`, :class:`RelevanceMLP`), and the conditional
mixture-density-network baseline (:class:`ConditionalMDN`).

The AC-KDE spatial factor is imported from its submodules:
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`
evaluates the kernel mixture at the observed shot coordinate, on top of
a support backend
(:class:`~shotcloud.models.retrieval_collaborative_kde.RetrievalCollaborativeKDE`
or :class:`~shotcloud.models.collaborative_kde.CollaborativeKDE`) and the
optional :class:`~shotcloud.models.pooling_gate.PoolingGate`.

Deprecated components are retained for reproducibility in
:mod:`shotcloud.legacy` (KDE product, learnable temperature and weights,
player encoders) and :mod:`shotcloud.legacy_pivot` (grid-cell Gibbs
decoder, low-rank tilt decoder, grid adaptive priors, Wasserstein
archetypes).
"""

from shotcloud.models.conditional_mdn import ConditionalMDN
from shotcloud.models.context_mlp import ContextMLP
from shotcloud.models.context_residual import ContextResidualEncoder
from shotcloud.models.count_head import NegBinCountHead
from shotcloud.models.relevance import RelevanceMLP, RelevanceScore
from shotcloud.models.timing_head import TimingSoftmaxHead

__all__ = [
    "ConditionalMDN",
    "ContextMLP",
    "ContextResidualEncoder",
    "NegBinCountHead",
    "RelevanceMLP",
    "RelevanceScore",
    "TimingSoftmaxHead",
]
