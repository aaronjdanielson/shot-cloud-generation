"""Kernel density estimation on the court grid.

This package provides the grid-based KDE building blocks used by the
spatial model:

* :class:`AdaptiveKDE` -- per-player causal shot histories (cells,
  coordinates, context, dates) that form the own-player support of the
  collaborative KDE, plus a dense cell-to-cell kernel matrix.
* :class:`AnisotropicKernelEvaluator` -- learned per-shot anisotropic
  Gaussian kernels in the rim-radial / tangential frame.
* :class:`HierarchicalKDE` -- player, position-group, and league KDEs
  with sample-size shrinkage.
* :class:`HierarchicalKDEBase` -- a shrunk player density exposed as a
  fixed spatial base measure.
"""

from shotcloud.kde.adaptive import AdaptiveKDE
from shotcloud.kde.anisotropic import AnisotropicKernelEvaluator
from shotcloud.kde.hierarchical import HierarchicalKDE
from shotcloud.kde.hierarchical_base import HierarchicalKDEBase

__all__ = [
    "AdaptiveKDE",
    "AnisotropicKernelEvaluator",
    "HierarchicalKDE",
    "HierarchicalKDEBase",
]
