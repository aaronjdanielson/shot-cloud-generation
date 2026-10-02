"""Evaluation suite: Wasserstein-only after the 2026-05-15 pivot cleanup.

Pre-pivot evaluation harness (``ablation``, ``diagnostics``,
``metrics``, ``retrieval``) moved to :mod:`shotcloud.legacy_pivot`
on 2026-05-15. The current production paper-train-evaluate path
uses only :func:`sliced_wasserstein_grid` from this package (via
``pretrain_snapshots`` and ``archetype_stability``).
"""

from shotcloud.evaluation.count_calibration import compute_count_calibration
from shotcloud.evaluation.energy_distance import energy_distance
from shotcloud.evaluation.support_masks import SupportSourceMasks, support_source_masks
from shotcloud.evaluation.timing_calibration import compute_timing_calibration
from shotcloud.evaluation.wasserstein import sliced_wasserstein, wasserstein_1d

__all__ = [
    "SupportSourceMasks",
    "compute_count_calibration",
    "compute_timing_calibration",
    "energy_distance",
    "sliced_wasserstein",
    "support_source_masks",
    "wasserstein_1d",
]
