"""Evaluation metrics and diagnostics for forecast shot clouds.

The package provides:

* distances between finite shot clouds --
  :func:`~shotcloud.evaluation.wasserstein.sliced_wasserstein` and
  :func:`~shotcloud.evaluation.energy_distance.energy_distance`;
* proper scoring rules on the predictive density surface
  (:mod:`shotcloud.evaluation.density_surfaces`);
* calibration summaries for the count and timing factors
  (:func:`compute_count_calibration`, :func:`compute_timing_calibration`);
* the own/pooled partition of a collaborative support set
  (:func:`support_source_masks`) and the own-history buckets used to
  stratify results (:mod:`shotcloud.evaluation.history_buckets`).

The grid-cell evaluation harness is retained in
:mod:`shotcloud.legacy_pivot`.
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
