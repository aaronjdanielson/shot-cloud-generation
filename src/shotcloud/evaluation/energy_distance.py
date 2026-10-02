"""Energy distance between two 2D point clouds (Székely--Rizzo, 2013).

The U-statistic estimator of the 2-sample energy distance:

.. math::

    \\widehat{\\mathcal E}_U(X, Y)
    = \\frac{2}{mn}\\sum_{i,j}\\|x_i - y_j\\|
      - \\frac{1}{m(m-1)}\\sum_{i\\ne i'}\\|x_i - x_{i'}\\|
      - \\frac{1}{n(n-1)}\\sum_{j\\ne j'}\\|y_j - y_{j'}\\|

with the Euclidean norm in :math:`\\mathbb R^2`. The raw energy (no
square root) is reported, in feet, matching
:func:`~shotcloud.evaluation.wasserstein.sliced_wasserstein`.

The U-statistic is used rather than the V-statistic, which divides the
within-sample sums by :math:`m^2` instead of :math:`m(m-1)`. The
V-statistic underestimates the within-sample mean pairwise distance by
a factor of :math:`(m-1)/m` and so biases the energy distance upward by
:math:`O(1/m)`, which is material for clouds of 10--25 shots.

Properties of the U-statistic estimator:

* :math:`\\widehat{\\mathcal E}_U(X, Y) \\ge 0` *in expectation*, but can
  go slightly negative in finite samples — especially when ``x`` and
  ``y`` are drawn from the same distribution. ``clamp_nonneg`` (default
  ``False``) optionally clamps such cases to 0 for display purposes.
* When ``x`` and ``y`` are **the same array** (e.g., in self-bootstrap
  diagnostics), the estimator returns :math:`-2\\bar d_x / m` where
  :math:`\\bar d_x` is the within-sample U-statistic mean pairwise
  distance. This is :math:`O(1/m)` and decays with sample size.
* O(m² + n² + mn) compute — fine for the typical NBA player-game shot
  count (~10–25 shots per cloud × ~30 bootstraps).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _within_mean_pairwise_u(a: NDArray[np.floating]) -> float:
    """U-statistic mean pairwise distance within a single cloud.

    Returns ``Σ_{i≠j} ‖a_i − a_j‖ / (m(m-1))`` for ``m = a.shape[0]``.
    The diagonal is exactly 0, so summing the full matrix and dividing
    by ``m(m-1)`` is equivalent to summing off-diagonal entries.
    Returns 0.0 for m ≤ 1 (no pairs).
    """
    m = a.shape[0]
    if m <= 1:
        return 0.0
    diff = a[:, None, :] - a[None, :, :]
    d = np.linalg.norm(diff, axis=-1)  # diag exactly 0
    return float(d.sum() / (m * (m - 1)))


def _cross_mean_pairwise(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Mean pairwise distance across two clouds: ``Σ_{i,j} ‖a_i − b_j‖ / (mn)``."""
    diff = a[:, None, :] - b[None, :, :]
    return float(np.linalg.norm(diff, axis=-1).mean())


def energy_distance(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    clamp_nonneg: bool = False,
) -> float:
    """2D energy distance (U-statistic estimator) between ``x`` and ``y``.

    Parameters
    ----------
    x : array of shape ``(m, 2)``
        First point cloud, in feet.
    y : array of shape ``(n, 2)``
        Second point cloud, in feet.
    clamp_nonneg : bool, default False
        If True, clamp small-sample negative values to 0. The
        U-statistic can dip slightly below 0 when the two samples come
        from the same distribution or are the same array. Clamping
        introduces a one-sided bias, so it is intended for display only;
        the default returns the raw, unbiased estimate.

    Returns
    -------
    float
        Energy distance in feet (same units as inputs). ``nan`` if
        either input is empty, or if a within-sample term cannot be
        formed (m ≤ 1 or n ≤ 1) — those degenerate cases have no
        unbiased estimate.

    Raises
    ------
    ValueError
        If a non-empty input does not have shape ``(·, 2)``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"x must have shape (m, 2); got {x.shape}")
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"y must have shape (n, 2); got {y.shape}")
    m, n = x.shape[0], y.shape[0]
    if m < 2 or n < 2:
        # The within-sample U-statistic needs at least two points.
        return float("nan")

    cross = _cross_mean_pairwise(x, y)
    within_x = _within_mean_pairwise_u(x)
    within_y = _within_mean_pairwise_u(y)
    e = 2.0 * cross - within_x - within_y
    if clamp_nonneg and e < 0.0:
        return 0.0
    return e


__all__ = ["energy_distance"]
