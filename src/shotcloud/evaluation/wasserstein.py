"""Wasserstein distances for shot-cloud comparison.

Provides a 1-D Wasserstein-1 distance between empirical samples, a
sliced Wasserstein-1 estimator for 2-D point clouds (averaged over random
1-D projections), and a sliced variant for probability vectors on a
shared grid of cells. The point-cloud estimator follows the convention of
the companion ``shot_flow`` project so that results are comparable;
quantile matching handles unequal sample sizes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def wasserstein_1d(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """1D Wasserstein-1 distance between two empirical samples.

    For equal-size samples, this is ``mean(|sort(a) - sort(b)|)``. For
    unequal sizes, both samples are interpolated to ``min(len(a), len(b))``
    quantile points before sorting.

    Returns ``nan`` if either input is empty.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.size == 0 or b_arr.size == 0:
        return float("nan")
    if a_arr.size != b_arr.size:
        n = min(a_arr.size, b_arr.size)
        qs = np.linspace(0.0, 1.0, n)
        a_arr = np.quantile(a_arr, qs)
        b_arr = np.quantile(b_arr, qs)
    return float(np.mean(np.abs(np.sort(a_arr) - np.sort(b_arr))))


def sliced_wasserstein(
    p_shots: NDArray[np.floating],
    q_shots: NDArray[np.floating],
    *,
    n_projections: int = 200,
    seed: int = 0,
) -> float:
    """Approximate 2D Wasserstein-1 between two shot-location point clouds.

    Draws ``n_projections`` random unit vectors in :math:`\\mathbb R^2`,
    projects both clouds onto each, and averages :func:`wasserstein_1d`
    over the projections.

    Parameters
    ----------
    p_shots : array of shape ``(N, 2)``
        First point cloud (e.g., real shots).
    q_shots : array of shape ``(M, 2)``
        Second point cloud (e.g., generated shots).
    n_projections : int, default 200
        Number of random 1D slices to average over.
    seed : int, default 0
        RNG seed for reproducibility.

    Returns
    -------
    float
        Approximated W1 in feet (the units of ``p_shots`` / ``q_shots``).
        ``nan`` when either input is empty.

    Raises
    ------
    ValueError
        If a non-empty input does not have shape ``(·, 2)``.
    """
    p = np.asarray(p_shots, dtype=np.float64)
    q = np.asarray(q_shots, dtype=np.float64)
    if p.size == 0 or q.size == 0:
        return float("nan")
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"p_shots must have shape (N, 2); got {p.shape}")
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError(f"q_shots must have shape (M, 2); got {q.shape}")

    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_projections, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    total = 0.0
    for d in directions:
        total += wasserstein_1d(p @ d, q @ d)
    return total / n_projections


def sliced_wasserstein_grid(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    cell_centers: NDArray[np.floating],
    *,
    n_projections: int = 50,
    seed: int = 0,
) -> float:
    """Sliced Wasserstein-1 between two simplex-valued grid distributions.

    Both inputs are probability distributions over a shared discrete
    support of cells with known 2D centers, such as grid densities. For
    each random unit direction :math:`d`, project
    cell centers onto :math:`d` to get a 1D support, then compute the
    weighted 1D Earth Mover's distance between :math:`(s, a)` and
    :math:`(s, b)` along the projected support. Average over
    projections.

    Parameters
    ----------
    a, b : array of shape ``(C,)``
        Probability mass on each of the ``C`` cells. Need not sum to
        exactly 1 — both are renormalized internally so the function
        is robust to small numerical drift in the simplex constraint.
    cell_centers : array of shape ``(C, 2)``
        2D Cartesian center of each cell, in feet.
    n_projections : int, default 50
        Number of random 1D slices to average. Lower than the point-cloud
        default because each projection costs O(C log C) and the
        function is typically called for every pair in a set of grid
        densities.
    seed : int, default 0
        RNG seed.

    Returns
    -------
    float
        Approximate sliced-W1 distance in feet; non-negative, and zero
        when ``a == b`` after renormalization. ``nan`` if either input
        has non-positive total mass.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` differ in shape, are not 1-D, or do not match
        ``cell_centers``.
    """
    from scipy.stats import wasserstein_distance

    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    centers = np.asarray(cell_centers, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"a, b shapes must match; got {a_arr.shape} vs {b_arr.shape}")
    if a_arr.ndim != 1:
        raise ValueError(f"a must be 1-D over cells; got shape {a_arr.shape}")
    if centers.ndim != 2 or centers.shape != (a_arr.size, 2):
        raise ValueError(f"cell_centers must have shape ({a_arr.size}, 2); got {centers.shape}")

    a_sum = float(a_arr.sum())
    b_sum = float(b_arr.sum())
    if a_sum <= 0.0 or b_sum <= 0.0:
        return float("nan")
    a_arr = a_arr / a_sum
    b_arr = b_arr / b_sum

    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_projections, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    total = 0.0
    for d in directions:
        proj = centers @ d  # (C,)
        total += float(wasserstein_distance(proj, proj, u_weights=a_arr, v_weights=b_arr))
    return total / n_projections
