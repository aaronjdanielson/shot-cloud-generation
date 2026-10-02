"""Shared KDE primitives used by ``HierarchicalKDE`` and ``AdaptiveKDE``.

* :func:`build_recency_weights` -- exponential-decay weights from shot
  dates to a reference date, given a half-life, so that a fitted density
  reflects recent behavior.
* :func:`build_kernel_matrix` -- the grid Gaussian kernel materialized as
  an explicit ``(n_cells, n_cells)`` matrix.
* :func:`fit_density_grid` -- histogram plus Gaussian blur on a
  :class:`~shotcloud.grids.CourtGrid`, ε-floored and L1-normalized so the
  result is a strictly positive probability mass function.

The histogram + Gaussian-blur estimator matches the approximation used by
``shot_flow``. For bandwidths that span multiple grid cells (the typical
regime) it is functionally equivalent to a per-shot Gaussian sum but runs
in ``O(N + n_cells · kernel_size)`` instead of ``O(N · n_cells)``.

The helpers are stateless so that any KDE class can call them without
coupling to its internals.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter

from shotcloud.grids import CourtGrid

LN_2 = float(np.log(2.0))


def build_recency_weights(
    date: ArrayLike | None,
    reference_date: np.datetime64 | None,
    n: int,
    *,
    half_life_days: float | None,
) -> NDArray[np.float64]:
    """Exponential-decay weights ``exp(-λ · Δdays)`` per shot.

    Returns ``np.ones(n)`` when ``date`` or ``half_life_days`` is
    ``None`` (recency disabled). Otherwise:

    .. math::

        w_j = \\exp\\!\\left(-\\frac{\\ln 2}{H} \\cdot \\max(\\Delta t_j, 0)\\right),

    where ``H`` is ``half_life_days`` and ``Δt_j`` is days from shot to
    ``reference_date`` (defaults to the latest date in the array).
    Shots dated *after* the reference get weight 1 (no upweighting).
    """
    if date is None or half_life_days is None:
        return np.ones(n, dtype=np.float64)

    date_arr = np.asarray(date, dtype="datetime64[D]")
    if date_arr.shape[0] != n:
        raise ValueError(f"length mismatch: date has {date_arr.shape[0]} elements, expected {n}")
    if reference_date is None:
        reference_date = date_arr.max()

    ref = np.asarray(reference_date, dtype="datetime64[D]")
    delta_days = (ref - date_arr) / np.timedelta64(1, "D")
    delta = delta_days.astype(np.float64)
    lam = LN_2 / float(half_life_days)
    return cast("NDArray[np.float64]", np.exp(-lam * np.maximum(delta, 0.0)))


def build_kernel_matrix(
    grid: CourtGrid, bandwidth: float, epsilon: float = 1e-12
) -> NDArray[np.float64]:
    """Per-cell ε-floored Gaussian kernel matrix ``M[c, k] = K_h(cell c, cell k)``.

    Lets :class:`~shotcloud.kde.AdaptiveKDE` express a relevance-weighted
    historical-shot density as a single matrix product,

    .. math::

        \\hat q_\\phi(\\cdot \\mid p, x) = M \\cdot \\pi_\\phi(x; Z[p]),

    where :math:`\\pi_\\phi(x; Z[p])` is the vector of relevance weights
    placed on the player's history cells. ``M`` is the kernel that
    :func:`fit_density_grid` applies, materialized as an explicit table so
    densities can be formed without re-running
    :func:`scipy.ndimage.gaussian_filter`.

    Parameters
    ----------
    grid : CourtGrid
        Spatial discretization.
    bandwidth : float
        Gaussian bandwidth in feet. ``0`` gives the identity kernel.
    epsilon : float, default 1e-12
        Floor relative to the grid: entries are clipped below at
        ``epsilon / n_cells`` before each column is normalized.

    Returns
    -------
    M : NDArray[float64], shape ``(n_cells, n_cells)``
        Column ``k`` is the blurred, floored, L1-normalized density of a
        unit point mass at cell ``k``; every column sums to 1.

    Notes
    -----
    All ``n_cells`` point-source images are blurred in one call (an
    identity matrix reshaped to ``(n_cells, ny, nx)``). The matrix holds
    ``n_cells²`` float64 entries, about 98 MB at ``n_cells = 3584``.
    """
    g = grid
    n_cells = g.n_cells
    # Identity over flat cells, reshaped to (n_cells, ny, nx) — each
    # "image" is a single 1.0 at one cell.
    eye = np.eye(n_cells, dtype=np.float64).reshape(n_cells, g.ny, g.nx)

    if bandwidth > 0:
        sigma = (bandwidth / g.dy, bandwidth / g.dx)
        # Apply 2D Gaussian blur over the (ny, nx) axes only.
        blurred = gaussian_filter(eye, sigma=(0.0, *sigma), mode="constant", cval=0.0)
    else:
        blurred = eye

    # Reshape back to (n_cells_source, n_cells_target) and ε-floor / normalize per source.
    cols = blurred.reshape(n_cells, n_cells)
    floor = epsilon / n_cells
    cols = np.maximum(cols, floor)
    # Each *column* of M (M[:, k]) is the density resulting from a point at k.
    # In our identity reshape, the source index k is the *row* of `cols`. So
    # M = cols.T. Normalize each column (= source) to sum to 1.
    cols = cols / cols.sum(axis=1, keepdims=True)
    return cast("NDArray[np.float64]", cols.T.astype(np.float64))


def fit_density_grid(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    grid: CourtGrid,
    bandwidth: float,
    epsilon: float,
) -> tuple[NDArray[np.float64], float]:
    """Fit a grid density by histogram plus Gaussian blur.

    The blur uses per-axis widths ``bandwidth / dy`` and ``bandwidth / dx``
    in cell units, so the kernel is isotropic in feet even when grid cells
    are not square.

    Parameters
    ----------
    x, y : float arrays of length ``N``
        Shot coordinates in feet, basket at origin.
    weights : float array of length ``N``
        Per-shot weights (e.g., recency). Pass ``np.ones(N)`` for
        unweighted.
    grid : CourtGrid
        Spatial discretization.
    bandwidth : float
        Gaussian bandwidth in feet. Set ``0`` to disable blurring.
    epsilon : float
        Density floor (relative to grid). Guarantees ``q > 0`` for
        log-domain operations downstream.

    Returns
    -------
    density : NDArray[float64], shape ``(ny, nx)``
        Image-layout grid, normalized so cells sum to 1, with floor
        ``epsilon / n_cells`` for strict positivity.
    effective_n : float
        ``weights.sum()`` — the (recency-weighted) sample size. Used by
        the shrinkage formula in :class:`HierarchicalKDE`.
    """
    g = grid
    effective_n = float(weights.sum())

    # np.histogram2d returns shape (nx, ny); .T gives image layout (ny, nx).
    hist, _, _ = np.histogram2d(
        x, y, bins=[g.xedges, g.yedges], range=[g.xlim, g.ylim], weights=weights
    )
    hist = hist.T

    if bandwidth > 0:
        sigma = (bandwidth / g.dy, bandwidth / g.dx)
        blurred = gaussian_filter(hist, sigma=sigma, mode="constant", cval=0.0)
    else:
        blurred = hist

    floor = epsilon / g.n_cells
    density = np.maximum(blurred, floor)
    density = density / density.sum()
    return cast("NDArray[np.float64]", density.astype(np.float64)), effective_n
