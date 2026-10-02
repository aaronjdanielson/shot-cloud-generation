"""Tier-1 evaluation metrics: zone distributions, KL divergence, NLL, KDE-gain.

Conventions
-----------
- Zone distributions return frequency vectors that sum to 1.
- ``zone_kl_divergence(gen, real)`` computes ``KL(gen || real)`` — generated
  in the numerator. Mirrors shot_flow's convention
  (``shot_flow/src/shot_flow/evaluation/metrics.py``) for cross-paper
  comparability.
- 5-zone (RA / Paint / Mid / Above-Break 3 / Corner 3) is the default for
  paper reporting; the 8-zone variant matches our :mod:`~shotcloud.data.zones`
  taxonomy and is exposed for finer analysis.
- All inputs are numpy arrays in the canonical coordinate convention
  (feet, basket at origin).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from shotcloud.data.zones import N_ZONES, zone_from_xy_vectorized

# Five-zone collapsed taxonomy (paper-readable). Order matters — this is the
# vector index for each zone.
ZONE_NAMES_5: tuple[str, ...] = (
    "RA",
    "Paint",
    "Mid-Range",
    "Above-Break 3",
    "Corner 3",
)
N_ZONES_5: int = len(ZONE_NAMES_5)


# ---------------------------------------------------------------------------
# Zone distributions
# ---------------------------------------------------------------------------


def zone_distribution_8(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> NDArray[np.float64]:
    """8-zone frequency vector matching :mod:`shotcloud.data.zones`.

    Returns ``(8,)`` floats summing to 1 (or 0 if the input is empty).
    Backcourt / out-of-bounds shots (zone -1) are excluded from the count
    and from the denominator.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size == 0:
        return np.zeros(N_ZONES, dtype=np.float64)
    zones = zone_from_xy_vectorized(x_arr, y_arr)
    valid = zones[zones >= 0]
    if valid.size == 0:
        return np.zeros(N_ZONES, dtype=np.float64)
    counts = np.bincount(valid, minlength=N_ZONES).astype(np.float64)
    return counts / valid.size


def zone_distribution_5(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> NDArray[np.float64]:
    """5-zone collapsed frequency vector (paper-readable).

    Mapping from the 8-zone taxonomy to the 5-zone collapsed one:

    - RA               ← zone 0
    - Paint            ← zone 1
    - Mid-Range        ← zone 2
    - Corner 3         ← zones 3, 4 (left + right corner)
    - Above-Break 3    ← zones 5, 6, 7 (wings + top of key)

    Returns ``(5,)`` floats summing to 1.
    """
    eight = zone_distribution_8(x, y)
    return np.array(
        [
            eight[0],  # RA
            eight[1],  # Paint
            eight[2],  # Mid-Range
            eight[5] + eight[6] + eight[7],  # Above-Break 3
            eight[3] + eight[4],  # Corner 3
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Zone KL divergence
# ---------------------------------------------------------------------------


def zone_kl_divergence(
    gen_freq: NDArray[np.floating],
    real_freq: NDArray[np.floating],
    eps: float = 1e-6,
) -> float:
    """``KL(gen || real)`` between two zone-frequency vectors.

    Both vectors are smoothed by ``eps`` and renormalized to avoid ``log 0``.
    A perfectly matched pair yields 0; larger values indicate the generated
    distribution is allocating mass to zones the real distribution
    underweights.

    Parameters
    ----------
    gen_freq : array, shape ``(K,)``
        Generated zone frequencies (must sum to 1, or close to it).
    real_freq : array, shape ``(K,)``
        Real zone frequencies. Must have the same length as ``gen_freq``.
    eps : float, default ``1e-6``
        Smoothing constant.
    """
    p = np.asarray(gen_freq, dtype=np.float64)
    q = np.asarray(real_freq, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: gen_freq {p.shape}, real_freq {q.shape}")
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log(p / q)).sum())


# ---------------------------------------------------------------------------
# NLL — model and base-measure
# ---------------------------------------------------------------------------


# Type alias for an observation: (player_id, cells_array).
ShotObservation = tuple[str, NDArray[np.int64]]


def nll_per_shot(
    process: object,  # ShotCloudProcess; loose typing avoids circular import
    observations: Sequence[ShotObservation],
) -> float:
    """Mean spatial NLL per shot under the full model ``p_θ``.

    NLL = ``-mean(log p_θ(c_i | x))`` across all observed shots.

    Parameters
    ----------
    process : ShotCloudProcess
        A fitted process. Must expose ``spatial_log_probs(player_id, cells)``.
    observations : sequence of (player_id, cells)
        Each entry is one player-game's observed cell sequence.
    """
    if not observations:
        return float("nan")
    log_probs = np.concatenate(
        [
            process.spatial_log_probs(pid, np.asarray(cells, dtype=np.int64))  # type: ignore[attr-defined]
            for pid, cells in observations
            if len(cells) > 0
        ]
        or [np.array([], dtype=np.float64)]
    )
    if log_probs.size == 0:
        return float("nan")
    return -float(log_probs.mean())


def base_measure_nll_per_shot(
    base_measure: object,  # KDEProduct; loose typing avoids circular import
    observations: Sequence[ShotObservation],
) -> float:
    """Mean spatial NLL per shot under the KDE-product base measure alone.

    NLL = ``-mean(log q_0(c_i))``. Used as the baseline for KDE-gain.
    """
    if not observations:
        return float("nan")
    log_probs_chunks: list[NDArray[np.float64]] = []
    for pid, cells in observations:
        cells_arr = np.asarray(cells, dtype=np.int64)
        if cells_arr.size == 0:
            continue
        log_q0 = base_measure.log_density(pid).ravel()  # type: ignore[attr-defined]
        log_probs_chunks.append(log_q0[cells_arr].astype(np.float64))
    if not log_probs_chunks:
        return float("nan")
    log_probs = np.concatenate(log_probs_chunks)
    return -float(log_probs.mean())


# ---------------------------------------------------------------------------
# KDE gain
# ---------------------------------------------------------------------------


def kde_gain(metric_kde: float, metric_model: float) -> float:
    """``Δ_KDE = metric(q_0) − metric(p_θ)``.

    For *distance* / *NLL* metrics (lower is better), positive ``kde_gain``
    means the model improves over the base KDE. For utility metrics where
    higher is better, the sign convention flips — handle externally.
    """
    return float(metric_kde - metric_model)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_metric(values: Sequence[float]) -> dict[str, float]:
    """Summary statistics over a list of per-player metric values."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "n": int(arr.size),
    }
