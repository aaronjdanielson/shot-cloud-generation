"""KDE diagnostics: descriptive statistics and per-player NLL slices.

Deprecated; retained to reproduce the classical-KDE and temperature
diagnostics on the court grid.

The classical-KDE helpers characterize the *prior geometry* before any learnable
parameter is introduced. They run on a fitted
:class:`~shotcloud.kde.HierarchicalKDE` and produce per-player
statistics that reveal:

* **Where shrinkage matters** — entropy gap between raw and
  hierarchical densities, by training-shot bucket.
* **Where temperature could help** — concentration metrics (top-k mass,
  entropy). Flat-density players are candidates for ``τ > 1``.
* **Where the model has held-out NLL headroom** — per-player NLL on a
  held-out frame, grouped by shot count.

The trained-model helper adds per-player held-out NLL under the
low-rank tilt decoder ``softmax_c[τ · log q_p^hier(c) + u_p^⊤ V_c]`` so the gain over
the classical Hier-KDE baseline can be sliced by the same buckets.

The outputs feed ``scripts/legacy_pivot/diagnose_kde.py`` and
``scripts/legacy/phase15_temperature_slice.py``.

The classical-KDE metrics are pure-numpy and operate on density grids in
image layout ``(ny, nx)`` or flat ``(n_cells,)``. The trained-NLL helper
takes torch ``encoder``/``decoder`` modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor

from shotcloud.grids import CourtGrid
from shotcloud.kde import HierarchicalKDE
from shotcloud.legacy import PlayerEmbeddingEncoder, PlayerPositionEncoder
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder
from shotcloud.training.dataset import PlayerVocab

EncoderModule = PlayerEmbeddingEncoder | PlayerPositionEncoder

# ---------------------------------------------------------------------------
# Scalar density statistics
# ---------------------------------------------------------------------------


def entropy(q: NDArray[np.floating]) -> float:
    """Shannon entropy ``H(q) = -Σ q log q`` in nats.

    The grid is assumed to be a normalized probability mass function
    (sums to 1). Cells with ``q == 0`` are skipped — ε-floored grids
    will not have any, but the guard makes the helper safe to call on
    arbitrary nonnegative inputs.
    """
    flat = np.asarray(q, dtype=np.float64).ravel()
    if flat.size == 0:
        return float("nan")
    if (flat < 0).any():
        raise ValueError("entropy: q must be nonnegative")
    pos = flat[flat > 0]
    return float(-(pos * np.log(pos)).sum())


def top_k_mass(q: NDArray[np.floating], k: int) -> float:
    """Sum of the ``k`` largest cell probabilities.

    A simple sharpness statistic: ``top_k_mass(q, 1)`` is the mass of
    the modal cell; values close to 1 mean the density is concentrated.
    For a uniform grid of ``N`` cells, ``top_k_mass(q, k) == k / N``.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    flat = np.asarray(q, dtype=np.float64).ravel()
    if k >= flat.size:
        return float(flat.sum())
    # np.partition is O(n); we only need the k-th largest threshold.
    part = np.partition(flat, -k)
    return float(part[-k:].sum())


def held_out_nll(q: NDArray[np.floating], cells: NDArray[np.integer]) -> float:
    """Mean negative log-density at observed cells (in nats per shot).

    Parameters
    ----------
    q : array-like
        Density grid (image layout or flat). Must be strictly positive
        at every cell touched by ``cells`` — pass ε-floored grids.
    cells : array-like of int
        Flat cell indices ``[0, n_cells)``. Negative indices (backcourt
        from :meth:`CourtGrid.coord_to_cell`) must be filtered before
        calling.

    Returns ``nan`` if ``cells`` is empty.
    """
    cells_arr = np.asarray(cells, dtype=np.int64)
    if cells_arr.size == 0:
        return float("nan")
    if (cells_arr < 0).any():
        raise ValueError("held_out_nll: filter backcourt cells (idx < 0) before calling")
    flat = np.asarray(q, dtype=np.float64).ravel()
    if (flat[cells_arr] <= 0).any():
        raise ValueError("held_out_nll: q must be strictly positive at observed cells")
    return float(-np.log(flat[cells_arr]).mean())


# ---------------------------------------------------------------------------
# Player-level bucketing
# ---------------------------------------------------------------------------

# Shared by every diagnostic in this module so the classical and trained
# slices are directly comparable.
SHOT_COUNT_BUCKET_EDGES: tuple[float, ...] = (0.0, 50.0, 500.0, 2000.0)
SHOT_COUNT_BUCKET_LABELS: tuple[str, ...] = ("<50", "50-500", "500-2000", ">=2000")


def shot_count_bucket(n: float) -> str:
    """Label a player by training-shot count.

    The bucket boundaries are 50, 500, 2000 (open intervals).
    """
    if n < 0:
        raise ValueError(f"n must be nonnegative, got {n}")
    edges = SHOT_COUNT_BUCKET_EDGES
    labels = SHOT_COUNT_BUCKET_LABELS
    for i in range(1, len(edges)):
        if n < edges[i]:
            return labels[i - 1]
    return labels[-1]


# ---------------------------------------------------------------------------
# Per-player diagnostic frame
# ---------------------------------------------------------------------------


def build_per_player_diagnostics(
    kde: HierarchicalKDE,
    held_out_df: pd.DataFrame,
    grid: CourtGrid,
) -> pd.DataFrame:
    """Compute one row per fitted player with descriptive + held-out stats.

    Parameters
    ----------
    kde : HierarchicalKDE
        Already fit on the training split.
    held_out_df : DataFrame
        Held-out shots (typically the val or test split). Must contain
        ``x``, ``y``, ``player_id`` columns. Players in ``held_out_df``
        but not fit by ``kde`` contribute nothing (the KDE has no
        density for them); players fit by ``kde`` but absent from the
        held-out frame still get descriptive stats with NaN held-out
        NLL.
    grid : CourtGrid
        Used to map held-out ``(x, y)`` to flat cell indices. Backcourt
        shots (``coord_to_cell == -1``) are dropped.

    Returns
    -------
    DataFrame with columns:

    ============================  ====================================
    column                         meaning
    ============================  ====================================
    ``player_id``                  fitted player id (string-normalized)
    ``position``                   position group from the KDE fit
    ``n_train``                    effective training shot count
    ``n_held``                     held-out shots in court
    ``bucket``                    one of :data:`SHOT_COUNT_BUCKET_LABELS`
    ``entropy_raw``                ``H(q_p)``, nats
    ``entropy_hier``               ``H(q_p^hier)``, nats
    ``entropy_gap``                ``entropy_hier - entropy_raw``
    ``top1_mass_raw/hier``         modal-cell mass
    ``top5_mass_raw/hier``         top-5-cell mass
    ``top25_mass_raw/hier``        top-25-cell mass
    ``nll_raw_held``               mean ``-log q_p`` on held-out shots
    ``nll_hier_held``              mean ``-log q_p^hier`` on held-out shots
    ``nll_gain_hier_over_raw``     ``nll_raw_held - nll_hier_held``
                                   (positive = shrinkage helps held-out)
    ============================  ====================================
    """
    if not kde.is_fitted:
        raise ValueError("kde must be fit before building diagnostics")
    for col in ("x", "y", "player_id"):
        if col not in held_out_df.columns:
            raise KeyError(f"held_out_df missing required column {col!r}")

    # Group held-out shots by stringified player id (matches kde keys).
    held = held_out_df.copy()
    held["_pid_key"] = held["player_id"].astype(str)
    held_groups = {pid: sub for pid, sub in held.groupby("_pid_key")}

    rows: list[dict[str, object]] = []
    for pid in sorted(kde.player_density_grid):
        q_raw = kde.player_density(pid, hierarchical=False)
        q_hier = kde.player_density(pid, hierarchical=True)
        n_train = float(kde.player_n[pid])
        position = kde.player_position[pid]

        sub = held_groups.get(pid)
        if sub is None or len(sub) == 0:
            n_held = 0
            nll_raw = float("nan")
            nll_hier = float("nan")
        else:
            cells = grid.coord_to_cell(
                sub["x"].to_numpy(dtype=np.float64),
                sub["y"].to_numpy(dtype=np.float64),
            )
            cells = cells[cells >= 0].astype(np.int64)
            n_held = int(cells.size)
            if n_held == 0:
                nll_raw = float("nan")
                nll_hier = float("nan")
            else:
                nll_raw = held_out_nll(q_raw, cells)
                nll_hier = held_out_nll(q_hier, cells)

        rows.append(
            {
                "player_id": pid,
                "position": position,
                "n_train": n_train,
                "n_held": n_held,
                "bucket": shot_count_bucket(n_train),
                "entropy_raw": entropy(q_raw),
                "entropy_hier": entropy(q_hier),
                "entropy_gap": entropy(q_hier) - entropy(q_raw),
                "top1_mass_raw": top_k_mass(q_raw, 1),
                "top5_mass_raw": top_k_mass(q_raw, 5),
                "top25_mass_raw": top_k_mass(q_raw, 25),
                "top1_mass_hier": top_k_mass(q_hier, 1),
                "top5_mass_hier": top_k_mass(q_hier, 5),
                "top25_mass_hier": top_k_mass(q_hier, 25),
                "nll_raw_held": nll_raw,
                "nll_hier_held": nll_hier,
                "nll_gain_hier_over_raw": (nll_raw - nll_hier)
                if not (np.isnan(nll_raw) or np.isnan(nll_hier))
                else float("nan"),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bucket-level summary
# ---------------------------------------------------------------------------


def _weighted_mean(values: NDArray[np.floating], weights: NDArray[np.floating]) -> float:
    """Mean of ``values`` weighted by ``weights``, ignoring NaN values
    and zero / negative weights. Returns NaN if no positive-weight
    finite values remain.
    """
    vals = np.asarray(values, dtype=np.float64)
    wts = np.asarray(weights, dtype=np.float64)
    mask = ~np.isnan(vals) & (wts > 0)
    if not mask.any():
        return float("nan")
    total = wts[mask].sum()
    if total == 0:
        return float("nan")
    return float((vals[mask] * wts[mask]).sum() / total)


def summarize_by_bucket(per_player: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-player frame to one row per shot-count bucket.

    Returns a frame indexed by bucket label (in canonical order) with
    means of the numeric columns and a ``n_players`` count. Held-out
    NLL means weight by ``n_held`` (so a single sparse player doesn't
    dominate the bucket mean).
    """
    if "bucket" not in per_player.columns:
        raise KeyError("per_player must have a 'bucket' column")

    rows: list[dict[str, object]] = []
    for label in SHOT_COUNT_BUCKET_LABELS:
        sub = per_player[per_player["bucket"] == label]
        if len(sub) == 0:
            rows.append({"bucket": label, "n_players": 0})
            continue
        weights = sub["n_held"].to_numpy(dtype=np.float64)
        row: dict[str, object] = {
            "bucket": label,
            "n_players": len(sub),
            "n_held_total": int(weights.sum()),
            "mean_n_train": float(sub["n_train"].mean()),
            "mean_entropy_raw": float(sub["entropy_raw"].mean()),
            "mean_entropy_hier": float(sub["entropy_hier"].mean()),
            "mean_top1_mass_hier": float(sub["top1_mass_hier"].mean()),
            "mean_top5_mass_hier": float(sub["top5_mass_hier"].mean()),
            "weighted_nll_raw_held": _weighted_mean(
                sub["nll_raw_held"].to_numpy(dtype=np.float64), weights
            ),
            "weighted_nll_hier_held": _weighted_mean(
                sub["nll_hier_held"].to_numpy(dtype=np.float64), weights
            ),
            "weighted_nll_gain_hier_over_raw": _weighted_mean(
                sub["nll_gain_hier_over_raw"].to_numpy(dtype=np.float64), weights
            ),
        }
        # Trained-model columns are present only if attached.
        if "nll_trained_held" in sub.columns:
            row["weighted_nll_trained_held"] = _weighted_mean(
                sub["nll_trained_held"].to_numpy(dtype=np.float64), weights
            )
        if "nll_gain_trained_over_hier" in sub.columns:
            row["weighted_nll_gain_trained_over_hier"] = _weighted_mean(
                sub["nll_gain_trained_over_hier"].to_numpy(dtype=np.float64), weights
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index("bucket").reindex(SHOT_COUNT_BUCKET_LABELS)


# ---------------------------------------------------------------------------
# Per-player trained NLL slice
# ---------------------------------------------------------------------------


def per_player_trained_nll(
    kde: HierarchicalKDE,
    held_out_df: pd.DataFrame,
    grid: CourtGrid,
    *,
    encoder: EncoderModule,
    decoder: LowRankTiltDecoder,
    vocab: PlayerVocab,
    tau: float = 1.0,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Mean held-out NLL **under the trained model** for each player.

    The trained model is

    .. math::

        p_\\theta(c \\mid p)
        = \\mathrm{softmax}_c\\!\\left[
            \\tau \\cdot \\log \\hat q_p^{\\mathrm{hier}}(c)
            + u_p^\\top v_c
        \\right].

    Parameters
    ----------
    kde : HierarchicalKDE
        Already fit on the training split (used for ``log q_p^hier``).
    held_out_df : DataFrame
        Held-out shots (val or test). Must contain ``x``, ``y``,
        ``player_id``.
    grid : CourtGrid
    encoder, decoder : trained modules
    vocab : PlayerVocab
        Same vocab the trained model was built against.
    tau : float, default 1.0
        Temperature applied to ``log q_p^hier``. Pass the value learned
        by :class:`~shotcloud.legacy.temperature.LearnableTemperature`.
    device : optional, default ``"cpu"``

    Returns
    -------
    dict mapping ``player_id`` (string-normalized) to mean held-out NLL.
    Players present in ``vocab`` but absent from ``held_out_df`` map to
    ``nan``. Players absent from ``vocab`` are skipped entirely.
    """
    if not kde.is_fitted:
        raise ValueError("kde must be fit before scoring")
    for col in ("x", "y", "player_id"):
        if col not in held_out_df.columns:
            raise KeyError(f"held_out_df missing required column {col!r}")
    if tau <= 0:
        raise ValueError(f"tau must be strictly positive, got {tau}")

    dev = torch.device(device)
    encoder = encoder.to(dev)
    decoder = decoder.to(dev)
    encoder.eval()
    decoder.eval()
    V = decoder.V.detach().to(dev)  # (n_cells, rank)

    held = held_out_df.copy()
    held["_pid_key"] = held["player_id"].astype(str)
    held_groups = {pid: sub for pid, sub in held.groupby("_pid_key")}

    out: dict[str, float] = {}
    for pid in vocab.ids:
        # log q_hier on the grid for this player.
        q_hier = kde.player_density(pid, hierarchical=True)
        log_q_hier = torch.from_numpy(np.log(q_hier).ravel()).to(dev).float()

        idx = vocab.to_idx(pid)
        with torch.no_grad():
            u = encoder(torch.tensor([idx], dtype=torch.long, device=dev)).squeeze(0)  # (rank,)
            logits = tau * log_q_hier + V @ u  # (n_cells,)
            log_p: Tensor = logits - torch.logsumexp(logits, dim=0)

        sub = held_groups.get(pid)
        if sub is None or len(sub) == 0:
            out[pid] = float("nan")
            continue
        cells = grid.coord_to_cell(
            sub["x"].to_numpy(dtype=np.float64),
            sub["y"].to_numpy(dtype=np.float64),
        )
        cells = cells[cells >= 0].astype(np.int64)
        if cells.size == 0:
            out[pid] = float("nan")
            continue
        nll = -log_p[torch.from_numpy(cells).to(dev)].mean().item()
        out[pid] = float(nll)

    return out


def attach_trained_nll(
    per_player: pd.DataFrame,
    trained_nll: dict[str, float],
) -> pd.DataFrame:
    """Add ``nll_trained_held`` + ``nll_gain_trained_over_hier`` columns.

    Returns a new DataFrame; the input is not mutated. Players in
    ``per_player`` but missing from ``trained_nll`` get NaN. The
    trained-over-hier gain is positive when the trained model improves
    on the classical Hier-KDE baseline.
    """
    out = per_player.copy()
    out["nll_trained_held"] = out["player_id"].map(trained_nll)
    out["nll_gain_trained_over_hier"] = out["nll_hier_held"] - out["nll_trained_held"]
    return out
