"""Timing-head calibration + baseline-comparison diagnostics (paper §5.2 timing).

Phase 3 of the 2026-06-07 audit. The timing factor is a 48-bin
categorical over per-shot game minute. We evaluate the trained
:class:`TimingSoftmaxHead` against three histogram baselines so the
paper can either validate timing as an empirical factor or demote it
with evidence:

* **global** — single 48-bin distribution fit to all training shots.
* **starter / bench** — two histograms, indexed by the starter slot of
  ``x_n_raw``.
* **minutes-conditioned** — four histograms, indexed by quartile of
  the per-shot minutes-zscored slot of ``x_n_raw``.

All three are built on the training set and evaluated on the held-out
val set (no leakage). Calibration metrics: per-shot NLL (the
optimization target), 48-bin L1 between the predicted aggregate and
the observed val histogram, quarter-aggregated 4-bin L1, and
predictive p10/p50/p90.

The trained head's NLL is the value to beat. The 48-bin and
quarter-aggregated L1 metrics characterize *calibration* (whether the
predicted distribution matches the observed marginal) independently
of conditional sharpness.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from shotcloud.models.context_mlp import ContextMLP
from shotcloud.models.timing_head import TimingSoftmaxHead

#: Per-context slot indices in the raw 27-dim ``x_n_raw`` vector — see
#: ``CLAUDE.md`` for the canonical layout.
_STARTER_IDX: int = 6
_MINUTES_Z_IDX: int = 7

#: Default number of minutes-conditioned quantile buckets.
_N_MINUTES_BUCKETS: int = 4

#: Default number of timing bins (paper §5: one per game minute).
_N_TIMING_BINS: int = 48


def _empirical_histogram(tau_bin: Tensor, n_bins: int) -> Tensor:
    """Build a smoothed empirical histogram from a 1-D bin index
    tensor. Returns a probability vector of shape ``(n_bins,)``.
    Adds a small additive count (``+1``) to every bin to keep
    log-probabilities finite under finite-sample queries; this is
    standard Laplace smoothing.
    """
    if tau_bin.dim() != 1:
        raise ValueError(f"tau_bin must be (N,); got {tuple(tau_bin.shape)}")
    counts = torch.bincount(tau_bin.to(torch.int64), minlength=n_bins).to(torch.float64)
    counts = counts[:n_bins]
    smoothed = counts + 1.0
    return (smoothed / smoothed.sum()).to(torch.float32)


def _starter_baseline_split(
    train_x_n_raw: Tensor, train_tau_bin: Tensor, n_bins: int
) -> tuple[Tensor, Tensor]:
    """Build (starter_hist, bench_hist) from training shots."""
    starter_mask = train_x_n_raw[:, _STARTER_IDX] >= 0.5
    if starter_mask.any():
        starter_hist = _empirical_histogram(train_tau_bin[starter_mask], n_bins)
    else:
        starter_hist = torch.full((n_bins,), 1.0 / n_bins, dtype=torch.float32)
    if (~starter_mask).any():
        bench_hist = _empirical_histogram(train_tau_bin[~starter_mask], n_bins)
    else:
        bench_hist = torch.full((n_bins,), 1.0 / n_bins, dtype=torch.float32)
    return starter_hist, bench_hist


def _minutes_baseline_split(
    train_x_n_raw: Tensor,
    train_tau_bin: Tensor,
    n_bins: int,
    n_buckets: int = _N_MINUTES_BUCKETS,
) -> tuple[Tensor, Tensor]:
    """Build (bucket_edges, bucket_hists) from training shots.

    Quantile-edges over the ``minutes_zscored`` slot of ``x_n_raw``.
    ``bucket_edges`` has shape ``(n_buckets+1,)`` and ``bucket_hists``
    has shape ``(n_buckets, n_bins)``.
    """
    minutes = train_x_n_raw[:, _MINUTES_Z_IDX].to(torch.float64)
    quantiles = torch.linspace(0.0, 1.0, n_buckets + 1, dtype=minutes.dtype, device=minutes.device)
    edges = torch.quantile(minutes, quantiles)
    # Guarantee monotone strictly-increasing edges (ties → tiny perturbation).
    for i in range(1, edges.shape[0]):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    hists = torch.zeros((n_buckets, n_bins), dtype=torch.float32)
    bucket_idx = torch.bucketize(minutes, edges[1:-1]).clamp_(0, n_buckets - 1)
    for b in range(n_buckets):
        mask = bucket_idx == b
        if mask.any():
            hists[b] = _empirical_histogram(train_tau_bin[mask], n_bins)
        else:
            hists[b] = torch.full((n_bins,), 1.0 / n_bins, dtype=torch.float32)
    return edges, hists


def _baseline_nll_per_shot(
    val_x_n_raw: Tensor,
    val_tau_bin: Tensor,
    *,
    global_hist: Tensor,
    starter_hist: Tensor,
    bench_hist: Tensor,
    minutes_edges: Tensor,
    minutes_hists: Tensor,
) -> dict[str, float]:
    """Per-shot NLL of each baseline on the val set."""
    # Global.
    log_p_global = torch.log(global_hist[val_tau_bin])
    # Starter/bench.
    starter_mask = val_x_n_raw[:, _STARTER_IDX] >= 0.5
    p_role = torch.where(
        starter_mask.unsqueeze(-1),
        starter_hist.unsqueeze(0).expand(val_tau_bin.shape[0], -1),
        bench_hist.unsqueeze(0).expand(val_tau_bin.shape[0], -1),
    )
    log_p_role = torch.log(p_role.gather(1, val_tau_bin.unsqueeze(-1)).squeeze(-1))
    # Minutes-conditioned.
    minutes = val_x_n_raw[:, _MINUTES_Z_IDX].to(torch.float64)
    n_buckets = minutes_hists.shape[0]
    bucket_idx = torch.bucketize(minutes, minutes_edges[1:-1]).clamp_(0, n_buckets - 1)
    p_min = minutes_hists[bucket_idx]
    log_p_min = torch.log(p_min.gather(1, val_tau_bin.unsqueeze(-1)).squeeze(-1))
    return {
        "global": float((-log_p_global).mean().item()),
        "starter_bench": float((-log_p_role).mean().item()),
        "minutes_conditioned": float((-log_p_min).mean().item()),
    }


def _bin_l1(p_pred: Tensor, p_obs: Tensor) -> float:
    """L1 distance between two probability vectors."""
    return float((p_pred - p_obs).abs().sum().item())


def _quarter_aggregate(p: Tensor) -> Tensor:
    """48-bin distribution → 4-bin quarter distribution (12 bins / Q)."""
    if p.shape[-1] != 48:
        raise ValueError(f"expected 48-bin distribution; got {tuple(p.shape)}")
    return p.view(*p.shape[:-1], 4, 12).sum(dim=-1)


def compute_timing_calibration(
    timing_head: TimingSoftmaxHead,
    context_mlp: ContextMLP,
    train_x_n_raw: Tensor,
    train_tau_bin: Tensor,
    val_x_n_raw: Tensor,
    val_tau_bin: Tensor,
    *,
    n_bins: int = _N_TIMING_BINS,
    n_minutes_buckets: int = _N_MINUTES_BUCKETS,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Calibration + baseline-comparison summary for ``timing_head``.

    Returns a JSON-serializable dict with timing NLL of the trained
    head + three baselines, 48-bin L1 between predicted aggregate
    distribution and observed val histogram, quarter-aggregated 4-bin
    L1, and predictive p10/p50/p90 (averaged across val shots).

    The verdict the caller wants from this is:

    * If trained-head NLL beats all three baselines by a meaningful
      margin AND its predicted aggregate matches the observed val
      histogram within tight 48-bin L1 → timing validates as an
      empirical marked-PP factor.
    * If trained-head NLL is at or near the minutes-conditioned
      baseline → timing is a scaffold that adds little beyond a
      simple lookup; the paper should demote it honestly.
    """
    if train_x_n_raw.dim() != 2 or train_x_n_raw.shape[1] < max(_STARTER_IDX, _MINUTES_Z_IDX) + 1:
        raise ValueError(
            f"train_x_n_raw must have shape (N, >= {max(_STARTER_IDX, _MINUTES_Z_IDX) + 1}); "
            f"got {tuple(train_x_n_raw.shape)}"
        )
    if val_x_n_raw.shape[1] != train_x_n_raw.shape[1]:
        raise ValueError("val_x_n_raw must have the same context dim as train_x_n_raw")

    if device is None:
        dev_obj: torch.device = next(timing_head.parameters()).device
    elif isinstance(device, str):
        dev_obj = torch.device(device)
    else:
        dev_obj = device
    train_x = train_x_n_raw.to(dev_obj)
    train_t = train_tau_bin.to(dev_obj)
    val_x = val_x_n_raw.to(dev_obj)
    val_t = val_tau_bin.to(dev_obj)

    was_training_th = timing_head.training
    was_training_ctx = context_mlp.training
    timing_head.eval()
    context_mlp.eval()
    try:
        # Trained head: per-shot NLL + aggregate predicted distribution.
        with torch.no_grad():
            x_n_val = context_mlp(val_x)
            log_p_val = timing_head.log_prob(val_t, x_n_val)
            head_nll = float((-log_p_val).mean().item())
            # Predicted aggregate distribution: average per-shot
            # softmax probability vector over val shots.
            logits = timing_head(x_n_val)
            probs = torch.softmax(logits, dim=-1)  # (N, n_bins)
            pred_agg = probs.mean(dim=0).to(torch.float32)

        # Observed val histogram (smoothed for comparability with pred_agg).
        obs_agg = _empirical_histogram(val_t, n_bins).to(dev_obj)

        # Baselines: fit on train set, evaluated on val.
        global_hist = _empirical_histogram(train_t, n_bins).to(dev_obj)
        starter_hist, bench_hist = _starter_baseline_split(train_x, train_t, n_bins)
        starter_hist = starter_hist.to(dev_obj)
        bench_hist = bench_hist.to(dev_obj)
        minutes_edges, minutes_hists = _minutes_baseline_split(
            train_x, train_t, n_bins, n_buckets=n_minutes_buckets
        )
        minutes_edges = minutes_edges.to(dev_obj)
        minutes_hists = minutes_hists.to(dev_obj)

        baseline_nll = _baseline_nll_per_shot(
            val_x,
            val_t,
            global_hist=global_hist,
            starter_hist=starter_hist,
            bench_hist=bench_hist,
            minutes_edges=minutes_edges,
            minutes_hists=minutes_hists,
        )

        # 48-bin L1 between predicted-aggregate and observed-aggregate.
        l1_48 = _bin_l1(pred_agg.detach().cpu(), obs_agg.detach().cpu())
        # Quarter-aggregated L1.
        l1_4 = _bin_l1(
            _quarter_aggregate(pred_agg).detach().cpu(), _quarter_aggregate(obs_agg).detach().cpu()
        )

        # Predictive p10/p50/p90 averaged across val shots: per-shot
        # predictive CDF on the bin axis, then quantile lookup.
        with torch.no_grad():
            cdf = probs.cumsum(dim=-1)  # (N, n_bins)

            # For each row find the smallest bin index with cdf >= q.
            def _q_idx(q: float) -> float:
                idx = (cdf >= q).float().argmax(dim=-1)
                return float(idx.float().mean().item())

            p10 = _q_idx(0.10)
            p50 = _q_idx(0.50)
            p90 = _q_idx(0.90)

        return {
            "n_shots_train": int(train_x.shape[0]),
            "n_shots_val": int(val_x.shape[0]),
            "head_nll_per_shot": head_nll,
            "baseline_nll_per_shot": baseline_nll,
            "nll_advantage_over_minutes_baseline": baseline_nll["minutes_conditioned"] - head_nll,
            "nll_advantage_over_global_baseline": baseline_nll["global"] - head_nll,
            "aggregate_48bin_l1": l1_48,
            "aggregate_quarter_l1": l1_4,
            "predicted_p10_p50_p90_mean_bin": [p10, p50, p90],
            "head_n_bins": n_bins,
        }
    finally:
        timing_head.train(was_training_th)
        context_mlp.train(was_training_ctx)
