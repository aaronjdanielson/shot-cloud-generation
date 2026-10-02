"""Tests for :func:`shotcloud.evaluation.compute_timing_calibration`
and the timing-only training loop :func:`shotcloud.training.train_timing_only`.
"""

from __future__ import annotations

import math

import pytest
import torch

from shotcloud.data.context import CONTEXT_DIM
from shotcloud.evaluation import compute_timing_calibration
from shotcloud.models.context_mlp import ContextMLP
from shotcloud.models.timing_head import TimingSoftmaxHead
from shotcloud.training import train_timing_only


def _make_synthetic_per_shot(
    n_shots: int = 600,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic per-shot data in which starter status (slot 6) determines ``tau_bin``:
    starters shoot in the first half of the game, bench players in the second.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_shots, CONTEXT_DIM, generator=g)
    x[: n_shots // 2, 6] = 1.0
    x[n_shots // 2 :, 6] = 0.0
    x[:, 7] = torch.randn(n_shots, generator=g)
    tau = torch.where(
        x[:, 6] >= 0.5,
        torch.randint(0, 24, (n_shots,), generator=g),
        torch.randint(24, 48, (n_shots,), generator=g),
    )
    return x, tau


def test_compute_timing_calibration_returns_expected_keys() -> None:
    x_train, tau_train = _make_synthetic_per_shot()
    x_val, tau_val = _make_synthetic_per_shot(seed=1)
    th = TimingSoftmaxHead()
    ctx = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    diag = compute_timing_calibration(th, ctx, x_train, tau_train, x_val, tau_val)
    for key in (
        "n_shots_train",
        "n_shots_val",
        "head_nll_per_shot",
        "baseline_nll_per_shot",
        "nll_advantage_over_minutes_baseline",
        "nll_advantage_over_global_baseline",
        "aggregate_48bin_l1",
        "aggregate_quarter_l1",
        "predicted_p10_p50_p90_mean_bin",
        "head_n_bins",
    ):
        assert key in diag, f"missing key {key!r}"
    for k in ("global", "starter_bench", "minutes_conditioned"):
        assert k in diag["baseline_nll_per_shot"], f"missing baseline {k!r}"
    assert len(diag["predicted_p10_p50_p90_mean_bin"]) == 3


def test_starter_bench_baseline_wins_when_starter_perfectly_encodes_timing() -> None:
    x_train, tau_train = _make_synthetic_per_shot()
    x_val, tau_val = _make_synthetic_per_shot(seed=1)
    th = TimingSoftmaxHead()
    ctx = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    diag = compute_timing_calibration(th, ctx, x_train, tau_train, x_val, tau_val)
    bn = diag["baseline_nll_per_shot"]
    # The starter/bench baseline matches the generating process.
    assert bn["starter_bench"] < bn["global"]
    assert bn["starter_bench"] < bn["minutes_conditioned"]
    # Optimal NLL for two 24-bin half-uniform distributions is log(24) ≈ 3.18.
    assert bn["starter_bench"] < math.log(48.0)  # ≈ 3.871


def test_uniform_head_yields_log48_nll() -> None:
    """An untrained ``TimingSoftmaxHead`` is uniform over bins, so its per-shot NLL is
    ``log(48)`` for any ``tau_bin`` distribution."""
    x_val, tau_val = _make_synthetic_per_shot(seed=1)
    x_train, tau_train = _make_synthetic_per_shot()
    th = TimingSoftmaxHead()
    ctx = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    diag = compute_timing_calibration(th, ctx, x_train, tau_train, x_val, tau_val)
    assert abs(diag["head_nll_per_shot"] - math.log(48.0)) < 1e-3


def test_train_timing_only_reduces_nll() -> None:
    x_train, tau_train = _make_synthetic_per_shot(n_shots=600, seed=0)
    x_val, tau_val = _make_synthetic_per_shot(n_shots=200, seed=1)
    th = TimingSoftmaxHead()
    ctx = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    hist = train_timing_only(
        timing_head=th,
        context_mlp=ctx,
        train_x_n_raw=x_train,
        train_tau_bin=tau_train,
        val_x_n_raw=x_val,
        val_tau_bin=tau_val,
        n_epochs=3,
        batch_size=64,
        learning_rate=5e-3,
    )
    assert hist.train_timing[-1] < hist.train_timing[0]


def test_train_timing_only_runs_without_val() -> None:
    x_train, tau_train = _make_synthetic_per_shot()
    th = TimingSoftmaxHead()
    ctx = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    hist = train_timing_only(
        timing_head=th,
        context_mlp=ctx,
        train_x_n_raw=x_train,
        train_tau_bin=tau_train,
        n_epochs=2,
        batch_size=64,
        learning_rate=5e-3,
    )
    assert hist.val_timing == []
    assert len(hist.train_timing) == 2


def test_validation_errors() -> None:
    x_train, tau_train = _make_synthetic_per_shot()
    x_val, tau_val = _make_synthetic_per_shot(seed=1)
    th = TimingSoftmaxHead()
    ctx = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    # context-dim mismatch.
    with pytest.raises(ValueError, match="same context dim"):
        compute_timing_calibration(th, ctx, x_train, tau_train, x_val[:, :10], tau_val)
