"""Tests for :func:`shotcloud.evaluation.compute_count_calibration` and
the count-only training loop :func:`shotcloud.training.train_count_only`.
"""

from __future__ import annotations

import torch

from shotcloud.data.context import CONTEXT_DIM
from shotcloud.evaluation import compute_count_calibration
from shotcloud.models.context_mlp import ContextMLP
from shotcloud.models.count_head import NegBinCountHead
from shotcloud.training import train_count_only


def _make_synthetic_per_game(
    n_games: int = 64,
    k_mean: float = 10.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return random raw contexts and Poisson shot counts with mean ``k_mean``."""
    g = torch.Generator().manual_seed(seed)
    x_raw = torch.randn(n_games, CONTEXT_DIM, generator=g)
    k_obs = torch.poisson(torch.full((n_games,), k_mean), generator=g).long()
    return x_raw, k_obs


def test_compute_count_calibration_returns_expected_keys() -> None:
    """The calibration report has every documented key and μ̄/K̄ ≈ 1 at init."""
    x_raw, k_obs = _make_synthetic_per_game()
    count_head = NegBinCountHead(init_mean=10.0)
    ctx = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    diag = compute_count_calibration(count_head, ctx, x_raw, k_obs)
    for key in (
        "n_games",
        "mean_K_obs",
        "mean_mu",
        "ratio_mu_over_K",
        "MAE",
        "RMSE",
        "log_kappa",
        "kappa",
        "nll_per_game",
        "calibration_slope",
        "calibration_intercept",
        "p10_p50_p90_predicted_mean",
    ):
        assert key in diag, f"missing key {key!r}"
    assert diag["n_games"] == x_raw.shape[0]
    assert len(diag["p10_p50_p90_predicted_mean"]) == 3
    # init_mean = K̄ → ratio near 1 at init.
    assert 0.8 < diag["ratio_mu_over_K"] < 1.25


def test_compute_count_calibration_handles_bad_shapes() -> None:
    """Malformed context or count inputs raise ``ValueError``."""
    x_raw, k_obs = _make_synthetic_per_game()
    count_head = NegBinCountHead(init_mean=10.0)
    ctx = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    # x_raw shape mismatch.
    try:
        compute_count_calibration(count_head, ctx, x_raw.flatten(), k_obs)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 1D x_raw")
    # k mismatched length.
    try:
        compute_count_calibration(count_head, ctx, x_raw, k_obs[:5])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for length-mismatched k")


def test_train_count_only_reduces_nll() -> None:
    """Count-only training does not raise the train NLL (beyond 0.05) and keeps μ̄ calibrated."""
    x_raw, k_obs = _make_synthetic_per_game(n_games=128, k_mean=10.0)
    val_x, val_k = _make_synthetic_per_game(n_games=32, k_mean=10.0, seed=1)
    count_head = NegBinCountHead(init_mean=k_obs.float().mean().item())
    ctx = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    hist = train_count_only(
        count_head=count_head,
        context_mlp=ctx,
        train_per_game_x_raw=x_raw,
        train_per_game_k=k_obs,
        val_per_game_x_raw=val_x,
        val_per_game_k=val_k,
        n_epochs=8,
        batch_size=32,
        learning_rate=1e-2,
    )
    # NLL should not blow up; this is a sanity check, not a tight bound.
    assert hist.train_count[-1] <= hist.train_count[0] + 0.05
    # Calibration: μ̄ stays in the right ballpark with init_mean = K̄.
    assert 5.0 < hist.train_mean_mu[-1] < 20.0
    # Diagnostic populated.
    final = compute_count_calibration(count_head, ctx, val_x, val_k)
    assert 0.5 < final["ratio_mu_over_K"] < 2.0
    assert final["MAE"] > 0.0  # Poisson dispersion guarantees some error.


def test_train_count_only_runs_without_val() -> None:
    """Training without validation data records an empty validation history."""
    x_raw, k_obs = _make_synthetic_per_game()
    count_head = NegBinCountHead(init_mean=10.0)
    ctx = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    hist = train_count_only(
        count_head=count_head,
        context_mlp=ctx,
        train_per_game_x_raw=x_raw,
        train_per_game_k=k_obs,
        n_epochs=3,
        batch_size=16,
        learning_rate=1e-2,
    )
    assert hist.val_count == []
    assert len(hist.train_count) == 3
