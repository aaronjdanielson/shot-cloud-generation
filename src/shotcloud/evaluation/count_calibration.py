"""Count-head calibration diagnostics (paper §5.2).

The 2026-06-07 audit established that the joint-trained count head
under the old per-shot-amortized loss collapsed to ``μ ≈ 1.1`` against
``K̄_train ≈ 9.57``. This module provides a single function that
summarizes the count head's calibration on a per-game test set, used
both by :mod:`scripts.train_count_head` (final pretrain diagnostic)
and by :mod:`scripts.train_gibbs` (joint-training-final-epoch
diagnostic).

The returned dict is JSON-serializable and intended for ``manifest /
diagnostics`` artifacts; printable summary is left to the caller.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from shotcloud.models.context_mlp import ContextMLP
from shotcloud.models.count_head import NegBinCountHead


def compute_count_calibration(
    count_head: NegBinCountHead,
    context_mlp: ContextMLP,
    per_game_x_raw: Tensor,
    per_game_k: Tensor,
    *,
    n_quantile_samples: int = 512,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Calibration summary for ``count_head`` on a per-game eval set.

    Parameters
    ----------
    count_head, context_mlp : the modules under evaluation. Both are
        switched to ``.eval()`` inside this function and restored on
        exit.
    per_game_x_raw : (n_games, CONTEXT_DIM) raw per-game context.
    per_game_k : (n_games,) integer observed counts.
    n_quantile_samples : Monte Carlo sample size for the predictive
        p10/p50/p90 (NegBin has no closed-form icdf).
    device : where to run the forward pass. Defaults to the count
        head's current device.

    Returns
    -------
    dict with keys:

    * ``n_games``: number of evaluated games.
    * ``mean_K_obs`` / ``mean_mu``: scalar means.
    * ``ratio_mu_over_K``: ``mean_mu / mean_K_obs``. ``≈ 1`` is
      calibrated; ``< 0.2`` is the pre-audit failure mode.
    * ``MAE`` / ``RMSE``: per-game prediction error of ``μ`` against
      ``K_obs``.
    * ``nll_per_game``: mean NegBin NLL on the eval set.
    * ``calibration_slope`` / ``calibration_intercept``: OLS fit
      ``K_obs ~ a + b·μ``. ``b ≈ 1, a ≈ 0`` is calibrated.
    * ``kappa`` / ``log_kappa``: scalar dispersion parameter.
    * ``p10_p50_p90_predicted_mean``: predictive quantile means
      across games (Monte Carlo).
    """
    if per_game_x_raw.dim() != 2:
        raise ValueError(
            f"per_game_x_raw must be (n_games, context_dim); got {tuple(per_game_x_raw.shape)}"
        )
    if per_game_k.shape[0] != per_game_x_raw.shape[0] or per_game_k.dim() != 1:
        raise ValueError(
            f"per_game_k must be (n_games,) matching x_raw; got "
            f"{tuple(per_game_k.shape)} vs {tuple(per_game_x_raw.shape)}"
        )

    dev_obj: torch.device
    if device is None:
        dev_obj = next(count_head.parameters()).device
    elif isinstance(device, str):
        dev_obj = torch.device(device)
    else:
        dev_obj = device

    was_training_ch = count_head.training
    was_training_ctx = context_mlp.training
    count_head.eval()
    context_mlp.eval()
    try:
        with torch.no_grad():
            x_raw = per_game_x_raw.to(dev_obj)
            k = per_game_k.to(dev_obj)
            x_n = context_mlp(x_raw)
            mu, kappa = count_head(x_n)
            log_p = count_head.log_prob(k, x_n)
            # ``NegativeBinomial.sample()`` routes through Gamma, whose
            # ``aten::_standard_gamma`` op is not implemented on MPS as of
            # PyTorch 2.4 (https://github.com/pytorch/pytorch/issues/141287).
            # Move μ, κ to CPU for the sampling step; the forward NLL
            # stays on the original device.
            mu_cpu = mu.detach().cpu()
            kappa_cpu = kappa.detach().cpu()
            nb_cpu = torch.distributions.NegativeBinomial(
                total_count=kappa_cpu, probs=mu_cpu / (mu_cpu + kappa_cpu)
            )
            samples = nb_cpu.sample((n_quantile_samples,))  # type: ignore[no-untyped-call]  # (S, G)
            p10 = samples.quantile(0.10, dim=0)
            p50 = samples.quantile(0.50, dim=0)
            p90 = samples.quantile(0.90, dim=0)

        k_float = k.float()
        mu_np = mu.detach().cpu().numpy()
        k_np = k_float.detach().cpu().numpy()
        # OLS: K_obs ~ intercept + slope · μ.
        A = np.vstack([np.ones_like(mu_np), mu_np]).T
        coef, *_ = np.linalg.lstsq(A, k_np, rcond=None)
        intercept, slope = float(coef[0]), float(coef[1])
        err = mu_np - k_np
        kappa_scalar = float(torch.nn.functional.softplus(count_head.log_kappa).item())
        return {
            "n_games": int(per_game_k.shape[0]),
            "mean_K_obs": float(k_float.mean().item()),
            "mean_mu": float(mu.mean().item()),
            "ratio_mu_over_K": float(mu.mean().item() / max(k_float.mean().item(), 1e-9)),
            "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "log_kappa": float(count_head.log_kappa.detach().item()),
            "kappa": kappa_scalar,
            "nll_per_game": float(-log_p.mean().item()),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "p10_p50_p90_predicted_mean": [
                float(p10.float().mean().item()),
                float(p50.float().mean().item()),
                float(p90.float().mean().item()),
            ],
        }
    finally:
        count_head.train(was_training_ch)
        context_mlp.train(was_training_ctx)
