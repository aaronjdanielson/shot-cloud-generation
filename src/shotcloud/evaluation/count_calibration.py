"""Calibration diagnostics for the negative-binomial count head.

:func:`compute_count_calibration` summarizes how well a
:class:`~shotcloud.models.count_head.NegBinCountHead` predicts per-game
shot counts on a held-out set: mean ratio, point-prediction error,
negative-binomial NLL, an OLS calibration line, and Monte Carlo
predictive quantiles. The returned dict is JSON-serializable, for
inclusion in run manifests and diagnostics files.
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

    The context MLP maps the raw context to ``x_n``, which the count
    head maps to ``(μ, κ)``.

    Parameters
    ----------
    count_head : NegBinCountHead
        Count head under evaluation.
    context_mlp : ContextMLP
        Context MLP producing the head's input. Both modules are
        switched to ``.eval()`` for the evaluation and their previous
        training mode is restored on exit.
    per_game_x_raw : Tensor of shape (n_games, CONTEXT_DIM)
        Raw per-game context vectors.
    per_game_k : Tensor of shape (n_games,)
        Observed integer shot counts.
    n_quantile_samples : int, default 512
        Monte Carlo sample size for the predictive p10/p50/p90 (the
        negative binomial has no closed-form inverse CDF).
    device : torch.device or str, optional
        Device for the forward pass. Defaults to the count head's
        current device.

    Returns
    -------
    dict
        With keys:

        * ``n_games``: number of evaluated games.
        * ``mean_K_obs`` / ``mean_mu``: scalar means.
        * ``ratio_mu_over_K``: ``mean_mu / mean_K_obs``; ``≈ 1`` for a
          calibrated head.
        * ``MAE`` / ``RMSE``: per-game prediction error of ``μ`` against
          ``K_obs``.
        * ``nll_per_game``: mean NegBin NLL on the eval set.
        * ``calibration_slope`` / ``calibration_intercept``: OLS fit
          ``K_obs ~ a + b·μ``; ``b ≈ 1, a ≈ 0`` for a calibrated head.
        * ``log_kappa``: the raw dispersion parameter; ``kappa``: the
          dispersion ``κ = softplus(log_kappa)``.
        * ``p10_p50_p90_predicted_mean``: predictive quantiles averaged
          across games (Monte Carlo).

    Raises
    ------
    ValueError
        If ``per_game_x_raw`` is not 2-D or ``per_game_k`` is not a 1-D
        tensor with one entry per game.
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
            # ``aten::_standard_gamma`` op is not implemented on MPS
            # (https://github.com/pytorch/pytorch/issues/141287). Sample
            # on CPU; the forward NLL stays on the original device.
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
