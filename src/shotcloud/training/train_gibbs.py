"""Trainers for the marked point-process model.

:func:`train_gibbs` fits the spatial, timing and count factors jointly on
a :class:`~shotcloud.training.GibbsShotDataset`, under the shared learned
context :math:`x_n = f_{\\mathrm{ctx}}(\\tilde x_n)`. The spatial factor
is selected by ``spatial_likelihood``:

* ``"continuous_mixture"`` -- the AC-KDE spatial factor,
  :class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`
  (or its mode-routed subclass when a ``mode_router`` is given), scored by
  the cell-free kernel-mixture likelihood;
* ``"mode_mixture"`` -- the mode-mixture decoder
  :class:`~shotcloud.models.collaborative_mode_mixture.CollaborativeModeMixtureSpatial`,
  an alternative evaluated as an ablation;
* ``"cell"`` and ``"continuous_cell"`` -- the deprecated grid-cell
  :class:`~shotcloud.legacy_pivot.gibbs_decoder.ConditionalGibbsDecoder`,
  scored by exact-cell or continuous-coordinate NLL.

With the default ``count_loss_normalization="per_game"`` the minibatch
objective is

.. math::

    L = \\frac{1}{B} \\sum_{n \\in \\mathrm{batch}} \\bigl[
          \\lambda_s \\ell^{\\mathrm{spatial}}_n
        + \\lambda_t \\ell^{\\mathrm{timing}}_n
        + \\lambda_{\\mathrm{tilt}} \\overline{R_\\theta^2}_n
        + \\lambda_D \\overline{D^2}_n \\bigr]
      + \\lambda_c \\frac{1}{G} \\sum_{g \\in \\mathrm{batch}}
          \\ell^{\\mathrm{count}}_g,

where the bars are means of the squared residual tilt and opponent
reweighting over the row's support shots (or cells), and the count NLL
is averaged over the ``G`` distinct player-games in the batch.

The module also provides single-factor trainers --
:func:`train_count_only`, :func:`train_timing_only` and
:func:`train_presence_only` -- used to pretrain a head before it is
loaded, and optionally frozen, in the joint trainer.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from shotcloud.features.defense_features import DefenseFeatures
from shotcloud.features.matchup_features import MatchupFeatures
from shotcloud.grids import CourtGrid
from shotcloud.legacy_pivot.adaptive_defensive import AdaptiveDefensiveField
from shotcloud.legacy_pivot.adaptive_prior import AdaptiveOffensivePrior
from shotcloud.legacy_pivot.gibbs_decoder import ConditionalGibbsDecoder
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder
from shotcloud.models import (
    ContextMLP,
    ContextResidualEncoder,
    NegBinCountHead,
    TimingSoftmaxHead,
)
from shotcloud.models.anisotropic_kernel import (
    FullCovarianceZoneKernel,
    RadialTangentZoneKernel,
)
from shotcloud.models.collaborative_kde import CollaborativeKDE, CollaborativeOutputs
from shotcloud.models.collaborative_mode_mixture import CollaborativeModeMixtureSpatial
from shotcloud.models.continuous_adaptive_defensive import ContinuousAdaptiveDefensiveField
from shotcloud.models.continuous_mixture_spatial import ContinuousMixtureSpatial
from shotcloud.models.defensive_retrieval_cache import DefensiveRetrievalCache
from shotcloud.models.mode_routed_spatial import ModeRoutedContinuousMixtureSpatial
from shotcloud.models.pooling_gate import PoolingGate
from shotcloud.models.presence import PresenceModel
from shotcloud.models.retrieval_collaborative_kde import RetrievalCollaborativeKDE
from shotcloud.models.zone_defense_reweighting import (
    MatchupReweightingDefense,
    ZoneReweightingDefense,
)
from shotcloud.models.zone_source_bandwidth import ZoneSourceBandwidth
from shotcloud.training.gibbs_dataset import GibbsShotDataset
from shotcloud.training.spatial_losses import (
    continuous_coordinate_nll,
    exact_cell_nll,
    expected_distance_ft,
)

#: Spatial likelihood selected by ``train_gibbs(spatial_likelihood=...)``.
#:
#:  - ``"cell"``: categorical NLL of the observed cell (grid-cell decoder).
#:  - ``"continuous_cell"``: continuous-coordinate NLL via a Gaussian
#:    observation kernel against the grid log-probabilities (grid-cell
#:    decoder).
#:  - ``"continuous_mixture"``: cell-free kernel-mixture NLL over the
#:    causal support shots (the AC-KDE spatial factor).
#:  - ``"mode_mixture"``: per-row K-mode Gaussian mixture extracted from
#:    the attended support shots.
SpatialLikelihood = str  # validated by train_gibbs

#: Alias of :data:`SpatialLikelihood`.
SpatialLoss = SpatialLikelihood

#: How the count NLL enters the joint loss.
#:
#: - ``"per_game"`` (default): the per-game count NLL is averaged over the
#:   distinct games in the batch and added to the mean per-shot loss,
#:   ``loss = loss_per_shot.mean() + lambda_count *
#:   count_loss_per_game.mean()``. With ``lambda_count = 1`` one game's
#:   count NLL is weighted like one shot's spatial NLL. This is the
#:   normalization of the paper's training objective.
#: - ``"per_shot"``: the count NLL is amortized as
#:   ``-log p(K_g | x_g) / K_g`` over the game's shots and added to the
#:   per-shot loss. At ``lambda_count = 1`` the count term then carries
#:   about ``1/K̄`` of the spatial weight; retained to reproduce models
#:   trained with this normalization.
CountLossNormalization = str  # "per_game" | "per_shot", validated by _epoch


@dataclass
class CountOnlyTrainHistory:
    """Per-epoch history for :func:`train_count_only`.

    ``train_count`` / ``val_count`` are the count NLL averaged over games;
    ``train_mean_mu`` / ``val_mean_mu`` the mean predicted count ``μ``; and
    ``log_kappa`` the dispersion parameter, each recorded once per epoch
    (training values accumulate over the epoch's minibatches, validation
    values are computed after the epoch). ``train_mean_k`` / ``val_mean_k``
    are the observed mean counts, for comparison with ``μ``.
    """

    train_count: list[float] = field(default_factory=list)
    val_count: list[float] = field(default_factory=list)
    train_mean_mu: list[float] = field(default_factory=list)
    val_mean_mu: list[float] = field(default_factory=list)
    train_mean_k: float = float("nan")
    val_mean_k: float = float("nan")
    log_kappa: list[float] = field(default_factory=list)


def train_count_only(
    *,
    count_head: NegBinCountHead,
    context_mlp: ContextMLP,
    train_per_game_x_raw: Tensor,
    train_per_game_k: Tensor,
    val_per_game_x_raw: Tensor | None = None,
    val_per_game_k: Tensor | None = None,
    n_epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str | torch.device = "cpu",
    shuffle: bool = True,
    progress: bool = False,
) -> CountOnlyTrainHistory:
    """Train the count head alone on per-game shot counts.

    Iterates over per-game ``(x_n_raw, K_obs)`` rows of a
    :class:`~shotcloud.training.PerGameTable`, computes
    ``μ = softplus(g_η(f_ctx(x_n_raw)))`` and minimizes the
    negative-binomial NLL ``-log p(K | μ, κ)`` averaged over games.

    This is the pretraining step for the count factor: the resulting
    ``count_head`` and ``context_mlp`` states can be loaded by
    :func:`train_gibbs` through ``count_checkpoint_path`` and held fixed
    with ``freeze_count`` (and ``freeze_context_mlp``), so the spatial
    decoder consumes a fixed count prediction.

    Parameters
    ----------
    count_head : NegBinCountHead
        Trained in place.
    context_mlp : ContextMLP
        The same residual-zero-init MLP the joint trainer uses, so the
        ``x_n`` representation matches. Trained in place.
    train_per_game_x_raw : Tensor, shape ``(n_games, CONTEXT_DIM)``
        Per-game raw context (typically ``train_set.per_game.x_n_raw``).
    train_per_game_k : Tensor, shape ``(n_games,)`` integer
        Per-game observed shot counts.
    val_per_game_x_raw, val_per_game_k : Tensor, optional
        Validation per-game tensors. If both are given, the validation NLL
        is computed at the end of every epoch.
    n_epochs, batch_size, learning_rate, weight_decay
        Adam training configuration; ``batch_size`` counts games.
    device : str or torch.device, default "cpu"
        Device to train on.
    shuffle : bool, default True
        Shuffle games each epoch.
    progress : bool, default False
        Print one line of losses per epoch.

    Returns
    -------
    CountOnlyTrainHistory
        Per-epoch losses and calibration diagnostics.

    Raises
    ------
    ValueError
        On mismatched input shapes or when no parameter requires grad.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    count_head.to(dev)
    context_mlp.to(dev)

    train_per_game_x_raw = train_per_game_x_raw.to(dev)
    train_per_game_k = train_per_game_k.to(dev)
    if val_per_game_x_raw is not None and val_per_game_k is not None:
        val_per_game_x_raw = val_per_game_x_raw.to(dev)
        val_per_game_k = val_per_game_k.to(dev)

    if train_per_game_x_raw.dim() != 2:
        raise ValueError(
            f"train_per_game_x_raw must be (n_games, context_dim); "
            f"got {tuple(train_per_game_x_raw.shape)}"
        )
    if train_per_game_k.dim() != 1 or train_per_game_k.shape[0] != train_per_game_x_raw.shape[0]:
        raise ValueError(
            f"train_per_game_k must be (n_games,) matching x_raw; got "
            f"{tuple(train_per_game_k.shape)} vs {tuple(train_per_game_x_raw.shape)}"
        )

    params = [p for m in (count_head, context_mlp) for p in m.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters in count_head + context_mlp")
    optimizer = Adam(params, lr=learning_rate, weight_decay=weight_decay)

    history = CountOnlyTrainHistory()
    history.train_mean_k = float(train_per_game_k.float().mean().item())
    if val_per_game_k is not None:
        history.val_mean_k = float(val_per_game_k.float().mean().item())

    n_games = int(train_per_game_x_raw.shape[0])
    for epoch in range(1, n_epochs + 1):
        count_head.train()
        context_mlp.train()
        order = (
            torch.randperm(n_games, device=dev) if shuffle else torch.arange(n_games, device=dev)
        )
        total_train_nll = 0.0
        total_train_mu = 0.0
        n_train_seen = 0
        for start in range(0, n_games, batch_size):
            idx = order[start : start + batch_size]
            x_raw_b = train_per_game_x_raw[idx]
            k_b = train_per_game_k[idx]
            x_n_b = context_mlp(x_raw_b)
            log_p = count_head.log_prob(k_b, x_n_b)
            loss = -log_p.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            with torch.no_grad():
                mu_b, _ = count_head(x_n_b)
                total_train_mu += float(mu_b.sum().item())
            total_train_nll += float(-log_p.sum().detach().item())
            n_train_seen += int(idx.shape[0])
        history.train_count.append(total_train_nll / max(n_train_seen, 1))
        history.train_mean_mu.append(total_train_mu / max(n_train_seen, 1))
        history.log_kappa.append(float(count_head.log_kappa.detach().item()))

        if val_per_game_x_raw is not None and val_per_game_k is not None:
            count_head.eval()
            context_mlp.eval()
            with torch.no_grad():
                x_n_val = context_mlp(val_per_game_x_raw)
                log_p_val = count_head.log_prob(val_per_game_k, x_n_val)
                mu_val, _ = count_head(x_n_val)
                history.val_count.append(float((-log_p_val).mean().item()))
                history.val_mean_mu.append(float(mu_val.mean().item()))
        if progress:
            line = (
                f"epoch {epoch}/{n_epochs}  train_nll={history.train_count[-1]:.4f}  "
                f"mean_mu={history.train_mean_mu[-1]:.3f}  "
                f"log_kappa={history.log_kappa[-1]:.3f}"
            )
            if history.val_count:
                line += f"  val_nll={history.val_count[-1]:.4f}"
            print(line, flush=True)

    return history


@dataclass
class PresenceTrainHistory:
    """Per-epoch history for :func:`train_presence_only`.

    Per-bin binary cross-entropy and mean absolute error on train and
    validation rows, plus the model's scalar parameters ``rho`` (recency
    decay per day), ``b0`` (gate bias) and ``beta_h`` (gate history
    slope) after each epoch.
    """

    train_bce: list[float] = field(default_factory=list)
    val_bce: list[float] = field(default_factory=list)
    train_mae: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    rho: list[float] = field(default_factory=list)
    b0: list[float] = field(default_factory=list)
    beta_h: list[float] = field(default_factory=list)


def _build_presence_training_inputs(
    table: pd.DataFrame,
    player_to_position: dict[int, int],
    *,
    n_bins: int,
    same_starter_only: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build per-row presence training tensors from the on-court table.

    For each player-game row the target is the row's ``n_bins`` on-court
    vector and the priors are the same player's rows with strictly
    earlier ``game_date`` (and matching starter status when
    ``same_starter_only=True``).

    Returns a tuple ``(prior_bins_pad, prior_ages_days_pad, prior_mask_pad,
    position_idx, starter_idx, history_count, target_bins)``.
    """
    bin_cols = [f"bin_{b}" for b in range(n_bins)]
    n = len(table)
    grouped = {
        int(pid): g.sort_values("game_date").reset_index(drop=True)
        for pid, g in table.groupby("player_id", sort=False)
    }
    prior_bins_list: list[np.ndarray] = []
    prior_ages_list: list[np.ndarray] = []
    starters: list[int] = []
    positions: list[int] = []
    targets: list[np.ndarray] = []
    history_counts: list[int] = []
    k_max = 0
    for _, row in table.iterrows():
        pid = int(row["player_id"])
        date = row["game_date"]
        starter = int(row["starter"])
        target_vec = row[bin_cols].to_numpy(dtype=np.float32)
        g = grouped[pid]
        prior = g[g["game_date"].values < date]
        if same_starter_only:
            prior = prior[prior["starter"] == starter]
        if not prior.empty:
            prior_bins_arr = prior[bin_cols].to_numpy(dtype=np.float32)
            date_np = np.datetime64(pd.Timestamp(date).to_datetime64(), "D")
            prior_dates = prior["game_date"].values.astype("datetime64[D]")
            prior_ages_arr = (date_np - prior_dates).astype("timedelta64[D]").astype(np.float32)
        else:
            prior_bins_arr = np.zeros((0, n_bins), dtype=np.float32)
            prior_ages_arr = np.zeros(0, dtype=np.float32)
        k_max = max(k_max, prior_bins_arr.shape[0])
        prior_bins_list.append(prior_bins_arr)
        prior_ages_list.append(prior_ages_arr)
        starters.append(starter)
        positions.append(int(player_to_position.get(pid, 0)))
        targets.append(target_vec)
        history_counts.append(int(prior_bins_arr.shape[0]))

    if k_max == 0:
        k_max = 1  # avoid zero-sized tensors when no row has any priors

    prior_bins_pad = np.zeros((n, k_max, n_bins), dtype=np.float32)
    prior_ages_pad = np.zeros((n, k_max), dtype=np.float32)
    prior_mask_pad = np.zeros((n, k_max), dtype=np.float32)
    for i, (bins_i, ages_i) in enumerate(zip(prior_bins_list, prior_ages_list, strict=False)):
        k_i = bins_i.shape[0]
        if k_i > 0:
            prior_bins_pad[i, :k_i] = bins_i
            prior_ages_pad[i, :k_i] = ages_i
            prior_mask_pad[i, :k_i] = 1.0

    return (
        torch.from_numpy(prior_bins_pad),
        torch.from_numpy(prior_ages_pad),
        torch.from_numpy(prior_mask_pad),
        torch.tensor(positions, dtype=torch.long),
        torch.tensor(starters, dtype=torch.long),
        torch.tensor(history_counts, dtype=torch.float32),
        torch.from_numpy(np.stack(targets, axis=0)),
    )


def train_presence_only(
    *,
    presence_model: PresenceModel,
    table: pd.DataFrame,
    player_to_position: dict[int, int],
    val_table: pd.DataFrame | None = None,
    n_epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    device: str | torch.device = "cpu",
    shuffle: bool = True,
    progress: bool = False,
) -> PresenceTrainHistory:
    """Train a :class:`~shotcloud.models.presence.PresenceModel` on on-court vectors.

    Each player-game row's target is its per-bin on-court fraction; the
    inputs are the same player's games with strictly earlier dates and the
    same starter status, so no row sees its own or later games. The loss
    is per-bin binary cross-entropy.

    Parameters
    ----------
    presence_model : PresenceModel
        Trained in place; must expose ``n_bins``.
    table : DataFrame
        Training rows with ``player_id``, ``game_date``, ``starter`` and
        on-court columns ``bin_0`` ... ``bin_{n_bins-1}``.
    player_to_position : dict of int to int
        Position index per player ID; unknown players map to 0.
    val_table : DataFrame, optional
        Validation rows in the same format.
    n_epochs, batch_size, learning_rate, weight_decay
        Adam training configuration.
    device : str or torch.device, default "cpu"
        Device to train on.
    shuffle : bool, default True
        Shuffle rows each epoch.
    progress : bool, default False
        Print one line of losses per epoch.

    Returns
    -------
    PresenceTrainHistory
        Per-epoch losses and parameter values.
    """
    if not hasattr(presence_model, "n_bins"):
        raise ValueError("presence_model must expose ``n_bins`` attribute")
    n_bins = int(presence_model.n_bins)
    dev = torch.device(device) if isinstance(device, str) else device
    presence_model.to(dev)
    print("[presence] building training tensors", flush=True)
    (
        train_prior_bins,
        train_prior_ages,
        train_prior_mask,
        train_pos,
        train_starter,
        train_history,
        train_target,
    ) = _build_presence_training_inputs(table, player_to_position, n_bins=n_bins)
    train_prior_bins = train_prior_bins.to(dev)
    train_prior_ages = train_prior_ages.to(dev)
    train_prior_mask = train_prior_mask.to(dev)
    train_pos = train_pos.to(dev)
    train_starter = train_starter.to(dev)
    train_history = train_history.to(dev)
    train_target = train_target.to(dev)
    n_train = train_target.shape[0]
    print(f"[presence] train rows = {n_train}, K_max = {train_prior_bins.shape[1]}", flush=True)

    val_tensors: tuple[Tensor, ...] | None = None
    if val_table is not None and not val_table.empty:
        print("[presence] building val tensors", flush=True)
        raw = _build_presence_training_inputs(val_table, player_to_position, n_bins=n_bins)
        val_tensors = tuple(t.to(dev) for t in raw)
        print(f"[presence] val rows = {val_tensors[6].shape[0]}", flush=True)

    optimizer = Adam(
        [p for p in presence_model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history = PresenceTrainHistory()

    for epoch in range(1, n_epochs + 1):
        presence_model.train()
        order = (
            torch.randperm(n_train, device=dev) if shuffle else torch.arange(n_train, device=dev)
        )
        total_loss = 0.0
        total_mae = 0.0
        n_seen = 0
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            out = presence_model(
                prior_bins=train_prior_bins[idx],
                prior_ages_days=train_prior_ages[idx],
                prior_mask=train_prior_mask[idx],
                position_idx=train_pos[idx],
                starter_idx=train_starter[idx],
                history_count=train_history[idx],
            )
            target = train_target[idx]
            loss = torch.nn.functional.binary_cross_entropy(out, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            with torch.no_grad():
                total_loss += float(loss.detach().item()) * int(idx.shape[0])
                total_mae += float((out - target).abs().mean(dim=-1).sum().detach().item())
                n_seen += int(idx.shape[0])
        history.train_bce.append(total_loss / max(n_seen, 1))
        history.train_mae.append(total_mae / max(n_seen, 1))
        history.rho.append(float(presence_model.rho.detach().item()))
        history.b0.append(float(presence_model.b0.detach().item()))
        history.beta_h.append(float(presence_model.beta_h.detach().item()))

        if val_tensors is not None:
            presence_model.eval()
            with torch.no_grad():
                out_val = presence_model(
                    prior_bins=val_tensors[0],
                    prior_ages_days=val_tensors[1],
                    prior_mask=val_tensors[2],
                    position_idx=val_tensors[3],
                    starter_idx=val_tensors[4],
                    history_count=val_tensors[5],
                )
                target_val = val_tensors[6]
                history.val_bce.append(
                    float(torch.nn.functional.binary_cross_entropy(out_val, target_val).item())
                )
                history.val_mae.append(float((out_val - target_val).abs().mean().item()))
        if progress:
            line = (
                f"epoch {epoch}/{n_epochs}  "
                f"train_bce={history.train_bce[-1]:.4f}  "
                f"train_mae={history.train_mae[-1]:.4f}  "
                f"rho={history.rho[-1]:.5f}  b0={history.b0[-1]:.3f}  "
                f"beta_h={history.beta_h[-1]:.3f}"
            )
            if history.val_bce:
                line += f"  val_bce={history.val_bce[-1]:.4f}  val_mae={history.val_mae[-1]:.4f}"
            print(line, flush=True)
    return history


@dataclass
class TimingOnlyTrainHistory:
    """Per-epoch history for :func:`train_timing_only`.

    ``train_timing`` / ``val_timing`` are the timing NLL of the 48-bin
    softmax head (one bin per game minute) averaged over shots.
    """

    train_timing: list[float] = field(default_factory=list)
    val_timing: list[float] = field(default_factory=list)


def train_timing_only(
    *,
    timing_head: TimingSoftmaxHead,
    context_mlp: ContextMLP,
    train_x_n_raw: Tensor,
    train_tau_bin: Tensor,
    val_x_n_raw: Tensor | None = None,
    val_tau_bin: Tensor | None = None,
    n_epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str | torch.device = "cpu",
    shuffle: bool = True,
    progress: bool = False,
) -> TimingOnlyTrainHistory:
    """Train the timing head alone on per-shot timing bins.

    Iterates over per-shot ``(x_n_raw, tau_bin)`` rows of a
    :class:`~shotcloud.training.GibbsShotDataset` and minimizes the
    per-shot NLL ``-log ρ_η(τ | x_n)`` of the 48-bin softmax head. Used to
    pretrain the timing factor separately from the spatial and count
    factors; the timing counterpart of :func:`train_count_only`.

    Parameters
    ----------
    timing_head : TimingSoftmaxHead
        Trained in place.
    context_mlp : ContextMLP
        Same residual-zero-init MLP the joint trainer uses so the
        ``x_n = f_ctx(x_n_raw)`` representation matches. Trained in
        place.
    train_x_n_raw : Tensor of shape ``(N_train, CONTEXT_DIM)``
        Per-shot raw context (typically ``train_set.x_n_raw``).
    train_tau_bin : Tensor of shape ``(N_train,)`` int64
        Per-shot timing bins (typically ``train_set.tau_bin``).
    val_x_n_raw, val_tau_bin : Tensor, optional
        Validation per-shot tensors. If both are given, the validation
        timing NLL is computed at the end of every epoch.
    n_epochs, batch_size, learning_rate, weight_decay
        Adam training configuration; ``batch_size`` counts shots.
    device : str or torch.device, default "cpu"
        Device to train on.
    shuffle : bool, default True
        Shuffle shots each epoch.
    progress : bool, default False
        Print one line of losses per epoch.

    Returns
    -------
    TimingOnlyTrainHistory
        Per-epoch train/val timing NLL.

    Raises
    ------
    ValueError
        On mismatched input shapes or when no parameter requires grad.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    timing_head.to(dev)
    context_mlp.to(dev)

    train_x_n_raw = train_x_n_raw.to(dev)
    train_tau_bin = train_tau_bin.to(dev)
    if val_x_n_raw is not None and val_tau_bin is not None:
        val_x_n_raw = val_x_n_raw.to(dev)
        val_tau_bin = val_tau_bin.to(dev)

    if train_x_n_raw.dim() != 2:
        raise ValueError(
            f"train_x_n_raw must be (N, context_dim); got {tuple(train_x_n_raw.shape)}"
        )
    if train_tau_bin.dim() != 1 or train_tau_bin.shape[0] != train_x_n_raw.shape[0]:
        raise ValueError(
            f"train_tau_bin must be (N,) matching x_n_raw; got "
            f"{tuple(train_tau_bin.shape)} vs {tuple(train_x_n_raw.shape)}"
        )

    params = [p for m in (timing_head, context_mlp) for p in m.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters in timing_head + context_mlp")
    optimizer = Adam(params, lr=learning_rate, weight_decay=weight_decay)

    history = TimingOnlyTrainHistory()
    n_shots = int(train_x_n_raw.shape[0])
    for epoch in range(1, n_epochs + 1):
        timing_head.train()
        context_mlp.train()
        order = (
            torch.randperm(n_shots, device=dev) if shuffle else torch.arange(n_shots, device=dev)
        )
        total_train_nll = 0.0
        n_train_seen = 0
        for start in range(0, n_shots, batch_size):
            idx = order[start : start + batch_size]
            x_raw_b = train_x_n_raw[idx]
            t_b = train_tau_bin[idx]
            x_n_b = context_mlp(x_raw_b)
            log_p = timing_head.log_prob(t_b, x_n_b)
            loss = -log_p.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total_train_nll += float(-log_p.sum().detach().item())
            n_train_seen += int(idx.shape[0])
        history.train_timing.append(total_train_nll / max(n_train_seen, 1))

        if val_x_n_raw is not None and val_tau_bin is not None:
            timing_head.eval()
            context_mlp.eval()
            with torch.no_grad():
                x_n_val = context_mlp(val_x_n_raw)
                log_p_val = timing_head.log_prob(val_tau_bin, x_n_val)
                history.val_timing.append(float((-log_p_val).mean().item()))
        if progress:
            line = f"epoch {epoch}/{n_epochs}  train_nll={history.train_timing[-1]:.4f}"
            if history.val_timing:
                line += f"  val_nll={history.val_timing[-1]:.4f}"
            print(line, flush=True)

    return history


@dataclass
class GibbsTrainHistory:
    """Per-epoch loss and diagnostic history of :func:`train_gibbs`.

    Each series holds one value per epoch. Every ``train_*`` series except
    ``train_reg`` has a ``val_*`` counterpart, filled only when a
    validation set is given. Series that do not apply to the configured
    model are ``NaN``.

    * ``spatial`` is the optimized spatial NLL per shot: the kernel-mixture
      NLL (also in ``spatial_mix_nll``) for the cell-free likelihoods, and
      ``spatial_cell`` or ``spatial_continuous`` for ``"cell"`` and
      ``"continuous_cell"``. On the grid paths both cell losses and
      ``expected_distance_ft`` are always recorded for comparison.
    * ``timing`` is the per-shot timing NLL; ``count`` is the count NLL
      amortized per shot (each game's NLL divided by its shot count);
      ``reg`` is the weighted residual-tilt and opponent-reweighting
      penalty per shot.
    * ``total`` is ``spatial + timing + count + reg`` per shot, without the
      ``lambda_spatial``, ``lambda_timing`` and ``lambda_count`` weights.
    * The remaining series are diagnostics of the support weights,
      bandwidth, pooling gate, mode mixture, opponent reweighting and
      matchup channel; see :class:`_EpochLosses`.

    ``best_epoch`` is the epoch whose parameters were restored by
    ``restore_best_val``.
    """

    train_total: list[float] = field(default_factory=list)
    train_spatial: list[float] = field(default_factory=list)
    train_spatial_cell: list[float] = field(default_factory=list)
    train_spatial_continuous: list[float] = field(default_factory=list)
    train_spatial_mix_nll: list[float] = field(default_factory=list)
    train_expected_distance_ft: list[float] = field(default_factory=list)
    train_alpha_entropy: list[float] = field(default_factory=list)
    train_beta_entropy: list[float] = field(default_factory=list)
    train_frac_sub_uniform: list[float] = field(default_factory=list)
    train_support_entropy: list[float] = field(default_factory=list)
    train_effective_support_size: list[float] = field(default_factory=list)
    train_expected_support_distance_ft: list[float] = field(default_factory=list)
    train_min_dist_to_support_ft: list[float] = field(default_factory=list)
    train_mode_to_support_dist_ft: list[float] = field(default_factory=list)
    train_effective_modes: list[float] = field(default_factory=list)
    train_mode_min_pair_dist_ft: list[float] = field(default_factory=list)
    train_tail_responsibility: list[float] = field(default_factory=list)
    train_gate_lambda_mean: list[float] = field(default_factory=list)
    train_sigma_mean: list[float] = field(default_factory=list)
    train_frac_cold_start: list[float] = field(default_factory=list)
    train_timing: list[float] = field(default_factory=list)
    train_count: list[float] = field(default_factory=list)
    train_reg: list[float] = field(default_factory=list)
    #: Opponent-reweighting diagnostics. NaN when no defensive field is wired.
    train_def_score_mean: list[float] = field(default_factory=list)
    train_def_score_abs_mean: list[float] = field(default_factory=list)
    train_def_score_max_abs: list[float] = field(default_factory=list)
    train_def_cold_start_fraction: list[float] = field(default_factory=list)
    train_def_beta: list[float] = field(default_factory=list)
    #: Matchup-channel diagnostics. NaN when no matchup field is wired.
    #: ``matchup_effect_abs_mean`` and ``matchup_effect_max_abs`` are the
    #: per-row magnitudes of ``β_match · Δ̂``; ``matchup_beta`` is the
    #: scalar parameter; ``matchup_n_eff_*`` summarize the per-row
    #: effective sample size of peer-versus-opponent evidence.
    train_matchup_effect_abs_mean: list[float] = field(default_factory=list)
    train_matchup_effect_max_abs: list[float] = field(default_factory=list)
    train_matchup_beta: list[float] = field(default_factory=list)
    train_matchup_n_eff_mean: list[float] = field(default_factory=list)
    train_matchup_n_eff_p10: list[float] = field(default_factory=list)
    train_matchup_n_eff_p50: list[float] = field(default_factory=list)
    train_matchup_n_eff_p90: list[float] = field(default_factory=list)
    val_total: list[float] = field(default_factory=list)
    val_spatial: list[float] = field(default_factory=list)
    val_spatial_cell: list[float] = field(default_factory=list)
    val_spatial_continuous: list[float] = field(default_factory=list)
    val_spatial_mix_nll: list[float] = field(default_factory=list)
    val_expected_distance_ft: list[float] = field(default_factory=list)
    val_alpha_entropy: list[float] = field(default_factory=list)
    val_beta_entropy: list[float] = field(default_factory=list)
    val_frac_sub_uniform: list[float] = field(default_factory=list)
    val_support_entropy: list[float] = field(default_factory=list)
    val_effective_support_size: list[float] = field(default_factory=list)
    val_expected_support_distance_ft: list[float] = field(default_factory=list)
    val_min_dist_to_support_ft: list[float] = field(default_factory=list)
    val_mode_to_support_dist_ft: list[float] = field(default_factory=list)
    val_effective_modes: list[float] = field(default_factory=list)
    val_mode_min_pair_dist_ft: list[float] = field(default_factory=list)
    val_tail_responsibility: list[float] = field(default_factory=list)
    val_gate_lambda_mean: list[float] = field(default_factory=list)
    val_sigma_mean: list[float] = field(default_factory=list)
    val_frac_cold_start: list[float] = field(default_factory=list)
    val_timing: list[float] = field(default_factory=list)
    val_count: list[float] = field(default_factory=list)
    val_def_score_mean: list[float] = field(default_factory=list)
    val_def_score_abs_mean: list[float] = field(default_factory=list)
    val_def_score_max_abs: list[float] = field(default_factory=list)
    val_def_cold_start_fraction: list[float] = field(default_factory=list)
    val_def_beta: list[float] = field(default_factory=list)
    val_matchup_effect_abs_mean: list[float] = field(default_factory=list)
    val_matchup_effect_max_abs: list[float] = field(default_factory=list)
    val_matchup_beta: list[float] = field(default_factory=list)
    val_matchup_n_eff_mean: list[float] = field(default_factory=list)
    val_matchup_n_eff_p10: list[float] = field(default_factory=list)
    val_matchup_n_eff_p50: list[float] = field(default_factory=list)
    val_matchup_n_eff_p90: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    #: Count calibration on the training set of the returned model, from
    #: :func:`shotcloud.evaluation.compute_count_calibration`. Set when
    #: ``lambda_count > 0``, else ``None``; ``{"error": ...}`` if the
    #: computation failed.
    final_train_count_calibration: dict[str, object] | None = None
    #: Count calibration on the validation set, under the same rules.
    final_val_count_calibration: dict[str, object] | None = None


@dataclass
class _EpochLosses:
    """Epoch-level means of the losses and diagnostics computed by :func:`_epoch`."""

    total: float
    spatial: float  # the optimized spatial loss
    # Grid-derived series are NaN for the cell-free likelihoods, which
    # compute no grid log-probabilities.
    spatial_cell: float  # exact-cell NLL (grid modes only)
    spatial_continuous: float  # continuous-coord NLL (grid modes only)
    spatial_mix_nll: float  # cell-free mixture NLL (mixture modes only)
    expected_distance_ft: float  # E_c[||x_c - y||] over grid (grid modes only)
    alpha_entropy: float
    beta_entropy: float
    frac_sub_uniform: float  # grid modes only
    # Cell-free-only diagnostics.
    support_entropy: float  # mean H(w) under the joint mixture softmax
    effective_support_size: float  # mean exp(H(w))
    expected_support_distance_ft: float  # Σ_m w_m ||s_m - y||
    min_dist_to_support_ft: float
    #: mode_mixture only: mean over rows of the average distance from
    #: each of the K mode centers to its nearest causal support shot.
    #: Small values mean the modes lie among the support; the soft
    #: k-means extractor keeps it small by construction.
    mode_to_support_dist_ft: float
    #: mode_mixture only: mean ``exp(H(π))`` across rows, where π is
    #: the K-mode softmax. ``≈ K`` means modes are used uniformly;
    #: ``≈ 1`` means one mode dominates per row (collapse).
    effective_modes: float
    #: mode_mixture only: mean over rows of the min pairwise distance
    #: between mode centers. Collapse indicator: modes converging on
    #: the same point drive it toward 0.
    mode_min_pair_dist_ft: float
    #: mode_mixture with tail_weight > 0: mean posterior responsibility
    #: of the support-tail component. Large values mean the K modes
    #: under-cover the data and the tail explains most shots.
    tail_responsibility: float
    #: continuous_mixture with a pooling gate: mean own-support weight λ
    #: over rows with support. Expected to grow with the depth of the
    #: player's own history.
    gate_lambda_mean: float
    sigma_mean: float
    frac_cold_start: float  # fraction of rows with no valid causal support
    timing: float
    count: float
    reg: float
    #: Opponent-reweighting diagnostics; ``NaN`` when no defensive field
    #: is wired. ``def_score_mean`` is the per-row mean of D over the
    #: support (near 0 when the field is centered per row).
    #: ``def_score_abs_mean`` and ``def_score_max_abs`` are the typical
    #: and largest per-row magnitudes, which reveal a growing ``β_D``.
    #: ``def_cold_start_fraction`` is the fraction of rows without causal
    #: defensive evidence. ``def_beta`` is the scalar ``β_D`` after the
    #: epoch.
    def_score_mean: float
    def_score_abs_mean: float
    def_score_max_abs: float
    def_cold_start_fraction: float
    def_beta: float
    #: Matchup-channel diagnostics; NaN when no matchup field is wired.
    #: ``matchup_effect_abs_mean`` and ``matchup_effect_max_abs`` are
    #: per-row magnitudes of ``β_match · Δ̂_{p,d,z(s_m)}``;
    #: ``matchup_beta`` is the scalar ``β_match`` after the epoch;
    #: ``matchup_n_eff_{mean,p10,p50,p90}`` summarize the per-row
    #: effective sample size of peer-versus-opponent evidence.
    matchup_effect_abs_mean: float
    matchup_effect_max_abs: float
    matchup_beta: float
    matchup_n_eff_mean: float
    matchup_n_eff_p10: float
    matchup_n_eff_p50: float
    matchup_n_eff_p90: float


def _module_snapshot(module: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _restore_snapshot(module: nn.Module, snapshot: dict[str, Tensor]) -> None:
    module.load_state_dict(snapshot)


def _collect_modules(
    *,
    offensive_prior: nn.Module,
    count_head: NegBinCountHead,
    timing_head: TimingSoftmaxHead,
    context_mlp: ContextMLP,
    defensive_field: AdaptiveDefensiveField | None,
    residual_encoder: ContextResidualEncoder | None,
    tilt_decoder: LowRankTiltDecoder | None,
) -> dict[str, nn.Module]:
    """Named dict of every module participating in training.

    Always-on modules first, then optional ones in fixed declaration
    order. Used as a single source of truth for device placement,
    Adam param assembly, best-val snapshot/restore, and grad-NaN scans.
    """
    modules: dict[str, nn.Module] = {
        "offensive_prior": offensive_prior,
        "count_head": count_head,
        "timing_head": timing_head,
        "context_mlp": context_mlp,
    }
    if defensive_field is not None:
        modules["defensive_field"] = defensive_field
    if residual_encoder is not None:
        modules["residual_encoder"] = residual_encoder
    if tilt_decoder is not None:
        modules["tilt_decoder"] = tilt_decoder
    return modules


def _check_finite(name: str, tensor: Tensor, batch_idx: int) -> None:
    """Raise a clear error if the tensor contains any NaN or +/-Inf.

    On by default in the training loop: the cost is two reductions per
    check, while a single non-finite value would otherwise propagate
    through the optimizer into every later step.
    """
    with torch.no_grad():
        finite_mask = torch.isfinite(tensor)
        if not bool(finite_mask.all()):
            n_nan = int(torch.isnan(tensor).sum())
            n_inf = int(torch.isinf(tensor).sum())
            n_finite = int(finite_mask.sum())
            fmin = tensor[finite_mask].min().item() if n_finite else float("nan")
            fmax = tensor[finite_mask].max().item() if n_finite else float("nan")
            raise RuntimeError(
                f"[train_gibbs] non-finite values in {name!r} at "
                f"batch_idx={batch_idx}: shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}, device={tensor.device}, "
                f"n_nan={n_nan}, n_inf={n_inf}, "
                f"finite_min={fmin}, finite_max={fmax}"
            )


def _check_param_grads(modules: dict[str, nn.Module], batch_idx: int) -> None:
    """After backward, scan every module's parameter gradients for non-finite values."""
    for module_name, module in modules.items():
        for pname, p in module.named_parameters():
            if p.grad is None:
                continue
            if not bool(torch.isfinite(p.grad).all()):
                n_nan = int(torch.isnan(p.grad).sum())
                n_inf = int(torch.isinf(p.grad).sum())
                raise RuntimeError(
                    f"[train_gibbs] non-finite grad in {module_name}.{pname} "
                    f"at batch_idx={batch_idx}: shape={tuple(p.grad.shape)}, "
                    f"n_nan={n_nan}, n_inf={n_inf}, param_norm={p.detach().norm().item():.4g}"
                )


def _collect_modules_for_spatial(
    *,
    spatial: (ConditionalGibbsDecoder | ContinuousMixtureSpatial | CollaborativeModeMixtureSpatial),
    count_head: NegBinCountHead,
    timing_head: TimingSoftmaxHead,
    context_mlp: ContextMLP,
) -> dict[str, nn.Module]:
    """Return the named trainable modules for a spatial decoder.

    The dict drives device placement, optimizer parameters, best-val
    snapshot/restore and gradient checks. Dispatches on the decoder type
    so the cell-free and grid paths share one bookkeeping function.
    """
    if isinstance(spatial, ConditionalGibbsDecoder):
        return _collect_modules(
            offensive_prior=spatial.offensive_prior,
            count_head=count_head,
            timing_head=timing_head,
            context_mlp=context_mlp,
            defensive_field=spatial.defensive_field,
            residual_encoder=spatial.residual_encoder,
            tilt_decoder=spatial.tilt_decoder,
        )
    # Cell-free paths (continuous_mixture or mode_mixture) share the
    # support backend; only the spatial-density head differs.
    modules: dict[str, nn.Module] = {
        "offensive_prior": spatial.offensive_prior,
        "count_head": count_head,
        "timing_head": timing_head,
        "context_mlp": context_mlp,
    }
    if spatial.residual_encoder is not None:
        modules["residual_encoder"] = spatial.residual_encoder
    if isinstance(spatial, ContinuousMixtureSpatial):
        if spatial.location_embedding is not None:
            modules["location_embedding"] = spatial.location_embedding
        if spatial.pooling_gate is not None:
            modules["pooling_gate"] = spatial.pooling_gate
        # Optional parameterized components of the AC-KDE decoder. The
        # defensive cache and the defense / matchup feature tensors are
        # parameter-free attributes of the decoder, moved to the device
        # when gathered in its forward pass, so they are not listed.
        if spatial.defensive_field is not None:
            modules["defensive_field"] = spatial.defensive_field
        if spatial.bandwidth_field is not None:
            modules["bandwidth_field"] = spatial.bandwidth_field
        if spatial.anisotropic_kernel is not None:
            modules["anisotropic_kernel"] = spatial.anisotropic_kernel
        if spatial.matchup_field is not None:
            modules["matchup_field"] = spatial.matchup_field
        if spatial.has_within_game_gru:
            assert spatial.within_game_gru is not None  # narrowed by the flag
            modules["within_game_gru"] = spatial.within_game_gru
        if spatial.causal_zone_bias is not None:
            modules["causal_zone_bias"] = spatial.causal_zone_bias
        if isinstance(spatial, ModeRoutedContinuousMixtureSpatial):
            modules["mode_router"] = spatial.mode_router
    else:
        # CollaborativeModeMixtureSpatial: the residual location embedding
        # is the residual tilt's ψ; the mode extractor holds its own
        # parameters (the context-bias MLP, plus a separate support
        # embedding and mode queries for the learned-query extractor).
        if spatial.residual_location_embedding is not None:
            modules["residual_location_embedding"] = spatial.residual_location_embedding
        modules["mode_extractor"] = spatial.mode_extractor
    return modules


def _epoch(
    *,
    spatial: (ConditionalGibbsDecoder | ContinuousMixtureSpatial | CollaborativeModeMixtureSpatial),
    count_head: NegBinCountHead,
    timing_head: TimingSoftmaxHead,
    context_mlp: ContextMLP,
    loader: DataLoader[
        tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
        ]
    ],
    dataset: GibbsShotDataset,
    optimizer: Adam | None,
    lambda_tilt: float,
    lambda_spatial: float,
    lambda_timing: float,
    lambda_count: float,
    lambda_defense: float,
    cell_centers: Tensor,
    spatial_loss: SpatialLoss,
    obs_kernel_tau: float,
    obs_kernel_normalize: bool,
    device: torch.device,
    count_loss_normalization: CountLossNormalization = "per_game",
    nan_check: bool = True,
    max_batches: int | None = None,
) -> _EpochLosses:
    """Run one pass over ``loader``; with ``optimizer=None``, evaluate without gradients."""
    if count_loss_normalization not in ("per_game", "per_shot"):
        raise ValueError(
            f"count_loss_normalization must be 'per_game' or 'per_shot', "
            f"got {count_loss_normalization!r}"
        )
    train_mode = optimizer is not None
    if train_mode:
        spatial.train()
        count_head.train()
        timing_head.train()
        context_mlp.train()
    else:
        spatial.eval()
        count_head.eval()
        timing_head.eval()
        context_mlp.eval()

    # Per-game shot counts and per-game context, moved to the device once.
    shots_per_game = (
        torch.bincount(dataset.game_idx, minlength=dataset.n_games)
        .to(device=device, dtype=torch.float32)
        .clamp_min(1.0)
    )
    per_game_x_raw = dataset.per_game.x_n_raw.to(device)
    per_game_k = dataset.per_game.k_obs.to(device)

    total_spatial = 0.0  # optimized spatial loss, used for `total` and best-val
    total_spatial_cell = 0.0  # grid modes only
    total_spatial_continuous = 0.0  # grid modes only
    total_spatial_mix_nll = 0.0  # mixture modes only
    total_expected_distance_ft = 0.0  # grid modes only
    total_alpha_entropy = 0.0
    total_beta_entropy = 0.0
    n_alpha_rows = 0  # rows contributing an α entropy
    n_beta_rows = 0  # (b, l) pairs with valid causal history
    total_sub_uniform = 0  # shots with q(c_obs) < 1/n_cells
    total_support_entropy = 0.0  # H(w) of the support weights (mixture modes only)
    total_eff_support = 0.0  # exp(H(w)) (mixture modes only)
    total_expected_support_dist = 0.0  # Σ_m w_m ||s_m - y|| (mixture modes only)
    total_min_dist_to_support = 0.0  # min_m ||s_m - y|| (mixture modes only)
    # mean over rows of avg min-dist mode→support (mode_mixture only):
    total_mode_to_support_dist = 0.0
    n_mode_to_support_rows = 0  # # rows that contributed to total_mode_to_support_dist
    total_effective_modes = 0.0  # mode_mixture only
    n_effective_modes_rows = 0
    total_mode_min_pair_dist = 0.0  # mode_mixture only, requires K ≥ 2
    n_mode_min_pair_rows = 0
    total_tail_responsibility = 0.0  # mode_mixture + tail_weight > 0
    n_tail_rows = 0
    total_gate_lambda = 0.0  # continuous_mixture + pooling gate
    n_gate_lambda_rows = 0
    total_sigma = 0.0  # σ_mean across batch
    total_cold_start = 0  # rows with no valid causal support
    total_timing = 0.0
    total_count = 0.0
    total_reg = 0.0
    # Opponent-reweighting diagnostics (continuous_mixture with defense).
    # Row sums, divided by ``n_def_rows`` at the end.
    total_def_score_abs_mean = 0.0
    total_def_score_max_abs = 0.0  # per-row max |D|, batch-summed
    total_def_score_mean = 0.0
    total_def_cold_start_rows = 0
    n_def_rows = 0
    # Matchup-channel diagnostics. ``total_matchup_*`` are row sums
    # divided by ``n_match_rows`` at the end; ``matchup_n_eff_pool`` keeps
    # the per-row N_eff values for the percentiles.
    total_matchup_effect_abs_mean = 0.0
    total_matchup_effect_max_abs = 0.0
    total_matchup_n_eff_mean = 0.0
    n_match_rows = 0
    matchup_n_eff_pool: list[float] = []
    n_shots_seen = 0
    n_cells = int(cell_centers.shape[0])
    log_uniform = float(np.log(n_cells))  # log(1/C) is -log_uniform

    is_mixture_mode = isinstance(
        spatial, ContinuousMixtureSpatial | CollaborativeModeMixtureSpatial
    )
    modules_for_grad_check = _collect_modules_for_spatial(
        spatial=spatial, count_head=count_head, timing_head=timing_head, context_mlp=context_mlp
    )

    ctx_mgr = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx_mgr:
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            (
                player_idx,
                opp_idx,
                snapshot_idx,
                cell_idx,
                tau_bin,
                x_n_raw,
                game_idx,
                shot_xy,
                h_n,
                o_n,
                prior_seq,
                prior_lengths,
            ) = (b.to(device) for b in batch)

            if nan_check:
                _check_finite("x_n_raw", x_n_raw, batch_idx)

            x_n = context_mlp(x_n_raw)
            if nan_check:
                _check_finite("x_n", x_n, batch_idx)

            spatial_nll_cell = torch.full((player_idx.shape[0],), float("nan"), device=device)
            spatial_nll_continuous = torch.full_like(spatial_nll_cell, float("nan"))
            spatial_nll_mix = torch.full_like(spatial_nll_cell, float("nan"))
            expected_dist = torch.full_like(spatial_nll_cell, float("nan"))
            sub_uniform = 0
            alpha_h = float("nan")
            beta_h = float("nan")
            n_beta_in_batch = 0
            support_h = float("nan")
            eff_support = float("nan")
            expected_support_dist = float("nan")
            min_dist_to_support = float("nan")
            mode_to_support_dist_b = float("nan")
            n_rows_with_mode_to_support = 0
            effective_modes_b = float("nan")
            n_rows_with_effective_modes = 0
            mode_min_pair_dist_b = float("nan")
            n_rows_with_mode_pairs = 0
            tail_resp_b = float("nan")
            n_rows_with_tail_resp = 0
            gate_lambda_b = float("nan")
            n_rows_with_gate_lambda = 0
            sigma_mean_b = float("nan")
            cold_start_b = 0
            r_theta: Tensor

            if is_mixture_mode:
                assert isinstance(
                    spatial, ContinuousMixtureSpatial | CollaborativeModeMixtureSpatial
                )
                # Optional inputs are passed only to a decoder whose
                # corresponding component is wired: opp_idx for defense
                # or matchup, the prior-shot sequence for the within-game
                # GRU, and the prior-outcome features for the outcome
                # residual branch.
                mix_kwargs: dict[str, Tensor] = {
                    "player_idx": player_idx,
                    "snapshot_idx": snapshot_idx,
                    "x_n_raw": x_n_raw,
                    "x_n": x_n,
                    "shot_xy": shot_xy,
                    "h_n": h_n,
                }
                if isinstance(spatial, ContinuousMixtureSpatial) and (
                    spatial.has_defense or spatial.has_matchup
                ):
                    mix_kwargs["opp_idx"] = opp_idx
                if isinstance(spatial, ContinuousMixtureSpatial) and spatial.has_within_game_gru:
                    mix_kwargs["prior_seq"] = prior_seq
                    mix_kwargs["prior_lengths"] = prior_lengths
                if isinstance(spatial, ContinuousMixtureSpatial) and spatial.has_outcome_residual:
                    mix_kwargs["o_n"] = o_n
                mix_out = spatial(**mix_kwargs)
                if nan_check:
                    _check_finite("log_lik_mixture", mix_out.log_lik, batch_idx)
                spatial_nll_mix = -mix_out.log_lik
                spatial_nll = spatial_nll_mix
                # Residual tilt R_θ(s_m) per support shot, for the L2
                # penalty (the cell-free analog of the grid tilt penalty).
                r_theta = mix_out.residual_logits  # (B, M)
                # Diagnostics off the joint mixture.
                with torch.no_grad():
                    cold_start_b = int(mix_out.cold_start.sum().item())
                    log_w = mix_out.log_weights  # (B, M), -inf on invalid
                    w = log_w.exp()
                    support_entropy_per_row = -(
                        w * log_w.clamp_min(-1e30) * mix_out.support_mask.to(w.dtype)
                    ).nansum(dim=-1)
                    support_h = float(support_entropy_per_row.nan_to_num(0.0).mean().item())
                    eff_support = float(support_entropy_per_row.nan_to_num(0.0).exp().mean().item())
                    diff = shot_xy.unsqueeze(1) - mix_out.support_xy  # (B, M, 2)
                    dist = diff.norm(dim=-1)  # (B, M)
                    # Mask invalid support with +inf for min, 0 for E[].
                    dist_masked_inf = torch.where(
                        mix_out.support_mask, dist, torch.full_like(dist, float("inf"))
                    )
                    min_d = dist_masked_inf.min(dim=-1).values
                    # Cold-start rows carry a placeholder weight, so
                    # ``expected_d`` is finite but meaningless for them;
                    # both diagnostics exclude rows without support.
                    expected_d = (w * dist).sum(dim=-1)
                    has_support = mix_out.support_mask.any(dim=-1)
                    min_dist_to_support = (
                        float(min_d[has_support].mean().item())
                        if has_support.any()
                        else float("nan")
                    )
                    expected_support_dist = (
                        float(expected_d[has_support].mean().item())
                        if has_support.any()
                        else float("nan")
                    )
                    sigma_mean_b = float(mix_out.sigma.mean().item())
                    # Mode-locality diagnostic (mode_mixture only): min
                    # distance from each mode center to the row's causal
                    # support, averaged over modes and over rows with
                    # support. Large values mean modes have drifted away
                    # from the support.
                    if isinstance(spatial, CollaborativeModeMixtureSpatial):
                        modes_xy = mix_out.support_xy  # (B, K, 2) — mode centers
                        sup_xy = mix_out.collab.support_xy  # (B, M, 2)
                        sup_mask = mix_out.collab.support_mask  # (B, M)
                        d_ms = (modes_xy.unsqueeze(2) - sup_xy.unsqueeze(1)).norm(
                            dim=-1
                        )  # (B, K, M)
                        d_ms = torch.where(
                            sup_mask.unsqueeze(1),
                            d_ms,
                            torch.full_like(d_ms, float("inf")),
                        )
                        min_per_mode = d_ms.min(dim=-1).values  # (B, K)
                        avg_per_row = min_per_mode.mean(dim=-1)  # (B,)
                        if has_support.any():
                            mode_to_support_dist_b = float(avg_per_row[has_support].mean().item())
                            n_rows_with_mode_to_support = int(has_support.sum().item())

                        # Effective modes = mean exp(H(π)) over rows
                        # with valid support. log_weights are already
                        # the normalized mode log-probs.
                        log_pi = mix_out.log_weights  # (B, K)
                        pi = log_pi.exp()
                        h_pi = -(pi * log_pi.clamp_min(-1e30)).sum(dim=-1)
                        eff_per_row = h_pi.exp()  # (B,)
                        if has_support.any():
                            effective_modes_b = float(eff_per_row[has_support].mean().item())
                            n_rows_with_effective_modes = int(has_support.sum().item())

                        # min pairwise distance between mode centers.
                        # Collapse indicator. Only computable with K ≥ 2.
                        k_modes = modes_xy.shape[1]
                        if k_modes >= 2:
                            d_pp = (modes_xy.unsqueeze(2) - modes_xy.unsqueeze(1)).norm(
                                dim=-1
                            )  # (B, K, K)
                            # Mask the diagonal (self-distances = 0).
                            eye = torch.eye(k_modes, dtype=torch.bool, device=d_pp.device)
                            d_pp_masked = d_pp.masked_fill(eye, float("inf"))
                            min_pair_per_row = d_pp_masked.amin(dim=(1, 2))  # (B,)
                            if has_support.any():
                                mode_min_pair_dist_b = float(
                                    min_pair_per_row[has_support].mean().item()
                                )
                                n_rows_with_mode_pairs = int(has_support.sum().item())

                        # Tail responsibility (only set when tail_weight > 0).
                        gamma = mix_out.tail_responsibility
                        if gamma is not None and has_support.any():
                            tail_resp_b = float(gamma[has_support].mean().item())
                            n_rows_with_tail_resp = int(has_support.sum().item())
                    # Pooling-gate λ (continuous_mixture + gate only).
                    gate_lam = mix_out.gate_lambda
                    if gate_lam is not None and has_support.any():
                        gate_lambda_b = float(gate_lam[has_support].mean().item())
                        n_rows_with_gate_lambda = int(has_support.sum().item())
                    # α, β entropies of the support attention (always
                    # M-shaped), factored over analogues L and history
                    # slots R. For continuous_mixture the support
                    # attention equals the mixture weights; for
                    # mode_mixture it precedes the K mode weights but has
                    # the same (L, R) factorization. The retrieval backend
                    # has no (L, R) factorization, so both stay NaN.
                    if isinstance(spatial.offensive_prior, CollaborativeKDE):
                        sup_log_w = mix_out.support_log_weights  # (B, M)
                        sup_w = sup_log_w.exp()
                        L_ = spatial.offensive_prior.L
                        R_ = spatial.offensive_prior.max_history
                        sup_w_blr = sup_w.view(-1, L_, R_)
                        alpha_marg = sup_w_blr.sum(dim=-1)  # (B, L)
                        alpha_h = float(_safe_entropy(alpha_marg).mean().item())
                        alpha_safe = alpha_marg.clamp_min(1e-20).unsqueeze(-1)
                        beta_cond = sup_w_blr / alpha_safe  # (B, L, R)
                        has_mass = alpha_marg > 1e-9
                        if has_mass.any():
                            beta_h_per = _safe_entropy(beta_cond)
                            beta_h = float(beta_h_per[has_mass].mean().item())
                            n_beta_in_batch = int(has_mass.sum().item())
            else:
                assert isinstance(spatial, ConditionalGibbsDecoder)
                log_p_spatial, components = spatial(
                    player_idx,
                    opp_idx,
                    snapshot_idx,
                    x_n_raw,
                    x_n,
                    h_n=h_n,
                    return_components=True,
                )
                r_theta = components.r_theta
                if nan_check:
                    _check_finite("log_p_spatial", log_p_spatial, batch_idx)
                # Both grid losses and the expected-distance diagnostic
                # are always computed for comparison; ``spatial_loss``
                # selects the one that is optimized.
                spatial_nll_cell = exact_cell_nll(log_p_spatial, cell_idx)
                spatial_nll_continuous = continuous_coordinate_nll(
                    log_p_spatial,
                    shot_xy,
                    cell_centers,
                    tau=obs_kernel_tau,
                    normalize_kernel=obs_kernel_normalize,
                )
                with torch.no_grad():
                    expected_dist = expected_distance_ft(log_p_spatial, shot_xy, cell_centers)
                    obs_log_p = log_p_spatial.gather(1, cell_idx.unsqueeze(1)).squeeze(1)
                    sub_uniform = int((obs_log_p < -log_uniform).sum().item())
                    if isinstance(components.prior_components, CollaborativeOutputs):
                        coll = components.prior_components
                        alpha_h = float(_safe_entropy(coll.alpha).mean().item())
                        valid = coll.has_analogue_history
                        if valid.any():
                            beta_entropy_blr = _safe_entropy(coll.beta)
                            beta_h = float(beta_entropy_blr[valid].mean().item())
                            n_beta_in_batch = int(valid.sum().item())
                if spatial_loss == "continuous":
                    spatial_nll = spatial_nll_continuous
                else:
                    spatial_nll = spatial_nll_cell

            # Timing NLL.
            timing_log_p = timing_head.log_prob(tau_bin, x_n)
            if nan_check:
                _check_finite("timing_log_p", timing_log_p, batch_idx)
            timing_nll = -timing_log_p

            # The count NLL is per game: evaluate it once per distinct game
            # in the batch, so each game contributes one term per update
            # regardless of K_g. The per-game context (the first shot's
            # x_n_raw, see PerGameTable) runs through the same f_ctx.
            unique_games, inverse_to_unique = torch.unique(game_idx, return_inverse=True)
            game_x_raw_unique = per_game_x_raw[unique_games]
            game_x_n_unique = context_mlp(game_x_raw_unique)
            game_k_unique = per_game_k[unique_games]
            count_log_p_unique = count_head.log_prob(game_k_unique, game_x_n_unique)
            if nan_check:
                _check_finite("count_log_p_unique", count_log_p_unique, batch_idx)
            count_loss_per_game = -count_log_p_unique
            # Per-shot amortized view: the game's NLL divided by its shot
            # count. Used by the per_shot normalization and by the
            # reported ``count`` series.
            shots_in_game = shots_per_game[game_idx]
            count_log_p = count_log_p_unique[inverse_to_unique]
            count_nll_per_shot = -count_log_p / shots_in_game

            # Tilt penalty: mean squared residual over the row's output
            # axis, r_θ(c) = u^T v_c per cell on the grid path and
            # R_θ(s_m) = u^T ψ(s_m) per support shot on the cell-free path.
            if spatial.has_residual:
                tilt_reg_per_shot = (r_theta.pow(2)).mean(dim=-1)
            else:
                tilt_reg_per_shot = torch.zeros_like(spatial_nll)

            # Defense penalty: mean of D² over the row's support shots.
            # Zero without a defensive field; the mode-mixture decoder
            # has no defensive term.
            defense_reg_per_shot = torch.zeros_like(spatial_nll)
            if (
                is_mixture_mode
                and isinstance(spatial, ContinuousMixtureSpatial)
                and spatial.has_defense
                and mix_out.defense_logits is not None
            ):
                defense_reg_per_shot = mix_out.defense_logits.pow(2).mean(dim=-1)

            # Count term by normalization mode (see CountLossNormalization):
            # ``per_game`` averages the per-game NLL separately and adds
            # it to the per-shot mean; ``per_shot`` mixes the amortized
            # per-shot NLL into ``loss_per_shot``.
            if count_loss_normalization == "per_shot":
                loss_per_shot = (
                    lambda_spatial * spatial_nll
                    + lambda_timing * timing_nll
                    + lambda_count * count_nll_per_shot
                    + lambda_tilt * tilt_reg_per_shot
                    + lambda_defense * defense_reg_per_shot
                )
                loss = loss_per_shot.mean()
            else:
                loss_per_shot = (
                    lambda_spatial * spatial_nll
                    + lambda_timing * timing_nll
                    + lambda_tilt * tilt_reg_per_shot
                    + lambda_defense * defense_reg_per_shot
                )
                loss = loss_per_shot.mean() + lambda_count * count_loss_per_game.mean()
            if nan_check:
                _check_finite("loss", loss, batch_idx)

            if train_mode:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if nan_check:
                    _check_param_grads(modules_for_grad_check, batch_idx)
                optimizer.step()

            b_size = int(player_idx.shape[0])
            n_shots_seen += b_size
            total_spatial += float(spatial_nll.sum().detach())
            # Accumulate only the metrics defined for this mode; the
            # others are reported as NaN at the end.
            if is_mixture_mode:
                total_spatial_mix_nll += float(spatial_nll_mix.sum().detach())
                total_support_entropy += support_h * b_size
                total_eff_support += eff_support * b_size
                total_expected_support_dist += expected_support_dist * b_size
                if not np.isnan(min_dist_to_support):
                    total_min_dist_to_support += min_dist_to_support * b_size
                if not np.isnan(mode_to_support_dist_b) and n_rows_with_mode_to_support > 0:
                    total_mode_to_support_dist += (
                        mode_to_support_dist_b * n_rows_with_mode_to_support
                    )
                    n_mode_to_support_rows += n_rows_with_mode_to_support
                if not np.isnan(effective_modes_b) and n_rows_with_effective_modes > 0:
                    total_effective_modes += effective_modes_b * n_rows_with_effective_modes
                    n_effective_modes_rows += n_rows_with_effective_modes
                if not np.isnan(mode_min_pair_dist_b) and n_rows_with_mode_pairs > 0:
                    total_mode_min_pair_dist += mode_min_pair_dist_b * n_rows_with_mode_pairs
                    n_mode_min_pair_rows += n_rows_with_mode_pairs
                if not np.isnan(tail_resp_b) and n_rows_with_tail_resp > 0:
                    total_tail_responsibility += tail_resp_b * n_rows_with_tail_resp
                    n_tail_rows += n_rows_with_tail_resp
                if not np.isnan(gate_lambda_b) and n_rows_with_gate_lambda > 0:
                    total_gate_lambda += gate_lambda_b * n_rows_with_gate_lambda
                    n_gate_lambda_rows += n_rows_with_gate_lambda
                total_sigma += sigma_mean_b * b_size
                total_cold_start += cold_start_b
            else:
                total_spatial_cell += float(spatial_nll_cell.sum().detach())
                total_spatial_continuous += float(spatial_nll_continuous.sum().detach())
                total_expected_distance_ft += float(expected_dist.sum())
                total_sub_uniform += sub_uniform
            if not np.isnan(alpha_h):
                # Aggregate as (per-row mean) × (n rows = batch size).
                total_alpha_entropy += alpha_h * b_size
                n_alpha_rows += b_size
            if not np.isnan(beta_h) and n_beta_in_batch > 0:
                total_beta_entropy += beta_h * n_beta_in_batch
                n_beta_rows += n_beta_in_batch
            total_timing += float(timing_nll.sum().detach())
            total_count += float(count_nll_per_shot.sum().detach())
            total_reg += float((lambda_tilt * tilt_reg_per_shot).sum().detach())
            total_reg += float((lambda_defense * defense_reg_per_shot).sum().detach())
            # Defense diagnostics, when the decoder emitted defense_logits.
            if (
                is_mixture_mode
                and isinstance(spatial, ContinuousMixtureSpatial)
                and spatial.has_defense
                and mix_out.defense_logits is not None
                and mix_out.defense_cold_start is not None
            ):
                with torch.no_grad():
                    d_logits = mix_out.defense_logits
                    total_def_score_mean += float(d_logits.mean(dim=-1).sum().detach())
                    total_def_score_abs_mean += float(d_logits.abs().mean(dim=-1).sum().detach())
                    total_def_score_max_abs += float(d_logits.abs().amax(dim=-1).sum().detach())
                    total_def_cold_start_rows += int(mix_out.defense_cold_start.sum().item())
                    n_def_rows += b_size
            # Matchup diagnostics, when the decoder emitted matchup_logits.
            if (
                is_mixture_mode
                and isinstance(spatial, ContinuousMixtureSpatial)
                and spatial.has_matchup
                and mix_out.matchup_logits is not None
                and mix_out.matchup_n_eff is not None
            ):
                with torch.no_grad():
                    m_logits = mix_out.matchup_logits
                    n_eff_row = mix_out.matchup_n_eff
                    total_matchup_effect_abs_mean += float(
                        m_logits.abs().mean(dim=-1).sum().detach()
                    )
                    total_matchup_effect_max_abs += float(
                        m_logits.abs().amax(dim=-1).sum().detach()
                    )
                    total_matchup_n_eff_mean += float(n_eff_row.sum().detach())
                    matchup_n_eff_pool.extend(n_eff_row.detach().cpu().tolist())
                    n_match_rows += b_size

    if n_shots_seen == 0:
        raise RuntimeError("epoch processed zero shots — empty dataset?")
    return _EpochLosses(
        total=(total_spatial + total_timing + total_count + total_reg) / n_shots_seen,
        spatial=total_spatial / n_shots_seen,
        spatial_cell=((total_spatial_cell / n_shots_seen) if not is_mixture_mode else float("nan")),
        spatial_continuous=(
            (total_spatial_continuous / n_shots_seen) if not is_mixture_mode else float("nan")
        ),
        spatial_mix_nll=(
            (total_spatial_mix_nll / n_shots_seen) if is_mixture_mode else float("nan")
        ),
        expected_distance_ft=(
            (total_expected_distance_ft / n_shots_seen) if not is_mixture_mode else float("nan")
        ),
        alpha_entropy=(total_alpha_entropy / n_alpha_rows) if n_alpha_rows else float("nan"),
        beta_entropy=(total_beta_entropy / n_beta_rows) if n_beta_rows else float("nan"),
        frac_sub_uniform=(
            (total_sub_uniform / n_shots_seen) if not is_mixture_mode else float("nan")
        ),
        support_entropy=(
            (total_support_entropy / n_shots_seen) if is_mixture_mode else float("nan")
        ),
        effective_support_size=(
            (total_eff_support / n_shots_seen) if is_mixture_mode else float("nan")
        ),
        expected_support_distance_ft=(
            (total_expected_support_dist / n_shots_seen) if is_mixture_mode else float("nan")
        ),
        min_dist_to_support_ft=(
            (total_min_dist_to_support / n_shots_seen) if is_mixture_mode else float("nan")
        ),
        mode_to_support_dist_ft=(
            (total_mode_to_support_dist / n_mode_to_support_rows)
            if n_mode_to_support_rows > 0
            else float("nan")
        ),
        effective_modes=(
            (total_effective_modes / n_effective_modes_rows)
            if n_effective_modes_rows > 0
            else float("nan")
        ),
        mode_min_pair_dist_ft=(
            (total_mode_min_pair_dist / n_mode_min_pair_rows)
            if n_mode_min_pair_rows > 0
            else float("nan")
        ),
        tail_responsibility=(
            (total_tail_responsibility / n_tail_rows) if n_tail_rows > 0 else float("nan")
        ),
        gate_lambda_mean=(
            (total_gate_lambda / n_gate_lambda_rows) if n_gate_lambda_rows > 0 else float("nan")
        ),
        sigma_mean=((total_sigma / n_shots_seen) if is_mixture_mode else float("nan")),
        frac_cold_start=((total_cold_start / n_shots_seen) if is_mixture_mode else float("nan")),
        timing=total_timing / n_shots_seen,
        count=total_count / n_shots_seen,
        reg=total_reg / n_shots_seen,
        def_score_mean=((total_def_score_mean / n_def_rows) if n_def_rows > 0 else float("nan")),
        def_score_abs_mean=(
            (total_def_score_abs_mean / n_def_rows) if n_def_rows > 0 else float("nan")
        ),
        def_score_max_abs=(
            (total_def_score_max_abs / n_def_rows) if n_def_rows > 0 else float("nan")
        ),
        def_cold_start_fraction=(
            (total_def_cold_start_rows / n_def_rows) if n_def_rows > 0 else float("nan")
        ),
        def_beta=(
            float(spatial.defensive_field.beta_D.detach().item())
            if (
                isinstance(spatial, ContinuousMixtureSpatial)
                and spatial.has_defense
                and spatial.defensive_field is not None
            )
            else float("nan")
        ),
        matchup_effect_abs_mean=(
            (total_matchup_effect_abs_mean / n_match_rows) if n_match_rows > 0 else float("nan")
        ),
        matchup_effect_max_abs=(
            (total_matchup_effect_max_abs / n_match_rows) if n_match_rows > 0 else float("nan")
        ),
        matchup_beta=(
            float(spatial.matchup_field.beta_match.detach().item())
            if (
                isinstance(spatial, ContinuousMixtureSpatial)
                and spatial.has_matchup
                and spatial.matchup_field is not None
            )
            else float("nan")
        ),
        matchup_n_eff_mean=(
            (total_matchup_n_eff_mean / n_match_rows) if n_match_rows > 0 else float("nan")
        ),
        matchup_n_eff_p10=(
            float(np.quantile(matchup_n_eff_pool, 0.10)) if matchup_n_eff_pool else float("nan")
        ),
        matchup_n_eff_p50=(
            float(np.quantile(matchup_n_eff_pool, 0.50)) if matchup_n_eff_pool else float("nan")
        ),
        matchup_n_eff_p90=(
            float(np.quantile(matchup_n_eff_pool, 0.90)) if matchup_n_eff_pool else float("nan")
        ),
    )


def _safe_entropy(p: Tensor, *, eps: float = 1e-20) -> Tensor:
    """Entropy ``-Σ p log p`` in nats over the last dimension.

    Each row of ``p`` is either a probability vector or all zeros (the
    collaborative β is zero for analogues with no causal history); zero
    entries contribute nothing, so all-zero rows have entropy 0.
    """
    return -(p.clamp_min(eps) * p.clamp_min(eps).log() * (p > 0).to(p.dtype)).sum(dim=-1)


def train_gibbs(
    *,
    offensive_prior: nn.Module,
    count_head: NegBinCountHead,
    timing_head: TimingSoftmaxHead,
    context_mlp: ContextMLP,
    train_set: GibbsShotDataset,
    grid: CourtGrid,
    val_set: GibbsShotDataset | None = None,
    defensive_field: AdaptiveDefensiveField | None = None,
    residual_encoder: ContextResidualEncoder | None = None,
    tilt_decoder: LowRankTiltDecoder | None = None,
    location_embedding: object | None = None,
    pooling_gate: PoolingGate | None = None,
    defensive_field_cellfree: (
        ContinuousAdaptiveDefensiveField | ZoneReweightingDefense | None
    ) = None,
    defensive_cache: DefensiveRetrievalCache | None = None,
    defensive_features: DefenseFeatures | None = None,
    matchup_field: MatchupReweightingDefense | None = None,
    matchup_features: MatchupFeatures | None = None,
    bandwidth_field: ZoneSourceBandwidth | None = None,
    anisotropic_kernel: RadialTangentZoneKernel | FullCovarianceZoneKernel | None = None,
    khat_log1p_mean: float | None = None,
    khat_log1p_std: float | None = None,
    within_game_gru: nn.Module | None = None,
    n_epochs: int = 30,
    batch_size: int = 512,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.0,
    lambda_tilt: float = 1e-3,
    lambda_spatial: float = 1.0,
    lambda_timing: float = 1.0,
    lambda_count: float = 1.0,
    lambda_defense: float = 0.0,
    count_loss_normalization: CountLossNormalization = "per_game",
    count_checkpoint_path: str | Path | None = None,
    freeze_count: bool = False,
    freeze_context_mlp: bool = False,
    count_lr: float | None = None,
    spatial_likelihood: SpatialLikelihood = "continuous_cell",
    obs_kernel_tau: float = 1.0,
    obs_kernel_normalize: bool = True,
    mode_mixture_n_court_modes: int = 6,
    mode_mixture_query_dim: int = 32,
    mode_mixture_sigma_ft: float = 3.0,
    mode_mixture_context_correction: bool = True,
    mode_mixture_lambda_omega: float = 0.0,
    mode_mixture_tail_weight: float = 0.0,
    mode_mixture_tail_sigma_ft: float = 1.0,
    mode_mixture_extractor_kind: str = "soft_kmeans",
    mode_mixture_kernel_bandwidth_ft: float = 5.0,
    mode_mixture_n_iterations: int = 2,
    snapshot_callback: (Callable[[int, dict[str, dict[str, Tensor]]], None] | None) = None,
    device: str | torch.device = "cpu",
    shuffle: bool = True,
    progress: bool = False,
    restore_best_val: bool = True,
    nan_check: bool = True,
    max_batches: int | None = None,
    stratified_epsilon: float = 1.0,
    causal_zone_bias: nn.Module | None = None,
    mode_router: nn.Module | None = None,
    court_bounds: tuple[float, float, float, float] | None = None,
    out_spatial: list[nn.Module] | None = None,
    **legacy_kwargs: object,
) -> GibbsTrainHistory:
    """Jointly train the spatial, timing and count factors.

    Builds the spatial decoder selected by ``spatial_likelihood`` around
    ``offensive_prior`` and the optional components, then optimizes all
    trainable modules with Adam on the objective described in
    :mod:`shotcloud.training.train_gibbs`. Modules are trained in place.

    Parameters
    ----------
    offensive_prior : CollaborativeKDE or RetrievalCollaborativeKDE or AdaptiveOffensivePrior
        Support backend of the spatial factor. The cell-free likelihoods
        require :class:`~shotcloud.models.collaborative_kde.CollaborativeKDE`
        or
        :class:`~shotcloud.models.retrieval_collaborative_kde.RetrievalCollaborativeKDE`
        (``"mode_mixture"`` only the former); the grid likelihoods require
        :class:`~shotcloud.legacy_pivot.adaptive_prior.AdaptiveOffensivePrior`
        or ``CollaborativeKDE``.
    count_head : NegBinCountHead
        Negative-binomial count head on the learned context.
    timing_head : TimingSoftmaxHead
        48-bin softmax timing head on the learned context.
    context_mlp : ContextMLP
        :math:`f_{\\mathrm{ctx}}`, shared by all factors. Its residual
        zero-initialization makes it the identity at the start of
        training.
    train_set : GibbsShotDataset
        Training shots.
    grid : CourtGrid
        Court grid; its cell centers are used by the grid likelihoods.
    val_set : GibbsShotDataset, optional
        Validation shots, evaluated after every epoch.
    defensive_field : AdaptiveDefensiveField, optional
        Grid-path opponent reweighting
        (:class:`~shotcloud.legacy_pivot.adaptive_defensive.AdaptiveDefensiveField`).
        Requires datasets with an opponent vocabulary; not supported by the
        cell-free likelihoods.
    residual_encoder : ContextResidualEncoder, optional
        Context encoder :math:`u_\\theta` of the residual tilt. Pair with
        ``tilt_decoder`` on the grid path and with ``location_embedding``
        on the cell-free paths.
    tilt_decoder : LowRankTiltDecoder, optional
        Grid-path cell embedding :math:`v_c` of the residual tilt.
    location_embedding : LocationEmbedding, optional
        Cell-free coordinate embedding :math:`\\psi(s)` of the residual
        tilt.
    pooling_gate : PoolingGate, optional
        Own/pooled gate, giving the density
        :math:`\\lambda f_{\\mathrm{own}} + (1 - \\lambda) f_{\\mathrm{pooled}}`.
        Only valid with ``"continuous_mixture"``.
    defensive_field_cellfree : ZoneReweightingDefense or ContinuousAdaptiveDefensiveField, optional
        Cell-free opponent reweighting :math:`D(s_m)`, used with
        ``"continuous_mixture"`` only. ``ZoneReweightingDefense`` requires
        ``defensive_features`` and no cache;
        ``ContinuousAdaptiveDefensiveField`` (an alternative evaluated as
        an ablation) requires both ``defensive_cache`` and
        ``defensive_features``. Requires datasets with an opponent
        vocabulary.
    defensive_cache : DefensiveRetrievalCache, optional
        Causal allowed-shot cache for ``ContinuousAdaptiveDefensiveField``.
    defensive_features : DefenseFeatures, optional
        Causal per-(opponent, snapshot) defensive features.
    matchup_field, matchup_features : MatchupReweightingDefense, MatchupFeatures, optional
        Player-versus-opponent zone reweighting and its causal features.
        Both or neither; ``"continuous_mixture"`` only. Composes
        additively with the opponent reweighting.
    bandwidth_field : ZoneSourceBandwidth, optional
        Per-(source, zone) kernel bandwidth replacing the backend's
        per-row bandwidth. ``"continuous_mixture"`` only; mutually
        exclusive with ``anisotropic_kernel``.
    anisotropic_kernel : RadialTangentZoneKernel or FullCovarianceZoneKernel, optional
        Per-zone anisotropic kernel replacing the isotropic kernel.
        ``"continuous_mixture"`` only.
    khat_log1p_mean, khat_log1p_std : float, optional
        Standardization of the predicted count fed to the residual,
        :math:`(\\log(1 + \\hat K) - \\mu) / \\sigma`. Used only when the
        residual encoder consumes :math:`\\hat K` (its ``usage_dim`` is
        ``USAGE_KHAT_DIM`` or 1). Statistics computed from observed
        training counts keep the transform independent of the count
        head's calibration.
    within_game_gru : nn.Module, optional
        Recurrent encoder over the earlier shots of the same game, added
        to the residual context (an alternative evaluated as an ablation).
        Used only with ``residual_encoder`` and ``"continuous_mixture"``.
    n_epochs : int, default 30
        Number of epochs.
    batch_size : int, default 512
        Shots per minibatch.
    learning_rate : float, default 5e-4
        Adam learning rate.
    weight_decay : float, default 0.0
        Adam weight decay.
    lambda_tilt : float, default 1e-3
        Weight of the mean squared residual tilt; no effect without a
        residual.
    lambda_spatial, lambda_timing, lambda_count : float, default 1.0
        Weights of the spatial, timing and count NLL terms.
    lambda_defense : float, default 0.0
        Weight of the mean squared opponent reweighting.
    count_loss_normalization : {"per_game", "per_shot"}, default "per_game"
        How the count NLL enters the loss; see
        :data:`CountLossNormalization`.
    count_checkpoint_path : str or Path, optional
        Checkpoint written by ``scripts/train_count_head.py``. Its
        ``"count_head"`` state (and ``"context_mlp"`` state, if present) is
        loaded before training.
    freeze_count : bool, default False
        Disable gradients of the count head. A residual that consumes
        :math:`\\hat K` always receives it detached; freezing additionally
        keeps the count head itself fixed.
    freeze_context_mlp : bool, default False
        Disable gradients of ``context_mlp`` as well. Freezing the count
        head alone does not fix its predictions, because spatial and
        timing gradients still update the shared :math:`f_{\\mathrm{ctx}}`;
        freezing it also fixes the context representation of every
        factor.
    count_lr : float, optional
        Separate learning rate for the count head's parameter group.
        Ignored when ``freeze_count`` is set.
    spatial_likelihood : {"continuous_cell", "cell", "continuous_mixture", "mode_mixture"}
        Spatial factor and its likelihood; see
        :data:`SpatialLikelihood`. Default ``"continuous_cell"``.
    obs_kernel_tau : float, default 1.0
        Observation-kernel bandwidth in feet of the continuous-coordinate
        grid loss.
    obs_kernel_normalize : bool, default True
        Normalize the observation kernel over cells.
    mode_mixture_n_court_modes, mode_mixture_query_dim, mode_mixture_sigma_ft
        Number of modes, learned-query dimension and mode bandwidth in
        feet of the mode-mixture decoder.
    mode_mixture_context_correction : bool, default True
        Enable the context bias on the mode logits.
    mode_mixture_lambda_omega : float, default 0.0
        Weight of the support-mass bias in the learned-query extractor.
    mode_mixture_tail_weight, mode_mixture_tail_sigma_ft : float
        Mixing weight and bandwidth of the support-KDE tail component.
    mode_mixture_extractor_kind : {"soft_kmeans", "learned_query"}
        Mode-extraction operator: weighted farthest-point seeding plus
        mean-shift over the row's support, or learned queries shared
        across rows.
    mode_mixture_kernel_bandwidth_ft, mode_mixture_n_iterations
        Mean-shift bandwidth in feet and iteration count of the soft
        k-means extractor.
    snapshot_callback : callable, optional
        Called as ``snapshot_callback(epoch, module_states)`` after every
        epoch, with detached copies of each trained module's
        ``state_dict()`` keyed by module name.
    device : str or torch.device, default "cpu"
        Device to train on.
    shuffle : bool, default True
        Shuffle training shots each epoch.
    progress : bool, default False
        Print one line of losses and diagnostics per epoch.
    restore_best_val : bool, default True
        With ``val_set``, keep the parameters of the epoch with the lowest
        validation spatial loss and restore them before returning.
    nan_check : bool, default True
        Raise on non-finite activations, losses or gradients.
    max_batches : int, optional
        Stop each epoch after this many batches.
    stratified_epsilon : float, default 1.0
        Cross-zone kernel attenuation of the AC-KDE decoder; 1.0
        disables it.
    causal_zone_bias : CausalZoneBias, optional
        Zone-pair bias on the support logits of the AC-KDE decoder.
    mode_router : ModeRouter, optional
        Builds
        :class:`~shotcloud.models.mode_routed_spatial.ModeRoutedContinuousMixtureSpatial`
        instead of ``ContinuousMixtureSpatial`` (an alternative evaluated
        as an ablation). ``pooling_gate``, ``causal_zone_bias`` and
        ``court_bounds`` are then not used.
    court_bounds : tuple of float, optional
        ``(x_min, x_max, y_min, y_max)`` in feet. Renormalizes each
        isotropic kernel of the AC-KDE decoder to the court rectangle.
    out_spatial : list, optional
        If given, the constructed spatial decoder is appended to it, so
        callers can save decoder-owned state that is not among the
        modules passed in.
    **legacy_kwargs
        Accepts only ``spatial_loss`` in ``{"cell", "continuous"}``, an
        alias for ``spatial_likelihood`` ``"cell"`` / ``"continuous_cell"``.

    Returns
    -------
    GibbsTrainHistory
        Per-epoch losses and diagnostics, plus the final count calibration
        when ``lambda_count > 0``.

    Raises
    ------
    ValueError
        On invalid option values or inconsistently paired components.
    TypeError
        On an unsupported ``offensive_prior`` type or unknown keyword.
    NotImplementedError
        On component combinations a spatial likelihood does not support.
    """
    # ``spatial_loss`` alias: "continuous" maps to "continuous_cell".
    if "spatial_loss" in legacy_kwargs:
        legacy_val = str(legacy_kwargs.pop("spatial_loss"))
        if legacy_val == "continuous":
            spatial_likelihood = "continuous_cell"
        elif legacy_val == "cell":
            spatial_likelihood = "cell"
        else:
            raise ValueError(
                f"legacy spatial_loss must be 'continuous' or 'cell'; got {legacy_val!r}"
            )
    if legacy_kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(legacy_kwargs)}")

    if defensive_field is not None and not train_set.has_opponents:
        raise ValueError(
            "defensive_field requires train_set to carry an opp_vocab "
            "(pass opp_vocab=... when constructing GibbsShotDataset)"
        )
    if defensive_field is not None and val_set is not None and not val_set.has_opponents:
        raise ValueError("defensive_field requires val_set to carry an opp_vocab")
    if spatial_likelihood not in (
        "cell",
        "continuous_cell",
        "continuous_mixture",
        "mode_mixture",
    ):
        raise ValueError(
            f"spatial_likelihood must be 'cell', 'continuous_cell', "
            f"'continuous_mixture', or 'mode_mixture'; got {spatial_likelihood!r}"
        )
    if obs_kernel_tau <= 0:
        raise ValueError(f"obs_kernel_tau must be > 0; got {obs_kernel_tau}")
    # Two-valued name used by _epoch to pick the optimized grid loss.
    spatial_loss: str = "continuous" if spatial_likelihood == "continuous_cell" else "cell"
    dev = torch.device(device)

    # Cell centers in flat image-layout order (c = iy*nx + ix), built once
    # as an (n_cells, 2) tensor and reused by the grid losses.
    cx = np.tile(grid.xcenters, grid.ny)
    cy = np.repeat(grid.ycenters, grid.nx)
    cell_centers = torch.from_numpy(np.stack([cx, cy], axis=1).astype(np.float32)).to(dev)
    if not isinstance(
        offensive_prior,
        AdaptiveOffensivePrior | CollaborativeKDE | RetrievalCollaborativeKDE,
    ):
        raise TypeError(
            f"offensive_prior must be AdaptiveOffensivePrior, CollaborativeKDE, or "
            f"RetrievalCollaborativeKDE; got {type(offensive_prior).__name__}"
        )

    if pooling_gate is not None and spatial_likelihood != "continuous_mixture":
        raise ValueError(
            f"pooling_gate is only supported with spatial_likelihood="
            f"'continuous_mixture'; got {spatial_likelihood!r}."
        )

    spatial: ConditionalGibbsDecoder | ContinuousMixtureSpatial | CollaborativeModeMixtureSpatial
    if spatial_likelihood in ("continuous_mixture", "mode_mixture"):
        if not isinstance(offensive_prior, CollaborativeKDE | RetrievalCollaborativeKDE):
            raise TypeError(
                f"spatial_likelihood={spatial_likelihood!r} requires offensive_prior "
                "to be a CollaborativeKDE or RetrievalCollaborativeKDE (the "
                "grid-based AdaptiveOffensivePrior has no support-set forward path)."
            )
        if spatial_likelihood == "mode_mixture" and isinstance(
            offensive_prior, RetrievalCollaborativeKDE
        ):
            raise NotImplementedError(
                "RetrievalCollaborativeKDE + mode_mixture is not yet supported; "
                "use spatial_likelihood='continuous_mixture' with the retrieval backend."
            )
        if defensive_field is not None:
            raise NotImplementedError(
                "The grid-side `defensive_field` (AdaptiveDefensiveField) is not "
                "applicable to the cell-free paths. Use the cell-free triple "
                "`defensive_field_cellfree` + `defensive_cache` + "
                "`defensive_features` instead, or switch to "
                "spatial_likelihood='continuous_cell' for the grid path."
            )
        if (residual_encoder is None) != (location_embedding is None):
            raise ValueError(
                f"spatial_likelihood={spatial_likelihood!r}: residual_encoder and "
                "location_embedding must both be provided or both None."
            )
        # Cell-free defense validation, mirroring ContinuousMixtureSpatial:
        # * ContinuousAdaptiveDefensiveField requires field, cache and
        #   features.
        # * ZoneReweightingDefense requires field and features; no cache.
        # * All three None means no defense.
        is_zone_lite = isinstance(defensive_field_cellfree, ZoneReweightingDefense)
        if defensive_field_cellfree is None:
            if defensive_cache is not None or defensive_features is not None:
                raise ValueError(
                    "defensive_cache and defensive_features must be None when "
                    "defensive_field_cellfree is None; got "
                    f"cache={defensive_cache is not None}, "
                    f"features={defensive_features is not None}"
                )
            n_defense = 0
        elif is_zone_lite:
            if defensive_cache is not None:
                raise ValueError(
                    "ZoneReweightingDefense (D-lite) does not consume a defensive "
                    "cache; pass defensive_cache=None."
                )
            if defensive_features is None:
                raise ValueError(
                    "ZoneReweightingDefense (D-lite) requires defensive_features "
                    "(it consumes the centered-zone block); got None."
                )
            n_defense = 3
        else:
            if defensive_cache is None or defensive_features is None:
                raise ValueError(
                    "ContinuousAdaptiveDefensiveField (D-field) requires both "
                    "defensive_cache and defensive_features; got "
                    f"cache={defensive_cache is not None}, "
                    f"features={defensive_features is not None}"
                )
            n_defense = 3
        # Matchup channel: field and features together, continuous_mixture
        # only; independent of the opponent reweighting.
        if (matchup_field is None) != (matchup_features is None):
            raise ValueError(
                "matchup_field and matchup_features must both be provided or both None; "
                f"got field={matchup_field is not None}, "
                f"features={matchup_features is not None}"
            )
        if matchup_field is not None and spatial_likelihood != "continuous_mixture":
            raise NotImplementedError(
                f"D-matchup is currently only supported with "
                f"spatial_likelihood='continuous_mixture'; got {spatial_likelihood!r}."
            )
        if matchup_field is not None and not train_set.has_opponents:
            raise ValueError(
                "D-matchup requires train_set to carry an opp_vocab and an opp_idx "
                "tensor (opponent identity is the row's defense d)."
            )
        if matchup_field is not None and val_set is not None and not val_set.has_opponents:
            raise ValueError("D-matchup requires val_set to carry an opp_vocab.")
        if n_defense == 3 and spatial_likelihood != "continuous_mixture":
            raise NotImplementedError(
                f"Cell-free defense is currently only supported with "
                f"spatial_likelihood='continuous_mixture'; got {spatial_likelihood!r}. "
                "(mode_mixture + defense is a future PR; the K-mode extractor "
                "would need a separate adaptation.)"
            )
        if n_defense == 3 and not train_set.has_opponents:
            raise ValueError(
                "Cell-free defense triple requires train_set to carry an opp_vocab "
                "and an opp_idx tensor."
            )
        if n_defense == 3 and val_set is not None and not val_set.has_opponents:
            raise ValueError("Cell-free defense triple requires val_set to carry an opp_vocab.")
        if spatial_likelihood == "continuous_mixture":
            # The decoder needs ``count_head`` when the residual encoder
            # consumes K̂: ``usage_dim`` is USAGE_KHAT_DIM (usage features
            # plus K̂) or 1 (K̂ only). The count head stays in the
            # trainer's own module dict, so it is optimized under the
            # count loss; the decoder only reads its detached output.
            from shotcloud.features.usage_features import USAGE_KHAT_DIM

            count_head_for_spatial: nn.Module | None = None
            if residual_encoder is not None and residual_encoder.usage_dim in (
                USAGE_KHAT_DIM,
                1,
            ):
                count_head_for_spatial = count_head
            # With a ``mode_router``, build the mode-routed subclass. Mode
            # routing replaces the pooling gate and the zone-pair bias, so
            # the subclass rejects both; they are passed as None here.
            if mode_router is not None:
                spatial = ModeRoutedContinuousMixtureSpatial(
                    offensive_prior=offensive_prior,
                    residual_encoder=residual_encoder,
                    location_embedding=location_embedding,
                    pooling_gate=None,
                    defensive_field=defensive_field_cellfree,
                    defensive_cache=defensive_cache,
                    defensive_features=defensive_features,
                    matchup_field=matchup_field,
                    matchup_features=matchup_features,
                    bandwidth_field=bandwidth_field,
                    anisotropic_kernel=anisotropic_kernel,
                    count_head=count_head_for_spatial,
                    khat_log1p_mean=(
                        khat_log1p_mean if count_head_for_spatial is not None else None
                    ),
                    khat_log1p_std=(khat_log1p_std if count_head_for_spatial is not None else None),
                    within_game_gru=(within_game_gru if residual_encoder is not None else None),
                    stratified_epsilon=stratified_epsilon,
                    causal_zone_bias=None,
                    mode_router=mode_router,
                ).to(dev)
            else:
                spatial = ContinuousMixtureSpatial(
                    offensive_prior=offensive_prior,
                    residual_encoder=residual_encoder,
                    location_embedding=location_embedding,  # type: ignore[arg-type]
                    pooling_gate=pooling_gate,
                    defensive_field=defensive_field_cellfree,
                    defensive_cache=defensive_cache,
                    defensive_features=defensive_features,
                    matchup_field=matchup_field,
                    matchup_features=matchup_features,
                    bandwidth_field=bandwidth_field,
                    anisotropic_kernel=anisotropic_kernel,
                    count_head=count_head_for_spatial,
                    khat_log1p_mean=(
                        khat_log1p_mean if count_head_for_spatial is not None else None
                    ),
                    khat_log1p_std=(khat_log1p_std if count_head_for_spatial is not None else None),
                    within_game_gru=(within_game_gru if residual_encoder is not None else None),
                    stratified_epsilon=stratified_epsilon,
                    causal_zone_bias=causal_zone_bias,  # type: ignore[arg-type]
                    court_bounds=court_bounds,
                ).to(dev)
        else:
            # mode_mixture rejects RetrievalCollaborativeKDE above; narrow
            # the type for the CollaborativeModeMixtureSpatial constructor.
            assert isinstance(offensive_prior, CollaborativeKDE)
            spatial = CollaborativeModeMixtureSpatial(
                offensive_prior=offensive_prior,
                residual_encoder=residual_encoder,
                residual_location_embedding=location_embedding,  # type: ignore[arg-type]
                n_court_modes=mode_mixture_n_court_modes,
                mode_query_dim=mode_mixture_query_dim,
                mode_sigma_ft=mode_mixture_sigma_ft,
                use_context_correction=mode_mixture_context_correction,
                lambda_omega=mode_mixture_lambda_omega,
                tail_weight=mode_mixture_tail_weight,
                tail_sigma_ft=mode_mixture_tail_sigma_ft,
                extractor_kind=mode_mixture_extractor_kind,
                mode_kernel_bandwidth_ft=mode_mixture_kernel_bandwidth_ft,
                mode_n_iterations=mode_mixture_n_iterations,
            ).to(dev)
    else:
        # Grid paths (cell / continuous_cell) require a grid-capable
        # offensive prior. RetrievalCollaborativeKDE has no grid forward.
        if isinstance(offensive_prior, RetrievalCollaborativeKDE):
            raise NotImplementedError(
                f"RetrievalCollaborativeKDE has no grid forward; "
                f"spatial_likelihood={spatial_likelihood!r} requires either "
                "AdaptiveOffensivePrior or CollaborativeKDE."
            )
        spatial = ConditionalGibbsDecoder(
            offensive_prior,
            defensive_field=defensive_field,
            residual_encoder=residual_encoder,
            tilt_decoder=tilt_decoder,
        ).to(dev)

    modules = _collect_modules_for_spatial(
        spatial=spatial, count_head=count_head, timing_head=timing_head, context_mlp=context_mlp
    )
    for m in modules.values():
        m.to(dev)

    # Optional pretrained count head (from scripts/train_count_head.py),
    # optionally frozen together with the shared context MLP.
    if count_checkpoint_path is not None:
        ckpt = torch.load(count_checkpoint_path, weights_only=False, map_location=dev)
        if "count_head" not in ckpt:
            raise ValueError(
                f"count_checkpoint_path={count_checkpoint_path} is missing the "
                f"'count_head' key; expected output of train_count_head.py"
            )
        count_head.load_state_dict(ckpt["count_head"])
        if "context_mlp" in ckpt:
            context_mlp.load_state_dict(ckpt["context_mlp"])
    if freeze_count:
        for p in count_head.parameters():
            p.requires_grad = False
    if freeze_context_mlp:
        for p in context_mlp.parameters():
            p.requires_grad = False

    # A single Adam group, or a separate group for the count head at
    # ``count_lr`` (e.g. to fine-tune a pretrained count head slowly).
    if count_lr is None or freeze_count:
        params: list[Tensor] = [
            p for m in modules.values() for p in m.parameters() if p.requires_grad
        ]
        if not params:
            raise ValueError("no trainable parameters; check requires_grad on submodules")
        optimizer = Adam(params, lr=learning_rate, weight_decay=weight_decay)
    else:
        count_param_ids = {id(p) for p in count_head.parameters() if p.requires_grad}
        rest_params = [
            p
            for m in modules.values()
            for p in m.parameters()
            if p.requires_grad and id(p) not in count_param_ids
        ]
        count_params = [p for p in count_head.parameters() if p.requires_grad]
        if not rest_params and not count_params:
            raise ValueError("no trainable parameters; check requires_grad on submodules")
        optimizer = Adam(
            [
                {"params": rest_params, "lr": learning_rate},
                {"params": count_params, "lr": count_lr},
            ],
            weight_decay=weight_decay,
        )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = (
        DataLoader(val_set, batch_size=batch_size, shuffle=False) if val_set is not None else None
    )

    history = GibbsTrainHistory()
    best_val_spatial = float("inf")
    best_state: dict[str, dict[str, Tensor]] | None = None

    for epoch in range(1, n_epochs + 1):
        tr = _epoch(
            spatial=spatial,
            count_head=count_head,
            timing_head=timing_head,
            context_mlp=context_mlp,
            loader=train_loader,
            dataset=train_set,
            optimizer=optimizer,
            lambda_tilt=lambda_tilt,
            lambda_spatial=lambda_spatial,
            lambda_timing=lambda_timing,
            lambda_count=lambda_count,
            lambda_defense=lambda_defense,
            cell_centers=cell_centers,
            spatial_loss=spatial_loss,
            obs_kernel_tau=obs_kernel_tau,
            obs_kernel_normalize=obs_kernel_normalize,
            device=dev,
            count_loss_normalization=count_loss_normalization,
            nan_check=nan_check,
            max_batches=max_batches,
        )
        history.train_total.append(tr.total)
        history.train_spatial.append(tr.spatial)
        history.train_spatial_cell.append(tr.spatial_cell)
        history.train_spatial_continuous.append(tr.spatial_continuous)
        history.train_spatial_mix_nll.append(tr.spatial_mix_nll)
        history.train_expected_distance_ft.append(tr.expected_distance_ft)
        history.train_alpha_entropy.append(tr.alpha_entropy)
        history.train_beta_entropy.append(tr.beta_entropy)
        history.train_frac_sub_uniform.append(tr.frac_sub_uniform)
        history.train_support_entropy.append(tr.support_entropy)
        history.train_effective_support_size.append(tr.effective_support_size)
        history.train_expected_support_distance_ft.append(tr.expected_support_distance_ft)
        history.train_min_dist_to_support_ft.append(tr.min_dist_to_support_ft)
        history.train_mode_to_support_dist_ft.append(tr.mode_to_support_dist_ft)
        history.train_effective_modes.append(tr.effective_modes)
        history.train_mode_min_pair_dist_ft.append(tr.mode_min_pair_dist_ft)
        history.train_tail_responsibility.append(tr.tail_responsibility)
        history.train_gate_lambda_mean.append(tr.gate_lambda_mean)
        history.train_sigma_mean.append(tr.sigma_mean)
        history.train_frac_cold_start.append(tr.frac_cold_start)
        history.train_timing.append(tr.timing)
        history.train_count.append(tr.count)
        history.train_reg.append(tr.reg)
        history.train_def_score_mean.append(tr.def_score_mean)
        history.train_def_score_abs_mean.append(tr.def_score_abs_mean)
        history.train_def_score_max_abs.append(tr.def_score_max_abs)
        history.train_def_cold_start_fraction.append(tr.def_cold_start_fraction)
        history.train_def_beta.append(tr.def_beta)
        history.train_matchup_effect_abs_mean.append(tr.matchup_effect_abs_mean)
        history.train_matchup_effect_max_abs.append(tr.matchup_effect_max_abs)
        history.train_matchup_beta.append(tr.matchup_beta)
        history.train_matchup_n_eff_mean.append(tr.matchup_n_eff_mean)
        history.train_matchup_n_eff_p10.append(tr.matchup_n_eff_p10)
        history.train_matchup_n_eff_p50.append(tr.matchup_n_eff_p50)
        history.train_matchup_n_eff_p90.append(tr.matchup_n_eff_p90)

        if val_loader is not None and val_set is not None:
            va = _epoch(
                spatial=spatial,
                count_head=count_head,
                timing_head=timing_head,
                context_mlp=context_mlp,
                loader=val_loader,
                dataset=val_set,
                optimizer=None,
                lambda_tilt=lambda_tilt,
                lambda_spatial=lambda_spatial,
                lambda_timing=lambda_timing,
                lambda_count=lambda_count,
                lambda_defense=lambda_defense,
                cell_centers=cell_centers,
                spatial_loss=spatial_loss,
                obs_kernel_tau=obs_kernel_tau,
                obs_kernel_normalize=obs_kernel_normalize,
                device=dev,
                count_loss_normalization=count_loss_normalization,
                nan_check=nan_check,
                max_batches=max_batches,
            )
            history.val_total.append(va.total)
            history.val_spatial.append(va.spatial)
            history.val_spatial_cell.append(va.spatial_cell)
            history.val_spatial_continuous.append(va.spatial_continuous)
            history.val_spatial_mix_nll.append(va.spatial_mix_nll)
            history.val_expected_distance_ft.append(va.expected_distance_ft)
            history.val_alpha_entropy.append(va.alpha_entropy)
            history.val_beta_entropy.append(va.beta_entropy)
            history.val_frac_sub_uniform.append(va.frac_sub_uniform)
            history.val_support_entropy.append(va.support_entropy)
            history.val_effective_support_size.append(va.effective_support_size)
            history.val_expected_support_distance_ft.append(va.expected_support_distance_ft)
            history.val_min_dist_to_support_ft.append(va.min_dist_to_support_ft)
            history.val_mode_to_support_dist_ft.append(va.mode_to_support_dist_ft)
            history.val_effective_modes.append(va.effective_modes)
            history.val_mode_min_pair_dist_ft.append(va.mode_min_pair_dist_ft)
            history.val_tail_responsibility.append(va.tail_responsibility)
            history.val_gate_lambda_mean.append(va.gate_lambda_mean)
            history.val_sigma_mean.append(va.sigma_mean)
            history.val_frac_cold_start.append(va.frac_cold_start)
            history.val_timing.append(va.timing)
            history.val_count.append(va.count)
            history.val_def_score_mean.append(va.def_score_mean)
            history.val_def_score_abs_mean.append(va.def_score_abs_mean)
            history.val_def_score_max_abs.append(va.def_score_max_abs)
            history.val_def_cold_start_fraction.append(va.def_cold_start_fraction)
            history.val_def_beta.append(va.def_beta)
            history.val_matchup_effect_abs_mean.append(va.matchup_effect_abs_mean)
            history.val_matchup_effect_max_abs.append(va.matchup_effect_max_abs)
            history.val_matchup_beta.append(va.matchup_beta)
            history.val_matchup_n_eff_mean.append(va.matchup_n_eff_mean)
            history.val_matchup_n_eff_p10.append(va.matchup_n_eff_p10)
            history.val_matchup_n_eff_p50.append(va.matchup_n_eff_p50)
            history.val_matchup_n_eff_p90.append(va.matchup_n_eff_p90)
            if restore_best_val and va.spatial < best_val_spatial:
                best_val_spatial = va.spatial
                history.best_epoch = epoch
                best_state = {name: _module_snapshot(m) for name, m in modules.items()}

        # Fires every epoch (callers filter); the states are detached
        # clones, so later in-place updates do not alter them.
        if snapshot_callback is not None:
            snapshot_callback(epoch, {name: _module_snapshot(m) for name, m in modules.items()})

        if progress:
            if spatial_likelihood in ("continuous_mixture", "mode_mixture"):
                line = (
                    f"[gibbs] epoch {epoch:>3}/{n_epochs}  "
                    f"tr_mix_nll={tr.spatial_mix_nll:.4f}  "
                    f"tr_sup_dist={tr.expected_support_distance_ft:.3f}ft  "
                    f"tr_min_dist={tr.min_dist_to_support_ft:.3f}ft  "
                    f"tr_H_sup={tr.support_entropy:.3f}  "
                    f"tr_eff_M={tr.effective_support_size:.1f}  "
                    f"tr_σ={tr.sigma_mean:.3f}  "
                    f"tr_cold={tr.frac_cold_start:.3f}  "
                    f"tr_Hα={tr.alpha_entropy:.3f}  tr_Hβ={tr.beta_entropy:.3f}  "
                    f"tr_timing={tr.timing:.4f}  tr_count={tr.count:.4f}"
                )
                if spatial_likelihood == "mode_mixture":
                    line += (
                        f"  tr_mode→sup={tr.mode_to_support_dist_ft:.3f}ft"
                        f"  tr_eff_K={tr.effective_modes:.2f}"
                        f"  tr_min_pair={tr.mode_min_pair_dist_ft:.2f}ft"
                        f"  tr_γ_tail={tr.tail_responsibility:.3f}"
                    )
                if not np.isnan(tr.gate_lambda_mean):
                    line += f"  tr_λ={tr.gate_lambda_mean:.3f}"
                if val_set is not None:
                    line += (
                        f"  va_mix_nll={history.val_spatial_mix_nll[-1]:.4f}  "
                        f"va_sup_dist={history.val_expected_support_distance_ft[-1]:.3f}ft  "
                        f"va_min_dist={history.val_min_dist_to_support_ft[-1]:.3f}ft  "
                        f"va_H_sup={history.val_support_entropy[-1]:.3f}  "
                        f"va_eff_M={history.val_effective_support_size[-1]:.1f}  "
                        f"va_σ={history.val_sigma_mean[-1]:.3f}  "
                        f"va_cold={history.val_frac_cold_start[-1]:.3f}"
                    )
                    if spatial_likelihood == "mode_mixture":
                        line += (
                            f"  va_mode→sup={history.val_mode_to_support_dist_ft[-1]:.3f}ft"
                            f"  va_eff_K={history.val_effective_modes[-1]:.2f}"
                            f"  va_min_pair={history.val_mode_min_pair_dist_ft[-1]:.2f}ft"
                            f"  va_γ_tail={history.val_tail_responsibility[-1]:.3f}"
                        )
                    if not np.isnan(history.val_gate_lambda_mean[-1]):
                        line += f"  va_λ={history.val_gate_lambda_mean[-1]:.3f}"
            else:
                line = (
                    f"[gibbs] epoch {epoch:>3}/{n_epochs}  "
                    f"tr_total={tr.total:.4f}  tr_spatial={tr.spatial:.4f}  "
                    f"tr_sp_cell={tr.spatial_cell:.4f}  "
                    f"tr_sp_cont={tr.spatial_continuous:.4f}  "
                    f"tr_dist={tr.expected_distance_ft:.3f}ft  "
                    f"tr_subU={tr.frac_sub_uniform:.3f}  "
                    f"tr_Hα={tr.alpha_entropy:.3f}  tr_Hβ={tr.beta_entropy:.3f}  "
                    f"tr_timing={tr.timing:.4f}  tr_count={tr.count:.4f}"
                )
                if val_set is not None:
                    line += (
                        f"  va_total={history.val_total[-1]:.4f}  "
                        f"va_spatial={history.val_spatial[-1]:.4f}  "
                        f"va_sp_cell={history.val_spatial_cell[-1]:.4f}  "
                        f"va_sp_cont={history.val_spatial_continuous[-1]:.4f}  "
                        f"va_dist={history.val_expected_distance_ft[-1]:.3f}ft  "
                        f"va_subU={history.val_frac_sub_uniform[-1]:.3f}  "
                        f"va_Hα={history.val_alpha_entropy[-1]:.3f}  "
                        f"va_Hβ={history.val_beta_entropy[-1]:.3f}"
                    )
            print(line, flush=True)

    if val_set is not None and restore_best_val and best_state is not None:
        for name, m in modules.items():
            _restore_snapshot(m, best_state[name])

    # Count calibration on train and val, computed after best-val
    # restoration so it describes the returned model. Skipped when
    # lambda_count == 0. A failure is recorded as ``{"error": ...}``
    # rather than raised, so a diagnostic cannot discard a training run.
    if lambda_count > 0:
        from shotcloud.evaluation import compute_count_calibration

        try:
            train_per_game_x_raw = train_set.per_game.x_n_raw.to(dev)
            train_per_game_k = train_set.per_game.k_obs.to(dev)
            history.final_train_count_calibration = compute_count_calibration(
                count_head, context_mlp, train_per_game_x_raw, train_per_game_k, device=dev
            )
        except Exception as exc:
            history.final_train_count_calibration = {"error": repr(exc)}
        if val_set is not None:
            try:
                val_per_game_x_raw = val_set.per_game.x_n_raw.to(dev)
                val_per_game_k = val_set.per_game.k_obs.to(dev)
                history.final_val_count_calibration = compute_count_calibration(
                    count_head, context_mlp, val_per_game_x_raw, val_per_game_k, device=dev
                )
            except Exception as exc:
                history.final_val_count_calibration = {"error": repr(exc)}

    # Expose the constructed decoder so callers can save decoder-owned
    # state that is not among the modules passed in.
    if out_spatial is not None:
        out_spatial.append(spatial)

    # Deepcopy guards against caller-side mutation of the lists.
    return deepcopy(history)
