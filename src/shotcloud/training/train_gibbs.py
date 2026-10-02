"""Joint Gibbs trainer.

Trains the spatial, timing, and count factors jointly under the
shared learned context :math:`x_n = f_{\\mathrm{ctx}}(\\tilde x_n)`.
The spatial factor is :class:`ConditionalGibbsDecoder` composing the
offensive prior, the opponent reweighting field (when present), and
the residual tilt (when present).

Per-shot total loss::

    L_shot = -log p_Θ(c_obs | x_n)
             -log ρ_η(τ_bin | x_n)
             -log p_count(K_obs | x_n_game) / shots_in_game
             + λ_tilt * ||u^T v||²

The amortized count contribution sums to ``-log p_count`` per game
across the game's shots, so an epoch-aggregate is exactly the
per-game count log-likelihood without a separate count pass.
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

#: Literal type for the ``--spatial-likelihood`` switch.
#:
#:  - ``"cell"``: legacy categorical NLL on the observed cell.
#:  - ``"continuous_cell"``: continuous-coordinate NLL via a Gaussian
#:    observation kernel against the grid log-probs.
#:  - ``"continuous_mixture"``: cell-free continuous-mixture NLL of
#:    the collaborative support shots — no grid in the loss path.
#:  - ``"mode_mixture"``: per-player-game K-mode Gaussian mixture
#:    extracted from the attended support shots; the headline
#:    cell-free spatial decoder for the v1.2 paper.
SpatialLikelihood = str  # constrained at CLI boundary

# Legacy alias retained for back-compat with old callers.
SpatialLoss = SpatialLikelihood

#: How the count NLL enters the joint loss.
#:
#: - ``"per_game"`` (default after 2026-06-07 audit): the per-game count
#:   NLL is averaged across the unique games in each batch and added to
#:   the per-shot loss mean as ``loss = loss_per_shot.mean() +
#:   lambda_count * count_loss_per_game.mean()``. ``lambda_count = 1.0``
#:   then puts the count head at parity with the per-shot spatial NLL
#:   on a per-update basis — which is what every other entry in the
#:   joint loss already does. Use this for any new run; it is the
#:   normalization the paper's loss equation describes.
#: - ``"per_shot"`` (legacy): the per-game count NLL is amortized as
#:   ``-log p(K_g | x_g) / K_g`` per shot and mixed into ``loss_per_shot``.
#:   At ``lambda_count = 1.0`` the count term contributes roughly
#:   ``1/K̄ ≈ 4.5%`` of the per-shot spatial weight, which under-supervises
#:   the count head; this normalization is preserved here only to
#:   reproduce checkpoints trained before the audit.
CountLossNormalization = str  # constrained at CLI boundary: "per_game" | "per_shot"


@dataclass
class CountOnlyTrainHistory:
    """Per-epoch history for :func:`train_count_only`.

    ``train_count`` / ``val_count`` are the per-game count NLL averaged
    over the epoch's games. Calibration diagnostics
    (``train_mean_mu`` / ``val_mean_mu`` etc.) are populated by the
    final ``finalize_diagnostics`` epoch over the full train/val sets.
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
    """Standalone count-head training loop (no spatial decoder, no support).

    Iterates over per-game ``(x_n_raw, K_obs)`` rows from
    :class:`PerGameTable`, runs ``μ = softplus(g_η(f_ctx(x_n_raw)))`` and
    minimizes ``-log p(K | μ, κ)`` averaged per game.

    Intended for the *count pretraining* step of the
    ``pretrain → freeze → spatial`` workflow (CLAUDE.md Phase 1.2c). The
    saved ``count_head`` and ``context_mlp`` state can be loaded by the
    joint trainer via ``--count-checkpoint`` and held fixed with
    ``--freeze-count`` so the spatial decoder consumes a calibrated
    detached ``μ`` rather than a co-adapted latent score.

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
    val_per_game_x_raw, val_per_game_k : optional
        Validation per-game tensors. If both provided, val NLL is
        computed at the end of every epoch.

    Returns
    -------
    CountOnlyTrainHistory
        Per-epoch loss + calibration diagnostics.
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
    """Per-epoch history for :func:`train_presence_only` (paper §5.2,
    Phase 3.4 PR-P1)."""

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

    For each row $(p, g)$ in the table sorted by ``game_date`` the
    *target* is the row's 30-bin vector and the *priors* are the rows
    in the player's group with strictly earlier dates (and matching
    starter status when ``same_starter_only=True``).

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
    """Train :class:`~shotcloud.models.PresenceModel` on per-(game, player)
    on-court vectors. Per-bin BCE loss; targets are the row's bin
    vector, inputs are the player's strictly-prior games with matching
    starter status — leave-one-out causality by construction.
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

    ``train_timing`` / ``val_timing`` are the per-shot timing NLL
    averaged over the epoch's shots. The 48-bin softmax timing head
    is trained against per-shot ``tau_bin`` from
    :class:`GibbsShotDataset` (one bin per game minute).
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
    """Standalone timing-head training loop (no spatial decoder, no count).

    Iterates over per-shot ``(x_n_raw, tau_bin)`` rows from
    :class:`GibbsShotDataset` and minimizes per-shot
    ``-log ρ_η(τ | x_n)`` against the 48-bin softmax head's predicted
    distribution.

    Intended for the timing pretrain step of the timing-validation
    Phase (paper §5 / docs/log Phase 3): solo-train the timing head
    on per-shot timing NLL alone with the correct loss normalization,
    then freeze and consume as a separately-validated marked-PP
    component. Mirrors :func:`train_count_only` for the count
    pathway.

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
    val_x_n_raw, val_tau_bin : optional
        Validation per-shot tensors. If both provided, val timing
        NLL is computed at the end of every epoch.

    Returns
    -------
    TimingOnlyTrainHistory
        Per-epoch train/val timing NLL.
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
    """Per-epoch loss history for the joint trainer.

    ``train_spatial`` / ``val_spatial`` are the **training objective**
    spatial loss — equal to ``train_spatial_cell`` when
    ``spatial_loss="cell"`` and to ``train_spatial_continuous`` when
    ``spatial_loss="continuous"``. The other ``_cell`` /
    ``_continuous`` / ``_distance_ft`` series are always logged for
    cross-run comparability regardless of which loss was optimized
    against.
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
    #: PR-D3 cell-free defense diagnostics. NaN when defense is off.
    train_def_score_mean: list[float] = field(default_factory=list)
    train_def_score_abs_mean: list[float] = field(default_factory=list)
    train_def_score_max_abs: list[float] = field(default_factory=list)
    train_def_cold_start_fraction: list[float] = field(default_factory=list)
    train_def_beta: list[float] = field(default_factory=list)
    #: Tier-2a D-matchup diagnostics. NaN when the matchup channel
    #: is off. ``matchup_effect_abs_mean`` and ``matchup_effect_max_abs``
    #: track the per-row magnitudes of ``β_match · Δ̂``;
    #: ``matchup_beta`` is the scalar parameter; the ESS quantiles
    #: ``matchup_n_eff_p10/p50/p90`` describe the distribution of
    #: per-row peer-vs-opponent evidence and are the load-bearing
    #: signal for the "matchup only helps where evidence is strong"
    #: falsification test.
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
    #: Final-epoch count calibration on the train set, as produced by
    #: :func:`shotcloud.evaluation.compute_count_calibration`. Populated
    #: by :func:`train_gibbs` after the last epoch when
    #: ``lambda_count > 0``. ``None`` otherwise.
    final_train_count_calibration: dict[str, object] | None = None
    #: Final-epoch count calibration on the val set, same population
    #: rule as the train counterpart.
    final_val_count_calibration: dict[str, object] | None = None


@dataclass
class _EpochLosses:
    total: float
    spatial: float  # the optimization-target spatial loss
    # Always-computed cross-mode metrics. In ``continuous_mixture`` mode
    # the grid-derived series are NaN (no grid log_probs are computed).
    spatial_cell: float  # exact-cell NLL (grid modes only)
    spatial_continuous: float  # continuous-coord NLL (grid modes only)
    spatial_mix_nll: float  # cell-free mixture NLL (continuous_mixture mode only)
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
    #: Tests "modes live among the support" — soft-k-means guarantees
    #: small values by construction; large values flag the
    #: global-anchor / drift pathology observed with the learned-query
    #: extractor. NaN in continuous_mixture / cell / continuous_cell.
    mode_to_support_dist_ft: float
    #: mode_mixture only: mean ``exp(H(π))`` across rows, where π is
    #: the K-mode softmax. ``≈ K`` means modes are used uniformly;
    #: ``≈ 1`` means one mode dominates per row (collapse).
    effective_modes: float
    #: mode_mixture only: mean over rows of the min pairwise distance
    #: between mode centers. Collapse indicator — pairs converging on
    #: the same point drive this toward 0.
    mode_min_pair_dist_ft: float
    #: mode_mixture + tail_weight > 0: mean posterior responsibility
    #: of the support-tail component. Large values mean the K-mode
    #: mixture is undercovering and the tail is doing the explanatory
    #: work — defeats the scientific story of compact per-row modes.
    tail_responsibility: float
    #: continuous_mixture + pooling gate: mean own-history mixing
    #: weight λ over non-cold-start rows. Should rise with own-history
    #: depth; a flat value flags the constant-pooling pathology.
    gate_lambda_mean: float
    sigma_mean: float
    frac_cold_start: float  # fraction of rows with no valid causal support
    timing: float
    count: float
    reg: float
    #: PR-D3 cell-free defense diagnostics. ``NaN`` when defense is
    #: off or the spatial mode doesn't carry a defensive field.
    #: ``def_score_mean`` is the per-row mean of D_Δ across M_off
    #: (should be ≈ 0 because the field is per-row centered).
    #: ``def_score_abs_mean`` and ``def_score_max_abs`` are the
    #: typical and worst-case per-row magnitudes — track them to
    #: watch for runaway β_D. ``def_cold_start_fraction`` is the
    #: fraction of rows whose defensive cache cell was empty.
    #: ``def_beta`` is the scalar ``β_D`` (read off the field after
    #: each epoch).
    def_score_mean: float
    def_score_abs_mean: float
    def_score_max_abs: float
    def_cold_start_fraction: float
    def_beta: float
    #: Tier-2a D-matchup diagnostics. NaN when the matchup channel
    #: is off. ``matchup_effect_abs_mean`` and ``matchup_effect_max_abs``
    #: track per-row ``β_match · Δ̂_{p,d,z(s_m)}`` magnitudes;
    #: ``matchup_beta`` is the scalar ``β_match`` after the epoch;
    #: ``matchup_n_eff_{mean,p10,p50,p90}`` describe the distribution
    #: of per-row peer-vs-opponent evidence (load-bearing for the
    #: ESS-bucket falsification).
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

    Used as a default-on diagnostic in the trainer's hot loop. The cost
    is two reductions per check, negligible compared to the rest of the
    forward/backward pass; we keep it default-on because a silent NaN
    corrupts every subsequent iteration via the optimizer and is the
    single most painful debugging mode of the trainer.
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
    """Build the named-module dict the trainer uses for device
    placement, Adam params, best-val snapshot/restore, and grad-NaN
    scans. Dispatches on the spatial-model type so the cell-free
    (continuous_mixture / mode_mixture) and grid paths share one
    bookkeeping function."""
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
    # Cell-free path (continuous_mixture or mode_mixture). Both share
    # the same upstream support attention; only the spatial-density
    # head differs.
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
        if spatial.defensive_field is not None:
            # PR-D2a: the cell-free defensive field (when wired) joins
            # the module dict so its parameters land in the optimizer
            # and best-val snapshot dict. The retrieval cache + feature
            # artifact stored on the wrapper are non-Module attributes
            # — they don't carry parameters and are handled via the
            # gather function's device transfer at forward time.
            modules["defensive_field"] = spatial.defensive_field
        if spatial.bandwidth_field is not None:
            # Tier-1a source/zone bandwidth: 2×N_ZONES learnable scalars
            # join the optimizer + best-val snapshot dict.
            modules["bandwidth_field"] = spatial.bandwidth_field
        if spatial.anisotropic_kernel is not None:
            # Tier-2 anisotropic kernel: per-zone covariance params
            # (16 for RT, 24 for FC) join the optimizer + best-val
            # snapshot dict. Mirrors how bandwidth_field is registered.
            modules["anisotropic_kernel"] = spatial.anisotropic_kernel
        if spatial.matchup_field is not None:
            # Tier-2a D-matchup: scalar β_match joins the optimizer +
            # best-val snapshot dict. The matchup-features artifact is
            # a non-Module attribute (no learnable parameters) and is
            # handled via the gather inside the wrapper forward.
            modules["matchup_field"] = spatial.matchup_field
        if spatial.has_within_game_gru:
            # G1 within-game shot GRU: GRU cell + zero-init projection.
            # Joins the optimizer + best-val snapshot dict so its weights
            # are trained and the eval-side reconstruction can rebuild
            # from the saved state_dict.
            assert spatial.within_game_gru is not None  # narrowed by the flag
            modules["within_game_gru"] = spatial.within_game_gru
        if spatial.causal_zone_bias is not None:
            # Phase 2 α1 (causal redesign, 2026-06-09): causal zone-pair
            # edge bias on support attention. Lives as its own submodule
            # so the optimizer trains its parameters, the best-val
            # snapshot/restore round-trips them, and the disk save state
            # captures them via state_dict. See
            # :class:`shotcloud.models.CausalZoneBias` and the locked
            # leakage rule on the wrapper class.
            modules["causal_zone_bias"] = spatial.causal_zone_bias
        # Phase 3 mode-routed AC-KDE (2026-06-09): mode router lives
        # on the ModeRoutedContinuousMixtureSpatial subclass. Register
        # it so the optimizer trains the router head and the save/load
        # pipeline round-trips its parameters.
        if isinstance(spatial, ModeRoutedContinuousMixtureSpatial):
            modules["mode_router"] = spatial.mode_router
    else:
        # CollaborativeModeMixtureSpatial: residual location embedding
        # is the per-shot-residual ψ; the mode-extractor owns a
        # separate support-embedding ψ for its Q-K product plus the
        # mode queries and context-bias MLP.
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
    """Run one epoch. ``optimizer=None`` runs evaluation (no_grad)."""
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

    # Per-game shot counts and per-game pregame context, moved once.
    shots_per_game = (
        torch.bincount(dataset.game_idx, minlength=dataset.n_games)
        .to(device=device, dtype=torch.float32)
        .clamp_min(1.0)
    )
    per_game_x_raw = dataset.per_game.x_n_raw.to(device)
    per_game_k = dataset.per_game.k_obs.to(device)

    total_spatial = 0.0  # optimization target — for `total` and best-val
    total_spatial_cell = 0.0  # NaN in continuous_mixture mode
    total_spatial_continuous = 0.0  # NaN in continuous_mixture mode
    total_spatial_mix_nll = 0.0  # only populated in continuous_mixture mode
    total_expected_distance_ft = 0.0  # grid modes only
    total_alpha_entropy = 0.0
    total_beta_entropy = 0.0
    n_alpha_rows = 0  # batches with collab α; mean reported per-row
    n_beta_rows = 0  # (b, l) pairs with valid causal history
    total_sub_uniform = 0  # shots with q(c_obs) < 1/n_cells
    total_support_entropy = 0.0  # H(w) under joint mixture (continuous_mixture only)
    total_eff_support = 0.0  # exp(H(w)) (continuous_mixture only)
    total_expected_support_dist = 0.0  # Σ_m w_m ||s_m - y|| (continuous_mixture only)
    total_min_dist_to_support = 0.0  # min_m ||s_m - y|| (continuous_mixture only)
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
    # Cell-free defense diagnostics (continuous_mixture + defense only).
    # All four are weighted sums; divide by ``n_shots_seen`` at the end.
    total_def_score_abs_mean = 0.0
    total_def_score_max_abs = 0.0  # per-row max |D|, batch-summed
    total_def_score_mean = 0.0
    total_def_cold_start_rows = 0
    n_def_rows = 0
    # Tier-2a D-matchup diagnostics. ``total_matchup_*`` are weighted
    # sums divided by ``n_match_rows`` at the epoch end;
    # ``matchup_n_eff_pool`` holds the per-row N_eff values for
    # percentile reporting at the end of the epoch.
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
                # PR-D2a: thread opp_idx through the cell-free spatial
                # call. The wrapper takes ``opp_idx=None`` by default
                # for back-compat; the continuous-mixture wrapper only
                # *consumes* it when ``has_defense`` is True. The
                # mode-mixture wrapper ignores it. Passing it
                # unconditionally is safe and keeps the call site
                # uniform.
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
                # Tilt regularizer for the cell-free residual: L2 on the
                # per-shot R_θ(s_m) contribution (broadcast α=ones over
                # M, so this is exactly the analog of the grid tilt L2).
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
                    # Cold-start rows now carry a dummy-logit ``w`` from
                    # the upstream patch, so ``expected_d`` is finite
                    # but meaningless for those rows. Filter them out
                    # of both diagnostics before the mean.
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
                    # Mode-locality diagnostic (mode_mixture only). For
                    # each row's K mode centers, compute min L2 distance
                    # to its causal support set; average across modes
                    # and across rows with valid support. Soft-k-means
                    # bounds this small by construction; the
                    # learned-query extractor's H2 pathology had modes
                    # drifting toward global anchors and this metric
                    # exploding.
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
                    # α, β entropies from the SUPPORT attention (always
                    # M-shaped). For continuous_mixture, support
                    # attention == mixture weights, so this matches the
                    # old behavior. For mode_mixture, support attention
                    # is upstream of the K-shaped mode weights, but the
                    # (L, R) factorization of the support is identical
                    # so α / β remain meaningful and comparable.
                    # The retrieval backend has no (L, R) factorization
                    # — leave α/β entropy at the default NaN sentinel.
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
                # Always compute both spatial losses + expected-distance
                # diagnostic for cross-run comparability. The chosen
                # ``spatial_likelihood`` selects which feeds the optimizer.
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

            # Count NLL is per-game by nature. Compute once per unique
            # game in the batch (deduplicated from the per-shot view) so
            # the count head receives a single gradient per game per
            # update, regardless of K_g. The per-game context is the
            # game's first-shot raw context, run through the same f_ctx.
            unique_games, inverse_to_unique = torch.unique(game_idx, return_inverse=True)
            game_x_raw_unique = per_game_x_raw[unique_games]
            game_x_n_unique = context_mlp(game_x_raw_unique)
            game_k_unique = per_game_k[unique_games]
            count_log_p_unique = count_head.log_prob(game_k_unique, game_x_n_unique)
            if nan_check:
                _check_finite("count_log_p_unique", count_log_p_unique, batch_idx)
            # Per-game NLL for the per_game-normalization loss term.
            count_loss_per_game = -count_log_p_unique
            # Per-shot amortized view: broadcast the per-game log-prob
            # back to (B,) and divide by shots-in-game. Used by the
            # per_shot legacy normalization and by the per-shot count
            # diagnostic that the _EpochLosses ``count`` field reports.
            shots_in_game = shots_per_game[game_idx]
            count_log_p = count_log_p_unique[inverse_to_unique]
            count_nll_per_shot = -count_log_p / shots_in_game

            # Tilt regularizer. Grid: r_θ(c) = u^T v_c per cell.
            # Cell-free: R_θ(s_m) = u^T ψ(s_m) per support shot. Same
            # L2 form per-row across the residual's "output axis"
            # (cells or support shots).
            if spatial.has_residual:
                tilt_reg_per_shot = (r_theta.pow(2)).mean(dim=-1)
            else:
                tilt_reg_per_shot = torch.zeros_like(spatial_nll)

            # Defense regularizer (PR-D3). Penalizes the per-shot
            # log-feasibility magnitude E[D_Δ²]; computed per-row as
            # the mean over the M_off support axis. Zero when defense
            # is off (mix_out is the cell-free wrapper output;
            # mode_mixture wrappers don't yet carry defense_logits).
            defense_reg_per_shot = torch.zeros_like(spatial_nll)
            if (
                is_mixture_mode
                and isinstance(spatial, ContinuousMixtureSpatial)
                and spatial.has_defense
                and mix_out.defense_logits is not None
            ):
                defense_reg_per_shot = mix_out.defense_logits.pow(2).mean(dim=-1)

            # The count contribution to the optimization loss depends on
            # the normalization mode. In ``per_game`` mode the per-game
            # NLL is averaged separately and added to the per-shot mean;
            # ``lambda_count = 1.0`` then weights one per-game count
            # update at parity with one per-shot spatial update. In
            # ``per_shot`` mode the amortized-per-shot count NLL is mixed
            # into ``loss_per_shot`` (legacy; reproduces pre-2026-06-07
            # checkpoints, where ``lambda_count = 1.0`` effectively
            # weighted count NLL at ``1/K̄`` relative to spatial NLL).
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
            # Grid-mode metrics: sum if defined, else leave at 0 (the
            # epoch-level mean will be NaN by construction below when
            # we divide by 0 — guard at output).
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
            # Defense diagnostics — only populate when defense is wired
            # and the wrapper emitted defense_logits this batch.
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
            # D-matchup diagnostics — populate when the matchup channel
            # is wired and the wrapper emitted matchup_logits this batch.
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
    """Per-row entropy ``-Σ p log p`` in nats. Last dim is the
    distribution; output is one rank lower.

    ``p`` is assumed to be a proper (rows sum-to-1) probability tensor,
    OR zero on entire rows (the collaborative β returns all-zero rows
    for analogues with no causal history). Zero rows produce 0 entropy
    (handled by the ``eps`` clamp inside the log).
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
    #: Optional history-dependent pooling gate; only consulted when
    #: ``spatial_likelihood == "continuous_mixture"``. When provided,
    #: the spatial density becomes the structured two-component
    #: mixture ``λ·f_own + (1-λ)·f_pooled``.
    pooling_gate: PoolingGate | None = None,
    #: Cell-free defense triple (PR-D2b). All three must be provided
    #: together or all ``None``. Only consulted when
    #: ``spatial_likelihood == "continuous_mixture"``. The trainer
    #: passes them through to :class:`ContinuousMixtureSpatial` which
    #: handles the gather + forward wiring (PR-D2a). Distinct from the
    #: ``defensive_field`` kwarg above, which is the *grid-side*
    #: :class:`AdaptiveDefensiveField` used by the cell-based path.
    defensive_field_cellfree: (
        ContinuousAdaptiveDefensiveField | ZoneReweightingDefense | None
    ) = None,
    defensive_cache: DefensiveRetrievalCache | None = None,
    defensive_features: DefenseFeatures | None = None,
    #: Tier-2a D-matchup channel: ``β_match · Δ̂_{p,d,z(s_m)}(t)``. Both
    #: ``matchup_field`` and ``matchup_features`` must be provided
    #: together or both ``None``. Composes with the D-lite cell-free
    #: channel for the ablation-C combined run; passes through to
    #: :class:`ContinuousMixtureSpatial` which handles the gather.
    matchup_field: MatchupReweightingDefense | None = None,
    matchup_features: MatchupFeatures | None = None,
    #: Optional Tier-1a source/zone bandwidth field. When provided,
    #: each support shot gets its own σ_{src,z} instead of the per-row
    #: collab σ. Only consulted when ``spatial_likelihood ==
    #: "continuous_mixture"``; the spatial loglik already accepts both
    #: (B,) and (B, M) σ shapes.
    bandwidth_field: ZoneSourceBandwidth | None = None,
    #: Optional Tier-2 anisotropic kernel (Option 1 RT or Option 3 FC).
    #: Mutually exclusive with ``bandwidth_field`` — both modify the
    #: kernel-shape axis. When wired, replaces the isotropic kernel
    #: entirely with per-zone covariance shaping. Only consulted when
    #: ``spatial_likelihood == "continuous_mixture"``.
    anisotropic_kernel: RadialTangentZoneKernel | FullCovarianceZoneKernel | None = None,
    #: K̂-standardization stats (paper 2026-06-05 calibration fix). When
    #: both are provided AND the residual encoder consumes K̂, CMS
    #: applies ``tilde_K = (log1p(K̂) − μ) / σ`` before the K̂ column
    #: reaches the residual. Use K_obs-based stats from the audit (e.g.
    #: μ=2.239, σ=0.490 for the current corpus) so the transform is
    #: invariant to count-head calibration drift.
    khat_log1p_mean: float | None = None,
    khat_log1p_std: float | None = None,
    #: G1 within-game shot GRU (paper §10). When provided, the module's
    #: output is added to the residual encoder output before the
    #: location embedding; CMS validates the shape match. Wired only
    #: when ``residual_encoder`` is active. Defaults ``None`` =
    #: backward-compatible no-GRU path.
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
    #: Count-loss normalization (see :data:`CountLossNormalization`).
    #: ``"per_game"`` (default) puts ``lambda_count`` at parity with
    #: per-shot spatial NLL; ``"per_shot"`` reproduces the pre-2026-06-07
    #: amortized-per-shot weighting where ``lambda_count = 1.0``
    #: effectively under-supervised the count head by ``≈ 1/K̄``.
    count_loss_normalization: CountLossNormalization = "per_game",
    #: Optional path to a checkpoint ``.pt`` produced by
    #: :mod:`scripts.train_count_head`. When provided, the joint
    #: trainer loads ``count_head`` (and ``context_mlp`` if present)
    #: state from the checkpoint before training starts. Use with
    #: ``freeze_count=True`` to consume a fixed calibrated μ from a
    #: pretrained count head — the canonical paper §5.2 calibration
    #: workflow.
    count_checkpoint_path: str | Path | None = None,
    #: If ``True``, disable gradients on ``count_head`` parameters.
    #: The spatial decoder still consumes the count head's detached
    #: μ via ``c_{p,t_n} = stopgrad(μ_η(x_n))``, so the residual still
    #: sees a count-supervised signal — it's just a *frozen* one.
    freeze_count: bool = False,
    #: If ``True``, disable gradients on ``context_mlp`` parameters
    #: too. Required for the count-pathway-frozen experiment of
    #: paper §5.2: the count head's calibration depends on
    #: ``f_ctx(x_n_raw)``, and freezing ``count_head`` alone does
    #: *not* freeze that pathway because spatial gradients flow
    #: through ``context_mlp``. Pin both to test "does calibrated
    #: K̂ improve spatial when the count pathway is genuinely
    #: frozen." Caveat: ``context_mlp`` is shared with retrieval,
    #: residual, and timing — freezing it constrains the spatial
    #: representation too. If the resulting spatial NLL regresses,
    #: the principled solution is dedicated context encoders for
    #: count vs spatial, not blanket freezing.
    freeze_context_mlp: bool = False,
    #: Optional separate learning rate for the count-head parameter
    #: group. ``None`` (default) puts every trainable parameter on a
    #: single Adam group at ``learning_rate``. When set (and
    #: ``freeze_count=False``), the count head gets its own LR while
    #: the rest of the model stays at ``learning_rate``. Useful for
    #: ``count_pretrain + low-LR-finetune`` (paper §5.2.C).
    count_lr: float | None = None,
    spatial_likelihood: SpatialLikelihood = "continuous_cell",
    obs_kernel_tau: float = 1.0,
    obs_kernel_normalize: bool = True,
    # Mode-mixture (cell-free, mode-extraction) hyperparameters; only
    # consulted when ``spatial_likelihood == "mode_mixture"``.
    mode_mixture_n_court_modes: int = 6,
    mode_mixture_query_dim: int = 32,
    mode_mixture_sigma_ft: float = 3.0,
    mode_mixture_context_correction: bool = True,
    mode_mixture_lambda_omega: float = 0.0,
    mode_mixture_tail_weight: float = 0.0,
    mode_mixture_tail_sigma_ft: float = 1.0,
    #: Which mode-extraction operator to instantiate inside the
    #: mode-mixture spatial decoder. ``"soft_kmeans"`` (the 2026-05-18
    #: default) clusters the row's attended support via weighted FPS
    #: init + a few mean-shift iterations; modes are local to the
    #: row by construction. ``"learned_query"`` is the legacy
    #: globally-parameterized extractor — kept for ablations.
    mode_mixture_extractor_kind: str = "soft_kmeans",
    #: Mean-shift kernel bandwidth ``ρ`` (feet). Only consulted when
    #: ``mode_mixture_extractor_kind == "soft_kmeans"``.
    mode_mixture_kernel_bandwidth_ft: float = 5.0,
    #: Number of mean-shift refinement iterations after the weighted-
    #: FPS init. Only consulted when ``mode_mixture_extractor_kind ==
    #: "soft_kmeans"``.
    mode_mixture_n_iterations: int = 2,
    #: Optional per-epoch snapshot callback. When provided, the
    #: trainer calls ``snapshot_callback(epoch, module_states)`` after
    #: each epoch's train + val pass with ``module_states`` a
    #: ``dict[str, dict[str, Tensor]]`` mirroring the participating
    #: modules' ``state_dict()``s. Callers (typically the training
    #: CLI) filter by epoch and persist to disk for downstream
    #: visualization / analysis tooling.
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
    """Train the joint Gibbs model.

    Parameters
    ----------
    offensive_prior : AdaptiveOffensivePrior
        The :math:`q_n^{\\mathrm{off}}` factor; consumes
        ``(player_idx, snapshot_idx, x_n)``.
    count_head, timing_head : NegBin / 48-bin softmax heads
        Operate on the learned ``x_n``.
    context_mlp : ContextMLP
        ``f_{\\mathrm{ctx}}``. Residual zero-init recommended so the
        model begins as the raw-context baseline.
    train_set, val_set : GibbsShotDataset
        When ``defensive_field`` is provided, both datasets must
        carry an ``opp_vocab`` and the dataset's ``opp_idx`` tensor
        must align with that vocab.
    defensive_field : optional
        Adds the opponent-conditioned reweighting :math:`a_\\delta`
        to the spatial energy: ``log q_off + log a_δ + r_θ``. The
        dataset must have ``has_opponents=True``.
    residual_encoder, tilt_decoder : optional
        Provide both or neither. When provided, the spatial logits
        gain :math:`r_\\theta = u_\\theta(x_n)^\\top v_c`.
    n_epochs, batch_size, learning_rate, weight_decay : Adam config.
    lambda_tilt : float
        Coefficient on the per-shot residual L2 penalty. Ignored
        when ``residual_encoder is None``.
    device : torch.device or str
    shuffle : bool
    progress : bool
        Print per-epoch losses if True.
    restore_best_val : bool
        If True and ``val_set`` is provided, snapshot the parameters
        whenever val spatial NLL improves and restore them before
        returning. Mirrors the legacy trainer's ``restore_best_val``
        contract.
    nan_check : bool, default True
        Enable per-batch NaN guards in the forward + backward paths.
    max_batches : int or None
        If set, stop each epoch after this many batches. Useful for
        diagnostic runs.
    """
    # Legacy alias support: callers passing ``spatial_loss=`` instead of
    # ``spatial_likelihood=`` get the legacy two-value semantics
    # mapped to the new three-value enum.
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
    # Legacy local name used inside _epoch for grid-mode dispatch.
    spatial_loss: str = "continuous" if spatial_likelihood == "continuous_cell" else "cell"
    dev = torch.device(device)

    # Cell centers in image-layout flat order (c = iy*nx + ix), as one
    # (n_cells, 2) float32 tensor on `dev`. Built once and reused every
    # batch — the spatial loss helpers don't take ``grid`` directly so
    # we don't have to rematerialize this each call.
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
        # Cell-free defense kind-aware validation. Mirrors the
        # constructor checks inside ContinuousMixtureSpatial:
        # * D-field (ContinuousAdaptiveDefensiveField) requires the
        #   full triple ``(field, cache, features)``.
        # * D-lite (ZoneReweightingDefense) requires ``(field, features)``
        #   only; cache must be None.
        # * All three None → no defense.
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
        # D-matchup wiring validation. Both ``matchup_field`` and
        # ``matchup_features`` must be provided together. Composes
        # freely with ``n_defense`` (zero or three) for the A/B/C
        # ablations. Only valid for continuous_mixture.
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
            # Count-residual wiring: CMS needs ``count_head`` whenever
            # the residual encoder's ``usage_dim`` consumes K̂ —
            # either ``USAGE_KHAT_DIM`` (B2 mainline: usage + K̂) or
            # ``1`` (K̂-only diagnostic). The count head's parameters
            # are still owned by the trainer's top-level module dict
            # (so they receive L_count gradient); CMS just consumes
            # the head's forward.
            from shotcloud.features.usage_features import USAGE_KHAT_DIM

            count_head_for_spatial: nn.Module | None = None
            if residual_encoder is not None and residual_encoder.usage_dim in (
                USAGE_KHAT_DIM,
                1,
            ):
                count_head_for_spatial = count_head
            # Phase 3 mode-routed AC-KDE (2026-06-09): when a
            # ``mode_router`` is provided, build the structural
            # mode-routed subclass instead of the single-softmax CMS.
            # The subclass rejects ``pooling_gate`` and
            # ``causal_zone_bias`` at construction time per the
            # 2026-06-09 design lock — those knobs are mutually
            # exclusive with mode routing.
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

    # Optional pretrain → freeze workflow (paper §5.2 calibration fix).
    # ``count_checkpoint_path`` loads a state_dict from a
    # :mod:`scripts.train_count_head` run; ``freeze_count`` disables
    # ``count_head`` parameter gradients so the spatial decoder
    # consumes a fixed, pre-calibrated μ via the detached
    # ``c_{p,t_n} = stopgrad(μ_η(x_n))``.
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

    # Optimizer with optional separate LR for the count head.
    # ``count_lr=None`` keeps the legacy single-group setup. When set,
    # the count-head params get their own AdamW group at ``count_lr``;
    # all other params stay at ``learning_rate``. Useful for the joint
    # ``count_pretrain + low-LR finetune`` experiment (paper §5.2.C).
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

        # Per-epoch snapshot callback for downstream viz / analysis.
        # Always fires (caller filters by epoch); module_states are
        # detached clones so they survive subsequent in-place updates.
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

    # Final count-head calibration on train + val (paper §5.2). Computed
    # after best-val restoration so the diagnostic reflects the model
    # the caller actually gets back. Skipped when count training is
    # disabled (lambda_count == 0) — the count head's parameters then
    # carry no meaningful signal. Wrapped in try/except so a diagnostic
    # failure cannot lose a full training run; any failure is reported
    # on the resulting history field via an ``{"error": ...}`` dict.
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

    # Expose the resolved spatial wrapper through the optional output
    # container so callers can serialize wrapper-owned state (e.g. the
    # zone-pair attention bias submodule) that doesn't live on any of
    # the inbound submodules. Backwards-compat: when ``out_spatial`` is
    # None this is a no-op.
    if out_spatial is not None:
        out_spatial.append(spatial)

    # Deepcopy guards against caller-side mutation of the lists.
    return deepcopy(history)
