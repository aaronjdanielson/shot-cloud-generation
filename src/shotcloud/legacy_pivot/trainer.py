"""Spatial NLL trainer for the low-rank tilt decoder.

Trains the decoder's ``V`` matrix and the player encoder's embedding
table jointly to minimize cross-entropy ``-log p_θ(c | x)`` over
observed shots. Optional learnable scalars compose orthogonally on top:

* ``τ = softplus(θ)`` from :class:`~shotcloud.models.LearnableTemperature`
  (Phase 1) sharpens / flattens the offensive prior.
* ``α_def = softplus(θ_d)`` from
  :class:`~shotcloud.models.LearnableDefensiveScale` (Phase 2) scales a
  defensive product factor ``log q_def(c | opp)``.
* The legacy :class:`~shotcloud.models.LearnableKDEProductWeights`
  (KDE-product mixture weights) — kept for the paper ablation row;
  empirically collapses to "temperature on player KDE", so prefer the
  single-scalar Phase-1 module.

The mixture-weights and temperature options are mutually exclusive
(both control the offensive prior). The defensive scale composes with
either offensive option.

This is the **v1 minimum viable trainer**: pure spatial NLL, Adam,
single-pass mini-batches, no learning-rate schedule. By default we
**track the best validation checkpoint** and restore it at the end of
training — without this, a long run that overfits returns an inferior
model. Toggle via ``restore_best_val=False`` if you want the final-epoch
state instead (useful when overfitting is genuinely desired, e.g., when
the val set is too small to trust).

Initialization gotcha
---------------------
The decoder defaults to ``V = 0`` (preserves the
``softmax(log q_0) == q_0`` invariant at step 0). The encoder defaults
to **random-init** because zero-initing both factors traps the model at
a saddle point: ``∂(u^T V)/∂V = u = 0`` and
``∂(u^T V)/∂u = V = 0``, so neither factor moves. The standard low-rank
pattern is exactly one zero factor — see
:class:`~shotcloud.legacy.PlayerEmbeddingEncoder` docstring for detail.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from shotcloud.legacy import (
    LearnableKDEProductWeights,
    LearnableTemperature,
    PlayerEmbeddingEncoder,
    PlayerPositionEncoder,
)
from shotcloud.legacy_pivot.adaptive_prior import AdaptiveOffensivePrior
from shotcloud.legacy_pivot.defensive_scale import LearnableDefensiveScale
from shotcloud.legacy_pivot.regularizers import (
    entropy_regularizer,
    ess_regularizer,
)
from shotcloud.legacy_pivot.shot_cell_dataset import ShotCellDataset
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder

EncoderModule = PlayerEmbeddingEncoder | PlayerPositionEncoder


@dataclass
class TrainHistory:
    """Per-epoch training trajectory."""

    train_nll: list[float] = field(default_factory=list)
    val_nll: list[float] = field(default_factory=list)
    best_epoch: int | None = None  # 1-indexed; None if no val_set was provided

    @property
    def best_val_nll(self) -> float:
        return min(self.val_nll) if self.val_nll else float("inf")

    @property
    def final_train_nll(self) -> float:
        return self.train_nll[-1] if self.train_nll else float("nan")


def _epoch(
    encoder: EncoderModule,
    decoder: LowRankTiltDecoder,
    loader: Iterable[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    learnable_weights: LearnableKDEProductWeights | None = None,
    learnable_temperature: LearnableTemperature | None = None,
    learnable_defensive_scale: LearnableDefensiveScale | None = None,
    adaptive_prior: AdaptiveOffensivePrior | None = None,
    lambda_entropy: float = 0.0,
    lambda_ess: float = 0.0,
    log_qp_table: Tensor | None = None,
    log_qg_table: Tensor | None = None,
    log_ql_vector: Tensor | None = None,
    log_qd_table: Tensor | None = None,
) -> float:
    """Run one pass over ``loader``. Returns mean NLL per shot.

    If ``optimizer`` is ``None`` runs in eval mode (no autograd).

    When ``learnable_weights`` is provided, the dataset's cached
    ``log_q0_table`` is **ignored** — instead the per-shot ``log q_0`` is
    recomputed from the per-component tables (``log_qp_table``,
    ``log_qg_table``, ``log_ql_vector``) using the current weights.

    When ``learnable_temperature`` is provided, the dataset's cached
    ``log_q0_table`` is multiplied by ``τ = softplus(θ)`` per minibatch
    before being passed to the decoder. The decoder's softmax absorbs
    the temperature partition function, so no separate normalization is
    needed.

    When ``learnable_defensive_scale`` is provided, ``α_def = softplus(θ_d)``
    times ``log q_def(c | opp)`` is added to the offensive log-density
    inside the decoder's softmax. Requires ``log_qd_table`` (per-opponent
    log defensive density).
    """
    is_train = optimizer is not None
    encoder.train(is_train)
    decoder.train(is_train)
    if learnable_weights is not None:
        learnable_weights.train(is_train)
    if learnable_temperature is not None:
        learnable_temperature.train(is_train)
    if learnable_defensive_scale is not None:
        learnable_defensive_scale.train(is_train)
    if adaptive_prior is not None:
        adaptive_prior.train(is_train)

    use_learnable_weights = learnable_weights is not None
    use_adaptive = adaptive_prior is not None
    if use_learnable_weights and use_adaptive:
        raise ValueError(
            "learnable_weights and adaptive_prior are mutually exclusive — "
            "the adaptive prior replaces the offensive base measure entirely."
        )
    if use_learnable_weights and (
        log_qp_table is None or log_qg_table is None or log_ql_vector is None
    ):
        raise ValueError("learnable_weights requires log_qp_table, log_qg_table, log_ql_vector")
    if learnable_defensive_scale is not None and log_qd_table is None:
        raise ValueError("learnable_defensive_scale requires log_qd_table")

    total_nll = 0.0
    total_n = 0
    nll_loss = nn.NLLLoss(reduction="sum")

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for player_idx, cell_idx, log_q0, opp_idx, x_n in loader:
            player_idx = player_idx.to(device)
            cell_idx = cell_idx.to(device)
            x_n_dev = x_n.to(device) if x_n.numel() > 0 else x_n

            pi: Tensor | None = None  # captured if adaptive_prior is on, for regularizers
            if use_adaptive:
                assert adaptive_prior is not None
                log_q0, pi, _omega = adaptive_prior(player_idx, x_n_dev)
                if learnable_temperature is not None:
                    if learnable_temperature.context_dim > 0:
                        log_q0 = learnable_temperature(log_q0, x_n_dev)
                    else:
                        log_q0 = learnable_temperature(log_q0)
            elif use_learnable_weights:
                assert (
                    learnable_weights is not None
                    and log_qp_table is not None
                    and log_qg_table is not None
                    and log_ql_vector is not None
                )
                log_qp = log_qp_table[player_idx]
                log_qg = log_qg_table[player_idx]
                log_q0 = learnable_weights(log_qp, log_qg, log_ql_vector)
            else:
                log_q0 = log_q0.to(device)
                if learnable_temperature is not None:
                    if learnable_temperature.context_dim > 0:
                        log_q0 = learnable_temperature(log_q0, x_n_dev)
                    else:
                        log_q0 = learnable_temperature(log_q0)

            if learnable_defensive_scale is not None:
                assert log_qd_table is not None
                opp_idx = opp_idx.to(device)
                log_q_def = log_qd_table[opp_idx]
                if learnable_defensive_scale.context_dim > 0:
                    log_q0 = log_q0 + learnable_defensive_scale(log_q_def, x_n_dev)
                else:
                    log_q0 = log_q0 + learnable_defensive_scale(log_q_def)

            u = encoder(player_idx)  # (B, rank)
            log_probs = decoder.log_probs(log_q0, u)  # (B, n_cells)
            loss = nll_loss(log_probs, cell_idx)

            # Regularizers on the adaptive relevance softmax.
            if pi is not None:
                if lambda_entropy > 0:
                    loss = loss + lambda_entropy * entropy_regularizer(pi) * cell_idx.shape[0]
                if lambda_ess > 0:
                    loss = loss + lambda_ess * ess_regularizer(pi) * cell_idx.shape[0]

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_nll += float(loss.detach().cpu())
            total_n += int(cell_idx.shape[0])

    return total_nll / max(total_n, 1)


def train_decoder(
    encoder: EncoderModule,
    decoder: LowRankTiltDecoder,
    train_set: ShotCellDataset,
    *,
    val_set: ShotCellDataset | None = None,
    learnable_weights: LearnableKDEProductWeights | None = None,
    learnable_temperature: LearnableTemperature | None = None,
    learnable_defensive_scale: LearnableDefensiveScale | None = None,
    adaptive_prior: AdaptiveOffensivePrior | None = None,
    lambda_entropy: float = 0.0,
    lambda_ess: float = 0.0,
    n_epochs: int = 5,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str | torch.device = "cpu",
    shuffle: bool = True,
    progress: bool = False,
    restore_best_val: bool = True,
) -> TrainHistory:
    """Train ``encoder`` + ``decoder`` jointly on spatial NLL.

    Parameters
    ----------
    encoder : PlayerEmbeddingEncoder
        Produces ``u`` from player indices. Must share ``rank`` with
        ``decoder``.
    decoder : LowRankTiltDecoder
        Must share ``n_cells`` with the dataset's grid.
    train_set : ShotCellDataset
    val_set : ShotCellDataset, optional
        Same vocab as ``train_set``. If provided, val NLL is logged each
        epoch.
    learnable_weights : LearnableKDEProductWeights, optional
        Legacy. If provided, the KDE-product weights ``(a_p, a_g, a_0)``
        are trained jointly. ``train_set`` (and ``val_set``, if provided)
        must be constructed with ``cache_components=True``. Mutually
        exclusive with ``learnable_temperature``.
    learnable_temperature : LearnableTemperature, optional
        Phase 1. If provided, a single scalar ``τ = softplus(θ)`` is
        trained jointly and applied to ``log q_0`` per minibatch. Works
        with any base measure; no component caching needed. Mutually
        exclusive with ``learnable_weights``.
    learnable_defensive_scale : LearnableDefensiveScale, optional
        Phase 2. If provided, a single scalar ``α_def = softplus(θ_d)``
        scales the per-shot log defensive density and is added to the
        offensive log-density inside the decoder's softmax. Requires
        the train (and val, if provided) datasets to be constructed
        with a ``defensive_kde``. Composes orthogonally with the
        offensive options (temperature or mixture weights).
    adaptive_prior : AdaptiveOffensivePrior, optional
        Phase 4. If provided, **replaces** the dataset's offensive
        ``log q_0`` with the relevance-weighted historical-shot
        density ``log q̂_φ^hier(c | p, x_n)``. Mutually exclusive with
        ``learnable_weights`` (which configures a different offensive
        prior). May still be paired with ``learnable_temperature`` and
        ``learnable_defensive_scale``. Requires train (and val) sets
        to be constructed with a ``context_encoder``.
    lambda_entropy : float, default 0.0
        Phase 4 regularizer weight on
        :func:`~shotcloud.training.entropy_regularizer`. Only effective
        when ``adaptive_prior`` is set. ``0.005`` is a sensible non-zero
        starting value (research_plan §7).
    lambda_ess : float, default 0.0
        Phase 4 regularizer weight on
        :func:`~shotcloud.training.ess_regularizer`. Off by default;
        turn on if attention collapses to a few shots.
    n_epochs : int, default 5
    batch_size : int, default 512
    learning_rate, weight_decay : Adam hyperparameters.
    device : str or torch.device, default "cpu"
    shuffle : bool, default True
    progress : bool, default False
        If True, prints per-epoch NLL.
    restore_best_val : bool, default True
        When ``val_set`` is provided, snapshot the encoder + decoder
        parameters whenever val NLL improves and restore them before
        returning. This makes the saved checkpoint correspond to the
        best validation epoch — without it, a long run that overfits
        returns the (worse) final-epoch state. Has no effect when
        ``val_set`` is ``None``.

    Returns
    -------
    TrainHistory
        ``history.best_epoch`` is set (1-indexed) when ``val_set`` is
        provided; otherwise ``None``.
    """
    if encoder.rank != decoder.rank:
        raise ValueError(f"encoder rank ({encoder.rank}) must match decoder rank ({decoder.rank})")
    if learnable_weights is not None and learnable_temperature is not None:
        raise ValueError(
            "learnable_weights and learnable_temperature are mutually exclusive — "
            "running both at once produces an over-specified parameterization. "
            "Pick one (Phase 1 recommends learnable_temperature)."
        )
    if learnable_weights is not None and adaptive_prior is not None:
        raise ValueError(
            "learnable_weights and adaptive_prior are mutually exclusive — "
            "the adaptive prior replaces the offensive base measure entirely."
        )
    if learnable_weights is not None and not train_set.has_components:
        raise ValueError("learnable_weights requires train_set built with cache_components=True")
    if learnable_weights is not None and val_set is not None and not val_set.has_components:
        raise ValueError("learnable_weights requires val_set built with cache_components=True")
    if learnable_defensive_scale is not None and not train_set.has_defensive:
        raise ValueError(
            "learnable_defensive_scale requires train_set built with defensive_kde=..."
        )
    if learnable_defensive_scale is not None and val_set is not None and not val_set.has_defensive:
        raise ValueError("learnable_defensive_scale requires val_set built with defensive_kde=...")

    # Phase-4 adaptive prior requires the dataset to carry x_n features.
    if adaptive_prior is not None:
        if not train_set.has_context:
            raise ValueError("adaptive_prior requires train_set built with context_encoder=...")
        if val_set is not None and not val_set.has_context:
            raise ValueError("adaptive_prior requires val_set built with context_encoder=...")
        # AA-KDE refactor (2026-04-29): AdaptiveOffensivePrior now requires
        # a per-row snapshot_idx and consumes the SnapshotStore + archetype
        # modules. The legacy trainer path here predates the refactor; the
        # AA-KDE joint training script is the supported entry point. Fail
        # loudly so callers don't silently use the broken path.
        raise NotImplementedError(
            "train_decoder() does not support the AA-KDE AdaptiveOffensivePrior; "
            "use the AA-KDE joint training pipeline (pending) instead. "
            "See docs/log.md 2026-04-29 for the refactor scope."
        )

    # Phase-3 context-conditioned-mode requires the dataset to carry x_n features.
    needs_context = (
        learnable_temperature is not None and learnable_temperature.context_dim > 0
    ) or (learnable_defensive_scale is not None and learnable_defensive_scale.context_dim > 0)
    if needs_context:
        if not train_set.has_context:
            raise ValueError(
                "context-conditioned learnable scalars require train_set built with "
                "context_encoder=..."
            )
        if val_set is not None and not val_set.has_context:
            raise ValueError(
                "context-conditioned learnable scalars require val_set built with "
                "context_encoder=..."
            )
        # Make sure dimensions agree.
        for module, name in (
            (learnable_temperature, "learnable_temperature"),
            (learnable_defensive_scale, "learnable_defensive_scale"),
        ):
            if (
                module is not None
                and module.context_dim > 0
                and module.context_dim != train_set.context_dim
            ):
                raise ValueError(
                    f"{name}.context_dim ({module.context_dim}) does not match "
                    f"train_set.context_dim ({train_set.context_dim})"
                )

    dev = torch.device(device)
    encoder.to(dev)
    decoder.to(dev)
    if learnable_weights is not None:
        learnable_weights.to(dev)
    if learnable_temperature is not None:
        learnable_temperature.to(dev)
    if learnable_defensive_scale is not None:
        learnable_defensive_scale.to(dev)
    if adaptive_prior is not None:
        adaptive_prior.to(dev)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] | None = (
        DataLoader(val_set, batch_size=batch_size, shuffle=False) if val_set is not None else None
    )

    # Move the per-component tables to device once (they're shared across
    # all batches; per-batch index_select pulls rows by player_idx).
    train_qp = train_qg = train_ql = None
    val_qp = val_qg = val_ql = None
    if learnable_weights is not None:
        assert train_set.log_qp_table is not None
        assert train_set.log_qg_table is not None
        assert train_set.log_ql_vector is not None
        train_qp = train_set.log_qp_table.to(dev)
        train_qg = train_set.log_qg_table.to(dev)
        train_ql = train_set.log_ql_vector.to(dev)
        if val_set is not None:
            assert val_set.log_qp_table is not None
            assert val_set.log_qg_table is not None
            assert val_set.log_ql_vector is not None
            val_qp = val_set.log_qp_table.to(dev)
            val_qg = val_set.log_qg_table.to(dev)
            val_ql = val_set.log_ql_vector.to(dev)

    train_qd: Tensor | None = None
    val_qd: Tensor | None = None
    if learnable_defensive_scale is not None:
        assert train_set.log_qd_table is not None
        train_qd = train_set.log_qd_table.to(dev)
        if val_set is not None:
            assert val_set.log_qd_table is not None
            val_qd = val_set.log_qd_table.to(dev)

    param_groups: list[dict[str, object]] = [
        {"params": list(encoder.parameters())},
        {"params": list(decoder.parameters())},
    ]
    if learnable_weights is not None:
        param_groups.append({"params": list(learnable_weights.parameters())})
    if learnable_temperature is not None:
        param_groups.append({"params": list(learnable_temperature.parameters())})
    if learnable_defensive_scale is not None:
        param_groups.append({"params": list(learnable_defensive_scale.parameters())})
    if adaptive_prior is not None:
        param_groups.append({"params": list(adaptive_prior.parameters())})
    optimizer = torch.optim.Adam(
        param_groups,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = TrainHistory()
    best_val: float = float("inf")
    best_state: dict[str, dict[str, Tensor]] | None = None

    for epoch in range(n_epochs):
        train_nll = _epoch(
            encoder,
            decoder,
            train_loader,
            optimizer,
            device=dev,
            learnable_weights=learnable_weights,
            learnable_temperature=learnable_temperature,
            learnable_defensive_scale=learnable_defensive_scale,
            adaptive_prior=adaptive_prior,
            lambda_entropy=lambda_entropy,
            lambda_ess=lambda_ess,
            log_qp_table=train_qp,
            log_qg_table=train_qg,
            log_ql_vector=train_ql,
            log_qd_table=train_qd,
        )
        history.train_nll.append(train_nll)

        val_nll: float | None = None
        if val_loader is not None:
            val_nll = _epoch(
                encoder,
                decoder,
                val_loader,
                None,
                device=dev,
                learnable_weights=learnable_weights,
                learnable_temperature=learnable_temperature,
                learnable_defensive_scale=learnable_defensive_scale,
                adaptive_prior=adaptive_prior,
                lambda_entropy=0.0,  # regularizers are train-only
                lambda_ess=0.0,
                log_qp_table=val_qp,
                log_qg_table=val_qg,
                log_ql_vector=val_ql,
                log_qd_table=val_qd,
            )
            history.val_nll.append(val_nll)

            if val_nll < best_val:
                best_val = val_nll
                history.best_epoch = epoch + 1
                if restore_best_val:
                    best_state = {
                        "encoder": copy.deepcopy(encoder.state_dict()),
                        "decoder": copy.deepcopy(decoder.state_dict()),
                    }
                    if learnable_weights is not None:
                        best_state["weights"] = copy.deepcopy(learnable_weights.state_dict())
                    if learnable_temperature is not None:
                        best_state["temperature"] = copy.deepcopy(
                            learnable_temperature.state_dict()
                        )
                    if learnable_defensive_scale is not None:
                        best_state["defensive_scale"] = copy.deepcopy(
                            learnable_defensive_scale.state_dict()
                        )
                    if adaptive_prior is not None:
                        best_state["adaptive_prior"] = copy.deepcopy(adaptive_prior.state_dict())

        if progress:
            msg = f"epoch {epoch + 1}/{n_epochs}  train_nll={train_nll:.4f}"
            if val_nll is not None:
                marker = "  *" if history.best_epoch == epoch + 1 else ""
                msg += f"  val_nll={val_nll:.4f}{marker}"
            if learnable_weights is not None:
                w = learnable_weights.weights_as_floats()
                msg += f"  a_p={w['a_p']:.3f} a_g={w['a_g']:.3f} a_0={w['a_0']:.3f}"
            # For context-conditioned scalars, log the mean over a small
            # reference batch (the first 256 train rows). For scalar mode,
            # tau_as_float / alpha_as_float ignore x_n.
            x_n_ref: Tensor | None = None
            if needs_context:
                ref_n = min(256, len(train_set))
                x_n_ref = train_set.context_features[:ref_n].to(dev)
            if learnable_temperature is not None:
                tau_val = (
                    learnable_temperature.tau_as_float(x_n_ref)
                    if learnable_temperature.context_dim > 0
                    else learnable_temperature.tau_as_float()
                )
                label = "tau(x).mean" if learnable_temperature.context_dim > 0 else "tau"
                msg += f"  {label}={tau_val:.3f}"
            if learnable_defensive_scale is not None:
                a_def_val = (
                    learnable_defensive_scale.alpha_as_float(x_n_ref)
                    if learnable_defensive_scale.context_dim > 0
                    else learnable_defensive_scale.alpha_as_float()
                )
                label = "a_def(x).mean" if learnable_defensive_scale.context_dim > 0 else "a_def"
                msg += f"  {label}={a_def_val:.3f}"
            if adaptive_prior is not None:
                p = adaptive_prior.relevance.params_as_floats()
                msg += (
                    f"  bq={p['beta_q']:+.2f} bm={p['beta_m']:+.2f} "
                    f"bt={p['beta_t']:+.2f} bo={p['beta_o']:+.2f} "
                    f"lg={p['lambda_g']:+.2f}"
                )
            print(msg)

    if restore_best_val and best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])
        if learnable_weights is not None and "weights" in best_state:
            learnable_weights.load_state_dict(best_state["weights"])
        if learnable_temperature is not None and "temperature" in best_state:
            learnable_temperature.load_state_dict(best_state["temperature"])
        if adaptive_prior is not None and "adaptive_prior" in best_state:
            adaptive_prior.load_state_dict(best_state["adaptive_prior"])
        if learnable_defensive_scale is not None and "defensive_scale" in best_state:
            learnable_defensive_scale.load_state_dict(best_state["defensive_scale"])
        if progress and history.best_epoch is not None:
            print(
                f"[best] restored encoder/decoder from epoch "
                f"{history.best_epoch} (val_nll={best_val:.4f})"
            )

    return history
