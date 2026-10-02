"""Tests for :class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`.

Covers output shapes, gradient flow, constructor validation, the cold-start floor, and
identity at initialization for every optional branch (residual tilt, pooling gate,
opponent and matchup reweighting, kernel-shape fields, usage and count residuals,
within-game GRU, causal zone bias), plus independence of the support weights from the
observed shot location.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud import CourtGrid
from shotcloud.data import ContextEncoder
from shotcloud.data.context import CONTEXT_DIM
from shotcloud.data.player_traits import build_player_traits_table
from shotcloud.data.role_profile import build_role_profiles
from shotcloud.data.snapshots import build_snapshot_store_from_shots
from shotcloud.kde.adaptive import AdaptiveKDE
from shotcloud.models.analogue_retrieval import build_analogue_cache
from shotcloud.models.collaborative_kde import CollaborativeKDE
from shotcloud.models.context_residual import ContextResidualEncoder
from shotcloud.models.continuous_mixture_spatial import (
    ContinuousMixtureOutputs,
    ContinuousMixtureSpatial,
)
from shotcloud.models.location_embedding import LocationEmbedding
from shotcloud.training.dataset import PlayerVocab


def _build_setup(seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=14, ny=12)
    base_date = pd.Timestamp("2024-01-01")
    n_players = 6
    rows: list[dict[str, object]] = []
    for pid in range(1, n_players + 1):
        cx, cy = (0.0, 5.0) if pid <= n_players // 2 else (2.0, 8.0)
        for i in range(35):
            rows.append(
                {
                    "x": float(cx + rng.normal(0, 3)),
                    "y": float(cy + rng.normal(0, 3)),
                    "player_id": pid,
                    "opponent": "BOS",
                    "made": int(rng.random() < 0.5),
                    "period": (i % 4) + 1,
                    "time_remaining_sec": int(60 * (i % 48)),
                    "date": base_date + pd.Timedelta(days=i),
                }
            )
    shots = pd.DataFrame(rows)
    anchors = [np.datetime64("2024-01-05", "D")]
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )
    enc = ContextEncoder.fit(shots)
    ctx = enc.transform(shots)
    akde = AdaptiveKDE(grid=grid, bandwidth=1.5, max_history=20, seed=seed)
    akde.fit(
        x=shots["x"].to_numpy(),
        y=shots["y"].to_numpy(),
        player_id=shots["player_id"].to_numpy(),
        context_features=ctx,
        date=shots["date"].to_numpy(),
    )
    vocab = PlayerVocab.from_ids(akde.players)
    vocab_ids = [int(pid) for pid in vocab.ids]
    bio_rows = [
        {
            "player_id": int(pid),
            "display_name": f"Player {pid}",
            "birthdate": pd.Timestamp("1990-01-15") + pd.Timedelta(days=i * 30),
            "height_inches": 72 + i,
            "weight_lbs": 190 + i * 5,
            "position_raw": "Guard" if i < n_players // 2 else "Center",
            "position_group": "SG" if i < n_players // 2 else "C",
            "status": "ok",
        }
        for i, pid in enumerate(akde.players)
    ]
    bio = pd.DataFrame(bio_rows)
    gl = shots[["player_id", "date"]].rename(columns={"date": "game_date"}).copy()
    gl["minutes"] = 25
    gl["fga"] = 8
    gl["fta"] = 3
    gl["tov"] = 1
    traits = build_player_traits_table(
        snapshot_store=store, vocab_ids=vocab_ids, bio_df=bio, game_logs_df=gl
    )
    cache = build_analogue_cache(traits, L=3, ensure_self=True)
    collab = CollaborativeKDE(
        adaptive_kde=akde,
        snapshot_store=store,
        traits_table=traits,
        analogue_cache=cache,
        vocab=vocab,
        grid=grid,
    )
    return {"collab": collab, "vocab": vocab, "ctx": ctx, "shots": shots, "grid": grid}


def _batch(setup: dict[str, object], n_batch: int = 4) -> dict[str, torch.Tensor]:
    vocab = setup["vocab"]
    ctx = setup["ctx"]
    n = min(n_batch, len(vocab))  # type: ignore[arg-type]
    player_idx = torch.arange(n, dtype=torch.long)
    snapshot_idx = torch.zeros(n, dtype=torch.long)
    x_n_raw = torch.from_numpy(ctx[:n]).float()  # type: ignore[index]
    x_n = x_n_raw.clone()
    shots = setup["shots"]
    shot_xy = torch.from_numpy(
        np.stack([shots["x"].to_numpy()[:n], shots["y"].to_numpy()[:n]], axis=-1).astype(np.float32)
    )
    return {
        "player_idx": player_idx,
        "snapshot_idx": snapshot_idx,
        "x_n_raw": x_n_raw,
        "x_n": x_n,
        "shot_xy": shot_xy,
    }


def test_forward_returns_outputs_with_correct_shapes() -> None:
    setup = _build_setup()
    spatial = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    batch = _batch(setup, n_batch=4)
    out = spatial(**batch)
    assert isinstance(out, ContinuousMixtureOutputs)
    b = batch["player_idx"].shape[0]
    L = setup["collab"].L  # type: ignore[attr-defined]
    R = setup["collab"].max_history  # type: ignore[attr-defined]
    assert out.log_lik.shape == (b,)
    assert out.log_weights.shape == (b, L * R)
    assert out.support_xy.shape == (b, L * R, 2)
    assert out.sigma.shape == (b,)
    assert out.support_mask.shape == (b, L * R)
    assert torch.isfinite(out.log_lik).all()


def test_forward_with_residual_changes_log_lik_only_off_step_zero() -> None:
    """A zero-initialized ``LocationEmbedding`` makes the residual tilt vanish, so the
    log-lik equals the residual-free model at initialization."""
    setup = _build_setup()
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    residual = ContextResidualEncoder(rank=4, within_game_dim=0)
    loc = LocationEmbedding(rank=4, zero_init=True)
    spatial_on = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=residual,
        location_embedding=loc,
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        log_lik_off = spatial_off(**batch).log_lik
        log_lik_on = spatial_on(**batch).log_lik
    torch.testing.assert_close(log_lik_off, log_lik_on, atol=1e-5, rtol=1e-5)


def test_gradient_flows_to_collab_and_residual_params() -> None:
    setup = _build_setup()
    residual = ContextResidualEncoder(rank=4, within_game_dim=0)
    loc = LocationEmbedding(rank=4, zero_init=False)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=residual,
        location_embedding=loc,
    )
    batch = _batch(setup, n_batch=3)
    out = spatial(**batch)
    loss = -out.log_lik.mean()
    loss.backward()
    # Collab scalar grads.
    assert setup["collab"].b_same.grad is not None  # type: ignore[attr-defined]
    # Residual encoder grads.
    assert residual.fc1.weight.grad is not None and residual.fc1.weight.grad.abs().sum() > 0
    # Location embedding grads (proj is non-zero init here).
    assert loc.proj.weight.grad is not None and loc.proj.weight.grad.abs().sum() > 0


def test_partial_defense_triple_rejected() -> None:
    """A cache-based ``defensive_field`` (e.g. ``ContinuousAdaptiveDefensiveField``)
    without both ``defensive_cache`` and ``defensive_features`` is rejected at
    construction. Zone-level reweighting has its own checks, tested below.
    """
    setup = _build_setup()
    with pytest.raises(ValueError, match="requires both defensive_cache"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            defensive_field=object(),  # type: ignore[arg-type]
            # missing defensive_cache + defensive_features
        )


def test_residual_encoder_without_location_embedding_rejected() -> None:
    setup = _build_setup()
    residual = ContextResidualEncoder(rank=4, within_game_dim=0)
    with pytest.raises(ValueError, match="both"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            residual_encoder=residual,
        )


def test_rank_mismatch_residual_vs_location_rejected() -> None:
    setup = _build_setup()
    residual = ContextResidualEncoder(rank=4, within_game_dim=0)
    loc = LocationEmbedding(rank=8)
    with pytest.raises(ValueError, match="rank"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            residual_encoder=residual,
            location_embedding=loc,
        )


def test_train_gibbs_continuous_mixture_path_runs_end_to_end() -> None:
    """One epoch of ``train_gibbs`` with ``spatial_likelihood='continuous_mixture'`` and
    the residual tilt records finite mixture diagnostics and NaN grid-mode series."""
    from shotcloud.models import ContextMLP, NegBinCountHead, TimingSoftmaxHead
    from shotcloud.training import GibbsShotDataset, train_gibbs

    setup = _build_setup()
    train_set = (
        GibbsShotDataset(
            shots_df=setup["shots"],  # type: ignore[arg-type]
            snapshot_store=setup["collab"].offensive_prior.__dict__.get("snapshot_store"),  # type: ignore[attr-defined]
            grid=setup["grid"],  # type: ignore[arg-type]
            player_vocab=setup["vocab"],  # type: ignore[arg-type]
            opp_vocab=None,
            context_encoder=ContextEncoder.fit(setup["shots"]),  # type: ignore[arg-type]
        )
        if False
        else None
    )  # unused placeholder
    # Build the dataset from a fresh snapshot store over the fixture shots.
    from shotcloud.data.role_profile import build_role_profiles
    from shotcloud.data.snapshots import build_snapshot_store_from_shots

    shots_df = setup["shots"]  # type: ignore[assignment]
    grid = setup["grid"]  # type: ignore[assignment]
    enc = ContextEncoder.fit(shots_df)
    anchors = [np.datetime64("2024-01-05", "D")]
    store = build_snapshot_store_from_shots(
        shots_df,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )
    # Add game_id to shots_df for the dataset.
    shots_df = shots_df.copy()
    shots_df["game_id"] = shots_df["player_id"].astype(str) + "_g"
    train_set = GibbsShotDataset(
        shots_df=shots_df,
        snapshot_store=store,
        grid=grid,
        player_vocab=setup["vocab"],  # type: ignore[arg-type]
        opp_vocab=None,
        context_encoder=enc,
    )
    residual_encoder = ContextResidualEncoder(rank=4, within_game_dim=0)
    location_embedding = LocationEmbedding(rank=4)
    history = train_gibbs(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        count_head=NegBinCountHead(),
        timing_head=TimingSoftmaxHead(),
        context_mlp=ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True),
        train_set=train_set,
        grid=grid,
        residual_encoder=residual_encoder,
        location_embedding=location_embedding,
        spatial_likelihood="continuous_mixture",
        lambda_timing=0.0,
        lambda_count=0.0,
        n_epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        progress=False,
        restore_best_val=False,
    )
    assert len(history.train_spatial_mix_nll) == 1
    assert np.isfinite(history.train_spatial_mix_nll[0])
    assert np.isfinite(history.train_support_entropy[0])
    assert np.isfinite(history.train_sigma_mean[0])
    # Grid-mode series should be NaN in continuous_mixture mode.
    assert np.isnan(history.train_spatial_cell[0])
    assert np.isnan(history.train_spatial_continuous[0])
    # No pooling gate → λ diagnostic stays NaN.
    assert np.isnan(history.train_gate_lambda_mean[0])


def test_train_gibbs_continuous_mixture_with_pooling_gate_runs_end_to_end() -> None:
    """Training with a ``PoolingGate`` moves the mixture NLL and records a per-epoch
    mean λ in [0, 1]."""
    from shotcloud.data.role_profile import build_role_profiles
    from shotcloud.data.snapshots import build_snapshot_store_from_shots
    from shotcloud.models import ContextMLP, NegBinCountHead, TimingSoftmaxHead
    from shotcloud.models.pooling_gate import PoolingGate
    from shotcloud.training import GibbsShotDataset, train_gibbs

    setup = _build_setup()
    shots_df = setup["shots"].copy()  # type: ignore[union-attr]
    grid = setup["grid"]  # type: ignore[assignment]
    enc = ContextEncoder.fit(shots_df)
    store = build_snapshot_store_from_shots(
        shots_df,
        [np.datetime64("2024-01-05", "D")],
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )
    shots_df["game_id"] = shots_df["player_id"].astype(str) + "_g"
    train_set = GibbsShotDataset(
        shots_df=shots_df,
        snapshot_store=store,
        grid=grid,
        player_vocab=setup["vocab"],  # type: ignore[arg-type]
        opp_vocab=None,
        context_encoder=enc,
    )
    gate = PoolingGate(history_dim=0)
    history = train_gibbs(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        count_head=NegBinCountHead(),
        timing_head=TimingSoftmaxHead(),
        context_mlp=ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True),
        train_set=train_set,
        grid=grid,
        pooling_gate=gate,
        spatial_likelihood="continuous_mixture",
        lambda_timing=0.0,
        lambda_count=0.0,
        n_epochs=3,
        batch_size=16,
        learning_rate=1e-3,
        progress=False,
        restore_best_val=False,
    )
    assert len(history.train_spatial_mix_nll) == 3
    assert all(np.isfinite(history.train_spatial_mix_nll))
    # λ diagnostic populated and in the unit interval.
    assert all(np.isfinite(history.train_gate_lambda_mean))
    assert all(0.0 <= v <= 1.0 for v in history.train_gate_lambda_mean)
    # Loss moved.
    assert history.train_spatial_mix_nll[-1] != history.train_spatial_mix_nll[0]


def test_train_gibbs_pooling_gate_rejects_non_continuous_mixture() -> None:
    """A pooling gate with any spatial likelihood other than ``continuous_mixture``
    raises."""
    from shotcloud.data.role_profile import build_role_profiles
    from shotcloud.data.snapshots import build_snapshot_store_from_shots
    from shotcloud.models import ContextMLP, NegBinCountHead, TimingSoftmaxHead
    from shotcloud.models.pooling_gate import PoolingGate
    from shotcloud.training import GibbsShotDataset, train_gibbs

    setup = _build_setup()
    shots_df = setup["shots"].copy()  # type: ignore[union-attr]
    grid = setup["grid"]  # type: ignore[assignment]
    enc = ContextEncoder.fit(shots_df)
    store = build_snapshot_store_from_shots(
        shots_df,
        [np.datetime64("2024-01-05", "D")],
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )
    shots_df["game_id"] = shots_df["player_id"].astype(str) + "_g"
    train_set = GibbsShotDataset(
        shots_df=shots_df,
        snapshot_store=store,
        grid=grid,
        player_vocab=setup["vocab"],  # type: ignore[arg-type]
        opp_vocab=None,
        context_encoder=enc,
    )
    with pytest.raises(ValueError, match="pooling_gate is only supported"):
        train_gibbs(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            count_head=NegBinCountHead(),
            timing_head=TimingSoftmaxHead(),
            context_mlp=ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True),
            train_set=train_set,
            grid=grid,
            pooling_gate=PoolingGate(history_dim=0),
            spatial_likelihood="mode_mixture",
            n_epochs=1,
            batch_size=16,
            progress=False,
            restore_best_val=False,
        )


def test_train_gibbs_continuous_mixture_rejects_legacy_offensive_prior() -> None:
    """The continuous-mixture path requires a ``CollaborativeKDE`` offensive prior and
    rejects the grid-only ``AdaptiveOffensivePrior``."""
    from shotcloud import CourtGrid
    from shotcloud.data.role_profile import build_role_profiles
    from shotcloud.data.snapshots import build_snapshot_store_from_shots
    from shotcloud.legacy_pivot.adaptive_prior import AdaptiveOffensivePrior
    from shotcloud.legacy_pivot.archetypes import ArchetypeDictionary, ArchetypeMixture
    from shotcloud.models import ContextMLP, NegBinCountHead, RelevanceScore, TimingSoftmaxHead
    from shotcloud.training import GibbsShotDataset, PlayerVocab, train_gibbs

    rng = np.random.default_rng(0)
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=10)
    base = pd.Timestamp("2024-01-01")
    rows = []
    for pid in range(1, 5):
        for i in range(30):
            rows.append(
                {
                    "x": float(rng.normal(0, 5)),
                    "y": float(rng.uniform(0, 25)),
                    "player_id": pid,
                    "opponent": "BOS",
                    "made": 0,
                    "period": (i % 4) + 1,
                    "time_remaining_sec": int(60 * i),
                    "date": base + pd.Timedelta(days=i),
                    "game_id": f"g{pid}_{i // 6}",
                }
            )
    shots = pd.DataFrame(rows)
    anchors = [np.datetime64("2024-01-05", "D")]
    n_archetypes = 2
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
        archetype_fit_fn=lambda _sub, _t: np.full(
            (n_archetypes, grid.n_cells), 1.0 / grid.n_cells, dtype=np.float32
        ),
    )
    enc = ContextEncoder.fit(shots)
    ctx = enc.transform(shots)
    from shotcloud.kde.adaptive import AdaptiveKDE

    akde = AdaptiveKDE(grid=grid, bandwidth=1.5, max_history=20)
    akde.fit(
        x=shots["x"].to_numpy(),
        y=shots["y"].to_numpy(),
        player_id=shots["player_id"].to_numpy(),
        context_features=ctx,
        date=shots["date"].to_numpy(),
    )
    vocab = PlayerVocab.from_ids(akde.players)
    arch_dict = ArchetypeDictionary.from_snapshot_store(store)
    arch_mix = ArchetypeMixture(n_archetypes=arch_dict.n_archetypes)
    offensive_prior = AdaptiveOffensivePrior(
        akde, store, arch_dict, arch_mix, vocab, RelevanceScore()
    )
    train_set = GibbsShotDataset(
        shots_df=shots,
        snapshot_store=store,
        grid=grid,
        player_vocab=vocab,
        opp_vocab=None,
        context_encoder=enc,
    )
    with pytest.raises(TypeError, match="CollaborativeKDE"):
        train_gibbs(
            offensive_prior=offensive_prior,
            count_head=NegBinCountHead(),
            timing_head=TimingSoftmaxHead(),
            context_mlp=ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True),
            train_set=train_set,
            grid=grid,
            spatial_likelihood="continuous_mixture",
            n_epochs=1,
            batch_size=16,
            progress=False,
        )


def test_cold_start_row_uses_floor_log_lik() -> None:
    """Rows with no causal support receive the cold-start floor (uniform-court
    density), not NaN."""
    setup = _build_setup()
    spatial = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    batch = _batch(setup, n_batch=3)
    # Force every causal mask to be zero by setting anchor before any shot.
    with torch.no_grad():
        setup["collab"].anchor_dates.fill_(  # type: ignore[attr-defined]
            int(setup["collab"].pool_dates.min().item()) - 1  # type: ignore[attr-defined]
        )
    with torch.no_grad():
        out = spatial(**batch)
    assert torch.isfinite(out.log_lik).all()
    # Should match the floor.
    np.testing.assert_allclose(
        out.log_lik.numpy(),
        np.full(batch["player_idx"].shape[0], spatial.cold_start_log_lik_floor),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# History-dependent pooling gate
# ---------------------------------------------------------------------------


def _component_loglik(setup: dict[str, object], batch: dict[str, torch.Tensor], own: bool):
    """Reference own-only or pooled-only mixture log-lik for the gate-limit tests."""
    from shotcloud.evaluation.support_masks import support_source_masks
    from shotcloud.training.spatial_losses import continuous_mixture_loglik

    collab = setup["collab"]
    with torch.no_grad():
        out = collab.forward_continuous(  # type: ignore[attr-defined]
            batch["player_idx"], batch["snapshot_idx"], batch["x_n_raw"], batch["x_n"]
        )
        r = out.support_logits.shape[1] // out.analogue_idx.shape[1]
        masks = support_source_masks(
            analogue_idx=out.analogue_idx,
            player_idx=batch["player_idx"],
            n_shots_per_analogue=r,
            valid_mask=out.support_mask,
        )
        subset = masks.own if own else masks.pooled
        return continuous_mixture_loglik(
            out.support_logits,
            out.support_xy,
            batch["shot_xy"],
            out.sigma,
            weights_are_log_probs=False,
            support_mask=subset,
        )


def test_gate_populates_lambda_in_unit_interval() -> None:
    from shotcloud.models.pooling_gate import PoolingGate

    setup = _build_setup()
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        pooling_gate=PoolingGate(),
    )
    batch = _batch(setup, n_batch=4)
    out = spatial(**batch)
    assert out.gate_lambda is not None
    assert out.gate_lambda.shape == (batch["player_idx"].shape[0],)
    assert (out.gate_lambda >= 0.0).all() and (out.gate_lambda <= 1.0).all()
    assert torch.isfinite(out.log_lik).all()


def test_gate_lambda_one_recovers_own_only_loglik() -> None:
    """Forcing λ→1 (huge intercept) makes the gated log-lik equal the
    own-only continuous-mixture log-lik."""
    from shotcloud.models.pooling_gate import PoolingGate

    setup = _build_setup()
    gate = PoolingGate()
    with torch.no_grad():
        gate.b0.fill_(40.0)  # logit ≫ 0 → λ ≈ 1
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        pooling_gate=gate,
    )
    batch = _batch(setup, n_batch=4)
    with torch.no_grad():
        out = spatial(**batch)
    own_ref = _component_loglik(setup, batch, own=True)
    torch.testing.assert_close(out.log_lik, own_ref, atol=1e-4, rtol=1e-4)


def test_gate_lambda_zero_recovers_pooled_only_loglik() -> None:
    """Forcing λ→0 (huge negative intercept) makes the gated log-lik
    equal the pooled-only continuous-mixture log-lik."""
    from shotcloud.models.pooling_gate import PoolingGate

    setup = _build_setup()
    gate = PoolingGate()
    with torch.no_grad():
        gate.b0.fill_(-40.0)  # logit ≪ 0 → λ ≈ 0
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        pooling_gate=gate,
    )
    batch = _batch(setup, n_batch=4)
    with torch.no_grad():
        out = spatial(**batch)
    pooled_ref = _component_loglik(setup, batch, own=False)
    torch.testing.assert_close(out.log_lik, pooled_ref, atol=1e-4, rtol=1e-4)


def test_gated_support_log_weights_sum_to_one() -> None:
    """The effective per-support weights (λ·ω_own on own slots,
    (1-λ)·ω_pooled on pooled slots) form a proper distribution over M."""
    from shotcloud.models.pooling_gate import PoolingGate

    setup = _build_setup()
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        pooling_gate=PoolingGate(),
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out = spatial(**batch)
    total = out.support_log_weights.exp().sum(dim=-1)
    torch.testing.assert_close(total, torch.ones_like(total), atol=1e-4, rtol=1e-4)


def test_gated_gradient_flows_to_gate_and_collab() -> None:
    from shotcloud.models.pooling_gate import PoolingGate

    setup = _build_setup()
    gate = PoolingGate()
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        pooling_gate=gate,
    )
    batch = _batch(setup, n_batch=4)
    out = spatial(**batch)
    (-out.log_lik.mean()).backward()
    assert gate.b0.grad is not None and gate.b0.grad.abs() > 0
    assert gate.b_h.grad is not None
    assert setup["collab"].b_same.grad is not None  # type: ignore[attr-defined]


def test_gated_cold_start_uses_floor() -> None:
    from shotcloud.models.pooling_gate import PoolingGate

    setup = _build_setup()
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        pooling_gate=PoolingGate(),
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        setup["collab"].anchor_dates.fill_(  # type: ignore[attr-defined]
            int(setup["collab"].pool_dates.min().item()) - 1  # type: ignore[attr-defined]
        )
        out = spatial(**batch)
    assert torch.isfinite(out.log_lik).all()
    np.testing.assert_allclose(
        out.log_lik.numpy(),
        np.full(batch["player_idx"].shape[0], spatial.cold_start_log_lik_floor),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Continuous adaptive defensive field
# ---------------------------------------------------------------------------


def _build_defense_setup(
    *,
    cold_start_anchor: bool = False,
    beta_init: float = 1e-3,
    seed: int = 0,
) -> dict[str, object]:
    """Extend the offensive fixture with defense pieces: opponent vocabulary,
    defensive retrieval cache, defense features, and the defensive field.

    ``cold_start_anchor=True`` places the snapshot anchor before every synthetic shot,
    so each (opponent, snapshot) cell is cold-start and the defense contribution is
    identically zero.
    """
    from shotcloud.features.defense_features import (
        DEFENSE_FEATURE_DIM,
        DefenseFeaturesConfig,
        build_defense_features,
    )
    from shotcloud.models.continuous_adaptive_defensive import (
        ContinuousAdaptiveDefensiveField,
    )
    from shotcloud.models.defensive_retrieval_cache import (
        DefensiveRetrievalCacheConfig,
        build_defensive_retrieval_cache,
    )
    from shotcloud.training.dataset import OpponentVocab

    setup = _build_setup(seed=seed)
    shots = setup["shots"]
    opp_ids = sorted(shots["opponent"].astype(str).unique().tolist())  # type: ignore[union-attr]
    opp_vocab = OpponentVocab.from_ids(opp_ids)
    # Anchor after the fixture shots (populated cache) or before them (all cold-start).
    anchor_str = "2023-01-01" if cold_start_anchor else "2024-01-25"
    anchor = int(np.datetime64(anchor_str, "D").astype(np.int64))
    anchors = np.array([anchor], dtype=np.int64)
    cache_cfg = DefensiveRetrievalCacheConfig(
        shots_fingerprint="fixture_v0",
        anchor_dates=tuple(int(d) for d in anchors),
        defensive_support_max=8,
        defensive_recency_window_days=365,
    )
    cache = build_defensive_retrieval_cache(
        shots_df=shots,
        opp_vocab=opp_vocab,
        anchor_dates=anchors,
        config=cache_cfg,
    )
    feat_cfg = DefenseFeaturesConfig(
        shots_fingerprint="fixture_v0",
        anchor_dates=tuple(int(d) for d in anchors),
        window_days=365,
        half_life_days=90.0,
    )
    features = build_defense_features(
        shots_df=shots,
        opp_vocab=opp_vocab,
        anchor_dates=anchors,
        config=feat_cfg,
    )
    field = ContinuousAdaptiveDefensiveField(
        n_opponents=len(opp_vocab),
        context_dim=CONTEXT_DIM,
        within_game_dim=0,
        defense_feature_dim=DEFENSE_FEATURE_DIM,
        opp_embed_dim=4,
        query_hidden_dim=16,
        key_hidden_dim=16,
        proj_dim=8,
        beta_init=beta_init,
        query_chunk_size=16,
    )
    setup["opp_vocab"] = opp_vocab
    setup["defensive_field"] = field
    setup["defensive_cache"] = cache
    setup["defensive_features"] = features
    return setup


def _defense_batch(setup: dict[str, object], n_batch: int = 4) -> dict[str, torch.Tensor]:
    """Extend the base batch with ``opp_idx`` (all-zeros since the
    fixture has a single opponent ``"BOS"``)."""
    batch = _batch(setup, n_batch=n_batch)
    batch["opp_idx"] = torch.zeros(batch["player_idx"].shape[0], dtype=torch.long)
    return batch


def test_defense_none_preserves_existing_behavior() -> None:
    """Without a defensive field the wrapper reports no defense diagnostics and a
    finite log-lik."""
    setup = _build_setup()
    spatial = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    assert not spatial.has_defense
    batch = _batch(setup, n_batch=3)
    out = spatial(**batch)
    assert torch.isfinite(out.log_lik).all()
    assert out.defense_logits is None
    assert out.defense_cold_start is None


def test_defense_beta_zero_equals_no_defense_exactly() -> None:
    """A defensive field with ``β_D = 0`` reproduces the no-defense per-row
    log-likelihood exactly."""
    setup = _build_defense_setup(beta_init=0.0)
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_on = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        defensive_field=setup["defensive_field"],  # type: ignore[arg-type]
        defensive_cache=setup["defensive_cache"],  # type: ignore[arg-type]
        defensive_features=setup["defensive_features"],  # type: ignore[arg-type]
    )
    batch = _defense_batch(setup, n_batch=3)
    with torch.no_grad():
        out_off = spatial_off(**{k: v for k, v in batch.items() if k != "opp_idx"})
        out_on = spatial_on(**batch)
    torch.testing.assert_close(out_off.log_lik, out_on.log_lik, atol=1e-6, rtol=1e-6)


def test_defense_warm_init_produces_finite_loglik_close_to_no_defense() -> None:
    """At the training-default ``β_D = 1e-3``, the wrapper's
    log-likelihood is finite and only slightly different from the
    no-defense path."""
    setup = _build_defense_setup(beta_init=1e-3)
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_on = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        defensive_field=setup["defensive_field"],  # type: ignore[arg-type]
        defensive_cache=setup["defensive_cache"],  # type: ignore[arg-type]
        defensive_features=setup["defensive_features"],  # type: ignore[arg-type]
    )
    batch = _defense_batch(setup, n_batch=3)
    with torch.no_grad():
        out_off = spatial_off(**{k: v for k, v in batch.items() if k != "opp_idx"})
        out_on = spatial_on(**batch)
    assert torch.isfinite(out_on.log_lik).all()
    # At β=1e-3 the per-row log-lik stays within 0.1 nats of the no-defense
    # baseline; the field is mean-centered per row, so its typical magnitude is
    # well below this bound.
    max_drift = float((out_on.log_lik - out_off.log_lik).abs().max().item())
    assert max_drift < 0.1, f"warm-init drift = {max_drift:.4f} nats — too large"
    # Diagnostic tensors populated.
    assert out_on.defense_logits is not None
    assert out_on.defense_logits.shape == out_on.collab.support_xy.shape[:2]
    assert out_on.defense_cold_start is not None


def test_gradients_flow_through_defensive_field() -> None:
    """With β_D > 0 the spatial loss reaches ``β_D``, the query and key networks, and
    the opponent embedding."""
    setup = _build_defense_setup(beta_init=1e-3)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        defensive_field=setup["defensive_field"],  # type: ignore[arg-type]
        defensive_cache=setup["defensive_cache"],  # type: ignore[arg-type]
        defensive_features=setup["defensive_features"],  # type: ignore[arg-type]
    )
    batch = _defense_batch(setup, n_batch=3)
    out = spatial(**batch)
    loss = -out.log_lik.mean()
    loss.backward()
    field = setup["defensive_field"]
    assert field.beta_D.grad is not None  # type: ignore[union-attr]
    assert field.beta_D.grad.abs().item() > 0  # type: ignore[union-attr]
    assert field.q_net[0].weight.grad is not None  # type: ignore[union-attr, index]
    assert field.q_net[0].weight.grad.abs().sum().item() > 0  # type: ignore[union-attr, index]
    assert field.k_net[0].weight.grad is not None  # type: ignore[union-attr, index]
    assert field.k_net[0].weight.grad.abs().sum().item() > 0  # type: ignore[union-attr, index]
    assert field.opp_embedding.weight.grad is not None  # type: ignore[union-attr]
    assert field.opp_embedding.weight.grad.abs().sum().item() > 0  # type: ignore[union-attr]


def test_defense_enters_before_own_pooled_subset_softmax() -> None:
    """Defense reweights support shots within the own and pooled subsets and leaves the
    gate's ``λ`` unchanged.

    Applied after the mixture combination, defense would only shift ``log_lik`` by a
    per-row constant. The test instead checks that ``defense_logits`` are per support
    shot, that zeroing ``β_D`` changes ``support_log_weights``, and that ``λ`` is the
    same with and without defense.
    """
    from shotcloud.models.pooling_gate import PoolingGate

    setup = _build_defense_setup(beta_init=1e-2)  # bigger β to make defense visible
    gate = PoolingGate()
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        defensive_field=setup["defensive_field"],  # type: ignore[arg-type]
        defensive_cache=setup["defensive_cache"],  # type: ignore[arg-type]
        defensive_features=setup["defensive_features"],  # type: ignore[arg-type]
        pooling_gate=gate,
    )
    batch = _defense_batch(setup, n_batch=3)
    with torch.no_grad():
        out = spatial(**batch)
    # Sanity: defense_logits is per-shot (B, M), not per-row scalar.
    # If defense entered AFTER mixture combination it would be a (B,)
    # tensor or wouldn't change support_log_weights at all.
    assert out.defense_logits is not None
    assert out.defense_logits.shape == out.support_log_weights.shape, (
        "defense_logits should be per-support-shot (B, M), not per-row"
    )
    # Construct a second wrapper with the SAME modules but β=0, and
    # confirm the support_log_weights differ — that proves defense
    # is influencing the within-subset normalization, not just an
    # external row-level constant.
    field_off = setup["defensive_field"]
    field_off.beta_D.data.zero_()  # type: ignore[union-attr]
    with torch.no_grad():
        out_off = spatial(**batch)
    # support_log_weights MUST differ between β≠0 and β=0 runs (the
    # defense reweights shots within each subset). If defense entered
    # after the mixture, support_log_weights would be identical.
    # Invalid slots are -inf in both, which subtract to NaN — filter
    # to valid slots before checking the diff.
    valid = out.support_mask
    diff_valid = (out.support_log_weights[valid] - out_off.support_log_weights[valid]).abs()
    assert diff_valid.max().item() > 1e-6, (
        "defense did not change support_log_weights — likely applied after "
        "subset softmax instead of before"
    )
    # And λ MUST be unchanged: defense is feasibility within a
    # subset, not a re-mixing knob.
    assert out.gate_lambda is not None and out_off.gate_lambda is not None
    torch.testing.assert_close(out.gate_lambda, out_off.gate_lambda, atol=1e-6, rtol=1e-6)


def test_cold_start_defensive_rows_match_no_defense() -> None:
    """When every row's defensive cache cell is cold-start (anchor before any
    allowed-shot data), the defensive field returns identically zero and the
    log-lik matches the no-defense baseline exactly."""
    setup = _build_defense_setup(cold_start_anchor=True, beta_init=1e-3)
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_on = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        defensive_field=setup["defensive_field"],  # type: ignore[arg-type]
        defensive_cache=setup["defensive_cache"],  # type: ignore[arg-type]
        defensive_features=setup["defensive_features"],  # type: ignore[arg-type]
    )
    batch = _defense_batch(setup, n_batch=3)
    with torch.no_grad():
        out_off = spatial_off(**{k: v for k, v in batch.items() if k != "opp_idx"})
        out_on = spatial_on(**batch)
    # All rows are cold-start in this fixture; defense_cold_start is
    # all True, defense_logits is all zero, log-lik exactly matches.
    assert out_on.defense_cold_start is not None
    assert out_on.defense_cold_start.all().item()
    assert out_on.defense_logits is not None
    assert (out_on.defense_logits == 0).all().item()
    torch.testing.assert_close(out_off.log_lik, out_on.log_lik, atol=1e-6, rtol=1e-6)


def test_missing_opp_idx_raises_when_defense_wired() -> None:
    """With defense wired, a forward call without ``opp_idx`` raises."""
    setup = _build_defense_setup(beta_init=1e-3)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        defensive_field=setup["defensive_field"],  # type: ignore[arg-type]
        defensive_cache=setup["defensive_cache"],  # type: ignore[arg-type]
        defensive_features=setup["defensive_features"],  # type: ignore[arg-type]
    )
    batch = _batch(setup, n_batch=3)  # no opp_idx
    with pytest.raises(ValueError, match="opp_idx"):
        spatial(**batch)


# ---------------------------------------------------------------------------
# Zone-level opponent reweighting (ZoneReweightingDefense)
# ---------------------------------------------------------------------------


def _build_zone_lite_setup(*, beta_init: float = 1e-3, seed: int = 0) -> dict[str, object]:
    """Counterpart of :func:`_build_defense_setup` for zone-level reweighting: a
    field and features, no cache."""
    from shotcloud.models.zone_defense_reweighting import ZoneReweightingDefense

    setup = _build_defense_setup(beta_init=beta_init, seed=seed)
    opp_vocab = setup["opp_vocab"]
    field = ZoneReweightingDefense(
        n_opponents=len(opp_vocab),  # type: ignore[arg-type]
        beta_init=beta_init,
    )
    setup["defensive_field_zone_lite"] = field
    return setup


def test_zone_lite_beta_zero_equals_no_defense_exactly() -> None:
    """Zone-level reweighting with ``β_D = 0`` reproduces the no-defense log-lik."""
    setup = _build_zone_lite_setup(beta_init=0.0)
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_on = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        defensive_field=setup["defensive_field_zone_lite"],  # type: ignore[arg-type]
        defensive_features=setup["defensive_features"],  # type: ignore[arg-type]
    )
    assert spatial_on.has_defense
    batch = _defense_batch(setup, n_batch=3)
    with torch.no_grad():
        out_off = spatial_off(**{k: v for k, v in batch.items() if k != "opp_idx"})
        out_on = spatial_on(**batch)
    torch.testing.assert_close(out_off.log_lik, out_on.log_lik, atol=1e-6, rtol=1e-6)


def test_zone_lite_rejects_cache_argument() -> None:
    """Zone-level reweighting does not consume a cache; passing ``defensive_cache``
    raises."""
    from shotcloud.models.zone_defense_reweighting import ZoneReweightingDefense

    setup = _build_defense_setup(beta_init=1e-3)
    field = ZoneReweightingDefense(n_opponents=len(setup["opp_vocab"]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not consume a defensive cache"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            defensive_field=field,
            defensive_cache=setup["defensive_cache"],  # type: ignore[arg-type]
            defensive_features=setup["defensive_features"],  # type: ignore[arg-type]
        )


def test_zone_lite_requires_features() -> None:
    """Zone-level reweighting requires ``defensive_features`` (it reads the
    centered-zone block)."""
    from shotcloud.models.zone_defense_reweighting import ZoneReweightingDefense

    setup = _build_defense_setup(beta_init=1e-3)
    field = ZoneReweightingDefense(n_opponents=len(setup["opp_vocab"]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires defensive_features"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            defensive_field=field,
            defensive_features=None,
        )


def test_zone_lite_warm_init_produces_finite_loglik() -> None:
    """Zone-level reweighting at ``β_D = 1e-3`` gives a finite log-lik close to the
    no-defense path."""
    setup = _build_zone_lite_setup(beta_init=1e-3)
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_on = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        defensive_field=setup["defensive_field_zone_lite"],  # type: ignore[arg-type]
        defensive_features=setup["defensive_features"],  # type: ignore[arg-type]
    )
    batch = _defense_batch(setup, n_batch=3)
    with torch.no_grad():
        out_off = spatial_off(**{k: v for k, v in batch.items() if k != "opp_idx"})
        out_on = spatial_on(**batch)
    assert torch.isfinite(out_on.log_lik).all()
    # Small β → small departure from the no-defense path.
    assert torch.allclose(out_off.log_lik, out_on.log_lik, atol=1e-2)


def test_zone_lite_defense_logits_shape_and_diagnostics() -> None:
    """Zone-level reweighting exposes ``defense_logits`` of shape (B, M) and a boolean
    ``defense_cold_start`` of shape (B,)."""
    setup = _build_zone_lite_setup(beta_init=1e-2)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        defensive_field=setup["defensive_field_zone_lite"],  # type: ignore[arg-type]
        defensive_features=setup["defensive_features"],  # type: ignore[arg-type]
    )
    batch = _defense_batch(setup, n_batch=3)
    with torch.no_grad():
        out = spatial(**batch)
    assert out.defense_logits is not None
    assert out.defense_logits.shape[0] == 3
    assert out.defense_cold_start is not None
    assert out.defense_cold_start.shape == (3,)
    assert out.defense_cold_start.dtype == torch.bool


def test_zone_lite_uses_opp_idx() -> None:
    """Rows that differ only in ``opp_idx`` receive different ``defense_logits``:
    zone-level reweighting depends on opponent identity through the per-row features.

    The fixture has a single opponent, so the test synthesizes two opponents with
    different centered-zone blocks and routes rows via ``opp_idx`` ∈ {0, 1}.
    """
    from shotcloud.features.defense_features import (
        _ZONE_CENTERED_SLICE,
        DEFENSE_FEATURE_DIM,
        DefenseFeatures,
        DefenseFeaturesConfig,
    )
    from shotcloud.models.zone_defense_reweighting import ZoneReweightingDefense

    setup = _build_defense_setup(beta_init=1.0)
    # Synthesize a 2-opponent features tensor: opp 0 = all zeros (cold),
    # opp 1 = nonzero centered-zone block.
    n_snap = 1
    feat = torch.zeros(2, n_snap, DEFENSE_FEATURE_DIM)
    feat[1, 0, _ZONE_CENTERED_SLICE] = torch.tensor([0.1, -0.1, 0.0, 0.2, -0.2, 0.05, -0.05, 0.0])
    fake_features = DefenseFeatures(
        features=feat,
        feature_names=tuple(["f"] * DEFENSE_FEATURE_DIM),
        config=DefenseFeaturesConfig(
            shots_fingerprint="fake",
            anchor_dates=(0,),
            window_days=365,
            half_life_days=90.0,
        ),
    )
    field = ZoneReweightingDefense(n_opponents=2, beta_init=1.0)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        defensive_field=field,
        defensive_features=fake_features,
    )
    # Two-row batch with snapshot_idx=0; differ only by opp_idx.
    batch = _batch(setup, n_batch=2)
    # Force snapshot_idx=0 so it indexes into our synthesized features.
    batch["snapshot_idx"] = torch.zeros_like(batch["snapshot_idx"])
    batch["opp_idx"] = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        out = spatial(**batch)
    assert out.defense_logits is not None
    d_row0 = out.defense_logits[0]
    d_row1 = out.defense_logits[1]
    # Opp 0 has all-zero features → the defense contribution is exactly zero.
    assert torch.allclose(d_row0, torch.zeros_like(d_row0))
    # Opp 1 has a nonzero centered-zone block → nonzero contribution.
    assert not torch.allclose(d_row1, torch.zeros_like(d_row1))


# ---------------------------------------------------------------------------
# Matchup reweighting (MatchupReweightingDefense)
# ---------------------------------------------------------------------------


def _build_matchup_setup(*, beta_init: float = 1e-3, seed: int = 0) -> dict[str, object]:
    """Synthesize the fixture pieces matchup reweighting needs: a small
    :class:`MatchupFeatures` aligned with the offensive fixture's
    player/snapshot/opponent dimensions, plus the
    :class:`MatchupReweightingDefense` module.

    The Δ̂ tensor is hand-built with nonzero values on a few cells so
    the wrapper's gather + β·Δ̂ path is exercised; the rest is zero
    (cold-start), letting tests check both the warm and cold branches
    in one fixture.
    """
    from shotcloud.data.zones import N_ZONES as _N_ZONES
    from shotcloud.features.matchup_features import (
        MatchupFeatures,
        MatchupFeaturesConfig,
    )
    from shotcloud.models.zone_defense_reweighting import MatchupReweightingDefense
    from shotcloud.training.dataset import OpponentVocab

    setup = _build_setup(seed=seed)
    shots = setup["shots"]
    opp_ids = sorted(shots["opponent"].astype(str).unique().tolist())  # type: ignore[union-attr]
    opp_vocab = OpponentVocab.from_ids(opp_ids)
    n_players = len(setup["vocab"])  # type: ignore[arg-type]
    n_snaps = 1
    n_opps = max(2, len(opp_vocab))  # ensure at least 2 for opp_idx={0,1}
    # Δ̂: zeros everywhere except a clear nonzero signal for
    # (player=0, snap=0, opp=0, zone=TopKey3=7) = +0.5 and
    # (player=0, snap=0, opp=1, zone=RA=0) = -0.4. Other cells stay
    # at zero (cold-start). N^eff matches: the warm cells have N^eff=5
    # so a downstream ESS-bucket diagnostic can distinguish them.
    delta_hat = torch.zeros((n_players, n_snaps, n_opps, _N_ZONES))
    delta_int = torch.zeros_like(delta_hat)
    n_eff = torch.zeros((n_players, n_snaps, n_opps))
    # Warm cell #1: (player=0, snap=0, opp=0). Set a recognizable
    # per-zone signature with at least one nonzero entry per zone
    # so any randomly-distributed support_xy lands on a nonzero Δ̂_z
    # and the matchup contribution is exercised regardless of where
    # the synthetic support shots fall.
    delta_hat[0, 0, 0, :] = torch.tensor([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, 0.5])
    delta_int[0, 0, 0, :] = delta_hat[0, 0, 0, :].clone()
    n_eff[0, 0, 0] = 5.0
    if n_opps > 1:
        # Warm cell #2: (player=0, snap=0, opp=1). Different signature
        # so test cases that route to opp=1 see a distinct contribution.
        delta_hat[0, 0, 1, :] = -delta_hat[0, 0, 0, :]
        delta_int[0, 0, 1, :] = -delta_int[0, 0, 0, :]
        n_eff[0, 0, 1] = 5.0
    cfg = MatchupFeaturesConfig(
        shots_fingerprint="matchup_fixture",
        traits_hash="th",
        opp_vocab_hash="ovh",
        anchor_dates=(0,),
        grouping_K=2,
    )
    features = MatchupFeatures(delta_hat=delta_hat, delta_int=delta_int, n_eff=n_eff, config=cfg)
    field = MatchupReweightingDefense(beta_init=beta_init)
    setup["matchup_opp_vocab"] = opp_vocab
    setup["matchup_features"] = features
    setup["matchup_field"] = field
    return setup


def _matchup_batch(setup: dict[str, object], n_batch: int = 4) -> dict[str, torch.Tensor]:
    """Base batch extended with ``opp_idx`` for the matchup path."""
    batch = _batch(setup, n_batch=n_batch)
    batch["opp_idx"] = torch.zeros(batch["player_idx"].shape[0], dtype=torch.long)
    return batch


def test_matchup_wrapper_requires_paired_field_and_features() -> None:
    setup = _build_matchup_setup()
    with pytest.raises(ValueError, match="must both be provided"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            matchup_field=setup["matchup_field"],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must both be provided"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            matchup_features=setup["matchup_features"],  # type: ignore[arg-type]
        )


def test_matchup_beta_zero_matches_no_matchup_path() -> None:
    """At ``β_match = 0`` the log-lik equals the no-matchup baseline (within float32
    rounding) and ``matchup_logits`` are zero."""
    setup = _build_matchup_setup(beta_init=0.0)
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_on = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        matchup_field=setup["matchup_field"],  # type: ignore[arg-type]
        matchup_features=setup["matchup_features"],  # type: ignore[arg-type]
    )
    batch = _matchup_batch(setup, n_batch=3)
    with torch.no_grad():
        out_off = spatial_off(**{k: v for k, v in batch.items() if k != "opp_idx"})
        out_on = spatial_on(**batch)
    torch.testing.assert_close(out_on.log_lik, out_off.log_lik, atol=1e-6, rtol=0)
    assert out_on.matchup_logits is not None
    assert torch.equal(out_on.matchup_logits, torch.zeros_like(out_on.matchup_logits))


def test_matchup_warm_init_changes_loglik_and_exposes_diagnostics() -> None:
    """With ``β_match > 0`` and a player-opponent cell with nonzero Δ̂, the log-lik
    differs from the ``β_match = 0`` baseline and the diagnostics are populated."""
    setup_off = _build_matchup_setup(beta_init=0.0)
    setup_on = _build_matchup_setup(beta_init=1.0)
    spatial_off = ContinuousMixtureSpatial(
        offensive_prior=setup_off["collab"],  # type: ignore[arg-type]
        matchup_field=setup_off["matchup_field"],  # type: ignore[arg-type]
        matchup_features=setup_off["matchup_features"],  # type: ignore[arg-type]
    )
    spatial_on = ContinuousMixtureSpatial(
        offensive_prior=setup_on["collab"],  # type: ignore[arg-type]
        matchup_field=setup_on["matchup_field"],  # type: ignore[arg-type]
        matchup_features=setup_on["matchup_features"],  # type: ignore[arg-type]
    )
    # Force player_idx=0 + opp_idx=0 + snap=0 (the warm cell).
    batch = _matchup_batch(setup_on, n_batch=3)
    batch["player_idx"] = torch.zeros_like(batch["player_idx"])
    batch["snapshot_idx"] = torch.zeros_like(batch["snapshot_idx"])
    batch["opp_idx"] = torch.zeros_like(batch["opp_idx"])
    with torch.no_grad():
        out_off = spatial_off(**batch)
        out_on = spatial_on(**batch)
    # The warm cell contributes a nonzero matchup term, which shifts the log-lik.
    assert not torch.allclose(out_off.log_lik, out_on.log_lik)
    assert out_on.matchup_logits is not None
    assert out_on.matchup_logits.shape[0] == 3
    assert out_on.matchup_n_eff is not None
    assert torch.equal(out_on.matchup_n_eff, torch.full((3,), 5.0))


def test_matchup_cold_cell_yields_zero_contribution() -> None:
    """Rows routed to a cold-start cell (Δ̂ = 0, N^eff = 0) get exactly zero matchup
    contribution, and the log-lik matches the no-matchup baseline."""
    setup = _build_matchup_setup(beta_init=2.0)
    # Pick player_idx that the fixture left at all-zero Δ̂. n_players ≥ 2
    # in _build_setup; the fixture only warmed player 0, so player 1 is
    # cold across every opp.
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_on = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        matchup_field=setup["matchup_field"],  # type: ignore[arg-type]
        matchup_features=setup["matchup_features"],  # type: ignore[arg-type]
    )
    batch = _matchup_batch(setup, n_batch=3)
    # Force player_idx=1 (cold across all opponents at snap 0).
    batch["player_idx"] = torch.ones_like(batch["player_idx"])
    batch["snapshot_idx"] = torch.zeros_like(batch["snapshot_idx"])
    batch["opp_idx"] = torch.zeros_like(batch["opp_idx"])
    with torch.no_grad():
        out_off = spatial_off(**{k: v for k, v in batch.items() if k != "opp_idx"})
        out_on = spatial_on(**batch)
    assert out_on.matchup_logits is not None
    assert torch.equal(out_on.matchup_logits, torch.zeros_like(out_on.matchup_logits))
    assert out_on.matchup_n_eff is not None
    assert torch.equal(out_on.matchup_n_eff, torch.zeros(3))
    torch.testing.assert_close(out_on.log_lik, out_off.log_lik, atol=1e-6, rtol=0)


def test_matchup_requires_opp_idx_when_wired() -> None:
    setup = _build_matchup_setup()
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        matchup_field=setup["matchup_field"],  # type: ignore[arg-type]
        matchup_features=setup["matchup_features"],  # type: ignore[arg-type]
    )
    batch = _matchup_batch(setup, n_batch=2)
    # Drop opp_idx — must error.
    del batch["opp_idx"]
    with pytest.raises(ValueError, match="matchup_field is wired"):
        spatial(**batch)


def test_matchup_composes_with_zone_lite_defense() -> None:
    """Zone-level and matchup reweighting can be wired together; both
    ``defense_logits`` and ``matchup_logits`` are populated and nonzero."""
    from shotcloud.features.defense_features import (
        _ZONE_CENTERED_SLICE,
        DEFENSE_FEATURE_DIM,
        DefenseFeatures,
        DefenseFeaturesConfig,
    )
    from shotcloud.models.zone_defense_reweighting import ZoneReweightingDefense

    setup = _build_matchup_setup(beta_init=0.5)
    # Synthesize two-opponent defense features with a nonzero centered-zone block
    # for opp 0, so the zone-level contribution is nonzero for the opp_idx=0 batch.
    n_snap = 1
    feat = torch.zeros(2, n_snap, DEFENSE_FEATURE_DIM)
    feat[0, 0, _ZONE_CENTERED_SLICE] = torch.tensor([0.1, -0.1, 0.0, 0.2, -0.2, 0.05, -0.05, 0.0])
    d_features = DefenseFeatures(
        features=feat,
        feature_names=tuple(["f"] * DEFENSE_FEATURE_DIM),
        config=DefenseFeaturesConfig(
            shots_fingerprint="fake",
            anchor_dates=(0,),
            window_days=365,
            half_life_days=90.0,
        ),
    )
    d_field = ZoneReweightingDefense(n_opponents=2, beta_init=1.0)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        defensive_field=d_field,
        defensive_features=d_features,
        matchup_field=setup["matchup_field"],  # type: ignore[arg-type]
        matchup_features=setup["matchup_features"],  # type: ignore[arg-type]
    )
    batch = _matchup_batch(setup, n_batch=2)
    batch["player_idx"] = torch.zeros_like(batch["player_idx"])
    batch["snapshot_idx"] = torch.zeros_like(batch["snapshot_idx"])
    batch["opp_idx"] = torch.zeros_like(batch["opp_idx"])
    with torch.no_grad():
        out = spatial(**batch)
    assert out.defense_logits is not None
    assert out.matchup_logits is not None
    # Both channels contributed: at least one row's defense_logits and
    # matchup_logits are nonzero (their support shots hit the right
    # zones).
    assert (out.defense_logits.abs() > 0).any()
    assert (out.matchup_logits.abs() > 0).any()


def test_matchup_field_lands_in_collected_modules_for_optimizer() -> None:
    """``_collect_modules_for_spatial`` includes ``matchup_field``, so ``β_match``
    reaches the optimizer and the best-validation snapshot."""
    from shotcloud.models.context_mlp import ContextMLP
    from shotcloud.models.count_head import NegBinCountHead
    from shotcloud.models.timing_head import TimingSoftmaxHead
    from shotcloud.training.train_gibbs import _collect_modules_for_spatial

    setup = _build_matchup_setup(beta_init=1e-3)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        matchup_field=setup["matchup_field"],  # type: ignore[arg-type]
        matchup_features=setup["matchup_features"],  # type: ignore[arg-type]
    )
    count_head = NegBinCountHead(context_dim=27, hidden_dim=4)
    timing_head = TimingSoftmaxHead(context_dim=2, n_bins=4, hidden_dim=4, zero_init_residual=True)
    context_mlp = ContextMLP(input_dim=2, hidden_dim=4, residual=True)

    modules = _collect_modules_for_spatial(
        spatial=spatial,
        count_head=count_head,
        timing_head=timing_head,
        context_mlp=context_mlp,
    )
    assert "matchup_field" in modules, (
        f"matchup_field absent from optimizer modules; got keys {sorted(modules)}"
    )
    assert modules["matchup_field"] is spatial.matchup_field
    # And the β_match parameter must be reachable via the dict's
    # .parameters() walk (what the trainer feeds Adam).
    all_params = [p for m in modules.values() for p in m.parameters() if p.requires_grad]
    assert any(p is spatial.matchup_field.beta_match for p in all_params), (
        "β_match parameter is not in the optimizer-bound parameter list"
    )


# ---------------------------------------------------------------------------
# Zone/source bandwidth field (ZoneSourceBandwidth)
# ---------------------------------------------------------------------------


def test_zone_source_bandwidth_at_init_matches_fixed_sigma() -> None:
    """With ``sigma_init`` equal to the fixed σ, the bandwidth field reproduces the
    fixed per-shot σ and the fixed-σ log-likelihood at initialization (up to float32
    round-off in the sigmoid composition)."""
    from shotcloud.models.zone_source_bandwidth import ZoneSourceBandwidth

    setup = _build_setup()
    fixed_sigma = float(setup["collab"].sigma_init)  # type: ignore[attr-defined]
    bw = ZoneSourceBandwidth(
        sigma_min=1.0, sigma_max=max(2.5, fixed_sigma + 0.1), sigma_init=fixed_sigma
    )

    spatial_fixed = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_zone = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        bandwidth_field=bw,
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out_fixed = spatial_fixed(**batch)
        out_zone = spatial_zone(**batch)
    # Per-shot σ at init equals fixed_sigma (within float32 precision).
    assert out_zone.sigma_per_shot is not None
    assert out_zone.sigma_per_shot.shape[0] == 3
    torch.testing.assert_close(
        out_zone.sigma_per_shot,
        torch.full_like(out_zone.sigma_per_shot, fixed_sigma),
        atol=1e-5,
        rtol=1e-5,
    )
    # log_lik matches the fixed-σ path within float32 sigmoid-composition
    # noise; loglik values are O(6), so atol=1e-3 is < 0.02% relative.
    torch.testing.assert_close(out_zone.log_lik, out_fixed.log_lik, atol=1e-3, rtol=1e-3)


def test_zone_source_bandwidth_shifts_log_lik_when_sigma_diverges() -> None:
    """Moving σ off its initial value changes the log-lik, so the per-shot σ reaches
    the likelihood."""
    from shotcloud.models.zone_source_bandwidth import ZoneSourceBandwidth

    setup = _build_setup()
    fixed_sigma = float(setup["collab"].sigma_init)  # type: ignore[attr-defined]
    bw = ZoneSourceBandwidth(sigma_min=0.5, sigma_max=4.0, sigma_init=fixed_sigma)
    with torch.no_grad():
        bw.raw.fill_(2.0)  # push σ toward σ_max

    spatial_fixed = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_zone = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        bandwidth_field=bw,
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out_fixed = spatial_fixed(**batch)
        out_zone = spatial_zone(**batch)
    assert not torch.allclose(out_zone.log_lik, out_fixed.log_lik)


# ---------------------------------------------------------------------------
# Anisotropic zone kernels
# ---------------------------------------------------------------------------


def test_radial_tangent_kernel_at_init_matches_fixed_sigma_loglik() -> None:
    """With ``sigma_init`` equal to the fixed σ, ``RadialTangentZoneKernel``
    reproduces the fixed-σ log-likelihood at initialization (within float32 noise)."""
    from shotcloud.models.anisotropic_kernel import RadialTangentZoneKernel

    setup = _build_setup()
    fixed_sigma = float(setup["collab"].sigma_init)  # type: ignore[attr-defined]
    rt = RadialTangentZoneKernel(
        sigma_min=1.0, sigma_max=max(2.5, fixed_sigma + 0.1), sigma_init=fixed_sigma
    )
    spatial_fixed = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_rt = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        anisotropic_kernel=rt,
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out_fixed = spatial_fixed(**batch)
        out_rt = spatial_rt(**batch)
    # log_lik matches fixed within float32 noise; the loglik path is
    # different (anisotropic decomposition vs scalar) so the round-off
    # budget is slightly looser than the σ-path test.
    torch.testing.assert_close(out_rt.log_lik, out_fixed.log_lik, atol=2e-3, rtol=2e-3)
    assert spatial_rt.has_anisotropic_kernel
    assert not spatial_rt.has_bandwidth_field


def test_full_cov_kernel_at_init_matches_fixed_sigma_loglik() -> None:
    """At ``σ_x = σ_y = σ_init`` and ``ρ = 0``, ``FullCovarianceZoneKernel``
    reproduces the fixed-σ log-likelihood."""
    from shotcloud.models.anisotropic_kernel import FullCovarianceZoneKernel

    setup = _build_setup()
    fixed_sigma = float(setup["collab"].sigma_init)  # type: ignore[attr-defined]
    fc = FullCovarianceZoneKernel(
        sigma_min=1.0,
        sigma_max=max(2.5, fixed_sigma + 0.1),
        sigma_init=fixed_sigma,
        rho_max=0.8,
        rho_init=0.0,
    )
    spatial_fixed = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_fc = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        anisotropic_kernel=fc,
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out_fixed = spatial_fixed(**batch)
        out_fc = spatial_fc(**batch)
    torch.testing.assert_close(out_fc.log_lik, out_fixed.log_lik, atol=2e-3, rtol=2e-3)


def test_anisotropic_and_bandwidth_field_are_mutually_exclusive() -> None:
    """Wiring both raises: each sets the kernel shape, so one would silently override
    the other."""
    from shotcloud.models.anisotropic_kernel import RadialTangentZoneKernel
    from shotcloud.models.zone_source_bandwidth import ZoneSourceBandwidth

    setup = _build_setup()
    with pytest.raises(ValueError, match="mutually exclusive"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            bandwidth_field=ZoneSourceBandwidth(),
            anisotropic_kernel=RadialTangentZoneKernel(),
        )


def test_radial_tangent_kernel_diverges_from_fixed_sigma_after_param_shift() -> None:
    """Moving σ_r off its initial value changes the log-lik, so the anisotropic kernel
    reaches the likelihood."""
    from shotcloud.models.anisotropic_kernel import RadialTangentZoneKernel

    setup = _build_setup()
    fixed_sigma = float(setup["collab"].sigma_init)  # type: ignore[attr-defined]
    rt = RadialTangentZoneKernel(sigma_min=0.5, sigma_max=4.0, sigma_init=fixed_sigma)
    with torch.no_grad():
        rt.raw_r.fill_(3.0)  # push σ_r toward σ_max across all zones
    spatial_fixed = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_rt = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        anisotropic_kernel=rt,
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out_fixed = spatial_fixed(**batch)
        out_rt = spatial_rt(**batch)
    assert not torch.allclose(out_rt.log_lik, out_fixed.log_lik, atol=1e-2)


def test_anisotropic_kernel_gradients_reach_kernel_params() -> None:
    """The log-lik gradient reaches the kernel's raw parameters."""
    from shotcloud.models.anisotropic_kernel import RadialTangentZoneKernel

    setup = _build_setup()
    fixed_sigma = float(setup["collab"].sigma_init)  # type: ignore[attr-defined]
    rt = RadialTangentZoneKernel(sigma_min=1.0, sigma_max=2.5, sigma_init=fixed_sigma)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        anisotropic_kernel=rt,
    )
    batch = _batch(setup, n_batch=3)
    out = spatial(**batch)
    (-out.log_lik.sum()).backward()
    assert rt.raw_r.grad is not None
    assert rt.raw_t.grad is not None
    assert rt.raw_r.grad.abs().sum() > 0
    assert rt.raw_t.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Causal usage residual
# ---------------------------------------------------------------------------


def test_usage_residual_at_init_matches_no_usage_residual_loglik() -> None:
    """The usage MLP's output layer is zero-initialized, so a residual encoder with
    ``usage_dim > 0`` reproduces the ``usage_dim = 0`` log-likelihood at
    initialization (within float32 noise)."""
    from shotcloud import ContextResidualEncoder
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    # Build matched pairs of (residual encoder, location embedding)
    # for the no-usage and with-usage variants. Both use the same
    # rank, so the location embeddings are identical in shape.
    rank = 8
    encoder_no_usage = ContextResidualEncoder(rank=rank, within_game_dim=0, usage_dim=0)
    encoder_with_usage = ContextResidualEncoder(rank=rank, within_game_dim=0, usage_dim=3)
    # Copy base-branch weights so the only difference is the usage
    # MLP (zero-init by construction).
    encoder_with_usage.fc1.weight.data.copy_(encoder_no_usage.fc1.weight.data)
    encoder_with_usage.fc1.bias.data.copy_(encoder_no_usage.fc1.bias.data)
    encoder_with_usage.fc2.weight.data.copy_(encoder_no_usage.fc2.weight.data)
    encoder_with_usage.fc2.bias.data.copy_(encoder_no_usage.fc2.bias.data)
    loc = LocationEmbedding(rank=rank, zero_init=True)

    spatial_no = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder_no_usage,
        location_embedding=LocationEmbedding(rank=rank, zero_init=True),
    )
    # Copy the location-embedding weights too, so the only difference is the usage
    # branch.
    spatial_with = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder_with_usage,
        location_embedding=loc,
    )
    spatial_with.location_embedding.proj.weight.data.copy_(
        spatial_no.location_embedding.proj.weight.data
    )
    spatial_with.location_embedding.proj.bias.data.copy_(
        spatial_no.location_embedding.proj.bias.data
    )

    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out_no = spatial_no(**batch)
        out_with = spatial_with(**batch)
    torch.testing.assert_close(out_with.log_lik, out_no.log_lik, atol=2e-5, rtol=0)


def test_usage_residual_gradient_reaches_usage_mlp() -> None:
    """Once off zero-init, the usage MLP receives the log-lik gradient."""
    from shotcloud import ContextResidualEncoder
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    encoder = ContextResidualEncoder(rank=8, within_game_dim=0, usage_dim=3)
    # Move usage_fc2 off zero so the chain rule has a non-zero path.
    with torch.no_grad():
        encoder.usage_fc2.weight.normal_(std=0.5)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder,
        location_embedding=LocationEmbedding(rank=8, zero_init=False),
    )
    batch = _batch(setup, n_batch=3)
    out = spatial(**batch)
    (-out.log_lik.sum()).backward()
    assert encoder.usage_fc1.weight.grad is not None
    assert encoder.usage_fc1.weight.grad.abs().sum() > 0
    assert encoder.usage_fc2.weight.grad is not None
    assert encoder.usage_fc2.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Count residual (K̂ from the count head)
# ---------------------------------------------------------------------------


def test_count_residual_at_init_matches_usage_only_loglik() -> None:
    """Appending the count head's K̂ to the usage vector leaves the log-likelihood
    unchanged at initialization, because ``usage_fc2`` is zero-initialized."""
    from shotcloud import ContextResidualEncoder
    from shotcloud.features.usage_features import USAGE_DIM, USAGE_KHAT_DIM
    from shotcloud.models.count_head import NegBinCountHead
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    rank = 8
    enc_no_k = ContextResidualEncoder(rank=rank, within_game_dim=0, usage_dim=USAGE_DIM)
    enc_with_k = ContextResidualEncoder(rank=rank, within_game_dim=0, usage_dim=USAGE_KHAT_DIM)
    # Copy base-branch weights so the only architectural difference is
    # the usage MLP's input width.
    enc_with_k.fc1.weight.data.copy_(enc_no_k.fc1.weight.data)
    enc_with_k.fc1.bias.data.copy_(enc_no_k.fc1.bias.data)
    enc_with_k.fc2.weight.data.copy_(enc_no_k.fc2.weight.data)
    enc_with_k.fc2.bias.data.copy_(enc_no_k.fc2.bias.data)
    # (Both encoders have usage_fc2 zero-init by construction.)

    count_head = NegBinCountHead(context_dim=27, hidden_dim=4)
    spatial_no_k = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=enc_no_k,
        location_embedding=LocationEmbedding(rank=rank, zero_init=True),
    )
    spatial_with_k = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=enc_with_k,
        location_embedding=LocationEmbedding(rank=rank, zero_init=True),
        count_head=count_head,
    )
    # Sync location-embedding weights too.
    spatial_with_k.location_embedding.proj.weight.data.copy_(
        spatial_no_k.location_embedding.proj.weight.data
    )
    spatial_with_k.location_embedding.proj.bias.data.copy_(
        spatial_no_k.location_embedding.proj.bias.data
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out_no = spatial_no_k(**batch)
        out_with = spatial_with_k(**batch)
    torch.testing.assert_close(out_with.log_lik, out_no.log_lik, atol=2e-5, rtol=0)


def test_count_residual_constructor_rejects_inconsistent_wiring() -> None:
    """Valid wiring is ``usage_dim == USAGE_KHAT_DIM`` with a count head, or
    ``usage_dim == USAGE_DIM`` without one; other combinations raise."""
    from shotcloud import ContextResidualEncoder
    from shotcloud.features.usage_features import USAGE_DIM, USAGE_KHAT_DIM
    from shotcloud.models.count_head import NegBinCountHead
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    enc_b1 = ContextResidualEncoder(rank=8, within_game_dim=0, usage_dim=USAGE_DIM)
    enc_b2 = ContextResidualEncoder(rank=8, within_game_dim=0, usage_dim=USAGE_KHAT_DIM)
    count_head = NegBinCountHead(context_dim=27, hidden_dim=4)
    loc = LocationEmbedding(rank=8, zero_init=True)

    # Usage + K̂ encoder without a count head → error.
    with pytest.raises(ValueError, match="usage_dim=4 expects a wired count_head"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            residual_encoder=enc_b2,
            location_embedding=loc,
        )
    # Usage-only encoder with a count head → error.
    with pytest.raises(ValueError, match=r"count_head wired but residual_encoder\.usage_dim"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            residual_encoder=enc_b1,
            location_embedding=loc,
            count_head=count_head,
        )
    # count_head wired with no residual_encoder at all → error.
    with pytest.raises(ValueError, match="count_head wired but residual_encoder is None"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            count_head=count_head,
        )


def test_count_residual_gradient_path_isolation() -> None:
    """The spatial loss sends no gradient to the count head: K̂ is detached before it
    enters the residual, so the count head is trained by the count loss alone."""
    from shotcloud import ContextResidualEncoder
    from shotcloud.features.usage_features import USAGE_KHAT_DIM
    from shotcloud.models.count_head import NegBinCountHead
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    enc = ContextResidualEncoder(rank=8, within_game_dim=0, usage_dim=USAGE_KHAT_DIM)
    # Move usage_fc2 off zero so the chain rule has a non-zero path
    # through the usage MLP (without this, all gradients downstream of
    # the usage branch are zero by the encoder's init invariant).
    with torch.no_grad():
        enc.usage_fc2.weight.normal_(std=0.5)
    count_head = NegBinCountHead(context_dim=27, hidden_dim=4)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=enc,
        location_embedding=LocationEmbedding(rank=8, zero_init=False),
        count_head=count_head,
    )
    batch = _batch(setup, n_batch=3)
    out = spatial(**batch)
    # Use only the spatial log_lik (no count_loss) and back-prop.
    (-out.log_lik.sum()).backward()
    # Usage MLP receives a gradient — that path is intact.
    assert enc.usage_fc2.weight.grad is not None
    assert enc.usage_fc2.weight.grad.abs().sum() > 0
    # The count head receives no gradient from the spatial loss.
    for name, p in count_head.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, (
            f"count_head parameter {name!r} received spatial-loss gradient — "
            "the detach invariant is broken; B2's count head would no longer "
            "behave as a count forecast."
        )


def test_khat_only_path_runs_and_produces_finite_loglik() -> None:
    """With ``usage_dim=1`` and a count head, the usage MLP consumes the (B, 1) K̂
    column alone and the log-likelihood is finite."""
    from shotcloud import ContextResidualEncoder
    from shotcloud.models.count_head import NegBinCountHead
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    rank = 8
    encoder = ContextResidualEncoder(rank=rank, within_game_dim=0, usage_dim=1)
    count_head = NegBinCountHead(context_dim=27, hidden_dim=4)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder,
        location_embedding=LocationEmbedding(rank=rank, zero_init=True),
        count_head=count_head,
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out = spatial(**batch)
    assert torch.isfinite(out.log_lik).all()


def test_khat_only_at_init_matches_no_residual_baseline() -> None:
    """With zero-initialized ``usage_fc2`` and location embedding, the K̂-only residual
    reproduces the no-residual log-likelihood at initialization."""
    from shotcloud import ContextResidualEncoder
    from shotcloud.models.count_head import NegBinCountHead
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    spatial_no_res = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    encoder = ContextResidualEncoder(rank=8, within_game_dim=0, usage_dim=1)
    count_head = NegBinCountHead(context_dim=27, hidden_dim=4)
    spatial_khat = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder,
        location_embedding=LocationEmbedding(rank=8, zero_init=True),
        count_head=count_head,
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out_no_res = spatial_no_res(**batch)
        out_khat = spatial_khat(**batch)
    # The base residual branch is small but nonzero; the zero-initialized usage_fc2
    # removes the K̂ contribution and the zero-initialized location embedding makes the
    # residual logits vanish, so the log-lik matches.
    torch.testing.assert_close(out_khat.log_lik, out_no_res.log_lik, atol=2e-5, rtol=0)


def test_within_game_gru_zero_init_matches_b2_loglik_exactly() -> None:
    """The within-game GRU's output projection is zero-initialized, so adding the GRU
    leaves the log-lik bit-identical at initialization."""
    from shotcloud import ContextResidualEncoder
    from shotcloud.data.within_game_history import MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM
    from shotcloud.models.location_embedding import LocationEmbedding
    from shotcloud.models.within_game_gru import WithinGameGRU

    setup = _build_setup()
    rank = 8
    torch.manual_seed(0)
    encoder = ContextResidualEncoder(rank=rank, within_game_dim=0, usage_dim=0)
    encoder_g1 = ContextResidualEncoder(rank=rank, within_game_dim=0, usage_dim=0)
    encoder_g1.load_state_dict(encoder.state_dict())
    loc = LocationEmbedding(rank=rank, zero_init=False)
    loc_g1 = LocationEmbedding(rank=rank, zero_init=False)
    loc_g1.load_state_dict(loc.state_dict())
    gru = WithinGameGRU(hidden_dim=16, out_dim=rank)

    spatial_b2 = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder,
        location_embedding=loc,
    )
    spatial_g1 = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder_g1,
        location_embedding=loc_g1,
        within_game_gru=gru,
    )
    batch = _batch(setup, n_batch=4)
    n_b = batch["player_idx"].shape[0]
    prior_seq = torch.randn(n_b, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)
    prior_lengths = torch.tensor([0, 3, 1, 7][:n_b], dtype=torch.int64)
    with torch.no_grad():
        out_b2 = spatial_b2(**batch)
        out_g1 = spatial_g1(**batch, prior_seq=prior_seq, prior_lengths=prior_lengths)
    torch.testing.assert_close(out_g1.log_lik, out_b2.log_lik, atol=0, rtol=0)


def test_within_game_gru_rejects_missing_prior_tensors() -> None:
    from shotcloud import ContextResidualEncoder
    from shotcloud.models.location_embedding import LocationEmbedding
    from shotcloud.models.within_game_gru import WithinGameGRU

    setup = _build_setup()
    rank = 8
    encoder = ContextResidualEncoder(rank=rank, within_game_dim=0, usage_dim=0)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder,
        location_embedding=LocationEmbedding(rank=rank, zero_init=True),
        within_game_gru=WithinGameGRU(hidden_dim=8, out_dim=rank),
    )
    batch = _batch(setup, n_batch=3)
    with pytest.raises(ValueError, match="prior_seq/prior_lengths were not provided"):
        spatial(**batch)


def test_within_game_gru_rejects_when_no_residual() -> None:
    from shotcloud.models.within_game_gru import WithinGameGRU

    setup = _build_setup()
    with pytest.raises(ValueError, match="within_game_gru wired but residual_encoder is None"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            within_game_gru=WithinGameGRU(hidden_dim=8, out_dim=8),
        )


def test_within_game_gru_prior_tensors_without_gru_raises() -> None:
    from shotcloud import ContextResidualEncoder
    from shotcloud.data.within_game_history import MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    rank = 8
    encoder = ContextResidualEncoder(rank=rank, within_game_dim=0, usage_dim=0)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder,
        location_embedding=LocationEmbedding(rank=rank, zero_init=True),
    )
    batch = _batch(setup, n_batch=3)
    n_b = batch["player_idx"].shape[0]
    prior_seq = torch.randn(n_b, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)
    prior_lengths = torch.zeros(n_b, dtype=torch.int64)
    with pytest.raises(ValueError, match="prior_seq/prior_lengths supplied"):
        spatial(**batch, prior_seq=prior_seq, prior_lengths=prior_lengths)


# ---------------------------------------------------------------------------
# Count-head calibration: ``init_mean`` warm-starts ``fc2.bias`` so μ at
# initialization matches the mean count K̄, and ``khat_log1p_mean/std``
# standardize log1p(K̂) before it enters the residual.
# ---------------------------------------------------------------------------


def test_count_head_init_mean_warm_starts_fc2_bias() -> None:
    """``NegBinCountHead(init_mean=K̄)`` sets ``fc2.bias`` to
    ``log(exp(K̄) − 1)`` (the inverse softplus); the default (``None``) leaves it at
    zero."""
    import math

    from shotcloud.models.count_head import NegBinCountHead

    default_head = NegBinCountHead(context_dim=27, hidden_dim=8)
    assert float(default_head.fc2.bias.item()) == 0.0

    target = 9.569  # a realistic mean shot count per player-game
    warm_head = NegBinCountHead(context_dim=27, hidden_dim=8, init_mean=target)
    expected_bias = math.log(math.expm1(target))
    assert float(warm_head.fc2.bias.item()) == pytest.approx(expected_bias, rel=1e-6)


def test_count_head_init_mean_produces_mu_near_target_at_init() -> None:
    """With a warm-started bias, ``μ_η(x_n)`` at initialization stays on the scale of
    ``init_mean`` for random ``x_n``, whereas the default head sits near
    softplus(0) ≈ 0.7."""
    from shotcloud.models.count_head import NegBinCountHead

    target = 9.569
    warm_head = NegBinCountHead(context_dim=27, hidden_dim=8, init_mean=target)
    default_head = NegBinCountHead(context_dim=27, hidden_dim=8)
    torch.manual_seed(0)
    x = torch.randn(64, 27)
    with torch.no_grad():
        warm_mu, _ = warm_head(x)
        default_mu, _ = default_head(x)
    # The default head sits near softplus(0) ≈ 0.7.
    assert 0.3 < float(default_mu.mean().item()) < 1.5
    # Warm-started head sits in a (target ± noise) band — well above
    # the default's range and clearly on the K̄ scale.
    assert target / 2.0 < float(warm_mu.mean().item()) < target * 2.0


def test_count_head_init_mean_rejects_non_positive() -> None:
    from shotcloud.models.count_head import NegBinCountHead

    with pytest.raises(ValueError, match="init_mean must be positive"):
        NegBinCountHead(context_dim=27, hidden_dim=8, init_mean=0.0)
    with pytest.raises(ValueError, match="init_mean must be positive"):
        NegBinCountHead(context_dim=27, hidden_dim=8, init_mean=-3.2)


def test_cms_khat_standardization_default_none_matches_raw_path() -> None:
    """By default ``khat_log1p_mean/std`` are ``None``, standardization is off, and the
    residual receives the raw detached μ."""
    from shotcloud import ContextResidualEncoder
    from shotcloud.models.count_head import NegBinCountHead
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    rank = 8
    torch.manual_seed(0)
    encoder = ContextResidualEncoder(rank=rank, within_game_dim=0, usage_dim=1)
    count_head = NegBinCountHead(context_dim=27, hidden_dim=4)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder,
        location_embedding=LocationEmbedding(rank=rank, zero_init=True),
        count_head=count_head,
    )
    assert spatial.khat_log1p_mean is None
    assert spatial.khat_log1p_std is None
    assert not spatial.has_khat_standardization
    # Forward succeeds and routes a real raw-K̂ tensor into the residual.
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out = spatial(**batch)
    assert torch.isfinite(out.log_lik).all()


def test_cms_khat_standardization_requires_count_head() -> None:
    from shotcloud import ContextResidualEncoder
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    encoder = ContextResidualEncoder(rank=8, within_game_dim=0, usage_dim=0)
    with pytest.raises(ValueError, match="khat_log1p_mean/std are only meaningful"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            residual_encoder=encoder,
            location_embedding=LocationEmbedding(rank=8, zero_init=True),
            khat_log1p_mean=2.24,
            khat_log1p_std=0.49,
        )


def test_cms_khat_standardization_requires_both_mean_and_std() -> None:
    from shotcloud import ContextResidualEncoder
    from shotcloud.models.count_head import NegBinCountHead
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    encoder = ContextResidualEncoder(rank=8, within_game_dim=0, usage_dim=1)
    count_head = NegBinCountHead(context_dim=27, hidden_dim=4)
    with pytest.raises(ValueError, match="must both be provided or both None"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            residual_encoder=encoder,
            location_embedding=LocationEmbedding(rank=8, zero_init=True),
            count_head=count_head,
            khat_log1p_mean=2.24,
            # std missing
        )


def test_cms_khat_standardization_requires_positive_std() -> None:
    from shotcloud import ContextResidualEncoder
    from shotcloud.models.count_head import NegBinCountHead
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    encoder = ContextResidualEncoder(rank=8, within_game_dim=0, usage_dim=1)
    count_head = NegBinCountHead(context_dim=27, hidden_dim=4)
    with pytest.raises(ValueError, match="khat_log1p_std must be positive"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            residual_encoder=encoder,
            location_embedding=LocationEmbedding(rank=8, zero_init=True),
            count_head=count_head,
            khat_log1p_mean=2.24,
            khat_log1p_std=0.0,
        )


def test_cms_khat_standardization_transform_matches_formula() -> None:
    """The ``_transform_khat_for_residual`` helper applies exactly
    ``(log1p(μ) − μ_K) / σ_K`` when standardization is wired and just
    detaches μ otherwise. Calling it directly avoids depending on the
    residual encoder's internal MLP shape."""
    from shotcloud import ContextResidualEncoder
    from shotcloud.models.count_head import NegBinCountHead
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    encoder = ContextResidualEncoder(rank=8, within_game_dim=0, usage_dim=1)
    count_head = NegBinCountHead(context_dim=27, hidden_dim=4)
    mu_K, sig_K = 2.24, 0.49
    spatial_std = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder,
        location_embedding=LocationEmbedding(rank=8, zero_init=True),
        count_head=count_head,
        khat_log1p_mean=mu_K,
        khat_log1p_std=sig_K,
    )
    assert spatial_std.has_khat_standardization
    mu = torch.tensor([0.5, 1.2, 9.5, 22.0])
    out = spatial_std._transform_khat_for_residual(mu)
    expected = (torch.log1p(mu) - mu_K) / sig_K
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    # And the detach invariant — grad does not flow back through μ.
    mu_with_grad = torch.tensor([1.2], requires_grad=True)
    transformed = spatial_std._transform_khat_for_residual(mu_with_grad)
    assert not transformed.requires_grad


def test_cms_khat_standardization_in_forward_changes_residual_input() -> None:
    """Two decoders that differ only in K̂ standardization give different log-liks once
    the usage path is nonzero. (With zero-initialized ``usage_fc2`` the K̂ path
    contributes nothing whatever K̂'s value, so the test perturbs it first.)"""
    from shotcloud import ContextResidualEncoder
    from shotcloud.models.count_head import NegBinCountHead
    from shotcloud.models.location_embedding import LocationEmbedding

    setup = _build_setup()
    torch.manual_seed(0)
    encoder = ContextResidualEncoder(rank=8, within_game_dim=0, usage_dim=1)
    # Force the usage pathway nonzero so the K̂ column actually reaches
    # the residual logits. Without this, usage_fc2 = 0 zeros the K̂
    # contribution at init regardless of any standardization.
    with torch.no_grad():
        encoder.usage_fc2.weight.normal_(std=0.1)
        encoder.usage_fc2.bias.normal_(std=0.1)
    count_head = NegBinCountHead(context_dim=27, hidden_dim=4, init_mean=9.569)
    loc_a = LocationEmbedding(rank=8, zero_init=False)
    loc_b = LocationEmbedding(rank=8, zero_init=False)
    loc_b.load_state_dict(loc_a.state_dict())
    spatial_raw = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder,
        location_embedding=loc_a,
        count_head=count_head,
    )
    spatial_std = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=encoder,
        location_embedding=loc_b,
        count_head=count_head,
        khat_log1p_mean=2.24,
        khat_log1p_std=0.49,
    )
    batch = _batch(setup, n_batch=4)
    with torch.no_grad():
        out_raw = spatial_raw(**batch)
        out_std = spatial_std(**batch)
    assert torch.isfinite(out_raw.log_lik).all()
    assert torch.isfinite(out_std.log_lik).all()
    # The standardized residual sees a different K̂ column (order ±2σ
    # away from the raw ~9.5), so log_lik differs across at least one row.
    assert not torch.allclose(out_raw.log_lik, out_std.log_lik, atol=1e-6)


def test_stratified_epsilon_one_is_bit_identical_to_unstratified() -> None:
    """``stratified_epsilon=1.0`` reproduces the unstratified log-lik bit-for-bit; the
    stratified-kernel path runs only when ε ≠ 1."""
    torch.manual_seed(0)
    setup = _build_setup()
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_on1 = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        stratified_epsilon=1.0,
    )
    batch = _batch(setup, n_batch=8)
    out_off = spatial_off(**batch)
    out_on1 = spatial_on1(**batch)
    assert torch.equal(out_off.log_lik, out_on1.log_lik)


def test_stratified_epsilon_lt_one_perturbs_log_lik() -> None:
    """``stratified_epsilon=0.1`` adds a log(0.1) ≈ -2.3 nat penalty to each
    cross-stratum (query zone ≠ support zone) pair, so no row's log-lik rises and some
    rows fall."""
    torch.manual_seed(0)
    setup = _build_setup()
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_on = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        stratified_epsilon=0.1,
    )
    batch = _batch(setup, n_batch=16)
    out_off = spatial_off(**batch)
    out_on = spatial_on(**batch)
    # Stratified decoder downweights cross-zone support, so the mixture
    # density at the observed shot can only be ≤ the unstratified
    # density (some pairs are penalized, no pair is rewarded).
    assert (out_on.log_lik <= out_off.log_lik + 1e-6).all(), (
        "stratified log-lik must not exceed unstratified at any row"
    )
    assert (out_off.log_lik - out_on.log_lik > 1e-3).any(), (
        "at least some rows should show a non-trivial penalty"
    )


def test_stratified_epsilon_invalid_raises() -> None:
    """``stratified_epsilon`` outside (0, 1] raises at construction."""
    import pytest

    setup = _build_setup()
    with pytest.raises(ValueError, match="stratified_epsilon"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            stratified_epsilon=0.0,
        )
    with pytest.raises(ValueError, match="stratified_epsilon"):
        ContinuousMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            stratified_epsilon=1.5,
        )


def test_causal_zone_bias_disabled_is_bit_identical() -> None:
    """With ``causal_zone_bias=None`` the decoder has no edge-bias path and its log-lik
    is bit-identical to the default."""
    from shotcloud.models.continuous_mixture_spatial import CausalZoneBias  # noqa: F401

    torch.manual_seed(0)
    setup = _build_setup()
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_disabled = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        causal_zone_bias=None,
    )
    batch = _batch(setup, n_batch=8)
    out_off = spatial_off(**batch)
    out_disabled = spatial_disabled(**batch)
    assert torch.equal(out_off.log_lik, out_disabled.log_lik)
    assert spatial_disabled.causal_zone_bias is None


def test_causal_zone_bias_zero_init_is_bit_identical() -> None:
    """``B`` is zero-initialized, so ``π_q @ B = 0`` for any ``π_q``, the edge bias
    vanishes, and the log-lik is bit-identical to the no-bias decoder."""
    from shotcloud.models.continuous_mixture_spatial import CausalZoneBias

    torch.manual_seed(0)
    setup = _build_setup()
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_on = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        causal_zone_bias=CausalZoneBias(),
    )
    batch = _batch(setup, n_batch=8)
    out_off = spatial_off(**batch)
    out_on = spatial_on(**batch)
    assert spatial_on.causal_zone_bias is not None
    assert spatial_on.causal_zone_bias.B.shape == (8, 8)
    assert torch.all(spatial_on.causal_zone_bias.B == 0.0)
    assert torch.equal(out_off.log_lik, out_on.log_lik)


def test_causal_zone_bias_learned_perturbs_log_lik() -> None:
    """With nonzero ``B`` and ``q_head`` parameters the log-lik differs from the
    no-bias decoder, so the bias reaches the forward pass."""
    from shotcloud.models.continuous_mixture_spatial import CausalZoneBias

    torch.manual_seed(0)
    setup = _build_setup()
    bias = CausalZoneBias()
    with torch.no_grad():
        bias.B.copy_(torch.randn(8, 8) * 2.0)
        for p in bias.q_head.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    spatial_off = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    spatial_on = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        causal_zone_bias=bias,
    )
    batch = _batch(setup, n_batch=16)
    out_off = spatial_off(**batch)
    out_on = spatial_on(**batch)
    assert not torch.allclose(out_off.log_lik, out_on.log_lik, atol=1e-6)


def test_causal_zone_bias_module_is_registered_child() -> None:
    """``causal_zone_bias`` is a registered child module and is collected by
    ``_collect_modules_for_spatial``, so the optimizer, best-validation snapshot, and
    checkpoint all include ``B``."""
    from shotcloud.models import ContextMLP, NegBinCountHead, TimingSoftmaxHead
    from shotcloud.models.continuous_mixture_spatial import CausalZoneBias
    from shotcloud.training.train_gibbs import _collect_modules_for_spatial

    setup = _build_setup()
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        causal_zone_bias=CausalZoneBias(),
    )
    child_names = {n for n, _ in spatial.named_children()}
    assert "causal_zone_bias" in child_names, (
        f"causal_zone_bias must be a named child of spatial; got {child_names}"
    )
    bias_param = spatial.causal_zone_bias.B  # type: ignore[union-attr]
    assert any(p is bias_param for p in spatial.parameters() if p.requires_grad), (
        "the B parameter must appear in spatial.parameters() so the optimizer trains it"
    )
    modules = _collect_modules_for_spatial(
        spatial=spatial,
        count_head=NegBinCountHead(),
        timing_head=TimingSoftmaxHead(),
        context_mlp=ContextMLP(input_dim=27, hidden_dim=4, residual=True),
    )
    assert "causal_zone_bias" in modules, (
        "_collect_modules_for_spatial must register the causal_zone_bias module so the "
        "optimizer + best-val snapshot + disk save pick it up"
    )


def test_causal_zone_bias_B_receives_gradient_at_init() -> None:
    """At zero-initialized ``B`` one backward pass gives ``B`` a nonzero, finite
    gradient. The ``q_head`` gradient is exactly zero here because it is multiplied by
    ``B``; the next test covers it once ``B`` is nonzero."""
    from shotcloud.models.continuous_mixture_spatial import CausalZoneBias

    setup = _build_setup()
    bias = CausalZoneBias()
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        causal_zone_bias=bias,
    )
    batch = _batch(setup, n_batch=16)
    out = spatial(**batch)
    loss = -out.log_lik.mean()
    loss.backward()
    assert bias.B.grad is not None, "B.grad must be populated after backward"
    assert torch.isfinite(bias.B.grad).all(), "B grad must be finite"
    assert bias.B.grad.abs().sum() > 0.0, "B grad must be non-zero so the optimizer can update it"


def test_causal_zone_bias_q_head_receives_gradient_after_warmstart() -> None:
    """Once ``B`` is nonzero, ``q_head`` receives gradient through
    ``loss → edge_bias → π_q → q_head``."""
    from shotcloud.models.continuous_mixture_spatial import CausalZoneBias

    setup = _build_setup()
    bias = CausalZoneBias()
    # A small nonzero B breaks the dual-zero saddle, as one optimizer step would.
    with torch.no_grad():
        bias.B.copy_(torch.randn(8, 8) * 0.1)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        causal_zone_bias=bias,
    )
    batch = _batch(setup, n_batch=16)
    out = spatial(**batch)
    loss = -out.log_lik.mean()
    loss.backward()
    qh_grads = [p.grad for p in bias.q_head.parameters()]
    assert all(g is not None for g in qh_grads), "q_head params must have grad"
    assert any(g.abs().sum() > 0.0 for g in qh_grads if g is not None), (
        "at least one q_head param must have non-zero gradient once B is non-zero"
    )


def test_causal_zone_bias_invariant_to_observed_shot() -> None:
    """With ``causal_zone_bias`` enabled, ``support_log_weights`` are bit-identical
    when only ``shot_xy`` changes.

    This is the no-leakage contract: no support logit, gate, residual, or kernel
    modifier may depend on the observed location y_n except through the normalized
    density evaluation itself. The bias derives its query-zone distribution from
    ``x_n`` alone.
    """
    from shotcloud.models.continuous_mixture_spatial import CausalZoneBias

    setup = _build_setup()
    bias = CausalZoneBias()
    # Random nonzero parameters; a zero bias would satisfy the invariant trivially.
    with torch.no_grad():
        bias.B.copy_(torch.randn(8, 8) * 1.5)
        for p in bias.q_head.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        causal_zone_bias=bias,
    )
    batch_a = _batch(setup, n_batch=8)
    # Shift shot_xy far enough to cross zone boundaries, so any dependence on
    # z(y_n) would change the bias; x_n, the support set, and gate inputs are fixed.
    perturbation = torch.tensor([[20.0, 5.0]], dtype=batch_a["shot_xy"].dtype)
    batch_b = {**batch_a, "shot_xy": batch_a["shot_xy"] + perturbation}
    out_a = spatial(**batch_a)
    out_b = spatial(**batch_b)
    # Support weights must not depend on the observed shot location.
    assert torch.equal(out_a.support_log_weights, out_b.support_log_weights), (
        "support_log_weights changed when shot_xy changed — LEAKAGE BUG. "
        "Some component is now conditioning on the observed shot's location "
        "in computing attention weights. See the locked rule in "
        "CausalZoneBias docstring."
    )
