"""Training smoke tests for ``train_gibbs`` with the continuous adaptive defensive field.

Runs ``train_gibbs`` on synthetic data with a small batch and defensive support cap
and checks that training finishes with finite losses, that the defensive field's
parameters (``β_D``, ``q_net``, ``k_net``, opponent embedding) receive gradients and
move, and that incomplete or unsupported defense configurations are rejected.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from shotcloud import CourtGrid
from shotcloud.data import ContextEncoder
from shotcloud.data.context import CONTEXT_DIM
from shotcloud.data.player_traits import build_player_traits_table
from shotcloud.data.role_profile import build_role_profiles
from shotcloud.data.snapshots import build_snapshot_store_from_shots
from shotcloud.features.defense_features import (
    DEFENSE_FEATURE_DIM,
    DefenseFeaturesConfig,
    build_defense_features,
)
from shotcloud.kde.adaptive import AdaptiveKDE
from shotcloud.models import (
    ContextMLP,
    NegBinCountHead,
    TimingSoftmaxHead,
)
from shotcloud.models.analogue_retrieval import build_analogue_cache
from shotcloud.models.collaborative_kde import CollaborativeKDE
from shotcloud.models.context_residual import ContextResidualEncoder
from shotcloud.models.continuous_adaptive_defensive import ContinuousAdaptiveDefensiveField
from shotcloud.models.defensive_retrieval_cache import (
    DefensiveRetrievalCacheConfig,
    build_defensive_retrieval_cache,
)
from shotcloud.models.location_embedding import LocationEmbedding
from shotcloud.models.pooling_gate import PoolingGate
from shotcloud.training import GibbsShotDataset, OpponentVocab, PlayerVocab, train_gibbs


def _build_setup(seed: int = 0) -> dict[str, object]:
    """Four players with 40 shots each, alternating between two opponents over 40
    consecutive days, plus the offensive ``CollaborativeKDE`` and the defensive
    cache, features, and field."""
    rng = np.random.default_rng(seed)
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=14, ny=12)
    base_date = pd.Timestamp("2024-01-01")
    n_players = 4
    opponents = ["BOS", "LAL"]
    rows: list[dict[str, object]] = []
    for pid in range(1, n_players + 1):
        cx, cy = (0.0, 5.0) if pid <= n_players // 2 else (2.0, 8.0)
        for i in range(40):
            rows.append(
                {
                    "x": float(cx + rng.normal(0, 3)),
                    "y": float(cy + rng.normal(0, 3)),
                    "player_id": pid,
                    "opponent": opponents[i % len(opponents)],
                    "made": int(rng.random() < 0.5),
                    "period": (i % 4) + 1,
                    "time_remaining_sec": int(60 * (i % 48)),
                    "date": base_date + pd.Timedelta(days=i),
                    "game_id": f"g{pid}_{i // 5}",
                }
            )
    shots = pd.DataFrame(rows)
    anchors_np = np.array([np.datetime64("2024-01-05", "D").astype(np.int64)], dtype=np.int64)
    anchors_dt = [np.datetime64("2024-01-05", "D")]
    store = build_snapshot_store_from_shots(
        shots,
        anchors_dt,
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
    player_vocab = PlayerVocab.from_ids(akde.players)
    opp_vocab = OpponentVocab.from_ids(sorted(shots["opponent"].unique().tolist()))
    vocab_ids = [int(pid) for pid in player_vocab.ids]
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
        for i, pid in enumerate(player_vocab.ids)
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
    analogue_cache = build_analogue_cache(traits, L=3, ensure_self=True)
    collab = CollaborativeKDE(
        adaptive_kde=akde,
        snapshot_store=store,
        traits_table=traits,
        analogue_cache=analogue_cache,
        vocab=player_vocab,
        grid=grid,
    )

    # Defensive cache, features, and field.
    cache_cfg = DefensiveRetrievalCacheConfig(
        shots_fingerprint="smoke_v0",
        anchor_dates=tuple(int(d) for d in anchors_np),
        defensive_support_max=16,
        defensive_recency_window_days=365,
    )
    def_cache = build_defensive_retrieval_cache(
        shots_df=shots,
        opp_vocab=opp_vocab,
        anchor_dates=anchors_np,
        config=cache_cfg,
    )
    feat_cfg = DefenseFeaturesConfig(
        shots_fingerprint="smoke_v0",
        anchor_dates=tuple(int(d) for d in anchors_np),
        window_days=365,
        half_life_days=90.0,
    )
    def_features = build_defense_features(
        shots_df=shots,
        opp_vocab=opp_vocab,
        anchor_dates=anchors_np,
        config=feat_cfg,
    )
    def_field = ContinuousAdaptiveDefensiveField(
        n_opponents=len(opp_vocab),
        context_dim=CONTEXT_DIM,
        within_game_dim=0,
        defense_feature_dim=DEFENSE_FEATURE_DIM,
        opp_embed_dim=4,
        query_hidden_dim=16,
        key_hidden_dim=16,
        proj_dim=8,
        beta_init=1e-3,
        query_chunk_size=16,
    )

    train_set = GibbsShotDataset(
        shots_df=shots,
        snapshot_store=store,
        grid=grid,
        player_vocab=player_vocab,
        opp_vocab=opp_vocab,
        context_encoder=enc,
    )
    return {
        "shots": shots,
        "grid": grid,
        "store": store,
        "encoder": enc,
        "player_vocab": player_vocab,
        "opp_vocab": opp_vocab,
        "collab": collab,
        "train_set": train_set,
        "def_cache": def_cache,
        "def_features": def_features,
        "def_field": def_field,
    }


def test_training_with_defense_runs_two_epochs_end_to_end() -> None:
    """Two epochs with defense wired give finite losses and move the defensive
    parameters."""
    setup = _build_setup()
    residual = ContextResidualEncoder(rank=4, within_game_dim=0)
    loc = LocationEmbedding(rank=4)
    gate = PoolingGate(history_dim=0)

    # Record β_D and a query-network weight before training to check that they move.
    def_field = setup["def_field"]
    beta_before = float(def_field.beta_D.detach().clone().item())  # type: ignore[union-attr]
    q_weight_before = def_field.q_net[0].weight.detach().clone()  # type: ignore[union-attr, index]

    history = train_gibbs(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        count_head=NegBinCountHead(),
        timing_head=TimingSoftmaxHead(),
        context_mlp=ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True),
        train_set=setup["train_set"],  # type: ignore[arg-type]
        grid=setup["grid"],  # type: ignore[arg-type]
        residual_encoder=residual,
        location_embedding=loc,
        pooling_gate=gate,
        defensive_field_cellfree=setup["def_field"],  # type: ignore[arg-type]
        defensive_cache=setup["def_cache"],  # type: ignore[arg-type]
        defensive_features=setup["def_features"],  # type: ignore[arg-type]
        spatial_likelihood="continuous_mixture",
        lambda_timing=0.0,
        lambda_count=0.0,
        n_epochs=2,
        batch_size=16,
        learning_rate=1e-3,
        progress=False,
        restore_best_val=False,
    )

    # Training ran for both epochs with finite losses.
    assert len(history.train_spatial_mix_nll) == 2
    for v in history.train_spatial_mix_nll:
        assert np.isfinite(v), f"train mix_nll not finite: {v}"

    # An optimizer step on a nonzero gradient changes β_D or the query network.
    beta_after = float(def_field.beta_D.detach().item())  # type: ignore[union-attr]
    q_weight_after = def_field.q_net[0].weight.detach()  # type: ignore[union-attr, index]
    moved = not np.isclose(beta_before, beta_after) or not torch.allclose(
        q_weight_before, q_weight_after, atol=1e-7, rtol=1e-7
    )
    assert moved, (
        "neither β_D nor q_net[0].weight changed across two training epochs — "
        "the optimizer is not stepping on defensive parameters"
    )


def test_defense_parameters_join_optimizer() -> None:
    """After one epoch, ``β_D``, ``q_net``, ``k_net`` and the opponent embedding all have
    gradients populated."""
    setup = _build_setup()
    residual = ContextResidualEncoder(rank=4, within_game_dim=0)
    loc = LocationEmbedding(rank=4)
    gate = PoolingGate(history_dim=0)
    def_field = setup["def_field"]
    # Train a deep copy so its gradients can be inspected afterwards.
    def_field = copy.deepcopy(def_field)
    setup["def_field"] = def_field

    train_gibbs(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        count_head=NegBinCountHead(),
        timing_head=TimingSoftmaxHead(),
        context_mlp=ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True),
        train_set=setup["train_set"],  # type: ignore[arg-type]
        grid=setup["grid"],  # type: ignore[arg-type]
        residual_encoder=residual,
        location_embedding=loc,
        pooling_gate=gate,
        defensive_field_cellfree=def_field,
        defensive_cache=setup["def_cache"],  # type: ignore[arg-type]
        defensive_features=setup["def_features"],  # type: ignore[arg-type]
        spatial_likelihood="continuous_mixture",
        lambda_timing=0.0,
        lambda_count=0.0,
        n_epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        progress=False,
        restore_best_val=False,
    )
    # Only check that backward populated the gradients; whether the trainer zeroes
    # them after the last step does not matter here.
    assert def_field.beta_D.grad is not None
    assert def_field.q_net[0].weight.grad is not None  # type: ignore[index]
    assert def_field.k_net[0].weight.grad is not None  # type: ignore[index]
    assert def_field.opp_embedding.weight.grad is not None


def test_train_gibbs_rejects_partial_defense_triple() -> None:
    """A continuous adaptive defensive field without ``defensive_cache`` and
    ``defensive_features`` raises ``ValueError``."""
    import pytest

    setup = _build_setup()
    with pytest.raises(ValueError, match="requires both defensive_cache"):
        train_gibbs(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            count_head=NegBinCountHead(),
            timing_head=TimingSoftmaxHead(),
            context_mlp=ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True),
            train_set=setup["train_set"],  # type: ignore[arg-type]
            grid=setup["grid"],  # type: ignore[arg-type]
            defensive_field_cellfree=setup["def_field"],  # type: ignore[arg-type]
            # defensive_cache and defensive_features omitted
            spatial_likelihood="continuous_mixture",
            n_epochs=1,
            batch_size=16,
            progress=False,
            restore_best_val=False,
        )


def test_train_gibbs_rejects_defense_with_non_continuous_mixture() -> None:
    """The defensive field requires ``spatial_likelihood='continuous_mixture'``; with
    ``'mode_mixture'`` the trainer raises ``NotImplementedError``."""
    import pytest

    setup = _build_setup()
    with pytest.raises(NotImplementedError, match="continuous_mixture"):
        train_gibbs(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            count_head=NegBinCountHead(),
            timing_head=TimingSoftmaxHead(),
            context_mlp=ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True),
            train_set=setup["train_set"],  # type: ignore[arg-type]
            grid=setup["grid"],  # type: ignore[arg-type]
            defensive_field_cellfree=setup["def_field"],  # type: ignore[arg-type]
            defensive_cache=setup["def_cache"],  # type: ignore[arg-type]
            defensive_features=setup["def_features"],  # type: ignore[arg-type]
            spatial_likelihood="mode_mixture",
            n_epochs=1,
            batch_size=16,
            progress=False,
            restore_best_val=False,
        )
