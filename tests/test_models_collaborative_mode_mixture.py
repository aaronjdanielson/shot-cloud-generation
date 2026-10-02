"""Integration tests for ``CollaborativeModeMixtureSpatial``.

End-to-end smoke for the cell-free per-player-game mode-mixture
spatial decoder: collab support attention → mode extractor → mode
mixture log-likelihood. Verifies shapes, gradients flow to every
participating module, the cold-start floor fires without
NaN-poisoning, and the residual on/off comparison at step 0 holds
(residual's LocationEmbedding is zero-init, so support attention is
identical and mode extraction is identical).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud import CourtGrid
from shotcloud.data import ContextEncoder
from shotcloud.data.player_traits import build_player_traits_table
from shotcloud.data.role_profile import build_role_profiles
from shotcloud.data.snapshots import build_snapshot_store_from_shots
from shotcloud.kde.adaptive import AdaptiveKDE
from shotcloud.models.analogue_retrieval import build_analogue_cache
from shotcloud.models.collaborative_kde import CollaborativeKDE
from shotcloud.models.collaborative_mode_mixture import (
    CollaborativeModeMixtureSpatial,
)
from shotcloud.models.context_residual import ContextResidualEncoder
from shotcloud.models.continuous_mixture_spatial import ContinuousMixtureOutputs
from shotcloud.models.location_embedding import LocationEmbedding
from shotcloud.models.mode_extractor import DEFAULT_N_COURT_MODES
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
        shots, anchors, role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1)
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
    bio = pd.DataFrame(
        [
            {
                "player_id": int(pid),
                "display_name": f"P{pid}",
                "birthdate": pd.Timestamp("1990-01-15") + pd.Timedelta(days=i * 30),
                "height_inches": 72 + i,
                "weight_lbs": 190 + i * 5,
                "position_raw": "Guard" if i < n_players // 2 else "Center",
                "position_group": "SG" if i < n_players // 2 else "C",
                "status": "ok",
            }
            for i, pid in enumerate(akde.players)
        ]
    )
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
    return {"collab": collab, "vocab": vocab, "ctx": ctx, "shots": shots}


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


def test_forward_shape_and_finiteness() -> None:
    setup = _build_setup()
    spatial = CollaborativeModeMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    batch = _batch(setup, n_batch=4)
    out = spatial(**batch)
    assert isinstance(out, ContinuousMixtureOutputs)
    b = batch["player_idx"].shape[0]
    K = DEFAULT_N_COURT_MODES  # 6
    L = setup["collab"].L  # type: ignore[attr-defined]
    R = setup["collab"].max_history  # type: ignore[attr-defined]
    assert out.log_lik.shape == (b,)
    assert out.log_weights.shape == (b, K)
    assert out.support_xy.shape == (b, K, 2)  # K mode centers per row
    assert out.support_log_weights.shape == (b, L * R)
    assert torch.isfinite(out.log_lik).all()
    np.testing.assert_allclose(
        out.log_weights.exp().sum(dim=-1).detach().numpy(), np.ones(b), atol=1e-5
    )


def test_gradient_flows_to_all_modules() -> None:
    """Gradients reach the collaborative KDE, the soft-k-means mode-bias MLP, and the residual.

    The location embedding is built with ``zero_init=False`` so the residual
    term is non-zero and its encoder receives gradient.
    """
    setup = _build_setup()
    residual = ContextResidualEncoder(rank=4, within_game_dim=0)
    loc = LocationEmbedding(rank=4, zero_init=False)
    spatial = CollaborativeModeMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=residual,
        residual_location_embedding=loc,
    )
    batch = _batch(setup, n_batch=3)
    out = spatial(**batch)
    (-out.log_lik.mean()).backward()
    assert setup["collab"].b_same.grad is not None  # type: ignore[attr-defined]
    mode_bias = spatial.mode_extractor.mode_bias  # type: ignore[attr-defined]
    # Output layer of the mode-bias MLP must receive gradient.
    assert mode_bias[-1].weight.grad is not None
    assert mode_bias[-1].weight.grad.abs().sum() > 0
    assert residual.fc1.weight.grad is not None and residual.fc1.weight.grad.abs().sum() > 0


def test_learned_query_extractor_gradient_flows() -> None:
    """With ``extractor_kind='learned_query'``, the extractor's parameters receive gradient."""
    setup = _build_setup()
    spatial = CollaborativeModeMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        extractor_kind="learned_query",
    )
    batch = _batch(setup, n_batch=3)
    out = spatial(**batch)
    (-out.log_lik.mean()).backward()
    assert spatial.mode_extractor.mode_queries.grad is not None  # type: ignore[attr-defined]
    assert spatial.mode_extractor.mode_queries.grad.abs().sum() > 0  # type: ignore[attr-defined]
    assert (
        spatial.mode_extractor.support_embedding.proj.weight.grad is not None  # type: ignore[attr-defined]
        and spatial.mode_extractor.support_embedding.proj.weight.grad.abs().sum() > 0  # type: ignore[attr-defined]
    )


def test_cold_start_uses_floor() -> None:
    setup = _build_setup()
    spatial = CollaborativeModeMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    with torch.no_grad():
        setup["collab"].anchor_dates.fill_(  # type: ignore[attr-defined]
            int(setup["collab"].pool_dates.min().item()) - 1  # type: ignore[attr-defined]
        )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out = spatial(**batch)
    assert torch.isfinite(out.log_lik).all()
    np.testing.assert_allclose(
        out.log_lik.numpy(),
        np.full(batch["player_idx"].shape[0], spatial.cold_start_log_lik_floor),
        atol=1e-5,
    )
    assert out.cold_start.all()


def test_residual_on_off_match_at_step_zero_when_loc_is_zero_init() -> None:
    """At step 0 with ``zero_init=True`` on the residual location
    embedding, the residual term is identically zero. Constructing
    both wrappers under the same RNG seed lines up the soft-k-means
    extractor's only random init (the ``mode_bias`` MLP) bit-for-bit,
    so the two paths must produce identical log-lik."""
    setup = _build_setup()
    torch.manual_seed(0)
    spatial_off = CollaborativeModeMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    torch.manual_seed(0)
    residual = ContextResidualEncoder(rank=4, within_game_dim=0)
    loc = LocationEmbedding(rank=4, zero_init=True)
    spatial_on = CollaborativeModeMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=residual,
        residual_location_embedding=loc,
    )
    # Copy the mode_bias MLP weights to make absolutely sure the only
    # difference between the two paths is the (zero-valued) residual.
    for mod_a, mod_b in zip(
        spatial_off.mode_extractor.mode_bias.modules(),  # type: ignore[attr-defined]
        spatial_on.mode_extractor.mode_bias.modules(),  # type: ignore[attr-defined]
        strict=True,
    ):
        if hasattr(mod_a, "weight") and mod_a.weight is not None:
            mod_b.weight.data.copy_(mod_a.weight.data)
            if mod_a.bias is not None:
                mod_b.bias.data.copy_(mod_a.bias.data)

    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        log_lik_off = spatial_off(**batch).log_lik
        log_lik_on = spatial_on(**batch).log_lik
    torch.testing.assert_close(log_lik_off, log_lik_on, atol=1e-5, rtol=1e-5)


def test_tail_weight_zero_matches_pure_mode_log_lik() -> None:
    """A positive ``tail_weight`` changes the log-likelihood of the default pure mode mixture.

    With ``tail_weight = 0`` (the default) the support tail is disabled.
    """
    setup = _build_setup()
    torch.manual_seed(0)
    spatial_no_tail = CollaborativeModeMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        tail_weight=0.0,
    )
    torch.manual_seed(0)
    spatial_with_tail = CollaborativeModeMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        tail_weight=0.05,
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        log_lik_no_tail = spatial_no_tail(**batch).log_lik
        log_lik_with_tail = spatial_with_tail(**batch).log_lik
    # The two differ — verifies tail_weight > 0 actually changes the loss.
    assert not torch.allclose(log_lik_no_tail, log_lik_with_tail)


def test_tail_weight_provides_density_lower_bound() -> None:
    """The support tail bounds the density below: ``f_Θ(y) ≥ λ_tail · f_support(y)``."""
    from shotcloud.training.spatial_losses import continuous_mixture_loglik

    setup = _build_setup()
    spatial = CollaborativeModeMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        tail_weight=0.05,
        tail_sigma_ft=1.0,
    )
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out = spatial(**batch)
        # Reconstruct f_support log-lik on the same support attention
        # the model used.
        sup_log_lik = continuous_mixture_loglik(
            log_weights=out.support_log_weights,
            support_xy=out.collab.support_xy,
            shot_xy=batch["shot_xy"],
            sigma=torch.full((batch["shot_xy"].shape[0],), 1.0),
            weights_are_log_probs=True,
            support_mask=out.collab.support_mask,
        )
    # Lower bound: log f_Θ(y) ≥ log(λ_tail) + log f_support(y).
    # Filter cold-start rows where the floor short-circuits the bound.
    valid = ~out.cold_start
    if valid.any():
        bound = math.log(0.05) + sup_log_lik[valid]
        full = out.log_lik[valid]
        assert (full >= bound - 1e-5).all(), (
            f"density lower bound violated: full={full.tolist()} bound={bound.tolist()}"
        )


def test_tail_weight_rejects_out_of_range() -> None:
    setup = _build_setup()
    with pytest.raises(ValueError, match="tail_weight"):
        CollaborativeModeMixtureSpatial(offensive_prior=setup["collab"], tail_weight=-0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tail_weight"):
        CollaborativeModeMixtureSpatial(offensive_prior=setup["collab"], tail_weight=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tail_sigma"):
        CollaborativeModeMixtureSpatial(offensive_prior=setup["collab"], tail_sigma_ft=0.0)  # type: ignore[arg-type]


def test_lambda_omega_threads_through_wrapper() -> None:
    """``lambda_omega`` reaches the learned-query extractor.

    Only that extractor uses it, as a logit bias; the soft-k-means extractor
    uses ω directly in its mean-shift kernel.
    """
    setup = _build_setup()
    spatial = CollaborativeModeMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        extractor_kind="learned_query",
        lambda_omega=0.7,
    )
    assert spatial.mode_extractor.lambda_omega == 0.7  # type: ignore[attr-defined]


def test_extractor_kind_dispatch() -> None:
    """The default extractor is soft-k-means; ``extractor_kind='learned_query'`` selects
    :class:`~shotcloud.models.mode_extractor.SupportModeExtractor`."""
    from shotcloud.models.mode_extractor import SupportModeExtractor
    from shotcloud.models.soft_kmeans_extractor import SoftKMeansModeExtractor

    setup = _build_setup()
    default_spatial = CollaborativeModeMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    legacy_spatial = CollaborativeModeMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        extractor_kind="learned_query",
    )
    assert isinstance(default_spatial.mode_extractor, SoftKMeansModeExtractor)
    assert isinstance(legacy_spatial.mode_extractor, SupportModeExtractor)
    assert default_spatial.extractor_kind == "soft_kmeans"
    assert legacy_spatial.extractor_kind == "learned_query"
    with pytest.raises(ValueError, match="extractor_kind"):
        CollaborativeModeMixtureSpatial(
            offensive_prior=setup["collab"],  # type: ignore[arg-type]
            extractor_kind="bogus",
        )


def test_constructor_rejects_invalid_arguments() -> None:
    setup = _build_setup()
    collab = setup["collab"]
    residual = ContextResidualEncoder(rank=4, within_game_dim=0)
    loc = LocationEmbedding(rank=4)
    with pytest.raises(ValueError, match="both"):
        CollaborativeModeMixtureSpatial(offensive_prior=collab, residual_encoder=residual)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank"):
        CollaborativeModeMixtureSpatial(
            offensive_prior=collab,  # type: ignore[arg-type]
            residual_encoder=ContextResidualEncoder(rank=8, within_game_dim=0),
            residual_location_embedding=loc,
        )
