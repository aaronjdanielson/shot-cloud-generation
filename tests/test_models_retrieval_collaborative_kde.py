"""Tests for :class:`shotcloud.models.RetrievalCollaborativeKDE`.

PR3.2 acceptance criteria:

1. Forward returns a :class:`CollaborativeContinuousOutputs` with the
   correct shapes for the retrieval support layout (M = own_max +
   pooled_max).
2. ``own_mask`` agrees with the cache's ``own_mask`` (concatenated
   into the ``(B, M)`` slot layout — True only on the own block).
3. ``support_mask = own_mask | pooled_mask`` (no overlap because the
   cache enforces disjointness by construction).
4. Padded slots (cache index ``-1``) produce ``support_mask = False``.
5. Gradient flows to all learnable parameters (``b_same``,
   ``lambda_age``, ``a_0``/``a_M``/``a_S``/``a_R``, ``phi``,
   ``f_x``/``h_z`` for bilinear).
6. Step-0 σ equals ``sigma_init`` (a_M, a_S init to large negative so
   softplus ≈ 0).
7. The :class:`ContinuousMixtureSpatial` wrapper consumes the
   retrieval backend without change.
8. The pooling gate works under retrieval (``own_mask`` is the
   backend-agnostic field; the wrapper does not branch on backend).
9. ``train_gibbs`` accepts ``RetrievalCollaborativeKDE`` as
   offensive_prior with ``spatial_likelihood='continuous_mixture'``
   and produces finite loss after one epoch.
10. ``train_gibbs`` rejects ``RetrievalCollaborativeKDE +
    spatial_likelihood='mode_mixture'`` with an explicit
    ``NotImplementedError`` (the mode-mixture variant has not been
    ported to the retrieval support layout).
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
from shotcloud.models.collaborative_kde import CollaborativeContinuousOutputs
from shotcloud.models.retrieval_cache import (
    RetrievalCacheConfig,
    build_retrieval_cache,
)
from shotcloud.models.retrieval_collaborative_kde import RetrievalCollaborativeKDE
from shotcloud.training.dataset import PlayerVocab


def _build_setup(
    n_players: int = 6,
    n_shots_per_player: int = 35,
    own_support_max: int = 8,
    pooled_support_max: int = 6,
    pooled_recency_window_days: int = 365,
    seed: int = 0,
) -> dict[str, object]:
    """Synthetic fixture parallel to ``test_models_continuous_mixture_spatial._build_setup``
    but with a retrieval cache + RetrievalCollaborativeKDE in place of
    the L×R CollaborativeKDE."""
    rng = np.random.default_rng(seed)
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=14, ny=12)
    base_date = pd.Timestamp("2024-01-01")
    rows: list[dict[str, object]] = []
    for pid in range(1, n_players + 1):
        cx, cy = (0.0, 5.0) if pid <= n_players // 2 else (2.0, 8.0)
        for i in range(n_shots_per_player):
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
                    "game_id": f"{pid}_g{i // 5}",
                }
            )
    shots = pd.DataFrame(rows)
    # Mid-window anchor: leaves enough causal history per player for
    # the cache to populate AND post-anchor shots for the GibbsShotDataset
    # smoke trainer test.
    anchors = [np.datetime64("2024-01-15", "D")]
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )
    enc = ContextEncoder.fit(shots)
    vocab = PlayerVocab.from_ids(shots["player_id"].unique())
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
        for i, pid in enumerate(vocab_ids)
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

    anchor_dates_np = np.array(
        [np.asarray(b.anchor_date, dtype="datetime64[D]").astype(np.int64) for b in store.bundles],
        dtype=np.int64,
    )
    config = RetrievalCacheConfig(
        shots_fingerprint="synthetic_v0",
        anchor_dates=tuple(int(d) for d in anchor_dates_np),
        own_support_max=own_support_max,
        pooled_support_max=pooled_support_max,
        pooled_recency_window_days=pooled_recency_window_days,
        pooled_recency_half_life_days=30.0,
    )
    traits_tensor = torch.from_numpy(traits.traits.copy()).to(torch.float32)
    cache = build_retrieval_cache(
        shots_df=shots,
        player_vocab=vocab,
        anchor_dates=anchor_dates_np,
        traits=traits_tensor,
        config=config,
    )
    kde = RetrievalCollaborativeKDE(
        retrieval_cache=cache,
        shots_df=shots,
        context_encoder=enc,
        traits_table=traits,
        vocab=vocab,
    )
    return {
        "shots": shots,
        "grid": grid,
        "store": store,
        "encoder": enc,
        "vocab": vocab,
        "traits": traits,
        "cache": cache,
        "kde": kde,
    }


def _batch(setup: dict[str, object], n_batch: int = 4) -> dict[str, torch.Tensor]:
    vocab = setup["vocab"]
    n = min(n_batch, len(vocab))  # type: ignore[arg-type]
    enc = setup["encoder"]
    shots = setup["shots"]
    ctx = enc.transform(shots)  # type: ignore[attr-defined]
    player_idx = torch.arange(n, dtype=torch.long)
    snapshot_idx = torch.zeros(n, dtype=torch.long)
    x_n_raw = torch.from_numpy(ctx[:n]).float()
    x_n = x_n_raw.clone()
    shot_xy = torch.from_numpy(
        np.stack(
            [
                shots["x"].to_numpy()[:n],  # type: ignore[union-attr]
                shots["y"].to_numpy()[:n],  # type: ignore[union-attr]
            ],
            axis=-1,
        ).astype(np.float32)
    )
    return {
        "player_idx": player_idx,
        "snapshot_idx": snapshot_idx,
        "x_n_raw": x_n_raw,
        "x_n": x_n,
        "shot_xy": shot_xy,
    }


# ---------------------------------------------------------------------------
# Forward shape + dataclass contract
# ---------------------------------------------------------------------------


def test_forward_returns_collaborative_continuous_outputs_with_correct_shapes() -> None:
    setup = _build_setup()
    kde: RetrievalCollaborativeKDE = setup["kde"]  # type: ignore[assignment]
    batch = _batch(setup, n_batch=4)
    with torch.no_grad():
        out = kde.forward_continuous(
            batch["player_idx"], batch["snapshot_idx"], batch["x_n_raw"], batch["x_n"]
        )
    assert isinstance(out, CollaborativeContinuousOutputs)
    b = batch["player_idx"].shape[0]
    m = kde.M
    assert out.support_xy.shape == (b, m, 2)
    assert out.support_logits.shape == (b, m)
    assert out.support_mask.shape == (b, m)
    assert out.own_mask.shape == (b, m)
    assert out.support_shooter.shape == (b, m)
    assert out.sigma.shape == (b,)
    # Retrieval backend produces empty-L diagnostic tensors.
    assert out.analogue_idx.shape == (b, 0)
    assert out.alpha_scores.shape == (b, 0)
    # Dtypes.
    assert out.support_mask.dtype == torch.bool
    assert out.own_mask.dtype == torch.bool
    assert out.support_shooter.dtype == torch.int64


def test_m_equals_own_plus_pooled_caps() -> None:
    setup = _build_setup(own_support_max=8, pooled_support_max=6)
    kde: RetrievalCollaborativeKDE = setup["kde"]  # type: ignore[assignment]
    assert kde.own_support_max == 8
    assert kde.pooled_support_max == 6
    assert kde.M == 14


# ---------------------------------------------------------------------------
# Mask invariants
# ---------------------------------------------------------------------------


def test_own_mask_matches_cache_own_block() -> None:
    """own_mask in the (B, M) slot layout is exactly the cache's
    per-player own_mask gathered for the batch's (player_idx,
    snapshot_idx) and concatenated with False on the pooled block."""
    setup = _build_setup()
    kde: RetrievalCollaborativeKDE = setup["kde"]  # type: ignore[assignment]
    cache = setup["cache"]
    batch = _batch(setup, n_batch=4)
    with torch.no_grad():
        out = kde.forward_continuous(
            batch["player_idx"], batch["snapshot_idx"], batch["x_n_raw"], batch["x_n"]
        )
    own_max = kde.own_support_max
    expected_own = cache.own_mask[batch["player_idx"], batch["snapshot_idx"]].to(torch.bool)  # type: ignore[attr-defined]
    torch.testing.assert_close(out.own_mask[:, :own_max], expected_own)
    # Pooled block of own_mask is identically False.
    assert not out.own_mask[:, own_max:].any().item()


def test_support_mask_is_own_or_pooled_block() -> None:
    """support_mask = (own block from cache.own_mask) ∪ (pooled block
    from cache.pooled_mask). The two blocks are disjoint by layout."""
    setup = _build_setup()
    kde: RetrievalCollaborativeKDE = setup["kde"]  # type: ignore[assignment]
    cache = setup["cache"]
    batch = _batch(setup, n_batch=4)
    with torch.no_grad():
        out = kde.forward_continuous(
            batch["player_idx"], batch["snapshot_idx"], batch["x_n_raw"], batch["x_n"]
        )
    own_max = kde.own_support_max
    expected_own = cache.own_mask[batch["player_idx"], batch["snapshot_idx"]].to(torch.bool)  # type: ignore[attr-defined]
    expected_pool = cache.pooled_mask[batch["player_idx"], batch["snapshot_idx"]].to(torch.bool)  # type: ignore[attr-defined]
    torch.testing.assert_close(out.support_mask[:, :own_max], expected_own)
    torch.testing.assert_close(out.support_mask[:, own_max:], expected_pool)
    # own_mask is a subset of support_mask.
    assert (~out.own_mask | out.support_mask).all().item()


def test_padded_slots_have_false_support_mask() -> None:
    """A small own/pooled cap that exceeds available causal shots
    leaves trailing slots padded (-1 indices, False masks). Verify
    no padded slot leaks through support_mask."""
    # Cap larger than available causal own-history per player; the
    # synthetic fixture gives ~35 own shots per player, so set cap=64.
    setup = _build_setup(own_support_max=64, pooled_support_max=64)
    kde: RetrievalCollaborativeKDE = setup["kde"]  # type: ignore[assignment]
    cache = setup["cache"]
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out = kde.forward_continuous(
            batch["player_idx"], batch["snapshot_idx"], batch["x_n_raw"], batch["x_n"]
        )
    # The cache must have some -1 slots given cap >> shots/player.
    own_idx_b = cache.own_idx[batch["player_idx"], batch["snapshot_idx"]]  # type: ignore[attr-defined]
    assert (own_idx_b == -1).any().item(), "expected some -1 padding in own_idx"
    # Every padded slot maps to support_mask=False.
    pad_slot = own_idx_b == -1
    assert not out.support_mask[:, : kde.own_support_max][pad_slot].any().item()


def test_pooled_block_shooters_are_not_target_player() -> None:
    """Pooled retrieval excludes the target player; verify the per-slot
    shooter identity respects that on the pooled block where the mask
    is True."""
    setup = _build_setup()
    kde: RetrievalCollaborativeKDE = setup["kde"]  # type: ignore[assignment]
    batch = _batch(setup, n_batch=4)
    with torch.no_grad():
        out = kde.forward_continuous(
            batch["player_idx"], batch["snapshot_idx"], batch["x_n_raw"], batch["x_n"]
        )
    own_max = kde.own_support_max
    pooled_mask = out.support_mask[:, own_max:]
    pooled_shooter = out.support_shooter[:, own_max:]
    for b_idx in range(batch["player_idx"].shape[0]):
        valid = pooled_mask[b_idx]
        if not valid.any():
            continue
        shooters = pooled_shooter[b_idx][valid]
        assert (shooters != batch["player_idx"][b_idx]).all().item(), (
            f"pooled block contains target shooter at row {b_idx}"
        )


def test_own_block_shooters_are_target_player_only() -> None:
    """On the own block, valid slots must point at the target shooter."""
    setup = _build_setup()
    kde: RetrievalCollaborativeKDE = setup["kde"]  # type: ignore[assignment]
    batch = _batch(setup, n_batch=4)
    with torch.no_grad():
        out = kde.forward_continuous(
            batch["player_idx"], batch["snapshot_idx"], batch["x_n_raw"], batch["x_n"]
        )
    own_max = kde.own_support_max
    own_mask = out.support_mask[:, :own_max]
    own_shooter = out.support_shooter[:, :own_max]
    for b_idx in range(batch["player_idx"].shape[0]):
        valid = own_mask[b_idx]
        if not valid.any():
            continue
        shooters = own_shooter[b_idx][valid]
        assert (shooters == batch["player_idx"][b_idx]).all().item()


# ---------------------------------------------------------------------------
# Step-0 invariants
# ---------------------------------------------------------------------------


def test_step0_sigma_equals_init() -> None:
    """At step 0 (a_M, a_S init = -10 so softplus ≈ 0), σ = sigma_init.
    Tolerance accommodates the residual softplus(-10) ≈ 4.5e-5 from the
    evidence terms (same convention as CollaborativeKDE)."""
    setup = _build_setup()
    kde: RetrievalCollaborativeKDE = setup["kde"]  # type: ignore[assignment]
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out = kde.forward_continuous(
            batch["player_idx"], batch["snapshot_idx"], batch["x_n_raw"], batch["x_n"]
        )
    np.testing.assert_allclose(out.sigma.numpy(), np.full(3, kde.sigma_init), atol=1e-2)


def test_step0_support_logits_equal_b_same_on_own_block_only() -> None:
    """At step 0, phi output layer is zero-init and h_z output layer is
    zero-init (bilinear default) → A and B contribute 0; the only
    score signal is ``b_same * is_self`` plus the recency term
    ``-lambda_age * Δt`` (lambda_age init = 0). So support_logits are
    zero everywhere when b_same=0, and exactly ``b_same`` on own slots
    when b_same > 0 (with lambda_age held at 0)."""
    setup = _build_setup()
    kde: RetrievalCollaborativeKDE = setup["kde"]  # type: ignore[assignment]
    batch = _batch(setup, n_batch=3)
    with torch.no_grad():
        out = kde.forward_continuous(
            batch["player_idx"], batch["snapshot_idx"], batch["x_n_raw"], batch["x_n"]
        )
    # b_same default is 0 → all logits are 0 at step 0 (the recency
    # term is also 0 because lambda_age init = 0).
    torch.testing.assert_close(out.support_logits, torch.zeros_like(out.support_logits))


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["bilinear", "concat"])
def test_gradient_flows_to_all_learnable_params(form: str) -> None:
    setup = _build_setup()
    # Rebuild kde with the requested shot_attention_form.
    cache = setup["cache"]
    traits = setup["traits"]
    enc = setup["encoder"]
    vocab = setup["vocab"]
    shots = setup["shots"]
    kde = RetrievalCollaborativeKDE(
        retrieval_cache=cache,  # type: ignore[arg-type]
        shots_df=shots,  # type: ignore[arg-type]
        context_encoder=enc,  # type: ignore[arg-type]
        traits_table=traits,  # type: ignore[arg-type]
        vocab=vocab,  # type: ignore[arg-type]
        shot_attention_form=form,  # type: ignore[arg-type]
    )
    batch = _batch(setup, n_batch=3)
    out = kde.forward_continuous(
        batch["player_idx"], batch["snapshot_idx"], batch["x_n_raw"], batch["x_n"]
    )
    # Mask invalid slots and take a simple loss over the support
    # logits + sigma so every learnable param is reachable.
    masked_logits = out.support_logits.masked_fill(~out.support_mask, 0.0)
    loss = masked_logits.sum() + out.sigma.sum()
    loss.backward()

    # Scalars.
    for name in ("b_same", "lambda_M", "lambda_S", "lambda_age", "a_0", "a_M", "a_S", "a_R"):
        p = getattr(kde, name)
        assert p.grad is not None, f"{name} has no grad"

    # phi first Linear: gradient tensor allocated. At step 0 the
    # output-layer zero-init zeros the gradient in expectation
    # (same convention as the L×R CollaborativeKDE step-0 test).
    assert kde.phi[0].weight.grad is not None  # type: ignore[union-attr]

    # Shot-attention path: gradient tensor allocated.
    if form == "bilinear":
        assert kde.f_x[0].weight.grad is not None  # type: ignore[union-attr]
        assert kde.h_z[0].weight.grad is not None  # type: ignore[union-attr]
    else:
        assert kde.g[0].weight.grad is not None  # type: ignore[union-attr]


def test_lambda_age_receives_gradient_when_perturbed() -> None:
    """lambda_age starts at 0 so its gradient at step 0 is zero in
    expectation through the loss above. Perturb it off zero and verify
    autograd is not blocked."""
    setup = _build_setup()
    kde: RetrievalCollaborativeKDE = setup["kde"]  # type: ignore[assignment]
    with torch.no_grad():
        kde.lambda_age.fill_(0.01)
    batch = _batch(setup, n_batch=3)
    out = kde.forward_continuous(
        batch["player_idx"], batch["snapshot_idx"], batch["x_n_raw"], batch["x_n"]
    )
    loss = out.support_logits.masked_fill(~out.support_mask, 0.0).sum()
    loss.backward()
    assert kde.lambda_age.grad is not None
    assert kde.lambda_age.grad.abs().item() > 0


# ---------------------------------------------------------------------------
# Wrapper integration (ContinuousMixtureSpatial)
# ---------------------------------------------------------------------------


def test_wrapper_consumes_retrieval_backend() -> None:
    """``ContinuousMixtureSpatial`` wraps either backend uniformly.
    The retrieval backend should produce a finite per-row log-lik
    via the same forward call (no branching at the wrapper)."""
    from shotcloud.models.continuous_mixture_spatial import (
        ContinuousMixtureOutputs,
        ContinuousMixtureSpatial,
    )

    setup = _build_setup()
    spatial = ContinuousMixtureSpatial(offensive_prior=setup["kde"])  # type: ignore[arg-type]
    batch = _batch(setup, n_batch=4)
    out = spatial(**batch)
    assert isinstance(out, ContinuousMixtureOutputs)
    assert out.log_lik.shape == (batch["player_idx"].shape[0],)
    assert torch.isfinite(out.log_lik).all()


def test_wrapper_with_pooling_gate_under_retrieval() -> None:
    """Pooling gate works under retrieval: ``own_mask`` is the
    backend-agnostic field that drives the gate's own/pooled
    partition. λ should populate and lie in [0, 1]."""
    from shotcloud.models.continuous_mixture_spatial import ContinuousMixtureSpatial
    from shotcloud.models.pooling_gate import PoolingGate

    setup = _build_setup()
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["kde"],  # type: ignore[arg-type]
        pooling_gate=PoolingGate(),
    )
    batch = _batch(setup, n_batch=4)
    out = spatial(**batch)
    assert out.gate_lambda is not None
    assert out.gate_lambda.shape == (batch["player_idx"].shape[0],)
    assert (out.gate_lambda >= 0.0).all().item()
    assert (out.gate_lambda <= 1.0).all().item()
    assert torch.isfinite(out.log_lik).all()


def test_wrapper_gate_lambda_one_recovers_own_only_loglik_under_retrieval() -> None:
    """Forcing λ→1 (huge intercept) makes the gated log-lik equal an
    own-only continuous-mixture log-lik for the retrieval backend
    (same invariant as the L×R case)."""
    from shotcloud.models.continuous_mixture_spatial import ContinuousMixtureSpatial
    from shotcloud.models.pooling_gate import PoolingGate
    from shotcloud.training.spatial_losses import continuous_mixture_loglik

    setup = _build_setup()
    gate = PoolingGate()
    with torch.no_grad():
        gate.b0.fill_(40.0)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["kde"],  # type: ignore[arg-type]
        pooling_gate=gate,
    )
    batch = _batch(setup, n_batch=4)
    with torch.no_grad():
        out = spatial(**batch)
    # Reference: own-only continuous-mixture log-lik from a raw forward.
    kde: RetrievalCollaborativeKDE = setup["kde"]  # type: ignore[assignment]
    with torch.no_grad():
        coll = kde.forward_continuous(
            batch["player_idx"], batch["snapshot_idx"], batch["x_n_raw"], batch["x_n"]
        )
        own_ref = continuous_mixture_loglik(
            coll.support_logits,
            coll.support_xy,
            batch["shot_xy"],
            coll.sigma,
            weights_are_log_probs=False,
            support_mask=coll.own_mask,
        )
    torch.testing.assert_close(out.log_lik, own_ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Trainer smoke
# ---------------------------------------------------------------------------


def test_train_gibbs_continuous_mixture_with_retrieval_backend_runs() -> None:
    """End-to-end smoke: continuous-mixture trainer with the retrieval
    backend produces finite NLL after one epoch and supports the
    pooling gate."""
    from shotcloud.models import ContextMLP, NegBinCountHead, TimingSoftmaxHead
    from shotcloud.models.pooling_gate import PoolingGate
    from shotcloud.training import GibbsShotDataset, train_gibbs

    setup = _build_setup()
    shots = setup["shots"]
    store = setup["store"]
    grid = setup["grid"]
    enc = setup["encoder"]
    vocab = setup["vocab"]
    train_set = GibbsShotDataset(
        shots_df=shots,  # type: ignore[arg-type]
        snapshot_store=store,  # type: ignore[arg-type]
        grid=grid,  # type: ignore[arg-type]
        player_vocab=vocab,  # type: ignore[arg-type]
        opp_vocab=None,
        context_encoder=enc,  # type: ignore[arg-type]
    )
    history = train_gibbs(
        offensive_prior=setup["kde"],  # type: ignore[arg-type]
        count_head=NegBinCountHead(),
        timing_head=TimingSoftmaxHead(),
        context_mlp=ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True),
        train_set=train_set,
        grid=grid,  # type: ignore[arg-type]
        pooling_gate=PoolingGate(history_dim=0),
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
    assert np.isfinite(history.train_gate_lambda_mean[0])


def test_train_gibbs_rejects_retrieval_with_mode_mixture() -> None:
    """``RetrievalCollaborativeKDE + mode_mixture`` is not supported;
    the trainer must raise an explicit ``NotImplementedError`` rather
    than silently producing a wrong forward."""
    from shotcloud.models import ContextMLP, NegBinCountHead, TimingSoftmaxHead
    from shotcloud.training import GibbsShotDataset, train_gibbs

    setup = _build_setup()
    shots = setup["shots"]
    store = setup["store"]
    grid = setup["grid"]
    enc = setup["encoder"]
    vocab = setup["vocab"]
    train_set = GibbsShotDataset(
        shots_df=shots,  # type: ignore[arg-type]
        snapshot_store=store,  # type: ignore[arg-type]
        grid=grid,  # type: ignore[arg-type]
        player_vocab=vocab,  # type: ignore[arg-type]
        opp_vocab=None,
        context_encoder=enc,  # type: ignore[arg-type]
    )
    with pytest.raises(NotImplementedError, match="mode_mixture"):
        train_gibbs(
            offensive_prior=setup["kde"],  # type: ignore[arg-type]
            count_head=NegBinCountHead(),
            timing_head=TimingSoftmaxHead(),
            context_mlp=ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True),
            train_set=train_set,
            grid=grid,  # type: ignore[arg-type]
            spatial_likelihood="mode_mixture",
            n_epochs=1,
            batch_size=16,
            progress=False,
            restore_best_val=False,
        )


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_construction_rejects_traits_with_mismatched_player_count() -> None:
    setup = _build_setup()
    cache = setup["cache"]
    traits = setup["traits"]
    enc = setup["encoder"]
    shots = setup["shots"]
    # Vocab with fewer players than the traits table.
    bad_vocab = PlayerVocab.from_ids([str(v) for v in setup["vocab"].ids[:-1]])  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="n_players"):
        RetrievalCollaborativeKDE(
            retrieval_cache=cache,  # type: ignore[arg-type]
            shots_df=shots,  # type: ignore[arg-type]
            context_encoder=enc,  # type: ignore[arg-type]
            traits_table=traits,  # type: ignore[arg-type]
            vocab=bad_vocab,
        )


def test_construction_rejects_invalid_sigma_bounds() -> None:
    setup = _build_setup()
    cache = setup["cache"]
    traits = setup["traits"]
    enc = setup["encoder"]
    vocab = setup["vocab"]
    shots = setup["shots"]
    with pytest.raises(ValueError, match="sigma"):
        RetrievalCollaborativeKDE(
            retrieval_cache=cache,  # type: ignore[arg-type]
            shots_df=shots,  # type: ignore[arg-type]
            context_encoder=enc,  # type: ignore[arg-type]
            traits_table=traits,  # type: ignore[arg-type]
            vocab=vocab,  # type: ignore[arg-type]
            sigma_min=4.0,
            sigma_max=1.0,
            sigma_init=1.5,
        )
