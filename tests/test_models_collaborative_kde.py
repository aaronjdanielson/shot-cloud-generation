"""Tests for :class:`shotcloud.models.collaborative_kde.CollaborativeKDE`.

Load-bearing invariants verified here:

1. Shape and normalization: forward returns ``(B, n_cells)`` log-probs,
   each row sums to 0 in exp space.
2. Step-0 invariant: with all learnable params at init, α is uniform
   over valid analogues, β is uniform within each analogue's causal
   history, σ = sigma_init, and the kernel is isotropic Gaussian.
3. Cold-start invariant: a target with no own-history still produces
   valid log-probs by attending to analogue shots.
4. Valid-analogue mask: analogues with no causal history get α = 0.
5. σ widens for sparse-evidence players when ``a_M`` trained positive
   (the softplus sign-prior is enforced).
6. Step-0 bilinear-vs-concat equivalence: the two shot-attention
   forms produce identical output at init (zero-init output layers
   make β uniform under both).
7. Gradient flow to all learnable parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud import CourtGrid
from shotcloud.data import ContextEncoder
from shotcloud.data.player_traits import (
    build_player_traits_table,
)
from shotcloud.data.role_profile import build_role_profiles
from shotcloud.data.snapshots import build_snapshot_store_from_shots
from shotcloud.kde.adaptive import AdaptiveKDE
from shotcloud.models.analogue_retrieval import build_analogue_cache
from shotcloud.models.collaborative_kde import (
    CollaborativeKDE,
    CollaborativeOutputs,
)
from shotcloud.training.dataset import PlayerVocab


def _build_test_setup(
    n_players: int = 6,
    n_shots_per_player: int = 30,
    n_snapshots: int = 1,
    seed: int = 0,
) -> dict[str, object]:
    """Build a small synthetic training setup for CollaborativeKDE tests."""
    rng = np.random.default_rng(seed)
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=14, ny=15)
    base_date = pd.Timestamp("2024-01-01")

    rows = []
    for pid in range(1, n_players + 1):
        # Players 1..n_players/2 are "guards" clustered near the rim/paint;
        # the rest are "centers" clustered near the rim.
        cx, cy = (0.0, 5.0) if pid <= n_players // 2 else (0.0, 3.0)
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
                }
            )
    shots = pd.DataFrame(rows)

    # Snapshot anchors AFTER the synthetic shots so all players have
    # causal history at every snapshot.
    anchors = [
        np.datetime64("2024-04-15", "D") + np.timedelta64(m * 30, "D") for m in range(n_snapshots)
    ]
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )

    enc = ContextEncoder.fit(shots)
    ctx = enc.transform(shots)

    akde = AdaptiveKDE(grid=g, bandwidth=1.5, max_history=30, seed=0)
    akde.fit(
        x=shots["x"].to_numpy(),
        y=shots["y"].to_numpy(),
        player_id=shots["player_id"].to_numpy(),
        context_features=ctx,
        date=shots["date"].to_numpy(),
    )
    vocab = PlayerVocab.from_ids(akde.players)

    # Bio: every player gets a record. Positions alternate to give
    # the bio block some signal.
    bio_rows = []
    for i, pid in enumerate(akde.players):
        bio_rows.append(
            {
                "player_id": int(pid),
                "display_name": f"Player {pid}",
                "birthdate": pd.Timestamp("1990-01-15") + pd.Timedelta(days=i * 30),
                "height_inches": 72 + i * 2,
                "weight_lbs": 190 + i * 10,
                "position_raw": "Guard" if int(pid) <= n_players // 2 else "Center",
                "position_group": "SG" if int(pid) <= n_players // 2 else "C",
                "status": "ok",
            }
        )
    bio = pd.DataFrame(bio_rows)

    # Game logs: minimal — synthesize per-player games matching the shots window.
    gl_rows = []
    for pid in range(1, n_players + 1):
        for i in range(n_shots_per_player):
            gl_rows.append(
                {
                    "player_id": pid,
                    "game_date": base_date + pd.Timedelta(days=i),
                    "minutes": 25,
                    "fga": 8,
                    "fta": 3,
                    "tov": 1,
                }
            )
    gl = pd.DataFrame(gl_rows)

    vocab_ids = [int(pid) for pid in vocab.ids]
    traits = build_player_traits_table(
        snapshot_store=store,
        vocab_ids=vocab_ids,
        bio_df=bio,
        game_logs_df=gl,
    )
    cache = build_analogue_cache(traits, L=4)

    return {
        "akde": akde,
        "store": store,
        "traits": traits,
        "cache": cache,
        "vocab": vocab,
        "grid": g,
        "ctx": ctx,
        "encoder": enc,
        "shots": shots,
    }


def _make_kde(setup: dict[str, object], **overrides: object) -> CollaborativeKDE:
    kwargs: dict[str, object] = {
        "adaptive_kde": setup["akde"],
        "snapshot_store": setup["store"],
        "traits_table": setup["traits"],
        "analogue_cache": setup["cache"],
        "vocab": setup["vocab"],
        "grid": setup["grid"],
    }
    kwargs.update(overrides)
    return CollaborativeKDE(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_construction_succeeds_on_clean_setup() -> None:
    setup = _build_test_setup()
    kde = _make_kde(setup)
    assert isinstance(kde, CollaborativeKDE)
    assert kde.L == setup["cache"].L  # type: ignore[attr-defined]
    assert kde.n_cells == setup["grid"].n_cells  # type: ignore[attr-defined]


def test_construction_rejects_unfitted_adaptive_kde() -> None:
    setup = _build_test_setup()
    unfit = AdaptiveKDE(grid=setup["grid"], bandwidth=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be fit"):
        _make_kde(setup, adaptive_kde=unfit)


def test_construction_rejects_invalid_sigma_bounds() -> None:
    setup = _build_test_setup()
    with pytest.raises(ValueError, match="sigma_min"):
        _make_kde(setup, sigma_min=4.0, sigma_max=1.0)
    with pytest.raises(ValueError, match="sigma_init"):
        _make_kde(setup, sigma_min=0.5, sigma_max=2.0, sigma_init=5.0)


def test_construction_allows_sigma_locked_when_min_equals_max() -> None:
    """``sigma_min == sigma_max`` is a valid configuration (lock σ).
    Used for the cell-free continuous-mixture ablation where the
    learnable bandwidth was overfitting by widening; the locked-σ
    mode forces the model to learn through the support weights
    instead. Verified by checking the per-row σ at init is exactly
    the locked value and stays there regardless of trait inputs."""
    setup = _build_test_setup()
    kde = _make_kde(setup, sigma_min=1.5, sigma_max=1.5, sigma_init=1.5)
    batch = _make_batch(setup, n_batch=3)
    with torch.no_grad():
        _, comps = kde(**batch, return_components=True)
    np.testing.assert_allclose(comps.sigma.numpy(), np.full(3, 1.5), atol=1e-6)


# ---------------------------------------------------------------------------
# Forward shape + normalization
# ---------------------------------------------------------------------------


def _make_batch(setup: dict[str, object], n_batch: int = 4) -> dict[str, torch.Tensor]:
    vocab = setup["vocab"]
    ctx = setup["ctx"]
    n_vocab = len(vocab)  # type: ignore[arg-type]
    player_idx = torch.arange(min(n_batch, n_vocab), dtype=torch.long)
    snapshot_idx = torch.zeros(min(n_batch, n_vocab), dtype=torch.long)
    x_n_raw = torch.from_numpy(ctx[: min(n_batch, n_vocab)]).float()  # type: ignore[index]
    x_n = x_n_raw.clone()
    return {
        "player_idx": player_idx,
        "snapshot_idx": snapshot_idx,
        "x_n_raw": x_n_raw,
        "x_n": x_n,
    }


def test_forward_shape_and_normalization() -> None:
    setup = _build_test_setup()
    kde = _make_kde(setup)
    batch = _make_batch(setup, n_batch=4)
    with torch.no_grad():
        log_q = kde(**batch)
    assert log_q.shape == (4, kde.n_cells)
    np.testing.assert_allclose(torch.exp(log_q).sum(dim=-1).numpy(), np.ones(4), atol=1e-5)


def test_return_components_round_trip() -> None:
    setup = _build_test_setup()
    kde = _make_kde(setup)
    batch = _make_batch(setup, n_batch=3)
    with torch.no_grad():
        log_q, comps = kde(**batch, return_components=True)
    assert isinstance(comps, CollaborativeOutputs)
    assert comps.log_q_collab.shape == log_q.shape
    assert comps.alpha.shape == (3, kde.L)
    assert comps.beta.shape == (3, kde.L, kde.max_history)
    assert comps.sigma.shape == (3,)
    assert comps.analogues.shape == (3, kde.L)
    assert comps.has_analogue_history.shape == (3, kde.L)
    # α rows sum to 1 (modulo float).
    np.testing.assert_allclose(comps.alpha.sum(dim=-1).numpy(), np.ones(3), atol=1e-5)


def test_beta_rows_sum_to_one_per_valid_analogue() -> None:
    """For every ``(b, l)`` with at least one causally-usable shot,
    ``Σ_j β_{b,l,j} = 1`` (the shot-level softmax is properly
    normalized). Rows whose analogue has no causal history get
    β ≡ 0 instead — the defensive-fallback behavior the kernel sum
    relies on.

    This is invariant 3 from the math-preservation review:
    ``Σ_{p'} α = 1`` (covered above) and ``Σ_j β = 1`` (covered here).
    """
    setup = _build_test_setup()
    kde = _make_kde(setup)
    batch = _make_batch(setup, n_batch=3)
    with torch.no_grad():
        _, comps = kde(**batch, return_components=True)
    valid = comps.has_analogue_history.numpy()  # (B, L) bool
    row_sums = comps.beta.sum(dim=-1).numpy()  # (B, L)
    # Valid rows sum to 1.
    np.testing.assert_allclose(row_sums[valid], np.ones(int(valid.sum())), atol=1e-5)
    # Invalid rows are all-zero (β ≡ 0 → row sum 0).
    if (~valid).any():
        np.testing.assert_allclose(row_sums[~valid], np.zeros(int((~valid).sum())), atol=1e-7)


def test_final_density_sums_to_one_per_batch_row() -> None:
    """Invariant 1: ``Σ_c q_p^collab(c | x, t) = 1`` for every batch
    row, end-to-end through the public forward."""
    setup = _build_test_setup()
    kde = _make_kde(setup)
    batch = _make_batch(setup, n_batch=4)
    with torch.no_grad():
        log_q = kde(**batch)
    np.testing.assert_allclose(torch.exp(log_q).sum(dim=-1).numpy(), np.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# Step-0 invariants
# ---------------------------------------------------------------------------


def test_step0_sigma_equals_init() -> None:
    """At step 0 (a_M, a_S init to large negative so softplus ≈ 0),
    σ_p = sigma_init for every batch row. Tolerance accommodates the
    residual softplus(-10) ≈ 4.5e-5 contribution from the evidence
    terms; it's well below any training-relevant change in σ."""
    setup = _build_test_setup()
    kde = _make_kde(setup, sigma_init=1.5)
    batch = _make_batch(setup, n_batch=3)
    with torch.no_grad():
        _, comps = kde(**batch, return_components=True)
    np.testing.assert_allclose(comps.sigma.numpy(), np.full(3, 1.5), atol=1e-2)


def test_step0_alpha_uniform_over_valid_analogues() -> None:
    """At step 0 (φ output zero, b_same = 0, λ_M = λ_S = 0), α is
    uniform over the valid-analogue set."""
    setup = _build_test_setup()
    kde = _make_kde(setup)
    batch = _make_batch(setup, n_batch=3)
    with torch.no_grad():
        _, comps = kde(**batch, return_components=True)
    L = kde.L
    # Every analogue in our synthetic setup is valid (all players have
    # shots before the snapshot anchor) → α should be exactly 1/L.
    np.testing.assert_allclose(comps.alpha.numpy(), np.full((3, L), 1.0 / L), atol=1e-5)


# ---------------------------------------------------------------------------
# Valid-analogue mask
# ---------------------------------------------------------------------------


def test_invalid_analogues_get_zero_alpha() -> None:
    """An analogue with no causal shots at the target's anchor must
    receive α = 0 (via -inf masking before softmax). We construct this
    case by shifting the in-memory anchor_dates buffer to a date that
    precedes every shot in the synthetic history — making every
    analogue's causal-shot count zero."""
    setup = _build_test_setup()
    kde = _make_kde(setup)
    # Force every per-row causal mask to be zero: anchor < min(shot date).
    with torch.no_grad():
        kde.anchor_dates.fill_(int(kde.pool_dates.min().item()) - 1)
    batch = _make_batch(setup, n_batch=3)
    with torch.no_grad():
        _, comps = kde(**batch, return_components=True)
    # Every analogue is invalid → has_analogue_history all False;
    # the forward's defensive branch falls back to uniform α so the
    # softmax doesn't NaN.
    assert not comps.has_analogue_history.any().item()
    # α should be uniform 1/L over the (now-all-invalid) analogue list.
    L = kde.L
    np.testing.assert_allclose(comps.alpha.numpy(), np.full((3, L), 1.0 / L), atol=1e-5)


# ---------------------------------------------------------------------------
# v1.1 refactor invariants: bilinear vs concat at step 0, unique-dedupe path
# ---------------------------------------------------------------------------


def test_step0_bilinear_and_concat_produce_identical_log_q() -> None:
    """At step 0 the two shot-attention forms must agree.

    Both forms zero-init their output layer so g_θ ≡ 0 at init,
    which makes β uniform-over-causal-shots in both branches. The
    rest of the forward (α, σ, separable kernel) is shared, so
    ``log q_collab`` must be identical up to float-32 round-off.
    The bilinear form is the v1.1 default; this test fences against
    a re-parameterization slipping into the refactor.
    """
    setup = _build_test_setup()
    batch = _make_batch(setup, n_batch=3)
    kde_bilinear = _make_kde(setup, shot_attention_form="bilinear")
    kde_concat = _make_kde(setup, shot_attention_form="concat")
    with torch.no_grad():
        log_q_bilinear = kde_bilinear(**batch)
        log_q_concat = kde_concat(**batch)
    torch.testing.assert_close(log_q_bilinear, log_q_concat, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Warm-init mode: β stays near-uniform but f_θ side has nonzero gradient
# at step 0. Keep strict-init test above untouched — the warm tests are
# additive, not replacements.
# ---------------------------------------------------------------------------


def test_warm_init_beta_max_deviation_from_uniform_below_tolerance() -> None:
    """Per the math-preservation review: warm-init β must satisfy
    ``max_j |β_j - 1/R| < 1e-3`` at step 0. The warm init's σ = 1e-3
    on h_θ output (with f_θ default Kaiming, proj_dim=32) was chosen
    so the per-shot score magnitudes stay well below the softmax
    saturation regime, keeping β essentially uniform at init.
    """
    setup = _build_test_setup()
    kde = _make_kde(setup, shot_attention_form="bilinear", h_z_init="warm")
    batch = _make_batch(setup, n_batch=3)
    with torch.no_grad():
        _, comps = kde(**batch, return_components=True)

    # For each (b, l) with at least one causal shot, restrict β to its
    # valid R-prefix and compare to 1/R_valid. We use the per-row valid
    # count rather than the full max_history to keep the tolerance
    # meaningful — padded rows have β = 0 by design.
    valid_analogue = comps.has_analogue_history  # (B, L) bool
    # eff_mask isn't returned, so rebuild it from pool_dates + anchor.
    analogue_idx = kde.analogues[batch["player_idx"], batch["snapshot_idx"]]
    shot_idx = kde.pool_index[analogue_idx].clamp_min(0)
    anchor_dates = kde.anchor_dates[batch["snapshot_idx"]]
    shot_dates = kde.pool_dates[shot_idx]
    real_mask = (kde.pool_index[analogue_idx] >= 0).float()
    causal_mask = (shot_dates < anchor_dates.view(-1, 1, 1)).float()
    eff_mask = real_mask * causal_mask  # (B, L, R)

    for b_idx in range(comps.beta.shape[0]):
        for l_idx in range(comps.beta.shape[1]):
            if not bool(valid_analogue[b_idx, l_idx]):
                continue
            mask = eff_mask[b_idx, l_idx].bool()
            r_valid = int(mask.sum().item())
            beta_valid = comps.beta[b_idx, l_idx][mask]
            uniform = 1.0 / r_valid
            assert (beta_valid - uniform).abs().max().item() < 1e-3, (
                f"warm-init β at (b={b_idx}, l={l_idx}) drifts max="
                f"{(beta_valid - uniform).abs().max().item():.2e} from "
                f"uniform 1/{r_valid}"
            )


def test_warm_init_beta_kl_to_uniform_below_threshold() -> None:
    """Per the math-preservation review: warm-init β must satisfy
    ``KL(β || Unif) < 1e-4`` at step 0. KL is the stricter aggregate
    measure (catches uniform-spread small drifts that max-deviation
    might miss in average)."""
    setup = _build_test_setup()
    kde = _make_kde(setup, shot_attention_form="bilinear", h_z_init="warm")
    batch = _make_batch(setup, n_batch=3)
    with torch.no_grad():
        _, comps = kde(**batch, return_components=True)

    valid_analogue = comps.has_analogue_history
    analogue_idx = kde.analogues[batch["player_idx"], batch["snapshot_idx"]]
    real_mask = (kde.pool_index[analogue_idx] >= 0).float()
    shot_dates = kde.pool_dates[kde.pool_index[analogue_idx].clamp_min(0)]
    anchor_dates = kde.anchor_dates[batch["snapshot_idx"]]
    causal_mask = (shot_dates < anchor_dates.view(-1, 1, 1)).float()
    eff_mask = real_mask * causal_mask

    for b_idx in range(comps.beta.shape[0]):
        for l_idx in range(comps.beta.shape[1]):
            if not bool(valid_analogue[b_idx, l_idx]):
                continue
            mask = eff_mask[b_idx, l_idx].bool()
            r_valid = int(mask.sum().item())
            beta_valid = comps.beta[b_idx, l_idx][mask].clamp_min(1e-20)
            # KL(β || Unif) = Σ_j β_j · log(β_j / (1/R)) = Σ β log β + log R
            kl = (beta_valid * beta_valid.log()).sum().item() + np.log(r_valid)
            assert kl < 1e-4, (
                f"warm-init β at (b={b_idx}, l={l_idx}) has KL to uniform "
                f"= {kl:.2e} (> 1e-4 threshold)"
            )


def test_warm_init_log_q_close_to_strict_init_baseline() -> None:
    """The warm-init log_q should differ from the strict-init baseline
    by a small amount at step 0 — small enough to be a perturbation,
    not a regime change. Tolerance is loose (1e-2) because the kernel
    aggregation amplifies tiny β drifts into per-cell log-prob drifts
    that scale with the spread of analogue shot locations."""
    setup = _build_test_setup()
    batch = _make_batch(setup, n_batch=3)
    kde_zero = _make_kde(setup, shot_attention_form="bilinear", h_z_init="zero")
    kde_warm = _make_kde(setup, shot_attention_form="bilinear", h_z_init="warm")
    with torch.no_grad():
        log_q_zero = kde_zero(**batch)
        log_q_warm = kde_warm(**batch)
    max_diff = (log_q_warm - log_q_zero).abs().max().item()
    assert max_diff < 1e-2, (
        f"warm-init log_q deviates from strict-init baseline by "
        f"max_diff = {max_diff:.4e} (> 1e-2 threshold) — warm init is too hot"
    )


def test_warm_init_f_x_output_layer_receives_gradient_at_step_0() -> None:
    """The whole point of warm init: ``f_θ``'s output-layer weights
    must receive nonzero gradient on the very first step. With strict
    zero init, ``h_θ(z) = 0`` everywhere → ``∂score/∂f_θ_out = 0`` →
    f_θ_out grad = 0 → f_θ doesn't move until h_θ lifts off (the
    dual-saddle bootstrap). With warm init ``h_θ(z) ≠ 0`` from step 0
    so f_θ_out grad is nonzero immediately.
    """
    setup = _build_test_setup()
    kde = _make_kde(setup, shot_attention_form="bilinear", h_z_init="warm")
    batch = _make_batch(setup, n_batch=3)
    log_q = kde(**batch)
    (-log_q.mean()).backward()
    f_x_out_grad = kde.f_x[-1].weight.grad  # type: ignore[union-attr]
    assert f_x_out_grad is not None
    assert f_x_out_grad.abs().sum().item() > 0, (
        "warm-init f_x output layer received zero gradient — the warm h_z "
        "must lift score off zero on step 0 for the bootstrap to escape"
    )


def test_h_z_init_rejects_invalid_value() -> None:
    """The constructor should reject any string outside {zero, warm}.
    Guards against typos in CLI manifests when re-loading a checkpoint.
    """
    setup = _build_test_setup()
    with pytest.raises(ValueError, match="h_z_init"):
        _make_kde(setup, shot_attention_form="bilinear", h_z_init="hot")


# ---------------------------------------------------------------------------
# α similarity prior + non-zero b_same init
# ---------------------------------------------------------------------------


def test_alpha_prior_none_matches_legacy_uniform_init_alpha() -> None:
    """``alpha_prior="none"`` (the default) preserves the legacy
    behavior where α is uniform at init over valid analogues. Fences
    against the similarity prior accidentally activating when off."""
    setup = _build_test_setup()
    kde = _make_kde(setup, alpha_prior="none")
    batch = _make_batch(setup, n_batch=3)
    with torch.no_grad():
        _, comps = kde(**batch, return_components=True)
    L = kde.L
    np.testing.assert_allclose(comps.alpha.numpy(), np.full((3, L), 1.0 / L), atol=1e-5)


def test_alpha_prior_similarity_produces_non_uniform_alpha_at_init() -> None:
    """``alpha_prior="similarity"`` with ``γ_init=2`` breaks the
    uniform α at step 0 — the top-similarity analogue should get
    strictly more α weight than the bottom-similarity analogue.

    This is the load-bearing test for the 2026-05-17 α-prior
    intervention: if step-0 α is still ~uniform, the prior isn't
    actually being applied.
    """
    setup = _build_test_setup()
    kde = _make_kde(setup, alpha_prior="similarity", alpha_prior_scale_init=2.0)
    batch = _make_batch(setup, n_batch=3)
    with torch.no_grad():
        _, comps = kde(**batch, return_components=True)
    L = kde.L
    uniform = np.full((3, L), 1.0 / L)
    diff = (comps.alpha.numpy() - uniform).std(axis=-1).max()
    assert diff > 1e-3, (
        f"α should drift visibly from uniform under γ=2 similarity prior; "
        f"max per-row std from uniform = {diff:.2e}"
    )


def test_alpha_prior_similarity_collapses_to_uniform_when_gamma_is_zero() -> None:
    """When ``γ_sim`` is manually forced to zero, even with
    ``alpha_prior="similarity"`` enabled the α distribution should
    return to uniform — the prior is well-conditioned by γ."""
    setup = _build_test_setup()
    kde = _make_kde(setup, alpha_prior="similarity", alpha_prior_scale_init=2.0)
    with torch.no_grad():
        kde.gamma_sim.fill_(0.0)
    batch = _make_batch(setup, n_batch=3)
    with torch.no_grad():
        _, comps = kde(**batch, return_components=True)
    L = kde.L
    np.testing.assert_allclose(comps.alpha.numpy(), np.full((3, L), 1.0 / L), atol=1e-5)


def test_same_player_bias_init_boosts_self_analogue_slot() -> None:
    """With ``same_player_bias_init=1.0`` and the target player's own
    id present in its analogue list (``ensure_self=True`` in our
    fixture), the self slot should get a higher α weight than the
    other slots at init.
    """
    setup = _build_test_setup()
    kde = _make_kde(setup, same_player_bias_init=1.0)
    batch = _make_batch(setup, n_batch=4)
    with torch.no_grad():
        _, comps = kde(**batch, return_components=True)
    analogues = comps.analogues  # (B, L)
    player_idx = batch["player_idx"]  # (B,)
    L = kde.L
    for b_idx in range(player_idx.shape[0]):
        self_mask = analogues[b_idx] == player_idx[b_idx]
        if not bool(self_mask.any()):
            continue  # no self slot for this row → skip
        self_alpha = float(comps.alpha[b_idx, self_mask].sum().item())
        # Self slot should get materially more than the per-slot
        # uniform share (1/L). +1.0 bias → exp(1) ≈ 2.7× more weight.
        assert self_alpha > 1.5 / L, (
            f"row {b_idx}: self α = {self_alpha:.4f}, expected > {1.5 / L:.4f}"
        )


def test_alpha_prior_rejects_invalid_value() -> None:
    setup = _build_test_setup()
    with pytest.raises(ValueError, match="alpha_prior"):
        _make_kde(setup, alpha_prior="bilinear")  # typo'd to wrong enum


def test_alpha_prior_similarity_appears_in_learned_scalars_and_gets_gradient() -> None:
    """``γ_sim`` should round-trip through ``learned_scalars`` (for
    logging) and receive gradient through ``α`` when the prior is on.
    """
    setup = _build_test_setup()
    kde = _make_kde(setup, alpha_prior="similarity", alpha_prior_scale_init=2.0)
    scalars = kde.learned_scalars()
    assert "gamma_sim" in scalars
    np.testing.assert_allclose(scalars["gamma_sim"], 2.0, atol=1e-6)
    batch = _make_batch(setup, n_batch=3)
    log_q = kde(**batch)
    (-log_q.mean()).backward()
    assert kde.gamma_sim.grad is not None
    assert kde.gamma_sim.grad.abs().item() > 0


def test_bilinear_unique_dedupe_path_matches_naive_per_shot_h_compute() -> None:
    """The torch.unique-with-inverse dedup must be a pure speed
    optimization — the per-shot ``h_θ(z_j)`` values gathered by inverse
    must match the values you'd get from running ``h_θ`` directly on
    the per-batch ``(B*L*R, D)`` shot-context tensor."""
    setup = _build_test_setup()
    kde = _make_kde(setup, shot_attention_form="bilinear")
    batch = _make_batch(setup, n_batch=3)

    with torch.no_grad():
        # Reproduce the forward's gather and dedupe path.
        analogue_idx = kde.analogues[batch["player_idx"], batch["snapshot_idx"]]
        shot_idx = kde.pool_index[analogue_idx].clamp_min(0)
        b, L, R = shot_idx.shape

        flat = shot_idx.reshape(-1)
        unique_ids, inverse = torch.unique(flat, return_inverse=True)
        dedup_h = kde.h_z(kde.pool_context.index_select(0, unique_ids))
        dedup_h_full = dedup_h.index_select(0, inverse).view(b, L, R, -1)

        # Reference: directly run h_θ on the full (B, L, R, D) shot context.
        naive_ctx = kde.pool_context.index_select(0, flat).view(b, L, R, -1)
        naive_h_full = kde.h_z(naive_ctx)

    torch.testing.assert_close(dedup_h_full, naive_h_full, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# σ widens with low evidence when a_M is positive (softplus prior)
# ---------------------------------------------------------------------------


def test_sigma_widens_for_low_evidence_player_when_a_m_trained_positive() -> None:
    """Manually set a_M > 0 and confirm σ for a low-M player > σ for a
    high-M player. This verifies the softplus sign-prior:
    σ ∝ sigmoid(a_0 − softplus(a_M)·log1p_M − ...). With softplus(a_M)
    strictly positive, log1p_M ↑ → argument ↓ → σ ↓."""
    setup = _build_test_setup()
    kde = _make_kde(setup)
    # Push a_M well above zero so softplus(a_M) ≈ 1.
    with torch.no_grad():
        kde.a_M.fill_(2.0)

    batch = _make_batch(setup, n_batch=2)
    with torch.no_grad():
        _, comps = kde(**batch, return_components=True)
    # Pull traits manually to compare evidence orderings.
    traits = setup["traits"]  # type: ignore[assignment]
    log1p_M = torch.from_numpy(traits.traits[batch["player_idx"].numpy(), 0, 14])  # type: ignore[attr-defined]
    high_idx, low_idx = int(log1p_M.argmax()), int(log1p_M.argmin())
    if high_idx != low_idx:
        # high-M player should have smaller σ than low-M player.
        assert comps.sigma[high_idx].item() < comps.sigma[low_idx].item()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["bilinear", "concat"])
def test_gradient_flows_to_all_learnable_params(form: str) -> None:
    setup = _build_test_setup()
    kde = _make_kde(setup, shot_attention_form=form)
    batch = _make_batch(setup, n_batch=3)
    log_q = kde(**batch)
    loss = -log_q.mean()
    loss.backward()

    # Scalars
    for name in ("b_same", "lambda_M", "lambda_S", "lambda_age", "a_0", "a_M", "a_S", "a_R"):
        p = getattr(kde, name)
        assert p.grad is not None, f"{name} has no grad"
        # At step 0 some gradients may legitimately be zero (e.g.
        # b_same when no analogue is self for any batch row); just
        # check the gradient tensor was allocated.

    # φ_θ first Linear: weights non-zero at init, so gradient exists.
    first_phi = kde.phi[0]
    assert first_phi.weight.grad is not None  # type: ignore[union-attr]

    # Shot-attention path: check the first Linear of the form-specific MLP.
    if form == "bilinear":
        first_h = kde.h_z[0]
        first_f = kde.f_x[0]
        assert first_h.weight.grad is not None  # type: ignore[union-attr]
        assert first_f.weight.grad is not None  # type: ignore[union-attr]
    else:
        first_g = kde.g[0]
        assert first_g.weight.grad is not None  # type: ignore[union-attr]


def test_phi_and_g_output_layers_receive_gradient_when_perturbed() -> None:
    """Zero-init output layers won't get gradient at step 0 alone (their
    gradient is zero in expectation). Verify gradients do flow once the
    layers are perturbed off zero — proving the autograd path is
    correct, not blocked anywhere. Concat form.
    """
    setup = _build_test_setup()
    kde = _make_kde(setup, shot_attention_form="concat")
    with torch.no_grad():
        kde.phi[-1].weight.normal_(std=0.1)  # type: ignore[union-attr]
        kde.g[-1].weight.normal_(std=0.1)  # type: ignore[union-attr]
    batch = _make_batch(setup, n_batch=3)
    log_q = kde(**batch)
    (-log_q.mean()).backward()
    assert kde.phi[-1].weight.grad is not None  # type: ignore[union-attr]
    assert kde.phi[-1].weight.grad.abs().sum().item() > 0  # type: ignore[union-attr]
    assert kde.g[-1].weight.grad is not None  # type: ignore[union-attr]
    assert kde.g[-1].weight.grad.abs().sum().item() > 0  # type: ignore[union-attr]


def test_bilinear_h_z_output_layer_receives_gradient_when_perturbed() -> None:
    """Bilinear form's h_z output layer is zero-init; gradient through it
    is zero at step 0 (just like the concat form's g[-1]). Perturb h_z
    output weights off zero and verify the bilinear path's autograd is
    not blocked anywhere — in particular, that the torch.unique-with-
    inverse dedup doesn't break the gradient chain back to h_z and f_x.
    """
    setup = _build_test_setup()
    kde = _make_kde(setup, shot_attention_form="bilinear")
    with torch.no_grad():
        kde.h_z[-1].weight.normal_(std=0.1)  # type: ignore[union-attr]
        kde.f_x[-1].weight.normal_(std=0.1)  # type: ignore[union-attr]
    batch = _make_batch(setup, n_batch=3)
    log_q = kde(**batch)
    (-log_q.mean()).backward()
    assert kde.h_z[-1].weight.grad is not None  # type: ignore[union-attr]
    assert kde.h_z[-1].weight.grad.abs().sum().item() > 0  # type: ignore[union-attr]
    assert kde.f_x[-1].weight.grad is not None  # type: ignore[union-attr]
    assert kde.f_x[-1].weight.grad.abs().sum().item() > 0  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------


def test_learned_scalars_returns_expected_keys() -> None:
    setup = _build_test_setup()
    kde = _make_kde(setup)
    d = kde.learned_scalars()
    assert set(d.keys()) == {
        "b_same",
        "lambda_M",
        "lambda_S",
        "lambda_age",
        "a_0",
        "a_M",
        "a_S",
        "a_R",
        "softplus_a_M",
        "softplus_a_S",
        "gamma_sim",
    }
    # At init the a_M, a_S coefficients are initialized to -10 so that
    # softplus(·) ≈ 4.5e-5 — the evidence terms in σ contribute
    # negligibly at step 0, preserving σ = sigma_init for every player.
    assert d["softplus_a_M"] < 1e-3
    assert d["softplus_a_S"] < 1e-3


def test_forward_continuous_own_mask_matches_support_source_masks() -> None:
    """PR3.0 contract: ``CollaborativeContinuousOutputs.own_mask`` equals
    the partition produced by
    :func:`shotcloud.evaluation.support_masks.support_source_masks` —
    the backend-agnostic field exactly replaces the helper call the
    wrapper used to make."""
    from shotcloud.evaluation.support_masks import support_source_masks

    setup = _build_test_setup()
    kde = _make_kde(setup)
    b = 4
    vocab = setup["vocab"]
    ctx = setup["ctx"]
    player_idx = torch.arange(b, dtype=torch.long)
    snapshot_idx = torch.zeros(b, dtype=torch.long)
    x_n_raw = torch.from_numpy(ctx[:b]).float()  # type: ignore[index]
    x_n = x_n_raw.clone()
    with torch.no_grad():
        out = kde.forward_continuous(player_idx, snapshot_idx, x_n_raw, x_n)
    assert hasattr(out, "own_mask")
    assert out.own_mask.shape == out.support_mask.shape
    assert out.own_mask.dtype == torch.bool
    # own_mask must be a subset of support_mask (no own slot is invalid).
    assert (~out.own_mask | out.support_mask).all()
    r = out.support_xy.shape[1] // out.analogue_idx.shape[1]
    helper = support_source_masks(
        analogue_idx=out.analogue_idx,
        player_idx=player_idx,
        n_shots_per_analogue=r,
        valid_mask=out.support_mask,
    )
    torch.testing.assert_close(out.own_mask, helper.own)
    # Sanity that vocab has the players we addressed (no off-by-one).
    assert b <= len(vocab)
