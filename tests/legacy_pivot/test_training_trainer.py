"""Tests for :func:`shotcloud.training.train_decoder`.

The headline test (``test_trainer_recovers_known_tilt_on_synthetic_data``)
is the load-bearing one: build a synthetic dataset whose true generating
tilt is a known low-rank deformation of the prior, then verify that
training recovers a matching tilt as evidenced by improved NLL and the
decoder's ``V`` becoming non-zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud import AdaptiveKDE, CourtGrid, HierarchicalKDE, HierarchicalKDEBase, RelevanceScore
from shotcloud.legacy import (
    KDEProduct,
    LearnableKDEProductWeights,
    LearnableTemperature,
    PlayerEmbeddingEncoder,
)
from shotcloud.legacy_pivot.adaptive_prior import AdaptiveOffensivePrior
from shotcloud.legacy_pivot.defensive_kde import DefensiveKDE
from shotcloud.legacy_pivot.defensive_scale import LearnableDefensiveScale
from shotcloud.legacy_pivot.shot_cell_dataset import ShotCellDataset
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder
from shotcloud.legacy_pivot.trainer import TrainHistory, train_decoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_fitted_kde(seed: int = 0) -> tuple[HierarchicalKDE, KDEProduct, CourtGrid]:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(seed)
    rows = []
    for pid in ("A", "B", "C", "D"):
        for _ in range(80):
            rows.append(
                {"x": float(rng.normal(0, 8)), "y": float(rng.normal(15, 8)), "player_id": pid}
            )
    df = pd.DataFrame(rows)
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=200.0, recency_half_life_days=None)
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        position=np.array(["G"] * len(df)),
    )
    base = KDEProduct(hierarchical_kde=kde, weights={"a_p": 1.0, "a_g": 0.0, "a_0": 0.0})
    return kde, base, grid


def _sample_shots_from_decoder(
    encoder: PlayerEmbeddingEncoder,
    decoder: LowRankTiltDecoder,
    base: KDEProduct,
    player_ids: list[str],
    *,
    shots_per_player: int,
    seed: int,
) -> pd.DataFrame:
    """Sample shots from a *known* decoder, returning a DataFrame for training."""
    rng = np.random.default_rng(seed)
    grid = base.grid
    rows = []
    for pid in player_ids:
        log_q0 = base.log_density(pid).ravel()
        log_q0_t = torch.from_numpy(log_q0.astype(np.float32)).unsqueeze(0)
        # Look up the player's u via the encoder vocab — simplified: pid → index 0..N-1.
        idx = sorted({pid for pid in player_ids}).index(pid)
        u = encoder(torch.tensor([idx], dtype=torch.long))
        with torch.no_grad():
            probs = decoder.probs(log_q0_t, u).numpy().ravel()
        cells = rng.choice(grid.n_cells, size=shots_per_player, p=probs)
        x, y = grid.dequantize(cells.astype(np.int64), rng)
        rows.extend(
            {"x": float(xi), "y": float(yi), "player_id": pid} for xi, yi in zip(x, y, strict=True)
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sanity tests
# ---------------------------------------------------------------------------


def test_history_dataclass_defaults() -> None:
    h = TrainHistory()
    assert h.train_nll == []
    assert h.val_nll == []
    assert np.isinf(h.best_val_nll)


def test_train_runs_on_zero_init_decoder() -> None:
    """Smoke test: training runs end-to-end and produces a TrainHistory."""
    _, base, grid = _build_fitted_kde()
    df = pd.DataFrame(
        {
            "x": np.random.default_rng(0).normal(0, 5, 200),
            "y": np.random.default_rng(0).normal(15, 5, 200),
            "player_id": np.tile(["A", "B", "C", "D"], 50),
        }
    )
    ds = ShotCellDataset(df, base, grid)
    encoder = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    hist = train_decoder(encoder, decoder, ds, n_epochs=2, batch_size=32)
    assert isinstance(hist, TrainHistory)
    assert len(hist.train_nll) == 2


def test_train_rank_mismatch_raises() -> None:
    _, base, grid = _build_fitted_kde()
    df = pd.DataFrame({"x": [0.0, 0.0], "y": [5.0, 5.0], "player_id": ["A", "B"]})
    ds = ShotCellDataset(df, base, grid)
    encoder = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=8)  # mismatch
    with pytest.raises(ValueError, match="rank"):
        train_decoder(encoder, decoder, ds, n_epochs=1, batch_size=2)


def test_val_nll_logged_when_val_set_provided() -> None:
    _, base, grid = _build_fitted_kde()
    rng = np.random.default_rng(0)
    df_train = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 200),
            "y": rng.normal(15, 5, 200),
            "player_id": np.tile(["A", "B", "C", "D"], 50),
        }
    )
    df_val = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 80),
            "y": rng.normal(15, 5, 80),
            "player_id": np.tile(["A", "B", "C", "D"], 20),
        }
    )
    ds = ShotCellDataset(df_train, base, grid)
    val = ShotCellDataset(df_val, base, grid, vocab=ds.vocab)
    encoder = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    hist = train_decoder(encoder, decoder, ds, val_set=val, n_epochs=2, batch_size=32)
    assert len(hist.val_nll) == 2


def test_best_epoch_recorded_when_val_set_provided() -> None:
    """``history.best_epoch`` should equal argmin(val_nll) + 1."""
    _, base, grid = _build_fitted_kde()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 200),
            "y": rng.normal(15, 5, 200),
            "player_id": np.tile(["A", "B", "C", "D"], 50),
        }
    )
    ds = ShotCellDataset(df, base, grid)
    val = ShotCellDataset(df, base, grid, vocab=ds.vocab)
    encoder = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    hist = train_decoder(encoder, decoder, ds, val_set=val, n_epochs=3, batch_size=32)

    assert hist.best_epoch is not None
    assert 1 <= hist.best_epoch <= 3
    expected = int(np.argmin(hist.val_nll)) + 1
    assert hist.best_epoch == expected


def test_best_epoch_is_none_when_no_val_set() -> None:
    _, base, grid = _build_fitted_kde()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 100),
            "y": rng.normal(15, 5, 100),
            "player_id": np.tile(["A", "B", "C", "D"], 25),
        }
    )
    ds = ShotCellDataset(df, base, grid)
    encoder = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    hist = train_decoder(encoder, decoder, ds, n_epochs=2, batch_size=32)
    assert hist.best_epoch is None


def test_restore_best_val_returns_best_epoch_weights() -> None:
    """After training, encoder/decoder must equal their best-val-epoch state."""
    torch.manual_seed(0)
    _, base, grid = _build_fitted_kde(seed=1)
    rng = np.random.default_rng(0)
    # Tiny train set, larger val set → more likely to overfit, revealing
    # the difference between best and final epoch.
    df_train = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 60),
            "y": rng.normal(15, 5, 60),
            "player_id": np.tile(["A", "B", "C", "D"], 15),
        }
    )
    df_val = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 200),
            "y": rng.normal(15, 5, 200),
            "player_id": np.tile(["A", "B", "C", "D"], 50),
        }
    )
    ds = ShotCellDataset(df_train, base, grid)
    val = ShotCellDataset(df_val, base, grid, vocab=ds.vocab)
    encoder = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)

    hist = train_decoder(
        encoder,
        decoder,
        ds,
        val_set=val,
        n_epochs=20,
        batch_size=16,
        learning_rate=5e-2,  # high lr → overshoot best-val epoch
        restore_best_val=True,
    )

    # After training, val NLL should equal the recorded best (because we
    # restored the best-val parameters).
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        u = encoder(val.player_idx)
        log_probs = decoder.log_probs(val.log_q0_table[val.player_idx], u)
        per_shot = -log_probs[torch.arange(len(val)), val.cell_idx]
    actual_val = float(per_shot.mean())

    assert abs(actual_val - hist.best_val_nll) < 1e-4, (
        f"after restore, val_nll={actual_val:.4f} should equal best_val_nll={hist.best_val_nll:.4f}"
    )


def test_restore_best_val_can_be_disabled() -> None:
    """With restore_best_val=False, the best epoch is still recorded but
    parameters are not rolled back."""
    torch.manual_seed(0)
    _, base, grid = _build_fitted_kde(seed=1)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 200),
            "y": rng.normal(15, 5, 200),
            "player_id": np.tile(["A", "B", "C", "D"], 50),
        }
    )
    ds = ShotCellDataset(df, base, grid)
    val = ShotCellDataset(df, base, grid, vocab=ds.vocab)
    encoder = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    hist = train_decoder(
        encoder,
        decoder,
        ds,
        val_set=val,
        n_epochs=3,
        batch_size=32,
        restore_best_val=False,
    )
    assert hist.best_epoch is not None


# ---------------------------------------------------------------------------
# THE recovery test — load-bearing
# ---------------------------------------------------------------------------


def test_trainer_recovers_known_tilt_on_synthetic_data() -> None:
    """Build a known random-init decoder, sample shots from it, train a fresh
    zero-init decoder on those shots, and verify NLL drops below the
    base-measure NLL — i.e., training has learned non-trivial tilt.
    """
    torch.manual_seed(0)
    _, base, grid = _build_fitted_kde(seed=42)
    rank = 4
    n_cells = grid.n_cells
    player_ids = ["A", "B", "C", "D"]

    # Ground-truth decoder + encoder — strong, varied tilts so the data
    # has signal for the trainer to recover.
    truth_decoder = LowRankTiltDecoder(n_cells=n_cells, rank=rank, zero_init=False)
    truth_encoder = PlayerEmbeddingEncoder(n_players=len(player_ids), rank=rank, zero_init=False)
    with torch.no_grad():
        # Scale up so the tilt is non-trivial.
        truth_decoder.V.mul_(1.5)
        truth_encoder.embedding.weight.mul_(1.5)

    # Sample lots of shots from the ground-truth model.
    shots_per_player = 1000
    df = _sample_shots_from_decoder(
        truth_encoder,
        truth_decoder,
        base,
        player_ids,
        shots_per_player=shots_per_player,
        seed=0,
    )
    ds = ShotCellDataset(df, base, grid)

    # Fresh student: zero-init decoder (preserves the q_0 invariant at step 0),
    # random-init encoder (escapes the dead-zero saddle so V can update).
    student_decoder = LowRankTiltDecoder(n_cells=n_cells, rank=rank, zero_init=True)
    student_encoder = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=rank, zero_init=False)

    # Pre-training NLL: V = 0 → output equals base measure for any u.
    pre_train_nll = _eval_nll(student_encoder, student_decoder, ds)

    # Train.
    hist = train_decoder(
        student_encoder,
        student_decoder,
        ds,
        n_epochs=10,
        batch_size=512,
        learning_rate=1e-2,
    )

    # Post-training NLL.
    post_train_nll = hist.train_nll[-1]

    assert post_train_nll < pre_train_nll, (
        f"NLL did not improve: pre={pre_train_nll:.4f}, post={post_train_nll:.4f}"
    )
    # Decoder V should have moved off zero.
    assert (student_decoder.V.detach().abs() > 1e-3).any()
    # Encoder embeddings should have moved off zero.
    assert (student_encoder.embedding.weight.detach().abs() > 1e-3).any()


def _eval_nll(
    encoder: PlayerEmbeddingEncoder,
    decoder: LowRankTiltDecoder,
    ds: ShotCellDataset,
) -> float:
    """Mean per-shot NLL under the current encoder/decoder."""
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        u = encoder(ds.player_idx)
        log_probs = decoder.log_probs(ds.log_q0_table[ds.player_idx], u)
        per_shot = -log_probs[torch.arange(len(ds)), ds.cell_idx]
    return float(per_shot.mean())


# ---------------------------------------------------------------------------
# Learnable KDE-product weights
# ---------------------------------------------------------------------------


def test_learnable_weights_require_components() -> None:
    """Trainer must reject a learnable_weights run when train_set lacks components."""
    _, base, grid = _build_fitted_kde()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 60),
            "y": rng.normal(15, 5, 60),
            "player_id": np.tile(["A", "B", "C", "D"], 15),
        }
    )
    ds = ShotCellDataset(df, base, grid)  # cache_components=False (default)
    enc = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    weights = LearnableKDEProductWeights()
    with pytest.raises(ValueError, match="cache_components"):
        train_decoder(enc, dec, ds, learnable_weights=weights, n_epochs=1, batch_size=16)


def test_learnable_weights_run_smoke() -> None:
    """End-to-end: trainer runs with learnable weights and the thetas update."""
    torch.manual_seed(0)
    _, base, grid = _build_fitted_kde()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 600),
            "y": rng.normal(15, 5, 600),
            "player_id": np.tile(["A", "B", "C", "D"], 150),
        }
    )
    ds = ShotCellDataset(df, base, grid, cache_components=True)
    enc = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    weights = LearnableKDEProductWeights()

    initial = {k: v for k, v in weights.weights_as_floats().items()}
    hist = train_decoder(
        enc,
        dec,
        ds,
        learnable_weights=weights,
        n_epochs=3,
        batch_size=128,
        learning_rate=1e-2,
    )
    assert isinstance(hist, TrainHistory)
    assert len(hist.train_nll) == 3

    final = weights.weights_as_floats()
    # At least one weight should have moved off its init.
    deltas = [abs(final[k] - initial[k]) for k in ("a_p", "a_g", "a_0")]
    assert max(deltas) > 1e-4, f"weights did not update; deltas={deltas}"


# ---------------------------------------------------------------------------
# Learnable temperature (Phase 1)
# ---------------------------------------------------------------------------


def _build_hier_base(seed: int = 0) -> tuple[HierarchicalKDEBase, CourtGrid]:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for pid in ("A", "B", "C", "D"):
        for _ in range(80):
            rows.append(
                {"x": float(rng.normal(0, 8)), "y": float(rng.normal(15, 8)), "player_id": pid}
            )
    df = pd.DataFrame(rows)
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=200.0, recency_half_life_days=None)
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        position=np.array(["G"] * len(df)),
    )
    return HierarchicalKDEBase(hierarchical_kde=kde), grid


def test_temperature_with_hierarchical_base_smoke() -> None:
    """Phase-1 happy path: hierarchical base + temperature trains end-to-end."""
    torch.manual_seed(0)
    base, grid = _build_hier_base()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 600),
            "y": rng.normal(15, 5, 600),
            "player_id": np.tile(["A", "B", "C", "D"], 150),
        }
    )
    ds = ShotCellDataset(df, base, grid)  # no component caching needed
    enc = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    temperature = LearnableTemperature(init=1.0)

    initial_tau = temperature.tau_as_float()
    hist = train_decoder(
        enc,
        dec,
        ds,
        learnable_temperature=temperature,
        n_epochs=3,
        batch_size=128,
        learning_rate=5e-2,
    )
    assert isinstance(hist, TrainHistory)
    assert len(hist.train_nll) == 3
    final_tau = temperature.tau_as_float()
    assert abs(final_tau - initial_tau) > 1e-4, (
        f"tau did not update; init={initial_tau}, final={final_tau}"
    )


def test_temperature_with_kde_product_base_smoke() -> None:
    """Temperature pathway also works with KDEProduct base — no component cache needed."""
    torch.manual_seed(0)
    _, base, grid = _build_fitted_kde()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 600),
            "y": rng.normal(15, 5, 600),
            "player_id": np.tile(["A", "B", "C", "D"], 150),
        }
    )
    ds = ShotCellDataset(df, base, grid)  # cache_components=False
    enc = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    temperature = LearnableTemperature(init=1.0)
    hist = train_decoder(
        enc, dec, ds, learnable_temperature=temperature, n_epochs=2, batch_size=128
    )
    assert len(hist.train_nll) == 2


def test_temperature_and_weights_mutually_exclusive() -> None:
    """Trainer must reject simultaneous learnable_weights and learnable_temperature."""
    _, base, grid = _build_fitted_kde()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 60),
            "y": rng.normal(15, 5, 60),
            "player_id": np.tile(["A", "B", "C", "D"], 15),
        }
    )
    ds = ShotCellDataset(df, base, grid, cache_components=True)
    enc = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    weights = LearnableKDEProductWeights()
    temperature = LearnableTemperature()
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_decoder(
            enc,
            dec,
            ds,
            learnable_weights=weights,
            learnable_temperature=temperature,
            n_epochs=1,
            batch_size=16,
        )


def test_temperature_best_val_rollback() -> None:
    """When restore_best_val=True, the temperature checkpoint also rolls back."""
    torch.manual_seed(0)
    base, grid = _build_hier_base(seed=1)
    rng = np.random.default_rng(0)
    df_train = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 60),
            "y": rng.normal(15, 5, 60),
            "player_id": np.tile(["A", "B", "C", "D"], 15),
        }
    )
    df_val = pd.DataFrame(
        {
            "x": rng.normal(0, 5, 200),
            "y": rng.normal(15, 5, 200),
            "player_id": np.tile(["A", "B", "C", "D"], 50),
        }
    )
    ds = ShotCellDataset(df_train, base, grid)
    val = ShotCellDataset(df_val, base, grid, vocab=ds.vocab)
    enc = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    temperature = LearnableTemperature(init=1.0)

    # Run long enough that the high LR will cause overshoot past best-val.
    hist = train_decoder(
        enc,
        dec,
        ds,
        val_set=val,
        learnable_temperature=temperature,
        n_epochs=12,
        batch_size=16,
        learning_rate=5e-2,
        restore_best_val=True,
    )

    # Recompute val NLL with the restored temperature; must equal best_val_nll.
    enc.eval()
    dec.eval()
    temperature.eval()
    with torch.no_grad():
        u = enc(val.player_idx)
        log_q0 = temperature(val.log_q0_table[val.player_idx])
        log_probs = dec.log_probs(log_q0, u)
        per_shot = -log_probs[torch.arange(len(val)), val.cell_idx]
    actual_val = float(per_shot.mean())
    assert abs(actual_val - hist.best_val_nll) < 1e-4, (
        f"after restore, val_nll={actual_val:.4f} should equal best_val_nll={hist.best_val_nll:.4f}"
    )


# ---------------------------------------------------------------------------
# Phase 2 — defensive product factor
# ---------------------------------------------------------------------------


def _build_defensive_setup(
    seed: int = 0,
) -> tuple[HierarchicalKDEBase, DefensiveKDE, CourtGrid, pd.DataFrame]:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for pid in ("A", "B", "C", "D"):
        for i in range(120):
            rows.append(
                {
                    "x": float(rng.normal(0, 8)),
                    "y": float(rng.normal(15, 8)),
                    "player_id": pid,
                    "opponent": "X" if i % 2 == 0 else "Y",
                }
            )
    df = pd.DataFrame(rows)
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=200.0, recency_half_life_days=None)
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        position=np.array(["G"] * len(df)),
    )
    base = HierarchicalKDEBase(hierarchical_kde=kde)
    def_kde = DefensiveKDE(grid=grid, bandwidth=1.5, recency_half_life_days=None)
    def_kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        opponent=df["opponent"].to_numpy(),
    )
    return base, def_kde, grid, df


def test_defensive_smoke_run() -> None:
    """End-to-end: trainer runs with defensive KDE + α_def and the scalar updates."""
    torch.manual_seed(0)
    base, def_kde, grid, df = _build_defensive_setup()
    ds = ShotCellDataset(df, base, grid, defensive_kde=def_kde)
    enc = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    alpha = LearnableDefensiveScale(init=0.5)
    initial = alpha.alpha_as_float()
    hist = train_decoder(
        enc,
        dec,
        ds,
        learnable_defensive_scale=alpha,
        n_epochs=3,
        batch_size=128,
        learning_rate=5e-2,
    )
    assert isinstance(hist, TrainHistory)
    assert len(hist.train_nll) == 3
    final = alpha.alpha_as_float()
    assert abs(final - initial) > 1e-4, f"alpha_def did not update; init={initial}, final={final}"


def test_defensive_requires_dataset_to_have_defense() -> None:
    """Trainer must reject defensive_scale when train_set has no defensive cache."""
    torch.manual_seed(0)
    base, _def_kde, grid, df = _build_defensive_setup()
    ds = ShotCellDataset(df, base, grid)  # no defensive_kde
    enc = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    alpha = LearnableDefensiveScale()
    with pytest.raises(ValueError, match="defensive_kde"):
        train_decoder(enc, dec, ds, learnable_defensive_scale=alpha, n_epochs=1, batch_size=16)


def test_defensive_composes_with_temperature() -> None:
    """Phase 1 (temperature) + Phase 2 (defensive scale) must train together."""
    torch.manual_seed(0)
    base, def_kde, grid, df = _build_defensive_setup()
    ds = ShotCellDataset(df, base, grid, defensive_kde=def_kde)
    enc = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    tau = LearnableTemperature(init=1.0)
    alpha = LearnableDefensiveScale(init=0.3)
    hist = train_decoder(
        enc,
        dec,
        ds,
        learnable_temperature=tau,
        learnable_defensive_scale=alpha,
        n_epochs=2,
        batch_size=128,
        learning_rate=5e-2,
    )
    assert len(hist.train_nll) == 2


def _build_defensive_setup_with_context(
    seed: int = 0,
) -> tuple[
    HierarchicalKDEBase,
    DefensiveKDE,
    CourtGrid,
    pd.DataFrame,
    ContextEncoder,  # noqa: F821
]:
    """Defensive setup augmented with the columns ContextEncoder needs."""
    from shotcloud.data import ContextEncoder

    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for pid in ("A", "B", "C", "D"):
        for i in range(120):
            rows.append(
                {
                    "x": float(rng.normal(0, 8)),
                    "y": float(rng.normal(15, 8)),
                    "player_id": pid,
                    "opponent": "X" if i % 2 == 0 else "Y",
                    "made": int(rng.random() < 0.5),
                    "period": (i % 4) + 1,
                    "time_remaining_sec": int(60 * (i % 48)),
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(i)),
                }
            )
    df = pd.DataFrame(rows)
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=200.0, recency_half_life_days=None)
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        position=np.array(["G"] * len(df)),
    )
    base = HierarchicalKDEBase(hierarchical_kde=kde)
    def_kde = DefensiveKDE(grid=grid, bandwidth=1.5, recency_half_life_days=None)
    def_kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        opponent=df["opponent"].to_numpy(),
    )
    enc = ContextEncoder.fit(df)
    return base, def_kde, grid, df, enc


def test_context_temperature_smoke() -> None:
    """Phase-3 happy path: context-conditioned τ trains end-to-end."""
    from shotcloud.data import CONTEXT_DIM

    torch.manual_seed(0)
    base, _, grid, df, enc = _build_defensive_setup_with_context()
    ds = ShotCellDataset(df, base, grid, context_encoder=enc)
    encm = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    tau = LearnableTemperature(init=1.0, context_dim=CONTEXT_DIM, mlp_hidden=8)
    initial_mean = tau.tau_as_float(ds.context_features[:64])
    hist = train_decoder(
        encm,
        dec,
        ds,
        learnable_temperature=tau,
        n_epochs=3,
        batch_size=128,
        learning_rate=5e-2,
    )
    assert len(hist.train_nll) == 3
    final_mean = tau.tau_as_float(ds.context_features[:64])
    assert abs(final_mean - initial_mean) > 1e-4, (
        f"context τ did not update; init={initial_mean:.4f}, final={final_mean:.4f}"
    )


def test_context_temperature_requires_dataset_context() -> None:
    """Trainer rejects context-conditioned τ when train_set has no x_n."""
    from shotcloud.data import CONTEXT_DIM

    base, _def_kde, grid, df, _enc = _build_defensive_setup_with_context()
    ds = ShotCellDataset(df, base, grid)  # no context_encoder
    encm = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    tau = LearnableTemperature(init=1.0, context_dim=CONTEXT_DIM)
    with pytest.raises(ValueError, match="context"):
        train_decoder(encm, dec, ds, learnable_temperature=tau, n_epochs=1, batch_size=16)


def test_context_dim_mismatch_raises() -> None:
    """train_set context_dim must match the scalar module's context_dim."""
    base, _def_kde, grid, df, enc = _build_defensive_setup_with_context()
    ds = ShotCellDataset(df, base, grid, context_encoder=enc)
    encm = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    tau = LearnableTemperature(init=1.0, context_dim=ds.context_dim + 1)
    with pytest.raises(ValueError, match="context_dim"):
        train_decoder(encm, dec, ds, learnable_temperature=tau, n_epochs=1, batch_size=16)


def test_context_temperature_and_defensive_scale_compose() -> None:
    """Context τ(x) and context α_def(x) compose end-to-end."""
    from shotcloud.data import CONTEXT_DIM

    torch.manual_seed(0)
    base, def_kde, grid, df, enc = _build_defensive_setup_with_context()
    ds = ShotCellDataset(df, base, grid, defensive_kde=def_kde, context_encoder=enc)
    encm = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    tau = LearnableTemperature(init=1.0, context_dim=CONTEXT_DIM)
    alpha = LearnableDefensiveScale(init=0.5, context_dim=CONTEXT_DIM)
    hist = train_decoder(
        encm,
        dec,
        ds,
        learnable_temperature=tau,
        learnable_defensive_scale=alpha,
        n_epochs=2,
        batch_size=128,
        learning_rate=5e-2,
    )
    assert len(hist.train_nll) == 2


def test_defensive_best_val_rollback() -> None:
    """When restore_best_val=True, the defensive scale must also roll back."""
    torch.manual_seed(0)
    base, def_kde, grid, df = _build_defensive_setup()
    ds = ShotCellDataset(df, base, grid, defensive_kde=def_kde)
    df_val = df.iloc[: len(df) // 4].reset_index(drop=True)
    val = ShotCellDataset(
        df_val, base, grid, vocab=ds.vocab, defensive_kde=def_kde, opponent_vocab=ds.opponent_vocab
    )
    enc = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    alpha = LearnableDefensiveScale(init=0.5)
    hist = train_decoder(
        enc,
        dec,
        ds,
        val_set=val,
        learnable_defensive_scale=alpha,
        n_epochs=8,
        batch_size=64,
        learning_rate=1e-1,  # high lr → overshoot best val
        restore_best_val=True,
    )
    # After rollback, recompute val NLL with the restored α and check it equals best.
    enc.eval()
    dec.eval()
    alpha.eval()
    with torch.no_grad():
        u = enc(val.player_idx)
        log_q0 = val.log_q0_table[val.player_idx]
        assert val.log_qd_table is not None
        log_q_def = val.log_qd_table[val.opponent_idx]
        log_q0 = log_q0 + alpha(log_q_def)
        log_probs = dec.log_probs(log_q0, u)
        per_shot = -log_probs[torch.arange(len(val)), val.cell_idx]
    actual_val = float(per_shot.mean())
    assert abs(actual_val - hist.best_val_nll) < 1e-4, (
        f"after restore, val_nll={actual_val:.4f} should equal best_val_nll={hist.best_val_nll:.4f}"
    )


# ---------------------------------------------------------------------------
# Phase 4 — adaptive prior
# ---------------------------------------------------------------------------


def _build_adaptive_setup(seed: int = 0):  # type: ignore[no-untyped-def]
    from shotcloud.data import ContextEncoder

    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for pid in ("A", "B", "C", "D"):
        for i in range(120):
            rows.append(
                {
                    "x": float(rng.normal(0, 8)),
                    "y": float(rng.normal(15, 8)),
                    "player_id": pid,
                    "opponent": "X" if i % 2 == 0 else "Y",
                    "made": int(rng.random() < 0.5),
                    "period": (i % 4) + 1,
                    "time_remaining_sec": int(60 * (i % 48)),
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(i)),
                }
            )
    df = pd.DataFrame(rows)
    enc = ContextEncoder.fit(df)
    ctx_arr = enc.transform(df)

    hkde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=200.0, recency_half_life_days=None)
    hkde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        position=np.array(["G"] * len(df)),
    )
    base = HierarchicalKDEBase(hierarchical_kde=hkde)
    akde = AdaptiveKDE(grid=grid, bandwidth=1.5, max_history=40)
    akde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx_arr,
    )
    return base, akde, hkde, grid, df, enc


@pytest.mark.skip(
    reason="legacy API; AdaptiveOffensivePrior was refactored to consume "
    "SnapshotBundle. The trainer rewrite for AA-KDE (todo: joint "
    "training run) will replace these tests."
)
def test_adaptive_prior_smoke() -> None:
    torch.manual_seed(0)
    base, akde, hkde, grid, df, enc = _build_adaptive_setup()
    ds = ShotCellDataset(df, base, grid, context_encoder=enc)
    relevance = RelevanceScore(init_beta_q=0.3)
    prior = AdaptiveOffensivePrior(akde, hkde, ds.vocab, relevance, kappa=20.0)
    encm = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)

    initial_beta_q = float(relevance.beta_q)
    hist = train_decoder(
        encm,
        dec,
        ds,
        adaptive_prior=prior,
        lambda_entropy=0.005,
        n_epochs=2,
        batch_size=128,
        learning_rate=5e-2,
    )
    assert isinstance(hist, TrainHistory)
    assert len(hist.train_nll) == 2
    final_beta_q = float(relevance.beta_q)
    assert abs(final_beta_q - initial_beta_q) > 1e-4


@pytest.mark.skip(reason="legacy API; see test_adaptive_prior_smoke skip note")
def test_adaptive_prior_requires_context() -> None:
    base, akde, hkde, grid, df, _enc = _build_adaptive_setup()
    ds = ShotCellDataset(df, base, grid)
    relevance = RelevanceScore()
    prior = AdaptiveOffensivePrior(akde, hkde, ds.vocab, relevance)
    encm = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    with pytest.raises(ValueError, match="context"):
        train_decoder(encm, dec, ds, adaptive_prior=prior, n_epochs=1, batch_size=16)


@pytest.mark.skip(reason="legacy API; see test_adaptive_prior_smoke skip note")
def test_adaptive_and_learnable_weights_mutually_exclusive() -> None:
    _base, akde, hkde, grid, df, enc = _build_adaptive_setup()
    kdep_base = KDEProduct(hierarchical_kde=hkde, weights={"a_p": 1.0, "a_g": 0.0, "a_0": 0.0})
    ds = ShotCellDataset(df, kdep_base, grid, cache_components=True, context_encoder=enc)
    relevance = RelevanceScore()
    prior = AdaptiveOffensivePrior(akde, hkde, ds.vocab, relevance)
    weights = LearnableKDEProductWeights()
    encm = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_decoder(
            encm,
            dec,
            ds,
            learnable_weights=weights,
            adaptive_prior=prior,
            n_epochs=1,
            batch_size=16,
        )


@pytest.mark.skip(reason="legacy API; see test_adaptive_prior_smoke skip note")
def test_adaptive_with_temperature_and_defense_compose() -> None:
    torch.manual_seed(0)
    base, akde, hkde, grid, df, enc = _build_adaptive_setup()
    def_kde = DefensiveKDE(grid=grid, bandwidth=1.5, recency_half_life_days=None)
    def_kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        opponent=df["opponent"].to_numpy(),
    )
    ds = ShotCellDataset(df, base, grid, defensive_kde=def_kde, context_encoder=enc)
    relevance = RelevanceScore()
    prior = AdaptiveOffensivePrior(akde, hkde, ds.vocab, relevance, kappa=20.0)
    encm = PlayerEmbeddingEncoder(n_players=len(ds.vocab), rank=4)
    dec = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4)
    tau = LearnableTemperature(init=1.0)
    alpha = LearnableDefensiveScale(init=0.5)
    hist = train_decoder(
        encm,
        dec,
        ds,
        adaptive_prior=prior,
        learnable_temperature=tau,
        learnable_defensive_scale=alpha,
        lambda_entropy=0.005,
        n_epochs=2,
        batch_size=128,
        learning_rate=5e-2,
    )
    assert len(hist.train_nll) == 2
