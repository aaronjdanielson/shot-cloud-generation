"""Tests for :func:`shotcloud.legacy_pivot.eval_ablation.run_ablation`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shotcloud import load_shots
from shotcloud.legacy_pivot.eval_ablation import AblationResult, model_order, run_ablation

# Optional smoke test against the real shot_flow CSV.
SHOT_FLOW_CSV = Path("/Users/aarondanielson/Dropbox/shot_flow/data/shot_data.csv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_shots_df(
    n_per_player_train: int = 200,
    n_per_player_test: int = 50,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train + test DataFrames for 4 players with distinct shot patterns."""
    rng = np.random.default_rng(seed)
    train_rows = []
    test_rows = []
    for pos, mu_x, mu_y in [("G", 0.0, 24.0), ("F", 0.0, 4.0)]:
        for player_idx in range(2):
            pid = f"{pos}{player_idx}"
            mu = (mu_x + 2.0 * (player_idx - 0.5), mu_y)
            x_train = rng.normal(mu[0], 2.0, n_per_player_train)
            y_train = rng.normal(mu[1], 2.0, n_per_player_train)
            x_test = rng.normal(mu[0], 2.0, n_per_player_test)
            y_test = rng.normal(mu[1], 2.0, n_per_player_test)
            for x_, y_ in zip(x_train, y_train, strict=True):
                train_rows.append({"x": x_, "y": y_, "player_id": pid})
            for x_, y_ in zip(x_test, y_test, strict=True):
                test_rows.append({"x": x_, "y": y_, "player_id": pid})
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


@pytest.fixture
def synthetic_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _synthetic_shots_df()


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


def test_returns_ablation_result(synthetic_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = synthetic_split
    out = run_ablation(train, test, R_samples=10, sw_n_projections=20, retrieval_n_projections=20)
    assert isinstance(out, AblationResult)
    assert isinstance(out.summary, pd.DataFrame)
    assert isinstance(out.per_player, pd.DataFrame)


def test_summary_has_one_row_per_model(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = synthetic_split
    out = run_ablation(train, test, R_samples=10, sw_n_projections=20, retrieval_n_projections=20)
    assert set(out.summary["model"]) == set(model_order())
    assert len(out.summary) == len(model_order())


def test_per_player_has_model_x_player_rows(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = synthetic_split
    out = run_ablation(train, test, R_samples=10, sw_n_projections=20, retrieval_n_projections=20)
    n_models = len(model_order())
    n_players = test["player_id"].nunique()
    assert len(out.per_player) == n_models * n_players


def test_summary_columns_present(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = synthetic_split
    out = run_ablation(train, test, R_samples=10, sw_n_projections=20, retrieval_n_projections=20)
    expected = {
        "model",
        "n_players",
        "n_test_shots",
        "nll_mean",
        "nll_median",
        "zone_kl_mean",
        "sw_mean",
        "sw_median",
        "retrieval_top1",
        "retrieval_topk",
    }
    assert expected.issubset(out.summary.columns)


# ---------------------------------------------------------------------------
# Metric sanity
# ---------------------------------------------------------------------------


def test_nll_finite_and_non_negative(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = synthetic_split
    out = run_ablation(train, test, R_samples=10, sw_n_projections=20, retrieval_n_projections=20)
    assert out.per_player["nll_per_shot"].notna().all()
    assert (out.per_player["nll_per_shot"] >= 0).all()


def test_zone_kl_non_negative(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = synthetic_split
    out = run_ablation(train, test, R_samples=10, sw_n_projections=20, retrieval_n_projections=20)
    assert (out.per_player["zone_kl_5"] >= 0).all()


def test_sw_non_negative(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = synthetic_split
    out = run_ablation(train, test, R_samples=10, sw_n_projections=20, retrieval_n_projections=20)
    assert (out.per_player["sliced_wasserstein"] >= 0).all()


def test_retrieval_in_unit_interval(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = synthetic_split
    out = run_ablation(train, test, R_samples=20, sw_n_projections=30, retrieval_n_projections=30)
    for col in ("retrieval_top1", "retrieval_topk"):
        assert out.summary[col].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# Ranking expectations
# ---------------------------------------------------------------------------


def test_player_kde_beats_league_kde_on_nll(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Per-player KDE should have lower NLL than League KDE on player-specific test shots."""
    train, test = synthetic_split
    out = run_ablation(train, test, R_samples=20, sw_n_projections=30, retrieval_n_projections=30)
    league_nll = out.summary.loc[out.summary["model"] == "League KDE", "nll_mean"].iloc[0]
    player_nll = out.summary.loc[out.summary["model"] == "Player KDE (raw)", "nll_mean"].iloc[0]
    assert player_nll < league_nll, (
        f"Player KDE (raw) NLL ({player_nll:.4f}) should beat League KDE NLL ({league_nll:.4f})"
    )


def test_player_kde_beats_league_kde_on_retrieval(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Per-player retrieval should be much better than league average."""
    train, test = synthetic_split
    out = run_ablation(train, test, R_samples=30, sw_n_projections=50, retrieval_n_projections=50)
    league_top1 = out.summary.loc[out.summary["model"] == "League KDE", "retrieval_top1"].iloc[0]
    player_top1 = out.summary.loc[
        out.summary["model"] == "Player KDE (raw)", "retrieval_top1"
    ].iloc[0]
    # League KDE retrieval is essentially chance (1/n_players); player KDE
    # should retrieve correctly far more often.
    assert player_top1 > league_top1


def test_ablation_accepts_position_map_and_recency(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """End-to-end: pass real position groups + recency, verify no errors."""
    train, test = synthetic_split
    train = train.copy()
    train["date"] = pd.date_range("2024-01-01", periods=len(train), freq="h")
    pos_map = {"G0": "guard", "G1": "guard", "F0": "big", "F1": "big"}

    out = run_ablation(
        train,
        test,
        position_map=pos_map,
        recency_half_life_days=180.0,
        R_samples=10,
        sw_n_projections=20,
        retrieval_n_projections=20,
    )
    assert len(out.summary) == 4
    assert (out.summary["nll_mean"] > 0).all()


def test_ablation_recency_disabled_when_no_date_column(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Passing recency_half_life_days when there's no date column: silently disabled."""
    train, test = synthetic_split
    # No 'date' column exists in the synthetic split.
    out = run_ablation(
        train,
        test,
        recency_half_life_days=365.0,
        R_samples=10,
        sw_n_projections=20,
        retrieval_n_projections=20,
    )
    assert len(out.summary) == 4


# ---------------------------------------------------------------------------
# Sparse-player filter
# ---------------------------------------------------------------------------


def _split_with_uneven_train_shots() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a 4-player split where two players have many train shots and two have few."""
    rng = np.random.default_rng(0)
    train_rows = []
    test_rows = []
    counts = {"DENSE_A": 600, "DENSE_B": 400, "SPARSE_A": 80, "SPARSE_B": 60}
    for pid, n in counts.items():
        x = rng.normal(0.0, 5.0, n)
        y = rng.normal(15.0, 5.0, n)
        for x_, y_ in zip(x, y, strict=True):
            train_rows.append({"x": x_, "y": y_, "player_id": pid})
        # Each player gets ~30 test shots regardless of training count.
        x_test = rng.normal(0.0, 5.0, 30)
        y_test = rng.normal(15.0, 5.0, 30)
        for x_, y_ in zip(x_test, y_test, strict=True):
            test_rows.append({"x": x_, "y": y_, "player_id": pid})
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def test_min_train_shots_keeps_only_dense_players() -> None:
    train, test = _split_with_uneven_train_shots()
    out = run_ablation(
        train,
        test,
        min_train_shots=200,
        R_samples=5,
        sw_n_projections=10,
        retrieval_n_projections=10,
    )
    keep = set(out.per_player["player_id"].unique())
    assert keep == {"DENSE_A", "DENSE_B"}


def test_max_train_shots_keeps_only_sparse_players() -> None:
    train, test = _split_with_uneven_train_shots()
    out = run_ablation(
        train,
        test,
        max_train_shots=200,
        R_samples=5,
        sw_n_projections=10,
        retrieval_n_projections=10,
    )
    keep = set(out.per_player["player_id"].unique())
    assert keep == {"SPARSE_A", "SPARSE_B"}


def test_train_shot_filter_combined_window() -> None:
    train, test = _split_with_uneven_train_shots()
    out = run_ablation(
        train,
        test,
        min_train_shots=70,
        max_train_shots=500,
        R_samples=5,
        sw_n_projections=10,
        retrieval_n_projections=10,
    )
    keep = set(out.per_player["player_id"].unique())
    # SPARSE_B has 60 train shots (below 70); DENSE_A has 600 (above 500).
    assert keep == {"SPARSE_A", "DENSE_B"}


def test_train_shot_filter_too_strict_raises() -> None:
    train, test = _split_with_uneven_train_shots()
    with pytest.raises(ValueError, match="no players satisfy"):
        run_ablation(train, test, min_train_shots=10_000)


# ---------------------------------------------------------------------------
# extra_models hook
# ---------------------------------------------------------------------------


def test_extra_models_appears_as_extra_summary_row(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A user-supplied density function should show up as a 5th model."""
    train, test = synthetic_split
    g = pd.read_csv  # only here to keep imports clean — replaced below.
    del g

    n_cells = 64 * 56  # default CourtGrid

    def uniform_density(_pid: object) -> np.ndarray:
        return np.full((56, 64), 1.0 / n_cells)

    out = run_ablation(
        train,
        test,
        R_samples=10,
        sw_n_projections=20,
        retrieval_n_projections=20,
        extra_models={"Uniform": uniform_density},
    )
    assert "Uniform" in set(out.summary["model"])
    assert len(out.summary) == 5  # 4 baselines + 1 extra


def test_extra_models_name_collision_raises(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = synthetic_split

    def fake_density(_pid: object) -> np.ndarray:
        return np.full((56, 64), 1.0 / (56 * 64))

    with pytest.raises(ValueError, match="collides"):
        run_ablation(
            train,
            test,
            extra_models={"League KDE": fake_density},  # already a built-in
            R_samples=5,
        )


# ---------------------------------------------------------------------------
# make_density_fn_from_checkpoint — wires a trained checkpoint into ablation
# ---------------------------------------------------------------------------


def test_make_density_fn_from_checkpoint_round_trips(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame], tmp_path: pytest.TempPathFactory
) -> None:
    """Build a checkpoint, wire it through `make_density_fn_from_checkpoint`,
    and verify the resulting density matches a hand-computed forward pass."""
    import torch

    from shotcloud import CourtGrid, HierarchicalKDE, PlayerVocab
    from shotcloud.legacy import KDEProduct, PlayerEmbeddingEncoder
    from shotcloud.legacy_pivot.checkpoint import (
        load_decoder_checkpoint,
        save_decoder_checkpoint,
    )
    from shotcloud.legacy_pivot.eval_ablation import make_density_fn_from_checkpoint
    from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder

    train, _ = synthetic_split
    grid = CourtGrid()
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, recency_half_life_days=None)
    kde.fit(
        x=train["x"].to_numpy(),
        y=train["y"].to_numpy(),
        player_id=train["player_id"].to_numpy(),
        position=np.array(["G"] * len(train)),
    )
    base = KDEProduct(hierarchical_kde=kde, weights={"a_p": 1.0, "a_g": 0.0, "a_0": 0.0})

    vocab = PlayerVocab.from_ids(["G0", "G1", "F0", "F1"])
    encoder = PlayerEmbeddingEncoder(n_players=len(vocab), rank=4, zero_init=False)
    decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4, zero_init=False)

    # Save then load to exercise the full flow.
    ckpt_path = tmp_path / "ckpt.pt"
    save_decoder_checkpoint(ckpt_path, encoder=encoder, decoder=decoder, vocab=vocab)
    loaded = load_decoder_checkpoint(ckpt_path)

    fn = make_density_fn_from_checkpoint(loaded, base, grid)

    # Reference computation by hand.
    pid = "G0"
    log_q0 = base.log_density(pid).ravel()
    log_q0_t = torch.as_tensor(log_q0, dtype=loaded.decoder.V.dtype).unsqueeze(0)
    idx = torch.tensor([loaded.vocab.to_idx(pid)], dtype=torch.long)
    with torch.no_grad():
        u = loaded.encoder(idx)
        expected_probs = loaded.decoder.probs(log_q0_t, u).numpy().ravel()
    expected = expected_probs.reshape(grid.ny, grid.nx)

    actual = fn(pid)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(actual.sum(), 1.0)
    assert (actual > 0).all()


def test_run_ablation_with_trained_extra_model(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame], tmp_path: pytest.TempPathFactory
) -> None:
    """End-to-end: trained checkpoint added as a 5th ablation row."""
    from shotcloud import CourtGrid, HierarchicalKDE, PlayerVocab
    from shotcloud.legacy import KDEProduct, PlayerEmbeddingEncoder
    from shotcloud.legacy_pivot.checkpoint import (
        load_decoder_checkpoint,
        save_decoder_checkpoint,
    )
    from shotcloud.legacy_pivot.eval_ablation import make_density_fn_from_checkpoint
    from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder

    train, test = synthetic_split
    grid = CourtGrid()
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, recency_half_life_days=None)
    kde.fit(
        x=train["x"].to_numpy(),
        y=train["y"].to_numpy(),
        player_id=train["player_id"].to_numpy(),
        position=np.array(["G"] * len(train)),
    )
    base = KDEProduct(hierarchical_kde=kde, weights={"a_p": 1.0, "a_g": 0.0, "a_0": 0.0})

    vocab = PlayerVocab.from_ids(train["player_id"].unique())
    encoder = PlayerEmbeddingEncoder(n_players=len(vocab), rank=4, zero_init=False)
    decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4, zero_init=True)

    ckpt_path = tmp_path / "ckpt.pt"
    save_decoder_checkpoint(ckpt_path, encoder=encoder, decoder=decoder, vocab=vocab)
    loaded = load_decoder_checkpoint(ckpt_path)
    fn = make_density_fn_from_checkpoint(loaded, base, grid)

    out = run_ablation(
        train,
        test,
        R_samples=10,
        sw_n_projections=20,
        retrieval_n_projections=20,
        extra_models={"Trained": fn},
    )
    assert "Trained" in set(out.summary["model"])
    assert len(out.summary) == 5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_missing_required_column_raises() -> None:
    train = pd.DataFrame({"x": [0.0], "y": [0.0]})  # no player_id
    test = pd.DataFrame({"x": [0.0], "y": [0.0], "player_id": ["A"]})
    with pytest.raises(KeyError, match="player_id"):
        run_ablation(train, test)


def test_no_player_overlap_raises() -> None:
    train = pd.DataFrame({"x": [0.0], "y": [0.0], "player_id": ["A"]})
    test = pd.DataFrame({"x": [0.0], "y": [0.0], "player_id": ["B"]})
    with pytest.raises(ValueError, match="no overlap"):
        run_ablation(train, test)


def test_no_in_court_test_shots_raises() -> None:
    train = pd.DataFrame({"x": [0.0, 1.0], "y": [5.0, 6.0], "player_id": ["A", "A"]})
    # Test shots all out-of-bounds (y > 47 → backcourt).
    test = pd.DataFrame({"x": [0.0, 0.0], "y": [50.0, 60.0], "player_id": ["A", "A"]})
    with pytest.raises(ValueError, match="no test players"):
        run_ablation(train, test)


# ---------------------------------------------------------------------------
# Real-data smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SHOT_FLOW_CSV.exists(), reason="shot_flow CSV not available")
def test_runs_on_real_shot_flow_data() -> None:
    """Smoke test: load a slice of real NBA data, fractional split, run."""
    df = load_shots(SHOT_FLOW_CSV, nrows=20_000)
    if "season" in df.columns:
        df = df.drop(columns=["season"])

    # Need players with shots in both train and test halves; keep only
    # players with at least 20 shots so the random split has overlap.
    counts = df["player_id"].value_counts()
    eligible = counts[counts >= 20].index
    df = df[df["player_id"].isin(eligible)].copy()

    rng = np.random.default_rng(0)
    df = df.iloc[rng.permutation(len(df))]
    n_train = int(len(df) * 0.7)
    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)

    common = set(train["player_id"]) & set(test["player_id"])
    assert len(common) >= 3, f"smoke test needs >= 3 overlapping players, got {len(common)}"

    out = run_ablation(
        train,
        test,
        R_samples=10,
        sw_n_projections=20,
        retrieval_n_projections=20,
    )
    assert len(out.summary) == 4
    assert (out.summary["nll_mean"] > 0).all()
    assert (out.summary["zone_kl_mean"] >= 0).all()
    assert (out.summary["sw_mean"] >= 0).all()
