"""Tests for Phase 0.5 + Phase 1.5 KDE diagnostic helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shotcloud import CourtGrid, HierarchicalKDE, PlayerVocab
from shotcloud.legacy import PlayerEmbeddingEncoder
from shotcloud.legacy_pivot.eval_diagnostics import (
    SHOT_COUNT_BUCKET_LABELS,
    attach_trained_nll,
    build_per_player_diagnostics,
    entropy,
    held_out_nll,
    per_player_trained_nll,
    shot_count_bucket,
    summarize_by_bucket,
    top_k_mass,
)
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder

# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------


def test_entropy_uniform_equals_log_n() -> None:
    """H(uniform over N cells) = log(N)."""
    n = 100
    q = np.full(n, 1.0 / n)
    assert abs(entropy(q) - np.log(n)) < 1e-10


def test_entropy_one_hot_is_zero() -> None:
    """A degenerate point mass has entropy 0."""
    q = np.zeros(10)
    q[3] = 1.0
    assert abs(entropy(q)) < 1e-10


def test_entropy_image_layout_matches_flat() -> None:
    """entropy is invariant under reshape."""
    rng = np.random.default_rng(0)
    q_flat = rng.dirichlet(np.ones(20))
    q_2d = q_flat.reshape(4, 5)
    assert abs(entropy(q_flat) - entropy(q_2d)) < 1e-12


def test_entropy_negative_input_raises() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        entropy(np.array([-0.1, 0.5, 0.6]))


def test_entropy_empty_input_is_nan() -> None:
    assert np.isnan(entropy(np.array([])))


def test_entropy_skips_zeros() -> None:
    """Zero-mass cells should contribute nothing (and not raise)."""
    q = np.array([0.0, 0.5, 0.5])
    expected = -2 * 0.5 * np.log(0.5)
    assert abs(entropy(q) - expected) < 1e-12


# ---------------------------------------------------------------------------
# top_k_mass
# ---------------------------------------------------------------------------


def test_top_k_mass_uniform_equals_k_over_n() -> None:
    n = 100
    q = np.full(n, 1.0 / n)
    assert abs(top_k_mass(q, 10) - 0.1) < 1e-12
    assert abs(top_k_mass(q, 1) - 0.01) < 1e-12


def test_top_k_mass_one_hot_peaks_at_one() -> None:
    q = np.zeros(10)
    q[5] = 1.0
    assert abs(top_k_mass(q, 1) - 1.0) < 1e-12
    assert abs(top_k_mass(q, 5) - 1.0) < 1e-12


def test_top_k_mass_k_exceeds_n_returns_total() -> None:
    q = np.array([0.2, 0.3, 0.5])
    assert abs(top_k_mass(q, 100) - 1.0) < 1e-12


def test_top_k_mass_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        top_k_mass(np.array([0.5, 0.5]), 0)
    with pytest.raises(ValueError, match="positive"):
        top_k_mass(np.array([0.5, 0.5]), -1)


def test_top_k_mass_image_layout() -> None:
    """top_k_mass works on either image-layout or flat input."""
    rng = np.random.default_rng(0)
    q_flat = rng.dirichlet(np.ones(20))
    q_2d = q_flat.reshape(4, 5)
    assert abs(top_k_mass(q_flat, 5) - top_k_mass(q_2d, 5)) < 1e-12


# ---------------------------------------------------------------------------
# held_out_nll
# ---------------------------------------------------------------------------


def test_held_out_nll_matches_manual() -> None:
    q = np.array([0.1, 0.2, 0.3, 0.4])
    cells = np.array([0, 2, 3, 3])
    expected = -np.mean(np.log([0.1, 0.3, 0.4, 0.4]))
    assert abs(held_out_nll(q, cells) - expected) < 1e-12


def test_held_out_nll_empty_is_nan() -> None:
    q = np.array([0.5, 0.5])
    assert np.isnan(held_out_nll(q, np.array([], dtype=np.int64)))


def test_held_out_nll_negative_cell_raises() -> None:
    q = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="backcourt"):
        held_out_nll(q, np.array([-1, 0]))


def test_held_out_nll_zero_density_cell_raises() -> None:
    """If q is 0 at an observed cell, log(0) is -inf — better to error."""
    q = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        held_out_nll(q, np.array([0]))


# ---------------------------------------------------------------------------
# shot_count_bucket
# ---------------------------------------------------------------------------


def test_shot_count_bucket_boundaries() -> None:
    assert shot_count_bucket(0) == "<50"
    assert shot_count_bucket(49) == "<50"
    assert shot_count_bucket(50) == "50-500"
    assert shot_count_bucket(499) == "50-500"
    assert shot_count_bucket(500) == "500-2000"
    assert shot_count_bucket(1999) == "500-2000"
    assert shot_count_bucket(2000) == ">=2000"
    assert shot_count_bucket(10_000) == ">=2000"


def test_shot_count_bucket_negative_raises() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        shot_count_bucket(-1)


# ---------------------------------------------------------------------------
# build_per_player_diagnostics — synthetic 2-player KDE
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_kde_two_players() -> HierarchicalKDE:
    """Two players with very different density geometry."""
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    # Player A — many shots, concentrated near the basket
    for _ in range(800):
        rows.append(
            {
                "x": float(rng.normal(0, 2)),
                "y": float(rng.normal(4, 2)),
                "player_id": "A",
                "position": "G",
            }
        )
    # Player B — sparse, broader
    for _ in range(40):
        rows.append(
            {
                "x": float(rng.normal(0, 8)),
                "y": float(rng.normal(20, 8)),
                "player_id": "B",
                "position": "G",
            }
        )
    df = pd.DataFrame(rows)
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=200.0, recency_half_life_days=None)
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        position=df["position"].to_numpy(),
    )
    return kde


def _held_out_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows: list[dict[str, object]] = []
    for _ in range(50):
        rows.append({"x": float(rng.normal(0, 2)), "y": float(rng.normal(4, 2)), "player_id": "A"})
    for _ in range(20):
        rows.append({"x": float(rng.normal(0, 8)), "y": float(rng.normal(20, 8)), "player_id": "B"})
    return pd.DataFrame(rows)


def test_per_player_frame_shape_and_columns(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    held = _held_out_frame()
    df = build_per_player_diagnostics(fitted_kde_two_players, held, fitted_kde_two_players.grid)
    assert set(df["player_id"]) == {"A", "B"}
    expected_cols = {
        "player_id",
        "position",
        "n_train",
        "n_held",
        "bucket",
        "entropy_raw",
        "entropy_hier",
        "entropy_gap",
        "top1_mass_raw",
        "top1_mass_hier",
        "top5_mass_raw",
        "top5_mass_hier",
        "top25_mass_raw",
        "top25_mass_hier",
        "nll_raw_held",
        "nll_hier_held",
        "nll_gain_hier_over_raw",
    }
    assert expected_cols.issubset(df.columns)


def test_sparse_player_has_larger_entropy_gap(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    """Shrinkage should affect the sparse player more than the dense one."""
    held = _held_out_frame()
    df = build_per_player_diagnostics(
        fitted_kde_two_players, held, fitted_kde_two_players.grid
    ).set_index("player_id")
    # B has 40 train shots → heavy shrinkage; A has 800 → light shrinkage.
    assert abs(df.loc["B", "entropy_gap"]) > abs(df.loc["A", "entropy_gap"])


def test_dense_player_is_sharper(fitted_kde_two_players: HierarchicalKDE) -> None:
    """A's near-basket density should have higher top-1 mass than B's broad cloud."""
    held = _held_out_frame()
    df = build_per_player_diagnostics(
        fitted_kde_two_players, held, fitted_kde_two_players.grid
    ).set_index("player_id")
    assert df.loc["A", "top1_mass_hier"] > df.loc["B", "top1_mass_hier"]


def test_held_out_nll_finite_when_in_court(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    held = _held_out_frame()
    df = build_per_player_diagnostics(fitted_kde_two_players, held, fitted_kde_two_players.grid)
    assert df["nll_raw_held"].notna().all()
    assert df["nll_hier_held"].notna().all()
    assert (df["nll_raw_held"] > 0).all()
    assert (df["nll_hier_held"] > 0).all()


def test_player_absent_from_held_out_gets_nan_nll(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    """A fitted player with no held-out shots should get descriptive stats but NaN NLL."""
    held = pd.DataFrame({"x": [0.0], "y": [4.0], "player_id": ["A"]})  # only A
    df = build_per_player_diagnostics(
        fitted_kde_two_players, held, fitted_kde_two_players.grid
    ).set_index("player_id")
    assert not np.isnan(df.loc["A", "nll_hier_held"])
    assert np.isnan(df.loc["B", "nll_hier_held"])
    # Descriptive stats still populated for B.
    assert not np.isnan(df.loc["B", "entropy_hier"])


def test_per_player_missing_required_column_raises(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    bad = pd.DataFrame({"x": [0.0], "y": [4.0]})  # no player_id
    with pytest.raises(KeyError, match="player_id"):
        build_per_player_diagnostics(fitted_kde_two_players, bad, fitted_kde_two_players.grid)


def test_unfitted_kde_raises() -> None:
    unfit = HierarchicalKDE(grid=CourtGrid())
    with pytest.raises(ValueError, match="must be fit"):
        build_per_player_diagnostics(unfit, pd.DataFrame(), CourtGrid())


# ---------------------------------------------------------------------------
# summarize_by_bucket
# ---------------------------------------------------------------------------


def test_summary_index_is_canonical_bucket_order(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    held = _held_out_frame()
    per_player = build_per_player_diagnostics(
        fitted_kde_two_players, held, fitted_kde_two_players.grid
    )
    summary = summarize_by_bucket(per_player)
    # Index should be the four canonical labels in order, even if some are empty.
    assert list(summary.index) == list(SHOT_COUNT_BUCKET_LABELS)


def test_summary_counts_match_bucket_assignment(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    held = _held_out_frame()
    per_player = build_per_player_diagnostics(
        fitted_kde_two_players, held, fitted_kde_two_players.grid
    )
    summary = summarize_by_bucket(per_player)
    # Each player's bucket population should match the per-player frame.
    for label in SHOT_COUNT_BUCKET_LABELS:
        expected = int((per_player["bucket"] == label).sum())
        assert int(summary.loc[label, "n_players"]) == expected


def test_summary_weighted_nll_uses_n_held() -> None:
    """If two players have different n_held, the weighted mean must reflect that."""
    per_player = pd.DataFrame(
        [
            {"bucket": ">=2000", "n_train": 3000, "n_held": 100, "nll_hier_held": 6.0},
            {"bucket": ">=2000", "n_train": 3000, "n_held": 1, "nll_hier_held": 100.0},
        ]
    )
    # Fill the columns expected by summarize_by_bucket (we only care about one).
    for col in ("nll_raw_held", "nll_gain_hier_over_raw"):
        per_player[col] = 0.0
    for col in (
        "entropy_raw",
        "entropy_hier",
        "top1_mass_hier",
        "top5_mass_hier",
    ):
        per_player[col] = 0.0
    summary = summarize_by_bucket(per_player)
    expected_weighted = (100 * 6.0 + 1 * 100.0) / 101
    assert abs(summary.loc[">=2000", "weighted_nll_hier_held"] - expected_weighted) < 1e-9


def test_summary_missing_bucket_column_raises() -> None:
    with pytest.raises(KeyError, match="bucket"):
        summarize_by_bucket(pd.DataFrame({"x": [1, 2]}))


# ---------------------------------------------------------------------------
# Phase 1.5 — per-player trained NLL
# ---------------------------------------------------------------------------


def _trained_setup(
    fitted_kde: HierarchicalKDE,
) -> tuple[PlayerEmbeddingEncoder, LowRankTiltDecoder, PlayerVocab]:
    """Build a vocab + zero-init encoder + zero-init decoder for trained-NLL tests."""
    vocab = PlayerVocab.from_ids(list(fitted_kde.player_density_grid))
    rank = 4
    n_cells = fitted_kde.grid.n_cells
    encoder = PlayerEmbeddingEncoder(n_players=len(vocab), rank=rank, zero_init=True)
    decoder = LowRankTiltDecoder(n_cells=n_cells, rank=rank, zero_init=True)
    return encoder, decoder, vocab


def test_trained_nll_with_zero_decoder_and_tau_one_matches_hier(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    """At V=0, u=0, τ=1, the trained model = the classical Hier-KDE.

    Therefore per-player trained NLL must equal the per-player
    ``nll_hier_held`` from the Phase 0.5 frame.
    """
    encoder, decoder, vocab = _trained_setup(fitted_kde_two_players)
    held = _held_out_frame()

    # Phase-0.5 reference frame.
    classical = build_per_player_diagnostics(
        fitted_kde_two_players, held, fitted_kde_two_players.grid
    ).set_index("player_id")

    trained = per_player_trained_nll(
        fitted_kde_two_players,
        held,
        fitted_kde_two_players.grid,
        encoder=encoder,
        decoder=decoder,
        vocab=vocab,
        tau=1.0,
    )

    for pid in classical.index:
        ref = float(classical.loc[pid, "nll_hier_held"])
        got = trained[pid]
        assert abs(got - ref) < 1e-4, f"{pid}: trained={got} hier_ref={ref}"


def test_trained_nll_uniform_when_tau_zero_and_decoder_zero(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    """At τ=0 and V=0, log_p is uniform → per-shot NLL = log(n_cells)."""
    encoder, decoder, vocab = _trained_setup(fitted_kde_two_players)
    held = _held_out_frame()

    # τ must be > 0 by API contract; demonstrate via tiny ε.
    trained = per_player_trained_nll(
        fitted_kde_two_players,
        held,
        fitted_kde_two_players.grid,
        encoder=encoder,
        decoder=decoder,
        vocab=vocab,
        tau=1e-8,
    )
    n_cells = fitted_kde_two_players.grid.n_cells
    expected = float(np.log(n_cells))
    for pid, val in trained.items():
        assert not np.isnan(val)
        assert abs(val - expected) < 5e-3, f"{pid}: got {val:.4f} vs uniform {expected:.4f}"


def test_trained_nll_player_with_no_held_out_is_nan(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    encoder, decoder, vocab = _trained_setup(fitted_kde_two_players)
    held = pd.DataFrame({"x": [0.0], "y": [4.0], "player_id": ["A"]})  # only A
    trained = per_player_trained_nll(
        fitted_kde_two_players,
        held,
        fitted_kde_two_players.grid,
        encoder=encoder,
        decoder=decoder,
        vocab=vocab,
    )
    assert not np.isnan(trained["A"])
    assert np.isnan(trained["B"])


def test_trained_nll_invalid_tau_raises(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    encoder, decoder, vocab = _trained_setup(fitted_kde_two_players)
    held = _held_out_frame()
    with pytest.raises(ValueError, match="strictly positive"):
        per_player_trained_nll(
            fitted_kde_two_players,
            held,
            fitted_kde_two_players.grid,
            encoder=encoder,
            decoder=decoder,
            vocab=vocab,
            tau=0.0,
        )


def test_trained_nll_unfitted_kde_raises() -> None:
    encoder = PlayerEmbeddingEncoder(n_players=1, rank=4)
    decoder = LowRankTiltDecoder(n_cells=10, rank=4)
    vocab = PlayerVocab.from_ids(["A"])
    with pytest.raises(ValueError, match="must be fit"):
        per_player_trained_nll(
            HierarchicalKDE(grid=CourtGrid()),
            pd.DataFrame({"x": [], "y": [], "player_id": []}),
            CourtGrid(),
            encoder=encoder,
            decoder=decoder,
            vocab=vocab,
        )


def test_attach_trained_nll_adds_columns(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    held = _held_out_frame()
    base = build_per_player_diagnostics(fitted_kde_two_players, held, fitted_kde_two_players.grid)
    trained = {"A": 6.0, "B": 8.0}
    out = attach_trained_nll(base, trained)
    assert "nll_trained_held" in out.columns
    assert "nll_gain_trained_over_hier" in out.columns
    a = out[out["player_id"] == "A"].iloc[0]
    assert abs(a["nll_trained_held"] - 6.0) < 1e-9
    # gain = nll_hier_held - nll_trained_held
    assert abs(a["nll_gain_trained_over_hier"] - (a["nll_hier_held"] - 6.0)) < 1e-9


def test_attach_trained_nll_does_not_mutate_input(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    held = _held_out_frame()
    base = build_per_player_diagnostics(fitted_kde_two_players, held, fitted_kde_two_players.grid)
    cols_before = list(base.columns)
    _ = attach_trained_nll(base, {"A": 6.0, "B": 8.0})
    assert list(base.columns) == cols_before  # no mutation


def test_summary_includes_trained_columns_when_present(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    held = _held_out_frame()
    encoder, decoder, vocab = _trained_setup(fitted_kde_two_players)
    base = build_per_player_diagnostics(fitted_kde_two_players, held, fitted_kde_two_players.grid)
    trained = per_player_trained_nll(
        fitted_kde_two_players,
        held,
        fitted_kde_two_players.grid,
        encoder=encoder,
        decoder=decoder,
        vocab=vocab,
        tau=1.0,
    )
    extended = attach_trained_nll(base, trained)
    summary = summarize_by_bucket(extended)
    assert "weighted_nll_trained_held" in summary.columns
    assert "weighted_nll_gain_trained_over_hier" in summary.columns


def test_summary_omits_trained_columns_when_absent(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    held = _held_out_frame()
    base = build_per_player_diagnostics(fitted_kde_two_players, held, fitted_kde_two_players.grid)
    summary = summarize_by_bucket(base)
    assert "weighted_nll_trained_held" not in summary.columns


def test_trained_nll_default_init_encoder_changes_when_decoder_zero(
    fitted_kde_two_players: HierarchicalKDE,
) -> None:
    """Sanity: with V=0, u doesn't matter — trained NLL still equals classical hier."""
    vocab = PlayerVocab.from_ids(list(fitted_kde_two_players.player_density_grid))
    rank = 4
    n_cells = fitted_kde_two_players.grid.n_cells
    encoder = PlayerEmbeddingEncoder(n_players=len(vocab), rank=rank, zero_init=False)  # random
    decoder = LowRankTiltDecoder(n_cells=n_cells, rank=rank, zero_init=True)  # zero
    held = _held_out_frame()
    trained = per_player_trained_nll(
        fitted_kde_two_players,
        held,
        fitted_kde_two_players.grid,
        encoder=encoder,
        decoder=decoder,
        vocab=vocab,
        tau=1.0,
    )
    classical = build_per_player_diagnostics(
        fitted_kde_two_players, held, fitted_kde_two_players.grid
    ).set_index("player_id")
    for pid in classical.index:
        assert abs(trained[pid] - classical.loc[pid, "nll_hier_held"]) < 1e-4
