"""Tests for :class:`shotcloud.training.ShotCellDataset` + :class:`PlayerVocab`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud import CourtGrid, HierarchicalKDE, PlayerVocab
from shotcloud.legacy import KDEProduct
from shotcloud.legacy_pivot.defensive_kde import DefensiveKDE
from shotcloud.legacy_pivot.shot_cell_dataset import ShotCellDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_shots(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pid, mu_y in [("A", 4.0), ("B", 24.0)]:
        for _ in range(n):
            rows.append(
                {
                    "x": float(rng.normal(0, 2)),
                    "y": float(rng.normal(mu_y, 2)),
                    "player_id": pid,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_kde() -> HierarchicalKDE:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    df = _synthetic_shots(n=80)
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=200.0, recency_half_life_days=None)
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        position=np.array(["G"] * len(df)),
    )
    return kde


@pytest.fixture
def base_measure(fitted_kde: HierarchicalKDE) -> KDEProduct:
    return KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.0, "a_0": 0.0})


# ---------------------------------------------------------------------------
# PlayerVocab
# ---------------------------------------------------------------------------


def test_vocab_round_trips_ids() -> None:
    v = PlayerVocab.from_ids(["B", "A", "C", "A"])  # duplicates dropped, sorted
    assert len(v) == 3
    assert v.ids == ("A", "B", "C")
    assert v.to_idx("A") == 0
    assert v.to_id(2) == "C"


def test_vocab_unknown_player_raises() -> None:
    v = PlayerVocab.from_ids(["A", "B"])
    with pytest.raises(KeyError, match="not in vocab"):
        v.to_idx("C")


# ---------------------------------------------------------------------------
# ShotCellDataset
# ---------------------------------------------------------------------------


def test_dataset_yields_correct_triples(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    df = _synthetic_shots(n=20)
    ds = ShotCellDataset(df, base_measure, fitted_kde.grid)
    assert len(ds) > 0

    pid_idx, cell, log_q0, opp_idx, x_n = ds[0]
    assert isinstance(pid_idx, torch.Tensor) and pid_idx.dtype == torch.int64
    assert isinstance(cell, torch.Tensor) and cell.dtype == torch.int64
    assert log_q0.shape == (fitted_kde.grid.n_cells,)
    # log_q0 should be normalized in probability space.
    np.testing.assert_allclose(torch.exp(log_q0).sum().item(), 1.0, atol=1e-5)
    # When no defensive_kde is attached, opponent_idx defaults to 0.
    assert isinstance(opp_idx, torch.Tensor) and opp_idx.dtype == torch.int64
    assert int(opp_idx.item()) == 0
    # When no context_encoder is attached, the per-shot x_n is empty (shape (0,)).
    assert isinstance(x_n, torch.Tensor)
    assert x_n.shape == (0,)


def test_dataset_drops_backcourt_shots(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    df = _synthetic_shots(n=20)
    df_extended = pd.concat(
        [df, pd.DataFrame({"x": [0.0], "y": [60.0], "player_id": ["A"]})],
        ignore_index=True,
    )
    ds = ShotCellDataset(df_extended, base_measure, fitted_kde.grid)
    # 60ft is beyond the half-court; the extra row should be dropped.
    assert len(ds) == len(df)


def test_dataset_drops_unknown_players(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    df = _synthetic_shots(n=20)
    df_extended = pd.concat(
        [df, pd.DataFrame({"x": [0.0], "y": [5.0], "player_id": ["UNKNOWN"]})],
        ignore_index=True,
    )
    ds = ShotCellDataset(df_extended, base_measure, fitted_kde.grid)
    assert len(ds) == len(df)


def test_dataset_log_q0_table_matches_base_measure(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    df = _synthetic_shots(n=20)
    ds = ShotCellDataset(df, base_measure, fitted_kde.grid)
    for pid in ("A", "B"):
        idx = ds.vocab.to_idx(pid)
        expected = base_measure.log_density(pid).ravel()
        actual = ds.log_q0_table[idx].numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_dataset_missing_required_column_raises(base_measure: KDEProduct) -> None:
    df = pd.DataFrame({"x": [0.0], "y": [0.0]})  # no player_id
    with pytest.raises(KeyError, match="player_id"):
        ShotCellDataset(df, base_measure, base_measure.grid)


def test_dataset_empty_after_filtering_raises(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    df = pd.DataFrame({"x": [0.0], "y": [60.0], "player_id": ["A"]})  # only backcourt
    with pytest.raises(ValueError, match="no in-court shots"):
        ShotCellDataset(df, base_measure, fitted_kde.grid)


# ---------------------------------------------------------------------------
# HierarchicalKDEBase support
# ---------------------------------------------------------------------------


def test_dataset_accepts_hierarchical_base(fitted_kde: HierarchicalKDE) -> None:
    from shotcloud import HierarchicalKDEBase

    base = HierarchicalKDEBase(hierarchical_kde=fitted_kde)
    df = _synthetic_shots(n=20)
    ds = ShotCellDataset(df, base, fitted_kde.grid)
    assert len(ds) > 0
    _, _, log_q0, _, _ = ds[0]
    np.testing.assert_allclose(torch.exp(log_q0).sum().item(), 1.0, atol=1e-5)


def test_dataset_log_q0_table_matches_hierarchical_base(
    fitted_kde: HierarchicalKDE,
) -> None:
    from shotcloud import HierarchicalKDEBase

    base = HierarchicalKDEBase(hierarchical_kde=fitted_kde)
    df = _synthetic_shots(n=20)
    ds = ShotCellDataset(df, base, fitted_kde.grid)
    for pid in ("A", "B"):
        idx = ds.vocab.to_idx(pid)
        expected = base.log_density(pid).ravel()
        actual = ds.log_q0_table[idx].numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Component caching for learnable weights
# ---------------------------------------------------------------------------


def test_dataset_components_cached_when_requested(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    df = _synthetic_shots(n=20)
    ds = ShotCellDataset(df, base_measure, fitted_kde.grid, cache_components=True)
    assert ds.has_components
    assert ds.log_qp_table is not None
    assert ds.log_qg_table is not None
    assert ds.log_ql_vector is not None
    n_cells = fitted_kde.grid.n_cells
    n_players = len(ds.vocab)
    assert ds.log_qp_table.shape == (n_players, n_cells)
    assert ds.log_qg_table.shape == (n_players, n_cells)
    assert ds.log_ql_vector.shape == (n_cells,)


def test_dataset_components_match_kde(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    df = _synthetic_shots(n=20)
    ds = ShotCellDataset(df, base_measure, fitted_kde.grid, cache_components=True)
    for pid in ("A", "B"):
        idx = ds.vocab.to_idx(pid)
        assert ds.log_qp_table is not None and ds.log_qg_table is not None
        expected_qp = np.log(fitted_kde.player_density(pid, hierarchical=False)).ravel()
        expected_qg = np.log(fitted_kde.position_density(fitted_kde.player_position[pid])).ravel()
        np.testing.assert_allclose(ds.log_qp_table[idx].numpy(), expected_qp, atol=1e-5)
        np.testing.assert_allclose(ds.log_qg_table[idx].numpy(), expected_qg, atol=1e-5)
    assert ds.log_ql_vector is not None
    expected_ql = np.log(fitted_kde.league_density()).ravel()
    np.testing.assert_allclose(ds.log_ql_vector.numpy(), expected_ql, atol=1e-5)


def test_dataset_components_default_off(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    df = _synthetic_shots(n=20)
    ds = ShotCellDataset(df, base_measure, fitted_kde.grid)
    assert not ds.has_components
    assert ds.log_qp_table is None


def test_dataset_components_require_kde_product(
    fitted_kde: HierarchicalKDE,
) -> None:
    from shotcloud import HierarchicalKDEBase

    base = HierarchicalKDEBase(hierarchical_kde=fitted_kde)
    df = _synthetic_shots(n=20)
    with pytest.raises(ValueError, match="cache_components"):
        ShotCellDataset(df, base, fitted_kde.grid, cache_components=True)


# ---------------------------------------------------------------------------
# Defensive caching (Phase 2)
# ---------------------------------------------------------------------------


def _synthetic_shots_with_opponent(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Synthetic shots with an 'opponent' column (two opponents)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid, mu_y in [("A", 4.0), ("B", 24.0)]:
        for i in range(n):
            rows.append(
                {
                    "x": float(rng.normal(0, 2)),
                    "y": float(rng.normal(mu_y, 2)),
                    "player_id": pid,
                    "opponent": "X" if i % 2 == 0 else "Y",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_defensive_kde() -> DefensiveKDE:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 2, n)
    y = np.concatenate([rng.normal(4, 2, n // 2), rng.normal(20, 2, n - n // 2)])
    opponent = np.array(["X"] * (n // 2) + ["Y"] * (n - n // 2))
    kde = DefensiveKDE(grid=grid, bandwidth=1.5, recency_half_life_days=None)
    kde.fit(x=x, y=y, opponent=opponent)
    return kde


def test_dataset_has_defensive_when_kde_provided(
    fitted_kde: HierarchicalKDE,
    base_measure: KDEProduct,
    fitted_defensive_kde: DefensiveKDE,
) -> None:
    df = _synthetic_shots_with_opponent(n=20)
    ds = ShotCellDataset(df, base_measure, fitted_kde.grid, defensive_kde=fitted_defensive_kde)
    assert ds.has_defensive
    assert ds.log_qd_table is not None
    assert ds.log_qd_table.shape == (2, fitted_kde.grid.n_cells)
    assert ds.opponent_vocab is not None
    assert ds.opponent_vocab.ids == ("X", "Y")


def test_dataset_opponent_idx_aligned_with_data(
    fitted_kde: HierarchicalKDE,
    base_measure: KDEProduct,
    fitted_defensive_kde: DefensiveKDE,
) -> None:
    """opponent_idx[i] must round-trip back to the original opponent label."""
    df = _synthetic_shots_with_opponent(n=20)
    ds = ShotCellDataset(df, base_measure, fitted_kde.grid, defensive_kde=fitted_defensive_kde)
    assert ds.opponent_vocab is not None
    # Spot-check a few rows.
    for i in (0, 5, 17):
        if i >= len(ds):
            continue
        opp_id = ds.opponent_vocab.to_id(int(ds.opponent_idx[i].item()))
        assert opp_id in {"X", "Y"}


def test_dataset_log_qd_table_matches_kde(
    fitted_kde: HierarchicalKDE,
    base_measure: KDEProduct,
    fitted_defensive_kde: DefensiveKDE,
) -> None:
    df = _synthetic_shots_with_opponent(n=20)
    ds = ShotCellDataset(df, base_measure, fitted_kde.grid, defensive_kde=fitted_defensive_kde)
    assert ds.log_qd_table is not None and ds.opponent_vocab is not None
    for oid in ds.opponent_vocab.ids:
        idx = ds.opponent_vocab.to_idx(oid)
        expected = fitted_defensive_kde.log_density(oid).ravel()
        np.testing.assert_allclose(ds.log_qd_table[idx].numpy(), expected, atol=1e-5)


def test_dataset_default_no_defensive(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    df = _synthetic_shots(n=20)
    ds = ShotCellDataset(df, base_measure, fitted_kde.grid)
    assert not ds.has_defensive
    assert ds.log_qd_table is None
    # opponent_idx is still present (sentinel zeros) so __getitem__ shape is invariant.
    assert ds.opponent_idx.shape == (len(ds),)
    assert (ds.opponent_idx == 0).all()


def test_dataset_drops_shots_with_unknown_opponent(
    fitted_kde: HierarchicalKDE,
    base_measure: KDEProduct,
    fitted_defensive_kde: DefensiveKDE,
) -> None:
    df = _synthetic_shots_with_opponent(n=20)
    df_extended = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "x": [0.0],
                    "y": [5.0],
                    "player_id": ["A"],
                    "opponent": ["UNKNOWN"],
                }
            ),
        ],
        ignore_index=True,
    )
    ds = ShotCellDataset(
        df_extended, base_measure, fitted_kde.grid, defensive_kde=fitted_defensive_kde
    )
    assert len(ds) == len(df)


def test_dataset_defensive_requires_opponent_column(
    fitted_kde: HierarchicalKDE,
    base_measure: KDEProduct,
    fitted_defensive_kde: DefensiveKDE,
) -> None:
    df = _synthetic_shots(n=20)  # no opponent column
    with pytest.raises(KeyError, match="opponent"):
        ShotCellDataset(df, base_measure, fitted_kde.grid, defensive_kde=fitted_defensive_kde)


def test_dataset_unfitted_defensive_raises(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    from shotcloud.legacy_pivot.defensive_kde import DefensiveKDE

    unfit = DefensiveKDE(grid=fitted_kde.grid)
    df = _synthetic_shots_with_opponent(n=20)
    with pytest.raises(ValueError, match="must be fit"):
        ShotCellDataset(df, base_measure, fitted_kde.grid, defensive_kde=unfit)


def test_dataset_opponent_vocab_without_defensive_raises(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    from shotcloud import OpponentVocab

    df = _synthetic_shots_with_opponent(n=20)
    vocab = OpponentVocab.from_ids(["X", "Y"])
    with pytest.raises(ValueError, match="opponent_vocab provided without"):
        ShotCellDataset(df, base_measure, fitted_kde.grid, opponent_vocab=vocab)


def test_dataset_getitem_shape_with_defense(
    fitted_kde: HierarchicalKDE,
    base_measure: KDEProduct,
    fitted_defensive_kde: DefensiveKDE,
) -> None:
    df = _synthetic_shots_with_opponent(n=20)
    ds = ShotCellDataset(df, base_measure, fitted_kde.grid, defensive_kde=fitted_defensive_kde)
    _, _, _, opp, _ = ds[0]
    assert isinstance(opp, torch.Tensor) and opp.dtype == torch.int64
    # opp_idx must be 0 or 1 (we have 2 opponents).
    assert int(opp.item()) in {0, 1}


# ---------------------------------------------------------------------------
# Phase 3 — context features
# ---------------------------------------------------------------------------


def _shots_with_full_context(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append(
            {
                "x": float(rng.normal(0, 5)),
                "y": float(rng.normal(15, 5)),
                "player_id": "A" if i % 2 == 0 else "B",
                "opponent": "X" if i % 4 < 2 else "Y",
                "made": int(rng.random() < 0.5),
                "period": (i % 4) + 1,
                "time_remaining_sec": int(60 * (i % 48)),
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(i)),
            }
        )
    return pd.DataFrame(rows)


def test_dataset_no_context_returns_empty_x_n(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    df = _synthetic_shots(n=20)
    ds = ShotCellDataset(df, base_measure, fitted_kde.grid)
    assert not ds.has_context
    assert ds.context_dim == 0
    _, _, _, _, x_n = ds[0]
    assert x_n.shape == (0,)


def test_dataset_with_context_emits_x_n(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    from shotcloud.data import CONTEXT_DIM, ContextEncoder

    df = _shots_with_full_context(n=30)
    enc = ContextEncoder.fit(df)
    ds = ShotCellDataset(df, base_measure, fitted_kde.grid, context_encoder=enc)
    assert ds.has_context
    assert ds.context_dim == CONTEXT_DIM
    _, _, _, _, x_n = ds[0]
    assert x_n.shape == (CONTEXT_DIM,)


def test_dataset_context_aligned_with_filtered_rows(
    fitted_kde: HierarchicalKDE, base_measure: KDEProduct
) -> None:
    """After dropping backcourt + unknown players, x_n must align with surviving rows."""
    from shotcloud.data import ContextEncoder

    df = _shots_with_full_context(n=30)
    enc = ContextEncoder.fit(df)
    # Append an out-of-court shot — it must be dropped from both player_idx and context.
    df_extended = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "x": 0.0,
                        "y": 60.0,  # backcourt
                        "player_id": "A",
                        "opponent": "X",
                        "made": 0,
                        "period": 4,
                        "time_remaining_sec": 0,
                        "date": pd.Timestamp("2024-01-01"),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ds = ShotCellDataset(df_extended, base_measure, fitted_kde.grid, context_encoder=enc)
    # The out-of-court row was dropped — context_features rows must match.
    assert ds.context_features.shape[0] == len(ds)


def test_dataset_with_defense_and_context(
    fitted_kde: HierarchicalKDE,
    base_measure: KDEProduct,
    fitted_defensive_kde: DefensiveKDE,
) -> None:
    from shotcloud.data import CONTEXT_DIM, ContextEncoder

    df = _shots_with_full_context(n=40)
    enc = ContextEncoder.fit(df)
    ds = ShotCellDataset(
        df,
        base_measure,
        fitted_kde.grid,
        defensive_kde=fitted_defensive_kde,
        context_encoder=enc,
    )
    assert ds.has_defensive
    assert ds.has_context
    _, _, _, opp, x_n = ds[0]
    assert x_n.shape == (CONTEXT_DIM,)
    assert int(opp.item()) in {0, 1}
