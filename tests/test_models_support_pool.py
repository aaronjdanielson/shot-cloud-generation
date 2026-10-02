"""Tests for :mod:`shotcloud.models._support_pool`.

:class:`GlobalSupportPool` stores every support shot once and
:class:`PerPlayerSupportIndex` maps each player to rows of the pool. The tests check
that:

* the builder concatenates per-player histories without losing or reordering shots;
* gathered ``(coords, context, dates)`` per player match the source
  :class:`AdaptiveKDE` arrays element for element;
* short-history players get ``-1`` tail entries and players absent from the
  ``AdaptiveKDE`` get an all-``-1`` row;
* malformed pools and indices are rejected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud import CourtGrid
from shotcloud.data import ContextEncoder
from shotcloud.data.context import CONTEXT_DIM
from shotcloud.kde.adaptive import AdaptiveKDE
from shotcloud.models._support_pool import (
    GlobalSupportPool,
    PerPlayerSupportIndex,
    build_support_pool_from_adaptive_kde,
)
from shotcloud.training.dataset import PlayerVocab

# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------


def _fit_adaptive_kde(
    *,
    n_players: int = 4,
    shots_per_player: tuple[int, ...] = (5, 12, 1, 8),
    max_history: int = 20,
    seed: int = 0,
) -> tuple[AdaptiveKDE, pd.DataFrame, ContextEncoder]:
    """Build a tiny AdaptiveKDE with uneven per-player history counts."""
    assert len(shots_per_player) == n_players
    rng = np.random.default_rng(seed)
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=12, ny=10)
    base_date = pd.Timestamp("2024-01-01")
    rows: list[dict[str, object]] = []
    for pid in range(1, n_players + 1):
        n_shots = shots_per_player[pid - 1]
        for i in range(n_shots):
            rows.append(
                {
                    "x": float(rng.normal(0, 5)),
                    "y": float(rng.uniform(0, 25)),
                    "player_id": pid,
                    "opponent": "BOS",
                    "made": int(rng.random() < 0.5),
                    "period": (i % 4) + 1,
                    "time_remaining_sec": int(60 * (i % 48)),
                    "date": base_date + pd.Timedelta(days=i),
                }
            )
    shots = pd.DataFrame(rows)
    enc = ContextEncoder.fit(shots)
    ctx = enc.transform(shots)
    akde = AdaptiveKDE(grid=grid, bandwidth=1.5, max_history=max_history, seed=seed)
    akde.fit(
        x=shots["x"].to_numpy(),
        y=shots["y"].to_numpy(),
        player_id=shots["player_id"].to_numpy(),
        context_features=ctx,
        date=shots["date"].to_numpy(),
    )
    return akde, shots, enc


# ---------------------------------------------------------------------------
# Builder: shape + content
# ---------------------------------------------------------------------------


def test_pool_shapes_match_total_shots_and_context_dim() -> None:
    akde, _shots, _enc = _fit_adaptive_kde()
    vocab = PlayerVocab.from_ids(akde.players)
    pool, idx = build_support_pool_from_adaptive_kde(akde, vocab)

    total = sum(int(akde.context[pid].shape[0]) for pid in akde.players)
    assert pool.coords.shape == (total, 2)
    assert pool.context.shape == (total, CONTEXT_DIM)
    assert pool.dates.shape == (total,)
    assert pool.coords.dtype == torch.float32
    assert pool.context.dtype == torch.float32
    assert pool.dates.dtype == torch.int64

    assert idx.n_players == len(vocab)
    assert idx.max_R == max(int(akde.context[pid].shape[0]) for pid in akde.players)
    assert idx.index.dtype == torch.int64


def test_gathered_per_player_matches_adaptive_kde() -> None:
    """Coords, context and dates gathered through each player's index equal the
    ``AdaptiveKDE`` per-player arrays."""
    akde, _shots, _enc = _fit_adaptive_kde()
    vocab = PlayerVocab.from_ids(akde.players)
    pool, idx = build_support_pool_from_adaptive_kde(akde, vocab)
    real_mask = idx.real_mask()  # (n_players, max_R) bool

    for pid in vocab.ids:
        p_idx = vocab.to_idx(pid)
        n_p = int(akde.context[pid].shape[0])

        row = idx.index[p_idx]  # (max_R,)
        assert int(real_mask[p_idx].sum().item()) == n_p

        if n_p == 0:
            assert (row == -1).all().item()
            continue

        valid_ids = row[real_mask[p_idx]]
        np.testing.assert_array_equal(
            pool.coords[valid_ids].numpy(),
            akde.coords[pid].astype(np.float32),
        )
        np.testing.assert_array_equal(
            pool.context[valid_ids].numpy(),
            akde.context[pid].astype(np.float32),
        )
        np.testing.assert_array_equal(
            pool.dates[valid_ids].numpy(),
            akde.dates[pid].astype(np.int64),
        )


def test_pad_rows_for_short_history_player_have_minus_one_tail() -> None:
    """A player with fewer shots than ``max_R`` has valid pool ids followed by ``-1``."""
    akde, _shots, _enc = _fit_adaptive_kde(shots_per_player=(5, 12, 1, 8))
    vocab = PlayerVocab.from_ids(akde.players)
    _pool, idx = build_support_pool_from_adaptive_kde(akde, vocab)
    max_r = idx.max_R
    # Player "1" has 5 shots; tail (5 .. max_R - 1) must be -1.
    p_idx = vocab.to_idx("1")
    row = idx.index[p_idx]
    n_p = int(akde.context["1"].shape[0])
    assert (row[:n_p] >= 0).all().item()
    assert (row[n_p:max_r] == -1).all().item()


def test_player_missing_from_adaptive_kde_gets_all_minus_one_row() -> None:
    """A vocabulary id absent from the fit gets an all-``-1`` row and no real shots."""
    akde, _shots, _enc = _fit_adaptive_kde()
    extra_ids = [*akde.players, 999]
    vocab = PlayerVocab.from_ids(extra_ids)
    _pool, idx = build_support_pool_from_adaptive_kde(akde, vocab)
    p_idx = vocab.to_idx(999)
    assert (idx.index[p_idx] == -1).all().item()
    assert idx.real_mask()[p_idx].sum().item() == 0


# ---------------------------------------------------------------------------
# Builder: rejection of unfit / malformed input
# ---------------------------------------------------------------------------


def test_rejects_unfit_adaptive_kde() -> None:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=8, ny=8)
    akde = AdaptiveKDE(grid=grid, bandwidth=1.5)
    vocab = PlayerVocab.from_ids(["1"])
    with pytest.raises(ValueError, match="fit"):
        build_support_pool_from_adaptive_kde(akde, vocab)


def test_rejects_empty_vocab() -> None:
    akde, _shots, _enc = _fit_adaptive_kde()
    vocab = PlayerVocab.from_ids([])
    with pytest.raises(ValueError, match="vocab"):
        build_support_pool_from_adaptive_kde(akde, vocab)


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


def test_pool_rejects_misshapen_inputs() -> None:
    coords = torch.zeros(5, 2)
    context = torch.zeros(5, 27)
    dates = torch.zeros(5, dtype=torch.int64)
    # Bad coords shape:
    with pytest.raises(ValueError, match="coords"):
        GlobalSupportPool(coords=torch.zeros(5), context=context, dates=dates)
    # Bad context first dim:
    with pytest.raises(ValueError, match="context"):
        GlobalSupportPool(coords=coords, context=torch.zeros(4, 27), dates=dates)
    # Wrong dates dtype:
    with pytest.raises(ValueError, match="int64"):
        GlobalSupportPool(coords=coords, context=context, dates=torch.zeros(5))


def test_index_rejects_wrong_rank_or_dtype() -> None:
    with pytest.raises(ValueError, match="2-D"):
        PerPlayerSupportIndex(index=torch.zeros(5, dtype=torch.int64))
    with pytest.raises(ValueError, match="int64"):
        PerPlayerSupportIndex(index=torch.zeros(2, 3))


def test_real_mask_matches_nonnegative_entries() -> None:
    idx = PerPlayerSupportIndex(
        index=torch.tensor([[0, 3, -1, -1], [-1, -1, -1, -1], [7, 8, 9, -1]], dtype=torch.int64)
    )
    expected = torch.tensor(
        [[True, True, False, False], [False, False, False, False], [True, True, True, False]]
    )
    assert torch.equal(idx.real_mask(), expected)
