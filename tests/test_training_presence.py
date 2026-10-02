"""Tests for :func:`shotcloud.training.train_presence_only`.

Phase 3.4 PR-P1 of the 2026-06-07 audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud.models.presence import PRESENCE_N_BINS, PresenceModel
from shotcloud.training import train_presence_only


def _make_synthetic_table(
    n_players: int = 10,
    n_games_per_player: int = 8,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[int, int]]:
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(100, 100 + n_players):
        for g in range(n_games_per_player):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=g * 3)
            starter = int(pid % 2 == 0)
            # Make starter and bench have distinct on-court patterns so
            # the model has signal to learn.
            base = np.zeros(PRESENCE_N_BINS, dtype=np.float32)
            if starter:
                base[:6] = 1.0  # on for opening Q1
                base[20:24] = 1.0  # late Q4
            else:
                base[6:18] = 0.6  # mid-game minutes
            base = np.clip(base + 0.05 * rng.standard_normal(PRESENCE_N_BINS), 0.0, 1.0)
            row = {
                "game_id": f"g{pid}-{g}",
                "game_date": date,
                "player_id": pid,
                "starter": starter,
                **{f"bin_{b}": float(base[b]) for b in range(PRESENCE_N_BINS)},
            }
            rows.append(row)
    table = pd.DataFrame(rows)
    player_to_position = {100 + i: i % 3 for i in range(n_players)}
    return table, player_to_position


def test_train_presence_only_reduces_bce() -> None:
    table, p2p = _make_synthetic_table()
    presence = PresenceModel()
    hist = train_presence_only(
        presence_model=presence,
        table=table,
        player_to_position=p2p,
        n_epochs=15,
        batch_size=32,
        learning_rate=5e-2,
    )
    assert len(hist.train_bce) == 15
    assert hist.train_bce[-1] < hist.train_bce[0]


def test_train_presence_only_handles_val_table() -> None:
    table, p2p = _make_synthetic_table(n_games_per_player=6)
    presence = PresenceModel()
    train_table = table[: int(0.7 * len(table))]
    val_table = table[int(0.7 * len(table)) :]
    hist = train_presence_only(
        presence_model=presence,
        table=train_table,
        player_to_position=p2p,
        val_table=val_table,
        n_epochs=3,
        batch_size=32,
        learning_rate=5e-2,
    )
    assert len(hist.val_bce) == 3
    for v in hist.val_bce:
        assert np.isfinite(v)


def test_train_presence_only_runs_without_val() -> None:
    table, p2p = _make_synthetic_table()
    presence = PresenceModel()
    hist = train_presence_only(
        presence_model=presence,
        table=table,
        player_to_position=p2p,
        n_epochs=2,
        batch_size=16,
        learning_rate=5e-2,
    )
    assert hist.val_bce == []
    assert len(hist.train_bce) == 2


def test_train_presence_only_requires_n_bins_attr() -> None:
    table, p2p = _make_synthetic_table()
    bogus = torch.nn.Linear(2, 2)  # no n_bins attr
    with pytest.raises(ValueError, match="n_bins"):
        train_presence_only(
            presence_model=bogus,
            table=table,
            player_to_position=p2p,
            n_epochs=1,
            batch_size=8,
            learning_rate=1e-2,
        )
