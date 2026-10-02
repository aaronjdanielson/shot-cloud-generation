"""Tests for :class:`shotcloud.legacy.PlayerPositionEncoder` plus its
checkpoint round-trip and end-to-end training behavior.

The encoder lives under :mod:`shotcloud.legacy` post-2026-05-pivot:
the residual tilt is now context-only (paper §3.5), so the player-id
encoders have moved to legacy and are exercised here only via the
legacy ``train_decoder`` flow that they were originally written for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud import CourtGrid, HierarchicalKDE
from shotcloud.legacy import KDEProduct, PlayerPositionEncoder
from shotcloud.legacy_pivot.checkpoint import (
    load_decoder_checkpoint,
    save_decoder_checkpoint,
)
from shotcloud.legacy_pivot.shot_cell_dataset import ShotCellDataset
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder
from shotcloud.legacy_pivot.trainer import train_decoder

# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_construction_default_random_init() -> None:
    enc = PlayerPositionEncoder(n_players=4, n_positions=3, rank=4, player_to_position=[0, 0, 1, 2])
    assert enc.zero_init_player is False
    assert enc.zero_init_position is False
    assert (enc.player_emb.weight != 0).any()
    assert (enc.position_emb.weight != 0).any()
    np.testing.assert_array_equal(enc.player_to_position.numpy(), [0, 0, 1, 2])


def test_zero_init_both_returns_zero_u() -> None:
    """With both factors zero, u is identically zero — the dead-zero
    saddle, but useful as a sanity check / for the q_0 invariant."""
    enc = PlayerPositionEncoder(
        n_players=3,
        n_positions=2,
        rank=4,
        player_to_position=[0, 1, 0],
        zero_init_player=True,
        zero_init_position=True,
    )
    pid = torch.arange(3, dtype=torch.long)
    assert torch.equal(enc(pid), torch.zeros(3, 4))


def test_forward_combines_player_and_position() -> None:
    """u_p must equal player_emb[p] + position_emb[g(p)]."""
    enc = PlayerPositionEncoder(n_players=4, n_positions=3, rank=4, player_to_position=[0, 1, 2, 0])
    pid = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    u = enc(pid)

    expected = enc.player_emb.weight[pid] + enc.position_emb.weight[enc.player_to_position[pid]]
    torch.testing.assert_close(u, expected)


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError, match="n_players"):
        PlayerPositionEncoder(n_players=0, n_positions=3, rank=4, player_to_position=[])
    with pytest.raises(ValueError, match="n_positions"):
        PlayerPositionEncoder(n_players=2, n_positions=0, rank=4, player_to_position=[0, 0])
    with pytest.raises(ValueError, match="rank"):
        PlayerPositionEncoder(n_players=2, n_positions=2, rank=0, player_to_position=[0, 1])
    with pytest.raises(ValueError, match="player_to_position has length"):
        PlayerPositionEncoder(n_players=4, n_positions=2, rank=4, player_to_position=[0, 1, 0])
    with pytest.raises(ValueError, match="out of range"):
        PlayerPositionEncoder(n_players=2, n_positions=2, rank=4, player_to_position=[0, 5])


def test_non_long_dtype_raises() -> None:
    enc = PlayerPositionEncoder(n_players=3, n_positions=2, rank=4, player_to_position=[0, 1, 0])
    with pytest.raises(ValueError, match="integer dtype"):
        enc(torch.tensor([0.0, 1.0]))


def test_2d_input_raises() -> None:
    enc = PlayerPositionEncoder(n_players=3, n_positions=2, rank=4, player_to_position=[0, 1, 0])
    with pytest.raises(ValueError, match="1-D"):
        enc(torch.tensor([[0, 1]], dtype=torch.long))


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_through_both_embeddings() -> None:
    enc = PlayerPositionEncoder(n_players=4, n_positions=2, rank=4, player_to_position=[0, 0, 1, 1])
    pid = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    u = enc(pid)
    u.pow(2).sum().backward()

    assert enc.player_emb.weight.grad is not None
    assert enc.position_emb.weight.grad is not None
    assert (enc.player_emb.weight.grad != 0).any()
    assert (enc.position_emb.weight.grad != 0).any()


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_position_encoder(tmp_path: Path) -> None:
    """Save → load → forward must produce identical u for the same input."""
    encoder = PlayerPositionEncoder(
        n_players=5, n_positions=3, rank=4, player_to_position=[0, 1, 2, 0, 1]
    )
    decoder = LowRankTiltDecoder(n_cells=120, rank=4, zero_init=True)
    from shotcloud import PlayerVocab

    vocab = PlayerVocab.from_ids(["A", "B", "C", "D", "E"])

    # Mutate so the saved values aren't defaults.
    with torch.no_grad():
        encoder.player_emb.weight.copy_(torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4))
        encoder.position_emb.weight.copy_(torch.linspace(-1, 1, 3 * 4).reshape(3, 4))

    save_decoder_checkpoint(
        tmp_path / "ckpt.pt",
        encoder=encoder,
        decoder=decoder,
        vocab=vocab,
        metadata={"kind": "position-aware"},
    )
    ckpt = load_decoder_checkpoint(tmp_path / "ckpt.pt")

    assert isinstance(ckpt.encoder, PlayerPositionEncoder)
    assert ckpt.encoder.n_players == 5
    assert ckpt.encoder.n_positions == 3

    torch.testing.assert_close(ckpt.encoder.player_emb.weight, encoder.player_emb.weight)
    torch.testing.assert_close(ckpt.encoder.position_emb.weight, encoder.position_emb.weight)
    np.testing.assert_array_equal(
        ckpt.encoder.player_to_position.numpy(),
        encoder.player_to_position.numpy(),
    )

    pid = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    torch.testing.assert_close(encoder(pid), ckpt.encoder(pid))

    assert ckpt.metadata == {"kind": "position-aware"}


# ---------------------------------------------------------------------------
# Trainer accepts both encoder kinds
# ---------------------------------------------------------------------------


def test_trainer_accepts_position_encoder() -> None:
    """End-to-end smoke test: train with PlayerPositionEncoder."""
    torch.manual_seed(0)
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(0)
    rows = []
    for pid, mu_y in [("G0", 24.0), ("G1", 24.0), ("F0", 4.0), ("F1", 4.0)]:
        for _ in range(80):
            rows.append(
                {
                    "x": float(rng.normal(0, 4)),
                    "y": float(rng.normal(mu_y, 2)),
                    "player_id": pid,
                }
            )
    df = pd.DataFrame(rows)
    pos_arr = np.array(["G", "G", "F", "F"]).repeat(80)

    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, recency_half_life_days=None)
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        position=pos_arr,
    )
    base = KDEProduct(hierarchical_kde=kde, weights={"a_p": 1.0, "a_g": 0.0, "a_0": 0.0})
    ds = ShotCellDataset(df, base, grid)

    # Map vocab → position via the fitted KDE.
    position_names = sorted(kde.position_density_grid.keys())
    position_to_idx = {n: i for i, n in enumerate(position_names)}
    player_to_position = [position_to_idx[kde.player_position[str(pid)]] for pid in ds.vocab.ids]

    encoder = PlayerPositionEncoder(
        n_players=len(ds.vocab),
        n_positions=len(position_names),
        rank=4,
        player_to_position=player_to_position,
    )
    decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4, zero_init=True)

    hist = train_decoder(encoder, decoder, ds, n_epochs=2, batch_size=32)
    assert len(hist.train_nll) == 2
    # Both embeddings should have updated.
    assert (encoder.player_emb.weight.detach().abs() > 1e-6).any()
    assert (encoder.position_emb.weight.detach().abs() > 1e-6).any()
