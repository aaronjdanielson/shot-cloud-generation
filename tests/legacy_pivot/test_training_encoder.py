"""Tests for :class:`shotcloud.legacy.PlayerEmbeddingEncoder`."""

from __future__ import annotations

import pytest
import torch

from shotcloud.legacy import PlayerEmbeddingEncoder


def test_zero_init_returns_zero_u() -> None:
    enc = PlayerEmbeddingEncoder(n_players=10, rank=4, zero_init=True)
    pid = torch.arange(10, dtype=torch.long)
    u = enc(pid)
    assert torch.equal(u, torch.zeros(10, 4))


def test_random_init_is_nonzero() -> None:
    enc = PlayerEmbeddingEncoder(n_players=10, rank=4, zero_init=False)
    assert (enc.embedding.weight != 0).any()


def test_default_init_is_random_not_zero() -> None:
    """Default ``zero_init=False``: the dead-zero saddle is the gotcha."""
    enc = PlayerEmbeddingEncoder(n_players=10, rank=4)
    assert enc.zero_init is False
    assert (enc.embedding.weight != 0).any()


def test_output_shape() -> None:
    enc = PlayerEmbeddingEncoder(n_players=20, rank=8)
    out = enc(torch.tensor([0, 5, 19], dtype=torch.long))
    assert out.shape == (3, 8)


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError, match="n_players"):
        PlayerEmbeddingEncoder(n_players=0, rank=4)
    with pytest.raises(ValueError, match="rank"):
        PlayerEmbeddingEncoder(n_players=10, rank=0)


def test_non_long_dtype_raises() -> None:
    enc = PlayerEmbeddingEncoder(n_players=5, rank=2)
    with pytest.raises(ValueError, match="integer dtype"):
        enc(torch.tensor([0.0, 1.0]))


def test_2d_input_raises() -> None:
    enc = PlayerEmbeddingEncoder(n_players=5, rank=2)
    with pytest.raises(ValueError, match="1-D"):
        enc(torch.tensor([[0, 1], [2, 3]], dtype=torch.long))


def test_gradients_flow_through_embedding() -> None:
    enc = PlayerEmbeddingEncoder(n_players=5, rank=2, zero_init=False)
    pid = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    u = enc(pid)
    loss = u.pow(2).sum()
    loss.backward()
    assert enc.embedding.weight.grad is not None
    # Players 0–3 should have nonzero grads; player 4 should have zero (not used).
    assert (enc.embedding.weight.grad[:4] != 0).any()
    assert (enc.embedding.weight.grad[4] == 0).all()
