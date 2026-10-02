"""Tests for :mod:`shotcloud.training.checkpoint`."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from shotcloud import PlayerVocab
from shotcloud.legacy import PlayerEmbeddingEncoder
from shotcloud.legacy_pivot.checkpoint import (
    DecoderCheckpoint,
    load_decoder_checkpoint,
    save_decoder_checkpoint,
)
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder


def _build_triple() -> tuple[PlayerEmbeddingEncoder, LowRankTiltDecoder, PlayerVocab]:
    encoder = PlayerEmbeddingEncoder(n_players=4, rank=4, zero_init=False)
    decoder = LowRankTiltDecoder(n_cells=120, rank=4, zero_init=True)
    vocab = PlayerVocab.from_ids(["A", "B", "C", "D"])
    return encoder, decoder, vocab


def test_round_trip_preserves_weights(tmp_path: Path) -> None:
    encoder, decoder, vocab = _build_triple()
    # Mutate so the saved values are not the defaults.
    with torch.no_grad():
        encoder.embedding.weight.copy_(torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4))
        decoder.V.copy_(torch.linspace(-1, 1, 120 * 4).reshape(120, 4))

    path = save_decoder_checkpoint(
        tmp_path / "ckpt.pt",
        encoder=encoder,
        decoder=decoder,
        vocab=vocab,
        metadata={"note": "round-trip test"},
    )
    assert path.exists()

    ckpt = load_decoder_checkpoint(path)
    assert isinstance(ckpt, DecoderCheckpoint)

    torch.testing.assert_close(ckpt.encoder.embedding.weight, encoder.embedding.weight)
    torch.testing.assert_close(ckpt.decoder.V, decoder.V)
    assert ckpt.vocab.id_to_idx == vocab.id_to_idx
    assert ckpt.vocab.ids == vocab.ids
    assert ckpt.metadata == {"note": "round-trip test"}


def test_loaded_modules_have_correct_shapes(tmp_path: Path) -> None:
    encoder, decoder, vocab = _build_triple()
    save_decoder_checkpoint(tmp_path / "ckpt.pt", encoder=encoder, decoder=decoder, vocab=vocab)
    ckpt = load_decoder_checkpoint(tmp_path / "ckpt.pt")

    assert ckpt.encoder.n_players == 4
    assert ckpt.encoder.rank == 4
    assert ckpt.decoder.n_cells == 120
    assert ckpt.decoder.rank == 4


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_decoder_checkpoint(tmp_path / "nope.pt")


def test_loaded_encoder_produces_same_u(tmp_path: Path) -> None:
    """Forward pass must match before/after save/load."""
    encoder, decoder, vocab = _build_triple()
    save_decoder_checkpoint(tmp_path / "ckpt.pt", encoder=encoder, decoder=decoder, vocab=vocab)
    ckpt = load_decoder_checkpoint(tmp_path / "ckpt.pt")

    pid = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    torch.testing.assert_close(encoder(pid), ckpt.encoder(pid))


def test_invalid_format_version_raises(tmp_path: Path) -> None:
    """Hand-craft a payload with a bogus format version."""
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 999, "junk": True}, path)
    with pytest.raises(ValueError, match="format_version"):
        load_decoder_checkpoint(path)
