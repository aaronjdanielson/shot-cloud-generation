"""Save / load helpers for trained ``(encoder, decoder, vocab)`` triples.

A checkpoint is a single ``.pt`` file containing the decoder's ``V``
matrix, the encoder's embedding table(s), the player vocabulary, and
the hyperparameters needed to rehydrate compatible module instances.

Encoder kinds supported (`encoder_kind` in the payload):

* ``"PlayerEmbeddingEncoder"`` — original v1, single embedding table.
* ``"PlayerPositionEncoder"`` — position-aware additive variant
  introduced 2026-04-26 to surface position into the neural correction.

Loading is forward-compatible: if a future version saves a different
``encoder_kind``, the loader raises a clear error rather than silently
loading the wrong shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from shotcloud.legacy import PlayerEmbeddingEncoder, PlayerPositionEncoder
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder
from shotcloud.training.dataset import PlayerVocab

EncoderModule = PlayerEmbeddingEncoder | PlayerPositionEncoder


@dataclass
class DecoderCheckpoint:
    """Materialized form of a saved checkpoint."""

    encoder: EncoderModule
    decoder: LowRankTiltDecoder
    vocab: PlayerVocab
    metadata: dict[str, Any]


def save_decoder_checkpoint(
    path: str | Path,
    *,
    encoder: EncoderModule,
    decoder: LowRankTiltDecoder,
    vocab: PlayerVocab,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist trained encoder + decoder + vocab to ``path``.

    Supports both :class:`PlayerEmbeddingEncoder` and
    :class:`PlayerPositionEncoder`. The encoder kind is recorded in the
    payload as ``encoder_kind`` so the loader can dispatch.

    Returns the path written (resolved).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    encoder_kind = type(encoder).__name__

    if isinstance(encoder, PlayerPositionEncoder):
        encoder_payload: dict[str, Any] = {
            "encoder_kind": encoder_kind,
            "player_emb": encoder.player_emb.weight.detach().cpu(),
            "position_emb": encoder.position_emb.weight.detach().cpu(),
            "player_to_position": encoder.player_to_position.detach().cpu(),
            "encoder_config": {
                "n_players": encoder.n_players,
                "n_positions": encoder.n_positions,
                "rank": encoder.rank,
                "zero_init_player": encoder.zero_init_player,
                "zero_init_position": encoder.zero_init_position,
            },
        }
    elif isinstance(encoder, PlayerEmbeddingEncoder):
        encoder_payload = {
            "encoder_kind": encoder_kind,
            "embedding": encoder.embedding.weight.detach().cpu(),
            "encoder_config": {
                "n_players": encoder.n_players,
                "rank": encoder.rank,
                "zero_init": encoder.zero_init,
            },
        }
    else:  # pragma: no cover — defensive
        raise TypeError(f"unsupported encoder type: {encoder_kind}")

    payload: dict[str, Any] = {
        "format_version": 2,
        **encoder_payload,
        "V": decoder.V.detach().cpu(),
        "decoder_config": {
            "n_cells": decoder.n_cells,
            "rank": decoder.rank,
            "zero_init": decoder.zero_init,
        },
        "vocab_id_to_idx": dict(vocab.id_to_idx),
        "vocab_ids": list(vocab.ids),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, out)
    return out


def load_decoder_checkpoint(path: str | Path) -> DecoderCheckpoint:
    """Load a checkpoint written by :func:`save_decoder_checkpoint`.

    Supports format_version 1 (PlayerEmbeddingEncoder only) and
    format_version 2 (either encoder kind, with ``encoder_kind`` field
    dispatching).
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"checkpoint not found: {src}")
    # weights_only=False because the payload includes the vocab dict;
    # callers should only load checkpoints they trust.
    payload = torch.load(src, map_location="cpu", weights_only=False)

    fmt = payload.get("format_version")
    if fmt not in (1, 2):
        raise ValueError(f"unsupported checkpoint format_version: {fmt}")

    enc_cfg = payload["encoder_config"]
    encoder: EncoderModule
    if fmt == 1 or payload.get("encoder_kind") == "PlayerEmbeddingEncoder":
        encoder = PlayerEmbeddingEncoder(
            n_players=enc_cfg["n_players"],
            rank=enc_cfg["rank"],
            zero_init=enc_cfg["zero_init"],
        )
        encoder.embedding.weight.data.copy_(payload["embedding"])
    elif payload.get("encoder_kind") == "PlayerPositionEncoder":
        # player_to_position lives in the buffer; provide it via a stub list
        # then overwrite the buffer to match the saved one exactly.
        encoder = PlayerPositionEncoder(
            n_players=enc_cfg["n_players"],
            n_positions=enc_cfg["n_positions"],
            rank=enc_cfg["rank"],
            player_to_position=[0] * enc_cfg["n_players"],
            zero_init_player=enc_cfg["zero_init_player"],
            zero_init_position=enc_cfg["zero_init_position"],
        )
        encoder.player_emb.weight.data.copy_(payload["player_emb"])
        encoder.position_emb.weight.data.copy_(payload["position_emb"])
        # Replace the placeholder buffer with the saved mapping.
        encoder.player_to_position = payload["player_to_position"].clone()
    else:  # pragma: no cover
        raise ValueError(f"unknown encoder_kind: {payload.get('encoder_kind')!r}")

    dec_cfg = payload["decoder_config"]
    decoder = LowRankTiltDecoder(
        n_cells=dec_cfg["n_cells"],
        rank=dec_cfg["rank"],
        zero_init=dec_cfg["zero_init"],
    )
    decoder.V.data.copy_(payload["V"])

    vocab = PlayerVocab(
        id_to_idx=dict(payload["vocab_id_to_idx"]),
        ids=tuple(payload["vocab_ids"]),
    )

    return DecoderCheckpoint(
        encoder=encoder, decoder=decoder, vocab=vocab, metadata=dict(payload["metadata"])
    )
