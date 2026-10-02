"""Tests for the opponent-reweighting wiring in ``scripts/train_gibbs.py``.

Covers the defense CLI flags and their defaults, the builder for the continuous
adaptive defensive field stack (field, allowed-shot cache, features, opponent
vocabulary, and config hashes), and the inclusion of either kind of defensive field
in the serialized ``modules.pt`` state. Training behavior is tested in
``tests/test_training_defense_smoke.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_gibbs as train_gibbs_script  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# CLI flag wiring
# ---------------------------------------------------------------------------


def _parse(argv: list[str]):
    return train_gibbs_script._parse_args(
        [
            # The required flags; every other flag keeps its default.
            "--snapshots",
            "data/snapshots.pt",
            "--shots",
            "shots.csv",
            "--output-dir",
            "/tmp/x",
            "--train-end-date",
            "2024-01-01",
            "--K",
            "4",
            *argv,
        ]
    )


def test_cli_exposes_all_defense_flags() -> None:
    """Every defense flag is registered and parses to its default."""
    args = _parse([])
    # Boolean + scalar defaults.
    assert args.with_defense is False
    assert args.defensive_support_max == 1000
    assert args.defensive_recency_window_days == 365
    assert args.defense_query_chunk_size == 128
    assert args.defense_beta_init == pytest.approx(1e-3)
    assert args.defense_recency_half_life_days == pytest.approx(90.0)
    assert args.lambda_defense == pytest.approx(1e-4)
    assert args.defense_rank == 32
    assert args.defense_hidden_dim == 64
    assert args.defense_opp_embed_dim == 16
    # Path defaults.
    assert args.defensive_cache_dir == Path("data/defensive_retrieval_cache")
    assert args.defensive_cache_rebuild is False


def test_cli_with_defense_flag_sets_true() -> None:
    args = _parse(["--with-defense"])
    assert args.with_defense is True


def test_cli_defense_override_values() -> None:
    args = _parse(
        [
            "--with-defense",
            "--defensive-support-max",
            "256",
            "--defensive-recency-window-days",
            "180",
            "--defense-query-chunk-size",
            "64",
            "--defense-beta-init",
            "0.01",
            "--lambda-defense",
            "1e-3",
            "--defense-rank",
            "16",
            "--defense-hidden-dim",
            "32",
            "--defense-opp-embed-dim",
            "8",
            "--defensive-cache-dir",
            "/tmp/my_def_cache",
        ]
    )
    assert args.defensive_support_max == 256
    assert args.defensive_recency_window_days == 180
    assert args.defense_query_chunk_size == 64
    assert args.defense_beta_init == pytest.approx(0.01)
    assert args.lambda_defense == pytest.approx(1e-3)
    assert args.defense_rank == 16
    assert args.defense_hidden_dim == 32
    assert args.defense_opp_embed_dim == 8
    assert args.defensive_cache_dir == Path("/tmp/my_def_cache")


def test_build_cellfree_defense_stack_returns_triple_and_hashes() -> None:
    """``_build_cellfree_defense_stack`` returns a consistent field, cache, features,
    opponent vocabulary, and 16-character config hashes on a synthetic fixture."""
    import numpy as np
    import pandas as pd

    from shotcloud.data.role_profile import build_role_profiles
    from shotcloud.data.snapshots import build_snapshot_store_from_shots
    from shotcloud.features.defense_features import DEFENSE_FEATURE_DIM
    from shotcloud.models.continuous_adaptive_defensive import (
        ContinuousAdaptiveDefensiveField,
    )
    from shotcloud.models.defensive_retrieval_cache import DefensiveRetrievalCache
    from shotcloud.training.dataset import OpponentVocab

    rng = np.random.default_rng(0)
    base = pd.Timestamp("2024-01-01")
    rows: list[dict[str, object]] = []
    for pid in range(1, 4):
        for i in range(20):
            rows.append(
                {
                    "player_id": pid,
                    "team": "Z",
                    "opponent": ["BOS", "LAL"][i % 2],
                    "x": float(rng.normal(0, 3)),
                    "y": float(rng.normal(0, 3)),
                    "date": base + pd.Timedelta(days=i),
                    "made": int(rng.random() < 0.5),
                    "period": 1,
                    "time_remaining_sec": 0,
                }
            )
    shots = pd.DataFrame(rows)
    store = build_snapshot_store_from_shots(
        shots,
        [pd.Timestamp("2024-01-15").to_datetime64()],
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
    )

    args = _parse(["--with-defense"])
    args.shots = Path("shots.csv")
    # defensive_shots_fingerprint hashes the file's stat(), so point it at a real file.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(b"x,y,player_id,opponent,date\n")
        args.shots = Path(tmp.name)

    field, cache, features, opp_vocab, cache_hash, features_hash = (
        train_gibbs_script._build_cellfree_defense_stack(
            shots=shots,
            shots_path=args.shots,
            snapshot_store=store,
            args=args,
        )
    )
    assert isinstance(field, ContinuousAdaptiveDefensiveField)
    assert isinstance(cache, DefensiveRetrievalCache)
    assert isinstance(opp_vocab, OpponentVocab)
    assert features.features.shape == (2, 1, DEFENSE_FEATURE_DIM)
    # Hashes are non-empty 16-char hex strings.
    assert len(cache_hash) == 16
    assert len(features_hash) == 16
    # Field's defense_feature_dim matches the feature artifact.
    assert field._defense_feature_dim == DEFENSE_FEATURE_DIM
    assert field._n_opponents == len(opp_vocab)


def test_no_defense_path_unchanged_in_cli() -> None:
    """A no-defense invocation parses without any defense flags (none is required)."""
    args = _parse([])  # no --with-defense
    # Defense kwargs all carry their defaults.
    assert args.with_defense is False
    # The defense attributes exist and keep their defaults.
    assert args.defensive_support_max == 1000
    assert args.lambda_defense == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# Serialization of the defensive field in modules.pt
# ---------------------------------------------------------------------------


def test_build_modules_state_includes_cellfree_defensive_field() -> None:
    """``_build_modules_state`` saves a continuous-path defensive field under
    ``defensive_field`` with its trained parameters.

    Without it, evaluation would rebuild the field at its initial ``β_D`` and score
    the model as if defense were disabled.
    """
    import torch
    from torch import nn

    from shotcloud.models.zone_defense_reweighting import ZoneReweightingDefense

    # Stand-in modules for the other inputs.
    op = nn.Linear(1, 1)
    ch = nn.Linear(1, 1)
    th = nn.Linear(1, 1)
    cm = nn.Linear(1, 1)
    cellfree = ZoneReweightingDefense(n_opponents=4, beta_init=0.5)

    state = train_gibbs_script._build_modules_state(
        offensive_prior=op,
        count_head=ch,
        timing_head=th,
        context_mlp=cm,
        defensive_field=None,
        cellfree_defensive_field=cellfree,
        residual_encoder=None,
        tilt_decoder=None,
        location_embedding=None,
        pooling_gate=None,
    )
    assert "defensive_field" in state, (
        "cellfree_defensive_field state must be included in modules.pt under "
        "the 'defensive_field' key"
    )
    # The saved state carries the current β_D, not the initial value.
    saved_beta = state["defensive_field"]["beta_D"]
    assert torch.isclose(saved_beta, torch.tensor(0.5))


def test_build_modules_state_no_defense_omits_defensive_field() -> None:
    """Without a defensive field the state has no ``defensive_field`` key.

    Evaluation decides whether defense is wired from
    ``manifest.defensive_features_hash``; a stray key would contradict it.
    """
    from torch import nn

    state = train_gibbs_script._build_modules_state(
        offensive_prior=nn.Linear(1, 1),
        count_head=nn.Linear(1, 1),
        timing_head=nn.Linear(1, 1),
        context_mlp=nn.Linear(1, 1),
        defensive_field=None,
        cellfree_defensive_field=None,
        residual_encoder=None,
        tilt_decoder=None,
        location_embedding=None,
        pooling_gate=None,
    )
    assert "defensive_field" not in state


def test_build_modules_state_includes_gridside_defensive_field() -> None:
    """A grid-path ``defensive_field`` (with no ``cellfree_defensive_field``) is also
    serialized."""
    from torch import nn

    legacy = nn.Linear(1, 1)
    state = train_gibbs_script._build_modules_state(
        offensive_prior=nn.Linear(1, 1),
        count_head=nn.Linear(1, 1),
        timing_head=nn.Linear(1, 1),
        context_mlp=nn.Linear(1, 1),
        defensive_field=legacy,
        cellfree_defensive_field=None,
        residual_encoder=None,
        tilt_decoder=None,
        location_embedding=None,
        pooling_gate=None,
    )
    assert "defensive_field" in state
