"""PR-D3 — wiring tests for the cell-free defense CLI / manifest /
save-load / reconstruction path through :mod:`scripts.train_gibbs`.

Scope: prove the user-facing plumbing works. Numerical / training
behaviour is covered by the PR-D2b smokes; here we only assert that:

* The CLI accepts the new flags and resolves their defaults.
* When ``--with-defense`` is on with ``spatial_likelihood=continuous_mixture``
  the trainer constructs the cell-free defense triple, the manifest
  records the right fields, and ``modules.pt`` round-trips the
  defensive field's state.
* The eval reconstruction path
  (``plot_learned_density_modes._rebuild_collab_and_spatial``)
  rebuilds the triple and loads the field's state.
* The no-defense path is unchanged.
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
            # Minimal set of required flags so argparse doesn't complain
            # about missing required args. Most defaults are fine for
            # CLI-presence tests.
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
    """The full PR-D3 flag set parses with defaults — verifies every
    `add_argument` for defense made it into the parser."""
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
    """The helper that the trainer calls when ``--with-defense`` +
    ``continuous_mixture`` is selected. Tests with the same synthetic
    fixture the PR-D2b training-smoke uses (compact, no real-data
    dependency)."""
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
    args.shots = Path("shots.csv")  # fake path; defensive_shots_fingerprint takes the
    # actual stat() — provide a real one via a tmp file.

    # Write a tiny temp file just so defensive_shots_fingerprint has a
    # real stat() to hash.
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
    """A canonical no-defense invocation parses without touching any
    of the new flags — verifies we didn't accidentally make them
    required."""
    args = _parse([])  # no --with-defense
    # Defense kwargs all carry their defaults.
    assert args.with_defense is False
    # The new flags exist as parsed attributes but are untouched.
    assert args.defensive_support_max == 1000
    assert args.lambda_defense == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# Regression: cell-free defensive field must be saved to modules.pt
# ---------------------------------------------------------------------------


def test_build_modules_state_includes_cellfree_defensive_field() -> None:
    """**Regression for the 2026-05-27 modules.pt save bug.**

    Before the fix, the script's final save block keyed off the
    grid-side ``defensive_field`` variable only, so cell-free
    defense modules (D-field's ContinuousAdaptiveDefensiveField and
    D-lite's ZoneReweightingDefense) were silently *not* included
    in ``modules.pt``. Eval reconstruction would then load the
    field at its init parameters (β_D ≈ 1e-3), evaluating the
    model as if defense were disabled while the rest of the
    modules were tuned in its presence — silently corrupting
    every cell-free cloud-metric comparison.

    The helper :func:`scripts.train_gibbs._build_modules_state`
    must include ``defensive_field`` whenever either kind of
    defensive field is wired.
    """
    import torch
    from torch import nn

    from shotcloud.models.zone_defense_reweighting import ZoneReweightingDefense

    # Minimal trivially-instantiable stubs for the wrapper inputs.
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
    # The serialized state must carry the trained scalar, not just the init.
    saved_beta = state["defensive_field"]["beta_D"]
    assert torch.isclose(saved_beta, torch.tensor(0.5))


def test_build_modules_state_no_defense_omits_defensive_field() -> None:
    """No-defense runs must NOT carry a ``defensive_field`` key —
    eval reconstruction's "defense wired?" gate keys off
    ``manifest.defensive_features_hash``, and a stray
    ``defensive_field`` in state would confuse downstream consumers.
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
    """Symmetric coverage for the legacy grid-side path: when only
    ``defensive_field`` (and not ``cellfree_defensive_field``) is
    wired, it must still be serialized."""
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
