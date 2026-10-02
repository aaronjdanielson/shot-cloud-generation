"""Tests for the fast-validation training utilities.

Covers:

* :meth:`GibbsShotDataset.subset`: deterministic sampling, the rebuilt per-game table,
  length, and batch-tuple shape.
* :func:`train_gibbs` loss weights: with ``lambda_timing=0`` and ``lambda_count=0``
  the timing and count heads receive no gradient and do not move.
* :class:`GibbsTrainHistory`: the ``*_alpha_entropy``, ``*_beta_entropy`` and
  ``*_frac_sub_uniform`` series are finite on the collaborative path, and the
  entropies are ``NaN`` for offensive priors without α/β.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud import CourtGrid
from shotcloud.data import ContextEncoder
from shotcloud.data.context import CONTEXT_DIM
from shotcloud.data.player_traits import build_player_traits_table
from shotcloud.data.role_profile import build_role_profiles
from shotcloud.data.snapshots import build_snapshot_store_from_shots
from shotcloud.kde.adaptive import AdaptiveKDE
from shotcloud.legacy_pivot.adaptive_prior import AdaptiveOffensivePrior
from shotcloud.legacy_pivot.archetypes import ArchetypeDictionary, ArchetypeMixture
from shotcloud.models import ContextMLP, NegBinCountHead, RelevanceScore, TimingSoftmaxHead
from shotcloud.models.analogue_retrieval import build_analogue_cache
from shotcloud.models.collaborative_kde import CollaborativeKDE
from shotcloud.training import GibbsShotDataset, PlayerVocab, train_gibbs

# ---------------------------------------------------------------------------
# Small synthetic setup shared across tests
# ---------------------------------------------------------------------------


def _build_setup(
    n_players: int = 6,
    shots_per_player: int = 35,
    use_collab: bool = False,
    seed: int = 0,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=14, ny=12)
    base_date = pd.Timestamp("2024-01-01")
    rows = []
    for pid in range(1, n_players + 1):
        for i in range(shots_per_player):
            cx, cy = (0.0, 5.0) if pid <= n_players // 2 else (2.0, 8.0)
            rows.append(
                {
                    "x": float(cx + rng.normal(0, 3)),
                    "y": float(cy + rng.normal(0, 3)),
                    "player_id": pid,
                    "opponent": "BOS",
                    "made": int(rng.random() < 0.5),
                    "period": (i % 4) + 1,
                    "time_remaining_sec": int(60 * (i % 48)),
                    "date": base_date + pd.Timedelta(days=i),
                    "game_id": f"g{pid}_{i // 6}",
                }
            )
    shots = pd.DataFrame(rows)

    # An early anchor leaves most shots after it, so they enter the dataset.
    anchors = [np.datetime64("2024-01-05", "D")]
    n_archetypes = 3
    store = build_snapshot_store_from_shots(
        shots,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
        archetype_fit_fn=lambda _sub, _t: np.full(
            (n_archetypes, grid.n_cells), 1.0 / grid.n_cells, dtype=np.float32
        ),
    )
    enc = ContextEncoder.fit(shots)
    ctx = enc.transform(shots)

    akde = AdaptiveKDE(grid=grid, bandwidth=1.5, max_history=30, seed=seed)
    akde.fit(
        x=shots["x"].to_numpy(),
        y=shots["y"].to_numpy(),
        player_id=shots["player_id"].to_numpy(),
        context_features=ctx,
        date=shots["date"].to_numpy(),
    )
    vocab = PlayerVocab.from_ids(akde.players)

    if use_collab:
        # Minimal bio and game-log tables matching the shot frame.
        bio_rows = [
            {
                "player_id": int(pid),
                "display_name": f"Player {pid}",
                "birthdate": pd.Timestamp("1990-01-15") + pd.Timedelta(days=i * 30),
                "height_inches": 72 + i,
                "weight_lbs": 190 + i * 5,
                "position_raw": "Guard" if i < n_players // 2 else "Center",
                "position_group": "SG" if i < n_players // 2 else "C",
                "status": "ok",
            }
            for i, pid in enumerate(akde.players)
        ]
        bio = pd.DataFrame(bio_rows)
        gl = shots[["player_id", "date"]].rename(columns={"date": "game_date"}).copy()
        gl["minutes"] = 25
        gl["fga"] = 8
        gl["fta"] = 3
        gl["tov"] = 1
        traits = build_player_traits_table(
            snapshot_store=store,
            vocab_ids=[int(pid) for pid in vocab.ids],
            bio_df=bio,
            game_logs_df=gl,
        )
        cache = build_analogue_cache(traits, L=3, ensure_self=True)
        offensive_prior: torch.nn.Module = CollaborativeKDE(
            adaptive_kde=akde,
            snapshot_store=store,
            traits_table=traits,
            analogue_cache=cache,
            vocab=vocab,
            grid=grid,
        )
    else:
        archetype_dict = ArchetypeDictionary.from_snapshot_store(store)
        archetype_mix = ArchetypeMixture(n_archetypes=archetype_dict.n_archetypes)
        offensive_prior = AdaptiveOffensivePrior(
            akde,
            store,
            archetype_dict,
            archetype_mix,
            vocab,
            RelevanceScore(),
        )

    count_head = NegBinCountHead()
    timing_head = TimingSoftmaxHead()
    context_mlp = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    train_set = GibbsShotDataset(
        shots_df=shots,
        snapshot_store=store,
        grid=grid,
        player_vocab=vocab,
        opp_vocab=None,
        context_encoder=enc,
    )
    return {
        "offensive_prior": offensive_prior,
        "count_head": count_head,
        "timing_head": timing_head,
        "context_mlp": context_mlp,
        "train_set": train_set,
        "grid": grid,
        "shots": shots,
        "vocab": vocab,
        "store": store,
        "encoder": enc,
    }


# ---------------------------------------------------------------------------
# subset()
# ---------------------------------------------------------------------------


def test_subset_returns_requested_size_and_is_deterministic_under_seed() -> None:
    setup = _build_setup()
    train_set: GibbsShotDataset = setup["train_set"]  # type: ignore[assignment]
    a = train_set.subset(40, seed=0)
    b = train_set.subset(40, seed=0)
    assert a.n_shots == 40
    assert torch.equal(a.cell_idx, b.cell_idx)
    assert torch.equal(a.shot_xy, b.shot_xy)


def test_subset_no_op_when_request_exceeds_dataset() -> None:
    setup = _build_setup()
    train_set: GibbsShotDataset = setup["train_set"]  # type: ignore[assignment]
    n = train_set.n_shots
    same = train_set.subset(n + 100)
    assert same is train_set


def test_subset_batch_tuple_has_same_twelve_fields() -> None:
    setup = _build_setup()
    train_set: GibbsShotDataset = setup["train_set"]  # type: ignore[assignment]
    sub = train_set.subset(20)
    item = sub[0]
    # 12 fields, including the within-game prior-shot sequence and its lengths and the
    # prior-outcome features.
    assert len(item) == 12
    assert sub.shot_xy.shape == (20, 2)
    from shotcloud.data.prior_outcomes import PRIOR_OUTCOME_DIM
    from shotcloud.data.within_game_history import (
        MAX_PRIOR_SHOTS,
        WITHIN_GAME_DIM,
        WITHIN_GAME_SEQ_DIM,
    )

    assert sub.h_within_game.shape == (20, WITHIN_GAME_DIM)
    assert sub.prior_outcome.shape == (20, PRIOR_OUTCOME_DIM)
    assert sub.prior_seq.shape == (20, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)
    assert sub.prior_lengths.shape == (20,)


def test_subset_per_game_table_matches_resampled_rows() -> None:
    """After subsetting, ``per_game.k_obs`` sums to the sampled shot count and
    ``per_game.x_n_raw`` has one row per surviving game."""
    setup = _build_setup(shots_per_player=20)
    train_set: GibbsShotDataset = setup["train_set"]  # type: ignore[assignment]
    sub = train_set.subset(30, seed=0)
    assert int(sub.per_game.k_obs.sum()) == 30
    assert sub.per_game.x_n_raw.shape[0] == sub.n_games
    # Game ids are contiguous in [0, n_games).
    assert int(sub.game_idx.min()) == 0
    assert int(sub.game_idx.max()) == sub.n_games - 1


def test_subset_rejects_zero_or_negative() -> None:
    setup = _build_setup()
    train_set: GibbsShotDataset = setup["train_set"]  # type: ignore[assignment]
    with pytest.raises(ValueError, match="n_shots"):
        train_set.subset(0)
    with pytest.raises(ValueError, match="n_shots"):
        train_set.subset(-5)


# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------


def test_lambda_zero_timing_and_count_freezes_those_heads() -> None:
    """With ``lambda_timing=0`` and ``lambda_count=0`` the timing and count heads receive
    zero gradient and do not move, so only the spatial path is optimized."""
    setup = _build_setup()
    timing_before = [p.detach().clone() for p in setup["timing_head"].parameters()]  # type: ignore[attr-defined]
    count_before = [p.detach().clone() for p in setup["count_head"].parameters()]  # type: ignore[attr-defined]
    train_gibbs(
        offensive_prior=setup["offensive_prior"],  # type: ignore[arg-type]
        count_head=setup["count_head"],  # type: ignore[arg-type]
        timing_head=setup["timing_head"],  # type: ignore[arg-type]
        context_mlp=setup["context_mlp"],  # type: ignore[arg-type]
        train_set=setup["train_set"],  # type: ignore[arg-type]
        grid=setup["grid"],  # type: ignore[arg-type]
        n_epochs=2,
        batch_size=32,
        learning_rate=1e-2,  # large enough that any leaked gradient moves the weights
        lambda_timing=0.0,
        lambda_count=0.0,
        spatial_loss="continuous",
        progress=False,
        restore_best_val=False,
    )
    for before, p in zip(timing_before, setup["timing_head"].parameters(), strict=True):  # type: ignore[attr-defined]
        torch.testing.assert_close(before, p.detach(), atol=1e-7, rtol=0)
    for before, p in zip(count_before, setup["count_head"].parameters(), strict=True):  # type: ignore[attr-defined]
        torch.testing.assert_close(before, p.detach(), atol=1e-7, rtol=0)


# ---------------------------------------------------------------------------
# Per-epoch diagnostics
# ---------------------------------------------------------------------------


def test_collaborative_path_populates_alpha_beta_entropy_and_sub_uniform() -> None:
    setup = _build_setup(use_collab=True)
    history = train_gibbs(
        offensive_prior=setup["offensive_prior"],  # type: ignore[arg-type]
        count_head=setup["count_head"],  # type: ignore[arg-type]
        timing_head=setup["timing_head"],  # type: ignore[arg-type]
        context_mlp=setup["context_mlp"],  # type: ignore[arg-type]
        train_set=setup["train_set"],  # type: ignore[arg-type]
        grid=setup["grid"],  # type: ignore[arg-type]
        n_epochs=1,
        batch_size=32,
        learning_rate=1e-3,
        spatial_loss="continuous",
        progress=False,
        restore_best_val=False,
    )
    h_alpha = history.train_alpha_entropy[0]
    h_beta = history.train_beta_entropy[0]
    sub_u = history.train_frac_sub_uniform[0]
    # The α and β entropies are bounded by log(L) and log(R); the check uses the
    # looser [0, log(100)].
    assert np.isfinite(h_alpha) and 0.0 <= h_alpha <= float(np.log(100))
    assert np.isfinite(h_beta) and 0.0 <= h_beta <= float(np.log(100))
    # frac_sub_uniform is a fraction.
    assert 0.0 <= sub_u <= 1.0


def test_legacy_offensive_prior_leaves_alpha_beta_as_nan() -> None:
    """For an offensive prior without α/β (the grid ``AdaptiveOffensivePrior``),
    ``H(α)`` and ``H(β)`` are NaN while ``frac_sub_uniform`` is still a fraction."""
    setup = _build_setup(use_collab=False)
    history = train_gibbs(
        offensive_prior=setup["offensive_prior"],  # type: ignore[arg-type]
        count_head=setup["count_head"],  # type: ignore[arg-type]
        timing_head=setup["timing_head"],  # type: ignore[arg-type]
        context_mlp=setup["context_mlp"],  # type: ignore[arg-type]
        train_set=setup["train_set"],  # type: ignore[arg-type]
        grid=setup["grid"],  # type: ignore[arg-type]
        n_epochs=1,
        batch_size=32,
        learning_rate=1e-3,
        progress=False,
        restore_best_val=False,
    )
    assert np.isnan(history.train_alpha_entropy[0])
    assert np.isnan(history.train_beta_entropy[0])
    assert 0.0 <= history.train_frac_sub_uniform[0] <= 1.0
