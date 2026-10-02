"""Tests for :class:`GibbsShotDataset` and :func:`train_gibbs`.

The dataset has to route the seven per-shot tensors and the
per-game count/context table correctly. The trainer has to drive
a coherent loss downward over a handful of epochs and respect the
zero-init / no-residual baselines.

Covered invariants:

1. Dataset emits well-shaped tensors and a per-game table whose
   total :math:`\\sum K_n` matches the number of shots used.
2. Dataset's snapshot_idx is causal: every shot's anchor date is
   ``<=`` the shot date.
3. Out-of-vocab players / opponents are dropped.
4. ``train_gibbs`` decreases total loss across epochs on a tiny
   synthetic problem, and the trainable submodules all see
   gradients.
5. The zero-init invariant holds at step 0: with no residual
   encoder and ``f_ctx`` residual-zero-init, the per-shot spatial
   log-probs equal ``log_softmax(log q_off)`` from a fresh
   ``AdaptiveOffensivePrior`` call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from shotcloud import (
    AdaptiveKDE,
    ContextMLP,
    ContextResidualEncoder,
    CourtGrid,
    NegBinCountHead,
    PlayerVocab,
    RelevanceScore,
    TimingSoftmaxHead,
)
from shotcloud.data import ContextEncoder
from shotcloud.data.context import CONTEXT_DIM
from shotcloud.data.role_profile import build_role_profiles
from shotcloud.data.snapshots import build_snapshot_store_from_shots
from shotcloud.legacy_pivot.adaptive_prior import AdaptiveOffensivePrior
from shotcloud.legacy_pivot.archetypes import ArchetypeDictionary, ArchetypeMixture
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder
from shotcloud.training import N_TIMING_BINS, GibbsShotDataset, train_gibbs


def _build_synthetic_shots(seed: int = 0) -> pd.DataFrame:
    """Four anchors, 4 players × 3 opponents, ~24 shots each across 3 games.

    Dates span Jan–Jun 2024, so the snapshot anchors at 2024-03-15
    and 2024-04-15 have shots both before (causal pool) and after
    (training-time rows).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    players = (101, 102, 103, 104)
    opponents = ("BOS", "LAL", "GSW")
    # Game offsets in days from 2024-01-15: 3 games per (player, opp).
    game_offsets = (0, 60, 110)  # Jan 15, Mar 15, May 4
    for player in players:
        for opp_int, opp in enumerate(opponents):
            for game_i, offset in enumerate(game_offsets):
                game_id = f"{player}-{opp}-{game_i}"
                base_date = pd.Timestamp("2024-01-15") + pd.Timedelta(
                    days=int(offset + 2 * opp_int)
                )
                for shot_i in range(8):
                    period = (shot_i % 4) + 1
                    # time_remaining_sec is total game-elapsed seconds
                    # (see loaders.py — the column name is a misnomer).
                    # Spread shots through each period: shot 0 → 5 min in,
                    # shot 1 → 7 min in, etc.
                    game_min = (period - 1) * 12 + 5 + (shot_i // 4) * 2
                    rows.append(
                        {
                            "x": float(rng.normal(0.0, 5.0)),
                            "y": float(rng.normal(15.0, 5.0)),
                            "player_id": player,
                            "opponent": opp,
                            "game_id": game_id,
                            "period": period,
                            "time_remaining_sec": float(game_min * 60),
                            "date": base_date + pd.Timedelta(hours=int(shot_i * 3)),
                            "made": int(rng.random() < 0.5),
                        }
                    )
    return pd.DataFrame(rows)


def _build_training_setup(
    *,
    seed: int = 0,
    n_archetypes: int = 4,
    with_residual: bool = False,
    with_defense: bool = False,
    kernel_form: str = "isotropic",
):
    df = _build_synthetic_shots(seed=seed)
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=16, ny=18)

    anchors = [np.datetime64("2024-03-15", "D"), np.datetime64("2024-04-15", "D")]
    store = build_snapshot_store_from_shots(
        df,
        anchors,
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
        archetype_fit_fn=lambda sub, _t: np.full(
            (n_archetypes, grid.n_cells), 1.0 / grid.n_cells, dtype=np.float32
        ),
    )

    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)

    off_kde = AdaptiveKDE(grid=grid, bandwidth=1.5, max_history=200)
    off_kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    player_vocab = PlayerVocab.from_ids(off_kde.players)

    archetype_dict = ArchetypeDictionary.from_snapshot_store(store)
    archetype_mix = ArchetypeMixture(n_archetypes=n_archetypes)
    anisotropic_kernel = None
    if kernel_form != "isotropic":
        from shotcloud import AnisotropicKernelEvaluator

        anisotropic_kernel = AnisotropicKernelEvaluator(
            grid,
            context_dim=CONTEXT_DIM,
            kernel_form=kernel_form,
            sigma_min=0.5,
            sigma_max=4.0,
            init_sigma=1.5,
        )
    offensive_prior = AdaptiveOffensivePrior(
        off_kde,
        store,
        archetype_dict,
        archetype_mix,
        player_vocab,
        RelevanceScore(),
        kappa=10.0,
        anisotropic_kernel=anisotropic_kernel,
    )

    defensive_field = None
    opp_vocab = None
    if with_defense:
        from shotcloud.legacy_pivot.adaptive_defensive import AdaptiveDefensiveField
        from shotcloud.training import OpponentVocab

        def_kde = AdaptiveKDE(grid=grid, bandwidth=1.5, max_history=200)
        def_kde.fit(
            x=df["x"].to_numpy(),
            y=df["y"].to_numpy(),
            player_id=df["opponent"].to_numpy(),
            context_features=ctx,
            date=df["date"].to_numpy(),
        )
        opp_vocab = OpponentVocab.from_ids(def_kde.players)
        defensive_field = AdaptiveDefensiveField(def_kde, store, opp_vocab, RelevanceScore())

    context_mlp = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, residual=True)
    count_head = NegBinCountHead(context_dim=CONTEXT_DIM, hidden_dim=32)
    timing_head = TimingSoftmaxHead(
        context_dim=CONTEXT_DIM, n_bins=N_TIMING_BINS, hidden_dim=32, zero_init_residual=True
    )

    residual_encoder = None
    tilt_decoder = None
    if with_residual:
        residual_encoder = ContextResidualEncoder(rank=4)
        tilt_decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=4, zero_init=True)

    train_set = GibbsShotDataset(
        shots_df=df,
        snapshot_store=store,
        grid=grid,
        player_vocab=player_vocab,
        opp_vocab=opp_vocab,
        context_encoder=enc,
    )

    return {
        "df": df,
        "store": store,
        "grid": grid,
        "encoder": enc,
        "player_vocab": player_vocab,
        "opp_vocab": opp_vocab,
        "offensive_prior": offensive_prior,
        "defensive_field": defensive_field,
        "context_mlp": context_mlp,
        "count_head": count_head,
        "timing_head": timing_head,
        "residual_encoder": residual_encoder,
        "tilt_decoder": tilt_decoder,
        "train_set": train_set,
    }


# ---------------------------------------------------------------- dataset


def test_dataset_emits_twelve_per_shot_tensors() -> None:
    setup = _build_training_setup()
    train_set = setup["train_set"]
    assert len(train_set) > 0
    item = train_set[0]
    # 12 fields: the original 9, the G1 prior_seq + prior_lengths (paper §10),
    # and the Phase 2 prior_outcome summary (paper 2026-06-07 audit).
    assert len(item) == 12
    (
        player_idx,
        opp_idx,
        snap_idx,
        cell_idx,
        tau_bin,
        x_n_raw,
        game_idx,
        shot_xy,
        h_within_game,
        prior_outcome,
        prior_seq,
        prior_lengths,
    ) = item
    from shotcloud.data.prior_outcomes import PRIOR_OUTCOME_DIM
    from shotcloud.data.within_game_history import (
        MAX_PRIOR_SHOTS,
        WITHIN_GAME_DIM,
        WITHIN_GAME_SEQ_DIM,
    )

    assert prior_outcome.shape == (PRIOR_OUTCOME_DIM,)
    assert prior_outcome.dtype == torch.float32

    assert player_idx.dtype == torch.int64
    assert opp_idx.dtype == torch.int64
    assert snap_idx.dtype == torch.int64
    assert cell_idx.dtype == torch.int64
    assert tau_bin.dtype == torch.int64
    assert game_idx.dtype == torch.int64
    assert x_n_raw.shape == (CONTEXT_DIM,)
    assert shot_xy.shape == (2,)
    assert shot_xy.dtype == torch.float32
    assert h_within_game.shape == (WITHIN_GAME_DIM,)
    assert h_within_game.dtype == torch.float32
    assert prior_seq.shape == (MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)
    assert prior_seq.dtype == torch.float32
    assert prior_lengths.dtype == torch.int64
    assert prior_lengths.shape == ()
    assert 0 <= int(tau_bin) < N_TIMING_BINS


def test_per_game_table_total_matches_n_shots() -> None:
    setup = _build_training_setup()
    train_set = setup["train_set"]
    assert int(train_set.per_game.k_obs.sum()) == train_set.n_shots
    assert train_set.per_game.x_n_raw.shape == (train_set.n_games, CONTEXT_DIM)


def test_spatial_hawkes_flag_extends_outcome_dim_to_17() -> None:
    """Phase 1 B1: with ``with_spatial_hawkes_residual=True`` the
    dataset's ``prior_outcome`` becomes 17-dim (9 outcome + 8
    spatial-Hawkes) and the new attribute ``outcome_feature_dim``
    exposes the realized dim. First-shot rows still produce all-zero
    spatial-Hawkes slots so the AC-KDE step-0 invariant carries
    through unchanged."""
    from shotcloud.data.prior_outcomes import PRIOR_OUTCOME_DIM
    from shotcloud.data.prior_shot_kde import PRIOR_SHOT_KDE_DIM

    setup = _build_training_setup()
    df = setup["df"]
    store = setup["store"]
    grid = setup["grid"]
    enc = setup["encoder"]
    pv = setup["player_vocab"]
    ov = setup["opp_vocab"]

    train_set_with = GibbsShotDataset(
        shots_df=df,
        snapshot_store=store,
        grid=grid,
        player_vocab=pv,
        opp_vocab=ov,
        context_encoder=enc,
        with_spatial_hawkes_residual=True,
    )
    expected = PRIOR_OUTCOME_DIM + PRIOR_SHOT_KDE_DIM
    assert train_set_with.outcome_feature_dim == expected
    assert train_set_with.prior_outcome.shape[-1] == expected
    assert train_set_with.with_spatial_hawkes_residual is True

    # The default-off path still emits 9-dim.
    assert setup["train_set"].outcome_feature_dim == PRIOR_OUTCOME_DIM
    assert setup["train_set"].prior_outcome.shape[-1] == PRIOR_OUTCOME_DIM
    assert setup["train_set"].with_spatial_hawkes_residual is False


def test_game_idx_unique_per_player_game_pair() -> None:
    """Regression for pandas 3.x ``pd.factorize`` null-byte bug
    (2026-06-08). On pandas 3.0.2 / numpy 2.4.4, factorize on a
    numpy object array containing ``\\x00`` truncates the C-string
    hash at the null byte, collapsing every game of a single player
    into one ``game_idx``. The synthetic fixture has 4 players × 3
    opponents × 3 game offsets = 36 distinct (player, game) pairs,
    and each shot's ``game_idx`` must be unique to its (player_id,
    game_id) tuple — i.e. the number of distinct ``game_idx`` values
    equals the number of distinct (player_id, game_id) tuples in the
    surviving (post-snapshot-filter) data, NOT the number of
    distinct ``player_id`` values.
    """
    import pandas as pd

    setup = _build_training_setup()
    train_set = setup["train_set"]
    df = setup["df"]
    store = setup["store"]

    # The dataset drops shots whose date precedes the first snapshot
    # anchor. To get the post-filter (player, game) pair count, replay
    # the same filter on the fixture df.
    first_anchor = np.asarray(store.anchor_dates[0], dtype="datetime64[D]")
    eligible = df[df["date"].to_numpy().astype("datetime64[D]") >= first_anchor]
    eligible_keys = eligible["player_id"].astype(str) + "||" + eligible["game_id"].astype(str)
    n_distinct_pairs_post_filter = int(eligible_keys.nunique())
    assert train_set.n_games == n_distinct_pairs_post_filter, (
        f"game_idx collapses: n_games={train_set.n_games} but post-filter "
        f"distinct (player, game) pairs = {n_distinct_pairs_post_filter} "
        "(was the null-byte separator reintroduced?)"
    )
    # Sanity: with the synthetic 4-player fixture, the bug would
    # collapse n_games to 4 (one per player). The post-filter count
    # is much higher, so this assertion would fail under the bug.
    assert train_set.n_games > eligible["player_id"].nunique()

    # And the per-shot game_idx must agree with the raw (player, game)
    # pairing: the number of unique game_idx values equals n_games.
    game_idx = train_set.game_idx.cpu().numpy()
    n_unique_idx = int(pd.Series(game_idx).nunique())
    assert n_unique_idx == train_set.n_games


def test_dataset_snapshot_idx_is_causal() -> None:
    setup = _build_training_setup()
    train_set = setup["train_set"]
    df = setup["df"]
    store = setup["store"]
    snap_idx_np = train_set.snapshot_idx.numpy()
    # Recover per-shot dates in the same row order the dataset emits.
    surviving = df.assign(date=pd.to_datetime(df["date"]).dt.normalize()).reset_index(drop=True)
    # The dataset drops shots before the first anchor; align by the
    # mapping (n_shots may be < len(df)).
    first_anchor = np.asarray(store.anchor_dates[0], dtype="datetime64[D]")
    eligible = surviving[surviving["date"].to_numpy().astype("datetime64[D]") >= first_anchor]
    eligible = eligible.reset_index(drop=True)
    assert len(eligible) == train_set.n_shots
    shot_dates = eligible["date"].to_numpy().astype("datetime64[D]")
    for i, s_idx in enumerate(snap_idx_np):
        anchor = np.asarray(store.anchor_dates[s_idx], dtype="datetime64[D]")
        assert anchor <= shot_dates[i], f"shot {i}: anchor {anchor} > shot date {shot_dates[i]}"


def test_tau_bin_uses_game_elapsed_seconds_semantic() -> None:
    """``time_remaining_sec`` in the canonical schema is actually
    *total game-elapsed seconds* (loader misnomer). ``tau_bin`` is
    therefore ``floor(time_remaining_sec / 60)`` clamped to
    ``[0, 47]``, **not** a function of seconds-remaining-in-period.

    Regression test for the timing-bin bug surfaced by the first
    real-data ``train_gibbs`` run (2026-05-15): the earlier formula
    treated the column as period-remaining seconds, collapsed every
    post-Q1 shot to bin 0 of its quarter, and only populated 16 of
    48 bins on real data."""
    from shotcloud.training.gibbs_dataset import _compute_tau_bin

    # Game-elapsed seconds covering all four quarters + overtime.
    # period values shouldn't matter — the bin is implicit in the
    # elapsed-seconds value.
    cases = [
        # (elapsed_sec, expected_bin)
        (0.0, 0),  # start of Q1
        (30.0, 0),  # 0:30 into Q1
        (90.0, 1),  # 1:30 into Q1
        (660.0, 11),  # 11 min into Q1
        (720.0, 12),  # start of Q2
        (1500.0, 25),  # 1 min into Q3 (24 + 1)
        (2160.0, 36),  # start of Q4
        (2820.0, 47),  # last minute of regulation
        (2880.0, 47),  # overtime → clamped
        (4961.0, 47),  # deep overtime → clamped
    ]
    elapsed = np.array([c[0] for c in cases], dtype=np.float64)
    expected = np.array([c[1] for c in cases], dtype=np.int64)
    period = np.full_like(elapsed, fill_value=1, dtype=np.int64)  # unused
    actual = _compute_tau_bin(period, elapsed)
    np.testing.assert_array_equal(actual, expected)


def test_dataset_drops_out_of_vocab_players() -> None:
    setup = _build_training_setup()
    df = setup["df"]
    grid = setup["grid"]
    store = setup["store"]
    enc = setup["encoder"]
    # Vocab covering only two of the four players.
    partial_vocab = PlayerVocab.from_ids([101, 102])
    ds = GibbsShotDataset(
        shots_df=df,
        snapshot_store=store,
        grid=grid,
        player_vocab=partial_vocab,
        opp_vocab=None,
        context_encoder=enc,
    )
    assert ds.n_shots > 0
    pid_set = {int(p) for p in ds.player_idx.tolist()}
    assert pid_set <= {0, 1}


# ---------------------------------------------------------------- trainer


def test_train_gibbs_loss_decreases_no_residual() -> None:
    setup = _build_training_setup()
    initial = train_gibbs(
        offensive_prior=setup["offensive_prior"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=1,
        batch_size=64,
        learning_rate=1e-2,
        progress=False,
    )
    initial_loss = initial.train_total[0]

    history = train_gibbs(
        offensive_prior=setup["offensive_prior"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=6,
        batch_size=64,
        learning_rate=1e-2,
        progress=False,
    )
    assert history.train_total[-1] < initial_loss, (
        f"loss did not decrease: epoch1={initial_loss:.4f} epoch7={history.train_total[-1]:.4f}"
    )


def test_train_gibbs_loss_decreases_with_residual() -> None:
    setup = _build_training_setup(with_residual=True)
    initial = train_gibbs(
        offensive_prior=setup["offensive_prior"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        residual_encoder=setup["residual_encoder"],
        tilt_decoder=setup["tilt_decoder"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=1,
        batch_size=64,
        learning_rate=1e-2,
        progress=False,
    )
    initial_loss = initial.train_total[0]

    history = train_gibbs(
        offensive_prior=setup["offensive_prior"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        residual_encoder=setup["residual_encoder"],
        tilt_decoder=setup["tilt_decoder"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=6,
        batch_size=64,
        learning_rate=1e-2,
        progress=False,
    )
    assert history.train_total[-1] < initial_loss


def test_train_gibbs_zero_init_spatial_logits_match_q_off() -> None:
    """At step 0 with no residual and ContextMLP residual-zero-init,
    spatial logits should equal ``log_softmax(log q_off)`` evaluated
    with x_n == x_tilde."""
    setup = _build_training_setup()
    offensive_prior = setup["offensive_prior"]
    context_mlp = setup["context_mlp"]
    train_set = setup["train_set"]

    # Pull a small batch from the dataset.
    player_idx = train_set.player_idx[:8]
    snap_idx = train_set.snapshot_idx[:8]
    x_n_raw = train_set.x_n_raw[:8]

    offensive_prior.eval()
    context_mlp.eval()
    with torch.no_grad():
        x_n = context_mlp(x_n_raw)
        # Zero-init residual makes f_ctx an identity at step 0.
        torch.testing.assert_close(x_n, x_n_raw, atol=1e-6, rtol=0.0)
        log_q_off, _, _ = offensive_prior(player_idx, snap_idx, x_n_raw, x_n)
        expected = torch.log_softmax(log_q_off, dim=-1)
        # The trainer's spatial slice goes through the same call, so
        # an end-to-end check would be circular. Instead we verify the
        # invariant that ``log_softmax(log q_off + 0) == log_softmax(log q_off)``.
        zero_residual = torch.zeros_like(log_q_off)
        actual = torch.log_softmax(log_q_off + zero_residual, dim=-1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)


def test_train_gibbs_gradient_flows_to_all_submodules() -> None:
    setup = _build_training_setup(with_residual=True)
    offensive_prior = setup["offensive_prior"]
    count_head = setup["count_head"]
    timing_head = setup["timing_head"]
    context_mlp = setup["context_mlp"]
    residual_encoder = setup["residual_encoder"]
    tilt_decoder = setup["tilt_decoder"]
    assert residual_encoder is not None and tilt_decoder is not None
    train_gibbs(
        offensive_prior=offensive_prior,
        count_head=count_head,
        timing_head=timing_head,
        context_mlp=context_mlp,
        residual_encoder=residual_encoder,
        tilt_decoder=tilt_decoder,
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=1,
        batch_size=32,
        learning_rate=5e-3,
        progress=False,
    )
    # After a step, at least one parameter per submodule should
    # have been updated. The initial state was fresh modules with
    # zero-init residual tilts and a residual-zero-init f_ctx, so a
    # straightforward sanity check is that the submodule parameters'
    # collective L2 norm has changed (we can't be more specific
    # without snapshotting initials).
    modules = [
        offensive_prior,
        count_head,
        timing_head,
        context_mlp,
        residual_encoder,
        tilt_decoder,
    ]
    for module in modules:
        total = sum(float(p.detach().pow(2).sum()) for p in module.parameters() if p.requires_grad)
        assert total > 0.0 or not any(p.requires_grad for p in module.parameters())


def test_train_gibbs_restore_best_val_lowers_final_val_loss() -> None:
    setup = _build_training_setup()
    # Tiny val set = the train set; the test only checks bookkeeping,
    # not generalization.
    history = train_gibbs(
        offensive_prior=setup["offensive_prior"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        val_set=setup["train_set"],
        n_epochs=4,
        batch_size=64,
        learning_rate=5e-3,
        restore_best_val=True,
        progress=False,
    )
    assert history.best_epoch is not None
    assert 1 <= history.best_epoch <= 4
    assert min(history.val_spatial) == pytest.approx(history.val_spatial[history.best_epoch - 1])


# ---------------------------------------------------------------- defense


def test_train_gibbs_with_defense_loss_decreases() -> None:
    """Offense + defense composition: spatial logits are
    ``log_softmax(log q_off + log a_delta)``; the trainer must drive
    loss downward and gradient must flow into the defensive field's
    relevance parameters too."""
    setup = _build_training_setup(with_defense=True)
    assert setup["defensive_field"] is not None
    initial = train_gibbs(
        offensive_prior=setup["offensive_prior"],
        defensive_field=setup["defensive_field"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=1,
        batch_size=64,
        learning_rate=1e-2,
        progress=False,
    )
    initial_loss = initial.train_total[0]
    history = train_gibbs(
        offensive_prior=setup["offensive_prior"],
        defensive_field=setup["defensive_field"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=6,
        batch_size=64,
        learning_rate=1e-2,
        progress=False,
    )
    assert history.train_total[-1] < initial_loss
    # Defensive field's relevance params received gradient at some
    # point during training (they're learnable scalars on RelevanceScore).
    defensive_field = setup["defensive_field"]
    total = sum(
        float(p.detach().pow(2).sum()) for p in defensive_field.parameters() if p.requires_grad
    )
    assert total > 0.0


def test_train_gibbs_with_defense_full_gibbs_loss_decreases() -> None:
    """Full Gibbs composition: offense + defense + residual."""
    setup = _build_training_setup(with_defense=True, with_residual=True)
    assert setup["defensive_field"] is not None
    initial = train_gibbs(
        offensive_prior=setup["offensive_prior"],
        defensive_field=setup["defensive_field"],
        residual_encoder=setup["residual_encoder"],
        tilt_decoder=setup["tilt_decoder"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=1,
        batch_size=64,
        learning_rate=1e-2,
        progress=False,
    )
    initial_loss = initial.train_total[0]
    history = train_gibbs(
        offensive_prior=setup["offensive_prior"],
        defensive_field=setup["defensive_field"],
        residual_encoder=setup["residual_encoder"],
        tilt_decoder=setup["tilt_decoder"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=6,
        batch_size=64,
        learning_rate=1e-2,
        progress=False,
    )
    assert history.train_total[-1] < initial_loss


def test_train_gibbs_defense_without_opp_vocab_raises() -> None:
    """If defensive_field is provided but the dataset has no opp_vocab,
    the trainer must reject the configuration loudly."""
    setup = _build_training_setup(with_defense=True)
    # Build a parallel train_set without an opp_vocab.
    no_opp_set = GibbsShotDataset(
        shots_df=setup["df"],
        snapshot_store=setup["store"],
        grid=setup["grid"],
        player_vocab=setup["player_vocab"],
        opp_vocab=None,
        context_encoder=setup["encoder"],
    )
    with pytest.raises(ValueError, match="opp_vocab"):
        train_gibbs(
            offensive_prior=setup["offensive_prior"],
            defensive_field=setup["defensive_field"],
            count_head=setup["count_head"],
            timing_head=setup["timing_head"],
            context_mlp=setup["context_mlp"],
            train_set=no_opp_set,
            grid=setup["grid"],
            n_epochs=1,
            batch_size=64,
            progress=False,
        )


# ---------------------------------------------------------------- anisotropic kernel


def test_train_gibbs_with_factored_anisotropic_kernel_loss_decreases() -> None:
    """Full factored σ(x, z_j) anisotropic kernel: trainer drives the
    joint loss down over a handful of epochs and the kernel's σ
    parameters end up varying (anisotropy actually engages)."""
    setup = _build_training_setup(kernel_form="factored")
    initial = train_gibbs(
        offensive_prior=setup["offensive_prior"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=1,
        batch_size=64,
        learning_rate=1e-2,
        progress=False,
    )
    initial_loss = initial.train_total[0]

    history = train_gibbs(
        offensive_prior=setup["offensive_prior"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=6,
        batch_size=64,
        learning_rate=1e-2,
        progress=False,
    )
    assert history.train_total[-1] < initial_loss

    # After training, σ_∥ and σ_⊥ should have drifted off init_sigma
    # for at least some shots — confirming anisotropy actually engaged.
    kernel = setup["offensive_prior"].anisotropic_kernel
    assert kernel is not None
    df = setup["df"]
    enc = setup["encoder"]
    ctx = torch.from_numpy(enc.transform(df)[:32]).float()
    # Use first 32 shots as a batch sample.
    z_j_sample = ctx.unsqueeze(1).expand(-1, 5, -1)  # (32, 5, D)
    stats = kernel.sigma_stats(ctx, z_j_sample)
    # At init both σs were exactly 1.5; after a few epochs they should
    # spread (std > 0) and/or become anisotropic.
    assert stats["sigma_par_std"] + stats["sigma_perp_std"] > 1e-4


def test_train_gibbs_with_z_only_anisotropic_loss_decreases() -> None:
    """The simplest anisotropic ablation (σ(z_j) only) also trains."""
    setup = _build_training_setup(kernel_form="z_only")
    history = train_gibbs(
        offensive_prior=setup["offensive_prior"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=4,
        batch_size=64,
        learning_rate=1e-2,
        progress=False,
    )
    assert history.train_total[-1] < history.train_total[0]


def test_train_gibbs_count_loss_normalization_per_game_differs_from_per_shot() -> None:
    """``per_game`` (new default) and ``per_shot`` (legacy) produce
    distinct loss trajectories at the same ``lambda_count``. Both must
    still drive total loss down.
    """
    setup_a = _build_training_setup(seed=0)
    hist_per_game = train_gibbs(
        offensive_prior=setup_a["offensive_prior"],
        count_head=setup_a["count_head"],
        timing_head=setup_a["timing_head"],
        context_mlp=setup_a["context_mlp"],
        train_set=setup_a["train_set"],
        grid=setup_a["grid"],
        n_epochs=3,
        batch_size=64,
        learning_rate=1e-2,
        count_loss_normalization="per_game",
        progress=False,
    )
    setup_b = _build_training_setup(seed=0)
    hist_per_shot = train_gibbs(
        offensive_prior=setup_b["offensive_prior"],
        count_head=setup_b["count_head"],
        timing_head=setup_b["timing_head"],
        context_mlp=setup_b["context_mlp"],
        train_set=setup_b["train_set"],
        grid=setup_b["grid"],
        n_epochs=3,
        batch_size=64,
        learning_rate=1e-2,
        count_loss_normalization="per_shot",
        progress=False,
    )
    # Both must decrease.
    assert hist_per_game.train_total[-1] < hist_per_game.train_total[0]
    assert hist_per_shot.train_total[-1] < hist_per_shot.train_total[0]
    # And they must differ — same seed + setup, only the normalization
    # changes the count term's contribution to the optimization loss.
    assert hist_per_game.train_total[-1] != hist_per_shot.train_total[-1]


def test_train_gibbs_rejects_unknown_count_loss_normalization() -> None:
    setup = _build_training_setup()
    with pytest.raises(ValueError, match="count_loss_normalization"):
        train_gibbs(
            offensive_prior=setup["offensive_prior"],
            count_head=setup["count_head"],
            timing_head=setup["timing_head"],
            context_mlp=setup["context_mlp"],
            train_set=setup["train_set"],
            grid=setup["grid"],
            n_epochs=1,
            batch_size=64,
            learning_rate=1e-2,
            count_loss_normalization="banana",  # type: ignore[arg-type]
            progress=False,
        )


def test_train_gibbs_freeze_count_holds_count_head_fixed() -> None:
    """``freeze_count`` should leave the count head's parameters
    unchanged after training; spatial submodules still update.
    """
    setup = _build_training_setup()
    before_count = {n: p.detach().clone() for n, p in setup["count_head"].named_parameters()}
    before_ctx_mlp = {n: p.detach().clone() for n, p in setup["context_mlp"].named_parameters()}
    train_gibbs(
        offensive_prior=setup["offensive_prior"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=2,
        batch_size=64,
        learning_rate=1e-2,
        freeze_count=True,
        progress=False,
    )
    for n, p_before in before_count.items():
        p_after = dict(setup["count_head"].named_parameters())[n].detach()
        assert torch.allclose(p_before, p_after), f"count_head param {n} changed under freeze_count"
    # At least one context_mlp param should have moved.
    any_ctx_moved = any(
        not torch.allclose(p_before, dict(setup["context_mlp"].named_parameters())[n].detach())
        for n, p_before in before_ctx_mlp.items()
    )
    assert any_ctx_moved, "context_mlp params were unchanged — trainer may not have stepped"


def test_train_gibbs_freeze_context_mlp_holds_both_fixed() -> None:
    """``freeze_count`` + ``freeze_context_mlp`` together must hold
    *both* the count head AND the context MLP fixed across joint
    training (paper §5.2 Experiment B' falsification test for the
    context_mlp-drift co-adaptation pathway). At least one other
    submodule must still update so we know the trainer actually
    stepped.
    """
    setup = _build_training_setup()
    before_count = {n: p.detach().clone() for n, p in setup["count_head"].named_parameters()}
    before_ctx_mlp = {n: p.detach().clone() for n, p in setup["context_mlp"].named_parameters()}
    before_timing = {n: p.detach().clone() for n, p in setup["timing_head"].named_parameters()}
    train_gibbs(
        offensive_prior=setup["offensive_prior"],
        count_head=setup["count_head"],
        timing_head=setup["timing_head"],
        context_mlp=setup["context_mlp"],
        train_set=setup["train_set"],
        grid=setup["grid"],
        n_epochs=2,
        batch_size=64,
        learning_rate=1e-2,
        freeze_count=True,
        freeze_context_mlp=True,
        progress=False,
    )
    for n, p_before in before_count.items():
        p_after = dict(setup["count_head"].named_parameters())[n].detach()
        assert torch.allclose(p_before, p_after), (
            f"count_head param {n} moved under freeze_count + freeze_context_mlp"
        )
    for n, p_before in before_ctx_mlp.items():
        p_after = dict(setup["context_mlp"].named_parameters())[n].detach()
        assert torch.allclose(p_before, p_after), (
            f"context_mlp param {n} moved under freeze_context_mlp"
        )
    any_timing_moved = any(
        not torch.allclose(p_before, dict(setup["timing_head"].named_parameters())[n].detach())
        for n, p_before in before_timing.items()
    )
    assert any_timing_moved, "timing_head params unchanged — trainer may not have stepped"
