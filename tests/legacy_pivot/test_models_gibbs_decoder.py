"""Tests for :class:`shotcloud.legacy_pivot.gibbs_decoder.ConditionalGibbsDecoder`.

The decoder is a pure composition of four upstream modules:

* :class:`~shotcloud.legacy_pivot.adaptive_prior.AdaptiveOffensivePrior`,
* :class:`~shotcloud.legacy_pivot.adaptive_defensive.AdaptiveDefensiveField`,
* :class:`~shotcloud.models.context_residual.ContextResidualEncoder`,
* :class:`~shotcloud.legacy_pivot.tilt_decoder.LowRankTiltDecoder`.

Load-bearing invariants tested here:

1. **Shape + normalization contract.** ``forward`` returns
   ``(B, n_cells)`` log-probabilities, each row summing to 0 in
   exp space.
2. **Zero-init invariant.** With ``V = 0`` in the tilt decoder, the
   residual term is identically zero and the decoder reduces to
   ``softmax(log q_off + log a_δ)``.
3. **Defensive uniform fallback.** Rows whose opponent has no
   causal history get ``log a_δ = -log C`` (a constant); the
   softmax absorbs the constant, leaving
   ``softmax(log q_off + r_θ)``.
4. **Gradient flow.** All four submodules' parameters receive
   nonzero gradients when ``V`` is nonzero.
5. **Construction validation.** Mismatched ``n_cells`` or rank
   between submodules raises clearly.
6. **Components dataclass round-trip.** ``return_components=True``
   returns a :class:`GibbsDecoderOutputs` whose ``energy_neg`` field
   reconstructs the log-probs exactly via softmax.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.typing import NDArray

from shotcloud import AdaptiveKDE, ContextResidualEncoder, CourtGrid, PlayerVocab, RelevanceScore
from shotcloud.data import ContextEncoder
from shotcloud.data.role_profile import build_role_profiles
from shotcloud.data.snapshots import SnapshotStore, build_snapshot_store_from_shots
from shotcloud.legacy_pivot.adaptive_defensive import AdaptiveDefensiveField
from shotcloud.legacy_pivot.adaptive_prior import AdaptiveOffensivePrior
from shotcloud.legacy_pivot.archetypes import ArchetypeDictionary, ArchetypeMixture
from shotcloud.legacy_pivot.gibbs_decoder import ConditionalGibbsDecoder, GibbsDecoderOutputs
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder
from shotcloud.training.dataset import OpponentVocab


def _build_setup(
    *,
    seed: int = 0,
    n_archetypes: int = 4,
    rank: int = 4,
) -> tuple[
    ConditionalGibbsDecoder,
    SnapshotStore,
    PlayerVocab,
    OpponentVocab,
    NDArray[np.float32],
    pd.DataFrame,
]:
    """Synthetic 3-player × 3-opponent setup, single anchor 2024-04-01.

    All shots fall within Jan-Mar 2024 (well before the anchor) so
    every row has both causal self-history and causal opponent
    history. The archetype surfaces are uniform — the offensive
    prior reduces to a relevance-weighted self-KDE in this control,
    isolating the Gibbs composition logic from archetype dynamics.
    """
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(seed)
    rows = []
    players = (1, 2, 3)
    opponents = ("BOS", "LAL", "GSW")
    for opp_int, opp in enumerate(opponents):
        for pid in players:
            for i in range(40):
                rows.append(
                    {
                        "x": float(rng.normal(0, 5)),
                        "y": float(rng.normal(15, 5)),
                        "player_id": pid,
                        "opponent": opp,
                        "made": int(rng.random() < 0.5),
                        "period": (i % 4) + 1,
                        "time_remaining_sec": int(60 * (i % 48)),
                        "date": pd.Timestamp("2024-01-01")
                        + pd.Timedelta(days=int(i + 5 * opp_int)),
                    }
                )
    df = pd.DataFrame(rows)

    anchor_date = np.datetime64("2024-04-01", "D")
    store = build_snapshot_store_from_shots(
        df,
        [anchor_date],
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
        archetype_fit_fn=lambda sub, _t: np.full(
            (n_archetypes, g.n_cells), 1.0 / g.n_cells, dtype=np.float32
        ),
    )

    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)

    # Offensive AdaptiveKDE: keyed by player_id.
    off_kde = AdaptiveKDE(grid=g, bandwidth=1.5, max_history=40)
    off_kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    p_vocab = PlayerVocab.from_ids(off_kde.players)

    # Defensive AdaptiveKDE: keyed by opponent.
    def_kde = AdaptiveKDE(grid=g, bandwidth=1.5, max_history=40)
    def_kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["opponent"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    o_vocab = OpponentVocab.from_ids(def_kde.players)

    archetype_dict = ArchetypeDictionary.from_snapshot_store(store)
    archetype_mix = ArchetypeMixture(n_archetypes=n_archetypes)

    offensive_prior = AdaptiveOffensivePrior(
        off_kde,
        store,
        archetype_dict,
        archetype_mix,
        p_vocab,
        RelevanceScore(),
        kappa=20.0,
    )
    defensive_field = AdaptiveDefensiveField(def_kde, store, o_vocab, RelevanceScore())
    residual_encoder = ContextResidualEncoder(rank=rank)
    tilt_decoder = LowRankTiltDecoder(n_cells=g.n_cells, rank=rank, zero_init=True)

    decoder = ConditionalGibbsDecoder(
        offensive_prior=offensive_prior,
        defensive_field=defensive_field,
        residual_encoder=residual_encoder,
        tilt_decoder=tilt_decoder,
    )
    return decoder, store, p_vocab, o_vocab, ctx, df


def test_forward_shape_and_normalization() -> None:
    decoder, _store, _pv, _ov, ctx, _df = _build_setup()
    B = 5
    x_n = torch.from_numpy(ctx[:B]).float()
    player_idx = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    opp_idx = torch.tensor([0, 1, 2, 0, 2], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)
    log_probs = decoder(player_idx, opp_idx, snapshot_idx, x_n, x_n)
    assert log_probs.shape == (B, decoder.n_cells)
    np.testing.assert_allclose(
        torch.exp(log_probs).sum(dim=-1).detach().numpy(),
        np.ones(B),
        atol=1e-5,
    )


def test_zero_init_invariant_collapses_to_q_off_plus_a_delta() -> None:
    """With V=0 in the tilt decoder, the residual term is zero and
    the decoder reduces to softmax(log q_off + log a_delta)."""
    decoder, _store, _pv, _ov, ctx, _df = _build_setup()
    # Confirm V is zero (the constructor used zero_init=True).
    assert torch.equal(decoder.tilt_decoder.V, torch.zeros_like(decoder.tilt_decoder.V))

    B = 4
    x_n = torch.from_numpy(ctx[:B]).float()
    player_idx = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    opp_idx = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)

    log_probs, comps = decoder(player_idx, opp_idx, snapshot_idx, x_n, x_n, return_components=True)
    # r_theta is zero everywhere.
    np.testing.assert_allclose(
        comps.r_theta.detach().numpy(), np.zeros_like(comps.r_theta.detach().numpy())
    )
    # log_probs == log_softmax(log_q_off + log_a_delta).
    expected = torch.log_softmax(comps.log_q_off + comps.log_a_delta, dim=-1)
    np.testing.assert_allclose(log_probs.detach().numpy(), expected.detach().numpy(), atol=1e-6)


def test_defensive_uniform_fallback_slides_through_softmax() -> None:
    """When a row has no causal opponent history, log_a_delta is a
    constant (-log C) across cells. The softmax absorbs the constant,
    so the row reduces to softmax(log q_off + r_theta)."""
    decoder, _store, _pv, _ov, ctx, _df = _build_setup()
    g = decoder.offensive_prior.M.shape[0]  # n_cells

    # Build a brand-new SnapshotStore with anchor before all shots so
    # every defensive history row is post-anchor → uniform fallback.
    early_rows = pd.DataFrame(
        [
            {
                "x": 0.0,
                "y": 10.0,
                "player_id": 1,
                "opponent": "BOS",
                "date": pd.Timestamp("2023-01-01"),
            }
        ]
    )
    early_store = build_snapshot_store_from_shots(early_rows, [np.datetime64("2023-01-15", "D")])
    # Construct a fresh decoder against the early-anchor store; this test
    # exercises the composition, not the underlying mask logic.
    early_off = AdaptiveOffensivePrior(
        decoder.offensive_prior.relevance.__class__()  # dead branch; the else arm builds the KDE
        if False
        else _force_relevance_only_setup(decoder, early_store)[0],
        early_store,
        decoder.offensive_prior.archetype_dictionary,
        decoder.offensive_prior.archetype_mixture,
        _force_relevance_only_setup(decoder, early_store)[1],
        RelevanceScore(),
        kappa=20.0,
    )
    early_def = AdaptiveDefensiveField(
        _force_relevance_only_setup(decoder, early_store)[2],
        early_store,
        _force_relevance_only_setup(decoder, early_store)[3],
        RelevanceScore(),
    )
    early_decoder = ConditionalGibbsDecoder(
        offensive_prior=early_off,
        defensive_field=early_def,
        residual_encoder=decoder.residual_encoder,
        tilt_decoder=decoder.tilt_decoder,
    )

    B = 3
    x_n = torch.from_numpy(ctx[:B]).float()
    player_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    opp_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)
    log_probs, comps = early_decoder(
        player_idx, opp_idx, snapshot_idx, x_n, x_n, return_components=True
    )

    # All rows fall back to uniform on the defensive side.
    assert not bool(comps.has_def_history.any())
    expected_const = float(np.log(1.0 / g))
    np.testing.assert_allclose(
        comps.log_a_delta.detach().numpy(),
        np.full_like(comps.log_a_delta.detach().numpy(), expected_const),
        atol=1e-5,
    )
    # The softmax of (log_q_off + const + r_theta) equals the softmax
    # of (log_q_off + r_theta). With V=0 also, that's just softmax(log_q_off).
    expected_log_probs = torch.log_softmax(comps.log_q_off, dim=-1)
    np.testing.assert_allclose(
        log_probs.detach().numpy(), expected_log_probs.detach().numpy(), atol=1e-5
    )


def _force_relevance_only_setup(
    decoder: ConditionalGibbsDecoder, store: SnapshotStore
) -> tuple[AdaptiveKDE, PlayerVocab, AdaptiveKDE, OpponentVocab]:
    """Reconstruct the offensive + defensive AdaptiveKDEs that produced
    `decoder`, swapping in `store` as the new SnapshotStore."""
    # The decoder does not hold its upstream KDEs, so rebuild them from a
    # synthetic dataset generated as in _build_setup. The uniform-fallback
    # test only needs the modules to be constructible against the new store.
    del decoder  # the input is only kept for signature parity
    rng = np.random.default_rng(0)
    rows = []
    for opp_int, opp in enumerate(("BOS", "LAL", "GSW")):
        for pid in (1, 2, 3):
            for i in range(40):
                rows.append(
                    {
                        "x": float(rng.normal(0, 5)),
                        "y": float(rng.normal(15, 5)),
                        "player_id": pid,
                        "opponent": opp,
                        "made": int(rng.random() < 0.5),
                        "period": (i % 4) + 1,
                        "time_remaining_sec": int(60 * (i % 48)),
                        "date": pd.Timestamp("2024-01-01")
                        + pd.Timedelta(days=int(i + 5 * opp_int)),
                    }
                )
    df = pd.DataFrame(rows)
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    off_kde = AdaptiveKDE(grid=grid, bandwidth=1.5, max_history=40)
    off_kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    p_vocab = PlayerVocab.from_ids(off_kde.players)
    def_kde = AdaptiveKDE(grid=grid, bandwidth=1.5, max_history=40)
    def_kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["opponent"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    o_vocab = OpponentVocab.from_ids(def_kde.players)
    return off_kde, p_vocab, def_kde, o_vocab


def test_returns_components_when_requested() -> None:
    decoder, _store, _pv, _ov, ctx, _df = _build_setup()
    B = 3
    x_n = torch.from_numpy(ctx[:B]).float()
    player_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    opp_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)
    log_probs, comps = decoder(player_idx, opp_idx, snapshot_idx, x_n, x_n, return_components=True)
    assert isinstance(comps, GibbsDecoderOutputs)
    assert comps.log_q_off.shape == (B, decoder.n_cells)
    assert comps.log_a_delta.shape == (B, decoder.n_cells)
    assert comps.r_theta.shape == (B, decoder.n_cells)
    assert comps.energy_neg.shape == (B, decoder.n_cells)
    assert comps.omega.shape == (B,)
    assert comps.has_def_history.shape == (B,)
    # energy_neg reconstructs log_probs exactly via softmax.
    expected = torch.log_softmax(comps.energy_neg, dim=-1)
    np.testing.assert_allclose(log_probs.detach().numpy(), expected.detach().numpy(), atol=1e-7)


def test_gradient_flows_through_all_components() -> None:
    """With V nonzero, every submodule's parameters receive a gradient."""
    decoder, _store, _pv, _ov, ctx, _df = _build_setup(rank=4)
    # Wake the tilt decoder up from V=0 so the residual gradient path
    # has nonzero output. Need to mutate `decoder.tilt_decoder.V`
    # directly (it's an nn.Parameter).
    with torch.no_grad():
        decoder.tilt_decoder.V.add_(0.01 * torch.randn_like(decoder.tilt_decoder.V))

    B = 4
    x_n = torch.from_numpy(ctx[:B]).float()
    player_idx = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    opp_idx = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)
    log_probs = decoder(player_idx, opp_idx, snapshot_idx, x_n, x_n)
    loss = -log_probs.mean()
    loss.backward()

    # Gather gradients from each submodule and assert each subsystem
    # received SOME nonzero gradient. (Some individual params can be
    # zero — e.g. the relevance score's input-bias on rows that only
    # touched a few of its named slices — but at least one param per
    # submodule should be nonzero.)
    def _has_nonzero_grad(module: torch.nn.Module) -> bool:
        return any(
            (p.grad is not None and p.grad.abs().sum().item() > 0) for p in module.parameters()
        )

    assert _has_nonzero_grad(decoder.offensive_prior), "offensive_prior got no grad"
    assert _has_nonzero_grad(decoder.defensive_field), "defensive_field got no grad"
    assert _has_nonzero_grad(decoder.residual_encoder), "residual_encoder got no grad"
    assert _has_nonzero_grad(decoder.tilt_decoder), "tilt_decoder got no grad"


def test_n_cells_mismatch_between_submodules_raises() -> None:
    decoder, _store, _pv, _ov, _ctx, _df = _build_setup()
    # Construct a tilt decoder with a different n_cells.
    bad_tilt = LowRankTiltDecoder(
        n_cells=decoder.n_cells + 1,
        rank=decoder.tilt_decoder.rank,
        zero_init=True,
    )
    with pytest.raises(ValueError, match=r"tilt_decoder\.n_cells"):
        ConditionalGibbsDecoder(
            offensive_prior=decoder.offensive_prior,
            defensive_field=decoder.defensive_field,
            residual_encoder=decoder.residual_encoder,
            tilt_decoder=bad_tilt,
        )


def test_rank_mismatch_between_residual_and_tilt_raises() -> None:
    decoder, _store, _pv, _ov, _ctx, _df = _build_setup(rank=4)
    bad_encoder = ContextResidualEncoder(rank=8)  # tilt_decoder.rank == 4
    with pytest.raises(ValueError, match="rank"):
        ConditionalGibbsDecoder(
            offensive_prior=decoder.offensive_prior,
            defensive_field=decoder.defensive_field,
            residual_encoder=bad_encoder,
            tilt_decoder=decoder.tilt_decoder,
        )


def test_decoder_owns_no_parameters_of_its_own() -> None:
    """All learnable state lives on the four submodules."""
    decoder, _store, _pv, _ov, _ctx, _df = _build_setup()
    own_param_ids = {id(p) for name, p in decoder.named_parameters(recurse=False)}
    # No parameters declared directly on the decoder — only via submodules.
    assert own_param_ids == set()
    # Total parameters equals the union of submodule parameters.
    total = sum(p.numel() for p in decoder.parameters())
    assert decoder.defensive_field is not None
    assert decoder.residual_encoder is not None
    assert decoder.tilt_decoder is not None
    submodule_total = sum(
        p.numel()
        for module in (
            decoder.offensive_prior,
            decoder.defensive_field,
            decoder.residual_encoder,
            decoder.tilt_decoder,
        )
        for p in module.parameters()
    )
    assert total == submodule_total


def test_offense_only_mode_equals_log_softmax_q_off() -> None:
    """Offense-only mode: with no defense and no residual, the decoder
    reduces to ``log_softmax(log q_off)``."""
    full, _store, _pv, _ov, ctx, _df = _build_setup()
    offense_only = ConditionalGibbsDecoder(offensive_prior=full.offensive_prior)
    assert not offense_only.has_defense
    assert not offense_only.has_residual

    B = 5
    x_n = torch.from_numpy(ctx[:B]).float()
    player_idx = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    opp_idx = torch.zeros(B, dtype=torch.long)  # ignored in offense-only mode
    snapshot_idx = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        log_p = offense_only(player_idx, opp_idx, snapshot_idx, x_n, x_n)
        log_q_off, _, _ = full.offensive_prior(player_idx, snapshot_idx, x_n, x_n)
        expected = torch.log_softmax(log_q_off, dim=-1)
    torch.testing.assert_close(log_p, expected, atol=1e-6, rtol=0)
    # Normalization sanity check.
    np.testing.assert_allclose(torch.exp(log_p).sum(dim=-1).numpy(), np.ones(B), atol=1e-5)


def test_offense_plus_defense_mode_equals_log_softmax_off_plus_def() -> None:
    """Offense-plus-defense mode: with no residual, the decoder reduces to
    ``log_softmax(log q_off + log a_δ)``."""
    full, _store, _pv, _ov, ctx, _df = _build_setup()
    off_plus_def = ConditionalGibbsDecoder(
        offensive_prior=full.offensive_prior, defensive_field=full.defensive_field
    )
    assert off_plus_def.has_defense
    assert not off_plus_def.has_residual

    B = 4
    x_n = torch.from_numpy(ctx[:B]).float()
    player_idx = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    opp_idx = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        log_p = off_plus_def(player_idx, opp_idx, snapshot_idx, x_n, x_n)
        log_q_off, _, _ = full.offensive_prior(player_idx, snapshot_idx, x_n, x_n)
        log_a_delta, _, _ = full.defensive_field(opp_idx, snapshot_idx, x_n)
        expected = torch.log_softmax(log_q_off + log_a_delta, dim=-1)
    torch.testing.assert_close(log_p, expected, atol=1e-6, rtol=0)


def test_residual_must_be_paired() -> None:
    """``residual_encoder`` and ``tilt_decoder`` must both be provided
    or both omitted; supplying only one is a construction error."""
    full, _store, _pv, _ov, _ctx, _df = _build_setup()
    with pytest.raises(ValueError, match="both provided or both None"):
        ConditionalGibbsDecoder(
            offensive_prior=full.offensive_prior,
            residual_encoder=full.residual_encoder,
            # tilt_decoder missing
        )
    with pytest.raises(ValueError, match="both provided or both None"):
        ConditionalGibbsDecoder(
            offensive_prior=full.offensive_prior,
            tilt_decoder=full.tilt_decoder,
            # residual_encoder missing
        )
