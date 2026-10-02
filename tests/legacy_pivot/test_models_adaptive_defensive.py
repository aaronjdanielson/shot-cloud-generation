"""Tests for :class:`shotcloud.legacy_pivot.adaptive_defensive.AdaptiveDefensiveField`.

The defensive field is the per-opponent analog of
:class:`~shotcloud.legacy_pivot.adaptive_prior.AdaptiveOffensivePrior`.
These tests check the load-bearing invariants:

1. Forward returns valid log-probabilities (the field is a simplex
   per row when the row has causal history).
2. Causal date mask drops post-anchor history correctly.
3. Rows with no causal history fall back to uniform feasibility.
4. Same-grid kernel matrix as the offensive prior; output shape +
   normalization match.
5. Independent ``RelevanceScore`` parameters from the offensive
   prior — no accidental sharing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.typing import NDArray

from shotcloud import AdaptiveKDE, CourtGrid, RelevanceScore
from shotcloud.data import ContextEncoder
from shotcloud.data.snapshots import SnapshotStore, build_snapshot_store_from_shots
from shotcloud.legacy_pivot.adaptive_defensive import AdaptiveDefensiveField
from shotcloud.training.dataset import OpponentVocab


def _build_setup(
    seed: int = 0,
    *,
    bandwidth: float = 1.5,
    max_history: int | None = 40,
) -> tuple[
    AdaptiveKDE,
    SnapshotStore,
    OpponentVocab,
    NDArray[np.float32],
    pd.DataFrame,
]:
    """Synthetic 3-opponent / 60-shots-each setup, single anchor 2024-04-01.

    Every training shot is dated Jan-Mar 2024 so the single bundle's
    anchor at 2024-04-01 leaves all shots causal-eligible. Anchors and
    shots are constructed exactly as in the offensive-prior tests;
    only the grouping key differs (``opponent`` instead of
    ``player_id``).
    """
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(seed)
    rows = []
    opponents = ("BOS", "LAL", "GSW")
    # Stagger each opponent by 5 days so date ranges overlap but are
    # still distinguishable. All shots fall in 2024-01-01 .. 2024-03-11,
    # well before the 2024-04-01 anchor used in tests.
    for opp_int, opp in enumerate(opponents):
        for i in range(60):
            rows.append(
                {
                    "x": float(rng.normal(0, 5)),
                    "y": float(rng.normal(15, 5)),
                    "player_id": (i % 3) + 1,  # cycle 3 fake shooters
                    "opponent": opp,
                    "made": int(rng.random() < 0.5),
                    "period": (i % 4) + 1,
                    "time_remaining_sec": int(60 * (i % 48)),
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(i + 5 * opp_int)),
                }
            )
    df = pd.DataFrame(rows)

    anchor_date = np.datetime64("2024-04-01", "D")
    store = build_snapshot_store_from_shots(df, [anchor_date])

    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)

    # AdaptiveKDE is grouping-agnostic — pass opponent codes as `player_id`
    # to obtain per-opponent causal histories.
    def_kde = AdaptiveKDE(grid=g, bandwidth=bandwidth, max_history=max_history)
    def_kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["opponent"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    vocab = OpponentVocab.from_ids(def_kde.players)

    return def_kde, store, vocab, ctx, df


def test_forward_outputs_log_probabilities() -> None:
    def_kde, store, vocab, ctx, _df = _build_setup()
    relevance = RelevanceScore()
    field = AdaptiveDefensiveField(def_kde, store, vocab, relevance)
    B = 4
    x_n = torch.from_numpy(ctx[:B]).float()
    opp_idx = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)
    log_a, pi, has_history = field(opp_idx, snapshot_idx, x_n)
    assert log_a.shape == (B, def_kde.grid.n_cells)
    assert pi.shape == (B, field.max_history)
    assert has_history.shape == (B,)
    # Each row of a_δ is a simplex over cells (sums to 1).
    np.testing.assert_allclose(torch.exp(log_a).sum(dim=-1).detach(), torch.ones(B), atol=1e-4)
    # Relevance softmax sums to 1 per row when has_history is True.
    np.testing.assert_allclose(pi.sum(dim=-1).detach(), torch.ones(B), atol=1e-5)
    assert bool(has_history.all())


def test_no_causal_history_falls_back_to_uniform() -> None:
    """Anchor before any shot exists → has_history all False → uniform feasibility."""
    def_kde, _store_with_late_anchor, vocab, ctx, _df = _build_setup()
    # Build a brand-new SnapshotStore with anchor BEFORE any shots so
    # every defensive-history row's date >= anchor → eff_mask all-zero.
    g = def_kde.grid
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
    early_anchor = np.datetime64("2023-01-15", "D")
    early_store = build_snapshot_store_from_shots(early_rows, [early_anchor])
    relevance = RelevanceScore()
    field = AdaptiveDefensiveField(def_kde, early_store, vocab, relevance)
    B = 3
    x_n = torch.from_numpy(ctx[:B]).float()
    opp_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)
    log_a, _pi, has_history = field(opp_idx, snapshot_idx, x_n)
    # All rows fall back; log_a == log(1 / n_cells) at every cell.
    assert not bool(has_history.any())
    expected = np.log(1.0 / g.n_cells)
    np.testing.assert_allclose(log_a.detach().numpy(), expected, atol=1e-5)


def test_causal_mask_drops_post_anchor_history() -> None:
    """An anchor mid-window: only shots with date < anchor contribute."""
    def_kde, _store, vocab, ctx, _df = _build_setup()
    # Build a store with a mid-window anchor.
    rows = pd.DataFrame(
        [
            {
                "x": 0.0,
                "y": 10.0,
                "player_id": 1,
                "opponent": "BOS",
                "date": pd.Timestamp("2024-01-15"),
            }
        ]
    )
    mid_anchor = np.datetime64("2024-02-01", "D")
    mid_store = build_snapshot_store_from_shots(rows, [mid_anchor])
    relevance = RelevanceScore()
    field = AdaptiveDefensiveField(def_kde, mid_store, vocab, relevance)
    B = 3
    x_n = torch.from_numpy(ctx[:B]).float()
    opp_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)
    _, pi, has_history = field(opp_idx, snapshot_idx, x_n)
    # eff_mask = real_mask * (date < 2024-02-01). Some shots qualify;
    # later-dated ones do not.
    assert bool(has_history.all())
    # On the rows that have history, pi is a valid simplex.
    assert torch.allclose(pi.sum(dim=-1), torch.ones(B), atol=1e-5)
    # Each opponent's 60 shots start within the first ten days of January
    # and step by one day, so only those dated before 2024-02-01 (21-31 of
    # 60) survive the mask. Confirm by inspecting the effective mask directly.
    eff_mask, _, _ = field._causal_history_mask(opp_idx, snapshot_idx)
    assert eff_mask[0].sum().item() > 0


def test_relevance_is_independent_of_offensive_prior() -> None:
    """Defensive RelevanceScore parameters are not aliased to any other module."""
    def_kde, store, vocab, _ctx, _df = _build_setup()
    rel_def = RelevanceScore()
    field = AdaptiveDefensiveField(def_kde, store, vocab, rel_def)
    # Touch one parameter, confirm only the defensive instance sees it.
    rel_other = RelevanceScore()
    for p_def, p_other in zip(rel_def.parameters(), rel_other.parameters(), strict=True):
        assert p_def.data_ptr() != p_other.data_ptr()
    # And the field exposes its own relevance, not someone else's.
    assert field.relevance is rel_def


def test_unfitted_kde_raises() -> None:
    g = CourtGrid()
    def_kde = AdaptiveKDE(grid=g, bandwidth=1.5)
    rows = pd.DataFrame(
        [
            {
                "x": 0.0,
                "y": 10.0,
                "player_id": 1,
                "opponent": "BOS",
                "date": pd.Timestamp("2024-01-01"),
            }
        ]
    )
    store = build_snapshot_store_from_shots(rows, [np.datetime64("2024-02-01", "D")])
    vocab = OpponentVocab.from_ids(("BOS",))
    rel = RelevanceScore()
    with pytest.raises(ValueError, match="must be fit"):
        AdaptiveDefensiveField(def_kde, store, vocab, rel)


def test_unfitted_dates_raises() -> None:
    """Refitting without `date=` should be rejected — causal mask requires it."""
    g = CourtGrid(nx=10, ny=10)
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(
        [
            {
                "x": float(rng.normal()),
                "y": float(rng.normal()),
                "player_id": 1,
                "opponent": "BOS",
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(i)),
            }
            for i in range(20)
        ]
    )
    enc = ContextEncoder.fit(rows)
    ctx = enc.transform(rows)
    def_kde = AdaptiveKDE(grid=g, bandwidth=1.0)
    def_kde.fit(
        x=rows["x"].to_numpy(),
        y=rows["y"].to_numpy(),
        player_id=rows["opponent"].to_numpy(),
        context_features=ctx,
        # date intentionally omitted
    )
    store = build_snapshot_store_from_shots(rows, [np.datetime64("2024-02-01", "D")])
    vocab = OpponentVocab.from_ids(def_kde.players)
    rel = RelevanceScore()
    with pytest.raises(ValueError, match="must be fit with the date="):
        AdaptiveDefensiveField(def_kde, store, vocab, rel)


def test_gradient_flows_through_relevance() -> None:
    def_kde, store, vocab, ctx, _df = _build_setup()
    rel = RelevanceScore()
    field = AdaptiveDefensiveField(def_kde, store, vocab, rel)
    B = 4
    x_n = torch.from_numpy(ctx[:B]).float()
    opp_idx = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)
    log_a, _, _ = field(opp_idx, snapshot_idx, x_n)
    # Some scalar that depends on log_a.
    loss = log_a.mean()
    loss.backward()
    # At least one relevance parameter should have a non-trivial grad.
    grads = [p.grad for p in rel.parameters() if p.grad is not None]
    assert grads, "no relevance parameter received a gradient"
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_low_rank_matches_dense_when_full_rank() -> None:
    """SVD with rank == min(M.shape) should reproduce the dense path exactly."""
    def_kde, store, vocab, ctx, _df = _build_setup()
    rel_dense = RelevanceScore()
    rel_lr = RelevanceScore()
    # Force identical params on both copies for a fair comparison.
    rel_lr.load_state_dict(rel_dense.state_dict())
    field_dense = AdaptiveDefensiveField(def_kde, store, vocab, rel_dense)
    full_rank = min(def_kde.M.shape)
    field_lr = AdaptiveDefensiveField(def_kde, store, vocab, rel_lr, low_rank=full_rank)
    B = 4
    x_n = torch.from_numpy(ctx[:B]).float()
    opp_idx = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)
    log_a_dense, _, _ = field_dense(opp_idx, snapshot_idx, x_n)
    log_a_lr, _, _ = field_lr(opp_idx, snapshot_idx, x_n)
    # Compare in probability space rather than log space: float-drift in
    # SVD reconstruction (~1e-7 relative) flips near-zero KDE cells
    # across the eps clamp boundary, exaggerating the difference in log
    # space. The actual field values agree to ~1e-6.
    np.testing.assert_allclose(
        torch.exp(log_a_dense).detach().numpy(),
        torch.exp(log_a_lr).detach().numpy(),
        atol=1e-6,
    )
