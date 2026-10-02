"""Tests for :class:`shotcloud.legacy_pivot.adaptive_prior.AdaptiveOffensivePrior`.

The prior consumes a :class:`SnapshotStore` and blends a
relevance-weighted self-KDE with the archetype prior produced by an
:class:`ArchetypeDictionary` and :class:`ArchetypeMixture`. Forward takes
``snapshot_idx`` and applies the causal date mask on the per-player
history pool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.typing import NDArray
from torch import Tensor

from shotcloud import AdaptiveKDE, CourtGrid, PlayerVocab, RelevanceScore
from shotcloud.data import ContextEncoder
from shotcloud.data.role_profile import build_role_profiles
from shotcloud.data.snapshots import SnapshotStore, build_snapshot_store_from_shots
from shotcloud.legacy_pivot.adaptive_prior import AdaptiveOffensivePrior
from shotcloud.legacy_pivot.archetypes import ArchetypeDictionary, ArchetypeMixture


def _build_setup(
    seed: int = 0,
    *,
    bandwidth: float = 1.5,
    max_history: int | None = 40,
    n_archetypes: int = 4,
) -> tuple[
    AdaptiveKDE,
    SnapshotStore,
    ArchetypeDictionary,
    ArchetypeMixture,
    PlayerVocab,
    NDArray[np.float32],
    pd.DataFrame,
]:
    """Synthetic 3-player setup with a 1-bundle SnapshotStore.

    The single bundle anchored at 2024-04-01 makes every training shot
    (Jan-Mar 2024) causal-eligible (every history date < 2024-04-01).
    Archetype surfaces are uniform, so the archetype prior reduces to a
    uniform fallback; this isolates the self-KDE + ESS path from the
    archetype layer. The encoder runs in no-snapshot mode so the
    snapshot-derived slices of x_n stay zero, which does not affect the
    prior's history/causal-mask machinery.
    """
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(seed)
    rows = []
    for pid_int, _pid in enumerate(("A", "B", "C"), start=1):
        for i in range(60):
            rows.append(
                {
                    "x": float(rng.normal(0, 5)),
                    "y": float(rng.normal(15, 5)),
                    "player_id": pid_int,
                    "opponent": "X" if i % 2 == 0 else "Y",
                    "made": int(rng.random() < 0.5),
                    "period": (i % 4) + 1,
                    "time_remaining_sec": int(60 * (i % 48)),
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(i)),
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

    akde = AdaptiveKDE(grid=g, bandwidth=bandwidth, max_history=max_history)
    akde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    vocab = PlayerVocab.from_ids(akde.players)

    archetype_dict = ArchetypeDictionary.from_snapshot_store(store)
    archetype_mix = ArchetypeMixture(n_archetypes=n_archetypes)

    return akde, store, archetype_dict, archetype_mix, vocab, ctx, df


def test_forward_outputs_log_probabilities() -> None:
    akde, store, ad, am, vocab, ctx, _df = _build_setup()
    relevance = RelevanceScore()
    prior = AdaptiveOffensivePrior(akde, store, ad, am, vocab, relevance, kappa=20.0)
    B = 4
    x_n = torch.from_numpy(ctx[:B]).float()
    player_idx = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)
    log_q, pi, omega = prior(player_idx, snapshot_idx, x_n, x_n)
    assert log_q.shape == (B, akde.grid.n_cells)
    assert pi.shape == (B, prior.max_history)
    assert omega.shape == (B,)
    np.testing.assert_allclose(torch.exp(log_q).sum(dim=-1).detach(), torch.ones(B), atol=1e-4)
    np.testing.assert_allclose(pi.sum(dim=-1).detach(), torch.ones(B), atol=1e-5)
    assert (omega > 0).all() and (omega < 1).all()


def test_uniform_relevance_matches_kde_uniform() -> None:
    """At init (all β = 0), relevance softmax is uniform → adaptive density
    matches AdaptiveKDE.density_with_uniform_relevance for each player."""
    akde, store, ad, am, vocab, ctx, _df = _build_setup()
    relevance = RelevanceScore()
    prior = AdaptiveOffensivePrior(akde, store, ad, am, vocab, relevance, kappa=1e-6)
    x_n = torch.from_numpy(ctx[:3]).float()
    pids = [vocab.to_idx(p) for p in akde.players[:3]]
    player_idx = torch.tensor(pids, dtype=torch.long)
    snapshot_idx = torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        log_q, _pi, omega = prior(player_idx, snapshot_idx, x_n, x_n)
    assert (omega > 0.99).all()
    for b, pid in enumerate(akde.players[:3]):
        expected = akde.density_with_uniform_relevance(pid)
        actual = torch.exp(log_q[b]).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-3)


def test_high_kappa_falls_back_to_archetype_prior() -> None:
    """Very large κ → ω → 0, and the output approaches the archetype prior."""
    akde, store, ad, am, vocab, ctx, _df = _build_setup()
    relevance = RelevanceScore()
    prior = AdaptiveOffensivePrior(akde, store, ad, am, vocab, relevance, kappa=1e6)
    x_n = torch.from_numpy(ctx[:1]).float()
    player_idx = torch.tensor([0], dtype=torch.long)
    snapshot_idx = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        log_q, _pi, omega = prior(player_idx, snapshot_idx, x_n, x_n)
        rho = am(x_n)
        q_arch = ad(snapshot_idx, rho).numpy()
    assert omega.item() < 0.001
    actual = torch.exp(log_q[0]).numpy()
    np.testing.assert_allclose(actual, q_arch[0], atol=1e-4)


def test_causal_mask_drops_post_anchor_history() -> None:
    """Shots dated after the anchor are excluded from the relevance pool.

    Build a setup where the player's first 30 shots are pre-anchor and the
    next 30 are post-anchor. Forward at the early anchor should have
    π summing to 1 only on the first 30 slots (post-anchor positions zero).
    """
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(60):
        rows.append(
            {
                "x": float(rng.normal(0, 5)),
                "y": float(rng.normal(15, 5)),
                "player_id": 1,
                "opponent": "X",
                "made": 1,
                "period": 1,
                "time_remaining_sec": 0,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(i)),
            }
        )
    df = pd.DataFrame(rows)
    # Anchor at day 30: first 30 shots are causal, last 30 are not.
    anchor = np.datetime64("2024-01-31", "D")
    store = build_snapshot_store_from_shots(
        df,
        [anchor],
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
        archetype_fit_fn=lambda sub, _t: np.full((4, g.n_cells), 1.0 / g.n_cells, dtype=np.float32),
    )
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    akde = AdaptiveKDE(grid=g, bandwidth=1.5, max_history=None)
    akde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    vocab = PlayerVocab.from_ids(akde.players)
    ad = ArchetypeDictionary.from_snapshot_store(store)
    am = ArchetypeMixture(n_archetypes=4)
    prior = AdaptiveOffensivePrior(akde, store, ad, am, vocab, RelevanceScore(), kappa=20.0)

    x_n = torch.from_numpy(ctx[:1]).float()
    player_idx = torch.tensor([0], dtype=torch.long)
    snapshot_idx = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        _log_q, pi, omega = prior(player_idx, snapshot_idx, x_n, x_n)
    # π still sums to 1 (rebalanced over the unmasked tail).
    np.testing.assert_allclose(pi.sum(dim=-1).numpy(), [1.0], atol=1e-5)
    # But all weight is on slots whose date < anchor.
    eff_mask = (prior.history_dates[player_idx] < prior.anchor_dates[snapshot_idx].unsqueeze(1)).to(
        pi.dtype
    )
    n_causal = int(eff_mask.sum().item())
    assert 0 < n_causal < akde.n_history[akde.players[0]], "test fixture must straddle the anchor"
    # No π weight on post-anchor slots.
    post_anchor_weight = (pi * (1.0 - eff_mask)).sum().item()
    assert post_anchor_weight < 1e-6
    assert (omega > 0).item()


def test_no_causal_history_falls_back_to_archetype_prior() -> None:
    """When every history shot is post-anchor, ω = 0 and output = archetype prior.

    The cleanest way to engineer this: build the prior normally, then
    override ``anchor_dates`` to epoch day 0 (1970-01-01). Every history
    shot is post-1970, so the causal mask zeroes the entire pool;
    ``has_history`` is False; ω = 0; output reduces to the archetype
    prior.
    """
    akde, store, ad, am, vocab, ctx, _df = _build_setup()
    relevance = RelevanceScore()
    prior = AdaptiveOffensivePrior(akde, store, ad, am, vocab, relevance, kappa=20.0)
    prior.anchor_dates.fill_(0)  # epoch day 0; every history slot is post-anchor

    x_n = torch.from_numpy(ctx[:1]).float()
    player_idx = torch.tensor([0], dtype=torch.long)
    snapshot_idx = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        log_q, pi, omega = prior(player_idx, snapshot_idx, x_n, x_n)
        rho = am(x_n)
        q_arch = ad(snapshot_idx, rho).numpy()
    assert omega.item() == 0.0
    np.testing.assert_allclose(pi.numpy(), 0.0, atol=0)
    actual = torch.exp(log_q[0]).numpy()
    np.testing.assert_allclose(actual, q_arch[0], atol=1e-5)


def test_gradient_flows_through_relevance_and_mixture() -> None:
    """NLL gradients flow back to both the relevance params and the mixture head."""
    akde, store, ad, am, vocab, ctx, _df = _build_setup()
    relevance = RelevanceScore(init_beta_q=0.5)
    prior = AdaptiveOffensivePrior(akde, store, ad, am, vocab, relevance, kappa=20.0)
    x_n = torch.from_numpy(ctx[:2]).float()
    player_idx = torch.tensor([0, 1], dtype=torch.long)
    snapshot_idx = torch.zeros(2, dtype=torch.long)
    log_q, _pi, _omega = prior(player_idx, snapshot_idx, x_n, x_n)
    cell = torch.tensor([5, 10], dtype=torch.long)
    loss = -log_q[torch.arange(2), cell].sum()
    loss.backward()
    assert relevance.beta_q.grad is not None
    assert relevance.beta_q.grad.abs() > 0
    # ArchetypeMixture also receives gradient (via ω-shrinkage path even
    # when the dictionary is uniform, the mixture parameters appear in
    # the computation graph).
    assert am.bias.grad is not None


def test_scatter_add_matches_dense_einsum() -> None:
    """The fast scatter_add path returns the same q_phi as the explicit dense gather."""
    torch.manual_seed(0)
    akde, store, ad, am, vocab, ctx, _df = _build_setup()
    relevance = RelevanceScore(
        init_beta_q=0.5,
        init_beta_m=1.0,
        init_beta_t=0.3,
        init_beta_o=0.7,
        init_lambda_g=0.0,
    )
    prior = AdaptiveOffensivePrior(akde, store, ad, am, vocab, relevance, kappa=20.0)

    B = 5
    x_n = torch.from_numpy(ctx[:B]).float()
    player_idx = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        log_q_fast, pi_fast, omega_fast = prior(player_idx, snapshot_idx, x_n, x_n)

    with torch.no_grad():
        eff_mask, has_history, z_j = prior._causal_history_mask(player_idx, snapshot_idx)
        pi_ref = prior.relevance.softmax(z_j, x_n, mask=eff_mask)
        pi_ref = torch.where(has_history.unsqueeze(-1), pi_ref, torch.zeros_like(pi_ref))
        cells_per_row = prior.history_cells[player_idx]
        gathered = prior.M.t()[cells_per_row]  # (B, max_N, n_cells)
        q_phi_ref = torch.einsum("bk,bkc->bc", pi_ref, gathered)
        n_eff_ref = 1.0 / pi_ref.pow(2).sum(dim=-1).clamp_min(prior.eps)
        omega_ref = n_eff_ref / (n_eff_ref + prior.kappa)
        omega_ref = torch.where(has_history, omega_ref, torch.zeros_like(omega_ref))
        rho = am(x_n)
        q_arch = ad(snapshot_idx, rho)
        mixed_ref = omega_ref.unsqueeze(-1) * q_phi_ref + (1.0 - omega_ref.unsqueeze(-1)) * q_arch
        log_q_ref = torch.log(mixed_ref.clamp_min(prior.eps / prior.n_cells))

    assert torch.equal(pi_fast, pi_ref)
    assert torch.allclose(log_q_fast, log_q_ref, atol=1e-5, rtol=1e-5), (
        f"max abs diff: {(log_q_fast - log_q_ref).abs().max().item():.2e}"
    )
    assert torch.allclose(omega_fast, omega_ref, atol=1e-6)


def _q_phi_from_prior(
    prior: AdaptiveOffensivePrior,
    player_idx: Tensor,
    snapshot_idx: Tensor,
    x_n: Tensor,
) -> Tensor:
    """Replicate the prior's pre-log ``q_φ(c)`` computation for testing."""
    eff_mask, has_history, z_j = prior._causal_history_mask(player_idx, snapshot_idx)
    pi = prior.relevance.softmax(z_j, x_n, mask=eff_mask)
    pi = torch.where(has_history.unsqueeze(-1), pi, torch.zeros_like(pi))
    cells_per_row = prior.history_cells[player_idx]
    pi_per_cell = torch.zeros(pi.shape[0], prior.n_cells, device=pi.device, dtype=pi.dtype)
    pi_per_cell.scatter_add_(dim=1, index=cells_per_row, src=pi)
    if prior.low_rank is None:
        return pi_per_cell @ prior.M.t()
    proj = pi_per_cell @ prior.M_V
    proj = proj * prior.M_S
    return proj @ prior.M_U.t()


def test_svd_full_rank_matches_dense_exactly() -> None:
    """At full rank, SVD ≈ exact matmul up to float roundoff."""
    torch.manual_seed(0)
    akde, store, ad, am, vocab, ctx, _df = _build_setup()
    rkw = dict(
        init_beta_q=0.5, init_beta_m=1.0, init_beta_t=0.3, init_beta_o=0.7, init_lambda_g=0.0
    )
    n_cells = akde.grid.n_cells
    prior_exact = AdaptiveOffensivePrior(
        akde, store, ad, am, vocab, RelevanceScore(**rkw), kappa=20.0, low_rank=None
    )
    prior_svd = AdaptiveOffensivePrior(
        akde, store, ad, am, vocab, RelevanceScore(**rkw), kappa=20.0, low_rank=n_cells
    )
    M_reconstructed = prior_svd.M_U @ torch.diag(prior_svd.M_S) @ prior_svd.M_V.t()
    M_exact = torch.from_numpy(akde.M.astype(np.float32))  # type: ignore[arg-type]
    rel_err = (M_reconstructed - M_exact).norm() / M_exact.norm()
    assert rel_err < 1e-5, f"Full-rank SVD reconstruction error: {rel_err.item():.2e}"

    B = 5
    x_n = torch.from_numpy(ctx[:B]).float()
    player_idx = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        q_phi_exact = _q_phi_from_prior(prior_exact, player_idx, snapshot_idx, x_n)
        q_phi_svd = _q_phi_from_prior(prior_svd, player_idx, snapshot_idx, x_n)
    max_abs_diff = (q_phi_exact - q_phi_svd).abs().max().item()
    assert max_abs_diff < 1e-4, f"Full-rank SVD q_φ diverges from exact: {max_abs_diff:.2e}"


def test_svd_low_rank_invalid_raises() -> None:
    akde, store, ad, am, vocab, _ctx, _df = _build_setup()
    relevance = RelevanceScore()
    with pytest.raises(ValueError, match="low_rank must be positive or None"):
        AdaptiveOffensivePrior(akde, store, ad, am, vocab, relevance, low_rank=0)
    with pytest.raises(ValueError, match="low_rank must be positive or None"):
        AdaptiveOffensivePrior(akde, store, ad, am, vocab, relevance, low_rank=-5)


def test_unfitted_kde_raises() -> None:
    g = CourtGrid()
    akde = AdaptiveKDE(grid=g)
    # Build a minimal store + dictionary so the constructor reaches the
    # akde.is_fitted check.
    rng = np.random.default_rng(0)
    n = 10
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 5, n),
            "y": rng.normal(15, 5, n),
            "player_id": [1] * n,
            "date": [pd.Timestamp("2024-01-01")] * n,
            "period": [1] * n,
            "time_remaining_sec": [0] * n,
            "opponent": ["X"] * n,
            "made": [1] * n,
        }
    )
    store = build_snapshot_store_from_shots(
        df,
        [np.datetime64("2024-04-01", "D")],
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
        archetype_fit_fn=lambda sub, _t: np.full((4, g.n_cells), 1.0 / g.n_cells, dtype=np.float32),
    )
    ad = ArchetypeDictionary.from_snapshot_store(store)
    am = ArchetypeMixture(n_archetypes=4)
    vocab = PlayerVocab.from_ids(["1"])
    relevance = RelevanceScore()
    with pytest.raises(ValueError, match="adaptive_kde must be fit"):
        AdaptiveOffensivePrior(akde, store, ad, am, vocab, relevance)


def test_unfitted_dates_raises() -> None:
    """AdaptiveKDE fit without ``date=`` cannot be used: causal mask requires dates."""
    g = CourtGrid()
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 5, n),
            "y": rng.normal(15, 5, n),
            "player_id": [1] * n,
            "date": [pd.Timestamp("2024-01-01")] * n,
            "period": [1] * n,
            "time_remaining_sec": [0] * n,
            "opponent": ["X"] * n,
            "made": [1] * n,
        }
    )
    store = build_snapshot_store_from_shots(
        df,
        [np.datetime64("2024-04-01", "D")],
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
        archetype_fit_fn=lambda sub, _t: np.full((4, g.n_cells), 1.0 / g.n_cells, dtype=np.float32),
    )
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    akde = AdaptiveKDE(grid=g, bandwidth=1.5, max_history=None)
    akde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        # date=  intentionally omitted
    )
    ad = ArchetypeDictionary.from_snapshot_store(store)
    am = ArchetypeMixture(n_archetypes=4)
    vocab = PlayerVocab.from_ids(akde.players)
    relevance = RelevanceScore()
    with pytest.raises(ValueError, match="must be fit with the date"):
        AdaptiveOffensivePrior(akde, store, ad, am, vocab, relevance)
