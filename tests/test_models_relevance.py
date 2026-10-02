"""Tests for the structured :class:`shotcloud.models.RelevanceScore` and its MLP
alternative :class:`shotcloud.models.RelevanceMLP`.
"""

from __future__ import annotations

import pytest
import torch

from shotcloud import RelevanceScore
from shotcloud.data import CONTEXT_DIM, FEATURE_LAYOUT


def _make_canonical_x(
    period: int, time_in_period: float, recency: float, opp_bucket: int
) -> torch.Tensor:
    """Construct a single canonical x_n vector for testing."""
    x = torch.zeros(CONTEXT_DIM, dtype=torch.float32)
    x[FEATURE_LAYOUT["period_onehot"].start + (period - 1)] = 1.0
    x[FEATURE_LAYOUT["time_in_period"].start] = time_in_period
    x[FEATURE_LAYOUT["season_recency"].start] = recency
    x[FEATURE_LAYOUT["opp_efficiency_onehot"].start + opp_bucket] = 1.0
    return x


def test_default_init_gives_zero_logits() -> None:
    """All five parameters start at 0, so logits are zero and the softmax is uniform."""
    m = RelevanceScore()
    z_j = torch.randn(2, 5, CONTEXT_DIM)
    x_n = torch.randn(2, CONTEXT_DIM)
    with torch.no_grad():
        logits = m(z_j, x_n)
        pi = m.softmax(z_j, x_n)
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-6)
    # Uniform over 5 positions.
    assert torch.allclose(pi, torch.full_like(pi, 0.2), atol=1e-6)


def test_period_match_responds_to_beta_q() -> None:
    """With β_q > 0, same-period historical shots get higher logits than mismatched ones."""
    m = RelevanceScore(init_beta_q=2.0)
    # Two history shots: shot 0 in Q1, shot 1 in Q2. Target in Q1.
    z_j = torch.stack(
        [_make_canonical_x(1, 0.5, 0.5, 0), _make_canonical_x(2, 0.5, 0.5, 0)]
    ).unsqueeze(0)
    x_n = _make_canonical_x(1, 0.5, 0.5, 0).unsqueeze(0)
    with torch.no_grad():
        logits = m(z_j, x_n)
    # Shot 0 (Q1, matches) should have higher logit than shot 1 (Q2, no match).
    assert logits[0, 0] > logits[0, 1] + 1.5  # β_q = 2.0 → ~2.0 difference


def test_time_similarity_responds_to_beta_m() -> None:
    """With β_m > 0, shots closer in time to the target get higher logits."""
    m = RelevanceScore(init_beta_m=5.0)
    z_j = torch.stack(
        [
            _make_canonical_x(1, 0.5, 0.5, 0),  # close in time
            _make_canonical_x(1, 0.0, 0.5, 0),  # far in time
        ]
    ).unsqueeze(0)
    x_n = _make_canonical_x(1, 0.5, 0.5, 0).unsqueeze(0)
    with torch.no_grad():
        logits = m(z_j, x_n)
    assert logits[0, 0] > logits[0, 1] + 2.0


def test_recency_similarity_responds_to_beta_t() -> None:
    m = RelevanceScore(init_beta_t=5.0)
    z_j = torch.stack(
        [
            _make_canonical_x(1, 0.5, 0.5, 0),  # same recency
            _make_canonical_x(1, 0.5, 0.0, 0),  # different recency
        ]
    ).unsqueeze(0)
    x_n = _make_canonical_x(1, 0.5, 0.5, 0).unsqueeze(0)
    with torch.no_grad():
        logits = m(z_j, x_n)
    assert logits[0, 0] > logits[0, 1] + 2.0


def test_opp_match_responds_to_beta_o() -> None:
    m = RelevanceScore(init_beta_o=2.0)
    z_j = torch.stack(
        [
            _make_canonical_x(1, 0.5, 0.5, 1),  # same bucket
            _make_canonical_x(1, 0.5, 0.5, 3),  # different bucket
        ]
    ).unsqueeze(0)
    x_n = _make_canonical_x(1, 0.5, 0.5, 1).unsqueeze(0)
    with torch.no_grad():
        logits = m(z_j, x_n)
    assert logits[0, 0] > logits[0, 1] + 1.5


def test_lambda_g_recency_decay() -> None:
    """With λ_g > 0, older (larger games_ago) shots get smaller logits."""
    m = RelevanceScore(init_lambda_g=1.0)
    z_j = torch.zeros(1, 3, CONTEXT_DIM)
    x_n = torch.zeros(1, CONTEXT_DIM)
    games_ago = torch.tensor([[0.0, 1.0, 2.0]])
    with torch.no_grad():
        logits = m(z_j, x_n, games_ago=games_ago)
    assert logits[0, 0] > logits[0, 1] > logits[0, 2]


def test_mask_zeros_padded_softmax() -> None:
    """Padded positions (mask=0) get -inf logits and zero softmax weight."""
    m = RelevanceScore(init_beta_q=1.0)
    z_j = torch.stack(
        [_make_canonical_x(1, 0.5, 0.5, 0), _make_canonical_x(1, 0.5, 0.5, 0)]
    ).unsqueeze(0)
    x_n = _make_canonical_x(1, 0.5, 0.5, 0).unsqueeze(0)
    mask = torch.tensor([[1.0, 0.0]])
    with torch.no_grad():
        pi = m.softmax(z_j, x_n, mask=mask)
    assert pi[0, 0] == 1.0
    assert pi[0, 1] == 0.0


def test_softmax_sums_to_one_per_row() -> None:
    m = RelevanceScore(init_beta_q=0.5, init_beta_m=0.5)
    z_j = torch.randn(3, 7, CONTEXT_DIM)
    x_n = torch.randn(3, CONTEXT_DIM)
    with torch.no_grad():
        pi = m.softmax(z_j, x_n)
    assert torch.allclose(pi.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_gradient_flows_through_all_params() -> None:
    m = RelevanceScore()
    z_j = torch.randn(2, 4, CONTEXT_DIM)
    x_n = torch.randn(2, CONTEXT_DIM)
    games_ago = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    pi = m.softmax(z_j, x_n, games_ago=games_ago)
    pi[:, 0].sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} has no grad"


def test_invalid_shapes_raise() -> None:
    m = RelevanceScore()
    with pytest.raises(ValueError, match="expected"):
        m(torch.randn(2, 5), torch.randn(2, 10))


def test_params_as_floats() -> None:
    m = RelevanceScore(init_beta_q=0.7, init_lambda_g=0.3)
    p = m.params_as_floats()
    assert abs(p["beta_q"] - 0.7) < 1e-5
    assert abs(p["lambda_g"] - 0.3) < 1e-5
    assert set(p.keys()) == {"beta_q", "beta_m", "beta_t", "beta_o", "lambda_g"}


def test_beta_max_init_reproduces_target_values() -> None:
    """With beta_max, init values flow through tanh and are reproduced exactly."""
    m = RelevanceScore(
        init_beta_q=0.5,
        init_beta_m=1.5,
        init_beta_t=-0.3,
        init_beta_o=0.0,
        init_lambda_g=0.7,
        beta_max=2.0,
    )
    p = m.params_as_floats()
    assert abs(p["beta_q"] - 0.5) < 1e-5
    assert abs(p["beta_m"] - 1.5) < 1e-5
    assert abs(p["beta_t"] - (-0.3)) < 1e-5
    assert abs(p["beta_o"] - 0.0) < 1e-5
    # λ_g is not bounded; should be passed through unchanged.
    assert abs(p["lambda_g"] - 0.7) < 1e-5


def test_beta_max_caps_effective_value() -> None:
    """Even when raw θ runs to a large magnitude, effective β stays in (-β_max, β_max)."""
    m = RelevanceScore(beta_max=2.0)
    # Manually push the raw parameter to a huge value to simulate gradient blow-up.
    with torch.no_grad():
        m.beta_q.fill_(50.0)
        m.beta_m.fill_(-50.0)
    p = m.params_as_floats()
    # tanh(50/2) saturates to 1, so β = 2 · 1 = 2.0; likewise -2.0.
    assert abs(p["beta_q"] - 2.0) < 1e-4
    assert abs(p["beta_m"] - (-2.0)) < 1e-4


def test_beta_max_bound_acts_in_logits() -> None:
    """The bounded β, not the raw θ, enters the logits."""
    z_j = torch.zeros(1, 2, CONTEXT_DIM)
    z_j[0, 0, FEATURE_LAYOUT["period_onehot"].start] = 1.0  # shot 0 matches period 1
    x_n = _make_canonical_x(period=1, time_in_period=0.0, recency=0.0, opp_bucket=0).unsqueeze(0)

    m = RelevanceScore(beta_max=2.0)
    with torch.no_grad():
        m.beta_q.fill_(100.0)  # would give huge logit if unbounded
        logits = m(z_j, x_n)
    # period_match for shot 0 is 1.0; effective_beta_q ≈ 2.0 → logit ≈ 2.0
    # (not 100.0). Tolerance loose because of float-in-tanh.
    assert abs(logits[0, 0].item() - 2.0) < 1e-3
    # shot 1 has period_match = 0 → logit = 0 regardless of β.
    assert abs(logits[0, 1].item() - 0.0) < 1e-5


def test_beta_max_invalid_raises() -> None:
    """Invalid ``beta_max`` values raise at construction."""
    with pytest.raises(ValueError, match="beta_max must be positive or None"):
        RelevanceScore(beta_max=0.0)
    with pytest.raises(ValueError, match="beta_max must be positive or None"):
        RelevanceScore(beta_max=-1.0)


def test_beta_max_init_at_boundary_raises() -> None:
    """An initial ``|β| ≥ β_max`` raises, since the tanh inverse would diverge."""
    with pytest.raises(ValueError, match="strictly within"):
        RelevanceScore(init_beta_q=2.0, beta_max=2.0)
    with pytest.raises(ValueError, match="strictly within"):
        RelevanceScore(init_beta_m=-3.0, beta_max=2.0)


def test_beta_max_gradient_flows() -> None:
    """Gradient through the tanh bound reaches the underlying θ parameters."""
    m = RelevanceScore(beta_max=2.0)
    z_j = torch.randn(2, 5, CONTEXT_DIM, requires_grad=False)
    x_n = torch.randn(2, CONTEXT_DIM, requires_grad=False)
    logits = m(z_j, x_n)
    loss = logits.pow(2).sum()
    loss.backward()
    # All four bounded θ should have non-None gradients.
    assert m.beta_q.grad is not None
    assert m.beta_m.grad is not None
    assert m.beta_t.grad is not None
    assert m.beta_o.grad is not None
    # λ_g not exercised here (no games_ago).


# ---------------------------------------------------------------------------
# RelevanceMLP
# ---------------------------------------------------------------------------


from shotcloud import RelevanceMLP  # noqa: E402


def test_mlp_default_init_gives_uniform_softmax() -> None:
    """The zero-initialized output layer gives zero logits and a uniform softmax,
    matching the ``RelevanceScore()`` default.
    """
    m = RelevanceMLP(context_dim=CONTEXT_DIM, hidden_dim=16)
    z_j = torch.randn(3, 7, CONTEXT_DIM)
    x_n = torch.randn(3, CONTEXT_DIM)
    with torch.no_grad():
        logits = m(z_j, x_n)
        pi = m.softmax(z_j, x_n)
    assert logits.shape == (3, 7)
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-6)
    assert torch.allclose(pi, torch.full_like(pi, 1.0 / 7.0), atol=1e-6)


def test_mlp_with_trained_output_layer_produces_nonuniform_logits() -> None:
    """With a nonzero ``fc2`` the logits vary across history shots."""
    m = RelevanceMLP(context_dim=CONTEXT_DIM, hidden_dim=16)
    torch.nn.init.normal_(m.fc2.weight, std=0.5)
    torch.nn.init.zeros_(m.fc2.bias)
    z_j = torch.randn(2, 5, CONTEXT_DIM)
    x_n = torch.randn(2, CONTEXT_DIM)
    with torch.no_grad():
        logits = m(z_j, x_n)
    # Logits should vary across the max_N dimension.
    row_stds = logits.std(dim=-1)
    assert (row_stds > 1e-3).all()


def test_mlp_mask_zeros_padded_positions() -> None:
    """Padded positions get -inf logits and zero softmax weight."""
    m = RelevanceMLP(context_dim=CONTEXT_DIM, hidden_dim=8)
    # Force fc2 nonzero so the unmasked logits aren't already all zero.
    torch.nn.init.normal_(m.fc2.weight, std=0.5)
    z_j = torch.randn(2, 6, CONTEXT_DIM)
    x_n = torch.randn(2, CONTEXT_DIM)
    mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    with torch.no_grad():
        logits = m(z_j, x_n, mask=mask)
        pi = m.softmax(z_j, x_n, mask=mask)
    assert torch.isinf(logits[0, 3:]).all() and (logits[0, 3:] < 0).all()
    assert torch.isinf(logits[1, 2:]).all() and (logits[1, 2:] < 0).all()
    # Softmax weight on padded positions is zero.
    assert (pi[0, 3:] == 0).all()
    assert (pi[1, 2:] == 0).all()
    # Per-row softmax sums to 1 on real positions.
    torch.testing.assert_close(pi.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=0)


def test_mlp_consumes_games_ago_when_supplied() -> None:
    """Changing ``games_ago`` changes the logits."""
    m = RelevanceMLP(context_dim=CONTEXT_DIM, hidden_dim=16)
    # Make fc2 nonzero so the MLP can react.
    torch.nn.init.normal_(m.fc2.weight, std=0.5)
    z_j = torch.randn(1, 4, CONTEXT_DIM)
    x_n = torch.randn(1, CONTEXT_DIM)
    games_ago_a = torch.zeros(1, 4)
    games_ago_b = torch.tensor([[0.1, 0.5, 1.0, 2.0]])
    with torch.no_grad():
        logits_a = m(z_j, x_n, games_ago=games_ago_a)
        logits_b = m(z_j, x_n, games_ago=games_ago_b)
    assert not torch.allclose(logits_a, logits_b)


def test_mlp_gradient_flows_to_all_params() -> None:
    """Backward through the MLP reaches both ``fc1`` and ``fc2``."""
    m = RelevanceMLP(context_dim=CONTEXT_DIM, hidden_dim=16)
    # fc2 starts at zero, which blocks gradient to fc1; perturb it so both layers
    # receive gradient.
    torch.nn.init.normal_(m.fc2.weight, std=0.5)
    z_j = torch.randn(2, 5, CONTEXT_DIM)
    x_n = torch.randn(2, CONTEXT_DIM)
    logits = m(z_j, x_n)
    loss = logits.pow(2).sum()
    loss.backward()
    assert m.fc1.weight.grad is not None and m.fc1.weight.grad.abs().sum() > 0
    assert m.fc1.bias.grad is not None
    assert m.fc2.weight.grad is not None and m.fc2.weight.grad.abs().sum() > 0
    assert m.fc2.bias.grad is not None


def test_mlp_raises_on_context_dim_mismatch() -> None:
    m = RelevanceMLP(context_dim=10, hidden_dim=8)
    z_j = torch.randn(1, 3, CONTEXT_DIM)  # CONTEXT_DIM != 10
    x_n = torch.randn(1, CONTEXT_DIM)
    with pytest.raises(ValueError, match="context_dim"):
        m(z_j, x_n)


def test_mlp_params_as_floats_returns_layer_norms() -> None:
    m = RelevanceMLP(context_dim=CONTEXT_DIM, hidden_dim=16)
    d = m.params_as_floats()
    assert set(d.keys()) == {"fc1_w_norm", "fc1_b_norm", "fc2_w_norm", "fc2_b_norm"}
    # fc2 is zero-init.
    assert d["fc2_w_norm"] == 0.0
    assert d["fc2_b_norm"] == 0.0
    # fc1 uses default Linear init → nonzero.
    assert d["fc1_w_norm"] > 0


def test_mlp_drop_in_replaces_structured_in_adaptive_offensive_prior() -> None:
    """``RelevanceMLP`` can replace ``RelevanceScore`` inside the grid-cell
    ``AdaptiveOffensivePrior`` (same forward and softmax contract)."""
    import numpy as np
    import pandas as pd

    from shotcloud import AdaptiveKDE, CourtGrid, PlayerVocab
    from shotcloud.data import ContextEncoder
    from shotcloud.data.role_profile import build_role_profiles
    from shotcloud.data.snapshots import build_snapshot_store_from_shots
    from shotcloud.legacy_pivot.adaptive_prior import AdaptiveOffensivePrior
    from shotcloud.legacy_pivot.archetypes import ArchetypeDictionary, ArchetypeMixture

    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=12)
    rng = np.random.default_rng(0)
    rows = []
    for pid in (1, 2, 3):
        for i in range(20):
            rows.append(
                {
                    "x": float(rng.normal(0, 5)),
                    "y": float(rng.normal(15, 5)),
                    "player_id": pid,
                    "opponent": "BOS",
                    "made": 0,
                    "period": (i % 4) + 1,
                    "time_remaining_sec": int(60 * (i % 48)),
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                }
            )
    df = pd.DataFrame(rows)
    anchor = np.datetime64("2024-04-01", "D")
    store = build_snapshot_store_from_shots(
        df,
        [anchor],
        role_profile_fn=lambda sub: build_role_profiles(sub, min_shots=1),
        archetype_fit_fn=lambda sub, _t: np.full((4, g.n_cells), 1.0 / g.n_cells, dtype=np.float32),
    )
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    akde = AdaptiveKDE(grid=g, bandwidth=1.5, max_history=20)
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
    mlp = RelevanceMLP(context_dim=CONTEXT_DIM, hidden_dim=16)
    prior = AdaptiveOffensivePrior(akde, store, ad, am, vocab, mlp, kappa=20.0)

    B = 3
    x_n = torch.from_numpy(ctx[:B]).float()
    player_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    snapshot_idx = torch.zeros(B, dtype=torch.long)
    log_q, pi, omega = prior(player_idx, snapshot_idx, x_n, x_n)
    assert log_q.shape == (B, g.n_cells)
    assert pi.shape == (B, prior.max_history)
    assert omega.shape == (B,)
    # exp(log_q) sums to 1 over the grid cells.
    import numpy as _np

    _np.testing.assert_allclose(torch.exp(log_q).sum(dim=-1).detach(), torch.ones(B), atol=1e-4)
