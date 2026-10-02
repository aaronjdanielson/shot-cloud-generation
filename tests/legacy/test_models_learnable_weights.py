"""Tests for :class:`shotcloud.legacy.LearnableKDEProductWeights`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shotcloud import CourtGrid, HierarchicalKDE
from shotcloud.legacy import KDEProduct, LearnableKDEProductWeights, _invert_softplus

# ---------------------------------------------------------------------------
# Reparameterization
# ---------------------------------------------------------------------------


def test_softplus_inverse_round_trips() -> None:
    """softplus(invert(a)) == a for typical positive weights."""
    for a in (1e-3, 0.1, 0.3, 1.0, 2.5):
        theta = _invert_softplus(a)
        recovered = float(torch.nn.functional.softplus(torch.tensor(theta)))
        assert abs(recovered - a) < 1e-6


def test_invert_softplus_rejects_zero() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _invert_softplus(0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        _invert_softplus(-0.1)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_default_init_matches_v1_baseline() -> None:
    m = LearnableKDEProductWeights()
    w = m.weights_as_floats()
    assert abs(w["a_p"] - 1.0) < 1e-5
    assert abs(w["a_g"] - 0.3) < 1e-5
    assert abs(w["a_0"] - 0.2) < 1e-5


def test_custom_init_weights() -> None:
    m = LearnableKDEProductWeights(init_weights={"a_p": 2.0, "a_g": 0.5, "a_0": 0.1})
    w = m.weights_as_floats()
    assert abs(w["a_p"] - 2.0) < 1e-5
    assert abs(w["a_g"] - 0.5) < 1e-5
    assert abs(w["a_0"] - 0.1) < 1e-5


def test_missing_init_weight_raises() -> None:
    with pytest.raises(ValueError, match="missing keys"):
        LearnableKDEProductWeights(init_weights={"a_p": 1.0, "a_g": 0.3})


def test_unknown_init_weight_raises() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        LearnableKDEProductWeights(init_weights={"a_p": 1.0, "a_g": 0.3, "a_0": 0.2, "a_d": 0.5})


def test_zero_init_weight_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        LearnableKDEProductWeights(init_weights={"a_p": 0.0, "a_g": 0.3, "a_0": 0.2})


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_kde() -> HierarchicalKDE:
    grid = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 5, n)
    y = rng.normal(15, 5, n)
    pid = np.array(["A"] * 100 + ["B"] * 100)
    pos = np.array(["G"] * n)
    kde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=200.0, recency_half_life_days=None)
    kde.fit(x=x, y=y, player_id=pid, position=pos)
    return kde


def _components_for(
    kde: HierarchicalKDE, player_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_qp = np.log(kde.player_density(player_id, hierarchical=False)).ravel()
    pos = kde.player_position[player_id]
    log_qg = np.log(kde.position_density(pos)).ravel()
    log_ql = np.log(kde.league_density()).ravel()
    return log_qp, log_qg, log_ql


def test_forward_normalizes_to_probabilities(fitted_kde: HierarchicalKDE) -> None:
    log_qp, log_qg, log_ql = _components_for(fitted_kde, "A")
    m = LearnableKDEProductWeights()
    log_q0 = m(
        torch.from_numpy(log_qp[None, :]).float(),
        torch.from_numpy(log_qg[None, :]).float(),
        torch.from_numpy(log_ql).float(),
    )
    assert log_q0.shape == (1, log_qp.size)
    np.testing.assert_allclose(torch.exp(log_q0).sum(dim=-1).item(), 1.0, atol=1e-5)


def test_forward_matches_kde_product_at_default_init(fitted_kde: HierarchicalKDE) -> None:
    """At default init weights (1.0, 0.3, 0.2), forward must match KDEProduct."""
    log_qp, log_qg, log_ql = _components_for(fitted_kde, "A")
    m = LearnableKDEProductWeights()  # default KDEProduct weights
    with torch.no_grad():
        log_q0 = (
            m(
                torch.from_numpy(log_qp[None, :]).double(),
                torch.from_numpy(log_qg[None, :]).double(),
                torch.from_numpy(log_ql).double(),
            )
            .squeeze(0)
            .numpy()
        )

    kdep = KDEProduct(hierarchical_kde=fitted_kde, weights={"a_p": 1.0, "a_g": 0.3, "a_0": 0.2})
    expected = kdep.log_density("A").ravel()
    np.testing.assert_allclose(log_q0, expected, atol=1e-5)


def test_forward_supports_batched_log_ql(fitted_kde: HierarchicalKDE) -> None:
    """log_ql may be passed as (n_cells,) or (B, n_cells); both must work."""
    log_qp, log_qg, log_ql = _components_for(fitted_kde, "A")
    m = LearnableKDEProductWeights()
    qp = torch.from_numpy(log_qp[None, :]).float().expand(3, -1).contiguous()
    qg = torch.from_numpy(log_qg[None, :]).float().expand(3, -1).contiguous()
    ql_1d = torch.from_numpy(log_ql).float()
    ql_2d = ql_1d.unsqueeze(0).expand(3, -1).contiguous()
    log_q0_a = m(qp, qg, ql_1d)
    log_q0_b = m(qp, qg, ql_2d)
    np.testing.assert_allclose(log_q0_a.detach().numpy(), log_q0_b.detach().numpy())


def test_forward_shape_mismatch_raises() -> None:
    m = LearnableKDEProductWeights()
    qp = torch.zeros(1, 100)
    qg = torch.zeros(1, 100)
    ql_bad = torch.zeros(50)
    with pytest.raises(ValueError, match="last-dim"):
        m(qp, qg, ql_bad)


def test_weights_have_grad() -> None:
    """The three thetas must be trainable parameters."""
    m = LearnableKDEProductWeights()
    params = list(m.parameters())
    assert len(params) == 3
    assert all(p.requires_grad for p in params)


def test_gradient_flows_through_weights(fitted_kde: HierarchicalKDE) -> None:
    """Backprop through the forward must produce gradients on theta_p/g/0."""
    log_qp, log_qg, log_ql = _components_for(fitted_kde, "A")
    m = LearnableKDEProductWeights()
    qp = torch.from_numpy(log_qp[None, :]).float()
    qg = torch.from_numpy(log_qg[None, :]).float()
    ql = torch.from_numpy(log_ql).float()
    log_q0 = m(qp, qg, ql)
    target = torch.tensor([5])  # arbitrary cell index
    loss = -log_q0[0, target]
    loss.sum().backward()
    assert m.theta_p.grad is not None and m.theta_p.grad.abs().item() > 0
    assert m.theta_g.grad is not None
    assert m.theta_0.grad is not None
