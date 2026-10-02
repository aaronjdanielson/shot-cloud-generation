"""Tests for :class:`shotcloud.legacy.LearnableTemperature`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shotcloud.legacy import LearnableTemperature
from shotcloud.legacy.temperature import DEFAULT_INIT_TAU

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_init_is_one() -> None:
    m = LearnableTemperature()
    assert abs(m.tau_as_float() - DEFAULT_INIT_TAU) < 1e-5


def test_custom_init_round_trips() -> None:
    for init in (0.1, 0.5, 1.0, 1.5, 2.5):
        m = LearnableTemperature(init=init)
        assert abs(m.tau_as_float() - init) < 1e-5


def test_zero_init_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        LearnableTemperature(init=0.0)


def test_negative_init_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        LearnableTemperature(init=-0.5)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def test_forward_with_tau_one_is_identity() -> None:
    """At τ = 1, forward(log_q0) == log_q0 (within float precision)."""
    m = LearnableTemperature(init=1.0)
    log_q0 = torch.randn(3, 100)
    with torch.no_grad():
        out = m(log_q0)
    assert torch.allclose(out, log_q0, atol=1e-5)


def test_forward_scales_correctly() -> None:
    """At τ = 2.0, forward returns 2 * log_q0."""
    m = LearnableTemperature(init=2.0)
    log_q0 = torch.randn(3, 50)
    with torch.no_grad():
        out = m(log_q0)
    assert torch.allclose(out, 2.0 * log_q0, atol=1e-4)


def test_forward_preserves_shape() -> None:
    m = LearnableTemperature()
    for shape in [(100,), (3, 100), (2, 4, 50)]:
        log_q0 = torch.randn(*shape)
        with torch.no_grad():
            out = m(log_q0)
        assert out.shape == log_q0.shape


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------


def test_gradient_flows_through_theta() -> None:
    """Backprop through forward must produce a gradient on θ."""
    m = LearnableTemperature(init=1.5)
    log_q0 = torch.randn(2, 10, requires_grad=False)
    out = m(log_q0)
    loss = out.sum()
    loss.backward()
    assert m.theta.grad is not None
    # ∂(τ * log_q0).sum() / ∂θ = log_q0.sum() * sigmoid(θ).
    expected = float(log_q0.sum()) * float(torch.sigmoid(m.theta))
    assert abs(float(m.theta.grad) - expected) < 1e-4


def test_theta_is_a_parameter() -> None:
    m = LearnableTemperature()
    params = list(m.parameters())
    assert len(params) == 1
    assert params[0].requires_grad


# ---------------------------------------------------------------------------
# tau methods
# ---------------------------------------------------------------------------


def test_tau_returns_tensor_with_grad() -> None:
    m = LearnableTemperature()
    t = m.tau()
    assert isinstance(t, torch.Tensor)
    assert t.requires_grad


def test_tau_as_float_is_detached() -> None:
    m = LearnableTemperature(init=1.7)
    val = m.tau_as_float()
    assert isinstance(val, float)
    assert abs(val - 1.7) < 1e-5


# ---------------------------------------------------------------------------
# Softmax-absorption identity (the math claim that justifies the
# "no separate normalization" implementation).
# ---------------------------------------------------------------------------


def test_softmax_absorbs_partition_function() -> None:
    """``softmax(τ log_q0) == softmax(τ log_q0 - logsumexp(τ log_q0))``.

    This is the mathematical identity that lets the temperature pathway
    skip an explicit re-normalization before the decoder sees its
    input. Without this, the CLI/trainer-side simplification would be
    wrong.
    """
    rng = np.random.default_rng(0)
    log_q0 = torch.from_numpy(np.log(rng.dirichlet(np.ones(50)))).float()
    tau = 1.7
    scaled = tau * log_q0
    direct = torch.softmax(scaled, dim=-1)
    normalized = torch.softmax(scaled - torch.logsumexp(scaled, dim=-1, keepdim=True), dim=-1)
    assert torch.allclose(direct, normalized, atol=1e-6)


# ---------------------------------------------------------------------------
# Context-conditioned mode
# ---------------------------------------------------------------------------


def test_context_dim_zero_is_default_scalar_mode() -> None:
    """Default constructor creates a scalar temperature (no MLP)."""
    m = LearnableTemperature(init=1.5)
    assert m.context_dim == 0
    assert m.context_mlp is None
    # 1 parameter total (θ).
    assert sum(p.numel() for p in m.parameters()) == 1


def test_context_dim_positive_creates_mlp() -> None:
    m = LearnableTemperature(init=1.0, context_dim=10, mlp_hidden=8)
    assert m.context_dim == 10
    assert m.context_mlp is not None
    # MLP params: 10*8 + 8 (fc1) + 8*1 + 1 (fc2) = 97
    n_params = sum(p.numel() for p in m.parameters())
    assert n_params == 97


def test_context_negative_dim_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LearnableTemperature(context_dim=-1)


def test_context_init_value_approximately_recovered() -> None:
    """At init, mean τ over random x_n should be close to ``init``."""
    torch.manual_seed(0)
    init_value = 1.4
    m = LearnableTemperature(init=init_value, context_dim=10, mlp_hidden=8)
    rng = np.random.default_rng(0)
    x_n = torch.from_numpy(rng.standard_normal((64, 10)).astype(np.float32))
    with torch.no_grad():
        tau = m.tau(x_n)
    assert tau.shape == (64,)
    # MLP weight init is small; outputs should cluster near init.
    assert abs(float(tau.mean()) - init_value) < 0.15


def test_context_forward_uses_per_row_tau() -> None:
    """Different x_n rows produce different τ values (so different scalings)."""
    torch.manual_seed(1)
    m = LearnableTemperature(init=1.0, context_dim=5, mlp_hidden=4)
    # Make MLP non-trivially varying.
    with torch.no_grad():
        m.context_mlp.fc1.weight.add_(torch.randn_like(m.context_mlp.fc1.weight))
        m.context_mlp.fc2.weight.add_(torch.randn_like(m.context_mlp.fc2.weight))
    x_n = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    log_q0 = torch.zeros(2, 20)
    log_q0[0, 5] = 1.0
    log_q0[1, 5] = 1.0
    with torch.no_grad():
        tau = m.tau(x_n)
        out = m(log_q0, x_n)
    # The two rows should get different per-row τ.
    assert abs(float(tau[0]) - float(tau[1])) > 1e-3
    # Forward equals τ_row * log_q0_row.
    assert torch.allclose(out[0], tau[0] * log_q0[0])
    assert torch.allclose(out[1], tau[1] * log_q0[1])


def test_context_forward_requires_x_n() -> None:
    m = LearnableTemperature(init=1.0, context_dim=4)
    log_q0 = torch.randn(3, 10)
    with pytest.raises(ValueError, match="x_n"):
        m(log_q0)


def test_scalar_forward_ignores_x_n() -> None:
    """In scalar mode, forward(log_q0, x_n) is equivalent to forward(log_q0)."""
    torch.manual_seed(0)
    m = LearnableTemperature(init=1.3)
    log_q0 = torch.randn(2, 10)
    x_n = torch.randn(2, 4)  # arbitrary, should be ignored
    with torch.no_grad():
        a = m(log_q0)
        b = m(log_q0, x_n)
    assert torch.allclose(a, b)


def test_context_gradient_flows_through_mlp() -> None:
    torch.manual_seed(0)
    m = LearnableTemperature(init=1.0, context_dim=4, mlp_hidden=4)
    x_n = torch.randn(2, 4)
    log_q0 = torch.randn(2, 6)
    out = m(log_q0, x_n)
    loss = out.sum()
    loss.backward()
    # Both MLP layers' weights should have gradients.
    assert m.context_mlp.fc1.weight.grad is not None
    assert m.context_mlp.fc1.weight.grad.abs().sum() > 0
    assert m.context_mlp.fc2.weight.grad is not None
    assert m.context_mlp.fc2.weight.grad.abs().sum() > 0


def test_context_tau_as_float_returns_mean() -> None:
    """In context mode, tau_as_float(x_n) returns the mean τ over rows."""
    torch.manual_seed(0)
    m = LearnableTemperature(init=1.0, context_dim=4)
    x_n = torch.randn(8, 4)
    with torch.no_grad():
        per_row = m.tau(x_n)
        mean_val = m.tau_as_float(x_n)
    assert abs(mean_val - float(per_row.mean())) < 1e-6
