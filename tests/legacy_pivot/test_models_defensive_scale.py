"""Tests for :class:`shotcloud.legacy_pivot.defensive_scale.LearnableDefensiveScale`."""

from __future__ import annotations

import pytest
import torch

from shotcloud.legacy_pivot.defensive_scale import DEFAULT_INIT_ALPHA_DEF, LearnableDefensiveScale


def test_default_init_is_half() -> None:
    m = LearnableDefensiveScale()
    assert abs(m.alpha_as_float() - DEFAULT_INIT_ALPHA_DEF) < 1e-5


def test_custom_init_round_trips() -> None:
    for init in (0.05, 0.1, 0.5, 1.0, 2.0):
        m = LearnableDefensiveScale(init=init)
        assert abs(m.alpha_as_float() - init) < 1e-5


def test_zero_init_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        LearnableDefensiveScale(init=0.0)


def test_negative_init_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        LearnableDefensiveScale(init=-0.1)


def test_forward_scales_input_by_alpha() -> None:
    m = LearnableDefensiveScale(init=0.5)
    log_q_def = torch.randn(2, 25)
    with torch.no_grad():
        out = m(log_q_def)
    assert torch.allclose(out, 0.5 * log_q_def, atol=1e-4)


def test_forward_preserves_shape() -> None:
    m = LearnableDefensiveScale()
    for shape in [(50,), (4, 50), (2, 3, 50)]:
        x = torch.randn(*shape)
        with torch.no_grad():
            out = m(x)
        assert out.shape == x.shape


def test_alpha_returns_tensor_with_grad() -> None:
    m = LearnableDefensiveScale()
    a = m.alpha()
    assert isinstance(a, torch.Tensor)
    assert a.requires_grad


def test_gradient_flows_through_theta() -> None:
    m = LearnableDefensiveScale(init=0.5)
    log_q_def = torch.randn(3, 10, requires_grad=False)
    out = m(log_q_def)
    loss = out.sum()
    loss.backward()
    assert m.theta.grad is not None
    expected = float(log_q_def.sum()) * float(torch.sigmoid(m.theta))
    assert abs(float(m.theta.grad) - expected) < 1e-4


def test_one_parameter_only() -> None:
    m = LearnableDefensiveScale()
    params = list(m.parameters())
    assert len(params) == 1
    assert params[0].requires_grad


# ---------------------------------------------------------------------------
# Context-conditioned mode (parallel to LearnableTemperature)
# ---------------------------------------------------------------------------


def test_context_dim_zero_is_default_scalar_mode() -> None:
    m = LearnableDefensiveScale(init=0.5)
    assert m.context_dim == 0
    assert m.context_mlp is None


def test_context_dim_positive_creates_mlp() -> None:
    m = LearnableDefensiveScale(init=0.5, context_dim=10, mlp_hidden=8)
    assert m.context_dim == 10
    assert m.context_mlp is not None
    n_params = sum(p.numel() for p in m.parameters())
    assert n_params == 97


def test_context_init_value_approximately_recovered() -> None:
    torch.manual_seed(0)
    init_value = 0.7
    m = LearnableDefensiveScale(init=init_value, context_dim=10, mlp_hidden=8)
    import numpy as np  # local import to avoid polluting top-level

    rng = np.random.default_rng(0)
    x_n = torch.from_numpy(rng.standard_normal((64, 10)).astype(np.float32))
    with torch.no_grad():
        a = m.alpha(x_n)
    assert a.shape == (64,)
    assert abs(float(a.mean()) - init_value) < 0.10


def test_context_forward_requires_x_n() -> None:
    m = LearnableDefensiveScale(init=0.5, context_dim=4)
    log_q_def = torch.randn(3, 10)
    with pytest.raises(ValueError, match="x_n"):
        m(log_q_def)


def test_scalar_forward_ignores_x_n() -> None:
    torch.manual_seed(0)
    m = LearnableDefensiveScale(init=0.4)
    log_q_def = torch.randn(2, 10)
    x_n = torch.randn(2, 4)
    with torch.no_grad():
        a = m(log_q_def)
        b = m(log_q_def, x_n)
    assert torch.allclose(a, b)


def test_context_gradient_flows() -> None:
    torch.manual_seed(0)
    m = LearnableDefensiveScale(init=0.5, context_dim=4, mlp_hidden=4)
    x_n = torch.randn(2, 4)
    log_q_def = torch.randn(2, 6)
    loss = m(log_q_def, x_n).sum()
    loss.backward()
    assert m.context_mlp.fc1.weight.grad is not None
    assert m.context_mlp.fc1.weight.grad.abs().sum() > 0
    assert m.context_mlp.fc2.weight.grad is not None
    assert m.context_mlp.fc2.weight.grad.abs().sum() > 0
