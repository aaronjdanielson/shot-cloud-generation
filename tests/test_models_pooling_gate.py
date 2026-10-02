"""Tests for ``shotcloud.models.pooling_gate``."""

from __future__ import annotations

import pytest
import torch

from shotcloud.data.context import CONTEXT_DIM
from shotcloud.models.pooling_gate import PoolingGate


def _inputs(b: int, h_hat: torch.Tensor, history_dim: int = 0):
    log1p_h = torch.log1p(h_hat)
    x_n = torch.zeros(b, CONTEXT_DIM)
    own_count = torch.full((b,), 25.0)
    own_avail = torch.ones(b, dtype=torch.bool)
    h_n = torch.zeros(b, history_dim) if history_dim > 0 else None
    return log1p_h, x_n, own_count, own_avail, h_n


def test_zero_init_g_theta_matches_history_schedule() -> None:
    """With g_θ zero-init, λ equals the closed-form (b_0, b_H)
    history schedule exactly."""
    gate = PoolingGate()
    h_hat = torch.tensor([0.0, 25.0, 100.0, 300.0, 1000.0])
    log1p_h, x_n, own_count, own_avail, _ = _inputs(5, h_hat)
    with torch.no_grad():
        lam = gate(log1p_h, x_n, own_count, own_avail)
    for i, h in enumerate(h_hat.tolist()):
        expected = gate.history_schedule(h)
        assert abs(float(lam[i]) - expected) < 1e-5


def test_lambda_monotone_increasing_in_history() -> None:
    """λ is non-decreasing in the own-history count for any value of
    the slope parameter (softplus keeps the slope ≥ 0)."""
    torch.manual_seed(0)
    gate = PoolingGate()
    # Perturb b_h to a few values, including negative.
    for bh in (-3.0, -0.31, 0.0, 2.0):
        with torch.no_grad():
            gate.b_h.fill_(bh)
        h_hat = torch.tensor([0.0, 10.0, 50.0, 200.0, 800.0])
        log1p_h, x_n, own_count, own_avail, _ = _inputs(5, h_hat)
        with torch.no_grad():
            lam = gate(log1p_h, x_n, own_count, own_avail)
        diffs = lam[1:] - lam[:-1]
        assert (diffs >= -1e-6).all(), f"λ not monotone at b_h={bh}: {lam.tolist()}"


def test_default_schedule_in_target_band() -> None:
    """The default init lands in the intended band: low at cold-start,
    rising past 0.5 around H=100."""
    gate = PoolingGate()
    assert gate.history_schedule(0.0) < 0.15
    assert 0.25 < gate.history_schedule(25.0) < 0.40
    assert 0.45 < gate.history_schedule(100.0) < 0.60
    assert gate.history_schedule(1000.0) > 0.70


def test_lambda_in_unit_interval() -> None:
    torch.manual_seed(1)
    gate = PoolingGate()
    # Random g_θ weights so the MLP contributes.
    for p in gate.g_theta.parameters():
        torch.nn.init.normal_(p, std=1.0)
    h_hat = torch.rand(64) * 1500.0
    log1p_h, x_n, own_count, own_avail, _ = _inputs(64, h_hat)
    x_n = torch.randn(64, CONTEXT_DIM)
    lam = gate(log1p_h, x_n, own_count, own_avail)
    assert (lam >= 0.0).all() and (lam <= 1.0).all()


def test_no_own_support_forces_lambda_zero() -> None:
    gate = PoolingGate()
    h_hat = torch.tensor([500.0, 500.0])
    log1p_h, x_n, own_count, own_avail, _ = _inputs(2, h_hat)
    own_avail = torch.tensor([True, False])
    own_count = torch.tensor([40.0, 0.0])
    with torch.no_grad():
        lam = gate(log1p_h, x_n, own_count, own_avail)
    assert float(lam[1]) == 0.0
    assert float(lam[0]) > 0.0


def test_no_pooled_support_forces_lambda_one() -> None:
    gate = PoolingGate()
    h_hat = torch.tensor([5.0, 5.0])
    log1p_h, x_n, own_count, own_avail, _ = _inputs(2, h_hat)
    pooled_avail = torch.tensor([True, False])
    with torch.no_grad():
        lam = gate(log1p_h, x_n, own_count, own_avail, pooled_available=pooled_avail)
    assert float(lam[1]) == 1.0
    assert float(lam[0]) < 1.0


def test_gradient_flows_to_all_params() -> None:
    gate = PoolingGate()
    h_hat = torch.tensor([10.0, 200.0, 600.0])
    log1p_h, x_n, own_count, own_avail, _ = _inputs(3, h_hat)
    x_n = torch.randn(3, CONTEXT_DIM)
    lam = gate(log1p_h, x_n, own_count, own_avail)
    lam.sum().backward()
    assert gate.b0.grad is not None and gate.b0.grad.abs() > 0
    assert gate.b_h.grad is not None and gate.b_h.grad.abs() > 0
    # First g_θ layer receives gradient (chain through the zero-init
    # output layer is zero at init, but the output layer's own weight
    # gets gradient from the GELU activations).
    out_layer = gate.g_theta[-1]
    assert out_layer.weight.grad is not None and out_layer.weight.grad.abs().sum() > 0


def test_history_dim_requires_h_n() -> None:
    gate = PoolingGate(history_dim=10)
    h_hat = torch.tensor([10.0])
    log1p_h, x_n, own_count, own_avail, _ = _inputs(1, h_hat)
    with pytest.raises(ValueError, match="h_n is required"):
        gate(log1p_h, x_n, own_count, own_avail)
    # With h_n it works.
    lam = gate(log1p_h, x_n, own_count, own_avail, h_n=torch.zeros(1, 10))
    assert lam.shape == (1,)


def test_rejects_bad_shapes() -> None:
    gate = PoolingGate()
    log1p_h, x_n, own_count, own_avail, _ = _inputs(3, torch.tensor([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="log1p_h_hat must be"):
        gate(torch.zeros(2), x_n, own_count, own_avail)
    with pytest.raises(ValueError, match="own_support_count must be"):
        gate(log1p_h, x_n, torch.zeros(2), own_avail)


def test_constructor_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="context_dim"):
        PoolingGate(context_dim=0)
    with pytest.raises(ValueError, match="history_dim"):
        PoolingGate(history_dim=-1)
    with pytest.raises(ValueError, match="hidden_dim"):
        PoolingGate(hidden_dim=0)
