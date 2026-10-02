"""Tests for :class:`shotcloud.models.TimingSoftmaxHead`, the 48-bin timing factor.

Covers normalized ``(B, n_bins)`` log-probabilities, the zero-initialized residual
(uniform at initialization, or ``softmax(a(t))`` for a nonzero baseline), gradient
flow, the context-only signature, and input validation.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from shotcloud import TimingSoftmaxHead
from shotcloud.data.context import CONTEXT_DIM


def test_forward_shape_and_normalization() -> None:
    head = TimingSoftmaxHead()
    x_n = torch.randn(6, CONTEXT_DIM)
    log_p = head(x_n)
    assert log_p.shape == (6, 48)
    np.testing.assert_allclose(torch.exp(log_p).sum(dim=-1).detach().numpy(), np.ones(6), atol=1e-5)


def test_zero_init_gives_uniform_distribution() -> None:
    """With ``zero_init_residual=True`` and ``a(t) = 0`` the distribution is uniform
    for any ``x_n``."""
    head = TimingSoftmaxHead(zero_init_residual=True)
    # Both the baseline and the residual output layer start at zero.
    assert torch.equal(head.bin_baseline, torch.zeros_like(head.bin_baseline))
    assert torch.equal(head.fc2.weight, torch.zeros_like(head.fc2.weight))
    assert torch.equal(head.fc2.bias, torch.zeros_like(head.fc2.bias))

    x_n = torch.randn(4, CONTEXT_DIM)
    log_p = head(x_n)
    expected = torch.full_like(log_p, np.log(1.0 / 48))
    np.testing.assert_allclose(log_p.detach().numpy(), expected.detach().numpy(), atol=1e-6)


def test_baseline_only_when_residual_is_zero() -> None:
    """With a zero-initialized residual every row equals ``softmax(a(t))``."""
    head = TimingSoftmaxHead(zero_init_residual=True)
    # Set a(t) to a structured non-uniform pattern.
    with torch.no_grad():
        head.bin_baseline.copy_(torch.linspace(-2.0, 2.0, 48))

    x_n = torch.randn(5, CONTEXT_DIM)
    log_p = head(x_n)
    expected = torch.log_softmax(head.bin_baseline, dim=-1)
    for row in range(5):
        np.testing.assert_allclose(
            log_p[row].detach().numpy(), expected.detach().numpy(), atol=1e-6
        )


def test_random_init_residual_gives_context_dependent_output() -> None:
    """With ``zero_init_residual=False``, different ``x_n`` give different
    log-probabilities."""
    head = TimingSoftmaxHead(zero_init_residual=False)
    x_a = torch.randn(1, CONTEXT_DIM) * 5
    x_b = -x_a
    log_pa = head(x_a)
    log_pb = head(x_b)
    # At least some bins should differ substantially.
    assert (log_pa - log_pb).abs().max().item() > 1e-3


def test_log_prob_shape_and_consistency() -> None:
    head = TimingSoftmaxHead()
    x_n = torch.randn(7, CONTEXT_DIM)
    t_bin = torch.tensor([0, 5, 10, 23, 35, 47, 1], dtype=torch.long)
    log_p = head.log_prob(t_bin, x_n)
    assert log_p.shape == (7,)
    # log_prob equals log_p[row, t_bin].
    full_log_p = head(x_n)
    expected = full_log_p.gather(-1, t_bin.unsqueeze(-1)).squeeze(-1)
    np.testing.assert_allclose(log_p.detach().numpy(), expected.detach().numpy(), atol=1e-7)


def test_gradient_flows_to_baseline_and_residual() -> None:
    head = TimingSoftmaxHead(zero_init_residual=False)
    x_n = torch.randn(8, CONTEXT_DIM)
    t_bin = torch.randint(0, 48, (8,), dtype=torch.long)
    nll = -head.log_prob(t_bin, x_n).mean()
    nll.backward()

    # Baseline receives a gradient.
    assert head.bin_baseline.grad is not None
    assert head.bin_baseline.grad.abs().sum().item() > 0
    # Residual MLP receives gradients.
    assert head.fc1.weight.grad is not None
    assert head.fc1.weight.grad.abs().sum().item() > 0
    assert head.fc2.weight.grad is not None
    assert head.fc2.weight.grad.abs().sum().item() > 0


def test_zero_init_residual_baseline_grad_only_at_init() -> None:
    """With a zero-initialized residual, the first backward pass reaches both the
    baseline and ``fc2.weight`` (through the nonzero hidden activations)."""
    head = TimingSoftmaxHead(zero_init_residual=True)
    x_n = torch.randn(8, CONTEXT_DIM)
    t_bin = torch.randint(0, 48, (8,), dtype=torch.long)
    nll = -head.log_prob(t_bin, x_n).mean()
    nll.backward()
    # The baseline is the only term that moves the distribution at initialization.
    assert head.bin_baseline.grad is not None
    assert head.bin_baseline.grad.abs().sum().item() > 0
    # fc2.weight's gradient is proportional to the nonzero hidden activations.
    assert head.fc2.weight.grad is not None
    assert head.fc2.weight.grad.abs().sum().item() > 0


def test_no_player_idx_in_signature() -> None:
    """``forward`` takes only ``x_n``: the timing head is context-only."""
    sig = inspect.signature(TimingSoftmaxHead.forward)
    params = set(sig.parameters.keys())
    assert "player_idx" not in params
    assert params == {"self", "x_n"}


def test_custom_n_bins() -> None:
    head = TimingSoftmaxHead(n_bins=24)
    x_n = torch.randn(3, CONTEXT_DIM)
    log_p = head(x_n)
    assert log_p.shape == (3, 24)


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        TimingSoftmaxHead(n_bins=0)
    with pytest.raises(ValueError, match="context_dim"):
        TimingSoftmaxHead(context_dim=0)
    with pytest.raises(ValueError, match="hidden_dim"):
        TimingSoftmaxHead(hidden_dim=0)


def test_forward_rejects_wrong_input_dim() -> None:
    head = TimingSoftmaxHead(context_dim=27)
    with pytest.raises(ValueError, match="context_dim=27"):
        head(torch.randn(5, 30))


def test_log_prob_shape_validation() -> None:
    head = TimingSoftmaxHead()
    x_n = torch.randn(4, CONTEXT_DIM)
    with pytest.raises(ValueError, match="t_bin must have shape"):
        head.log_prob(torch.tensor([1, 2]), x_n)  # mismatched batch
    with pytest.raises(ValueError, match="t_bin must have shape"):
        head.log_prob(torch.zeros(4, 2, dtype=torch.long), x_n)  # wrong dim
