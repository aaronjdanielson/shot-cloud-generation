"""Tests for :class:`shotcloud.models.TimingSoftmaxHead` (paper §5).

Load-bearing invariants:

1. **Shape contract.** Forward returns ``(B, n_bins)`` log-probs
   (rows sum to 0 in exp space). ``log_prob(t_bin, x_n)`` returns
   ``(B,)``.
2. **Zero-init invariant.** With ``zero_init_residual=True`` (default)
   and ``a(t) = 0``, the timing distribution is uniform at step 0.
3. **Baseline-only at step 0.** With ``zero_init_residual=True`` but
   ``a(t)`` nonzero, the distribution at step 0 equals
   ``softmax(a(t))`` regardless of ``x_n``.
4. **Gradient flow.** Both the baseline and the residual MLP weights
   receive nonzero gradients from log-likelihood.
5. **No `player_idx` parameter.** Context-only by design.
6. **Validation.** Invalid ``n_bins``, ``context_dim``, ``hidden_dim``
   raise; out-of-range ``t_bin`` indexing surfaces clearly.
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
    """At step 0 with zero_init_residual=True and a(t)=0, the
    distribution is uniform across bins regardless of x_n."""
    head = TimingSoftmaxHead(zero_init_residual=True)
    # Confirm both factors are zero at init.
    assert torch.equal(head.bin_baseline, torch.zeros_like(head.bin_baseline))
    assert torch.equal(head.fc2.weight, torch.zeros_like(head.fc2.weight))
    assert torch.equal(head.fc2.bias, torch.zeros_like(head.fc2.bias))

    x_n = torch.randn(4, CONTEXT_DIM)
    log_p = head(x_n)
    expected = torch.full_like(log_p, np.log(1.0 / 48))
    np.testing.assert_allclose(log_p.detach().numpy(), expected.detach().numpy(), atol=1e-6)


def test_baseline_only_when_residual_is_zero() -> None:
    """When the residual is zero-init, the per-row distribution
    equals softmax(a(t)) regardless of x_n."""
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
    """With zero_init_residual=False, two different x_n produce
    different log-probabilities."""
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
    """With zero_init residual, the residual MLP's *output* is zero at
    step 0 — so the residual's contribution to the gradient passes
    through fc2.weight via x_n's hidden activations. Gradient should
    flow to both baseline AND fc2.weight on the first step."""
    head = TimingSoftmaxHead(zero_init_residual=True)
    x_n = torch.randn(8, CONTEXT_DIM)
    t_bin = torch.randint(0, 48, (8,), dtype=torch.long)
    nll = -head.log_prob(t_bin, x_n).mean()
    nll.backward()
    # Baseline grad: yes (this is the only signal that moves the
    # distribution at step 0).
    assert head.bin_baseline.grad is not None
    assert head.bin_baseline.grad.abs().sum().item() > 0
    # fc2.weight grad: nonzero — the cross-entropy gradient at step 0
    # depends on the hidden activation, which is nonzero.
    assert head.fc2.weight.grad is not None
    assert head.fc2.weight.grad.abs().sum().item() > 0


def test_no_player_idx_in_signature() -> None:
    """Architectural check: the timing head is context-only by design."""
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
