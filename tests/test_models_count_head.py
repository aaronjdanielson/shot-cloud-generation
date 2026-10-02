"""Tests for :class:`shotcloud.models.NegBinCountHead`, the negative-binomial count factor.

Covers output shapes and positivity of ``(μ, κ)``, agreement with an explicit
:class:`torch.distributions.NegativeBinomial`, gradient flow, numerical stability at tiny
``μ``, the context-only signature, and input validation.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch.distributions import NegativeBinomial

from shotcloud import NegBinCountHead
from shotcloud.data.context import CONTEXT_DIM


def test_forward_shape_and_positivity() -> None:
    head = NegBinCountHead()
    x_n = torch.randn(5, CONTEXT_DIM)
    mu, kappa = head(x_n)
    assert mu.shape == (5,)
    assert kappa.shape == (5,)
    assert (mu > 0).all()
    assert (kappa > 0).all()
    # κ is shared across rows (a single scalar broadcast).
    assert torch.allclose(kappa, kappa[0].expand_as(kappa))


def test_forward_rejects_wrong_input_dim() -> None:
    head = NegBinCountHead(context_dim=27)
    with pytest.raises(ValueError, match="context_dim=27"):
        head(torch.randn(5, 30))


def test_log_prob_shape_and_finite() -> None:
    head = NegBinCountHead()
    x_n = torch.randn(8, CONTEXT_DIM)
    K = torch.tensor([0, 1, 5, 10, 15, 20, 25, 50], dtype=torch.long)
    log_p = head.log_prob(K, x_n)
    assert log_p.shape == (8,)
    assert torch.isfinite(log_p).all()
    # Log-prob of any valid count is non-positive.
    assert (log_p <= 0).all()


def test_log_prob_matches_explicit_negative_binomial() -> None:
    """The internal (μ, κ) → (total_count, probs) conversion matches a directly
    constructed ``NegativeBinomial``."""
    head = NegBinCountHead()
    x_n = torch.randn(4, CONTEXT_DIM)
    K = torch.tensor([3, 7, 12, 20], dtype=torch.long)

    log_p_head = head.log_prob(K, x_n)

    # Reconstruct the conversion the head does internally.
    mu, kappa = head(x_n)
    probs = (mu / (mu + kappa)).clamp(max=1.0 - 1e-7)
    dist = NegativeBinomial(total_count=kappa, probs=probs)
    log_p_explicit = dist.log_prob(K.float())

    np.testing.assert_allclose(
        log_p_head.detach().numpy(), log_p_explicit.detach().numpy(), atol=1e-6
    )


def test_gradient_flows_to_mu_net_and_log_kappa() -> None:
    head = NegBinCountHead()
    x_n = torch.randn(10, CONTEXT_DIM)
    K = torch.full((10,), 12, dtype=torch.long)
    nll = -head.log_prob(K, x_n).mean()
    nll.backward()

    # log_kappa receives a gradient.
    assert head.log_kappa.grad is not None
    assert head.log_kappa.grad.abs().item() > 0

    # MLP layers receive gradients on at least one parameter each.
    fc1_grad = head.fc1.weight.grad is not None and head.fc1.weight.grad.abs().sum().item() > 0
    fc2_grad = head.fc2.weight.grad is not None and head.fc2.weight.grad.abs().sum().item() > 0
    assert fc1_grad and fc2_grad


def test_kappa_is_contiguous() -> None:
    """``forward`` returns a contiguous ``kappa`` equal to ``softplus(log_kappa)``.

    ``softplus(log_kappa).expand_as(mu)`` alone is a stride-0 view, and the MPS backend
    of ``NegativeBinomial.log_prob`` returns ±Inf when ``total_count`` has stride 0
    (CPU and CUDA are unaffected), so the head materializes a real (B,) tensor.
    """
    head = NegBinCountHead()
    x_n = torch.randn(8, CONTEXT_DIM)
    _, kappa = head.forward(x_n)
    assert kappa.is_contiguous(), (
        f"kappa must be contiguous (MPS workaround); got stride={kappa.stride()}"
    )
    # And the values should all equal softplus(log_kappa).
    expected = torch.nn.functional.softplus(head.log_kappa)
    torch.testing.assert_close(kappa, expected.expand(8), atol=0.0, rtol=0.0)


def test_log_prob_finite_grad_when_mu_is_tiny() -> None:
    """Log-prob and gradients stay finite when ``μ`` is near zero.

    In that limit the ``value * log(probs)`` term has a divergent gradient while
    ``d(probs)/d(kappa) ≈ 0``, giving ``inf * 0 = NaN`` without the clamp in
    :meth:`NegBinCountHead.log_prob`.
    """
    head = NegBinCountHead(init_log_kappa=0.0)
    # Force fc2 to drive log_mu very negative so softplus(log_mu) is
    # essentially zero; this puts probs at the lower edge of the
    # clamped interval.
    with torch.no_grad():
        head.fc2.weight.zero_()
        head.fc2.bias.fill_(-50.0)  # softplus(-50) ≈ 1.9e-22
    x_n = torch.randn(8, CONTEXT_DIM)
    K = torch.tensor([1, 5, 10, 20, 0, 3, 15, 8], dtype=torch.long)
    log_p = head.log_prob(K, x_n)
    assert torch.isfinite(log_p).all(), f"log_prob has non-finite values: {log_p}"
    loss = -log_p.mean()
    loss.backward()
    assert head.log_kappa.grad is not None
    assert torch.isfinite(head.log_kappa.grad), (
        f"log_kappa grad is non-finite: {head.log_kappa.grad}"
    )
    assert head.fc2.bias.grad is not None
    assert torch.isfinite(head.fc2.bias.grad).all()


def test_no_player_idx_in_signature() -> None:
    """``forward`` takes only ``x_n``: the count head is context-only."""
    sig = inspect.signature(NegBinCountHead.forward)
    params = set(sig.parameters.keys())
    assert "player_idx" not in params
    assert params == {"self", "x_n"}


def test_constant_input_gives_constant_mu() -> None:
    """Identical ``x_n`` rows produce identical ``μ`` and ``κ``."""
    head = NegBinCountHead()
    head.eval()
    x_n = torch.zeros(8, CONTEXT_DIM)
    mu, kappa = head(x_n)
    np.testing.assert_allclose(mu.detach().numpy(), mu[0].detach().expand_as(mu).numpy(), atol=1e-6)
    np.testing.assert_allclose(
        kappa.detach().numpy(), kappa[0].detach().expand_as(kappa).numpy(), atol=1e-7
    )


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="context_dim"):
        NegBinCountHead(context_dim=0)
    with pytest.raises(ValueError, match="hidden_dim"):
        NegBinCountHead(hidden_dim=0)


def test_log_prob_shape_validation() -> None:
    head = NegBinCountHead()
    x_n = torch.randn(4, CONTEXT_DIM)
    with pytest.raises(ValueError, match="K must have shape"):
        head.log_prob(torch.tensor([1, 2]), x_n)  # mismatched batch
    with pytest.raises(ValueError, match="K must have shape"):
        head.log_prob(torch.zeros(4, 2), x_n)  # wrong dim
