"""Tests for :class:`shotcloud.models.NegBinCountHead` (paper §5).

Load-bearing invariants:

1. **Shape contract.** Forward returns ``(μ, κ)`` both shape ``(B,)``,
   both positive (post-softplus). ``log_prob(K, x_n)`` returns ``(B,)``.
2. **Numerical sanity.** Log-likelihood is finite for typical
   non-negative integer counts at default init.
3. **Distribution agreement.** Internal conversion to torch's
   ``(total_count, probs)`` matches an explicit
   :class:`torch.distributions.NegativeBinomial` constructed from
   the same ``(μ, κ)``.
4. **Gradient flow.** The MLP weights and ``log_kappa`` both
   receive nonzero gradients from the log-likelihood.
5. **No `player_idx` parameter.** Architectural — the count head is
   context-only, mirroring the residual decoder.
6. **Construction validation.** Invalid ``context_dim``,
   ``hidden_dim`` raise.
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
    """Internal (μ, κ) → (total_count, probs) conversion produces
    the same log-likelihood as constructing NegativeBinomial directly."""
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
    """Load-bearing on MPS: ``softplus(log_kappa).expand_as(mu)`` returns a
    stride-0 broadcast view of a scalar, and the MPS backend of
    ``torch.distributions.NegativeBinomial.log_prob`` produces ±Inf for
    nearly all batch entries when ``total_count`` has stride 0 (PyTorch
    MPS bug; CPU and CUDA are unaffected). The ``.contiguous()`` call
    inside :meth:`NegBinCountHead.forward` materializes a real (B,)
    tensor and works around the bug.

    This regression test pins the contract: ``kappa`` returned from
    ``forward`` must be contiguous so downstream consumers (NegBin,
    further reductions) see a normal-strided tensor regardless of
    device.
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
    """Stability check: when the predicted mean μ is near zero, the
    NegBin log_prob has a ``value * log(probs)`` term whose gradient
    ``value / probs`` diverges, and ``d(probs)/d(kappa) ≈ 0`` in the
    small-μ limit produces a classic ``inf * 0 = NaN`` backward.
    The clamp inside :meth:`NegBinCountHead.log_prob` must prevent
    this — a regression test for the failure surfaced by the first
    real-data ``train_gibbs`` run (2026-05-14).
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
    """Architectural check: the count head is context-only by design."""
    sig = inspect.signature(NegBinCountHead.forward)
    params = set(sig.parameters.keys())
    assert "player_idx" not in params
    assert params == {"self", "x_n"}


def test_constant_input_gives_constant_mu() -> None:
    """All-equal x_n rows must produce equal μ — sanity check on
    determinism + statelessness."""
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
