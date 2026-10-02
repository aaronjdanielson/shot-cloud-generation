"""Tests for :class:`shotcloud.models.ContextMLP`."""

from __future__ import annotations

import pytest
import torch

from shotcloud import ContextMLP
from shotcloud.data.context import CONTEXT_DIM


def test_shape_residual_default() -> None:
    """Default residual mode preserves the input dimension."""
    mlp = ContextMLP(input_dim=CONTEXT_DIM)
    x = torch.randn(8, CONTEXT_DIM)
    out = mlp(x)
    assert out.shape == (8, CONTEXT_DIM)


def test_residual_init_is_identity() -> None:
    """At step 0 in residual mode, ``f_ctx(x) == x`` exactly.

    This is the load-bearing invariant: every downstream consumer of
    ``x_n`` reproduces its raw-feature behavior at init, so the trainer
    sees the same starting point as a model with no MLP.
    """
    torch.manual_seed(0)
    mlp = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=64, residual=True)
    mlp.eval()  # disable any potential dropout
    x = torch.randn(16, CONTEXT_DIM)
    with torch.no_grad():
        out = mlp(x)
    # Float32 exact equality: zero-init final layer makes h(x) all-zero,
    # so x + 0 == x bit-for-bit.
    assert torch.equal(out, x)


def test_non_residual_mode_returns_mlp_output() -> None:
    """``residual=False`` returns ``h(x)`` directly with standard init."""
    torch.manual_seed(0)
    mlp = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=32, output_dim=12, residual=False)
    x = torch.randn(4, CONTEXT_DIM)
    out = mlp(x)
    assert out.shape == (4, 12)
    # With non-residual + non-zero init, output is generally != input.
    # No exactness check; just verify it's not the all-zero tensor (which
    # would indicate the MLP collapsed at init).
    assert out.abs().sum() > 0


def test_residual_requires_matching_dims() -> None:
    """``residual=True`` with mismatched dims is a configuration error."""
    with pytest.raises(ValueError, match="residual=True requires output_dim == input_dim"):
        ContextMLP(input_dim=27, output_dim=32, residual=True)


def test_invalid_dimensions_raise() -> None:
    with pytest.raises(ValueError, match="input_dim must be positive"):
        ContextMLP(input_dim=0)
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        ContextMLP(input_dim=27, hidden_dim=0)
    with pytest.raises(ValueError, match="output_dim must be positive"):
        ContextMLP(input_dim=27, output_dim=-1, residual=False)
    with pytest.raises(ValueError, match="dropout must be in"):
        ContextMLP(input_dim=27, dropout=1.0)


def test_input_dim_mismatch_at_forward_raises() -> None:
    mlp = ContextMLP(input_dim=27)
    x_wrong = torch.randn(8, 30)
    with pytest.raises(ValueError, match="last-dim 30, expected input_dim=27"):
        mlp(x_wrong)


def test_gradient_flow_residual_first_then_full() -> None:
    """Residual + zero-init means fc1 gradient is zero on the very first
    backward (fc2.weight=0 blocks the chain rule), but non-zero after one
    optimizer step. fc2 always gets gradient on the first backward.
    """
    torch.manual_seed(0)
    mlp = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=16, residual=True)
    x = torch.randn(4, CONTEXT_DIM)
    target = torch.zeros(4, CONTEXT_DIM)

    out = mlp(x)
    loss = (out - target).pow(2).sum()
    loss.backward()

    # fc2 sees gradient (its weight enters as a coefficient on the activation).
    assert mlp.fc2.weight.grad is not None
    assert mlp.fc2.weight.grad.abs().sum() > 0
    # fc1's gradient is zero because the chain rule passes through fc2.weight==0.
    # This is correct behavior, not a bug --- once the optimizer perturbs fc2,
    # fc1 starts learning on subsequent backward passes.
    assert mlp.fc1.weight.grad is not None
    assert torch.allclose(mlp.fc1.weight.grad, torch.zeros_like(mlp.fc1.weight.grad))

    # One SGD step perturbs fc2; after that, fc1 receives gradient.
    optim = torch.optim.SGD(mlp.parameters(), lr=0.01)
    optim.step()
    optim.zero_grad()
    out2 = mlp(x)
    loss2 = (out2 - target).pow(2).sum()
    loss2.backward()
    assert mlp.fc1.weight.grad is not None
    assert mlp.fc1.weight.grad.abs().sum() > 0


def test_gradient_flow_non_residual_first_backward() -> None:
    """In non-residual mode the standard Kaiming init means both layers
    get gradient on the very first backward.
    """
    torch.manual_seed(0)
    mlp = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=16, output_dim=8, residual=False)
    x = torch.randn(4, CONTEXT_DIM)
    target = torch.randn(4, 8)
    loss = (mlp(x) - target).pow(2).sum()
    loss.backward()
    assert mlp.fc1.weight.grad is not None
    assert mlp.fc1.weight.grad.abs().sum() > 0
    assert mlp.fc2.weight.grad is not None
    assert mlp.fc2.weight.grad.abs().sum() > 0


def test_residual_diverges_from_identity_after_optimization() -> None:
    """After a single SGD step the residual is no longer identity."""
    torch.manual_seed(0)
    mlp = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=16, residual=True)
    x = torch.randn(4, CONTEXT_DIM)
    target = torch.zeros(4, CONTEXT_DIM)  # push output toward 0 ≠ input
    optim = torch.optim.SGD(mlp.parameters(), lr=0.1)

    optim.zero_grad()
    loss = (mlp(x) - target).pow(2).sum()
    loss.backward()
    optim.step()

    with torch.no_grad():
        out_after = mlp(x)
    # Output now differs from input.
    assert not torch.equal(out_after, x)


def test_3d_input_supported() -> None:
    """``ContextMLP`` accepts ``(B, T, D)`` inputs (e.g., per-shot histories)."""
    mlp = ContextMLP(input_dim=CONTEXT_DIM, hidden_dim=16, residual=True)
    mlp.eval()
    x = torch.randn(4, 7, CONTEXT_DIM)
    with torch.no_grad():
        out = mlp(x)
    assert out.shape == (4, 7, CONTEXT_DIM)
    # Residual + zero init → identity even on 3-D inputs.
    assert torch.equal(out, x)
