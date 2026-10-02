"""Tests for :class:`shotcloud.models.ContextResidualEncoder`.

The encoder produces the context input ``u_θ`` of the low-rank residual
tilt ``R_θ`` (see *Support-logit components* in the paper). Invariants:

1. **Output shape contract.** ``forward(x_n) -> (B, rank)`` for any batch
   size, with ``context_dim`` validation on the input.
2. **No player identity.** ``forward`` takes ``x_n`` plus the optional causal
   channels ``h_n``, ``usage``, and ``outcome``, and no ``player_idx``; the
   residual cannot carry player identity because the encoder cannot receive
   it.
3. **Zero-init composition.** Paired with the grid-cell
   :class:`~shotcloud.legacy_pivot.tilt_decoder.LowRankTiltDecoder`, a zero
   basis ``V`` gives ``softmax(log q_0 + u^T V) == q_0`` exactly, whatever the
   encoder outputs. The encoder's own output is small but non-zero, so ``V``
   still receives gradient.
4. **Gradient flow** through both factors when ``V`` is non-zero.
5. **Optional channels.** The within-game channel ``h_n`` is validated and
   used when configured; the usage branch is an exact no-op at init.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from shotcloud import ContextResidualEncoder
from shotcloud.data.context import CONTEXT_DIM
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder


def test_forward_shape_contract() -> None:
    enc = ContextResidualEncoder(rank=8)
    x_n = torch.randn(5, CONTEXT_DIM)
    u = enc(x_n)
    assert u.shape == (5, 8)
    assert u.dtype == torch.float32


def test_forward_rejects_wrong_input_dim() -> None:
    enc = ContextResidualEncoder(rank=8, context_dim=27)
    bad = torch.randn(5, 30)  # wrong dim
    with pytest.raises(ValueError, match="context_dim=27"):
        enc(bad)


def test_no_player_idx_in_signature() -> None:
    """``forward`` has no ``player_idx`` parameter: the residual is context-only by construction."""
    import inspect

    sig = inspect.signature(ContextResidualEncoder.forward)
    params = set(sig.parameters.keys())
    assert "player_idx" not in params
    # x_n plus the optional causal channels: within-game history h_n,
    # usage state, and prior outcomes.
    assert params == {"self", "x_n", "h_n", "usage", "outcome"}


def test_zero_init_invariant_via_decoder() -> None:
    """With decoder basis ``V = 0``, the output equals ``q_0`` whatever the encoder outputs."""
    rank = 4
    n_cells = 50
    enc = ContextResidualEncoder(rank=rank)
    dec = LowRankTiltDecoder(n_cells=n_cells, rank=rank, zero_init=True)
    # Random (non-uniform) base measure that sums to 1.
    log_q0 = torch.log_softmax(torch.randn(3, n_cells), dim=-1)
    x_n = torch.randn(3, CONTEXT_DIM)
    u = enc(x_n)
    # Encoder is small-but-nonzero by default, so u != 0.
    assert u.abs().sum().item() > 0
    probs = dec.probs(log_q0, u)
    np.testing.assert_allclose(
        probs.detach().numpy(), torch.softmax(log_q0, dim=-1).detach().numpy(), atol=1e-6
    )


def test_init_default_is_small_but_nonzero() -> None:
    """The last-layer weight std is about ``1/sqrt(rank)``, so ``u`` is small but non-zero.

    A zero ``u`` together with a zero basis ``V`` would give
    ``∂(u^T V)/∂V = u = 0`` and ``V`` would never move.
    """
    rank = 16
    enc = ContextResidualEncoder(rank=rank)
    # Final layer std should be ~init_std_scale / sqrt(rank).
    std = enc.fc2.weight.std().item()
    expected = 1.0 / math.sqrt(rank)
    assert 0.3 * expected < std < 3.0 * expected
    # Output is nonzero on random input.
    x_n = torch.randn(8, CONTEXT_DIM)
    u = enc(x_n)
    assert u.abs().mean().item() > 0


def test_init_std_scale_smaller_means_smaller_output() -> None:
    """A smaller ``init_std_scale`` gives smaller outputs at step 0."""
    torch.manual_seed(0)
    enc_default = ContextResidualEncoder(rank=8, init_std_scale=1.0)
    torch.manual_seed(0)
    enc_small = ContextResidualEncoder(rank=8, init_std_scale=0.1)
    x_n = torch.randn(64, CONTEXT_DIM)
    u_default = enc_default(x_n)
    u_small = enc_small(x_n)
    assert u_small.abs().mean().item() < 0.5 * u_default.abs().mean().item()


def test_gradient_flows_through_encoder_and_decoder() -> None:
    """When V is nonzero, both encoder and decoder receive gradients."""
    rank = 4
    n_cells = 50
    enc = ContextResidualEncoder(rank=rank)
    dec = LowRankTiltDecoder(n_cells=n_cells, rank=rank, zero_init=False)
    log_q0 = torch.log_softmax(torch.randn(8, n_cells), dim=-1)
    x_n = torch.randn(8, CONTEXT_DIM, requires_grad=False)
    u = enc(x_n)
    logits = dec(log_q0, u)
    loss = -torch.log_softmax(logits, dim=-1).mean()
    loss.backward()
    assert dec.V.grad is not None and dec.V.grad.abs().sum().item() > 0
    enc_grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert enc_grads, "encoder received no gradient"
    assert any(g.abs().sum().item() > 0 for g in enc_grads)


def test_invalid_constructor_args_raise() -> None:
    with pytest.raises(ValueError, match="rank"):
        ContextResidualEncoder(rank=0)
    with pytest.raises(ValueError, match="rank"):
        ContextResidualEncoder(rank=-1)
    with pytest.raises(ValueError, match="context_dim"):
        ContextResidualEncoder(rank=8, context_dim=0)
    with pytest.raises(ValueError, match="hidden_dim"):
        ContextResidualEncoder(rank=8, hidden_dim=0)
    with pytest.raises(ValueError, match="init_std_scale"):
        ContextResidualEncoder(rank=8, init_std_scale=0.0)


def test_default_hidden_dim_is_capacity_limited() -> None:
    """The default encoder has fewer than 1500 parameters.

    The residual is kept capacity-limited so it cannot dominate the
    KDE-derived geometry.
    """
    enc = ContextResidualEncoder(rank=8)
    n_params = sum(p.numel() for p in enc.parameters())
    # 27 -> 32 -> 8: 27*32 + 32 + 32*8 + 8 = 864 + 32 + 256 + 8 = 1160 params
    assert n_params < 1500


# ---------------------------------------------------------------------------
# Within-game-history channel (h_n)
# ---------------------------------------------------------------------------


def test_within_game_dim_zero_default_rejects_h_n() -> None:
    """With the default ``within_game_dim=0``, passing ``h_n`` raises rather than being ignored."""
    import torch as torch_

    enc = ContextResidualEncoder(rank=4)
    x_n = torch_.randn(3, CONTEXT_DIM)
    h_n = torch_.randn(3, 10)
    with pytest.raises(ValueError, match="within_game_dim=0"):
        enc(x_n, h_n)


def test_within_game_dim_positive_requires_h_n() -> None:
    """With ``within_game_dim > 0``, omitting ``h_n`` raises rather than falling back to ``x_n``."""
    import torch as torch_

    enc = ContextResidualEncoder(rank=4, within_game_dim=10)
    x_n = torch_.randn(3, CONTEXT_DIM)
    with pytest.raises(ValueError, match="h_n is required"):
        enc(x_n)


def test_within_game_channel_changes_output_for_different_h_n() -> None:
    """Batches that share ``x_n`` but differ in ``h_n`` give different outputs."""
    import torch as torch_

    enc = ContextResidualEncoder(rank=4, within_game_dim=10)
    x_n = torch_.randn(2, CONTEXT_DIM)
    h_a = torch_.zeros(2, 10)
    h_b = torch_.randn(2, 10)
    u_a = enc(x_n, h_a)
    u_b = enc(x_n, h_b)
    assert not torch_.allclose(u_a, u_b)


def test_within_game_channel_gradient_flows_to_h_n() -> None:
    """Backward through ``u_θ(x_n, h_n)`` populates a non-zero ``h_n.grad``."""
    import torch as torch_

    enc = ContextResidualEncoder(rank=4, within_game_dim=10)
    x_n = torch_.randn(3, CONTEXT_DIM, requires_grad=False)
    h_n = torch_.randn(3, 10, requires_grad=True)
    u = enc(x_n, h_n)
    u.sum().backward()
    assert h_n.grad is not None
    assert h_n.grad.abs().sum().item() > 0


def test_within_game_dim_rejects_negative() -> None:
    with pytest.raises(ValueError, match="within_game_dim"):
        ContextResidualEncoder(rank=4, within_game_dim=-1)


def test_within_game_dim_rejects_shape_mismatch_in_forward() -> None:
    import torch as torch_

    enc = ContextResidualEncoder(rank=4, within_game_dim=10)
    x_n = torch_.randn(3, CONTEXT_DIM)
    h_bad = torch_.randn(3, 5)
    with pytest.raises(ValueError, match="within_game_dim=10"):
        enc(x_n, h_bad)


# --------------------------------------------------------------------------- #
# Causal usage-state branch
# --------------------------------------------------------------------------- #


def test_usage_branch_is_no_op_at_init_for_any_usage_vector() -> None:
    """At init, an encoder with a usage branch matches the no-usage encoder for any usage input.

    The usage MLP's output layer is zero-initialized, so enabling the
    channel leaves the step-0 model unchanged.
    """
    import torch as t

    t.manual_seed(0)
    rank = 8
    enc_no_usage = ContextResidualEncoder(rank=rank, within_game_dim=0, usage_dim=0)
    enc_with_usage = ContextResidualEncoder(rank=rank, within_game_dim=0, usage_dim=3)
    # Copy the base-branch weights so both encoders have an identical
    # u_base path; only the usage branch differs.
    enc_with_usage.fc1.weight.data.copy_(enc_no_usage.fc1.weight.data)
    enc_with_usage.fc1.bias.data.copy_(enc_no_usage.fc1.bias.data)
    enc_with_usage.fc2.weight.data.copy_(enc_no_usage.fc2.weight.data)
    enc_with_usage.fc2.bias.data.copy_(enc_no_usage.fc2.bias.data)

    B = 4
    x_n = t.randn(B, 27)
    usage = t.randn(B, 3) * 5.0  # extreme usage values
    out_no = enc_no_usage(x_n)
    out_with = enc_with_usage(x_n, usage=usage)
    t.testing.assert_close(out_with, out_no, atol=1e-6, rtol=0)


def test_usage_branch_diverges_after_param_shift() -> None:
    """Once the usage MLP leaves zero init, the output differs from the no-usage path."""
    import torch as t

    t.manual_seed(0)
    enc = ContextResidualEncoder(rank=8, within_game_dim=0, usage_dim=3)
    x_n = t.randn(3, 27)
    usage = t.tensor([[1.0, 0.0, -1.0], [-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
    out_zero_init = enc(x_n, usage=usage)
    # Move the usage MLP's last layer off zero.
    with t.no_grad():
        enc.usage_fc2.weight.normal_(std=0.5)
    out_moved = enc(x_n, usage=usage)
    assert not t.allclose(out_zero_init, out_moved, atol=1e-4)


def test_usage_dim_0_rejects_usage_argument() -> None:
    import pytest
    import torch as t

    enc = ContextResidualEncoder(rank=4, within_game_dim=0, usage_dim=0)
    x_n = t.randn(2, 27)
    with pytest.raises(ValueError, match="usage_dim=0"):
        enc(x_n, usage=t.zeros(2, 3))


def test_usage_dim_positive_requires_usage_argument() -> None:
    import pytest
    import torch as t

    enc = ContextResidualEncoder(rank=4, within_game_dim=0, usage_dim=3)
    x_n = t.randn(2, 27)
    with pytest.raises(ValueError, match="usage is required"):
        enc(x_n)


def test_usage_branch_gradient_flows_to_usage_mlp() -> None:
    """After perturbing the usage_fc2 layer off zero, gradient flow
    through the usage branch must reach the usage MLP parameters."""
    import torch as t

    t.manual_seed(0)
    enc = ContextResidualEncoder(rank=8, within_game_dim=0, usage_dim=3)
    # Move usage_fc2 off zero so the chain rule has a non-zero
    # path back to usage_fc1.
    with t.no_grad():
        enc.usage_fc2.weight.normal_(std=0.5)
    x_n = t.randn(3, 27, requires_grad=False)
    usage = t.randn(3, 3, requires_grad=False)
    out = enc(x_n, usage=usage)
    out.sum().backward()
    assert enc.usage_fc1.weight.grad is not None
    assert enc.usage_fc1.weight.grad.abs().sum() > 0
    assert enc.usage_fc2.weight.grad is not None
    assert enc.usage_fc2.weight.grad.abs().sum() > 0


def test_usage_dim_rejects_bad_usage_shape() -> None:
    import pytest
    import torch as t

    enc = ContextResidualEncoder(rank=4, within_game_dim=0, usage_dim=3)
    x_n = t.randn(3, 27)
    with pytest.raises(ValueError, match="usage must have shape"):
        enc(x_n, usage=t.zeros(3, 5))  # wrong usage dim
    with pytest.raises(ValueError, match="batch sizes must match"):
        enc(x_n, usage=t.zeros(2, 3))  # wrong batch dim
