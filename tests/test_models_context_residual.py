"""Tests for :class:`shotcloud.models.ContextResidualEncoder`.

The encoder is the context-only residual-tilt input ``u_θ(x_n)``
(paper §3.5). The load-bearing invariants:

1. **Output shape contract.** ``forward(x_n) -> (B, rank)`` for any
   batch size, with ``CONTEXT_DIM`` validation on the input.
2. **No player identity.** The forward signature is
   ``(x_n,) -> u`` only — there is no ``player_idx`` parameter. This
   is structural, not runtime: the residual cannot leak player
   identity because the encoder has no way to receive it.
3. **Composition with `LowRankTiltDecoder` preserves the zero-init
   invariant.** When the decoder's basis ``V`` is zero,
   ``softmax(log q_0 + u^T V) == q_0`` exactly, regardless of the
   encoder's output (which is small but nonzero by default,
   avoiding the dual-zero saddle).
4. **Gradient flows through both factors when V is nonzero.**
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
    """Architectural check: the residual is context-only by construction.

    Calling ``forward`` with the canonical context input must succeed
    without any player identity threaded through. Any future regression
    that adds a ``player_idx`` parameter would force a signature change
    and break this test.
    """
    import inspect

    sig = inspect.signature(ContextResidualEncoder.forward)
    params = set(sig.parameters.keys())
    assert "player_idx" not in params
    # Forward accepts (self, x_n) plus the optional within-game-history
    # channel h_n (added 2026-05-17 per paper §3.5), the optional
    # causal usage-state channel `usage` (added 2026-06-04 for the
    # count-location coupling ablation), and the optional causal
    # prior-outcome channel `outcome` (added 2026-06-07 Phase 2 of
    # the audit). Player identity is still architecturally excluded.
    assert params == {"self", "x_n", "h_n", "usage", "outcome"}


def test_zero_init_invariant_via_decoder() -> None:
    """When V=0 in the decoder, the full network output equals q_0
    exactly regardless of the encoder's output."""
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
    """Last-layer std should be ~1/sqrt(rank); output magnitude is small but nonzero.

    Avoids the dual-zero saddle: encoder zero + decoder V=0 would
    leave ∂(u^T V)/∂V = u = 0, so V never moves. The default init
    keeps u nonzero so V receives a gradient signal.
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
    """Decreasing init_std_scale should produce smaller outputs at step 0.

    Useful when the user wants a tighter ``u`` to keep the residual
    closer to zero in early training.
    """
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
    """The paper requires the residual to be capacity-limited so it
    cannot dominate the KDE-derived geometry. Default hidden_dim=32
    keeps the parameter count small."""
    enc = ContextResidualEncoder(rank=8)
    n_params = sum(p.numel() for p in enc.parameters())
    # 27 -> 32 -> 8: 27*32 + 32 + 32*8 + 8 = 864 + 32 + 256 + 8 = 1160 params
    assert n_params < 1500


# ---------------------------------------------------------------------------
# Within-game-history channel (h_n) — paper §3.5 short-horizon term
# ---------------------------------------------------------------------------


def test_within_game_dim_zero_default_rejects_h_n() -> None:
    """Default constructor (within_game_dim=0) preserves legacy
    behavior; passing an h_n tensor should error loudly so callers
    don't silently lose the short-horizon signal."""
    import torch as torch_

    enc = ContextResidualEncoder(rank=4)
    x_n = torch_.randn(3, CONTEXT_DIM)
    h_n = torch_.randn(3, 10)
    with pytest.raises(ValueError, match="within_game_dim=0"):
        enc(x_n, h_n)


def test_within_game_dim_positive_requires_h_n() -> None:
    """When the encoder is constructed with within_game_dim > 0, h_n
    is required. Forgetting to pass it should error, not silently
    fall back to x_n-only."""
    import torch as torch_

    enc = ContextResidualEncoder(rank=4, within_game_dim=10)
    x_n = torch_.randn(3, CONTEXT_DIM)
    with pytest.raises(ValueError, match="h_n is required"):
        enc(x_n)


def test_within_game_channel_changes_output_for_different_h_n() -> None:
    """The encoder must actually use h_n — if two batches share x_n
    but differ in h_n, the outputs should differ."""
    import torch as torch_

    enc = ContextResidualEncoder(rank=4, within_game_dim=10)
    x_n = torch_.randn(2, CONTEXT_DIM)
    h_a = torch_.zeros(2, 10)
    h_b = torch_.randn(2, 10)
    u_a = enc(x_n, h_a)
    u_b = enc(x_n, h_b)
    assert not torch_.allclose(u_a, u_b)


def test_within_game_channel_gradient_flows_to_h_n() -> None:
    """Backward through ``u_θ(x_n, h_n)`` should populate ``h_n.grad``
    when ``within_game_dim > 0`` — the short-horizon channel is
    actually carrying gradient, not just being read."""
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
# Causal usage-state branch (2026-06-04)
# --------------------------------------------------------------------------- #


def test_usage_branch_is_no_op_at_init_for_any_usage_vector() -> None:
    """**Load-bearing invariant.** With ``usage_dim > 0`` and the
    usage MLP's output layer zero-init, the encoder must be bit-
    identical to the no-usage encoder at step 0 — for *every* usage
    input. This is the equivalent of the Tier-1a / Tier-2 isotropic-
    collapse invariant for the residual axis: turning the channel on
    cannot disturb the existing mainline at step 0.
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
    """After moving the usage MLP off zero-init, the encoder's
    output must differ from the no-usage path — confirming the usage
    signal actually reaches the output."""
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
