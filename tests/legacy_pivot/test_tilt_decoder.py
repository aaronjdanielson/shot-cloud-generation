"""Tests for :class:`shotcloud.legacy_pivot.tilt_decoder.LowRankTiltDecoder`.

The headline test is :func:`test_zero_init_reproduces_base_measure_exactly`,
which checks the load-bearing zero-init invariant: at ``V = 0`` the decoder
returns the base measure ``q_0`` exactly.
"""

from __future__ import annotations

import pytest
import torch

from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalized_log_q0(batch: int, n_cells: int, *, seed: int = 0) -> torch.Tensor:
    """Build a batch of valid log-probability vectors (each row sums to 1 in prob space)."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(batch, n_cells, generator=g)
    return torch.log_softmax(raw, dim=-1)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_default_construction_has_zero_init_v() -> None:
    decoder = LowRankTiltDecoder(n_cells=10, rank=4)
    assert decoder.n_cells == 10
    assert decoder.rank == 4
    assert decoder.zero_init is True
    assert torch.equal(decoder.V, torch.zeros(10, 4))


def test_random_init_v_is_nonzero() -> None:
    decoder = LowRankTiltDecoder(n_cells=10, rank=4, zero_init=False)
    assert decoder.zero_init is False
    assert (decoder.V != 0).any()
    # Glorot scale roughly 1/sqrt(rank) ≈ 0.5; std should be in that ballpark.
    std = decoder.V.std().item()
    assert 0.05 < std < 5.0


def test_v_is_a_parameter() -> None:
    decoder = LowRankTiltDecoder(n_cells=10, rank=4)
    assert isinstance(decoder.V, torch.nn.Parameter)
    assert decoder.V.requires_grad
    # Should appear in .parameters().
    params = list(decoder.parameters())
    assert any(p is decoder.V for p in params)


def test_zero_or_negative_n_cells_raises() -> None:
    with pytest.raises(ValueError, match="n_cells must be positive"):
        LowRankTiltDecoder(n_cells=0, rank=4)
    with pytest.raises(ValueError, match="n_cells must be positive"):
        LowRankTiltDecoder(n_cells=-1, rank=4)


def test_zero_or_negative_rank_raises() -> None:
    with pytest.raises(ValueError, match="rank must be positive"):
        LowRankTiltDecoder(n_cells=10, rank=0)
    with pytest.raises(ValueError, match="rank must be positive"):
        LowRankTiltDecoder(n_cells=10, rank=-1)


def test_v_has_expected_shape() -> None:
    decoder = LowRankTiltDecoder(n_cells=64 * 56, rank=8)
    assert decoder.V.shape == (64 * 56, 8)


# ---------------------------------------------------------------------------
# THE CRITICAL ZERO-INIT INVARIANT
# ---------------------------------------------------------------------------


def test_zero_init_reproduces_base_measure_exactly() -> None:
    """At zero-init, softmax(forward(log_q0, u)) must equal q_0 for any u.

    This load-bearing property lets training start from the KDE-product
    prior with no random distortion.
    """
    n_cells, rank, batch = 50, 4, 8
    decoder = LowRankTiltDecoder(n_cells=n_cells, rank=rank, zero_init=True)

    log_q0 = _normalized_log_q0(batch, n_cells, seed=0)
    q0 = torch.exp(log_q0)
    u = torch.randn(batch, rank)  # arbitrary, including non-zero

    probs = decoder.probs(log_q0, u)
    torch.testing.assert_close(probs, q0, atol=1e-6, rtol=1e-5)


def test_zero_init_log_probs_recover_log_q0_exactly() -> None:
    """Log-space version of the same invariant."""
    n_cells, rank, batch = 30, 4, 4
    decoder = LowRankTiltDecoder(n_cells=n_cells, rank=rank, zero_init=True)

    log_q0 = _normalized_log_q0(batch, n_cells, seed=42)
    u = torch.randn(batch, rank)

    log_probs = decoder.log_probs(log_q0, u)
    torch.testing.assert_close(log_probs, log_q0, atol=1e-5, rtol=1e-5)


def test_zero_init_tilt_is_zero() -> None:
    decoder = LowRankTiltDecoder(n_cells=10, rank=4, zero_init=True)
    u = torch.randn(2, 4)
    tilt = decoder.tilt(u)
    assert torch.equal(tilt, torch.zeros(2, 10))


def test_random_init_changes_output_from_q0() -> None:
    """Confirm the invariant test would actually catch a non-zero V."""
    n_cells, rank, batch = 20, 4, 3
    decoder = LowRankTiltDecoder(n_cells=n_cells, rank=rank, zero_init=False)

    log_q0 = _normalized_log_q0(batch, n_cells, seed=0)
    q0 = torch.exp(log_q0)
    u = torch.randn(batch, rank)

    probs = decoder.probs(log_q0, u)
    # Non-zero V + non-zero u → non-trivial tilt → output should differ from q_0.
    assert not torch.allclose(probs, q0, atol=1e-3)


# ---------------------------------------------------------------------------
# Output shapes and normalization
# ---------------------------------------------------------------------------


def test_forward_output_shape() -> None:
    decoder = LowRankTiltDecoder(n_cells=20, rank=4)
    log_q0 = _normalized_log_q0(7, 20)
    u = torch.randn(7, 4)
    out = decoder(log_q0, u)
    assert out.shape == (7, 20)


def test_probs_normalize_along_cells() -> None:
    decoder = LowRankTiltDecoder(n_cells=20, rank=4, zero_init=False)
    log_q0 = _normalized_log_q0(5, 20)
    u = torch.randn(5, 4)
    probs = decoder.probs(log_q0, u)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(5))


def test_log_probs_normalize_in_probability_space() -> None:
    decoder = LowRankTiltDecoder(n_cells=20, rank=4, zero_init=False)
    log_q0 = _normalized_log_q0(5, 20)
    u = torch.randn(5, 4)
    log_probs = decoder.log_probs(log_q0, u)
    torch.testing.assert_close(torch.exp(log_probs).sum(dim=-1), torch.ones(5))


def test_singleton_batch_works() -> None:
    decoder = LowRankTiltDecoder(n_cells=10, rank=2)
    log_q0 = _normalized_log_q0(1, 10)
    u = torch.randn(1, 2)
    probs = decoder.probs(log_q0, u)
    assert probs.shape == (1, 10)


# ---------------------------------------------------------------------------
# Shape mismatch errors
# ---------------------------------------------------------------------------


def test_wrong_log_q0_n_cells_raises() -> None:
    decoder = LowRankTiltDecoder(n_cells=10, rank=4)
    log_q0 = _normalized_log_q0(2, 11)  # wrong: should be 10
    u = torch.randn(2, 4)
    with pytest.raises(ValueError, match="log_q0 must have shape"):
        decoder(log_q0, u)


def test_wrong_u_rank_raises() -> None:
    decoder = LowRankTiltDecoder(n_cells=10, rank=4)
    log_q0 = _normalized_log_q0(2, 10)
    u = torch.randn(2, 5)  # wrong: should be 4
    with pytest.raises(ValueError, match="u must have shape"):
        decoder(log_q0, u)


def test_batch_dim_mismatch_raises() -> None:
    decoder = LowRankTiltDecoder(n_cells=10, rank=4)
    log_q0 = _normalized_log_q0(2, 10)
    u = torch.randn(3, 4)  # different batch size
    with pytest.raises(ValueError, match="batch dim mismatch"):
        decoder(log_q0, u)


def test_one_d_log_q0_raises() -> None:
    decoder = LowRankTiltDecoder(n_cells=10, rank=4)
    log_q0 = _normalized_log_q0(1, 10).squeeze(0)  # shape (10,) — needs to be (B, 10)
    u = torch.randn(1, 4)
    with pytest.raises(ValueError, match="log_q0 must have shape"):
        decoder(log_q0, u)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_gradients_flow_to_v() -> None:
    decoder = LowRankTiltDecoder(n_cells=10, rank=4, zero_init=True)
    log_q0 = _normalized_log_q0(4, 10, seed=0)
    u = torch.randn(4, 4, requires_grad=False)
    target = torch.tensor([3, 7, 1, 8])

    log_probs = decoder.log_probs(log_q0, u)
    loss = torch.nn.functional.nll_loss(log_probs, target)
    loss.backward()

    assert decoder.V.grad is not None
    assert decoder.V.grad.shape == decoder.V.shape
    assert (decoder.V.grad != 0).any()


def test_zero_init_step_zero_log_likelihood_equals_kde_log_likelihood() -> None:
    """At zero-init, NLL of the decoder equals NLL of the KDE prior alone.

    This is the practical version of the invariant: 'training starts from
    the KDE baseline'. The first-step loss should match what you'd get if
    you just used q_0 directly.
    """
    n_cells, rank, batch = 30, 4, 16
    decoder = LowRankTiltDecoder(n_cells=n_cells, rank=rank, zero_init=True)

    log_q0 = _normalized_log_q0(batch, n_cells, seed=99)
    u = torch.randn(batch, rank)
    target = torch.randint(0, n_cells, (batch,))

    decoder_nll = torch.nn.functional.nll_loss(decoder.log_probs(log_q0, u), target)
    base_nll = torch.nn.functional.nll_loss(log_q0, target)
    torch.testing.assert_close(decoder_nll, base_nll, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Import path
# ---------------------------------------------------------------------------


def test_top_level_import_works() -> None:
    """The class is importable from :mod:`shotcloud.legacy_pivot.tilt_decoder`."""
    from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder as TopLevel

    assert TopLevel is LowRankTiltDecoder
