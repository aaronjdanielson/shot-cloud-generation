"""Tests for ``shotcloud.simulation.sampler``."""

from __future__ import annotations

import pytest
import torch

from shotcloud.simulation.sampler import (
    DEFAULT_COURT_XLIM,
    DEFAULT_COURT_YLIM,
    sample_locations,
)


def _uniform_log_weights(b: int, m: int) -> torch.Tensor:
    logits = torch.zeros(b, m)
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def test_output_shape_and_inside_court() -> None:
    torch.manual_seed(0)
    b, m, n = 3, 8, 5
    support_xy = torch.tensor([[[0.0, 5.0]] * m, [[10.0, 20.0]] * m, [[-15.0, 3.0]] * m])
    log_w = _uniform_log_weights(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    sigma = torch.full((b,), 1.5)
    out = sample_locations(support_xy, log_w, mask, sigma, n_samples=n)
    assert out.shape == (b, n, 2)
    x_lo, x_hi = DEFAULT_COURT_XLIM
    y_lo, y_hi = DEFAULT_COURT_YLIM
    assert (out[..., 0] >= x_lo).all() and (out[..., 0] <= x_hi).all()
    assert (out[..., 1] >= y_lo).all() and (out[..., 1] <= y_hi).all()


def test_concentrated_omega_samples_near_chosen_support() -> None:
    """When ω is concentrated on a single support shot, samples
    cluster around it within ~3σ."""
    torch.manual_seed(0)
    b, m, n = 1, 5, 200
    support_xy = torch.tensor([[[-15.0, 5.0], [0.0, 25.0], [15.0, 5.0], [0.0, 8.0], [10.0, 15.0]]])
    # Mass entirely on slot 1 (0, 25).
    log_w = torch.full((b, m), -50.0)
    log_w[0, 1] = 0.0
    log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
    mask = torch.ones(b, m, dtype=torch.bool)
    sigma = torch.full((b,), 1.5)
    out = sample_locations(support_xy, log_w, mask, sigma, n_samples=n)
    # Mean of samples should be near (0, 25).
    mean = out[0].mean(dim=0)
    assert abs(float(mean[0]) - 0.0) < 0.5
    assert abs(float(mean[1]) - 25.0) < 0.5


def test_clip_fallback_keeps_samples_inside_court_even_for_edge_centers() -> None:
    """A support shot at the court boundary + a fat σ could in
    principle put samples outside on every attempt. The clip
    fallback guarantees the return is always inside the court."""
    torch.manual_seed(0)
    b, m, n = 1, 1, 100
    # Center exactly on the right sideline; huge σ → most draws are off-court.
    support_xy = torch.tensor([[[25.0, 5.0]]])
    log_w = torch.zeros(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    sigma = torch.full((b,), 10.0)
    out = sample_locations(support_xy, log_w, mask, sigma, n_samples=n, max_attempts=2)
    x_lo, x_hi = DEFAULT_COURT_XLIM
    y_lo, y_hi = DEFAULT_COURT_YLIM
    assert (out[..., 0] >= x_lo).all() and (out[..., 0] <= x_hi).all()
    assert (out[..., 1] >= y_lo).all() and (out[..., 1] <= y_hi).all()


def test_deterministic_with_generator() -> None:
    b, m, n = 2, 6, 4
    support_xy = torch.randn(b, m, 2) * 5.0
    log_w = _uniform_log_weights(b, m)
    mask = torch.ones(b, m, dtype=torch.bool)
    sigma = torch.full((b,), 1.5)
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    out1 = sample_locations(support_xy, log_w, mask, sigma, n_samples=n, generator=gen1)
    out2 = sample_locations(support_xy, log_w, mask, sigma, n_samples=n, generator=gen2)
    torch.testing.assert_close(out1, out2)


def test_cold_start_row_samples_without_error() -> None:
    """A row with all log_omega == -inf would otherwise crash
    multinomial. The sampler patches it with a uniform-on-slot-0
    fallback so the call is safe; the caller is expected to filter
    cold rows from downstream aggregation."""
    b, m, n = 2, 4, 3
    support_xy = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]] * b)
    log_w = torch.zeros(b, m)
    log_w[1] = float("-inf")  # row 1 cold
    mask = torch.ones(b, m, dtype=torch.bool)
    mask[1] = False
    sigma = torch.full((b,), 1.0)
    out = sample_locations(support_xy, log_w, mask, sigma, n_samples=n)
    assert out.shape == (b, n, 2)
    assert torch.isfinite(out).all()


def test_rejects_invalid_shapes() -> None:
    support_xy = torch.zeros(2, 5, 2)
    log_w = torch.zeros(2, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    sigma = torch.ones(2)
    with pytest.raises(ValueError, match="support_xy must be"):
        sample_locations(torch.zeros(2, 5), log_w, mask, sigma, n_samples=1)
    with pytest.raises(ValueError, match="log_omega must be"):
        sample_locations(support_xy, torch.zeros(2, 4), mask, sigma, n_samples=1)
    with pytest.raises(ValueError, match="sigma must be"):
        sample_locations(support_xy, log_w, mask, torch.ones(3), n_samples=1)
    with pytest.raises(ValueError, match="n_samples must be positive"):
        sample_locations(support_xy, log_w, mask, sigma, n_samples=0)
