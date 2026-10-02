"""Tests for :class:`shotcloud.models.location_embedding.LocationEmbedding`.

Covers the zero-initialized projection (``ψ(s) ≡ 0`` at initialization, so the residual
tilt starts at zero), gradient flow to the projection and the input coordinates,
``(..., 2) → (..., rank)`` shape handling, and distinct embeddings for distinct
coordinates once the projection is nonzero.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shotcloud.models.location_embedding import LocationEmbedding


def test_zero_init_yields_zero_output_for_any_coordinate() -> None:
    enc = LocationEmbedding(rank=8)
    rng = np.random.default_rng(0)
    xy = torch.from_numpy(
        np.stack([rng.uniform(-25, 25, 50), rng.uniform(-5, 47, 50)], axis=-1).astype(np.float32)
    )
    with torch.no_grad():
        psi = enc(xy)
    assert psi.shape == (50, 8)
    torch.testing.assert_close(psi, torch.zeros_like(psi))


def test_output_shape_supports_arbitrary_leading_batch_shape() -> None:
    enc = LocationEmbedding(rank=6, zero_init=False)
    rng = np.random.default_rng(0)
    for shape in [(4, 2), (3, 5, 2), (2, 4, 7, 2)]:
        xy = torch.from_numpy(rng.normal(size=shape).astype(np.float32))
        psi = enc(xy)
        assert psi.shape == (*shape[:-1], 6)


def test_distinct_coords_produce_distinct_embeddings_when_proj_nonzero() -> None:
    """With nonzero projection weights, distinct coordinates map to distinct
    embeddings."""
    enc = LocationEmbedding(rank=8, zero_init=False)
    a = torch.tensor([[0.0, 5.0]])
    b = torch.tensor([[10.0, 20.0]])
    psi_a = enc(a)
    psi_b = enc(b)
    assert not torch.allclose(psi_a, psi_b)


def test_gradient_flows_to_proj_and_input_coords() -> None:
    enc = LocationEmbedding(rank=8, zero_init=False)
    xy = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    psi = enc(xy)
    psi.sum().backward()
    assert xy.grad is not None and xy.grad.abs().sum() > 0
    assert enc.proj.weight.grad is not None and enc.proj.weight.grad.abs().sum() > 0


def test_n_frequencies_zero_skips_fourier_features() -> None:
    """``n_frequencies=0`` reduces the embedding to a linear projection of
    ``(x_norm, y_norm)``."""
    enc = LocationEmbedding(rank=4, n_frequencies=0, zero_init=False)
    # Raw feature dim is 2 (just linear x/y).
    assert enc.proj.in_features == 2
    xy = torch.tensor([[5.0, 10.0]])
    psi = enc(xy)
    assert psi.shape == (1, 4)


def test_rejects_invalid_constructor_args() -> None:
    with pytest.raises(ValueError, match="rank"):
        LocationEmbedding(rank=0)
    with pytest.raises(ValueError, match="n_frequencies"):
        LocationEmbedding(rank=4, n_frequencies=-1)
    with pytest.raises(ValueError, match="x_scale"):
        LocationEmbedding(rank=4, x_scale=0.0)


def test_rejects_wrong_last_dim() -> None:
    enc = LocationEmbedding(rank=4)
    with pytest.raises(ValueError, match="last dim"):
        enc(torch.zeros(3, 3))


def test_normalization_brings_court_extent_close_to_unit_box() -> None:
    """The default scales map the court to ``|x_norm| ≤ 1`` and ``|y_norm| ≤ 47/26``.

    Fourier wavelengths are defined in this normalized space.
    """
    enc = LocationEmbedding(rank=4, zero_init=False)
    corners = torch.tensor([[-25.0, -5.0], [25.0, 47.0]])
    x_norm = corners[..., 0] / enc.x_scale
    y_norm = corners[..., 1] / enc.y_scale
    assert x_norm.abs().max().item() <= 1.0 + 1e-6
    assert y_norm.abs().max().item() <= 47.0 / 26.0 + 1e-6  # ~1.81
