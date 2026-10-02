"""Tests for the torch-vectorized zone_from_xy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shotcloud.data.zones import (
    zone_from_xy,
    zone_from_xy_torch,
    zone_from_xy_vectorized,
)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"must share shape"):
        zone_from_xy_torch(torch.zeros(3), torch.zeros(4))


def test_matches_numpy_on_grid() -> None:
    """The torch version must produce identical labels to
    ``zone_from_xy_vectorized`` on a regular grid spanning the court."""
    xs = np.linspace(-26.0, 26.0, 60, dtype=np.float64)
    ys = np.linspace(-6.0, 48.0, 60, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    np_out = zone_from_xy_vectorized(gx, gy)
    t_out = zone_from_xy_torch(torch.from_numpy(gx), torch.from_numpy(gy)).numpy()
    np.testing.assert_array_equal(np_out, t_out)


def test_matches_scalar_on_random_points() -> None:
    """1000 random points should give the same zone in both forms."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(-27.0, 27.0, size=1000)
    ys = rng.uniform(-7.0, 49.0, size=1000)
    scalar = np.array([zone_from_xy(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])
    torch_out = zone_from_xy_torch(torch.from_numpy(xs), torch.from_numpy(ys)).numpy()
    np.testing.assert_array_equal(scalar, torch_out)


def test_zone_specific_landmarks() -> None:
    """Hand-picked points within each zone (avoiding the boundaries
    that the B1 centroid choices sit on) must assign correctly."""
    points = torch.tensor(
        [
            (0.0, 1.0),  # RA
            (0.0, 7.5),  # paint
            (0.0, 16.0),  # midrange (avoid y=15 paint boundary)
            (-23.0, 5.0),  # corner-3 L
            (23.0, 5.0),  # corner-3 R
            (-18.0, 22.0),  # wing-3 L
            (18.0, 22.0),  # wing-3 R
            (0.0, 26.0),  # top-key 3
        ]
    )
    expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
    zones = zone_from_xy_torch(points[:, 0], points[:, 1])
    assert torch.equal(zones, expected), f"got {zones.tolist()} expected {expected.tolist()}"


def test_out_of_bounds_returns_neg1() -> None:
    pts = torch.tensor(
        [
            [0.0, 50.0],  # past halfcourt
            [0.0, -6.0],  # past baseline
            [-26.0, 0.0],  # past left sideline
            [26.0, 0.0],  # past right sideline
        ]
    )
    out = zone_from_xy_torch(pts[:, 0], pts[:, 1])
    assert (out == -1).all()
