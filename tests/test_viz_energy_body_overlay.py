"""Tests for the hero shot-cloud overlay renderers.

Scope: shape + file-creation + metadata-formatting checks for the
two new renderers in :mod:`shotcloud.viz.energy_body_overlay`. We
don't pixel-compare the PNGs (matplotlib output is platform-noisy);
we just verify the figures are produced without error and that the
title-block / sidecar metadata round-trips cleanly.
"""

from __future__ import annotations

# Headless backend before any pyplot import (the viz module imports it).
import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pytest

from shotcloud.viz.energy_body_overlay import (
    GameMetadata,
    OverlayPaletteConfig,
    _bilinear_at_points,
    _bootstrap_resamples,
    render_predicted_vs_bootstrap,
    render_predicted_with_observed_shots,
)


@pytest.fixture
def fake_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    # Predicted cloud: tight bump near the rim.
    pred = np.stack(
        [
            rng.normal(0, 4, size=400),
            rng.normal(8, 5, size=400).clip(-5, 47),
        ],
        axis=1,
    )
    # Observed shots: 14 points near the rim and the right corner.
    obs_rim = np.stack(
        [rng.normal(0, 3, size=8), rng.normal(3, 2, size=8).clip(-5, 47)],
        axis=1,
    )
    obs_corner = np.stack(
        [rng.normal(22, 1, size=6), rng.normal(5, 1, size=6).clip(-5, 47)],
        axis=1,
    )
    obs = np.concatenate([obs_rim, obs_corner], axis=0).astype(np.float32)
    return {"predicted": pred.astype(np.float32), "observed": obs}


@pytest.fixture
def fake_metadata() -> GameMetadata:
    return GameMetadata(
        player_name="Test Player",
        away_team="LAL",
        home_team="BOS",
        date="2024-03-15",
        opponent="BOS",
        n_shots=14,
        model_checkpoint="test_ckpt",
        selection_rule="largest_shot_count",
        energy_distance=1.85,
        zone_l1=0.31,
        rim_distance_w1=2.40,
    )


def test_render_overlay_writes_png(
    tmp_path: Path, fake_data: dict[str, np.ndarray], fake_metadata: GameMetadata
) -> None:
    """Figure A produces a non-empty PNG."""
    out = tmp_path / "overlay.png"
    written = render_predicted_with_observed_shots(
        predicted_xy=fake_data["predicted"],
        observed_xy=fake_data["observed"],
        output_path=out,
        metadata=fake_metadata,
    )
    assert written.exists()
    assert written.stat().st_size > 0


def test_render_bootstrap_writes_png(
    tmp_path: Path, fake_data: dict[str, np.ndarray], fake_metadata: GameMetadata
) -> None:
    """Figure B produces a non-empty PNG with the dual-palette body."""
    out = tmp_path / "bootstrap.png"
    written = render_predicted_vs_bootstrap(
        predicted_xy=fake_data["predicted"],
        observed_xy=fake_data["observed"],
        output_path=out,
        metadata=fake_metadata,
        n_bootstraps=20,
        bootstrap_seed=0,
    )
    assert written.exists()
    assert written.stat().st_size > 0


def test_render_overlay_with_2d_companion(
    tmp_path: Path, fake_data: dict[str, np.ndarray], fake_metadata: GameMetadata
) -> None:
    """The 2D companion panel adds a second axes; the output should
    be wider than the 3D-only variant."""
    out_solo = tmp_path / "solo.png"
    render_predicted_with_observed_shots(
        predicted_xy=fake_data["predicted"],
        observed_xy=fake_data["observed"],
        output_path=out_solo,
        metadata=fake_metadata,
        companion_2d=False,
    )
    out_dual = tmp_path / "dual.png"
    render_predicted_with_observed_shots(
        predicted_xy=fake_data["predicted"],
        observed_xy=fake_data["observed"],
        output_path=out_dual,
        metadata=fake_metadata,
        companion_2d=True,
    )
    # Both written; 2D companion should produce a larger file (more
    # rendered pixels).
    assert out_solo.exists() and out_dual.exists()
    assert out_dual.stat().st_size > out_solo.stat().st_size


def test_palette_config_defaults_are_complementary() -> None:
    """Default palette pairs a warm cmap (predicted) with a cool
    cmap (observed) so the dual-cloud figure remains visually
    decodable."""
    p = OverlayPaletteConfig()
    assert p.predicted_cmap == "plasma"
    assert p.observed_cmap == "cividis"
    # Bootstrap alphas should be capped (per user spec: <=0.4).
    assert max(p.observed_shell_alphas) <= 0.4


def test_metadata_title_block_includes_all_fields(
    fake_metadata: GameMetadata,
) -> None:
    """Title block carries every paper-grade metadata line."""
    title, subtitle, extras = fake_metadata.title_block()
    assert "Test Player" in title
    assert "LAL @ BOS" in title
    assert "2024-03-15" in subtitle
    assert "Shots: 14" in subtitle
    assert "test_ckpt" in subtitle
    assert "Opponent: BOS" in extras
    assert "Split: validation" in extras
    assert "Energy:" in extras
    assert "Zone L1:" in extras
    assert "Rim W1:" in extras


def test_metadata_dict_roundtrip(fake_metadata: GameMetadata) -> None:
    """Metadata to_dict produces a JSON-serializable mapping."""
    import json

    d = fake_metadata.to_dict()
    s = json.dumps(d)
    back = json.loads(s)
    assert back["player_name"] == "Test Player"
    assert back["matchup"] == "LAL @ BOS"
    assert back["energy_distance"] == pytest.approx(1.85)


def test_bilinear_at_points_endpoint_consistency() -> None:
    """Bilinear interp at grid-cell-center coords reproduces the
    grid value exactly."""
    density = np.array([[0.0, 1.0], [2.0, 3.0]])
    xedges = np.array([0.0, 1.0, 2.0])
    yedges = np.array([0.0, 1.0, 2.0])
    # Cell centers: x=0.5, 1.5; y=0.5, 1.5.
    # density[0,0]=0 is at (0.5, 0.5). density[1,1]=3 is at (1.5, 1.5).
    xs = np.array([0.5, 1.5, 0.5, 1.5])
    ys = np.array([0.5, 0.5, 1.5, 1.5])
    got = _bilinear_at_points(density, xedges, yedges, xs, ys)
    np.testing.assert_allclose(got, [0.0, 1.0, 2.0, 3.0])


def test_bilinear_at_points_clamps_oob() -> None:
    """Out-of-bounds points clamp to the nearest in-grid value."""
    density = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    xedges = np.array([0.0, 1.0, 2.0])
    yedges = np.array([0.0, 1.0, 2.0])
    # Far-OOB → clamped to boundary cell.
    xs = np.array([-100.0, 100.0])
    ys = np.array([0.5, 1.5])
    got = _bilinear_at_points(density, xedges, yedges, xs, ys)
    # Clamped to xcenter[0]=0.5 (left) and xcenter[-1]=1.5 (right).
    np.testing.assert_allclose(got, [0.0, 3.0])


def test_bootstrap_resamples_shape() -> None:
    """Bootstrap returns ``n_bootstraps * K`` rows in (?, 2)."""
    obs = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    out = _bootstrap_resamples(obs, n_bootstraps=10, seed=0)
    assert out.shape == (30, 2)
    # All resampled points must be one of the originals.
    for row in out:
        assert any(np.allclose(row, o) for o in obs)


def test_bootstrap_empty_observed_passthrough() -> None:
    """Empty observed → empty bootstrap (the dual renderer falls
    back to predicted-only in this case)."""
    empty = np.empty((0, 2))
    out = _bootstrap_resamples(empty, n_bootstraps=10, seed=0)
    assert out.shape == (0, 2)
