"""Tests for the energy-body renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shotcloud import CourtGrid
from shotcloud.viz.energy_body import (
    EnergyBodyConfig,
    estimate_density,
    render_energy_body,
)

# A small config for fast tests — coarse grid, fewer particles, low DPI.
FAST_CONFIG = EnergyBodyConfig(
    grid=CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22),
    n_particles=500,
    n_halo=200,
    dpi=60,
    figsize=(4.0, 3.0),
)


def _fake_shots(n: int = 800) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    # Two clusters: paint and an above-break-3 arc-ish blob.
    x1 = rng.normal(0.0, 2.5, n // 2)
    y1 = rng.normal(4.0, 2.0, n // 2)
    x2 = rng.normal(0.0, 6.0, n - n // 2)
    y2 = rng.normal(24.0, 1.5, n - n // 2)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    return np.clip(x, -25, 25), np.clip(y, -5, 47)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults_match_canonical_hero_figure() -> None:
    """The default config must keep producing the canonical hero figure."""
    cfg = EnergyBodyConfig()
    assert cfg.grid.xlim == (-25.0, 25.0)
    assert cfg.grid.ylim == (-5.0, 47.0)
    assert (cfg.grid.nx, cfg.grid.ny) == (100, 104)
    assert cfg.density_power == pytest.approx(0.62)
    assert cfg.density_smoothing == pytest.approx(2.6)
    assert cfg.n_particles == 35_000
    assert cfg.n_halo == 15_000


def test_config_default_grid_is_independent_per_instance() -> None:
    """default_factory must produce a fresh CourtGrid per instance, not a shared one."""
    a = EnergyBodyConfig()
    b = EnergyBodyConfig()
    assert a.grid is not b.grid


def test_config_is_frozen() -> None:
    cfg = EnergyBodyConfig()
    # FrozenInstanceError is a subclass of Exception; dataclasses.FrozenInstanceError
    # would be the precise type, but Exception keeps the test independent of that import.
    with pytest.raises(Exception):  # noqa: B017
        cfg.dpi = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# estimate_density
# ---------------------------------------------------------------------------


def test_estimate_density_normalized() -> None:
    """Returned density grid must be normalized so its peak is 1.0."""
    x, y = _fake_shots(1000)
    density, height, gx, gy, (_xedges, _yedges) = estimate_density(x, y, FAST_CONFIG)

    assert density.shape == (FAST_CONFIG.grid.ny, FAST_CONFIG.grid.nx)
    assert height.shape == density.shape
    assert gx.shape == density.shape
    assert gy.shape == density.shape

    assert float(density.max()) == pytest.approx(1.0)
    assert float(density.min()) >= 0.0


def test_estimate_density_height_is_density_power() -> None:
    """height = density ** cfg.density_power, elementwise."""
    x, y = _fake_shots(500)
    density, height, _, _, _ = estimate_density(x, y, FAST_CONFIG)

    np.testing.assert_allclose(height, np.power(density, FAST_CONFIG.density_power))


def test_estimate_density_edges_cover_extent() -> None:
    x, y = _fake_shots(500)
    _, _, _, _, (xedges, yedges) = estimate_density(x, y, FAST_CONFIG)

    assert xedges[0] == pytest.approx(FAST_CONFIG.grid.xlim[0])
    assert xedges[-1] == pytest.approx(FAST_CONFIG.grid.xlim[1])
    assert yedges[0] == pytest.approx(FAST_CONFIG.grid.ylim[0])
    assert yedges[-1] == pytest.approx(FAST_CONFIG.grid.ylim[1])


def test_estimate_density_uses_default_config_when_omitted() -> None:
    x, y = _fake_shots(300)
    density_default, *_ = estimate_density(x, y)
    density_explicit, *_ = estimate_density(x, y, EnergyBodyConfig())
    np.testing.assert_array_equal(density_default, density_explicit)


# ---------------------------------------------------------------------------
# render_energy_body
# ---------------------------------------------------------------------------


def test_render_energy_body_writes_nonempty_png(tmp_path: Path) -> None:
    """The renderer should produce a non-empty PNG at the requested path."""
    x, y = _fake_shots(800)
    out = tmp_path / "cloud.png"
    written = render_energy_body(x, y, out, config=FAST_CONFIG)

    assert written == out
    assert out.exists()
    assert out.stat().st_size > 1024, f"output PNG suspiciously small: {out.stat().st_size} bytes"

    # PNG magic bytes.
    with out.open("rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_render_energy_body_deterministic_under_fixed_seed(tmp_path: Path) -> None:
    """Two renders with the same config and same input data should be byte-identical."""
    x, y = _fake_shots(800)
    out1 = tmp_path / "a.png"
    out2 = tmp_path / "b.png"

    render_energy_body(x, y, out1, config=FAST_CONFIG)
    render_energy_body(x, y, out2, config=FAST_CONFIG)

    # Matplotlib font hinting can introduce small nondeterminism in the
    # antialiased output, so compare file sizes within a generous tolerance
    # rather than exact bytes. The seed governs particle sampling, which is
    # the dominant source of variance.
    s1, s2 = out1.stat().st_size, out2.stat().st_size
    assert abs(s1 - s2) / max(s1, s2) < 0.02, f"renders differ in size by >2%: {s1} vs {s2} bytes"


def test_render_energy_body_accepts_no_annotation(tmp_path: Path) -> None:
    x, y = _fake_shots(400)
    out = tmp_path / "no_ann.png"
    render_energy_body(x, y, out, config=FAST_CONFIG, annotation=None)
    assert out.exists()
