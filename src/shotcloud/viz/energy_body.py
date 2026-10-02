"""3D "energy body" renderer for shot clouds.

Renders a smoothed 2D shot density on the NBA half-court as a layered
volumetric plasma surface plus density-proportional particles and a halo.

Coordinates are in feet. Default extent matches the package convention:
    x ∈ [-25, 25],  y ∈ [-5, 47]   (basket at origin).

Public API
----------
:class:`EnergyBodyConfig`
    Frozen dataclass of rendering parameters.
:func:`estimate_density`
    Smoothed 2D density grid (used by other viz modules too).
:func:`render_energy_body`
    Top-level renderer; writes a PNG and returns its path.

Adapted from ``plotting_instructions/shot_cloud_energy_body_final.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.patches import Arc, Circle, Rectangle
from mpl_toolkits.mplot3d import art3d
from scipy.ndimage import gaussian_filter

from shotcloud.grids import CourtGrid


def _default_render_grid() -> CourtGrid:
    """Canonical render-resolution grid (100x104 over the full half-court).

    Higher resolution than the model grid (typically 64x56) because the
    renderer wants smooth surfaces; the model wants a tractable softmax.
    """
    return CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=100, ny=104)


@dataclass(frozen=True)
class EnergyBodyConfig:
    """Rendering parameters for :func:`render_energy_body`.

    The ``grid`` field carries the spatial discretization (extent + resolution).
    Other fields control the volumetric rendering style. Defaults reproduce
    the canonical hero figure.
    """

    grid: CourtGrid = field(default_factory=_default_render_grid)
    density_smoothing: float = 2.6
    density_power: float = 0.62
    n_particles: int = 35_000
    n_halo: int = 15_000
    elev: float = 31.0
    azim: float = -64.0
    figsize: tuple[float, float] = (13.0, 9.0)
    dpi: int = 180
    figure_bg: str = "#EEF1F7"
    axes_bg: str = "#F7F8FC"
    text_color: str = "#1C2230"
    court_color: str = "#2D3340"
    seed: int = 101


# ---------------------------------------------------------------------------
# Density estimation
# ---------------------------------------------------------------------------


def estimate_density(
    x: np.ndarray,
    y: np.ndarray,
    cfg: EnergyBodyConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Smoothed normalized 2D density grid.

    Parameters
    ----------
    x, y : np.ndarray
        Shot coordinates in feet, basket at origin. Same length, non-empty.
    cfg : EnergyBodyConfig, optional
        Provides ``grid`` (a :class:`~shotcloud.grids.CourtGrid`),
        ``density_smoothing``, and ``density_power``. Defaults are used when
        not provided.

    Returns
    -------
    density : np.ndarray, shape ``(ny, nx)``
        Normalized to ``density.max() == 1`` (or ``0`` if input is empty).
    Z : np.ndarray, shape ``(ny, nx)``
        ``density ** density_power`` — the rendering height field.
    X, Y : np.ndarray, shape ``(ny, nx)``
        Cell-center coordinate grids.
    edges : tuple of (xedges, yedges)
        Histogram bin edges.
    """
    cfg = cfg or EnergyBodyConfig()
    g = cfg.grid

    hist, _, _ = np.histogram2d(x, y, bins=[g.xedges, g.yedges], range=[g.xlim, g.ylim])
    density = gaussian_filter(hist.T, sigma=cfg.density_smoothing)
    peak = float(density.max())
    density = density / max(peak, 1e-12)

    grid_x, grid_y = g.mesh
    height = np.power(density, cfg.density_power)

    return density, height, grid_x, grid_y, (g.xedges, g.yedges)


# ---------------------------------------------------------------------------
# Court drawing on z=0 plane
# ---------------------------------------------------------------------------


def _add_patch_3d(ax: Any, patch: Any, z: float = 0.0) -> None:
    ax.add_patch(patch)
    art3d.pathpatch_2d_to_3d(patch, z=z, zdir="z")


def _draw_halfcourt(
    ax: Any, cfg: EnergyBodyConfig, z: float = 0.0, lw: float = 1.05, alpha: float = 0.72
) -> None:
    """Draw NBA half-court lines on the z=0 plane."""
    color = cfg.court_color

    _add_patch_3d(
        ax, Rectangle((-25, -5), 50, 52, fill=False, edgecolor=color, linewidth=lw, alpha=alpha), z
    )
    _add_patch_3d(
        ax, Circle((0, 0), 0.75, fill=False, edgecolor=color, linewidth=lw, alpha=alpha), z
    )
    ax.plot([-3, 3], [-1.25, -1.25], [z, z], color=color, lw=lw, alpha=alpha)

    _add_patch_3d(
        ax, Rectangle((-8, -5), 16, 19, fill=False, edgecolor=color, linewidth=lw, alpha=alpha), z
    )
    _add_patch_3d(
        ax,
        Rectangle(
            (-6, -5),
            12,
            19,
            fill=False,
            edgecolor=color,
            linewidth=lw * 0.75,
            alpha=alpha * 0.72,
        ),
        z,
    )

    _add_patch_3d(
        ax,
        Arc((0, 14), 12, 12, theta1=0, theta2=180, edgecolor=color, linewidth=lw, alpha=alpha),
        z,
    )
    _add_patch_3d(
        ax,
        Arc(
            (0, 14),
            12,
            12,
            theta1=180,
            theta2=360,
            edgecolor=color,
            linewidth=lw,
            alpha=alpha * 0.42,
            linestyle="--",
        ),
        z,
    )
    _add_patch_3d(
        ax,
        Arc((0, 0), 8, 8, theta1=0, theta2=180, edgecolor=color, linewidth=lw, alpha=alpha),
        z,
    )

    ax.plot([-22, -22], [-5, 14], [z, z], color=color, lw=lw, alpha=alpha)
    ax.plot([22, 22], [-5, 14], [z, z], color=color, lw=lw, alpha=alpha)
    _add_patch_3d(
        ax,
        Arc((0, 0), 47.5, 47.5, theta1=22, theta2=158, edgecolor=color, linewidth=lw, alpha=alpha),
        z,
    )


# ---------------------------------------------------------------------------
# Energy shells (translucent surfaces)
# ---------------------------------------------------------------------------


def _render_energy_shells(
    ax: Any, grid_x: np.ndarray, grid_y: np.ndarray, height: np.ndarray, density: np.ndarray
) -> None:
    """Stack of nested translucent shells plus a brighter canopy."""
    shell_offsets = [0.00, -0.03, -0.06, -0.09]
    shell_alphas = [0.24, 0.19, 0.15, 0.11]

    magma = colormaps["magma"]
    plasma = colormaps["plasma"]

    for offset, alpha in zip(shell_offsets, shell_alphas, strict=True):
        z_shell = np.maximum(height + offset, 0)
        colors = magma(z_shell)
        colors[..., :3] = colors[..., :3] * 0.55 + 0.45
        colors[..., -1] = np.clip(z_shell * alpha, 0, alpha)

        ax.plot_surface(
            grid_x,
            grid_y,
            z_shell,
            facecolors=colors,
            linewidth=0,
            antialiased=True,
            shade=False,
        )

    ridge = np.power(density, 0.38)
    colors = plasma(ridge)
    colors[..., :3] = colors[..., :3] * 0.50 + 0.50
    colors[..., -1] = np.clip(ridge * 0.38, 0, 0.42)

    ax.plot_surface(
        grid_x,
        grid_y,
        height * 1.02,
        facecolors=colors,
        linewidth=0,
        antialiased=True,
        shade=False,
    )


# ---------------------------------------------------------------------------
# Density-proportional particle clouds
# ---------------------------------------------------------------------------


def _sample_density_particles(
    density: np.ndarray,
    height: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    n: int,
    seed: np.random.SeedSequence | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``n`` particles with probability proportional to ``density``."""
    rng = np.random.default_rng(seed)

    probs = density.ravel()
    total = probs.sum()
    if total <= 0:
        raise ValueError("density grid sums to zero; cannot sample particles")
    probs = probs / total

    ids = rng.choice(len(probs), size=n, p=probs)
    grid_iy, grid_ix = np.unravel_index(ids, density.shape)

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    px = xcenters[grid_ix] + rng.normal(0, dx * 0.45, n)
    py = ycenters[grid_iy] + rng.normal(0, dy * 0.45, n)
    pz = height[grid_iy, grid_ix] * rng.beta(0.85, 2.3, n)

    return px, py, pz, density[grid_iy, grid_ix], height[grid_iy, grid_ix]


def _render_particles(
    ax: Any,
    density: np.ndarray,
    height: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    cfg: EnergyBodyConfig,
    seed: np.random.SeedSequence | int,
) -> None:
    px, py, pz, dval, _ = _sample_density_particles(
        density, height, xedges, yedges, cfg.n_particles, seed
    )

    colors = colormaps["plasma"](dval)
    colors[..., :3] = colors[..., :3] * 0.45 + 0.55
    colors[..., -1] = 0.030 + 0.095 * dval

    ax.scatter(
        px,
        py,
        pz,
        s=1.2 + 4.6 * dval,
        c=colors,
        depthshade=False,
        linewidths=0,
    )


def _render_halo(
    ax: Any,
    density: np.ndarray,
    height: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    cfg: EnergyBodyConfig,
    seed_seed: np.random.SeedSequence | int,
    jitter_seed: np.random.SeedSequence | int,
) -> None:
    px, py, _, _, zval = _sample_density_particles(
        density, height, xedges, yedges, cfg.n_halo, seed_seed
    )

    rng = np.random.default_rng(jitter_seed)
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    halo_x = px + rng.normal(0, dx * 1.25, cfg.n_halo)
    halo_y = py + rng.normal(0, dy * 1.25, cfg.n_halo)
    halo_z = zval * rng.uniform(0.05, 1.0, cfg.n_halo)

    colors = np.zeros((cfg.n_halo, 4))
    colors[:, 0] = 0.70
    colors[:, 1] = 0.40
    colors[:, 2] = 0.95
    colors[:, 3] = 0.013

    ax.scatter(
        halo_x,
        halo_y,
        halo_z,
        s=3.4,
        c=colors,
        depthshade=False,
        linewidths=0,
    )


def _render_floor_contours(
    ax: Any, grid_x: np.ndarray, grid_y: np.ndarray, density: np.ndarray
) -> None:
    for level in [0.08, 0.18, 0.32, 0.50, 0.70]:
        ax.contour(
            grid_x,
            grid_y,
            density,
            levels=[level],
            zdir="z",
            offset=0.002,
            colors=[(0.50, 0.20, 0.82, 0.24)],
            linewidths=1.45,
        )


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------


def render_energy_body(
    x: np.ndarray,
    y: np.ndarray,
    output_path: str | Path,
    config: EnergyBodyConfig | None = None,
    title: str = "Shot Cloud as Condensed Energy Field",
    annotation: str | None = (
        "Volumetric plasma rendering\n"
        "Lighter background + denser energy body\n"
        "Density visualized as condensed probability energy"
    ),
) -> Path:
    """Render the 3D energy-body shot cloud and save to ``output_path``.

    Parameters
    ----------
    x, y : np.ndarray
        Shot coordinates in feet, basket at origin. Same length.
    output_path : str | Path
        Where to write the PNG. Parent directory must exist.
    config : EnergyBodyConfig, optional
        Rendering parameters; defaults to canonical hero-figure settings.
    title : str
        Figure title.
    annotation : str, optional
        Top-left annotation text. Pass ``None`` to suppress.

    Returns
    -------
    Path
        The path the figure was written to.
    """
    cfg = config or EnergyBodyConfig()
    out = Path(output_path)

    density, height, grid_x, grid_y, (xedges, yedges) = estimate_density(x, y, cfg)

    seed_seq = np.random.SeedSequence(cfg.seed)
    s_particles, s_halo, s_jitter = seed_seq.spawn(3)

    fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor(cfg.figure_bg)
    ax.set_facecolor(cfg.axes_bg)

    _render_energy_shells(ax, grid_x, grid_y, height, density)
    _render_particles(ax, density, height, xedges, yedges, cfg, s_particles)
    _render_halo(ax, density, height, xedges, yedges, cfg, s_halo, s_jitter)
    _render_floor_contours(ax, grid_x, grid_y, density)
    _draw_halfcourt(ax, cfg)

    g = cfg.grid
    width = g.xlim[1] - g.xlim[0]
    height_extent = g.ylim[1] - g.ylim[0]

    ax.set_xlim(*g.xlim)
    ax.set_ylim(*g.ylim)
    ax.set_zlim(0, 1.02)
    ax.set_xlabel("X (feet)", color=cfg.text_color, labelpad=10)
    ax.set_ylabel("Y (feet)", color=cfg.text_color, labelpad=10)
    ax.set_zlabel("Energy density", color=cfg.text_color, labelpad=10)
    ax.tick_params(colors=cfg.text_color, labelsize=8)
    ax.grid(True, alpha=0.16)
    ax.view_init(elev=cfg.elev, azim=cfg.azim)
    ax.set_box_aspect((width, height_extent, 18))

    ax.set_title(title, color=cfg.text_color, fontsize=16, pad=20, weight="bold")

    if annotation:
        ax.text2D(
            0.02,
            0.94,
            annotation,
            transform=ax.transAxes,
            color=cfg.text_color,
            fontsize=9.5,
            va="top",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#FFFFFF",
                "edgecolor": "#C3C8D4",
                "alpha": 0.92,
            },
        )

    plt.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out
