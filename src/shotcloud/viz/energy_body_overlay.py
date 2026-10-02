"""Energy-body overlays comparing predicted and observed shot clouds.

Companion to :mod:`shotcloud.viz.energy_body`, which renders a single
energy body. This module provides two overlay renderers with the same
visual style:

* :func:`render_predicted_with_observed_shots` — the predicted
  density surface plus discrete observed-shot markers placed at
  ``(x_i, y_i, ẑ_i)`` where ``ẑ_i`` is the predicted-density value
  bilinearly interpolated at the observed shot's location. Markers
  on the predicted ridge mean "the model put high density here";
  markers on the floor mean "the model missed this shot".

* :func:`render_predicted_vs_bootstrap` — the predicted density
  surface (warm palette, default plasma) **plus** a bootstrap density
  surface of the observed shots (cool palette, default cividis) in
  the same 3D frame. The two palettes are perceptually orthogonal,
  so overlap regions read as muted desaturated tones and mismatches
  show as one-color blooms.

Both renderers support an optional 2D-contour companion panel
(``companion_2d=True``) — useful when location precision matters more
than the visual hook. The 2D panel sits to the left of the 3D body in
a 2-axes figure.

``scripts/plot_hero_shot_clouds.py`` drives both renderers: it selects
players and games from a trained checkpoint and writes a metadata JSON
sidecar alongside each PNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from shotcloud.viz.energy_body import (
    EnergyBodyConfig,
    _draw_halfcourt,
    _render_floor_contours,
    estimate_density,
)

__all__ = [
    "OverlayPaletteConfig",
    "render_predicted_vs_bootstrap",
    "render_predicted_with_observed_shots",
]


@dataclass
class OverlayPaletteConfig:
    """Colors, transparency, and marker style for the overlay renderers.

    Defaults pair plasma (predicted, warm) with cividis (bootstrap,
    cool). Both palettes are perceptually uniform; their hue axes are
    near-orthogonal so transparency-blended overlap reads as a muted
    desaturated tone rather than mud.

    Attributes
    ----------
    predicted_cmap : str
        Matplotlib colormap name for the predicted (warm) surface.
    observed_cmap : str
        Matplotlib colormap name for the bootstrap-observed (cool)
        surface. Pair with a cool palette to maintain visual contrast
        with the predicted surface.
    predicted_shell_alphas : tuple of float
        Per-shell alpha for the warm stack. Length determines the
        number of nested shells.
    observed_shell_alphas : tuple of float
        Per-shell alpha for the cool stack. Lower than the predicted
        alphas by default so the bootstrap surface stays visibly
        secondary to the predicted surface (bootstraps over small K
        are noisy).
    observed_marker_color : str
        Color for observed-shot markers in
        :func:`render_predicted_with_observed_shots`. Default cyan
        on the plasma palette reads cleanly against the warm
        background.
    observed_marker_edge : str
        Edge color for observed-shot markers.
    observed_marker_size : float
        Marker size (matplotlib scatter ``s``).
    """

    predicted_cmap: str = "plasma"
    observed_cmap: str = "cividis"
    predicted_shell_alphas: tuple[float, ...] = (0.24, 0.19, 0.15, 0.11)
    observed_shell_alphas: tuple[float, ...] = (0.20, 0.15, 0.11, 0.08)
    observed_marker_color: str = "#5DE2FF"
    observed_marker_edge: str = "#0C1828"
    observed_marker_size: float = 36.0


# ---------------------------------------------------------------------------
# Shared geometry: density estimation reuses the base module's helper.
# ---------------------------------------------------------------------------


def _bilinear_at_points(
    density: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Bilinear-interpolate a 2D density grid at scatter points.

    Out-of-bounds points are clamped to the boundary cell (density
    on the court edge is typically near zero anyway).
    """
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    # Clamp to grid range.
    xs_c = np.clip(xs, xcenters[0], xcenters[-1])
    ys_c = np.clip(ys, ycenters[0], ycenters[-1])
    # Locate the four neighbors.
    ix = np.searchsorted(xcenters, xs_c) - 1
    iy = np.searchsorted(ycenters, ys_c) - 1
    ix = np.clip(ix, 0, density.shape[1] - 2)
    iy = np.clip(iy, 0, density.shape[0] - 2)
    fx = (xs_c - xcenters[ix]) / (xcenters[ix + 1] - xcenters[ix])
    fy = (ys_c - ycenters[iy]) / (ycenters[iy + 1] - ycenters[iy])
    d00 = density[iy, ix]
    d10 = density[iy, ix + 1]
    d01 = density[iy + 1, ix]
    d11 = density[iy + 1, ix + 1]
    interp = d00 * (1 - fx) * (1 - fy) + d10 * fx * (1 - fy) + d01 * (1 - fx) * fy + d11 * fx * fy
    return np.asarray(interp, dtype=np.float64)


# ---------------------------------------------------------------------------
# Shell renderer with a configurable colormap, so two palettes can share
# one set of axes.
# ---------------------------------------------------------------------------


def _render_shells_palette(
    ax: Any,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    height: np.ndarray,
    density: np.ndarray,
    *,
    cmap_name: str,
    shell_alphas: tuple[float, ...],
    canopy_alpha: float = 0.42,
    ridge_power: float = 0.38,
    lightness_pull: float = 0.50,
) -> None:
    """Translucent shells in any matplotlib colormap.

    Mirrors :func:`shotcloud.viz.energy_body._render_energy_shells`
    but with the colormap (and per-shell alpha) parameterized so two
    surfaces in different palettes can share the same axes.
    """
    cmap = colormaps[cmap_name]
    shell_offsets = np.linspace(0.0, -0.09, len(shell_alphas))

    for offset, alpha in zip(shell_offsets, shell_alphas, strict=True):
        z_shell = np.maximum(height + offset, 0)
        colors = cmap(z_shell)
        colors[..., :3] = colors[..., :3] * (1.0 - lightness_pull) + lightness_pull
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

    # Bright canopy on top of the stack.
    ridge = np.power(density, ridge_power)
    canopy_colors = cmap(ridge)
    canopy_colors[..., :3] = canopy_colors[..., :3] * (1.0 - lightness_pull) + lightness_pull
    canopy_colors[..., -1] = np.clip(ridge * 0.38, 0, canopy_alpha)
    ax.plot_surface(
        grid_x,
        grid_y,
        height * 1.02,
        facecolors=canopy_colors,
        linewidth=0,
        antialiased=True,
        shade=False,
    )


# ---------------------------------------------------------------------------
# 2D companion panel
# ---------------------------------------------------------------------------


def _add_patch_2d(ax: Axes, patch: Any) -> None:
    ax.add_patch(patch)


def _draw_halfcourt_2d(ax: Axes, cfg: EnergyBodyConfig) -> None:
    """Draw the same half-court lines as the 3D version, in 2D."""
    color = cfg.court_color
    lw = 1.0
    _add_patch_2d(
        ax, mpatches.Rectangle((-25, -5), 50, 52, fill=False, edgecolor=color, linewidth=lw)
    )
    # Three-point line: straight corner segments at x = ±22 ft joined by
    # the 23.75 ft arc. The arc starts where it meets the corner lines
    # (x = ±22, θ = arccos(22/23.75) ≈ 22°) so the two join without a gap.
    corner_x, arc_r = 22.0, 23.75
    theta_start = float(np.arccos(corner_x / arc_r))
    theta = np.linspace(theta_start, np.pi - theta_start, 120)
    ax.plot(arc_r * np.cos(theta), arc_r * np.sin(theta), color=color, linewidth=lw)
    corner_y = float(np.sqrt(arc_r**2 - corner_x**2))  # arc/corner junction ≈ 8.95 ft
    ax.plot([-corner_x, -corner_x], [-5, corner_y], color=color, linewidth=lw)
    ax.plot([corner_x, corner_x], [-5, corner_y], color=color, linewidth=lw)
    # Free-throw key.
    ax.plot([-8, -8], [-5, 19], color=color, linewidth=lw)
    ax.plot([8, 8], [-5, 19], color=color, linewidth=lw)
    ax.plot([-8, 8], [19, 19], color=color, linewidth=lw)
    # Restricted area.
    theta_ra = np.linspace(0, np.pi, 60)
    ax.plot(4 * np.cos(theta_ra), 4 * np.sin(theta_ra), color=color, linewidth=lw)
    # Backboard + rim.
    ax.plot([-3, 3], [-1.25, -1.25], color=color, linewidth=lw)
    ax.add_patch(mpatches.Circle((0, 0), 0.75, fill=False, edgecolor=color, linewidth=lw))


def _draw_2d_companion(
    ax: Axes,
    density: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    cfg: EnergyBodyConfig,
    *,
    cmap_name: str,
    observed_xy: np.ndarray | None,
    observed_marker_color: str,
    observed_marker_edge: str,
    observed_marker_size: float,
    overlay_density: np.ndarray | None = None,
    overlay_cmap_name: str | None = None,
) -> None:
    """2D view of the predicted density with observed-shot markers.

    An optional second density (the bootstrap surface) is drawn as
    contour lines.
    """
    extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
    # Filled predicted density with a *density-proportional alpha* so
    # low-density court regions stay transparent and the court lines
    # show through. A flat alpha floods the whole court with the
    # colormap's dark low end and hides the geometry.
    norm_d = density / max(float(density.max()), 1e-12)
    rgba = colormaps[cmap_name](norm_d)
    rgba[..., -1] = np.clip(np.power(norm_d, 0.6), 0.0, 0.92)
    ax.imshow(
        rgba,
        origin="lower",
        extent=extent,
        interpolation="bilinear",
    )
    # Optional second density (bootstrap), shown as contour lines so it
    # reads against the filled background.
    if overlay_density is not None and overlay_cmap_name is not None:
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        grid_xs, grid_ys = np.meshgrid(xcenters, ycenters)
        levels = np.linspace(overlay_density.max() * 0.10, overlay_density.max() * 0.95, 6)
        ax.contour(
            grid_xs,
            grid_ys,
            overlay_density,
            levels=levels,
            cmap=overlay_cmap_name,
            linewidths=1.2,
            alpha=0.95,
        )
    _draw_halfcourt_2d(ax, cfg)
    if observed_xy is not None and observed_xy.shape[0] > 0:
        ax.scatter(
            observed_xy[:, 0],
            observed_xy[:, 1],
            s=observed_marker_size,
            c=observed_marker_color,
            edgecolors=observed_marker_edge,
            linewidths=0.8,
            zorder=5,
        )
    g = cfg.grid
    ax.set_xlim(*g.xlim)
    ax.set_ylim(*g.ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("X (feet)", color=cfg.text_color)
    ax.set_ylabel("Y (feet)", color=cfg.text_color)
    ax.tick_params(colors=cfg.text_color, labelsize=8)


# ---------------------------------------------------------------------------
# Title block: centralized so both renderers carry identical metadata.
# ---------------------------------------------------------------------------


@dataclass
class GameMetadata:
    """Per-game metadata for the figure title block and the sidecar JSON.

    Built by the caller, typically from the validation rows and the
    joined game logs. ``extra`` entries are merged into :meth:`to_dict`.
    """

    player_name: str
    away_team: str
    home_team: str
    date: str
    opponent: str
    n_shots: int
    model_checkpoint: str
    selection_rule: str = "largest_shot_count"
    split: str = "validation"
    energy_distance: float | None = None
    zone_l1: float | None = None
    rim_distance_w1: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def matchup(self) -> str:
        """Matchup string ``"AWAY @ HOME"``."""
        return f"{self.away_team} @ {self.home_team}"

    def title_block(self) -> tuple[str, str, str]:
        """``(title, subtitle, extras)`` strings for figure annotation."""
        title = f"{self.player_name} — {self.matchup}"
        subtitle = f"{self.date} | Shots: {self.n_shots} | Model: {self.model_checkpoint}"
        extras_parts = [f"Opponent: {self.opponent}", f"Split: {self.split}"]
        if self.energy_distance is not None:
            extras_parts.append(f"Energy: {self.energy_distance:.2f} ft")
        if self.zone_l1 is not None:
            extras_parts.append(f"Zone L1: {self.zone_l1:.3f}")
        if self.rim_distance_w1 is not None:
            extras_parts.append(f"Rim W1: {self.rim_distance_w1:.2f} ft")
        extras = " | ".join(extras_parts)
        return title, subtitle, extras

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict of all fields, with ``extra`` merged in."""
        d = {
            "player_name": self.player_name,
            "away_team": self.away_team,
            "home_team": self.home_team,
            "matchup": self.matchup,
            "date": self.date,
            "opponent": self.opponent,
            "n_shots": self.n_shots,
            "model_checkpoint": self.model_checkpoint,
            "selection_rule": self.selection_rule,
            "split": self.split,
            "energy_distance": self.energy_distance,
            "zone_l1": self.zone_l1,
            "rim_distance_w1": self.rim_distance_w1,
        }
        d.update(self.extra)
        return d


def _apply_title_block(
    fig: Figure, ax3d: Any, metadata: GameMetadata, cfg: EnergyBodyConfig
) -> None:
    """Add the title and subtitle, plus the extras line in a 3D-axes box."""
    title, subtitle, extras = metadata.title_block()
    fig.suptitle(title, color=cfg.text_color, fontsize=16, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.945,
        subtitle,
        ha="center",
        va="top",
        color=cfg.text_color,
        fontsize=11,
    )
    ax3d.text2D(
        0.02,
        0.96,
        extras,
        transform=ax3d.transAxes,
        color=cfg.text_color,
        fontsize=9,
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFFFFF",
            "edgecolor": "#C3C8D4",
            "alpha": 0.92,
        },
    )


# ---------------------------------------------------------------------------
# 3D axes setup shared by both renderers
# ---------------------------------------------------------------------------


def _setup_3d_axes(
    fig: Figure, position: tuple[float, float, float, float], cfg: EnergyBodyConfig
) -> Any:
    # The 3D axes type is matplotlib.mplot3d.Axes3D; we treat it as
    # ``Any`` because the public matplotlib stubs don't reflect that
    # ``add_axes(projection="3d")`` returns the 3D subclass.
    ax: Any = fig.add_axes(position, projection="3d")
    ax.set_facecolor(cfg.axes_bg)
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
    return ax


# ---------------------------------------------------------------------------
# Observed-shot markers with floor stems (3D)
# ---------------------------------------------------------------------------


def _render_observed_stems(
    ax: Any,
    observed_xy: np.ndarray,
    z_tops: np.ndarray,
    *,
    marker_color: str,
    marker_edge: str,
    marker_size: float,
    stem_color: str = "#0C1828",
    stem_alpha: float = 0.55,
) -> None:
    """Scatter observed shots at ``(x, y, z_top)`` with stems to the floor.

    Each marker gets a vertical stem from the court floor (z=0) and a
    faint floor dot at ``(x, y, 0)``; a floating marker alone is hard to
    localize under the 3D projection.
    """
    for (x, y), z in zip(observed_xy, z_tops, strict=True):
        ax.plot(
            [x, x],
            [y, y],
            [0.0, float(z)],
            color=stem_color,
            linewidth=0.8,
            alpha=stem_alpha,
            zorder=9,
        )
    # Faint floor footprints.
    ax.scatter(
        observed_xy[:, 0],
        observed_xy[:, 1],
        np.zeros(observed_xy.shape[0]),
        s=marker_size * 0.35,
        c=stem_color,
        alpha=0.45,
        depthshade=False,
        linewidths=0,
        zorder=9,
    )
    # Markers on the canopy.
    ax.scatter(
        observed_xy[:, 0],
        observed_xy[:, 1],
        z_tops,
        s=marker_size,
        c=marker_color,
        edgecolors=marker_edge,
        linewidths=0.9,
        depthshade=False,
        zorder=10,
    )


# ---------------------------------------------------------------------------
# Predicted body + observed shot markers
# ---------------------------------------------------------------------------


def render_predicted_with_observed_shots(
    predicted_xy: np.ndarray,
    observed_xy: np.ndarray,
    output_path: str | Path,
    metadata: GameMetadata,
    *,
    config: EnergyBodyConfig | None = None,
    palette: OverlayPaletteConfig | None = None,
    companion_2d: bool = False,
) -> Path:
    """Render the predicted energy body with observed shots as markers.

    Observed shots are placed at ``(x_i, y_i, ẑ_i)``, where ``ẑ_i`` is
    the rendered height of the predicted surface at the shot.

    Parameters
    ----------
    predicted_xy : ndarray of shape ``(N, 2)``
        Shot coordinates sampled from the trained spatial model,
        typically ``n_samples`` conditional draws for each of the ``K``
        observed shots, ravelled to ``N = n_samples × K``.
    observed_xy : ndarray of shape ``(K, 2)``
        The validation game's observed shot coordinates.
    output_path : str | Path
        Where to write the PNG. Parent directory must exist.
    metadata : GameMetadata
        Title-block and sidecar fields.
    config : EnergyBodyConfig, optional
        Rendering parameters; defaults to the hero-figure settings.
    palette : OverlayPaletteConfig, optional
        Surface colormap and observed-shot marker style.
    companion_2d : bool, default False
        When True, the figure has two axes side-by-side: a 2D view of
        the predicted density (left) and the 3D energy body (right).
        When False, just the 3D body.

    Returns
    -------
    Path
        The path the figure was written to.
    """
    cfg = config or EnergyBodyConfig()
    pal = palette or OverlayPaletteConfig()
    out = Path(output_path)

    density, height, grid_x, grid_y, (xedges, yedges) = estimate_density(
        predicted_xy[:, 0], predicted_xy[:, 1], cfg
    )

    if companion_2d:
        fig = plt.figure(figsize=(cfg.figsize[0] * 1.6, cfg.figsize[1]), dpi=cfg.dpi)
        fig.patch.set_facecolor(cfg.figure_bg)
        ax2d = fig.add_axes((0.05, 0.08, 0.38, 0.78))
        _draw_2d_companion(
            ax2d,
            density,
            xedges,
            yedges,
            cfg,
            cmap_name=pal.predicted_cmap,
            observed_xy=observed_xy,
            observed_marker_color=pal.observed_marker_color,
            observed_marker_edge=pal.observed_marker_edge,
            observed_marker_size=pal.observed_marker_size,
        )
        ax2d.set_title("2D predicted density + observed shots", color=cfg.text_color, fontsize=11)
        ax3d = _setup_3d_axes(fig, (0.46, 0.05, 0.52, 0.82), cfg)
    else:
        fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)
        fig.patch.set_facecolor(cfg.figure_bg)
        ax3d_local: Any = fig.add_subplot(111, projection="3d")
        ax3d = ax3d_local
        ax3d.set_facecolor(cfg.axes_bg)
        g = cfg.grid
        width = g.xlim[1] - g.xlim[0]
        height_extent = g.ylim[1] - g.ylim[0]
        ax3d.set_xlim(*g.xlim)
        ax3d.set_ylim(*g.ylim)
        ax3d.set_zlim(0, 1.02)
        ax3d.set_xlabel("X (feet)", color=cfg.text_color, labelpad=10)
        ax3d.set_ylabel("Y (feet)", color=cfg.text_color, labelpad=10)
        ax3d.set_zlabel("Energy density", color=cfg.text_color, labelpad=10)
        ax3d.tick_params(colors=cfg.text_color, labelsize=8)
        ax3d.grid(True, alpha=0.16)
        ax3d.view_init(elev=cfg.elev, azim=cfg.azim)
        ax3d.set_box_aspect((width, height_extent, 18))

    # Predicted body
    _render_shells_palette(
        ax3d,
        grid_x,
        grid_y,
        height,
        density,
        cmap_name=pal.predicted_cmap,
        shell_alphas=pal.predicted_shell_alphas,
    )
    _render_floor_contours(ax3d, grid_x, grid_y, density)
    _draw_halfcourt(ax3d, cfg)

    # Observed shot markers at (x_i, y_i, ẑ_i). Interpolating the
    # smoothed density rather than the raw histogram keeps markers on
    # the rendered canopy near the shot's coordinate.
    if observed_xy.shape[0] > 0:
        ẑ = _bilinear_at_points(density, xedges, yedges, observed_xy[:, 0], observed_xy[:, 1])
        # Power-scale to match the rendered height field.
        ẑ_h = np.power(ẑ, cfg.density_power) * 1.03
        _render_observed_stems(
            ax3d,
            observed_xy,
            ẑ_h,
            marker_color=pal.observed_marker_color,
            marker_edge=pal.observed_marker_edge,
            marker_size=pal.observed_marker_size,
        )

    _apply_title_block(fig, ax3d, metadata, cfg)
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Predicted energy body + bootstrap-observed energy body
# ---------------------------------------------------------------------------


def _bootstrap_resamples(
    observed_xy: np.ndarray,
    n_bootstraps: int,
    seed: int | np.random.SeedSequence,
) -> np.ndarray:
    """Resample observed shots ``n_bootstraps`` times; returns ``(n_bootstraps × K, 2)``."""
    if observed_xy.shape[0] == 0:
        return observed_xy
    rng = np.random.default_rng(seed)
    k = observed_xy.shape[0]
    idx = rng.integers(0, k, size=(n_bootstraps, k))
    return observed_xy[idx.ravel()]


def render_predicted_vs_bootstrap(
    predicted_xy: np.ndarray,
    observed_xy: np.ndarray,
    output_path: str | Path,
    metadata: GameMetadata,
    *,
    config: EnergyBodyConfig | None = None,
    palette: OverlayPaletteConfig | None = None,
    n_bootstraps: int = 200,
    bootstrap_seed: int = 0,
    companion_2d: bool = False,
) -> Path:
    """Render the predicted and bootstrap-observed energy bodies together.

    The predicted body (warm palette) and a bootstrap density of the
    observed shots (cool palette) share one 3D frame.

    Parameters
    ----------
    predicted_xy : ndarray of shape ``(N, 2)``
        Predicted samples (see :func:`render_predicted_with_observed_shots`).
    observed_xy : ndarray of shape ``(K, 2)``
        Observed shots; the bootstrap surface is built by resampling
        with replacement.
    output_path : str | Path
        Where to write the PNG. Parent directory must exist.
    metadata : GameMetadata
        Title-block and sidecar fields.
    config : EnergyBodyConfig, optional
        Rendering parameters; defaults to the hero-figure settings.
    palette : OverlayPaletteConfig, optional
        Predicted vs observed colormap pairing.
    n_bootstraps : int, default 200
        Number of resamples of the observed shots. The bootstrap
        density is the smoothed histogram of all ``n_bootstraps × K``
        resampled points; larger ``n_bootstraps`` yields a smoother
        cool-palette surface.
    bootstrap_seed : int, default 0
        Seed for the bootstrap resampling.
    companion_2d : bool, default False
        When True, adds a 2D panel on the left showing the predicted
        density (filled) and the bootstrap density (contours).

    Returns
    -------
    Path
        The path the figure was written to.
    """
    cfg = config or EnergyBodyConfig()
    pal = palette or OverlayPaletteConfig()
    out = Path(output_path)

    pred_density, pred_height, grid_x, grid_y, (xedges, yedges) = estimate_density(
        predicted_xy[:, 0], predicted_xy[:, 1], cfg
    )

    boot_xy = _bootstrap_resamples(observed_xy, n_bootstraps, bootstrap_seed)
    if boot_xy.shape[0] == 0:
        # Degenerate: no observed shots → just the predicted body.
        boot_density = np.zeros_like(pred_density)
        boot_height = np.zeros_like(pred_height)
    else:
        boot_density, boot_height, _, _, _ = estimate_density(boot_xy[:, 0], boot_xy[:, 1], cfg)

    if companion_2d:
        fig = plt.figure(figsize=(cfg.figsize[0] * 1.6, cfg.figsize[1]), dpi=cfg.dpi)
        fig.patch.set_facecolor(cfg.figure_bg)
        ax2d = fig.add_axes((0.05, 0.08, 0.38, 0.78))
        _draw_2d_companion(
            ax2d,
            pred_density,
            xedges,
            yedges,
            cfg,
            cmap_name=pal.predicted_cmap,
            observed_xy=observed_xy,
            observed_marker_color=pal.observed_marker_color,
            observed_marker_edge=pal.observed_marker_edge,
            observed_marker_size=pal.observed_marker_size,
            overlay_density=boot_density,
            overlay_cmap_name=pal.observed_cmap,
        )
        ax2d.set_title(
            "2D: predicted (filled) vs bootstrap (contour)",
            color=cfg.text_color,
            fontsize=11,
        )
        ax3d = _setup_3d_axes(fig, (0.46, 0.05, 0.52, 0.82), cfg)
    else:
        fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)
        fig.patch.set_facecolor(cfg.figure_bg)
        ax3d_local: Any = fig.add_subplot(111, projection="3d")
        ax3d = ax3d_local
        ax3d.set_facecolor(cfg.axes_bg)
        g = cfg.grid
        width = g.xlim[1] - g.xlim[0]
        height_extent = g.ylim[1] - g.ylim[0]
        ax3d.set_xlim(*g.xlim)
        ax3d.set_ylim(*g.ylim)
        ax3d.set_zlim(0, 1.02)
        ax3d.set_xlabel("X (feet)", color=cfg.text_color, labelpad=10)
        ax3d.set_ylabel("Y (feet)", color=cfg.text_color, labelpad=10)
        ax3d.set_zlabel("Energy density", color=cfg.text_color, labelpad=10)
        ax3d.tick_params(colors=cfg.text_color, labelsize=8)
        ax3d.grid(True, alpha=0.16)
        ax3d.view_init(elev=cfg.elev, azim=cfg.azim)
        ax3d.set_box_aspect((width, height_extent, 18))

    # Bootstrap-observed body first so the predicted body sits on top
    # (slightly higher z-offset).
    _render_shells_palette(
        ax3d,
        grid_x,
        grid_y,
        boot_height,
        boot_density,
        cmap_name=pal.observed_cmap,
        shell_alphas=pal.observed_shell_alphas,
    )
    _render_shells_palette(
        ax3d,
        grid_x,
        grid_y,
        pred_height,
        pred_density,
        cmap_name=pal.predicted_cmap,
        shell_alphas=pal.predicted_shell_alphas,
    )
    _render_floor_contours(ax3d, grid_x, grid_y, pred_density)
    _draw_halfcourt(ax3d, cfg)

    # Legend: small in-axes patch with the two-palette key.
    pred_color = colormaps[pal.predicted_cmap](0.7)
    obs_color = colormaps[pal.observed_cmap](0.7)
    ax3d.text2D(
        0.78,
        0.96,
        f"■ {pal.predicted_cmap}: model\n■ {pal.observed_cmap}: bootstrap",
        transform=ax3d.transAxes,
        color=cfg.text_color,
        fontsize=9,
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFFFFF",
            "edgecolor": "#C3C8D4",
            "alpha": 0.92,
        },
    )
    # Use the colors locally to avoid lint complaints about unused vars.
    del pred_color, obs_color

    _apply_title_block(fig, ax3d, metadata, cfg)
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out


# Re-export the metadata dataclass.
__all__.insert(0, "GameMetadata")
__all__.sort()
