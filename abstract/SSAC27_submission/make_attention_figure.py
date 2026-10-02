"""Build the shot-level attention figure of the SSAC27 abstract.

    python3 abstract/SSAC27_submission/make_attention_figure.py

Reads evidence/hero_wembanyama.npz (a copy of outputs/attention_diagnostics/hero_wembanyama.npz,
written by scripts/extract_attention_for_hero.py) and writes figures/fig_attention.pdf (and
.png). Three half-court panels: the predicted density with the observed shots, the attention on
the player's own past shots, and the attention on shots borrowed from similar players. Same
quantities and encodings as scripts/build_attention_figure.py, arranged in one row with the
own/borrowed split in the panel titles.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COURT_X = (-25.0, 25.0)
COURT_Y = (-5.0, 33.0)
INK = "#0b0b0b"
MUTED = "#52514e"
LINE = "#3a3a38"


def draw_half_court(ax: plt.Axes, color: str = LINE) -> None:
    ax.set_xlim(COURT_X)
    ax.set_ylim(COURT_Y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    lw = 0.7
    ax.plot([COURT_X[0], COURT_X[1], COURT_X[1], COURT_X[0], COURT_X[0]], [-5, -5, 33, 33, -5], color=color, lw=lw)
    theta = np.linspace(np.pi - math.acos(22 / 23.75), math.acos(22 / 23.75), 200)
    ax.plot(23.75 * np.cos(theta), 23.75 * np.sin(theta), color=color, lw=lw)
    ax.plot([-22, -22], [-5, 23.75 * math.cos(math.asin(22 / 23.75))], color=color, lw=lw)
    ax.plot([22, 22], [-5, 23.75 * math.cos(math.asin(22 / 23.75))], color=color, lw=lw)
    ax.add_patch(patches.Rectangle((-8, -5), 16, 19, fill=False, lw=lw, ec=color))
    ax.add_patch(patches.Circle((0, 0), 0.75, fill=False, lw=lw, ec=color))
    theta_ra = np.linspace(0, math.pi, 100)
    ax.plot(4 * np.cos(theta_ra), 4 * np.sin(theta_ra), color=color, lw=lw)


def gaussian_density(grid_xy: np.ndarray, centers: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    sq = ((grid_xy[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    log_kernel = -0.5 * sq / (sigma * sigma) - math.log(2 * math.pi * sigma * sigma)
    return (weights[None, :] * np.exp(log_kernel)).sum(axis=-1)


def attention_sizes(w: np.ndarray, max_pt: float = 150.0) -> np.ndarray:
    return 2.0 + max_pt * (w / max(w.max(), 1e-12)) ** 0.55


def main() -> None:
    z = np.load(HERE / "evidence" / "hero_wembanyama.npz", allow_pickle=True)
    shot_xy = z["shot_xy"]
    support_xy0 = z["support_xy"][0]
    support_mask = z["support_mask"]
    own_mask = z["own_mask"]
    valid, own0 = support_mask[0], own_mask[0]
    pooled0 = valid & ~own0

    w = np.exp(z["support_log_weights"]) * support_mask.astype(np.float64)
    w = w / w.sum(axis=-1, keepdims=True).clip(min=1e-30)
    avg_w = w.mean(axis=0)
    own_mass = float((w * own_mask.astype(np.float64)).sum(axis=-1).mean())
    pooled_mass = float((w * pooled0.astype(np.float64)[None, :]).sum(axis=-1).mean())
    sigma = float(z["sigma"].mean())
    print(f"shots in game {len(shot_xy)}, own support {int(own0.sum())}, borrowed support {int(pooled0.sum())}")
    print(f"own mass {own_mass:.3f}, borrowed mass {pooled_mass:.3f}, mean bandwidth {sigma:.2f} ft")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.4, 2.6), facecolor="white")

    gx = np.linspace(*COURT_X, 140)
    gy = np.linspace(*COURT_Y, 140)
    GX, GY = np.meshgrid(gx, gy, indexing="xy")
    dens = gaussian_density(np.stack([GX.ravel(), GY.ravel()], axis=-1), support_xy0[valid], avg_w[valid], sigma)
    log_dens = np.log(np.clip(dens.reshape(GX.shape), dens.max() * 1e-5, None))
    levels = np.unique(np.quantile(log_dens.ravel(), np.linspace(0.05, 0.99, 11)))
    ax_a.contourf(GX, GY, log_dens, levels=levels, cmap="magma", extend="both")
    draw_half_court(ax_a, color="#f2f1ed")
    ax_a.scatter(shot_xy[:, 0], shot_xy[:, 1], s=16, facecolor="white", edgecolor="black", linewidth=0.6, zorder=4)
    ax_a.set_title("Predicted density\nand the shots he took", fontsize=8.5, color=INK)

    for ax, mask, cmap, title in (
        (ax_b, own0, "Blues", f"Attention on his own {int(own0.sum())} past shots\n{own_mass:.0%} of the weight"),
        (ax_c, pooled0, "Oranges", f"Attention on {int(pooled0.sum())} shots from similar players\n{pooled_mass:.0%} of the weight"),
    ):
        draw_half_court(ax)
        ws = avg_w[mask]
        order = np.argsort(ws)
        ax.scatter(
            support_xy0[mask][order, 0], support_xy0[mask][order, 1], s=attention_sizes(ws[order]), c=ws[order],
            cmap=cmap, vmin=-0.25 * ws.max(), vmax=ws.max(), edgecolors="white", linewidths=0.3, alpha=0.85,
        )
        ax.set_title(title, fontsize=8.5, color=INK)

    fig.tight_layout(w_pad=0.6)
    out = HERE / "figures"
    fig.savefig(out / "fig_attention.pdf", facecolor="white", bbox_inches="tight")
    fig.savefig(out / "fig_attention.png", dpi=200, facecolor="white", bbox_inches="tight")
    print("wrote figures/fig_attention.pdf")


if __name__ == "__main__":
    main()
