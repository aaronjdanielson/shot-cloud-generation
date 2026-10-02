"""Build the method-comparison figure of the SSAC27 abstract from stored per-game records.

    python3 abstract/SSAC27_submission/make_comparison_figure.py

Reads evidence/*.json and writes figures/fig_method_comparison.pdf (and .png). One panel per
finite-cloud measure; each panel ranks four methods against the self-bootstrap noise floor on
the validation player-games that all four evaluations score (evidence/mdn_game_match.json,
written by match_mdn_games.py: games with 19+ shots by players present in the training window):

    position chart   causal grid KDE with kappa = 1e12, i.e. the position-group density only
    shot chart       the reference KDE: the player's own causal shots shrunk toward his position
    neural network   the conditional mixture density network
    AC-KDE           the mainline model (joint_b2_outcome_residual_v1)

Also prints, for every measure, each method's mean and the paired bootstrap interval of its
difference from AC-KDE, so the caption and the Results text can be checked against the records.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
N_BOOT = 5000
SEED = 0

# (record key, panel title, unit suffix for value labels, decimals)
METRICS = [
    ("mean_shot_distance_err_ft", "Mean shot-distance error", " ft", 2),
    ("rim_distance_w1", "Rim-distance Wasserstein", " ft", 2),
    ("sliced_wasserstein", "Sliced Wasserstein", " ft", 2),
    ("zone_l1", "Zone-share $L_1$", "", 3),
    ("rim_distance_ks", "Rim-distance KS", "", 3),
    ("energy_distance", "Energy distance", "", 3),
]

# (evidence file, row label, records nest metrics under "model")
METHODS = [
    ("position_chart_cloud_metrics.json", "Position chart", False),
    ("reference_kde_cloud_metrics.json", "Player's shot chart", False),
    ("mdn_cloud_metrics.json", "Neural network", True),
    ("ackde_cloud_metrics.json", "AC-KDE", True),
]

INK = "#0b0b0b"
MUTED = "#52514e"
BASELINE = "#8a8983"
ACCENT = "#2a78d6"
GRID = "#e4e3df"
SURFACE = "#ffffff"


def load(name: str) -> list[dict]:
    with open(EVIDENCE / name) as fh:
        return json.load(fh)["per_game"]


def values(records: list[dict], key: str, nested: bool) -> np.ndarray:
    return np.array([(g["model"] if nested else g)[key] for g in records])


def paired_ci(delta: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    n = len(delta)
    means = delta[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    records = {label: (load(name), nested) for name, label, nested in METHODS}
    full = records["AC-KDE"][0]
    games = [(g["game_idx"], g["k_obs"]) for g in full]
    with open(EVIDENCE / "mdn_game_match.json") as fh:
        pairs = json.load(fh)["pairs_ackde_pos_mdn_pos"]
    for label, (recs, nested) in records.items():
        if label == "Neural network":
            recs = [recs[m] for _, m in pairs]
        else:
            if [(g["game_idx"], g["k_obs"]) for g in recs] != games:
                raise SystemExit(f"{label}: records are not aligned game by game with AC-KDE")
            recs = [recs[a] for a, _ in pairs]
        records[label] = (recs, nested)
    ack = records["AC-KDE"][0]
    if [g["k_obs"] for g in ack] != [g["k_obs"] for g in records["Neural network"][0]]:
        raise SystemExit("matched MDN games do not have the AC-KDE games' shot counts")
    print(f"games scored by all four methods: {len(ack)}, players {len({g['player_id'] for g in ack})}, mean shots {np.mean([g['k_obs'] for g in ack]):.1f}")

    rng = np.random.default_rng(SEED)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.9), facecolor=SURFACE)
    labels = [label for _, label, _ in METHODS]
    ys = np.arange(len(labels))[::-1]

    for ax, (key, title, unit, d) in zip(axes.ravel(), METRICS):
        a = values(ack, key, True)
        floor = float(np.mean([g["self_bootstrap"][key] for g in ack]))
        means = []
        print(f"{key}: floor {floor:.4f}")
        for label in labels:
            recs, nested = records[label]
            v = values(recs, key, nested)
            means.append(float(v.mean()))
            if label != "AC-KDE":
                mu, lo, hi = paired_ci(a - v, rng)
                print(f"   {label:20s} {v.mean():.4f}   AC-KDE minus it {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]  AC-KDE closer in {np.mean(a < v):.3f} of games")
            else:
                print(f"   {label:20s} {v.mean():.4f}")

        gap = means[0] - floor
        print("   share of the position chart's gap to the floor closed: " + ", ".join(f"{lab} {(means[0] - m) / gap:.3f}" for lab, m in zip(labels[1:], means[1:])))
        span = max(means) - floor
        ax.set_xlim(floor - 0.07 * span, max(means) + 0.30 * span)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.axvline(floor, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for y, label, m in zip(ys, labels, means):
            accent = label == "AC-KDE"
            color = ACCENT if accent else BASELINE
            ax.plot([floor, m], [y, y], color=color, lw=1.6 if accent else 1.2, solid_capstyle="round", zorder=2)
            ax.plot(m, y, "o", ms=6, color=color, mec=SURFACE, mew=1.0, zorder=3)
            ax.text(
                m + 0.035 * span, y, f"{m:.{d}f}{unit}".replace("-", "−"),
                va="center", ha="left", color=INK, fontsize=7.5, fontweight="bold" if accent else "normal",
            )
        ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=4)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels if ax in axes[:, 0] else [], color=INK)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.tick_params(axis="x", colors=MUTED, length=2, labelsize=7)
        ax.locator_params(axis="x", nbins=4)
        ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:g}".replace("-", "−"))
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_facecolor(SURFACE)

    lo, hi = axes[0, 0].get_xlim()
    floor0 = float(np.mean([g["self_bootstrap"][METRICS[0][0]] for g in ack]))
    axes[0, 0].text(floor0 + 0.02 * (hi - lo), len(labels) - 0.45, "noise floor", color=MUTED, fontsize=7, va="top", ha="left")
    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig_method_comparison.pdf", facecolor=SURFACE)
    fig.savefig(out / "fig_method_comparison.png", dpi=200, facecolor=SURFACE)
    print("wrote figures/fig_method_comparison.pdf")

    with open(EVIDENCE / "count_head_diagnostics.json") as fh:
        val = json.load(fh)["val"]
    print(f"count head (val): {val['n_games']} player-games, MAE {val['MAE']:.3f}, slope {val['calibration_slope']:.3f}")


if __name__ == "__main__":
    main()
