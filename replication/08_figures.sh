#!/usr/bin/env bash
# Stage 8: paper figures.
#
# Writes to outputs/figures/paper/:
#   self_bootstrap_floor.png, ablation_summary.png   (scripts/build_paper_figures.py)
#   dynamics_history.png                             (scripts/build_dynamics_figure.py)
#   attention_over_support.png                       (scripts/build_attention_figure.py)
# The attention figure reads outputs/attention_diagnostics/hero_wembanyama.npz,
# extracted here from the mainline checkpoint.
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
need "$MAINLINE/modules.pt" "$MAINLINE/cloud_metrics.json" "$MAINLINE/density_metrics.json" \
    outputs/baseline_kde/cloud_metrics.json
begin "$@"

FIGURES=outputs/figures/paper
HERO=outputs/attention_diagnostics/hero_wembanyama.npz

have "$FIGURES/self_bootstrap_floor.png" "$FIGURES/ablation_summary.png" \
    || run uv run python scripts/build_paper_figures.py

have "$FIGURES/dynamics_history.png" || run uv run python scripts/build_dynamics_figure.py

have "$HERO" || run bash scripts/runs/extract_attention_for_hero.sh --output "$HERO"
have "$FIGURES/attention_over_support.png" \
    || run bash scripts/runs/build_attention_figure.sh --output "$FIGURES/attention_over_support.png"
