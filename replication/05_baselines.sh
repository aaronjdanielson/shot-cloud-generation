#!/usr/bin/env bash
# Stage 5: baselines -- the causal reference KDE and the conditional MDN.
#
# The reference KDE has no training step; it is fitted and scored in one pass.
# The MDN is trained and then scored. Both are evaluated on the same 2000
# validation player-games as AC-KDE. Writes cloud_metrics.json and
# density_metrics.json to outputs/baseline_kde/ and outputs/mdn_v1/.
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
need "$SNAPSHOTS" "$PLAYER_BIO"
begin "$@"

have outputs/baseline_kde/cloud_metrics.json outputs/baseline_kde/density_metrics.json \
    || run bash scripts/runs/evaluate_baseline_kde.sh \
        --output-cloud outputs/baseline_kde/cloud_metrics.json \
        --output-density outputs/baseline_kde/density_metrics.json

have outputs/mdn_v1/modules.pt outputs/mdn_v1/cloud_metrics.json outputs/mdn_v1/density_metrics.json \
    || run bash scripts/run_experiment_mdn.sh
