#!/usr/bin/env bash
# Stage 4: train the mainline AC-KDE model.
#
# Writes outputs/joint_b2_outcome_residual_v1/ (checkpoint, manifest, training
# history). This is the longest single stage.
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
need "$SNAPSHOTS" "$PLAYER_BIO" outputs/count_head_v1/modules.pt
begin "$@"

have "$MAINLINE/modules.pt" || run bash scripts/run_experiment_o1.sh
