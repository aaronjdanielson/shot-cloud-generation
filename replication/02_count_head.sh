#!/usr/bin/env bash
# Stage 2: pretrain the negative-binomial count head and the context MLP.
#
# Writes outputs/count_head_v1/ (checkpoint, training history, calibration
# diagnostics). The joint model loads this checkpoint and keeps it frozen.
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
need "$SNAPSHOTS" "$PLAYER_BIO"
begin "$@"

have outputs/count_head_v1/modules.pt || run bash scripts/run_experiment_count_head.sh
