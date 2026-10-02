#!/usr/bin/env bash
# Stage 3: pretrain the 48-bin timing head on pregame context.
#
# Writes outputs/timing_head_v2_masked/ (checkpoint, training history, and the
# calibration diagnostics against the histogram baselines).
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
need "$SNAPSHOTS" "$PLAYER_BIO"
begin "$@"

have outputs/timing_head_v2_masked/modules.pt || run bash scripts/run_experiment_timing_pretrain_masked.sh
