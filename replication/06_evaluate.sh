#!/usr/bin/env bash
# Stage 6: evaluate the mainline AC-KDE model.
#
# Writes to outputs/joint_b2_outcome_residual_v1/:
#   cloud_metrics.json        finite-cloud metrics with the self-bootstrap floor
#   density_metrics.json      density-surface scores
#   autonomous_rollout.json   count-plus-location rollout metrics
#   support_audit.json        support-set descriptors against cloud error
# plus the count-head audit (outputs/ablation_b2/count_head_audit.json) and the
# per-game support-attention statistics (outputs/attention_diagnostics/per_game.json).
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
need "$MAINLINE/modules.pt" "$PLAYER_BIO"
begin "$@"

have "$MAINLINE/cloud_metrics.json" \
    || run bash scripts/runs/evaluate_shot_clouds.sh --output "$MAINLINE/cloud_metrics.json"

have "$MAINLINE/density_metrics.json" \
    || run bash scripts/runs/evaluate_density_surfaces.sh \
        --output "$MAINLINE/density_metrics.json" \
        --device cpu

have "$MAINLINE/autonomous_rollout.json" \
    || run uv run python scripts/evaluate_autonomous_rollout.py \
        --run-dir "$MAINLINE" \
        --shots "$SHOTS" \
        --player-bio "$PLAYER_BIO" \
        --game-logs "$GAME_LOGS" \
        --epoch latest \
        --max-games 2000 \
        --min-shots-per-game 3 \
        --n-rollouts 10 \
        --sliced-w-projections 200 \
        --output "$MAINLINE/autonomous_rollout.json" \
        --device "$DEVICE" \
        --seed 0

have "$MAINLINE/support_audit.json" \
    || run bash scripts/runs/support_retrieval_audit.sh --output "$MAINLINE/support_audit.json"

have outputs/ablation_b2/count_head_audit.json \
    || run bash scripts/runs/audit_count_head.sh --output outputs/ablation_b2/count_head_audit.json

have outputs/attention_diagnostics/per_game.json \
    || run uv run python scripts/extract_attention_diagnostics.py \
        --run-dir "$MAINLINE" \
        --shots "$SHOTS" \
        --player-bio "$PLAYER_BIO" \
        --game-logs "$GAME_LOGS" \
        --epoch latest \
        --max-games 0 \
        --min-shots-per-game 3 \
        --output-dir outputs/attention_diagnostics \
        --device cpu \
        --seed 0
