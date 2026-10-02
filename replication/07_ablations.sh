#!/usr/bin/env bash
# Stage 7: comparison runs for the outcome-residual and stress-test tables.
#
# Trains, and scores against the mainline, each variant that has a launcher in
# scripts/:
#   joint_b2_with_frozen_count_and_ctx_v1     mainline without the outcome branch
#   joint_b2_outcome_spatial_hawkes_v1        within-game spatial self-excitation
#   joint_b2_outcome_stratified_kernel_v1     zone-stratified mixture kernel
#   joint_b2_outcome_causal_zone_bias_v1      causal zone-pair support bias
#   joint_b2_outcome_mode_routed_v1           mode-routed spatial decoder
# Five further training runs; skip with SKIP_ABLATIONS=1 in run_all.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
need "$MAINLINE/cloud_metrics.json" "$MAINLINE/density_metrics.json" outputs/count_head_v1/modules.pt
begin "$@"

NO_OUTCOME=outputs/joint_b2_with_frozen_count_and_ctx_v1
have "$NO_OUTCOME/modules.pt" || run bash scripts/run_experiment_b2.sh
have "$NO_OUTCOME/cloud_metrics.json" \
    || run bash scripts/runs/evaluate_shot_clouds.sh \
        --run-dir "$NO_OUTCOME" \
        --output "$NO_OUTCOME/cloud_metrics.json"

# variant directory : training script : evaluation script
VARIANTS=(
    "joint_b2_outcome_spatial_hawkes_v1:run_experiment_b1_spatial_hawkes.sh:eval_b1_spatial_hawkes.sh"
    "joint_b2_outcome_stratified_kernel_v1:run_experiment_c1_stratified_kernel.sh:eval_c1_stratified_kernel.sh"
    "joint_b2_outcome_causal_zone_bias_v1:run_experiment_alpha1_causal_zone_bias.sh:eval_alpha1_causal_zone_bias.sh"
    "joint_b2_outcome_mode_routed_v1:run_experiment_mode_routed_v1.sh:eval_mode_routed_v1.sh"
)
for variant in "${VARIANTS[@]}"; do
    IFS=: read -r name train_script eval_script <<< "$variant"
    have "outputs/$name/modules.pt" || run bash "scripts/$train_script"
    have "outputs/$name/cloud_metrics.json" "outputs/$name/density_metrics.json" \
        || run bash "scripts/$eval_script"
done
