# shellcheck shell=bash
#
# Shared setup for the replication stages. Source it, then call ``begin "$@"``:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
#     begin "$@"
#
# Sourcing loads scripts/runs/_common.sh (repository root as working directory,
# data locations, device). ``begin`` detaches the stage under
# ``nohup caffeinate`` unless it is already part of a detached run or
# ``SHOTCLOUD_FOREGROUND=1`` is set; every script the stage calls then runs
# inline, in order, in the stage's own log.
#
# Stages write to the paths the paper and the figure scripts read. A step whose
# outputs already exist is skipped, so finished work is never recomputed or
# overwritten; set ``FORCE=1`` to run it anyway. With ``DRY_RUN=1`` a stage
# prints the commands it would run and executes nothing.

_STAGE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)/$(basename "${BASH_SOURCE[1]}")"
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/runs/_common.sh"
_RUN_SCRIPT="$_STAGE_SCRIPT"

MAINLINE=outputs/joint_b2_outcome_residual_v1

begin() {
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        detach "$@"
    fi
    export SHOTCLOUD_DETACHED=1
    echo "=== $(basename "$_STAGE_SCRIPT" .sh) ==="
}

# have PATH...: succeed (and say so) when every path exists and FORCE is not 1.
have() {
    if [[ "${FORCE:-0}" == 1 ]]; then
        return 1
    fi
    local path
    for path in "$@"; do
        [[ -e "$path" ]] || return 1
    done
    echo "skip: $* present (FORCE=1 to recompute)"
}

# need PATH...: exit with an error naming the first missing prerequisite.
need() {
    local path
    for path in "$@"; do
        if [[ ! -e "$path" ]]; then
            echo "error: missing prerequisite $path" >&2
            [[ "${DRY_RUN:-0}" == 1 ]] || exit 1
        fi
    done
}

# run COMMAND...: execute the command, or only print it when DRY_RUN=1.
run() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "would run: $*"
    else
        "$@"
    fi
}
