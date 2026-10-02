#!/usr/bin/env bash
# Run the replication pipeline end to end, stopping at the first failure.
#
# Detaches once under nohup + caffeinate and writes a single log to logs/.
# Steps whose outputs already exist are skipped (FORCE=1 recomputes them).
# Set SKIP_ABLATIONS=1 to leave out stage 7, the five comparison training runs.
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
begin "$@"

for stage in replication/[0-9][0-9]_*.sh; do
    if [[ "${SKIP_ABLATIONS:-0}" == 1 && "$stage" == *_ablations.sh ]]; then
        echo "skip: $stage (SKIP_ABLATIONS=1)"
        continue
    fi
    bash "$stage"
done
echo "=== replication complete ==="
