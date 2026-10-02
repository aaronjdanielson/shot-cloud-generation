#!/usr/bin/env bash
# Stage 1: build the causal snapshot store.
#
# Writes $SNAPSHOTS and its manifest: one bundle per monthly anchor, each built
# only from shots before the anchor. Every later stage reads it.
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
need "$SHOTS" "$GAME_LOGS" "$STARTERS"
begin "$@"

have "$SNAPSHOTS" || run bash scripts/runs/pretrain_snapshots.sh --output "$SNAPSHOTS"
