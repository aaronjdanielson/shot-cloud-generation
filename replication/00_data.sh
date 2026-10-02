#!/usr/bin/env bash
# Stage 0: check the raw inputs and fetch the player bio table.
#
# Requires the shot data, per-game logs, and starters tables ($SHOTS,
# $GAME_LOGS, $STARTERS). Writes $PLAYER_BIO (height, weight, position, birth
# date per player) from the NBA Stats API; the fetch is rate-limited and
# resumable.
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
need "$SHOTS" "$GAME_LOGS" "$STARTERS"
begin "$@"

have "$PLAYER_BIO" || run bash scripts/runs/fetch_player_bio.sh --output "$PLAYER_BIO"
