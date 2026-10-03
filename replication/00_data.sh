#!/usr/bin/env bash
# Stage 0: fetch the raw data from NBA Stats.
#
# Writes, in order, each skipped when already present:
#   $GAME_LOGS    per-player game logs, 2014-15 to 2024-25 (one request per season)
#   $SHOTS        every shot of every player-season in the logs (about an hour)
#   $STARTERS     starter status per player-game (one request per game, several hours)
#   $PLAYER_BIO   height, weight, position and birth date per player
# The play-by-play events of the optional presence model are not fetched here;
# see scripts/runs/fetch_play_by_play.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
begin "$@"

have "$GAME_LOGS" || run bash scripts/runs/fetch_game_logs.sh --output "$GAME_LOGS"
have "$SHOTS" || run bash scripts/runs/fetch_shots.sh --output "$SHOTS"
have "$STARTERS" || run bash scripts/runs/fetch_starters.sh --output "$STARTERS"
have "$PLAYER_BIO" || run bash scripts/runs/fetch_player_bio.sh --output "$PLAYER_BIO"
