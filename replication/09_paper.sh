#!/usr/bin/env bash
# Stage 9: compile the paper.
#
# Writes paper/shot_cloud.pdf. Requires latexmk and the figures from stage 8.
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
need outputs/figures/paper/self_bootstrap_floor.png
begin "$@"

cd paper
run latexmk -pdf -interaction=nonstopmode shot_cloud.tex
