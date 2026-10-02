#!/usr/bin/env bash
# Stage 10: build the conference abstract.
#
# Redraws the abstract's figures from the stored evidence files and compiles
# abstract/SSAC27_submission/SSAC27_abstract.pdf. Requires pdflatex and
# pdftotext.
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
begin "$@"

run sh abstract/SSAC27_submission/build.sh
