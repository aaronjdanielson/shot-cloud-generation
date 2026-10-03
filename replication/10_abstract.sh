#!/usr/bin/env bash
# Stage 10: build the conference abstract.
#
# Redraws the abstract's figures from the stored evidence files and compiles
# abstract/SSAC27_submission/SSAC27_abstract.pdf. Requires pdflatex and
# pdftotext. Skipped when the abstract source is not present.
source "$(dirname "${BASH_SOURCE[0]}")/_stage.sh"
if [[ ! -f abstract/SSAC27_submission/build.sh ]]; then
    echo "skip: abstract/SSAC27_submission/build.sh not present"
    exit 0
fi
begin "$@"

run sh abstract/SSAC27_submission/build.sh
