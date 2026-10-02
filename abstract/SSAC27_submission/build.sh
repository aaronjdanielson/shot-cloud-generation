#!/bin/sh
# Build the SSAC27 abstract from this folder: redraw the comparison figure from evidence/,
# compile twice, keep the PDF beside the source and print the word counts.
#
#   sh abstract/SSAC27_submission/build.sh     (from anywhere; needs pdflatex, pdftotext, python3 + numpy + matplotlib)
#
# What it does not do: refresh evidence/ or figures/attention_over_support.png. Those are copies
# of outputs/<run>/{cloud,density}_metrics.json and outputs/figures/paper/attention_over_support.png;
# evidence/mdn_game_match.json is written by match_mdn_games.py (needs the shot data and the package).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
TEX=SSAC27_abstract.tex

# 1. Figure 2 and the numbers quoted in Results.
python3 make_attention_figure.py
python3 make_comparison_figure.py

# 2. Compile twice into build/ (clutter stays there); the PDF is copied beside the source.
mkdir -p build
for i in 1 2; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$TEX" > build/compile.log 2>&1 \
        || { echo "COMPILE FAILED:"; grep -n -A6 '^!' build/compile.log | head -40; exit 1; }
done
cp build/SSAC27_abstract.pdf SSAC27_abstract.pdf
echo "built   SSAC27_abstract.pdf: $(pdfinfo SSAC27_abstract.pdf | awk '/Pages/{print $2}') pages, $(grep -c '^!' build/compile.log || true) errors, $(grep -c 'Overfull' build/compile.log || true) overfull boxes"

# 3. Counts. The rule is fewer than 500 words including the title. The strict reading counts
#    everything printed outside the figures: title, author block, body, both captions,
#    references and the link. Labels inside the comparison figure are reported separately
#    (the attention figure is a bitmap, so its labels are not in the PDF text).
ALL=$(pdftotext SSAC27_abstract.pdf - | wc -w | tr -d ' ')
FIG=$(pdftotext figures/fig_method_comparison.pdf - | wc -w | tr -d ' ')
echo "printed words: $ALL in all, $((ALL - FIG)) outside the figures (title, authors, body, captions, references, link), $FIG inside the comparison figure"
