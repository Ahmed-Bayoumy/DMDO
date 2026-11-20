#!/bin/sh
# Rebuild the SBJ formulation diagrams (needs pdflatex with TikZ, and pdftocairo from poppler).
cd "$(dirname "$0")" || exit 1
for f in a b c d; do
  pdflatex -interaction=nonstopmode -halt-on-error "fig6$f.tex" > /dev/null && pdftocairo -png -r 160 -singlefile "fig6$f.pdf" "../fig6$f"
done
rm -f ./*.aux ./*.log ./*.pdf
