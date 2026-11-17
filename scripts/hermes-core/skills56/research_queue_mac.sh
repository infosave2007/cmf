#!/usr/bin/env bash
# The Mac's measurements of the research arms, one after another (one run at a time on the Mac):
#   research_queue_mac.sh NAME...       each ~/embryo-data/hermes-core/skills56/research/NAME.cmf
# Held-out English perplexity (16 windows of 512, as measure.sh) and the KLM index (measure_klm.sh);
# results appended to ~/embryo-data/hermes-core/skills56/research/results.txt
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../../.." && pwd)"
R="$HOME/embryo-data/hermes-core/skills56/research"; C="$REPO/target/release/cortiq"
# wait for a KLM run already going (its own queue)
while pgrep -f "measure_klm.s[h]" >/dev/null; do sleep 30; done
for n in "$@"; do
  p=$(CMF_GPU=0 CMF_GROWTH=off "$C" ppl --file "$HOME/embryo-data/heldout-en.txt" --windows 16 --window-len 512 "$R/$n.cmf" 2>&1 | tail -1)
  echo "$n heldout-en: $p" | tee -a "$R/results.txt"
  "$HERE/measure_klm.sh" "$n" "$R/$n.cmf" 2>&1 | tail -1 | tee -a "$R/results.txt"
done
echo "QUEUE DONE $*" >> "$R/results.txt"
