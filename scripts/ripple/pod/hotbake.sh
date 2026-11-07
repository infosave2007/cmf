#!/bin/bash
# hotbake.sh NAME STAT COLD [FLAGS...]: hf/COLD.cmf (a q2tp or q4tp file) with a layer's busiest FFN
# neurons (hot/STAT.mass, hotprobe.sh) kept in q8_2f from the 8-bit file, as out/ripple-65b-NAME.cmf;
# then its numbers over the first 2048 tokens of wikitext-2 test against the published model's
# (ppl.sh). FLAGS go to `cortiq hot-bake` (--frac 0.05, --pick random, --no-readout).
set -uo pipefail
cd /root/rip && mkdir -p out
export XDG_RUNTIME_DIR=/tmp
n=$1; s=$2; c=$3; shift 3
t0=$(date +%s)
./cortiq hot-bake hf/$c.cmf hf/ripple-65b-q8_2f.cmf hot/$s.mass out/ripple-65b-$n.cmf "$@" 2>&1 | grep -a "hot-bake\|GB\|rror" | cut -c1-220
echo "bake $n: $(( $(date +%s) - t0 )) s, $(ls -l out/ripple-65b-$n.cmf | awk '{print $5}') bytes"
bash scripts/ppl.sh ripple-65b-$n ${HOT_TOKENS:-2048}
echo HOT_DONE
