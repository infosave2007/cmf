#!/bin/bash
# hotprobe.sh STAT [TOKENS]: which FFN neurons the q4tp file uses most on wikitext-2 test text far
# from the scored part (100 000 tokens in): hot/STAT.mass.
set -uo pipefail
cd /root/rip && mkdir -p hot
export XDG_RUNTIME_DIR=/tmp
s=$1; t=${2:-8192}
./cortiq hot-probe hf/ripple-65b-q4tp.cmf data/wiki.test.raw hot/$s.mass --tokens $t --window 2048 --skip 100000 --stat $s 2>&1 | grep -a "hot-probe\|share\|layer \|rror" | cut -c1-200
echo PROBE_DONE
