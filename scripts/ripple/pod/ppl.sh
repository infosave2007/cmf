#!/bin/bash
# ppl.sh NAME [TOKENS] [ENV=VALUE ...]: out/NAME.cmf (CMF_DIR=hf for the published files) over the first TOKENS tokens of wikitext-2 test
# (default 2048), position by position against the published model's numbers (ref/wiki2048.npz).
set -uo pipefail
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
n=$1; t=${2:-2048}; shift; shift 2>/dev/null || true
t0=$(date +%s)
env "$@" CMF_PPL_TRACE=1 ./cortiq ppl ${CMF_DIR:-out}/$n.cmf -f data/wiki.test.raw --tokens $t > logs/ppl_$n.log 2>&1
echo "$n: $(grep -a 'PPL =' logs/ppl_$n.log | tail -1) | $(( $(date +%s) - t0 )) s | $(grep -a -c -i 'declin\|refus' logs/ppl_$n.log) decline lines"
python3 -I scripts/ppl_cmp.py ref/wiki2048.npz logs/ppl_$n.log
echo PPL_DONE
