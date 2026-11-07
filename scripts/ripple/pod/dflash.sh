#!/bin/bash
# dflash.sh NAME PROMPT TOKENS [ENV=VALUE ...]: a greedy answer of FILE (default hf/ripple-65b-q4tp.cmf,
# its DFlash sidecar beside it) to the prompt in data/PROMPT.txt with the drafter's rounds on:
# drafts kept, what a round costs, the decode speed. `plain` as NAME runs without rounds.
# The answers' ids land in logs/dflash_NAME.ids for a comparison with the plain run's.
set -uo pipefail
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
n=$1; p=$2; t=$3; shift 3
f=${FILE:-hf/ripple-65b-q4tp.cmf}
if [ "$n" = plain ]; then set -- CMF_GRAPH_SPEC=0 "$@"; else set -- CMF_DFLASH=1 "$@"; fi
env "$@" RUST_LOG=${RUST_LOG:-info} CMF_GRAPH_SPEC_TIME=1 CMF_GEN_IDS_OUT=logs/dflash_${n}_$p.ids ./cortiq run $f --prompt "$(cat data/$p.txt)" --greedy --max-tokens $t > logs/dflash_${n}_$p.out 2> logs/dflash_${n}_$p.err
e=logs/dflash_${n}_$p.err
grep -a "DFlash" $e | sed -e 's/\x1b\[[0-9;]*m//g' | cut -c1-220 | head -3
echo "$n $p: $(grep -a -c 'solo-round:' $e) rounds | kept $(grep -a 'solo-round:' $e | awk '{k += $(NF-2); d += $NF} END {print k+0 " of " d+0}') | draft/verify/commit ms $(grep -a 'solo-round:' $e | awk '{a += $3; b += $7; c += $11; n++} END {if (n) printf "%.0f/%.0f/%.0f", a/n, b/n, c/n}') | $(grep -a -i -c 'declin\|refus\|failed' $e) decline or failure lines"
tail -c 400 logs/dflash_${n}_$p.out | tr '\n' ' ' | sed -e 's/\x1b\[[0-9;]*m//g' | grep -a -o "\[[^]]*tok/s[^]]*\]" | tail -1
[ -f logs/dflash_plain_$p.ids ] && [ "$n" != plain ] && (cmp -s logs/dflash_plain_$p.ids logs/dflash_${n}_$p.ids && echo "the plain run's ids" || echo "ids differ from the plain run's")
echo DFLASH_DONE
