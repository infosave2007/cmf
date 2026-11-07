#!/bin/bash
# research2.sh: after rebake + ft4 build — the shared expert as a q8 tube (device ppl + host parity),
# pick=err and pick=random 5 % variants, 5 % + shared tube -> logs/research.txt
cd /root/rip && . fnx/env.sh
until grep -q REBAKE_DONE logs/rebake.txt 2>/dev/null && grep -q BUILD_DONE logs/build_ft4.log 2>/dev/null; do sleep 20; done
BIN=/root/rip/cortiq.ft4
TURBO=/root/rip/fn/qwen38-flash-next-q2tp-turbo.cmf
HF=Qwen/Qwen3.8-Flash-Next
bake() { # NAME OUT ARGS...
  local n=$1 out=$2; shift 2; local t0=$(date +%s)
  $BIN hot-bake --experts $TURBO $HF hot/experts.stat $out "$@" --scratch /root/rip/src > logs/bake_$n.log 2>&1
  echo "bake $n: exit $? | $(( $(date +%s) - t0 )) s | $(sed -e "s/\x1b\[[0-9;]*m//g" logs/bake_$n.log | grep -a "holding\|->" | tr "\n" " " | cut -c1-300)"
}
hostppl() { # NAME FILE TOKENS
  local t0=$(date +%s)
  CMF_GPU=0 CMF_QWEN_DEVICE=0 $BIN ppl $2 -f data/expert_store_v089.rs --tokens $3 > logs/hostppl_$1.log 2>&1
  echo "host $1: $(grep -a "PPL" logs/hostppl_$1.log | tail -1 | cut -c1-100) | $(( $(date +%s) - t0 )) s"
}
{
date -u; rm -f fn/hot25.cmf; df -h /root | tail -1
bake sharedt fn/sharedt.cmf --frac 0 --shared-q8
BIN=$BIN FILE=fn/sharedt.cmf fnx/ppl.sh sharedt 1536
BIN=$BIN FILE=fn/sharedt.cmf fnx/ppl.sh sharedt_128 128
hostppl sharedt_128 fn/sharedt.cmf 128
rm -f fn/sharedt.cmf
bake err5 fn/err5.cmf --frac 0.05 --pick err
BIN=$BIN FILE=fn/err5.cmf fnx/ppl.sh err5 1536
rm -f fn/err5.cmf
bake rnd5 fn/rnd5.cmf --frac 0.05 --pick random
BIN=$BIN FILE=fn/rnd5.cmf fnx/ppl.sh rnd5 1536
rm -f fn/rnd5.cmf
bake hot5s fn/hot5s.cmf --frac 0.05 --shared-q8
BIN=$BIN FILE=fn/hot5s.cmf fnx/ppl.sh hot5s 1536
df -h /root | tail -1
echo RESEARCH_DONE
} > logs/research.txt 2>&1
