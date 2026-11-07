#!/bin/bash
# research3.sh: after rebake + ft5 build — the owner's whole-experts idea (28 busiest experts a layer in q8, same
# bytes as 5 % neurons), the shared expert as a q8 tube (device + host parity), both together, a random-experts
# control, pick=err -> logs/research.txt
cd /root/rip && . fnx/env.sh
until grep -q REBAKE_DONE logs/rebake.txt 2>/dev/null && grep -q BUILD_DONE logs/build_ft5.log 2>/dev/null; do sleep 20; done
BIN=/root/rip/cortiq.ft5
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
bake whole28 fn/whole28.cmf --whole-experts 28
BIN=$BIN FILE=fn/whole28.cmf fnx/ppl.sh whole28 1536
BIN=$BIN FILE=fn/whole28.cmf fnx/ppl.sh whole28_128 128
hostppl whole28_128 fn/whole28.cmf 128
rm -f fn/whole28.cmf
bake sharedt fn/sharedt.cmf --frac 0 --shared-q8
BIN=$BIN FILE=fn/sharedt.cmf fnx/ppl.sh sharedt 1536
BIN=$BIN FILE=fn/sharedt.cmf fnx/ppl.sh sharedt_128 128
hostppl sharedt_128 fn/sharedt.cmf 128
rm -f fn/sharedt.cmf
bake whole28s fn/whole28s.cmf --whole-experts 28 --shared-q8
BIN=$BIN FILE=fn/whole28s.cmf fnx/ppl.sh whole28s 1536
bake rndw28 fn/rndw28.cmf --whole-experts 28 --pick random
BIN=$BIN FILE=fn/rndw28.cmf fnx/ppl.sh rndw28 1536
rm -f fn/rndw28.cmf
bake err5 fn/err5.cmf --frac 0.05 --pick err
BIN=$BIN FILE=fn/err5.cmf fnx/ppl.sh err5 1536
rm -f fn/err5.cmf
df -h /root | tail -1
echo RESEARCH_DONE
} > logs/research.txt 2>&1
