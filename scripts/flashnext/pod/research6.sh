#!/bin/bash
# research6.sh: with ft8 (banks flushed per layer) — whole28 + shared (bake left running by research5), then the
# random-experts control, then whole28 alone again on the device -> appended to logs/research.txt
cd /root/rip && . fnx/env.sh
until grep -q BUILD_DONE logs/build_ft8.log 2>/dev/null; do sleep 15; done
until ! pgrep -x cortiq.ft7 > /dev/null; do sleep 15; done
BIN=/root/rip/cortiq.ft8
TURBO=/root/rip/fn/qwen38-flash-next-q2tp-turbo.cmf
HF=Qwen/Qwen3.8-Flash-Next
bake() { # NAME OUT ARGS...
  local n=$1 out=$2; shift 2; local t0=$(date +%s)
  $BIN hot-bake --experts $TURBO $HF hot/experts.stat $out "$@" --scratch /root/rip/src > logs/bake_$n.log 2>&1
  echo "bake $n: exit $? | $(( $(date +%s) - t0 )) s | $(sed -e "s/\x1b\[[0-9;]*m//g" logs/bake_$n.log | grep -a "holding\|->\|Error" | tr "\n" " " | cut -c1-300)"
}
{
echo "--- research6 $(date -u)"; df -h /root | tail -1
echo "whole28s bake: $(sed -e "s/\x1b\[[0-9;]*m//g" logs/bake_whole28s.log | grep -a "holding\|->\|Error" | tr "\n" " " | cut -c1-300)"
BIN=$BIN FILE=fn/whole28s.cmf fnx/ppl.sh whole28s 1536 RUST_BACKTRACE=1
bake rndw28 fn/rndw28.cmf --whole-experts 28 --pick random
BIN=$BIN FILE=fn/rndw28.cmf fnx/ppl.sh rndw28 1536
rm -f fn/rndw28.cmf
bake whole28 fn/whole28.cmf --whole-experts 28
BIN=$BIN FILE=fn/whole28.cmf fnx/ppl.sh whole28 1536
rm -f fn/whole28.cmf
df -h /root | tail -1
echo RESEARCH_DONE
} >> logs/research.txt 2>&1
