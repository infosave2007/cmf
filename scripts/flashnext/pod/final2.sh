#!/bin/bash
# final2.sh RECIPE...: after research6 — bake the final hot file with RECIPE (e.g. --frac 0 --shared-q8), MTP copy,
# sha256, parity check, code ppl, wiki 12x512 (turbo + hot), old-binary behaviour, generation sanity, then the quiet
# benches (exact vs far field 1k/4k/16k/32k, hot 1k/4k) -> logs/final2.txt
cd /root/rip && . fnx/env.sh
until grep -q RESEARCH_DONE logs/research.txt 2>/dev/null; do sleep 20; done
BIN=/root/rip/cortiq.ft8
HOT=/root/rip/fn/qwen38-flash-next-q2tp-hot-turbo.cmf
TURBO=/root/rip/fn/qwen38-flash-next-q2tp-turbo.cmf
HF=Qwen/Qwen3.8-Flash-Next
warm() { t0=$(date +%s); cat $1 > /dev/null; echo "warmed $(basename $1) in $(( $(date +%s) - t0 )) s"; }
{
echo "--- final2 $(date -u) recipe: $*"; df -h /root | tail -1
t0=$(date +%s)
[ -f $HOT ] && echo "final file present, bake skipped" || $BIN hot-bake --experts $TURBO $HF hot/experts.stat $HOT "$@" --scratch /root/rip/src > logs/bake_final.log 2>&1
echo "bake final: exit $? | $(( $(date +%s) - t0 )) s | $(sed -e "s/\x1b\[[0-9;]*m//g" logs/bake_final.log | grep -a "holding\|->\|Error" | tr "\n" " " | cut -c1-300)"
cp fn/qwen38-flash-next-q2tp-turbo.mtp.cmf fn/qwen38-flash-next-q2tp-hot-turbo.mtp.cmf
(cd fn && sha256sum qwen38-flash-next-q2tp-hot-turbo.cmf > qwen38-flash-next-q2tp-hot-turbo.cmf.sha256 && sha256sum qwen38-flash-next-q2tp-hot-turbo.mtp.cmf > qwen38-flash-next-q2tp-hot-turbo.mtp.cmf.sha256); cat fn/qwen38-flash-next-q2tp-hot-turbo*.sha256
ls -l fn/qwen38-flash-next-q2tp-hot-turbo.cmf
BIN=$BIN FILE=$HOT fnx/ppl.sh final 1536
BIN=$BIN FILE=$HOT fnx/pplf.sh final_w12 data/wiki.test.raw 512 --windows 12 --window-len 512
# device vs host on the same short prompt (the device check needs one token a chunk)
for pair in turbo:$TURBO hot:$HOT; do
  n=${pair%%:*}; f=${pair#*:}
  CMF_QWEN_DEVICE_CHECK=1 CMF_QWEN_PREFILL_CHUNK=1 CMF_QWEN_PROFILE=$PROFILE $BIN run $f --prompt "Write a Rust function that reverses a string, with a test." --max-tokens 12 --greedy --no-think > logs/check2_$n.out 2> logs/check2_$n.err
  echo "check $n: $(grep -a -c qwen4-check logs/check2_$n.err) rows; $(grep -a qwen4-check logs/check2_$n.err | awk "{print \$3, \$4}" | tr "\n" " " | cut -c1-300)"
done
# generation sanity, greedy, no thinking
for pair in turbo:$TURBO hot:$HOT; do
  n=${pair%%:*}; f=${pair#*:}
  CMF_QWEN_PROFILE=$PROFILE $BIN run $f --prompt "Write a Rust function that reverses a string, with a test." --max-tokens 160 --greedy --no-think > logs/gen2_$n.out 2> logs/gen2_$n.err
  echo "gen $n: $(grep -a -i "tok/s" logs/gen2_$n.err | tail -1 | cut -c1-120)"; echo "   $(grep -a -v "^\[" logs/gen2_$n.out | tr "\n" " " | cut -c1-400)"
done
echo "old binary (0.8.60) on the hot file:"; BIN=/root/rip/cortiq FILE=$HOT fnx/ppl.sh hot_old 256
warm $TURBO
for c in 1000 4000 16000 32000; do
  BIN=$BIN FILE=$TURBO fnx/bench.sh exactq_$c $c CMF_QWEN_O1=off
  BIN=$BIN FILE=$TURBO fnx/bench.sh turboq_$c $c
done
warm $HOT
for c in 1000 4000; do BIN=$BIN FILE=$HOT fnx/bench.sh hotq_$c $c; done
echo FINAL2_DONE
} > logs/final2.txt 2>&1
