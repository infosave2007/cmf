#!/bin/bash
# after.sh: after the hot chain — plain generations hot vs turbo, the parity check with a 1-token prefill, the quiet benches, 64k -> logs/after.txt
cd /root/rip && . fnx/env.sh
until grep -q HOTCHAIN_DONE logs/hotchain.txt 2>/dev/null; do sleep 20; done
HOT=/root/rip/fn/qwen38-flash-next-q2tp-hot-turbo.cmf
TURBO=/root/rip/fn/qwen38-flash-next-q2tp-turbo.cmf
warm() { t0=$(date +%s); cat $1 > /dev/null; echo "warmed $(basename $1) in $(( $(date +%s) - t0 )) s"; }
{
date -u
for pair in turbo:$TURBO hot:$HOT; do
  n=${pair%%:*}; f=${pair#*:}
  CMF_QWEN_PROFILE=$PROFILE $BIN run $f --prompt "Write a Rust function that reverses a string, with a test." --max-tokens 160 --greedy --no-think > logs/gen_$n.out 2> logs/gen_$n.err
  echo "gen $n: $(grep -a "tokens ·" logs/gen_$n.out | tail -1 | cut -c1-120) $(grep -a "MTP draft" logs/gen_$n.out | tail -1)"
  echo "   text: $(grep -a -v "^\[\|^Loading\|^Ready\|^Prompt\|^$" logs/gen_$n.out | head -c 400 | tr "\n" " ")"
done
for pair in turbo:$TURBO hot:$HOT; do
  n=${pair%%:*}; f=${pair#*:}
  CMF_QWEN_DEVICE_CHECK=1 CMF_QWEN_PREFILL_CHUNK=1 CMF_QWEN_PROFILE=$PROFILE $BIN run $f --prompt "fn reverse(s: &str) -> String {" --max-tokens 8 --greedy --raw > logs/check2_$n.out 2> logs/check2_$n.err
  echo "check $n: $(grep -a -c qwen4-check logs/check2_$n.err) rows; $(grep -a qwen4-check logs/check2_$n.err | awk "{print \$3, \$4, \$5, \$6}" | tr "\n" "|" | cut -c1-600)"
  echo "   text: $(grep -a -v "^\[\|^Loading\|^Ready\|^Prompt\|^$" logs/check2_$n.out | head -c 200 | tr "\n" " ")"
done
warm $TURBO
for c in 1000 4000 16000 32000; do
  FILE=$TURBO fnx/bench.sh exactq_$c $c CMF_QWEN_O1=off
  FILE=$TURBO fnx/bench.sh turboq_$c $c
done
warm $HOT
for c in 1000 4000; do FILE=$HOT fnx/bench.sh hotq_$c $c; done
echo FINAL_DONE
warm $TURBO
FILE=$TURBO fnx/bench.sh turboq_64000 64000
FILE=$TURBO fnx/bench.sh exactq_64000 64000 CMF_QWEN_O1=off
echo AFTER_DONE
} > logs/after.txt 2>&1
