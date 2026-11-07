#!/bin/bash
# hotchain.sh: after the bake and the bench chain — parity check, code ppl, wiki 12x512, benches of the hot file -> logs/hotchain.txt
cd /root/rip && . fnx/env.sh
until grep -q BAKE_DONE logs/bake.txt 2>/dev/null && grep -q BENCHCHAIN2_DONE logs/benchchain2.txt 2>/dev/null; do sleep 20; done
HOT=/root/rip/fn/qwen38-flash-next-q2tp-hot-turbo.cmf
TURBO=/root/rip/fn/qwen38-flash-next-q2tp-turbo.cmf
warm() { t0=$(date +%s); cat $1 > /dev/null; echo "warmed $(basename $1) in $(( $(date +%s) - t0 )) s"; }
{
cp fn/qwen38-flash-next-q2tp-v2.mtp.cmf fn/qwen38-flash-next-q2tp-hot-turbo.mtp.cmf
for pair in turbo:$TURBO hot:$HOT; do
  n=${pair%%:*}; f=${pair#*:}
  CMF_QWEN_DEVICE_CHECK=1 CMF_QWEN_PROFILE=$PROFILE $BIN run $f --prompt "Write a Rust function that reverses a string, with a test." --max-tokens 12 --greedy --no-think > logs/check_$n.out 2> logs/check_$n.err
  echo "check $n: $(grep -a -c qwen4-check logs/check_$n.err) rows; $(grep -a qwen4-check logs/check_$n.err | awk "{print \$3, \$4}" | tr "\n" " " | cut -c1-300)"
  echo "   text: $(grep -a -v "^\[" logs/check_$n.out | tr "\n" " " | cut -c1-160)"
done
FILE=$HOT fnx/ppl.sh hot 1536
FILE=$TURBO fnx/pplf.sh turbo_w12 data/wiki.test.raw 512 --windows 12 --window-len 512
FILE=$HOT fnx/pplf.sh hot_w12 data/wiki.test.raw 512 --windows 12 --window-len 512
warm $TURBO
FILE=$TURBO fnx/bench.sh exact_1000 1000 CMF_QWEN_O1=off
warm $HOT
FILE=$HOT fnx/bench.sh hot_1000 1000
FILE=$HOT fnx/bench.sh hot_4000 4000
echo "old binary on the hot file:"; FILE=$HOT BIN=/root/rip/cortiq fnx/ppl.sh hot_old 256
echo HOTCHAIN_DONE
} > logs/hotchain.txt 2>&1
