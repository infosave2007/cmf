#!/usr/bin/env bash
# A short birth on the real base1 recipe from the final 340M checkpoint; per-step log, peak memory, checkpoint sha256.
#   bench.sh NAME BATCH STEPS [extra birth flags]      env: SEQ [1024], SAVE (1 = keep the checkpoint hash)
set -u
NAME="$1"; B="$2"; N="$3"; shift 3
cd /root/hc/data
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
E=${E:-/root/hc/cmf/target/release/cortiq-embryo}
R=/root/hc/run
S0=426200
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits; sleep 1; done > $R/mem-$NAME.log 2>&1 ) &
W=$!
EMBRYO_LOG_EVERY=${LOG:-1} timeout 1800 $E birth --resume base1-step426200.ckpt \
  --shard en.u16:0.07 ru.u16:0.06 code.u16:0.02 math.u16:0.015 se.u16:0.07 docs.u16:0.03 b_hermes_code.u16:0.04 b_harness_code.u16:0.07 \
  b_ultrachat.u16:0.05 b_magpie.u16:0.06 b_chat_misc.u16:0.03 b_ot_answers.u16:0.03 b_multilingual.u16:0.04 b_multilingual_think.u16:0.05 \
  b_ot_think.u16:0.03 b_think_misc.u16:0.035 b_toolmind.u16:0.08 b_toucan.u16:0.08 b_nemotron.u16:0.045 b_tools_misc.u16:0.035 \
  b_lookup16.u16:0.04 b_q27_answers.u16:0.006 b_q27_chat.u16:0.007 b_q27_code.u16:0.007 --val rawval.u16 \
  --batch "$B" --seq ${SEQ:-1024} --steps $((S0 + N)) --warmup 4 --lr 1e-4 --lr-restart --eval-every 1000000 --save-every 1000000 \
  --carry --carry-reset-every 16 --carry-eot 32760 --dropless --seed 31 --out "$R/$NAME.ckpt" "$@" > "$R/$NAME.log" 2>&1
echo "exit $?"
kill $W 2>/dev/null
echo "peak MiB: $(cut -d, -f1 $R/mem-$NAME.log | sort -n | tail -1)  power W max: $(cut -d, -f3 $R/mem-$NAME.log | sort -n | tail -1)"
grep -E "panicked|OUT_OF_MEMORY|error" -A1 "$R/$NAME.log" | cut -c1-200 | head -6
grep -E "^step" "$R/$NAME.log" | tail -3 | cut -c1-160
grep -E "^step" "$R/$NAME.log" | awk "NR>8 {for(i=1;i<=NF;i++) if(\$i==\"ms\") print \$(i-1)}" | sort -n | awk "{a[NR]=\$1} END {m=a[int((NR+1)/2)]; printf \"median %d ms over %d steps -> %.0f tok/s\n\", m, NR, $B*${SEQ:-1024}*1000/m}"
if [ "${SAVE:-0}" = 1 ] && [ -f "$R/$NAME.ckpt" ]; then sha256sum "$R/$NAME.ckpt" | cut -c1-24; fi
rm -f /root/hc/run/"$NAME".ckpt
