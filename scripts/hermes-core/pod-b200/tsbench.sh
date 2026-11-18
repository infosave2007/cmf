#!/bin/bash
# tsbench.sh NAME BATCH STEPS: the real recipe with a wall-clock stamp on every step line;
# prints the median train-step ms (the log) and the median wall ms between consecutive steps (steps 9+).
cd /root/hc/data
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
E=${E:-/root/hc/cmf/target/release/cortiq-embryo}
NAME=$1; B=$2; N=$3
EMBRYO_LOG_EVERY=${LOG:-1} $E birth --resume base1-step426200.ckpt \
  --shard en.u16:0.07 ru.u16:0.06 code.u16:0.02 math.u16:0.015 se.u16:0.07 docs.u16:0.03 b_hermes_code.u16:0.04 b_harness_code.u16:0.07 \
  b_ultrachat.u16:0.05 b_magpie.u16:0.06 b_chat_misc.u16:0.03 b_ot_answers.u16:0.03 b_multilingual.u16:0.04 b_multilingual_think.u16:0.05 \
  b_ot_think.u16:0.03 b_think_misc.u16:0.035 b_toolmind.u16:0.08 b_toucan.u16:0.08 b_nemotron.u16:0.045 b_tools_misc.u16:0.035 \
  b_lookup16.u16:0.04 b_q27_answers.u16:0.006 b_q27_chat.u16:0.007 b_q27_code.u16:0.007 --val rawval.u16 \
  --batch $B --seq 1024 --steps $((426200 + N)) --warmup 4 --lr 1e-4 --lr-restart --eval-every ${EVAL:-1000000} --save-every 1000000 \
  --carry --carry-reset-every 16 --carry-eot 32760 --dropless --seed 31 --out /root/hc/run/$NAME.ckpt 2>&1 \
  | python3 -u -c "
import sys,time
for l in sys.stdin: print(f\"{time.time():.4f} {l}\", end=\"\")" > /root/hc/run/$NAME.tlog
rm -f /root/hc/run/"$NAME".ckpt
python3 - /root/hc/run/$NAME.tlog $B <<PY
import sys,statistics
ts=[];ms=[]
for l in open(sys.argv[1]):
    p=l.split()
    if len(p)>2 and p[1]=="step":
        ts.append(float(p[0])); ms.append(float(p[p.index("ms")-1]))
lg=int(__import__("os").environ.get("LOG","1")); d=[(b-a)/lg for a,b in zip(ts,ts[1:])][max(8//lg,1):]
b=int(sys.argv[2]); w=statistics.median(d)*1000; m=statistics.median(ms[min(9,len(ms)-1):])
print(f"B{b}: train_step {m:.0f} ms ({b*1024/m*1000:.0f} tok/s), wall {w:.0f} ms/step ({b*1024/w*1000:.0f} tok/s) over {len(d)} steps")
PY
