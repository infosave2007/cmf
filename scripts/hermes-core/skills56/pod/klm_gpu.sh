#!/bin/bash
# The KLM-1.6 index of trainer checkpoints on this pod's GPU (`cortiq-embryo loglik`, the numbers of the
# engine's CPU `cortiq loglik`: 0 different answers of 10 042 HellaSwag and 2376 ARC-Easy questions on f0):
#   klm_gpu.sh NAME=CKPT ...       runs under /root/sk/bench/runs/gpu-NAME; one summary line per model
# into /root/sk/run/klm.txt
B=/root/sk/bench; C=/root/sk/cmf/target/release/cortiq; H=/root/sk/cmf/scripts/embryo/bench; V=/root/sk/data/f0.cmf
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 HF_HOME=$B/hf-home CMF_GROWTH=off
for nc in "$@"; do
  NAME=${nc%%=*}; CK=${nc#*=}; O=$B/runs/gpu-$NAME; SECONDS=0
  EMBRYO_LOGLIK_CKPT=$CK $B/venv/bin/python -I $H/run_lmeval.py --cortiq $C --model $V --out-dir $O \
    --tasks arc_easy,arc_challenge,hellaswag,piqa > $O.lmeval.log 2>&1 || { echo "$NAME: lm-eval FAILED"; tail -3 $O.lmeval.log; continue; }
  EMBRYO_LOGLIK_CKPT=$CK $B/venv/bin/python -I $H/arithmark3_cortiq.py --cortiq $C --model $V --script $B/dl-arithmark3/bencharithmark-3.py \
    --data $B/dl-arithmark3/arithmark-3.jsonl --out-dir $O > $O.arith.log 2>&1 || { echo "$NAME: ArithMark FAILED"; tail -3 $O.arith.log; }
  python3 -I - "$O" "$NAME" "$SECONDS" <<'PY' | tee -a /root/sk/run/klm.txt
import json, sys
o, name, secs = sys.argv[1], sys.argv[2], sys.argv[3]
acc = lambda t: 100 * json.load(open(f"{o}/{t}.raw.results.json"))["results"][t]["acc_norm,none"]
n = lambda s, c: 100 * (s - c) / (100 - c)
ae, ac, hs, pq = acc("arc_easy"), acc("arc_challenge"), acc("hellaswag"), acc("piqa")
am = json.load(open(f"{o}/arithmark-3.results.json"))["results"]["arithmark-3"]["acc_norm"]
idx = (n(hs, 25) + n((ae + ac) / 2, 25) + n(pq, 50) + 0.65 * n(am, 25)) / 3.65
print(f"{name}: ARC-E {ae:.2f} ARC-C {ac:.2f} HellaSwag {hs:.2f} PIQA {pq:.2f} ArithMark-3 {am:.2f} -> index {idx:.2f} ({secs} s)")
PY
done
