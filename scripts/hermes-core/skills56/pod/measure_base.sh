#!/bin/bash
# A 56M checkpoint on general text and the KLM-1.6 tasks, on this pod's CPU:
#   measure_base.sh NAME [CKPT]     CKPT: a trainer checkpoint, exported to /root/sk/run/NAME.cmf (f32);
#                                   without it /root/sk/run/NAME.cmf (or /root/sk/data/NAME.cmf) is measured.
# Writes /root/sk/run/NAME.ppl (held-out English, 16 windows of 512, as on the Mac) and the lm-eval runs under
# /root/sk/bench/runs/NAME (ARC-Easy, ARC-Challenge, HellaSwag, PIQA zero-shot; ArithMark-3).
set -u
R=/root/sk/run; D=/root/sk/data; B=/root/sk/bench; C=/root/sk/cmf/target/release/cortiq
E=/root/sk/cmf/target/release/cortiq-embryo; H=/root/sk/cmf/scripts/embryo/bench
NAME="$1"; CK="${2:-}"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 HF_HOME=$B/hf-home CMF_GPU=0 CMF_GROWTH=off
M=$R/$NAME.cmf; [ -f "$M" ] || M=$D/$NAME.cmf
if [ -n "$CK" ]; then M=$R/$NAME.cmf; rm -f "$M"; $E export --ckpt "$CK" --tokenizer $D/gen1.tokenizer.json --out "$M" | tail -1; fi
SECONDS=0
"$C" ppl --file $D/heldout-en.txt --windows 16 --window-len 512 "$M" 2>&1 | tail -1 | tee $R/$NAME.ppl
CORTIQ_LM_JOBS="${JOBS:-48}" CMF_THREADS="${THREADS:-2}" $B/venv/bin/python -I $H/run_lmeval.py --cortiq "$C" --model "$M" \
  --out-dir $B/runs/$NAME --tasks arc_easy,arc_challenge,hellaswag,piqa > $B/runs.$NAME.lmeval.log 2>&1
tail -12 $B/runs.$NAME.lmeval.log
CMF_THREADS=8 $B/venv/bin/python -I $H/arithmark3_cortiq.py --cortiq "$C" --model "$M" --script $B/dl-arithmark3/bencharithmark-3.py \
  --data $B/dl-arithmark3/arithmark-3.jsonl --out-dir $B/runs/$NAME > $B/runs.$NAME.arith.log 2>&1
tail -4 $B/runs.$NAME.arith.log
echo "MEASURED $NAME in $SECONDS s"
