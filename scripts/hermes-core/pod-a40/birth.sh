#!/usr/bin/env bash
# A short `birth` on the pod: the grown 340M checkpoint on the text corpus,
# with the card's peak memory and the sha256 of the saved checkpoint (the
# bit-identity gate of a kernel change: same steps, same seed, one hash).
#
#   birth.sh NAME BATCH STEPS [extra birth flags, e.g. --dropless]
#
# Environment: SEQ [1024], RESET (--carry-reset-every) [16], EVAL
# (--eval-every) [100000], WARMUP [4]; the trainer's own switches pass
# through (CMF_CUDA_TF32=0, CMF_CUDA_NATIVE=0, CMF_CUDA_NATIVE_SKIP=...,
# CMF_CUDA_MAIN_SPEC=0, CMF_CUDA_CUBLAS=0, CMF_VULKAN_OP_TS=1).
set -u
NAME="$1"; B="$2"; STEPS="$3"; shift 3
cd /root/hc/data
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
E=/root/hc/cmf/target/release/cortiq-embryo
R=/root/hc/run
( for i in $(seq 1 400); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done > $R/mem-$NAME.log 2>&1 ) &
W=$!
timeout 900 $E birth --resume grown340.ckpt --shard se.u16:0.884 docs.u16:0.076 hermes_code.u16:0.040 --val val.u16 \
  --batch "$B" --seq ${SEQ:-1024} --steps "$STEPS" --warmup ${WARMUP:-4} --lr 1e-4 --lr-restart --eval-every ${EVAL:-100000} --save-every 100000 \
  --carry --carry-reset-every ${RESET:-16} --carry-eot 32760 --seed 17 --out $R/$NAME.ckpt "$@" > $R/$NAME.log 2>&1
echo "exit $?"
kill $W 2>/dev/null
echo "peak MiB: $(sort -n $R/mem-$NAME.log | tail -1)"
grep -E "panicked|OUT_OF_MEMORY" -A1 $R/$NAME.log | cut -c1-200
grep -E "^step|step " $R/$NAME.log | tail -6 | cut -c1-200
[ -f $R/$NAME.ckpt ] && sha256sum $R/$NAME.ckpt | cut -c1-24
