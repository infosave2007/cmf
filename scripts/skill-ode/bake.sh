#!/bin/bash
# bake.sh TAG [extra bake args...]: cortiq skill bake of the ODE skill (all-token loss) from train.txt / held.txt
# in the current directory; BASE = the model (default cmf/qwen3-0.6b-f16.cmf), B = the cortiq binary
set -u
tag=$1; shift
B=${B:-cortiq}
BASE=${BASE:-cmf/qwen3-0.6b-f16.cmf}
mkdir -p runs/$tag
timeout ${BAKE_TIMEOUT:-5400} $B skill bake $BASE --files train.txt --held-files held.txt \
  --held ${HELD:-16} --calib-chunks ${CALIB:-127} --chunk ${CHUNK:-64} \
  --soft-fold --no-defrag --mask-name ode --checkpoint-dir runs/$tag/ckpt --skip-runtime-gate \
  --output runs/$tag/specialist.cmf "$@" > runs/$tag/bake.log 2>&1
echo "BAKE_EXIT $tag $?"
