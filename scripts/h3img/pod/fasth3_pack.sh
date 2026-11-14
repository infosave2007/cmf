#!/bin/bash
# FastH3 (FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) into CMF on the pod:
# the release build of the branch, the last shard, the DiT packed on its own
# (the shards deleted right after: 70 GB on a 100 GB disk), then the encoder
# and both decoders carried from the MiniMax-H3 Turbo container.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
src=fasth3
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  bash scripts/build.sh release > logs/fh_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -40 logs/fh_build.log; exit 1; }
  echo "build ok"
fi
if [ ! -s $src/transformer/diffusion_pytorch_model-00014-of-00014.safetensors ]; then
  timeout 1800 venv/bin/hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
    --include transformer/diffusion_pytorch_model-00014-of-00014.safetensors --local-dir $src > logs/fh_dl2.log 2>&1 \
    || { echo SHARD14_FAILED; tail -3 logs/fh_dl2.log; exit 1; }
fi
mkdir -p cmf/fh
if [ ! -s cmf/fh/fasth3-dit-q4tp.cmf ]; then
  t0=$(date +%s)
  timeout 7200 bin/cortiq animate-pack --dit-diffusers $src/transformer --quant q4tp --out cmf/fh/fasth3-dit-q4tp.cmf > logs/fh_pack_dit.log 2>&1 \
    || { echo PACK_DIT_FAILED; tail -20 logs/fh_pack_dit.log; exit 1; }
  echo "dit packed in $(( $(date +%s) - t0 )) s: $(ls -l cmf/fh/fasth3-dit-q4tp.cmf | awk '{print $5}') bytes"
  grep -E "adaLN curve|dit packed" logs/fh_pack_dit.log | tail -3
fi
# the shards are no longer needed
rm -f $src/transformer/diffusion_pytorch_model-000*-of-00014.safetensors
rm -rf $src/.cache
echo "disk $(df -h /root | awk 'NR==2{print $4}') free"
if [ ! -s cmf/fh/mmh3-turbo-q4tp.cmf ]; then
  timeout 3600 venv/bin/hf download infosave/MiniMax-H3-Turbo-cmf mmh3-turbo-q4tp.cmf --local-dir cmf/fh > logs/fh_dl_turbo.log 2>&1 \
    || { echo TURBO_DL_FAILED; tail -3 logs/fh_dl_turbo.log; exit 1; }
fi
timeout 3600 bin/cortiq animate-pack --in cmf/fh/mmh3-turbo-q4tp.cmf --dit-from cmf/fh/fasth3-dit-q4tp.cmf \
  --quant q4tp --out cmf/fh/fasth3-q4tp.cmf > logs/fh_pack_full.log 2>&1 \
  || { echo PACK_FULL_FAILED; tail -20 logs/fh_pack_full.log; exit 1; }
ls -l cmf/fh
bin/cortiq info cmf/fh/fasth3-q4tp.cmf 2>&1 | head -20
echo "disk $(df -h /root | awk 'NR==2{print $4}') free"
echo FH_PACK_DONE
