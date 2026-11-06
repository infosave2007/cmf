#!/bin/bash
# dflash-wgpu.sh [TOKENS] [ARM...]: the DFlash drafter on a card (wgpu). The sidecar from the
# published drafter with the router fitted on the Mac's rows, its block against the kit with
# the stage timers where the kit's rows were copied over (kit/*.f16 are links on the Mac), then
# the five prompts, one arm after another (q27_dflash_ab.sh): plain, the MTP chain, the block
# pinned. On wgpu the chain keeps the rounds unless the block is pinned. p/p_NAME.txt, rows/
# (packed with COPYFILE_DISABLE=1, or `._*` twins come along) and kit/ are in place.
cd /root/q27; export XDG_RUNTIME_DIR=/tmp
n=${1:-220}; shift; arms=("$@"); [ ${#arms[@]} -eq 0 ] && arms=(plain mtp dflash)
M=m/qwen38-27b-turbo.cmf
if [ ! -s m/qwen38-27b-turbo.dflash.cmf ]; then
  timeout 900 ./cortiq convert --model ext/dflash2 --output $M --dflash-sidecar --quant q4tp 2>&1 | tail -1 | cut -c1-200
  timeout 600 ./cortiq dflash-route-fit $M rows --edge 1.05 2>&1 | grep -E "^all sequences|^rank|^written" | cut -c1-240
fi
ls -la m/qwen38-27b-turbo.dflash.cmf
if [ -e "$(ls kit/*.f16 2> /dev/null | head -1)" ]; then
  echo "== the block against the kit"
  CMF_DFLASH_TIME=1 timeout 600 ./cortiq dflash-check $M kit --starts 72 2>&1 | grep -E "^lowest|^drafts|^a round|^a block"
  echo "== per matrix"
  CMF_DFLASH_TIME=1 CMF_DFLASH_FUSED=0 timeout 600 ./cortiq dflash-check $M kit --starts 72 2>&1 | grep -E "^lowest|^a round|^a block"
fi
rm -f ab.log ab.log.out ab.log.err
./q27_dflash_ab.sh ./cortiq $M p ab.log 0 "$n" code edit quote prose ru -- "${arms[@]}"
cat ab.log
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo DFLASH_WGPU_DONE
