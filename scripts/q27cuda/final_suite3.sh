#!/bin/bash
# DFlash 2 on CUDA, release build: the ten prompts with the MTP chain (the
# default) and with the DFlash sidecar beside the file (CMF_DFLASH=1, its
# router fitted at this card's prices), chat0 twice each (the first run pays
# the PTX cache).
#   final_suite3.sh <bin>
set -u
bin=$1
cd /root/cuda
m=/root/models/qwen38-27b-q4tp.cmf
r=src/scripts/q27cuda/run_prompts.sh
$r fin3-chain "$bin" $m 512 chat0 chat0 chat1 chat2 chat3 chat4 chat5 chat6 chat7 copy0 edit0 -- CMF_GPU=cuda
$r fin3-dflash "$bin" $m 512 chat0 chat0 chat1 chat2 chat3 chat4 chat5 chat6 chat7 copy0 edit0 -- CMF_GPU=cuda CMF_DFLASH=1
echo FINAL3_DONE > out/final3.done
