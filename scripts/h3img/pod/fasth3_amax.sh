#!/bin/bash
# Stored f16 maxima of the Vulkan chain's sites (CMF_H3_WGPU_AMAX=1): VSA with
# its gated branch, without the branch, and dense — is a site near the f16
# limit once the branch is on?
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
m=cmf/fh/fasth3-q4tp.cmf
prompt="A corgi runs along a sunny beach, waves rolling in behind it, the camera tracking alongside; the dog barks twice."
for v in "gate:CMF_VSA_SPARSITY=0" "nogate:CMF_VSA_GATE=0" "dense:CMF_VSA=0"; do
  tag=${v%%:*}; envv=${v#*:}
  flock gpu.lock env CMF_GPU=1 CMF_H3_WGPU_AMAX=1 $envv timeout 3600 bin/cortiq animate $m --prompt "$prompt" \
    --width 512 --height 288 --frames 39 --steps 4 --seed 7 --out out/amax_$tag.avi > logs/amax_$tag.log 2>&1
  echo "== $tag: exit $?"
  grep "h3 wgpu amax" logs/amax_$tag.log | grep -v "encoder\|te " | head -8
done
echo AMAX_DONE
