#!/bin/bash
# sidecar_spec.sh: the q4tp trunk sidecar on this card, then the drafter against plain decode on code and prose.
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
CMF_GPU=0 ./cortiq convert --model hfdl/dflash2 --output out/ripple-65b-q4kv8.cmf --dflash-sidecar --quant q4tp 2>&1 | grep -a "wrote\|rror" | cut -c1-200
ls -la out/*.dflash.cmf
for p in code prose; do
  FILE=out/ripple-65b-q4kv8.cmf CMF_GPU_PROBE=0 bash scripts/dflash.sh plain $p 160
  FILE=out/ripple-65b-q4kv8.cmf CMF_GPU_PROBE=0 bash scripts/dflash.sh auto $p 160
  FILE=out/ripple-65b-q4kv8.cmf CMF_GPU_PROBE=0 bash scripts/dflash.sh k7 $p 160 CMF_GRAPH_SPEC_K=7
  FILE=out/ripple-65b-q4kv8.cmf CMF_GPU_PROBE=0 bash scripts/dflash.sh k4 $p 160 CMF_GRAPH_SPEC_K=4
done
echo SPEC_DONE
