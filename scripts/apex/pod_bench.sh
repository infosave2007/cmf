#!/bin/bash
# Speed of an Apex-35B CMF on the pod's GPU (Vulkan), as shipped (no env):
# decode after a short prompt, then prompt ingest and decode at 4k and 16k
# tokens of context. One model process at a time; each bench's untimed
# warmup pays shader compile and weight upload.
#
# usage: pod_bench.sh MODEL.cmf [CORTIQ]
set -u
M=$1
B=${2:-/root/apex/bin/cortiq}
export XDG_RUNTIME_DIR=/tmp
echo "== $(basename "$M") $(date -u +%FT%TZ)"
"$B" --version
echo "-- short prompt, 256 tokens"
"$B" bench "$M" --core --ignore-eos --tokens 256 2>&1 | grep -a -E "Prompt:|Decode:|MTP:|GPU:"
for ctx in 4096 16384; do
  echo "-- ctx $ctx, 128 tokens"
  "$B" bench "$M" --core --ignore-eos --tokens 128 --ctx "$ctx" 2>&1 | grep -a -E "Prompt:|Decode:|MTP:|Memory:"
done
echo "BENCH_DONE"
