#!/bin/bash
# graph_ab.sh TAG CTX...: prompt ingest of the chunked host walk against the batched wgpu
# graph (scalar GEMMs, coop GEMMs, coop with 512-row submits) on Vulkan, `bench --core`.
# BIN= the binary (default ./cortiq.graph), F= the model file. Output logs/gab_TAG.txt.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=${CMF_MAX_SEQ:-140000}
B=${BIN:-./cortiq.graph}; F=${F:-hf/cmf/spark-x2.6-4b-q4tp.cmf}; T=$1; shift; O=logs/gab_$T.txt; : > "$O"
VARIANTS=${VARIANTS:-"walk:CMF_DENSE_GRAPH_PREFILL=0 graph:CMF_DENSE_GRAPH_PREFILL=1 coop:CMF_DENSE_GRAPH_PREFILL=1,CMF_BATCH_COOP=1 coop512:CMF_DENSE_GRAPH_PREFILL=1,CMF_BATCH_COOP=1,CMF_BATCH_K=512"}
for n in "$@"; do
  for v in $VARIANTS; do
    name=${v%%:*}; envs=${v#*:}
    env ${envs//,/ } $B bench $F --core --ignore-eos --tokens ${TOKENS:-8} --ctx $n > logs/gab_$T.$name.$n.log 2>&1
    echo "$n | $name | $(grep -E "TTFT|Prompt|Decode|error|panick" logs/gab_$T.$name.$n.log | tr "\n" " " | tr -s " " | cut -c1-300)" | tee -a "$O"
  done
done
echo GAB_DONE >> "$O"
