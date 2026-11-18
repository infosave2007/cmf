#!/bin/bash
# decode_prof.sh: Vulkan decode, GPU span per token against the wall, host encode per layer part.
cd /root/sn
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=270000 CMF_MTP=0
B=${1:-bin/cortiq-fast}
CMF_GPU_TS=1 CMF_GRAPH_DEVICE_SPAN=1 CMF_GRAPH_HOSTPROF=1 $B bench out/sensenova-loop-9b-q4tp.cmf --core --ignore-eos --tokens 64 --ctx 512 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > logs/decode_prof.txt
grep -E "Decode:" logs/decode_prof.txt
grep -oE "span_ms=[0-9.]+ slots=[0-9]+ submits=[0-9]+" logs/decode_prof.txt | tail -40 | awk -F'[= ]' '{s+=$2; n++; sub_=$6} END {printf "device span %.2f ms/token over %d tokens, submits %s\n", s/n, n, sub_}'
grep hostprof logs/decode_prof.txt | tail -5
echo DPROF_DONE
