#!/bin/bash
# Keep-one stage table of the wgpu token path: tokens per second with only these
# dispatches kept; `none` keeps the head and the host's part of a token.
# stages.sh MODEL [TOKENS] [CTX]
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
M=$1; N=${2:-80}; C=$3
B=./src/target/release/cortiq
ctx=""; [ -n "$C" ] && ctx="--ctx $C"
for k in all none qkv g prep score soft sum gate o gu silu d slerp mix; do
    if [ "$k" = all ]; then unset CMF_DORY_KEEP; else export CMF_DORY_KEEP=$k; fi
    r=$(timeout 120 $B bench $M --core --tokens $N --ignore-eos $ctx 2>&1 | grep -oE "[0-9.]+ tok/s steady" | head -1 | cut -d' ' -f1)
    awk -v k=$k -v r="$r" 'BEGIN{printf "%-7s %8.1f tok/s %7.2f ms\n", k, r, 1000/r}'
done
