#!/bin/bash
# convert.sh [QUANT] [NAME] [extra convert flags]: the BF16 source in /root/rip/src as out/NAME.cmf
# (default q4tp, ripple-65b-q4tp). The log keeps the wall time and the peak of the container's memory.
set -uo pipefail
cd /root/rip && mkdir -p out
q=${1:-q4tp}; n=${2:-ripple-65b-$q}; shift 2 2>/dev/null || shift $# 
t0=$(date +%s)
./cortiq convert --model src --output out/$n.cmf --quant "$q" "$@" 2>&1 | grep -v "@PROGRESS" | tail -8
echo "convert $n: $(( $(date +%s) - t0 )) s, $(du -h out/$n.cmf | cut -f1), container peak $(( $(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null || echo 0) / 1048576 )) MiB"
echo CONVERT_DONE
