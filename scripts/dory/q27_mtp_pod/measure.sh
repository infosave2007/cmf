#!/bin/bash
# measure.sh FILE LOG: a draft's hold rates per depth on the held-out windows, chains laid out as rounds
cd /root/q27; export XDG_RUNTIME_DIR=/tmp
timeout 900 ./cortiq3 mtp-heal "$1" --data code/data/q27-code-t27.tdata@400000,own.tdata,own2.tdata --chain "${3:-5}" --stride "${4:-8}" --epochs 0 --trunk-cache /root/q27/tc --output /root/q27/out/none-$(basename "$2" .log).cmf > "$2" 2>&1
