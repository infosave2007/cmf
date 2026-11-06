#!/bin/bash
# train.sh NAME HARD: the draft head as a chain of five on the three sets, the result as f16
cd /root/q27; export XDG_RUNTIME_DIR=/tmp
timeout 5400 ./cortiq3 mtp-heal m/qwen38-27b-turbo.cmf --data code/data/q27-code-t27.tdata@400000,own.tdata,own2.tdata \
  --chain 5 --epochs 2 --eval-every 127 --refresh 400 --patience 0 --hard "$2" --trunk-cache /root/q27/tc \
  --codec f16 --output /root/q27/out/$1-f16.cmf > train$1.log 2>&1
