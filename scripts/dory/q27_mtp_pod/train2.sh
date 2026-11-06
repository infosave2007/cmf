#!/bin/bash
# train2.sh NAME HARD STRIDE EPOCHS REFRESH: the chain of five laid out as rounds, the result as f16
cd /root/q27; export XDG_RUNTIME_DIR=/tmp
timeout 5400 ./cortiq3 mtp-heal m/qwen38-27b-turbo.cmf --data code/data/q27-code-t27.tdata@400000,own.tdata,own2.tdata \
  --chain 5 --stride "$3" --epochs "$4" --eval-every 127 --refresh "$5" --patience 0 --hard "$2" --trunk-cache /root/q27/tc \
  --codec f16 --output /root/q27/out/$1-f16.cmf > train$1.log 2>&1
