#!/bin/bash
# sets.sh NAME FILE…: the chain of each file, 7 deep in the round layout, per data set
cd /root/q27; export XDG_RUNTIME_DIR=/tmp
for f in "$@"; do
  n=$(basename "$f" .cmf)
  timeout 900 ./cortiq3 mtp-heal "$f" --data code/data/q27-code-t27.tdata@400000,own.tdata,own2.tdata --chain 7 --stride 12 --epochs 0 --trunk-cache /root/q27/tc --output /root/q27/out/none-sets-$n.cmf > sets-$n.log 2>&1
  rm -f /root/q27/out/none-sets-$n.draft.cmf
done
echo done > sets.done
