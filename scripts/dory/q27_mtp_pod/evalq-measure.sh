#!/bin/bash
# evalq-measure.sh: the trunk's answers to the 36 fresh prompts (evalq-gen.sh) as text no
# training read, rounds in order, 7 deep: the stock block, head C, DFlash 2 as published
# and with its matrices rounded to 4 bits
cd /root/q27; export XDG_RUNTIME_DIR=/tmp
D=evalq-code.tdata,evalq-en.tdata,evalq-ru.tdata
timeout 1200 bash -c 'until [ -f evalq-gen.done ]; do sleep 5; done' || { echo "no evalq-gen.done" > evalq-measure.done; exit 1; }
CMF_MTP_HOLD_ALL=1 timeout 900 ./cortiq3 mtp-heal m/qwen38-27b-turbo.cmf --data $D --chain 7 --stride 12 --epochs 0 --trunk-cache /root/q27/tc --taps /root/q27/evalq-taps=5,19,33,47,61 --output /root/q27/out/none-evalq-stock.cmf > evalq-stock.log 2>&1
CMF_MTP_HOLD_ALL=1 timeout 900 ./cortiq3 mtp-heal out/C-q8.cmf --data $D --chain 7 --stride 12 --epochs 0 --trunk-cache /root/q27/tc --output /root/q27/out/none-evalq-C.cmf > evalq-C.log 2>&1
rm -f /root/q27/out/none-evalq-*.cmf
# the data sets by their windows, in the order mtp-heal read them
S=$(sed -nE 's/^mtp heal: evalq-([a-z]+): ([0-9]+) windows.*/\1:\2/p' evalq-stock.log | paste -sd, -)
echo "sets $S" > evalq-sets.txt
SETS="$S" timeout 900 python3 q27_ext_draft.py evalq-taps dump ext/dflash2 12 7 > evalq-dflash2.log 2>&1
QUANT=q4 SETS="$S" timeout 900 python3 q27_ext_draft.py evalq-taps dump ext/dflash2 12 7 > evalq-dflash2-q4.log 2>&1
echo done > evalq-measure.done
