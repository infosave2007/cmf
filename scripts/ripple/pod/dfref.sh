#!/bin/bash
# dfref.sh: the engine rows of a plain greedy answer to data/code.txt, then the published drafter
# over them in Python (scripts/ripple_dflash_ref.py), both ways of reading the block input rows.
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh > logs/build.out 2>&1
timeout 1500 bash -c "until grep -q FINALBAKE_DONE logs/finalbake.out; do sleep 5; done"
rm -rf dump/code
CMF_DFLASH_DUMP=dump/code bash scripts/dflash.sh dump code 100 CMF_GRAPH_SPEC=0
ls -la dump/code
python3 -I scripts/ripple_dflash_ref.py dump/code logs/dflash_dump_code.ids dflash2 7 40 2>&1 | grep -v Warning
echo DFREF_DONE
