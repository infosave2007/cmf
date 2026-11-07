#!/bin/bash
# cardq4.sh: the drafter beside the q2tp file, one code prompt: plain and the adaptive depth.
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
timeout 9000 bash -c "until grep -q CARDQ3_DONE logs/cardq3.out; do sleep 15; done"
FILE=out/ripple-65b-q2hot5kv.cmf bash scripts/dflash.sh q2plain code 160
FILE=out/ripple-65b-q2hot5kv.cmf bash scripts/dflash.sh q2auto code 160 CMF_DFLASH_TIME=1
echo CARDQ4_DONE
