#!/bin/bash
# cardq3b.sh: cardq3 without the fact check (it runs on the second pod) — the drafter sidecars
# bound to the two files that go out; then CARDQ5_DONE for the upload queue once cardq4 is over
# (the drafter's device memory is measured on the second pod too).
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
timeout 7200 bash -c "until grep -q CARDQ2_DONE logs/cardq2.out; do sleep 15; done"
for f in out/ripple-65b-q4kv8.cmf out/ripple-65b-q2hot5kv.cmf; do
  CMF_GPU=0 ./cortiq convert --model dflash2 --output $f --dflash-sidecar --quant q4tp 2>&1 | grep -a "wrote\|rror" | cut -c1-200
done
ls -la out/*.dflash.cmf
echo CARDQ3_DONE
timeout 7200 bash -c "until grep -q CARDQ4_DONE logs/cardq4.out; do sleep 15; done"
echo CARDQ5_DONE >> logs/cardq5.out
