#!/bin/bash
# cardq3.sh: after the drafter runs — the drafter sidecars bound to the two files that go out, the
# exact-rows dump and the fact check of the q4tp file.
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
timeout 7200 bash -c "until grep -q CARDQ2_DONE logs/cardq2.out; do sleep 15; done"
for f in out/ripple-65b-q4kv8.cmf out/ripple-65b-q2hot5kv.cmf; do
  CMF_GPU=0 ./cortiq convert --model dflash2 --output $f --dflash-sidecar --quant q4tp 2>&1 | grep -a "wrote\|rror" | cut -c1-200
done
ls -la out/*.dflash.cmf
bash scripts/quality.sh out/ripple-65b-q4kv8.cmf logs/quality_q4tp.log
echo CARDQ3_DONE
