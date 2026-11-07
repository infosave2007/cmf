#!/bin/bash
# sidecars.sh: the DFlash 2 drafter sidecars of the two low-bit files that go out (bound to each trunk hash).
cd /root/rip
for f in out/ripple-65b-q4kv8.cmf out/ripple-65b-q2hot5kv.cmf; do
  echo "== $f"
  CMF_GPU=0 ./cortiq convert --model dflash2 --output $f --dflash-sidecar --quant q4tp 2>&1 | grep -a "wrote\|rror\|sidecar" | cut -c1-200
done
ls -la out/*.dflash.cmf
echo SIDECARS_DONE
