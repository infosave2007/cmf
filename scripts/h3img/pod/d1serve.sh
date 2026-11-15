#!/bin/bash
# d1-2B warm: `serve` and four /v1/system_one calls in a row a backend (the
# first pays the weights' upload), CUDA (bin/cortiq-fast) / Vulkan / CPU.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p cmf/dl logs/d1s
f=cmf/dl/d1-2b-q4tp.cmf
[ -s $f ] || timeout 3600 venv/bin/hf download infosave/d1-2B-cmf d1-2b-q4tp.cmf --local-dir cmf/dl > logs/d1s/dl.log 2>&1
body='{"state": "I was charged twice and my parcel never arrived. Please refund one charge.", "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}, "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Delivery and parcels", "technical": "App or site faults"}}}, "media": []}'
for be in cuda 1 0; do
  flock gpu.lock bash -c "
    CMF_GPU=$be CMF_CUDA_PROF=1 timeout 600 bin/cortiq-fast serve $f --port 8091 > logs/d1s/serve_$be.log 2>&1 &
    sp=\$!
    for i in \$(seq 1 120); do curl -s -o /dev/null http://127.0.0.1:8091/v1/models && break; sleep 1; done
    for i in 1 2 3 4; do
      curl -s -o logs/d1s/ans_${be}_\$i.json -w '$be call '\$i': %{time_total} s\n' -H 'Content-Type: application/json' -d '$body' http://127.0.0.1:8091/v1/system_one
    done
    kill \$sp; wait \$sp 2>/dev/null
  "
  head -c 300 logs/d1s/ans_${be}_4.json; echo
done
rm -f $f
echo D1S_DONE
