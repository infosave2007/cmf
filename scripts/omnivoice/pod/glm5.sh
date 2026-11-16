#!/bin/bash
# GLM-5.3 on CUDA with DSA's attention absorbed (kv_b through the query and
# the output, not over every selected latent): texts against the CPU, a
# 64-token bench, 512+64, and the last token's per-layer profile. After orn3.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glm5 out/glm5
end=$((SECONDS + 14400))
until grep -qE "ORN3_DONE|ORN3_DOWNLOAD_FAILED" logs/orn3.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/glm5/build.log 2>&1
grep -q BUILD_FAST_DONE logs/glm5/build.log || { echo GLM5_BUILD_FAILED; tail -30 logs/glm5/build.log; exit 1; }
B=bin/cortiq-fast; f=cmf/big/glm-5.3-flash-q2tp.cmf
P="Explain in two sentences why the sky is blue."
for be in 0 cuda; do
  flock gpu.lock env CMF_GPU=$be timeout 3600 $B run $f --prompt "What is 2+2? Answer briefly." --no-think -n 16 --greedy 2>/dev/null | grep -v '^\[' > out/glm5/two_$be.txt
  flock gpu.lock env CMF_GPU=$be timeout 3600 $B run $f --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > out/glm5/sky_$be.txt
  echo "$be: 2+2 $(tail -n +5 out/glm5/two_$be.txt | tr '\n' ' ' | cut -c1-60) | sky $(tail -n +5 out/glm5/sky_$be.txt | tr '\n' ' ' | cut -c1-200)"
done
cmp -s out/glm5/sky_0.txt out/glm2/sky_cpu.txt && echo "absorbed CPU sky == expanded CPU sky" || echo "absorbed CPU sky DIFFERS from expanded CPU sky"
flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 3600 $B bench $f --tokens 64 --ignore-eos --json > logs/glm5/d64.log 2>&1
echo "cuda 64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glm5/d64.log | tr '\n' ' ')"
flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 CMF_GLM_CPU_PROF=1 timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/glm5/p512.log 2>&1
echo "cuda 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glm5/p512.log | tr '\n' ' ')"
grep -hE "glm-prof layer=" logs/glm5/p512.log | tail -45 | python3 -c "
import sys,re
a={'kda':[0,0,0],'dsa':[0,0,0]}
for l in sys.stdin:
    m=re.search(r'mixer=(\w+) attn_ms=([0-9.]+) ffn_ms=([0-9.]+)',l)
    if m: k=m.group(1); a[k][0]+=1; a[k][1]+=float(m.group(2)); a[k][2]+=float(m.group(3))
for k,(n,at,ff) in a.items(): print(f'      {k}: {n} layers, mixer {at:.0f} ms, ffn {ff:.0f} ms')
"
grep -hE "cuda moe rows" logs/glm5/p512.log | tail -1 | cut -c1-200 | sed 's/^/      /'
echo GLM5_DONE
