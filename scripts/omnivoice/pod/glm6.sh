#!/bin/bash
# GLM-5.3 on CUDA: the slots leave room for the host-run trunk's matvecs on
# the card (default) or take it all (CMF_MOE_ROW_TRUNK=0): 512+64 each, the
# last token's per-layer profile, the slots' hit line, and the CUDA text.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glm6 out/glm6
end=$((SECONDS + 14400))
until grep -qE "ORN5_DONE|ORN5_BUILD_FAILED" logs/orn5.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/glm6/build.log 2>&1
grep -q BUILD_FAST_DONE logs/glm6/build.log || { echo GLM6_BUILD_FAILED; tail -30 logs/glm6/build.log; exit 1; }
B=bin/cortiq-fast; f=cmf/big/glm-5.3-flash-q2tp.cmf
for v in trunk all; do
  x=""; [ $v = all ] && x="CMF_MOE_ROW_TRUNK=0"
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 CMF_GLM_CPU_PROF=1 RUST_LOG=info $x timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/glm6/$v.log 2>&1
  echo "$v 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glm6/$v.log | tr '\n' ' ')"
  grep -hoE "cuda moe rows: [0-9]+ card slots.*" logs/glm6/$v.log | head -1 | cut -c1-160 | sed 's/^/      /'
  grep -hE "glm-prof layer=" logs/glm6/$v.log | tail -45 | python3 -c "
import sys,re
a={'kda':[0,0,0],'dsa':[0,0,0]}
for l in sys.stdin:
    m=re.search(r'mixer=(\w+) attn_ms=([0-9.]+) ffn_ms=([0-9.]+)',l)
    if m: k=m.group(1); a[k][0]+=1; a[k][1]+=float(m.group(2)); a[k][2]+=float(m.group(3))
for k,(n,at,ff) in a.items(): print(f'      {k}: {n} layers, mixer {at:.0f} ms, ffn {ff:.0f} ms')
"
  grep -hE "cuda moe rows: [0-9]+ slots" logs/glm6/$v.log | tail -1 | cut -c1-200 | sed 's/^/      /'
done
flock gpu.lock env CMF_GPU=cuda timeout 3600 $B run $f --prompt "What is 2+2? Answer briefly." --no-think -n 16 --greedy 2>/dev/null | grep -v '^\[' > out/glm6/two.txt
echo "cuda 2+2: $(tail -n +5 out/glm6/two.txt | tr '\n' ' ' | cut -c1-60)"
echo GLM6_DONE
