#!/bin/bash
# Where GLM-5.3's token goes on CUDA after a long prompt: CMF_GLM_CPU_PROF=1
# per-layer mixer / FFN milliseconds at the last tokens of a 512-token prompt
# plus 8 decoded, with the dense matvecs on the card (CMF_GPU=cuda) and with
# the trunk held on the CPU (CMF_GLM_GPU_MOE_ONLY=1 keeps only the experts
# on the card).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glm4
B=bin/cortiq-fast; f=cmf/big/glm-5.3-flash-q2tp.cmf
for v in plain moeonly; do
  x=""; [ $v = moeonly ] && x="CMF_GLM_GPU_MOE_ONLY=1"
  flock gpu.lock env CMF_GPU=cuda CMF_GLM_CPU_PROF=1 $x timeout 3600 $B bench $f --tokens 8 --ctx 512 --ignore-eos --json > logs/glm4/$v.log 2>&1
  echo "$v 512+8: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glm4/$v.log | tr '\n' ' ')"
  # the last 45 layer lines (the last token)
  grep -hE "glm-prof layer=" logs/glm4/$v.log | tail -45 | python3 -c "
import sys,re
a={'kda':[0,0,0],'dsa':[0,0,0]}
for l in sys.stdin:
    m=re.search(r'mixer=(\w+) attn_ms=([0-9.]+) ffn_ms=([0-9.]+)',l)
    if m: k=m.group(1); a[k][0]+=1; a[k][1]+=float(m.group(2)); a[k][2]+=float(m.group(3))
for k,(n,at,ff) in a.items(): print(f'      {k}: {n} layers, mixer {at:.0f} ms, ffn {ff:.0f} ms')
"
done
echo GLM4_DONE
