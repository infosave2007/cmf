#!/bin/bash
# LFM3-Audio-2B's long-prompt greedy text: the 0.8.98 release on Vulkan
# (per-position prefill), the new build on Vulkan (batch short conv) and the
# CPU, 32 tokens, and where they part. After cudareg.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/lfm7 out/lfm7
end=$((SECONDS + 14400))
until grep -qE "CUDAREG_DONE" logs/cudareg.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
f=cmf/dl/lfm3-audio-2b-q4tp.cmf
[ -s $f ] || timeout 3600 venv/bin/hf download infosave/LFM3-Audio-2B-cmf lfm3-audio-2b-q4tp.cmf --local-dir cmf/dl > logs/lfm7/dl.log 2>&1
run() { flock gpu.lock env CMF_GPU=$2 timeout 1800 $1 run $f --prompt-file prompts/long.txt -n 32 --greedy 2>/dev/null | grep -v '^\[' | grep -v '^Loading' | tail -n +4 | tr '\n' ' '; }
r=$(run bin/cortiq 1); n=$(run bin/cortiq-fast 1); cpu=$(run bin/cortiq-fast 0)
python3 - "$r" "$n" "$cpu" <<'PY'
import sys
r, n, c = sys.argv[1:4]
def part(a, b):
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    return "same" if a == b else f"part at {i}: …{a[max(0,i-40):i+40]!r} | …{b[max(0,i-40):i+40]!r}"
print("release vulkan vs cpu:", part(r, c))
print("new vulkan vs cpu:    ", part(n, c))
print("new vs release vulkan:", part(n, r))
PY
rm -f $f
echo LFM7_DONE
