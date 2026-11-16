#!/bin/bash
# Flash-Next (qwen4_exp) on Vulkan: `forward_ids` now reads a prompt in the
# frames generation uses. A 4096-token bench with the 0.8.98 release
# (per-position forward_ids) and the new build, and the greedy text of a long
# prompt from both on Vulkan. After orn4.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/flash1 out/flash1 cmf/big
end=$((SECONDS + 14400))
until grep -qE "ORN4_DONE|ORN4_BUILD_FAILED" logs/orn4.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
f=cmf/big/qwen38-flash-next-fast-code.cmf
[ -s $f ] || timeout 3600 venv/bin/hf download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf --local-dir cmf/big > logs/flash1/dl.log 2>&1
[ -s $f ] || { echo FLASH1_DOWNLOAD_FAILED; exit 1; }
for B in bin/cortiq bin/cortiq-fast; do
  t=$(basename $B)
  flock gpu.lock env CMF_GPU=1 timeout 2400 $B bench $f --tokens 32 --ctx 4096 --json > logs/flash1/4k_$t.log 2>&1
  echo "$t vulkan 4k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s|prefill_measurement)": [^,]+' logs/flash1/4k_$t.log | tr '\n' ' ')"
  flock gpu.lock env CMF_GPU=1 timeout 1800 $B run $f --prompt-file prompts/long.txt -n 32 --greedy 2>/dev/null | grep -v '^\[' | grep -v '^Loading' > out/flash1/long_$t.txt
done
if cmp -s out/flash1/long_cortiq.txt out/flash1/long_cortiq-fast.txt; then echo "long prompt: new text == release"; else echo "long prompt: texts differ"; diff out/flash1/long_cortiq.txt out/flash1/long_cortiq-fast.txt | head -4 | cut -c1-160; fi
echo FLASH1_DONE
