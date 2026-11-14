#!/bin/bash
# After probe3: why Kimi-Linear-48B decodes at 9 tok/s over Vulkan — the
# declines and the stage split of one decode bench.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
end=$(( $(date +%s) + 30000 ))
until grep -q -E 'P3_DONE|_TIMEOUT' logs/probe3.out 2>/dev/null; do
  [ "$(date +%s)" -gt $end ] && { echo P3_WAIT_TIMEOUT; exit 1; }
  sleep 30
done
f=cmf/dl/kimi48-code.cmf
keys='"(decode_tok_s_steady|prefill_tok_s|ttft_s)": *[0-9.]*'
flock gpu.lock env CMF_GPU=1 CMF_GPU_DEBUG=1 CMF_BATCH_DEBUG=1 RUST_LOG=info timeout 900 bin/cortiq bench $f --tokens 32 --ignore-eos --json > logs/kimi_vk_diag.log 2>&1
echo "kimi vulkan decode: exit $?; $(grep -oE "$keys" logs/kimi_vk_diag.log | tr '\n' ' ')"
grep -ihE "declin|refused|unsupported|→ cpu|fallback|not mine|batch graph" logs/kimi_vk_diag.log | sed -E 's/^.*Z[^ ]* +//' | sort | uniq -c | sort -rn | head -20
echo KVD_DONE
