#!/bin/bash
# LFM3's short convolutions on wgpu's batch graph (sconv_batch): a 4096-token
# prompt on Vulkan with the 0.8.98 release (per-op prefill) and the new build,
# their declines, and the new build's greedy text on Vulkan against the CPU's
# over a long prompt. lfm5: after the MiMo round, its own build (the batch
# builder hands the short convs and sandwich norms to wgpu too).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/lfm4 out/lfm4
end=$((SECONDS + 14400))
until grep -qE "mimo_DONE|BUILD FAILED" logs/mimo.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/lfm4/build.log 2>&1
grep -q BUILD_FAST_DONE logs/lfm4/build.log || { echo LFM5_BUILD_FAILED; tail -30 logs/lfm4/build.log; exit 1; }
f=cmf/dl/lfm3-1.7b-q4tp.cmf
[ -s $f ] || timeout 3600 venv/bin/hf download infosave/LFM3-1.7B-cmf lfm3-1.7b-q4tp.cmf --local-dir cmf/dl > logs/lfm4/dl.log 2>&1
for B in bin/cortiq-fast; do
  t=$(basename $B)
  flock gpu.lock env CMF_GPU=1 timeout 1800 $B bench $f --tokens 32 --ctx 4096 --json > logs/lfm4/4k_$t.log 2>&1
  echo "$t vulkan 4k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/lfm4/4k_$t.log | tr '\n' ' ')"
  sed -E 's/\x1b\[[0-9;]*m//g' logs/lfm4/4k_$t.log | grep -iE 'declin|refus|batch graph' | sed -E 's/^.*(INFO|WARN|ERROR) //' | sort | uniq -c | sort -rn | head -5 | cut -c1-200 | sed 's/^/      /'
done
B=bin/cortiq-fast
for be in 1 0; do
  flock gpu.lock env CMF_GPU=$be timeout 1800 $B run $f --prompt-file prompts/long.txt -n 48 --greedy 2>/dev/null | grep -v '^\[' | grep -v '^Loading' > out/lfm4/long_$be.txt
done
if cmp -s out/lfm4/long_1.txt out/lfm4/long_0.txt; then echo "LFM3 long prompt: Vulkan text == CPU"; else echo "LFM3 long prompt: Vulkan text DIFFERS from CPU"; diff out/lfm4/long_1.txt out/lfm4/long_0.txt | head -6 | cut -c1-200; fi
tail -3 out/lfm4/long_1.txt | cut -c1-200
echo LFM5_DONE
