#!/bin/bash
# The LFM3 batch-graph short convolution on the Audio and Omni towers'
# text stacks: a 4096-token prompt on Vulkan with the 0.8.98 release binary
# and the new build, and the new build's greedy text on Vulkan against the
# CPU over a long prompt. After bigs3.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/lfm6 out/lfm6 cmf/dl
end=$((SECONDS + 14400))
until grep -qE "BIGS3_DONE|BIGS3_BUILD_FAILED" logs/bigs3.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
for rf in "infosave/LFM3-Audio-2B-cmf lfm3-audio-2b-q4tp.cmf" "infosave/LFM3-Omni-2B-cmf lfm3-omni-2b-q4tp.cmf"; do
  read -r repo file <<< "$rf"
  f=cmf/dl/$file; tag=${file%.cmf}
  [ -s $f ] || timeout 3600 venv/bin/hf download $repo $file --local-dir cmf/dl > logs/lfm6/${tag}_dl.log 2>&1
  [ -s $f ] || { echo "$tag: DOWNLOAD FAILED"; continue; }
  for B in bin/cortiq bin/cortiq-fast; do
    t=$(basename $B)
    flock gpu.lock env CMF_GPU=1 timeout 1800 $B bench $f --tokens 32 --ctx 4096 --json > logs/lfm6/${tag}_4k_$t.log 2>&1
    echo "$tag $t vulkan 4k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/lfm6/${tag}_4k_$t.log | tr '\n' ' ')"
  done
  for be in 1 0; do
    flock gpu.lock env CMF_GPU=$be timeout 1800 bin/cortiq-fast run $f --prompt-file prompts/long.txt -n 32 --greedy 2>/dev/null | grep -v '^\[' | grep -v '^Loading' > out/lfm6/${tag}_$be.txt
  done
  if cmp -s out/lfm6/${tag}_1.txt out/lfm6/${tag}_0.txt; then echo "$tag: Vulkan text == CPU"; else echo "$tag: Vulkan text DIFFERS from CPU"; diff out/lfm6/${tag}_1.txt out/lfm6/${tag}_0.txt | head -4 | cut -c1-160; fi
  rm -f $f
done
echo LFM6_DONE
