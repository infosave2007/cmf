#!/bin/bash
# The big models again on the new build (bin/cortiq-fast): Ripple-65B q4tp
# (value residual and clamp on the prompt path too) and Ornith-122B q4tp
# (routed experts past the card in mapped host memory). Each file waits for
# disk room, runs decode x2, a 4096-token prompt and the greedy text against
# the probe's Vulkan text, and is deleted.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh fast > logs/big2_build.log 2>&1
grep -q BUILD_FAST_DONE logs/big2_build.log || { echo BIG2_BUILD_FAILED; tail -40 logs/big2_build.log; exit 1; }
B=bin/cortiq-fast
mkdir -p cmf/b2 logs/big2 out/big2
while read -r repo file gb; do
  [ -z "$repo" ] && continue
  end=$(( $(date +%s) + 14400 ))
  while [ $(( $(df -k --output=avail / | tail -1) / 1048576 )) -lt $(( gb + 25 )) ]; do [ "$(date +%s)" -gt $end ] && break; sleep 60; done
  f=cmf/b2/$file; tag=${file%.cmf}
  [ -s $f ] || timeout 7200 venv/bin/hf download $repo $file --local-dir cmf/b2 > logs/big2/${tag}_dl.log 2>&1 || { echo "$tag: DOWNLOAD FAILED"; continue; }
  echo "== $tag"
  for k in 1 2; do
    flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 2400 $B bench $f --tokens 64 --ignore-eos --json > logs/big2/${tag}_d_$k.log 2>&1
    echo "$tag decode run$k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/big2/${tag}_d_$k.log | tr '\n' ' ')"
  done
  sed -E 's/\x1b\[[0-9;]*m//g' logs/big2/${tag}_d_1.log | grep -iE 'declin|refus|mapped host|hybrid graphs|on the CPU|error' | sed -E 's/^.*(INFO|WARN|ERROR) //' | sort | uniq -c | head -8
  flock gpu.lock env CMF_GPU=cuda timeout 3600 $B bench $f --tokens 32 --ctx 4096 --json > logs/big2/${tag}_4k.log 2>&1
  echo "$tag 4k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/big2/${tag}_4k.log | tr '\n' ' ')"
  flock gpu.lock env CMF_GPU=cuda timeout 2400 $B run $f --prompt "Explain in two sentences why the sky is blue." -n 48 --greedy 2>/dev/null | grep -v '^\[' | grep -v '^Loading model' > out/big2/${tag}_cuda.txt
  if diff -q out/big2/${tag}_cuda.txt <(grep -v '^Loading model' out/big/${tag}_1.txt) > /dev/null; then echo "$tag: CUDA text == Vulkan"; else echo "$tag: CUDA text differs:"; sed -n '5,9p' out/big2/${tag}_cuda.txt | cut -c1-200; fi
  rm -f $f
done <<'LIST'
infosave/Ripple-65B-Preview-cmf ripple-65b-q4tp.cmf 35
infosave/Ornith-1.5-122B-cmf ornith-1.5-122b-q4tp.cmf 64
LIST
echo BIG2_DONE
