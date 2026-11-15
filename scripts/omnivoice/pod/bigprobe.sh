#!/bin/bash
# The big models on the A40 (46 GB card, a 46.6 GiB container): decode (64
# tokens, two runs) and a 4096-token prompt on CUDA and Vulkan with the
# release binary (bin/cortiq, 0.8.98), the hybrid arm's declines from the
# logs, greedy text CUDA == Vulkan. One file on disk at a time, the next one
# fetched while the current one is measured; every run under gpu.lock.
#   bigprobe.sh [LIST]   lines "repo file" (default: the list below)
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
B=${B:-bin/cortiq}
mkdir -p cmf/big logs/big out/big
LIST=${1:-scripts/bigprobe.list}
fetch() { # repo file
  [ -s cmf/big/$2 ] && return 0
  timeout 7200 venv/bin/hf download $1 $2 --local-dir cmf/big > logs/big/${2%.cmf}_dl.log 2>&1
}
mapfile -t rows < <(grep -vE '^\s*(#|$)' "$LIST")
pf=""
for i in "${!rows[@]}"; do
  read -r repo file <<< "${rows[$i]}"
  [ -n "$pf" ] && wait $pf
  fetch $repo $file || { echo "$file: DOWNLOAD FAILED"; continue; }
  # prefetch the next file while this one is measured
  pf=""
  if [ $((i + 1)) -lt ${#rows[@]} ]; then
    read -r nrepo nfile <<< "${rows[$((i + 1))]}"
    fetch $nrepo $nfile & pf=$!
  fi
  f=cmf/big/$file; tag=${file%.cmf}
  echo "== $tag $(du -h $f | cut -f1)"
  for be in cuda 1; do
    for k in 1 2; do
      flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 2400 $B bench $f --tokens 64 --ignore-eos --json > logs/big/${tag}_d_${be}_$k.log 2>&1
      echo "$tag $be decode run$k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/big/${tag}_d_${be}_$k.log | tr '\n' ' ')"
    done
    flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 3600 $B bench $f --tokens 32 --ctx 4096 --json > logs/big/${tag}_4k_${be}.log 2>&1
    echo "$tag $be 4k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/big/${tag}_4k_${be}.log | tr '\n' ' ')"
    sed -E 's/\x1b\[[0-9;]*m//g' logs/big/${tag}_d_${be}_1.log logs/big/${tag}_4k_${be}.log | grep -iE 'declin|refus|fall.*back|per-op|not supported|out of memory|error' | sed -E 's/^.*(INFO|WARN|ERROR) //' | sort | uniq -c | sort -rn | head -5 | sed 's/^/      /'
    flock gpu.lock env CMF_GPU=$be timeout 2400 $B run $f --prompt "Explain in two sentences why the sky is blue." -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/big/${tag}_$be.txt
  done
  if cmp -s out/big/${tag}_cuda.txt out/big/${tag}_1.txt; then echo "$tag: CUDA text == Vulkan"; else echo "$tag: CUDA text DIFFERS from Vulkan"; fi
  rm -f $f
done
[ -n "$pf" ] && wait $pf
echo BIGPROBE_DONE
