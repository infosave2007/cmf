#!/bin/bash
# After the Vulkan VSA round: Hy-Memory-30B-A3B, Spark-X2.5 4B (q4mix, q8_2f)
# and MiniCPM-V-4.7-35B-A3B q4tp on CUDA and Vulkan — decode (64 tokens, two
# runs), the 4k prompt, CUDA's decline reasons, and a greedy answer CUDA
# against Vulkan. Each file deleted after its turn.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
end=$(( $(date +%s) + 14400 ))
until grep -q -E 'FHVK_DONE|_FAILED|TIMEOUT' logs/fhvk_fix.out 2>/dev/null; do
  [ "$(date +%s)" -gt $end ] && { echo VK_WAIT_TIMEOUT; exit 1; }
  sleep 30
done
mkdir -p cmf/dl logs/p3 out/p3
keys='"(decode_tok_s_steady|prefill_tok_s|ttft_s)": *[0-9.]*'
while read -r repo file; do
  [ -z "$repo" ] && continue
  tag=${file%.cmf}
  f=cmf/dl/$file
  [ -s $f ] || timeout 3600 venv/bin/hf download $repo $file --local-dir cmf/dl > logs/p3/${tag}_dl.log 2>&1 || { echo "$tag: DOWNLOAD FAILED"; continue; }
  for be in cuda 1; do
    for i in 1 2; do
      flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 900 bin/cortiq bench $f --tokens 64 --ignore-eos --json > logs/p3/${tag}_d_${be}_$i.log 2>&1
      echo "$tag $be decode run$i: $(grep -oE "$keys" logs/p3/${tag}_d_${be}_$i.log | tr '\n' ' ')"
    done
    flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 900 bin/cortiq bench $f --tokens 32 --ctx 4096 --json > logs/p3/${tag}_4k_$be.log 2>&1
    echo "$tag $be 4k: exit $?; $(grep -oE "$keys" logs/p3/${tag}_4k_$be.log | tr '\n' ' ')"
    grep -ihE "declin|refused|failed|→ cpu" logs/p3/${tag}_d_${be}_1.log logs/p3/${tag}_4k_$be.log | sed -E 's/.*(INFO|WARN|ERROR)[^ ]* //' | sort | uniq -c | sort -rn | head -5
    flock gpu.lock env CMF_GPU=$be timeout 900 bin/cortiq run $f --prompt "Explain in two sentences why the sky is blue." -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/p3/${tag}_$be.txt
  done
  cmp -s out/p3/${tag}_cuda.txt out/p3/${tag}_1.txt && echo "$tag: CUDA text == Vulkan" || { echo "$tag: CUDA text differs from Vulkan"; tail -2 out/p3/${tag}_cuda.txt; tail -2 out/p3/${tag}_1.txt; }
  rm -f $f
done <<'LIST'
infosave/Spark-X2.5-cmf Spark-X2.5-4B-q4mix.cmf
infosave/Spark-X2.5-cmf Spark-X2.5-4B-q8_2f.cmf
infosave/Hy-Memory-30B-A3B-cmf hy-memory-30b-a3b-q4tp.cmf
infosave/MiniCPM-V-4.7-35B-A3B-cmf MiniCPM-V-4.7-35B-A3B-q4tp.cmf
LIST
rm -rf cmf/dl/.cache
echo P3_DONE
