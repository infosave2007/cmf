#!/bin/bash
# Spark-X2.5-4B on the CUDA hybrid arm with the erf GELU pair (after the rotation fix)
# (its sliding layers rotate 256 dims, its full layers 64): declines, decode,
# the 4k prompt, and a long-prompt answer against the CPU.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
end=$(( $(date +%s) + 14400 ))
until grep -q -E "GD2_DONE" logs/gd2.out 2>/dev/null; do
  [ "$(date +%s)" -gt $end ] && { echo GD2_WAIT_TIMEOUT; exit 1; }
  sleep 30
done
bash scripts/build.sh release > logs/spark_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -30 logs/spark_build.log; exit 1; }
echo "build ok"
mkdir -p cmf/dl logs/sp2 out/sp2
keys='"(decode_tok_s_steady|prefill_tok_s|ttft_s)": *[0-9.]*'
long="$(head -c 2400 srcrs/docs/ppl_nat.txt) Summarize the text above in three sentences."
for file in Spark-X2.5-4B-q4mix.cmf Spark-X2.5-4B-q8_2f.cmf; do
  tag=${file%.cmf}
  f=cmf/dl/$file
  [ -s $f ] || timeout 1800 venv/bin/hf download infosave/Spark-X2.5-cmf $file --local-dir cmf/dl > logs/sp2/${tag}_dl.log 2>&1 || { echo "$tag: DOWNLOAD FAILED"; continue; }
  for i in 1 2; do
    flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 900 bin/cortiq bench $f --tokens 64 --ignore-eos --json > logs/sp2/${tag}_d_$i.log 2>&1
    echo "$tag cuda decode run$i: $(grep -oE "$keys" logs/sp2/${tag}_d_$i.log | tr '\n' ' ')"
  done
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 900 bin/cortiq bench $f --tokens 32 --ctx 4096 --json > logs/sp2/${tag}_4k.log 2>&1
  echo "$tag cuda 4k: $(grep -oE "$keys" logs/sp2/${tag}_4k.log | tr '\n' ' ')"
  grep -ihE "declin|refused|→ cpu" logs/sp2/${tag}_d_1.log logs/sp2/${tag}_4k.log | sed -E 's/^.*Z[^ ]* +//' | sort | uniq -c | sort -rn | head -4
  for be in cuda 1 0; do
    flock gpu.lock env CMF_GPU=$be timeout 1800 bin/cortiq run $f --prompt "$long" -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/sp2/${tag}_$be.txt
  done
  for o in 1 0; do
    cmp -s out/sp2/${tag}_cuda.txt out/sp2/${tag}_$o.txt && echo "$tag: long prompt CUDA text == $o" || echo "$tag: long prompt CUDA text differs from $o"
  done
  cmp -s out/sp2/${tag}_1.txt out/sp2/${tag}_0.txt && echo "$tag: Vulkan == CPU" || echo "$tag: Vulkan differs from CPU"
  rm -f $f
done
echo SPARK2_DONE
