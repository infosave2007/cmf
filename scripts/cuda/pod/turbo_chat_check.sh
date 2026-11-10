#!/bin/bash
# A chat that starts past the seal: the long document as the first turn, two
# short questions after it, in one process. The CUDA path (the far field on
# the card) and the wgpu path; the answers and each turn's time.
#   turbo_chat_check.sh [binary]
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
bin=${1:-bin/cortiq-cuda}
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0 CMF_MAX_SEQ=131072
m=models/spark-x2.6-4b-q4tp-turbo.cmf
{
  tr '\n' ' ' < prompts_long/p6_long_16k.txt; echo
  echo "In one sentence: what does the CUDA backend use to compile its kernels?"
  echo "And which environment variable selects that backend?"
} > out/turbo_chat_turns.txt
for path in cu vk; do
  env=(); [ $path = cu ] && env=(CMF_GPU=cuda)
  t0=$(date +%s.%N)
  env "${env[@]}" "$bin" run $m --greedy --no-think -n 48 < out/turbo_chat_turns.txt 2> out/turbo_chat_$path.log > out/turbo_chat_$path.txt
  echo "== $path: rc=$? $(printf '%.1f' "$(echo "$(date +%s.%N) - $t0" | bc)") s"
  grep -E "first token|tokens ·" out/turbo_chat_$path.txt | cut -c1-160
  grep -v "^\[" out/turbo_chat_$path.txt | grep -v "^$" | cut -c1-300 | tail -8
  sed 's/\x1b\[[0-9;]*m//g' out/turbo_chat_$path.log | grep -E "kv-reuse|far field|declin|refus|error|ERROR|WARN" | cut -c1-200 | sort | uniq -c | sort -rn | head -8
done
echo TURBO_CHAT_DONE
