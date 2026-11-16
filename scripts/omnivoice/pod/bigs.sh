#!/bin/bash
# A MoE past the card and the host's memory on the CUDA hybrid's expert
# stream: CPU greedy text (the reference), CUDA text with the declines and the
# stream's lines, decode (64 tokens, two runs) and a 512-token prompt with the
# stream's hit rate (CMF_CUDA_DEBUG).
#   bigs.sh TAG REPO FILE [WAIT_FOR_LOG WAIT_FOR_MARK]
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
tag=$1 repo=$2 file=$3
mkdir -p logs/$tag out/$tag cmf/big
if [ $# -ge 5 ]; then
  end=$((SECONDS + 14400))
  until grep -qE "$5" "$4" 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
fi
m=cmf/big/$file
[ -s $m ] || timeout 7200 venv/bin/hf download $repo $file --local-dir cmf/big > logs/$tag/dl.log 2>&1
[ -s $m ] || { echo "${tag}: DOWNLOAD FAILED"; tail -3 logs/$tag/dl.log; exit 1; }
echo "$tag $(du -h $m | cut -f1)"
bash scripts/build.sh fast > logs/$tag/build.log 2>&1
grep -q BUILD_FAST_DONE logs/$tag/build.log || { echo "${tag}: BUILD FAILED"; tail -30 logs/$tag/build.log; exit 1; }
B=bin/cortiq-fast
P="Explain in two sentences why the sky is blue."
[ -s out/$tag/cpu.txt ] || flock gpu.lock env CMF_GPU=0 timeout 3600 $B run $m --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > out/$tag/cpu.txt
echo "cpu text: $(tail -n +6 out/$tag/cpu.txt | tr '\n' ' ' | cut -c1-200)"
flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 2400 $B run $m --prompt "$P" -n 24 --greedy > out/$tag/cuda.txt 2> logs/$tag/run_cuda.log
echo "cuda text exit $?: $(grep -v '^\[' out/$tag/cuda.txt | tail -n +6 | tr '\n' ' ' | cut -c1-200)"
if cmp -s <(grep -v '^\[' out/$tag/cuda.txt) out/$tag/cpu.txt; then echo "CUDA text == CPU"; else echo "CUDA text DIFFERS from CPU"; fi
sed -E 's/\x1b\[[0-9;]*m//g' logs/$tag/run_cuda.log | grep -iE 'declin|refus|stream|hybrid:|error|panick' | sed -E 's/^.*(INFO|WARN|ERROR) //' | sort | uniq -c | sort -rn | head -10 | cut -c1-230 | sed 's/^/      /'
for k in 1 2; do
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 2400 $B bench $m --tokens 64 --ignore-eos --json > logs/$tag/d_$k.log 2>&1
  echo "decode run$k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/$tag/d_$k.log | tr '\n' ' ') | $(grep -hoE 'expert stream: .*' logs/$tag/d_$k.log | tail -1)"
done
flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 3600 $B bench $m --tokens 32 --ctx 512 --json > logs/$tag/p512.log 2>&1
echo "512: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/$tag/p512.log | tr '\n' ' ') | $(grep -hoE 'expert stream: .*' logs/$tag/p512.log | tail -1)"
echo "${tag}_DONE"
