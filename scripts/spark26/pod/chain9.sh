#!/bin/bash
# chain9.sh: the comparable perplexity protocol on the pod — after the rsync and the rebuild
# (`--window-score-from`), cortiq q4tp / q8_2f on Vulkan over 64 chunks of 512 tokens scoring
# the second half of each (stride 512, from position 256), and llama-perplexity's default walk
# (`-c 512 --chunks 64`: the same chunks, the same second halves) for Q4_K_M and Q8_0 on CUDA.
# Waits for the recall chain (chain8) before touching the GPU.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp
bash build.sh > logs/build9.out 2>&1
grep -q "status=0" logs/build9.out || { echo "build failed"; tail -20 logs/build.log; echo CHAIN9_DONE > logs/chain9.done; exit 1; }
until [ -f logs/chain8.done ]; do sleep 20; done
{
for q in q4tp q8_2f; do
  echo "== cortiq ppl $q (vulkan, chunks of 512, second half)"
  CMF_GPU=1 ./cortiq ppl hf/cmf/spark-x2.6-4b-$q.cmf data/wiki.test.raw --windows 64 --window-len 512 --window-stride 512 --window-score-from 256 2>&1 | grep -E "windows|PPL"
done
for q in Q4_K_M Q8_0; do
  echo "== llama ppl $q (cuda, -c 512 --chunks 64)"
  llama.cpp/build/bin/llama-perplexity -m hf/gguf/Spark-X2.6-4B-$q.gguf -f data/wiki.test.raw -c 512 --chunks 64 -ngl 99 2>&1 | grep -E "estimate|chunks|\[64\]" | tail -3
done
} > logs/chain9.out 2>&1
echo CHAIN9_DONE > logs/chain9.done
