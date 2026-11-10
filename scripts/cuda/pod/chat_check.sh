#!/bin/bash
# A three-turn chat in one process on both paths (the K/V reuse between turns): the texts must agree.
cd /root/cuda
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
turns="My name is Oleg and my favourite number is 47. Remember both.
What is my favourite number plus three? Answer in one short sentence.
And what is my name spelled backwards? Answer in one short sentence."
for q in q4tp q8_2f; do
  m=models/spark-x2.6-4b-$q-turbo.cmf
  printf "%s\n" "$turns" | CMF_O1=off ./bin/cortiq-vk run $m --greedy --no-think -n 64 2>out/chat_vk_$q.log | grep -v "^\[" > out/chat_vk_$q.txt
  printf "%s\n" "$turns" | CMF_GPU=cuda ${CUDA_BIN:-./bin/cortiq-cuda} run $m --greedy --no-think -n 64 2>out/chat_cu_$q.log | grep -v "^\[" > out/chat_cu_$q.txt
  if cmp -s out/chat_vk_$q.txt out/chat_cu_$q.txt; then echo "chat $q: SAME ($(wc -c < out/chat_cu_$q.txt) bytes)"; else echo "chat $q: DIFFERENT"; diff out/chat_vk_$q.txt out/chat_cu_$q.txt | head -20; fi
done
tail -12 out/chat_cu_q4tp.txt
grep -c -i "kv-reuse\|reuse" out/chat_cu_q4tp.log
