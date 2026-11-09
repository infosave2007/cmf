#!/bin/bash
# chain6.sh: after chain5, the perplexity of the CMF files on Vulkan under the shared window protocol
# (64 x 512 at stride 511, raw tokens), with the cooperative-matrix GEMM on and off (its tf32-class
# accuracy against the f32 kernel), and the token ids for the tokenizer parity check.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp
until [ -f logs/chain5.done ]; do sleep 20; done
for f in q4tp q8_2f; do
  echo "== cortiq ppl $f"
  CMF_PPL_IDS_OUT=logs/cortiq_ids.txt ./cortiq ppl hf/cmf/spark-x2.6-4b-$f.cmf -f data/wiki.test.raw --windows 64 --window-len 512 --window-stride 511 2>&1 | grep -E "windows|PPL|rror"
done
echo "== cortiq ppl q4tp CMF_COOP=0"
CMF_COOP=0 ./cortiq ppl hf/cmf/spark-x2.6-4b-q4tp.cmf -f data/wiki.test.raw --windows 64 --window-len 512 --window-stride 511 2>&1 | grep -E "windows|PPL|rror"
# llama.cpp under the same windows: -c 257 because llama-perplexity adds ppl_stride/2 to the context
# when a stride is set (257 + 255 = 512-token windows, the last 511 positions scored).
for q in Q4_K_M Q8_0; do echo "== llama ppl $q"; llama.cpp/build-cuda/bin/llama-perplexity -m hf/gguf/Spark-X2.6-4B-$q.gguf -f data/wiki.test.raw -c 257 -b 512 --ppl-stride 511 --chunks 64 --override-kv tokenizer.ggml.add_bos_token=bool:false 2>&1 | grep -E "chunks|Final|\[64\]" | tail -3; done
echo CHAIN6_DONE > logs/chain6.done
