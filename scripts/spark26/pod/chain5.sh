#!/bin/bash
# chain5.sh: the GPU queue after the facts test and the 0.8.43 build: the 0.8.43 vs new prompt A/B
# (Vulkan, CMF_PREFILL_PROF splits), llama.cpp CUDA bench with peak VRAM, perplexity of the GGUF
# files under the shared window protocol (64 windows x 512 tokens at stride 511, raw tokens, no BOS),
# the token ids for the tokenizer parity check, the standard model's facts test at 60k chars, and
# llama.cpp Vulkan decode at 32k/64k.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp
busy() { pgrep -f "passkey_file.py|cargo build|target0843.*rustc" > /dev/null; }
while busy; do sleep 15; done
for b in /root/spark26/target0843/release/cortiq ./cortiq; do for n in 2000 16000; do echo "== $b $n"; CMF_PREFILL_PROF=1 $b bench hf/cmf/spark-x2.6-4b-q4tp.cmf --core --ignore-eos --tokens 32 --ctx $n 2>&1 | grep -E "Prompt|Decode|prefill-split" | tail -3; done; done > logs/ab0843.txt 2>&1
# peak VRAM (MiB) of a command, sampled every 0.5 s
vpeak() { "$@" & local pid=$! p=0; while kill -0 $pid 2>/dev/null; do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$v" -gt "$p" ] && p=$v; sleep 0.5; done; echo "peak_vram_mib=$p"; }
LB=llama.cpp/build-cuda/bin/llama-bench
for q in Q4_K_M Q8_0; do
  echo "== CUDA $q prompt"; vpeak $LB -m hf/gguf/Spark-X2.6-4B-$q.gguf -p 2000,16000 -n 0 -r 1 -o md 2>/dev/null | grep -E "pp|model|peak"
  echo "== CUDA $q decode"; vpeak $LB -m hf/gguf/Spark-X2.6-4B-$q.gguf -p 0 -n 64 -d 0,16000,32000,64000 -r 1 -o md 2>/dev/null | grep -E "tg|peak"
done > logs/llama_cuda_bench.txt 2>&1
until grep -q TOOLS_DONE logs/llama_cuda_tools.log; do sleep 15; done
for q in Q4_K_M Q8_0; do echo "== $q"; llama.cpp/build-cuda/bin/llama-perplexity -m hf/gguf/Spark-X2.6-4B-$q.gguf -f data/wiki.test.raw -c 512 -b 512 --ppl-stride 511 --chunks 64 --override-kv tokenizer.ggml.add_bos_token=bool:false 2>&1 | grep -E "chunks|estimate|tokeniz|Final|\[" | tail -6; done > logs/llama_cuda_ppl.txt 2>&1
llama.cpp/build-cuda/bin/llama-tokenize -m hf/gguf/Spark-X2.6-4B-Q4_K_M.gguf -f data/wiki.test.raw --ids --no-bos --log-disable > logs/llama_ids.txt 2> logs/llama_ids.err
venv/bin/python passkey_file.py ./cortiq hf/cmf/spark-x2.6-4b-q4tp.cmf data/wiki.test.raw 60000 8 > logs/pk_std_vk.log 2>&1
llama.cpp/build-vk/bin/llama-bench -m hf/gguf/Spark-X2.6-4B-Q4_K_M.gguf -p 0 -n 64 -d 32000,64000 -r 1 -o md > logs/llama_vk_deep.txt 2>&1
echo CHAIN5_DONE > logs/chain5.done
