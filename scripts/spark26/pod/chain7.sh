#!/bin/bash
# chain7.sh: after chain6, llama.cpp's Vulkan numbers again on a quiet GPU (the first tg run at 16k
# gave 9.7 tok/s while a deeper run gave 44 at 32k): prompt 2000/16000 and decode at 0/16k/32k/64k
# for Q4_K_M and Q8_0, with the peak VRAM of each run.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp
until [ -f logs/chain6.done ]; do sleep 20; done
vpeak() { "$@" & local pid=$! p=0; while kill -0 $pid 2>/dev/null; do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$v" -gt "$p" ] && p=$v; sleep 0.5; done; echo "peak_vram_mib=$p"; }
LB=llama.cpp/build-vk/bin/llama-bench
for q in Q4_K_M Q8_0; do
  echo "== Vulkan $q prompt"; vpeak $LB -m hf/gguf/Spark-X2.6-4B-$q.gguf -p 2000,16000 -n 0 -r 1 -o md 2>/dev/null | grep -E "pp|peak"
  echo "== Vulkan $q decode"; vpeak $LB -m hf/gguf/Spark-X2.6-4B-$q.gguf -p 0 -n 64 -d 0,16000,32000,64000 -r 1 -o md 2>/dev/null | grep -E "tg|peak"
done > logs/llama_vk_bench2.txt 2>&1
echo CHAIN7_DONE > logs/chain7.done
