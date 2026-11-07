#!/bin/bash
# Vulkan speed + accuracy of one file: vbench.sh FILE TAG [ENV=V ...]
# decode bench x3, prompt 2k and 8k (TTFT, decode at depth, peak VRAM), wikitext-2 ppl 12x512.
cd /root/cherrypit
F=$1; T=$2; shift 2
B=${BIN:-/root/cherrypit/cortiq}
mkdir -p logs/vb
S=logs/vb/$T.summary; : > $S
peak() { local p=0; while kill -0 $1 2>/dev/null; do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$v" -gt "$p" ] && p=$v; sleep 0.5; done; echo $p; }
one() { local label=$1; shift
  env "$@" > logs/vb/$T.$label.log 2>&1 & local pid=$!
  local p=$(peak $pid); wait $pid
  echo "== $label peak_vram_mib=$p $(grep -E "Prompt|Decode" logs/vb/$T.$label.log | tr "\n" " " | tr -s " ")" | tee -a $S
}
for i in 1 2 3; do one dec$i "$@" $B bench $F --core --ignore-eos --tokens 128; done
one ctx2k "$@" $B bench $F --core --ignore-eos --tokens 64 --ctx 2048
one ctx8k "$@" $B bench $F --core --ignore-eos --tokens 64 --ctx 8192
echo "== ppl $(env "$@" $B ppl $F --file /root/rip/data/wiki.test.raw --windows 12 --window-len 512 2>/dev/null | grep PPL)" | tee -a $S
echo VBENCH_DONE >> $S
