#!/bin/bash
# f10.sh: build f10 gently (split-K pair GEMM, arena-slot-sorted experts for prompt frames), check the greedy
# text against 8-token frames (correctness only: chain7's server shares the card meanwhile)
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
cd cmf && . "$HOME/.cargo/env" && git checkout -q -- . && git clean -qfd crates scripts docs && git apply --whitespace=nowarn /root/fc/hybrid.patch && git apply --whitespace=nowarn /root/fc/fc.patch || { log "f10 PATCH_FAILED"; exit 1; }
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 nice -n 19 cargo build -j 6 --release -p cortiq-cli --features gpu > ../logs/build_f10.log 2>&1
cd ..; grep -q "Finished" logs/build_f10.log || { log "BUILD f10 FAILED $(grep -a -E "^error" -A3 logs/build_f10.log | head -5 | tr '\n' ' ')"; exit 1; }
cp cmf/target/release/cortiq cortiq.f10
P="CMF_QWEN_PROFILE=out/fc256.profile"
for CH in 8 128; do
  env $P CMF_QWEN_PREFILL_CHUNK=$CH CMF_QWEN_MTP=0 timeout 900 ./cortiq.f10 run out/fc256.cmf --prompt-file data/p_9k.txt --max-tokens 64 --greedy --no-think > logs/eq10_ch$CH.log 2>&1
  sed -e "s/\x1b\[[0-9;]*m//g" logs/eq10_ch$CH.log | grep -a -v -E "^[0-9]{4}-|INFO|WARN|^\[64 tokens" > logs/eq10_ch$CH.txt
done
log "f10 greedy text, 8- vs 128-token frames (sorted experts): $(cmp -s logs/eq10_ch8.txt logs/eq10_ch128.txt && echo IDENTICAL || echo "DIFFER: $(diff logs/eq10_ch8.txt logs/eq10_ch128.txt | head -4 | tr '\n' ' ' | cut -c1-300)") | $(grep -a -c -i panic logs/eq10_ch128.log) panics"
log F10_BUILT
