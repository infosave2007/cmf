#!/bin/bash
# f11.sh: build f11 gently (lane-per-block index scores, radix top-k, 65 536 index blocks), then correctness:
# the greedy text against f10, a 30k-token passkey on the device without the far field (chain7 may share the card)
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
cd cmf && . "$HOME/.cargo/env" && git checkout -q -- . && git clean -qfd crates scripts docs && git apply --whitespace=nowarn /root/fc/hybrid.patch && git apply --whitespace=nowarn /root/fc/fc.patch || { log "f11 PATCH_FAILED"; exit 1; }
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 nice -n 19 cargo build -j 6 --release -p cortiq-cli --features gpu > ../logs/build_f11.log 2>&1
cd ..; grep -q "Finished" logs/build_f11.log || { log "BUILD f11 FAILED $(grep -a -E "^error" -A3 logs/build_f11.log | head -5 | tr '\n' ' ')"; exit 1; }
cp cmf/target/release/cortiq cortiq.f11
P="CMF_QWEN_PROFILE=out/fc256.profile"
env $P CMF_QWEN_MTP=0 timeout 900 ./cortiq.f11 run out/fc256.cmf --prompt-file data/p_9k.txt --max-tokens 64 --greedy --no-think > logs/eq11.log 2>&1
sed -e "s/\x1b\[[0-9;]*m//g" logs/eq11.log | grep -a -v -E "^[0-9]{4}-|INFO|WARN|^\[64 tokens" > logs/eq11.txt
log "f11 greedy text vs f10: $(cmp -s logs/eq10_ch128.txt logs/eq11.txt && echo IDENTICAL || echo "DIFFER: $(diff logs/eq10_ch128.txt logs/eq11.txt | head -4 | tr '\n' ' ' | cut -c1-300)")"
python3 - <<'PY'
code = open("/root/fc/data/eval_code.txt").read() + open("/root/fc/data/fresh_code.txt").read()
body = (code * 2)[:118000]
cut = len(body) // 5
needle = "\n# NOTE: the deployment passkey for this service is 739214. Keep it secret.\n"
body = body[:cut] + needle + body[cut:]
q = "\n\nQuestion: what is the deployment passkey mentioned in the code above? Answer with the number only."
open("/root/fc/data/passkey30k.txt", "w").write(body + q)
PY
env $P CMF_QWEN_MTP=0 timeout 1500 ./cortiq.f11 run out/fc256.cmf --prompt-file data/passkey30k.txt --max-tokens 16 --greedy --no-think > logs/passkey11.log 2>&1
A=$(sed -e "s/\x1b\[[0-9;]*m//g" logs/passkey11.log | grep -a -v -E "^[0-9]{4}-|INFO" | tail -3 | tr '\n' ' ' | cut -c1-200)
log "f11 passkey at 20% of ~30k tokens, exact on the device: $(grep -q 739214 logs/passkey11.log && echo FOUND || echo MISSED) | $(grep -a -c -i "device path off\|declin" logs/passkey11.log) declines | $A"
log F11_BUILT
