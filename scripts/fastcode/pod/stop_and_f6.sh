#!/bin/bash
# stop_and_f6.sh: stop chain3c and its fc256 server test (the f5 arena left no room for KV past 16k), build f6,
# run the split-attention device test on this card
cd /root/fc; J=/root/fc/JOURNAL.md
log() { echo "$(date +%T) $*" | tee -a $J; }
for p in $(pgrep -f "chain3c[.]sh") $(pgrep -f "serve_tests[.]sh") $(pgrep -f "speedtest[.]py"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "cortiq[.]f5 serve"); do kill -- -$p 2>/dev/null; kill $p 2>/dev/null; done
sleep 5
log "STOP: fc256 server test on f5 fell to the host path at 16384 tokens (arena took the whole budget, no room for K/V); restarting on f6"
./build.sh f6 > logs/build_f6.log 2>&1
grep -q BUILD_DONE logs/build_f6.log || { log "BUILD f6 FAILED"; exit 1; }
. "$HOME/.cargo/env"; cd cmf
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 timeout 1200 cargo test --release -p cortiq-engine --features gpu --lib split_attend_matches -- --nocapture > ../logs/test_split.log 2>&1
cd ..; log "f6 built; split attention device test: $(grep -a -E "test result|max \|diff|panicked" logs/test_split.log | tr '\n' ' ' | cut -c1-300)"
