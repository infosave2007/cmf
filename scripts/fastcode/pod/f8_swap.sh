#!/bin/bash
# f8_swap.sh: once Strata's speed test is done (its HumanEval only measures accuracy), build f8 gently (4 jobs, nice 19)
# and swap it in under chain6's binary name for the fc256 / imported-file server tests.
cd /root/fc; J=/root/fc/JOURNAL.md
for i in $(seq 1 720); do [ -s res/strata_speed.json ] && break; sleep 10; done
cd cmf && . "$HOME/.cargo/env" && git checkout -q -- . && git apply --whitespace=nowarn /root/fc/fc.patch || { echo "$(date +%T) f8 PATCH_FAILED" >> $J; exit 1; }
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 nice -n 19 cargo build -j 4 --release -p cortiq-cli --features gpu > ../logs/build_f8.log 2>&1
cd ..; if grep -q "Finished" logs/build_f8.log; then cp cmf/target/release/cortiq cortiq.f8 && cp cortiq.f8 cortiq.f7.tmp && mv -f cortiq.f7.tmp cortiq.f7 && echo "$(date +%T) f8 built (sealed QSA layers skip the index-table cap) and swapped in as cortiq.f7" >> $J; else echo "$(date +%T) BUILD f8 FAILED" >> $J; fi
