#!/bin/bash
# chain2.sh: standard vs turbo at depth on Vulkan (q4tp, then q8_2f), then the 0.8.43 release tree for the prefill A/B.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp
bash ctxbench.sh hf/cmf/spark-x2.6-4b-q4tp.cmf std_q4 1000 16000 32000 64000
bash ctxbench.sh out/spark-x2.6-4b-q4tp-turbo.cmf turbo_q4 1000 16000 32000 64000
bash ctxbench.sh hf/cmf/spark-x2.6-4b-q8_2f.cmf std_q8 1000 16000 64000
bash ctxbench.sh out/spark-x2.6-4b-q8_2f-turbo.cmf turbo_q8 1000 16000 64000
[ -d old0843 ] || git clone -q --depth 1 --branch v0.8.43 https://github.com/infosave2007/cmf old0843
( cd old0843 && . "$HOME/.cargo/env" && CARGO_TARGET_DIR=/root/spark26/target0843 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release -p cortiq-cli --features gpu > /root/spark26/logs/build0843.log 2>&1 )
for b in /root/spark26/target0843/release/cortiq ./cortiq; do echo "== $b"; $b bench hf/cmf/spark-x2.6-4b-q4tp.cmf --core --ignore-eos --tokens 32 --ctx 2000 2>&1 | grep -E "Prompt|Decode"; done > logs/ab0843.txt
echo CHAIN2_DONE > logs/chain2.done
