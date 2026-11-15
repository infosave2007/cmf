#!/bin/bash
# The CUDA batch chunk on the Gemma-4 / Ripple prompt path (fast build):
# Ripple-65B 4k by default (now 1024 rows), humanizer-12B (Gemma-4) 4k at
# 128 rows and by default, and their greedy texts (128 vs default).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh fast > logs/q8_build.log 2>&1
grep -q BUILD_FAST_DONE logs/q8_build.log || { echo Q8_BUILD_FAILED; tail -30 logs/q8_build.log; exit 1; }
B=bin/cortiq-fast
mkdir -p logs/q8 out/q8
b4k() { flock gpu.lock env CMF_GPU=cuda "${@:3}" timeout 1800 $B bench $1 --tokens 32 --ctx 4096 --json > logs/q8/$2.log 2>&1; echo "$2: exit $? | $(grep -hoE '"(prefill_tok_s|ttft_s|decode_tok_s_steady)": [0-9.e+-]+' logs/q8/$2.log | tr '\n' ' ')"; }
txt() { flock gpu.lock env CMF_GPU=cuda "${@:3}" timeout 1800 $B run $1 --prompt-file prompts/long.txt -n 48 --greedy 2>/dev/null | grep -v '^\[' | grep -v '^Loading' > out/q8/$2.txt; }
python3 - <<'PY'
import os
os.makedirs("prompts", exist_ok=True)
para = ("The printing press changed how ideas moved. Scribes had copied books by hand for centuries, and a single volume "
        "could take months. Movable type made pages cheap, and within decades books spread across Europe. ")
open("prompts/long.txt", "w").write(para * 40 + "\nSummarize the passage above in one sentence.")
PY
b4k cmf/big/ripple-65b-q4tp.cmf rip_default CMF_X=1
f=cmf/big/humanizer-12b-q4tp-ffn.cmf
[ -s $f ] || timeout 3600 venv/bin/hf download infosave/humanizer-cmf humanizer-12b-q4tp-ffn.cmf --local-dir cmf/big > logs/q8/hum_dl.log 2>&1
b4k $f hum_128 CMF_BATCH_K=128
b4k $f hum_default CMF_X=1
txt $f hum_128 CMF_BATCH_K=128
txt $f hum_default CMF_X=1
if cmp -s out/q8/hum_128.txt out/q8/hum_default.txt; then echo "humanizer: text at 128 == default"; else echo "humanizer: text differs"; diff out/q8/hum_128.txt out/q8/hum_default.txt | head -6; fi
grep -v '^$' out/q8/hum_default.txt | tail -3 | cut -c1-200
echo Q8_DONE
