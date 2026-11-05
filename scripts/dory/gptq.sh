#!/bin/bash
# q4tp-v2: GPTQ of every projection against wiki, the model's own answers and code.
# The Hessian hook needs a quantized calibration model, f16 is the full-precision source.
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
B=./bin_gptq
cp src/target/release/cortiq $B
python3 - <<'PY'
import json
S=json.load(open("router/samples.json"))
open("data_chat.txt","w").write("\n".join(s["text"] for s in S))
PY
find src/crates -name '*.rs' | head -60 | xargs cat > data_code.txt
t0=$(date +%s)
timeout 7000 $B quantize-gptq out/dory-f16.cmf --codec q4tp --calib-model out/dory-q8_2f.cmf \
  --calib wiki_train.txt=0.6 --calib data_chat.txt=0.25 --calib data_code.txt=0.15 \
  --tokens ${1:-32768} --window 2048 --lambda 0.01 --threads 5 \
  --output out/dory-q4tp-v2.cmf > logs/gptq.log 2>&1
echo "gptq rc $? in $(( $(date +%s) - t0 )) s" >> logs/gptq.log
ls -la out/dory-q4tp-v2.cmf >> logs/gptq.log
timeout 300 ./src/target/release/cortiq ppl out/dory-q4tp-v2.cmf -f wiki_test.txt --windows 12 --window-len 512 2>&1 | grep -E "PPL" >> logs/gptq.log
echo DONE >> logs/gptq.log
