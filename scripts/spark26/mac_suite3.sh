#!/bin/bash
# mac_suite3.sh DIR SCRIPTS: the final Mac rows of the Spark-X2.6-4B Turbo page, one GPU job at a
# time, a minute of rest before each run (a fanless MacBook Air slows under back-to-back runs).
#  1. llama.cpp perplexity on the native arm64 build (the cortiq rows are in suite3.log of the first run).
#  2. The depth sweep: prompts of 2k, 14k, 32k and 64k tokens (wiki text + one instruction), 96 greedy
#     tokens, for cortiq q4tp, cortiq q4tp Turbo, cortiq q4tp + Neural Engine, stock mlx-lm 4-bit and
#     llama.cpp Q4_K_M on Metal (-c sized to the prompt). Each run under fp_run.py: peak phys_footprint,
#     peak resident, wall. A second p2k round of cortiq and mlx at the end shows the thermal noise.
#  3. The row scores of a planted fact on the turbo file for the exact-rows figure (CMF_DORY_EXACT_LOG).
# Run by mac_llama_native.sh after the native llama.cpp build.
D=$1; S=$2; cd "$D"; PY=cmp/venv/bin/python; LL=cmp/llama.cpp/build-arm64/bin
Q=hf/spark-x2.6-4b-q4tp.cmf; T=hf/spark-x2.6-4b-q4tp-turbo.cmf
fp() { python3 "$S/fp_run.py" "$@"; }
rest() { python3 -c "import time; time.sleep(60)"; }
file $LL/llama-completion | grep -q arm64 || { echo "llama.cpp is not an arm64 build"; exit 1; }
for q in Q4_K_M Q8_0; do
  echo "== llama ppl $q (metal arm64, -c 512 --chunks 64)"
  $LL/llama-perplexity -m cmp/hf/gguf/Spark-X2.6-4B-$q.gguf -f data/wiki.test.raw -c 512 --chunks 64 -ngl 99 2>&1 | grep -E "estimate" | tail -1
done
cq() { # tag file prompt env...  -> one row
  local tag=$1 f=$2 p=$3; shift 3; rest
  local m=$(env "$@" CMF_PREFILL_PROF=1 timeout 1500 python3 "$S/fp_run.py" cmp/s3_${tag}_$p ./cortiq run "$f" --prompt-file $p.txt --no-think --greedy -n 96)
  local n=$(grep -o "metal-prefill: [0-9]* of [0-9]*" cmp/s3_${tag}_$p.err | tail -1 | awk '{print $4}')
  [ -z "$n" ] && n=$(grep -o "prefill: [0-9]* tokens" cmp/s3_${tag}_$p.err | tail -1 | awk '{print $2}')
  local line=$(tail -1 cmp/s3_${tag}_$p.out)
  local ft=$(echo "$line" | grep -o "first token [0-9.]*" | awk '{print $3}')
  echo "$tag | $p | prompt ${n:-?} tok | first token $ft s | prompt $(python3 -c "print(round(${n:-0}/${ft:-1},1))") tok/s | $(echo "$line" | grep -o "decode [0-9.]* tok/s") | $m"
}
mlx() { local p=$1; rest
  local m=$(timeout 1500 python3 "$S/fp_run.py" cmp/s3_mlx4_$p $PY "$S/mlx_bench.py" cmp/hf/mlx4 $p.txt 96)
  echo "mlx4 | $p | $(head -1 cmp/s3_mlx4_$p.out) | $m"
}
lcpp() { local p=$1 c=$2; rest
  local m=$(timeout 1500 python3 "$S/fp_run.py" cmp/s3_lcpp_$p $LL/llama-completion -m cmp/hf/gguf/Spark-X2.6-4B-Q4_K_M.gguf -f $p.txt -n 96 -c $c --temp 0 -no-cnv --no-warmup)
  echo "llama Q4_K_M | $p | -c $c | $(grep -E 'eval time' cmp/s3_lcpp_$p.err | sed 's/.*print: *//' | tr -s ' ' | tr '\n' ';') | $m"
}
echo "== depth sweep"
for p in p2k p14k p32k p64k; do
  case $p in p2k) c=2560;; p14k) c=16384;; p32k) c=36864;; p64k) c=69632;; esac
  cq cortiq $Q $p X=1
  cq turbo $T $p X=1
  mlx $p
  lcpp $p $c
  cq ane $Q $p CMF_ANE=1
done
echo "== p2k second round"
cq cortiq $Q p2k X=1
mlx p2k
echo "== fact scores (turbo, 60 000 characters, fact at 36 000)"
python3 - <<'PY' > fact.txt
text = open("data/wiki.test.raw", encoding="utf-8").read().replace("\r", "")
a = text[5000:65000]
print(a[:36000] + "\n\nRemember this pass key: 653269. It is important.\n\n" + a[36000:] + "\n\nWhat was the pass key given above? Reply with the digits only.")
PY
rest
CMF_DORY_EXACT_LOG=cmp/fact_scores.bin CMF_O1_TOKENS_DUMP=cmp/fact_tokens.bin timeout 900 ./cortiq run $T --prompt-file fact.txt --greedy --no-think -n 12 2>/dev/null | tail -c 200 | tr '\n' ' '
echo
$PY - <<'PY'
import numpy as np
from tokenizers import Tokenizer
t = Tokenizer.from_file("cmp/hf/mlx4/tokenizer.json")
ids = np.fromfile("cmp/fact_tokens.bin", dtype=np.uint32).tolist()
sc = np.fromfile("cmp/fact_scores.bin", dtype=np.float32)
hits = [i for i in range(len(ids) - 6) if "pass key" in t.decode(ids[i:i + 4]) and i < len(sc)]
print("rows", len(ids), "scores", len(sc), "fact rows from", hits[:3])
if hits:
    h = hits[0]; r = np.argsort(np.argsort(-sc))
    print("places", [int(r[j]) for j in range(h, min(h + 24, len(sc)))], repr(t.decode(ids[h:h + 24])))
PY
echo SUITE3_DONE
