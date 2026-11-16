#!/bin/bash
# GLM-5.3's DSA on the card past index_topk (the indexer's pools, the top-k
# selection, the absorbed attention over the selected latents): with the
# selection cut to 64 positions (CMF_GLM_DSA_TOPK=64) a 256-token perplexity
# runs the selection on both the host and the card; then the file's 2048 on a
# 3000-token prompt (TTFT and decode), the texts against the card below 2048.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glmd1 out/glmd1
end=$((SECONDS + 7200))
until grep -qE "GLMG3_DONE|GLMG3_BUILD_FAILED" logs/glmg3.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/glmd1/build.log 2>&1
grep -q BUILD_FAST_DONE logs/glmd1/build.log || { echo GLMD1_BUILD_FAILED; grep -E '^error|error:' -A8 logs/glmd1/build.log | head -60; exit 1; }
strings bin/cortiq-fast | grep -q CMF_GLM_DSA_TOPK || { echo GLMD1_STALE_BINARY; exit 1; }
B=bin/cortiq-fast; f=cmf/big/glm-5.3-flash-q2tp.cmf
for v in cuda cpu; do
  case $v in cpu) e="CMF_GPU=0";; cuda) e="CMF_GPU=cuda CMF_CUDA_DEBUG=1";; esac
  flock gpu.lock env $e CMF_GLM_DSA_TOPK=64 RUST_LOG=info timeout 5400 $B ppl $f --file prompts/rail.txt --tokens 256 > logs/glmd1/ppl64_$v.log 2>&1
  echo "ppl topk=64 $v: exit $? | $(grep -hiE 'ppl|perplexity' logs/glmd1/ppl64_$v.log | tail -1 | cut -c1-200)"
done
sed -E 's/\x1b\[[0-9;]*m//g' logs/glmd1/ppl64_cuda.log | grep -iE 'glm5_next|declin|replay|error|panic' | sort | uniq -c | head -8
python3 - <<'PY'
import random
random.seed(7)
words = open("prompts/rail.txt").read().split()
text = " ".join(random.choice(words) for _ in range(2600))
open("prompts/long3k.txt", "w").write("Read this text and then say in one sentence what it is about.\n\n" + text)
PY
flock gpu.lock env CMF_GPU=cuda RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 3072 --ignore-eos --json > logs/glmd1/p3k.log 2>&1
echo "cuda 3072+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glmd1/p3k.log | tr '\n' ' ')"
sed -E 's/\x1b\[[0-9;]*m//g' logs/glmd1/p3k.log | grep -iE 'glm5_next|replay|declin' | sort | uniq -c | head -5
echo GLMD1_DONE
