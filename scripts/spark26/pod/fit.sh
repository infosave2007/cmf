#!/bin/bash
# fit.sh FILE N CHARS: exact-row weights for Spark-X2.6-4B (9 full layers x 4 KV heads x 2 = 72 rank features)
# over N prompts with a planted fact; writes keep_spark26.bin.
cd /root/spark26
export XDG_RUNTIME_DIR=/tmp CMF_O1_M=32 ${CMF_O1_PREFILL:+CMF_O1_PREFILL=$CMF_O1_PREFILL}
venv/bin/python cmf/scripts/dory/router/q27_feat.py ./cortiq $1 hf/src/tokenizer.json data/wiki.test.raw keep_spark26.bin ${2:-24} ${3:-21000} 72
echo FIT_DONE
