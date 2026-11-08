#!/bin/bash
# chain8.sh: recall against context depth on Vulkan — the facts test (passkey_file.py, 8 trials of
# 4 kinds) at 120 000 and 240 000 chars (~28k / ~56k tokens) for the standard q4tp file and the
# turbo file; the 60 000-char rows are in logs/pk_std_vk.log and logs/pk_turbo_vk.log already.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp
for n in 120000 240000; do
  echo "== std $n"; venv/bin/python passkey_file.py ./cortiq hf/cmf/spark-x2.6-4b-q4tp.cmf data/wiki.test.raw $n 8 2>&1 | tail -9
  echo "== turbo $n"; venv/bin/python passkey_file.py ./cortiq out/spark-x2.6-4b-q4tp-turbo.cmf data/wiki.test.raw $n 8 2>&1 | tail -9
done > logs/chain8.out 2>&1
echo CHAIN8_DONE > logs/chain8.done
