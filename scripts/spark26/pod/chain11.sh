#!/bin/bash
# chain11.sh: the recall rows at 240 000 characters (standard and turbo q4tp, 8 trials each) that
# chain8 lost to the 1500 s subprocess timeout while the card was shared with the A/B runs;
# 3000 s a trial. Output logs/chain11.out, chain11.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp
sed -i 's/timeout=1500/timeout=3000/' passkey_file.py
{
echo "== std 240000"; venv/bin/python passkey_file.py ./cortiq hf/cmf/spark-x2.6-4b-q4tp.cmf data/wiki.test.raw 240000 8 2>&1 | tail -9
echo "== turbo 240000"; venv/bin/python passkey_file.py ./cortiq out/spark-x2.6-4b-q4tp-turbo.cmf data/wiki.test.raw 240000 8 2>&1 | tail -9
} > logs/chain11.out 2>&1
echo CHAIN11_DONE > logs/chain11.done
