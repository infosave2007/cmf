#!/bin/bash
# lm-eval 0.4.12 (MIT, no torch) and ArithMark-3 (Apache-2.0) for the KLM-1.6 index on this pod, each in its own
# directory under /root/sk/bench. Log /root/sk/bench/setup.log, marker /root/sk/bench/setup.done
B=/root/sk/bench; mkdir -p $B
{
  command -v uv >/dev/null || { apt-get install -y -qq python3-venv >/dev/null 2>&1; }
  [ -x $B/venv/bin/python ] || python3 -m venv $B/venv
  $B/venv/bin/pip install -q "lm-eval==0.4.12" && echo PIP_OK
  mkdir -p $B/dl-arithmark3
  for f in README.md bencharithmark-3.py arithmark-3.jsonl; do
    curl -sL -o $B/dl-arithmark3/$f "https://huggingface.co/datasets/AxiomicLabs/ArithMark-3.0/resolve/main/$f"
  done
  ls -la $B/dl-arithmark3
  touch $B/setup.done
} > $B/setup.log 2>&1
