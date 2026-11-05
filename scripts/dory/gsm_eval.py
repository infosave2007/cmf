#!/usr/bin/env python3
"""GSM8K accuracy of a configuration, greedy, through `cortiq run`. Prints the rate and
the mean tokens per answer, so the cost of a correct answer shows.
gsm_eval.py CORTIQ MODEL PARQUET N NAME [ENV=VAL ...]"""
import os, re, subprocess, sys, time
import pyarrow.parquet as pq

cortiq, model, parquet, n, name = sys.argv[1:6]
n = int(n)
env = dict(os.environ, XDG_RUNTIME_DIR="/tmp", **dict(a.split("=", 1) for a in sys.argv[6:]))
t = pq.read_table(parquet)
Q = t.column("question").to_pylist()[:n]
A = t.column("answer").to_pylist()[:n]


def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


ok = 0
tokens = 0
t0 = time.time()
for q, a in zip(Q, A):
    gold = num(a.split("####")[-1])
    r = subprocess.run([cortiq, "run", model, "--prompt", q + "\nPut the final answer after ####.", "--greedy", "--no-think", "--max-tokens", "500"], capture_output=True, text=True, env=env, timeout=600)
    m = re.search(r"\[(\d+) tokens", r.stdout)
    tokens += int(m.group(1)) if m else 0
    body = re.sub(r"\n?\[\d+ tokens ·.*$", "", r.stdout, flags=re.S)
    after = body.split("</think>")[-1]
    pred = num(after.split("####")[-1]) if "####" in after else num(after)
    ok += pred == gold
dt = time.time() - t0
print(f"{name:20s} GSM8K {ok}/{n}  tokens/answer {tokens/n:.0f}  wall {dt:.0f} s  {tokens/dt:.1f} tok/s", flush=True)
