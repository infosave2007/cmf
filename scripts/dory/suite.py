#!/usr/bin/env python3
"""Real prompts through `cortiq run`: tokens per second and how far the text stays the
same as the baseline's. suite.py CORTIQ MODEL NAME [ENV=VAL ...] (the first run of a
MODEL is its baseline, kept in suite_base.json next to the script's cwd)."""
import json, os, re, subprocess, sys

cortiq, model, name = sys.argv[1:4]
extra = dict(a.split("=", 1) for a in sys.argv[4:])
env = dict(os.environ, XDG_RUNTIME_DIR="/tmp", **extra)
P = [
    "What is 17 * 23? Answer briefly.",
    "A train leaves at 3 pm at 60 km/h. Another leaves the same station at 4 pm at 80 km/h in the same direction. When does the second catch the first?",
    "If 3 pencils cost 45 cents, how much do 7 pencils cost?",
    "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation.",
    "Write a Rust function that reverses a singly linked list.",
    "Write a SQL query that returns the top 3 customers by total order amount per country.",
    "Why is the sky blue? Explain it to a ten year old.",
    "What were the main causes of the First World War?",
    "Give me five tips for writing a good cover letter.",
    "Translate to French: The library closes at nine, but the reading room stays open until midnight.",
    "What is the difference between weather and climate?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]
tok_s, texts, tokens = [], [], 0
for q in P:
    r = subprocess.run([cortiq, "run", model, "--prompt", q, "--greedy", "--max-tokens", os.environ.get("SUITE_TOKENS", "200")], capture_output=True, text=True, env=env, timeout=900)
    o = r.stdout
    m = re.search(r"\[(\d+) tokens · first token ([\d.]+) s · decode ([\d.]+) tok/s", o)
    if not m:
        print("no stats:", (r.stderr or o)[-300:])
        continue
    n, dec = int(m.group(1)), float(m.group(3))
    tokens += n
    tok_s.append((n, dec))
    head = "Prompt: " + q + "\n"
    a = o.find(head)
    texts.append(re.sub(r"\n?\[\d+ tokens ·.*$", "", o[a + len(head):], flags=re.S))
total_t = sum(n / d for n, d in tok_s if d > 0)
rate = sum(n for n, _ in tok_s) / total_t if total_t else 0
base_path = "suite_base_" + os.path.basename(model) + ".json"
if os.path.exists(base_path):
    base = json.load(open(base_path))
    same = []
    for a, b in zip(texts, base):
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        same.append(i / max(len(b), 1))
    agree = sum(same) / len(same)
else:
    json.dump(texts, open(base_path, "w"))
    agree = 1.0
print(f"{name:28s} {rate:6.1f} tok/s over {tokens} tokens, text same as baseline {agree*100:5.1f} %", flush=True)
