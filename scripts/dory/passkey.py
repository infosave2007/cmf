#!/usr/bin/env python3
"""Does the far field keep a fact? A pass key at the head of a long filler, asked at
the end: `passkey.py CORTIQ MODEL WIKI LENGTHS...` (characters of filler)."""
import os, random, re, subprocess, sys

cortiq, model, wiki = sys.argv[1:4]
text = open(wiki, encoding="utf-8").read().replace("\r", "")
env = dict(os.environ, XDG_RUNTIME_DIR="/tmp")
random.seed(7)
for n in [int(x) for x in sys.argv[4:]]:
    ok = 0
    trials = 4
    for t in range(trials):
        key = "".join(random.choice("0123456789") for _ in range(6))
        at = random.randrange(0, len(text) - n - 1)
        q = f"Remember this pass key: {key}. It is important.\n\n{text[at:at+n]}\n\nWhat was the pass key given at the very beginning of this message? Reply with the six digits only."
        r = subprocess.run([cortiq, "run", model, "--prompt", q, "--greedy", "--no-think", "--max-tokens", "24"], capture_output=True, text=True, env=env, timeout=900)
        tail = r.stdout[-260:]
        hit = key in tail
        ok += hit
        print(f"  chars {n} trial {t}: key {key} {'OK' if hit else 'miss'} {tail[-120:]!r}", flush=True)
    print(f"chars {n}: {ok}/{trials}", flush=True)
