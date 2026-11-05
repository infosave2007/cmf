#!/usr/bin/env python3
"""Facts of four kinds at random places of a long text, asked at the end through `cortiq run`.
passkey2.py CORTIQ MODEL WIKI CHARS TRIALS [ENV=VAL ...]"""
import os, random, subprocess, sys

cortiq, model, wiki, chars, trials = sys.argv[1:6]
chars, trials = int(chars), int(trials)
env = dict(os.environ, XDG_RUNTIME_DIR="/tmp", **dict(a.split("=", 1) for a in sys.argv[6:]))
text = open(wiki, encoding="utf-8").read().replace("\r", "")
random.seed(41)
KINDS = [
    lambda k: (f"Remember this pass key: {k}. It is important.", "What was the pass key given above? Reply with the digits only.", k),
    lambda k: (f"Note: the access code for the archive is {k}.", "What is the access code for the archive mentioned above? Reply with the digits only.", k),
    lambda k: (f"The password of the staging server is qX{k}vT.", "What is the password of the staging server mentioned above? Reply with the password only.", f"qX{k}vT"),
    lambda k: (f"The meeting with Ilmari Voskresky was moved to room {k}.", "Which room was the meeting with Ilmari Voskresky moved to? Reply with the number only.", k),
]
ok = [0] * len(KINDS)
seen = [0] * len(KINDS)
for t in range(trials):
    i = t % len(KINDS)
    key = "".join(random.choice("0123456789") for _ in range(6))
    note, q, ans = KINDS[i](key)
    at = random.randrange(0, len(text) - chars - 1)
    fill = text[at:at + chars]
    cut = random.randrange(0, chars * 8 // 10)
    msg = f"{fill[:cut]}\n\n{note}\n\n{fill[cut:]}\n\n{q}"
    r = subprocess.run([cortiq, "run", model, "--prompt", msg, "--greedy", "--no-think", "--max-tokens", "24"], capture_output=True, text=True, env=env, timeout=1500)
    tail = r.stdout[-200:]
    hit = ans in tail
    ok[i] += hit
    seen[i] += 1
    print(f"  trial {t} kind {i} at {cut}/{chars}: {'OK' if hit else 'miss'} {ans} {tail[-110:]!r}", flush=True)
print(f"chars {chars}: total {sum(ok)}/{sum(seen)}, by kind " + " ".join(f"{a}/{b}" for a, b in zip(ok, seen)), flush=True)
