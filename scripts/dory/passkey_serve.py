#!/usr/bin/env python3
"""Facts in a text too long for an argument, through a running `cortiq serve`.
passkey_serve.py PORT WIKI CHARS TRIALS [FIRST]"""
import json, random, sys, time, urllib.error, urllib.request

port, wiki, chars, trials = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
text = open(wiki, encoding="utf-8").read().replace("\r", "")
random.seed(43)
KINDS = [
    lambda k: (f"Remember this pass key: {k}. It is important.", "What was the pass key given above? Reply with the digits only.", k),
    lambda k: (f"Note: the access code for the archive is {k}.", "What is the access code for the archive mentioned above? Reply with the digits only.", k),
    lambda k: (f"The password of the staging server is qX{k}vT.", "What is the password of the staging server mentioned above? Reply with the password only.", f"qX{k}vT"),
    lambda k: (f"The meeting with Ilmari Voskresky was moved to room {k}.", "Which room was the meeting with Ilmari Voskresky moved to? Reply with the number only.", k),
]
first = int(sys.argv[5]) if len(sys.argv) > 5 else 0
ok = 0
for t in range(trials):
    key = "".join(random.choice("0123456789") for _ in range(6))
    note, q, ans = KINDS[t % len(KINDS)](key)
    at = random.randrange(0, len(text) - chars - 1)
    fill = text[at:at + chars]
    cut = random.randrange(0, chars * 8 // 10)
    msg = f"{fill[:cut]}\n\n{note}\n\n{fill[cut:]}\n\n{q}"
    if t < first:
        continue
    body = {"model": "dory", "messages": [{"role": "user", "content": msg}], "temperature": 0, "max_tokens": 24, "enable_thinking": False}
    t0 = time.time()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(r, timeout=1500))
    except urllib.error.HTTPError as e:
        print(f"  trial {t}: HTTP {e.code} {e.read()[:300]!r}", flush=True)
        continue
    out = d["choices"][0]["message"]["content"]
    hit = ans in out
    ok += hit
    print(f"  trial {t} at {cut}/{chars} rows {d['usage']['prompt_tokens']}: {'OK' if hit else 'miss'} {ans} {out[-80:]!r} {time.time()-t0:.0f} s", flush=True)
print(f"chars {chars}: {ok}/{trials}", flush=True)
