#!/usr/bin/env python3
"""The prompt of one trial of passkey_file.py (same seed, same draws): passkey_prompt.py WIKI CHARS TRIAL OUT"""
import random, sys
wiki, chars, trial, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
text = open(wiki, encoding="utf-8").read().replace("\r", "")
random.seed(41)
KINDS = [
    lambda k: (f"Remember this pass key: {k}. It is important.", "What was the pass key given above? Reply with the digits only.", k),
    lambda k: (f"Note: the access code for the archive is {k}.", "What is the access code for the archive mentioned above? Reply with the digits only.", k),
    lambda k: (f"The password of the staging server is qX{k}vT.", "What is the password of the staging server mentioned above? Reply with the password only.", f"qX{k}vT"),
    lambda k: (f"The meeting with Ilmari Voskresky was moved to room {k}.", "Which room was the meeting with Ilmari Voskresky moved to? Reply with the number only.", k),
]
for t in range(trial + 1):
    i = t % len(KINDS)
    key = "".join(random.choice("0123456789") for _ in range(6))
    note, q, ans = KINDS[i](key)
    at = random.randrange(0, len(text) - chars - 1)
    fill = text[at:at + chars]
    cut = random.randrange(0, chars * 8 // 10)
    msg = f"{fill[:cut]}\n\n{note}\n\n{fill[cut:]}\n\n{q}"
open(out, "w").write(msg)
print(ans)
