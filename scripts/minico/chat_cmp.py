"""Compare `cortiq run --greedy` answers (files PREFIX_<i>.txt) with chat_ref.py
JSON lines: identical text after whitespace normalisation, else where it diverges.

usage: python -I chat_cmp.py REF.jsonl PREFIX
"""
import json
import re
import sys


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


for i, line in enumerate(open(sys.argv[1]), 1):
    ref = json.loads(line)
    want = norm(ref["answer"].replace("<|im_end|>", ""))
    out = open(f"{sys.argv[2]}_{i}.txt").read()
    body = out.split("Prompt: " + ref["prompt"], 1)[1]
    body = norm(re.sub(r"\n\[\d+ tokens[^\]]*\]\s*$", "", body.strip()))
    n = min(len(body), len(want))
    k = next((j for j in range(n) if body[j] != want[j]), n)
    same = body == want or (k == n and n > 0)
    tag = "identical" if body == want else ("same prefix (length cut)" if k == n else f"diverges at char {k}/{len(want)}")
    print(f"[{i}] {ref['effort']} {ref['prompt'][:40]!r}: {tag}")
    if not same:
        print(f"    cortiq: ...{body[max(0, k - 30):k + 50]}")
        print(f"    ref:    ...{want[max(0, k - 30):k + 50]}")
