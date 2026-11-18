#!/usr/bin/env python3
"""Chat-format calibration text for GPTQ: the model's own greedy answers (no thinking) to a
prompt set, rendered through its chat template.

usage: chat_calib.py CORTIQ MODEL.cmf HF_DIR PROMPTS.json OUT.txt [MAX_TOKENS]
PROMPTS.json: {"kind": ["prompt", ...], ...}. Runs `cortiq run` with the caller's env.
"""
import json, re, subprocess, sys
from transformers import AutoTokenizer

cortiq, model, hf, prompts, out = sys.argv[1:6]
n = sys.argv[6] if len(sys.argv) > 6 else "400"
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
P = json.load(open(prompts))
docs = []
for kind, qs in P.items():
    for q in qs:
        r = subprocess.run([cortiq, "run", model, "--prompt", q, "--greedy", "--no-think", "--max-tokens", n],
                           capture_output=True, text=True, timeout=600)
        o = r.stdout
        head = "Prompt: " + q + "\n"
        m = o.find(head)
        body = o[m + len(head):] if m >= 0 else o
        body = re.sub(r"\n?\[\d+ tokens ·.*$", "", body, flags=re.S).strip()
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": body}]
        docs.append(tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False))
        print(kind, len(body), body[:80].replace("\n", " "), flush=True)
open(out, "w").write("\n".join(docs))
print("CALIB_DONE", len(docs), flush=True)
