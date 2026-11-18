#!/usr/bin/env python3
"""Chat-format calibration text for GPTQ through a running `cortiq serve`: the model's own greedy answers
(no thinking) to a prompt set, rendered through its chat template.
usage: chat_calib_serve.py PORT HF_DIR PROMPTS.json OUT.txt [MAX_TOKENS]"""
import json, sys, urllib.request
from transformers import AutoTokenizer

port, hf, prompts, out = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 400
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
docs = []
for kind, qs in json.load(open(prompts)).items():
    for q in qs:
        req = {"model": "sn", "messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": n,
               "enable_thinking": False}
        r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        body = json.load(urllib.request.urlopen(r, timeout=900))["choices"][0]["message"]["content"].strip()
        docs.append(tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": body}],
                                            tokenize=False, enable_thinking=False))
        print(kind, len(body), body[:80].replace("\n", " "), flush=True)
open(out, "w").write("\n".join(docs))
print("CALIB_DONE", len(docs), flush=True)
