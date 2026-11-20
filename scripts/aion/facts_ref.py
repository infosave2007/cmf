#!/usr/bin/env python3
"""Planted facts in long wikitext for the Transformers reference (bf16 on CUDA), and the
prompts as JSON for the engine (facts_engine.py reads them).

usage: facts_ref.py MODEL_DIR WIKI CHARS N OUT.json
FACTS_NO_REF=1 writes the prompts only (no Transformers run).
"""
import json, random, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, wiki, chars, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
text = open(wiki, encoding="utf-8").read().replace("\r", "")
random.seed(7 + chars)
KINDS = [
    lambda k: (f"Remember this pass key: {k}. It is important.", "What was the pass key given above? Reply with the digits only.", k),
    lambda k: (f"Note: the access code for the archive is {k}.", "What is the access code for the archive mentioned above? Reply with the digits only.", k),
    lambda k: (f"The password of the staging server is qX{k}vT.", "What is the password of the staging server mentioned above? Reply with the password only.", f"qX{k}vT"),
    lambda k: (f"The meeting with Ilmari Voskresky was moved to room {k}.", "Which room was the meeting with Ilmari Voskresky moved to? Reply with the number only.", k),
]
items = []
for s in range(n):
    key = "".join(random.choice("0123456789") for _ in range(6))
    at = random.randrange(0, len(text) - chars - 1)
    fill = text[at:at + chars]
    note, q, fact = KINDS[s % len(KINDS)](key)
    cut = random.randrange(chars // 10, chars * 9 // 10)
    msg = f"{fill[:cut]}\n\n{note}\n\n{fill[cut:]}\n\n{q}"
    items.append({"msg": msg, "fact": fact, "depth": cut / chars})
import os
if os.environ.get("FACTS_NO_REF") == "1":
    json.dump(items, open(out, "w"), ensure_ascii=False)
    print("FACTS_PROMPTS", len(items), flush=True)
    sys.exit(0)
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
ok = 0
for it in items:
    p = tok.apply_chat_template([{"role": "user", "content": it["msg"]}], tokenize=False, add_generation_prompt=True, current_date="2026-11-20")
    ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    t0 = time.time()
    with torch.inference_mode():
        g = model.generate(input_ids=ids, max_new_tokens=40, do_sample=False)
    ans = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
    it.update({"tokens": ids.shape[1], "ref": ans, "ref_ok": it["fact"] in ans, "ref_s": time.time() - t0})
    ok += it["ref_ok"]
    print(f"tokens {ids.shape[1]} depth {it['depth']:.2f} ok {it['ref_ok']} {time.time()-t0:.0f}s: {ans[:120]!r}", flush=True)
json.dump(items, open(out, "w"), ensure_ascii=False)
print(f"FACTS_REF_DONE {ok}/{len(items)}", flush=True)
