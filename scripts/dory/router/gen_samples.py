#!/usr/bin/env python3
"""The model's own answers on a mixed prompt set: the tokens a decode really sees.
gen_samples.py CORTIQ MODEL HF_DIR GSM_PARQUET PROMPTS OUT [N_GSM] [LOOPS]"""
import json, os, re, subprocess, sys, time
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast

cortiq, model, hf, gsm, prompts, out = sys.argv[1:7]
n_gsm = int(sys.argv[7]) if len(sys.argv) > 7 else 40
tok = PreTrainedTokenizerFast(tokenizer_file=hf + "/tokenizer.json", bos_token="<s>", eos_token="<|im_end|>", unk_token="<unk>", pad_token="<|im_end|>")
tok.chat_template = open(hf + "/chat_template.jinja").read()
t = pq.read_table(gsm)
Q = [("gsm", q + "\nPut the final answer after ####.", a) for q, a in zip(t.column("question").to_pylist()[:n_gsm], t.column("answer").to_pylist()[:n_gsm])]
P = json.load(open(prompts))
for k, v in P.items():
    Q += [(k, q, None) for q in v]
env = dict(os.environ, XDG_RUNTIME_DIR="/tmp")
res = []
t0 = time.time()
for i, (kind, q, a) in enumerate(Q):
    r = subprocess.run([cortiq, "run", model, "--prompt", q, "--greedy", "--max-tokens", "420"], capture_output=True, text=True, env=env, timeout=120)
    o = r.stdout
    head = "Prompt: " + q + "\n"
    m = o.find(head)
    body = o[m + len(head):] if m >= 0 else o
    body = re.sub(r"\n?\[\d+ tokens ·.*$", "", body, flags=re.S).lstrip("\n")
    prompt_text = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
    pid = tok(prompt_text, add_special_tokens=False)["input_ids"]
    aid = tok(body, add_special_tokens=False)["input_ids"]
    res.append({"kind": kind, "q": q, "gold": a, "text": body, "ids": pid + aid, "prompt_len": len(pid)})
    print(i, kind, len(pid), len(aid), f"{time.time()-t0:.0f}s", repr(body[:70]), flush=True)
    json.dump(res, open(out, "w"))
