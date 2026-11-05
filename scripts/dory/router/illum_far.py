#!/usr/bin/env python3
"""Illumination of long-range recall: which KV heads of the full-attention layers read a
planted fact. A fact in a long filler, a question at the end, the answer teacher-forced; at
the answer's rows, the attention mass of each KV head on the fact's rows.
illum_far.py HF_DIR WIKI_TXT OUT.json [SAMPLES] [FILLER_CHARS]"""
import json, random, sys
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
sys.path.insert(0, "/root/dory")
from ref_dory import Dory

hf, wiki, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
chars = int(sys.argv[5]) if len(sys.argv) > 5 else 22000
tok = PreTrainedTokenizerFast(tokenizer_file=hf + "/tokenizer.json", bos_token="<s>", eos_token="<|im_end|>", unk_token="<unk>", pad_token="<|im_end|>")
tok.chat_template = open(hf + "/chat_template.jinja").read()
text = open(wiki, encoding="utf-8").read().replace("\r", "")
random.seed(11)
m = Dory(hf, "cuda", torch.bfloat16)
hd, nk, nh = m.hd, m.nk, m.nh
rep = nh // nk
full = [li for li in range(m.n) if m.types[li] == "full_attention"]
print("full layers", full, flush=True)
fact = {(li, kh): [] for li in full for kh in range(nk)}
far = {(li, kh): [] for li in full for kh in range(nk)}
state = {"span": None, "near": 2048, "on": False}


def probe(li, kind, a):
    if not state["on"] or kind != "full_attention":
        return
    T = a.shape[2]
    lo, hi = state["span"]
    fm = a[:, :, lo:hi].sum(-1)               # [nh, t]
    old = a[:, :, :max(T - state["near"], 0)].sum(-1)
    for kh in range(nk):
        fact[(li, kh)].append(fm[kh * rep:(kh + 1) * rep].max(0).values.mean().item())
        far[(li, kh)].append(old[kh * rep:(kh + 1) * rep].mean().item())


m.probe = probe
hits = 0
for s in range(n):
    key = "".join(random.choice("0123456789") for _ in range(6))
    at = random.randrange(0, len(text) - chars - 1)
    fill = text[at:at + chars]
    if s % 2 == 0:
        note = f"Remember this pass key: {key}. It is important."
        msg = f"{note}\n\n{fill}\n\nWhat was the pass key given at the very beginning of this message? Reply with the six digits only."
    else:
        cut = random.randrange(chars // 5, chars // 2)
        note = f"Note: the access code for the archive is {key}."
        msg = f"{fill[:cut]}\n\n{note}\n\n{fill[cut:]}\n\nWhat is the access code for the archive mentioned above? Reply with the six digits only."
    prompt = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    c0 = prompt.index(key)
    rows = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < c0 + len(key) and b > c0]
    state["span"] = (rows[0], rows[-1] + 1)
    ans = tok(key, add_special_tokens=False)["input_ids"]
    caches = {}
    pos = 0
    t = torch.tensor(ids[:-1], device="cuda")
    state["on"] = False
    for c in t.split(256):
        m.forward(c, pos, caches, loops=2, kv="last_loop")
        pos += c.shape[0]
    state["on"] = True
    right = 0
    seq = [ids[-1]] + ans[:-1]
    for i, tid in enumerate(seq):
        lg = m.forward(torch.tensor([tid], device="cuda"), pos, caches, loops=2, kv="last_loop")
        pos += 1
        right += int(lg[-1].argmax().item() == ans[i])
    hits += right == len(ans)
    print(f"sample {s}: rows {len(ids)}, fact rows {state['span']}, answer {right}/{len(ans)}", flush=True)
    del caches
    torch.cuda.empty_cache()
tab = sorted(((float(np.mean(fact[k])), float(np.mean(far[k])), k[0], k[1]) for k in fact), reverse=True)
print(f"recalled {hits}/{n}")
print("fact mass, far mass, layer, kv head")
for f, o, li, kh in tab:
    print(f"{f:.4f} {o:.4f} {li} {kh}")
json.dump([{"fact": f, "far": o, "layer": li, "head": kh} for f, o, li, kh in tab], open(out, "w"), indent=0)
