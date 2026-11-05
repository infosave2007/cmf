#!/usr/bin/env python3
"""Row features of prompts with a planted fact, for the fit in illum_fit.py: per KV head of
each full layer the residual against the landmarks, the value's norm and the raw key's norm,
and the row's surprise. illum_feat.py HF_DIR WIKI_TXT OUT.npz [SAMPLES] [FILLER_CHARS]"""
import random, sys
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
sys.path.insert(0, "/root/dory")
from ref_dory import Dory

hf, wiki, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
chars = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
tok = PreTrainedTokenizerFast(tokenizer_file=hf + "/tokenizer.json", bos_token="<s>", eos_token="<|im_end|>", unk_token="<unk>", pad_token="<|im_end|>")
tok.chat_template = open(hf + "/chat_template.jinja").read()
text = open(wiki, encoding="utf-8").read().replace("\r", "")
random.seed(31)
m = Dory(hf, "cuda", torch.bfloat16)
full = [li for li in range(m.n) if m.types[li] == "full_attention"]


def residual(K, mm=32):
    t, nk, hd = K.shape
    cut = torch.linspace(0, t, mm + 1).long().tolist()
    res = []
    for kh in range(nk):
        k = K[:, kh].float()
        L = torch.stack([k[cut[i]:cut[i + 1]].mean(0) for i in range(mm)])
        G = L @ L.T + 1e-4 * torch.eye(mm, device=L.device)
        c = torch.linalg.solve(G, L @ k.T)
        r = k - (L.T @ c).T
        res.append((r * r).sum(-1))
    return torch.stack(res, 1)


KINDS = [
    lambda key: (f"Remember this pass key: {key}. It is important.", key),
    lambda key: (f"Note: the access code for the archive is {key}.", key),
    lambda key: (f"The password of the staging server is qX{key}vT.", f"qX{key}vT"),
    lambda key: (f"The meeting with Ilmari Voskresky was moved to room {key}.", "Ilmari Voskresky"),
    lambda key: (f"Deploy with the token sk-{key}a{key[::-1]} and nothing else.", f"sk-{key}a{key[::-1]}"),
]
raws = {}


def rawk(li, kind, k, q):
    if kind == "full_attention":
        raws.setdefault(li, []).append(torch.linalg.vector_norm(k, dim=-1).float())


m.raw = rawk
Xs, ys, ks = [], [], []
for s in range(n):
    key = "".join(random.choice("0123456789") for _ in range(6))
    at = random.randrange(0, len(text) - chars - 1)
    fill = text[at:at + chars]
    note, fact = KINDS[s % len(KINDS)](key)
    cut = random.randrange(chars // 10, chars // 2)
    msg = f"{fill[:cut]}\n\n{note}\n\n{fill[cut:]}\n\nWhat was noted above? Reply briefly."
    prompt = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    c0 = prompt.index(fact)
    frows = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < c0 + len(fact) and b > c0]
    caches = {}
    pos = 0
    sur = []
    t = torch.tensor(ids, device="cuda")
    raws.clear()
    for c in t.split(256):
        lg = m.forward(c, pos, caches, loops=2, kv="last_loop")
        lp = torch.log_softmax(lg.float(), -1)
        nxt = t[pos + 1:pos + c.shape[0] + 1]
        sur.append(-lp[torch.arange(nxt.shape[0], device="cuda"), nxt])
        pos += c.shape[0]
    sur = torch.cat([torch.zeros(1, device="cuda"), torch.cat(sur)])[:len(ids)]
    F = []
    for li in full:
        K, V = caches[(li, 0)]["k"], caches[(li, 0)]["v"]
        F.append(residual(K))                                         # [t, nk]
        F.append(torch.linalg.vector_norm(V.float(), dim=-1))         # [t, nk]
        R = raws[li]
        # The last write of a row: a recurrent layer's pass 2, chunk by chunk.
        step = len(R) // len(list(t.split(256)))
        F.append(torch.cat(R[step - 1::step], 0))
    X = torch.cat(F + [sur[:, None]], 1).cpu().numpy().astype(np.float32)
    y = np.zeros(len(ids), dtype=bool)
    y[frows] = True
    Xs.append(X)
    ys.append(y)
    ks.append(s % len(KINDS))
    print(s, X.shape, frows[:1], len(frows), flush=True)
    del caches
    torch.cuda.empty_cache()
np.savez(out, X=np.array(Xs, dtype=object), y=np.array(ys, dtype=object), kind=np.array(ks), full=np.array(full))
print("DONE")
