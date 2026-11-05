#!/usr/bin/env python3
"""Which rows of a prompt hold what the far field would lose. A fact in a long filler; each
signal ranks all rows, and the fact's rows should come out on top. No word lists: the fact is
found by what the model and the sketch do with it.
  residual  how far a key is from the span of the far field's 32 landmarks (full layers)
  surprise  the row's token against the model's own prediction
  router    the state after one pass against the two topologies of patent 1
illum_rows.py HF_DIR WIKI_TXT TOPO.npz OUT.json [SAMPLES] [FILLER_CHARS]"""
import json, random, sys
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
sys.path.insert(0, "/root/dory")
from ref_dory import Dory

hf, wiki, topo, out = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 12
chars = int(sys.argv[6]) if len(sys.argv) > 6 else 22000
tok = PreTrainedTokenizerFast(tokenizer_file=hf + "/tokenizer.json", bos_token="<s>", eos_token="<|im_end|>", unk_token="<unk>", pad_token="<|im_end|>")
tok.chat_template = open(hf + "/chat_template.jinja").read()
text = open(wiki, encoding="utf-8").read().replace("\r", "")
random.seed(23)
m = Dory(hf, "cuda", torch.bfloat16)
full = [li for li in range(m.n) if m.types[li] == "full_attention"]
T = np.load(topo)
me, be, mh, bh = [torch.tensor(T[k], device="cuda", dtype=torch.float32) for k in ("me", "be", "mh", "bh")]


def recon(x, mean, basis):
    d = x - mean
    return (d * d).sum(-1) - ((d @ basis) ** 2).sum(-1)


def residual(K, mm=32):
    """Per row and KV head: the part of the key outside the span of segment means."""
    t, nk, hd = K.shape
    cut = torch.linspace(0, t, mm + 1).long().tolist()
    res = []
    for kh in range(nk):
        k = K[:, kh].float()
        L = torch.stack([k[cut[i]:cut[i + 1]].mean(0) for i in range(mm)])
        G = L @ L.T + 1e-4 * torch.eye(mm, device=L.device)
        c = torch.linalg.solve(G, L @ k.T)  # [m, t]
        r = k - (L.T @ c).T
        res.append((r * r).sum(-1))
    return torch.stack(res, 1)  # [t, nk]


KINDS = [
    lambda key: (f"Remember this pass key: {key}. It is important.", "What was the pass key given in this message? Reply with the six digits only.", key),
    lambda key: (f"Note: the access code for the archive is {key}.", "What is the access code for the archive mentioned above? Reply with the six digits only.", key),
    lambda key: (f"The password of the staging server is qX{key}vT.", "What is the password of the staging server mentioned above? Reply with the password only.", f"qX{key}vT"),
    lambda key: (f"The meeting with Ilmari Voskresky was moved to room {key}.", "Which room was the meeting with Ilmari Voskresky moved to? Reply with the number only.", key),
]
rows = []
for s in range(n):
    key = "".join(random.choice("0123456789") for _ in range(6))
    at = random.randrange(0, len(text) - chars - 1)
    fill = text[at:at + chars]
    note, q, ans_text = KINDS[s % len(KINDS)](key)
    cut = random.randrange(chars // 10, chars // 2)
    msg = f"{fill[:cut]}\n\n{note}\n\n{fill[cut:]}\n\n{q}"
    prompt = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    c0 = prompt.index(ans_text)
    frows = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < c0 + len(ans_text) and b > c0]
    caches = {}
    pos = 0
    sur, rt = [], []
    t = torch.tensor(ids, device="cuda")
    for c in t.split(256):
        dump = {}
        lg = m.forward(c, pos, caches, loops=2, kv="last_loop", dump=dump)
        lp = torch.log_softmax(lg.float(), -1)
        nxt = t[pos + 1:pos + c.shape[0] + 1]
        s1 = -lp[torch.arange(nxt.shape[0], device="cuda"), nxt]
        sur.append(s1)
        x = dump["loop0"]
        rt.append(torch.min(recon(x, me, be), recon(x, mh, bh)))
        pos += c.shape[0]
    sur = torch.cat(sur)
    # The surprise of row r is about the token of row r+1.
    sur = torch.cat([torch.zeros(1, device="cuda"), sur])[:len(ids)]
    rt = torch.cat(rt)[:len(ids)]
    res = torch.stack([residual(caches[(li, 0)]["k"]) for li in full])  # [layers, t, nk]
    sig = {
        "residual_mean": res.mean((0, 2)),
        "residual_max": res.amax((0, 2)),
        "surprise": sur,
        "router": rt,
    }
    # Rank-sum of the independent ones: a row must stand out to more than one sensor.
    def pct(v):
        return torch.argsort(torch.argsort(v)).float() / (v.shape[0] - 1)
    sig["surprise+residual"] = pct(sig["surprise"]) + pct(sig["residual_mean"])
    sig["surprise+router"] = pct(sig["surprise"]) + pct(sig["router"])
    rec = {"sample": s, "kind": s % len(KINDS), "rows": len(ids), "fact_rows": frows}
    for name, v in sig.items():
        order = torch.argsort(v, descending=True)
        rank = torch.zeros_like(order)
        rank[order] = torch.arange(order.shape[0], device=order.device)
        rr = rank[torch.tensor(frows, device=rank.device)].float() / len(ids) * 100
        rec[name] = {"best": rr.min().item(), "worst": rr.max().item(), "mean": rr.mean().item()}
    rows.append(rec)
    print(s, "kind", s % len(KINDS), "rows", len(ids), "fact", frows[:1], len(frows), " ".join(f"{k}={rec[k]['best']:.2f}/{rec[k]['mean']:.1f}" for k in sig), flush=True)
    del caches
    torch.cuda.empty_cache()
print("signal: mean over samples of the fact's best row and of its mean row, % of the prompt above them")
for name in sig:
    print(f"{name:20s} best {np.mean([r[name]['best'] for r in rows]):6.2f}  mean {np.mean([r[name]['mean'] for r in rows]):6.2f}  worst {np.mean([r[name]['worst'] for r in rows]):6.2f}")
json.dump(rows, open(out, "w"))
