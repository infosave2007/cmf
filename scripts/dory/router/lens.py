#!/usr/bin/env python3
"""The head read straight off the state after one pass, skipping the output block: is
its confidence a cheap stand-in for the read-out's?"""
import json, numpy as np, torch
from safetensors import safe_open

d = np.load("router/probe.npz")
S, ans, kinds = d["sample"], d["answer"], d["kinds"]
test = (S % 2 == 1) & ans & (kinds[S] != "wiki")
idx = np.where(test)[0]
idx = idx[np.linspace(0, len(idx) - 1, 6000).astype(int)]
with safe_open("hf/bf16/model.safetensors", "pt") as f:
    H = f.get_tensor("lm_head.weight").float().cuda()
    sc = f.get_tensor("lm_head.logit_scale").float().cuda() / 0.0177
easy = d["t1"][idx] == d["t2"][idx]
def auc(s, y):
    p, n = s[y], s[~y]
    r = np.random.default_rng(0)
    i, j = r.integers(0, len(p), 300000), r.integers(0, len(n), 300000)
    return float(np.mean(p[i] > n[j]))
for sig in ("phi_in", "phi_1"):
    x = torch.tensor(d[sig][idx].astype(np.float32)).cuda()
    out = []
    for b in range(0, len(x), 256):
        lg = (x[b:b+256] @ H.T) * sc
        out.append(torch.softmax(lg, -1).max(-1).values.cpu())
    p = torch.cat(out).numpy()
    print(sig, "head read directly: AUC for t1==t2", round(auc(p, easy), 3), "p quantiles", np.percentile(p, [10, 50, 90]).round(3))
print("read-out after the output block: AUC", round(auc(d["p1"][idx], easy), 3))
