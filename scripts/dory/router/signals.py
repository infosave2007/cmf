#!/usr/bin/env python3
"""Which frozen signal lets two isolated topologies call a token easy? AUC on held-out
answers for the state after one pass and simple functions of it."""
import numpy as np, torch

d = np.load("router/probe.npz")
S, ans, kinds = d["sample"], d["answer"], d["kinds"]
own = kinds[S] != "wiki"
fit = (S % 2 == 0) & ans & own
test = (S % 2 == 1) & ans & own
easy = d["t1"] == d["t2"]
dev = "cuda"
def topo(x, r):
    x = torch.tensor(x, device=dev, dtype=torch.float32)
    mu = x.mean(0); y = x - mu
    w, v = torch.linalg.eigh((y.T @ y).double())
    return mu, v[:, -r:].float()
def err(x, mu, b):
    y = torch.tensor(x, device=dev, dtype=torch.float32) - mu
    return ((y * y).sum(1) - ((y @ b) ** 2).sum(1)).cpu().numpy()
def auc(s, y):
    p, n = s[y], s[~y]
    r = np.random.default_rng(0)
    i, j = r.integers(0, len(p), 400000), r.integers(0, len(n), 400000)
    return float(np.mean(p[i] > n[j]))
a = d["phi_1"].astype(np.float32)
b = d["phi_in"].astype(np.float32)
sigs = {"phi_1": a, "phi_1 - phi_in": a - b, "phi_1 & phi_in": np.concatenate([a, b], 1), "cos-free phi_1*sign": np.abs(a)}
for name, x in sigs.items():
    for r in (16,):
        me, be = topo(x[fit & easy], r)
        mh, bh = topo(x[fit & ~easy], r)
        s = err(x, mh, bh) - err(x, me, be)
        print(f"{name:22s} rank {r}: AUC {auc(s[test], easy[test]):.3f}")
    # class balance: the easy class fit on the confident part only
conf = d["p1"] >= 0.9
hard2 = d["p1"] < 0.5
x = a
me, be = topo(x[fit & conf], 16)
mh, bh = topo(x[fit & hard2], 16)
s = err(x, mh, bh) - err(x, me, be)
print(f"phi_1, classes by confidence (>=0.9 vs <0.5): AUC for t1==t2 {auc(s[test], easy[test]):.3f}, for p1>=0.7 {auc(s[test], (d['p1']>=0.7)[test]):.3f}")
