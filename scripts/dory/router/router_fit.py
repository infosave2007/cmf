#!/usr/bin/env python3
"""Patent-1 routing of the loop count, offline. Two isolated topologies over a frozen
hidden state: the affine PCA subspace of the tokens one loop answers as well as two
("easy") and of the rest ("hard"). The class that reconstructs the state better wins,
with a bias set on the fit split. No shared transform, no trained gate.
router_fit.py probe.npz [--ranks 16,64] [--out PREFIX]"""
import argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("probe")
ap.add_argument("--ranks", default="16,64")
ap.add_argument("--out", default="")
a = ap.parse_args()
d = np.load(a.probe)
S, ans = d["sample"], d["answer"]
kinds = d["kinds"]
kof = kinds[S]
own = kof != "wiki"
fit = (S % 2 == 0) & ans
test = (S % 2 == 1) & ans
dev = "cuda" if torch.cuda.is_available() else "cpu"
n1, n2, t1, t2 = d["n1"], d["n2"], d["t1"], d["t2"]
easy = t1 == t2


def topology(x, r):
    x = torch.tensor(x, device=dev, dtype=torch.float32)
    mu = x.mean(0)
    y = x - mu
    w, v = torch.linalg.eigh((y.T @ y).double())
    return mu, v[:, -r:].float()


def err(x, mu, b):
    y = torch.tensor(x, device=dev, dtype=torch.float32) - mu
    return ((y * y).sum(1) - ((y @ b) ** 2).sum(1)).cpu().numpy()


def auc(s, lab):
    p, n = s[lab], s[~lab]
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(p), 400000), rng.integers(0, len(n), 400000)
    return float(np.mean(p[i] > n[j]))


def table(name, score, mask=test):
    t = np.where(mask)[0]
    fs = np.sort(score[fit & (kof == kof) & (own if mask is test_own else True)])
    print(f"-- {name}: AUC {auc(score[t], easy[t]):.3f}")
    print("   1-loop share  layers  speed  nll    always2  always1  top1 same as 2 loops  on routed")
    a2, a1 = n2[t].mean(), n1[t].mean()
    for q in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        th = fs[int(len(fs) * (1 - q))]
        one = score[t] >= th
        nll = np.where(one, n1[t], n2[t]).mean()
        same = np.where(one, t1[t] == t2[t], True).mean()
        routed = (t1[t][one] == t2[t][one]).mean() if one.any() else 0
        layers = 72 + 40 * (1 - one.mean())
        print(f"   {one.mean():.3f}         {layers:5.1f}   x{112/layers:.2f}  {nll:.3f}  {a2:.3f}    {a1:.3f}    {same:.4f}               {routed:.4f}")


test_own = test & own
t = test_own
for sig in ("phi_in", "phi_1"):
    x = d[sig].astype(np.float32)
    for r in [int(v) for v in a.ranks.split(",")]:
        me, be = topology(x[fit & own & easy], r)
        mh, bh = topology(x[fit & own & ~easy], r)
        score = err(x, mh, bh) - err(x, me, be)
        print(f"{sig} rank {r}: fit on {np.sum(fit&own&easy)} easy, {np.sum(fit&own&~easy)} hard tokens")
        table(f"recon router, {sig}, rank {r}", score, test_own)
        if a.out and r == 16:
            np.savez(f"{a.out}.{sig}.npz", me=me.cpu().numpy(), be=be.cpu().numpy(), mh=mh.cpu().numpy(), bh=bh.cpu().numpy())
prev = np.roll(d["phi_2"].astype(np.float32), 1, axis=0)
ok = np.roll(S, 1) == S
for r in (16,):
    me, be = topology(prev[fit & own & easy & ok], r)
    mh, bh = topology(prev[fit & own & ~easy & ok], r)
    table(f"recon router, the previous token's state, rank {r}", err(prev, mh, bh) - err(prev, me, be), test_own)
table("baseline: the previous token's top probability", np.roll(d["p2"], 1), test_own)
table("ceiling: top probability after one loop (needs the output block and the head)", d["p1"], test_own)
