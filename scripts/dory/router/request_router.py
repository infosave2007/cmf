#!/usr/bin/env python3
"""Can a request's loop count be routed before it runs? Patent-1 topologies over a
frozen signal: Embryo's φ of the prompt, or Dory's state at the prompt's last token.
Leave-one-out over 80 prompts: the label is whether one loop flips more of this
answer's tokens than the median."""
import json, sys
import numpy as np

d = np.load("router/probe.npz")
S = json.load(open("router/samples.json"))
L = json.load(open("router/per_request.json"))
hard = np.array([o["hard"] for o in L])
gain = np.array([o["gain"] for o in L])
lab = hard > np.median(hard)
kinds = np.array([s["kind"] for s in S])
emb = np.fromfile("router/embryo_phi.f32", dtype="<f4").reshape(80, -1)
sample = d["sample"]
phi_in = d["phi_in"].astype(np.float32)
pos = d["pos"]
dory = np.zeros((80, phi_in.shape[1]), dtype=np.float32)
for i, s in enumerate(S):
    m = np.where((sample == i) & (pos == s["prompt_len"] - 1))[0]
    dory[i] = phi_in[m[0]]
dory /= np.linalg.norm(dory, axis=1, keepdims=True)


def topo(x, r):
    mu = x.mean(0)
    u, s, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[: min(r, vt.shape[0])]


def err(x, mu, b):
    y = x - mu
    return (y * y).sum(-1) - ((y @ b.T) ** 2).sum(-1)


def auc(s, y):
    p, n = s[y], s[~y]
    return float(np.mean(p[:, None] > n[None, :]))


for name, x in (("embryo", emb), ("dory-input-block", dory)):
    for r in (2, 4, 8):
        sc = np.zeros(80)
        for i in range(80):
            keep = np.arange(80) != i
            me, be = topo(x[keep & ~lab], r)
            mh, bh = topo(x[keep & lab], r)
            sc[i] = err(x[i], me, be) - err(x[i], mh, bh)
        print(f"{name:18s} rank {r}: request 'needs the second loop' AUC {auc(sc, lab):.3f}")
    # type of request, argmin over one topology per kind
    uk = sorted(set(kinds.tolist()))
    for r in (2, 4, 8):
        ok = 0
        for i in range(80):
            keep = np.arange(80) != i
            errs = []
            for k in uk:
                mu, b = topo(x[keep & (kinds == k)], r)
                errs.append(err(x[i], mu, b))
            ok += uk[int(np.argmin(errs))] == kinds[i]
        print(f"{name:18s} rank {r}: type of request {ok}/80")
for k in sorted(set(kinds.tolist())):
    print(f"{k:9s} one-loop flips {hard[kinds==k].mean()*100:.1f} %  nll gain of the second loop {gain[kinds==k].mean():.3f}")
# What it buys: half the requests on one loop, the half the router calls easy.
for name, x, r in (("embryo", emb, 4), ("dory-input-block", dory, 2)):
    sc = np.zeros(80)
    for i in range(80):
        keep = np.arange(80) != i
        me, be = topo(x[keep & ~lab], r)
        mh, bh = topo(x[keep & lab], r)
        sc[i] = err(x[i], me, be) - err(x[i], mh, bh)
    order = np.argsort(sc)
    for share in (0.25, 0.5, 0.75):
        easy = order[: int(80 * share)]
        print(f"{name:18s} {share:.2f} of requests on one loop: flips on them {hard[easy].mean()*100:.2f} % (all {hard.mean()*100:.2f} %), second-loop gain lost {gain[easy].mean():.4f} (all {gain.mean():.4f})")
