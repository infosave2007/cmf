#!/usr/bin/env python3
"""What the loops change, token by token. Over each teacher-forced sequence, one pass
through four loops gives the logits of 1, 2 and 4 loops (per-loop caches) and the
states a router could look at. loop_probe.py HF_DIR SAMPLES OUT [WIKI_IDS N L]"""
import json, math, sys
import numpy as np
import torch
sys.path.insert(0, "/root/dory")
from ref_dory import Dory, justnorm

hf, samples, out = sys.argv[1:4]
m = Dory(hf, "cuda", torch.float32)
S = json.load(open(samples))
if len(sys.argv) > 4:
    ids = [int(x) for x in open(sys.argv[4]).read().split()]
    n, ln = int(sys.argv[5]), int(sys.argv[6])
    step = (len(ids) - ln - 1) // n
    S += [{"kind": "wiki", "ids": ids[w * step: w * step + ln + 1], "prompt_len": 1} for w in range(n)]
KS = (1, 2, 4)
rows = {k: [] for k in ("sample", "pos", "answer", "t1", "t2", "t4", "p1", "p2", "p4", "e1", "e2", "e4", "n1", "n2", "n4")}
phi_in, phi_1, phi_2 = [], [], []
n_in, n_rec = m.n_in, m.n_rec
for si, smp in enumerate(S):
    x = torch.tensor(smp["ids"], device="cuda")
    T = len(smp["ids"]) - 1
    caches = {}
    with torch.no_grad():
        for b in range(0, T, 64):
            e = min(b + 64, T)
            h = m.w["backbone.embeddings.weight"][x[b:e]]
            for li in range(n_in):
                h = m.layer(li, h, b, caches, (li, 0))
            inp = h
            state = inp
            lg = {}
            sts = []
            for i in range(4):
                h = justnorm(state + inp)
                for li in range(n_in, n_in + n_rec):
                    h = m.layer(li, h, b, caches, (li, i))
                state = justnorm(h)
                sts.append(state)
                K = i + 1
                if K in KS:
                    o = state
                    for li in range(n_in + n_rec, m.n):
                        o = m.layer(li, o, b, caches, (li, 100 + K))
                    lg[K] = (o @ m.w["lm_head.weight"].T).float() * (m.w["lm_head.logit_scale"].float() / m.std)
            y = x[b + 1:e + 1]
            rows["sample"].append(np.full(e - b, si))
            rows["pos"].append(np.arange(b, e))
            rows["answer"].append(np.arange(b, e) >= smp["prompt_len"] - 1)
            for K in KS:
                lp = torch.log_softmax(lg[K], -1)
                p, t = lp.max(-1)
                rows[f"t{K}"].append(t.cpu().numpy())
                rows[f"p{K}"].append(p.exp().cpu().numpy())
                rows[f"e{K}"].append((-(lp.exp() * lp).sum(-1)).cpu().numpy())
                rows[f"n{K}"].append((-lp.gather(1, y[:, None])[:, 0]).cpu().numpy())
            phi_in.append(inp.float().half().cpu().numpy())
            phi_1.append(sts[0].float().half().cpu().numpy())
            phi_2.append(sts[1].float().half().cpu().numpy())
    print(si, smp["kind"], T, flush=True)
np.savez(out, phi_in=np.concatenate(phi_in), phi_1=np.concatenate(phi_1), phi_2=np.concatenate(phi_2),
         kinds=np.array([s["kind"] for s in S]), **{k: np.concatenate(v) for k, v in rows.items()})
print("saved", out)
