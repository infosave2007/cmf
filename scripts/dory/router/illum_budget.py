#!/usr/bin/env python3
"""From a rank-feature fit to a budget of exact rows: rows above a score, each with its
neighbours, on prompts the fit did not see. Writes the weights for `cortiq dory-turbo`.
illum_budget.py FEAT.npz WEIGHTS.bin"""
import sys
import numpy as np

d = np.load(sys.argv[1], allow_pickle=True)
Xs, ys, kinds, full = d["X"], d["y"], d["kind"], d["full"]
n = len(Xs)
names = []
for li in full:
    for what in ("res", "vnorm", "knorm"):
        names += [f"{what} L{li} h{h}" for h in range(4)]
names.append("surprise")


def pct(v):
    return np.argsort(np.argsort(v, axis=0), axis=0) / (v.shape[0] - 1)


P = [pct(x) for x in Xs]
Y = [y.astype(bool) for y in ys]


def fit(train, sel):
    pos = np.concatenate([P[i][Y[i]][:, sel] for i in train])
    neg = np.concatenate([P[i][~Y[i]][::7][:, sel] for i in train])
    return np.linalg.solve(np.cov(neg.T) + 1e-2 * np.eye(len(sel)), pos.mean(0) - neg.mean(0))


def pick(s, budget, dil):
    keep = np.zeros(len(s), dtype=bool)
    for r in np.argsort(-s):
        if keep.sum() >= budget:
            break
        keep[max(0, r - dil):r + dil + 1] = True
    return keep


SEL = {
    "vnorm": [i for i, x in enumerate(names) if x.startswith("vnorm")],
    "res": [i for i, x in enumerate(names) if x.startswith("res")],
    "res+vnorm": [i for i, x in enumerate(names) if x.startswith(("res", "vnorm"))],
}
for nm, sel in SEL.items():
    print(nm)
    S = []
    for i in range(n):
        w = fit([j for j in range(n) if j != i], sel)
        S.append(P[i][:, sel] @ w)
    for budget in (64, 128, 248, 512):
        for dil in (0, 2, 4):
            cov, whole = [], []
            for i in range(n):
                k = pick(S[i], budget, dil)
                cov.append(k[Y[i]].mean())
                whole.append(k[Y[i]].all())
            print(f"  budget {budget:4d} neighbours {dil}: fact rows kept {np.mean(cov)*100:5.1f} %, whole fact kept {np.mean(whole)*100:5.1f} %")
sel = SEL["res+vnorm"]
w = fit(list(range(n)), sel)
out = np.concatenate([[float(len(sel))], w]).astype(np.float32)
out.tofile(sys.argv[2])
print("weights", len(sel), "by layer, then res h0..3, vnorm h0..3; largest:")
for i in np.argsort(-np.abs(w))[:10]:
    print(f"  {w[i]:8.2f} {names[sel[i]]}")
