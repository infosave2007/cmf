#!/usr/bin/env python3
"""Row features of a generic model's far-field seal over prompts with a planted fact, read from
the engine, and the fit. q27_feat.py CORTIQ MODEL TOKENIZER_JSON WIKI OUT_WEIGHTS [N] [CHARS]"""
import os, random, subprocess, sys
import numpy as np
from tokenizers import Tokenizer

cortiq, model, tokj, wiki, out = sys.argv[1:6]
n = int(sys.argv[6]) if len(sys.argv) > 6 else 20
chars = int(sys.argv[7]) if len(sys.argv) > 7 else 20000
tok = Tokenizer.from_file(tokj)
text = open(wiki, encoding="utf-8").read().replace("\r", "")
random.seed(31)
KINDS = [
    lambda k: (f"Remember this pass key: {k}. It is important.", k),
    lambda k: (f"Note: the access code for the archive is {k}.", k),
    lambda k: (f"The password of the staging server is qX{k}vT.", k),
    lambda k: (f"The meeting with Ilmari Voskresky was moved to room {k}.", k),
    lambda k: (f"Deploy with the token sk-{k}a{k[::-1]} and nothing else.", k),
]
env = dict(os.environ, XDG_RUNTIME_DIR="/tmp", CMF_O1="all", CMF_O1_GPU="1", CMF_O1_WINDOW="1800", CMF_O1_SINK="4",
           CMF_O1_EXACT="none:0", CMF_DORY_EXACT_FEATURES="/tmp/f.bin", CMF_O1_TOKENS_DUMP="/tmp/ids.bin", CMF_MAX_SEQ="70000")
Xs, Ys, Ks = [], [], []
for s in range(n):
    key = "".join(random.choice("0123456789") for _ in range(6))
    at = random.randrange(0, len(text) - chars - 1)
    fill = text[at:at + chars]
    note, fact = KINDS[s % len(KINDS)](key)
    cut = random.randrange(chars // 10, chars // 2)
    msg = f"{fill[:cut]}\n\n{note}\n\n{fill[cut:]}\n\nWhat was noted above? Reply briefly."
    for f in ("/tmp/f.bin", "/tmp/ids.bin"):
        if os.path.exists(f):
            os.remove(f)
    r = subprocess.run([cortiq, "run", model, "--prompt", msg, "--greedy", "--no-think", "--max-tokens", "1"], capture_output=True, text=True, env=env, timeout=900)
    if not os.path.exists("/tmp/f.bin"):
        print("no features", s, r.stderr[-400:], flush=True)
        continue
    ids = np.fromfile("/tmp/ids.bin", dtype=np.uint32)
    F = np.fromfile("/tmp/f.bin", dtype=np.float32)
    # Rows of the fact: where the decoded prompt holds it.
    pieces = [tok.decode([int(i)], skip_special_tokens=False) for i in ids]
    pos, c0 = [], 0
    for p in pieces:
        pos.append(c0)
        c0 += len(p)
    joined = "".join(pieces)
    i0 = joined.find(fact)
    if i0 < 0:
        print("fact not found", s, flush=True)
        continue
    rows = [j for j in range(len(pos)) if pos[j] < i0 + len(fact) and pos[j] + len(pieces[j]) > i0]
    nrows = F.shape[0] // 128 if F.shape[0] % 128 == 0 else None
    X = F.reshape(nrows, 128)
    y = np.zeros(nrows, dtype=bool)
    y[[j for j in rows if j < nrows]] = True
    Xs.append(X)
    Ys.append(y)
    Ks.append(s % len(KINDS))
    print(s, "kind", s % len(KINDS), "prompt", len(ids), "scored", nrows, "fact rows", rows[:1], len(rows), flush=True)


def score(sv, y):
    r = np.argsort(np.argsort(-sv)) / len(sv) * 100
    return r[y].min(), r[y].mean(), r[y].max()


def fit(train):
    pos = np.concatenate([Xs[i][Ys[i]] for i in train])
    neg = np.concatenate([Xs[i][~Ys[i]][::7] for i in train])
    return np.linalg.solve(np.cov(neg.T) + 1e-2 * np.eye(pos.shape[1]), pos.mean(0) - neg.mean(0))


m = len(Xs)
loo = np.array([score(Xs[i] @ fit([j for j in range(m) if j != i]), Ys[i]) for i in range(m)])
print(f"linear, leave one prompt out: best {loo[:,0].mean():.2f} mean {loo[:,1].mean():.2f} worst {loo[:,2].mean():.2f}")
for k in sorted(set(Ks)):
    te = [i for i in range(m) if Ks[i] == k]
    w = fit([i for i in range(m) if Ks[i] != k])
    a = np.array([score(Xs[i] @ w, Ys[i]) for i in te]).mean(0)
    print(f"  kind {k} left out: best {a[0]:.2f} mean {a[1]:.2f} worst {a[2]:.2f}")
w = fit(list(range(m)))
np.concatenate([[float(len(w))], w]).astype(np.float32).tofile(out)
print("weights", len(w), "written to", out)
