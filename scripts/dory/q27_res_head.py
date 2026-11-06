#!/usr/bin/env python3
"""A draft head by resonance (cmf-decision's mechanism) on Qwen3.8-27B Turbo.

    q27_res_head.py ROOT [STAGES=base,global,anchor,train] [R=256]

ROOT holds `dump/` (`cortiq mtp-heal --dump`, chain 3) and the trunk caches `tc/`
of the same windows. The decision is cmf-decision's: one isolated affine topology
per label (the mean and the top K principal directions of the label's own rows,
`fit.rs`), the label with the smallest reconstruction error, a gate on
p = softmax(−E/T) of the winner. What changes is the signal and what a label is:

  global   a label is the next token, the signal is the trunk's row and the
           anchor token's embedding, or the draft block's output row
  anchor   a skill per anchor token: its labels are the tokens seen after it,
           the signal is the trunk's row (the context that picks among them)
  train    the same topologies trained on the trunk's own tokens (cross-entropy
           on −E/T), in place of the vocabulary head over the block's output

Rows are the ones a round can start at: the token after the row is the trunk's
own. Windows with index % 10 == 9 are held out, % 10 == 8 calibrate the gate.
Every number is the share of held-out rows where the drafted token is the trunk's.
"""
import math
import sys
import time

import numpy as np
import torch

root = sys.argv[1]
stages = (sys.argv[2] if len(sys.argv) > 2 else "base,global,anchor,train").split(",")
R = int(sys.argv[3]) if len(sys.argv) > 3 else 256
dev = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(0)
T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


t, hs, vocab = 2048, 5120, 248320
sets = [("tc/q27-code-t27-bcbe83680bddd9bd", 196), ("tc/own-1dd426a4a343be5b", 58)]
W = sum(n for _, n in sets)
ids = np.fromfile(f"{root}/dump/ids.u32", dtype=np.uint32).reshape(W, t).astype(np.int64)
lens = np.fromfile(f"{root}/dump/len.u32", dtype=np.uint32).astype(np.int64)
own = np.array(np.memmap(f"{root}/dump/ti.u32", dtype=np.uint32, mode="r", shape=(W, t, 64))[:, :, 0]).astype(np.int64)
hmap = [np.memmap(f"{root}/{d}/h.f32", dtype=np.float32, mode="r", shape=(n, t, hs)) for d, n in sets]
first = np.cumsum([0] + [n for _, n in sets])


def hrow(w):
    s = int(np.searchsorted(first, w, side="right") - 1)
    return hmap[s][w - first[s]]


def chain(j):
    z = np.memmap(f"{root}/dump/before/x{j}.f16", dtype=np.float16, mode="r", shape=(W, t, hs))
    d = np.array(np.memmap(f"{root}/dump/before/d{j}.u32", dtype=np.uint32, mode="r", shape=(W, t, 4))[:, :, 0]).astype(np.int64)
    return z, d


# rows of step j: row p drafts token p + j + 1 from the anchor p + j; every token fed is the trunk's own
def rows_of(j):
    ok = np.zeros((W, t), dtype=bool)
    for w in range(W):
        n = int(lens[w])
        p = np.arange(max(n - j - 1, 0))
        on = np.ones(len(p), dtype=bool)
        for i in range(1, j + 1):
            on &= ids[w, p + i] == own[w, p + i - 1]
        ok[w, : len(p)] = on
    return ok


split = np.arange(W) % 10
embed = torch.from_numpy(np.fromfile(f"{root}/dump/embed.f16", dtype=np.float16).reshape(vocab, hs)).to(dev)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def reduce_fit(get, wins, r, cap=120000):
    """A label-free map: the mean and the top r principal directions of unit rows."""
    acc, s, n = torch.zeros(hs, hs, dtype=torch.float64, device=dev), torch.zeros(hs, dtype=torch.float64, device=dev), 0
    per = max(cap // len(wins), 64)
    for w in wins:
        x = unit(torch.from_numpy(np.ascontiguousarray(get(w)[: int(lens[w])][:: max(int(lens[w]) // per, 1)])).to(dev).float()).double()
        acc += x.T @ x
        s += x.sum(0)
        n += x.shape[0]
    mu = s / n
    cov = acc / n - torch.outer(mu, mu)
    val, vec = torch.linalg.eigh(cov)
    return mu.float(), vec[:, -r:].flip(1).float().contiguous(), float(val[-r:].sum() / val.sum())


def reduce(get, mu, basis, mask):
    """Unit rows of the masked positions through the map, unit again: [rows][r]."""
    out = []
    for w in range(W):
        p = np.nonzero(mask[w])[0]
        if len(p) == 0:
            continue
        x = unit(torch.from_numpy(np.ascontiguousarray(get(w)[p])).to(dev).float())
        out.append(unit((x - mu) @ basis))
    return torch.cat(out)


def fit(x, k):
    """fit.rs: the mean and the top k principal directions of a label's rows."""
    mu = x.mean(0)
    if x.shape[0] < 2 or k == 0:
        return mu, x.new_zeros(0, x.shape[1])
    _, s, vt = torch.linalg.svd(x - mu, full_matrices=False)
    keep = min(k, x.shape[0] - 1, int((s * s / x.shape[0] > 1e-8).sum()))
    return mu, vt[:keep]


def errors(x, mu, basis, live):
    """E [rows][labels]: mu [labels][r], basis [labels][k][r] (zero rows past a label's rank)."""
    d = (x * x).sum(-1, keepdim=True) - 2 * x @ mu.T + (mu * mu).sum(-1)[None, :]
    c = torch.einsum("br,lkr->blk", x, basis) - torch.einsum("lr,lkr->lk", mu, basis)[None]
    return (d - (c * c).sum(-1)).clamp_min(0)


def topologies(x, y, k, least, cap=1024):
    """Per label of y with at least `least` rows: (labels, mu, basis)."""
    order = torch.argsort(y, stable=True)
    ys, xs = y[order], x[order]
    lab, cnt = torch.unique_consecutive(ys, return_counts=True)
    start = torch.cumsum(cnt, 0) - cnt
    keep = [(int(l), int(s), int(n)) for l, s, n in zip(lab, start, cnt) if n >= least]
    mu = x.new_zeros(len(keep), x.shape[1])
    basis = x.new_zeros(len(keep), k, x.shape[1])
    for i, (_, s, n) in enumerate(keep):
        rows = xs[s : s + n]
        if n > cap:
            rows = rows[torch.randperm(n, device=dev)[:cap]]
        m, b = fit(rows, k)
        mu[i] = m
        basis[i, : b.shape[0]] = b
    return torch.tensor([l for l, _, _ in keep], device=dev), mu, basis


def decide(x, labels, mu, basis, bs=2048):
    """The label of the smallest error per row, the error and the margin to the runner-up."""
    win, e0, e1 = [], [], []
    for b in range(0, x.shape[0], bs):
        e = errors(x[b : b + bs], mu, basis, None)
        v, i = torch.topk(e, min(2, e.shape[1]), dim=1, largest=False)
        win.append(labels[i[:, 0]])
        e0.append(v[:, 0])
        e1.append(v[:, -1])
    return torch.cat(win), torch.cat(e0), torch.cat(e1)


def selective(conf, right, name, mtp_right=None):
    """Accuracy of the rows the gate accepts, by the share it accepts."""
    order = torch.argsort(conf, descending=True)
    r = right[order].float()
    cum = torch.cumsum(r, 0) / torch.arange(1, len(r) + 1, device=dev)
    parts = []
    for share in (0.1, 0.25, 0.5, 0.75, 1.0):
        n = max(int(len(r) * share), 1)
        s = f"{share:.2f}: {cum[n - 1]:.3f}"
        if mtp_right is not None:
            s += f" (block {mtp_right[order][:n].float().mean():.3f})"
        parts.append(s)
    say(f"    {name}: accepted share: accuracy  " + "   ".join(parts))


for j in (1, 2, 3):
    ok = rows_of(j)
    tr, ca, te = ok & (split[:, None] < 8), ok & (split[:, None] == 8), ok & (split[:, None] == 9)
    wi, pi = np.nonzero(ok)
    part = torch.from_numpy(split[wi]).to(dev)
    y = torch.from_numpy(own[wi, pi + j]).to(dev)  # the trunk's token after the anchor
    anchor = torch.from_numpy(ids[wi, pi + j]).to(dev)
    z, d = chain(j)
    block = torch.from_numpy(d[wi, pi]).to(dev)
    TR, CA, TE = part < 8, part == 8, part == 9
    say(f"== step {j}: {int(TR.sum())} train rows, {int(CA.sum())} calibration, {int(TE.sum())} held out; {len(torch.unique(y[TR]))} tokens drafted in train")
    say(f"  the draft block as it is: {(block[TE] == y[TE]).float().mean():.3f}")
    if "base" in stages:
        # the token seen most after the anchor: what no context gives
        pair = anchor[TR] * vocab + y[TR]
        u, c = torch.unique(pair, return_counts=True)
        best = {}
        for a, tok, n in zip((u // vocab).tolist(), (u % vocab).tolist(), c.tolist()):
            if n > best.get(a, (0, 0))[1]:
                best[a] = (tok, n)
        guess = torch.tensor([best.get(a, (-1, 0))[0] for a in anchor[TE].tolist()], device=dev)
        say(f"  the token seen most after the anchor: {(guess == y[TE]).float().mean():.3f} of all rows, an answer for {(guess >= 0).float().mean():.3f}")
    if j > 1 and "deep" not in stages:
        continue

    hmu, hb, hv = reduce_fit(hrow, np.nonzero(split < 8)[0], R)
    zmu, zb, zv = reduce_fit(lambda w: z[w], np.nonzero(split < 8)[0], R)
    say(f"  label-free maps to {R}: the trunk's rows keep {hv:.3f} of their variance, the block's {zv:.3f}")
    H = reduce(hrow, hmu, hb, ok)
    Z = reduce(lambda w: z[w], zmu, zb, ok)
    # the anchor's embedding through its own label-free map
    em = unit(embed.float())
    emu = em.mean(0)
    _, _, ev = torch.pca_lowrank(em - emu, q=64, niter=4)
    E = unit((em[anchor] - emu) @ ev)
    HE = torch.cat([H, 0.5 * E], 1)
    del em
    # the block's row through the vocabulary head's own strongest directions
    hw = torch.from_numpy(np.fromfile(f"{root}/dump/head.f16", dtype=np.float16).reshape(vocab, hs)).to(dev)
    g = torch.zeros(hs, hs, dtype=torch.float64, device=dev)
    for b in range(0, vocab, 16384):
        c = hw[b : b + 16384].double()
        g += c.T @ c
    wb = torch.linalg.eigh(g)[1][:, -R:].flip(1).float().contiguous()
    del hw, g
    ZW = reduce(lambda w: z[w], torch.zeros(hs, device=dev), wb, ok)
    HT = H[TE]

    if "global" in stages:
        for name, X in (("the trunk's row and the anchor's embedding", HE), ("the block's output row", Z), ("the block's row in the head's directions", ZW)):
            for k in (0, 4, 16):
                lab, mu, basis = topologies(X[TR], y[TR], k, 8)
                win, e0, e1 = decide(X[TE], lab, mu, basis)
                right = win == y[TE]
                cover = torch.isin(y[TE], lab).float().mean()
                say(f"  global, {name}, K={k}: {len(lab)} topologies hold {cover:.3f} of the rows; right on {right.float().mean():.3f} of all rows")
                if k == 16:
                    selective(e1 - e0, right, "gate by the margin of errors", block[TE] == y[TE])

    if "anchor" in stages:
        for k in (4, 16):
            for least in (2, 4):
                # a skill per anchor: its labels are the tokens that follow it
                pair = anchor[TR] * vocab + y[TR]
                lab, mu, basis = topologies(H[TR], pair, k, least)
                la = lab // vocab
                order = torch.argsort(la, stable=True)
                lab, mu, basis, la = lab[order], mu[order], basis[order], la[order]
                ua, cnt = torch.unique_consecutive(la, return_counts=True)
                start = torch.cumsum(cnt, 0) - cnt
                where = {int(a): (int(s), int(n)) for a, s, n in zip(ua, start, cnt)}
                xa, ya, ba = anchor[TE], y[TE], block[TE]
                win = torch.full_like(ya, -1)
                conf = torch.zeros(len(ya), device=dev)
                oa = torch.argsort(xa, stable=True)
                u2, c2 = torch.unique_consecutive(xa[oa], return_counts=True)
                s2 = torch.cumsum(c2, 0) - c2
                for a, s, n in zip(u2.tolist(), s2.tolist(), c2.tolist()):
                    if a not in where:
                        continue
                    ls, ln = where[a]
                    rows = oa[s : s + n]
                    e = errors(HT[rows], mu[ls : ls + ln], basis[ls : ls + ln], None)
                    v, i = torch.topk(e, min(2, ln), dim=1, largest=False)
                    win[rows] = lab[ls + i[:, 0]] % vocab
                    conf[rows] = (v[:, -1] - v[:, 0]) if ln > 1 else 1e3
                right = win == ya
                has = win >= 0
                say(f"  a skill per anchor, K={k}, {least}+ rows a label: {len(ua)} skills, {len(lab)} topologies; an answer for {has.float().mean():.3f} of the rows, right on {right[has].float().mean():.3f} of those, {right.float().mean():.3f} of all")
                both = torch.where(has, win, ba)
                say(f"    the block where no skill answers: {(both == ya).float().mean():.3f}; the block alone {(ba == ya).float().mean():.3f}; right where the block is wrong: {(right & (ba != ya)).float().mean():.3f} of all rows")
                selective(conf[has], right[has], "gate by the margin of errors", (ba == ya)[has])

    if "rerank" in stages:
        # cmf-decision's own task: the block names four options, the topologies choose among them
        cand = torch.from_numpy(np.array(np.memmap(f"{root}/dump/before/d{j}.u32", dtype=np.uint32, mode="r", shape=(W, t, 4)))[wi, pi].astype(np.int64)).to(dev)
        prob = torch.from_numpy(np.array(np.memmap(f"{root}/dump/before/dp{j}.f32", dtype=np.float32, mode="r", shape=(W, t, 4)))[wi, pi]).to(dev)
        say(f"  the trunk's token is among the block's four options on {(cand[TE] == y[TE][:, None]).any(1).float().mean():.3f} of the rows, first on {(cand[TE][:, 0] == y[TE]).float().mean():.3f}")
        for name, X in (("the trunk's row and the anchor's embedding", HE), ("the block's output row", Z)):
            for k in (16, 48):
                lab, mu, basis = topologies(X[TR], y[TR], k, 4)
                slot = torch.full((vocab,), -1, device=dev, dtype=torch.long)
                slot[lab] = torch.arange(len(lab), device=dev)

                def option_errors(m):
                    out = []
                    xs, cs = X[m], slot[cand[m]]
                    for b in range(0, xs.shape[0], 2048):
                        x, c = xs[b : b + 2048], cs[b : b + 2048]
                        i = c.clamp_min(0)
                        r = x[:, None, :] - mu[i]
                        proj = torch.einsum("bor,bokr->bok", r, basis[i])
                        e = ((r * r).sum(-1) - (proj * proj).sum(-1)).clamp_min(0)
                        out.append(torch.where(c >= 0, e, torch.full_like(e, float("nan"))))
                    return torch.cat(out)

                def choose(m, lam):
                    e = option_errors(m)
                    known = ~torch.isnan(e)
                    # an option with no topology keeps the block's order: it gets the row's mean error
                    fill = torch.nan_to_num(e, nan=0.0).sum(1, keepdim=True) / known.sum(1, keepdim=True).clamp_min(1)
                    e = torch.where(known, e, fill)
                    s = torch.log(prob[m].clamp_min(1e-9)) - lam * e
                    return cand[m].gather(1, s.argmax(1, keepdim=True))[:, 0]

                alone = choose(TE, 1e6)
                grid = [0.0, 0.5, 1, 2, 4, 8, 16, 32, 64]
                best = max(grid, key=lambda lam: (choose(CA, lam) == y[CA]).float().mean().item())
                both = choose(TE, best)
                say(f"  options by resonance, {name}, K={k}: the topologies alone {(alone == y[TE]).float().mean():.3f}; with the block's own probability (weight {best}, set on the calibration windows) {(both == y[TE]).float().mean():.3f}; the block {(cand[TE][:, 0] == y[TE]).float().mean():.3f}")

    if "train" in stages:
        torch.set_grad_enabled(True)
        for name, X, k, top in (("the block's row in the head's directions", ZW, 4, 8192), ("the block's output row", Z, 4, 8192), ("the trunk's row and the anchor's embedding", HE, 4, 8192)):
            freq = torch.bincount(y[TR], minlength=vocab)
            lab = torch.argsort(freq, descending=True)[:top]
            slot = torch.full((vocab,), -1, device=dev, dtype=torch.long)
            slot[lab] = torch.arange(top, device=dev)
            dim = X.shape[1]
            mu = torch.zeros(top, dim, device=dev)
            basis = torch.zeros(top, k, dim, device=dev)
            l0, m0, b0 = topologies(X[TR], y[TR], k, 2)
            mu[slot[l0][slot[l0] >= 0]] = m0[slot[l0] >= 0]
            basis[slot[l0][slot[l0] >= 0]] = b0[slot[l0] >= 0]
            basis += 1e-3 * torch.randn_like(basis)
            mu, basis = mu.requires_grad_(), basis.requires_grad_()
            logt = torch.tensor(math.log(0.05), device=dev, requires_grad=True)
            opt = torch.optim.Adam([{"params": [mu, basis], "lr": 2e-3}, {"params": [logt], "lr": 2e-2}])
            xt, yt = X[TR], slot[y[TR]]
            seen = yt >= 0
            xt, yt = xt[seen], yt[seen]

            def score(x):
                q = torch.linalg.qr(basis.transpose(1, 2))[0].transpose(1, 2)  # orthonormal rows
                return -errors(x, mu, q, None) / logt.exp()

            def held():
                with torch.no_grad():
                    out = torch.cat([score(X[TE][b : b + 2048]).argmax(1) for b in range(0, int(TE.sum()), 2048)])
                    conf = torch.cat([torch.softmax(score(X[TE][b : b + 2048]), 1).max(1)[0] for b in range(0, int(TE.sum()), 2048)])
                return lab[out], conf

            win, _ = held()
            say(f"  trained, {name}: {top} topologies of K={k} as fitted: {(win == y[TE]).float().mean():.3f} of all rows")
            for epoch in range(6):
                perm = torch.randperm(xt.shape[0], device=dev)
                for b in range(0, len(perm), 1024):
                    i = perm[b : b + 1024]
                    loss = torch.nn.functional.cross_entropy(score(xt[i]), yt[i])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                win, conf = held()
                say(f"    epoch {epoch + 1}: loss {loss.item():.3f}, T {logt.exp().item():.4f}, right on {(win == y[TE]).float().mean():.3f} of all rows (the block {(block[TE] == y[TE]).float().mean():.3f})")
            right = win == y[TE]
            say(f"    right where the block is wrong: {(right & (block[TE] != y[TE])).float().mean():.3f} of all rows; wrong where it is right: {(~right & (block[TE] == y[TE])).float().mean():.3f}")
            selective(conf, right, "gate by p of the winner", block[TE] == y[TE])
        torch.set_grad_enabled(False)
