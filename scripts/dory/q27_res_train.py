#!/usr/bin/env python3
"""Resonance topologies trained in place of the draft head of Qwen3.8-27B Turbo.

    q27_res_train.py ROOT MODE [TOP=16384] [K=0] [EPOCHS=6] [DEPTH=3]

ROOT holds `dump/` (`cortiq mtp-heal --dump`, chain 3) and the trunk caches `tc/`
of the same windows. A label is one of the TOP tokens the trunk drafts most. Its
topology is cmf-decision's: a centre mu and K directions U, the error

    E(x) = |x - mu|^2 - gamma * |U (x - mu)|^2,     answer = the smallest E / T

at the full width of the signal, no reduction. What is new against the fitted
topologies of `q27_res_head.py` is that the centres, the directions, gamma and a
bias a label are then trained on the trunk's own tokens (cross-entropy on -E/T),
so the topologies are no longer isolated: the report must call this variant
"trained". At K = 0 a trained topology is a linear head with a bias.

MODE
  head   in place of the vocabulary head. The signal is the draft block's output
         row after the block's own norm. A topology starts as the head's own row
         (mu = T*s/2 * w, bias |mu|^2/T), so before training the answers are the
         block's; tokens outside TOP keep the head's row, frozen.
  free   in place of the whole draft block. The signal is the trunk's row and the
         embeddings of the tokens fed so far, unit length each. A topology starts
         as the mean and the top K directions of its label's own rows (fit.rs).

Rows are the ones a round can reach: every token fed is the trunk's own. Windows
with index % 10 == 9 are held out, % 10 == 8 pick the epoch. The last block walks
the held-out windows as rounds in order, each starting after what the one before
kept, for the trained topologies and for the draft block as it is.

Environment: SAVE=DIR keeps the topologies of the picked epoch, LOAD=DIR measures
kept ones with no training and every window of ROOT held out (text the topologies
never met), SETS="tc/NAME:WINDOWS,..." names ROOT's trunk caches in the dump's order.
"""
import math
import os
import sys
import time

import numpy as np
import torch

root, mode = sys.argv[1], sys.argv[2]
TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
K = int(sys.argv[4]) if len(sys.argv) > 4 else 0
EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 6
DEPTH = int(sys.argv[6]) if len(sys.argv) > 6 else 3
LR = float(os.environ.get("LR", "0.003" if mode == "head" else "0.01"))
BS = 2048 if mode == "head" else 4096
LOAD, SAVE = os.environ.get("LOAD"), os.environ.get("SAVE")
if LOAD:
    EPOCHS = 0
dev = "cuda"
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


t, hs, vocab = 2048, 5120, 248320
sets = [(a, int(n)) for a, n in (s.split(":") for s in os.environ.get("SETS", "tc/q27-code-t27-bcbe83680bddd9bd:196,tc/own-1dd426a4a343be5b:58").split(","))]
W = sum(n for _, n in sets)
ids = np.fromfile(f"{root}/dump/ids.u32", dtype=np.uint32).reshape(W, t).astype(np.int64)
lens = np.fromfile(f"{root}/dump/len.u32", dtype=np.uint32).astype(np.int64)
own = np.array(np.memmap(f"{root}/dump/ti.u32", dtype=np.uint32, mode="r", shape=(W, t, 64))[:, :, 0]).astype(np.int64)
hmap = [np.memmap(f"{root}/{d}/h.f32", dtype=np.float32, mode="r", shape=(n, t, hs)) for d, n in sets]
first = np.cumsum([0] + [n for _, n in sets])
split = np.full(W, 9) if LOAD else np.arange(W) % 10


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


def load_rows(get, wi, pi):
    """The rows (wi, pi) as f16 on the device; wi is sorted."""
    out = torch.empty(len(wi), hs, dtype=torch.float16, device=dev)
    edge = np.searchsorted(wi, np.arange(W + 1))
    for w in range(W):
        a, b = int(edge[w]), int(edge[w + 1])
        if b > a:
            out[a:b] = torch.from_numpy(np.ascontiguousarray(get(w)[pi[a:b]])).to(dev).half()
    return out


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def in_order(pred, oks, lo=0, hi=None):
    """Rounds in order over the held-out windows lo..hi: per depth the share kept and the rows it was asked on."""
    depth = len(pred)
    hit, tot = np.zeros(depth), np.zeros(depth)
    for w in np.nonzero(split == 9)[0]:
        if w < lo or (hi is not None and w >= hi):
            continue
        n, end = 0, int(lens[w])
        while n < end - 2:
            if ids[w, n + 1] != own[w, n]:
                n += 1
                continue
            kept = 0
            for j in range(1, depth + 1):
                if not oks[j - 1][w, n]:
                    break
                tot[j - 1] += 1
                if pred[j - 1][w, n] != own[w, n + j]:
                    break
                hit[j - 1] += 1
                kept += 1
            n += kept + 1
    rate = hit / np.maximum(tot, 1)
    return " / ".join(f"{r:.3f}" for r in rate) + f" (of {' / '.join(str(int(n)) for n in tot)} rows), {float(np.cumprod(rate).sum()):.2f} drafts a round"


def selective(conf, right, name, other):
    order = torch.argsort(conf, descending=True)
    r = right[order].float()
    cum = torch.cumsum(r, 0) / torch.arange(1, len(r) + 1, device=dev)
    parts = []
    for share in (0.1, 0.25, 0.5, 0.75, 1.0):
        n = max(int(len(r) * share), 1)
        parts.append(f"{share:.2f}: {cum[n - 1]:.3f} (block {other[order][:n].float().mean():.3f})")
    say(f"    {name}: accepted share: accuracy  " + "   ".join(parts))


embed = torch.from_numpy(np.fromfile(f"{root}/dump/embed.f16", dtype=np.float16).reshape(vocab, hs)).to(dev)
if mode == "head":
    head = torch.from_numpy(np.fromfile(f"{root}/dump/head.f16", dtype=np.float16).reshape(vocab, hs)).to(dev)
    gain = {n: torch.from_numpy(np.fromfile(f"{root}/dump/{n}.f32", dtype=np.float32)).to(dev) for n in ("mtp_norm", "norm")}
    s0 = math.sqrt(hs)

say(f"{mode}: {TOP} topologies of K={K}, {EPOCHS} epochs, {DEPTH} deep, step size {LR} of a tensor's scale")
preds, blocks, oks = [], [], []
for j in range(1, DEPTH + 1):
    ok = rows_of(j)
    wi, pi = np.nonzero(ok)
    part = torch.from_numpy(split[wi]).to(dev)
    y = torch.from_numpy(own[wi, pi + j]).to(dev)  # the trunk's token after the anchor
    z, d = chain(j)
    block = torch.from_numpy(d[wi, pi]).to(dev)
    TR, CA, TE = (torch.nonzero(m)[:, 0] for m in (part < 8, part == 8, part == 9))
    if LOAD:
        kept = torch.load(f"{LOAD}/{mode}-k{K}-s{j}.pt", map_location=dev)
        lab = kept["lab"]
    else:
        freq = torch.bincount(y[TR], minlength=vocab)
        lab = torch.argsort(freq, descending=True)[:TOP]
        lab = lab[freq[lab] > 0]
    L = len(lab)
    slot = torch.full((vocab,), -1, device=dev, dtype=torch.long)
    slot[lab] = torch.arange(L, device=dev)
    say(f"== step {j}: {len(TR)} train rows, {len(CA)} calibration, {len(TE)} held out; {L} topologies hold {(slot[y[TE]] >= 0).float().mean():.3f} of the held-out rows; the draft block as it is {(block[TE] == y[TE]).float().mean():.3f}")

    if mode == "head":
        rows = load_rows(lambda w: z[w], wi, pi)
        # which norm the block's row meets before the head: the one whose answers are the block's
        probe = TE[:: max(len(TE) // 4096, 1)]
        best = None
        for name, gn in gain.items():
            x = rows[probe].float()
            n = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * (1 + gn)
            same = ((n.half() @ head.T).argmax(1) == block[probe]).float().mean().item()
            if best is None or same > best[1]:
                best = (name, same, gn)
        say(f"  the head over `{best[0]}` of the block's row gives the block's own answer on {best[1]:.4f} of the rows")
        if best[1] < 0.98:
            sys.exit("the dump's rows do not reproduce the block's answers: stop")
        gn = best[2]
        D = hs

        def normed(i):
            x = rows[i].float()
            return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * (1 + gn)

        def sig(i):
            return normed(i) / s0

        if not LOAD:
            wl = head[lab].float()
            temp = 2.0 / (s0 * wl.norm(dim=1).mean().item())
            MU = (temp * s0 / 2) * wl
            B = (MU * MU).sum(-1) / temp
            fitted_gamma = 0.0
    else:
        rows = load_rows(hrow, wi, pi)
        fed = [torch.from_numpy(ids[wi, pi + i]).to(dev) for i in range(1, j + 1)]
        D = hs * (j + 1)

        def sig(i):
            return torch.cat([unit(rows[i].float())] + [unit(embed[f[i]].float()) for f in fed], 1)

        if not LOAD:
            MU = torch.zeros(L, D, device=dev)
            for b in range(0, len(TR), BS):
                i = TR[b : b + BS]
                s = slot[y[i]]
                MU.index_add_(0, s[s >= 0], sig(i)[s >= 0])
            MU /= freq[lab].float()[:, None]
            B = torch.zeros(L, device=dev)
            fitted_gamma = 1.0

    if LOAD:
        MU, B, GAM, U, temp = kept["mu"].float(), kept["b"], kept["gamma"], kept["u"].float() if K else None, kept["t"]
        del kept
    else:
        U = unit(torch.randn(L, K, D, device=dev)) if K else None
        GAM = torch.zeros(L, device=dev)
    if K and not LOAD:
        # fit.rs: the top K directions of a label's own rows about its centre (gram-eigh)
        s = slot[y[TR]]
        order = torch.argsort(s, stable=True)
        order = order[s[order] >= 0]
        cnt = torch.bincount(s[order], minlength=L)
        start = torch.cumsum(cnt, 0) - cnt
        fitted = 0
        for c in torch.nonzero(cnt >= 8)[:, 0].tolist():
            r = order[start[c] : start[c] + cnt[c]][:512]
            x = sig(TR[r]) - MU[c]
            val, vec = torch.linalg.eigh(x @ x.T)
            k = min(K, int((val > 1e-8 * val[-1]).sum()))
            U[c, :k] = unit(vec[:, -k:].flip(1).T @ x)
            GAM[c] = fitted_gamma
            fitted += 1
        say(f"  {fitted} topologies have directions fitted from 8+ rows of their own")
    if mode == "free" and not LOAD:
        # T as cmf-decision certifies it: the one the calibration rows' answers are likeliest under
        i = CA[:: max(len(CA) // 8192, 1)]
        x, s = sig(i), slot[y[i]]
        e = (x * x).sum(-1, keepdim=True) - 2 * x @ MU.T + (MU * MU).sum(-1)
        if K:
            c = (x @ U.reshape(L * K, D).T).view(-1, L, K) - (U * MU[:, None, :]).sum(-1)
            e = e - GAM * (c * c).sum(-1)
        grid = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        temp = min(grid, key=lambda v: torch.nn.functional.cross_entropy(-e[s >= 0] / v, s[s >= 0]).item())
        del e
    logt = torch.tensor(math.log(temp), device=dev)
    say(f"  T {temp:.4f}; {L * D * (1 + K) / 2**20:.0f} MiB at a byte a weight")

    train = [MU, B, GAM] + ([U] if K else []) + ([logt] if mode == "free" else [])
    for p in train:
        p.requires_grad_()
    rms = lambda p: p.detach().pow(2).mean().sqrt().item()
    groups = [{"params": [MU], "lr": LR * rms(MU)}, {"params": [B], "lr": 0.02}, {"params": [GAM], "lr": 0.02}]
    if K:
        groups.append({"params": [U], "lr": LR * rms(U)})
    if mode == "free":
        groups.append({"params": [logt], "lr": 0.01})
    opt = torch.optim.Adam(groups)
    base = [grp["lr"] for grp in groups]

    def score(x):
        temp = logt.exp()
        s = (2 * x @ MU.T - (MU * MU).sum(-1)) / temp + B
        if K:
            c = (x @ U.reshape(L * K, D).T).view(-1, L, K) - (U * MU[:, None, :]).sum(-1)
            s = s + GAM.clamp(0, 1) * (c * c).sum(-1) / temp
        return s

    def logits(i):
        s = score(sig(i))
        if mode == "free":
            return s
        with torch.no_grad():
            rest = (normed(i).half() @ head.T).float()
        return rest.index_copy(1, lab, s)

    def answers(idx):
        out, conf = [], []
        with torch.no_grad():
            for b in range(0, len(idx), BS):
                p = torch.softmax(logits(idx[b : b + BS]), 1).max(1)
                out.append(p.indices if mode == "head" else lab[p.indices])
                conf.append(p.values)
        return torch.cat(out), torch.cat(conf)

    win, conf = answers(TE)
    say(f"  {'the kept topologies' if LOAD else 'before training'}: right on {(win == y[TE]).float().mean():.3f} of the held-out rows, the block's own answer on {(win == block[TE]).float().mean():.3f}")
    best = (-1.0 if LOAD else (answers(CA)[0] == y[CA]).float().mean().item(), 0, win, conf)

    def keep():
        if SAVE:
            os.makedirs(SAVE, exist_ok=True)
            torch.save({"lab": lab, "mu": MU.detach().half(), "b": B.detach(), "gamma": GAM.detach().clamp(0, 1), "u": U.detach().half() if K else None, "t": logt.exp().item()}, f"{SAVE}/{mode}-k{K}-s{j}.pt")

    steps, step = EPOCHS * math.ceil(len(TR) / BS), 0
    for epoch in range(EPOCHS):
        perm = TR[torch.randperm(len(TR), device=dev)]
        total, seen = 0.0, 0
        for b in range(0, len(perm), BS):
            i = perm[b : b + BS]
            target = y[i] if mode == "head" else slot[y[i]]
            i, target = i[target >= 0], target[target >= 0]
            for grp, lr in zip(opt.param_groups, base):
                grp["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
            loss = torch.nn.functional.cross_entropy(logits(i), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            total += loss.item() * len(i)
            seen += len(i)
        if K:
            with torch.no_grad():
                U.copy_(torch.linalg.qr(U.transpose(1, 2))[0].transpose(1, 2))
        ca = (answers(CA)[0] == y[CA]).float().mean().item()
        win, conf = answers(TE)
        say(f"    epoch {epoch + 1}: loss {total / seen:.3f}, T {logt.exp().item():.4f}, gamma {GAM.clamp(0, 1).mean().item():.3f}; calibration {ca:.3f}, held out {(win == y[TE]).float().mean():.3f}")
        if ca > best[0]:
            best = (ca, epoch + 1, win, conf)
            keep()
    _, epoch, win, conf = best
    right, was = win == y[TE], block[TE] == y[TE]
    say(f"  step {j}, the epoch the calibration windows pick ({epoch}): right on {right.float().mean():.3f} of the held-out rows, the block {was.float().mean():.3f}; right where the block is wrong {(right & ~was).float().mean():.3f}, wrong where it is right {(~right & was).float().mean():.3f}")
    selective(conf, right, "gate by p of the winner", was)

    te = TE.cpu().numpy()
    for src, into in ((win, preds), (block[TE], blocks)):
        a = np.full((W, t), -1, dtype=np.int64)
        a[wi[te], pi[te]] = src.cpu().numpy()
        into.append(a)
    oks.append(ok)
    for p in train:
        p.requires_grad_(False)
    del rows, MU, U, B, GAM, opt, best, win, conf
    torch.cuda.empty_cache()

say(f"rounds in order, each starting after what the one before kept, {DEPTH} deep:")
say(f"  the trained topologies: {in_order(preds, oks)}")
say(f"  the draft block as it is: {in_order(blocks, oks)}")
if len(sets) > 1:
    for i, (name, _) in enumerate(sets):
        name = name.split("/")[-1].rsplit("-", 1)[0]
        say(f"    {name}: the trained topologies {in_order(preds, oks, first[i], first[i + 1])}; the block {in_order(blocks, oks, first[i], first[i + 1])}")
