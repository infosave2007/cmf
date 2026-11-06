#!/usr/bin/env python3
"""Resonance beside the draft block of Qwen3.8-27B Turbo: a depth gate and a trajectory memory.

    q27_res_gate.py ROOT [STAGES=depth,memory] [R=256]

ROOT holds `dump/` (`cortiq mtp-heal --dump`, chain 3) and the trunk caches `tc/`
of the same windows. Both stages are cmf-decision's mechanism as it is: isolated
topologies fitted from their own rows (`fit.rs`: the mean and the top K principal
directions), the smallest reconstruction error, a gate; the signal is label-free
(unit rows through a PCA fitted without labels). Nothing is trained across labels.

  depth   Four labels: how many of the block's three drafts the trunk keeps from
          a round's first row (0..3). The decision picks how many drafts the
          round sends to the trunk. Compared with a fixed depth, with the block's
          own probability as the gate (thresholds set on the calibration windows)
          and with the depth known in advance.
  memory  A row of the train windows is a remembered trajectory: its key is the
          trunk's row with the anchor token, its value the tokens that followed.
          A round whose key is close enough to a remembered one drafts that
          continuation in place of the block. Also the same with only the
          window's own past as the memory (what a session could remember).

A round is scored in the M4's measured cost: a round of k drafts costs
COST[k] plain tokens and yields 1 + the drafts kept. Speed is tokens over cost,
1.00 being plain decode. Windows % 10 == 9 are held out, == 8 calibrate.
"""
import itertools
import sys
import time

import numpy as np
import torch

root = sys.argv[1]
stages = (sys.argv[2] if len(sys.argv) > 2 else "depth,memory").split(",")
R = int(sys.argv[3]) if len(sys.argv) > 3 else 256
COST = np.array([1.0, 1.10, 1.275, 1.54])  # plain tokens a round of k drafts costs on the M4 Air
J, M = 3, 8
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
split = np.arange(W) % 10


def hrow(w):
    s = int(np.searchsorted(first, w, side="right") - 1)
    return hmap[s][w - first[s]]


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


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def reduce_fit(get, wins, r, cap=120000):
    acc, s, n = torch.zeros(hs, hs, dtype=torch.float64, device=dev), torch.zeros(hs, dtype=torch.float64, device=dev), 0
    per = max(cap // len(wins), 64)
    for w in wins:
        x = unit(torch.from_numpy(np.array(get(w)[: int(lens[w])][:: max(int(lens[w]) // per, 1)])).to(dev).float()).double()
        acc += x.T @ x
        s += x.sum(0)
        n += x.shape[0]
    mu = s / n
    vec = torch.linalg.eigh(acc / n - torch.outer(mu, mu))[1]
    return mu.float(), vec[:, -r:].flip(1).float().contiguous()


def reduce(get, mu, basis, wi, pi):
    out = torch.empty(len(wi), basis.shape[1], device=dev)
    edge = np.searchsorted(wi, np.arange(W + 1))
    for w in range(W):
        a, b = int(edge[w]), int(edge[w + 1])
        if b > a:
            out[a:b] = unit((unit(torch.from_numpy(np.array(get(w)[pi[a:b]])).to(dev).float()) - mu) @ basis)
    return out


def shifted(a, j):
    out = np.full_like(a, -1)
    out[:, : t - j] = a[:, j:]
    return out


# what a round starting at a row keeps of the block's three drafts, and how deep the dump lets it be asked
oks = [rows_of(j) for j in range(1, M + 1)]
hit, prob = [], []
for j in range(1, J + 1):
    d = np.array(np.memmap(f"{root}/dump/before/d{j}.u32", dtype=np.uint32, mode="r", shape=(W, t, 4))[:, :, 0]).astype(np.int64)
    hit.append(d == shifted(own, j))
    prob.append(np.array(np.memmap(f"{root}/dump/before/dp{j}.f32", dtype=np.float32, mode="r", shape=(W, t, 4))[:, :, 0]))
kmax = np.zeros((W, t), dtype=np.int64)
kept = np.zeros((W, t), dtype=np.int64)
alive = oks[0].copy()
cut = np.zeros((W, t), dtype=bool)  # the text leaves the trunk's own tokens before the chain ends
for j in range(1, J + 1):
    can = alive & oks[j - 1]
    cut |= alive & ~oks[j - 1]
    kmax[can] = j
    alive = can & hit[j - 1]
    kept[alive] = j
start = oks[0]


def speed(policy, wins, kept=None):
    """Rounds in order over `wins`: tokens over cost for a per-row choice of depth."""
    kept = globals()["kept"] if kept is None else kept
    tokens = cost = rounds = drafts = 0.0
    for w in wins:
        n, end = 0, int(lens[w])
        while n < end - 2:
            if not start[w, n]:
                n += 1
                continue
            k = min(int(policy[w, n]), int(kmax[w, n]))
            got = min(int(kept[w, n]), k)
            tokens += 1 + got
            cost += COST[k]
            drafts += k
            rounds += 1
            n += got + 1
    return tokens / cost, tokens / rounds, drafts / rounds


def fit(x, k):
    mu = x.mean(0)
    if x.shape[0] < 2 or k == 0:
        return mu, x.new_zeros(0, x.shape[1])
    _, s, vt = torch.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[: min(k, x.shape[0] - 1)]


def error(x, mu, basis):
    r = x - mu
    return ((r * r).sum(-1) - ((r @ basis.T) ** 2).sum(-1)).clamp_min(0)


held, cal = np.nonzero(split == 9)[0], np.nonzero(split == 8)[0]
wi, pi = np.nonzero(start)
part = torch.from_numpy(split[wi]).to(dev)
TR, CA, TE = part < 8, part == 8, part == 9
anchor = torch.from_numpy(ids[wi, pi + 1]).to(dev)
hmu, hb = reduce_fit(hrow, np.nonzero(split < 8)[0], R)
H = reduce(hrow, hmu, hb, wi, pi)
embed = torch.from_numpy(np.fromfile(f"{root}/dump/embed.f16", dtype=np.float16).reshape(vocab, hs)).to(dev)
em = unit(embed.float())
emu = em.mean(0)
ev = torch.pca_lowrank(em - emu, q=64, niter=4)[2]
HE = torch.cat([H, 0.5 * unit((em[anchor] - emu) @ ev)], 1)
del em, embed
say(f"{len(wi)} rows a round can start at: {int(TR.sum())} train, {int(CA.sum())} calibration, {int(TE.sum())} held out; label-free signal of {HE.shape[1]}")

if "depth" in stages:
    z = np.memmap(f"{root}/dump/before/x1.f16", dtype=np.float16, mode="r", shape=(W, t, hs))
    zmu, zb = reduce_fit(lambda w: z[w], np.nonzero(split < 8)[0], R)
    Z = reduce(lambda w: z[w], zmu, zb, wi, pi)
    y = torch.from_numpy(kept[wi, pi]).to(dev)
    full = torch.from_numpy(~cut[wi, pi]).to(dev)  # rows whose depth the dump knows to the end
    share = [(y[TE & full] == c).float().mean().item() for c in range(J + 1)]
    say("== depth gate: drafts kept of three, share of the held-out rows: " + " / ".join(f"{c}: {s:.3f}" for c, s in enumerate(share)))

    def table(name, policy):
        s, tok, dr = speed(policy, held)
        say(f"  {name}: speed {s:.3f} of plain, {tok:.2f} tokens a round, {dr:.2f} drafts sent a round")

    for k in range(J + 1):
        table(f"always {k} drafts", np.full((W, t), k))
    table("the depth known in advance", kept)
    # the block's own probability: send draft j while every p up to j clears its threshold
    grid = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    def by_prob(th):
        k = np.zeros((W, t), dtype=np.int64)
        on = np.ones((W, t), dtype=bool)
        for j in range(J):
            on &= prob[j] >= th[j]
            k[on] = j + 1
        return k

    best = max(itertools.product(grid, repeat=J), key=lambda th: speed(by_prob(th), cal)[0])
    table(f"the block's own probability, thresholds {best} set on the calibration windows", by_prob(best))

    for name, X in (("the trunk's row and the anchor", HE), ("the block's first row", Z), ("both", torch.cat([HE, Z], 1))):
        for k in (4, 16, 64):
            tops = [fit(X[TR & full & (y == c)][torch.randperm(int((TR & full & (y == c)).sum()), device=dev)[:40000]], k) for c in range(J + 1)]
            E = torch.stack([error(X, mu, b) for mu, b in tops], 1)
            win = E.argmin(1)
            acc = (win[TE & full] == y[TE & full]).float().mean().item()
            policy = np.zeros((W, t), dtype=np.int64)
            policy[wi, pi] = win.cpu().numpy()
            # the same errors as a posterior: T as the calibration rows' answers are likeliest under, then
            # the depth with the best tokens minus rate * cost, the rate being the policy's own
            m = CA & full
            temp = min([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], key=lambda v: torch.nn.functional.cross_entropy(-E[m] / v, y[m]).item())
            post = torch.softmax(-E / temp, 1)
            reach = torch.flip(torch.cumsum(torch.flip(post, [1]), 1), [1])  # P(kept >= c)
            gain = torch.cumsum(reach, 1) - reach[:, :1] + 1  # tokens expected of sending c drafts
            rate = 1.5
            for _ in range(6):
                pick = (gain - rate * torch.from_numpy(COST).to(dev).float()).argmax(1)
                ev_policy = np.zeros((W, t), dtype=np.int64)
                ev_policy[wi, pi] = pick.cpu().numpy()
                rate = speed(ev_policy, cal)[0]
            say(f"  resonance, {name}, K={k}: the label right on {acc:.3f} of the held-out rows (the commonest label {max(share):.3f}), T {temp}")
            table("    the label as the depth", policy)
            table("    the depth of the best expected speed", ev_policy)

if "memory" in stages:
    cont = np.stack([shifted(ids, 2 + i) for i in range(M)], -1)[wi, pi]  # what followed the anchor
    truth = np.stack([shifted(own, 1 + i) for i in range(M)], -1)[wi, pi]
    valid = np.stack([oks[i][wi, pi] for i in range(M)], -1)
    cont, truth, valid = (torch.from_numpy(a).to(dev) for a in (cont, truth, valid))
    blk = torch.from_numpy(kept[wi, pi]).to(dev)
    win_of = torch.from_numpy(wi).to(dev)
    pos_of = torch.from_numpy(pi).to(dev)

    def recall(q, mem, own_past):
        """Per query row: the error of the closest remembered key with the same anchor, and how many of its tokens the trunk keeps."""
        qi, mi = torch.nonzero(q)[:, 0], torch.nonzero(mem)[:, 0]
        e = torch.full((len(qi),), float("inf"), device=dev)
        got = torch.zeros(len(qi), dtype=torch.long, device=dev)
        km, am = HE[mi], anchor[mi]
        for b in range(0, len(qi), 1024):
            i = qi[b : b + 1024]
            d = (HE[i] ** 2).sum(-1, keepdim=True) - 2 * HE[i] @ km.T + (km**2).sum(-1)
            d = d + 1e6 * (anchor[i][:, None] != am[None, :])
            if own_past:  # only the same window's rows far enough back to have their continuation seen
                d = d + 1e6 * ((win_of[i][:, None] != win_of[mi][None, :]) | (pos_of[mi][None, :] + M + 1 > pos_of[i][:, None]))
            v, a = d.min(1)
            same = (cont[mi[a]] == truth[i]) & valid[i]
            e[b : b + 1024] = v
            got[b : b + 1024] = torch.cumprod(same.long(), 1).sum(1)
        return qi, e, got

    for name, mem, own_past in (("the train windows as the memory", TR, False), ("the window's own past as the memory", TE | CA, True)):
        say(f"== trajectory memory, {name}")
        qc, ec, gc = recall(CA, mem, own_past)
        qt, et, gt = recall(TE, mem, own_past)
        found = et < 1e5
        say(f"  a remembered row with the same anchor exists for {found.float().mean():.3f} of the held-out rows; its first token is the trunk's on {(gt[found] >= 1).float().mean():.3f} of those")
        order = torch.argsort(et)
        for sh in (0.02, 0.05, 0.1, 0.25, 0.5):
            n = max(int(len(order) * sh), 1)
            i = order[:n]
            say(f"    the closest {sh:.2f} of the rows (error up to {et[i[-1]]:.4f}): first token right {(gt[i] >= 1).float().mean():.3f}, {gt[i].float().mean():.2f} of {M} kept; the block on the same rows keeps {blk[qt[i]].float().mean():.2f} of {J}, on the others {blk[qt[order[n:]]].float().mean():.2f}")
        # a round drafts the remembered three where the key is close enough, the block's three elsewhere
        base = speed(np.full((W, t), J), held)
        def mixed(theta, q, e, got):
            k = kept.copy()
            use = (e <= theta).cpu().numpy()
            rows = q.cpu().numpy()[use]
            k[wi[rows], pi[rows]] = np.minimum(got.cpu().numpy()[use], J)
            return k

        three = np.full((W, t), J)
        thetas = [torch.quantile(ec[ec < 1e5], x).item() for x in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)] if (ec < 1e5).any() else []
        best = (None, 0.0)
        for th in thetas:
            s = speed(three, cal, mixed(th, qc, ec, gc))[0]
            if s > best[1]:
                best = (th, s)
        if best[0] is not None and best[1] > speed(three, cal)[0]:
            s = speed(three, held, mixed(best[0], qt, et, gt))
            say(f"  the memory where the error is under {best[0]:.4f} ({(et <= best[0]).float().mean():.3f} of the rows), the block elsewhere, three drafts a round: speed {s[0]:.3f} of plain, {s[1]:.2f} tokens a round; the block alone {base[0]:.3f}, {base[1]:.2f}")
        else:
            say(f"  no error threshold makes the memory beat the block on the calibration windows (the block alone: speed {base[0]:.3f} of plain, {base[1]:.2f} tokens a round)")
