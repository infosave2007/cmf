#!/usr/bin/env python3
"""How a draft model published for Qwen3.8-27B holds on this trunk.

    q27_ext_draft.py TAPS_DIR DUMP_DIR DRAFT_DIR [STRIDE=12] [DEPTH=7]

TAPS_DIR   `cortiq mtp-heal --taps DIR=5,19,33,47,61`: the trunk's layer rows,
           tokens and own next tokens of the held-out windows
DUMP_DIR   `cortiq mtp-heal --dump`: embed.f16 and head.f16 of the same file
DRAFT_DIR  config.json and model.safetensors of a DFlash 2 or DSpark drafter

The forward is the drafter's published one (z-lab/dflash `model.py`, the
`dflash.py` and `dspark.py` beside RadixArk's weights), written for a batch of
round starts of one window. The count is `mtp-heal`'s round layout: a chain
starts every STRIDE rows at offsets 0 and STRIDE/2 where the token after it is
the trunk's own, and a draft counts at depth j while every draft before it held.

REF=DIR with REF_WINDOWS=0,20,26 (indices into the taps' windows) writes what a
port of the drafter is checked against: per window its taps, tokens, the round
starts and the drafts of each, and for the first eight starts the block's rows
after the last norm, the sixteen candidates a row and their scores.

QUANT=q8|q4|q4all rounds the drafter's matrices first, a stand-in for a CMF codec:
q8 is eight bits with a scale a row, q4 four bits with a scale a row and 32 columns
on the layers' matrices (eight bits on the rest), q4all four bits on everything.
SETS="code:3,en:3,ru:3" names the data sets by their windows, in order.
"""
import json
import math
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

taps_dir, dump_dir, draft_dir = sys.argv[1:4]
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 12
depth = int(sys.argv[5]) if len(sys.argv) > 5 else 7
dev = "cuda"
torch.set_grad_enabled(False)
REF = os.environ.get("REF")
ONLY = [int(x) for x in os.environ["REF_WINDOWS"].split(",")] if REF else None


def safetensors(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    head = json.loads(f.read(n))
    base = 8 + n
    kinds = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    out = {}
    for name, e in head.items():
        if name == "__metadata__":
            continue
        a, b = e["data_offsets"]
        f.seek(base + a)
        t = torch.frombuffer(bytearray(f.read(b - a)), dtype=kinds[e["dtype"]]).reshape(e["shape"])
        out[name] = t.to(dev)
    return out


cfg = json.load(open(f"{draft_dir}/config.json"))
kind = cfg["architectures"][0]
dc = cfg.get("dflash_config", {})
W = safetensors(f"{draft_dir}/model.safetensors")
QUANT = os.environ.get("QUANT")


def rounded(w, bits):
    kind, w = w.dtype, w.float()
    if bits == 8:
        s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
        return (torch.round(w / s).clamp(-127, 127) * s).to(kind)
    r, c = w.shape
    x = F.pad(w, (0, (-c) % 32)).view(r, -1, 32)
    s = x.abs().amax(2, keepdim=True).clamp_min(1e-12) / 7
    return (torch.round(x / s).clamp(-7, 7) * s).view(r, -1)[:, :c].to(kind)


if QUANT:
    for name in list(W):
        if W[name].dim() != 2 or "base_kernel" in name:
            continue
        body = name.startswith("layers.") and "kernel_projection" not in name
        W[name] = rounded(W[name], 4 if QUANT == "q4all" or (QUANT == "q4" and body) else 8)
    print(f"the drafter's matrices rounded as {QUANT}", flush=True)
meta = json.load(open(f"{taps_dir}/meta.json"))
t, hs, layers, held = meta["window"], meta["hidden"], meta["layers"], meta["windows"]
want = dc.get("target_layer_ids", cfg.get("target_layer_ids"))
assert want == layers, f"the drafter reads layers {want}, the taps hold {layers}"
nw, nl = len(held), len(layers)
lens = np.fromfile(f"{taps_dir}/len.u32", dtype=np.uint32)
ids = np.fromfile(f"{taps_dir}/ids.u32", dtype=np.uint32).reshape(nw, t).astype(np.int64)
own = np.fromfile(f"{taps_dir}/ti.u32", dtype=np.uint32).reshape(nw, t).astype(np.int64)
taps = np.memmap(f"{taps_dir}/taps.f16", dtype=np.float16, mode="r", shape=(nw, nl, t, hs))
vocab = cfg["vocab_size"]
embed = torch.from_numpy(np.fromfile(f"{dump_dir}/embed.f16", dtype=np.float16).reshape(vocab, hs)).to(dev)
head = torch.from_numpy(np.fromfile(f"{dump_dir}/head.f16", dtype=np.float16).reshape(vocab, hs)).to(dev).to(torch.bfloat16)

eps = cfg["rms_norm_eps"]
nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
rep = nh // nkv
nlay = cfg["num_hidden_layers"]
mask_id = dc.get("mask_token_id", cfg.get("mask_token_id"))
is2 = kind == "DFlash2DraftModel"
block = int(dc.get("block_size", cfg.get("block_size")))
# rows a block runs, and the row that drafts the token after the anchor
rows, first = (block, 1) if is2 else (block, 0)
window = cfg.get("sliding_window") if cfg.get("use_sliding_window") else None
dt = torch.bfloat16


def rms(x, w):
    y = x.float()
    return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def lin(x, name):
    return F.linear(x, W[name])


def inv_freq():
    rp = cfg.get("rope_parameters") or {}
    base, dim = rp.get("rope_theta") or cfg["rope_theta"], hd
    pos = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    if rp.get("rope_type", "default") != "yarn":
        return 1.0 / pos, 1.0
    # transformers' _compute_yarn_parameters
    factor = rp["factor"]
    orig = rp.get("original_max_position_embeddings") or cfg["max_position_embeddings"]
    fast, slow = rp.get("beta_fast") or 32, rp.get("beta_slow") or 1
    scale = rp.get("attention_factor") or (0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0)

    def cdim(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))

    low, high = cdim(fast), cdim(slow)
    if rp.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    ext = 1 - ramp
    return (1.0 / (factor * pos)) * (1 - ext) + (1.0 / pos) * ext, scale


INV, SCALE = inv_freq()
INV = INV.to(dev)


def rope(x, pos):
    """x [..., n, hd] at integer positions pos [..., n]."""
    fr = pos.float()[..., None] * INV
    emb = torch.cat([fr, fr], -1)
    cos, sin = (emb.cos() * SCALE).to(x.dtype), (emb.sin() * SCALE).to(x.dtype)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def conv(h, dyn, base, gs):
    b, n, _ = h.shape
    g = hs // gs
    blocks = h.view(b, n, g, gs)
    dyn = dyn.reshape(b, n, base.shape[0], g, 1)
    out = torch.zeros_like(blocks)
    for o in range(base.shape[0]):
        v = blocks if o == 0 else F.pad(blocks[:, :-o], (0, 0, 0, 0, o, 0))
        out = out + base[o].view(1, 1, g, gs).to(h.dtype) * v
        out = torch.addcmul(out, dyn[:, :, o], v)
    return out.view_as(h)


class Conv:
    def __init__(self, pre):
        self.base, self.proj = W[f"{pre}.base_kernel"], f"{pre}.kernel_projection.weight"
        self.k, self.gs = int(dc["conv_kernel_size"]), int(dc["conv_group_size"])

    def prepare(self, h):
        dyn = lin(h, self.proj).view(*h.shape[:-1], 2, self.k, hs // self.gs)
        return conv(h, dyn[..., 0, :, :], self.base[0], self.gs), dyn[..., 1, :, :]

    def finish(self, h, dyn):
        return conv(h, dyn, self.base[1], self.gs)


def context(win):
    """The window's rows as every layer's context keys and values."""
    x = torch.from_numpy(np.ascontiguousarray(taps[win])).to(dev)  # [nl][t][hs]
    x = torch.cat([x[i] for i in range(nl)], -1).to(dt)
    tgt = rms(lin(x, "fc.weight"), W["hidden_norm.weight"])
    context.tgt = tgt
    pos = torch.arange(t, device=dev)
    out = []
    for i in range(nlay):
        p = f"layers.{i}.self_attn"
        k = rms(lin(tgt, f"{p}.k_proj.weight").view(t, nkv, hd), W[f"{p}.k_norm.weight"]).transpose(0, 1)
        v = lin(tgt, f"{p}.v_proj.weight").view(t, nkv, hd).transpose(0, 1)
        out.append((rope(k, pos[None, :]), v))  # [nkv][t][hd]
    return out


def draft(ctx, anchors, tokens):
    """Rows of a block per round: anchors [s] (the anchor token's position), tokens [s][rows]."""
    s = anchors.shape[0]
    h = embed[tokens].to(dt)
    pos = anchors[:, None] + torch.arange(rows, device=dev)[None, :]
    key = torch.arange(t, device=dev)
    # a context row is visible before the anchor, and inside the layer's window
    hide = key[None, None, :] >= anchors[:, None, None]
    if window is not None:
        hide = hide | (pos[:, :, None] - key[None, None, :] >= window)
    hide = hide[:, None, None]  # [s][1][1][rows][t]
    for i in range(nlay):
        p = f"layers.{i}"
        x = rms(h, W[f"{p}.input_layernorm.weight"])
        ak = None
        if is2:
            ac = Conv(f"{p}.attention_conv")
            x, ak = ac.prepare(x)
        q = rms(lin(x, f"{p}.self_attn.q_proj.weight").view(s, rows, nh, hd), W[f"{p}.self_attn.q_norm.weight"]).transpose(1, 2)
        kn = rms(lin(x, f"{p}.self_attn.k_proj.weight").view(s, rows, nkv, hd), W[f"{p}.self_attn.k_norm.weight"]).transpose(1, 2)
        vn = lin(x, f"{p}.self_attn.v_proj.weight").view(s, rows, nkv, hd).transpose(1, 2)
        q = rope(q, pos[:, None, :]).reshape(s, nkv, rep, rows, hd)
        kn = rope(kn, pos[:, None, :])
        kc, vc = ctx[i]
        sc = torch.einsum("sgqrd,gtd->sgqrt", q, kc).float() * hd**-0.5
        sc = sc.masked_fill(hide, float("-inf"))
        sn = torch.einsum("sgqrd,sgud->sgqru", q, kn).float() * hd**-0.5
        a = torch.softmax(torch.cat([sc, sn], -1), -1).to(dt)
        o = torch.einsum("sgqrt,gtd->sgqrd", a[..., :t], vc) + torch.einsum("sgqru,sgud->sgqrd", a[..., t:], vn)
        o = o.reshape(s, nh, rows, hd).transpose(1, 2).reshape(s, rows, nh * hd)
        o = lin(o, f"{p}.self_attn.o_proj.weight")
        if ak is not None:
            o = ac.finish(o, ak)
        h = h + o
        x = rms(h, W[f"{p}.post_attention_layernorm.weight"])
        mk = None
        if is2:
            mc = Conv(f"{p}.mlp_conv")
            x, mk = mc.prepare(x)
        x = lin(F.silu(lin(x, f"{p}.mlp.gate_proj.weight")) * lin(x, f"{p}.mlp.up_proj.weight"), f"{p}.mlp.down_proj.weight")
        if mk is not None:
            x = mc.finish(x, mk)
        h = h + x
    return rms(h, W["norm.weight"])


def propose(hid, anchor_tok):
    """The drafted tokens [s][n] of the rows after `first`."""
    hid = hid[:, first:]
    logits = F.linear(hid, head).float()
    if is2:
        top = int(dc["selector_top_k"])
        unary, cand = torch.topk(logits, top, dim=-1)
        hp = lin(hid, "candidate_selector.hidden_projection.weight").float()
        pc, scb = W["candidate_selector.predecessor_codebook"], W["candidate_selector.successor_codebook"]
        prev, path = anchor_tok, []
        for j in range(hid.shape[1]):
            sc = unary[:, j] + torch.einsum("br,bkr->bk", pc[prev].float() * hp[:, j], scb[cand[:, j]].float())
            prev = cand[:, j].gather(-1, sc.argmax(-1)[:, None])[:, 0]
            path.append(prev)
        return torch.stack(path, 1), logits.argmax(-1)
    plain = logits.argmax(-1)
    if "markov_head.markov_w1.weight" not in W:
        return plain, plain
    w1, w2 = W["markov_head.markov_w1.weight"], W["markov_head.markov_w2.weight"]
    prev, path = anchor_tok, []
    for j in range(hid.shape[1]):
        prev = (logits[:, j] + F.linear(w1[prev], w2).float()).argmax(-1)
        path.append(prev)
    return torch.stack(path, 1), plain


names = ["as published", "its rows' own top token", "rounds in order"]
cnt = np.zeros((3, depth, 2), dtype=np.int64)
group = {}
for wi in range(nw):
    if ONLY is not None and wi not in ONLY:
        continue
    n_len = int(lens[wi])
    starts = [n for off in (0, stride // 2) for n in range(off, n_len - 1, stride) if ids[wi, n + 1] == own[wi, n]]
    if not starts:
        continue
    ctx = context(wi)
    wid, wown = torch.from_numpy(ids[wi]).to(dev), torch.from_numpy(own[wi]).to(dev)
    got = [[], []]
    for b in range(0, len(starts), 96):
        st = torch.tensor(starts[b : b + 96], device=dev)
        anchors = st + 1
        toks = torch.full((st.shape[0], rows), mask_id, device=dev, dtype=torch.long)
        toks[:, 0] = wid[anchors]
        out = propose(draft(ctx, anchors, toks), wid[anchors])
        for v in range(2):
            got[v].append(out[v].cpu().numpy())
    got = [np.concatenate(g) for g in got]
    if REF:
        os.makedirs(REF, exist_ok=True)
        st = torch.tensor(starts[:8], device=dev)
        toks = torch.full((st.shape[0], rows), mask_id, device=dev, dtype=torch.long)
        toks[:, 0] = wid[st + 1]
        hid = draft(ctx, st + 1, toks)
        unary, cand = torch.topk(F.linear(hid[:, first:], head).float(), 16, dim=-1)
        np.savez(
            f"{REF}/ref-{held[wi]}.npz",
            ids=ids[wi], own=own[wi], length=n_len, starts=np.array(starts), drafts=got[0], plain=got[1],
            hidden=hid.float().cpu().numpy(), unary=unary.cpu().numpy(), cand=cand.cpu().numpy(),
            target=context.tgt[:64].float().cpu().numpy(),
            key0=ctx[0][0][:, :64].float().cpu().numpy(), value0=ctx[0][1][:, :64].float().cpu().numpy(),
        )
        np.ascontiguousarray(taps[wi]).tofile(f"{REF}/taps-{held[wi]}.f16")
    local = np.zeros((3, depth, 2), dtype=np.int64)
    # rounds as a decode has them: the next one starts after what this one kept
    n = 0
    while n + 1 < n_len:
        if ids[wi, n + 1] != own[wi, n]:
            n += 1
            continue
        toks = torch.full((1, rows), mask_id, device=dev, dtype=torch.long)
        toks[0, 0] = wid[n + 1]
        d = propose(draft(ctx, torch.tensor([n + 1], device=dev), toks), wid[n + 1 : n + 2])[0][0].cpu().numpy()
        kept = 0
        for j in range(1, min(depth, d.shape[0]) + 1):
            q = n + j - 1
            if q + 1 >= n_len or ids[wi, q + 1] != own[wi, q]:
                break
            hit = d[j - 1] == own[wi, q + 1]
            local[2, j - 1, 0] += 1
            local[2, j - 1, 1] += hit
            if not hit:
                break
            kept += 1
        n += kept + 1
    for v in range(2):
        for si, n in enumerate(starts):
            for j in range(1, min(depth, got[v].shape[1]) + 1):
                q = n + j - 1
                if q + 1 >= n_len or ids[wi, q + 1] != own[wi, q]:
                    break
                hit = got[v][si, j - 1] == own[wi, q + 1]
                local[v, j - 1, 0] += 1
                local[v, j - 1, 1] += hit
                if not hit:
                    break
    cnt += local
    # the data sets in the order mtp-heal read them: code, own, own2, or SETS="name:windows,..."
    g = "code" if held[wi] < 196 else "prose"
    if os.environ.get("SETS"):
        edge = 0
        for part in os.environ["SETS"].split(","):
            g, n = part.split(":")
            edge += int(n)
            if held[wi] < edge:
                break
    group[g] = group.get(g, 0) + local
    print(f"window {held[wi]:3d}: {len(starts)} rounds", flush=True)


def line(c):
    r = [c[j, 1] / max(c[j, 0], 1) for j in range(depth)]
    a, keep = 1.0, 0.0
    for x in r:
        a *= x
        keep += a
    return " / ".join(f"{x:.3f}" for x in r) + f"  ({' / '.join(str(int(c[j, 0])) for j in range(depth))} rows)  {keep:.2f} drafts a round"


print(f"{kind} from {draft_dir}: {rows} rows a block, a chain every {stride} rows")
for v in range(3):
    print(f"  {names[v]:24s} {line(cnt[v])}")
    for g in sorted(group):
        print(f"    {g:22s} {line(group[g][v])}")
