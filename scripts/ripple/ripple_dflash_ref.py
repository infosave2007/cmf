#!/usr/bin/env python3
"""The published DFlash2 drafter of Ripple-65B on the engine's own rows.

    ripple_dflash_ref.py DUMP_DIR IDS_FILE DRAFT_DIR [DEPTH=7] [ROUNDS=60]

DUMP_DIR   `CMF_DFLASH_DUMP=DIR` of a plain greedy run (`cortiq run … --greedy`
           with the sidecar beside the model): taps.bin, embed.f32, head.f32, meta.json
IDS_FILE   `CMF_GEN_IDS_OUT=FILE` of the same run: every id of the sequence
DRAFT_DIR  config.json and model.safetensors of ricursiveAI/Ripple-65B-Preview-DFlash2

The forward is the drafter's published one (z-lab/dflash `model.py`) with this
checkpoint's per-layer attention pattern: four sliding layers and a full layer
without rotary embeddings. A round starts at every position after the prompt as a
decode would (the next round after what this one kept); a draft counts at depth j
while every draft before it held. Both ways of reading the block's input rows are
tried: the embedding table as it is, and the trunk's scaled input rows.
Set REF_ROWS=n to print the first n rounds' drafts beside the trunk's own ids.
REF_KIT=DIR writes the kit `cortiq dflash-check` reads instead (the whole sequence as
window 0, a round at every position after the prompt, table rows as the block's input).
"""
import json
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

dump_dir, ids_file, draft_dir = sys.argv[1:4]
depth = int(sys.argv[4]) if len(sys.argv) > 4 else 7
rounds_most = int(sys.argv[5]) if len(sys.argv) > 5 else 60
torch.set_grad_enabled(False)
dev = "cuda" if os.environ.get("REF_CUDA") == "1" and torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if dev == "cuda" else torch.float32


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
        out[name] = t.to(dev).to(dt) if t.dim() == 2 else t.to(dev).to(dt)
    return out


cfg = json.load(open(f"{draft_dir}/config.json"))
dc = cfg["dflash_config"]
meta = json.load(open(f"{dump_dir}/meta.json"))
hs, vocab = meta["hidden"], meta["vocab"]
mult = float(meta["embed_multiplier"])
layers = dc["target_layer_ids"]
nl = len(layers)
ids = np.array([int(x) for x in open(ids_file).read().split()], dtype=np.int64)

# the kept chunks, by position (a later chunk at a position replaces an earlier one)
rows, prompt = {}, None
with open(f"{dump_dir}/taps.bin", "rb") as f:
    while True:
        h = f.read(16)
        if len(h) < 16:
            break
        pos0, n = struct.unpack("<QQ", h)
        prompt = prompt or n  # the first chunk is the prompt
        a = np.frombuffer(f.read(n * nl * hs * 4), dtype=np.float32).reshape(n, nl, hs)
        for r in range(n):
            rows[pos0 + r] = a[r]
n_rows = 0
while n_rows in rows:
    n_rows += 1
taps = torch.from_numpy(np.stack([rows[p] for p in range(n_rows)])).to(dev)  # [n][nl][hs]
print(f"{n_rows} tapped rows of {nl} layers, {len(ids)} ids, embed ×{mult:.3f}", flush=True)
assert n_rows + 1 >= len(ids) - 1, "the taps stop before the ids do"

embed = torch.from_numpy(np.fromfile(f"{dump_dir}/embed.f32", dtype=np.float32).reshape(vocab, hs)).to(dev).to(dt)
head = torch.from_numpy(np.fromfile(f"{dump_dir}/head.f32", dtype=np.float32).reshape(vocab, hs)).to(dev).to(dt)
W = safetensors(f"{draft_dir}/model.safetensors")

eps = cfg["rms_norm_eps"]
nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
rep = nh // nkv
nlay = cfg["num_hidden_layers"]
mask_id = dc["mask_token_id"]
block = int(dc["block_size"])
rows_n, first = block, 1
kinds = cfg.get("layer_types") or ["full_attention"] * nlay
windows = [cfg["sliding_window"] if k == "sliding_attention" and cfg.get("use_sliding_window", True) else None for k in kinds]
norope = set(cfg.get("no_rope_layers", []))
theta = (cfg.get("rope_parameters") or {}).get("rope_theta") or cfg["rope_theta"]
INV = (1.0 / theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)).to(dev)


def rms(x, w):
    y = x.float()
    return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def lin(x, name):
    return F.linear(x, W[name])


def rope(x, pos):
    fr = pos.float()[..., None] * INV
    emb = torch.cat([fr, fr], -1)
    cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
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


def context():
    """Every tapped row as each layer's context keys and values."""
    x = torch.cat([taps[:, i] for i in range(nl)], -1).to(dt)
    tgt = rms(lin(x, "fc.weight"), W["hidden_norm.weight"])
    context.tgt = tgt
    pos = torch.arange(n_rows, device=dev)
    out = []
    for i in range(nlay):
        p = f"layers.{i}.self_attn"
        k = rms(lin(tgt, f"{p}.k_proj.weight").view(n_rows, nkv, hd), W[f"{p}.k_norm.weight"]).transpose(0, 1)
        v = lin(tgt, f"{p}.v_proj.weight").view(n_rows, nkv, hd).transpose(0, 1)
        out.append((k if i in norope else rope(k, pos[None, :]), v))
    return out


def draft(ctx, anchor, tokens, scale):
    """The block's rows after the last norm: anchor position, tokens [rows]."""
    h = (embed[tokens] * scale).to(dt)[None]
    pos = anchor + torch.arange(rows_n, device=dev)[None, :]
    key = torch.arange(n_rows, device=dev)
    for i in range(nlay):
        p = f"layers.{i}"
        hide = key[None, None, :] >= anchor
        if windows[i] is not None:
            hide = hide | (pos[:, :, None] - key[None, None, :] >= windows[i])
        hide = hide[:, None, None]
        x = rms(h, W[f"{p}.input_layernorm.weight"])
        ac = Conv(f"{p}.attention_conv")
        x, ak = ac.prepare(x)
        q = rms(lin(x, f"{p}.self_attn.q_proj.weight").view(1, rows_n, nh, hd), W[f"{p}.self_attn.q_norm.weight"]).transpose(1, 2)
        kn = rms(lin(x, f"{p}.self_attn.k_proj.weight").view(1, rows_n, nkv, hd), W[f"{p}.self_attn.k_norm.weight"]).transpose(1, 2)
        vn = lin(x, f"{p}.self_attn.v_proj.weight").view(1, rows_n, nkv, hd).transpose(1, 2)
        if i not in norope:
            q, kn = rope(q, pos[:, None, :]), rope(kn, pos[:, None, :])
        q = q.reshape(1, nkv, rep, rows_n, hd)
        kc, vc = ctx[i]
        sc = torch.einsum("sgqrd,gtd->sgqrt", q, kc).float() * hd**-0.5
        sc = sc.masked_fill(hide, float("-inf"))
        sn = torch.einsum("sgqrd,sgud->sgqru", q, kn).float() * hd**-0.5
        a = torch.softmax(torch.cat([sc, sn], -1), -1).to(dt)
        o = torch.einsum("sgqrt,gtd->sgqrd", a[..., :n_rows], vc) + torch.einsum("sgqru,sgud->sgqrd", a[..., n_rows:], vn)
        o = o.reshape(1, nh, rows_n, hd).transpose(1, 2).reshape(1, rows_n, nh * hd)
        o = ac.finish(lin(o, f"{p}.self_attn.o_proj.weight"), ak)
        h = h + o
        x = rms(h, W[f"{p}.post_attention_layernorm.weight"])
        mc = Conv(f"{p}.mlp_conv")
        x, mk = mc.prepare(x)
        x = lin(F.silu(lin(x, f"{p}.mlp.gate_proj.weight")) * lin(x, f"{p}.mlp.up_proj.weight"), f"{p}.mlp.down_proj.weight")
        h = h + mc.finish(x, mk)
    return rms(h, W["norm.weight"])


def propose(hid, anchor_tok):
    """The selector's path and the rows' own top tokens, rows after `first`."""
    hid = hid[:, first:]
    logits = F.linear(hid, head).float()
    top = int(dc["selector_top_k"])
    unary, cand = torch.topk(logits, top, dim=-1)
    hp = lin(hid, "candidate_selector.hidden_projection.weight").float()
    pc, scb = W["candidate_selector.predecessor_codebook"], W["candidate_selector.successor_codebook"]
    prev, path = anchor_tok, []
    for j in range(hid.shape[1]):
        sc = unary[:, j] + torch.einsum("br,bkr->bk", pc[prev].float() * hp[:, j], scb[cand[:, j]].float())
        prev = cand[:, j].gather(-1, sc.argmax(-1)[:, None])[:, 0]
        path.append(prev)
    return torch.stack(path, 1)[0], logits.argmax(-1)[0]


ctx = context()
show = int(os.environ.get("REF_ROWS", "6"))
start = int(os.environ.get("REF_START", str(prompt)))
ids_t = torch.from_numpy(ids).to(dev)
kit = os.environ.get("REF_KIT")
if kit:
    os.makedirs(kit, exist_ok=True)
    starts = [p - 1 for p in range(start, n_rows)]  # the row before each anchor
    hidden, drafts_all, plain_all = [], [], []
    for si, s0 in enumerate(starts):
        a = s0 + 1
        toks = torch.full((rows_n,), mask_id, device=dev, dtype=torch.long)
        toks[0] = ids_t[a]
        hid = draft(ctx, torch.tensor([a], device=dev), toks, 1.0)
        path, plain = propose(hid, ids_t[a : a + 1])
        drafts_all.append(path.cpu().numpy())
        plain_all.append(plain.cpu().numpy())
        if si < 8:
            hidden.append(hid[0].float().cpu().numpy())
    n_first = len(hidden)
    ids[:n_rows].astype(np.uint32).tofile(f"{kit}/w0.ids.u32")
    ids[1 : n_rows + 1].astype(np.uint32).tofile(f"{kit}/w0.own.u32")
    np.array(starts, dtype=np.uint32).tofile(f"{kit}/w0.starts.u32")
    np.stack(hidden).astype(np.float32).tofile(f"{kit}/w0.hidden.f32")
    np.stack(drafts_all).astype(np.uint32).tofile(f"{kit}/w0.drafts.u32")
    np.stack(plain_all).astype(np.uint32).tofile(f"{kit}/w0.plain.u32")
    rows_k = min(64, n_rows)
    context.tgt[:rows_k].float().cpu().numpy().astype(np.float32).tofile(f"{kit}/w0.target.f32")
    ctx[0][0][:, :rows_k].float().cpu().numpy().astype(np.float32).tofile(f"{kit}/w0.key0.f32")
    ctx[0][1][:, :rows_k].float().cpu().numpy().astype(np.float32).tofile(f"{kit}/w0.value0.f32")
    taps.permute(1, 0, 2).contiguous().cpu().numpy().astype(np.float16).tofile(f"{kit}/w0.taps.f16")
    json.dump(
        {"windows": [0], "window": n_rows, "hidden": hs, "layers": layers, "starts": n_first, "block": rows_n, "rows": rows_k, "length": {"0": n_rows}},
        open(f"{kit}/kit.json", "w"),
        indent=1,
    )
    print(f"kit in {kit}: {len(starts)} starts, block rows of the first {n_first}", flush=True)
    sys.exit(0)
for name, scale in (("table rows", 1.0), ("scaled rows", mult)):
    cnt = np.zeros((depth, 2), dtype=np.int64)
    kept_all, drafted_all, nr = 0, 0, 0
    p = start  # the first tip is the first id after the prompt
    while p + 1 < len(ids) and p <= n_rows and nr < rounds_most:
        toks = torch.full((rows_n,), mask_id, device=dev, dtype=torch.long)
        toks[0] = ids_t[p]
        path, plain = propose(draft(ctx, torch.tensor([p], device=dev), toks, scale), ids_t[p : p + 1])
        d = path.cpu().numpy()[:depth]
        truth = ids[p + 1 : p + 1 + depth]
        kept = 0
        for j in range(min(len(d), len(truth))):
            cnt[j, 0] += 1
            if d[j] == truth[j]:
                cnt[j, 1] += 1
                kept += 1
            else:
                break
        if nr < show:
            print(f"  {name}: tip {ids[p]} at {p} | drafts {d.tolist()} | own top {plain.cpu().numpy()[:depth].tolist()} | truth {truth.tolist()} | kept {kept}")
        kept_all += kept
        drafted_all += min(len(d), len(truth))
        nr += 1
        p += kept + 1
    held = " ".join(f"{int(cnt[j, 1])}/{int(cnt[j, 0])}" for j in range(depth))
    print(f"{name}: {nr} rounds | kept {kept_all} of {drafted_all} | by depth {held}", flush=True)
