#!/usr/bin/env python3
"""The DSpark drafter of Inkling 2 / TML-d24-varloop-research in f32 PyTorch, against cortiq's first draft round.

dspark_ref.py SRC_DIR DUMP_DIR
  SRC_DIR   the release's checkpoint (mtp.safetensors, model safetensors with the index)
  DUMP_DIR  what `CMF_INKLING_DS_DUMP=DUMP_DIR cortiq run ...` wrote on CUDA: taps.bin / taps.idx (every target call's
            taps: the inputs of applications 5, 23, 41 and the final normed state, [tap][rows][hidden] f32), and the
            first round's h.bin (the block's normed rows), logits.bin (each draft's logits without fold_b) and round.json.

The drafter as Basalt's dspark.hpp describes vLLM's reference: context mixes mix_l = rms(sum_k softmax(fuse[l])_k f_k) *
dflare_norm[l]; keys rope(rms_head(k_ctx mix) * k_norm_ctx, p), values v_ctx mix; a block of the anchor's normed
embedding and seven mask rows at p .. p + 7 through three layers (q/k normed per head, NeoX rope theta 1e6, each row
attends the context [p - window, p) or [0, p) and all eight block rows); h = norm(x); the Markov head's recurrence over
W_h h_{i+1} + W_f f(p - 1) and W_sp [state; w1[prev]]; logits = head(h_{i+1}) + head(W_out lat + b_out)."""
import json, math, os, sys

import numpy as np
import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
src, dump = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(src, "config.json")))
mc = cfg.get("mtp_config") or cfg["text_config"]["mtp_config"]
tc = cfg.get("text_config", cfg)
H, NH, NKV, HD = mc["hidden_size"], mc["num_attention_heads"], mc["num_key_value_heads"], mc["head_dim"]
EPS, THETA, WIN, B = mc["rms_norm_eps"], mc["rope_theta"], mc["sliding_window_size"], mc["block_size"]
LOCAL = set(mc["local_layer_ids"])
NL, R = mc["num_hidden_layers"], mc["markov_rank"]

mtp = safe_open(os.path.join(src, "mtp.safetensors"), "pt")
W = lambda k: mtp.get_tensor(k).float()
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
_open = {}


def T(k):
    f = idx[k]
    if f not in _open:
        _open[f] = safe_open(os.path.join(src, f), "pt")
    return _open[f].get_tensor(k).float()


# the target's taps at every position before the round (a later call's rows overwrite an earlier one's)
rd = json.load(open(os.path.join(dump, "round.json")))
P, anchor, K = rd["pos"], rd["anchor"], rd["k"]
raw = np.fromfile(os.path.join(dump, "taps.bin"), dtype=np.float32)
NT = len(mc["target_layer_ids"])
taps = torch.zeros(NT, P, H)
at = 0
for line in open(os.path.join(dump, "taps.idx")):
    p0, n = map(int, line.split())
    blk = torch.from_numpy(raw[at:at + NT * n * H].copy()).view(NT, n, H)
    at += NT * n * H
    for r in range(n):
        if p0 + r < P:
            taps[:, p0 + r] = blk[:, r]
assert at == raw.size, (at, raw.size)


def rms(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos):  # x [..., n, HD], pos [n]
    half = HD // 2
    inv = THETA ** (-torch.arange(0, half, dtype=torch.float64) * 2 / HD)
    ang = pos[:, None].double() * inv[None]
    cs, sn = ang.cos().float(), ang.sin().float()
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cs - b * sn, b * cs + a * sn], -1)


alpha = torch.softmax(W("fuse_weight"), -1)  # [mixes][targets]
mix = lambda j, w: rms(torch.einsum("k,kph->ph", alpha[j], taps), w)
pos_ctx = torch.arange(P)
emb = T("model.llm.embed.weight")
x = torch.zeros(B, H)
x[0] = rms(emb[anchor], T("model.llm.embed_norm.weight"))
x[1:] = W("mask_embed")
pos_blk = torch.arange(P, P + B)
for l in range(NL):
    p = f"layers.{l}."
    m = mix(l, W(p + "dflare_norm.weight"))
    kc = (m @ W(p + "self_attn.k_proj_ctx.weight").T).view(P, NKV, HD)
    kc = rope(rms(kc, W(p + "self_attn.k_norm_ctx.weight")).transpose(0, 1), pos_ctx)  # [NKV][P][HD]
    vc = (m @ W(p + "self_attn.v_proj_ctx.weight").T).view(P, NKV, HD).transpose(0, 1)
    y = rms(x, W(p + "input_layernorm.weight"))
    q = (y @ W(p + "self_attn.q_proj.weight").T).view(B, NH, HD)
    k = (y @ W(p + "self_attn.k_proj.weight").T).view(B, NKV, HD)
    v = (y @ W(p + "self_attn.v_proj.weight").T).view(B, NKV, HD).transpose(0, 1)
    q = rope(rms(q, W(p + "self_attn.q_norm.weight")).transpose(0, 1), pos_blk)  # [NH][B][HD]
    k = rope(rms(k, W(p + "self_attn.k_norm.weight")).transpose(0, 1), pos_blk)
    lo = max(0, P - WIN) if l in LOCAL else 0
    keys = torch.cat([kc[:, lo:], k], 1)
    vals = torch.cat([vc[:, lo:], v], 1)
    rep = NH // NKV
    keys, vals = keys.repeat_interleave(rep, 0), vals.repeat_interleave(rep, 0)
    att = torch.softmax(q @ keys.transpose(1, 2) / math.sqrt(HD), -1) @ vals  # [NH][B][HD]
    x = x + att.transpose(0, 1).reshape(B, NH * HD) @ W(p + "self_attn.o_proj.weight").T
    z = rms(x, W(p + "post_attention_layernorm.weight"))
    g, u = z @ W(p + "mlp.gate_proj.weight").T, z @ W(p + "mlp.up_proj.weight").T
    x = x + (torch.nn.functional.silu(g) * u) @ W(p + "mlp.down_proj.weight").T
h = rms(x, W("norm.weight"))

ours_h = torch.from_numpy(np.fromfile(os.path.join(dump, "h.bin"), dtype=np.float32)).view(B, H)
cos = torch.nn.functional.cosine_similarity(h, ours_h, -1)
print("h rows cos:", " ".join(f"{c:.5f}" for c in cos.tolist()), " max |d|:", f"{(h - ours_h).abs().max():.4f}")

# the Markov head
head = T("model.llm.unembed.weight")
f = mix(NL, W("markov_head.dflare_norm.weight"))[P - 1]
rw, rb = W("markov_head.rnn_proj.weight"), W("markov_head.rnn_proj.bias")
w_sp, w_h, w_f = rw[:, :2 * R], rw[:, 2 * R:2 * R + H], rw[:, 2 * R + H:]
w1 = W("markov_head.markov_w1.weight")
wo, bo = W("markov_head.hidden_out.weight"), W("markov_head.hidden_out.bias")
ours_lg = torch.from_numpy(np.fromfile(os.path.join(dump, "logits.bin"), dtype=np.float32)).view(K, -1)


def drafts(follow):
    state, prev, out, rows = torch.zeros(R), anchor, [], []
    for i in range(K):
        raw = rb + w_h @ h[i + 1] + w_f @ f + w_sp @ torch.cat([state, w1[prev]])
        gate = torch.sigmoid(raw[:R])
        state = gate * state + (1 - gate) * torch.tanh(raw[R:2 * R])
        lat = torch.tanh(raw[2 * R:])
        lg = head @ (h[i + 1] + wo @ lat)  # fold_b = head @ bo is added below
        full = lg + head @ bo
        pick = int(full.argmax())
        out.append(pick)
        rows.append(lg)
        prev = follow[i] if follow is not None else pick
    return out, rows


ref_ids, ref_rows = drafts(None)
print("anchor", anchor, "pos", P, "k", K)
print("cortiq drafts:", rd["ids"])
print("ref drafts:   ", ref_ids)
_, rows_on_ours = drafts(rd["ids"])
for i in range(K):
    a, b = rows_on_ours[i], ours_lg[i]
    print(f"draft {i}: logits cos {torch.nn.functional.cosine_similarity(a, b, 0):.5f}, max |d| {(a - b).abs().max():.3f}, "
          f"ref top5 {torch.topk(a + head @ bo, 5).indices.tolist()}")
