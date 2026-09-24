#!/usr/bin/env python3
"""MiMo-V2 (model_type "mimo_v2") text-tower reference: a layer-streaming
PyTorch oracle that dequantizes the release checkpoint exactly.

It is the ground truth the converter and engine are diffed against, so it
does NOT use the HF modeling file for the checkpoint layout. That file splits
the fused qkv_proj globally as [Q|K|V], which is wrong for this release: the
rows are stored as `ckpt_tp = num_key_value_heads` tensor-parallel chunks,
each [Q_c | K_c | V_c], with the FP8 block scales tiled per chunk (vLLM /
SGLang loaders; 108 scale rows on a 13568-row full layer = 4 x ceil(3392/128)).
Inside a chunk the scale blocks are CONTIGUOUS 128-row blocks of the chunk
(rows 3200..3327 of a full-layer chunk share one scale across the K/V
boundary), not one block run per Q/K/V segment: `tiles` proves it (all
188160 FP8 tiles have max |code| = 448 under this mapping; the per-segment
reading leaves 266 qkv tiles of layers 23/35/41/47 below 448). Reading the
SWA layers contiguously gives wikitext ppl ~1e5; per-segment scales give
3.1821 on the first 256 tokens, this layout 3.1433. The HF modules are used
only in `selfcheck`, in SPLIT layout, to prove the attention / MoE / norm
math on a random tiny config.

Math (per layer): h += o_proj(attn(rms(h))); h += mlp(rms(h)).
  * RMSNorm x * rsqrt(mean(x^2) + eps) * w (no 1+w), eps = layernorm_epsilon.
  * q/k heads 192 wide, first rope_dim = int(hd * partial_rotary_factor) = 64
    dims rotated NeoX-style (pairs i, i+rope_dim/2); theta rope_theta on full
    layers, swa_rope_theta on SWA layers.
  * V multiplied by attention_value_scale (0.707) before attention.
  * softmax scale head_dim^-0.5; SWA layers see `sliding_window` keys
    INCLUDING the current token (k > q - window), plus a per-head sink logit
    (max includes the sink, the sink column carries no value).
  * MoE: sigmoid scores, top-k on scores + e_score_correction_bias, weights =
    chosen UNBIASED scores renormalized (+1e-20), no shared expert.
  * Weights: FP8 e4m3 x 128x128 block scale_inv; MXFP4 experts (U8 .weight
    low nibble = even element, E2M1, U8 .weight_scale E8M0 = 2^(k-127)).

Commands (python tools/mimo_ref.py <cmd> --help for all flags):
  ppl       --text FILE --tokens N                  one prefix, like `cortiq ppl --tokens N`
  ppl       --text FILE --windows W --window-len L  the exact `cortiq ppl --windows` offsets
  dump      --ids JSON --out DIR                    raw f32 p{pos:06}_l{li:02}.f32 + logits
  gen       --prompt TEXT | --ids JSON --n N        greedy, full recompute
  selfcheck [--hf-dir DIR]                          oracle math vs HF modules (split layout)
  tiles                                             FP8 scale-tiling proof on the checkpoint
  pt2raw    --in DIR --ids JSON --out DIR           old h{li}.pt dumps -> raw per-position files
  mtp       --seqs JSON --cache F --out R.json      MTP draft acceptance (variants A..C_pre)
  mtpcheck                                          MTP explicit-KV path == causal block

Dump layout (the engine's CMF_LAYER_DUMP contract, one comparator for both):
  p{pos:06}_l{li:02}.f32          hidden state AFTER layer li, raw little-endian f32 [hidden]
  p{pos:06}_l{li:02}_attn.f32     (--sub) self_attn output (after o_proj, before the residual)
  p{pos:06}_l{li:02}_ffn.f32      (--sub) MLP / MoE output (before the residual)
  p{pos:06}_logits.f32            logits at pos, raw f32 [vocab]
  logit_dump.f32                  CMF_LOGIT_DUMP format for the last position: final hidden
                                  (pre-norm) followed by its logits
  moe_trace.txt                   CMF_MOE_TRACE format "li:e1,...,ek", position-major
  picks.jsonl                     per (pos, layer): experts, weights, top-k margin
  meta.json                       ids, positions, geometry
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings

import torch

torch.set_grad_enabled(False)

DEFAULT_SRC = os.environ.get("MIMO_SRC", "/root/mimo/src")
FP8_BLOCK = 128
MX_BLOCK = 32


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
class Geo:
    """Attention geometry of one layer."""

    def __init__(self, swa, nq, nkv, hd, vd, rope, theta, sink, window):
        self.swa, self.nq, self.nkv, self.hd, self.vd = swa, nq, nkv, hd, vd
        self.rope, self.theta, self.sink, self.window = rope, theta, sink, window

    def __repr__(self):
        return (f"Geo(swa={self.swa} nq={self.nq} nkv={self.nkv} hd={self.hd} vd={self.vd} "
                f"rope={self.rope} theta={self.theta:g} sink={self.sink} window={self.window})")


class MimoCfg:
    """The subset of a mimo_v2 config.json the text tower needs, with the
    same defaults as configuration_mimo_v2.MiMoV2Config."""

    def __init__(self, cfg: dict):
        self.raw = cfg
        self.H = cfg["hidden_size"]
        self.NL = cfg["num_hidden_layers"]
        self.vocab = cfg["vocab_size"]
        self.nq = cfg["num_attention_heads"]
        self.nkv = cfg.get("num_key_value_heads") or self.nq
        self.hd = cfg.get("head_dim") or self.H // self.nq
        self.vd = cfg.get("v_head_dim") or self.hd
        self.swa_nq = cfg.get("swa_num_attention_heads") or self.nq
        self.swa_nkv = cfg.get("swa_num_key_value_heads") or self.nkv
        self.swa_hd = cfg.get("swa_head_dim") or self.hd
        self.swa_vd = cfg.get("swa_v_head_dim") or self.swa_hd
        rp = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
        self.rope_theta = float(cfg.get("rope_theta", rp.get("rope_theta", 10000.0)))
        self.swa_theta = float(cfg.get("swa_rope_theta") or self.rope_theta)
        self.partial = float(cfg.get("partial_rotary_factor", rp.get("partial_rotary_factor", 1.0)))
        self.window = cfg.get("sliding_window") or cfg.get("sliding_window_size")
        self.pattern = cfg.get("hybrid_layer_pattern") or [0] * self.NL
        moe = cfg.get("moe_layer_freq")
        if isinstance(moe, int):
            moe = [moe > 0 and i % moe == 0 for i in range(self.NL)]
        self.moe = [bool(x) for x in (moe or [0] * self.NL)]
        self.eps = float(cfg.get("layernorm_epsilon", 1e-6))
        self.vscale = cfg.get("attention_value_scale")
        self.full_sink = bool(cfg.get("add_full_attention_sink_bias", False))
        self.swa_sink = bool(cfg.get("add_swa_attention_sink_bias", False))
        self.ne = cfg.get("n_routed_experts")
        self.topk = cfg.get("num_experts_per_tok")
        self.norm_topk = bool(cfg.get("norm_topk_prob", True))
        self.rsf = float(cfg.get("routed_scaling_factor") or 1.0)
        # The fused qkv is pre-sharded for this many ranks (vLLM ckpt_tp).
        self.ckpt_tp = self.nkv
        for k in ("n_group", "topk_group"):
            if cfg.get(k) not in (None, 1):
                raise SystemExit(f"{k}={cfg.get(k)} is not supported (group routing)")
        if cfg.get("scoring_func", "sigmoid") != "sigmoid":
            raise SystemExit("only sigmoid routing is implemented")
        if cfg.get("n_shared_experts"):
            raise SystemExit("shared experts are not implemented (MiMo-V2 has none)")

    def first_swa(self) -> int:
        """Index of a sliding-window layer: the MTP blocks use that geometry
        (swa heads / KV heads / head dims, swa theta, window, sinks)."""
        for li, p in enumerate(self.pattern):
            if p == 1:
                return li
        raise SystemExit("no sliding-window layer: MTP geometry undefined")

    def geom(self, li: int) -> Geo:
        swa = self.pattern[li] == 1
        if swa:
            nq, nkv, hd, vd, theta, sink = (self.swa_nq, self.swa_nkv, self.swa_hd,
                                            self.swa_vd, self.swa_theta, self.swa_sink)
        else:
            nq, nkv, hd, vd, theta, sink = (self.nq, self.nkv, self.hd, self.vd,
                                            self.rope_theta, self.full_sink)
        rope = int(hd * self.partial)
        if rope % 2:
            raise SystemExit(f"odd rope dim {rope}")
        return Geo(swa, nq, nkv, hd, vd, rope, theta, sink, self.window if swa else None)


# --------------------------------------------------------------------------
# weight sources
# --------------------------------------------------------------------------
class DirSource:
    """A safetensors checkpoint directory (sharded with an index, or single)."""

    def __init__(self, d: str):
        from safetensors import safe_open

        self._open = safe_open
        self.dir = d
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.map = dict(json.load(open(idx))["weight_map"])
        else:
            self.map = {}
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".safetensors"):
                    with safe_open(os.path.join(d, fn), "pt") as f:
                        for k in f.keys():
                            self.map[k] = fn
        # The release index omits nothing, but a partial download may lack
        # the MTP file; MTP tensors are never read by the text oracle.
        self._files = {}

    def has(self, name):
        return name in self.map

    def get(self, name):
        fn = self.map.get(name)
        if fn is None:
            raise KeyError(name)
        if fn not in self._files:
            self._files[fn] = self._open(os.path.join(self.dir, fn), "pt")
        return self._files[fn].get_tensor(name)


class DictSource:
    """In-memory tensors under checkpoint names (toys, the HF self-check)."""

    def __init__(self, tensors: dict):
        self.t = tensors

    def has(self, name):
        return name in self.t

    def get(self, name):
        return self.t[name]


# --------------------------------------------------------------------------
# dequantization (exact)
# --------------------------------------------------------------------------
def cdiv(a, b):
    return -(-a // b)


def fp8_deq(w, sinv, row_index=None, block=FP8_BLOCK):
    """FP8 e4m3 [R, C] x per-block scale_inv [RB, CB] -> f32. `row_index`
    maps every stored row to its scale row (per-chunk tiling for qkv)."""
    R, C = w.shape
    if row_index is None:
        row_index = torch.arange(R) // block
    col_index = torch.arange(C) // block
    sinv = sinv.float()
    if int(row_index.max()) + 1 != sinv.shape[0] or int(col_index.max()) + 1 != sinv.shape[1]:
        raise ValueError(f"scale grid {tuple(sinv.shape)} does not tile weight {tuple(w.shape)}")
    return w.float() * sinv[row_index][:, col_index]


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def mxfp4_deq(packed, scale):
    """U8 [R, C/2] (low nibble = even element) x U8 E8M0 [R, C/32] -> f32."""
    R, half = packed.shape
    if scale.shape != (R, half * 2 // MX_BLOCK):
        raise ValueError(f"mxfp4 scale {tuple(scale.shape)} vs packed {tuple(packed.shape)}")
    if bool((scale == 255).any()):
        raise ValueError("E8M0 scale 255 (NaN) present")
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    vals = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).reshape(R, half * 2)
    sc = torch.pow(2.0, scale.float() - 127.0).repeat_interleave(MX_BLOCK, dim=1)
    return vals * sc


def qkv_rows(cfg: MimoCfg, li: int):
    g = cfg.geom(li)
    return g.nq * g.hd, g.nkv * g.hd, g.nkv * g.vd


def qkv_scale_rows(cfg: MimoCfg, li: int, n_scale_rows: int, block=FP8_BLOCK):
    """Scale-row index of every stored qkv row. Per-chunk tiling when the
    scale count says so (vLLM precedence), else one continuous grid."""
    q, k, v = qkv_rows(cfg, li)
    R, tp = q + k + v, cfg.ckpt_tp
    rpc = R // tp
    rows = torch.arange(R)
    if n_scale_rows == tp * cdiv(rpc, block):
        return (rows // rpc) * cdiv(rpc, block) + (rows % rpc) // block
    if n_scale_rows == cdiv(R, block):
        return rows // block
    raise ValueError(f"layer {li}: qkv scale rows {n_scale_rows}, expected "
                     f"{tp * cdiv(rpc, block)} (per chunk) or {cdiv(R, block)}")


def split_fused_qkv(cfg: MimoCfg, li: int, W, layout="chunked"):
    """Dequantized fused rows -> (Wq, Wk, Wv) in head order.

    chunked: stored as ckpt_tp chunks [Q_c | K_c | V_c]; de-interleave by
             concatenating Q_c over c, K_c over c, V_c over c (the release).
    contig:  one global [Q | K | V] (the HF modeling file's reading; WRONG
             for the release, kept for the layout A/B)."""
    q, k, v = qkv_rows(cfg, li)
    if W.shape[0] != q + k + v:
        raise ValueError(f"layer {li}: qkv has {W.shape[0]} rows, expected {q}+{k}+{v}")
    if layout == "contig":
        return W[:q], W[q:q + k], W[q + k:]
    tp = cfg.ckpt_tp
    if q % tp or k % tp or v % tp:
        raise ValueError(f"layer {li}: q/k/v rows {q}/{k}/{v} not divisible by ckpt_tp {tp}")
    qc, kc, vc = q // tp, k // tp, v // tp
    rpc = qc + kc + vc
    Q = torch.cat([W[c * rpc: c * rpc + qc] for c in range(tp)])
    K = torch.cat([W[c * rpc + qc: c * rpc + qc + kc] for c in range(tp)])
    V = torch.cat([W[c * rpc + qc + kc: (c + 1) * rpc] for c in range(tp)])
    return Q, K, V


def interleave_qkv(cfg: MimoCfg, li: int, Q, K, V):
    """Inverse of split_fused_qkv(chunked): (Q, K, V) -> stored chunk order."""
    tp = cfg.ckpt_tp
    qc, kc, vc = Q.shape[0] // tp, K.shape[0] // tp, V.shape[0] // tp
    parts = []
    for c in range(tp):
        parts += [Q[c * qc:(c + 1) * qc], K[c * kc:(c + 1) * kc], V[c * vc:(c + 1) * vc]]
    return torch.cat(parts)


def dense_weight(src, name, dt):
    """A projection weight in any of the release's storage forms."""
    w = src.get(name)
    if w.dtype == torch.uint8:
        return mxfp4_deq(w, src.get(name + "_scale")).to(dt)
    if w.dtype == torch.float8_e4m3fn:
        return fp8_deq(w, src.get(name + "_scale_inv")).to(dt)
    return w.to(dt)


# --------------------------------------------------------------------------
# layer weights
# --------------------------------------------------------------------------
class LayerW:
    def __init__(self, src, cfg: MimoCfg, li: int, dt=torch.float32, full_layout="chunked",
                 swa_layout="chunked", prefix="model.layers."):
        p = f"{prefix}{li}."
        self.li, self.src, self.dt, self.p = li, src, dt, p
        self.g = g = cfg.geom(li)
        a = p + "self_attn."
        if src.has(a + "qkv_proj.weight"):
            w = src.get(a + "qkv_proj.weight")
            if w.dtype == torch.float8_e4m3fn:
                sinv = src.get(a + "qkv_proj.weight_scale_inv")
                W = fp8_deq(w, sinv, qkv_scale_rows(cfg, li, sinv.shape[0]))
            else:
                W = w.float()
            layout = swa_layout if g.swa else full_layout
            wq, wk, wv = split_fused_qkv(cfg, li, W, layout)
        else:
            wq = dense_weight(src, a + "q_proj.weight", torch.float32)
            wk = dense_weight(src, a + "k_proj.weight", torch.float32)
            wv = dense_weight(src, a + "v_proj.weight", torch.float32)
        self.wq, self.wk, self.wv = wq.to(dt), wk.to(dt), wv.to(dt)
        self.wo = dense_weight(src, a + "o_proj.weight", dt)
        want = {"q": g.nq * g.hd, "k": g.nkv * g.hd, "v": g.nkv * g.vd}
        got = {"q": self.wq.shape[0], "k": self.wk.shape[0], "v": self.wv.shape[0]}
        if want != got or self.wo.shape[1] != g.nq * g.vd:
            raise ValueError(f"layer {li}: q/k/v rows {got} != {want} or o_proj {tuple(self.wo.shape)}")
        self.sink = src.get(a + "attention_sink_bias").to(dt) if g.sink else None
        self.n_in = src.get(p + "input_layernorm.weight").to(dt)
        self.n_post = src.get(p + "post_attention_layernorm.weight").to(dt)
        self.moe = cfg.moe[li] and cfg.ne is not None
        if self.moe:
            self.gw = src.get(p + "mlp.gate.weight").to(dt)
            self.gb = src.get(p + "mlp.gate.e_score_correction_bias").to(dt)
        else:
            self.dg = dense_weight(src, p + "mlp.gate_proj.weight", dt)
            self.du = dense_weight(src, p + "mlp.up_proj.weight", dt)
            self.dd = dense_weight(src, p + "mlp.down_proj.weight", dt)

    def expert(self, e):
        # One expert at a time, never cached: a release layer's 256 experts are
        # 25.8 GB in f32, and every expert is visited once per layer anyway.
        q = f"{self.p}mlp.experts.{e}."
        return tuple(dense_weight(self.src, q + f"{n}_proj.weight", self.dt)
                     for n in ("gate", "up", "down"))


# --------------------------------------------------------------------------
# math
# --------------------------------------------------------------------------
def rms(x, w, eps):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rope_tables(g: Geo, pos, dt):
    """cos/sin [S, rope] exactly as MiMoV2RotaryEmbedding (f32 tables),
    or in f64 when the whole run is f64."""
    tdt = torch.float64 if dt == torch.float64 else torch.float32
    inv = 1.0 / (g.theta ** (torch.arange(0, g.rope, 2, dtype=torch.int64).to(tdt) / g.rope))
    f = pos.to(tdt)[:, None] * inv[None, :]
    emb = torch.cat([f, f], -1)
    return emb.cos().to(dt), emb.sin().to(dt)


def apply_rope(t, cos, sin, rope):
    r, n = t[..., :rope], t[..., rope:]
    h = rope // 2
    rot = torch.cat([-r[..., h:], r[..., :h]], -1)
    return torch.cat([r * cos + rot * sin, n], -1)


def attention(x, L: LayerW, pos, cfg: MimoCfg):
    """x [B, S, H] (already normed) -> self_attn output [B, S, H]."""
    g = L.g
    B, S, _ = x.shape
    q = (x @ L.wq.T).view(B, S, g.nq, g.hd).transpose(1, 2)
    k = (x @ L.wk.T).view(B, S, g.nkv, g.hd).transpose(1, 2)
    v = (x @ L.wv.T).view(B, S, g.nkv, g.vd).transpose(1, 2)
    if cfg.vscale is not None:
        v = v * cfg.vscale
    cos, sin = rope_tables(g, pos, x.dtype)
    q, k = apply_rope(q, cos, sin, g.rope), apply_rope(k, cos, sin, g.rope)
    rep = g.nq // g.nkv
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    s = (q @ k.transpose(-1, -2)) * (g.hd ** -0.5)          # [B, nq, S, S]
    i = torch.arange(S)[:, None]
    j = torch.arange(S)[None, :]
    ok = j <= i
    if g.window is not None:
        ok = ok & (j > i - g.window)
    s = s.masked_fill(~ok, float("-inf"))
    if L.sink is not None:
        s = torch.cat([s, L.sink.view(1, g.nq, 1, 1).expand(B, g.nq, S, 1)], -1)
    s = s - s.amax(-1, keepdim=True)
    pr = torch.softmax(s, -1)
    if L.sink is not None:
        pr = pr[..., :-1]
    o = (pr @ v).transpose(1, 2).reshape(B, S, g.nq * g.vd)
    return o @ L.wo.T


def route(x, L: LayerW, cfg: MimoCfg):
    """x [T, H] -> (idx [T, k] sorted by biased score, weights [T, k],
    margin [T] = k-th minus (k+1)-th biased score)."""
    sc = torch.sigmoid(x @ L.gw.T)
    choice = sc + L.gb
    k = cfg.topk
    top = torch.topk(choice, min(k + 1, choice.shape[-1]), dim=-1)
    ix = top.indices[:, :k]
    margin = (top.values[:, k - 1] - top.values[:, k]) if choice.shape[-1] > k else \
        torch.full((x.shape[0],), float("inf"), dtype=x.dtype)
    w = sc.gather(1, ix)
    if k > 1 and cfg.norm_topk:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return ix, w * cfg.rsf, margin


def mlp(x, L: LayerW, cfg: MimoCfg, picks=None):
    """x [T, H] (normed) -> [T, H]."""
    silu = torch.nn.functional.silu
    if not L.moe:
        return (silu(x @ L.dg.T) * (x @ L.du.T)) @ L.dd.T
    ix, w, margin = route(x, L, cfg)
    if picks is not None:
        picks.append((ix, w, margin))
    out = torch.zeros_like(x)
    for e in ix.unique().tolist():
        tok, slot = (ix == e).nonzero(as_tuple=True)
        gw, uw, dw = L.expert(e)
        xe = x[tok]
        ye = (silu(xe @ gw.T) * (xe @ uw.T)) @ dw.T
        out.index_add_(0, tok, ye * w[tok, slot][:, None])
    return out


def layer_forward(h, L: LayerW, pos, cfg: MimoCfg, sub=None, picks=None):
    """h [B, S, H] -> [B, S, H]. `sub` collects (attn_out, ffn_out)."""
    a = attention(rms(h, L.n_in, cfg.eps), L, pos, cfg)
    h = h + a
    B, S, H = h.shape
    f = mlp(rms(h, L.n_post, cfg.eps).reshape(B * S, H), L, cfg, picks).view(B, S, H)
    if sub is not None:
        sub.append((a, f))
    return h + f


class Oracle:
    def __init__(self, src, cfg: MimoCfg, dt=torch.float32, full="chunked", swa="chunked",
                 verbose=True):
        self.src, self.cfg, self.dt = src, cfg, dt
        self.full, self.swa, self.verbose = full, swa, verbose

    def layer(self, li):
        return LayerW(self.src, self.cfg, li, self.dt, self.full, self.swa)

    def embed(self, ids):
        emb = self.src.get("model.embed_tokens.weight")
        return emb[torch.as_tensor(ids)].to(self.dt)

    def forward(self, ids, on_layer=None, want_sub=False, logits_rows=None, layers=None):
        """ids: [S] or [B, S] ints. Returns (final hidden pre-norm [B,S,H],
        logits fn). on_layer(li, h, sub, picks) is called after each layer."""
        ids = torch.as_tensor(ids)
        if ids.dim() == 1:
            ids = ids[None]
        t0 = time.time()
        h = self.embed(ids)
        pos = torch.arange(ids.shape[1])
        for li in range(self.cfg.NL if layers is None else layers):
            L = self.layer(li)
            sub = [] if want_sub else None
            picks = []
            h = layer_forward(h, L, pos, self.cfg, sub, picks)
            if on_layer:
                on_layer(li, h, sub[0] if sub else None, picks[0] if picks else None)
            del L
            if self.verbose:
                print(f"  layer {li} done {time.time() - t0:.0f}s |h| {h.norm(dim=-1).mean():.2f}",
                      file=sys.stderr, flush=True)
        return h

    def head(self):
        return (self.src.get("model.norm.weight").to(self.dt),
                self.src.get("lm_head.weight").to(self.dt))

    def logits(self, h, head=None):
        nw, lm = head or self.head()
        return rms(h, nw, self.cfg.eps) @ lm.T


# --------------------------------------------------------------------------
# MTP draft layers (model.mtp.layers.K.*, model_mtp.safetensors)
# --------------------------------------------------------------------------
# HF modeling_mimo_v2.py ignores the MTP head. The forward below is the one
# vLLM (vllm/model_executor/models/mimo_v2_mtp.py, MiMoV2MTPLayer.forward)
# and SGLang (sglang/srt/models/mimo_v2_nextn.py, MiMoV2ModelNextN.forward)
# run for ONE layer:
#     u = eh_proj(cat[enorm(embed(tok)), hnorm(hid)])      embedding FIRST
#     u = u + o_proj(swa_attn(input_layernorm(u)))         SWA geometry, sinks
#     u = u + mlp(pre_mlp_layernorm(u))                    dense FFN
#     logits = lm_head(final_layernorm(u))                 shared lm_head
# `hid` is the backbone's POST-final-norm hidden (both servers' target model
# returns model.norm(h)). How the three layers chain differs between servers,
# hence the variants of `cmd_mtp`.
class MtpLayer:
    def __init__(self, src, cfg: MimoCfg, k: int, dt=torch.float32):
        p = f"model.mtp.layers.{k}."
        li = cfg.first_swa()
        L = LayerW.__new__(LayerW)
        L.li, L.src, L.dt, L.p = li, src, dt, p
        L.g = g = cfg.geom(li)
        a = p + "self_attn."
        w = src.get(a + "qkv_proj.weight")
        if w.dtype == torch.float8_e4m3fn:
            sinv = src.get(a + "qkv_proj.weight_scale_inv")
            W = fp8_deq(w, sinv, qkv_scale_rows(cfg, li, sinv.shape[0]))
        else:
            W = w.float()
        wq, wk, wv = split_fused_qkv(cfg, li, W, "chunked")
        L.wq, L.wk, L.wv = wq.to(dt), wk.to(dt), wv.to(dt)
        L.wo = dense_weight(src, a + "o_proj.weight", dt)
        L.sink = src.get(a + "attention_sink_bias").to(dt) if g.sink else None
        L.n_in = src.get(p + "input_layernorm.weight").to(dt)
        L.n_post = src.get(p + "pre_mlp_layernorm.weight").to(dt)
        L.moe = False
        L.dg = dense_weight(src, p + "mlp.gate_proj.weight", dt)
        L.du = dense_weight(src, p + "mlp.up_proj.weight", dt)
        L.dd = dense_weight(src, p + "mlp.down_proj.weight", dt)
        self.L = L
        self.k = k
        self.enorm = src.get(p + "enorm.weight").to(dt)
        self.hnorm = src.get(p + "hnorm.weight").to(dt)
        self.eh = dense_weight(src, p + "eh_proj.weight", dt)
        self.fnorm = src.get(p + "final_layernorm.weight").to(dt)

    def fuse(self, emb, hid, cfg):
        return torch.cat([rms(emb, self.enorm, cfg.eps), rms(hid, self.hnorm, cfg.eps)], -1) @ self.eh.T

    def forward(self, emb, hid, pos, cfg):
        """emb/hid [B, S, H] at positions `pos` (teacher-forced, causal over
        the S rows) -> (block output pre-final-norm, post-final-norm)."""
        y = layer_forward(self.fuse(emb, hid, cfg), self.L, pos, cfg)
        return y, rms(y, self.fnorm, cfg.eps)


def kv_rows(xn, L: LayerW, pos, cfg: MimoCfg):
    """Normed rows [n, H] at positions `pos` [n] -> rotated K [nkv, n, hd]
    and scaled V [nkv, n, vd]."""
    g = L.g
    n = xn.shape[0]
    k = (xn @ L.wk.T).view(n, g.nkv, g.hd).transpose(0, 1)
    v = (xn @ L.wv.T).view(n, g.nkv, g.vd).transpose(0, 1)
    if cfg.vscale is not None:
        v = v * cfg.vscale
    cos, sin = rope_tables(g, pos, xn.dtype)
    return apply_rope(k, cos, sin, g.rope), v


def attend_kv(xn, qpos, K, V, kpos, L: LayerW, cfg: MimoCfg):
    """Queries from normed rows xn [n, H] at qpos [n] over explicit keys
    K/V at kpos (key visible iff kp <= qp and kp > qp - window) with the
    layer's sinks -> o_proj output [n, H]. Same math as `attention` for any
    key set, so a row can attend to rows computed in other passes."""
    g = L.g
    n = xn.shape[0]
    q = (xn @ L.wq.T).view(n, g.nq, g.hd).transpose(0, 1)
    cos, sin = rope_tables(g, qpos, xn.dtype)
    q = apply_rope(q, cos, sin, g.rope)
    rep = g.nq // g.nkv
    Kr, Vr = K.repeat_interleave(rep, 0), V.repeat_interleave(rep, 0)
    s = (q @ Kr.transpose(-1, -2)) * (g.hd ** -0.5)              # [nq, n, m]
    ok = kpos[None, :] <= qpos[:, None]
    if g.window is not None:
        ok = ok & (kpos[None, :] > qpos[:, None] - g.window)
    s = s.masked_fill(~ok[None], float("-inf"))
    if L.sink is not None:
        s = torch.cat([s, L.sink.view(g.nq, 1, 1).expand(g.nq, n, 1)], -1)
    s = s - s.amax(-1, keepdim=True)
    pr = torch.softmax(s, -1)
    if L.sink is not None:
        pr = pr[..., :-1]
    o = (pr @ Vr).transpose(0, 1).reshape(n, g.nq * g.vd)
    return o @ L.wo.T


def mtp_rows(m: MtpLayer, x, qpos, ctxK, ctxV, ctxpos, cfg: MimoCfg):
    """Fused MTP inputs x [n, H] (after eh_proj) at qpos, causal among
    themselves and over a context of earlier K/V rows -> (pre-norm out,
    post-final-norm out, own K, own V)."""
    L = m.L
    xn = rms(x, L.n_in, cfg.eps)
    Kn, Vn = kv_rows(xn, L, qpos, cfg)
    K = torch.cat([ctxK, Kn], 1) if ctxK is not None else Kn
    V = torch.cat([ctxV, Vn], 1) if ctxV is not None else Vn
    kpos = torch.cat([ctxpos, qpos]) if ctxpos is not None else qpos
    h = x + attend_kv(xn, qpos, K, V, kpos, L, cfg)
    h = h + mlp(rms(h, L.n_post, cfg.eps), L, cfg)
    return h, rms(h, m.fnorm, cfg.eps), Kn, Vn


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def load_cfg(src_dir):
    return MimoCfg(json.load(open(os.path.join(src_dir, "config.json"))))


def tokenizer(src_dir):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(src_dir, "tokenizer.json"))


def encode_raw(src_dir, text):
    """The RAW token stream, no special tokens: what `cortiq ppl` encodes
    (MiMo has bos_token null, so --tokens mode adds nothing either)."""
    return tokenizer(src_dir).encode(text, add_special_tokens=False).ids


def parse_positions(spec, n):
    """'all' | 'last' | '0,1,127-129,-1' (negative counts from the end)."""
    if spec in (None, "", "all"):
        return list(range(n))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part == "last":
            out.append(n - 1)
        elif re.fullmatch(r"-?\d+", part):
            p = int(part)
            out.append(p + n if p < 0 else p)
        elif re.fullmatch(r"\d+-\d+", part):
            a, b = map(int, part.split("-"))
            out += list(range(a, b + 1))
        else:
            raise SystemExit(f"bad position spec {part!r}")
    return sorted({p for p in out if 0 <= p < n})


def ppl_window_offsets(n, windows, window_len):
    """Mirror of cortiq-cli PplWindows::offsets (main.rs): stride =
    (n - len - 1) / windows, offsets k * stride, k = 0..windows-1."""
    if n <= window_len + 1:
        raise SystemExit(f"corpus has {n} tokens < window_len+2 = {window_len + 2}")
    stride = (n - window_len - 1) // windows
    if stride <= 0:
        raise SystemExit(f"{windows} windows of {window_len} do not fit in {n} tokens")
    return [k * stride for k in range(windows)]


def write_f32(path, t):
    with open(path, "wb") as f:
        f.write(t.detach().to(torch.float32).contiguous().numpy().tobytes())


class Dumper:
    """Writes the raw per-position files as layers stream past."""

    def __init__(self, out, positions, sub=False):
        self.out, self.positions, self.sub = out, positions, sub
        self.trace = []   # (pos, li, experts, weights, margin)
        os.makedirs(out, exist_ok=True)

    def __call__(self, li, h, sub, picks):
        for p in self.positions:
            write_f32(os.path.join(self.out, f"p{p:06d}_l{li:02d}.f32"), h[0, p])
            if self.sub and sub is not None:
                write_f32(os.path.join(self.out, f"p{p:06d}_l{li:02d}_attn.f32"), sub[0][0, p])
                write_f32(os.path.join(self.out, f"p{p:06d}_l{li:02d}_ffn.f32"), sub[1][0, p])
        if picks is not None:
            ix, w, m = picks
            for t in range(ix.shape[0]):
                self.trace.append((t, li, ix[t].tolist(), [float(x) for x in w[t]], float(m[t])))

    def finish(self, ids, h, logits_fn, logits_positions, extra=None):
        S = h.shape[1]
        for p in logits_positions:
            write_f32(os.path.join(self.out, f"p{p:06d}_logits.f32"), logits_fn(h[0, p:p + 1])[0])
        last = logits_fn(h[0, S - 1:S])[0]
        write_f32(os.path.join(self.out, "logit_dump.f32"), torch.cat([h[0, S - 1], last]))
        self.trace.sort(key=lambda r: (r[0], r[1]))
        with open(os.path.join(self.out, "moe_trace.txt"), "w") as f:
            for t, li, ex, _, _ in self.trace:
                f.write(f"{li}:{','.join(map(str, ex))}\n")
        with open(os.path.join(self.out, "picks.jsonl"), "w") as f:
            for t, li, ex, w, m in self.trace:
                f.write(json.dumps({"pos": t, "layer": li, "experts": ex, "weights": w,
                                    "margin": m}) + "\n")
        meta = {"ids": list(map(int, ids)), "positions": self.positions,
                "logits_positions": logits_positions, "hidden": int(h.shape[-1]),
                "vocab": int(last.shape[0]), "sub": self.sub,
                "min_route_margin": min((r[4] for r in self.trace), default=None)}
        meta.update(extra or {})
        json.dump(meta, open(os.path.join(self.out, "meta.json"), "w"), indent=1)
        return last


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_ppl(a):
    cfg = load_cfg(a.src)
    ids = encode_raw(a.src, open(a.text).read())
    orc = Oracle(DirSource(a.src), cfg, full=a.full, swa=a.swa)
    if a.windows:
        offs = ppl_window_offsets(len(ids), a.windows, a.window_len)
        seqs = torch.tensor([ids[o:o + a.window_len] for o in offs])
        print(f"windows: {len(offs)} x {a.window_len} tokens at stride "
              f"{offs[1] if len(offs) > 1 else 0} over {len(ids)} tokens", flush=True)
    else:
        seq = ([a.bos] if a.bos is not None else []) + ids
        seqs = torch.tensor([seq[: a.tokens]])
    h = orc.forward(seqs)
    head = orc.head()
    nll, cnt = 0.0, 0
    for b in range(seqs.shape[0]):
        lg = orc.logits(h[b, :-1], head).double()
        nll += float(torch.nn.functional.cross_entropy(lg, seqs[b, 1:], reduction="sum"))
        cnt += seqs.shape[1] - 1
    ppl = math.exp(nll / cnt)
    mode = (f"windows={a.windows}x{a.window_len}" if a.windows else f"tokens={seqs.shape[1]}")
    print(f"PPL full={a.full} swa={a.swa} {mode} scored={cnt}: {ppl:.4f}  (nll_sum {nll:.6f})")


def cmd_dump(a):
    cfg = load_cfg(a.src)
    ids = json.load(open(a.ids))
    positions = parse_positions(a.positions, len(ids))
    lpos = parse_positions(a.logits_positions, len(ids))
    orc = Oracle(DirSource(a.src), cfg, full=a.full, swa=a.swa)
    d = Dumper(a.out, positions, a.sub)
    h = orc.forward(ids, on_layer=d, want_sub=a.sub)
    head = orc.head()
    last = d.finish(ids, h, lambda x: orc.logits(x, head), lpos,
                    {"src": a.src, "full": a.full, "swa": a.swa, "dtype": "float32"})
    print(f"dumped {len(positions)} positions x {cfg.NL} layers to {a.out}; "
          f"last-position top5 {last.topk(5).indices.tolist()}")


def cmd_gen(a):
    cfg = load_cfg(a.src)
    tk = tokenizer(a.src)
    ids = json.load(open(a.ids)) if a.ids else tk.encode(a.prompt, add_special_tokens=False).ids
    orc = Oracle(DirSource(a.src), cfg, full=a.full, swa=a.swa, verbose=False)
    head = orc.head()
    for _ in range(a.n):
        h = orc.forward(ids)
        lg = orc.logits(h[0, -1:], head)[0]
        top2 = lg.topk(2)
        ids.append(int(top2.indices[0]))
        print(f"{int(top2.indices[0])} margin {float(top2.values[0] - top2.values[1]):.4f} "
              f"{tk.decode(ids)!r}", flush=True)


PAD_ID = 151643
MTP_VARIANTS = ("A", "A_pre", "B_pre", "B_post", "C", "C_pre")
MTP_VARIANT_DOC = {
    "A": "layer k at round start t: (embed x[t+k+1], POST-norm backbone h[t]); each layer its own "
         "KV cache at positions <= t (SGLang multi-layer MTP, MiMoV2MTP not in its chain list)",
    "A_pre": "A with the PRE-final-norm backbone hidden (control)",
    "B_pre": "DeepSeek-V3 chain: layer k at position t takes layer k-1's PRE-final-norm output at t",
    "B_post": "chain with layer k-1's POST-final-norm output",
    "C": "vLLM: layer 0 only, recursive: step s at position t+s with (x[t+s+1], its own "
         "post-final-norm output of step s-1)",
    "C_pre": "C with the pre-final-norm recursion (control)",
}


def load_seqs(path, src_dir):
    """--seqs JSON: [{"name", "ids" | "text" | "text_file"+"tokens"[+"offset"],
    optional "continuation" (ids appended), "eval_from"}]."""
    tk = None
    out = []
    for s in json.load(open(path)):
        if "ids" in s:
            ids = [int(x) for x in s["ids"]]
        else:
            tk = tk or tokenizer(src_dir)
            text = s["text"] if "text" in s else open(s["text_file"]).read()
            ids = tk.encode(text, add_special_tokens=False).ids
            off = s.get("offset", 0)
            ids = ids[off:off + s["tokens"]] if "tokens" in s else ids[off:]
        ids = ids + [int(x) for x in s.get("continuation", [])]
        out.append({"name": s["name"], "ids": ids, "eval_from": int(s.get("eval_from", 0))})
    return out


def head_argmax(x, lm, chunk=64):
    """Rows [n, H] (post-norm) -> (argmax [n], top1-top2 margin [n])."""
    am, mg = [], []
    for i in range(0, x.shape[0], chunk):
        lg = x[i:i + chunk] @ lm.T
        t2 = lg.topk(2, dim=-1)
        am.append(t2.indices[:, 0])
        mg.append(t2.values[:, 0] - t2.values[:, 1])
    return torch.cat(am), torch.cat(mg)


def mtp_main_pass(orc: Oracle, seqs, cache):
    """Backbone pass over every sequence (right-padded into one batch: the
    causal mask keeps real rows exact). Cached by ids."""
    if cache and os.path.exists(cache):
        c = torch.load(cache)
        if c["ids"] == [s["ids"] for s in seqs]:
            print(f"main pass: cached {cache}", flush=True)
            return c
        print("main pass: cache holds other ids, recomputing", flush=True)
    S = max(len(s["ids"]) for s in seqs)
    batch = torch.tensor([s["ids"] + [PAD_ID] * (S - len(s["ids"])) for s in seqs])
    t0 = time.time()
    picks = {}   # MoE layer -> routed experts [B, S, k]

    def on_layer(li, hh, sub, p):
        if p is not None:
            picks[li] = p[0].view(batch.shape[0], S, -1).to(torch.int16).clone()

    h = orc.forward(batch, on_layer=on_layer)
    nw, lm = orc.head()
    c = {"ids": [s["ids"] for s in seqs], "h": [], "m": [], "margin": [], "picks": []}
    for b, s in enumerate(seqs):
        hb = h[b, :len(s["ids"])].clone()
        am, mg = head_argmax(rms(hb, nw, orc.cfg.eps), lm)
        c["h"].append(hb)
        c["m"].append(am)
        c["margin"].append(mg)
        c["picks"].append({li: pk[b, :len(s["ids"])] for li, pk in picks.items()})
    print(f"main pass: {len(seqs)} x {S} positions in {time.time() - t0:.0f}s", flush=True)
    if cache:
        torch.save(c, cache)
    return c


def mtp_drafts(mtps, cfg, emb_w, nw, lm, ids, h, variant, K):
    """Teacher-forced drafts D[k][t] (k < K) of one variant: the token the
    round starting at t (backbone at t, pending x[t+1]) drafts for position
    t+k+2. Positions whose inputs run past the sequence are -1."""
    S = len(ids)
    ids_t = torch.tensor(ids)
    g_post = rms(h, nw, cfg.eps)
    back = h if variant == "A_pre" else g_post
    D = torch.full((K, S), -1, dtype=torch.long)
    W = cfg.geom(cfg.first_swa()).window or S
    if variant in ("A", "A_pre", "B_pre", "B_post"):
        prev = None
        for k in range(K):
            n = S - k - 1
            if n <= 0:
                break
            emb = emb_w[ids_t[k + 1:k + 1 + n]].to(h.dtype)
            if variant.startswith("B") and k > 0:
                hid = prev[:n]
            else:
                hid = back[:n]
            y, yn = mtps[k].forward(emb[None], hid[None], torch.arange(n), cfg)
            D[k, :n] = head_argmax(yn[0], lm)[0]
            prev = y[0] if variant == "B_pre" else yn[0]
        return D
    # C / C_pre: layer 0 recursion, per round start t
    m0 = mtps[0]
    n = S - 1
    x0 = m0.fuse(emb_w[ids_t[1:]].to(h.dtype), g_post[:n], cfg)
    pos0 = torch.arange(n)
    y0, yn0, K0, V0 = mtp_rows(m0, x0, pos0, None, None, None, cfg)
    D[0, :n] = head_argmax(yn0, lm)[0]
    for t in range(S - 2):
        ctxK, ctxV, ctxpos = K0[:, :t + 1], V0[:, :t + 1], pos0[:t + 1]
        lo = max(0, t + 1 - W)
        ctxK, ctxV, ctxpos = ctxK[:, lo:], ctxV[:, lo:], ctxpos[lo:]
        rec = (yn0 if variant == "C" else y0)[t]
        for s in range(1, K):
            if t + s + 1 >= S:
                break
            x = m0.fuse(emb_w[ids_t[t + s + 1:t + s + 2]].to(h.dtype), rec[None], cfg)
            qp = torch.tensor([t + s])
            y, yn, Kn, Vn = mtp_rows(m0, x, qp, ctxK, ctxV, ctxpos, cfg)
            D[s, t] = head_argmax(yn, lm)[0][0]
            ctxK, ctxV, ctxpos = torch.cat([ctxK, Kn], 1), torch.cat([ctxV, Vn], 1), torch.cat([ctxpos, qp])
            rec = (yn if variant == "C" else y)[0]
    return D


def mtp_stats(D, ids, m, lo, K):
    """Acceptance of a draft table against the backbone's own argmax m
    (m[j] = argmax of the logits at j-1, i.e. the greedy token FOR j).
    Round starts t in [lo, S-K-2]. `cons` = the teacher-forced inputs are
    the backbone's greedy path, so the round is exactly what speculative
    greedy decoding would do there."""
    S = len(ids)
    ids_t = torch.tensor(ids)
    m_for = torch.full((S + 1,), -2, dtype=torch.long)
    m_for[1:S + 1] = m            # m_for[j] = greedy token for position j
    hi = S - K - 1
    ts = list(range(max(lo, 0), hi))
    out = {"rounds": len(ts)}
    if not ts:
        return out
    acc = torch.zeros(len(ts), K, dtype=torch.bool)
    marg = torch.zeros(len(ts), K, dtype=torch.bool)
    cons = torch.zeros(len(ts), K, dtype=torch.bool)
    for r, t in enumerate(ts):
        ok, c = True, True
        for k in range(K):
            hit = int(D[k, t]) == int(m_for[t + k + 2])
            marg[r, k] = hit
            ok = ok and hit
            acc[r, k] = ok
            c = c and int(ids_t[t + k + 1]) == int(m_for[t + k + 1])
            cons[r, k] = c
    out["marginal"] = [float(marg[:, k].float().mean()) for k in range(K)]
    out["chain"] = [float(acc[:, k].float().mean()) for k in range(K)]
    out["conditional"] = [float(acc[:, 0].float().mean())] + [
        float(acc[:, k].sum()) / max(1, int(acc[:, k - 1].sum())) for k in range(1, K)]
    ex = cons[:, K - 1]
    out["exact_rounds"] = int(ex.sum())
    out["exact_chain"] = [float(acc[ex, k].float().mean()) if ex.any() else None for k in range(K)]
    # greedy walk: rounds of depth K' from `lo` (exact where consistent)
    walk = {}
    for kk in range(1, K + 1):
        t, steps, toks = ts[0], 0, 0
        while t < hi:
            r = t - ts[0]
            a = 0
            for k in range(kk):
                if bool(acc[r, k]):
                    a = k + 1
                else:
                    break
            steps += 1
            toks += a + 1
            t += a + 1
        walk[f"K{kk}"] = {"steps": steps, "tokens": toks, "tokens_per_step": toks / max(1, steps)}
    out["walk"] = walk
    return out


def cmd_mtp(a):
    cfg = load_cfg(a.src)
    src = DirSource(a.src)
    seqs = load_seqs(a.seqs, a.src)
    orc = Oracle(src, cfg, full=a.full, swa=a.swa)
    c = mtp_main_pass(orc, seqs, a.cache)
    if a.main_only:
        return
    nw, lm = orc.head()
    emb_w = src.get("model.embed_tokens.weight")
    K = a.depth
    mtps = [MtpLayer(src, cfg, k) for k in range(K)]
    variants = a.variants.split(",")
    tk = tokenizer(a.src)
    report = {"variants": {v: MTP_VARIANT_DOC[v] for v in variants}, "depth": K, "seqs": {}}
    rows_out = open(a.out + ".positions.jsonl", "w") if a.out else None
    for si, s in enumerate(seqs):
        ids, h, m = s["ids"], c["h"][si], c["m"][si]
        lo = s["eval_from"]
        cons_rate = float((torch.tensor(ids[lo + 1:]) == m[lo:len(ids) - 1]).float().mean()) \
            if len(ids) > lo + 1 else None
        rep = {"tokens": len(ids), "eval_from": lo, "greedy_consistency": cons_rate, "variants": {}}
        # Expert union of a verify batch: w consecutive rows route to how
        # many distinct experts per MoE layer (8 for one row). A K-draft
        # round verifies w = K+1 rows; its expert bytes scale with this.
        pk = c.get("picks", [None] * len(seqs))[si]
        if pk:
            union = {}
            for w in range(1, K + 2):
                tot, cnt = 0, 0
                for li, p in pk.items():
                    for t in range(lo, len(ids) - w + 1):
                        tot += int(torch.unique(p[t:t + w].reshape(-1)).numel())
                        cnt += 1
                union[f"w{w}"] = tot / max(1, cnt)
            rep["expert_union_per_layer"] = union
            print(f"[{s['name']}] experts per MoE layer for a {K + 1}-row window: "
                  + ", ".join(f"{k} {v:.2f}" for k, v in union.items()), flush=True)
        tables = {}
        for v in variants:
            t0 = time.time()
            D = mtp_drafts(mtps, cfg, emb_w, nw, lm, ids, h, v, K)
            tables[v] = D
            st = mtp_stats(D, ids, m, lo, K)
            st["seconds"] = round(time.time() - t0, 1)
            rep["variants"][v] = st
            w = st.get("walk", {}).get(f"K{K}", {})
            print(f"[{s['name']}] {v:6s} marginal {['%.3f' % x for x in st.get('marginal', [])]} "
                  f"chain {['%.3f' % x for x in st.get('chain', [])]} "
                  f"exact {['%.3f' % x if x is not None else '-' for x in st.get('exact_chain', [])]} "
                  f"({st.get('exact_rounds')}/{st['rounds']} rounds) "
                  f"walk K{K} {w.get('tokens_per_step', 0):.2f} tok/step", flush=True)
        report["seqs"][s["name"]] = rep
        if rows_out:
            for t in range(lo, len(ids) - K - 1):
                rows_out.write(json.dumps({
                    "seq": s["name"], "t": t, "pending": ids[t + 1],
                    "pending_text": tk.decode([ids[t + 1]]),
                    "main_greedy": [int(m[t + k + 1]) for k in range(K)],
                    "teacher": [ids[t + k + 2] for k in range(K)],
                    "drafts": {v: [int(tables[v][k, t]) for k in range(K)] for v in variants},
                }) + "\n")
    if rows_out:
        rows_out.close()
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"wrote {a.out} and {a.out}.positions.jsonl")


def cmd_mtpcheck(a):
    """Self-check of the MTP helpers on a random tiny config: the explicit-KV
    path (`mtp_rows`, used by the recursive variants) equals the causal
    teacher-forced block (`MtpLayer.forward`), row by row, and the window
    and sinks are live."""
    torch.manual_seed(a.seed)
    c = tiny_cfg(2, 4, False)
    c["num_nextn_predict_layers"] = 2
    cfg = MimoCfg(c)
    H, g = cfg.H, cfg.geom(cfg.first_swa())
    sd = {}
    rows = g.nq * g.hd + g.nkv * (g.hd + g.vd)
    for k in range(2):
        p = f"model.mtp.layers.{k}."
        sd[p + "self_attn.qkv_proj.weight"] = torch.randn(rows, H) / math.sqrt(H)
        sd[p + "self_attn.o_proj.weight"] = torch.randn(H, g.nq * g.vd) / math.sqrt(g.nq * g.vd)
        sd[p + "self_attn.attention_sink_bias"] = 0.5 + torch.randn(g.nq)
        sd[p + "eh_proj.weight"] = torch.randn(H, 2 * H) / math.sqrt(2 * H)
        for n in ("enorm", "hnorm", "final_layernorm", "input_layernorm", "pre_mlp_layernorm"):
            sd[p + f"{n}.weight"] = 1.0 + 0.2 * torch.randn(H)
        I = c["intermediate_size"]
        sd[p + "mlp.gate_proj.weight"] = torch.randn(I, H) / math.sqrt(H)
        sd[p + "mlp.up_proj.weight"] = torch.randn(I, H) / math.sqrt(H)
        sd[p + "mlp.down_proj.weight"] = torch.randn(H, I) / math.sqrt(I)
    src = DictSource(sd)
    m = MtpLayer(src, cfg, 1)
    S = 3 * (g.window or 8)
    emb, hid = torch.randn(1, S, H), torch.randn(1, S, H)
    y, yn = m.forward(emb, hid, torch.arange(S), cfg)
    x = m.fuse(emb[0], hid[0], cfg)
    worst = 0.0
    K = V = P = None
    for t in range(S):
        yt, ynt, Kn, Vn = mtp_rows(m, x[t:t + 1], torch.tensor([t]), K, V, P, cfg)
        worst = max(worst, float((ynt[0] - yn[0, t]).abs().max()))
        K = Kn if K is None else torch.cat([K, Kn], 1)
        V = Vn if V is None else torch.cat([V, Vn], 1)
        P = torch.tensor([t]) if P is None else torch.cat([P, torch.tensor([t])])
    # all rows at once through the explicit path too
    ya, yna, _, _ = mtp_rows(m, x, torch.arange(S), None, None, None, cfg)
    worst = max(worst, float((yna - yn[0]).abs().max()))
    # sensitivity: sinks and window change the output
    m.L.sink = None
    _, yn_ns = m.forward(emb, hid, torch.arange(S), cfg)
    d_sink = float((yn_ns - yn).abs().max())
    ok = worst < a.tol and d_sink > 100 * a.tol
    print(f"MTPCHECK {'PASS' if ok else 'FAIL'}: explicit-KV vs causal block max|d| {worst:.3e} "
          f"(tol {a.tol:g}); no-sink change {d_sink:.3e}")
    sys.exit(0 if ok else 1)


def segment_scale_rows(cfg: MimoCfg, li: int, block=FP8_BLOCK):
    """The WRONG alternative kept for the probe: every Q_c / K_c / V_c segment
    starts a new scale block (the scratch oracle before 2026-09-24). Same
    27-per-chunk count on full layers, different rows."""
    q, k, v = qkv_rows(cfg, li)
    tp = cfg.ckpt_tp
    rows, b = [], 0
    for _ in range(tp):
        for s in (q // tp, k // tp, v // tp):
            for o in range(0, s, block):
                rows += [b] * min(block, s - o)
                b += 1
    return torch.tensor(rows)


def tile_max(codes, row_index, block=FP8_BLOCK):
    """max |code| of every (scale row, column block) tile."""
    a = codes.float().abs()
    R, C = a.shape
    nrb, ncb = int(row_index.max()) + 1, cdiv(C, block)
    colmax = torch.zeros(R, ncb)
    for cb in range(ncb):
        colmax[:, cb] = a[:, cb * block:(cb + 1) * block].amax(1)
    out = torch.zeros(nrb, ncb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out.index_reduce_(0, row_index, colmax, "amax", include_self=True)
    return out


def cmd_tiles(a):
    """Static proof of the FP8 scale tiling: block quantization maps each
    tile's amax to the e4m3 maximum 448, so under the TRUE row -> scale-row
    mapping every tile's max |code| is 448."""
    cfg = load_cfg(a.src)
    src = DirSource(a.src)
    bad_ok, bad_alt, tiles = 0, 0, 0
    for li in range(cfg.NL):
        n = f"model.layers.{li}.self_attn.qkv_proj.weight"
        w = src.get(n)
        sinv = src.get(n + "_scale_inv")
        m_ok = tile_max(w, qkv_scale_rows(cfg, li, sinv.shape[0]))
        m_alt = tile_max(w, segment_scale_rows(cfg, li))
        k_ok, k_alt = int((m_ok < 448).sum()), int((m_alt < 448).sum())
        bad_ok += k_ok
        bad_alt += k_alt
        tiles += m_ok.numel()
        print(f"layer {li:2d} {'SWA ' if cfg.geom(li).swa else 'full'} qkv {tuple(w.shape)} scales "
              f"{tuple(sinv.shape)}: tiles below 448: per-chunk {k_ok}/{m_ok.numel()}, "
              f"per-segment {k_alt}/{m_alt.numel()}", flush=True)
    for nm in ("gate_proj", "up_proj", "down_proj"):
        for li in range(cfg.NL):
            if cfg.moe[li]:
                continue
            n = f"model.layers.{li}.mlp.{nm}.weight"
            w = src.get(n)
            m = tile_max(w, torch.arange(w.shape[0]) // FP8_BLOCK)
            print(f"layer {li} dense {nm} {tuple(w.shape)}: tiles below 448: {int((m < 448).sum())}/{m.numel()}")
            bad_ok += int((m < 448).sum())
            tiles += m.numel()
    print(f"TILES {'PASS' if bad_ok == 0 else 'FAIL'}: {bad_ok} of {tiles} tiles below 448 under the loader's "
          f"mapping; the per-segment alternative leaves {bad_alt} qkv tiles below 448")
    sys.exit(0 if bad_ok == 0 else 1)


def cmd_pt2raw(a):
    ids = json.load(open(a.ids))
    positions = parse_positions(a.positions, len(ids))
    os.makedirs(a.out, exist_ok=True)
    li = 0
    while os.path.exists(os.path.join(a.inp, f"h{li:02d}.pt")):
        h = torch.load(os.path.join(a.inp, f"h{li:02d}.pt"))
        assert h.shape[0] == len(ids), (h.shape, len(ids))
        for p in positions:
            write_f32(os.path.join(a.out, f"p{p:06d}_l{li:02d}.f32"), h[p])
        li += 1
    lpath = os.path.join(a.inp, "logits.pt")
    if os.path.exists(lpath):
        lg = torch.load(lpath)
        for p in parse_positions(a.logits_positions, len(ids)):
            write_f32(os.path.join(a.out, f"p{p:06d}_logits.f32"), lg[p])
    json.dump({"ids": ids, "positions": positions, "layers": li, "from": a.inp},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"converted {li} layers x {len(positions)} positions -> {a.out}")


# --------------------------------------------------------------------------
# selfcheck: oracle math vs the HF modules (split layout, eager, fp32)
# --------------------------------------------------------------------------
def import_hf(hf_dir):
    import importlib
    import shutil
    import tempfile

    root = tempfile.mkdtemp(prefix="mimo_hf_")
    pkg = os.path.join(root, "mimo_hf_pkg")
    os.makedirs(pkg)
    for fn in ("configuration_mimo_v2.py", "modeling_mimo_v2.py"):
        shutil.copy(os.path.join(hf_dir, fn), pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    sys.path.insert(0, root)
    conf = importlib.import_module("mimo_hf_pkg.configuration_mimo_v2")
    model = importlib.import_module("mimo_hf_pkg.modeling_mimo_v2")
    return conf, model


def tiny_cfg(nkv_full, nkv_swa, full_sink, vscale=0.707):
    return dict(
        vocab_size=320, hidden_size=128, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=nkv_full, head_dim=48, v_head_dim=32,
        swa_num_attention_heads=8, swa_num_key_value_heads=nkv_swa, swa_head_dim=48,
        swa_v_head_dim=32, sliding_window=8, sliding_window_size=8,
        add_swa_attention_sink_bias=True, add_full_attention_sink_bias=full_sink,
        hybrid_layer_pattern=[0, 1, 1, 0], moe_layer_freq=[0, 1, 1, 1],
        partial_rotary_factor=0.334, rope_theta=1e7, swa_rope_theta=1e4,
        rope_parameters={"partial_rotary_factor": 0.334, "rope_theta": 1e7,
                         "rope_type": "default", "type": "default"},
        n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, n_group=1,
        topk_group=1, norm_topk_prob=True, scoring_func="sigmoid", topk_method="noaux_tc",
        routed_scaling_factor=None, attention_value_scale=vscale, layernorm_epsilon=1e-6,
        attention_projection_layout="split", max_position_embeddings=4096,
        tie_word_embeddings=False, attention_bias=False, hidden_act="silu",
    )


def _rope_adjacent(t, cos, sin, rope):
    # Mutation: GPT-J pairing (2i, 2i+1) instead of NeoX halves.
    r, n = t[..., :rope], t[..., rope:]
    x1, x2 = r[..., 0::2], r[..., 1::2]
    c, s = cos[..., : rope // 2], sin[..., : rope // 2]
    out = torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], -1).flatten(-2)
    return torch.cat([out, n], -1)


MUTATIONS = {
    # name: (cfg mutator, module-level function overrides)
    "window+1": (lambda c: setattr(c, "window", c.window + 1), {}),
    "window-1": (lambda c: setattr(c, "window", c.window - 1), {}),
    "no sink": (lambda c: setattr(c, "swa_sink", False), {}),
    "no value scale": (lambda c: setattr(c, "vscale", None), {}),
    "swa theta = full theta": (lambda c: setattr(c, "swa_theta", c.rope_theta), {}),
    "rope pairs adjacent": (None, {"apply_rope": _rope_adjacent}),
    "rope on all dims": (lambda c: setattr(c, "partial", 1.0), {}),
}


def selfcheck_one(conf, modeling, c, S, seed, mutate=False):
    torch.manual_seed(seed)
    config = conf.MiMoV2Config(**c)
    config._attn_implementation = "eager"
    model = modeling.MiMoV2Model(config).float().eval()
    # Randomize every parameter so no path is identity or zero.
    for name, p in model.named_parameters():
        if name.endswith("layernorm.weight") or name == "norm.weight":
            p.data = 1.0 + 0.2 * torch.randn_like(p)
        elif name.endswith("attention_sink_bias"):
            p.data = 0.5 + torch.randn_like(p)
        elif name.endswith("e_score_correction_bias"):
            p.data = 0.1 * torch.randn_like(p)
        elif name == "embed_tokens.weight":
            p.data = torch.randn_like(p)
        elif name.endswith("mlp.gate.weight"):
            p.data = 2.0 * torch.randn_like(p) / math.sqrt(p.shape[1])
        else:
            p.data = torch.randn_like(p) / math.sqrt(p.shape[1])
    lm = torch.randn(c["vocab_size"], c["hidden_size"]) / math.sqrt(c["hidden_size"])
    ids = torch.randint(0, c["vocab_size"], (1, S))

    got = {}
    hooks = [layer.register_forward_hook(lambda m, i, o, li=li: got.__setitem__(li, o[0] if isinstance(o, tuple) else o))
             for li, layer in enumerate(model.layers)]
    out = model(input_ids=ids, use_cache=False).last_hidden_state
    for hk in hooks:
        hk.remove()
    hf_logits = out @ lm.T

    sd = {"model." + k: v.detach().clone() for k, v in model.state_dict().items()}
    sd["lm_head.weight"] = lm
    cfg = MimoCfg(c)
    orc = Oracle(DictSource(sd), cfg, torch.float32, verbose=False)
    mine = {}
    h = orc.forward(ids, on_layer=lambda li, hh, s, p: mine.__setitem__(li, hh.clone()))
    my_logits = orc.logits(h)
    worst = 0.0
    rows = []
    for li in range(cfg.NL):
        g = cfg.geom(li)
        err = float((mine[li] - got[li]).abs().max())
        worst = max(worst, err)
        rows.append(f"    layer {li} {'SWA ' if g.swa else 'full'} nkv={g.nkv} sink={int(g.sink)} "
                    f"{'moe  ' if cfg.moe[li] else 'dense'} max|h|={float(got[li].abs().max()):8.3f} "
                    f"max|dh|={err:.3e}")
    lerr = float((my_logits - hf_logits).abs().max())
    worst = max(worst, lerr)
    rows.append(f"    logits max|logit|={float(hf_logits.abs().max()):.3f} max|dlogit|={lerr:.3e} "
                f"top1 equal at {int((my_logits.argmax(-1) == hf_logits.argmax(-1)).sum())}/{S}")
    if mutate:
        # Sensitivity: every deliberate convention error must be far above tol.
        mod = sys.modules[__name__]
        for name, (fn, patch) in MUTATIONS.items():
            mc = MimoCfg(c)
            if fn:
                fn(mc)
            saved = {k: getattr(mod, k) for k in patch}
            for k, v in patch.items():
                setattr(mod, k, v)
            try:
                mh = {}
                Oracle(DictSource(sd), mc, torch.float32, verbose=False).forward(
                    ids, on_layer=lambda li, hh, s, p: mh.__setitem__(li, hh.clone()))
            finally:
                for k, v in saved.items():
                    setattr(mod, k, v)
            err = max(float((mh[li] - got[li]).abs().max()) for li in range(cfg.NL))
            rows.append(f"    mutation {name:24s} max|dh|={err:.3e}  (must be >> tol)")
    return worst, rows


def cmd_selfcheck(a):
    conf, modeling = import_hf(a.hf_dir)
    variants = [
        ("nkv full 2 / swa 4, sinks on SWA only (release-like)", tiny_cfg(2, 4, False)),
        ("nkv full 4 / swa 2, sinks on both layer kinds", tiny_cfg(4, 2, True)),
        ("nkv full 2 / swa 4, no value scale", tiny_cfg(2, 4, False, vscale=None)),
    ]
    worst_all = 0.0
    for i, (title, c) in enumerate(variants):
        worst, rows = selfcheck_one(conf, modeling, c, a.positions, a.seed + i, mutate=(i == 0))
        worst_all = max(worst_all, worst)
        print(f"[{title}] {a.positions} positions, window {c['sliding_window']}: max abs err {worst:.3e}")
        print("\n".join(rows))
    ok = worst_all < a.tol
    print(f"SELFCHECK {'PASS' if ok else 'FAIL'}: worst max abs err {worst_all:.3e} (tol {a.tol:g})")
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--src", default=DEFAULT_SRC, help="HF checkpoint dir (default $MIMO_SRC or /root/mimo/src)")
        p.add_argument("--full", default="chunked", choices=["chunked", "shard4", "contig"],
                       help="qkv reading on full-attention layers (shard4 = chunked)")
        p.add_argument("--swa", default="chunked", choices=["chunked", "shard4", "contig"],
                       help="qkv reading on SWA layers")
        p.add_argument("--threads", type=int, default=24)

    p = sp.add_parser("ppl")
    common(p)
    p.add_argument("--text", required=True)
    p.add_argument("--tokens", type=int, default=256, help="prefix length (ignored with --windows)")
    p.add_argument("--windows", type=int, help="score N evenly spaced windows (cortiq ppl --windows)")
    p.add_argument("--window-len", type=int, default=512)
    p.add_argument("--bos", type=int, help="prepend this id in --tokens mode (MiMo: none)")
    p = sp.add_parser("dump")
    common(p)
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--positions", default="all", help="'all' or '0,1,127-129,-1'")
    p.add_argument("--logits-positions", default="all")
    p.add_argument("--sub", action="store_true", help="also dump attn / ffn outputs per layer")
    p = sp.add_parser("gen")
    common(p)
    p.add_argument("--prompt")
    p.add_argument("--ids")
    p.add_argument("--n", type=int, default=16)
    p = sp.add_parser("tiles")
    common(p)
    p = sp.add_parser("mtp", help="MTP draft acceptance vs the backbone's greedy (teacher-forced)")
    common(p)
    p.add_argument("--seqs", required=True, help="JSON list of sequences (see load_seqs)")
    p.add_argument("--cache", help="torch file caching the backbone pass (ids-keyed)")
    p.add_argument("--out", help="report JSON (+ .positions.jsonl with per-position drafts)")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--variants", default=",".join(MTP_VARIANTS))
    p.add_argument("--main-only", action="store_true", help="only run and cache the backbone pass")
    p = sp.add_parser("mtpcheck")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--threads", type=int, default=8)
    p = sp.add_parser("pt2raw")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--positions", default="all")
    p.add_argument("--logits-positions", default="all")
    p.add_argument("--threads", type=int, default=8)
    p = sp.add_parser("selfcheck")
    p.add_argument("--hf-dir", default=DEFAULT_SRC, help="dir with modeling_mimo_v2.py + configuration_mimo_v2.py")
    p.add_argument("--positions", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    for attr in ("full", "swa"):
        if getattr(a, attr, None) == "shard4":
            setattr(a, attr, "chunked")
    {"ppl": cmd_ppl, "dump": cmd_dump, "gen": cmd_gen, "pt2raw": cmd_pt2raw,
     "selfcheck": cmd_selfcheck, "tiles": cmd_tiles, "mtp": cmd_mtp,
     "mtpcheck": cmd_mtpcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
