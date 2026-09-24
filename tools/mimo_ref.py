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
     "selfcheck": cmd_selfcheck, "tiles": cmd_tiles}[a.cmd](a)


if __name__ == "__main__":
    main()
