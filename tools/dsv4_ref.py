#!/usr/bin/env python3
"""A reference DeepSeek-V4 decode, transcribed from `inference/model.py`.

    python3 tools/mk_dsv4_toy.py /tmp/reftoy
    cortiq convert --model /tmp/reftoy --quant f16 --output /tmp/reftoy.cmf
    CMF_DSV4_DUMP=/tmp/rust.jsonl cortiq run /tmp/reftoy.cmf --prompt abcde --max-tokens 2
    python3 tools/dsv4_ref.py /tmp/reftoy /tmp/rust.jsonl

The upstream forward cannot be run here: its attention is a tilelang kernel
that wants CUDA, and the model it was written for is 304B. This transcribes
the same arithmetic in NumPy at toy scale, so the port can be diffed against
it layer by layer — the numerical parity the port has never had.

It follows model.py, NOT our Rust. Where the two disagree, that is the
finding. Deliberate omissions, each argued: the Hadamard rotation and the
FP4/FP8 activation simulation are absent, because the rotation is orthogonal
and lands on both sides of the same dot product (it cancels) and the
quantization only loses precision we are trying to measure against.
"""
import json
import math
import struct
import sys

import numpy as np

# Pinned exactly as the loader pins them — inference/config.json, not
# config.json, is where the release states these.
COMPRESS_ROPE_THETA = 160_000.0
ROPE_THETA = 10_000.0
YARN_FACTOR = 16.0
YARN_ORIGINAL = 65_536
BETA_FAST, BETA_SLOW = 32.0, 1.0


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        blob = f.read()
    out = {}
    for name, m in head.items():
        if name == "__metadata__":
            continue
        a, b = m["data_offsets"]
        out[name] = np.frombuffer(blob[a:b], dtype=np.float32).reshape(m["shape"]).astype(np.float64)
    return out


def rope_freqs(dim, base, yarn):
    """precompute_freqs_cis, but returning the per-pair frequencies."""
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    if yarn:
        def corr(rot):
            return dim * math.log(YARN_ORIGINAL / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr(BETA_FAST)), 0)
        high = min(math.ceil(corr(BETA_SLOW)), dim - 1)
        lin = (np.arange(dim // 2, dtype=np.float64) - low) / max(high - low, 1e-3)
        smooth = 1 - np.clip(lin, 0, 1)
        freqs = freqs / YARN_FACTOR * (1 - smooth) + freqs * smooth
    return freqs


def rope_tail(v, freqs, pos, rd, inverse=False):
    """apply_rotary_emb on the LAST rd dims, half-split pairing."""
    out = v.copy()
    base = len(v) - rd
    th = pos * freqs[:rd // 2]
    s, c = np.sin(th), np.cos(th)
    if inverse:
        s = -s
    # view_as_complex(unflatten(-1, (-1, 2))): adjacent pairs.
    a = v[base:base + rd:2]
    b = v[base + 1:base + rd:2]
    out[base:base + rd:2] = a * c - b * s
    out[base + 1:base + rd:2] = a * s + b * c
    return out


def rms(v, eps):
    return v / math.sqrt(float((v * v).mean()) + eps)


def sinkhorn(mixes, scale, base, hc, iters, eps):
    pre = 1 / (1 + np.exp(-(mixes[:hc] * scale[0] + base[:hc]))) + eps
    post = 2 / (1 + np.exp(-(mixes[hc:2 * hc] * scale[1] + base[hc:2 * hc])))
    comb = (mixes[2 * hc:] * scale[2] + base[2 * hc:]).reshape(hc, hc)
    e = np.exp(comb - comb.max(axis=-1, keepdims=True))
    comb = e / e.sum(axis=-1, keepdims=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


class Ref:
    def __init__(self, w, cfg):
        self.w, self.c = w, cfg
        self.hc = cfg["hc_mult"]
        self.dim = cfg["hidden_size"]
        self.hd = cfg["head_dim"]
        # qk_rope_head_dim, not a guess from head_dim: the rope tail is 64 of
        # 512 in the release and the ratio is not derivable from head_dim.
        self.rd = cfg.get("qk_rope_head_dim", min(64, self.hd))
        self.nh = cfg["num_attention_heads"]
        self.eps = cfg["rms_norm_eps"]
        self.nl = cfg["num_hidden_layers"]
        self.window = cfg["sliding_window"]
        self.limit = cfg.get("swiglu_limit", 0.0)
        self.topk = cfg["num_experts_per_tok"]
        self.nexp = cfg["n_routed_experts"]
        self.rscale = cfg["routed_scaling_factor"]
        self.nhash = cfg["num_hash_layers"]
        self.win_kv = [[] for _ in range(self.nl)]
        self.cmp_kv = [[] for _ in range(self.nl)]
        self.ix_kv = [[] for _ in range(self.nl)]
        self.pend = [[] for _ in range(self.nl)]
        self.prev = [None] * self.nl
        self.ix_pend = [[] for _ in range(self.nl)]
        self.ix_prev = [None] * self.nl
        self.f_cmp = rope_freqs(self.rd, COMPRESS_ROPE_THETA, True)
        self.f_win = rope_freqs(self.rd, ROPE_THETA, False)
        self.picked = []

    def has(self, name):
        return name in self.w

    # ── one compressor step; returns the folded entry when the window closes
    def compress(self, pfx, li, x, pos, pend, prev, freqs):
        wkv, wgate = self.w[f"{pfx}.wkv.weight"], self.w[f"{pfx}.wgate.weight"]
        ape, norm = self.w[f"{pfx}.ape"], self.w[f"{pfx}.norm.weight"]
        width = wkv.shape[0]
        ratio = ape.size // width
        overlap = ratio == 4
        ew = width // 2 if overlap else width
        kv, sc = wkv @ x, wgate @ x
        if overlap:
            sc = sc + ape.reshape(ratio, -1)[pos % ratio]
        pend.append((kv, sc))
        if len(pend) < ratio:
            return None, prev
        if overlap:
            slots_kv, slots_sc = [], []
            for t in range(ratio):
                if prev is not None:
                    slots_kv.append(prev[t][0][:ew]); slots_sc.append(prev[t][1][:ew])
                else:
                    slots_kv.append(np.zeros(ew)); slots_sc.append(np.full(ew, -np.inf))
            for t in range(ratio):
                slots_kv.append(pend[t][0][ew:]); slots_sc.append(pend[t][1][ew:])
            K = np.stack(slots_kv); S = np.stack(slots_sc)
            new_prev = list(pend)
        else:
            K = np.stack([k for k, _ in pend])
            S = np.stack([s for _, s in pend]) + ape.reshape(ratio, -1)
            new_prev = prev
        m = S.max(axis=0)
        e = np.where(np.isfinite(S), np.exp(S - m), 0.0)
        folded = (K * (e / np.maximum(e.sum(axis=0), 1e-30))).sum(axis=0)
        folded = rms(folded, self.eps) * norm
        folded = rope_tail(folded, freqs, pos + 1 - ratio, self.rd)
        pend.clear()
        return folded, new_prev

    def attention(self, li, x, pos):
        p = f"layers.{li}"
        cmp_pfx = f"{p}.attn.compressor"
        has_cmp = self.has(f"{cmp_pfx}.wkv.weight")
        freqs = self.f_cmp if has_cmp else self.f_win

        qr = rms(self.w[f"{p}.attn.wq_a.weight"] @ x, self.eps) * self.w[f"{p}.attn.q_norm.weight"]
        q = self.w[f"{p}.attn.wq_b.weight"] @ qr
        heads = []
        for h in range(self.nh):
            hq = q[h * self.hd:(h + 1) * self.hd]
            heads.append(rope_tail(rms(hq, self.eps), freqs, pos, self.rd))
        kv = rms(self.w[f"{p}.attn.wkv.weight"] @ x, self.eps) * self.w[f"{p}.attn.kv_norm.weight"]
        kv = rope_tail(kv, freqs, pos, self.rd)

        if has_cmp:
            e, self.prev[li] = self.compress(cmp_pfx, li, x, pos, self.pend[li], self.prev[li], freqs)
            if e is not None:
                self.cmp_kv[li].append(e)
        ix_pfx = f"{p}.attn.indexer"
        if self.has(f"{ix_pfx}.wq_b.weight"):
            e, self.ix_prev[li] = self.compress(
                f"{ix_pfx}.compressor", li, x, pos, self.ix_pend[li], self.ix_prev[li], freqs)
            if e is not None:
                self.ix_kv[li].append(e)

        self.win_kv[li].append(kv)
        if len(self.win_kv[li]) > self.window:
            self.win_kv[li] = self.win_kv[li][-self.window:]
        cache = list(self.win_kv[li])
        nwin = len(cache)
        picked = []
        if self.cmp_kv[li]:
            if self.has(f"{ix_pfx}.wq_b.weight") and self.ix_kv[li]:
                iw = self.w[f"{ix_pfx}.weights_proj.weight"]
                ih = iw.shape[0]
                qi = self.w[f"{ix_pfx}.wq_b.weight"] @ qr
                idim = qi.size // ih
                qi = [rope_tail(qi[h * idim:(h + 1) * idim], freqs, pos, self.rd) for h in range(ih)]
                hw = (iw @ x) * (idim ** -0.5) * (ih ** -0.5)
                n = min(len(self.ix_kv[li]), len(self.cmp_kv[li]))
                score = np.zeros(n)
                for h in range(ih):
                    for t in range(n):
                        score[t] += max(float(qi[h] @ self.ix_kv[li][t]), 0.0) * hw[h]
                picked = list(np.argsort(-score)[:min(self.c["index_topk"], n)])
            else:
                picked = list(range(len(self.cmp_kv[li])))
            cache += [self.cmp_kv[li][i] for i in picked]

        scale = self.hd ** -0.5
        import os
        dbg = os.environ.get("REF_DEBUG") == "1" and li == 0 and pos <= 1
        sink = self.w[f"{p}.attn.attn_sink"]
        attn = np.zeros(self.nh * self.hd)
        C = np.stack(cache)
        for h in range(self.nh):
            s = (C @ heads[h]) * scale
            m = max(s.max(), sink[h])
            e = np.exp(s - m)
            o = (C * e[:, None]).sum(axis=0) / (e.sum() + math.exp(sink[h] - m))
            if dbg:
                print(f"    [ref] поз {pos} голова {h}: позиций={len(cache)} score={[round(float(x),4) for x in s]} "
                      f"sink={sink[h]:.4f} вес={float(e.sum()/(e.sum()+math.exp(sink[h]-m))):.4f} "
                      f"|q|={np.linalg.norm(heads[h]):.3f} |kv|={np.linalg.norm(C[0]):.3f}")
            attn[h * self.hd:(h + 1) * self.hd] = rope_tail(o, freqs, pos, self.rd, inverse=True)

        wo_a, wo_b = self.w[f"{p}.attn.wo_a.weight"], self.w[f"{p}.attn.wo_b.weight"]
        groups = self.c["o_groups"]
        per = attn.size // groups
        lora = wo_a.shape[0] // groups
        mid = np.concatenate([wo_a[g * lora:(g + 1) * lora] @ attn[g * per:(g + 1) * per]
                              for g in range(groups)])
        return wo_b @ mid

    def moe(self, li, x, tok):
        p = f"layers.{li}"
        raw = self.w[f"{p}.ffn.gate.weight"] @ x
        sc = np.sqrt(np.log1p(np.exp(-np.abs(raw))) + np.maximum(raw, 0))
        if self.has(f"{p}.ffn.gate.tid2eid"):
            idx = self.w[f"{p}.ffn.gate.tid2eid"][tok].astype(int).tolist()
        else:
            shift = sc + self.w[f"{p}.ffn.gate.bias"] if self.has(f"{p}.ffn.gate.bias") else sc
            idx = list(np.argsort(-shift)[:self.topk])
        self.picked.append([int(i) for i in idx])
        wts = np.array([sc[i] for i in idx])
        wts = wts / wts.sum() * self.rscale
        out = np.zeros(self.dim)
        for e, i in zip(idx, range(len(idx))):
            out += self.expert(f"{p}.ffn.experts.{e}", x, wts[i])
        return out + self.expert(f"{p}.ffn.shared_experts", x, 1.0)

    def expert(self, pfx, x, weight):
        g = self.w[f"{pfx}.w1.weight"] @ x
        u = self.w[f"{pfx}.w3.weight"] @ x
        if self.limit > 0:
            u = np.clip(u, -self.limit, self.limit)
            g = np.minimum(g, self.limit)
        return self.w[f"{pfx}.w2.weight"] @ (g / (1 + np.exp(-g)) * u * weight)

    def block(self, state, fn, base, scale, norm, body):
        flat = state.reshape(-1)
        mixes = (fn @ flat) / math.sqrt(float((flat * flat).mean()) + self.eps)
        pre, post, comb = sinkhorn(mixes, scale, base, self.hc, self.c["hc_sinkhorn_iters"],
                                   self.c["hc_eps"])
        folded = (pre[:, None] * state).sum(axis=0)
        out = body(rms(folded, self.eps) * norm)
        return post[:, None] * out[None, :] + np.einsum("kj,kd->jd", comb, state)

    def forward(self, tok, pos):
        emb = self.w["embed.weight"][tok]
        state = np.tile(emb, (self.hc, 1))
        layers = []
        self.picked = []
        for li in range(self.nl):
            p = f"layers.{li}"
            state = self.block(state, self.w[f"{p}.hc_attn_fn"], self.w[f"{p}.hc_attn_base"],
                               self.w[f"{p}.hc_attn_scale"], self.w[f"{p}.attn_norm.weight"],
                               lambda h: self.attention(li, h, pos))
            layers.append(state.reshape(-1).copy())
            state = self.block(state, self.w[f"{p}.hc_ffn_fn"], self.w[f"{p}.hc_ffn_base"],
                               self.w[f"{p}.hc_ffn_scale"], self.w[f"{p}.ffn_norm.weight"],
                               lambda h: self.moe(li, h, tok))
            layers.append(state.reshape(-1).copy())
        flat = state.reshape(-1)
        mixes = (self.w["hc_head_fn"] @ flat) / math.sqrt(float((flat * flat).mean()) + self.eps)
        pre = 1 / (1 + np.exp(-(mixes * self.w["hc_head_scale"][0] + self.w["hc_head_base"]))) \
            + self.c["hc_eps"]
        h = (pre[:, None] * state).sum(axis=0)
        h = rms(h, self.eps) * self.w["norm.weight"]
        return emb, layers, h, self.w["head.weight"] @ h, self.picked


def main():
    toy, dump = sys.argv[1], sys.argv[2]
    cfg = json.load(open(f"{toy}/config.json"))
    w = read_safetensors(f"{toy}/model.safetensors")
    ref = Ref(w, cfg)
    rust = [json.loads(l) for l in open(dump)]

    # Feed the reference the port's OWN attention input, so only the body
    # can account for a difference.
    if all("attn_io" in r for r in rust):
        print("=== только тело внимания, на входе порта ===")
        # The caches must advance with the tokens, exactly as the port's do —
        # resetting them per record leaves the reference with a one-entry
        # window while the port has many, which compares nothing.
        for rec in rust:
            io = rec["attn_io"]
            for li in range(ref.nl):
                x = np.asarray(io[2 * li], dtype=np.float64)
                got = np.asarray(io[2 * li + 1], dtype=np.float64)
                want = ref.attention(li, x, rec["pos"])
                d = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12)
                print(f"  поз {rec['pos']:>3} слой {li}: {d:.3e}"
                      f"   |порт|={np.linalg.norm(got):.4f} |эталон|={np.linalg.norm(want):.4f}"
                      f"   cos={float(got @ want / max(np.linalg.norm(got) * np.linalg.norm(want), 1e-12)):+.4f}")
        print()

    ref = Ref(w, cfg)   # fresh caches for the end-to-end pass
    print(f"{'токен':>6} {'поз':>4} {'место':<12} {'отн.расхождение':>16}")
    worst = (0.0, "")
    for rec in rust:
        emb, layers, head, logits, picked = ref.forward(rec["tok"], rec["pos"])
        if "experts" in rec and rec["experts"] != picked:
            for li, (a, b) in enumerate(zip(rec["experts"], picked)):
                if a != b:
                    print(f"  ⚠ слой {li}: выбраны РАЗНЫЕ эксперты — порт {a}, эталон {b}")
        def cmp(name, a, b):
            nonlocal worst
            a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            d = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
            if d > worst[0]:
                worst = (d, f"токен {rec['tok']} поз {rec['pos']} {name}")
            return d
        de = cmp("embed", rec["embed"], emb)
        print(f"{rec['tok']:>6} {rec['pos']:>4} {'embed':<12} {de:>16.3e}")
        for k, (r, m) in enumerate(zip(rec["layers"], layers)):
            half = "внимание" if k % 2 == 0 else "эксперты"
            name = f"{k // 2}·{half}"
            print(f"{'':>6} {'':>4} {name:<14} {cmp(name, r, m):>16.3e}")
        print(f"{'':>6} {'':>4} {'head':<12} {cmp('head', rec['head'], head):>16.3e}")
        print(f"{'':>6} {'':>4} {'logits':<12} {cmp('logits', rec['logits'], logits):>16.3e}")
    print(f"\nхудшее расхождение: {worst[0]:.3e} — {worst[1]}")


if __name__ == "__main__":
    main()
