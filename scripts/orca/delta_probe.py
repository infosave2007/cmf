#!/usr/bin/env python3
"""Is OrcaCyber-Nano a low-rank edit of Qwen3.8-27B? delta_probe.py GGUF QWEN_DIR [TENSOR ...]

For each named GGUF matrix: dequantize (gguf-py), take the same bf16 matrix of
the base, D = G - B. Quantization error alone is a full-rank, noise-like D whose
top singular values sit near the Marchenko-Pastur edge sigma*(sqrt m + sqrt n);
a LoRA / abliteration edit adds a few singular values far above it.
Prints rel |D|/|B|, the top singular values over that edge, and the energy share
of the top 8/32/128 components.
"""
import json, os, struct, sys, time
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def st_bf16(path, name):
    """One BF16 tensor of a safetensors shard as float32 (numpy has no bf16)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    e = hdr[name]
    assert e["dtype"] == "BF16", e["dtype"]
    a, b = e["data_offsets"]
    raw = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + a, shape=((b - a) // 2,))
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])

gpath, qdir = sys.argv[1], sys.argv[2]
names = sys.argv[3:] or [
    "blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.3.attn_output.weight", "blk.3.attn_q.weight",
    "blk.32.ffn_down.weight", "blk.32.ffn_gate.weight", "blk.63.ffn_down.weight", "blk.63.attn_v.weight",
    "token_embd.weight",
]
HF = {"ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj", "ffn_gate": "mlp.gate_proj",
      "attn_output": "self_attn.o_proj", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
      "attn_v": "self_attn.v_proj"}
idx = json.load(open(os.path.join(qdir, "model.safetensors.index.json")))["weight_map"]
r = GGUFReader(gpath)
tmap = {t.name: t for t in r.tensors}


def hf_name(g):
    if g == "token_embd.weight":
        return "model.language_model.embed_tokens.weight"
    if g == "output.weight":
        return "lm_head.weight"
    _, l, k, _ = g.split(".")
    return f"model.language_model.layers.{l}.{HF[k]}.weight"


for g in names:
    t0 = time.time()
    t = tmap[g]
    G = dequantize(t.data, t.tensor_type).astype(np.float32).reshape([int(x) for x in reversed(t.shape)])
    h = hf_name(g)
    B = st_bf16(os.path.join(qdir, idx[h]), h)
    assert G.shape == B.shape, (g, G.shape, B.shape)
    D = G - B
    rel = np.linalg.norm(D) / np.linalg.norm(B)
    m, n = D.shape
    sigma = np.linalg.norm(D) / np.sqrt(m * n)
    edge = sigma * (np.sqrt(m) + np.sqrt(n))
    small = D if m <= n else D.T
    ev = np.linalg.eigvalsh(small.astype(np.float64) @ small.T.astype(np.float64))[::-1]
    sv = np.sqrt(np.clip(ev, 0, None))
    e = ev / ev.sum()
    over = int((sv > 1.5 * edge).sum())
    corr = float((G * B).sum() / (np.linalg.norm(G) * np.linalg.norm(B)))
    print(f"{g} {t.tensor_type.name} {m}x{n}: rel {rel:.4f} corr {corr:.6f} | sv0/edge {sv[0]/edge:.2f} "
          f"sv1 {sv[1]/edge:.2f} sv4 {sv[4]/edge:.2f} sv16 {sv[16]/edge:.2f} sv64 {sv[64]/edge:.2f} | "
          f">1.5 edge: {over} | energy top8 {e[:8].sum():.3f} top32 {e[:32].sum():.3f} top128 {e[:128].sum():.3f} "
          f"| {time.time()-t0:.0f} s", flush=True)
print("PROBE_DONE", flush=True)
