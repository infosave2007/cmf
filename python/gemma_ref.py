#!/usr/bin/env python3
"""Numpy reference for the Gemma-2 encoder forward (textenc.rs).

    python3 python/gemma_ref.py <text_encoder_dir> <out_dir>

Dumps token ids (u32) and the final-normed hidden states (f32 LE) for
tests/textenc_parity.rs. Pure numpy + stdlib.
"""
import json
import struct
import sys

import numpy as np


def load_shards(d):
    idx = json.load(open(f"{d}/model.safetensors.index.json"))
    out = {}
    for shard in sorted(set(idx["weight_map"].values())):
        with open(f"{d}/{shard}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            blob = f.read()
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            dt = meta["dtype"]
            if dt == "F32":
                # zero-copy view into the shard blob — no astype copy
                arr = np.frombuffer(blob, dtype=np.float32,
                                    count=(e - s) // 4, offset=s)
            elif dt == "BF16":
                raw = np.frombuffer(blob, dtype=np.uint16,
                                    count=(e - s) // 2, offset=s)
                arr = (raw.astype(np.uint32) << 16).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(blob, dtype=np.float16,
                                    count=(e - s) // 2, offset=s).astype(np.float32)
            else:
                raise SystemExit(f"dtype {dt}")
            out[name] = arr.reshape(meta["shape"])
    return out


def rms_norm(x, w, eps):
    inv = 1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
    return ((x.astype(np.float64) * inv).astype(np.float32)) * (1.0 + w)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def rope(x, theta):
    n, heads, hd = x.shape
    i = np.arange(hd // 2)
    freq = 1.0 / theta ** (2.0 * i / hd)
    ang = np.arange(n)[:, None] * freq[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    a, b = x[..., : hd // 2], x[..., hd // 2 :]
    return np.concatenate(
        [a * cos[:, None, :] - b * sin[:, None, :], a * sin[:, None, :] + b * cos[:, None, :]],
        axis=-1,
    ).astype(np.float32)


def main():
    d, out_dir = sys.argv[1], sys.argv[2]
    cfg = json.load(open(f"{d}/config.json"))
    t = load_shards(d)
    # Some exports (the Lumina one) drop the "model." prefix.
    if "model.embed_tokens.weight" not in t:
        t = {f"model.{k}": v for k, v in t.items()}
    hs, nh, nkv, hd = (
        cfg["hidden_size"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
    )
    eps = cfg["rms_norm_eps"]
    cap = cfg["attn_logit_softcapping"]
    scale = 1.0 / np.sqrt(cfg["query_pre_attn_scalar"])
    ids = np.array([2, 106, 1645, 108, 6176, 3855, 235265, 107], dtype=np.uint32)
    n = len(ids)
    h = t["model.embed_tokens.weight"][ids] * np.sqrt(np.float32(hs))
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}"
        xn = rms_norm(h, t[f"{p}.input_layernorm.weight"], eps)
        q = (xn @ t[f"{p}.self_attn.q_proj.weight"].T).reshape(n, nh, hd)
        k = (xn @ t[f"{p}.self_attn.k_proj.weight"].T).reshape(n, nkv, hd)
        v = (xn @ t[f"{p}.self_attn.v_proj.weight"].T).reshape(n, nkv, hd)
        q, k = rope(q, cfg["rope_theta"]), rope(k, cfg["rope_theta"])
        hpk = nh // nkv
        outs = np.zeros((n, nh, hd), dtype=np.float32)
        for hh in range(nh):
            kv = hh // hpk
            s = (q[:, hh] @ k[:, kv].T) * scale
            s = cap * np.tanh(s / cap)
            mask = np.triu(np.full((n, n), -np.inf), 1)
            s = s + mask
            s -= s.max(-1, keepdims=True)
            e = np.exp(s)
            a = e / e.sum(-1, keepdims=True)
            outs[:, hh] = a @ v[:, kv]
        proj = outs.reshape(n, nh * hd) @ t[f"{p}.self_attn.o_proj.weight"].T
        h = h + rms_norm(proj, t[f"{p}.post_attention_layernorm.weight"], eps)
        xn = rms_norm(h, t[f"{p}.pre_feedforward_layernorm.weight"], eps)
        g = xn @ t[f"{p}.mlp.gate_proj.weight"].T
        u = xn @ t[f"{p}.mlp.up_proj.weight"].T
        dn = (gelu_tanh(g) * u) @ t[f"{p}.mlp.down_proj.weight"].T
        h = h + rms_norm(dn, t[f"{p}.post_feedforward_layernorm.weight"], eps)
    h = rms_norm(h, t["model.norm.weight"], eps)
    ids.tofile(f"{out_dir}/gemma_ids.bin")
    h.astype(np.float32).tofile(f"{out_dir}/gemma_ref.bin")
    print(f"ids {ids.tolist()} -> hidden {h.shape}; |h| mean {np.abs(h).mean():.4f}")


if __name__ == "__main__":
    main()
