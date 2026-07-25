#!/usr/bin/env python3
"""Numpy reference for the Lumina2 Next-DiT forward (dit.rs).

    python3 python/nextdit_ref.py <transformer_dir> <out_dir>

Runs one denoising forward on deterministic inputs (seeded rng latent
16x16x16, 8 caption tokens, t=0.7) over the real Lumina-Image 2.0
transformer weights and dumps latent/caption/output f32 LE blobs for
tests/dit_parity.rs. Semantics mirror diffusers Lumina2Transformer2DModel:
plain-w RMSNorm, per-head qk-norm, 3-axis complex-interleaved RoPE,
AdaLN with tanh gates, LayerNorm(eps 1e-6, no affine) in norm_out.
Pure numpy + stdlib.
"""
import json
import mmap
import struct
import sys

import numpy as np

HID, NH, NKV, HD = 2304, 24, 8, 96
AXES_DIM, PATCH, EPS = (32, 32, 32), 2, 1e-5
C, H, W, CAP_N, T = 16, 16, 16, 8, 0.7


def load_shards(d):
    idx = json.load(open(f"{d}/diffusion_pytorch_model.safetensors.index.json"))
    out = {}
    for shard in sorted(set(idx["weight_map"].values())):
        # mmap: a plain read() short-reads on multi-GB shards on macOS,
        # and the mapping is zero-copy anyway.
        with open(f"{d}/{shard}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        base = 8 + n
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            if meta["dtype"] != "F32":
                raise SystemExit(f"dtype {meta['dtype']}")
            out[name] = np.frombuffer(mm, dtype=np.float32,
                                      count=(e - s) // 4, offset=base + s).reshape(meta["shape"])
    return out


def rms(x, w, eps=EPS):
    inv = 1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
    return (x * inv).astype(np.float32) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_table(pos_ids):
    """[n,3] int -> complex128 [n, 48] (16 freqs per axis, θ=10000)."""
    n = len(pos_ids)
    tab = np.zeros((n, sum(AXES_DIM) // 2), dtype=np.complex128)
    o = 0
    for a, d in enumerate(AXES_DIM):
        f = 1.0 / 10000.0 ** (np.arange(0, d, 2, dtype=np.float64) / d)
        ang = pos_ids[:, a : a + 1].astype(np.float64) * f[None, :]
        tab[:, o : o + d // 2] = np.cos(ang) + 1j * np.sin(ang)
        o += d // 2
    return tab


def rope_apply(x, tab):
    """x [n, heads, 96], pairs interleaved (even=re, odd=im)."""
    z = x[..., 0::2].astype(np.float64) + 1j * x[..., 1::2].astype(np.float64)
    z = z * tab[:, None, :]
    out = np.empty_like(x)
    out[..., 0::2] = z.real.astype(np.float32)
    out[..., 1::2] = z.imag.astype(np.float32)
    return out


def attention(t, x, pfx, tab):
    n = x.shape[0]
    q = (x @ t[f"{pfx}.attn.to_q.weight"].T).reshape(n, NH, HD)
    k = (x @ t[f"{pfx}.attn.to_k.weight"].T).reshape(n, NKV, HD)
    v = (x @ t[f"{pfx}.attn.to_v.weight"].T).reshape(n, NKV, HD)
    q = rms(q, t[f"{pfx}.attn.norm_q.weight"])
    k = rms(k, t[f"{pfx}.attn.norm_k.weight"])
    q, k = rope_apply(q, tab), rope_apply(k, tab)
    hpk = NH // NKV
    out = np.zeros((n, NH, HD), dtype=np.float32)
    for hh in range(NH):
        kv = hh // hpk
        s = (q[:, hh] @ k[:, kv].T) / np.sqrt(np.float32(HD))
        s -= s.max(-1, keepdims=True)
        e = np.exp(s)
        out[:, hh] = (e / e.sum(-1, keepdims=True)) @ v[:, kv]
    return out.reshape(n, NH * HD) @ t[f"{pfx}.attn.to_out.0.weight"].T


def block(t, x, pfx, tab, temb=None):
    if temb is not None:
        mod = silu(temb) @ t[f"{pfx}.norm1.linear.weight"].T + t[f"{pfx}.norm1.linear.bias"]
        s_msa, g_msa, s_mlp, g_mlp = mod.reshape(4, HID)
        xn = rms(x, t[f"{pfx}.norm1.norm.weight"]) * (1.0 + s_msa)
        x = x + np.tanh(g_msa) * rms(attention(t, xn, pfx, tab), t[f"{pfx}.norm2.weight"])
        y = rms(x, t[f"{pfx}.ffn_norm1.weight"]) * (1.0 + s_mlp)
    else:
        xn = rms(x, t[f"{pfx}.norm1.weight"])
        x = x + rms(attention(t, xn, pfx, tab), t[f"{pfx}.norm2.weight"])
        y = rms(x, t[f"{pfx}.ffn_norm1.weight"])
    mlp = (silu(y @ t[f"{pfx}.feed_forward.linear_1.weight"].T)
           * (y @ t[f"{pfx}.feed_forward.linear_3.weight"].T)
           ) @ t[f"{pfx}.feed_forward.linear_2.weight"].T
    gate = np.tanh(g_mlp) if temb is not None else 1.0
    return x + gate * rms(mlp, t[f"{pfx}.ffn_norm2.weight"])


def main():
    d, out_dir = sys.argv[1], sys.argv[2]
    t = load_shards(d)
    rng = np.random.default_rng(42)
    latent = rng.standard_normal((C, H, W)).astype(np.float32)
    cap = rng.standard_normal((CAP_N, 2304)).astype(np.float32)
    hp, wp = H // PATCH, W // PATCH

    # timestep -> temb [1024]  (flip_sin_to_cos: cos first)
    half = 128
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    ang = T * freqs
    emb = np.concatenate([np.cos(ang), np.sin(ang)]).astype(np.float32)
    temb = emb @ t["time_caption_embed.timestep_embedder.linear_1.weight"].T \
        + t["time_caption_embed.timestep_embedder.linear_1.bias"]
    temb = silu(temb) @ t["time_caption_embed.timestep_embedder.linear_2.weight"].T \
        + t["time_caption_embed.timestep_embedder.linear_2.bias"]

    cap_e = rms(cap, t["time_caption_embed.caption_embedder.0.weight"]) \
        @ t["time_caption_embed.caption_embedder.1.weight"].T \
        + t["time_caption_embed.caption_embedder.1.bias"]

    # patchify [C,H,W] -> [hp*wp, p*p*C] (dy, dx, ch inner order) + embed
    tok = latent.reshape(C, hp, PATCH, wp, PATCH).transpose(1, 3, 2, 4, 0) \
        .reshape(hp * wp, PATCH * PATCH * C)
    img = tok @ t["x_embedder.weight"].T + t["x_embedder.bias"]

    cap_ids = np.stack([np.arange(CAP_N), np.zeros(CAP_N), np.zeros(CAP_N)], 1).astype(np.int32)
    rr, cc = np.divmod(np.arange(hp * wp), wp)
    img_ids = np.stack([np.full(hp * wp, CAP_N), rr, cc], 1).astype(np.int32)
    cap_tab, img_tab = rope_table(cap_ids), rope_table(img_ids)

    for l in range(2):
        cap_e = block(t, cap_e, f"context_refiner.{l}", cap_tab)
    for l in range(2):
        img = block(t, img, f"noise_refiner.{l}", img_tab, temb)

    x = np.concatenate([cap_e, img], 0)
    tab = np.concatenate([cap_tab, img_tab], 0)
    for l in range(26):
        x = block(t, x, f"layers.{l}", tab, temb)

    # norm_out: LayerNorm(eps 1e-6, no affine) * (1+scale) -> linear_2
    scale = silu(temb) @ t["norm_out.linear_1.weight"].T + t["norm_out.linear_1.bias"]
    x64 = x.astype(np.float64)
    xn = ((x64 - x64.mean(-1, keepdims=True))
          / np.sqrt(x64.var(-1, keepdims=True) + 1e-6)).astype(np.float32)
    out = (xn * (1.0 + scale)) @ t["norm_out.linear_2.weight"].T + t["norm_out.linear_2.bias"]

    # unpatchify image tokens -> [C, H, W]
    pred = out[CAP_N:].reshape(hp, wp, PATCH, PATCH, C).transpose(4, 0, 2, 1, 3) \
        .reshape(C, H, W)

    latent.tofile(f"{out_dir}/dit_latent.bin")
    cap.tofile(f"{out_dir}/dit_cap.bin")
    pred.astype(np.float32).tofile(f"{out_dir}/dit_out.bin")
    print(f"pred {pred.shape}: |x| mean {np.abs(pred).mean():.4f} max {np.abs(pred).max():.4f}")


if __name__ == "__main__":
    main()
