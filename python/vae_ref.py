#!/usr/bin/env python3
"""Numpy reference for the FLUX-class VAE decoder (crates/.../vae.rs).

Pure numpy + stdlib (no torch): reads the diffusers vae/ directory,
decodes a seeded random latent, and dumps latent.bin + ref.bin (f32 LE)
for tests/vae_parity.rs.

    python3 python/vae_ref.py <vae_dir> <out_dir> [latent_hw]
"""
import json
import struct
import sys

import numpy as np


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        raw = blob[s:e]
        dt = {"F32": np.float32, "F16": np.float16}.get(meta["dtype"])
        if dt is None:
            raise SystemExit(f"unsupported dtype {meta['dtype']}")
        out[name] = (
            np.frombuffer(raw, dtype=dt).astype(np.float32).reshape(meta["shape"])
        )
    return out


def conv2d(x, w, b):
    oc, ic, k, _ = w.shape
    c, h, ww = x.shape
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    cols = np.zeros((ic * k * k, h * ww), dtype=np.float32)
    i = 0
    for ci in range(ic):
        for ky in range(k):
            for kx in range(k):
                cols[i] = xp[ci, ky : ky + h, kx : kx + ww].reshape(-1)
                i += 1
    y = w.reshape(oc, -1).astype(np.float64) @ cols.astype(np.float64)
    return (y + b[:, None]).reshape(oc, h, ww).astype(np.float32)


def group_norm(x, w, b, g=32):
    c, h, ww = x.shape
    per = c // g
    y = x.reshape(g, per * h * ww).astype(np.float64)
    mean = y.mean(axis=1, keepdims=True)
    var = y.var(axis=1, keepdims=True)
    y = ((y - mean) / np.sqrt(var + 1e-6)).reshape(c, h, ww)
    return (y * w[:, None, None] + b[:, None, None]).astype(np.float32)


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def upsample2x(x):
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def resnet(t, n, x):
    h = conv2d(silu(group_norm(x, t[f"{n}.norm1.weight"], t[f"{n}.norm1.bias"])),
               t[f"{n}.conv1.weight"], t[f"{n}.conv1.bias"])
    h = conv2d(silu(group_norm(h, t[f"{n}.norm2.weight"], t[f"{n}.norm2.bias"])),
               t[f"{n}.conv2.weight"], t[f"{n}.conv2.bias"])
    if f"{n}.conv_shortcut.weight" in t:
        x = conv2d(x, t[f"{n}.conv_shortcut.weight"], t[f"{n}.conv_shortcut.bias"])
    return x + h


def attn(t, n, x):
    c, h, ww = x.shape
    hw = h * ww
    xn = group_norm(x, t[f"{n}.group_norm.weight"], t[f"{n}.group_norm.bias"])
    flat = xn.reshape(c, hw)
    q = t[f"{n}.to_q.weight"] @ flat + t[f"{n}.to_q.bias"][:, None]
    k = t[f"{n}.to_k.weight"] @ flat + t[f"{n}.to_k.bias"][:, None]
    v = t[f"{n}.to_v.weight"] @ flat + t[f"{n}.to_v.bias"][:, None]
    scores = (q.T @ k) / np.sqrt(c)
    scores -= scores.max(axis=1, keepdims=True)
    e = np.exp(scores)
    a = e / e.sum(axis=1, keepdims=True)
    o = v @ a.T  # [c, hw]
    o = t[f"{n}.to_out.0.weight"] @ o + t[f"{n}.to_out.0.bias"][:, None]
    return x + o.reshape(c, h, ww)


def main():
    vae_dir, out_dir = sys.argv[1], sys.argv[2]
    hw = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    cfg = json.load(open(f"{vae_dir}/config.json"))
    t = load_safetensors(f"{vae_dir}/diffusion_pytorch_model.safetensors")
    rng = np.random.default_rng(7)
    z = rng.standard_normal((cfg["latent_channels"], hw, hw)).astype(np.float32)
    x = z / cfg["scaling_factor"] + cfg["shift_factor"]
    x = conv2d(x, t["decoder.conv_in.weight"], t["decoder.conv_in.bias"])
    x = resnet(t, "decoder.mid_block.resnets.0", x)
    x = attn(t, "decoder.mid_block.attentions.0", x)
    x = resnet(t, "decoder.mid_block.resnets.1", x)
    b = 0
    while f"decoder.up_blocks.{b}.resnets.0.conv1.weight" in t:
        r = 0
        while f"decoder.up_blocks.{b}.resnets.{r}.conv1.weight" in t:
            x = resnet(t, f"decoder.up_blocks.{b}.resnets.{r}", x)
            r += 1
        upn = f"decoder.up_blocks.{b}.upsamplers.0.conv"
        if f"{upn}.weight" in t:
            x = conv2d(upsample2x(x), t[f"{upn}.weight"], t[f"{upn}.bias"])
        b += 1
    x = silu(group_norm(x, t["decoder.conv_norm_out.weight"], t["decoder.conv_norm_out.bias"]))
    x = conv2d(x, t["decoder.conv_out.weight"], t["decoder.conv_out.bias"])
    z.tofile(f"{out_dir}/vae_latent.bin")
    x.astype(np.float32).tofile(f"{out_dir}/vae_ref.bin")
    print(f"latent {z.shape} -> image {x.shape}; range [{x.min():.3f}, {x.max():.3f}]")


if __name__ == "__main__":
    main()
