#!/usr/bin/env python3
"""Scale probe: GPU memory and time of metric-refit training steps at Qwen3.8-27B layer
geometry, without the model. A random q4tp qwen3_5 stack with the real shapes (hidden 5120,
FFN 17408, 24/4 x 256 attention heads, 16/48 x 128 GDN heads) but only --layers decoder layers
and a --vocab head is trained for a few steps in streaming residency; per-layer cost and the
fixed costs (head, one prefetched layer) extrapolate to the 64-layer model.

  scale_probe.py --layers 4 --vocab 32768 --mb 1 --steps 2 --gpu-cap-gb 1.9
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cmfcodec as cc  # noqa: E402
import qmodel  # noqa: E402
from test_quant_tools import FakeCmf  # noqa: E402


def fast_tiled(rng, rows, cols):
    """a valid random q4tp payload without the slow generic encoder"""
    gpr = cols // cc.G
    nib = rng.integers(0, 256, size=rows * gpr * 16, dtype=np.uint8)
    par = np.stack([cc.f16_bits(rng.uniform(-9.5, -8.5, rows)), cc.f16_bits(np.full(rows, 0.045))], 1)
    codes = rng.integers(0, 256, size=rows * cc.stride_of(gpr), dtype=np.uint8)
    return nib.tobytes() + par.astype("<u2").tobytes() + codes.tobytes()


def build(layers, vocab, seed=0):
    rng = np.random.default_rng(seed)
    H, I, nh, nkv, hd = 5120, 17408, 24, 4, 256
    nk, nv, dk, dv = 16, 48, 128, 128
    kinds = ["FullAttention" if (l + 1) % 4 == 0 else "LinearAttention" for l in range(layers)]
    arch = {"arch_name": "qwen3_5_text", "hidden_size": H, "intermediate_size": I, "num_layers": layers,
            "num_attention_heads": nh, "num_kv_heads": nkv, "head_dim": hd, "vocab_size": vocab, "layer_types": kinds,
            "rms_norm_eps": 1e-6, "norm_style": "gemma", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
            "linear_num_key_heads": nk, "linear_num_value_heads": nv, "linear_key_head_dim": dk,
            "linear_value_head_dim": dv, "linear_conv_kernel_dim": 4, "linear_core": {"kind": "gated_delta_net"}}
    m = FakeCmf(arch)
    q4 = lambda n, r, c: m.add(n, "Q4TiledP", (r, c), fast_tiled(rng, r, c))
    f16 = lambda n, shape, sd, c=0.0: m.add(n, "F16", shape, cc.encode_float("F16", c + rng.normal(0, sd, shape)))
    for l, t in enumerate(kinds):
        p = f"model.layers.{l}."
        f16(p + "input_layernorm.weight", (H,), 0.02)
        f16(p + "post_attention_layernorm.weight", (H,), 0.02)
        if t == "FullAttention":
            q4(p + "self_attn.q_proj.weight", 2 * nh * hd, H)
            q4(p + "self_attn.k_proj.weight", nkv * hd, H)
            q4(p + "self_attn.v_proj.weight", nkv * hd, H)
            q4(p + "self_attn.o_proj.weight", H, nh * hd)
            f16(p + "self_attn.q_norm.weight", (hd,), 0.02)
            f16(p + "self_attn.k_norm.weight", (hd,), 0.02)
        else:
            c = 2 * nk * dk + nv * dv
            q4(p + "linear_attn.in_proj_qkv.weight", c, H)
            q4(p + "linear_attn.in_proj_z.weight", nv * dv, H)
            f16(p + "linear_attn.in_proj_b.weight", (nv, H), 0.01)
            f16(p + "linear_attn.in_proj_a.weight", (nv, H), 0.01)
            q4(p + "linear_attn.out_proj.weight", H, nv * dv)
            f16(p + "linear_attn.conv1d.weight", (c, 1, 4), 0.3)
            m.add(p + "linear_attn.A_log", "F32", (nv,), cc.encode_float("F32", np.log(rng.uniform(1, 16, nv))))
            m.add(p + "linear_attn.dt_bias", "F32", (nv,), cc.encode_float("F32", rng.normal(0, 0.5, nv)))
            f16(p + "linear_attn.norm.weight", (dv,), 0.02, 1.0)
        q4(p + "mlp.gate_proj.weight", I, H)
        q4(p + "mlp.up_proj.weight", I, H)
        q4(p + "mlp.down_proj.weight", H, I)
    f16("model.norm.weight", (H,), 0.02)
    q4("model.embed_tokens.weight", vocab, H)
    q4("lm_head.weight", vocab, H)
    return m, kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--bs", type=int, default=0)
    ap.add_argument("--wl", type=int, default=512)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--residency", default="stream")
    ap.add_argument("--gpu-cap-gb", type=float, default=0)
    args = ap.parse_args()
    bs = args.bs or args.mb
    qmodel.setup_torch(0, tf32=True)
    dev = qmodel.resolve_device("cuda", args.gpu_cap_gb)
    t0 = time.time()
    m, kinds = build(args.layers, args.vocab)
    print(f"built {args.layers} layers {kinds} in {time.time() - t0:.0f}s", flush=True)
    model = qmodel.Model(m, dev, residency=args.residency, train=True, step=True, train_norms=True, vchunk=8192)
    per_layer = model.layers[0].device_bytes(), model.layers[-1].device_bytes()
    print(f"compact bytes per layer: GDN {per_layer[0] / 2**20:.0f} MiB, attention {per_layer[1] / 2**20:.0f} MiB; "
          f"head {model.head.lin.device_bytes() / 2**20:.0f} MiB; GPU after load "
          f"{torch.cuda.memory_allocated(dev) / 2**20:.0f} MiB", flush=True)
    rng = np.random.default_rng(1)
    w = rng.integers(0, args.vocab, size=(bs, args.wl))
    K = 64
    lp = torch.log_softmax(torch.randn(bs, args.wl - 1, K), -1) - 0.5
    ix = torch.from_numpy(rng.integers(0, args.vocab, size=(bs, args.wl - 1, K)).astype(np.int32))
    opt = torch.optim.Adam(model.params(), lr=1e-3)
    for s in range(args.steps):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        t1 = time.time()
        opt.zero_grad(set_to_none=True)
        model.train_step(w, lp, ix, args.mb, bs * (args.wl - 1))
        opt.step()
        torch.cuda.synchronize(dev)
        dt = time.time() - t1
        peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        print(f"step {s}: {dt:.2f} s for {bs}x{args.wl} tokens through {args.layers} layers "
              f"({dt / args.layers:.2f} s/layer), GPU peak {peak:.2f} GiB", flush=True)
    torch.cuda.reset_peak_memory_stats(dev)
    t1 = time.time()
    model.evaluate(w, mb=args.mb)
    torch.cuda.synchronize(dev)
    print(f"forward-only pass: {time.time() - t1:.2f} s, GPU peak {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
