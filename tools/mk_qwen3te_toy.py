#!/usr/bin/env python3
"""A toy Qwen3-VL prompt encoder and its golden forward.

MiniMax-H3's conditioning is the unnormalized hidden state after layer
50 of a truncated Qwen3-VL-32B. The port has to agree on four things
that a one-token check cannot see: q/k RMSNorm happens per head and
BEFORE the rotation, the rotation is split-half over the whole head at
θ=5e6, the block is llama-ordered (not Gemma's sandwich), and there is
no final norm and no embedding scale.

    python3 tools/mk_qwen3te_toy.py --comfyui /root/ComfyUI --out /root/qwentoy
"""

import argparse
import json
import os
import sys

import torch

CFG = dict(hidden_size=64, num_hidden_layers=3, num_attention_heads=8,
           num_key_value_heads=2, intermediate_size=176, vocab_size=97)
HEAD_DIM = 8
SEQ = [5, 41, 7, 3, 88, 12, 60, 1, 33, 19]


def build(comfyui, seed=99):
    sys.path.insert(0, comfyui)
    import comfy.ops
    from comfy.text_encoders.llama import Llama2_, Qwen3VL_32BConfig

    torch.manual_seed(seed)
    cfg = Qwen3VL_32BConfig(**CFG)
    cfg.head_dim = HEAD_DIM
    model = Llama2_(cfg, device="cpu", dtype=torch.float32, ops=comfy.ops.disable_weight_init)
    sd = model.state_dict()
    for k, v in sd.items():
        if not v.dtype.is_floating_point:
            continue
        sd[k] = torch.ones_like(v) + 0.1 * torch.randn_like(v) if k.endswith("norm.weight") \
            else 0.08 * torch.randn_like(v)
    model.load_state_dict(sd)
    model.eval().requires_grad_(False)
    return model, sd, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, sd, cfg = build(a.comfyui)
    os.makedirs(a.out, exist_ok=True)

    ids = torch.tensor([SEQ], dtype=torch.long)
    with torch.no_grad():
        out, _ = model(ids, dtype=torch.float32)

    from safetensors.torch import save_file
    save_file({("model." + k): v.contiguous() for k, v in sd.items()},
              os.path.join(a.out, "te.safetensors"))
    out.detach().float().contiguous().numpy().astype("<f4").tofile(
        os.path.join(a.out, "hidden.bin"))
    meta = dict(CFG)
    meta.update(head_dim=HEAD_DIM, rope_theta=float(cfg.rope_theta),
                rms_norm_eps=float(cfg.rms_norm_eps), final_norm=bool(cfg.final_norm),
                ids=SEQ)
    with open(os.path.join(a.out, "golden.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(a.out, tuple(out.shape), "rms", float(out.pow(2).mean().sqrt()))


if __name__ == "__main__":
    main()
