#!/usr/bin/env python3
"""Small BF16 oracle for the first Qwen4-exp text layer.

Only shards 1-3 and 130 of Qwen3.8-Flash-Next are needed.  The script is
intended for converter/runtime parity work: it avoids constructing the other
47 layers and writes selected intermediate activations as raw little-endian
f32 files that can be compared with CMF_QWEN_DUMP.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel


def as_f32(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().astype("<f4", copy=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--prompt", default="What is 2+2? Answer with one digit.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    if getattr(encoded, "encodings", None):
        ids = encoded.encodings[0].ids
    elif hasattr(encoded, "input_ids"):
        ids = encoded.input_ids
    else:
        ids = encoded
    print("token ids:", ids)
    print("first token:", ids[0], repr(tokenizer.decode([ids[0]])))

    cfg = copy.deepcopy(AutoConfig.from_pretrained(args.model_dir).text_config)
    cfg.num_hidden_layers = 1
    cfg.layer_types = cfg.layer_types[:1]
    cfg.ple_layer_ids = []
    # One layer is about 5 GiB in BF16 (roughly twice that during the short
    # float32 construction); unlike the full 176B model this fits comfortably.
    model = Qwen4ExpTextModel(cfg)

    wanted = ("embed_tokens.", "layers.0.", "hyper_connection_mixer.")
    state: dict[str, torch.Tensor] = {}
    for shard in (1, 2, 3, 130):
        path = args.model_dir / f"model-{shard:05d}-of-00131.safetensors"
        for name, tensor in load_file(path, device="cpu").items():
            prefix = "model.language_model."
            if name.startswith(prefix) and name[len(prefix) :].startswith(wanted):
                state[name[len(prefix) :]] = tensor
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    model = model.to(args.device).eval()

    captures: dict[str, torch.Tensor] = {}

    def save_output(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            captures[name] = output

        return hook

    model.embed_tokens.register_forward_hook(save_output("embedding"))
    layer = model.layers[0]
    layer.attn_hyper_connection.register_forward_hook(save_output("attn_in"))
    layer.linear_attn.in_proj_qkv.register_forward_hook(save_output("gdn_qkv"))
    layer.linear_attn.in_proj_z.register_forward_hook(save_output("gdn_z"))
    layer.linear_attn.in_proj_a.register_forward_hook(save_output("gdn_a"))
    layer.linear_attn.in_proj_b.register_forward_hook(save_output("gdn_b"))

    def save_input(name: str):
        def hook(_module, inputs):
            captures[name] = inputs[0]

        return hook

    layer.linear_attn.out_proj.register_forward_pre_hook(save_input("gdn_core"))
    layer.linear_attn.register_forward_hook(save_output("attn_out"))
    layer.mlp_hyper_connection.register_forward_hook(save_output("moe_in"))
    layer.mlp.register_forward_hook(save_output("moe_out"))
    layer.register_forward_hook(save_output("post_moe"))
    model.hyper_connection_mixer.register_forward_hook(save_output("head_in"))

    with torch.inference_mode():
        model(input_ids=torch.tensor([[ids[0]]], device=args.device), use_cache=False)

    for name, tensor in captures.items():
        values = as_f32(tensor).reshape(-1)
        values.tofile(args.out / f"p000000_l00_{name}.f32")
        print(
            f"{name:10s} shape={tuple(tensor.shape)} "
            f"rms={float(np.sqrt(np.mean(values.astype(np.float64) ** 2))):.9f} "
            f"max={float(np.max(np.abs(values))):.9f}"
        )


if __name__ == "__main__":
    main()
