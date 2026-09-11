#!/usr/bin/env python3
"""Seeded official Diffusers oracle for the native Qwen denoiser fixture.

This deliberately uses the installed Diffusers implementation as the numeric
oracle.  It emits only a tiny, nonuniform one-block case; production never
imports Python or Torch.  The Rust fixture uses the same xorshift stream when
it writes a real CMF and compares its output with the JSON emitted here.

Run with the reference environment, for example:

    /private/tmp/cmf-qwen-image-ref-venv/bin/python \
      qwen_image_denoiser_oracle.py /tmp/qwen-image-oracle.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from diffusers import QwenImageTransformer2DModel


CONFIG = {
    "patch_size": 1,
    "in_channels": 2,
    "out_channels": 2,
    "num_layers": 2,
    "attention_head_dim": 6,
    "num_attention_heads": 2,
    "joint_attention_dim": 4,
    "axes_dims_rope": (2, 2, 2),
}
HIDDEN = 12
INTER = 48
SHAPES = [[1, 2, 3], [1, 1, 2]]
TEXT_LEN = 3
TIMESTEP = 0.375
MASK64 = (1 << 64) - 1


class Stream:
    def __init__(self) -> None:
        self.state = 0x9E3779B97F4A7C15

    def next(self) -> float:
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & MASK64
        x ^= x >> 27
        self.state = x & MASK64
        z = (self.state * 0x2545F4914F6CDD1D) & MASK64
        unit = ((z >> 40) & 0xFFFFFF) / float(1 << 24)
        return (unit * 2.0 - 1.0)

    def values(self, count: int, scale: float, offset: float = 0.0) -> list[float]:
        return [offset + scale * self.next() for _ in range(count)]


def tensor_specs() -> list[tuple[str, tuple[int, ...], str]]:
    # Keep this order in lockstep with qwen_image_denoiser_fixture.rs.  It is
    # the model state order from Diffusers 0.36 for the focused configuration.
    in_channels = int(CONFIG["in_channels"])
    joint_attention_dim = int(CONFIG["joint_attention_dim"])
    head_dim = int(CONFIG["attention_head_dim"])
    output_features = int(CONFIG["patch_size"]) ** 2 * int(CONFIG["out_channels"])
    return [
        ("time_text_embed.timestep_embedder.linear_1.weight", (HIDDEN, 256), "w"),
        ("time_text_embed.timestep_embedder.linear_1.bias", (HIDDEN,), "b"),
        ("time_text_embed.timestep_embedder.linear_2.weight", (HIDDEN, HIDDEN), "w"),
        ("time_text_embed.timestep_embedder.linear_2.bias", (HIDDEN,), "b"),
        ("txt_norm.weight", (joint_attention_dim,), "n"),
        ("img_in.weight", (HIDDEN, in_channels), "w"),
        ("img_in.bias", (HIDDEN,), "b"),
        ("txt_in.weight", (HIDDEN, joint_attention_dim), "w"),
        ("txt_in.bias", (HIDDEN,), "b"),
        ("transformer_blocks.0.img_mod.1.weight", (6 * HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.0.img_mod.1.bias", (6 * HIDDEN,), "b"),
        ("transformer_blocks.0.attn.norm_q.weight", (head_dim,), "n"),
        ("transformer_blocks.0.attn.norm_k.weight", (head_dim,), "n"),
        ("transformer_blocks.0.attn.to_q.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.0.attn.to_q.bias", (HIDDEN,), "b"),
        ("transformer_blocks.0.attn.to_k.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.0.attn.to_k.bias", (HIDDEN,), "b"),
        ("transformer_blocks.0.attn.to_v.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.0.attn.to_v.bias", (HIDDEN,), "b"),
        ("transformer_blocks.0.attn.add_k_proj.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.0.attn.add_k_proj.bias", (HIDDEN,), "b"),
        ("transformer_blocks.0.attn.add_v_proj.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.0.attn.add_v_proj.bias", (HIDDEN,), "b"),
        ("transformer_blocks.0.attn.add_q_proj.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.0.attn.add_q_proj.bias", (HIDDEN,), "b"),
        ("transformer_blocks.0.attn.to_out.0.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.0.attn.to_out.0.bias", (HIDDEN,), "b"),
        ("transformer_blocks.0.attn.to_add_out.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.0.attn.to_add_out.bias", (HIDDEN,), "b"),
        ("transformer_blocks.0.attn.norm_added_q.weight", (head_dim,), "n"),
        ("transformer_blocks.0.attn.norm_added_k.weight", (head_dim,), "n"),
        ("transformer_blocks.0.img_mlp.net.0.proj.weight", (INTER, HIDDEN), "w"),
        ("transformer_blocks.0.img_mlp.net.0.proj.bias", (INTER,), "b"),
        ("transformer_blocks.0.img_mlp.net.2.weight", (HIDDEN, INTER), "w"),
        ("transformer_blocks.0.img_mlp.net.2.bias", (HIDDEN,), "b"),
        ("transformer_blocks.0.txt_mod.1.weight", (6 * HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.0.txt_mod.1.bias", (6 * HIDDEN,), "b"),
        ("transformer_blocks.0.txt_mlp.net.0.proj.weight", (INTER, HIDDEN), "w"),
        ("transformer_blocks.0.txt_mlp.net.0.proj.bias", (INTER,), "b"),
        ("transformer_blocks.0.txt_mlp.net.2.weight", (HIDDEN, INTER), "w"),
        ("transformer_blocks.0.txt_mlp.net.2.bias", (HIDDEN,), "b"),
        ("transformer_blocks.1.img_mod.1.weight", (6 * HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.1.img_mod.1.bias", (6 * HIDDEN,), "b"),
        ("transformer_blocks.1.attn.norm_q.weight", (head_dim,), "n"),
        ("transformer_blocks.1.attn.norm_k.weight", (head_dim,), "n"),
        ("transformer_blocks.1.attn.to_q.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.1.attn.to_q.bias", (HIDDEN,), "b"),
        ("transformer_blocks.1.attn.to_k.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.1.attn.to_k.bias", (HIDDEN,), "b"),
        ("transformer_blocks.1.attn.to_v.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.1.attn.to_v.bias", (HIDDEN,), "b"),
        ("transformer_blocks.1.attn.add_k_proj.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.1.attn.add_k_proj.bias", (HIDDEN,), "b"),
        ("transformer_blocks.1.attn.add_v_proj.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.1.attn.add_v_proj.bias", (HIDDEN,), "b"),
        ("transformer_blocks.1.attn.add_q_proj.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.1.attn.add_q_proj.bias", (HIDDEN,), "b"),
        ("transformer_blocks.1.attn.to_out.0.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.1.attn.to_out.0.bias", (HIDDEN,), "b"),
        ("transformer_blocks.1.attn.to_add_out.weight", (HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.1.attn.to_add_out.bias", (HIDDEN,), "b"),
        ("transformer_blocks.1.attn.norm_added_q.weight", (head_dim,), "n"),
        ("transformer_blocks.1.attn.norm_added_k.weight", (head_dim,), "n"),
        ("transformer_blocks.1.img_mlp.net.0.proj.weight", (INTER, HIDDEN), "w"),
        ("transformer_blocks.1.img_mlp.net.0.proj.bias", (INTER,), "b"),
        ("transformer_blocks.1.img_mlp.net.2.weight", (HIDDEN, INTER), "w"),
        ("transformer_blocks.1.img_mlp.net.2.bias", (HIDDEN,), "b"),
        ("transformer_blocks.1.txt_mod.1.weight", (6 * HIDDEN, HIDDEN), "w"),
        ("transformer_blocks.1.txt_mod.1.bias", (6 * HIDDEN,), "b"),
        ("transformer_blocks.1.txt_mlp.net.0.proj.weight", (INTER, HIDDEN), "w"),
        ("transformer_blocks.1.txt_mlp.net.0.proj.bias", (INTER,), "b"),
        ("transformer_blocks.1.txt_mlp.net.2.weight", (HIDDEN, INTER), "w"),
        ("transformer_blocks.1.txt_mlp.net.2.bias", (HIDDEN,), "b"),
        ("norm_out.linear.weight", (2 * HIDDEN, HIDDEN), "w"),
        ("norm_out.linear.bias", (2 * HIDDEN,), "b"),
        ("proj_out.weight", (output_features, HIDDEN), "w"),
        ("proj_out.bias", (output_features,), "b"),
    ]


def main(path: str) -> None:
    stream = Stream()
    model = QwenImageTransformer2DModel(**CONFIG)
    state = model.state_dict()
    weights: dict[str, list[float]] = {}
    for name, shape, kind in tensor_specs():
        count = 1
        for dim in shape:
            count *= dim
        if kind == "n":
            values = stream.values(count, 0.05, 1.0)
        elif kind == "b":
            values = stream.values(count, 0.025)
        else:
            values = stream.values(count, 0.08)
        tensor = torch.tensor(values, dtype=torch.float32).reshape(shape)
        state[name].copy_(tensor)
        weights[name] = [float(x) for x in tensor.flatten()]
    model.load_state_dict(state)

    in_channels = int(CONFIG["in_channels"])
    joint_attention_dim = int(CONFIG["joint_attention_dim"])
    image_tokens = sum(f * h * w for f, h, w in SHAPES)
    image = torch.tensor(
        stream.values(image_tokens * in_channels, 0.2), dtype=torch.float32
    ).reshape(
        1, image_tokens, in_channels
    )
    text = torch.tensor(
        stream.values(TEXT_LEN * joint_attention_dim, 0.2), dtype=torch.float32
    ).reshape(
        1, TEXT_LEN, joint_attention_dim
    )
    with torch.no_grad():
        output = model(
            image,
            encoder_hidden_states=text,
            timestep=torch.tensor([TIMESTEP], dtype=torch.float32),
            # Diffusers carries a batch dimension around the ordered target
            # then reference shapes; the native one-batch API flattens this
            # outer list to `&[[usize; 3]]`.
            img_shapes=[[tuple(x) for x in SHAPES]],
            txt_seq_lens=[TEXT_LEN],
            return_dict=False,
        )[0]

    payload = {
        "config": {**CONFIG, "axes_dims_rope": list(CONFIG["axes_dims_rope"])},
        "shapes": SHAPES,
        "text_len": TEXT_LEN,
        "timestep": TIMESTEP,
        "image": [float(x) for x in image.flatten()],
        "text": [float(x) for x in text.flatten()],
        "weights": weights,
        "expected": [float(x) for x in output.flatten()],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}: {output.numel()} output values")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} OUTPUT.json")
    main(sys.argv[1])
