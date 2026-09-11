#!/usr/bin/env python3
"""Seeded Diffusers oracle for the native Qwen Image VAE.

This is a tiny full encoder/decoder fixture.  It writes the exact official
state-dict names and config needed by a CMF packer, then records one
``encode(...).latent_dist.mode()`` result and one raw-latent decode result.
The Rust parity example consumes the resulting CMF; this script is only the
Torch/Diffusers numerical oracle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage
from safetensors.torch import save_file


BASE_DIM = 4
Z_DIM = 2
DIM_MULT = [1, 2, 2, 2]
NUM_RES_BLOCKS = 1
TEMPORAL_DOWNSAMPLE = [False, True, True]
HEIGHT = 16
WIDTH = 24
LATENT_HEIGHT = 2
LATENT_WIDTH = 3


def write_f32(path: Path, value: torch.Tensor) -> None:
    value.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy().astype("<f4").tofile(path)


def make_fixture(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(71)
    latents_mean = [0.11, -0.23]
    latents_std = [1.17, 0.83]
    model = AutoencoderKLQwenImage(
        base_dim=BASE_DIM,
        z_dim=Z_DIM,
        dim_mult=DIM_MULT,
        num_res_blocks=NUM_RES_BLOCKS,
        attn_scales=[],
        temperal_downsample=TEMPORAL_DOWNSAMPLE,
        latents_mean=latents_mean,
        latents_std=latents_std,
    )

    # Every floating parameter is deterministic, non-uniform, and finite.
    # Gamma is kept away from zero so an all-zero or all-one implementation
    # cannot accidentally pass the full-network gate.
    state = model.state_dict()
    with torch.no_grad():
        for name, value in state.items():
            if not value.dtype.is_floating_point:
                continue
            index = torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape)
            if name.endswith(".gamma"):
                values = 0.82 + 0.13 * torch.sin(index * 0.173 + 0.31)
            elif name.endswith(".bias"):
                values = 0.035 * torch.cos(index * 0.097 + 0.21)
            else:
                values = 0.045 * torch.sin(index * 0.071 + 0.47)
            state[name] = values.to(dtype=value.dtype)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)

    frame = torch.linspace(-0.9, 0.9, 3 * HEIGHT * WIDTH, dtype=torch.float32).reshape(
        1, 3, 1, HEIGHT, WIDTH
    )
    latent = torch.linspace(-0.7, 0.7, Z_DIM * LATENT_HEIGHT * LATENT_WIDTH, dtype=torch.float32).reshape(
        1, Z_DIM, 1, LATENT_HEIGHT, LATENT_WIDTH
    )
    with torch.no_grad():
        encoded = model.encode(frame).latent_dist.mode()
        decoded = model.decode(latent).sample

    # Remove batch and singleton temporal dimensions: the Rust API is NCHW.
    write_f32(out_dir / "frame_in.bin", frame[0, :, 0])
    write_f32(out_dir / "encode_mean.bin", encoded[0, :, 0])
    write_f32(out_dir / "decode_latent.bin", latent[0, :, 0])
    write_f32(out_dir / "decode_ref.bin", decoded[0, :, 0])
    save_file({k: v.detach().contiguous().cpu() for k, v in state.items()}, str(out_dir / "vae.safetensors"))

    config = {
        "_class_name": "AutoencoderKLQwenImage",
        "base_dim": BASE_DIM,
        "z_dim": Z_DIM,
        "dim_mult": DIM_MULT,
        "num_res_blocks": NUM_RES_BLOCKS,
        "attn_scales": [],
        "temperal_downsample": TEMPORAL_DOWNSAMPLE,
        "latents_mean": latents_mean,
        "latents_std": latents_std,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "weights": "vae.safetensors",
        "config": "config.json",
        "encode_input": "frame_in.bin",
        "encode_reference": "encode_mean.bin",
        "decode_input": "decode_latent.bin",
        "decode_reference": "decode_ref.bin",
        "encode_input_shape": [3, HEIGHT, WIDTH],
        "encode_output_shape": [Z_DIM, LATENT_HEIGHT, LATENT_WIDTH],
        "decode_input_shape": [Z_DIM, LATENT_HEIGHT, LATENT_WIDTH],
        "decode_output_shape": [3, HEIGHT, WIDTH],
        "seed": 71,
        "reference": "diffusers-0.36.0 AutoencoderKLQwenImage",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(out_dir), **manifest}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    make_fixture(args.out)


if __name__ == "__main__":
    main()
