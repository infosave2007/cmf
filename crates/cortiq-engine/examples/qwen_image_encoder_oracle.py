#!/usr/bin/env python3
"""Create a deterministic, download-free Qwen2.5-VL tiny encoder oracle.

The fixture uses the official Transformers model class with a deliberately
small text/vision geometry and a byte-level tokenizer.  It still traverses
the real image placeholder splice, vision tower, MRoPE, causal text layer,
and final hidden-state output.  The Rust companion packs the same state into
a CMF and compares ``hidden_states[-1][:, 64:]``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from tokenizers import models, pre_tokenizers
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)


PROMPT_HEAD = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image that meets "
    "the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n"
)
PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
PROMPT = "Turn this cat into a dog"


def bytes_to_unicode() -> list[str]:
    chars: list[str] = []
    extra = 0
    for value in range(256):
        printable = 0x21 <= value <= 0x7E or 0xA1 <= value <= 0xAC or 0xAE <= value <= 0xFF
        if printable:
            chars.append(chr(value))
        else:
            chars.append(chr(256 + extra))
            extra += 1
    return chars


def make_tokenizer(path: Path) -> dict:
    byte_chars = bytes_to_unicode()
    vocab = {token: i for i, token in enumerate(byte_chars)}
    special = {
        "<|image_pad|>": 300,
        "<|vision_start|>": 301,
        "<|vision_end|>": 302,
        "<|im_start|>": 303,
        "<|im_end|>": 304,
    }
    for token, token_id in special.items():
        vocab[token] = token_id
    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": token_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for token, token_id in special.items()
        ],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "vocab": vocab,
            "merges": [],
        },
    }
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


def condition_dimensions(width: int, height: int) -> tuple[int, int]:
    area = 384 * 384
    ratio = width / height
    # The fixture is square-free but this is the same ties-to-even operation
    # used by the pinned Diffusers helper.
    return round(math.sqrt(area * ratio) / 32) * 32, round(math.sqrt(area / ratio) / 32) * 32


def smart_resize(height: int, width: int) -> tuple[int, int]:
    factor, min_pixels, max_pixels = 4, 16, 320
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = max(factor, math.ceil(height * beta / factor) * factor)
        w = max(factor, math.ceil(width * beta / factor) * factor)
    return h, w


def make_pixel_values(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = smart_resize(image.height, image.width)
    image = image.resize((width, height), Image.Resampling.NEAREST)
    array = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    grid_h, grid_w = height // 2, width // 2
    # [T,C,H,W] with the singleton frame repeated to temporal_patch_size=2.
    patches = array.unsqueeze(0).repeat(2, 1, 1, 1)
    patches = patches.view(1, 2, 3, grid_h // 2, 2, 2, grid_w // 2, 2, 2)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    patches = patches.reshape(grid_h * grid_w, 3 * 2 * 2 * 2)
    return patches, torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: qwen_image_encoder_oracle.py OUTPUT.json")
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output.with_name("qwen_image_encoder_tokenizer.json")
    tokenizer_json = make_tokenizer(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    torch.manual_seed(1729)
    text = Qwen2_5_VLTextConfig(
        vocab_size=305,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        rope_scaling={"mrope_section": [1, 1, 1], "rope_type": "default"},
        use_sliding_window=False,
    )
    vision = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        tokens_per_second=4,
        # 8 / (merge 2) / (patch 2) = 2 merged groups per window.  The
        # [6,10] patch grid below therefore exercises 2×3 windows, including
        # one-row/one-column edge windows; block 1 uses that path.
        window_size=8,
        out_hidden_size=12,
        fullatt_block_indexes=[0],
    )
    config = Qwen2_5_VLConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=300,
        vision_start_token_id=301,
        vision_end_token_id=302,
    )
    model = Qwen2_5_VLForConditionalGeneration(config).eval()

    # Constant RGB values keep the image-resample part exactly comparable
    # between PIL and the native image crate while retaining a non-square
    # image/grid fixture.  Every model weight remains seeded and non-uniform.
    width, height = 3, 2
    rgb = [32, 96, 160] * (width * height)
    image = Image.new("RGB", (width, height), tuple(rgb[:3]))
    condition_width, condition_height = condition_dimensions(width, height)
    condition = image.resize((condition_width, condition_height), Image.Resampling.LANCZOS)
    # One placeholder is expanded by the processor in the official path;
    # spelling the expanded run is equivalent for this direct model call.
    prompt_text = PROMPT_HEAD + "Picture 1: <|vision_start|>"
    prompt_text += "<|image_pad|>" * 15
    prompt_text += "<|vision_end|>" + PROMPT + PROMPT_TAIL
    ids = tokenizer.encode(prompt_text).ids
    pixel_values, image_grid_thw = make_pixel_values(condition)
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
    hidden = outputs.hidden_states[-1][0, 64:, :].float().reshape(-1).tolist()

    weights: dict[str, list[float]] = {}
    shapes: dict[str, list[int]] = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("model.visual."):
            target = name[len("model.") :]
        elif name.startswith("model.language_model."):
            target = "model." + name[len("model.language_model.") :]
        elif name == "lm_head.weight":
            # The encoder never loads lm_head; omit it to keep the fixture
            # focused on the conditioning contract.
            continue
        else:
            continue
        value = tensor.detach().float().contiguous()
        shapes[target] = list(value.shape)
        weights[target] = value.reshape(-1).tolist()

    processor_json = {
        "image_processor_type": "Qwen2VLImageProcessorFast",
        "min_pixels": 16,
        "max_pixels": 320,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "rescale_factor": 1.0,
        "resample": 0,
    }
    fixture = {
        "config": config.to_dict(),
        "processor": processor_json,
        "tokenizer": tokenizer_json,
        "prompt": PROMPT,
        "image_width": width,
        "image_height": height,
        "image_rgb": rgb,
        "ids": ids,
        "grid": image_grid_thw[0].tolist(),
        "shapes": shapes,
        "weights": weights,
        "expected": hidden,
    }
    output.write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"output": str(output), "tokens": len(ids), "grid": fixture["grid"], "rows": len(hidden) // 12}))


if __name__ == "__main__":
    main()
