#!/usr/bin/env python3
"""Large native-Metal Qwen denoiser oracle fixture.

This reuses the seeded official Diffusers 0.36 oracle generator, changing only
the geometry so the native test crosses the real Metal attention and Q4TP
matmat thresholds.  It remains a tiny synthetic model: hidden=128, two layers,
and 4,102 image tokens (one target plus one reference shape).
"""

from __future__ import annotations

import json
import struct
import sys

import numpy as np
import torch
from diffusers import QwenImageTransformer2DModel

import qwen_image_denoiser_oracle as base


base.CONFIG = {
    "patch_size": 1,
    "in_channels": 32,
    "out_channels": 64,
    "num_layers": 2,
    "attention_head_dim": 64,
    "num_attention_heads": 2,
    "joint_attention_dim": 64,
    "axes_dims_rope": (16, 24, 24),
}
base.HIDDEN = 128
base.INTER = 512
base.SHAPES = [[1, 64, 64], [1, 2, 3]]
base.TEXT_LEN = 8
base.TIMESTEP = 0.375


def f16_to_f32(value: float) -> float:
    bits = int.from_bytes(struct.pack("<e", float(value)), "little")
    return float(np.float32(struct.unpack("<e", bits.to_bytes(2, "little"))[0]))


def f16_bits(value: float) -> int:
    return int.from_bytes(struct.pack("<e", float(value)), "little")


def q4tp_dequant(values: list[float], rows: int, cols: int) -> list[float]:
    """Mirror the fixture's Q4TP encoder and core decoder in float32."""
    assert cols % 32 == 0
    groups = cols // 32
    out: list[float] = []
    src = np.asarray(values, dtype=np.float32).reshape(rows, cols)
    for row in src:
        logs: list[float] = []
        lo = float("inf")
        hi = float("-inf")
        for g in range(groups):
            tile = row[g * 32 : (g + 1) * 32]
            absmax = float(np.max(np.abs(tile)))
            scale = max(f16_to_f32(np.float32(absmax / 7.0)), np.float32(6.1035156e-5))
            log = float(np.float32(np.log2(np.float32(scale))))
            logs.append(log)
            if absmax != 0.0:
                lo = min(lo, log)
                hi = max(hi, log)
        if not np.isfinite(lo):
            lo = logs[0]
            hi = lo
        lo_h = f16_bits(lo)
        lo_r = f16_to_f32(lo)
        span = max(float(np.float32(hi - lo_r)), 0.0)
        step_h = f16_bits(np.float32(span / 31.0))
        for _ in range(64):
            step = f16_to_f32(struct.unpack("<e", step_h.to_bytes(2, "little"))[0])
            if step > 0.0 and lo_r + 31.0 * step >= hi:
                break
            step_h = min(step_h + 1, 0xFFFF)
        lo_encoded = struct.unpack("<e", lo_h.to_bytes(2, "little"))[0]
        step_encoded = struct.unpack("<e", step_h.to_bytes(2, "little"))[0]
        ratio = float(np.float32(np.exp2(np.float32(step_encoded))))
        scale0 = float(np.float32(np.exp2(np.float32(lo_encoded))))
        ladder = [scale0]
        for _ in range(1, 32):
            ladder.append(float(np.float32(ladder[-1] * ratio)))
        step = f16_to_f32(step_encoded)
        for g, log in enumerate(logs):
            if step <= 0.0:
                code = 0
            else:
                raw = float(np.float32((log - lo_r) / step))
                code = int(np.rint(np.float32(raw)))
                code = max(0, min(31, code))
            scale = ladder[code]
            inv = float(np.float32(1.0 / scale)) if scale > 0.0 else 0.0
            tile = row[g * 32 : (g + 1) * 32]
            for k in range(16):
                q0 = int(np.rint(np.float32(tile[2 * k] * inv)))
                q1 = int(np.rint(np.float32(tile[2 * k + 1] * inv)))
                q0 = max(-8, min(7, q0))
                q1 = max(-8, min(7, q1))
                out.extend(
                    [
                        float(np.float32(q0 * scale)),
                        float(np.float32(q1 * scale)),
                    ]
                )
    return out


def q4tp_expected(payload: dict[str, object]) -> list[float]:
    model = QwenImageTransformer2DModel(**base.CONFIG)
    state = model.state_dict()
    weights = payload["weights"]
    assert isinstance(weights, dict)
    with torch.no_grad():
        for name, values in weights.items():
            target = state[name]
            vals = values
            if target.ndim == 2:
                vals = q4tp_dequant(values, target.shape[0], target.shape[1])
            state[name].copy_(torch.tensor(vals, dtype=torch.float32).reshape(target.shape))
    model.load_state_dict(state)
    image = torch.tensor(payload["image"], dtype=torch.float32).reshape(
        1, -1, base.CONFIG["in_channels"]
    )
    text = torch.tensor(payload["text"], dtype=torch.float32).reshape(
        1, payload["text_len"], base.CONFIG["joint_attention_dim"]
    )
    with torch.no_grad():
        output = model(
            image,
            encoder_hidden_states=text,
            timestep=torch.tensor([payload["timestep"]], dtype=torch.float32),
            img_shapes=[[tuple(x) for x in payload["shapes"]]],
            txt_seq_lens=[payload["text_len"]],
            return_dict=False,
        )[0]
    return [float(x) for x in output.flatten()]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} OUTPUT.json")
    base.main(sys.argv[1])
    output_path = sys.argv[1]
    payload = json.loads(open(output_path).read())
    payload["expected_q4tp"] = q4tp_expected(payload)
    with open(output_path, "w") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
