#!/usr/bin/env python3
"""Regenerate the independent Diffusers 0.36 pipeline-math fixture (no weights)."""
import argparse
import json
from pathlib import Path
import diffusers
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_shift

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
assert diffusers.__version__ == '0.36.0'
schedules = []
for steps, tokens in [(2, 256), (8, 1024), (40, 4096), (30, 16384)]:
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256, max_image_seq_len=8192,
        base_shift=0.5, max_shift=0.9, shift_terminal=0.02,
        use_dynamic_shifting=True, time_shift_type='exponential')
    mu = calculate_shift(tokens, 256, 8192, 0.5, 0.9)
    scheduler.set_timesteps(steps, sigmas=np.linspace(1.0, 1.0 / steps, steps), mu=mu)
    schedules.append(dict(steps=steps, tokens=tokens, sigmas=scheduler.sigmas.tolist()))
raw = torch.arange(48, dtype=torch.float32).reshape(1, 2, 4, 6) / 7.0
packed = QwenImageEditPlusPipeline._pack_latents(raw, 1, 2, 4, 6)
unpacked = QwenImageEditPlusPipeline._unpack_latents(packed, 32, 48, 8)
index = torch.arange(192, dtype=torch.float32).reshape(1, 3, 64)
cond = torch.sin(index * 0.07) * 1.2
uncond = torch.cos(index * 0.031) * 0.6
combined = uncond + 4.0 * (cond - uncond)
combined *= torch.norm(cond, dim=-1, keepdim=True) / torch.norm(combined, dim=-1, keepdim=True)
result = dict(source='diffusers==0.36.0 / torch==2.8.0', schedules=schedules,
              nchw=raw.flatten().tolist(), packed=packed.flatten().tolist(), unpacked=unpacked.flatten().tolist(),
              cond=cond.flatten().tolist(), uncond=uncond.flatten().tolist(), cfg=combined.flatten().tolist())
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, separators=(',', ':')) + '\n')
print(f'{args.out}: {len(schedules)} schedules, non-square packing, per-token CFG reference')
