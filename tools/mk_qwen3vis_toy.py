#!/usr/bin/env python3
"""A toy Qwen3-VL vision tower and its golden forward.

The tower is the half of `fl2va` that rides in the TEXT stream: the
keyframe goes to the prompt encoder as `"<Picture 1>: "` plus a vision
block, while its VAE latent goes to the DiT separately. Three of its
conventions pass any single-patch check and fail on a real image — the
bilinearly interpolated 48×48 position grid, the 2×2 merge-block order
the patches are permuted into, and the two different GELUs (tanh in the
blocks, exact in the mergers) — so the fixture uses a grid big enough
that all three are exercised.

    python3 tools/mk_qwen3vis_toy.py --comfyui <ComfyUI> --out <dir>

`--comfyui` may be a checkout or just a directory holding
comfy/text_encoders/{qwen35,qwen3vl,qwen_vl}.py; only the vision classes
are imported.
"""

import argparse
import json
import os
import sys

import torch

CFG = dict(hidden_size=96, intermediate_size=192, depth=4, num_heads=4,
           patch_size=16, temporal_patch_size=2, in_channels=3,
           spatial_merge_size=2, num_position_embeddings=2304,
           out_hidden_size=128, deepstack_visual_indexes=[1, 3])
IMG_H, IMG_W = 96, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.comfyui)
    import comfy.ops
    from comfy.text_encoders.qwen3vl import Qwen3VLVisionModel
    import comfy.text_encoders.qwen_vl as qvl

    torch.manual_seed(5)
    model = Qwen3VLVisionModel(CFG, device="cpu", dtype=torch.float32,
                               ops=comfy.ops.disable_weight_init)
    sd = model.state_dict()
    for k, v in sd.items():
        if not v.dtype.is_floating_point:
            continue
        if k.endswith("norm.weight") or ".norm1.weight" in k or ".norm2.weight" in k:
            sd[k] = torch.ones_like(v) + 0.1 * torch.randn_like(v)
        elif k.endswith(".bias"):
            sd[k] = 0.05 * torch.randn_like(v)
        else:
            sd[k] = 0.08 * torch.randn_like(v)
    model.load_state_dict(sd)
    model.eval().requires_grad_(False)

    # An image in [0, 1], HWC, as the tokenizer hands it over.
    img = torch.rand(1, IMG_H, IMG_W, 3)
    patches, grid = qvl.process_qwen2vl_images(
        img, patch_size=CFG["patch_size"], image_mean=[0.5] * 3, image_std=[0.5] * 3)
    with torch.no_grad():
        merged, deep = model(patches.to(torch.float32), grid)

    os.makedirs(a.out, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in sd.items()},
              os.path.join(a.out, "vis.safetensors"))
    # The port preprocesses from CHW in [0, 1]; hand it the same picture.
    img[0].permute(2, 0, 1).contiguous().numpy().astype("<f4").tofile(
        os.path.join(a.out, "image.bin"))
    patches.float().numpy().astype("<f4").tofile(os.path.join(a.out, "patches.bin"))
    merged.float().numpy().astype("<f4").tofile(os.path.join(a.out, "merged.bin"))
    for i, d in enumerate(deep):
        d.float().numpy().astype("<f4").tofile(os.path.join(a.out, f"deep{i}.bin"))
    meta = dict(CFG, img_h=IMG_H, img_w=IMG_W,
                grid_h=int(grid[0][1]), grid_w=int(grid[0][2]),
                n_patches=int(patches.shape[0]), merged_n=int(merged.shape[0]),
                n_deep=len(deep))
    with open(os.path.join(a.out, "golden.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(a.out, "grid", tuple(grid[0].tolist()), "patches", tuple(patches.shape),
          "merged", tuple(merged.shape), "deep", len(deep),
          "rms", float(merged.pow(2).mean().sqrt()))


if __name__ == "__main__":
    main()
