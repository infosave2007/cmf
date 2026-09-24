#!/usr/bin/env python3
"""MiMo-V2.6 vision oracle: preprocessing, prompt ids, frame sampling and the
ViT, as fixtures for crates/cortiq-engine/tests/mimo_vision_parity.rs.

Needs torch + numpy (+ tokenizers/transformers/jinja2 for prompt ids,
safetensors for the toy and the real tower). No PIL: images are generated
here as numpy arrays and written as lossless PNGs with the stdlib, so the
engine decodes exactly the pixels the oracle preprocesses.

The processor functions below are copied from the upstream MiMo processor
(`MiMoProcessor.smart_resize`, `get_visual_transform[_batch]`,
`standardize_batch`, `_flatten_visual_inputs`, `format_timestamp`,
`_decode_frames_and_timestamps`, the video pixel budget) and sglang's
`smart_nframes`, verbatim where possible. The ViT is the HF
`MiMoVisionTransformer` from the checkpoint's modeling file, run in fp32,
patched as the serving stacks build it: `merger.ln_q` = RMSNorm(eps 1e-6)
and zero merger biases (the checkpoint has neither LayerNorm bias nor MLP
biases).

Commands:
  fixtures --out DIR [--src HF_DIR]   cheap fixtures: resize cases, images and
                                      their pixel rows, frame-sampling cases,
                                      prompt ids, the toy tower + its output
  vit      --out DIR [--src HF_DIR]   real-weight ViT outputs (fp32, patched)
                                      for the images/video in DIR/manifest.json,
                                      plus the unpatched-merger cosine (G3.3)
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import zlib
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_SRC = "/root/mimo/src"
SHARD = "model_pp0_ep0_shard0.safetensors"

# ─────────────── processor code (upstream MiMoProcessor, verbatim) ───────────────

_QWEN2VL_PIXEL_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
_QWEN2VL_PIXEL_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    if min(height, width) < factor:
        scale = factor / min(height, width)
        height = int(round(height * scale))
        width = int(round(width * scale))
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def standardize_batch(images):
    mean = _QWEN2VL_PIXEL_MEAN.view(1, -1, 1, 1)
    std = _QWEN2VL_PIXEL_STD.view(1, -1, 1, 1)
    return (images - mean) / std


def get_visual_transform_batch(frames, factor, min_pixels, max_pixels):
    _, _, h, w = frames.shape
    h_bar, w_bar = smart_resize(h, w, factor, min_pixels, max_pixels)
    resized = F.interpolate(frames.float(), size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    return standardize_batch(resized), w_bar, h_bar


def get_visual_transform(img_tensor, factor, min_pixels, max_pixels):
    img_tensor = img_tensor.float()
    _, h, w = img_tensor.shape
    h_bar, w_bar = smart_resize(h, w, factor, min_pixels, max_pixels)
    img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    return standardize_batch(img_resized).squeeze(0), w_bar, h_bar


def flatten_visual_inputs(visual, visual_type, patch_size=16, merge_size=2, temporal_patch_size=2):
    if visual_type == "image":
        resized_height, resized_width = visual.shape[-2:]
        patches = visual.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)
    else:
        assert len(visual) % temporal_patch_size == 0
        patches = visual
        resized_height, resized_width = patches.shape[-2:]
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    patches = patches.contiguous().view(
        grid_t, temporal_patch_size, channel,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
    flatten = patches.view(grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
    return flatten, [int(grid_t), int(grid_h), int(grid_w)]


def format_timestamp(timestamp):
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    return f"{minutes:02d}:{seconds:02d}"


# sglang qwen_vl.smart_nframes with the MiMo processor defaults.
FRAME_FACTOR = 2


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def smart_nframes(ele, total_frames, video_fps):
    fps = ele.get("fps", 2.0)
    min_frames = ceil_by_factor(ele.get("min_frames", 4), FRAME_FACTOR)
    max_frames = floor_by_factor(ele.get("max_frames", min(768, total_frames)), FRAME_FACTOR)
    nframes = total_frames / video_fps * fps
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


VIDEO_ELE = {"fps": 1.0, "min_frames": 8, "max_frames": 3600}
IMAGE_MIN, IMAGE_MAX = 8192, 8388608
VIDEO_MIN, VIDEO_MAX, VIDEO_TOTAL = 8192, 8388608, 268435456


def decode_indices_and_timestamps(total_frames, video_fps):
    nframes = smart_nframes(VIDEO_ELE, total_frames=total_frames, video_fps=video_fps)
    idx = list(np.unique(np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)))
    timestamps = torch.as_tensor(idx, dtype=torch.float32) / video_fps
    return nframes, [int(i) for i in idx], timestamps


def smart_resize_video(num_total_frames, min_pixels=VIDEO_MIN, max_pixels=VIDEO_MAX, total_max_pixels=VIDEO_TOTAL):
    max_pixels_per_frame = total_max_pixels * 2 * 1 // num_total_frames
    max_pixels = max(min_pixels, min(max_pixels_per_frame, max_pixels))
    return min_pixels, max_pixels


# ─────────────────────────────── helpers ───────────────────────────────


def write_png(path, arr):
    """Lossless 8-bit PNG (gray, RGB or RGBA) with filter 0 rows."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim == 2:
        ctype, h, w = 0, arr.shape[0], arr.shape[1]
    elif arr.shape[2] == 3:
        ctype, h, w = 2, arr.shape[0], arr.shape[1]
    elif arr.shape[2] == 4:
        ctype, h, w = 6, arr.shape[0], arr.shape[1]
    else:
        raise ValueError(arr.shape)
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def to_rgb(arr):
    """PIL convert('RGB') without PIL: drop alpha, replicate gray."""
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    return arr[:, :, :3]


def synth_image(h, w, seed, channels=3):
    """Structured content: gradients, rectangles, stripes and noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 255 * (x / max(w - 1, 1))
    img[..., 1] = 255 * (y / max(h - 1, 1))
    img[..., 2] = 127 + 120 * np.sin(x / 7.0 + y / 11.0)
    for _ in range(6):
        y0, x0 = rng.integers(0, h), rng.integers(0, w)
        hh, ww = rng.integers(h // 10 + 1, h // 3 + 2), rng.integers(w // 10 + 1, w // 3 + 2)
        img[y0:y0 + hh, x0:x0 + ww] = rng.integers(0, 256, 3)
    img[(x.astype(int) // 5) % 7 == 0] *= 0.5
    img += rng.normal(0, 6, img.shape)
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    if channels == 4:
        alpha = rng.integers(0, 256, (h, w, 1), dtype=np.uint8)
        return np.concatenate([img, alpha], axis=2)
    if channels == 1:
        return img.mean(axis=2).round().astype(np.uint8)
    return img


def save_f32(path, t):
    np.ascontiguousarray(np.asarray(t, dtype=np.float32)).tofile(path)


def image_rows(rgb):
    """P's image path on a raw HxWx3 uint8 array."""
    t = torch.from_numpy(rgb).permute(2, 0, 1).float()
    img, _, _ = get_visual_transform(t, 32, IMAGE_MIN, IMAGE_MAX)
    return flatten_visual_inputs(img, "image")


def load_hf(src):
    tmp = tempfile.mkdtemp(prefix="mimo_vis_hf_")
    pkg = os.path.join(tmp, "mimo_vis_hf_pkg")
    os.makedirs(pkg)
    for fn in ("configuration_mimo_v2.py", "modeling_mimo_v2.py"):
        shutil.copy(os.path.join(src, fn), pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    sys.path.insert(0, tmp)
    return importlib.import_module("mimo_vis_hf_pkg.modeling_mimo_v2")


def build_vit(modeling, vcfg, patched=True):
    vit = modeling.MiMoVisionTransformer(SimpleNamespace(**vcfg)).float().eval()
    hidden = vcfg["hidden_size"]
    if patched:
        vit.merger.ln_q = torch.nn.RMSNorm(hidden, eps=1e-6)
    return vit


def load_tower(vit, get, prefix="visual."):
    """Copy weights by name; merger biases / ln_q bias absent → zero."""
    missing = []
    with torch.no_grad():
        for k, p in vit.state_dict().items():
            t = get(prefix + k)
            if t is None:
                missing.append(k)
                p.zero_()
            else:
                assert tuple(t.shape) == tuple(p.shape), (k, t.shape, p.shape)
                p.copy_(t.float())
    return missing


# ─────────────────────────────── fixtures ───────────────────────────────

RESIZE_SIZES = [
    (20, 300), (300, 20), (5, 7), (31, 1000), (1, 1),
    (5000, 4000), (4000, 5000), (10000, 10000), (3000, 3000),
    (64, 64), (90, 90), (50, 180), (32, 32),
    (48, 80), (80, 48), (112, 144), (16 + 32 * 5, 16 + 32 * 7),
    (448, 448), (352, 640), (768, 1024), (1080, 1920), (1920, 1080), (333, 517), (300, 500),
    (32, 6432), (33, 6600), (32, 6400), (100, 20101), (2, 401), (7, 1500),
    (4096, 2048), (2049, 4097), (1023, 777),
]


def cmd_fixtures(a):
    out = a.out
    os.makedirs(out, exist_ok=True)
    man = {}

    # G2.1 smart_resize cases.
    cases = []
    for h, w in RESIZE_SIZES:
        for mn, mx in ((IMAGE_MIN, IMAGE_MAX), (8192, 200704), (0, 1 << 40)):
            try:
                cases.append({"h": h, "w": w, "min": mn, "max": mx, "out": list(smart_resize(h, w, 32, mn, mx))})
            except ValueError:
                cases.append({"h": h, "w": w, "min": mn, "max": mx, "error": True})
    man["resize_cases"] = cases

    # G2.2 images (+ the video for G7.3). PNG for the engine, rows for both.
    specs = [
        ("img448", 448, 448, 3, 1),
        ("img640x352", 352, 640, 3, 2),
        ("rgba300x500", 300, 500, 4, 3),
        ("gray333x517", 333, 517, 1, 4),
        ("img1024x768", 768, 1024, 3, 5),
        ("tiny20x300", 20, 300, 3, 6),
    ]
    imgs = []
    for name, h, w, ch, seed in specs:
        arr = synth_image(h, w, seed, ch)
        write_png(os.path.join(out, name + ".png"), arr)
        rgb = to_rgb(arr)
        np.save(os.path.join(out, name + ".npy"), rgb)
        rows, grid = image_rows(rgb)
        save_f32(os.path.join(out, name + ".rows.f32"), rows)
        imgs.append({"name": name, "png": name + ".png", "rows": name + ".rows.f32", "grid": grid})
    man["images"] = imgs

    # G7.3 video: 8 PNG frames at 1 fps.
    vdir = os.path.join(out, "video8")
    os.makedirs(vdir, exist_ok=True)
    frames = []
    for i in range(8):
        arr = synth_image(224, 320, 100 + i)
        arr[16 * i:16 * i + 40, 20 * i:20 * i + 60] = [255, 32 * i, 0]
        write_png(os.path.join(vdir, f"frame_{i + 1:03d}.png"), arr)
        frames.append(arr)
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    nframes, idx, ts = decode_indices_and_timestamps(len(frames), 1.0)
    sel = video[idx]
    mn, mx = smart_resize_video(sel.shape[0])
    vt, _, _ = get_visual_transform_batch(sel, 32, mn, mx)
    vrows, vgrid = flatten_visual_inputs(vt, "video")
    save_f32(os.path.join(out, "video8.rows.f32"), vrows)
    man["video"] = {"dir": "video8", "fps": 1.0, "rows": "video8.rows.f32", "grid": vgrid,
                    "indices": idx, "timestamps": [float(x) for x in ts],
                    "labels": [format_timestamp(t) for t in ts[::2]]}

    # G7.1 frame sampling.
    fcases = []
    for total, fps in [(5, 30.0), (7, 1.0), (9, 1.0), (1, 1.0), (2, 5.0), (3, 1.0), (8, 1.0),
                       (240, 24000 / 1001), (1001, 30000 / 1001), (600, 60.0), (3601, 1.0),
                       (4000, 1.0), (7200, 1.0), (100000, 30000 / 1001), (216000, 30.0),
                       (125, 25.0), (13, 0.5), (4801, 1.25), (86399, 23.976)]:
        try:
            n, idx, ts = decode_indices_and_timestamps(total, fps)
            idx_pad, ts_l = list(idx), [float(x) for x in ts]
            if len(idx_pad) % 2:
                idx_pad.append(idx_pad[-1])
                ts_l.append(ts_l[-1])
            _, mxp = smart_resize_video(len(idx))
            fcases.append({"total": total, "fps": fps, "n": n, "indices": idx, "timestamps": ts_l,
                           "labels": [format_timestamp(t) for t in torch.tensor(ts_l)[::2]],
                           "max_pixels": mxp})
        except ValueError:
            fcases.append({"total": total, "fps": fps, "error": True})
    man["frame_cases"] = fcases

    # G2.3 / G7.2 prompt ids.
    man["prompt_cases"] = prompt_cases(a.src, imgs)

    # G3.1 toy tower.
    man["toy"] = toy(a.src, out)

    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"wrote {out}/manifest.json: {len(cases)} resize, {len(imgs)} images, "
          f"{len(fcases)} frame cases, {len(man['prompt_cases'])} prompts")


def prompt_cases(src, imgs):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    at = AutoTokenizer.from_pretrained(src)
    enc = Tokenizer.from_file(os.path.join(src, "tokenizer.json"))

    def encode(text):
        return at.encode(text)

    ids_of = {t: at.convert_tokens_to_ids(t) for t in [
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
        "<|mimo_video_start|>", "<|mimo_video_end|>"]}
    rx = re.compile(r"(<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>|"
                    r"<\|vision_start\|>(?:<\|video_pad\|>)+<\|vision_end\|>)")

    grids = {i["name"]: i["grid"] for i in imgs}
    vgrid = [3, 14, 20]
    vts = [7.0, 7.5, 65.2, 66.0, 6000.0, 6000.5]
    cases = [
        {"name": "one_image", "images": ["img448"], "videos": [],
         "messages": [{"role": "user", "content": [{"type": "image", "image": "img448.png"},
                                                   {"type": "text", "text": "Describe this image."}]}]},
        {"name": "two_images_text_between", "images": ["img448", "img640x352"], "videos": [],
         "messages": [{"role": "user", "content": [
             {"type": "text", "text": "Compare "}, {"type": "image", "image": "a.png"},
             {"type": "text", "text": " with "}, {"type": "image_url", "image_url": {"url": "b.png"}},
             {"type": "text", "text": " — which one is brighter?"}]}]},
        {"name": "multi_turn_image", "images": ["img640x352"], "videos": [],
         "messages": [{"role": "system", "content": "You are a careful assistant."},
                      {"role": "user", "content": [{"type": "image", "image": "c.png"},
                                                   {"type": "text", "text": "What is shown?"}]},
                      {"role": "assistant", "content": "Coloured rectangles over a gradient."},
                      {"role": "user", "content": "How many rectangles?"}]},
        {"name": "video_timestamps", "images": [], "videos": [{"grid": vgrid, "timestamps": vts}],
         "messages": [{"role": "user", "content": [{"type": "video", "video": "v.y4m"},
                                                   {"type": "text", "text": "What happens at 01:05?"}]}]},
        {"name": "image_and_video", "images": ["gray333x517"], "videos": [{"grid": [4, 14, 20], "timestamps": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}],
         "messages": [{"role": "user", "content": [{"type": "image", "image": "g.png"},
                                                   {"type": "text", "text": "\n"},
                                                   {"type": "video", "video": "frames/"},
                                                   {"type": "text", "text": "Same scene?"}]}]},
    ]
    out = []
    for c in cases:
        text = at.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True)
        parts = rx.split(text)
        ids = []
        ii = vi = 0
        for part in parts:
            if rx.fullmatch(part):
                if "<|image_pad|>" in part:
                    g = grids[c["images"][ii]]
                    ii += 1
                    n = g[0] * g[1] * g[2] // 4
                    ids += [ids_of["<|vision_start|>"]] + [ids_of["<|image_pad|>"]] * n + [ids_of["<|vision_end|>"]]
                else:
                    v = c["videos"][vi]
                    vi += 1
                    gt, gh, gw = v["grid"]
                    per = gh * gw // 4
                    ts = torch.tensor(v["timestamps"], dtype=torch.float32)
                    ids.append(ids_of["<|mimo_video_start|>"])
                    for t in ts[::2]:
                        ids += encode(format_timestamp(t))
                        ids += [ids_of["<|vision_start|>"]] + [ids_of["<|video_pad|>"]] * per + [ids_of["<|vision_end|>"]]
                    ids.append(ids_of["<|mimo_video_end|>"])
            elif part:
                ids += encode(part)
        assert ii == len(c["images"]) and vi == len(c["videos"]), c["name"]
        # The raw ids of the rendered text (tokenize-once path) for reference.
        raw = enc.encode(text, add_special_tokens=False).ids
        out.append({**c, "text": text, "ids": ids, "raw_ids": raw})
    return out


TOY = dict(depth=4, hidden_size=64, intermediate_size=96, num_heads=4, num_key_value_heads=2,
           qk_channels=16, patch_size=4, temporal_patch_size=2, spatial_merge_size=2, in_chans=3,
           out_hidden_size=48, fullatt_block_indexes=[0, 3], vit_window_attn_types=[-1, 0, 1, -1],
           use_sink=True, visual_token_window_size=5, window_size=128, hidden_act="silu",
           rms_norm_eps=1e-6)


def toy(src, out):
    from safetensors.torch import save_file

    modeling = load_hf(src)
    torch.manual_seed(0)
    vit = build_vit(modeling, TOY, patched=True)
    sd = {}
    with torch.no_grad():
        for k, p in vit.named_parameters():
            if k.startswith("merger.mlp.") and k.endswith(".bias"):
                p.zero_()
                continue
            if k.endswith("norm1.weight") or k.endswith("norm2.weight") or k == "merger.ln_q.weight":
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif k.endswith("sinks"):
                p.copy_(torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * (0.5 / math.sqrt(p.shape[-1] if p.dim() > 1 else 16)))
            sd["visual." + k] = p.detach().clone().contiguous()
    if "patch_embed.proj.weight" not in dict(vit.named_parameters()):
        raise SystemExit("toy: patch embed missing")
    grid = [2, 8, 12]
    n = grid[0] * grid[1] * grid[2]
    rows = torch.randn(n, 3 * 2 * 4 * 4)
    with torch.no_grad():
        y = vit(rows, torch.tensor([grid]))
    save_file(sd, os.path.join(out, "toy.safetensors"))
    with open(os.path.join(out, "toy_config.json"), "w") as f:
        json.dump({"vision_config": TOY}, f)
    save_f32(os.path.join(out, "toy.rows.f32"), rows)
    save_f32(os.path.join(out, "toy.out.f32"), y)
    return {"safetensors": "toy.safetensors", "config": "toy_config.json", "rows": "toy.rows.f32",
            "out": "toy.out.f32", "grid": grid, "out_shape": list(y.shape)}


# ─────────────────────────────── real ViT ───────────────────────────────


def cmd_vit(a):
    from safetensors import safe_open

    torch.set_num_threads(a.threads)
    man_path = os.path.join(a.out, "manifest.json")
    man = json.load(open(man_path))
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    vcfg = cfg["vision_config"]
    modeling = load_hf(a.src)
    sf = safe_open(os.path.join(a.src, SHARD), framework="pt")
    keys = set(sf.keys())
    get = lambda k: sf.get_tensor(k) if k in keys else None
    vit = build_vit(modeling, vcfg, patched=True)
    missing = load_tower(vit, get)
    print("patched: zero-filled (absent in checkpoint):", missing)
    vit_lnq = build_vit(modeling, vcfg, patched=False)
    missing2 = load_tower(vit_lnq, get)
    print("unpatched: zero-filled:", missing2)

    vit_bf = None
    if a.bf16:
        # The serving stacks run the tower in bf16 (weights and activations):
        # its distance from fp32 is the reference's own noise floor.
        import copy
        vit_bf = copy.deepcopy(vit).to(torch.bfloat16)
    items = [(i["name"], i["rows"], i["grid"]) for i in man["images"] if i["name"] in a.images.split(",")]
    v = man["video"]
    items.append(("video8", v["rows"], v["grid"]))
    res = {}
    for name, rows_file, grid in items:
        n = grid[0] * grid[1] * grid[2]
        rows = torch.from_numpy(np.fromfile(os.path.join(a.out, rows_file), dtype=np.float32).reshape(n, -1))
        with torch.no_grad():
            y = vit(rows, torch.tensor([grid]))
            y2 = vit_lnq(rows, torch.tensor([grid]))
        save_f32(os.path.join(a.out, name + ".vit.f32"), y)
        cos = F.cosine_similarity(y, y2, dim=1)
        res[name] = {"out": name + ".vit.f32", "shape": list(y.shape),
                     "g3_3_ln_vs_rms_row_cos_mean": float(cos.mean()),
                     "g3_3_ln_vs_rms_row_cos_min": float(cos.min())}
        if vit_bf is not None:
            with torch.no_grad():
                yb = vit_bf(rows.to(torch.bfloat16), torch.tensor([grid])).float()
            cb = F.cosine_similarity(yb, y, dim=1)
            res[name]["bf16_vs_fp32_row_cos_mean"] = float(cb.mean())
            res[name]["bf16_vs_fp32_row_cos_min"] = float(cb.min())
        print(name, res[name])
    man["vit"] = res
    with open(man_path, "w") as f:
        json.dump(man, f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fixtures")
    f.add_argument("--out", required=True)
    f.add_argument("--src", default=DEFAULT_SRC)
    v = sub.add_parser("vit")
    v.add_argument("--out", required=True)
    v.add_argument("--src", default=DEFAULT_SRC)
    v.add_argument("--images", default="img448,img640x352,rgba300x500,gray333x517,img1024x768")
    v.add_argument("--threads", type=int, default=16)
    v.add_argument("--bf16", action="store_true", help="also run the tower in bf16 and record its row cosine to fp32")
    a = p.parse_args()
    {"fixtures": cmd_fixtures, "vit": cmd_vit}[a.cmd](a)


if __name__ == "__main__":
    main()
