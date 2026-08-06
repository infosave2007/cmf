#!/usr/bin/env python3
"""A toy MiniMax-H3 and its golden forward, from ComfyUI's own module.

The port in `crates/cortiq-engine/src/mmh3.rs` has to agree with the
reference on a dozen conventions that all pass at a single token and all
fail differently at a hundred: the packed layout's cursor, the video t
axis's 1,4,4,4,4 spans, which 96 of 128 head dims rotate, the adaLN row
order (timestep-major, modality-minor), and the audio stream's separate
clock. This builds a small checkpoint carrying the release's REAL tensor
names in the pruned/curve form, runs `MiniMaxH3Model` on it, and writes
inputs and outputs as flat f32 for `tests/mmh3_parity.rs`.

    python3 tools/mk_mmh3_toy.py --comfyui /root/ComfyUI --out /root/mmh3toy

The audio output is written with the schedule slope DIVIDED OUT: the
reference pre-multiplies it by d(sigma_a)/d(sigma_v) so a single-schedule
sampler is approximately right, and this port returns each stream on its
own clock instead.
"""

import argparse
import json
import math
import os
import sys

import torch


CFG = dict(
    hidden_size=96,
    num_layers=2,
    token_refiner_num_layers=1,
    num_attention_heads=6,
    attention_head_dim=16,
    ffn_hidden_size=64,
    latents_dim=4,
    audio_latents_dim=6,
    patch_size=(1, 2, 2),
    text_dim=8,
    time_embed_dim=24,
    rope_inv_freq_len=2,
    norm_eps=1e-5,
    qk_norm_eps=1e-5,
    final_norm_eps=1e-5,
    sigma_shift_video=12.0,
    sigma_shift_audio=3.0,
    adaln_curve_grid=1025,
)

SHAPE = dict(text_len=7, latent_t=3, lat_h=8, lat_w=12, audio_t=5)
SIGMA = 0.923077  # the third rung of the 4-step simple schedule at shift 12


def install_ck_shim():
    """A pure-torch stand-in for `comfy_kitchen`'s fused rms+rope.

    The real kernel is CUDA-only, so a host without one cannot run the
    reference at all. This is NOT taken on trust: the t2va golden it
    produces is compared against the same port that already matched the
    REAL kernel on a CUDA box, so a shim that got the convention wrong
    would move those numbers and fail the existing test.
    """
    import types

    def _apply(x, freqs, w, eps, rot_dim):
        # x: [1, s, heads, hd]; freqs: [1, S, 1, rot/2, 2, 2]
        hd = x.shape[-1]
        var = x.float().pow(2).mean(-1, keepdim=True)
        x.copy_((x.float() * torch.rsqrt(var + eps)).to(x.dtype) * w)
        half = rot_dim // 2
        c = freqs[..., 0, 0]              # [1, S, 1, half]
        sn = freqs[..., 1, 0]             # sin
        a = x[..., :half]
        b = x[..., half:rot_dim]
        na = a * c - b * sn
        nb = a * sn + b * c
        x[..., :half] = na
        x[..., half:rot_dim] = nb

    def rms_rope_split_half_(q, k, freqs_cis, q_scale, k_scale, epsilon=1e-5, rot_dim=None):
        rot_dim = rot_dim or q.shape[-1]
        _apply(q, freqs_cis, q_scale, epsilon, rot_dim)
        _apply(k, freqs_cis, k_scale, epsilon, rot_dim)

    def rms_rope_split_half(q, k, freqs_cis, q_scale, k_scale, epsilon=1e-5, rot_dim=None):
        q, k = q.clone(), k.clone()
        rms_rope_split_half_(q, k, freqs_cis, q_scale, k_scale, epsilon, rot_dim)
        return q, k

    ck = types.SimpleNamespace(
        rms_rope_split_half_=rms_rope_split_half_,
        rms_rope_split_half=rms_rope_split_half,
    )
    import comfy.quant_ops
    if not hasattr(comfy.quant_ops, "ck"):
        comfy.quant_ops.ck = ck
        print("comfy_kitchen absent — using the torch stand-in "
              "(the t2va golden below is what checks it)", file=sys.stderr)


def build(comfyui, seed=1234):
    sys.path.insert(0, comfyui)
    import comfy.ops
    import comfy.quant_ops
    install_ck_shim()
    import comfy.ldm.minimax.model as mm

    torch.manual_seed(seed)
    ops = comfy.ops.disable_weight_init
    model = mm.MiniMaxH3Model(dtype=torch.float32, device="cpu", operations=ops, **CFG)

    # Random but well-scaled: adaLN gates near zero would make every
    # block a no-op and hide exactly the bugs this is for.
    sd = model.state_dict()
    for k, v in sd.items():
        if v.dtype.is_floating_point:
            if k.endswith("norm1.weight") or k.endswith("norm2.weight") or k.endswith("_norm.weight"):
                sd[k] = torch.ones_like(v) + 0.1 * torch.randn_like(v)
            elif k.endswith("final_norm.weight") or k == "final_layer.norm.weight":
                sd[k] = torch.ones_like(v) + 0.1 * torch.randn_like(v)
            elif k == "rope.inv_freq":
                sd[k] = 1.0 / (10000.0 ** (torch.arange(v.numel(), dtype=torch.float32) / v.numel()))
            elif k == "adaln_t_table":
                # a smooth curve, as the real table is
                t = torch.linspace(0, 1, v.shape[0])[:, None]
                f = torch.arange(1, v.shape[1] + 1, dtype=torch.float32)[None]
                sd[k] = torch.sin(t * f * 1.7) + 0.3 * torch.cos(t * f * 0.9)
            else:
                sd[k] = 0.05 * torch.randn_like(v)
    model.load_state_dict(sd)
    model.eval()
    # The fused rms+rope kernel refuses to run in place under autograd,
    # and every parameter here is a leaf that wants grad by default.
    model.requires_grad_(False)
    return model, sd


def golden(model, out_dir):
    s = SHAPE
    torch.manual_seed(7)
    video = torch.randn(1, CFG["latents_dim"], s["latent_t"], s["lat_h"], s["lat_w"])
    audio = torch.randn(1, CFG["audio_latents_dim"], 2, s["audio_t"])
    text = torch.randn(1, s["text_len"], CFG["text_dim"])
    ts = torch.tensor([SIGMA * 1000.0])

    with torch.no_grad():
        # The refined text stream, so the port can be checked in two
        # halves rather than one.
        refined = model.preprocess_text_embeds(text)
        v_out, a_out = model._forward([video, audio], ts, text, {}, minimax_payload={})

        # fl2va: the same clip conditioned on a first and a last frame.
        # `visual_cond_noise_aug=1.0` turns off the 0.1% noise blend,
        # whose stream comes from a torch-seeded generator this port
        # does not reproduce — everything else is compared exactly.
        frame_count = 39
        kf = [torch.randn(1, CFG["latents_dim"], 1, s["lat_h"], s["lat_w"]) for _ in range(2)]
        payload = {
            "keyframes": [{"resolved_frame_index": 0},
                          {"resolved_frame_index": frame_count - 1}],
            "frame_count": frame_count,
            "cond_video_latents": kf,
            "visual_cond_noise_aug": 1.0,
        }
        v_kf, a_kf = model._forward([video, audio], ts, text, {}, minimax_payload=payload)

    slope = model_slope(SIGMA)
    a_unscaled = a_out / (-slope) * (-1.0)  # undo the reference's (-slope) factor
    os.makedirs(out_dir, exist_ok=True)

    def dump(name, t):
        t.detach().float().contiguous().numpy().astype("<f4").tofile(
            os.path.join(out_dir, name))

    dump("video_in.bin", video)
    dump("audio_in.bin", audio)
    dump("text_in.bin", text)
    dump("text_refined.bin", refined)
    dump("video_out.bin", v_out)
    dump("audio_out.bin", a_unscaled)
    for i, k in enumerate(kf):
        dump(f"kf{i}.bin", k)
    dump("kf_video_out.bin", v_kf)
    dump("kf_audio_out.bin", a_kf / (-slope) * (-1.0))
    meta = dict(CFG)
    meta["patch_size"] = list(CFG["patch_size"])
    meta.update(s)
    meta["sigma"] = SIGMA
    meta["slope"] = slope
    meta["frame_count"] = frame_count
    meta["n_keyframes"] = len(kf)
    with open(os.path.join(out_dir, "golden.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("video_out", tuple(v_out.shape), "rms", float(v_out.pow(2).mean().sqrt()))
    print("audio_out", tuple(a_out.shape), "rms", float(a_out.pow(2).mean().sqrt()))


def model_slope(sv):
    fr, to = CFG["sigma_shift_video"], CFG["sigma_shift_audio"]
    base = sv / (fr + sv * (1.0 - fr))
    return (to * (1.0 + (fr - 1.0) * base) ** 2) / (fr * (1.0 + (to - 1.0) * base) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, sd = build(a.comfyui)
    os.makedirs(a.out, exist_ok=True)
    from safetensors.torch import save_file
    # `mask_token`-style buffers and anything the packer ignores ride
    # along; the packer picks by name.
    save_file({k: v.contiguous() for k, v in sd.items()},
              os.path.join(a.out, "dit.safetensors"))
    golden(model, a.out)
    print(a.out)
    print(f"{sum(v.numel() for v in sd.values())} parameters, "
          f"seq_len {SHAPE['text_len'] + 2 * SHAPE['audio_t'] + SHAPE['latent_t'] * (SHAPE['lat_h'] // 2) * (SHAPE['lat_w'] // 2)}")


if __name__ == "__main__":
    main()
