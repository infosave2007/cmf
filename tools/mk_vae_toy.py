#!/usr/bin/env python3
"""Toy video and audio VAE decoders, and their golden outputs.

Both decoders have a schedule the port has to reproduce exactly rather
than approximately. The video one always decodes in 256-pixel tiles and
17-frame clips — because it is global attention, a tile is a different
computation from a whole frame, so the tiling is part of the output.
The audio one wraps every nonlinearity in a kaiser-sinc resample whose
filter is designed at load time, not shipped in the checkpoint.

The toys keep the real spatial/temporal ratios (16 and 4) and the real
upsample rates, so the schedules are the release's; only the widths
shrink.

    python3 tools/mk_vae_toy.py --comfyui /root/ComfyUI --out /root/vaetoy
"""

import argparse
import json
import os
import sys

import torch

# z_channels is pinned at 24: the module's latents_mean/std buffers are
# the release's literal 24-entry lists whatever the constructor is told.
VID = dict(z_channels=24, num_layers=2, heads=2, dim_head=16,
           latent_t=12, lat_h=20, lat_w=20)
AUD = dict(latent_dim=32, decoder_dim=128, vae_latent_channels=4, audio_t=4)


def video(comfyui, out_dir):
    import comfy.ldm.minimax.vae as V

    torch.manual_seed(11)
    # ch is bounded below by the encoder's GroupNorm(32); the encoder is
    # never run here, but it is still constructed.
    vae = V.MiniMaxH3VideoVAE(
        ch=32, ch_mult=(1, 1, 1, 1, 1, 1), num_res_blocks=1,
        embed_dim=VID["z_channels"], z_channels=VID["z_channels"],
        space_down=(2, 2, 2, 2, 1, 1), time_down=(1, 2, 2, 1, 1, 1))
    vae.decoder = V.ViT3DDecoder(
        patch_size=vae.vae_ratio, patch_size_t=vae.vae_ratio_t,
        in_channels=VID["z_channels"], out_channels=3,
        num_layers=VID["num_layers"], heads=VID["heads"], dim_head=VID["dim_head"])
    sd = vae.state_dict()
    for k, v in sd.items():
        if not v.dtype.is_floating_point:
            continue
        if k.endswith("norm1.weight") or k.endswith("norm2.weight") or k.endswith("norm_out.weight"):
            sd[k] = torch.ones_like(v) + 0.1 * torch.randn_like(v)
        elif k.endswith("scale1") or k.endswith("scale2"):
            sd[k] = 0.3 + 0.1 * torch.randn_like(v)
        elif k == "latents_std":
            sd[k] = 1.0 + 0.2 * torch.rand_like(v)
        else:
            sd[k] = 0.1 * torch.randn_like(v)
    vae.load_state_dict(sd)
    vae.eval().requires_grad_(False)

    z = torch.randn(1, VID["z_channels"], VID["latent_t"], VID["lat_h"], VID["lat_w"])
    with torch.no_grad():
        dec = vae.decode(z)          # [-1, 1]
    rgb = (dec + 1.0) / 2.0          # what the port returns

    # A single keyframe through the encoder — the only path fl2va needs,
    # and the one where a causal 3-D kernel collapses to its last tap.
    fh, fw = VID["lat_h"] * 16, VID["lat_w"] * 16
    frame = torch.randn(1, 3, 1, fh, fw).clamp(-1, 1)
    with torch.no_grad():
        zf = vae.encode(frame)
    frame.numpy().astype("<f4").tofile(os.path.join(out_dir, "frame_in.bin"))
    zf.float().numpy().astype("<f4").tofile(os.path.join(out_dir, "frame_z.bin"))
    print("encode", tuple(zf.shape), "rms", float(zf.pow(2).mean().sqrt()))

    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in sd.items()},
              os.path.join(out_dir, "video_vae.safetensors"))
    z.numpy().astype("<f4").tofile(os.path.join(out_dir, "video_z.bin"))
    rgb.float().numpy().astype("<f4").tofile(os.path.join(out_dir, "video_rgb.bin"))
    print("video", tuple(dec.shape), "rms", float(rgb.pow(2).mean().sqrt()))
    return dict(VID, frames=int(dec.shape[2]),
                height=int(dec.shape[3]), width=int(dec.shape[4]),
                frame_h=fh, frame_w=fw,
                enc_zh=int(zf.shape[3]), enc_zw=int(zf.shape[4]))


def audio(comfyui, out_dir):
    import comfy.ldm.minimax.audio_vae as A

    torch.manual_seed(13)
    vae = A.MiniMaxH3AudioVAE(encoder_dim=8, latent_dim=AUD["latent_dim"],
                              decoder_dim=AUD["decoder_dim"],
                              vae_latent_channels=AUD["vae_latent_channels"])
    sd = vae.state_dict()
    for k, v in sd.items():
        if not v.dtype.is_floating_point:
            continue
        if k.endswith(".alpha") or k.endswith(".beta"):
            sd[k] = 0.2 * torch.randn_like(v)      # log scale
        elif k == "latents_std":
            sd[k] = 1.0 + 0.2 * torch.rand_like(v)
        elif k == "latents_mean":
            sd[k] = 0.1 * torch.randn_like(v)
        else:
            sd[k] = 0.1 * torch.randn_like(v)
    vae.load_state_dict(sd)
    vae.eval().requires_grad_(False)

    c, t = AUD["vae_latent_channels"], AUD["audio_t"]
    z = torch.randn(1, c, 2, t)
    with torch.no_grad():
        wav = vae.decode(z)          # [1, 2, L]

    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in sd.items()},
              os.path.join(out_dir, "audio_vae.safetensors"))
    z.numpy().astype("<f4").tofile(os.path.join(out_dir, "audio_z.bin"))
    wav.float().numpy().astype("<f4").tofile(os.path.join(out_dir, "audio_wav.bin"))
    print("audio", tuple(wav.shape), "rms", float(wav.pow(2).mean().sqrt()))
    return dict(AUD, samples=int(wav.shape[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.comfyui)
    os.makedirs(a.out, exist_ok=True)
    meta = {"video": video(a.comfyui, a.out), "audio": audio(a.comfyui, a.out)}
    with open(os.path.join(a.out, "golden.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(a.out)


if __name__ == "__main__":
    main()
