#!/usr/bin/env python3
"""Reference latents and reconstructions of ostris/primal_vae (float32, CUDA).

usage: primal_vae_ref.py <repo_dir> <out_dir> <image> [<image> ...]
Writes <out>/<variant>_<name>_{in,z,rec}.npy and prints PSNR per image.
"""
import sys, os, importlib.util
import numpy as np
import torch
from PIL import Image

repo, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
spec = importlib.util.spec_from_file_location("modeling_primal_vae", os.path.join(repo, "modeling_primal_vae.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
for variant in ("2x1", "2x2"):
    vae = mod.PrimalVAE.from_pretrained(repo, variant=variant, torch_dtype=torch.float32).cuda().eval()
    for path in sys.argv[3:]:
        name = os.path.splitext(os.path.basename(path))[0]
        im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        x = torch.from_numpy(im / 127.5 - 1.0).permute(2, 0, 1)[None].contiguous().cuda()
        with torch.inference_mode():
            z = vae.encode(x)
            rec = vae.decode(z)
        q = lambda t: ((t.clamp(-1, 1) + 1) * 127.5).round()
        mse = ((q(rec) - q(x)) ** 2).mean().item()
        psnr = 10 * np.log10(255.0 ** 2 / mse)
        np.save(f"{out}/{variant}_{name}_in.npy", x.cpu().numpy())
        np.save(f"{out}/{variant}_{name}_z.npy", z.float().cpu().numpy())
        np.save(f"{out}/{variant}_{name}_rec.npy", rec.float().cpu().numpy())
        print(f"{variant} {name} {tuple(x.shape)} z {tuple(z.shape)} mean {z.mean().item():.3f} std {z.std().item():.3f} PSNR {psnr:.2f} dB")
