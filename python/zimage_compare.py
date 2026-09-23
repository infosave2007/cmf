#!/usr/bin/env python3
"""Compare a cortiq Z-Image run (CMF_ZIMAGE_TRACE dir + output PNG) against a
diffusers oracle run (`run_*.safetensors` / `.png` from zimage_oracle.py).

  python zimage_compare.py --oracle /root/zimage/oracles/v1/turbo/run_r512_p0_t8_fp32 \
      --trace /root/za/trace/t512 --png out.png [--ref-png other.png ...]

Prints per-step rel/cos of v_i (guided prediction), vpos_i/vneg_i under CFG,
lat_i, and PSNR (u8 RGB) of the PNG against the oracle PNG and any extra
reference PNGs (e.g. the diffusers bf16 image, to show the bf16 floor).
"""
import argparse
import json
import os

import numpy as np
from safetensors.numpy import load_file


def rel(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)), \
        float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def load_png(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, help="run_* path without extension")
    ap.add_argument("--trace", help="CMF_ZIMAGE_TRACE dir")
    ap.add_argument("--png", help="cortiq output image")
    ap.add_argument("--ref-png", action="append", default=[])
    ap.add_argument("--json", help="append a JSON line with the numbers here")
    a = ap.parse_args()
    o = load_file(a.oracle + ".safetensors")
    res = {"oracle": os.path.basename(a.oracle)}
    shape = o["noise"].shape
    if a.trace:
        steps = len([k for k in o if k.startswith("v_")])
        for i in range(steps):
            for nm in ("vpos_%d" % i, "vneg_%d" % i, "v_%d" % i, "lat_%d" % (i + 1)):
                p = os.path.join(a.trace, nm + ".f32")
                if nm in o and os.path.exists(p):
                    got = np.fromfile(p, dtype="<f4").reshape(shape)
                    r, c = rel(got, o[nm])
                    res[nm] = r
                    print("%-8s rel %.3e  cos %.9f" % (nm, r, c))
    if a.png:
        img = load_png(a.png)
        ref = o["img_u8"]
        res["psnr_vs_oracle"] = psnr(img, ref)
        print("PSNR vs %s: %.2f dB" % (os.path.basename(a.oracle), res["psnr_vs_oracle"]))
        for rp in a.ref_png:
            v = psnr(img, load_png(rp))
            res["psnr_vs_" + os.path.basename(rp)] = v
            print("PSNR vs %s: %.2f dB" % (os.path.basename(rp), v))
    for rp in a.ref_png:
        v = psnr(load_png(rp), o["img_u8"])
        res["floor_" + os.path.basename(rp)] = v
        print("(floor) %s vs %s: %.2f dB" % (os.path.basename(rp), os.path.basename(a.oracle), v))
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
