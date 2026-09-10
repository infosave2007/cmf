"""Lumina-Image 2.0: cortiq against the diffusers reference, same machine.

    pip install torch diffusers transformers accelerate
    hf download Alpha-VLLM/Lumina-Image-2.0 --local-dir lumina-ref
    hf download infosave/Lumina-Image-2.0cmf lumina-q4tp.cmf --local-dir .

    python3 tools/bench_lumina.py --ref lumina-ref --cmf lumina-q4tp.cmf \
        --cortiq ./target/release/cortiq

Prints one JSON line per arm. Three things are measured and each is
reported separately, because they answer different questions:

  * warm render — what a second image costs
  * cold total — what the first one costs, load included
  * peak memory — whether the machine can run it at all

`--same-noise` additionally writes the reference's starting latent to a
file and hands it to cortiq through `CMF_INIT_LATENT`, so the two images
can be compared pixel to pixel. Without it the two draw different noise
and any such comparison measures the generators, not the arithmetic.
"""

import argparse
import json
import os
import re
import subprocess
import time

PROMPT = "a red fox sitting in snow at sunset, photorealistic, detailed fur"


def du_bytes(path):
    out = subprocess.run(["du", "-sb", path], capture_output=True, text=True).stdout
    return int(out.split()[0]) if out else os.path.getsize(path)


def run_reference(args, latents=None):
    import torch
    from diffusers import Lumina2Pipeline

    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    dtype = torch.bfloat16 if cuda else torch.float32
    if not cuda:
        torch.set_num_threads(os.cpu_count() or 1)

    t = time.time()
    pipe = Lumina2Pipeline.from_pretrained(args.ref, torch_dtype=dtype).to(dev)
    load = time.time() - t
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    def once(tag, lat):
        g = None if lat is not None else torch.Generator(dev).manual_seed(args.seed)
        t = time.time()
        img = pipe(
            prompt=PROMPT,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=g,
            latents=lat,
        ).images[0]
        if cuda:
            torch.cuda.synchronize()
        dt = time.time() - t
        img.save(f"ref_{tag}.png")
        with open(f"ref_{tag}.ppm", "wb") as f:
            f.write(b"P6\n%d %d\n255\n" % img.size)
            f.write(img.tobytes())
        return dt

    if latents is not None:
        lat = latents.to(dev, dtype)
        return {"arm": "reference", "device": dev, "render_s": round(once("fixed", lat), 1)}

    first = once("cold", None)
    warm = min(once(f"warm{i}", None) for i in range(2))
    peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
    return {
        "arm": "reference",
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "dtype": str(dtype).replace("torch.", ""),
        "load_s": round(load, 1),
        "first_render_s": round(first, 1),
        "warm_render_s": round(warm, 1),
        "cold_total_s": round(load + first, 1),
        "peak_vram_gb": round(peak, 2),
        "disk_gb": round(du_bytes(args.ref) / 2**30, 2),
    }


def run_cortiq(args, env_extra=None):
    env = dict(os.environ, CMF_GPU="0" if args.cpu else "1")
    env.update(env_extra or {})
    t = time.time()
    out = subprocess.run(
        [
            args.cortiq, "imagine", args.cmf,
            "--prompt", PROMPT,
            "--height", str(args.size), "--width", str(args.size),
            "--steps", str(args.steps), "--cfg", str(args.cfg),
            "--seed", str(args.seed), "--out", "cmf.ppm",
        ],
        capture_output=True, text=True, env=env,
    )
    dt = time.time() - t
    if out.returncode != 0:
        raise SystemExit(out.stderr[-2000:])
    steps = re.search(r"steps in ([0-9.]+)s", out.stdout + out.stderr)
    return {
        "arm": "cortiq",
        "device": "cpu" if args.cpu else "gpu",
        "cold_total_s": round(dt, 1),
        "reported_s": float(steps.group(1)) if steps else None,
        "disk_gb": round(du_bytes(args.cmf) / 2**30, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="diffusers checkout")
    p.add_argument("--cmf", required=True, help="lumina-q4tp.cmf")
    p.add_argument("--cortiq", default="cortiq")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cpu", action="store_true", help="cortiq on the CPU")
    p.add_argument("--same-noise", action="store_true")
    args = p.parse_args()

    if args.same_noise:
        import torch

        g = torch.Generator("cpu").manual_seed(args.seed)
        lat = torch.randn(1, 16, args.size // 8, args.size // 8, generator=g)
        # Round once here, so the file holds exactly what the pipeline sees.
        lat = lat.to(torch.bfloat16).to(torch.float32)
        lat.numpy().tofile("init_latent.f32")
        print(json.dumps(run_reference(args, lat)))
        print(json.dumps(run_cortiq(args, {"CMF_INIT_LATENT": "init_latent.f32"})))
        print("compare ref_fixed.ppm against cmf.ppm")
        return

    print(json.dumps(run_reference(args)))
    print(json.dumps(run_cortiq(args)))


if __name__ == "__main__":
    main()
