#!/usr/bin/env python3
"""Generate compact, source-authoritative MOSS-Image validation fixtures.

The native CMF implementation deliberately has separate tests for the parts
that are easy to subtly change while optimizing: the custom V2 snapped Flow
Match schedule, Qwen3.5 chat formatting, and Flux2 BatchNorm/unpatch decode
prelude.  This script invokes the *pinned source pipeline* for the schedule
and tokenizer and writes a deterministic small VAE fixture.  It does not load
the 8B text tower or 5B DiT, so it is cheap enough to run on a rented pod
before a full end-to-end parity run.

Typical pod use::

  python3 python/moss_image_oracle.py \
      --root /workspace/models/moss-image-base \
      --out /workspace/oracles/moss-image-base

It needs the source-compatible ``torch``, ``diffusers``, ``transformers``,
``safetensors`` and ``Pillow`` packages.  The output directory is deliberately
not checked into git: it contains binary numerical oracle data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer


DEFAULT_PROMPTS = [
    "A small red fox reading a book in a sunlit library, highly detailed.",
    "A stainless-steel refrigerator with the Chinese text '新鲜每一天' on its door.",
    "",  # CFG negative default must travel through the exact same template.
]


def f32(path: Path, value: torch.Tensor) -> None:
    """Write a CPU float32 tensor in the Rust runtime's raw little-endian form."""
    value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype("<f4").tofile(path)


def module_from_source(root: Path) -> Any:
    source = root / "pipeline_moss_image.py"
    if not source.is_file():
        # The Hub snapshot currently puts an identical custom component below
        # transformer/.  Keeping the fallback makes a component export usable.
        source = root / "transformer" / "pipeline_moss_image.py"
    if not source.is_file():
        raise FileNotFoundError(f"no pipeline_moss_image.py beneath {root}")
    spec = importlib.util.spec_from_file_location("moss_image_source", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def schedule_fixture(source: Any) -> dict[str, Any]:
    out = []
    # Include an awkward small count plus official 50-step quality default.
    for steps in (1, 2, 8, 50):
        s = source.MOSSImageScheduler()
        s.set_timesteps(steps, device="cpu")
        # Diffusers' current scheduler stores a terminal zero in sigmas.  Keep
        # both tensors so a future diffusers behavior change cannot silently
        # mutate the native contract.
        out.append(
            {
                "steps_requested": steps,
                "steps_effective": int(s.timesteps.numel()),
                "timesteps": [float(x) for x in s.timesteps.detach().cpu().float().tolist()],
                "sigmas": [float(x) for x in s.sigmas.detach().cpu().float().tolist()],
            }
        )
    return {"class": "MOSSImageScheduler", "schedules": out}


def tokenizer_fixture(root: Path, prompts: list[str]) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    rows = []
    for prompt in prompts:
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        # MOSS calls tokenizer(formatted), rather than asking template helper
        # to tokenize, so preserve that exact path and no special truncation.
        ids = tok(formatted, truncation=True, max_length=4096)["input_ids"]
        rows.append({"prompt": prompt, "formatted": formatted, "ids": [int(x) for x in ids]})
    return {
        "class": tok.__class__.__name__,
        "vocab_size": int(getattr(tok, "vocab_size", 0)),
        "rows": rows,
    }


def flux2_fixture(root: Path, out: Path) -> dict[str, Any]:
    """Save exact Flux2 BN/unpatch (and small full decode when available)."""
    vae_dir = root / "vae"
    cfg = json.loads((vae_dir / "config.json").read_text())
    weights = vae_dir / "diffusion_pytorch_model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as sf:
        mean = sf.get_tensor("bn.running_mean").float()
        var = sf.get_tensor("bn.running_var").float()
    c4 = int(mean.numel())
    if c4 % 4:
        raise AssertionError(f"Flux2 BN has {c4} channels, expected a multiple of four")
    # Non-square grid catches accidental (2,2,H,W) permutation mistakes.
    h, w = 3, 5
    packed = torch.linspace(-0.75, 0.75, c4 * h * w, dtype=torch.float32).reshape(1, c4, h, w)
    eps = float(cfg["batch_norm_eps"])
    z = packed * (var + eps).sqrt().view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    unpatched = z.reshape(1, c4 // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, c4 // 4, 2 * h, 2 * w)
    f32(out / "vae_packed_latent.f32", packed[0])
    f32(out / "vae_unpatched.f32", unpatched[0])
    result: dict[str, Any] = {
        "batch_norm_eps": eps,
        "packed_shape": list(packed.shape[1:]),
        "unpatched_shape": list(unpatched.shape[1:]),
        "packed": "vae_packed_latent.f32",
        "unpatched": "vae_unpatched.f32",
        "decoder_reference": None,
    }
    # Full Flux2 VAE is only ~238 MiB.  Its decoded tiny image catches a
    # missing post_quant_conv as well as BN/unpatch ordering errors.
    try:
        from diffusers import AutoencoderKLFlux2

        vae = AutoencoderKLFlux2.from_pretrained(vae_dir, local_files_only=True, torch_dtype=torch.float32)
        vae.eval().requires_grad_(False)
        with torch.inference_mode():
            decoded = vae.decode(unpatched, return_dict=False)[0]
        f32(out / "vae_decode_ref.f32", decoded[0])
        result["decoded_shape"] = list(decoded.shape[1:])
        result["decoder_reference"] = "vae_decode_ref.f32"
    except Exception as exc:  # fixture still useful if an older Diffusers lacks Flux2
        result["decoder_unavailable"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="pinned MOSS-Image source snapshot")
    ap.add_argument("--out", required=True, type=Path, help="new/existing oracle output directory")
    ap.add_argument("--prompt", action="append", default=None, help="additional prompt; repeatable")
    args = ap.parse_args()
    root = args.root.resolve()
    if not (root / "model_index.json").is_file():
        ap.error(f"{root} is not a MOSS diffusers root")
    index = json.loads((root / "model_index.json").read_text())
    if index.get("_class_name") != "MOSSImagePipeline":
        ap.error(f"{root}: model_index is {index.get('_class_name')!r}, not MOSSImagePipeline")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = module_from_source(root)
    prompts = DEFAULT_PROMPTS + (args.prompt or [])
    manifest = {
        "source_root": str(root),
        "model_index": index,
        "torch": torch.__version__,
        "schedule": schedule_fixture(source),
        "tokenizer": tokenizer_fixture(root, prompts),
        "flux2_vae": flux2_fixture(root, out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "schedules": len(manifest["schedule"]["schedules"]), "prompts": len(prompts), **manifest["flux2_vae"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
