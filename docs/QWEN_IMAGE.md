# Qwen Image Edit 2509

This guide covers the native Rust path for
[Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509).
Provide one or more reference images with `--image` and a text instruction.
The ready release is a self-contained CMF; the standalone component layout
remains available when users need to stage or override components independently.

## One-file use

The default release file is `qwen-image-edit-2509-q4tp.cmf`. Download the
[ready CMF](https://huggingface.co/infosave/Qwen-Image-Edit-2509-cmf/resolve/main/qwen-image-edit-2509-q4tp.cmf),
save it under that filename, and run it directly; no conversion or companion
files are required:

```sh
cortiq imagine qwen-image-edit-2509-q4tp.cmf \
  --image docs/media/fox-512.png \
  --prompt "Add a vivid blue knitted scarf while preserving the fox, pose, and snowy background." \
  --height 512 --width 512 --steps 30 --cfg 4 --seed 7 \
  --reference-size 1024 --out fox-scarf.png
```

## Standalone components

Place the pinned transformer GGUF and companion files under
`qwen-image/source/` (see [Source references](#source-references)), then import
the GGUF and pack the official companion directories:

```sh
cortiq import-gguf qwen-image/source/Qwen-Image-Edit-2509-Q6_K.gguf \
  --quant q4tp --output qwen-image/transformer.cmf
cortiq imagine-pack --component qwen-text-encoder --quant q4tp \
  --out qwen-image/text_encoder.cmf qwen-image/source
cortiq imagine-pack --component qwen-vae --quant f16 \
  --out qwen-image/vae.cmf qwen-image/source
cp qwen-image/source/scheduler/scheduler_config.json qwen-image/
cortiq imagine-pack --bundle qwen-image \
  --out qwen-image/qwen-image-edit-2509-q4tp.cmf
```

Run with explicit component overrides:

```sh
cortiq imagine qwen-image/transformer.cmf \
  --text-encoder qwen-image/text_encoder.cmf \
  --vae qwen-image/vae.cmf \
  --scheduler qwen-image/scheduler_config.json \
  --image docs/media/fox-512.png \
  --prompt "Turn the scene into a watercolor illustration." \
  --height 512 --width 512 --steps 30 --cfg 4 --seed 7 \
  --reference-size 1024 --out watercolor.png
```

The bundle defaults its embedded transformer, encoder, VAE, and scheduler;
explicit `--text-encoder`, `--vae`, and `--scheduler` options remain available.

## Input limits

Repeat `--image` for multiple references; the prompt uses the official
Qwen2.5-VL processor/template. The canonical `--reference-size 1024` is the
square root of the VAE reference area and is independent of output dimensions.
Keep output dimensions on multiples of 16. CFG at or below 1 omits the
unconditional branch and changes the guidance profile. PNG, JPEG, and PPM output
are supported.

## Backends and measured profile

Set `CMF_GPU=0` for the portable CPU path. For a headless Vulkan run, prefix the
command with `XDG_RUNTIME_DIR=/tmp WGPU_BACKEND=vulkan CMF_GPU=1`. On a capable
Vulkan device, the runtime automatically selects the memory-budgeted resident
transformer forward: hidden state stays on the device across transformer blocks
and each forward performs one final hidden-state readback. CPU and Metal paths
remain available as fallbacks. The encoder, reference VAE, transformer, and
decode VAE are loaded in stages so large mappings can be released between stages.

A current RTX 3090 component-folder profile recorded 109.487 s for a two-step
edit, 9.8 s steady forward, and 18,603 MiB peak device memory. These are
hardware- and workload-specific profile figures.

## Source references

- [Qwen-Image-Edit-2509 pinned source](https://huggingface.co/Qwen/Qwen-Image-Edit-2509/tree/983d8d220ec4cf16278ef80bf3f30fe0378c8263)
- [Qwen Image Edit GGUF pinned source](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/tree/84a3006979126011422eeeefe0c9485ddf431ef5)
- [Cortiq Apache-2.0 license](../LICENSE)
- [CMF v2 specification](CMF_V2_SPEC.md)
