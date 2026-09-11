# Qwen Image Edit 2509

This guide covers the native Rust path for
[Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509).
It is an image editing pipeline: provide one or more reference images with
`--image` and a text instruction. The runtime uses three separate CMF files so
the text/vision encoder, denoiser, and VAE can be staged independently.

## Components

Keep this layout for the defaults used by `cortiq imagine`:

```text
qwen-image/
  transformer.cmf
  text_encoder.cmf
  vae.cmf
  scheduler_config.json       # optional; the built-in FlowMatch defaults apply otherwise
```

The transformer is the diffusion model imported from the pinned Q6_K GGUF.
`text_encoder.cmf` contains Qwen2.5-VL text and vision weights plus the
tokenizer and image processor metadata. `vae.cmf` contains the Qwen Image VAE.
The files are independent: `cortiq verify` can check each one on its own.

The source components and revisions used for the reproducible conversion are:

| component | source | revision | source size |
|---|---|---|---:|
| transformer | `QuantStack/Qwen-Image-Edit-2509-GGUF/Qwen-Image-Edit-2509-Q6_K.gguf` | `84a3006979126011422eeeefe0c9485ddf431ef5` | 16,824,990,240 bytes |
| text encoder | `Qwen/Qwen-Image-Edit-2509/text_encoder/model-00001-of-00004.safetensors` | `983d8d220ec4cf16278ef80bf3f30fe0378c8263` | 4,968,243,304 bytes |
| text encoder | `Qwen/Qwen-Image-Edit-2509/text_encoder/model-00002-of-00004.safetensors` | same | 4,991,495,816 bytes |
| text encoder | `Qwen/Qwen-Image-Edit-2509/text_encoder/model-00003-of-00004.safetensors` | same | 4,932,751,040 bytes |
| text encoder | `Qwen/Qwen-Image-Edit-2509/text_encoder/model-00004-of-00004.safetensors` | same | 1,691,924,384 bytes |
| VAE | `Qwen/Qwen-Image-Edit-2509/vae/diffusion_pytorch_model.safetensors` | same | 253,806,966 bytes |

The source GGUF SHA-256 is
`ec5694f11a2908c10ef5324c50c79b1bb433547a39a211996551417b4b16f0ce`.
The companion weight SHA-256 values are:

```text
text_encoder/model-00001-of-00004.safetensors  d725335e4ea2399be706469e4b8807716a8fa64bd03468252e9f7acf2415fee4
text_encoder/model-00002-of-00004.safetensors  b1830db6908dcc76df3a71492acbcf2b8cac130114cf1f3c2d9edae8de8c6de3
text_encoder/model-00003-of-00004.safetensors  09c1807c6d00d7cab94f7db39d4c02ebb8537225ccde383861ac48db97945aa6
text_encoder/model-00004-of-00004.safetensors  5dd068336d14d45ffb43cef374d286cc6ba9d8741b028f90a7d040d847961f4a
vae/diffusion_pytorch_model.safetensors        0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344
```

Verify these values before packing when the source is transferred outside
the Hub client.

## Acquire and pack

The following uses the `hf` command-line client and an explicit revision. It
downloads only the transformer source and the companion files required by the
native path:

This complete-tree recipe materializes about 33.7 GB of source weights before
packing. Use it on a disk with that headroom plus the CMF outputs. On a small
development disk, pack the transformer first and remove its verified source,
then use the pinned remote companion commands below; never delete a source
shard before its output is durable and verified.

```sh
set -eu
mkdir -p qwen-image/source

GGUF_REV=84a3006979126011422eeeefe0c9485ddf431ef5
QWEN_REV=983d8d220ec4cf16278ef80bf3f30fe0378c8263

hf download QuantStack/Qwen-Image-Edit-2509-GGUF \
  Qwen-Image-Edit-2509-Q6_K.gguf \
  --revision "$GGUF_REV" --local-dir qwen-image/source

hf download Qwen/Qwen-Image-Edit-2509 \
  text_encoder/config.json \
  text_encoder/model.safetensors.index.json \
  text_encoder/model-00001-of-00004.safetensors \
  text_encoder/model-00002-of-00004.safetensors \
  text_encoder/model-00003-of-00004.safetensors \
  text_encoder/model-00004-of-00004.safetensors \
  vae/config.json \
  vae/diffusion_pytorch_model.safetensors \
  scheduler/scheduler_config.json \
  processor/tokenizer.json \
  processor/preprocessor_config.json \
  processor/tokenizer_config.json \
  --revision "$QWEN_REV" --local-dir qwen-image/source
```

Check the large source files before creating output files. For the companion
weights, compare against the SHA-256 values in the acquisition receipt for
the selected revision:

```sh
shasum -a 256 qwen-image/source/Qwen-Image-Edit-2509-Q6_K.gguf
shasum -a 256 qwen-image/source/text_encoder/model-*.safetensors
shasum -a 256 qwen-image/source/vae/diffusion_pytorch_model.safetensors
```

Build a matching `cortiq` binary from this repository, then pack the three
components. The transformer uses the existing Q4TP codec. The text encoder
uses Q4TP for its rank-2 projection/embedding weights while retaining the
other source tensors and metadata. The VAE command uses `f16`; its source
floating tensors are retained by the component packer.

```sh
cargo build --release -p cortiq-cli

./target/release/cortiq import-gguf \
  qwen-image/source/Qwen-Image-Edit-2509-Q6_K.gguf \
  --quant q4tp --output qwen-image/transformer.cmf

./target/release/cortiq imagine-pack \
  --component qwen-text-encoder --quant q4tp \
  --out qwen-image/text_encoder.cmf qwen-image/source

./target/release/cortiq imagine-pack \
  --component qwen-vae --quant f16 \
  --out qwen-image/vae.cmf qwen-image/source

cp qwen-image/source/scheduler/scheduler_config.json qwen-image/

./target/release/cortiq verify qwen-image/transformer.cmf
./target/release/cortiq verify qwen-image/text_encoder.cmf
./target/release/cortiq verify qwen-image/vae.cmf
```

`imagine-pack` can also stream a component directly from the pinned Hub
resolve base, using `HF_TOKEN` when the source requires authentication:

```sh
QWEN_REV=983d8d220ec4cf16278ef80bf3f30fe0378c8263
QWEN_BASE="https://huggingface.co/Qwen/Qwen-Image-Edit-2509/resolve/$QWEN_REV"
./target/release/cortiq imagine-pack \
  --component qwen-text-encoder --quant q4tp \
  --out qwen-image/text_encoder.cmf "$QWEN_BASE"
./target/release/cortiq imagine-pack \
  --component qwen-vae --quant f16 \
  --out qwen-image/vae.cmf "$QWEN_BASE"
```

The remote form still needs the `processor/` and component files at that
revision; the local form makes the complete source tree and its hashes easier
to audit. It reads the companion source by bounded HTTP ranges, so it avoids
materializing the roughly 16.6 GB text-encoder tree locally. For a bounded
disk, the safe sequence is: import and verify the 16.8 GB GGUF, retain the
verified transformer CMF, remove that GGUF, then stream-pack the text encoder
and VAE from the same pinned resolve base. Do not mix files from different
revisions in one model directory.

## Edit an image

Pass the model directory and at least one reference image. The CLI defaults
to 512×512 output, 30 denoising steps, CFG 4, and seed 42. The example fixes
all generation settings so it can be reproduced:

```sh
./target/release/cortiq imagine qwen-image \
  --image docs/media/fox-512.png \
  --prompt "Add a vivid blue knitted scarf while preserving the fox, pose, and snowy background." \
  --height 512 --width 512 --steps 30 --cfg 4 --seed 7 \
  --reference-size 1024 --out fox-scarf.png
```

Repeat `--image` to supply multiple references in prompt order. If the
components are stored elsewhere, pass a transformer file as the model path
and override the companions explicitly:

```sh
./target/release/cortiq imagine qwen-image/transformer.cmf \
  --text-encoder /models/qwen/text_encoder.cmf \
  --vae /models/qwen/vae.cmf \
  --image docs/media/fox-512.png \
  --prompt "Turn the scene into a watercolor illustration." \
  --height 512 --width 512 --steps 30 --cfg 4 --seed 7 \
  --reference-size 1024 --out watercolor.png
```

The official profile uses `--reference-size 1024`. This value is the square
root of the VAE reference image area; it is independent of the output
height/width. The Qwen2.5-VL image-conditioning stage has its own 384²-area
resize before smart patch resizing, so changing the VAE reference area does
not change that encoder rule. Keep output dimensions on multiples of 16.

The default negative prompt is a single space, which keeps true classifier
free guidance enabled. Set `--negative-prompt` to provide a deliberate
negative prompt. A CFG value at or below 1 disables the unconditional branch
and reduces work, but it is a different guidance profile.

The optional scheduler file is auto-discovered as
`qwen-image/scheduler_config.json`. Use `--scheduler PATH` when it lives
elsewhere. The file must describe the Qwen Image exponential FlowMatch Euler
contract from the pinned source.

## Backends and resource use

The same command selects the available native backend. Use `CMF_GPU=0` for a
portable CPU run. On a headless Vulkan host, set the loader environment before
running:

```sh
CMF_GPU=0 ./target/release/cortiq imagine qwen-image \
  --image docs/media/fox-512.png --prompt "Add a blue scarf." \
  --height 512 --width 512 --steps 30 --cfg 4 --seed 7 \
  --reference-size 1024 --out fox-scarf-cpu.png

XDG_RUNTIME_DIR=/tmp WGPU_BACKEND=vulkan CMF_GPU=1 \
  ./target/release/cortiq imagine qwen-image \
  --image docs/media/fox-512.png --prompt "Add a blue scarf." \
  --height 512 --width 512 --steps 30 --cfg 4 --seed 7 \
  --reference-size 1024 --out fox-scarf-vulkan.png
```

The pipeline reports text encoder, reference VAE, denoiser, and decode VAE
stages on stderr. It releases each large component stage before opening the
next one and keeps only conditioning data and latents between stages. A
missing companion, malformed metadata, or invalid tensor shape fails before
image generation starts.

This guide documents the canonical 1024² reference profile. The CLI accepts
other `--reference-size` values for controlled experiments, but this source
tree does not advertise a lower-quality/speed profile without a measured
quality and memory result for that exact setting.

## What is verified here

The component format preserves the official tensor names and embeds the
configuration required by the native loader. The encoder path follows the
Qwen2.5-VL processor/template, two-stage image preprocessing, vision windows,
MRoPE, and the 64-token prefix drop. The transformer follows the Qwen Image
double-stream denoiser and FlowMatch schedule; the VAE uses the official
Qwen Image scaling metadata. Focused seeded oracles cover these contracts.

Full-device image quality and timing are workload-specific. Record the exact
binary, component SHA-256 values, backend, output dimensions, steps, CFG,
seed, reference size, elapsed stages, and output SHA when publishing a
comparison. The repository guide intentionally makes no universal quality or
performance claim.

## Source references

- [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509/tree/983d8d220ec4cf16278ef80bf3f30fe0378c8263)
- [Qwen Image Edit GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/tree/84a3006979126011422eeeefe0c9485ddf431ef5)
- [CMF v2 specification](CMF_V2_SPEC.md)
