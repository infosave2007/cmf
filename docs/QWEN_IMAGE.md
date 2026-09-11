# Qwen Image Edit 2509

This guide covers the native Rust path for
[Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509).
Provide one or more reference images with `--image` and a text instruction.
The ready release is a self-contained CMF; the original three-file layout
remains available when users need to stage components independently.

## Ready layouts

The default release file is `qwen-image-edit-2509-q4tp.cmf`. It contains the
Qwen Image transformer, Qwen2.5-VL text/vision encoder, tokenizer, processor,
component configurations, VAE, scheduler configuration, and bundle manifest.
It is memory-mapped directly and needs no sibling CMF files at inference time.

The accepted bundle has 2,864 tensors, is 15,374,173,939 bytes, and has SHA-256
`6e0795e3f63a5e38083ca8b03c2abaee16621763b1cbf01456081bbbfe931c24`.
The merge preserved all 2,862 component tensor shapes, dtypes, and payload
bytes. Two configuration entries have distinct names in the bundle; the only
extra payloads are the embedded scheduler and manifest.

For component staging, retain this directory:

```text
qwen-image/
  transformer.cmf
  text_encoder.cmf
  vae.cmf
  scheduler_config.json
```

The verified standalone files are:

| file | profile | size | SHA-256 |
|---|---|---:|---|
| `transformer.cmf` | Q4TP Qwen Image transformer | 10,719,610,880 bytes | `7c477f464e6eda50d6c73eb9c8c4cd275c8852ba9eaf1c47252a97b612350aaa` |
| `text_encoder.cmf` | Q4TP + Q8_2f Qwen2.5-VL encoder | 4,403,056,640 bytes | `15745717be7fee53499a425a240cc25b8b122848a1c1e1c4a3fc50ea06b64532` |
| `vae.cmf` | source-float Qwen Image VAE | 257,213,312 bytes | `af10c1a55bf8e47b8eaea79dfdfb0cd6dadb9ac2a65cbdf1c383a5b1b8b5ee87` |
| `scheduler_config.json` | FlowMatch Euler metadata | 485 bytes | `7ee767e37bae4af31d4eb935e125bb20a2237eeecafd22af6610093865c6f587` |

A bundle defaults the transformer, encoder, and VAE paths to itself. The
`--text-encoder`, `--vae`, and `--scheduler` options remain available for
an explicit component or scheduler override. The standalone payloads and the
bundle therefore use the same tensor bytes and loader configuration.

## Acquire the pinned source

Use the following immutable revisions:

| source | revision | source bytes |
|---|---|---:|
| `QuantStack/Qwen-Image-Edit-2509-GGUF/Qwen-Image-Edit-2509-Q6_K.gguf` | `84a3006979126011422eeeefe0c9485ddf431ef5` | 16,824,990,240 |
| `Qwen/Qwen-Image-Edit-2509` text encoder shards | `983d8d220ec4cf16278ef80bf3f30fe0378c8263` | 16,584,414,544 |
| `Qwen/Qwen-Image-Edit-2509` VAE safetensors | same | 253,806,966 |

The pinned GGUF SHA-256 is
`ec5694f11a2908c10ef5324c50c79b1bb433547a39a211996551417b4b16f0ce`.
The companion receipt records these source hashes:

```text
text_encoder/model-00001-of-00004.safetensors  d725335e4ea2399be706469e4b8807716a8fa64bd03468252e9f7acf2415fee4
text_encoder/model-00002-of-00004.safetensors  b1830db6908dcc76df3a71492acbcf2b8cac130114cf1f3c2d9edae8de8c6de3
text_encoder/model-00003-of-00004.safetensors  09c1807c6d00d7cab94f7db39d4c02ebb8537225ccde383861ac48db97945aa6
text_encoder/model-00004-of-00004.safetensors  5dd068336d14d45ffb43cef374d286cc6ba9d8741b028f90a7d040d847961f4a
vae/diffusion_pytorch_model.safetensors        0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344
```

For a complete local source tree, download only the pinned files needed by the
native path:

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

Hash the transferred source before packing:

```sh
shasum -a 256 qwen-image/source/Qwen-Image-Edit-2509-Q6_K.gguf
shasum -a 256 qwen-image/source/text_encoder/model-*.safetensors
shasum -a 256 qwen-image/source/vae/diffusion_pytorch_model.safetensors
```

The complete source tree is about 33.7 GB before CMF outputs. On a smaller
disk, import and verify the GGUF first, remove that source only after its CMF
and receipt are durable, then stream-pack the pinned companion files. Never
mix source revisions.

## Pack standalone components and merge the bundle

Build the matching CLI, then create the standalone components:

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
```

For the text encoder, `--quant q4tp` applies to every rank-2 `.weight`
tensor, including embeddings and the retained `lm_head`. Aligned matrices use
Q4TP; shapes that cannot satisfy its tile group use Q8_2f. The verified profile
contains 328 Q4TP and 32 Q8_2f projection payloads. Norms, biases, patch
convolutions, and other non-rank-2 tensors keep their source F32/F16/BF16
storage. The VAE `--quant f16` path keeps its source floating-point tensors.

Verify the standalone files and merge them with the native streaming command:

```sh
./target/release/cortiq verify qwen-image/transformer.cmf
./target/release/cortiq verify qwen-image/text_encoder.cmf
./target/release/cortiq verify qwen-image/vae.cmf

./target/release/cortiq imagine-pack \
  --bundle qwen-image \
  --out qwen-image/qwen-image-edit-2509-q4tp.cmf

./target/release/cortiq verify qwen-image/qwen-image-edit-2509-q4tp.cmf
```

`imagine-pack --bundle ROOT --out FILE.cmf` validates all three CMFs and the
scheduler JSON, then opens one component at a time and streams each encoded
payload through the existing CMF writer. It aliases the colliding component
configs as `image.text_encoder.config_json` and
`image.vae.config_json`, embeds `image.scheduler_config_json` and
`image.bundle_config_json`, and preserves tokenizer/processor assets. It
does not requantize or materialize all component weights.

The component packer can also read directly from the pinned Hub resolve base.
This uses bounded HTTP ranges and keeps any required Hub credential local:

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

## Run an edit

The one-file command is the default and needs only the bundle plus a reference
image:

```sh
./target/release/cortiq imagine qwen-image/qwen-image-edit-2509-q4tp.cmf \
  --image docs/media/fox-512.png \
  --prompt "Add a vivid blue knitted scarf while preserving the fox, pose, and snowy background." \
  --height 512 --width 512 --steps 30 --cfg 4 --seed 7 \
  --reference-size 1024 --out fox-scarf.png
```

The standalone layout supports directory discovery:

```sh
./target/release/cortiq imagine qwen-image \
  --image docs/media/fox-512.png \
  --prompt "Add a vivid blue knitted scarf while preserving the fox, pose, and snowy background." \
  --height 512 --width 512 --steps 30 --cfg 4 --seed 7 \
  --reference-size 1024 --out fox-scarf.png
```

When components live elsewhere, pass the explicit overrides:

```sh
./target/release/cortiq imagine qwen-image/transformer.cmf \
  --text-encoder /models/qwen/text_encoder.cmf \
  --vae /models/qwen/vae.cmf \
  --scheduler /models/qwen/scheduler_config.json \
  --image docs/media/fox-512.png \
  --prompt "Turn the scene into a watercolor illustration." \
  --height 512 --width 512 --steps 30 --cfg 4 --seed 7 \
  --reference-size 1024 --out watercolor.png
```

Repeat `--image` in reference order for multiple images. The required edit
prompt is passed through the official Qwen2.5-VL processor/template. The
default negative prompt is one space, which keeps true CFG enabled; use
`--negative-prompt` for an explicit negative prompt. CFG at or below 1 omits
the unconditional branch and is a different guidance profile.

The canonical `--reference-size 1024` value is the square root of the VAE
reference area (1024²), independent of output height and width. Qwen2.5-VL
conditioning first applies the official 384²-area image resize and then its
smart patch resize, so changing the VAE reference area does not alter that
encoder preprocessing rule. Keep output dimensions on multiples of 16. PNG,
JPEG, and PPM output are supported.

## Backends, lifetimes, and measured operator paths

Use `CMF_GPU=0` for the portable CPU path. On a headless Vulkan host, set
the loader explicitly:

```sh
CMF_GPU=0 ./target/release/cortiq imagine qwen-image/qwen-image-edit-2509-q4tp.cmf \
  --image docs/media/fox-512.png --prompt "Add a blue scarf." \
  --height 512 --width 512 --steps 30 --cfg 4 --seed 7 \
  --reference-size 1024 --out fox-scarf-cpu.png

XDG_RUNTIME_DIR=/tmp WGPU_BACKEND=vulkan CMF_GPU=1 \
  ./target/release/cortiq imagine qwen-image/qwen-image-edit-2509-q4tp.cmf \
  --image docs/media/fox-512.png --prompt "Add a blue scarf." \
  --height 512 --width 512 --steps 30 --cfg 4 --seed 7 \
  --reference-size 1024 --out fox-scarf-vulkan.png
```

The pipeline opens the encoder, reference VAE, transformer, and decode VAE in
stages. It retains conditioning and latents across stage boundaries and
releases each large component before opening the next one. A missing companion,
malformed metadata, or invalid tensor shape fails before generation.

Accepted component measurements on the RTX 3090 Vulkan gate are useful for
operator attribution, not whole-pipeline promises: the optimized single-frame
VAE encoded a 1024² reference in 23.098 s versus 128.411 s in the frozen
baseline (5.56x), and decoded a 512² result in 8.582 s versus 54.775 s
(6.38x). The Q4TP GELU FFN fixture at 5,120 × 3,072 × 12,288 measured
205.570 ms with cooperative f16 versus 2,105.088 ms scalar, with relative RMS
error 1.94e-7. The fused QKV path remains opt-in; its measured operator result
was 514.364 ms versus 522.502 ms scalar with relative RMS 3.90e-4.

These are component/operator gates on the stated hardware and shapes. Record
the exact binary, CMF hashes, backend, output dimensions, steps, CFG, seed,
reference size, stage timings, and output SHA for any end-to-end comparison.
This guide makes no universal image-quality or whole-pipeline timing claim.

## Source references

- [Qwen-Image-Edit-2509 pinned source](https://huggingface.co/Qwen/Qwen-Image-Edit-2509/tree/983d8d220ec4cf16278ef80bf3f30fe0378c8263)
- [Qwen Image Edit GGUF pinned source](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/tree/84a3006979126011422eeeefe0c9485ddf431ef5)
- [CMF v2 specification](CMF_V2_SPEC.md)
