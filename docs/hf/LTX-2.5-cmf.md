---
library_name: cortiq
license: other
license_name: ltx-2-community-license-agreement
license_link: https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md
base_model:
- Lightricks/LTX-2.5
base_model_relation: quantized
pipeline_tag: text-to-video
tags:
- cmf
- cortiq
- video
- text-to-video
- ltx-video
- ltx-2.5
- rust
- 4-bit
---

# LTX-2.5 — the whole pipeline in one 22 GB file, rendered by Rust

<p align="center">
  <img src="assets/corgi.gif" width="49%" alt="A corgi in a chef hat flips a pancake in a sunlit kitchen">
  <img src="assets/neon.gif" width="49%" alt="Neon rain on a Tokyo side street at night">
</p>
<p align="center">
  <img src="assets/whale.gif" width="49%" alt="A humpback whale glides through a shaft of sunlight">
  <img src="assets/glass.gif" width="49%" alt="Molten glass blown into a bulb over an orange furnace">
</p>

**Every frame above was produced by `cortiq`** — a single Rust binary with no
PyTorch, no diffusers, no CUDA toolkit and no Python anywhere in the process —
reading one memory-mapped [CMF](https://github.com/infosave2007/cmf) file.

[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) renders video **and its
soundtrack** from one prompt: a 21 B audio-video diffusion transformer that
denoises picture and sound in the same 48 blocks, a Gemma-4 12 B prompt
encoder, a 3-D video VAE, an audio VAE, two latent upscalers and a duration
head. The reference checkout is **71.35 GB across six safetensors** plus a
PyTorch stack.

Here it is **one file of 22.07 GB** — every component, the Gemma-4 tokenizer
and every config inside it.

| | reference | this file |
|---|---|---|
| files | 6 safetensors + configs + tokenizer | **1** |
| bytes | 71.35 GB | **22.07 GB** (3.2× smaller) |
| weights | 35.65 B | 35.65 B — all of them |
| loader | diffusers / ComfyUI + PyTorch | `mmap` |
| renderer | Python | **one Rust binary** |

## Quick start

```bash
# 1 — the runtime (Rust 1.85+; nothing else)
cargo install cortiq-cli

# 2 — the model
hf download infosave/LTX-2.5-cmf ltx25-q4tp.cmf --local-dir .
cortiq verify ltx25-q4tp.cmf        # every tensor is hashed in the directory

# 3 — a video
cortiq ltx-video --model ltx25-q4tp.cmf \
  --prompt "A corgi in a chef hat flips a pancake in a sunlit kitchen. \
Warm morning light, static camera." \
  --height 256 --width 384 --frames 49 --fps 24 --seed 42 \
  --out corgi.y4m

ffmpeg -i corgi.y4m -pix_fmt yuv420p corgi.mp4
```

That is the whole thing: prompt in, frames out, one process, one file. The
GPU is found at run time — Vulkan on Linux and Windows, Metal on Apple
silicon — and everything falls back to the CPU when there is none.

> **Keep the file on local storage.** It is memory-mapped, so every weight is
> a page fault. On a network filesystem (NFS, MooseFS, a rented pod's
> `/workspace` volume) that is a network round trip per weight and the
> process will sit at 1 % CPU looking hung. Copy it to a local disk — or to
> `/dev/shm` if you have the RAM.

### The examples above, exactly

```bash
M=ltx25-q4tp.cmf
cortiq ltx-video --model $M --seed 42 --height 256 --width 384 --frames 49 \
  --out-dir corgi/ --prompt \
  "A corgi in a chef hat flips a pancake in a sunlit kitchen. Warm morning light, static camera."

cortiq ltx-video --model $M --seed 7 --height 256 --width 384 --frames 49 \
  --out-dir neon/ --prompt \
  "Neon rain on a Tokyo side street at night, a lone figure with a translucent umbrella \
walks past ramen shop signs, reflections rippling in the puddles, slow dolly."

cortiq ltx-video --model $M --seed 11 --height 256 --width 384 --frames 49 \
  --out-dir whale/ --prompt \
  "A humpback whale glides through a shaft of sunlight in deep blue water, plankton \
drifting like dust, the camera rises with it toward the surface."

cortiq ltx-video --model $M --seed 23 --height 256 --width 384 --frames 49 \
  --out-dir glass/ --prompt \
  "Molten glass is blown into a bulb over an orange furnace, the glowing gather \
stretching and rotating, sparks drifting in the dark workshop."

# frames → mp4 → gif
ffmpeg -framerate 24 -i corgi/frame_%04d.ppm -pix_fmt yuv420p -crf 18 corgi.mp4
ffmpeg -i corgi.mp4 -vf "fps=12,scale=384:-1:flags=lanczos,split[s0][s1];\
[s0]palettegen[p];[s1][p]paletteuse" corgi.gif
```

`--out-dir` writes `frame_0000.ppm …`; `--out file.y4m` writes one
[YUV4MPEG2](https://wiki.multimedia.cx/index.php/YUV4MPEG2) stream instead,
which every tool reads — so the renderer needs no video encoder of its own.

### Higher resolution

```bash
cortiq ltx-video --model $M --two-stage \
  --height 512 --width 768 --frames 49 --seed 42 \
  --prompt "…" --out hq.y4m
```

`--two-stage` samples the way the distilled model was trained: eight
ancestral Euler steps at half resolution, the learned latent upscaler ×2,
then three deterministic steps that refine what the upscale invented.

Resolution must be a multiple of 32 (the video VAE's spatial stride) and the
frame count `8k + 1` (its temporal stride plus the standalone first frame).

### Measured

RTX 5090, `/dev/shm`, 49 frames at 24 fps:

| stage | 384×256 | 768×512 (`--two-stage`) |
|---|---|---|
| prompt encode (Gemma-4 12 B + connectors) | 28 s | 28 s |
| denoise | 8 × 30 s | 8 × 30 s + 3 × 120 s |
| latent upscale | — | 25 s |
| video VAE | 50 s | 200 s |

Nothing here is tuned yet — the transformer runs one 48-block forward per
step against `mmap`ped 4-bit weights, and the VAE is a straight
im2col + GEMM.

## The stages, separately

Each stage is its own command. That is how the port was gated: every one of
them can be run against a dump of the reference implementation's own
activations and will report the first place it diverges.

```bash
# prompt → the two context tensors the transformer cross-attends to
cortiq ltx-encode --model $M --prompt "…" --out context.safetensors

# context → latent → frames
cortiq ltx-render --model $M --context context.safetensors \
  --height 256 --width 384 --frames 49 --out-latent latent.safetensors --out out.y4m

# latent → frames, the 3-D convolutional decoder alone
cortiq ltx-decode --model $M --latent latent.safetensors --out-dir frames/
```

## What is inside

| component | weights | in the file | codec |
|---|---|---|---|
| `dit.*` — LTX-2.5 22B audio-video DiT (distilled) | 21.004 B | 10.84 GiB | q4tp + exact adaLN |
| `te.*` — Gemma-4 12B prompt encoder, aggregates, vision tower | 13.116 B | 6.87 GiB | q4tp + q8 embeddings |
| `vvae.*` — video VAE (3-D conv, encoder + decoder) | 0.726 B | 1.35 GiB | f16 |
| `avae.*` — audio VAE | 0.182 B | 0.34 GiB | f16 |
| `ups.*` — latent spatial upscaler ×2 | 0.498 B | 0.93 GiB | f16 |
| `upt.*` — latent temporal upscaler ×2 | 0.131 B | 0.24 GiB | f16 |
| `dhead.*` — duration head | 1.9 M | 3.8 MB | f16 |
| configs, HF assets | — | 55 KB | raw |
| tokenizer (`tokenizer_json`, 32 MB) | — | VOCAB section | raw |

### What the codec does per tensor, and why

Four bits is not applied by fiat. `cortiq ltx-pack` decides per tensor, and
two of those decisions were made by measuring against the reference rather
than by taste:

* **2-D planes of at least 2²⁰ weights → q4tp**, 4.16 bits with a per-row
  scale ladder. Every projection in the transformer and in the encoder.
* **The adaLN-single stacks stay exact.** Their output is not a residual —
  it is the scale and the shift applied to every token in every block, so an
  error there is multiplied into the whole stream instead of being averaged
  away by anything downstream. Quantized, they put 3.6·10⁻² of relative
  error into the very first normalization of block 0; exact, 5.9·10⁻³.
  0.56 GB.
* **The token table stays 8-bit.** It *is* the residual stream at layer zero
  and it carries through forty-eight residual additions. q4tp put 11 % into
  every hidden state the prompt encoder produced; q8 puts 0.5 % there, for
  0.5 GB.
* **The adaLN tables, the connector's learnable registers and the VAE's
  `per_channel_statistics` stay exact** — 19 MB in total, read once a step,
  modulating everything.
* **Convolutions stay f16.** Both VAEs and both upscalers are convolutional,
  and the decoder is what the eye actually sees.

## The architecture it carries

`AVTransformer3DModel`, from the release's own config (kept verbatim in the
file as `ltx.config_json`):

* **48 blocks**, video stream 4096 (32 heads × 128), **audio stream 2048**
  (32 × 64), joint audio↔video cross-attention with adaLN-gated fusion that
  reads the *pre-fusion* state of both streams, so the order the two
  directions run in cannot bias the result.
* Per block: self-attention, cross-attention to the prompt with its own
  adaLN pair on the query *and* on the prompt's keys and values, **RMS
  q/k-norm across the whole inner dimension**, gated attention
  (`2·sigmoid` per head), and a gelu-approximate feed-forward — all
  modulated from per-block `[9, 4096]` / `[9, 2048]` tables.
* **Split 3-D RoPE** over (seconds, pixel row, pixel column) evaluated at the
  *middle* of each patch's bounds, θ = 10000, with the causal correction that
  gives the first latent frame one pixel frame where every later one gets
  eight. The audio stream shares the time axis in seconds, which is what lets
  the two cross-attend positionally.
* **The prompt encoder** is Gemma-4 12 B — forty sliding-window layers at head
  256 and eight full-attention layers at head 512 whose value projection *is*
  the key projection — and the features are not its last hidden state: all
  **forty-nine** layer outputs are RMS-normalized per token per layer,
  concatenated to 188160 numbers and projected once to 4096 (video) and once
  to 2048 (audio).
* **Embeddings connectors** — 8 gated-attention blocks each for video and
  audio, with **128 learnable registers** that replace every padded position,
  which is why the transformer needs no prompt mask at all.

## Packing it yourself

Three passes, each one able to delete its source before the next lands — the
stand this was packed on had a 50 GB disk quota and the sources are 71 GB:

```bash
cortiq ltx-pack --out p1.cmf --dit ltx-2.5-22b-distilled-transformer-bf16.safetensors
cortiq ltx-pack --out p2.cmf --in p1.cmf --te gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
cortiq ltx-pack --out ltx25-q4tp.cmf --in p2.cmf \
  --video-vae ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upscaler  ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --temporal-upscaler ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors \
  --duration-head     ltx-2.5-duration-head-bf16.safetensors
cortiq verify ltx25-q4tp.cmf && cortiq info ltx25-q4tp.cmf
```

Measured on a 32-core pod: **five minutes** for 71 GB of bf16, single
machine, no Python, no GPU. `--quant` picks the codec for the big planes
(`q4tp`, `q8`, `f16`, `f32`), `--vae-quant` the one for convolutions.

## Status

* ✅ **Text → video *and sound* runs end to end on the Rust engine**: the
  Gemma-4 prompt encoder, the aggregate projections, the connectors, the
  48-block audio-video transformer, the sampler, the latent upscaler, the
  video VAE, the audio VAE with its BigVGAN vocoder and bandwidth extension,
  and the duration head.
* ⏳ **Image and video conditioning, LoRAs and the IC-LoRA upscaler** are next.
  The weights for conditioning are in this file (the video VAE's encoder half);
  the LoRAs and the IC-LoRA upscaler are separate releases.

Everything above is honest about what it is: a 4-bit repack. The reference at
bf16 is the quality ceiling, and the codec's cost was measured stage by stage
rather than assumed — see the numbers in the codec section.

## Provenance

The weights are Lightricks' LTX-2.5 release and remain under the
[LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md)
— the licence text ships inside the source checkpoints and applies to this
repack unchanged. The CMF container format and the `cortiq` runtime are
Apache-2.0 ([repository](https://github.com/infosave2007/cmf), `PATENTS.md`).

No weight was altered: the pack is a codec change and a container change.
Every tensor's bytes are hashed in the directory, so `cortiq verify` proves
the file is the one that was written.
