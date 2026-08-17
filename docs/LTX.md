# LTX-2.5 on cortiq

[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) renders video **and its
soundtrack** from one prompt. This document is how to run it on `cortiq` —
a Rust binary with no Python, no PyTorch and no CUDA toolkit underneath —
from a single memory-mapped CMF container.

## Install

```bash
# from crates.io
cargo install cortiq-cli

# or from the repository
git clone https://github.com/infosave2007/cmf && cd cmf
cargo install --path crates/cortiq-cli
```

`cortiq --version` should now answer. The GPU paths are optional and
detected at run time: Vulkan on Linux/Windows (NVIDIA, AMD, Intel) and Metal
on Apple Silicon. Without either, everything still runs on the CPU.

## Get the model

```bash
huggingface-cli download infosave/LTX-2.5-cmf ltx25-q4tp.cmf --local-dir .
cortiq verify ltx25-q4tp.cmf      # every tensor hashed in the directory
cortiq info   ltx25-q4tp.cmf
```

One file, ~22 GB, and it holds the whole pipeline: the 21 B audio-video
diffusion transformer, the Gemma-4 12 B prompt encoder with its tokenizer,
both VAEs, both latent upscalers and the duration head.

> **Read the model from local storage.** The container is memory-mapped, so
> every weight is a page fault. On a network filesystem (NFS, MooseFS, a
> RunPod `/workspace` volume) that is a network round trip per weight and the
> process will look hung at 1% CPU. Copy it to a local disk — or to
> `/dev/shm` if you have the RAM — before running.

## Render a video

```bash
cortiq ltx-video \
  --model ltx25-q4tp.cmf \
  --prompt "A corgi in a chef hat flips a pancake in a sunlit kitchen. \
Warm morning light, static camera." \
  --height 256 --width 384 --frames 49 --fps 24 --seed 42 \
  --out corgi.y4m

ffmpeg -i corgi.y4m -pix_fmt yuv420p corgi.mp4
```

The renderer writes [YUV4MPEG2](https://wiki.multimedia.cx/index.php/YUV4MPEG2),
a raw stream every tool understands, so `cortiq` needs no video encoder of
its own. To get frames instead, pass `--out-dir frames/` and it writes
`frame_0000.ppm` and so on.

Resolution must be a multiple of 32 (the video VAE's spatial stride) and the
frame count `8k + 1` (its temporal stride, plus the standalone first frame).

## The stages, one at a time

Each stage is also a command of its own, which is how the port was gated
against the reference implementation.

### Encode a prompt

```bash
cortiq ltx-encode --model ltx25-q4tp.cmf \
  --prompt "A corgi in a chef hat flips a pancake" \
  --out context.safetensors
```

Runs Gemma-4 12 B over the 1024-token window, takes the per-token RMS of
**all forty-nine** hidden states, projects the concatenation once to 4096
(video) and once to 2048 (audio), and pushes both through the eight-block
connectors. Writes `enc.video` and `enc.audio` — the two tensors the
transformer cross-attends to.

### Denoise from a prepared context

```bash
cortiq ltx-render --model ltx25-q4tp.cmf \
  --context context.safetensors \
  --height 256 --width 384 --frames 49 \
  --out-y4m out.y4m --out-latent latent.safetensors
```

Eight ancestral Euler steps on the distilled schedule, then the video VAE.
`--skip-decode` stops after the latent.

### Decode a latent

```bash
cortiq ltx-decode --model ltx25-q4tp.cmf \
  --latent latent.safetensors --out-dir frames/
```

The 3-D convolutional decoder on its own: causal-free convolutions, PixelNorm,
depth-to-space upsamples and the `per_channel_statistics` un-normalization.

## Pack the container yourself

```bash
# 1 — the 22B transformer (42.0 GB → 11.0 GB)
cortiq ltx-pack --out p1.cmf \
  --dit ltx-2.5-22b-distilled-transformer-bf16.safetensors

# 2 — the Gemma-4 12B encoder on top (--in carries pass 1 byte for byte)
cortiq ltx-pack --out p2.cmf --in p1.cmf \
  --te gemma4-12b-with-proj-ltx-2.5-bf16.safetensors

# 3 — the VAEs, the upscalers and the duration head
cortiq ltx-pack --out ltx25-q4tp.cmf --in p2.cmf \
  --video-vae ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upscaler  ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --temporal-upscaler ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors \
  --duration-head     ltx-2.5-duration-head-bf16.safetensors

cortiq verify ltx25-q4tp.cmf
```

Multi-pass because the sources are 71 GB and the stand this was built on had
a 50 GB disk: each pass carries the previous file through byte for byte, so
you can delete each source before the next lands. `--quant` picks the codec
for the big planes (`q4tp`, `q8`, `f16`, `f32`); `--vae-quant` does the same
for convolutions.

### What the codec does per tensor

Four bits is not applied by fiat. `ltx-pack` decides per tensor, and two of
those decisions were made by measurement against the reference:

* **2-D planes of at least 2²⁰ weights → q4tp**, 4.16 bits with a per-row
  scale ladder. That is every projection in the transformer and in the
  encoder.
* **The adaLN-single stacks stay exact.** Their output is not a residual —
  it is the scale and shift applied to every token in every block, so a codec
  error there is multiplied into the whole stream rather than averaged away.
  Quantized, they put 3.6·10⁻² of relative error into the very first
  normalization of block 0; exact, 5.9·10⁻³. 0.56 GB.
* **The token table stays 8-bit.** It *is* the residual stream at layer zero
  and it carries through forty-eight residual additions. q4tp put 11% into
  every hidden state; q8 puts 0.5% there for 0.5 GB.
* **The adaLN tables, the connector registers and `per_channel_statistics`
  stay exact** — 19 MB in total, read once a step, modulating everything.
* **Convolutions stay f16.** Both VAEs are convolutional and the decoder is
  what the eye sees.

## Numerical gates

Every stage can be run against a dump of the reference implementation's
activations and reports the first place it diverges:

```bash
cortiq ltx-decode --model m.cmf --latent vae_gate.safetensors --gate
cortiq ltx-dit    --model m.cmf --oracle dit_oracle.safetensors --gate
cortiq ltx-encode --model m.cmf --prompt "…" --oracle te_oracle.safetensors
```

The oracle files are safetensors of the reference's own forward-hook output;
`docs/ltx-oracles.md` has the hook scripts that produce them.
