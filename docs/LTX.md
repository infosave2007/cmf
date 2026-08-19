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
  --out corgi.y4m --out-audio corgi.wav

# picture and soundtrack into one file
ffmpeg -i corgi.y4m -i corgi.wav -pix_fmt yuv420p -c:v libx264 -crf 18 \
  -c:a aac -shortest corgi.mp4
```

The transformer denoises picture and sound in the same 48 blocks, so the
soundtrack costs nothing extra to generate — only the eight seconds the audio
VAE and its vocoder take to turn the latent into 48 kHz stereo.

The renderer writes [YUV4MPEG2](https://wiki.multimedia.cx/index.php/YUV4MPEG2),
a raw stream every tool understands, so `cortiq` needs no video encoder of
its own. To get frames instead, pass `--out-dir frames/` and it writes
`frame_0000.ppm` and so on.

Resolution must be a multiple of 32 (the video VAE's spatial stride) and the
frame count `8k + 1` (its temporal stride, plus the standalone first frame).

### Higher resolution

```bash
cortiq ltx-video --model ltx25-q4tp.cmf --two-stage \
  --height 512 --width 768 --frames 49 --seed 42 \
  --prompt "…" --out hq.y4m
```

`--two-stage` samples the way the distilled model was trained: eight
ancestral Euler steps at half the requested resolution, the learned ×2
latent upscaler, then three deterministic steps that refine what the upscale
invented. It costs roughly four times a single-stage render.

`--steps N` / `--steps2 N` resample that schedule. The shipped ladder is
distilled — 8 is it exactly, other counts land on sigmas the model never saw
and usually soften the frame; detail comes from resolution instead.

### Measured

RTX 5090, container in `/dev/shm`, 49 frames at 24 fps:

| stage | RTX 5090, 384×256 | M4 MacBook, 24 GB, 384×256 |
|---|---|---|
| prompt encode (Gemma-4 12 B + connectors) | 26 s | 32 s |
| denoise | 8 × 19 s | 8 × 16 s |
| audio VAE + vocoder | 8 s | 16 s |
| video VAE | 50 s | 27 s |
| **total** | **3 min** | **4 min** |

## Every mode the model has

LTX-2.5 is one network with two streams; a *mode* is which parts you hold
fixed. Conditioning is encoded into the model's own latent space and frozen
there — the sampler is handed a timestep of zero for those tokens and leaves
them alone — so all of this is the same command with different inputs.

| mode | how |
|---|---|
| text → video + sound | `ltx-video --prompt "…" --out-audio track.wav` |
| text → video | the same, without `--out-audio` |
| text → sound | the same, keeping only the wav |
| image + text → video (+ sound) | `--image still.ppm` |
| video → video | `--video frames/` |
| video → sound | `--video frames/ --video-to-audio --out-audio track.wav` |
| sound → video | `--audio-in track.wav` |
| sound → sound | `--audio-in track.wav --out-audio out.wav` |
| image + sound → video | `--image still.ppm --audio-in track.wav` |

Each of these has been run end to end and the output looked at (or listened
to), not just compiled. `--video-to-audio` follows the prompt the way it
should: the same frozen clip with "a dog panting softly, a quiet kitchen"
gives an even track at 0.007 RMS, and with "loud sizzling and clattering
pans, a dog barking sharply twice" gives 0.035 with transients at 0.046,
0.056 and 0.040 — two barks and the pans.

One real limitation rather than a caveat: `--audio-in` freezes the *whole*
soundtrack, so sound → sound returns what you gave it. A partial freeze —
condition on the first seconds and continue — is what the machinery already
supports; the CLI does not expose it yet.

```bash
# a still into a shot, with its soundtrack
ffmpeg -i photo.jpg -vf scale=384:256 -pix_fmt rgb24 still.ppm
cortiq ltx-video --model ltx25-q4tp.cmf --image still.ppm \
  --prompt "the camera pushes in slowly as the light shifts" \
  --height 256 --width 384 --frames 49 --out-dir out/ --out-audio out.wav

# a clip's own soundtrack, written for the picture that is already there
cortiq ltx-video --model ltx25-q4tp.cmf --video frames/ --video-to-audio \
  --prompt "footsteps on gravel, distant traffic" \
  --height 256 --width 384 --frames 49 --out-audio track.wav
```

`--video-strength` is the video-to-video dial and it only applies when the
clip covers the whole render: 1.0 keeps little more than the composition,
0.2 barely touches it. The schedule starts at exactly that noise level, so a
lower strength is also a shorter render. A clip *shorter* than the render is
not re-noised at all — it is frozen and the rest is generated after it, which
is continuation rather than transformation.

A stream that is frozen everywhere is handed a sigma of zero, not the
schedule's. The other stream's fusion gate reads that sigma and closes on
noise, so leaving the schedule's value there makes the transformer discount a
picture it was given intact — measured on `--video-to-audio`, the soundtrack
came out three times quieter.

The conditioning still must match the render's resolution, and a PPM is what
the CLI reads — `ffmpeg -i anything.jpg -pix_fmt rgb24 still.ppm` is the
whole conversion. Frames for `--video` are `frame_0000.ppm …`, exactly what
`--out-dir` writes.

Image conditioning goes through the video VAE's **encoder**, audio
conditioning through the audio VAE's, and both are in the same container as
everything else.

### LoRA adapters, and multi-subject references

```sh
cortiq ltx-video --model $M --lora adapter.safetensors --lora-strength 0.8 \
  --prompt "…" --out clip.y4m
```

The container's weights are q4tp, so an adapter cannot be folded into them
without dequantizing the whole DiT. The branch is evaluated beside them
instead — `y = x·Wᵀ + s·(x·Aᵀ)·Bᵀ`, on every path including the fused
Metal q/k/v submission. At rank 128 against a 4096×4096 projection that is
about 6% more arithmetic. On Metal the branch is fused into the base GEMM's
own submission — it reads the activation already uploaded and accumulates
into the output already written — so it costs no transfer: a 384-token step
goes 8.6 s to 10.1 s on an M4. The file itself is the only extra memory.

Three naming conventions are read as they ship — `diffusion_model.…`
(ComfyUI single-file), `base_model.model.…` (PEFT) and the bare module path.
`CMF_LORA_PROBE=1` and `CMF_LORA_ROUTE=<r>` are in [LORA.md](LORA.md).

Adapters that carry a `reference_slot_embedding` also take reference stills:

```sh
cortiq ltx-video --model $M --lora msr.safetensors \
  --ref a.ppm --ref b.ppm --ref c.ppm --ref-frames 25 \
  --prompt "Image 1: … Image 2: … Image 3: …" --out clip.y4m
```

Each still is held for `--ref-frames` pixel frames (25 or 33, whichever the
adapter was trained on), encoded by the same video VAE the render uses, given
its slot's learned per-channel bias, and placed at a negative frame offset —
slot 1 furthest back, the last reference nearest the clip. Those tokens ride
in the same sequence as the clip, frozen, and are cropped off the result. The
stills must already be the render's width and height; this build does not
guess an aspect fit.

References cost sequence length: three of them at 384×256 add 1152 tokens to
a 384-token clip, so the step is four times the work. Name them in the prompt
the way the adapter expects (`Image 1`, `Image 2`, …).

Both halves are refused rather than approximated when a file does not match:
an adapter with a lone `lora_A`, a slot embedding whose width is not the
latent's channel count, or metadata asking for a token order this build does
not implement, all stop with a sentence instead of rendering something that
looks plausible.

### The prompt is encoded once

The 12 B prompt encoder depends on nothing but the token ids and the
container, so its output is cached under `~/.cache/cortiq/ltx-context/` and a
second render of the same prompt reads it back instead of recomputing:

```
prompt: 24 tokens → 1024-token context in 29.1s      # first
prompt: 24 tokens → 1024-token context from cache in 0.03s
```

Twenty-five megabytes a prompt. `--no-context-cache` turns it off; the key
covers the ids and the container's path, size and mtime, so a different pack
never reuses another's context.

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

### Decode a soundtrack

```bash
cortiq ltx-audio --model ltx25-q4tp.cmf \
  --latent latent.safetensors --out out.wav --stats
```

`ltx-video --out-latent` writes the audio latent alongside the video one, so
the audio tail — seconds of work behind minutes of denoising — can be
re-run on its own. `--stats` prints what every stage produced: the latent, the
log-mel, and the waveform's envelope over time.

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
