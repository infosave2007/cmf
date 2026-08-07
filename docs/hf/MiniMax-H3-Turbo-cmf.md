---
license: apache-2.0
base_model:
- Comfy-Org/MiniMax-H3
- larryvrh/MiniMax-H3-Turbo-Lora
base_model_relation: quantized
pipeline_tag: text-to-video
tags:
- cmf
- cortiq
- video
- audio
- 4-bit
---

# MiniMax-H3 Turbo — one 23.5 GB file, no Python

[MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) renders video and
synchronized stereo audio from one prompt, in one transformer, on two flow
schedules. [larryvrh's Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
brings it to four sampling steps. This is both of them in the
[CMF container](https://github.com/infosave2007/cmf) — the DiT, the Qwen3-VL
prompt encoder, the video VAE decoder and the audio vocoder in a single
memory-mapped file — running on `cortiq`, a Rust binary with no ML framework
underneath.

| | reference checkout | here |
|---|---|---|
| diffusion model | 66.3 GB (bf16) | — |
| prompt encoder | 51.5 GB (bf16) | — |
| video + audio VAE | 5.8 GB | — |
| Turbo LoRA | 0.8 GB | — |
| **total** | **124.4 GB, four files + a ComfyUI checkout** | **23.5 GB, one file** |

47.83 B parameters, 2 361 tensors, `cortiq verify` clean.

## What comes out

![A corgi in a chef hat over a pan, four-step render](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/samples/corgi_512x288_4step.gif)

*"A corgi in a chef hat flipping a pancake, sizzling sounds and a cheerful bark."*
— 512×288, 39 frames at 24 fps, seed 42, **four steps**, nothing but the prompt.

The GIF is silent; the audio is the point, so take the
**[mp4](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/samples/corgi_512x288_4step.mp4)**.
It is not a second model: the same transformer denoises both streams in one
packed sequence, on two different flow schedules.
[`samples/`](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/tree/main/samples)
also holds the AVI `cortiq animate` actually wrote and its `.wav` — the mp4 and
the GIF are remuxes for the browser, and the runtime itself never touches
ffmpeg.

The LoRA is not a separate download: it is merged into the weights, so the file
IS the 4-step model.

**Text-to-video and keyframe-to-video.** Prompt in, video and audio out; or
give it a first and/or last frame and it continues from there. The release's
third path — `ref2va`, conditioning on reference images, clips and audio — is
not ported.

| file | size | what differs |
|---|---|---|
| `mmh3-turbo-fl2va-q4tp.cmf` | 23.94 GB | every projection at 4 bits |
| `mmh3-turbo-fl2va-q2tp.cmf` | 18.74 GB | the gate/up planes at 2 |

## Keyframe to video

![The corgi flipping the pancake, started from one frame](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/samples/i2v_corgi_flip.gif)

```bash
cortiq animate mmh3-turbo-fl2va-q4tp.cmf \
  --prompt "the corgi lifts the pan and flips the pancake high, sizzling" \
  --first-frame keyframe.ppm --out flip.avi
```

One picture conditions the run twice, and both halves matter. Its VAE latent
becomes a row the DiT holds at a timestep of its own near 1 — a condition, not
noise being removed — and never denoises. The picture ITSELF goes to the prompt
encoder through Qwen3-VL's vision tower, as `"<Picture 1>: "` and a vision
block: at 512×288 that is 144 tokens of the 168 the prompt above carries.
Leave one out and the model is conditioned on something the reference never
conditions on.

`--last-frame` anchors the other end. The first frame is a geometry anchor and
is stretched to the canvas; the last one follows and is cover-cropped, which is
what the reference does with each. Frames come in as binary P6 PPM.

## Running it

### 1. Get the runtime

`cortiq` is one Rust binary. Either install it —

```bash
cargo install cortiq-cli          # needs Rust 1.85+; brings the GPU backend
```

— or take a prebuilt archive from the
[latest release](https://github.com/infosave2007/cmf/releases/latest)
(Linux x86-64, macOS on Apple Silicon and Intel, Windows x86-64 and ARM64;
each ships a `.sha256`). Nothing else is required: no Python, no PyTorch, no
CUDA toolkit, no ffmpeg.

Check it took:

```bash
cortiq --version
```

### 2. Get the weights

One file, 23.5 GB.

```bash
pip install -U "huggingface_hub[cli]"      # only to fetch the file
hf download infosave/MiniMax-H3-Turbo-cmf mmh3-turbo-q4tp.cmf --local-dir .
```

Confirm it arrived whole — the container carries a hash per tensor:

```bash
cortiq verify mmh3-turbo-q4tp.cmf     # → ✓ all tensor hashes match
cortiq info   mmh3-turbo-q4tp.cmf     # → arch, layers, 47.83B params
```

### 3. Render

```bash
cortiq animate mmh3-turbo-q4tp.cmf \
  --prompt "A corgi in a chef hat flipping a pancake, sizzling sounds and a cheerful bark." \
  --width 512 --height 288 --frames 39 --steps 4 --seed 42 \
  --out corgi.avi
```

That writes `corgi.avi` — MJPEG video with PCM stereo, playable in VLC, mpv,
QuickTime and Windows Media Player — and `corgi.wav` beside it. The JPEG
encoder and the RIFF muxer are inside the binary: a pipeline that ends in a
shell-out to a 20 MB dependency is not a pipeline you can ship. If you want an
mp4 for a browser, remux it yourself; the model never needs one.

**On a GPU.** The device path is opt-in for this model while its kernels earn
their keep (see below):

```bash
CMF_MMH3_GPU=1 cortiq animate mmh3-turbo-q4tp.cmf --prompt "…" --out corgi.avi
```

### Options that matter

| flag | default | what it does |
|---|---|---|
| `--width` / `--height` | 512 × 288 | multiples of 32. The trained short edge is 768; below ~256 the model drifts off-distribution |
| `--frames` | 39 | at 24 fps, snapped **up** to the model's 17k+5 grid: 5, 22, 39, 56, … 124. 124 ≈ 5 s, and 124–362 is the validated range |
| `--steps` | 4 | what the Turbo LoRA is trained for. More still helps a little |
| `--seed` | 42 | same seed, same prompt, same size → the same clip, byte for byte |
| `--quality` | 92 | JPEG quality of the AVI's frames |
| `--stock-sampler` | off | integrate the audio on the video's clock, as a single-schedule sampler does. Wrong at 4 steps — it is here to hear how wrong |

| environment | what it does |
|---|---|
| `CMF_MMH3_GPU=1` | opt into the device path |
| `CMF_THREADS=n` | cap the worker pool (defaults to the machine's cores) |
| `CMF_ANIM_PROF=1` | per-step rms of both latent streams and both velocities |

### What it needs

RAM at least the file's size — 24 GB — or every step faults on non-resident
pages; the weights are memory-mapped, not read. Disk: 24 GB. A GPU is optional
and wants ~14 GB of VRAM for the DiT's planes. No network access at run time.

## What it costs to run

The file is memory-mapped, so plan on RAM at least its size or every step
touches non-resident pages.

512×288, 39 frames, 4 steps, one machine — 48 CPU cores and one RTX PRO 6000
Blackwell:

| | denoise | decode | total |
|---|---|---|---|
| host | 198.2 s | 147.8 s | **346.5 s** |
| `CMF_MMH3_GPU=1` | 96.8 s | 74.7 s | **172.0 s** |

and the smaller size, on the device: 256×160 over 22 frames in **29.2 s**.

Nearly all of the decode is the video VAE — the vocoder is 4 s of it. The
packed sequence is `[text | audio | video]` and everything attends to
everything, so cost grows with the token count and then with its square: a
512×288 second is five times the tokens of a 256×160 one.

**A free 2× on the decoder, if you want it.** The video VAE decodes in
256-pixel tiles, always, and grows the OVERLAP rather than the tile count —
so a 288-pixel edge is covered by two 256-pixel tiles overlapping by 224, and
you pay for 512 rows to get 288. An edge of exactly 256 is one tile. 512×256
therefore decodes three tiles where 512×288 decodes six, for 89% of the
pixels. The schedule is the reference's and this port reproduces it exactly;
picking an edge that lands on it is free.

**Host and device do not agree to the last bit, and neither is wrong.** The
host arm quantizes activations to int8 (`CMF_SDOT`) where the device
dequantizes to f32, so the two renders differ by a few per cent in latent rms
and visibly in fine texture. Set `CMF_SDOT=0` on both sides to compare
arithmetic instead of that approximation.

**Why the device is opt-in.** Getting it right took three fixes, and one
thing is still held back.

The engine's blocked f32 GEMM cached its weight-side device buffer **by
pointer address**. Every batched attention allocates one k/v scratch pair per
call and refills it per head — same address, different matrix — so head 0's
keys came back for every head, on the GPU only, silently. It is keyed on a
content fingerprint now. The same GEMM also took every job over 4 M MACs on
sight with no CPU arm to lose to, which on this model's decoder was three
times *slower* than the host it displaced; it goes through the same
measure-don't-assume probe as every other op class now, and on this stack the
probe hands that work back (0.24 ms device against 0.13 host) while sending
the weight GEMMs to the card (25.8 ms against 92.0).

Still held: **the cooperative-matrix kernel runs this model out of f16 range.**
At 256×160 the render is correct; at 512×288 the audio stream goes NaN on the
second sampling step and the video follows. Bisected — `CMF_BAKE_GPU=0` does
not help, `CMF_COOP=0` does — so `cortiq animate` pins `CMF_COOP=0`.

That hold is specific to this model, not a verdict on the kernel: the image
model on the same card and the same kernel renders 20.5 s without it against
14.8 with, and the two agree to 42.6 dB — the price of f16 operands, which
the kernel documents, not a fault. MiniMax-H3's activations are simply larger.
Giving that kernel a scale is the next real speedup here.

## What the conversion did

**The adaLN collapse.** Forty per cent of the released DiT is one matrix per
block: `adaln_proj.linear` is `[96768, 2688]`, 520 MB at bf16, **13 B of the
model's 33 B parameters** — for a map whose input is one number, the timestep.
Its output over the whole schedule is a one-dimensional curve in R^96768, and
Comfy-Org's `pruned` checkpoints already ship it as one: an `adaln_t_table` of
`[1025, 8]` shared by every block and per-block weights of `[96768, 8]`.

Measured against the full matrix on block 0 (`tools/mmh3_fetch.py check`, which
range-reads 520 MB out of the 66 GB file rather than downloading it):

```
adaln  max|Δ| 8.0e-4   rms 8.7e-5   against a signal of rms 0.464
time-curve singular values 1..12, relative:
  1.00e0 2.96e-1 1.05e-1 6.63e-2 6.60e-3 2.11e-3 5.61e-4 2.92e-4
  3.67e-5 2.73e-5 1.32e-5 1.34e-6
```

The ninth singular value is already 3.7e-5 of the first. Rank eight is not an
approximation anyone should feel nervous about; the 26 GB is redundant.

The Turbo LoRA is written against the FULL matrix (`lora_A` is `[16, 2688]`),
which is why the ComfyUI node re-injects the time conditioning at run time when
the base is pruned. `cortiq animate-pack` does it once, at conversion:

```
adaln(t) = W_p · u(t) + b + B · (A · silu(e(t)))
         = [W_p | B] · [u(t) ; A · silu(e(t))]
```

— a rank-24 curve, driven by a `[1025, 24]` table per block. 4.6 MB a block
instead of 520, with the LoRA already inside it.

**The rest.**

- **Backbone** — bf16 → `q4tp`, 4.16 bits a weight with a predicted per-row
  scale ladder. The LoRA's rank-64 update is merged before quantizing.
- **Prompt encoder** — Qwen3-VL-32B truncated to 50 layers, 51.5 GB → 12.2 GB.
  It is the largest single component of the file and it runs once per
  generation.
- **Video VAE — decoder only.** It is a ViT3D, not a conv stack: 36 transformer
  blocks over the latent grid and one linear that expands each cell into a
  4×16×16 block of pixels. The 3-D causal CNN encoder is a third of the
  checkpoint and text-to-video never runs it.
- **Audio VAE — decoder only**, f16. Quantizing a vocoder buys 45 MB and costs
  audible hiss. Its 254 kaiser-sinc resampling filters are read from the
  checkpoint rather than re-derived — the design formula is in the code as a
  fallback, but a filter you compute is a filter that can drift from the one
  the weights were trained against.
- Integrity: 47.83 B parameters over 2 361 tensors; `cortiq verify` checks
  every one against the directory's hashes.

## On parity

Established, not assumed, and separately for each of the four stacks. The
reference is ComfyUI's own module, run on a toy checkpoint carrying the
release's real tensor names and the release's real schedules — `tools/`
builds them, `tools/mmh3_toy_gate.sh` runs the diff. The packs are exact f32
on purpose: `q4tp`'s noise floor sits an order of magnitude above the
arithmetic difference these are looking for, so quantizing here would pass a
broken port.

| stack | worst | rms | signal rms |
|---|---|---|---|
| DiT — video velocity | 8.8e-5 | 2.1e-5 | 0.515 |
| DiT — audio velocity | 5.2e-5 | 2.5e-5 | 0.409 |
| DiT — token refiner | 8.3e-7 | 2.6e-7 | 1.003 |
| Qwen3-VL encoder | 1.1e-6 | 3.3e-7 | 0.812 |
| video VAE decoder | 4.2e-7 | 4.0e-8 | 0.470 |
| audio VAE decoder | 1.7e-9 | 3.5e-10 | 8.9e-4 |

A dozen conventions in this model pass at one token and fail differently at a
hundred, which is why the toys are not one-vector unit tests: the packed
layout's cursor, the video time axis's 1,4,4,4,4 span pattern, which 96 of 128
head dimensions rotate, the adaLN row order (timestep-major, modality-minor),
the video VAE's 256-pixel tiling — global attention makes a tile a different
computation from a whole frame, so the tiling is part of the output, not a
memory strategy — and the audio stream's separate clock.

## Two clocks

The video and audio latents ride different flow schedules (shift 12 and 3).
The sampler walks the video grid, which at four steps is
`1, 0.973, 0.923, 0.8, 0`, and integrates the audio on its own remap of it.
Stepping both on the video grid is what a stock sampler does; it is fine at
twenty steps and wrong at four, because over the last interval Δσ_a and Δσ_v
differ by a factor of three and no per-step slope correction survives a step
that large. `--stock-sampler` reproduces the broken behaviour if you want to
hear it.

## Provenance

Weights derive from MiniMax's H3 release as repackaged by Comfy-Org, and from
larryvrh's Turbo LoRA; both remain under their own licences. The Turbo LoRA is
a **preview** — its own card notes plastic-looking skin and over-sharp grain at
`ckpt850`, and nothing here changes that. The CMF container and the cortiq
runtime are Apache-2.0 (see the repository's LICENSE and PATENTS.md).
