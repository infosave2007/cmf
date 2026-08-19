---
library_name: cortiq
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

# MiniMax-H3 Turbo — one file, no Python

[MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) renders video and
synchronized stereo audio from one prompt, in one transformer, on two flow
schedules. [larryvrh's Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
brings it to four sampling steps. This is both of them in the
[CMF container](https://github.com/infosave2007/cmf) — the DiT, the Qwen3-VL
prompt encoder, the video VAE decoder and the audio vocoder in a single
memory-mapped file — running on `cortiq`, a Rust binary with no ML framework
underneath.

The reference checkout is 124.4 GB across four files plus a ComfyUI install.
Here it is **one file between 13.2 and 23.9 GB**, and which one you take is
decided by your VRAM.

## Same prompt, three files

*"A corgi in a chef hat flipping a pancake, sizzling sounds and a cheerful bark."*
— 512×288, 39 frames, seed 42, four steps, nothing but the prompt.

| ![32B](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/samples/ab_q4tp.gif) | ![ClipProj](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/samples/ab_clipproj.gif) | ![two bits](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/samples/ab_q2tp.gif) |
|---|---|---|
| **23.9 GB** — full 32B prompt encoder | **13.2 GB** — 4B encoder + ClipProj | **18.7 GB** — two-bit encoder |
| the reference | same scene, plainer set | a different animal, no pan |

The 13.2 GB file is the one to take if you have 16–20 GB of VRAM: it holds the
whole run resident instead of paging, and it is still four bits everywhere.
Two bits is published as a dead end, not a choice —
[why, with numbers](#making-it-smaller).

The GIFs are silent and the audio is half the model, so take an
**[mp4](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/samples/ab_clipproj.mp4)**.
One transformer denoises picture and sound together in one packed sequence.

### Which file

| file | size | VRAM | keyframes | |
|---|---|---|---|---|
| `mmh3-turbo-fl2va-q4tp.cmf` | 23.94 GB | 24 GB+ | yes | **the default** |
| `mmh3-turbo-q4tp.cmf` | 23.47 GB | 24 GB+ | no | same, without the vision tower |
| **`mmh3-turbo-clipproj4b-q4tp.cmf`** | **13.16 GB** | **16–20 GB** | no | **the small one**, peaks at 15.1 GB |
| **`mmh3-turbo-clipproj4b-fl2va-q4tp.cmf`** | **14.48 GB** | **16–24 GB** | **yes** | the small one WITH start/end frames — the 4B vision tower and the VAE encoder join the compact build |
| **`mmh3-turbo-clipproj4b-fl2va-v2-q8_2f.cmf`** | **26.90 GB** | **32 GB+** | **yes** | **eight bits**: the two-field int8, `w = q·row[o]·col[i]`. More weight fidelity, and it runs on the card as of 0.5.94 — still without the fused kernels q4tp has |
| `mmh3-turbo-fl2va-q2tp.cmf` | 18.74 GB | — | yes | don't render with this |

Text-to-video everywhere; the `fl2va` files also take a first and/or last frame
(`--first-frame`/`--last-frame`, binary P6 PPM). The compact
`clipproj4b-fl2va` file was verified frame-for-frame: the conditioning
image comes back as frame 0 of the render. One honest caveat: its
ClipProj projection was fitted on text-only encoder activations, so with
a picture in the prompt the conditioning quality on complex scenes is
still being compared against the full-encoder fl2va files — report what
you see in the discussions.
([keyframes](#keyframe-to-video)). The release's third path, `ref2va`, is not
ported. The Turbo LoRA is merged into the weights, so the file IS the 4-step
model — nothing else to download. 47.83 B parameters, 2 361 tensors,
`cortiq verify` clean.

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

One file. Pick it from [Which file](#which-file) above — this is the
text-to-video default; swap the name for `mmh3-turbo-clipproj4b-q4tp.cmf` if
you are on a 20 GB card.

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

**On a GPU.** Nothing to opt into any more: `cortiq` probes this file's own
first qkv weight against the host on startup and takes the device arm only if
they agree, so the arm that renders is the arm that was checked.

```bash
cortiq animate mmh3-turbo-q4tp.cmf --prompt "…" --out corgi.avi
```

On one RTX 5090, 512×288, 39 frames: **60.2 s at the default four steps**
(91.6 s at eight, 42.8 at two). `RUST_LOG=info` prints a per-stage breakdown
for every run, and the full table is under *What it costs to run* below. The
whole pipeline stays on the card —
the DiT block, both VAE decoders, and the vocoder's dilated convolutions — so
nothing but the finished frames crosses the bus. `CMF_MMH3_GPU=0` forces the
host path if you want to compare.

**Two cards.** A render does not split across them, and should not: the DiT
block and both decoders are already resident, so a second card would only add
a bus crossing to a pipeline that no longer has one. Two cards double your
*clips*, not your clip — run two processes, one pinned to each:

```bash
CMF_GPU_ADAPTER=0 cortiq animate model.cmf --prompt "…" --seed 1 --out a.avi &
CMF_GPU_ADAPTER=1 cortiq animate model.cmf --prompt "…" --seed 2 --out b.avi &
wait
```

`cortiq gpu` lists the cards and their indices. For text models the same
binary both splits and replicates across cards — see
[docs/MULTI_GPU.md](https://github.com/infosave2007/cmf/blob/master/docs/MULTI_GPU.md).

### Options that matter

| flag | default | what it does |
|---|---|---|
| `--width` / `--height` | 512 × 288 | multiples of 32. The trained short edge is 768; below ~256 the model drifts off-distribution |
| `--frames` | 39 | at 24 fps, snapped **up** to the model's 17k+5 grid: 5, 22, 39, 56, … 124. 124 ≈ 5 s, and 124–362 is the validated range |
| `--steps` | 4 | what the Turbo LoRA is trained for. More still helps a little |
| `--seed` | 42 | same seed, same prompt, same size → the same clip, byte for byte. This is now true on the GPU too: the op arbitration used to alternate arms on real data while it made up its mind, and two runs of one binary could differ |
| `--quality` | 92 | JPEG quality of the AVI's frames |
| `--stock-sampler` | off | integrate the audio on the video's clock, as a single-schedule sampler does. Wrong at 4 steps — it is here to hear how wrong |

| environment | what it does |
|---|---|
| `CMF_MMH3_GPU=1` / `=0` | force the device or the host path instead of letting the parity probe choose |
| `CMF_GPU_PROBE=0` | pin the op arbitration (already the default for `animate`, so a seed reproduces) |
| `CMF_THREADS=n` | cap the worker pool (defaults to the machine's cores) |
| `CMF_ANIM_PROF=1` | per-step rms of both latent streams and both velocities |
| `CMF_MM_KILL=0` | never fall back to the host on a slow device op. The engine treats three consecutive over-budget GEMMs as "another process owns the card" and finishes the run on the CPU; a weight paging in from disk is exempt, but on a machine where the file exceeds RAM the first steps can be slow for reasons that are not contention (see the field notes) |

### What it needs

RAM at least the file's size — 24 GB — or every step faults on non-resident
pages; the weights are memory-mapped, not read. Disk: 24 GB. A GPU is optional
and wants ~14 GB of VRAM for the DiT's planes. No network access at run time.

`mmh3-turbo-clipproj4b-q4tp.cmf` asks for 14 GB of RAM and disk instead, and
peaks at 15.1 GB of VRAM — a whole-run maximum, polled, not a snapshot. That is
what makes it the one to reach for on a 20 GB card: below that the prompt
encoder pages, and on this workload paging costs more than any kernel.

### Field notes: 24 GB Macs and 20 GB cards

Two users measured what this card could not, and the engine changed on
their reports (HF discussions #1, #2 and #4). The numbers, as they sent them:

| machine | file | render | result |
|---|---|---|---|
| RTX 3080 20 GB, Windows | `mmh3-turbo-q4tp` (25.2 GB) | 512×288×39, 4 steps | **1218 s** — the encoder did not fit the 19 GB budget and paged every step |
| RTX 3080 20 GB, Windows | `mmh3-turbo-clipproj4b-q4tp` (13.2 GB) | same | **105 s** — resident; denoise 62.8 s, video VAE 30.6 s |
| Mac mini M4 24 GB | `mmh3-turbo-clipproj4b-q4tp` | 512×256×39, 4 steps | **174 s** — denoise 122 s, video VAE 41 s |
| Mac mini M4 24 GB | `mmh3-turbo-clipproj4b-fl2va-q4tp` (14.5 GB) | 22 frames from a keyframe | **92 s**, no swap |
| Mac mini M4 24 GB | `mmh3-turbo-fl2va-q4tp` (25.7 GB), 0.5.79 | 10 / 40 frames | **48 s / 140 s per denoise step** on the GPU, 20 GB resident, 0.3 GB swap |
| Mac mini M4 24 GB | same, 150 frames | | 8 GB of swap and a sawtooth — the activation cache no longer fits |
| Mac mini M4 24 GB | `mmh3-turbo-clipproj4b-fl2va-v2-q4tp` (14.5 GB) | 448×768, 50 frames | **136 s per denoise step**, faces hold through the clip; 90 frames at that size is the paging threshold |
| Mac mini M4 24 GB | `mmh3-turbo-fl2va-q4tp` (25.7 GB) | 512×256 | safe to ~70–90 frames; **768×448 caps at 40–50** (90 frames: 22.2 GB RAM + 1.1 GB swap, the GPU stalls) |
| Mac mini M4 24 GB | any file, `--first-frame` | | the video-VAE **encoder** of the keyframe ran ~100–140 s on the CPU (`encode 0/1 (140.8s)`); text-to-video skips it (0.1 s) |

What follows from them, if you are on such a machine:

- **On a 24 GB Mac the 25.7 GB file works after 0.5.79, for clips of
  ≤ 40 frames.** The prompt encoder's pages are released after the text
  encode; what remains — DiT, VAE decoders and the activation cache — fits
  up to ~40 frames. Past that the cache spills to swap. For longer clips
  chain 40-frame chunks (`--last-frame` of one render becomes
  `--first-frame` of the next) or use the `clipproj4b` files, which leave
  the room.
- **The Metal driver budgets by buffer size, not resident pages.** The
  weight arena's overlapping windows read as ~27 GB to the driver even
  after the encoder is released, so the first ops after the encode page
  from the SSD and can take seconds. Before 0.5.80 a single such op tripped
  the contention kill and the rest of the run walked the CPU (>60 s a
  step); the kill now needs three consecutive strikes, exempts weights
  that were not resident, and `CMF_MM_KILL=0` turns it off. If a run still
  says `device contended, CPU for the rest of the process` on a machine
  nobody else is using, that variable is the answer.
- **The kill is disarmed during the prompt encode (0.5.80).** The
  encoder is a one-shot pass over 12 GB of weights; on a machine the file
  does not fit it streams from disk, and its GEMMs run over any budget
  for reasons that are not contention — the report that had to gut
  `mm_kill` in the source was on exactly that. The kill now arms only
  when the denoise loop starts, and three strikes are still needed.
- **The keyframe encode is CPU work today.** With `--first-frame` the
  video-VAE encoder runs the reference picture on the host (~100–140 s
  on an M4 for a 448×768 frame; 0.1 s without a keyframe). 0.5.80 puts
  the pool's workers on the performance cores (they were landing on the
  E-cores with the P-cores asleep — that report's `asitop`), which
  shortens it; a device path for the encoder's 3-D convolutions is the
  real fix and is on the list.
- **Frame budgets on 24 GB, from that user's sweep:** with the 25.7 GB
  file 512×256 is safe to ~70–90 frames and 768×448 caps at 40–50; with
  the 14.5 GB v2 file 448×768 at 50 frames is the sweet spot (136 s per
  step) and 90 frames is where paging starts. Past those the machine
  swaps and the GPU stalls to nothing — better to chain clips than to
  push the frame count.
- **The draft → final workflow.** Block the shot on `clipproj4b-fl2va-v2`
  (90 s on the Mac; v2 holds faces through a clip — "25 GB-level face
  consistency at 14 GB speed", that user's words), then pay the full
  encoder once for the final take. Identity of *specific* real people is
  still the full encoder's territory.
- **`--height 256` instead of 288** halves the video-VAE decode on every
  machine (three 256-pixel tiles instead of six).
- **Voices are prompt space *here*, not in the release.** Speaker identity,
  timbre, pace and emotion respond to stage directions in the prompt, and a
  fixed seed keeps the same actor across takes — which is how to work with
  this file today. But the earlier claim on this card that "H3 has no
  reference-audio input" was **wrong**, and a user was right to push back
  (discussion #3): the release is tagged `audio-to-audio-video` and
  `video-to-audio-video`, and the DiT takes both as conditioning rows. What
  is missing is on our side and it is not training — see
  [What this port does not do yet](#what-this-port-does-not-do-yet).

## Making it smaller

Half this file is the PROMPT ENCODER — 12.2 GB of Qwen3-VL against the DiT that
actually draws. Two ways to act on that were tried, and the three clips at the
top of this card are the result: **squeezing the encoder does not work,
replacing it does.** Here is why, in both directions.

### Two bits: smaller, faster, and answering a different question

The obvious next cut is the DeepSeek-V4 policy: gate/up at two bits,
everything else at four. It builds — 23.94 GB down to **18.74**, and
with a device kernel of its own it renders *faster* than the four-bit
file — 217.3 s against 258.4 when the pair was measured, because there is
less weight to move. (Both numbers are from the build of that day; the
four-bit file renders the same clip in 60.2 s now. The ratio is what
carries over, not the seconds.)
`cortiq verify` passes. The file is here.

It also stops following the prompt, which is why it is not the one to
reach for.

| | |
|---|---|
| ![four bits](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/samples/ab_q4tp.gif) | ![two bits](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/samples/ab_q2tp.gif) |
| `q4tp` — 23.94 GB | `q2tp` — 18.74 GB |

Same prompt, same seed, same four steps. On the left the corgi is behind
a pan with batter in it, drawn flat and clean, which is what was asked
for. On the right it is a different animal in a different style with
**no pan and no pancake at all**, over a washed-out ground with visible
texture noise. That is not a quantizer trading detail for size; that is
a model answering a different question.

The likely culprit is where the two bits landed. Half this file is the
PROMPT ENCODER, and the policy put two bits on its gate/up planes along
with the DiT's — so the loss falls on the part that decides what the
clip is about, not on the part that draws it. A two-bit build confined
to the DiT would save ~2.9 GB instead of 5.2 and is the version worth
measuring next — the packer's policy is one predicate,
`is_wide_plane`, if you want to try it. The file above is published so
that experiment starts from something rather than nothing; four bits is
what to render with.

### A smaller encoder: replace it, don't squeeze it

The two-bit section above ends on a diagnosis: half this file is the PROMPT
ENCODER, and squeezing it is what broke the prompt. There is a second way to act
on that diagnosis — **don't compress the encoder, replace it.**

[ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj) fits a ridge
regression from a small Qwen3-VL's hidden state into the space the DiT was
conditioned on. Same tokenizer, same family, one affine map:

```text
cond = ((h - mean_in) / std_in) @ W [+ GELU residual] * std_out + mean_out
```

`mmh3-turbo-clipproj4b-q4tp.cmf` is the text-to-video file with Qwen3-VL-4B
tapped at layer 24 standing in for the 32B tapped at 50, plus the 304 MB
projection kept **exact** — f32, no quantization on the piece that carries the
whole substitution. Everything else is copied through byte for byte: DiT and
both VAE decoders, still `q4tp`. Nothing anywhere is two bits.

**23.47 GB → 13.16 GB.** The encoder went from 552 tensors to 277.

#### Does it still mean the same thing?

That is measurable without rendering a frame: dump what the DiT actually
receives (`CMF_TE_DUMP=<path>`) from both files on one prompt, and take the
cosine per token.

| | mean cosine to the 32B | worst token |
|---|---|---|
| 4B tapped at **24** | **0.9198** | 0.7982 |
| 4B tapped at 25 | 0.8452 | 0.5910 |
| *floor — two different tokens of the 32B itself* | *0.5914* | |

The tap index is 0-based, and off-by-one is a real failure mode rather than a
theoretical one: at 25 the worst token sits **on the floor**, uncorrelated.
Token 0 is not projected at all but replaced by a stored `sink_out` — the
attention sink is an outlier no regression fits — and it lands at cosine
**0.9999** against the teacher's, which is the sharpest single confirmation
that the mechanism is wired right.

#### What it buys

The one number that transfers between machines is the **encode stage**, because
that is the only stage ClipProj replaced. Same prompt, seed and four steps,
512x288x39, measured inside one run:

| | prompt encode |
|---|---|
| `mmh3-turbo-q4tp` — 32B tapped at 50 | 49.2 s |
| `mmh3-turbo-clipproj4b-q4tp` — 4B tapped at 24 | **1.9 s** |

Everything after that is residency, and residency is a property of YOUR card,
not of this file. On the 24 GB RTX 3090 these were taken on, the whole render
came out roughly half the time of the 23.47 GB file and the video VAE decode
more than halved — but totals on that box drifted between 140 s and 210 s for
one unchanged configuration as the card warmed, so treat the ratio as a
direction and measure your own.

The point is the 20 GB card this variant exists for, where the difference is
not a ratio but a cliff: a reader of this repo measured **20 minutes** for one
512x288 clip on an RTX 3080 20 GB, paging the 23.47 GB file through a ~19 GB
weight budget. At 13.16 GB, with a polled whole-run peak of 15.1 GB of VRAM,
there is nothing left to page.

The parity probe takes the device arm on both files at the same `rel rms
4.65e-3`, which is its own small proof that the DiT came through untouched.

#### What it costs in quality

Judge it on the clip at the top of this section, not on a frame: a still
catches the pancake at rest and reads as a loss that is not there.

Across the 39 frames the 4B does the whole job: corgi, chef hat, flat clean
style, and the pancake **leaves the surface and comes back**, the dish empty
underneath at the top of the toss. The verb survives — that is what `q2tp`
lost, along with the animal and the pan.

What drifts is set dressing. The 32B renders the cooking surface as a griddle
and puts patterned wallpaper behind; the 4B gives a rimmed white plate and a
plain ground. A 0.92 cosine is close, not equal, and where it is not equal is
the scene's furniture rather than its subject or its action.

So the ordering is not "smaller is worse". The two-bit file answers a different
question; this one answers the right question with a plainer set, at 56% of
the size. If you have the VRAM for the 32B encoder, use it —
its framing is richer. If you are paging, this is the better trade, and unlike
the two-bit build it is a trade rather than a loss.

Sound is untouched either way: the audio branch never sees the prompt encoder.
[The clip with its audio.](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/samples/ab_clipproj.mp4)

#### Building one

```bash
cortiq animate-pack \
  --in  mmh3-turbo-q4tp.cmf \
  --te  qwen3vl_4b_bf16.safetensors --te-layers 24 \
  --clip-proj mmh3-4b-ClipProj-celeb-mlp.safetensors \
  --quant q4tp --out mmh3-turbo-clipproj4b-q4tp.cmf
```

`--te-layers` is not a size knob, it is the tap: layers above it never execute,
so packing them is pure file. `--clip-proj` carries the projection in exact and
replaces the encoder **as a whole component** — matching tensor names alone
would leave `te.layers.24..49` of the 32B behind, six gigabytes that
`num_hidden_layers` then excludes from the forward while disk and VRAM budget
still pay for them.

The encoder is [`Comfy-Org/Qwen3-VL`](https://huggingface.co/Comfy-Org/Qwen3-VL)
`text_encoders/qwen3vl_4b_bf16.safetensors`, and the projection is
[`NicoLab28/ClipProj-MiniMax-H3`](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3)
`mmh3-4b-ClipProj-celeb-mlp.safetensors` — use the file the projection was
fitted on, since the map is only valid for those exact weights. An 8B
projection is published too and scores higher on its author's corpus (0.8037
against 0.7930); it costs about 4 GB more.

## What it costs to run

The file is memory-mapped, so plan on RAM at least its size or every step
touches non-resident pages.

One RTX 5090, 4 steps, 39 frames. `RUST_LOG=info` prints this breakdown for
every run:

| | text | denoise | video VAE | audio VAE | total |
|---|---|---|---|---|---|
| 512×288 | 2.5 s | 34.8 s | 16.6 s | 4.3 s | **60.2 s** |
| 512×256 | 2.3 s | 31.4 s | 8.4 s | 4.3 s | **48.5 s** |
| 256×160, 22 frames | | | | | **15.9 s** |
| 512×288, host only (`CMF_MMH3_GPU=0`) | 2.6 s | 363.5 s | 271.5 s | 7.1 s | **646.1 s** |

The card is **10.7× the host path** on the same machine, same seed. At 8
steps a 512×288 clip is **91.6 s**; at 2 it is 42.8. The same 4-step
render took 172 s when this card was written — the pipeline has since moved
onto the card end to end, both VAE decoders with it.

Nearly all of the decode is the video VAE — the vocoder is 4.3 s of it. The
packed sequence is `[text | audio | video]` and everything attends to
everything, so cost grows with the token count and then with its square: a
512×288 second is five times the tokens of a 256×160 one.

**A free 2× on the decoder, if you want it.** The video VAE decodes in
256-pixel tiles, always, and grows the OVERLAP rather than the tile count —
so a 288-pixel edge is covered by two 256-pixel tiles overlapping by 224, and
you pay for 512 rows to get 288. An edge of exactly 256 is one tile. 512×256
therefore decodes three tiles where 512×288 decodes six, for 89% of the
pixels. Measured on the current build: the video VAE goes 16.6 s → 8.4 s and
the whole render 60.2 s → 48.5 s. The schedule is the reference's and this
port reproduces it exactly; picking an edge that lands on it is free.

**Host and device do not agree to the last bit, and neither is wrong.** The
host arm quantizes activations to int8 (`CMF_SDOT`) where the device
dequantizes to f32, so the two renders differ by a few per cent in latent rms
and visibly in fine texture. Set `CMF_SDOT=0` on both sides to compare
arithmetic instead of that approximation.

**Why the device took a while to trust.** It is no longer opt-in — the parity
probe decides per file — but getting there took three fixes, and one thing is
still held back.

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

## What this port does not do yet

The release is tagged for six conditioning paths. This container packs one of
them — `fl2va`, text and/or keyframes → video + audio. What the others need is
listed here so nobody has to guess whether it is a missing feature or a
missing possibility. **None of them needs training.**

| the release's path | what it takes | what is missing here |
|---|---|---|
| `audio-to-audio-video` | a reference soundtrack | the audio VAE's **encoder half**. The container carries the decoder only — `pack_audio_vae` skips the encoder, `pre_block` and the mean/logs heads as unused. The DiT side already exists: the packed layout has a reference-audio segment kind and its own condition timestep |
| `video-to-audio-video` | a reference clip | the sampler plumbing. The video VAE **encoder is already packed** in the `fl2va` files — it is what encodes `--first-frame` — so this is a layout and CLI change, not a weight change |
| `ref2va` (subject / character references) | 1–N reference images | a **different DiT checkpoint** (`minimax_h3_ref2va_*`) with its own turbo LoRA. It would be a second container, not a flag on this one |

Two more, on the runtime side: the latent upscaler published for H3
(a 345 M-parameter 3-D conv net) is not ported, and there is no fused
device kernel for adapter branches on this model — see below.

## The eight-bit build — now the fastest file here

`mmh3-turbo-clipproj4b-fl2va-v2-q8_2f.cmf` (26.90 GB, 26.21 B parameters,
`cortiq verify` clean) packs the DiT as **`q8_2f`** — the two-field int8,
`w = q·row[o]·col[i]`: eight bits with a second scale field along the *input*
axis, which is where an activation-outlier channel shows up from the weight
side. As a codec it is strictly more faithful than the four-bit ladder.

Until today it was also the slow one. RTX 5090, 512×288, 22 frames, four
steps, one machine, one sitting:

| | denoise | video VAE | wall |
|---|---|---|---|
| q8_2f, 0.5.94 | 103.3 s | 62.2 s | 171.5 s |
| q8_2f, scalar int8 kernel | 56.2 s | 19.5 s | 81 s |
| **q8_2f, 0.5.95** | **31.7 s** | **9.0 s** | **46 s** |
| q4tp, same session | 59.0 s | 37.2 s | ~101 s |

Three things were in the way, and none of them was the codec:

1. **The fused chains asked for a four-bit weight by name.**
   `QTensor::mapped_q4tp` was the door to every fused submission in the DiT
   and the 3D VAE, so an eight-bit container walked past all of them into
   per-op GEMMs with a readback between each. `mapped_device_gemm` asks
   instead whether the codec *has* a device GEMM.
2. **int8 never reached the matrix units.** The four-bit path unpacks its
   weight into an f16 plane once and hands that to the cooperative GEMM;
   int8 ran the scalar kernel on the scalar ALUs. `q8_dq_f16` writes the same
   plane — and folds the column field into it, so the activation is no longer
   multiplied by that field on the host before every projection (a 40 MB pass,
   four hundred times a render).
3. **The panels stayed on the card.** qkv → attention → output projection and
   the FFN pair are one submission for this codec now, as they always were for
   four-bit.

`CMF_Q8_COOP=0` puts the scalar kernel back, `CMF_FUSED_ANY=0` the per-op
path — both arms above are from those switches, and both render the same
picture. The startup probe agrees with the CPU arm to 5.78e-3.

**The bug this shook out, since it is the more useful half of the story:**
widening that gate sent int8 weights to `vae_qkv_attn_out`, which still called
the four-bit entry points by name. It read the int8 payload as four-bit tiles,
finished three times faster than it should have, and produced a **flat grey
frame** — with every switch that could have explained it turned off. The
four-bit GEMM now refuses a tensor that is not four-bit rather than returning
plausible garbage, which is what turns that class of mistake into a fallback
instead of a mystery.

## The latent upscaler

Render small, resize the **latent**, decode once:

```bash
cortiq animate mmh3-turbo-clipproj4b-fl2va-v2-q4tp.cmf \
  --prompt "…" --width 512 --height 288 --frames 39 \
  --upscale minimax_h3_latent_upscaler_3d_fp16.safetensors --upscale-by 2.0 \
  --out big.avi
```

The net is [LBH-123-AI's](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
— 345 M parameters, twelve residual blocks and six temporal convolutions on
each side of a trilinear resize, with a scalar scale embedding modulating
every block. It is ported into the engine and gated against the node's own
torch module on the same weights: **worst 6.68e-6, relative rms 3.92e-7**.

| ![lanczos](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/images/upscaler_lanczos_1024.png) | ![the net](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/resolve/main/images/upscaler_net_1024.png) |
|---|---|
| the 512×288 render, resampled to 1024×576 | the same latent through the net, decoded at 1024×576 |

Same seed, same prompt, same denoise. On the left a resampler has nothing to
add and softens hair, coat seams and the wet road; on the right the detail is
generated. **Measured on an M4 (24 GB), 22 frames: denoise 76.9 s, upscale
13.8 s, VAE 98.2 s — 193 s for a 1024×576 clip** whose denoise only ever paid
for 512×288. The upscale itself is the cheapest stage in that list.

Why it matters on a 24 GB machine: the alternative is decoding through a
5 B-parameter VAE, resizing pixels and encoding again — the expensive path —
or interpolating the latent, which is the cheap path that ghosts. This is
neither. You keep the frame budget of the small render and get the detail of
the large one, and the file is loaded beside the container like an adapter,
so it stays under its own licence.

## Reference clips

```bash
cortiq animate model.cmf --prompt "…" --video ref_frames/ --video-stride 8
```

Every eighth `.ppm` of the directory becomes a condition pinned to its own
moment of the render, and the source clip is mapped onto the render's length
by position — a 100-frame reference conditions a 39-frame clip at the same
*moments*, not the same indices. This is the `fl2va` keyframe path with more
than two frames. It is **not** a port of the release's own `v2v` node, which
conditions differently; what it gives you is composition and motion carried
across a chain of shots, which is what the keyframe hack in discussion #6 was
reaching for.

## Streaming: a clip rendered in chunks

`mmh3-raven-streaming-q4tp.cmf` — **14.88 GB**, 2698 tensors, the
[RAVEN streaming adapter](https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA)
folded into the weights at pack time. It renders a clip the way that adapter
was trained: chunk by chunk, each one extrapolated from the frames already
finished, instead of denoising the whole sequence at once.

```bash
cortiq animate mmh3-raven-streaming-q4tp.cmf \
  --prompt "a woman in a red raincoat walks down a neon-lit street at night" \
  --width 384 --height 256 --frames 161 --steps 4 \
  --stream-chunk 13 --stream-sink 2 --stream-window 2 --out take.avi
```

`--stream-chunk` is the chunk length in *latent* frames (one latent frame is
four video frames). `--stream-sink` and `--stream-window` are counted in
latent frames too — a couple pinned at the start, a couple trailing the
current chunk, the attention-sink pattern the adapter trains — and the
already-finished frames enter the sequence at timestep 0, as clean context
rather than as noise being removed. (`CMF_STREAM_UNIT=chunks` counts them in
whole chunks instead — the reading this port shipped first. Same pictures,
and measured **109.6 s of denoise against 58.8 s** on one machine back to
back, because a chunk then attends to four chunks' worth of rows.)

**The chunk length is the knob that matters.** RTX 5090, 384×256, 41 frames
in (56 out), four steps, `sink 2 / window 2` throughout:

| | wall | denoise | picture |
|---|---|---|---|
| no streaming | 53 s | 36.8 s | coherent |
| `--stream-chunk 13` | 63 s | 47.0 s | coherent, motion holds |
| `--stream-chunk 9` | 69 s | 62.9 s | coherent |
| `--stream-chunk 5` | 75 s | 58.8 s | **drifts**: the scene changes at chunk boundaries |
| `--stream-chunk 5`, `sink 8 / window 8` | 122 s | 106.3 s | drifts *identically* |

The last row is the one worth reading twice: quadrupling the context costs
2.3× the denoise and changes nothing about the drift. Five latent frames is
simply shorter than what the adapter learned; nine to thirteen is the range
that holds.

Streaming costs about **1.3× the denoise** of the bidirectional path at this
clip length (47.0 s against 36.8 s), because a chunk's context rows are
recomputed at every one of its four steps. A KV cache over those rows is what
removes that, and it is an optimization of this path rather than a
prerequisite for it: not packing the rows a chunk may not see produces the
same attention pattern the reference gets from its cache.

**Where streaming stops being a trade and starts being the only way.** Longer
clips, same card and settings:

| clip | bidirectional | streaming, chunk 13 |
|---|---|---|
| 41 frames | 53 s | 63 s |
| 161 frames | 148 s (97.8 s denoise) | 190 s (142.8 s denoise) |
| 321 frames | **does not render** | 354 s (258.8 s denoise) |

The 321-frame bidirectional render does not fail for want of memory — it asks
for a `[gate|up]` FFN panel of exactly 2 GiB and hits the storage-binding
limit (`2147483648 exceeds 2147483644`). 0.5.95 splits that panel into passes
that fit, so it renders; but the chunked path never builds a panel that size
in the first place, and its cost grows with the clip instead of with the
square of it. Identity holds across the chunks: the same woman in the same
red raincoat at frame 4 and at frame 150.

A chunk that would leave a tail shorter than half a chunk swallows it — two
latent frames with four frames of context was the one shape that reliably
wandered into a different street half a second before the clip ended.

## Adapters at runtime

Community LoRAs for H3 run against this container as they ship:

```bash
cortiq animate mmh3-turbo-clipproj4b-fl2va-v2-q4tp.cmf \
  --prompt "r34l1sm a woman in a red raincoat on a neon street, close-up" \
  --lora h3-realism-people.safetensors --lora-strength 0.8 \
  --out take.avi
```

`--lora` reads a `.safetensors` in any of the three conventions in the wild
(`diffusion_model.…`, `base_model.model.dit.…`, or the bare module path), at
F32/F16/BF16, with either `lora_A`/`lora_B` or `lora_down`/`lora_up` naming.
It binds `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2` on all fifty
blocks and on the two token-refiner blocks, and it prints what it bound:

```
lora: rank 32, 104/104 branches bound
```

Branches it cannot bind are named rather than dropped in silence. The one real
gap is `adaln_proj.linear`: this container carries the modulation as a rank-24
curve over the timestep (that collapse is 40% of the released model's weights
and most of why the file is 14 GB), and folding an adaLN update into it needs
the time embedding, which only the packer has. Adapters that touch adaLN —
the spatial-physics and streaming ones — therefore land partially, and say so.
Bake them in instead with `animate-pack --lora … --time-embedder …`, which is
exactly how the Turbo LoRA got into these files.

**What it costs.** On an M4 (24 GB), 512×288, 22 frames, four steps, with
`fal/MiniMax-H3-Realism-People-LoRA` (rank 32, 104 branches, all of which
bind), the two runs taken back to back with memory free: **denoise 133.6 s
with the adapter against 126.1 s without — 1.06×.** The video VAE, which no
adapter touches, moved 53.4 → 56.9 s between those same runs, so on this
machine the adapter costs at or below the noise. The branch itself is half a
per cent of the projection's arithmetic and rides inside the base GEMM's Metal
submission; what it can cost is the *attention* fusion standing down, because
a branch on `qkv_proj` or `out_proj` needs the panels that fusion keeps on the
card — visible where the DiT is fully resident, hidden where weights stream.
`--lora-strength 0` reproduces the base render byte for byte.

**Which parts of an adapter matter.** `CMF_LORA_PROBE=1` prints every branch by
its measured contribution `‖s·ΔY‖/‖Y‖`, and `CMF_LORA_ROUTE=<r>` switches off
the ones below `r`. For the Realism adapter the loudest branch is 150× the
quietest, blocks 0–2 contribute nothing it would miss, and 41 of its 104
branches carry the look:

```
lora branches by contribution ‖sΔY‖/‖Y‖ (41 of 104 live):
    0.1633  on   blocks.13.attn.qkv_proj
    0.0946  on   blocks.9.attn.qkv_proj
    …
    0.0011  off  blocks.1.attn.qkv_proj
```

Rendered on those 41, the clip still looks like the adapter and not like the
base (13.34 dB PSNR against the base, where the full adapter is 13.76, and
16.75 dB against the full adapter). It did not make the step shorter on Metal
— the reason, and where routing does pay, is in
[docs/LORA.md](https://github.com/infosave2007/cmf/blob/master/docs/LORA.md).

## Provenance

Weights derive from MiniMax's H3 release as repackaged by Comfy-Org, and from
larryvrh's Turbo LoRA; both remain under their own licences. The Turbo LoRA is
a **preview** — its own card notes plastic-looking skin and over-sharp grain at
`ckpt850`, and nothing here changes that. The CMF container and the cortiq
runtime are Apache-2.0 (see the repository's LICENSE and PATENTS.md).
