---
library_name: cortiq
license: apache-2.0
base_model:
- MiniMaxAI/MiniMax-Music3
base_model_relation: quantized
pipeline_tag: text-to-audio
tags:
- cmf
- cortiq
- music
- audio
- 4-bit
---

# MiniMax-Music-3 → CMF — 20.3 GB of weights in one 5.55 GB file

```bash
cortiq music minimax-music3-q4tp.cmf \
  --prompt "Classic 1960s soul, passionate male tenor with rich vibrato, \
lush female backing vocals, gospel choir harmonies, vintage Motown and \
Stax atmosphere, groovy bassline, warm Hammond organ, horn section, \
analog tape saturation, romantic nighttime mood." \
  --lyrics "[verse]
Baby when the midnight comes around
I still hear your footsteps on the ground
[chorus]
Oh, come back to me" \
  --seconds 20 --steps 32 --seed 7 --out song.wav
```

**[Listen to that command's output.](https://huggingface.co/infosave/MiniMax-Music-3-cmf/resolve/main/samples/soul_20s_32steps.wav)**
One binary, one file, no Python.

The lyrics are not English-only — the same command
[in Russian](https://huggingface.co/infosave/MiniMax-Music-3-cmf/resolve/main/samples/russian_20s_32steps.wav),
caption and all. And steps buy audible quality: the
[same soul prompt at 16](https://huggingface.co/infosave/MiniMax-Music-3-cmf/resolve/main/samples/soul_16steps.wav)
is where the sibilance stops being distracting, 32 is where it settles.

[MiniMax-Music-3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
generates music with vocals from a caption and lyrics. This is its
[Comfy-Org repackage](https://huggingface.co/Comfy-Org/MiniMax-Music-3)
converted into the [CMF container](https://github.com/infosave2007/cmf):
one memory-mapped file holding the AR stack, the DiT and the vocoder,
read by `cortiq`, a Rust binary with no ML framework underneath.

| | source | here |
|---|---|---|
| AR stack (Qwen3-8B + RVQ depth decoder) | 15.56 GiB bf16 | — |
| flow-matching DiT | 4.58 GiB fp16 | — |
| DAV vocoder | 207 MiB | — |
| **total** | **20.34 GiB, three files** | **5.55 GiB, one file** |

915 tensors, `cortiq verify` clean. The tokenizer travels inside it.

## What is in the file

| stack | packed as | status |
|---|---|---|
| **AR** — Qwen3-8B backbone (36 layers, 32 q heads / 8 kv of 128), three embedding tables, pruned audio head, 4-layer RVQ depth decoder | `q4tp`, norms exact | **implemented, tested** |
| **DiT** — 36 layers, 32×64 heads, GEGLU 8192, flow matching | `q4tp`, norms and both 1×1 convs exact | **implemented, tested** |
| **DAV vocoder** — ×8 ×8 ×4 ×2 to 44.1 kHz stereo | **exact throughout** | **implemented, tested** |

The vocoder is not quantized on purpose. It buys tens of megabytes and
costs audible hiss — a lesson the
[H3 conversion](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf)
already paid for once.

## What the conversion had to get right

None of these are visible in a tensor name, and each is wrong in a way
that still produces plausible output. They came from reading ComfyUI's
`comfy/ldm/minimax_music/`, and several of them corrected a guess made
from the weights alone.

**The vocoder's residual dilations are 1, 3, 9** — not BigVGAN's usual
1, 3, 5.

**Its Snake activation reads α verbatim.** The H3 vocoder in the same
engine keeps α and β in log scale and exponentiates on load; feeding one
model's parameters to the other's loader silently raises the activation
to an exponent. Music-3 gets its own plain `Snake` rather than a shared
loader.

**The 128 latent channels are a stereo pair of 64.** `decode()` folds
`[b,128,t]` to `[b·2,64,t]` and unfolds to `[b,2,-1]`. The vocoder
config's `latent_channels: 128` against `dec_in_proj [1024,64,1]` reads
like evidence of a second VAE sitting between them. There is none — it
is left and right.

**The DiT's input is `[x | zeros_like(x) | condition]` on the channel
axis**, 128 + 128 + 2048 = 2304, which is what `preprocess_conv`'s width
was saying. The middle plane is a slot the reference leaves empty, not
padding.

**Both 1×1 convs are residual** (`conv(x) + x`), the **timestep
embedding is prepended as a token** — carried through all 36 blocks,
dropped before `project_out`, and shifting every latent frame's rotary
position by one — and **the output is negated**. A sampler stepping the
wrong way still moves and still decodes.

**RoPE covers only the first 32 of each head's 64 dims**, split-half:
the pair for `i < 16` is `x[i], x[i+16]`.

**The two fused projections are not the same shape.** Every projection
in the AR checkpoint is fused where the engine wants them split, and the
LM's `qkv_proj` is a GQA fuse — 32 query heads and 8 key/value heads of
128, hence 6144 rows rather than 3×4096 — while the depth decoder's is
12288 = 3×4096 with no grouping. Splitting the first as if it were the
second gives a model that loads, runs and is wrong, so the packer
DERIVES the key/value head count from the row count instead of assuming
it.

## What is verified, and how

Not "it produced something". Each stack is gated on quantities the
reference fixes exactly, so a stack that has quietly lost an input fails
rather than degrades.

**Vocoder** — 12 latent frames must decode to exactly 6144 samples per
side (512 per frame), inside a `tanh` range, above silence, and **left
must differ from right**. That last one is what catches the stereo fold
being read as one wide latent, which decodes noise at half the length.
Measured: 6144 samples/side, rms 0.065, L−R max 0.177.

**DiT** — the velocity must be `[128, n]`, finite, and must *respond*:
zeroing the condition has to change it (catches a wrong concatenation)
and moving the timestep has to change it (catches a dropped token).
Measured: rms 0.563, d/dcond 0.254, d/dt 2.445.

**Depth decoder** — it must be CAUSAL, since that is the only thing its
attention mask does and losing it lets a codebook level attend to its
own answer while the model keeps sampling plausible codes. Measured: 8
codebooks, head spread 6.188.

**The chain** — noise → sampler → vocoder has to land on the sample
count all three agree about, `frames × 512 × 2`, because the σ walk, the
DiT's timestep convention and the vocoder's hop all feed it.

And then the ear, which is the only judge of the last mile. The sample
above is what came out; by numbers it is 3382 zero crossings a second
(music sits in the low thousands, white noise above ten), band energy
0.42/0.25/0.25/0.08 from low to high, and an envelope with real onsets
— sd/mean 0.71, 124 of 149 windows active.

## How it generates

The AR stack does not encode the prompt, it **generates** the
conditioning. A Qwen3-8B backbone is prefilled at batch two — the words,
and a copy whose middle is replaced by `<|audio_cfg|>` — then sampled one
audio frame at a time at 25 fps: `c0` from the pruned head under
classifier-free guidance at 1.5, with the top-k mask taken from the
CONDITIONED logits, then seven more codebooks through the depth decoder,
each fed back through its own embedding table. The eight hidden states
of that frame are what the DiT sees, softmax-mixed by
`cond_layer_logits`.

Then an ordinary Euler flow walk over the latent — σ from 1 to 0, the
DiT asked at `1 − σ`, windowed 689 frames at a time with a 344 hop and
the overlap averaged, exactly as the reference does it — and the vocoder
turns each latent frame into 512 stereo samples.

Two places where this deliberately is not the reference, both marked in
the source:

- **The top-k sampler is a plain xorshift, not torch's seeded
  `Generator`.** Reproducing `torch.multinomial` bit-for-bit is its own
  project, and nothing here needs one seed to mean the same song across
  implementations — only that a seed means one song in this one.
- **The lyrics normaliser skips the reference's markdown scrubbing.**
  That step only ever removes characters a caption should not carry.

### What it costs

Every render prints where its time went. On an RTX 3090, 4 s at 4
steps:

| stage | GPU | CPU only |
|---|---|---|
| AR | 52.3 s | 44.2 s |
| denoise | 32.9 s | 36.2 s |
| vocoder | **18.2 s** | 31.3 s |
| total | **103.5 s** | 111.8 s |

The vocoder was 54.0 s until its convolutions stopped shipping their
column matrix across the bus — the host built it, transposed it into a
second buffer of the same size and uploaded that, up to 2.37 GB for a
20-second song, when the input it expands from is `k` times smaller.
It is expanded on the card now.

Two things that table will not tell you. The CPU column is a 256-core
EPYC, so an ordinary machine's fallback is far slower than this and the
device gap far wider. And the shape that matters for real songs is not
this one: attention over latent frames is quadratic, so a 20-second
render at 32 steps spends about **80% of its time in the denoise**,
around 32-40 s a step. `CMF_MUSIC3_PROF=1` splits a step four ways if
you want to see it.

Earlier, on an Apple M4, 5 s at 8 steps took 206 s: AR 0.45 s/frame,
denoise 6.5 s/step over 430 latent frames, vocoder 19 s — and the
vocoder was 75 s before it was handed the thread pool.

## Running it

```bash
cargo install cortiq-cli          # 0.5.74+
hf download infosave/MiniMax-Music-3-cmf minimax-music3-q4tp.cmf --local-dir .
cortiq verify minimax-music3-q4tp.cmf     # → ✓ all tensor hashes match
cortiq music minimax-music3-q4tp.cmf --prompt "..." --lyrics "..." \
  --seconds 20 --steps 32 --seed 7 --out song.wav
```

`--seconds` is a ceiling: the model can stop earlier. Same seed, same
prompt, same song. Repacking from the original sources:

```bash
cortiq animate-pack \
  --music-te  minimax_music3_text_encoder_pruned_bf16.safetensors \
  --music-dit minimax_music3_dit_fp16.safetensors \
  --music-vae minimax_music3_dav.safetensors \
  --quant q4tp --out minimax-music3-q4tp.cmf
```

## Provenance

Weights derive from MiniMax's Music-3 release as repackaged by
Comfy-Org, under their own licence. The conversion conventions were read
from ComfyUI's `comfy/ldm/minimax_music/` and MiniMax's own configs. The
CMF container and the cortiq runtime are Apache-2.0.
