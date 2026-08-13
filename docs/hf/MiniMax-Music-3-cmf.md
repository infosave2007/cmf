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

> **Read this first.** This is a **weight conversion, not a runnable
> generator yet.** All three stacks are packed and every tensor hash
> verifies; two of the three — the flow-matching DiT and the DAV vocoder
> — are implemented in `cortiq` and pass their tests out of this file.
> The autoregressive driver that produces the conditioning is **not
> written**, so `cortiq` cannot yet turn a prompt into music from it.
> Published because the conversion itself is finished, checkable and
> useful; not published as a working text-to-music model. What is left
> is spelled out at the bottom.

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
| **AR** — Qwen3-8B backbone (36 layers, 32 q heads / 8 kv of 128), three embedding tables, pruned audio head, 4-layer RVQ depth decoder | `q4tp`, norms exact | packed, **driver not written** |
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

Both run out of this single file with the same numbers as their
standalone packs.

## What is left

The remaining stack is the autoregressive driver, and it is not a
forward pass. ComfyUI's node calls
`clip.tokenize(caption, lyrics, seed, cfg_scale, top_k)`, so the DiT's
conditioning is **generated**: audio tokens sampled frame by frame at 25
frames/s with classifier-free guidance at 1.5 over a batch of two and
top-k 50, running the 8B backbone once per frame plus seven passes of
the depth decoder, and the conditioning the DiT sees is the eight RVQ
CODEBOOK levels of that generation — `c0`'s hidden state concatenated
with the depth decoder's seven, 8 x 4096 per frame — mixed by a learned
`cond_layer_logits`. (Not eight transformer layers, which is what the
name and the condition encoder's `num_condition_layers: 8` both suggest
until you read the loop.)

For two minutes of music that is 3000 frames — a full LLM sampling loop
with a KV cache, not something to bolt on quickly.

Then the sampler: `FlowMatchEulerDiscreteScheduler` with
`invert_sigmas`, `shift 1.0`, which is a named diffusers class and can
be implemented exactly rather than inferred.

The weights for all of it are in this file and verified. The driver is
the work.

## Using what does work today

```bash
cargo install cortiq-cli          # 0.5.73+
hf download infosave/MiniMax-Music-3-cmf minimax-music3-q4tp.cmf --local-dir .
cortiq verify minimax-music3-q4tp.cmf     # → ✓ all tensor hashes match
```

The DiT and the vocoder are `cortiq_engine::music3::Music3Dit` and
`cortiq_engine::audiovae::Music3Dav`; both take this file. Repacking
from the original sources:

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
