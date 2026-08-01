---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- cmf
- cortiq
- moe
- 4-bit
- 2-bit
---

# DeepSeek-V4-Flash-0731 — CMF

[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
— 304B parameters, 43 layers, 256 routed experts top-6 plus a shared one —
in the [CMF container](https://github.com/infosave2007/cmf), decoding on
`cortiq`. Two variants, both working:

| variant | size | expert planes | folder |
|---|---|---|---|
| `q4tp` | 158 GB | 4-bit throughout | `parts-q4tp/` |
| `q2tp` | 112 GB | 2-bit gate/up, 4-bit down | `parts-q2tp-v2/` |

**Requires cortiq 0.5.44 or newer.** Earlier builds rotate positions into
the wrong basis and the model repeats itself, cannot count, and drifts past
a few dozen tokens. The weights are unaffected — that was a runtime bug, and
nothing here was re-converted to fix it.

## Measured

Teacher-forced perplexity over ordinary English prose (212 words), greedy
decode, 48 CPU cores. Lower is better; falling with length is the sign that
the model is using its context.

| tokens scored | 16 | 32 | 64 | 128 | 200 |
|---|---|---|---|---|---|
| `q4tp` | 14.6 | 12.3 | 6.8 | 4.9 | **5.1** |
| `q2tp` | 15.5 | 20.1 | 9.9 | 6.4 | **6.5** |

Two bits on the expert gate/up planes cost about 1.3× the perplexity for
0.71× the file.

## What they answer

Greedy, in the prompt format below. `[stop]` means the model ended its own
turn.

**`q4tp`:**

```
What is 2 + 2? Answer with just the number.
→ 4                                                                  [stop]

What is the capital of France? Answer in one sentence.
→ The capital of France is Paris, a city renowned for its history,
  culture, and iconic landmarks like the Eiffel Tower.                [stop]

If a train leaves at 3pm and takes 2 hours, when does it arrive?
→ The train would arrive at **5:00 PM**. Since it leaves at 3:00 PM and
  the trip lasts 2 hours, you simply add 2 hours to the departure time.
                                                                      [stop]
Write one sentence about the sea.
→ The sea is a restless, breathing expanse of deep blue that holds both
  our gentlest dreams and our most ancient fears within its endless,
  shifting embrace.                                                   [stop]
```

**`q2tp`** reasons and writes, and loses the arithmetic:

```
If a train leaves at 3pm and takes 2 hours, when does it arrive?
→ Adding that to the departure time: 3:00 PM + 2 hours = **5:00 PM**

Write one sentence about the sea.
→ The sea, a vast and breathing expanse of indigo, cradles the world's
  forgotten secrets beneath its ever-shifting surface.                [stop]

What is 2 + 2? Answer with just the number.
→ 2                                                     ← wrong, q4tp says 4
```

## Prompting

It is a chat model with its own message format, it needs the BOS token, and
ordinary chat closes the reasoning block IN THE PROMPT:

```
<｜begin▁of▁sentence｜><｜User｜>your question<｜Assistant｜></think>
```

The trailing `</think>` is not a typo — the reference does
`prompt += thinking_end_token if thinking_mode != "thinking"`. Leave it out
and the model opens its reply by closing a block nobody opened (`2</think>2`
instead of `2`). For reasoning mode, end with `<think>` instead. Without the
BOS token the output is high-frequency noise: that is out-of-distribution
input, not a broken model.

## Getting a file

Each variant is past what the Hub takes as a single upload, so it ships in
slices that concatenate back into it byte for byte:

```bash
huggingface-cli download infosave/DeepSeek-V4-Flash-0731-cmf \
  --include 'parts-q4tp/part_*' --local-dir .
cat parts-q4tp/part_* > dsv4-flash-q4tp.cmf

cortiq run dsv4-flash-q4tp.cmf \
  --prompt $'<｜begin▁of▁sentence｜><｜User｜>What is 2 + 2? Answer with just the number.<｜Assistant｜></think>' \
  --max-tokens 8
```

Swap `parts-q4tp` for `parts-q2tp-v2` to get the 2-bit file. `cortiq info`
prints the architecture; `cortiq verify` checks every tensor hash against the
directory.

## What it costs to run

The file is memory-mapped, so plan on RAM at least its size or every token
touches non-resident pages.

**On CPU** (48 cores, `q2tp`): 2.7 tok/s on a long generation.

**On one GPU**, since 0.5.44: **12.7 tok/s**, with perplexity 5.211 either
way — the same number the CPU produces, to the digit.

```bash
CMF_GPU=wgpu CMF_DSV4_GPU_ATTN=1 CMF_DSV4_GPU_MOE2=1 CMF_GPU_VRAM_MB=94000 \
  cortiq run dsv4-flash-q2tp.cmf --prompt "..." --max-tokens 512
```

Measured on an RTX PRO 6000 Blackwell (96 GB) against the `q2tp` file.
Three things are worth knowing before you read a number off your own run:

- **The experts are uploaded to VRAM on first touch** — about 94 GB, a
  one-time cost per process. It shows up as a slow start: 512 tokens
  measures 10.4 tok/s, a long run 12.7. A server pays it once.
- **`CMF_GPU_VRAM_MB` matters more than it looks.** At 90000 roughly five
  layers do not fit and run on the CPU entirely, which costs about 20 ms a
  token; 94000 is the sweet spot on a 96 GB card. Too high and the driver
  refuses the allocation outright.
- **Do not use `CMF_MOE_MASK` with this model.** Restricting routing to the
  experts holding 95% of the mass looks safe and is not: generation degrades
  into fragments within a few dozen tokens. Measured with the experts on the
  CPU, so it is the restriction and not a kernel.

Where the 76 ms of a token goes: attention 42 (25 of that waiting on
submission barriers), experts 23, hyper-connections 4, head 2. One
submission per token would remove most of the first figure; the compressor
and the indexer still run on the host, which is what prevents it.

## What the conversion did

- **Skeleton** — FP8 E4M3 with one E8M0 scale per 128×128 tile → f32 →
  `q4tp`. The E4M3 decoder was checked against `torch.float8_e4m3fn` on this
  model's own weights and matched bit for bit.
- **Experts** — the checkpoint's `expert_dtype: fp4` is OCP MXFP4 (two E2M1
  values per byte, one E8M0 scale per 32), read directly, then requantized.
  Both layouts put a predicted per-row scale ladder under the weights, and
  the group scale and the ladder span are each chosen by reconstruction
  error rather than by the group maximum — worth 40% of the 2-bit error at
  identical file size.
- **No expert defrag.** The hash table reaches all 256 experts within the
  first 8 000 vocabulary ids — measured — so dropping unreachable experts
  saves nothing here.
- Integrity: 36 599 tensors per file, every hash verified against the
  directory.

## On parity

Established, not assumed. The upstream forward cannot be run outside a
CUDA box — its attention is a tilelang kernel — so it was transcribed into
NumPy (`tools/dsv4_ref.py` in the repository) and diffed against the port
layer by layer on a toy checkpoint carrying the release's real tensor names.
Worst divergence over a ten-token run: 1.6e-3, against an f16 quantization
floor of 2.2e-4 on the embedding alone.

That harness is what found the rotation bug above. Every check that passes
at position 0 passes for either pairing convention, so short prompts and
single-vector unit tests were blind to it.

The GPU path is held to the same standard: every kernel is diffed against
the CPU on its own (rotation, sparse attention with the sink, the compressor
pool, the indexer, the router, the grouped output projection), then the
assembled frames are diffed end to end. Compare with `CMF_SDOT=0` on both
sides — without it the CPU arm quantizes activations to int8 and you measure
that approximation instead of the device.

## Provenance

Weights derive from DeepSeek's release and remain under its licence. The CMF
container and the cortiq runtime are Apache-2.0 (see the repository's LICENSE
and PATENTS.md).
