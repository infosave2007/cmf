# Block-diffusion models: Clover-35B-A3B-Preview

[DiffuSpace Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview)
is a diffusion language model on the Qwen3.6-35B-A3B backbone (40 layers:
30 Gated DeltaNet + 10 gated full attention, 256 experts top-8 with a shared
expert, ~3B active parameters). It writes its answer in 32-token blocks:
each block starts as mask tokens and is resolved by repeated *denoising
passes* over the whole block; a resolved block is committed causally and the
next one starts.

## Convert

```bash
cortiq convert --model Diffuspace/Clover-35B-A3B-Preview --quant q4tp \
  --sampling temperature=0,top_p=1,top_k=0,min_p=0,repetition_penalty=1 \
  --output clover-35b-a3b-preview-q4tp.cmf
```

`model_type: clover_moe` converts as the `qwen3_5_moe_text` trunk plus a
`block_diffusion` record in the header, which sets the required feature bit
15 (`BLOCK_DIFFUSION`) — a reader without block-diffusion support refuses the
file instead of decoding it causally. The record holds:

| field | source | Clover |
|---|---|---|
| `algorithm`, `block_size`, `threshold` | `denoising_config`, `block_size` | `low_confidence`, 32, 0.95 |
| `mask_token_id` | `config.json` | 151669 (`<\|mask\|>`) |
| `coreset_enabled`, `coreset_top_p` | `moe_block_coreset_*` | true, 0.9 |
| presence / frequency / repetition | `generation_config.json` | 0.9 / 0 / 1.05 |
| stop-id ramp | `dllm_length_penalty*` | +10 from response token 8192 over 8192 |
| `</think>` ramp | `dllm_think_end_*` | +10 from 4096 over 4096 |
| `stop_token_ids` | EOS ∪ `stop_token_ids` ∪ pad | 248044, 248046 |

The preview tensors (`linear_attn.conv_right_weight`, `bd_softmax_logit_scale`,
`bd_softmax_history_bias`, `bd_softmax_history_c`) are stored in f32.

Higher precision at the same size: GPTQ the experts with `quantize-moe`
(Qwen3.5/3.6-MoE capture on the batched graph, original safetensors as the
full-precision source):

```bash
cortiq quantize-moe clover-35b-a3b-preview-q4tp.cmf --hf Diffuspace/Clover-35B-A3B-Preview \
  --calib wiki.train.txt=0.4 --calib chat.txt=0.4 --calib code.txt=0.2 \
  --tokens 262144 --codec q4tp --output clover-35b-a3b-preview-q4tp-v2.cmf
```

## Generation

A Clover file holds one model with two read paths: the causal trunk and the
block preview. `CMF_CLOVER_MODE` picks how a request uses them:

| mode | how | when |
|---|---|---|
| `ar` | the ordinary causal decode, with the release's response penalties and ramps | the default; the fastest everywhere it was measured. On Apple silicon the prompt runs on the Metal executor and the rows on the CPU |
| `draft` | self-speculation, the vendor's `draft-self`: one preview of `[pending token, masks]` proposes a block, one causal forward verifies it, the accepted prefix plus one token go out and the state rewinds to them | wgpu whole-stack graph only |
| `denoise` | LowConfidence block denoising, the release's `generate` | the original algorithm |

Self-speculation knobs: `CMF_CLOVER_DRAFT` (rows of a round, a divisor of 32,
default 16), `CMF_CLOVER_ACCEPT=exact` (keep only drafts equal to the verify's
penalized argmax, the causal decode's text), `CMF_CLOVER_ACCEPT_RATIO` and
`CMF_CLOVER_ACCEPT_DELTA` (the threshold rule `p ≥ min(ratio, delta·e^-H)`,
defaults 0.9 and 0.6 as in the vendor's llama.cpp example).

In every mode the penalties count response tokens only and the mask token
never comes out.

The rest of this section describes the denoising path:

* **Preview** (one denoising pass, the committed state read only): full
  attention sees the prefix and every row of the block; a GDN layer runs the
  release's block operator — causal conv taps over [conv state | block] plus
  right taps over the next three rows of the block, then per value head a
  softmax over the block's l2-normalized keys (logits scaled per head, no β)
  and a history slot whose value is the clean state's read-out decayed by
  the row's gate; the MoE keeps, for the whole block, the smallest set of
  experts holding 90% of the block's routed probability (on Clover ~15 of
  256 experts a layer) and every row routes inside it.
* **LowConfidence**: after each pass every masked row whose greedy token has
  probability above the threshold is fixed — at least the single most
  confident one — until the block is resolved. Presence, frequency and
  repetition penalties count committed response tokens only.
* **Commit**: the resolved block through the ordinary causal prefill.

Knobs: `CMF_CLOVER_THRESHOLD` (lower fixes more rows a pass: faster, less
careful), `CMF_CLOVER_BLOCK` (a divisor of 32), `CMF_CLOVER_GRAPH=0` (host
path), `CMF_CLOVER_DEVICE_PICK=0` (logits back to the host),
`CMF_CLOVER_DUMP=PATH` (first-pass top-10 per row, for parity checks with
`scripts/clover/clover_ref.py` / `clover_cmp.py`). Temperature sampling works
(`--temperature`, `--top-k`, `--top-p`); the confidence is always the T=1
probability, as in the release.

`cortiq run` prints the denoising statistics after the answer: blocks,
passes, tokens per pass, the rate after the prompt and the time of a pass.

## Device paths

* **Vulkan / DX12 (wgpu)**: the prompt, every preview and every commit are one
  batched-graph submission each. Preview kernels: `gpu_wgpu/clover.rs` (GDN
  preview conv and mixer, two-phase coreset, the expert tile list for the
  grouped MoE kernels, the penalties and greedy pick). A stack that does not
  fit the card runs its leading layers on it and the rest on the host
  (`device layers` in the statistics).
* **Apple silicon (Metal)**: `gpu_metal/clover.rs` takes the prompt and, in
  `denoise`, the passes and commits; the causal rows run on the CPU (below).
  `CMF_CLOVER_METAL=0` leaves the request to the general paths, whose token
  graph takes the model where the file fits Metal's working set (not measured
  here). The general Metal graphs
  cannot take the model (the rows graph has no MoE arm, and binding the
  19 GB mapping exceeds the M4's 17.8 GB working set), so a pass is one
  command buffer per layer: the mixer, the post norm and the router, ONE
  wait, the block coreset and every row's top-k on the host with the CPU
  path's own functions, then the expert panels with the next mixer. Every
  tensor but the routed experts is bound through a no-copy view of its
  page-rounded span; a routed expert gets its views when routed, kept in an
  LRU bounded in bytes (`CMF_CLOVER_METAL_CACHE_MB`, 6 GB) — nothing is
  copied, the page cache is the only copy (owned copies doubled the
  footprint and paged the file in from disk every pass: 1.3 s against
  0.37 s). Kernels: `clover_msl.metal` (preview conv, preview mixer with
  the block's keys staged in threadgroup memory, preview attend over the
  prefix and the whole block), `moe_q8`'s blob GEMM with half X tiles for
  the q8_2f projections and the head (`CMF_CLOVER_Q8=f32` for f32 tiles),
  `q4tp_mul_mm_n8` for the expert panels (8 rows a dispatch beat the 32-row
  tile GEMM and 16/32-row variants). The prompt and the block commits run
  the causal step on the same executor (the rows graph's GDN kernels, the
  delta rule writing S, chunks of 128 rows) and hand the states and K/V rows
  back to the host owners; `cortiq ppl` with `CMF_GPU=1` scores through it.
  `CMF_CLOVER_METAL=0` keeps a request on the CPU; `CMF_CLOVER_METAL_PROF=1`
  books the device time per stage; `CMF_CLOVER_METAL_CHECK=1` commits the
  prompt on both and compares every layer.
* **CPU**: the host path everywhere.

## Measured: the three modes (cortiq 0.8.52)

RTX 5090 (32 GB), Vulkan, `q4tp-v2`, tokens per second after the prompt over
512 tokens:

| mode | prose | code | cost |
|---|---:|---:|---|
| `ar` | 120 | 121 | 8.3 ms a token (5.2 ms on the device) |
| `draft`, 4 rows | 61 | 78 | 46–51 ms a round, 2.8 / 3.9 tokens |
| `draft`, 8 rows | 54 | 99 | 65–74 ms a round, 3.5 / 7.3 tokens |
| `draft`, 16 rows | 42 | 104 | 80–98 ms a round, 3.4 / 10.1 tokens |
| `draft`, 32 rows | 20 | 94 | 106–147 ms a round, 2.9 / 10 tokens |
| `denoise` | 18–26 | 63–95 | 35–59 ms a pass, 1.1 / 2.6 tokens |

The causal decode wins because the card reads one row in 8 ms, while a
self-speculative round costs a whole preview plus a verify: it would need ten
accepted tokens a round just to draw level, and free text gives three. A
denoising pass fixes about one token on prose. `bench --core` puts the causal
decode at 144 tok/s near the start of the context; a 2048-token prompt reads
at 1143 tok/s and decode there runs at 116.

Smaller budgets (`CMF_GPU_VRAM_MB`), causal against denoise, prose and code:
16 GB 54 / 53 against 5.6 / 14.3; 12 GB 29 / 29. The CPU (15 Xeon cores):
13.8 / 10.9 against 0.7 / 1.3. `q8_2f` on the 32 GB card, whose stack does
not fit: 49 / 46, a 2048-token prompt at 60.

The causal text equals `draft` with `CMF_CLOVER_ACCEPT=exact` byte for byte on
the prose prompt. The code prompt splits after about eight tokens: the verify
runs int8 activations, so a near-tie can resolve differently.

Apple M4 (24 GB, 10-core GPU), `q4tp-v2`: 13–18 tok/s on prose and 14–18 on
code, against 3.2 and 6.5–9 for `denoise`; a 2048-token prompt in 11.8 s
(~170 tok/s, the CPU alone 66), decode there 16.2. Why the rows stay on the
CPU: a row takes 55 ms there and 160 ms on the executor (40 host round trips a
token), and the token graph cannot take the file: 19.4 GB against a 17.8 GB
working set ends in `Insufficient Memory`, and the MoE pager's items do not
fit a page in this file's layout.

## Measured: denoising (cortiq 0.8.40 / 0.8.41)

RTX 5090 (32 GB), whole stack and head on the card, 18.8 GB of VRAM. A pass
is one 32-row denoising forward: 34–36 ms. Tokens per second after the
prompt, q4tp:

| text | threshold 0.95 (default) | 0.9 | 0.8 |
|---|---:|---:|---:|
| prose (0.9–1.1 tokens a pass) | 26 | 22–27 | 25–30 |
| code (3.7–6.6 tokens a pass) | 80–95 | 94–110 | 124–150 |

Apple M4 (24 GB, 10-core GPU), Metal, q4tp-v2: a pass 0.3 s on prose, 0.4–0.5 s
on code (more experts a block, some read from the SSD); a 32-row block
commit 0.35–0.75 s; the prompt ~57 tok/s. Tokens per second after the
prompt:

| text | threshold 0.95 (default) | 0.8 |
|---|---:|---:|
| prose | 3.2 | 3.7 |
| code | 6.5–9 | 9.7 |

The CPU path on the same Mac: a pass 1.13 s, prose 0.8, code 2.9 tok/s.
Device time of a prose pass (`CMF_CLOVER_METAL_PROF=1`): GDN mixers ~45 %
(the q8_2f projections ~1.4 ms a layer), expert panels ~40 %, attention and
the head the rest; 40 host round trips (routing ~12 ms a pass). Commit-path
perplexity (README, 1024 tokens): Metal 12.407, CPU 12.446.

Parity with the PyTorch release (bf16, transformers 5.21, CPU) on three chat
prompts: identical prompt ids; first-pass top-1 agreement q8_2f 32/32 · 30/32,
q4tp 31 · 32 · 30 /32, q4tp-v2 31 · 32 · 30 /32 — the disagreements are
near-ties. wikitext-2 perplexity of the causal path (12 × 512): q8_2f 15.86,
q4tp 18.22, q4tp-v2 16.80.
