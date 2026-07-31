# Qwen3.6-35B-A3B on cortiq — the ready file and every flag that matters

This page covers **Qwen3.6-35B-A3B** (40 layers, 10 full-attention +
30 GatedDeltaNet linear, 256 routed experts top-8 plus a shared expert,
vocab 248 320) as a single `q4tp` `.cmf` — including the model's **MTP
head** (35.51B parameters total), which the converter keeps as of 0.5.42.

Verified end-to-end on:

| | hardware | decode (steady) | prompt ingest |
|---|---|---|---|
| Vulkan | RTX 5090 32 GB | **64.5 tok/s** | — |
| Vulkan | RTX PRO 6000 Blackwell 96 GB | **67.8 tok/s** | **99 tok/s** batched |
| CPU | 48-core (same box) | 19.2 tok/s | 33 tok/s |
| CPU | Apple M4, 24 GB | 18.2 tok/s (`CMF_THREADS=8`) | — |

The RTX 5090 row predates 0.5.42's final q4tp matvec (vec4 loads +
register-blocked row pairs, +19% on the RTX PRO 6000) — expect higher.

Output is token-for-token identical to the CPU path at short context on
every step above. On long prompts (hundreds of positions) greedy decoding
can pick a different-but-coherent continuation near probability ties —
accumulated float ordering, common to every GPU offload, not a defect of
this file.

## 1. Get the file

The converted file is published and size-verified:

```sh
wget https://huggingface.co/infosave/Qwen3.6-35B-A3Bcmf/resolve/main/qwen36-35b-a3b-q4tp.cmf
```

Or convert yourself — streaming, straight from the hub, no 72 GB staging
(peak disk = the 18.7 GB output):

```sh
cortiq convert --model Qwen/Qwen3.6-35B-A3B --quant q4tp --output qwen36-35b-a3b-q4tp.cmf
# ~25 min on a datacenter link; needs nothing but the output space
```

**Version gate: everything on this page needs cortiq 0.5.42+.** Earlier
runtimes (0.5.41 and below, the current `cargo install` / GitHub release)
**cannot load this file at all** — their loader only knows dense-FFN MTP
heads and errors out on this model's MoE head — and their converter drops
`mtp.*` tensors, so a 0.5.41 conversion produces a different, smaller
file. Their whole-token graph also declines the q4tp layout, so none of
the GPU numbers above apply to them.

## 2. Run

```sh
# chat / one-shot
CMF_GPU=1 CMF_GPU_VRAM_MB=26000 cortiq run qwen36-35b-a3b-q4tp.cmf

# OpenAI-compatible server (Cline/Roo-style block-array `content` accepted)
CMF_GPU=1 CMF_GPU_VRAM_MB=26000 cortiq serve qwen36-35b-a3b-q4tp.cmf
```

**`CMF_GPU_VRAM_MB` is not optional on discrete cards.** The default
weight budget is 8 GB; this model's experts want **16.9 GB resident**.
Under budget the whole-token graph declines (loudly, since 0.5.42) and
every expert runs on the CPU. 26000 fits a 32 GB card with KV mirrors;
give it more on bigger cards.

## 3. Fast prompt ingest (batched prefill)

```sh
CMF_BATCH_K=32 CMF_MTP=0 CMF_GPU=1 CMF_GPU_VRAM_MB=26000 cortiq run ...
```

Chunks of 32 positions go through the graph in one submit — token-axis
MoE kernels route each token individually while everything around them
stays batched. Measured 99 tok/s ingest against 33 per-position. Two
gates to know about: batched prefill needs `CMF_MTP=0` while the file
carries an MTP head (the speculative warm-up wants per-position hidden
states), and it refuses while `--o1` is active (o1's calibration needs
the exact CPU prefill).

## 4. Long context: `--o1` now rides the GPU

O(1) Nyström attention replaces the KV-cache attention of the 10 full
layers with a sealed landmark skeleton — and since 0.5.42 it runs inside
the wgpu graph (`CMF_O1_GPU=1`) instead of dragging the whole model to
the CPU:

```sh
CMF_O1_GPU=1 CMF_GPU=1 CMF_GPU_VRAM_MB=26000 cortiq run ... --o1 all
```

| ctx | exact attention (graph) | o1 (graph) |
|---|---|---|
| 4 096 | 54.2 tok/s | ~53 |
| 16 384 | 37.8 tok/s | **53.8 — flat** |

The o1 path is depth-invariant by construction; exact attention keeps
falling past 16K. The price is one-time: the prompt must run through
exact CPU attention to collect the calibration trace (O(n²) — ~30 min at
16K on 48 cores), and o1 is an approximation — no bit-parity claim even
CPU-vs-CPU.

## 5. The MTP head

The file carries Qwen3.6's multi-token-prediction head (one full MoE
layer, +444 MB). The loader reads it (`MTP: 1 block(s)`), CPU speculative
decode uses it; on the GPU graph speculative decode is not wired yet —
measured economics on launch-bound decode put the ceiling at ~+13%, so
it is documented rather than chased. `CMF_MTP=0` disables the head
entirely.

## 6. If something is slow, read the refusals

Every silent fallback found while bringing this model up now says why:
the graph logs `ACTIVE`/`declined` with a reason (`RUST_LOG=info`), the
MoE block names the check that sent experts to the CPU, and the VRAM
budget prints the numbers it compared. If you see 15 tok/s on a big
card, the log will contain the reason on its first token.

## 7. The dense sibling: Qwen3.6-27B

[infosave/Qwen3.6-27Bcmf](https://huggingface.co/infosave/Qwen3.6-27Bcmf) —
one 14.3 GB q4tp file, MTP head kept, same flags. Dense hybrid: 64 layers
(16 full attention + 48 GDN linear), hidden 5120, FFN 17408.

| | hardware | decode | prompt ingest |
|---|---|---|---|
| Vulkan | RTX PRO 6000 Blackwell | **32.3 tok/s** | ~10 tok/s |
| CPU | 48-core (~55 GB/s RAM) | 2.1 tok/s | 10.4 tok/s |

Two dense-specific notes. CPU decode reads all 14.3 GB per token —
memory-bandwidth bound, budget `RAM GB/s / 14.3` tok/s. And `CMF_BATCH_K`
buys nothing here yet: dense prefill is limited by the batched GEMM kernel
(a known follow-up), not by dispatch structure, so batched and per-position
ingest run at the same speed.

`CMF_GPU_VRAM_MB=20000` fits the weights with room for KV; scale up on
bigger cards.
