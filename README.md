Русский: [README.ru.md](README.ru.md) · 中文: [README.zh.md](README.zh.md)

# CMF — Cortiq Model Format

**A single-file LLM format whose attention memory stops growing with the context.**

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![downloads](https://img.shields.io/crates/d/cortiq-cli.svg)](https://crates.io/crates/cortiq-cli)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/infosave2007/cmf/blob/master/LICENSE)

A `.cmf` file carries the weights, the tokenizer and the chat template together,
checks its own integrity, and memory-maps straight off disk. The runtime is a
small Rust core with no ML framework under it — no torch, no BLAS, no ONNX, no
CUDA install, no C++ toolchain — running on CPU everywhere, and on GPU via wgpu
(Vulkan / DX12 / Metal) out of the box. Converting a model takes one command
and no Python.

What makes it different: **you can convert a model's attention into a
constant-memory streaming operator with one flag** — no retraining, weights
byte-identical — so a long conversation stops costing more memory than a short
one.

## It renders, too

The same container, the same binary, and no Python at inference for any of it.

| `cortiq animate` — video **and its soundtrack** | `--first-frame` — continue from a picture |
|---|---|
| ![a corgi in a chef hat flipping a pancake](docs/media/corgi.gif) | ![the same clip continued from a still frame](docs/media/keyframe.gif) |

512×288, 39 frames, **four sampling steps**, out of one 23.9 GB file. The audio
is not dubbed on afterwards: it is denoised in the same packed sequence as the
video, on its own flow schedule, so it arrives already in sync. Clips with
sound, and the keyframe that produced the second one, are in the
[model repo](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf/tree/main/samples);
[the section below](#text-to-video-with-the-sound-minimax-h3--turbo-lora) says
how 124.4 GB of reference tree became one file.

| `cortiq imagine` — Lumina-Image 2.0, a 19 GB diffusers tree in a 3.2 GB `.cmf` |
|---|
| <img src="docs/media/fox-512.png" width="380" alt="a red fox in a snowy forest"> |

Try either without installing anything: [convert a model](https://huggingface.co/spaces/infosave/cmf-converter),
[render an image](https://huggingface.co/spaces/infosave/cmf-imagine),
[watch the clips](https://huggingface.co/spaces/infosave/cmf-animate).

> **Note on Hub download counters.** `.cmf` files are not yet in the Hub's
> download-counting registry, so CMF model repos currently show `0` even
> under real traffic. Upstream fix pending:
> [huggingface.js#2354](https://github.com/huggingface/huggingface.js/pull/2354).

## Try it

```sh
# prebuilt binary: github.com/infosave2007/cmf/releases/latest
# or, with a Rust toolchain:
cargo install cortiq-cli

cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --output qwen.cmf
cortiq run qwen.cmf --prompt "What is the capital of France?" --greedy --no-think
```

```console
Loading model: qwen.cmf
Ready: qwen3 | Task: general | Sparsity: 0%

Prompt: What is the capital of France?

The capital of France is **Paris**.
[10 tokens, 40.1 tok/s, finish: stop]
```

**Android**: the `aarch64-linux-android` release binary runs on-device in
[Termux](https://termux.dev) or an `adb shell` — download, `chmod +x cortiq`,
and the same `convert` / `run` / `serve` commands work (CPU path; wgpu
Vulkan rides along and the runtime probe keeps whichever side wins).
0.3.9 ships the mobile package: a blocked SDOT prefill GEMM (×2.1 on the
portable path), batched causal attention off Apple silicon (a
pool-parallel NEON micro-GEMM — pp1024 +77%, pp2048 +82% on the
mobile-sim stack), and a big.LITTLE-aware thread default that keeps
efficiency cores out of the pool.

`convert` pulls the checkpoint from Hugging Face (shards in parallel), quantizes
it and writes one self-contained file — native Rust, no torch, no numpy. Already
have a GGUF? `cortiq import-gguf <file-or-repo-id> --output model.cmf` reads it
natively too.

`run` applies the chat template stored in the file, so this is a real chat turn
and the model stops on its own. Qwen3 is a reasoning model — drop `--no-think`
and it shows its `<think>` reasoning first. `--raw` skips the template entirely
(completion mode). `Task` and `Sparsity` report the skill overlay; with no skill
selected they read `general` / `0%` — more on [skills](#many-specialists-one-backbone) below.

**Does it run your model?** Native conversion today: qwen2 · qwen3 · qwen3.5
(including the fused qwen3_next / AgentWorld layout) · llama · mistral ·
qwen-moe · gemma / gemma-3 (GeGLU, sandwich norms, 512-token sliding window
with dual RoPE) · gemma-4 dense 12B/31B (dual-geometry attention: sliding GQA
+ global MQA with V=K, proportional RoPE, layer scalars, final-logit softcap)
· phi-3 / phi-4 (fused qkv/gate_up splits, longrope served at the native
window) · gemma-2 (attention + final-logit soft-capping, alternating
sliding/global layers) · DeepSeek-R1 distills (qwen2/llama layouts) —
dense, MoE and GatedDeltaNet · DeepSeek-V2 MLA (V2-Lite: latent
attention expanded to MHA, interleaved-rope undone at convert; gated
end-to-end — Paris, ppl 8.8) · gemma-4 MoE 26B-A4B (dual-branch
dense+expert FFN off the raw residual, per-expert scales folded at
convert; gated by scorer/decoder parity — the scorer reproduces the
model's own greedy tokens 40/40) · Kimi Linear 48B-A3B (KDA: delta
rule with per-channel decay, per-projection short convs, sigmoid-gated
output norm; NoPE full-attention MLA; the tiktoken rank table becomes
a standard tokenizer.json at convert — gated end-to-end, streamed
98 GB → 27.7 GB q4t on a MacBook) · MiniCPM3 (compressed-q MLA
(q_a→rms→q_b) gated end-to-end; MiniCPM's scale_emb / scale_depth /
dim_model_base fold into the weights and header, longrope's per-dim
short factors divide inv_freq at load). MXFP4-packed checkpoints (Kimi-K3
experts) decode natively at convert. Gemma-3n E4B (the E-series:
AltUp's four hidden replicas with predict/correct, LAuReL low-rank
residuals, per-layer embeddings, KV sharing across the last 15
layers, gaussian-top-k activation sparsity — gated end-to-end on
E4B-it). Not yet: the Kimi-K3-only extras (residual streams, latent
MoE, MLA output gate — named refusals until the modeling code is
public). Anything else, try `import-gguf` — and if
it refuses, that is a bug worth filing.

## Plug it into what you already use

`cortiq serve` speaks the OpenAI API, so existing clients and SDKs work unchanged
— just point them at it:

```sh
cortiq serve qwen.cmf --port 8080        # + a web dashboard on /
```

```sh
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "cmf",
  "messages": [{"role": "user", "content": "Explain mmap in one sentence."}]
}'
```

`/v1/models`, `/v1/completions` and `/healthz` are there too, and streaming
(`"stream": true`) works. The `model` field is required by the schema but is not
matched against anything — send whatever your client sends.

Scope it honestly before you deploy: **requests are serialized** (one at a time
per model) and **there is no authentication** — this is a local-first server, not
a multi-tenant gateway. Don't expose it to a network you don't trust.

## Why CMF

### Attention that stops growing with the context

Normally every token you add to a conversation adds to the KV cache, forever.
`--o1` replaces a layer's softmax attention with a streaming operator that keeps
a **fixed-size state** instead: a few exact anchor keys, an exact recent window,
and a landmark sketch of everything older, all under one shared softmax
denominator. Conversion is instant and **the weights never change** — the flag
only records a hint in the header.

Measured on **Qwen3.5-4B** (24 GatedDeltaNet + 8 softmax layers; `--o1 all`
converts the 8; 16 query heads / 4 KV heads, head_dim 256; q8_2f). Apple M4, the
machine allowed to cool between runs:

| context | attention memory, `--o1 off` | `--o1 all` | decode, `off` → `all` |
|---:|---:|---:|---:|
| 543 | 141.0 MB | **124.1 MB** | 15.7 → 16.5 tok/s |
| 1055 | 174.5 MB | **124.1 MB** | 15.5 → 16.5 tok/s |
| 4127 | 380.3 MB | **124.1 MB** — 3.1× less | 8.2 → 10.7 tok/s |

**124.1 MB at every context length** — that is the whole point. It breaks down as
a constant recurrent-layer floor plus a fixed **18.8 MB** stand-in for the softmax
layers' KV cache. That KV would otherwise grow at ~64 KiB/token, so the two curves
cross at about **290 tokens**: below that, `--o1` costs you a few MB; above it, it
only saves — 3.1× less at 4k, and ~17× at 32k by extrapolation (the state is
constant, so the ratio keeps climbing; we have benchmarked to 4k — run
`cortiq bench model.cmf --ctx 32768` on your own box).

**What it costs.** The sketch is an approximation, and you pay for it in quality:
perplexity rises **1.13×** on Qwen3.5-4B and **1.30×** on Qwen3-0.6B (28/28 layers
converted) — measured on held-out wikitext through the real streaming kernel on
the harshest region (landmarks sealed from a 256-token prefill, scoring only the
drift rows). The more of the model is softmax attention, the more `--o1` costs: a
hybrid has recurrent layers to carry long-range state, a pure-attention model
makes the sketch do all the work. Treat `--o1` as a memory/quality dial, not a
free win. The cost doesn't grow with context — the state doesn't either. Don't
take our word for any of it; measure your own model:

```sh
cortiq ppl model.cmf --file wiki.txt --o1 all
```

It scores the converted model through the real streaming kernel and prints the
exact-attention baseline over the identical tokens next to it, so the ratio is a
like-for-like measurement rather than a claim.

If that cost is too high for your use case, `cortiq fcd` recovers part of it with
a bounded native training pass — see [O(1) in depth](#o1-in-depth). We haven't
published a clean before/after figure for it yet.

To be clear about the axis: `llama.cpp` is the yardstick we measure against.
One like-for-like run (2026-07-17: Qwen2.5-0.5B-Instruct, Apple Silicon M4,
exact attention for both, native arm64 `llama.cpp` master vs CMF 0.3.9,
interleaved runs from fresh processes, each side at its best measured
thread count — theirs is `-t 6`, ours the default; CMF timed with
`cortiq bench --core`, which matches `llama-bench`'s core contract: no
sampler copy, no per-token confidence pass):

| Apple M4 | `llama.cpp` (q8_0) | CMF (q8) | Δ |
|---|---|---|---|
| tg128, CPU, their best `-t 6` | 165.5 ± 0.3 tok/s | 151–158 tok/s | **−5%** |
| tg128, CPU, their default `-t 4` | 129.4 ± 0.2 tok/s | 151–158 tok/s | **+18%** |
| tg128, their GPU (Metal `-ngl 99`) | 150.9 ± 0.4 tok/s | 151–158 tok/s (CPU) | **CMF CPU ≥ their Metal** |
| pp512, CPU only | 1168 ± 5 tok/s | 1017–1051 tok/s | **−12%** |
| pp512, GPU prefill graph (`CMF_GPU=1`) | 3333–3396 tok/s (Metal) | 2742–3215 tok/s | **2.3–2.8× their CPU; −5% best-vs-best, −18% steady (same-minute interleaved)** |
| pp1024 (`CMF_GPU=1`) | — | 2432 tok/s | flat curve (was 390 in 0.3.3) |
| pp2048 / pp4096 (`CMF_GPU=1`) | — | 2109 / 1651 tok/s | GEMM attention scales with depth |
| Quant quality (PPL vs own f16, 12×512 windows) | near-lossless | +0.38% | matched |
| File size | 644 MB | 479 MB | **−26%** |

Two releases ago this table read −38% tg128 and −67% pp512. What closed it:
prefill rides Apple's AMX through Accelerate GEMM and attends the whole
chunk as GEMMs with a causal masked softmax; decode drops the sampler copy
and per-token confidence pass from the timed loop (`--core`; the default
`bench` still times the full production loop). 0.3.6–0.3.7 add the
**GPU prefill chunk graph**: under `CMF_GPU=1`, whole runs of layers
execute per chunk in one Metal submission — a ggml-layout simdgroup
GEMM over the q8 weights in place, RoPE with the K/V append fused into
the cache mirror, two-GEMM causal attention (scores → masked softmax →
P·V, the same shape the CPU AMX path uses), and the FFN activation
fused into the down-GEMM's operand load. One wait per chunk; the CPU
cache remains the owner of record. Perplexity stays in the half-GEMM
tolerance class (+0.16%). A per-stage GPU profiler ships with it
(`CMF_CHUNK_PROF=1`) — it is what found the attention stage eating 47%
of the chunk while the standalone kernel benchmark was mis-crediting
the GEMMs. The Vulkan/DX12 (wgpu) path carries the same tiled GEMM,
gated by the runtime probe per machine. 0.3.8 also blocks the x86
prefill GEMMs (q8 / q4 / q4_tiled / q4tp / vbit): weight tiles and nibble
unpacks stay in registers across four activation streams — +37% (q8) to
×4.4 (q4_block) on an EPYC AVX2 host, exact parity, `CMF_X86_BLOCKED=0`
reverts. **0.4.0** extends the whole-token decode graph (Metal) to drive
`q1`, the training-free ternary `q1t`, **and** `q4_block` projections — so a
q1t 14.8B GDN-hybrid decodes 2.7× faster on the graph and beats the same model
in `q4` on the CPU, and a plain q4 model now runs the graph too (3.0 → 5.6
tok/s on an M4, where q4 had no GPU kernel before). The wgpu backend gained
WGSL `q1t`/`q4_block` matvec plus a `q1t` prefill GEMM, so those quants
GPU-accelerate on NVIDIA / AMD / Intel as well (validated against the CPU
reference).

Beyond the drag race: the file is 26% smaller at matched quality, attention
memory can be O(1) (`--o1` holds ~16.5 tok/s at contexts where exact
attention decays from 15.7 to 8.2), 1-bit-trained models run on a GPU graph
`llama.cpp` has no equivalent for (see [1-bit models](#1-bit-models-bonsai--bitnet-class)),
and the whole engine is portable Rust with no C++ toolchain. Reproduce with
`cortiq bench --json` (add `--core` for the llama-bench contract).

### One file, nothing on the side

The tokenizer (HF byte-level BPE) and the chat template (Jinja) travel **inside**
the model — GGUF does this too, and it was right to: the file, not your runtime
binary, defines chat behavior, and there are no sidecars to lose or let drift out
of sync. What a `.cmf` adds on top is integrity: a fixed 128-byte envelope plus a
64-bit hash per tensor means a `.cmf` is either valid or `open()` fails loudly. It
detects truncation and bit-rot; it is not a signature.

```sh
cortiq verify model.cmf     # envelope, sections, every tensor hash
cortiq info   model.cmf     # arch, tensors, quantization, skills
```

Weights are memory-mapped and read in place, so startup is instant and unused
weights never touch RAM. Quantization is per tensor and mixable — `q8`
(1 byte/param) · `q8_2f` (int8 with both a per-row and a per-column scale — better
quality at the same byte count) · `q4` (0.5) · `q4t` (the same 4-bit grid in
interleaved tiles) · `q4tp` (**q4t with predicted scales** — 0.52 B/param, see
below) · `q1t` (**training-free ternary**,
`{−s,0,+s}` + a sparse outlier overlay, ~2.25–3.5 bit/param — below `q4`, made by
`quantize-gptq`, GPU-accelerated on Metal and wgpu) · `q1` (1-bit) · `f16` ·
`vbit` (variable 3–8 bit, ~4.25 avg ≈ 0.53) — so you can keep attention at q8 and
push the FFN to q4/q1t in the same file. See [q1t PTQ](docs/Q1T_PTQ.md).

#### `q4tp` — 7% off any q4t file, for ~0.1% of the error budget

A `q4t` tile spends 16 bits on a standalone f16 scale for 32 weights: 0.5 of
its 4.5 bits/weight, **11% of the file**. `q4tp` keeps the nibbles
byte-identical and makes that scale a 5-bit rung on a per-row geometric
ladder, for 4.17 bits/weight.

It is nearly free. Quantizing the *same* fp32 weights both ways, `q4tp`'s
error against the source is 9.71% where `q4t`'s is 9.71% — a **+0.1%**
relative increase at the median within-row scale spread measured on
KAT-Coder-V2.5, +0.3% at its 90th percentile. A coarser scale barely matters
because the nibbles simply re-round against it; the 4-bit grid dominates
either way. Nanbeige-3B keeps its top-5 next tokens in the same order with
logits within 0.7%.

Existing `.cmf` files convert in place — no original checkpoint needed, which
matters when the checkpoint behind a published model is tens of gigabytes:

```sh
cortiq requant model.cmf --output model-q4tp.cmf --quant q4tp
# KAT-Coder-V2.5: 12.65 → 11.80 GB (19254 tensors) in 2 min
# Nanbeige-3B:     2.36 →  2.19 GB, peak RSS 2.42 → 2.19 GB
```

`cortiq convert --quant q4tp` produces it straight from a HF checkpoint.
Kernels cover every path — CPU (scalar, sdot/AVX2/VNNI int8, blocked 1×4,
Accelerate), Metal and wgpu, decode and batched prefill alike. Speed is at
parity or better: a 16 B nibble stride is 4-aligned, so the matvec loads four
uints where `q4t` needs nine unaligned ushorts, and the batched GEMM at b=64
runs 9.51 vs 10.13 ms on Metal and 9.83 vs 10.76 on wgpu.

### Many specialists, one backbone

Shipping *N* fine-tunes normally means *N* full copies on disk and in RAM. CMF
keeps **one backbone plus one small skill per specialist**: a skill stores only
the tensors it actually replaces, and at inference the runtime reads those *in
place of* the backbone's — no separate model is ever assembled. Storage is
`|backbone| + Σ|skills|`, not `N × |model|`, and a skill you don't use costs
**zero RAM**.

A skill isn't just cheaper to ship — on its own task it beats the backbone it sits
on: on held-out data, a skill overlay cuts task perplexity by **24.9%**
([spec §9](docs/CMF_V2_SPEC.md)). Skills pay off most where the backbone is
weakest; on domains it already handles well, expect less.

```sh
cortiq run model.cmf --prompt "SELECT ..." --skill sql
```

Don't want to pick by hand? `cortiq route` chooses a skill from the prompt, and
`cortiq explain` shows you why.

**Try it in three commands**: [the skills guide](docs/SKILLS.md) bakes three
real skills from public Hugging Face fine-tunes into one 0.5B file with
`cortiq skill add` — a text-to-SQL assistant, a Russian assistant (−7.1%
measured PPL on Russian prose) and a step-by-step verifier — then routes
fresh prompts to the right one 6/6, blends them, and switches mid-stream.
The same guide covers the full DTG-MA bake: a trained task mask + FCD +
physical defrag turned a 1.6 GB checkpoint into a **705 MB specialist
that is 14.7% better on its domain and faster** — measured end-to-end
through this runtime on held-out text. With commands, measurements, and
the failure modes spelled out.

Serving *N* task-specialists:

| | N full fine-tunes | base + N external LoRAs | **CMF** |
|---|---|---|---|
| On disk | N × full model | base + N adapters (sidecars) | one backbone + N small skills, **one file** |
| Tokenizer + chat template | per copy / sidecar | embedded if the base is GGUF, else sidecar | **embedded** |
| Per-tensor integrity hash | — | — | **yes** |
| Unused skill in RAM | loaded | 0 with an adapter-paging server (S-LoRA / vLLM); loaded otherwise | **0**, paged on use, no serving stack required |
| Skill ships inside the model file | — | no (separate adapter files) | **yes, under the same hash chain** |

A full format-by-format comparison — GGUF, safetensors, ONNX, PyTorch, GGML,
TensorRT, with the trade-offs spelled out — is in
[docs/COMPARISON.md](docs/COMPARISON.md).

## Install

```sh
cargo install cortiq-cli                 # the `cortiq` command-line tool
cargo add cortiq-core                    # or use the format from your own Rust code
```

Prebuilt binaries are on the [latest release](https://github.com/infosave2007/cmf/releases/latest)
— Linux x86-64, macOS (Apple Silicon and Intel), Windows (x86-64 and ARM64); every
archive ships a `.sha256`. Since 0.3.1 they include the wgpu GPU backend —
set `CMF_GPU=1` to use it (see [GPU](#gpu)).

## Commands

| command | what it does |
|---|---|
| `cortiq convert --model <hf-repo\|dir>` | Hugging Face checkpoint → `.cmf` (native Rust) |
| `cortiq import-gguf <file\|hf-repo>` | GGUF → `.cmf`, every common ggml quant |
| `cortiq run model.cmf` | chat, or `--prompt` for one shot |
| `cortiq serve model.cmf` | OpenAI-compatible HTTP server + dashboard |
| `cortiq info` · `masks` · `verify` | inspect arch, tensors, skills; check integrity |
| `cortiq bench --ctx 4096` | tok/s and memory at a given context |
| `cortiq ppl --file f.txt` | teacher-forced perplexity — the quality gate |
| `cortiq fcd` | restoration trainer for `--o1` models (KL-anchored, generation-gated) |
| `cortiq diff a.cmf b.cmf` | what changed between two model versions |
| `cortiq route` · `explain` | which skill the router picks, and why |
| `cortiq skill add` · `list` | bake a skill from a donor checkpoint ([guide](docs/SKILLS.md)); list a file's skills |
| `cortiq moe-defrag` | drop the MoE experts a task never routes to — a smaller, faster specialist |
| `cortiq moe-mask` | the switchable twin: bake expert restrictions as task masks — one file, many specialists (`run --task`) |
| `cortiq sign` · `verify` | detached Ed25519 signature over the file's SHA-256; verify checks integrity + authenticity |
| `cortiq compact` | tight container rewrite after append-only skill growth |
| `cortiq imagine model.cmf --prompt "…"` | text → image (Lumina-Image 2.0), native Rust, CPU or Metal |
| `cortiq imagine-pack <diffusers-dir>` | pack text encoder + DiT + VAE + tokenizer into one quantized `.cmf` |
| `cortiq animate model.cmf --prompt "…"` | text → video **with synchronized stereo audio** (MiniMax-H3 + Turbo LoRA); `--first-frame`/`--last-frame` to continue from a picture |
| `cortiq animate-pack` | pack the audio-video DiT, its Qwen3-VL encoder, both VAE decoders and the vision tower into one `.cmf` |

`cortiq <command> --help` documents every flag.

### Converting

```sh
cortiq convert --model Qwen/Qwen2.5-0.5B-Instruct --quant q8    --output model.cmf
cortiq convert --model ./my-hf-checkpoint         --quant q8_2f --output model.cmf
cortiq import-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF --output model.cmf --quant q8
```

GGUF import covers `Q4_0/1`, `Q5_0/1`, `Q8_0`, `Q2_K`…`Q6_K`, `IQ4_NL/XS` and
`BF16` — including the Qwen3.6-MoE / KAT-Coder class (`qwen35moe`:
GatedDeltaNet hybrid + routed experts), where every llama.cpp storage
convention is undone on import. A 34.7B-A3B coder decodes at 16.6 tok/s
on a 32-core CPU where llama.cpp does 4.7 on the same file — and at
**32.8 tok/s on an RTX 5090**, where the router, the top-k expert
selection and every selected expert run inside the single-submit GPU
graph (0.5.25). `CMF_MOE_TAU=0.9` (confidence-adaptive expert routing)
adds ~12% on the CPU path at equal-or-better perplexity than the
model's own fixed top-k. And since expert usage is strongly
task-conditional (code and prose route to near-disjoint expert sets —
top-64 Jaccard 0.25), `cortiq moe-defrag` (0.5.27) physically drops
the experts a task never uses: code-calibrated at 95% routing mass,
the same coder shrinks **19.6 → 12.7 GB (−35%)** at +2.8% code
perplexity — and on a 24 GB MacBook, where the full model paged,
the specialist fits in memory and decodes **×1.8 faster** (prefill
×3.3). Prefer one file with SWITCHABLE specialists? `cortiq moe-mask`
(0.5.28) bakes the same restriction as a named task mask — the full
expert set stays, `run --task coder` narrows routing at inference,
token-identical to the physical cut. A step-by-step walkthrough —
download, convert, run on Vulkan and Metal, carve a specialist — is in
[docs/KAT_CODER.md](docs/KAT_CODER.md).

### 1-bit models (Bonsai / BitNet class)

Checkpoints **trained** with binary weights convert losslessly into `q1`
(1.5 bits/weight — per-group weights already sit on two levels ±s, so the
encoding just recovers them). A 27B becomes a 4.8 GB file that runs on a
24 GB MacBook — and on Apple silicon, `CMF_GPU=1` runs the whole token as
a Metal graph (weights no-copy from the mmap, attention attends on the
device, one sync per token): Bonsai-27B decodes at **11–12 tok/s** on an
M4 with a ~3.2 s first token (0.3.3 did 5); Bonsai-1.7B does ~80–87 tok/s.
CPU-only — the path phones run — does 5–6.6 tok/s on the same machine
(the 0.3.10-era NEON kernel was load-port-bound at 2.5–3.2; a TBL
unpack doubled it), which puts a mid-range phone (Snapdragon 778G
class) at an estimated 2–3 tok/s, DRAM-capped around 5.

Requires **cortiq ≥ 0.3.2** — check with `cortiq --version`; an older
binary answers `unknown quant 'q1'`. Update with
`cargo install cortiq-cli --force` (plain `cargo install` keeps the old
one) or grab the [latest release](https://github.com/infosave2007/cmf/releases/latest).

```sh
cortiq convert --model prism-ml/Bonsai-27B-unpacked --quant q1 --output bonsai27b-q1.cmf
CMF_GPU=1 CMF_THREADS=10 cortiq run bonsai27b-q1.cmf -p "What is 84 * 3 / 2?"
```

Notes: `--quant q1` is an explicit opt-in for 1-bit-trained models only —
as post-training quantization of a normal checkpoint it destroys quality.
Convert from the `*-unpacked` (safetensors) repo, not the GGUF one: hybrid
architectures (qwen3_5: GatedDeltaNet linear layers + full attention every
4th) are supported natively there, and 1-bit decode is compute-heavy, so
give it every core (`CMF_THREADS=10` on a 10-core machine).

The native converter writes **backbones**. The Python tooling in `converter/` is
still what produces the per-skill replacement tensors and task masks described
above, and the GPTQ-calibrated v-bit variant, which needs an activation Hessian.
The weight-only v-bit path is native.

### Training-free ternary (`q1t`)

Where `q1` needs a checkpoint that was *trained* binary, **`q1t` is the
post-training path** — it takes a normal f16/bf16 checkpoint down to ternary
`{−s,0,+s}` with a small per-row sparse **outlier overlay** (`(u16 col, f16 val)`
pairs), landing at ~2.25–3.5 bit/param depending on how many outliers you keep.
It is GPTQ-style: a two-field importance mask (`|W|·RMS(x)`) picks which weights
must stay in full precision, an output-stabilizing per-row rescale corrects the
residual, and embed / `lm_head` / `down_proj` stay heavier by default. No
training, no activation Hessian required — a calibration pass over a few hundred
tokens is enough.

```sh
CMF_GPTQ_TERNARY=1 CMF_GPTQ_SKIP=embed_tokens,lm_head,down_proj \
cortiq quantize-gptq model-q8.cmf --calib corpus.txt --output model-q1t.cmf \
    --keep 0.03 --tokens 1024
CMF_GPU=1 cortiq run model-q1t.cmf -p "What is 84 * 3 / 2?"
```

It carries the same GPU acceleration as the built-in quants: the whole-token
decode graph on Metal and WGSL matvec + prefill GEMM on wgpu (Vulkan / DX12 →
NVIDIA / AMD / Intel). A 14.8B GDN-hybrid in `q1t` decodes 2.7× faster on the
Metal graph than per-op and beats the same model in `q4` on the CPU. Full method,
knobs (`CMF_GPTQ_*`), and quality numbers are in [docs/Q1T_PTQ.md](docs/Q1T_PTQ.md).

### Text to image (Lumina-Image 2.0)

The same format and engine also run a diffusion pipeline — Gemma-2-2B
encodes the prompt, a Next-DiT denoiser runs the flow-matching loop, the
FLUX VAE decodes the latent — all native Rust, no Python at inference.
`imagine-pack` folds the 19 GB [Lumina-Image 2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0)
diffusers tree into one **3.2 GB** `.cmf` (projections q4t, VAE f16,
tokenizer embedded) that `imagine` runs straight off the mmap:

```sh
cortiq imagine-pack ./Lumina-Image-2.0 --quant q4t --out lumina-q4t.cmf
CMF_GPU=1 cortiq imagine lumina-q4t.cmf \
    --prompt "a red fox in a snowy forest, photo" \
    --height 512 --width 512 --out fox.ppm
```

On macOS every modulated DiT block executes as a single Metal command
buffer — quantized GEMMs decode their tiles in the K loop, attention
softmax and the SwiGLU FFN stay on the device, only the hidden state
crosses the CPU boundary — and the VAE decoder runs on the GPU too
(conv2d as implicit GEMM, whole resnet blocks per command buffer). An
M4 renders 256×256 / 30 steps in **~37 s** and 512×512 in ~2.5 min;
CPU-only takes about twice as long, and 1024×1024 is practical on the
block graph.

On everything else, `CMF_GPU=1` runs the quantized GEMMs and the fused
FFN through wgpu (Vulkan / DX12 → NVIDIA / AMD / Intel, and Adreno /
Mali on phones): an RTX 4090 renders 512×512 in **less than half** the
time its host's 32 CPU cores need. The GPU arm is probed against the CPU
exactly like LLM inference — a device that loses keeps the CPU path, so
enabling the GPU never makes you slower — and GPU renders stay visually
identical. For apps, the C ABI exports `cortiq_imagine`
(text → RGB8 with a per-step progress callback); `--cfg 1` disables
guidance and halves the work — the right default on phones.

### Text to video, with the sound (MiniMax-H3 + Turbo LoRA)

The same container carries a joint audio-video model. One prompt, one
transformer, and out comes a clip **and its synchronized stereo track** —
the two are denoised together in one packed sequence, on two different
flow schedules, in four sampling steps.

```sh
CMF_MMH3_GPU=1 cortiq animate mmh3-turbo-fl2va-q4tp.cmf \
    --prompt "A corgi in a chef hat flipping a pancake, sizzling sounds and a cheerful bark." \
    --width 512 --height 288 --frames 39 --out corgi.avi
# continue from a picture instead:
CMF_MMH3_GPU=1 cortiq animate mmh3-turbo-fl2va-q4tp.cmf \
    --prompt "…" --first-frame frame.ppm --out clip.avi
```

At 512×288, 39 frames, on one RTX 5090: **91.6 s for an 8-step render,
42.8 s for 2** — denoise 67.4 s, video VAE 16.6 s, audio VAE 5.0 s, and
`RUST_LOG=info` prints that breakdown for every run. The DiT block, both
VAE decoders and the vocoder's convolutions all run on the card; a given
seed reproduces bit for bit.

Four files and a ComfyUI checkout — 124.4 GB — become **one 23.9 GB
`.cmf`**: the 33 B DiT, its Qwen3-VL-32B prompt encoder, the ViT3D video
decoder, the BigVGAN vocoder, Qwen3-VL's vision tower and the VAE
encoder that a keyframe needs. The 4-step LoRA is merged in, so the file
IS the turbo model. Output is MJPEG+PCM in an AVI plus a `.wav` — the
JPEG encoder and the RIFF muxer are in the binary, because a pipeline
that ends in a shell-out to ffmpeg is not one you can ship.

Most of that reduction is not quantization. **Forty per cent of the
released model is one matrix per block** — `adaln_proj.linear`, 13 B
parameters, for a map whose only input is the timestep. Over the
schedule its output traces a curve, so `animate-pack` collapses it onto
a rank-24 basis with the Turbo LoRA already folded in: 4.6 MB a block
instead of 520, at rms 8.7e-5 against a signal of rms 0.464.

Every stack is diffed against ComfyUI's own modules on toy checkpoints
carrying the release's real tensor names and schedules (`tools/mmh3_toy_gate.sh`):
DiT velocities 8.8e-5, prompt encoder 1.1e-6, video decoder 4.2e-7
including its 256-pixel tiling, vocoder 1.7e-9, vision tower 6.2e-6.
512×288 over 39 frames renders in 346.5 s on 48 CPU cores and **172.0 s
on one RTX PRO 6000** — the device arm is opt-in for this model, hence
the `CMF_MMH3_GPU=1` above. Weights and a model card:
[infosave/MiniMax-H3-Turbo-cmf](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf).

## O(1) in depth

Record the hint at convert time, or decide at load time — the runtime picks the
header hint up automatically:

```sh
# at convert time: all softmax layers, the deepest N, or an explicit list
cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --o1 all    --output model.cmf
cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --o1 deep12 --output model.cmf

# or override at load time, without reconverting
cortiq run   model.cmf --o1 all      # force-convert every softmax layer
cortiq run   model.cmf --o1 off      # back to exact attention
cortiq bench model.cmf --ctx 4096    # memory + tok/s, with and without
CMF_O1=deep6 cortiq serve model.cmf  # env override, same syntax

# tuning (validated defaults: 32 landmarks, window 128, 4 anchor keys)
cortiq run model.cmf --o1 all --o1-m 32 --o1-window 128 --o1-sink 4
```

On hybrid models (e.g. qwen3.5: GatedDeltaNet layers with softmax islands)
`--o1 all` converts just the softmax layers, which makes the whole model's
attention state constant in context length.

**Restoration.** `cortiq fcd` is a bounded native training pass — no Python, no ML
framework — that tunes only the converted layers' norm/FFN tensors against the
same model running exact attention (KL-anchored), and keeps a checkpoint only if
long-context generation stays loop-free:

```sh
cortiq fcd model.cmf --corpus corpus.txt --gen-check --gen-gate --out model.fcd.cmf
# knobs: --steps 300 --eval-every 25 --kl 0.7 --lr 5e-5 --o1 all|deepN|i,j,k
#        --val-corpus val.txt --gate-threshold 0.35 --gate-slack 0.10
```

### Speculative decode off the model's own MTP head

A model that ships an MTP head drafts with it and verifies the whole
chain in one batched submit. **On by default for greedy decoding of
q4tp files** (since 0.5.80): the draft block runs on the token graph,
the verify's rows go through an int8-activation batched matvec
(`dot4I8Packed`), and a monitor keeps averaging tokens-per-round
against the plain token — it stops speculation after four losing rounds
(free prose often does not pay) and retries 128 tokens later, so a
prompt that gains keeps the gain and one that does not sits at the
plain rate. Greedy — with or without repetition/presence penalties —
verifies by argmax equality; with `CMF_VERIFY_I8=0` the f32 verify makes
the stream bit-identical to the plain path (the int8 default can
resolve a near-tie differently — a different but equally greedy
continuation). `CMF_GRAPH_SPEC=0` turns speculation off; `=1` forces it
on for other layouts (q4t/q8_2f verify through tile GEMMs and measured
a loss). Speculative *sampling* (temperature > 0) exists behind a
second switch: the draft is a draw from the MTP head's own post-chain
distribution q, accepted with min(1, p/q) against the verified position's
p, a rejected draft replaced by a draw from max(0, p − q); the emitted
stream is distributed exactly as the plain sampler's (a 400k-trial test
holds the empirical law within L1 0.01 of the target). It stays opt-in:
at the Qwen instruct row (0.7 / 0.8 / 20 / presence 1.5) acceptance is
60% and the round breaks even with the plain token.

```sh
cortiq bench model.cmf --tokens 200 --core --json          # speculation on by default (q4tp)
CMF_GRAPH_SPEC=0 cortiq bench model.cmf --tokens 200 --core --json   # plain, for the A/B
CMF_VERIFY_I8=0 cortiq run model.cmf --prompt "..." --greedy         # f32 verify: bit-exact with plain
# CMF_GRAPH_SPEC_K=5 (default with the int8 verify, 4 with f32) — drafts per round
# CMF_GRAPH_SPEC_SAMPLE=1 extends speculation to sampling (opt-in, see above)
```

On Qwen3.8-27B q4tp / RTX 5090, `bench --core`: **76 tok/s against a
plain 48.5** (int8 verify, k=5; the f32 verify 66), 90% of drafts
accepted; on a 2.3k-token code prompt in `run` 56.5 against 45.7, on an
essay the monitor stops speculation and the rate is the plain 46–47.
`bench --json` reports `mtp_drafted` and `mtp_accepted` — watch the
ratio, not just tok/s, because a broken draft path degrades silently
into a lower acceptance rate rather than into wrong output.

**On Apple silicon** (since 0.5.82) the same round runs natively on Metal:
a b-row whole-model graph verifies the chain in one submit (a
simdgroup-matrix GEMM that is flat in the batch up to 8 rows, the GDN
recurrence over the b positions in registers with the accepted prefix
replayed on commit), the MTP block drafts as one submit with a
vocabulary shortlist for its head, and the round's warm-ups are one
batched run. Qwen3.8-27B q4tp on an M4 mini (24 GB): plain **6.7
tok/s**, a code body **12.2** with speculation, prose at the plain rate
by the monitor's choice; the prompt goes through the same graph in
256-token chunks (447 tokens: **11.6 s** to first token, was 42). Two
oracles guard it — `CMF_METAL_VERIFY_CHECK=1` diffs every verify row
against the plain path, `=2` the replayed state after commit — and the
0.5.82 release also fixed two silent Metal bugs found on the way (an
unbound GEMM constant that made every q4tp batched prefill noise, and a
head-index slip that skipped the K heads' norm+RoPE on the device
attend of every Qwen3.5-family model). See the CHANGELOG.

It is worth knowing why the plain number is what it is: that decode is
**bus-bound**. Two decode processes on one card aggregate 52.8 tok/s
against a single process's 48.8, and the matvec stripped of all its
arithmetic runs the same token to within 4% — so the remaining levers
are reading fewer bytes (quantization) or amortizing the read
(speculation), not a faster kernel. The proofs are in
[the kernel recipes](docs/GPU_KERNEL_RECIPES.md).

**Model walkthroughs**: [KAT-Coder 35B MoE](docs/KAT_CODER.md) · [Qwen3.6-35B-A3B (q4tp, MTP head, o1 on GPU)](docs/QWEN36_MOE.md)

**GPU performance**: [kernel recipes and the measured failures](docs/GPU_KERNEL_RECIPES.md) ·
[where the next speed is](docs/OPTIMIZATION_ROADMAP.md)

## The format

A `.cmf` is a fixed 128-byte envelope followed by sections that a reader addresses
**only** through that envelope, never by assuming order:

- **header JSON** — arch, quant defaults, chat bundle, skill registry, provenance
- **tensor directory** — 56-byte binary records (name, dtype, shape, offset, nbytes, hash64), readable without touching the JSON
- **weight blob** — page-aligned, mapped and read in place
- **skills** — bit-packed task masks and per-skill replacement tensors
- **tokenizer** — the verbatim Hugging Face file
- **sparse index** — precomputed

Also supported: multi-token-prediction (MTP) heads, MoE FFN layers, Laguna's
mixed global/sliding attention with per-layer head counts, dual RoPE/YaRN and
softplus attention gates, append-only skill growth with compaction, and
sharding a model across `N` standalone-valid files.

**You are not locked in.** `python/cmf_reader.py` is a complete reader in ~300
lines of stdlib + numpy that shares no code with the Rust runtime — it was written
from the spec, on purpose, to prove the format outlives this implementation:

```python
from cmf_reader import CmfReader
r = CmfReader("model.cmf")
w = r.tensor("model.layers.0.mlp.gate_proj.weight")   # np.ndarray, dequantized
assert r.verify() == []                               # every tensor hash checks
```

If this project disappeared tomorrow, your weights are still readable from the
spec alone. The complete normative specification is in
[docs/CMF_V2_SPEC.md](docs/CMF_V2_SPEC.md).

## Status

CMF is **0.2.x** and young — first public release July 2026, one author. The crate
APIs may still move before 1.0. The **format** is the settled part: it is v2,
readers navigate only through the envelope, unknown header fields are ignored
(additive evolution), and a breaking change costs a feature bit or a `version`
bump — never a silent reinterpretation. A `.cmf` written today stays readable;
`cortiq verify` is the contract. Every change is in [CHANGELOG.md](CHANGELOG.md).

Bugs and feature requests: [open an issue](https://github.com/infosave2007/cmf/issues).
Security problems: **do not** open a public issue — see [SECURITY.md](SECURITY.md).
A model that won't convert is a bug report, not a user error.

## Build from source

```sh
cargo build --release --workspace   # the CLI's wgpu backend (Vulkan / DX12 / Metal) is on by default
cargo build --release -p cortiq-cli --no-default-features   # CPU-only build
```

```
crates/
  cortiq-core     format reader: envelope, directory, quant, masks, mmap
  cortiq-engine   portable CPU/GPU inference runtime, tokenizer, chat, skills
  cortiq-server   OpenAI-compatible HTTP serving
  cortiq-cli      the `cortiq` command-line tool
converter/        Python: DTG-MA skills/masks + the GPTQ-calibrated v-bit path
python/           reference reader — stdlib plus numpy, nothing else
docs/             format specification and comparison
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## GPU

```sh
CMF_GPU=1 cortiq run model.cmf
```

### More than one GPU

Two flags, two different problems:

```sh
cortiq serve model.cmf --gpus 2   # throughput: one full replica per card
cortiq run   model.cmf --gpus 2   # capacity: the layer stack split across cards
```

**Replicas** (`serve --gpus N`) put a whole copy of the model on each
card and serve N requests at once. On 2×RTX 5090 with a 34.7B MoE:
115.3 tok/s for one request, **218.5 tok/s aggregate for two** — the
mode that actually scales, and the one to reach for when the model
fits on a single card.

**Layer split** (`run/bench --gpus N`) runs segment *i* pinned to card
*i* inside ONE process; only a hidden vector crosses a boundary, and it
never leaves the address space. It exists for models bigger than one
card, and on a model that already fits it costs a few percent rather
than saving any — measured on 2×RTX 5090, single-stream decode: Bonsai
1.7B 191.6 → 174.8 tok/s, Nanbeige 3B 77.4 → 71.8, the 34.7B MoE 129.6
→ 118.9. All three run correctly on the split; all three are slower for
one stream. A layer split buys room, not speed. `--peer-split N` moves
the boundary; `cortiq gpu` lists the cards and their indices,
`CMF_GPU_ADAPTER=<index|name>` pins one.

[docs/MULTI_GPU.md](docs/MULTI_GPU.md) has the full table, the
across-machines form, and what to check when a card is not used.

For a model bigger than one HOST, `--peer` speaks to a `cortiq worker`
over the network instead — same split, one socket further away. Add
`--peer-head` and the worker also runs the head and the sampler and
answers token ids: the caller keeps only its tokenizer, and the wire
drops to 16 bytes a token. That is what lets a weak machine drive a
strong one — an Android phone with 2 GB of free RAM generates from a
34.7B MoE at 16.3 tok/s, against 9.1 with the head at home.
`--peer-run-ahead K` returns K tokens for one round trip, which is what
makes that work over Wi-Fi rather than a cable (13.3 → 23.6 tok/s); it
is not speculation, just the handshakes removed, and the output is
identical. `--peer-prefill` is the mode where the wire leaves: the peer
absorbs the prompt, its KV comes home, and the conversation continues
with nothing plugged in — an 1800-token prompt goes 86.9 s → 16.3 s on
that phone. `cortiq peers` finds a worker on the LAN.
[docs/MOBILE_SPLIT.ru.md](docs/MOBILE_SPLIT.ru.md) has the roles, the
numbers behind them, and the mobile ABI.

The backend is picked automatically: wgpu chooses Vulkan on Linux/Windows,
DX12 on Windows if Vulkan is absent, Metal on macOS — nothing to configure
(`WGPU_BACKEND=vulkan|dx12|metal|gl` overrides). Weights stay in VRAM up to
a budget (`CMF_GPU_VRAM_MB`, default 8192 on discrete cards); layers are made
resident in first-touch order, so the budget behaves like llama.cpp's `-ngl`
without a flag: first N layers on the GPU, the rest on the CPU.

On macOS, `q1`, `q1t`, and `q4_block` models run the whole token as a Metal
graph (0.4.0 extended it past `q1`, so a plain q4 model gets it too): hidden
state lives on the device across every layer, attention attends **on the GPU**
(rope, qk-norms, KV append, grouped online-softmax attend), and command
buffers commit as they are encoded — one wait per token. The CPU cache
remains the owner of record, so eviction, speculative rollback and
serialization behave exactly as on the CPU path. The graph is
distribution-equivalent to the CPU path (first-token probabilities within
~0.3%, PPL matches), not bit-identical on every prompt — floating-point
reductions run in a different order, as with any GPU offload.
`CMF_GPU_ATTEND=0` keeps the attention core on the CPU, `CMF_GPU_BLOCK=0`
disables the graph. `q4_tiled` joined the list in 0.5.35, on both the
decode graph and the batched prefill; `q4tp` in 0.5.38, on Metal and wgpu
alike.

Since 0.5.35 the Metal decode attend is flash-decoding shaped — the
simdgroups of one threadgroup split a head's positions and combine their
partials in threadgroup memory (`CMF_GQA_SPLIT`, default 8) — so
throughput holds at depth instead of falling off a cliff, and the whole
attention layer rides in ONE compute encoder. Prompt ingest for Looped
Transformers goes through the batched chunk-GEMM rather than the
per-position graph. On a **fanless MacBook Air (M4, 10-core GPU, 24 GB)**
the looped Nanbeige4.2-3B — 22 physical layers run twice, so every token
pays for 44 layer passes and 3.9 GB of weight reads — decodes at
**22.4 tok/s** at short context, 17.6 at ctx 512, 14.2 at ctx 1024, and
ingests a prompt at **181 tok/s** (2.7 s to first token on 512 tokens).
That is ~72% of what the machine's 120 GB/s of memory bandwidth allows at
all; before 0.5.35 the same file did 9 tok/s at ctx 512 and took 40 s to
absorb the prompt.

On discrete cards (Vulkan/DX12 via wgpu), 0.5.18 runs the same
whole-token graph by default: every layer — including the GDN recurrence
of hybrid models and the loop boundaries of Looped Transformers — executes
in one submit with one readback per token, and attention uses split-K
flash-decoding so throughput holds at depth. Measured on an RTX 4090:
Bonsai-27B q1 decodes at **40 tok/s** (7.7× the CPU path, still 38 at a
4K-token context), Bonsai-1.7B q1 at 154 tok/s (95+ out to ctx 2048),
Nanbeige4.2-3B at 31 tok/s. Since 0.5.25 routed MoE rides the same
graph: the router matvec, the on-device top-k selection and every
selected expert (plus the shared expert) execute in the same one submit
per token — KAT-Coder 34.7B-A3B decodes at **32.8 tok/s on an RTX 5090**
vs 14.4 on its 32-core host CPU
([step-by-step guide](docs/KAT_CODER.md)). Output is
distribution-equivalent to the CPU path, same as the Metal graph.
`CMF_GPU_WGPU_GRAPH=0` reverts to per-op offload.

For everything else, enabling the GPU never makes you slower: per-op
offload pays a fixed submit+poll latency that differs by an order of
magnitude between driver stacks, so at startup the engine *measures* — for
each op class (FFN chain, large matvec, prefill GEMM, QKV batch) the first
calls alternate between GPU and CPU, both timed, and the faster arm is
kept. `RUST_LOG=cortiq_engine=info` shows the verdicts; `CMF_GPU_PROBE=0`
trusts the GPU unconditionally.

## License

**Apache-2.0** ([LICENSE](LICENSE)) — use it, modify it, ship it commercially.

This software practices methods claimed in four pending US patent applications by
the author, listed in [PATENTS.md](PATENTS.md). Apache-2.0 Section 3 grants you a
perpetual, worldwide, royalty-free patent license to those applications' claims
that are necessarily infringed by this software as distributed: **running, forking
and shipping this software is covered**, and the grant lapses only if you sue the
project over patents.

That grant is scoped to this Work, as Apache-2.0 §3 always is — it does not by
itself extend to an independent reimplementation of the container. If you want to
implement CMF in another language or embed it in your own runtime, email
urevich55@gmail.com: an implementer's grant is available, and the format is meant
to be implemented widely.

## Where this came from

The design ideas came out of the author's separate work on a physics theory — the
Vacuum Mass Fraction (VMF) within Null-Vector Gravity (NVG): the
shared-backbone-plus-perturbations model, the two-field quantization. Nothing in
the format depends on that theory being right; it stands on the spec and the
numbers above. The mapping, with a hard line drawn between what is *measured* and
what stays a metaphor: [the VMF/NVG principles behind CMF](VMF_principles_in_CMF.md)
([Русский](VMF_principles_in_CMF.ru.md) · [中文](VMF_principles_in_CMF.zh.md)).
The physics itself lives in [its own repository](https://github.com/infosave2007/vmf).
