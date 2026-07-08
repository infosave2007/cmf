# CMF — Cortiq Model Format

**Serve many specialized LLMs from one shared model — in a single file, on CPU or GPU.**

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![downloads](https://img.shields.io/crates/d/cortiq-cli.svg)](https://crates.io/crates/cortiq-cli)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![stars](https://img.shields.io/github/stars/infosave2007/cmf?style=flat)](https://github.com/infosave2007/cmf)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/infosave2007/cmf/blob/master/LICENSE)

---

## The problem

Shipping a fleet of task-specialized models is expensive. *N* fine-tunes usually mean *N* full copies on disk and in RAM — plus loose `config.json` / `tokenizer.json` / adapter sidecars to keep in sync, and no built-in way to tell a corrupt file from a good one.

## The idea

**CMF keeps one backbone and layers lightweight per-skill overlays on top of it.** A skill stores only the tensors it actually changes; at inference the runtime reads a selected skill's tensors *in place of* the backbone — no separate model is ever assembled. So a whole set of specialists lives in **one self-describing file** and runs from a laptop, with weights read straight off disk (`mmap`, zero-copy) and unused skills costing no RAM.

And the specialist is not just cheaper — it's *better on its task*: measured on held-out data, a skill overlaid on its backbone cuts **task perplexity by 24.9%** versus the backbone alone (see [spec §9](docs/CMF_V2_SPEC.md)).

## Who it's for

- **Agent / plugin builders** — one model carrying 20 skills (SQL, code, translation…) instead of 20 models to store, load, and route between.
- **Edge / local deployment** — fit a routed multi-skill model into the RAM budget of a single model; weights are paged from disk on demand.
- **Anyone shipping quantized LLMs** — one integrity-checked file carries weights **+ tokenizer + chat template**, so there are no sidecars to lose and corruption is caught by per-tensor hashes.

## See it run

```console
$ cortiq run model.cmf --prompt "What is the capital of France?" --greedy
Ready: qwen2 | Task: general | Sparsity: 0%
Prompt: What is the capital of France?
 The capital of France is Paris.
[8 tokens, 33.6 tok/s, finish: stop]
```

## Why CMF — what you get

- **Add a skill without copying the model.** One backbone + small per-skill deltas: storage is `|backbone| + Σ|deltas|`, not `N × |model|`.
- **Starts instantly, light on RAM.** Weights are memory-mapped and read in place; masked-out or unused weights never touch RAM.
- **Smaller on disk, honestly.** Mix quantizations per tensor — `q8`, `q4`, two-field `q8_2f`, variable-bit (3–8 bit) — down to ~1 byte/param and below. The two-field and variable-bit codecs recover most of the int8→fp16 quality gap at the same file size, and the accuracy trade is *measured*, never declared.
- **One file, no sidecars.** The HF tokenizer (byte-level BPE) and the chat template (Jinja) travel inside the model — the file defines chat behavior, not your runtime binary.
- **Trust the file.** A fixed 128-byte envelope plus a 64-bit hash per tensor mean a `.cmf` is either valid or `open()` returns an error; `cortiq verify` checks the whole chain.
- **Runs anywhere.** A dependency-free Rust core on CPU, plus an optional GPU backend (wgpu → Vulkan · Metal · DX12).
- **Convert in one command.** `cortiq convert --model <hf-repo>` — native Rust, no Python/numpy/torch; the model is downloaded (in parallel) and quantized in one step.

## How it compares

Serving **N task-specialists**:

| | N full fine-tunes | Base + N external LoRA | **CMF — one backbone + N skills** |
|---|---|---|---|
| On disk | N × full model | base + N adapters (sidecars) | one backbone + N small deltas, **one file** |
| Tokenizer + chat template | per copy / sidecar | sidecar | **embedded** |
| Per-tensor integrity hash | — | — | **yes** |
| Cold / unused skill in RAM | loaded | loaded | **0** (paged on use) |

The full, honest format-by-format comparison — GGUF, safetensors, ONNX, PyTorch, GGML, TensorRT, with the trade-offs spelled out — is in [docs/COMPARISON.md](docs/COMPARISON.md).

## Install

Install the command-line tool:

```sh
cargo install cortiq-cli
```

Use the format from your own Rust project:

```sh
cargo add cortiq-core
```

## Quick start

Inspect a `.cmf` — arch, tensors, quantization, masks and skills:

```sh
cortiq info  model.cmf
cortiq masks model.cmf
cortiq verify model.cmf     # envelope, sections, per-tensor hashes
```

Convert a model to `.cmf` — **native Rust, no Python/numpy/torch**. Pass a
Hugging Face repo id (downloaded in parallel) or a local model directory:

```sh
cortiq convert --model Qwen/Qwen2.5-0.5B-Instruct --quant q8    --output model.cmf
cortiq convert --model ./my-hf-checkpoint         --quant q8_2f --output model.cmf
```

Or import a GGUF directly — a local file, or a Hugging Face GGUF **repo id**
(the best `.gguf` is picked and downloaded). Every common ggml quant is
dequantized natively (`Q4_0/1`, `Q5_0/1`, `Q8_0`, `Q2_K`…`Q6_K`, `IQ4_NL/XS`,
`BF16`) — no Python:

```sh
cortiq import-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF --output model.cmf --quant q8
cortiq import-gguf model.gguf                      --output model.cmf --quant q8
```

Quantization: `q8` · `q8_2f` (two-field, best quality/size) · `q4` · `f16` ·
`vbit` (variable 3–8 bit, ~4.25 avg).
Dense, **mixture-of-experts**, and **GatedDeltaNet** models (qwen2 / qwen3 /
qwen3.5 / llama / mistral / qwen-moe) convert natively — including the fused
qwen3_next / AgentWorld layout. The Python converter (`converter/`) is now only
needed for the **GPTQ-calibrated** v-bit variant (which needs an activation
Hessian) — the weight-only v-bit path is native.

Run inference:

```sh
# Interactive chat
cortiq run model.cmf

# Single prompt, greedy decoding, capped length
cortiq run model.cmf --prompt "Write a haiku about memory-mapped files." --greedy --max-tokens 64

# Overlay a specific skill — its replacement tensors are read in place of the backbone
cortiq run model.cmf --prompt "SELECT ..." --skill sql
```

## Container layout

```
 .cmf file
 ┌──────────────────────────────────────────────────────────┐
 │ Envelope        128 bytes, fixed                          │
 │   magic "CMF\x01" · version · feature bits · section       │
 │   offsets+lengths (header, dir, data, masks, vocab, index)│
 ├──────────────────────────────────────────────────────────┤
 │ Header JSON     arch, quant defaults, chat bundle,        │
 │                 skill registry, provenance                │
 ├──────────────────────────────────────────────────────────┤
 │ Tensor directory   binary 56-byte records:                │
 │                 name · dtype · shape · offset · nbytes ·  │
 │                 hash64  (read without parsing)            │
 ├──────────────────────────────────────────────────────────┤
 │ Weight blob     page-aligned (4096); every tensor 64-byte │
 │                 aligned; quantized; mmap zero-copy        │
 ├──────────────────────────────────────────────────────────┤
 │ Masks / Skills  bit-packed per-task masks (1 bit/neuron)  │
 │                 + per-skill replacement tensors           │
 ├──────────────────────────────────────────────────────────┤
 │ Tokenizer       HF tokenizer.json, verbatim               │
 ├──────────────────────────────────────────────────────────┤
 │ Sparse index    precomputed mask → active groups/heads    │
 └──────────────────────────────────────────────────────────┘
```

A reader addresses sections **only** through the envelope — never by assuming order.

## Features

- Single-file, memory-mappable, self-validating binary container.
- Binary tensor directory with 1:1 source-model tensor names and a per-tensor 64-bit hash for corruption detection.
- Mixed quantization per tensor: `f32`, `f16`, `bf16`, `q8_row`, `q4_block`, `q8_2f`, `vbit`.
- Embedded tokenizer (HF byte-level BPE parity) and chat template (Jinja, HF semantics).
- Per-task masks (bit-packed) and a precomputed sparse index.
- Multi-skill swarm: one backbone + per-skill full-shape replacement tensors, overlaid at forward time; append-only growth and compaction.
- Optional multi-token-prediction (MTP) head and mixture-of-experts (MoE) FFN layers.
- Sharding: a model split across `N` standalone-valid `.cmf` files.
- Dependency-free Rust runtime on **CPU and GPU** (optional `gpu` feature: wgpu → Vulkan / DX12 / Metal).
- Reference implementations in Rust (reader + runtime) and Python (writer + a stdlib+numpy reader).

## Format overview

The complete normative specification — envelope, header JSON, tensor directory, quant layouts, masks, tokenizer bundle, sparse index, `hash64`, skills and sharding — is in [docs/CMF_V2_SPEC.md](docs/CMF_V2_SPEC.md).

## Theory & background

CMF's design is derived from the author's physical theory — the **Vacuum Mass Fraction (VMF)**, within **Null-Vector Gravity (NVG)**. Twelve NVG/VMF principles map to concrete format elements (one shared backbone, two-field `q8_2f`, task masks, the held-out quality contract, resonance routing, the variable-bit codec…), with a hard line between what is *measured* and what stays a metaphor.

- **[The VMF/NVG principles behind CMF](VMF_principles_in_CMF.md)** — the full mapping ([Русский](VMF_principles_in_CMF.ru.md) · [中文](VMF_principles_in_CMF.zh.md)).
- **[NVG/VMF theory repository](https://github.com/infosave2007/vmf)** — the physics itself.

## Build from source

```sh
cargo build --release --workspace
```

Optional cross-platform GPU backend (wgpu → Vulkan / DX12 / Metal):

```sh
cargo build --release --workspace --features gpu
```

## Project layout

```
crates/
  cortiq-core     format reader: envelope, directory, quant, masks, mmap
  cortiq-engine   portable CPU/GPU inference runtime, tokenizer, chat, skill overlay
  cortiq-server   OpenAI-compatible HTTP serving
  cortiq-cli      the `cortiq` command-line tool (inspect/convert/run/serve)
converter/        Python converters for exotic archs (MoE / linear-attention)
python/           dependency-free reader (stdlib + numpy)
docs/             format specification and comparison
```

## License & patents

Licensed under the **Apache License, Version 2.0** — see [LICENSE](LICENSE).

This software implements methods that are the subject of three United States patent applications; details are in [PATENTS.md](PATENTS.md). The Apache-2.0 Section 3 patent grant applies to those three referenced applications, giving every user a royalty-free license to the patent claims necessarily infringed by this software as distributed.
