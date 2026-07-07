# CMF — Cortiq Model Format

**One self-describing file for a quantized LLM — weights, tokenizer, chat template, task masks and per-skill overlays — with a portable, dependency-free Rust runtime that runs on CPU and GPU (Vulkan · Metal · DX12).**

[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/infosave2007/cmf/blob/master/LICENSE)

---

## Why CMF

- **One file.** Weights, tokenizer, chat template, per-task masks and per-skill delta records ship as a single `.cmf` container — one unit of distribution, no sidecars.
- **Self-describing.** A fixed 128-byte envelope addresses every section; a binary tensor directory is the single source of truth for the layout. Files are validated, never guessed — a `.cmf` is either valid or `open()` returns an error.
- **mmap, zero-copy, runs anywhere.** The weight blob is page-aligned and every tensor is 64-byte aligned for SIMD. The runtime memory-maps the file and reads weights in place — cold (masked-out) weights cost no RSS. A dependency-free CPU core needs nothing else; an optional GPU backend targets Vulkan, Metal and DX12.
- **Quantized.** Per-tensor dtypes include `q8_row`, `q4_block`, two-field `q8_2f`, and variable-bit `vbit` (3–8 bit), mixed freely within one model.
- **Multi-skill overlay.** A single shared backbone plus per-skill full-shape replacement tensors. At forward time the runtime reads a selected skill's tensors *in place of* the backbone — without materializing a separate model. Storage scales as `|backbone| + Σ|deltas|`.

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
- Embedded tokenizer (HF byte-level BPE parity) and chat template (Jinja, HF semantics) — the file defines chat behavior, not the binary.
- Per-task masks (bit-packed) and a precomputed sparse index.
- Multi-skill swarm: one backbone + per-skill full-shape replacement tensors, overlaid at forward time via tensor-source indirection; append-only growth and compaction.
- Optional multi-token-prediction (MTP) head and mixture-of-experts (MoE) FFN layers.
- Sharding: a model split across `N` standalone-valid `.cmf` files.
- Dependency-free Rust runtime that runs on **CPU and GPU** (optional `gpu` feature: wgpu → Vulkan / DX12 / Metal).
- Reference implementations in Rust (reader + runtime) and Python (writer + a stdlib+numpy reader).

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
cortiq info model.cmf
cortiq masks model.cmf
```

Convert a Hugging Face checkpoint to `.cmf`:

```sh
python converter/convert_dtgma_to_cmf.py \
    --model  ./my-hf-checkpoint \
    --quant  Q8_ROW \
    --output model.cmf
```

Import a GGUF model:

```sh
python converter/import_gguf.py --input model.gguf --output model.cmf
```

Run inference:

```sh
# Interactive chat
cortiq run model.cmf

# Single prompt, greedy decoding
cortiq run model.cmf --prompt "Write a haiku about memory-mapped files." --greedy

# Overlay a specific skill (replacement tensors read in place of the backbone)
cortiq run model.cmf --prompt "SELECT ..." --skill sql
```

## Format overview

The complete normative specification — envelope, header JSON, tensor directory, quant layouts, masks, tokenizer bundle, sparse index, `hash64`, skills and sharding — is in [docs/CMF_V2_SPEC.md](docs/CMF_V2_SPEC.md).

## Comparison

How CMF relates to safetensors, GGUF and raw HF checkpoints — the one-file, self-describing, mmap-quantized, multi-skill trade-offs — is in [docs/COMPARISON.md](docs/COMPARISON.md).

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
converter/        Python writers: HF → .cmf, GGUF → .cmf
python/           dependency-free reader (stdlib + numpy)
docs/             format specification and comparison
```

## License & patents

Licensed under the **Apache License, Version 2.0** — see [LICENSE](LICENSE).

This software implements methods that are the subject of three United States patent applications; details are in [PATENTS.md](PATENTS.md). The Apache-2.0 Section 3 patent grant applies to those three referenced applications, giving every user a royalty-free license to the patent claims necessarily infringed by this software as distributed.
