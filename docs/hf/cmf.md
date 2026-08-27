---
license: apache-2.0
tags:
- cortiq
- cmf
---

# CMF — Cortiq Model Format

**One file. No Python, no torch, no CUDA install, no C++ toolchain.**

A `.cmf` carries the quantized weights, the tokenizer and the chat template
together, checks its own integrity, and memory-maps straight off disk. The
runtime is a small Rust core with no ML framework under it, running on CPU
everywhere and on GPU through wgpu — Vulkan, DX12, Metal — out of the box.

This repository is the format's front page. It holds the specification and
the index of published models; it carries no weights of its own.

- **Which file to take** — [`FORMATS.md`](FORMATS.md), the quantization ladder
  in plain terms: what `q4tp`, `q2tp`, `q1t` and the rest cost you, and how a
  video container decides precision per tensor family
- **Specification** — [`SPEC.md`](SPEC.md), the normative document
- **Source** — [github.com/infosave2007/cmf](https://github.com/infosave2007/cmf) (Apache-2.0)
- **Runtime** — [`cortiq-cli` on crates.io](https://crates.io/crates/cortiq-cli), and prebuilt binaries for six targets on the [releases page](https://github.com/infosave2007/cmf/releases/latest)

## Run one

```sh
cargo install cortiq-cli                    # or take a prebuilt binary
hf download infosave/Nanbeige4.2-3Bcmf nanbeige42-3b-q4t.cmf --local-dir .
cortiq verify nanbeige42-3b-q4t.cmf         # per-tensor hashes
cortiq run nanbeige42-3b-q4t.cmf --prompt "Write a haiku about memory-mapped files."
```

Converting a checkpoint is one command and no Python:

```sh
cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --output qwen.cmf
```

## What is actually in the file

A CMF file is an envelope, not a tarball: readers navigate through the header
and never guess at offsets. Unknown header fields are ignored, so the format
grows additively — a breaking change costs a feature bit or a version bump,
never a silent reinterpretation. A file written today stays readable, and
`cortiq verify` is the contract that says so.

Beside the weights it carries the things that are usually scattered across a
repository or lost entirely: the tokenizer and chat template, per-tensor
hashes, an optional Ed25519 signature, per-task **masks** that select an
active subset of the shared weights, and a **swarm of skills** sharing one
backbone — several specialists in one file, chosen at load time rather than
downloaded separately.

Two things it does that other single-file formats do not:

**Attention that stops growing with the context.** Converting with `--o1`
replaces a layer's softmax attention with a streaming operator holding a
fixed-size state — a few exact anchor keys, an exact recent window, and a
landmark sketch of everything older, under one shared denominator. The
weights are byte-identical; the flag only records a hint in the header.

**Generative models in the same container.** The format is not
LLM-only. `cortiq imagine` renders images and `cortiq animate` renders video
with synchronized stereo audio, from `.cmf` files packed the same way, on the
same runtime.

## On a phone

The same file on Android and iOS. **[Cortiq: Local AI Models](https://play.google.com/store/apps/details?id=ai.cortiq.cmf_mobile)**
carries this runtime as a native library: chat against a `.cmf` on the device,
convert a Hugging Face repo to CMF on the handset itself, and serve what is
loaded to your LAN over the same OpenAI-compatible API. Paired with
`cortiq worker` on a desktop it runs a model larger than the phone's memory —
measured, a 34.7B MoE at 16.3 tok/s with 2 GB free.

**[What it does →](https://huggingface.co/spaces/infosave/cortiq-mobile)** ·
[source](https://github.com/infosave2007/cmfmobile) (Apache-2.0)

## Published models

| model | what it does | size |
|---|---|---|
| [Nanbeige4.2-3B](https://huggingface.co/infosave/Nanbeige4.2-3Bcmf) | looped transformer, 22 layers run twice | 2.36 GB |
| [Bonsai-1.7B](https://huggingface.co/infosave/Bonsai-1.7Bcmf) | 1-bit BitNet, phone-sized | 0.33 GB |
| [Bonsai-8B](https://huggingface.co/infosave/Bonsai-8B_2bit_cmf) | 1-bit ternary at 2 bits stored | 2.32 GB |
| [Bonsai-27B](https://huggingface.co/infosave/Bonsai-27Bcmf) | 1-bit, 40 tok/s on an RTX 4090 | 5.10 GB |
| [KAT-Coder-V2.5](https://huggingface.co/infosave/KAT-Coder-V2.5-CMF) | 34.7B-A3B MoE for code | 12.65 GB |
| [Kimi-Linear-48B-A3B](https://huggingface.co/infosave/Kimi-Linear-48B-A3B-Code-CMF) | KDA linear attention, MoE | 17.75 GB |
| [Qwen3.6-27B](https://huggingface.co/infosave/Qwen3.6-27Bcmf) | dense, q4tp | 14.26 GB |
| [Qwen3.6-35B-A3B](https://huggingface.co/infosave/Qwen3.6-35B-A3Bcmf) | MoE, q4tp | 18.68 GB |
| [Qwen3.6-35B-A3B-Escha](https://huggingface.co/infosave/Qwen3.6-35B-A3B-Escha-W2cmf) | the same at two bits | 12.87 GB |
| [DeepSeek-V4-Flash](https://huggingface.co/infosave/DeepSeek-V4-Flash-0731-cmf) | 1T-A32B, split into parts | 455 GB |
| [Lumina-Image-2.0](https://huggingface.co/infosave/Lumina-Image-2.0cmf) | text → image, 19 GB tree in 3.2 GB | 6.72 GB |
| [MiniMax-H3 Turbo](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf) | text → video **with sound**, 4 steps | 71.05 GB |
| [LTX-2.5](https://huggingface.co/infosave/LTX-2.5-cmf) | text → video **with sound**, 21B DiT + audio VAE | 20.53 GB |
| [Qwen3.8-27B](https://huggingface.co/infosave/Qwen3.8-27B-cmf) | dense, q4tp | 14.44 GB |
| [Qwen3.8-Flash-Next](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf) | 176.9B hybrid MoE/PLE, mixed q4tp + q8_2f | 97.12 GB |

Every one of them runs from the same binary: `cortiq run` for the text
models, `cortiq imagine` for Lumina, `cortiq animate` for MiniMax-H3, `cortiq ltx-video` for LTX-2.5.

## Where it is honest

The **format** is settled at v2 and evolves additively. The **crate APIs**
may still move before 1.0. Quality claims in the model cards are measured
held-out numbers with the corpus named, not estimates — where a number was
not measured, the card says so rather than guessing, and the container
refuses to record a quality field without a measurement.
