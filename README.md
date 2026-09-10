Русский: [README.ru.md](README.ru.md) · 中文: [README.zh.md](README.zh.md)

# CMF — Cortiq Model Format

CMF is an auditable model container: one file can hold weights, tokenizer,
chat metadata, task masks and skill overlays for inference without a large
framework runtime.

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Status

CMF v2 is the current on-disk format. Readers validate the envelope, section
bounds, tensor metadata and hashes; incompatible changes require a feature bit
or version bump. The Rust crate APIs are still pre-1.0 and may change.

The project is usable for local inference and format experiments. Treat model
quality and speed numbers as workload-specific measurements, not guarantees;
see the focused guides for the test setup behind each claim.

## Quick start

Install the CLI and convert a small public checkpoint:

```sh
cargo install cortiq-cli
cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --output qwen.cmf
cortiq run qwen.cmf --prompt "What is the capital of France?" --greedy --no-think
```

Already have GGUF? Import it directly:

```sh
cortiq import-gguf model.gguf --output model.cmf
cortiq verify model.cmf
```

The CLI also exposes `info`, `bench`, `ppl`, `serve`, `skill`, `moe-mask`,
`moe-defrag`, `requant`, `compact`, `sign`, `imagine`, `animate` and
`ltx-video`. Run `cortiq <command> --help` for flags and current limitations.

## What a CMF file contains

- A fixed 128-byte envelope addressing every section.
- Header JSON with architecture, quantization defaults, chat metadata and
  provenance.
- A binary tensor directory with dtype, shape, offsets, lengths and `hash64`.
- A page-aligned weight blob that can be memory-mapped and read in place.
- Optional task masks, skill replacement tensors, tokenizer bytes and sparse
  indexes.

`cortiq verify` fails on malformed bounds or a hash mismatch; `python/cmf_reader.py`
is a small independent reader for inspection. The normative layout is in the
[CMF v2 specification](docs/CMF_V2_SPEC.md).

## Quantization

Quantization is selected per tensor, so sensitive tensors can stay at a higher
precision while large matrix blocks use a compact codec.

| Codec | Typical use |
|---|---|
| `f16`, `f32` | norms, embeddings and exact control tensors |
| `q8`, `q8_2f` | high-fidelity weights; `q8_2f` adds input-channel scales |
| `q4`, `q4t`, `q4tp` | general dense and MoE weights |
| `q2tp`, `vbit`, `vbit_ro` | size-constrained or mixed-bit profiles |
| `q1`, `q1t`, `q1s` | trained binary or experimental ternary/PTQ paths |

See [Q1T/PTQ](docs/Q1T_PTQ.md) for the experimental low-bit path and the
[quantization coverage matrix](docs/QUANT_COVERAGE.ru.md) for codec support by
execution path.

## Runtime capabilities

- **CPU inference:** portable Rust implementation with memory-mapped weights.
- **GPU inference:** native Metal on macOS and wgpu backends (Vulkan/DX12)
  where the device and build support them. Use `CMF_GPU=1` to request GPU.
- **Long context:** optional `--o1` attention uses fixed-size state and trades
  memory growth for a measured quality delta; measure on your model.
- **Skills:** one backbone can carry task masks and replacement-tensor overlays;
  inactive overlays do not need a second full model copy.
- **MoE:** task masks and physical defragmentation can reduce the active expert
  set. Validate perplexity on held-out data before shipping a restriction.
- **Speculative decode:** supported MTP/draft paths are enabled only where the
  model metadata and measured acceptance make them useful.
- **Serving:** `cortiq serve` provides an OpenAI-compatible local HTTP API.
- **Media:** `imagine`, `animate` and `ltx-video` pack model assets and run
  image, video or audio-capable pipelines when their model guides apply.

## Platforms and model families

| Target | Support |
|---|---|
| Linux/macOS/Windows | CPU builds; GPU through the available wgpu backend |
| Apple Silicon | CPU and Metal paths; memory is shared with the system |
| NVIDIA/AMD GPUs | Vulkan or DX12 through wgpu where tested |
| Android/iOS integrations | See the companion mobile project and split guide |

Native conversion covers Qwen, Llama, Mistral, Gemma, Phi, DeepSeek, Kimi
Linear, MiniCPM and several MoE/video families. Model-specific constraints and
published files are indexed in the [model guides](docs/hf/README.md). For an
unsupported checkpoint, try `import-gguf` and file a reproducible issue if it
fails.

## Focused documentation

- [CMF v2 specification](docs/CMF_V2_SPEC.md) — normative file layout.
- [Format comparison](docs/COMPARISON.md) — criteria and evidence.
- [Skills](docs/SKILLS.md) — masks, overlays and routing workflows.
- [GPU kernel recipes](docs/GPU_KERNEL_RECIPES.md) — backend measurements.
- [Multi-GPU execution](docs/MULTI_GPU.md) and [mobile split](docs/MOBILE_SPLIT.ru.md).
- [FCD restoration](docs/RUST_FCD.md) and [low-bit PTQ](docs/Q1T_PTQ.md).
- [Model cards and conversion notes](docs/hf/README.md).
- [Cortiq Spectra](docs/SPECTRA.ru.md) — deterministic CPU streaming colorization
  for dual-energy X-ray captures, scanner profiles, refusal masks, and measured
  limits (Russian).

## Build from source

```sh
cargo build --release --workspace
cargo build --release -p cortiq-cli --no-default-features  # CPU-only CLI
cargo test --workspace
```

The workspace is Apache-2.0. Read [LICENSE](LICENSE), [PATENTS.md](PATENTS.md),
[CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) before
redistributing or reporting a vulnerability. Releases and checksums are listed
on the [GitHub releases page](https://github.com/infosave2007/cmf/releases).
