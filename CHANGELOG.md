# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Native GatedDeltaNet (Qwen3.5 linear-attention) conversion — currently still
  uses the Python path.

## [0.1.6] — 2026-07-07

### Added

- **`cortiq import-gguf <file.gguf> --output model.cmf`** — a native Rust GGUF
  importer (F32 / F16 / Q8_0; llama / qwen2 / qwen3), which also reconstructs a
  Hugging Face tokenizer.json from the embedded ggml metadata. No Python.
  K-quants (Q4_K / Q5_K / Q6_K) still use the Python importer.
- **Mixture-of-experts** in `cortiq convert` — the router and per-expert matrices
  are converted and the runtime dispatches the sparse FFN (qwen2-moe / qwen3-moe).

## [0.1.5] — 2026-07-07

### Added

- `cortiq convert --quant q8_2f` — the two-field (𝒲×θ) int8 quantization that
  recovers most of the int8→fp16 quality gap at the same file size.
- Converter round-trip tests (q8 / q8_2f / q4 encoders + a tiny end-to-end
  convert) run in CI.
- A release workflow that attaches prebuilt `cortiq` binaries (Linux x86_64,
  macOS arm64 / x86_64) to each GitHub Release — usable with no Rust toolchain.

### Changed

- **Byte-faithful, lighter conversion**: round-half-to-even quantization (matches
  numpy — weights are now byte-identical to the reference converter), and the
  input safetensors are memory-mapped and processed one tensor at a time, so peak
  RAM is ≈ the output size rather than the whole model.
- **Resilient downloads**: each byte-range chunk retries with exponential backoff
  and shows a live percentage.

## [0.1.4] — 2026-07-07

### Added

- `cortiq convert --model <owner/name>` now accepts a **Hugging Face repo id**
  directly and downloads it (config, tokenizer, and safetensors weights) before
  converting — the whole HF → `.cmf` pipeline lives in one place, no external
  tooling. `--hf-token` for gated/private repos.
- **Parallel downloads**: weight files are fetched in concurrent 32 MiB
  byte-range chunks over reused connections (saturates bandwidth for both a
  single large file and sharded models). Tunable via `CORTIQ_HF_THREADS`
  (default 8). Downloads are cached under `~/.cache/cortiq/hf`.

## [0.1.3] — 2026-07-07

### Added

- **`cortiq convert`** — a native Rust converter from a Hugging Face checkpoint
  (`config.json` + `*.safetensors` + `tokenizer.json`) to `.cmf`, with **no
  Python / numpy / torch dependency**. Reads safetensors and quantizes in Rust
  (q8 / q4 / f16), embeds the tokenizer and chat template, and writes via
  `cortiq_core::CmfModel::write`. Standard dense transformers (qwen2 / qwen3 /
  llama / mistral-style); output is generation-identical to the reference
  Python converter. MoE / linear-attention models still use the Python path.

## [0.1.2] — 2026-07-07

### Added

- `cortiq serve --host <HOST>` to control the bind address (default `0.0.0.0`;
  set `127.0.0.1` for a local-only server).
- A `/healthz` liveness endpoint on the server — for process managers that embed
  `cortiq serve` as a local model backend (e.g. an LLM gateway).

## [0.1.1] — 2026-07-07

### Added

- `cortiq run --max-tokens <N>` (short `-n`) to cap the number of generated
  tokens (default 256); previously the generation length was fixed at 256.

## [0.1.0] — 2026-07-07

Initial public release.

### Added

- **`cortiq-core`** — the CMF v2 on-disk format: 128-byte envelope, section
  table, memory-mappable tensor directory, tokenizer and chat-template records,
  per-task mask records, and per-skill full-shape replacement-tensor delta
  records with a byte-offset delta index.
- **Quantization codecs** — including the two-field `q8_2f` (scale × phase)
  path and v-bit stacking, with golden round-trip and parity tests.
- **`cortiq-engine`** — a dependency-free runtime that memory-maps a container
  and runs inference on **CPU or GPU**. Overlay execution reads per-skill
  replacement tensors *in place of* the shared backbone at forward time without
  materializing a separate model. Optional `gpu` feature uses a portable wgpu
  backend (Vulkan / Metal / DX12) with CPU/GPU parity.
- **`cortiq-server`** — an optional axum-based HTTP serving layer.
- **`cortiq-cli`** — the `cortiq` command-line binary for inspecting, converting,
  and running containers.
- **Converters** — self-contained Python tooling to produce `.cmf` files from
  source models, plus a pure-Python reader for inspecting containers.
- **Documentation** — the CMF v2 specification and a comparison against GGUF,
  safetensors, ONNX, PyTorch `.pt`, GGML, and TensorRT, in English, Russian,
  and Chinese.
- **Developer tooling** — `Makefile` and `justfile` shortcuts, a pinned
  `rust-toolchain.toml`, GitHub Actions CI (build + test on Linux and macOS,
  clippy, rustfmt), and contributor / community-health docs
  (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR templates).
- **Licensing** — Apache-2.0 with an explicit patent-grant explanation
  (`LICENSE`, `NOTICE`, `PATENTS.md`).

[Unreleased]: https://github.com/infosave2007/cmf/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/infosave2007/cmf/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/infosave2007/cmf/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/infosave2007/cmf/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/infosave2007/cmf/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/infosave2007/cmf/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/infosave2007/cmf/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/infosave2007/cmf/releases/tag/v0.1.0
