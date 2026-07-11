# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **q4 SDOT decode path** — `q4_block` matvec now runs through the A8W8
  int8 `sdot` kernel on ARMv8.2+ (nibbles → centered i8 per 32-group, exact
  outlier correction), replacing the scalar inner loop. Measured on
  Qwen3.5-0.8B q4 (M4, interleaved runs): decode 5.3 → 14.4 tok/s (×2.7),
  prefill 7.3 → 24 tok/s (×3.3), PPL 4.008 → 4.022 (+0.35%, bounded A8W8
  noise — the same contract as q8/vbit). `CMF_SDOT=0` keeps the exact
  scalar path.

## [0.1.10] — 2026-07-09

### Added

- **Physical defragmentation** — `cortiq convert --defrag <skill_dir>` drops
  pruned FFN neurons so they are neither stored nor computed (Patent 2 claims
  9/10; spec §11). The mask overlay (§5) is virtual sparsity — the full tensors
  stay on disk; defrag bakes one task's keep-set into the weights and emits a
  standalone, smaller dense `.cmf`. Per-layer variable: each layer shrinks to
  its own live-neuron count (no global-max bottleneck). The keep-set comes from
  an explicit `ffn_keep.npy`, or is autodetected from zeroed `down_proj` columns.
  Native Rust (minimal `.npy` reader); masks are dropped; provenance records the
  pre/post neuron counts. FFN output is bit-identical to the masked model before
  quantization.

### Changed

- The FFN dims are derived from tensor shapes throughout; the loader now
  enforces the FFN triple invariant (`gate.rows == up.rows == down.cols`,
  `down.rows == hidden`) loudly, and runtime telemetry reports per-layer neuron
  counts from the actual shapes rather than the nominal `intermediate_size`.

## [0.1.9] — 2026-07-08

### Added

- **Native v-bit quantization** — `cortiq convert --quant vbit` /
  `cortiq import-gguf … --quant vbit` now encode the grouped variable-bit format
  in Rust (no Python): per-row bit-width (3–8, water-filled by log2 row
  amplitude toward a 4.25-bit budget), per-32-group f16 scale, MSB-first packing
  — byte-compatible with the `cortiq-core` v-bit reader. A round-trip unit test
  and a real-model convert→run confirm it (≈40% smaller than q8, coherent
  output). Only the **GPTQ-calibrated** v-bit variant (which needs an activation
  Hessian) still uses the Python converter; the weight-only path is fully native.

## [0.1.8] — 2026-07-08

### Fixed

- **f16 subnormal decode bug** (`cortiq-core`) — `f16_to_f32` computed the
  subnormal exponent as `127-15-e`, one too small, which **halved every
  subnormal half-float**. This corrupted GGUF K-quant super-block scales (which
  are frequently subnormal), producing garbage output. The biased exponent is
  now `113-e`; covered by new round-trip tests. It also slightly affects any
  runtime f16 weight that happened to be subnormal.

### Added

- **Full GGUF quant coverage** in `cortiq import-gguf` — every common ggml type
  is now dequantized natively (no Python): `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`,
  `Q8_0`, the K-quants `Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/`Q6_K`, `Q8_K`, `BF16`, and
  the non-linear-codebook `IQ4_NL` / `IQ4_XS` (used inside `q2_k`/`q3_k` mixes).
  Each codec is a faithful port of ggml `dequantize_row_*`; Q4_K/Q5_K/Q6_K have
  unit tests against fp16 ground truth, and all nine Qwen2.5 GGUF quantizations
  convert and generate coherently. Only the `IQ1`/`IQ2`/`IQ3` grid codebooks
  remain unsupported (an honest error, never silent garbage).
- **`cortiq import-gguf <owner/repo>`** now accepts a Hugging Face repo id (the
  best natively-supported `.gguf` is picked and downloaded in parallel), or
  `owner/repo/file.gguf` for a specific file, or a local path. `--hf-token` for
  gated repos. A linear-attention / SSM (GatedDeltaNet) GGUF is refused with a
  clear message pointing at the safetensors path — never silently mangled.
- **Native fused-GatedDeltaNet split** in `cortiq convert` — qwen3_next /
  AgentWorld checkpoints that fuse the GDN projections (`in_proj_qkvz` /
  `in_proj_ba`, group-interleaved) are split into the canonical hub tensors
  natively, so those models no longer need the Python converter. The split is a
  pure row permutation with a unit test; it is not yet generation-verified on
  real fused weights (no small public fused checkpoint exists).
- A GGUF-only repo passed to `cortiq convert` now returns an actionable error
  (use `import-gguf`, or convert the source safetensors repo) instead of a raw
  404 on the missing `config.json`.

## [0.1.7] — 2026-07-07

### Added

- **GatedDeltaNet linear attention** (Qwen3.5 hub layout) in `cortiq convert` —
  the per-layer linear/full schedule, the canonical GatedDeltaNet core, the
  zero-centered `(1+w)` norms, and the multimodal-wrapper tensor names are all
  handled natively. Validated: Qwen3.5-0.8B converts and generates identically
  to the reference Python converter. Fused qwen3_next / AgentWorld checkpoints
  (interleaved `in_proj_qkvz`/`in_proj_ba`) still use the Python path.

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

[Unreleased]: https://github.com/infosave2007/cmf/compare/v0.1.9...HEAD
[0.1.9]: https://github.com/infosave2007/cmf/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/infosave2007/cmf/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/infosave2007/cmf/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/infosave2007/cmf/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/infosave2007/cmf/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/infosave2007/cmf/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/infosave2007/cmf/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/infosave2007/cmf/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/infosave2007/cmf/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/infosave2007/cmf/releases/tag/v0.1.0
