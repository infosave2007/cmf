# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-07-16

The performance release: ten waves of engine work guided by the internal
performance roadmap, verified on three machines (Apple Silicon, Intel AVX2,
Xeon Granite Rapids). First like-for-like run against `llama.cpp` (b9310,
Qwen2.5-0.5B, CPU-only, 8 threads, exact attention both): **pp512 +70%,
tg128 +60%, file −26%**, with quantization quality matched (CMF q8 vs own
f16: +0.38% PPL over 12×512 windows). One model on one machine — the full
matrix is still open; reproduce with `cortiq bench --json`.

### Added

- **x86 SIMD kernels** (the engine previously had explicit SIMD only on
  AArch64): AVX2/FMA i8×f32 and f32 dots, `maddubs` A8W8 int8 path for q8,
  register-level q4 nibble kernels, SIMD unpack for the dominant vbit width
  (B=4), and an AVX-512 VNNI q8 path (bias-trick `vpdpbusd`, four
  accumulators). Runtime-detected; `CMF_AVX2=0` / `CMF_AVX512=0` opt out,
  `CMF_SDOT=0` keeps exact kernels on every architecture.
- **Multi-matrix jobs**: Q/K/V and gate+up projections run under a single
  worker-pool dispatch (`Pool::run_many`, `QTensor::matvec_many` /
  `matvec2_many`) on all codecs, in both the single and the MTP-pair decode
  paths.
- **Fused multi-token q4/vbit kernels**: true `matvec2` (weights unpacked
  once per activation pair) and batched `matmat` (weight row decoded once
  per prefill microbatch) — bit-identical to the per-position kernels.
- **Chunk-GEMM prefill attention**: Q/K/V and O projections run as
  chunk-level GEMMs inside `prefill_batch`; generation now uses the same
  batched prefill as `bench`/`ppl` (the pair path remains for
  dynamic-routing prompts).
- **Grouped exact-GQA attention**: all Q-heads of a KV group stream the
  shared K/V storage once per position (bit-identical per head, covered by
  parity tests in both f32 and q8 KV modes).
- **`cortiq bench --json`** with steady-state counters: allocations/token
  and pool dispatches/token are sampled per token over the same inter-token
  window as the steady tok/s.

### Changed

- **Worker pool rewritten**: shared job slot + atomic epoch + park/unpark
  instead of an `Arc<Latch>` and an mpsc message per worker per matvec; the
  caller joins the work as an extra participant. Dispatch no longer
  heap-allocates (`CMF_POOL_SPIN` tunes the spin budget, default 0).
- **Steady-state allocations cut from hundreds to ~26 per token**: reusable
  norm/projection/FFN buffers, a crate-wide buffer freelist (attention and
  FFN outputs, vocab-sized lm_head logits), allocation-free activation
  splitting, vbit row offsets precomputed at load, `select_nth_unstable`
  sampler top-k with candidates-only top-p.
- Release profile now builds with thin LTO and a single codegen unit.

### Fixed

- Metal no-copy buffers on Macs without unified memory (Intel-era discrete
  GPUs) silently returned stale data — such devices are now refused at
  init with a CPU fallback.
- Batched q4 `matmat` on non-SDOT platforms rounded differently from the
  per-position kernel (flat vs pairwise accumulation) — bit-parity restored.
- `QTensor::from_model` no longer scans the tensor directory linearly
  (O(N²) pipeline build on MoE/skills files).

## [0.2.2] — 2026-07-15

### Added

- **`cortiq ppl --o1 all|deepN|list|off`** (with `--o1-m` / `--o1-window` /
  `--o1-sink` / `--o1-prefill`, `--windows`, `--window-len`) — scores the
  **converted** model through the real streaming kernel and prints the exact
  baseline over the identical tokens next to it. The O(1) path's quality had
  never been measurable natively: the scoring path ran exact attention by
  design, so the only published numbers came from the reference probe, which
  rectifies every estimated far weight individually — a step a streaming
  operator cannot perform — and derives landmarks from the whole scored
  window. Each window's first `--o1-prefill` tokens run the exact pass that
  freezes the landmarks; every scored position then goes through
  `NystromState::step()`, the same code decode runs. The default is
  unchanged: `ppl` scores the backbone exactly even for a model carrying an
  `--o1` hint.
- **`--o1-rect agg|fm`** (and `CMF_O1_RECT`) — selects how the indefinite
  skeleton is rectified. `agg` (default) clamps only the aggregate far
  denominator; `fm` clamps `FM = F_u·M_u` per query row, which is the
  intuitively "correct" per-key guarantee and, measured, the worse one
  (×1.296 vs ×1.414 at the default landmark budget). `agg` wins at every m.
- Prebuilt **Windows** binaries in GitHub Releases — `x86_64-pc-windows-msvc`
  and `aarch64-pc-windows-msvc`, shipped as `.zip` + `.sha256` (the
  convention there) rather than `.tar.gz`; the ARM64 row cross-compiles from
  the x86_64 runner. The runtime needed no porting: Metal is gated behind
  `cfg(target_os = "macos")`, the NEON/SDOT kernels behind
  `cfg(target_arch = "aarch64")`, and `memmap2` covers Windows.
- The release workflow accepts `workflow_dispatch`, so the binaries for an
  existing tag can be rebuilt on demand.

### Changed

- **The O(1) exact window, sink buffer and landmark keys (K̃) are now shared
  per KV group.** The window ring and sink buffer hold the *group's* keys and
  values, and K̃ is `seg_means` over those same keys, so under grouped-query
  attention every query head in a group was storing byte-identical copies.
  `NystromState` is now one state per KV group — a shared `NystromGroup`
  (ring, sinks, K̃, `m_eff`, geometry) plus a `Vec<NystromHead>` for what
  genuinely depends on the head's queries: the far accumulators and their
  running maxima, Q̃, and the mixing matrix `M = pinv(exp(Q̃K̃ᵀ/√d))`.
  Eviction becomes a group event — `advance()` evicts a position once and
  each head then absorbs that key into its own accumulators before the slot
  is reused (one eviction, one insertion per head, which is the invariant the
  partition rests on). **Arithmetic is untouched and the output is
  bit-identical**, proven three ways: a 4-head group and 4 independent
  single-head states agree on `to_bits()`; on a real 4B hybrid, greedy
  generation from a 370-token prompt matches on token ids and top-1
  confidences to 1e-6 — also with a narrow `W=16 m=8 sink=2` window that
  maximizes evictions; and `ppl --o1 all` reproduces to the digit.
  `fcd_runtime_parity` is unmoved at 9.373e-7 against its pinned 9.4e-7. A
  dedicated test asserts each head's `far_len` equals the eviction count and
  closes the books with `far_len + w + sink == t`; it was verified to have
  teeth by injecting a double insert (the bit-identity test alone does *not*
  catch that mutant, since both paths share `advance`). Measured (qwen3_5 4B
  hybrid, 16 q-heads / 4 kv-heads, head_dim 256, `W=128 m=32 sink=4`, Apple
  M4): nystrom state **47.9 → 18.8 MB** (÷2.55), KV+state **153.2 → 124.1
  MB**, and against plain KV at ctx 4096 **÷2.48 → ÷3.06**; the crossover
  where `--o1` starts *saving* memory moves **731 → 287 tokens**.
- **Dynamic row chunking in the thread pool** — `Pool::run_rows` hands out
  row ranges from an atomic cursor instead of a static 1/n split, so a
  performance core takes several chunks per efficiency-core chunk instead of
  waiting at the latch; on an asymmetric-core machine the cores no longer
  wait on each other. Rows stay disjoint, so output is bit-identical.
  Measured: weight-path bandwidth 54.5 → 58.9 GB/s (+8%), decode +4–5% at
  every thread count on a 4B q8_2f model.
- **Corrected O(1) conversion quality figures.** Measured through the shipped
  streaming kernel on held-out wikitext, landmarks sealed from a 256-token
  prefill, scoring only the drift rows (the harshest region): Qwen3-0.6B with
  28/28 layers converted ×1.296; a Qwen3.5-4B hybrid with 8/32
  converted ×1.132. The ×1.177 previously in the docs was the reference
  operator with whole-window landmarks — an upper bound this runtime cannot
  reach by construction. Corrected in the module docs and the
  `convert --o1` help.

- `cortiq run` defaults to the `warn` log level — the loader's INFO lines are
  noise in front of an answer. `RUST_LOG` overrides; every other command
  keeps `info`.
- `convert` / `import-gguf` paint one in-place progress line on a terminal
  instead of several hundred `@PROGRESS` lines. The markers are byte-for-byte
  unchanged when stdout is not a terminal, which is where supervisors parse
  them.

### Fixed

- **`cortiq run` is a chat again.** It advertised "Interactive chat mode" but
  never rendered the container's chat template — `generate()` encodes the
  prompt verbatim — and `generate_from_ids` clears the KV cache per call
  ("Fresh sequence"), so the interactive loop carried no history either. The
  first command a new user runs answered correctly and then repeated "The
  answer is correct." until `max_tokens`; `finish: stop` was unreachable,
  because raw completion never emits `<|im_end|>`. `run` now renders the
  file's template through `apply_chat_template_opts` — the same call the
  server makes — and carries the conversation across turns. The gate is
  `chat_template.is_some()`, **not** the template call itself: with no
  template that helper falls back to hardcoded ChatML, which is not what a
  base model wants, so those still run completion — as does `--state`, whose
  frozen prefix is a raw token replay. A long chat drops its oldest exchange
  (never a system turn) rather than prefill past the RoPE range.
  - `--raw` — skip the template: the previous behavior, verbatim.
  - `--no-think` — render with `enable_thinking=false`; Qwen3/3.5 answer
    directly instead of emitting a `<think>` block.

- **`cortiq fcd` polished an operator the runtime never serves** — the
  trainer built its far field from whole-window landmarks and the per-(t,j)
  clamp. It now seals landmarks from a prompt prefix (`NysCfg.prefill`,
  default `t/2` — the same discipline `ppl --o1` uses), derives `m_eff` from
  the sealed prompt, runs the aggregate far-denominator guard with raw
  negative mass kept on passing rows, and leaves pre-seal rows exact. A new
  `fcd_runtime_parity` test pins the trainer forward against the live
  `NystromState` at 9.4e-7 (tol 2e-5), while the per-key rectifier differs by
  5.7e-2 on the same fixture — the test cannot pass a trainer that reinstates
  the clamp. The trainer-reported zero-shot ratio moves ×1.168 → ×1.146 on
  its own windows (teacher identical, a clean control).
- **`o1_seal`** now requires `num_heads % num_kv_heads == 0` and degrades to
  exact attention instead of panicking on an index overflow.

## [0.2.1] — 2026-07-14

### Added

- **`enable_thinking`** — `/v1/chat/completions` accepts `enable_thinking`
  (top-level) or the vLLM-style `chat_template_kwargs.enable_thinking`.
  `false` renders the chat template with `enable_thinking=false` — Qwen3/3.5
  prefill an empty `<think>` block and answer directly. Absent = the
  template's default. The tokenizer gains `apply_chat_template_opts`; the
  render context defines the variable only when it is set.

### Changed

- README: an O(1) conversion quick-start — the `convert --o1` commands, the
  `run` / `serve` / `bench` overrides, `CMF_O1`, the tuning knobs, and the
  `cortiq fcd` polish stage.

### Fixed

- **Corrupt published crate tarball** — `cargo package` deterministically
  corrupted the tarball on the previous `README.md` byte layout; a trailing
  newline works around it.

## [0.2.0] — 2026-07-14

### Added

- **O(1) constant-memory streaming attention conversion** — `cortiq convert
  --o1 all|deepN|list` (with `--o1-m` / `--o1-window` / `--o1-sink`) converts
  any softmax checkpoint to per-layer O(1) attention in seconds, with the
  **weights byte-identical**: the conversion records a hint in provenance and
  the binary envelope is unchanged. The kernel (`nystrom.rs`) is an exact
  sliding window plus a PSD far-field skeleton under a single joint
  denominator, with permanent sink tokens (the first `S=4`, which never enter
  the far field), per-landmark flash-style running-max accumulators, and
  delayed insertion — a key enters the far state only when it leaves the
  exact window. Guards: short-prompt exact mode, `m_eff = clamp(T/8, 4, m)`,
  and a ridge pseudo-inverse (f64 Cholesky) with jitter fallback. At runtime
  prefill runs exact attention, then `seal()` builds the landmarks and `M`
  per head, replays the prompt into the state and **drops the layer's full
  KV**; seal refuses on q8 KV and masked-sparse heads, the speculative pair
  path is disabled under o1, and eviction no-ops on sealed layers. Dispatch
  priority: CLI > `CMF_O1` env > the `provenance.o1_attn` header hint. Golden
  parity vs the validated reference math: max 1.1e-6 (sink=4). Measured (M4,
  Qwen3-0.6B q8, `--o1 all`): ctx 4096 decode 19.6 → 68.6 tok/s (×3.5) at
  84.9 MB constant state vs 954 MB KV (÷11.2); ctx 1024 ×1.5 / ÷2.9 — decode
  is near-flat in context length. (The zero-shot quality ratios published
  with this release came from the reference probe rather than the shipped
  kernel; corrected in 0.2.2.)
- **Native FCD restoration trainer** — `cortiq fcd <model.cmf> --corpus …`
  (`--steps`, `--eval-every`, `--kl`, `--gen-check`, `--gen-gate`,
  `--gate-threshold`, `--gate-slack`, `--out`): the bounded KL-anchored
  polish stage for `--o1` conversions, with **no ML framework** — one binary
  end to end. `fcd_ops.rs` is a fixed-graph op library with hand-derived
  backwards over an `Fp` trait (pooled f32 GEMMs, RMSNorm plain and
  zero-centered, RoPE, SwiGLU, segment means, exact causal attention,
  Nyström-joint attention, GatedDeltaNet BPTT, and CE + KL(teacher‖student)
  loss); every op carries a central finite-difference gradcheck (rel err
  1e-9…1e-12; whole-graph block checks ≤ 8.9e-4; GDN forward parity vs the
  runtime kernel 3.4e-8). Teacher and student share one frozen mmap and the
  trainable set is only the normalization gains and FFN tensors of converted
  layers (AdamW, grad clip, deterministic held-out eval, best-checkpoint
  restore, `provenance.fcd` on the written tensors). **Generation-gated
  selection**: each eval probes greedy long-context generation through the
  real streaming kernel and admits a checkpoint only if no prompt loops — if
  none passes, the zero-shot state is restored, so the stage cannot make
  generation worse than conversion alone. The motive is measured: on a
  6/24-softmax hybrid, ppl-only selection reached ×0.86 teacher ppl yet
  regressed all three generation probes into loops.
- **hybrid_k core support** — the vmf_phase linear core now honors an
  optional selective-write gate: `model.layers.{i}.vmf_attn.k_gate.weight`
  `[nh, hidden]` + `.bias [nh]`; κ_h = σ(W_k·x + b)_h multiplies the state
  write (`S = decay·S + κ·φk⊗v`). Presence-driven: files without the
  tensors run the classic phase kernel unchanged. Mechanism-level basis
  («phase + input gate», stage 71): fastest convergence and best/tied
  accuracy across the recall grid, correlated-noise robustness the bare
  phase kernel lacks, and an LM crossover vs softmax at SEQ 512.

- **NEON decode attention** — `attention_head` score/weighted-sum loops and
  the q8-KV `attend` branches now run through NEON kernels (`dot_f32`,
  `axpy_f32`, per-group `dot_i8_f32`, `axpy_i8_f32`). Measured on
  Qwen3-0.6B q8 (28 full-attention layers, teacher-forced 1536 tokens,
  interleaved rounds): **×1.61 wall-time** (29.5 s → 18.3 s); the gain grows
  linearly with context depth. PPL 22.053 → 22.084 (+0.14%, summation
  regrouping only).
- **Long-context bench mode** — `cortiq bench --ctx N` builds a synthetic
  N-token prompt, raises `CMF_MAX_SEQ` so eviction cannot skew the curve,
  and prints `Memory: KV+state X MB at seq_len N` (O(context) KV for
  full-attention vs O(1) state for the linear core, measured).
- Hot-path hygiene: `row_dot` (active-neuron path) NEON for q8_row/q8_2f
  (new `dot_i8_col_f32` folds the θ col-field without a prescaled copy);
  vbit SDOT per-row heap allocation replaced by a per-worker scratch
  (lm_head ≈ 150k rows/token); `prescale` returns borrowed activations
  for non-q8_2f dtypes (was an unconditional copy per matvec). Short-ctx
  q4 decode +4% (64.0 vs 61.6 tok/s, interleaved).

- **q4 SDOT decode path** — `q4_block` matvec now runs through the A8W8
  int8 `sdot` kernel on ARMv8.2+ (nibbles → centered i8 per 32-group, exact
  outlier correction), replacing the scalar inner loop. Measured on
  Qwen3.5-0.8B q4 (M4, interleaved runs): decode 5.3 → 14.4 tok/s (×2.7),
  prefill 7.3 → 24 tok/s (×3.3), PPL 4.008 → 4.022 (+0.35%, bounded A8W8
  noise — the same contract as q8/vbit). `CMF_SDOT=0` keeps the exact
  scalar path.

### Fixed

- **The `bench` memory line under-reported a fully-folded model** — an
  all-linear model reported `KV+state 0.0 MB` because the recurrent state
  (f64, constant in context) was not counted. Both cache kinds are now
  honest: the folded 0.6B reports its analytic 58.7 MB constant state against
  242 → 946 MB of growing KV for the softmax original.
- **The `x86_64-apple-darwin` release binary is published again** — the
  retired `macos-13` runner pool left the Intel job queued with zero steps
  for 24 h before being auto-cancelled, losing that asset on v0.1.8, v0.1.9
  and v0.1.10. It now cross-compiles on `macos-latest`, with a 30-minute
  timeout so a stuck pool fails loudly instead of silently dropping the
  binary.

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

[Unreleased]: https://github.com/infosave2007/cmf/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/infosave2007/cmf/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/infosave2007/cmf/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/infosave2007/cmf/compare/v0.1.10...v0.2.0
[0.1.10]: https://github.com/infosave2007/cmf/compare/v0.1.9...v0.1.10
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
