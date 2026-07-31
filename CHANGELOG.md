# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.42] — 2026-07-31

The Qwen3.6 release. The hybrid MoE flagship (35B-A3B) and the dense 27B run
whole-token on discrete GPUs — 64.5 tok/s decode on an RTX 5090, 99 tok/s
batched prompt ingest — keep their multi-token-prediction heads through
conversion, and O(1) Nyström attention now rides the GPU graph instead of
dragging the model back to the CPU.

### Added
- **Qwen3.6 MTP heads survive conversion.** The converter keeps `mtp.*`
  (renamed to the loader layout) instead of dropping it, and the loader
  builds MoE FFNs inside the MTP block — previously it only knew dense
  heads, so an MTP-bearing MoE file failed to load.
- **O(1) attention on the wgpu graph** (`CMF_O1_GPU=1`): the sealed
  landmark skeleton, far-field flash accumulators and FM rectifier run as
  graph kernels. Qwen3.6-35B-A3B holds a flat 53.8 tok/s at ctx 16 384
  where exact attention has fallen to 37.8.
- **Batched prompt ingest through the graph** (`CMF_BATCH_K`): token-axis
  MoE kernels route each position individually while the surrounding
  skeleton stays batched — 99 tok/s ingest against 33 per-position.
- Metal: the whole MoE block in four dispatches (M4 35B-A3B decode
  10.6 → 18.2 tok/s), with a loud `maxBufferLength` refusal.
- Measurement probes for the decode frame: `CMF_TOPK_PROBE`,
  `CMF_SKIP_PROBE`, `CMF_LAYERS_PROBE` — measure before optimizing.

### Fixed
- **q4t/q4tp files never entered the whole-token graph** — a hole in the
  layout prep silently sent every quantized-tile model to the CPU.
- **The graph's GDN device state started zeroed after a CPU prefill**
  (o1 calibration, `--state` resume): now seeded from the CPU recurrence.
- The OpenAI-compatible server accepts `content` as a block array as well
  as a string (Cline/Zoo-style clients got 422).
- `bench --ctx` decoded without the synthetic context it had just built.
- CPU MoE: every routed expert now runs under one pool dispatch instead
  of one dispatch each.

### Changed
- Graph refusals are loud: the graph logs ACTIVE/declined with the
  reason, and the VRAM budget prints the numbers it compared
  (`RUST_LOG=info`).

## [0.5.41] — 2026-07-30

A hotfix: **0.5.40 shipped a Metal backend that does not initialize.** Update.

### Fixed
- **The q4tp GEMMs referenced a variable that 0.5.40 deleted.** Hoisting the
  per-row scale out of the K loop removed `uint wr`, and two lines still used
  it. The MSL then fails to compile, `ctx()` returns None, and every model on
  macOS silently falls back to the CPU — not just q4tp ones.

  It reached a release because the parity test returned early when
  `gpu_metal::enabled()` was false, which is exactly what a broken shader
  causes. A test that skips on the condition it is meant to detect is worse
  than no test; it now asserts the backend came up.

- **Packed 3-D expert tensors doubled the `mlp.` prefix for the qwen
  lineage.** gemma-4 names the tensor `layers.N.experts.gate_up_proj` and
  qwen3.5/3.6 `layers.N.mlp.experts.gate_up_proj`; the splitter stripped the
  suffix and appended `.mlp.experts`, so the qwen form came out as
  `layers.N.mlp.mlp.experts.0.*`. Conversion succeeded — valid file, hashes
  matched — and the model failed at load with "router present but no expert
  tensors", which points nowhere near the naming. Qwen3.6-35B-A3B converts
  and runs now.


## [0.5.40] — 2026-07-29

q4tp reaches parity on the image path, plus the AWNP tooling and the
measurements that closed it. Measured on a fanless MacBook Air M4.

### Added
- **`cortiq awnp`** — activation-weighted nullspace projection: drop the
  weakest input channels of every MoE expert and refit the survivors through
  `C[:,S]·C[S,S]⁻¹` so they absorb what was removed. `--drop`, `--ridge`,
  `--rescale`. Output keeps the original shapes, so the result runs on
  today's runtime and the quality question is settled before any format
  change.
- **`CMF_RMS_TRACE`** (per-channel activation energy) and **`CMF_ACT_DUMP`**
  (raw FFN-input rows) — the calibration AWNP needs. Nothing in the runtime
  produced activation statistics before, so "is the flat weight-energy curve
  a property of the weights or of the criterion" could only be asserted.
- **q4tp in the Russian and Chinese README and spec.** The English pair had
  the new dtype and the localized ones did not.

### Fixed
- **The q4tp GEMMs read their per-row scale inside the K loop.** `lo`/`step`
  are loop-invariant — a thread's row is fixed across the whole loop — so
  that was a dependent chain (load → exp2 → scale) paid per staged tile for
  nothing. On Lumina's DiT the runtime probe measured the q4tp GEMM as the
  slower arm and routed the whole render to the CPU: 18 s against q4t's 13.
  Hoisted, interleaved at 256px/8 steps: 26 and 27 s for q4tp against 27 and
  26 for q4t. The image is unchanged at 43.6 dB PSNR.
- **The fused DiT SwiGLU chain had no q4tp twin**, so a q4tp image model
  shipped its `[b, inter]` intermediates across the CPU boundary twice per
  layer — 28 s before the twin, 18 s after, parity after the hoist above.
- **`sgemm_public` did not compile off macOS/aarch64.** `sgemm_rm` exists
  only where there is an f32 GEMM to call; x86 has quantized kernels and no
  such path. A portable triple loop now backs it — only the offline AWNP
  pass reaches it, where correctness matters and throughput does not.

### Notes
- **AWNP measured and not productized.** On KAT-Coder-V2.5 (qwen3_5_moe),
  25000 calibration rows over 2048 channels from hundreds of distinct
  dialogues, evaluated on a held-out half sharing no record with the
  calibration: baseline PPL 23.4, −12.5% channels 50.1, −25% channels 194.8.
  Widening the calibration from 2.9 to 12.2 samples per channel and adding a
  real ridge halved the damage, and the patent's variance-preserving rescale
  does not help — a least-squares refit already returns the minimum-error
  weights, so scaling them back up moves away from that optimum. The
  projection alone, without a healing pass, does not transfer out of sample.
  A skill overlay is the better lever for the same goal.


## [0.5.39] — 2026-07-29

Same code as 0.5.38 plus the documentation it should have shipped with.
0.5.38 was tagged but never published to crates.io: its CI was red because
`examples/q4tp_gpu_ab.rs` referenced the Metal module unconditionally and so
failed `--all-targets` on Linux, and the README and spec said nothing about
the new dtype. Both fixed here; 0.5.39 is the release to use.

### Added
- **`q4tp` in the README** — what it costs, what it saves, and `requant` for
  files already published.
- **`q4tp` in `docs/CMF_V2_SPEC.md`** as dtype 15, with its byte layout and
  the two encoder requirements (round `lo`/`step` to f16 before choosing
  codes; quantize nibbles against the reconstructed scale) that keep a
  writer and a reader on the same scale.

### Fixed
- `examples/q4tp_gpu_ab.rs` is a stub outside macOS. It times Metal encoders
  directly; the wgpu kernels are covered by `tests/gpu_q4tp.rs`, which runs
  on every backend.


## [0.5.38] — 2026-07-29

A new weight layout and the kernels to run it, plus a negative result that
is the reason the layout looks the way it does. Measured on a fanless
MacBook Air M4 (Metal and, via `CMF_GPU=wgpu`, the WGSL path).

### Added
- **`q4tp` (dtype 15) — q4_tiled with predicted scales.** A q4t tile spends
  16 bits on a standalone f16 scale for 32 weights: 0.5 of its 4.5
  bits/weight, 11% of the file. q4tp keeps the nibbles byte-identical and
  makes that scale a 5-bit rung on a per-row geometric ladder
  (`[nibbles][row (f16 lo, f16 step)][5-bit codes]`), for 4.17 bits/weight.

  Quantizing the SAME fp32 weights both ways, q4tp's error against the
  source is 9.71% where q4t's is 9.71% — +0.1% relative at the median
  within-row scale spread measured on KAT-Coder-V2.5 (1.27 in log2), +0.3%
  at its 90th percentile, +0.8% past the tail. A coarser scale costs almost
  nothing because the nibbles simply re-round against it; the 4-bit grid
  dominates either way. Nanbeige-3B keeps its top-5 next tokens in the same
  order with logits within 0.7%.

  Design points, each settled by measurement rather than taste: the row's
  `lo`/`step` are rounded to f16 FIRST and the codes chosen against those
  values, so `lo`'s rounding is absorbed by the code instead of stacking on
  it; codes round to nearest, not up (1.19% vs 1.60% RMS — letting the top
  tile clip one nibble beats coarsening the whole row); a per-tensor Lloyd
  codebook was tried and rejected (0.0576% vs 0.0591% NMSE at 4 bits, since
  within a row the log-scales are near-uniform); and `lo`/`step` come from
  the row's exact min/max, so no code is ever out of range and the format
  needs no escape hatch.

- **`cortiq requant <model> --output <out> --quant q4tp`** — recodes a
  published `.cmf` in place, without the original checkpoint. The
  checkpoints behind published CMF files are tens of gigabytes;
  re-converting them is the expensive path this avoids. KAT-Coder-V2.5:
  12.65 → 11.80 GB (19254 tensors) in 2 min. Nanbeige-3B: 2.36 → 2.19 GB,
  peak RSS 2.42 → 2.19 GB. `convert --quant q4tp` also works from HF.

- **q4tp kernels on every path.** CPU: scalar, sdot/avx2/vnni int8, blocked
  1×4, Accelerate sgemm, and the fused SwiGLU pair. Metal: `q4tp_matvec`,
  `q4tp_mul_mm`, `q4tp_mul_mm_silu`, plus the chunk-graph plumbing. wgpu:
  `q4tp_matvec` and `q4tp_mul_mm` under codec kind 6 — deliberately not
  sharing q4t's 5, since sharing a kind is exactly how q4t once got fed to
  the q4_block kernel and produced garbage on real Vulkan.

  The layout suits a GPU better than q4t's: a 16 B nibble stride is
  4-aligned, so Metal loads four uints where q4t needs nine unaligned
  ushorts and WGSL reads words straight off the storage array. Decode is at
  parity model-level (Metal 12.4 vs 12.4 tok/s) and the kernels themselves
  are faster — 0.100 vs 0.117 ms on o_proj, 23.8 vs 26.2 ms for a cold
  sweep over all 156 weight tensors. The batched GEMM is faster on both
  backends at b=64: Metal 9.51 vs 10.13 ms (2096 vs 1968 GFLOP/s), wgpu
  9.83 vs 10.76 (2027 vs 1853).

### Notes
- **MoE dictionary compression: gate failed, and the mechanism is known.**
  The hypothesis was that fine-grained MoE experts are one function with
  local variations — keep one as a base, code the rest as residuals against
  a shared dictionary. On KAT-Coder-V2.5 (~202 experts/layer) it does not
  hold: `sigma(W_e − W_base)/sigma(W_e)` is 1.34–1.38 against 1.414 for
  independent matrices, so the residual is WIDER than the weight and a base
  costs more bits than it saves. After optimal bipartite matching of neuron
  triplets (gate-row ⊕ up-row ⊕ down-column, 6144 dims) the aligned cosine
  is 0.037–0.047 against a chance level of 0.045 — there is no permutation
  bringing experts together, so tuning the metric, block shape or clustering
  has nothing to find. Load balancing decorrelates experts on purpose.

  What survived is the only redundancy that measured real: a shared
  per-channel scale profile (neuron-profile correlation 0.000 across
  experts, channel-profile +0.16…+0.62). That is what q4tp exploits — and
  per-row coding turned out to subsume the channel profile entirely, which
  is why the format has no profile plane.

  Structural pruning by raw magnitude is also empty here (dropping the
  weakest 12.5% of neurons costs 11% of Frobenius energy — proportional).
  That is NOT a verdict on AWNP, whose criterion is activation-weighted; no
  calibration traces were run.

## [0.5.37] — 2026-07-29

A batch of "the kernel was already there, nobody wired it up" and two
copies of the same defect in two languages. Measured on an RTX 3090 with
a 256-core EPYC beside it, and on a fanless MacBook Air M4.

### Added
- **DiT attention on wgpu** (`gpu::dit_attention` had a Metal arm and
  nothing else, so Vulkan ran the whole attention block on the CPU).
  Three WGSL kernels on the tile machinery the quantized GEMMs use —
  scores, row softmax, P·V — plus a panel unstack, all in one
  submission with the intermediates resident.
- **Causal chunk attention on wgpu** for LLM prefill. The CPU twin needs
  Accelerate or the NEON micro-GEMM, so x86 had no batched attend at all
  and fell back to a per-position scalar loop.
- **Fused QKV** for the batched attention path: one upload of the normed
  chunk, three GEMMs, one readback.

### Fixed
- **The tiled WGSL GEMMs kept their register block in memory.** All four
  (q8, q1, q1t, q4t) declared the 4×4 accumulator as
  `array<array<f32,4>,4>` and indexed it with loop variables; a
  dynamically indexed private array does not stay in registers on this
  backend. The GEMM ran at 373 GFLOP/s of an RTX 3090's ~35 TFLOP/s.
  Sixteen named scalars instead — bit-identical output, and the stage
  itself went ×4.1.
- **The same defect in AVX2**: the blocked q4t kernel folded each group's
  eight i32 lanes to a scalar inside the loop, 288 cross-lane reductions
  per weight row. Four f32 accumulator vectors and one reduction per row.
  Also more accurate — perplexity 9.777 → 9.572, because summing 72
  scalars in sequence loses more than accumulating in eight lanes.
  (`avx2` does not imply `fma`: without it declared, LLVM lowered
  `_mm256_fmadd_ps` to a libm call per lane and the "optimized" kernel
  ran 2× slower than the one it replaced.)
- **The batched dense FFN never used the fused GPU kernel**, so prefill
  paid three round trips per layer and shipped the 22 MB gate/up panels
  across the bus for nothing. The kernel had existed since the image DiT.
- **The default thread cap left big machines idle.** `min(available−1, 8)`
  meant eight threads on a 256-core box. Raised to 32 — a ceiling, not a
  target: decode is memory-bound and collapses past it (14.8 tok/s at 32,
  5.5 at 64, 1.6 at 256). Machines with nine cores or fewer, and
  heterogeneous ARM, are untouched.
- **A coin flip decided whether image generation ran at half speed**: the
  fused DiT paths gated themselves on a per-op probe that is a tie on
  Metal (2.62 vs 2.56 ms) and mere overhead on a discrete card. They now
  trust the device on native Metal and on discrete wgpu adapters;
  integrated and mobile keep probing.

### Changed
- Lumina's caption embedding and context refiner are hoisted out of the
  denoise loop — they depend on the prompt alone, so 30 CFG steps were
  evaluating 60 times a value with two distinct instances. Bit-identical,
  and honestly small (~0.25% of a step).
- `cortiq imagine`'s help caught up with reality: it still said "CPU f32
  (experimental; CMF packaging comes later)" long after `imagine-pack`
  shipped and the DiT moved onto the device.

### Measured
Lumina-Image 2.0 at 512², two steps, RTX 3090: **67.7 s → 24.75 s**.
Nanbeige 4.2 on the same box: Vulkan prefill **15.1 → 25.0 tok/s** at
ctx 512 and **15.0 → 23.0** at ctx 1024; CPU decode **7.8 → 14.3**, CPU
prefill **11.8 → 16.7**. Perplexity is unchanged at every step.

## [0.5.36] — 2026-07-28

### Fixed
- **A coin flip decided whether image generation ran at half speed.**
  The fused DiT paths (whole block, all-heads attention, SwiGLU FFN)
  gated themselves on the wide-GEMM probe, which asks "is ONE wide
  matmat faster on the GPU" — and for Lumina that is a tie: 2.62 ms GPU
  against 2.56 ms CPU, landing on either arm run to run. But a fused
  block's advantage is not per-op speed, it is that the hidden state,
  the packs and the attention panels never leave the device. Measured
  on a fanless MacBook Air M4, 512² Lumina-Image 2.0: a step takes
  ~4.5–5.8 s when the probe picks the GPU and **7.7–9.0 s when it picks
  the CPU**, and it picked the CPU in a third to two thirds of runs. On
  native Metal the fused paths now trust the device
  (`gpu::fused_block_trusted`); other backends keep probing, where the
  submit latency is real. Renders are bit-identical, just no longer
  randomly half-speed.

### Changed
- Lumina's caption embedding and its context-refiner blocks are hoisted
  out of the denoise loop. They depend on the prompt alone — not the
  timestep, not the latents — so 30 CFG steps evaluated 60 times what
  has two distinct values. Output is bit-identical; the saving is small
  (~0.25% of a step: the caption is ~60 tokens against 1024 image
  tokens), this is about not recomputing a constant.

## [0.5.35] — 2026-07-28

### Added
- **q4_tiled weights in the Metal chunk-prefill graph.** The graph
  required `q8_row` on all seven projections of a layer, so every q4t
  model — the whole Nanbeige/Bonsai class — silently fell back to the
  CPU prefill. An empty `row_scale` now marks q4t per weight, and the
  new `q4t_mul_mm_silu` kernel gives the fused down-GEMM its q4t twin.
  A non-`q8_row` embedding matrix no longer refuses the whole run
  either: the CPU fills the hidden and the graph starts from it.
  Nanbeige 4.2 on M4: 512-token prefill 6.1 s → 2.8 s (≈185 tok/s),
  1024-token 13.1 s → 6.6 s.
- `CMF_GQA_SPLIT` — how many simdgroups may split one Q-head's
  positions in the Metal decode attend (default 8).

### Fixed
- **Decode collapsed with context depth on Metal.** Two causes, both in
  `gqa_attend`: its Born-importance pass recomputed the QK dot with one
  position per lane, so each lane walked a whole K row and the reads
  never coalesced; and the kernel ran one simdgroup per Q-head, putting
  only `num_heads` simdgroups on the device — nowhere near enough to
  hide the per-position `simd_sum` latency. The importance pass now
  reuses the main loop's lane-sliced layout, and the attend is
  flash-decoding shaped (one threadgroup per head, its simdgroups
  splitting the positions, partials combined through threadgroup
  memory). Nanbeige 4.2 on M4, decode: 9.1 → 13.8 tok/s at ctx 512,
  6.0 → 11.6 at ctx 1024. `chunk_attend` got the same importance fix.
- **Looped Transformers prefilled one position at a time.**
  `graph_prefill_preferred()` sent them through the decode graph on the
  grounds that it beat the CPU *pair* path; the real competitor is the
  batched chunk-GEMM, which amortizes each weight over the whole chunk
  (512-token prompt: 85 tok/s chunked vs 14 through the graph). TTFT at
  ctx 1024 fell from 107 s to 7.9 s.
- **Metaspace `prepend_scheme` in the PRE-tokenizer was ignored.** The
  leading `▁` was read only off a `Prepend` normalizer (the llama
  shape), so tokenizers that carry Metaspace in the pre-tokenizer with
  a null normalizer — Nanbeige 4.2 — lost it: every raw prompt encoded
  `'Hello'` where HF encodes `'▁Hello'`, and `decode` kept the leading
  space the decoder's `Strip` removes. Chat prompts hid it (they open
  with an added token, so no section is at offset 0 and neither side
  prepends). Nanbeige perplexity over the same text: 10.13 → 9.81.
  `tokenizer_parity` is green on Nanbeige 4.2, gemma-3n-E4B(-it),
  gemma-4-26B and MiniCPM3.

### Added
- **q4_tiled weights in the Metal chunk-prefill graph.** The graph
  required `q8_row` on all seven projections of a layer, so every q4t
  model — the whole Nanbeige/Bonsai class — silently fell back to the
  CPU prefill. An empty `row_scale` now marks q4t per weight, and the
  new `q4t_mul_mm_silu` kernel gives the fused down-GEMM its q4t twin.
  A non-`q8_row` embedding matrix no longer refuses the whole run
  either: the CPU fills the hidden and the graph starts from it.
  Nanbeige 4.2 on M4: 512-token prefill 6.1 s → 2.8 s (≈181 tok/s),
  1024-token 13.1 s → 6.6 s.
- `CMF_GQA_SPLIT` — how many simdgroups may split one Q-head's
  positions in the Metal decode attend (default 8).

### Fixed
- **Decode collapsed with context depth on Metal.** Two causes, both in
  `gqa_attend`: its Born-importance pass recomputed the QK dot with one
  position per lane, so each lane walked a whole K row and the reads
  never coalesced; and the kernel ran one simdgroup per Q-head, putting
  only `num_heads` simdgroups on the device — nowhere near enough to
  hide the per-position `simd_sum` latency. The importance pass now
  reuses the main loop's lane-sliced layout, and the attend is
  flash-decoding shaped (one threadgroup per head, its simdgroups
  splitting the positions, partials combined through threadgroup
  memory). Nanbeige 4.2 on a fanless MacBook Air M4: decode 9.1 → 17.6
  tok/s at ctx 512, 6.0 → 14.2 at ctx 1024. `chunk_attend` got the same
  importance fix.
- **Looped Transformers prefilled one position at a time.**
  `graph_prefill_preferred()` sent them through the decode graph on the
  grounds that it beat the CPU *pair* path; the real competitor is the
  batched chunk-GEMM, which amortizes each weight over the whole chunk
  (512-token prompt: 85 tok/s chunked vs 14 through the graph). Time to
  first token at ctx 1024 fell from 107 s to 7.9 s.

### Changed
- One Metal compute encoder per attention layer instead of eleven.
  Dispatches inside an encoder are serial on Apple Silicon and already
  see the previous dispatch's writes, so the per-step encoder was a
  GPU pass with nothing to show for it (44 virtual layers × 11 = 484
  passes per token on Nanbeige 4.2).

## [0.5.34] — 2026-07-28

### Added
- **Gemma-3n E-series** (E4B/E2B): the dedicated stack — AltUp's four
  hidden replicas with tanh-router predict/correct, LAuReL low-rank
  residuals, per-layer embeddings, KV sharing across the last 15
  layers (shared layers attend over the source layer's cache and
  never append), and gaussian-top-k activation sparsity (population
  std + Acklam's inverse normal CDF). Formulas 1:1 from transformers'
  reference; the coefficient transpose is oracle-tested against a
  literal index port. Gated end-to-end on E4B-it (7.9 GB streamed →
  3.96 GB q4t): factual answer + clean stop, correct code generation
  at 11.7 tok/s on M4. v1 runs the sequential path (batched/pair
  prefill and GPU graphs stay off for this arch).

### Fixed
- Gemma-3n declares its multi-space ▁-runs as ADDED tokens — decode
  passed them through verbatim and leaked ▁ into generated code.
  Added non-special tokens now de-metaspace like vocabulary ones in
  both the full and the streaming decoder.

## [0.5.33] — 2026-07-28

### Added
- **`cortiq_execution_info()`**: one JSON line for status/About
  surfaces — `{"simd":"neon","threads":4,"gpu_backend":true}` — with
  the thread count from the pool's own resolution (forced >
  `CMF_THREADS` > big-core topology), valid before and after load.

### Fixed
- `cortiq_worker_tids` raced pool startup: tid registration ran on
  the worker threads while the embedder reads the registry right
  after `cortiq_load` — a phone saw "· 1 threads" with four workers
  alive. Pool construction now waits for every registration (thread
  start is milliseconds); the registry is complete the moment the
  pool exists.

## [0.5.32] — 2026-07-28

### Added
- **`cortiq_cancel(handle)`** in the C ABI: thread-safe cooperative
  cancellation — the engine honours it at every prefill chunk AND
  decode step (a 50-second mobile prefill is exactly where it
  matters), finishes with `finish_reason: "cancelled"`, clears the
  flag itself, and invalidates the KV-reuse history on a mid-prefill
  abort. A server can now honour a dropped connection.

### Fixed
- The forward-layers GPU-graph gate's unset branch read "is the GPU
  on" instead of the discrete-only `wgpu_graph_default()` — the one
  site of three that skipped the rule. `cortiq_set_gpu(true)` alone
  made the 0.2 tok/s whole-token graph race-eligible on mobile
  adapters (a measured 12–14× first-token cost on Adreno). Unset now
  means the same everywhere: discrete adapters only — the app-side
  `CMF_GPU_WGPU_GRAPH=0` workaround can be deleted.
- `execution_mode` reported a stub ("Avx2 · available_parallelism"
  even on an ARM phone with a 4-worker pool). It now reports the REAL
  pool size (the same forced > `CMF_THREADS` > big-core-topology
  resolution the pool itself uses, via `Pool::effective_threads`) and
  the target's actual SIMD.

## [0.5.31] — 2026-07-28

### Added
- **Cross-turn KV reuse**: chat apps resend the whole conversation
  every turn — the engine now remembers which ids the KV cache holds
  and, when the new turn strictly EXTENDS them, prefills only the new
  tail. Turn latency becomes proportional to the new text instead of
  the whole session (previously linear per turn, quadratic per
  conversation). Extension-only — no rollback — so it is exact for
  every layer kind including recurrent state (GDN/KDA/conv rings) and
  sliding windows; an edited turn, a new chat, a skill swap or any
  scorer run falls back to the fresh-sequence path. Gated off under
  MTP, O(1) attention and task masks; `CMF_KV_REUSE=0` disables;
  `CMF_PREFILL_PROF` logs the cache hit. Gate: a reused second turn
  is token-identical to a fresh prefill of the same ids.

### Fixed
- Release pipeline: every Android artifact links with 16 KB
  max-page-size (Android 15+ rejects 4 KB-aligned .so) and a readelf
  verify step FAILS the release on a regression; the iOS static
  library ships the wgpu-Metal backend so `cortiq_gpu_available()`
  can turn true on an iPhone.

## [0.5.30] — 2026-07-28

### Added
- **Mobile GPU support in the C ABI** (cmfmobile TUNING.md engine
  items): `cortiq_gpu_available()` — true when the build carries a
  GPU backend AND the device brings an adapter up (Vulkan on Android,
  Metal on iOS/macOS), so an app can tell "GPU off" from "GPU
  impossible"; `cortiq_set_threads(n)` — worker-pool size from the
  embedder instead of the process-wide `CMF_THREADS`;
  `cortiq_worker_tids(out, cap)` — kernel tids of the pool workers
  for ADPF work-duration reporting. Verified on real cross builds
  (NDK 28, 16 KB pages): arm64-v8a `--features gpu` imports dlopen
  and carries the wgpu Vulkan backend; armeabi-v7a and x86_64 build
  too; the aarch64-apple-ios staticlib carries wgpu-Metal.
- **Native MXFP4 decode** (OCP Microscaling FP4,
  compressed-tensors "mxfp4-pack-quantized" — Kimi-K3 experts): E2M1
  nibbles + per-32-group E8M0 scales decode straight into the normal
  quantize path; E8M0 NaN refuses loudly. Validated on a real Kimi-K3
  expert fetched by HTTP Range and by an exact synthetic roundtrip.
- **Big-core detection on EAS-less kernels**: when
  `cpu_capacity` is absent, the pool sizing AND the worker affinity
  pin fall back to `cpufreq/cpuinfo_max_freq` — same cluster
  ordering, so the 62.5% big-core rule keeps working.

### Fixed
- `moe-defrag` now slices the noaux_tc selection bias
  (`mlp.expert_bias`) with its experts and accepts F16 router rows —
  a renumbered expert used to read another expert's bias on
  Kimi/DeepSeek-V3/LFM2-class routers. Gated on Kimi-Linear-48B:
  27.7 → 17.7 GB (−36%), held-out code ppl +2.7%, decode ×2.3 on a
  24 GB MacBook (the specialist fits the page cache).

## [0.5.29] — 2026-07-28

### Added
- **DeepSeek-V2 MLA** (E12): latent attention executed as
  expand-to-MHA — per-token latent projection, K heads laid out
  [rope|nope] with the DeepSeek pair-interleave undone at convert, V
  zero-padded to the K head dim and sliced before O; YaRN mscale²
  softmax-scale correction. Gated on DeepSeek-V2-Lite (ppl 8.78).
- **gemma-4 MoE 26B-A4B** (E13b): dual-branch `DenseMoe` FFN — dense
  branch from the normed input plus expert branch from the RAW
  residual through its own norm sandwich; softmax→top-8→renorm router
  with per-expert scales, its scale-less input gain folded into the
  projection columns at convert; dual-geometry attention (sliding GQA
  hd=256 + every-6th global MQA hd=512). Gated by scorer/decoder
  parity: the scorer reproduces the model's own greedy tokens 40/40.
- **Kimi Delta Attention (KDA)** — Kimi Linear / Kimi-K3 linear mixer:
  delta rule with a PER-CHANNEL log decay (diagonal, vs GDN's
  per-head scalar), separate q/k/v projections each behind its own
  causal short conv, low-rank decay stage, sigmoid-gated output
  RMSNorm; both decay-gate formulas (standard softplus and the K3
  lower-bound sigmoid). Validated against a literal port of the FLA
  reference kernels; **Kimi-Linear-48B-A3B gated end-to-end** (98 GB
  streamed → 27.7 GB q4t on a MacBook, coherent generation).
- **Kimi-K3 machinery**: compressed-q MLA (q_a→rms→q_b), the `situ`
  activation, MLA NoPE (full-attention layers apply no rotary — the
  KDA layers carry position). K3-only pieces with no public modeling
  reference (mxfp4 packing, `attn_res_block_size` residual streams,
  latent MoE, MLA output gate) refuse with named errors.
- **tiktoken → tokenizer.json in Rust**: Kimi ships a tiktoken rank
  table instead of tokenizer.json; the converter synthesizes a
  standard one — byte-level BPE vocab plus the merge list RECOVERED
  from the ranks (transformers' algorithm), specials from
  `added_tokens_decoder`. 163k-entry table converts in ~320 ms.
- **MiniCPM3** — the compressed-q gate carrier: `scale_depth`/√L
  residual scaling folded into o_proj/down_proj at convert,
  `scale_emb` via the embedding multiplier, `dim_model_base` logit
  scale as a new `logit_multiplier` header field (the head is tied to
  the embedding — the divisor cannot be baked into the shared
  tensor), and longrope's per-dim short factors as
  `rope_freq_factors` dividing inv_freq at load (they are the TRAINED
  rope inside the native window — nothing like phi's ≈1 factors).
- **Streaming hub convert**: hub checkpoints convert one shard on
  disk at a time (fetch → process → delete) — a 98 GB release
  converts on a laptop whose free disk only fits the output plus one
  shard.

### Fixed
- Teacher-forced scorers double-applied the gemma final-logit softcap
  in the sequential branches (`lm_head_forward` already caps) —
  tanh∘tanh flattered every reported gemma-class ppl; branches now
  agree within int8-activation noise.
- The fused-pair prefill cached RAW V on v-norm architectures
  (gemma-4): q/k were normalized, V was not — pair prefill diverged
  from singles (argmax parity 28/40 → 40/40). Regression-tested
  against two singles.
- Pair prefill dispatched dual-branch FFNs into an `unreachable!`;
  models whose layers have no pair arm (MLA, KDA) now fall back to
  the single-position path instead of panicking.
- gemma tokenizer.json declares `<bos>` only as an added token —
  detect it there and set add_bos, or scoring ran bos-less.
- DeepSeek MLA rope pairs are interleaved in the checkpoint (view
  d/2,2 → transpose): the converter stores rope dims even-first so
  the runtime's half-split rotation reproduces their math; the
  interleave is correctly SKIPPED for half-split (MiniCPM3) and NoPE
  (Kimi) families.

## [0.5.28] — 2026-07-27

### Added
- **`cortiq moe-mask`** — the runtime-switchable twin of `moe-defrag`:
  bakes a task's expert restriction as a first-class task mask (new
  optional expert-bitfield area in the masks section, spec §5) instead
  of cutting the file. The full expert set stays on disk; `run --task
  <name>` narrows MoE routing to the mask's experts — one file, many
  specialists. Gated: decode through a baked mask is token-identical
  to the equivalent `CMF_MOE_MASK` runtime restriction.
- **`cortiq sign`** — detached model signing: Ed25519 over the file's
  SHA-256 written to `<model>.sig` (spec §8.2); `cortiq verify` checks
  it automatically when present. Authenticity on top of the format's
  integrity hash chain, no container rewrite.
- **`cortiq compact`** — native-Rust port of the container compactor:
  reclaims dead directory/header tails left by append-only skill
  growth, streaming payloads from the source mmap.
- `moe-defrag` writes an honest provenance block
  (`provenance.moe_defrag`: tool, cover, stats hash, kept_per_layer)
  and embeds the routing B-field remapped to the new expert numbering —
  `--stats` becomes optional when re-defragging a specialist tighter.

### Changed
- Spec synchronized with the implementation: dtypes 13 (`q1s`) / 14
  (`q1t`) fully specified, MoE expert defrag contract (§11.1), pipeline
  containers (§12), expert role-contiguity SHOULD (§2.2), sparse index
  marked deprecation-pending (§7); the RU/ZH specs — a whole section
  behind — brought level with EN.

## [0.5.27] — 2026-07-27

### Added
- **`cortiq moe-defrag` — carve a task specialist out of a MoE model.**
  MoE expert usage turns out to be strongly task-conditional (measured
  on KAT-Coder 34.7B-A3B: the top-64 expert sets for code vs prose
  overlap with Jaccard 0.25 — near-disjoint), so a model serving ONE
  task carries hundreds of experts it never routes to. The pipeline:
  run the task's representative corpus once with `CMF_MOE_STATS=f.json`
  (per-layer expert-selection counts), then
  `cortiq moe-defrag model.cmf --stats f.json --cover 0.95 --output
  specialist.cmf` keeps, per layer, the smallest top expert set
  reaching that fraction of the routing mass, renumbers the kept
  experts into a contiguous prefix, slices the router's rows to match
  and drops the rest from the file. Softmax renormalizes over the kept
  set. Measured on KAT-Coder, code-calibrated at 95% cover, on an M4
  24 GB: **19.6 → 12.7 GB (−35%)**, held-out code perplexity 5.058 →
  5.198 (+2.8%) — and because the specialist now fits the machine's
  memory where the full model paged, decode goes **7.6 → 13.7 tok/s
  (×1.8)** and prefill **6.1 → 20.0 tok/s (×3.3)**.
- **`CMF_MOE_MASK=<stats.json>`** (+ `CMF_MOE_MASK_COVER`, default
  0.9): the same expert restriction applied at RUNTIME, no file
  rewrite — selection happens over the allowed set only. Use it to
  ppl-gate a mask before committing to the physical defrag (the two
  are semantically identical; measured ppl matches to 3 digits).
- `CmfModel::write_ref` (+ `TensorSpecRef`): container rewrite with
  borrowed payloads sliced from the source mmap — a 19.6 GB model
  repacks in 94 s without materializing its tensors in RAM.

### Changed
- MoE expert lists now enumerate by tensor presence up to the header
  count (a defrag'd layer keeps fewer than `arch.moe.num_experts`);
  the router's row count must match the expert count, and `top_k`
  clamps to it. Files with the full expert set load exactly as before.

## [0.5.26] — 2026-07-27

### Changed
- **`cargo install cortiq-cli` now includes the GPU backend by
  default.** The `gpu` feature (wgpu → Vulkan / DX12) is on by default
  for the CLI crate — a bare install from crates.io previously built
  CPU-only and silently missed Vulkan, while the release binaries had
  always shipped with it. `--no-default-features` restores the
  CPU-only build; native Metal on macOS was and remains always in.

## [0.5.25] — 2026-07-27

### Added
- **Routed MoE inside the whole-token GPU graph** (wgpu / Vulkan &
  DX12). A MoE layer no longer breaks the one-submit-per-token decode
  graph: the router matvec, the shared-expert gate, the top-k
  selection, the fused gate+up+SiLU over every selected expert and the
  weighted down-projection all execute on-device, in one compute pass
  per layer. Expert weights live in per-layer concatenated buffers
  (the always-on shared expert rides as the trailing block, pinned by
  the select kernel with its sigmoid weight), counted against the
  usual VRAM budget. The select kernel is fully parallel — k rounds of
  an argmax reduction with lowest-index tie-breaking that matches the
  CPU scan on routing decisions; the serial one-thread top-k it
  replaces cost 22 ms/token at k=8 across 40 layers. Scope: softmax
  routers with a shared expert and q4t expert weights, ≤256 routed
  experts, top-k <16 (the KAT-Coder / Qwen3.6-MoE class);
  sigmoid/biased routers and `CMF_MOE_TAU` keep the CPU path.
  Measured on KAT-Coder-V2.5-Dev (34.7B-A3B) on an RTX 5090: decode
  **32.8 tok/s steady vs 14.4 on the host's 32-core CPU (2.3×)**,
  15.6 ms/token forward, output token-coherent. A step-by-step
  walkthrough from GGUF download to GPU decode on both Vulkan and
  Metal is in [docs/KAT_CODER.md](docs/KAT_CODER.md).
- **q4_tiled MoE expert blocks on both per-op GPU backends** (Metal +
  wgpu) — MoE models whose experts quantize to q4t now qualify for the
  per-op expert offload path (previously q1-only), with the runtime
  probe arbitrating as usual.
- `CMF_GRAPH_PROF=1` now also prints the caller-side total per token,
  so graph-build cost is visible next to encode and submit+readback.

### Fixed
- **GPU-vs-CPU probe verdicts now compare per-arm minima** instead of
  means. A first CPU op measured mmap-cold (page-fault storm, ~3×
  steady state) could poison the CPU arm's mean and lock a losing GPU
  path for the whole process; minima measure honest steady state.
- Uploading many large weight buffers before the first submit no
  longer transiently doubles memory (the staging belt is flushed per
  MoE layer) — fixes a device OOM on discrete cards when a 17 GB
  expert set was staged all at once.
- `import-gguf --quant` help text now lists the full codec set
  (`q4t`, `q1` were accepted but undocumented).

## [0.5.24] — 2026-07-27

### Added
- **qwen35moe GGUF import** — `cortiq import-gguf` now handles the
  Qwen3.6-MoE / KAT-Coder class (GDN linear-attention hybrid +
  256-expert MoE + shared expert) natively in Rust, instead of
  refusing all SSM hybrids. The layer schedule is derived from tensor
  presence; the 3-D routed-expert tensors split into per-expert
  matrices; and every llama.cpp storage convention is undone on
  import (the baked +1 in RMS norm weights — except the GDN gated
  norm, whose weights live near 1 naturally; `ssm_a` stored as
  −exp(A_log); the tiled V-head order on every V-indexed tensor,
  including the out_proj columns). GGUF files are memory-mapped (a
  21 GB MoE file imports on a 24 GB machine) and tied-embedding
  GGUFs work. Measured on KAT-Coder-V2.5-Dev (34.7B-A3B, Q4_K_M →
  q4t): a 32-core EPYC-class CPU decodes at **16.6 tok/s where
  llama.cpp does 4.7 on the same file — 3.5× faster**; `--o1 all`
  runs the whole model O(1)-in-context (30 of 40 layers are already
  linear) and cuts KV+state at 4K context from 238 MB to **83 MB**.
- **Adaptive MoE routing** (`CMF_MOE_TAU=0.x`, opt-in): keep the
  smallest prefix of the router's top-k whose renormalized mass
  reaches τ — confident tokens touch 1–2 experts, flat ones keep all
  k. MoE decode is memory-bound (every routed expert streams its
  three matrices per token), so skipped experts are skipped weight
  traffic. Preliminary numbers on KAT (M4): τ=0.9 decodes **12%
  faster at better perplexity than the model's own fixed top-8**
  (trailing near-zero experts contribute mostly noise); τ=0.8 is
  +58% at a modest quality cost and beats fixed k=4 on both axes.
  `CMF_MOE_TOPK=N` provides the fixed-k variant. Defaults untouched.
- **Bring-up diagnostics**: `CMF_DEBUG_LAYERS=1` prints per-layer
  hidden-state rms/max with layer kinds, and the `cmpcmf` example
  numerically diffs two `.cmf` files worst-cosine-first — together
  they turned a "healthy dynamics, garbage output" import bug into a
  one-tensor pinpoint.

## [0.5.23] — 2026-07-26

### Added
- **Text-to-image on Vulkan / DX12 (wgpu)** — the DiT's quantized
  GEMMs now run on every wgpu adapter (NVIDIA / AMD / Intel desktop
  cards, Adreno / Mali phone GPUs), not just Apple Metal. The WGSL
  `q4t_mul_mm` is the register-blocked cousin of the Metal kernel
  (18-byte tile decode in the W staging; parity on an RTX 4090:
  5.2e-6 vs an exact f64 reference), and the SwiGLU FFN runs fused —
  w1/w3/silu·u/w2 as four passes in ONE submission with one readback,
  because on discrete cards the per-op submits and the PCIe round
  trips of the intermediates dominate the GEMM itself. Weights stay
  cached in VRAM. The same per-process CPU-vs-GPU probe and
  contention kill-switch gate the path everywhere: a phone GPU that
  loses simply keeps the CPU arm — enabling the GPU never makes
  generation slower.
- **`cortiq_imagine` in the C FFI** (cortiq-ffi): text → interleaved
  RGB8 into a caller buffer, with a per-step progress callback — the
  entry point mobile apps need. `guidance ≤ 1` disables CFG and
  halves the work (the right default on phones); weights stream from
  the mmap, so peak RSS stays far below the model size.

### Performance
RunPod RTX 4090 (Vulkan) vs the same pod's 32-core CPU, 30 steps,
CFG 4, the one 3.2 GB q4t file:
- 512×512: **454 s** fused GPU / ~700 s per-op GPU / ~960 s CPU
- 256×256: **147 s** GPU / 222 s CPU
Renders are visually identical to the CPU path. The remaining wall on
discrete cards is CPU-side attention and the portable f32 GEMM — the
next port targets.

## [0.5.22] — 2026-07-26

### Added
- **VAE decoder on the Metal GPU**. conv2d runs as an implicit GEMM —
  the receptive field is gathered straight from the NCHW image inside
  the kernel's staging, where the CPU path materializes a ≥2 GB im2col
  patch matrix per high-resolution conv (that matrix, not the GEMM,
  was the VAE's wall). Whole resnet blocks execute in one command
  buffer (new GroupNorm reduce/apply kernels with SiLU fused, residual
  add on-device), and the nearest-2× upsample is fused with its
  following conv so only the small pre-upsample image is uploaded.
  Conv/norm weights upload once per process. VAE decode on an M4:
  512px **7.8 → 2.9 s**, 1024px **33.4 → 14.7 s**; renders visually
  identical (parity 9.3e-4 vs the CPU conv, pixel drift unchanged).
- **Flash attention for the DiT** — experimental, opt-in
  (`CMF_DIT_FLASH=1`): online softmax, no n×n scores in device
  memory, essentially exact (3.4e-8 vs an f64 reference). The V2
  kernel loads operand blocks straight from device memory and runs
  its KV loop with zero threadgroup barriers, but still trails the
  default GEMM attention chain on M4 (15.5 vs 12 ms at 512px shapes,
  270 vs 124 ms at 1024px) — the measured dead ends are documented in
  the gate comment, and the default path is unchanged.
- **GPU micro-bench harness** (`CMF_BENCH=1 cargo test --test
  gpu_q4t_bench`): the q4t GEMM at the Lumina FFN shape, the
  attention chain at 512px/1024px sequence lengths, and the VAE
  decode (`CMF_VAE_BENCH_HW` picks the resolution). Written for a
  noisy machine: alternate A/B runs, trust medians.

### Performance
- 256×256 / 30 steps / CFG 4 now renders in **~37 s** end to end on
  an M4 (0.5.20 CPU baseline: 164 s). The 512px render is VAE-bound
  no longer; the remaining wall is the q4t GEMM itself, measured at
  ~75% of the device's practical fp32 peak (a grid swizzle and a
  pre-transposed K both landed inside noise — the X panel already
  lives in the SLC).

## [0.5.21] — 2026-07-25

### Added
- **Text → image, end to end** (`cortiq imagine`): a native Rust
  pipeline for **Lumina-Image 2.0 (2.6B)** — Gemma-2-2B prompt encoder,
  Next-DiT flow-matching denoiser, FLUX VAE decoder — each with a numpy
  parity reference on the real weights (rel/abs max|Δ| 1.4e-4 / 2.1e-5 /
  6.6e-6). CFG with per-row norm rescale, σ′=6σ/(1+5σ) shift schedule,
  P6 PPM output.
- **`cortiq imagine-pack`**: folds a diffusers tree into ONE quantized
  `.cmf` (text encoder + DiT + VAE + tokenizer + configs) — 19 GB →
  **3.2 GB**. Projections q4t (default) or q8; AdaLN modulation and
  embeddings stay q8, VAE f16, norms f32. The runtime runs the
  quantized projections straight off the mmap.
- **DiT on the Metal GPU**: every modulated DiT block executes as one
  command buffer — norms/modulation, q4t GEMMs, qk-norm + 3-axis RoPE,
  all-heads attention (scores → row softmax → P·V), and the SwiGLU FFN
  all device-resident; only the hidden state crosses the CPU boundary
  (~10 MB/block instead of ~100 MB over ~7 roundtrips). The new
  `q4t_mul_mm` kernel decodes the 18-byte tiles inside the GEMM's
  K loop — the two-pass dequant variant was bandwidth-bound on ~2.8 GB
  of scratch re-reads per FFN-shaped op. The CPU/GPU probe gained a
  wide-batch class (prompt-encode and DiT batches have opposite
  winners in the same process), and a work-proportional contention
  kill-switch (cold PSO builds exempt) drops to CPU when another
  process owns the device.
- `CMF_DIT_PROF=1` — per-stage wall-time profile of the denoiser,
  dumped when the model unloads.

### Performance
M4, 30 steps, CFG 4, the one 3.2 GB q4t file; CPU/GPU renders visually
identical (pixel drift ≤ 10/255 at 512px):
- 256×256: **48 s** GPU / 79 s CPU-only
- 512×512: **161 s** GPU / 322 s CPU-only
- 1024×1024 is now practical: the DiT forward takes 13.6 s at 4136
  tokens on the block graph.
The CPU DiT path is pool-parallel end to end (bit-exact — same-seed
renders are byte-identical to the serial code).

### Fixed
- Gemma tokenizer picked `<s>` (id 204) as BOS instead of `<bos>`
  (id 2) — both live in added_tokens and the scan order chose the
  wrong one. BOS now comes from the post_processor template; golden
  test against HF tokenizers.
- A stale probe cold-flag could permanently disarm the GPU contention
  kill-switch on a thread, and a first-use pipeline compile counted as
  device contention — cold one-off costs are now exempt everywhere.

## [0.5.20] — 2026-07-25

### Fixed
- **Mobile GPU ~75× slower than the CPU**: 0.5.19 made the wgpu
  whole-token graph both correct and default-on everywhere, but tiled
  mobile GPUs (Adreno/Mali) drain their pipeline at every barrier and
  the ~300-dispatch graph collapsed to ~0.2 tok/s. The graph now **races
  the normal path** on integrated adapters at generation granularity:
  generations alternate arms, per-token wall times accumulate, and the
  faster side wins for the process — a fast phone GPU keeps the graph, a
  slow one costs a single discarded token (the first graph token bails
  immediately when it is already >4× the measured CPU pace). Discrete
  adapters and GDN hybrids trust the graph as before;
  `CMF_GPU_WGPU_GRAPH=1/0` still forces either way.

## [0.5.19] — 2026-07-25

### Fixed
- **Mobile GPU decoded garbage**: the wgpu attend kernels need 33 KB of
  workgroup memory, but phone GPUs (Adreno/Mali) and wgpu-on-Metal cap it
  at 32 KB — the invalid pipeline silently turned attention into a no-op
  (slow AND wrong). Stride-129 kernel twins (16.5 KB) now serve
  `head_dim ≤ 128` on every device; `head_dim > 128` models fall back to
  the CPU cleanly on 32 KB devices; and pipeline validation errors now
  fail GPU init into an honest CPU fallback instead of corrupting output.

## [0.5.18] — 2026-07-25

### Fixed
- **wgpu whole-token graph never engaged from the CLI**: the gate's default
  read only the FFI toggle (`cortiq_set_gpu`), so `CMF_GPU=1` on discrete
  Vulkan/DX12 cards silently ran the slow per-op path. The graph now
  defaults ON whenever the wgpu backend is active. RTX 4090 decode:
  Bonsai-27B q1 (GDN hybrid) **5.2 → 40.2 tok/s (×7.7)**, Bonsai-1.7B q1
  65 → 154 tok/s, Nanbeige4.2-3B (Looped Transformer) 10.6 → 30.8 tok/s —
  greedy output token-identical to the CPU path on all three.
- **q4_tiled on the wgpu graph produced garbage**: `Q4Tiled` and `Q4Block`
  shared graph codec kind 2, so 18-byte interleaved tiles were fed to the
  split-layout `q4b_matvec` shader. `Q4Tiled` is now kind 5 with its own
  `q4t_matvec` WGSL kernel (u16-assembled tile reads).
- clippy `erasing_op` in the wgpu GDN ring-size expression
  (compiled only under `--features gpu`).

### Added
- **wgpu attend depth work**: vec4 K/V/Q loads (+12% short-context decode)
  and split-K flash-decoding past 256 cached positions — one 128-position
  chunk per workgroup plus a log-sum-exp merge in the same compute pass.
  4090, 1.7B q1: ctx1024 47 → 99.5 tok/s (×2.1), ctx2048 94.9 vs CPU ~40.
- **Metal whole-token graph: Q4Tiled** — new `q4t_matvec` MSL kernel
  (ushort tile reads, parity with the q4b kernel); q4t models used to
  silently fall off the graph.
- **CPU kernels**: ARM 1×4 SDOT blocking for q4_tiled (prefill GEMM ×2.1
  on Apple Silicon, mirroring the ×2.1 EPYC result), fused ternary pair
  `q1t_matvec2` (×1.83 on the MTP verify path), fused `silu(gate)·up` for
  q4_tiled pairs (+7% on AVX2).
- **AVX-512 VNNI tile kernels** for q4t/q4b/q1/q1t (`vpdpbusd`, 256-bit VL,
  bit-identical sums), default ON where VNNI exists — measured on Zen 4
  (7950X): q4t +8%, q1 +6%, q4b +4%. `CMF_VNNI_TILES=0` opts out.
- **`skill bake`**: `--target-sparsity`, `--l1-aggression`, `--ffn-align`
  (default 32 — keeps the defragged FFN on grouped codecs and SIMD fast
  paths) and `--uniform-inter` (one FFN width across layers, required by
  the whole-token GPU graphs).

## [0.5.17] — 2026-07-23

### Added
- **Sampler-Level Token Suppression (`suppress_tokens`)**: Added `suppress_tokens: Vec<u32>` field to `SamplerConfig`. Suppressed tokens have their logits set to `-infinity` prior to sampling, preventing models from generating banned initial tokens (e.g. `<think>`) on CPU/GPU.

### Fixed
- **Reasoning / Thinking Suppression**:
  - Implemented token logit suppression for reasoning tokens (`<think>`) in `cortiq-server` when `enable_thinking=false` or `think_budget=0`, forcing immediate direct answer generation in 0.1s without wasting completion tokens on thinking.
  - Added stream and non-stream `<think>...</think>` block filtering fallback in `cortiq-server`.
  - Added `chatml_fallback_opts` thinking suppression support to `Tokenizer` when no custom Jinja template is embedded.

## [0.5.16] — 2026-07-23

### Fixed
- **Reasoning / Thinking Suppression**: Improved chat template `enable_thinking=false` prefill handling in `cortiq-engine` tokenizer to match `assistant` markers without requiring a trailing newline. Fixes reasoning mode suppression for ChatML models like Nanbeige 4.2 / Qwen 3.5.

## [0.5.12] — 2026-07-23

### Added
- **Looped Transformer GPU acceleration** (Metal, Apple Silicon): both loop
  iterations now execute in a single Metal graph submission with the
  `loop_final_norm` applied on-device via `encode_loop_norm` (RMS norm + blit).
  Eliminates the CPU round-trip at loop boundaries. Nanbeige4.2-3B Q4: 5.6 →
  18.6 tok/s steady decode (3.3×), TTFT 5.2 → 1.8 s (2.9×).
- **GPU graph prefill for looped models**: `graph_prefill_preferred()` returns
  true for `loop_final_norm` models on macOS — each prompt token goes through
  the same device-attend graph as decode, doubling prefill throughput.

### Fixed
- **GPU loop_final_norm insertion**: the `continue` after `q1_graph_gpu` /
  `chunk_run_gpu` returns no longer skips the per-loop norm — the norm is
  applied either on-device (fused graph) or on CPU (fallback) before the
  next loop iteration.
- **Clippy**: resolved all CI errors — `div_ceil`, `needless_borrow`,
  `needless_range_loop`, missing struct fields in tests.

## [0.5.11] — 2026-07-23

### Fixed
- **Looped Transformer prefill**: `loop_final_norm` was only applied in the decode
  path (`forward_layers_upto`) but missing from `prefill_batch` and `forward_pair`
  (MTP speculative). This corrupted the KV cache at loop boundaries during prompt
  processing, producing garbage output for Nanbeige4.2-3B. Now all forward paths
  apply the per-loop final norm correctly.
- **GPU graph guards**: `q1_graph_gpu`, `chunk_run_gpu`, and `try_token_graph_wgpu`
  now refuse looped models (`loop_final_norm=true`) — the flat layer graph cannot
  express mid-stack norm insertion. Falls through to the correct CPU path.

### Added
- **`enable_thinking` fallback**: Templates that ignore `enable_thinking` (e.g.
  Nanbeige/Qwen-legacy) get `</think>\n\n` injected after `assistant\n` when
  thinking is explicitly disabled — the model answers directly without reasoning.

## [0.5.10] — 2026-07-22

### Fixed
- **Metal GPU Q4Block matvec**: Fixed nibble extraction order in `q4_dot8_fast`
  and `q4_dot8_half` — the v0.5.7 ILP refactor swapped lo/hi nibble lanes,
  producing garbage output for all Q4Block models on the whole-token graph path.

## [0.5.9] — 2026-07-22

### Added
- **Looped Transformer support (Nanbeige 4.2)**: Native `num_loops` + `loop_final_norm`
  architecture fields. The 22-layer Nanbeige4.2-3B re-applies its layer stack twice
  (44 virtual layers) with per-loop final normalization — 4.17B effective parameters
  from 2.1B physical weights. Conversion, inference (CPU + Metal GPU), and O(1)
  Nyström attention all work with looped models.
- **Metal GPU whole-token graph for looped models**: `q1_graph_gpu` iterates
  `total_layers()` (num_layers × num_loops) with per-loop final-norm insertion,
  device-attend KV mirror handles the growing cache across loops.

### Performance (Nanbeige4.2-3B, Apple M4, CMF_GPU=1)
- Q8 decode: **13.2 tok/s** (92% of theoretical bandwidth limit)
- Q4 decode: **20.4 tok/s** (best throughput, 2.4 GB model)
- O(1) mode: **10.2 tok/s constant** at any context (vs 2.8 tok/s exact at ctx=2048 — ×3.7 speedup)
- Q8 GPU prefill: **211 tok/s** (chunk graph)

## [0.5.7] — 2026-07-21

### Added & Optimized
- **Metal GPU Shader Optimizations (Q4 & Q8)**: Implemented 4-way ILP unrolling and register activation vector caching (`float4 xv[8]`) in `q4b_matvec`, `q8_matvec`, and `q8_matmat`, achieving up to **+21.9% decode speedup** on Apple Silicon (M4).
- **Q1T CPU Performance**: Added zero-stack bitwise register unpacking (`q1t_unpack_reg_u64s`) and 2-way ILP unrolling along with macOS physical P-core auto-discovery via `sysctlbyname("hw.perflevel0.physicalcpu")`, yielding **14.85 tok/s** on Bonsai-8B (**8.25x speedup** over single-thread baseline).
- **Quantization Parity & Verification**: Verified 100% text generation accuracy and coherence across Q1T (1.58-bit), Q4 (4-bit), VBIT (4.25-bit), and Q8 (8-bit) models.

## [0.5.5] — 2026-07-21

### Fixed
- **GPU Metal/WGSL**: Fixed MSL compilation crash caused by invalid `packed_uint` memory access in the `q1t_matvec` kernel which was forcing a silent fallback to CPU.
- **WGSL Backend**: Replaced undefined `pow3t` usage with `Q1T_LUT` table in `q1t_matmat`.
- **CPU Path**: Resolved vector allocation bottlenecks by reusing thread-local buffers (`PRESCALE_BUF`), dramatically improving inference speed.

## [0.5.3] — 2026-07-21

### Fixed
- Fixed the global GPU toggle: `cortiq_set_gpu(bool)` now correctly enables/disables the `wgpu` and `metal` device initializers. Previously they ignored the toggle because the backend selection only checked `CMF_GPU`.

## [0.5.2] — 2026-07-21

### Added
- **Mobile GPU Toggle**: Added `cortiq_set_gpu(bool)` to the FFI C ABI to allow mobile apps (e.g. Flutter) to enable/disable the discrete Vulkan/Metal graph dynamically at runtime before loading a model.

## [0.5.1] — 2026-07-21

### Added
- **GPU Metal optimizations**: ported fused `add_rmsnorm` from Vulkan.
- **Metal `TokenGraph` Q8 support**: `q8_row` tensors are now supported natively in the Metal decode graph, eliminating CPU fallback.
- **CPU `add_rmsnorm` fusion**: integrated residual addition and RMSNorm into a single SIMD pass (`add_rmsnorm_fused_into`), reducing memory bandwidth overhead.

### Added

- **Whole-token wgpu decode graph on discrete GPUs (Vulkan / DX12 / Metal).**
  The entire layer stack for one decode token is encoded into a single command
  buffer with the hidden state resident in VRAM and exactly one readback per
  token — covering q1 / q8 / q4_tiled / q1t projections and Gated-DeltaNet
  attention hybrids (Bonsai-27B, Qwen3.5), with the final RMSNorm + lm_head
  folded into the same submit so the graph hands logits straight to the sampler.
  Opt-in via `CMF_GPU_WGPU_GRAPH=1`; token-identical to the CPU f32-activation
  path. Ships portable WGSL kernels (rmsnorm, RoPE + q/k-norm, flash-decode GQA
  attention, GDN conv + delta-rule step, 1-bit and int8 matvecs).

### Changed

- **Blob laid out in execution order.** The converter now writes tensors in the
  order the engine touches them (embed → per-layer: norm/attn/norm/router/experts/
  ffn → final norm → lm_head → mtp → tail), with each layer's — and each MoE
  expert's — tensors contiguous. The kernel's up-front `madvise(WILLNEED)` readahead
  now streams the file in the same order the forward pass consumes it, so page
  faults are hidden behind compute instead of thrashing. Byte-for-byte identical
  weights; only their on-disk position changed. Readers are unaffected.
- **Large tensors are page-aligned.** Tensors ≥ 16 KB are aligned to 4096 in the
  blob (was a uniform 64 B). Cold skill / MoE-expert / mask weights now sit on
  their own page(s), so "unused weights cost 0 RSS" holds at page granularity.
  Small tensors keep the 64 B packing (no size bloat). `4096 % 64 == 0`, so
  existing readers accept the files unchanged.

### Performance

- **~2× faster GPU decode on 1-bit models (RTX 4090, Vulkan).** A 16-way
  shared-memory bank conflict in the q1 matvec's activation tile was inflating
  the FFN kernel ~8× (all 16 lanes of a row hit the same 4 banks). Padding each
  32-column group to 33 slots spreads them across 16 distinct banks; identical
  math and accumulation order, so still token-identical. Bonsai-27B q1 decode
  ~18 → ~36 tok/s pure decode; the whole-token submit halved (51 → 25 ms).
- **Token-invariant graph state cached across decode tokens.** Per-layer norm
  weights, the GDN f32 in-projections (~63 MB re-uploaded every token), q8 row
  scales, and matvec param uniforms are now uploaded once and reused, cutting
  per-token host work from ~33 ms to ~1 ms and re-upload traffic to zero.
- **Independent projections share one compute pass.** QKV, the GDN in-projection
  (qkv/z/a/b), and FFN gate/up — all reading the same normed hidden — are issued
  in a single compute pass so the GPU overlaps them instead of draining between
  per-op barriers (+5–8%, token-identical).
- **Model open touches less memory.** The in-memory tensor directory is now indexed
  by a 64-bit hash of the tensor name (with a collision-safe overflow list) instead
  of a `String`-keyed map, so opening a model no longer allocates a copy of every
  tensor name. Lookups verify the full name on hit, so there are no false matches.

## [0.4.1] — 2026-07-19

### Added

- **`cortiq_set_options` gains `enable_thinking`** (C FFI) — a sticky per-handle
  flag for reasoning-model chat templates (Qwen3/3.5). `false` makes the model
  answer directly with no `<think>` block; `true` re-enables it; absent or
  `null` leaves the current value untouched. Lets embedders (the CMF Mobile app)
  expose a "disable thinking" toggle without a bespoke API. The `cortiq_chat` /
  `cortiq_chat_messages` render path now honors it.

## [0.4.0] — 2026-07-19

Training-free **q1t** ternary post-training quantization — take an ordinary
checkpoint to ~2.25–3.5 bits/weight (below `q4`) with no retraining — and full
GPU acceleration for it on **both** engine backends: native Metal and wgpu
(Vulkan / DX12 / Intel). On a 14.8B GDN-hybrid the q1t model is 6.27 GB (−25 %
vs `q4`) and, on the GPU, *faster* than the same model in `q4` on the CPU:
decode 3.9 tok/s, TTFT 6.0 s, PPL identical to the CPU path.

### Added

- **q1t codec** (`TensorDtype::Q1T`) — per 32-group ternary `{−s,0,+s}` packed
  base-3 (5 values/byte → ~2.25 bpw) with a sparse per-row outlier overlay
  `[u32 row_ptr[rows+1]][(u16 col, f16 val)]` (4 B/outlier, no binary search).
  Built on the holographic-transfer idea: preserve the layer output `W·x`.
- **`quantize-gptq` command** — a calibration-driven, training-free path
  (`CMF_GPTQ_TERNARY=1`): two-field outlier mask (`|W|·RMS(x)`), a closed-form
  per-row output-stabilising rescale (*докрутка*), and a keep-precise skip-list
  (`CMF_GPTQ_SKIP`, `CMF_GPTQ_DOWN_KEEP`). Streams one tensor per worker;
  diagonal Hessian capture fits a 12B.
- **q1t CPU kernels** — fused sign-LUT decode (no per-weight base-3 divide),
  int8 SDOT on **ARM dotprod and x86 AVX2**, a u64-store group unpack, and a
  `value·x` overlay correction. Decode + batched prefill both accelerated;
  `CMF_SDOT=0` keeps the exact f32 path.
- **q1t GPU (Metal)** — `q1t_matvec`/`q1t_overlay` (full-precision decode),
  `q1t_mul_mm` register-blocked prefill GEMM, and integration into the
  **whole-token GPU graph** so a q1t decode token runs entirely on-device.
- **q1t GPU (wgpu)** — WGSL `q1t_matvec` + `q1t_overlay`, `q4b_matvec`, and the
  `q1t_mul_mm` prefill GEMM + `q1t_overlay_mm`, with weights resident in VRAM.
  q1t/q4 now GPU-accelerate on NVIDIA / AMD / Intel via Vulkan / DX12.
- **`q4_block` GPU kernels** (Metal `q4b_matvec`, wgpu `q4b_matvec`) — a precise
  4-bit weight (e.g. `down_proj`, `lm_head`) stays on the GPU without
  ternarizing.

### Changed

- The whole-token GPU graph's projection dispatch (`proj_abs`/`encode_proj`)
  now accepts **Q1, Q1T or Q4-block** (was Q1-only). Consequence: **q4 models
  get the whole-token GPU decode path too** — 12B `q4` decode 3.0 → 5.6 tok/s
  on an M4, where before `q4` had no GPU kernel at all.
- `dequant_q1t` takes `(rows, cols)` (like `dequant_q8_row`/`vbit`) for the
  per-row overlay.

## [0.3.12] — 2026-07-18

LFM2-MoE support: the LiquidAI **LFM2.5-8B-A1B** hybrid — short-convolution
mixers, a sparse Mixture-of-Experts FFN, and a handful of full-attention
layers — converts and runs natively. Coherent generation verified
end-to-end (q4, `<think>` reasoning + correct answers).

### Added

- **LFM2 / LFM2-MoE architecture** (`lfm2_moe`). A new `ShortConv` token
  mixer (`AttnKind::ShortConv`, `LayerType::ShortConv`): the gated short
  convolution `out_proj(C ⊙ conv1d(B ⊙ x))` with a causal depthwise
  kernel and a per-channel ring state kept in the layer's linear state —
  decode and chunked prefill share one path, verified bit-identical. The
  full-attention layers reuse the existing per-head qk-norm → RoPE path
  with no new code.
- **Sigmoid MoE routing** (DeepSeek-V3 `noaux_tc` family): a shared
  `moe_route` scores each expert with a sigmoid, adds an optional
  per-expert selection bias (`mlp.expert_bias`) to the top-k *choice*
  only (the gathered weights stay unbiased), then renormalizes with a
  1e-6 floor and a routed scale. The Qwen softmax-over-all path is
  unchanged, bit-identical.
- **Converter** maps the LFM2 vendor tensor names onto CMF's canonical
  layout (`operator_norm`→`input_layernorm`, `conv.*`→`short_conv.*`,
  `feed_forward.wN`→`mlp.{gate,up,down}_proj`, `embedding_norm`→`norm`,
  `self_attn.out_proj`→`o_proj`, …) and reads the `lfm2_moe` config
  (`conv` → `ShortConv` layers, sigmoid routing, `conv_L_cache` kernel,
  `norm_eps`).

### Fixed

- **Chat template not bundled** when a checkpoint ships it as a sidecar
  `chat_template.jinja` (LFM2, newer Qwen3 releases) rather than
  embedding it in `tokenizer_config.json`: the downloader now fetches the
  file and the converter ignores an empty one. Without it, `run` fell
  back to a generic ChatML default that did not match the model and
  produced degenerate output.

### Changed

- `cortiq info` / `story` report conv-mixer layers distinctly (e.g.
  `24 (6 full / 18 conv)`) instead of lumping them under "linear".

## [0.3.4] — 2026-07-17

The whole token on the GPU, and the prefill on the AMX. Bonsai-27B (q1)
decode on an Apple M4 goes 5 → 10–11 tok/s with the first token 12.4 → 3.5 s;
Bonsai-1.7B goes 28 → ~75–79. On q8 (Qwen2.5-0.5B, same M4, interleaved
runs, both sides at their best measured configs) CMF now decodes faster
than llama.cpp's Metal backend and its default CPU config, within 5% of
its best CPU config, and prefills within 12% (pp512 377 → ~1030).

### Added

- **Whole-token Metal graph for q1** (macOS): full-attention layers join
  the GDN block graph. First as a sandwich — norm+QKV on the GPU, one
  sync, the CPU attends (it owns the KV cache), O+FFN encode into the
  next buffer with the following GDN run (~17 syncs/token instead of
  ~64) — and then all the way: new MSL kernels for per-head qk-norm +
  partial RoPE with gate split, KV append into per-layer shared-memory
  mirrors, grouped online-softmax attend with Born importance banked via
  `atomic_float`, and the sigmoid output gate. One wait per token. The
  CPU cache stays the owner of record — any divergence (eviction,
  rollback, a CPU-path append) re-uploads the mirror, and after each
  token the appended row replays through the normal `append` +
  importance bookkeeping. Guards fall back per-layer to the sandwich
  (`CMF_GPU_ATTEND=0` forces it; `CMF_GPU_BLOCK=0` disables the graph).
- **Early commit**: the graph submits each command buffer as soon as it
  is encoded, so the GPU crunches layer N while the CPU encodes N+1 —
  continuous submission also keeps the Metal clocks warm (measured 5.8 ms
  warm vs 8.8 ms mixed per block). The token's single `sync` waits only
  on the last buffer (queue order covers the rest).
- **Hybrid q1 prefill rides the token graph**: the chunked CPU
  prefill-GEMM is walled by the sequential scalar GDN recurrence on q1
  hybrids, so the prompt now runs position-by-position through the same
  graph as decode (Bonsai-27B TTFT 12.4 → 3.5 s). Pure-attention models
  keep the batched path, where chunk-GEMM amortization wins.
- **Prefill on the Apple matrix units** (macOS): big prefill batches
  route through Accelerate `cblas_sgemm` over dequantized f32 tiles
  (scale folded in, pool-parallel dequant, tiles stay in cache) — the
  same engine llama.cpp's `-ngl 0` prefill uses. The prefill chunk is
  platform-adaptive (512 on macOS; `CMF_PREFILL_CHUNK` overrides), and
  `CMF_ACCEL=0` opts out. Decode (M=1) never takes this path.
- **Batched causal attention for prefill**: the chunk preps and appends
  every position first, then attends per KV group in two fat GEMMs
  (scores `Q·Kᵀ` with the group's Q-heads stacked into one panel, and
  `P·V` after a causal masked softmax that zeroes the invisible tail),
  with Born importance from the masked column sums. Softmax `exp` is a
  NEON Cephes-style polynomial — scalar `expf` over a long prefill's
  ~10⁸ calls would have eaten the GEMM win. The quadratic wall is gone:
  pp1024 390 → 976 tok/s. Chunks under 32 positions and non-F32 KV
  modes keep the exact per-position order.
- **`cortiq bench --core`** — llama-bench-contract timing: greedy argmax
  without the sampler's full-vocab working copy, no repetition-penalty
  pass, no per-token confidence softmax (`Pipeline::set_confidence`).
  The default `bench` still measures the full production loop. The
  clone-free greedy argmax also lands in production for every
  greedy-with-no-penalty caller.
- q1 Metal matvec goes four rows per simdgroup with per-tile processing
  (halves the L1 activation traffic per weight byte; the earlier
  four-row attempt cached the whole x block and spilled).

### Changed

- **Numerics contract, stated plainly**: GPU-graph decode and GEMM-path
  prefill are distribution-equivalent to the CPU path (first-token
  probabilities match to ~0.3%, PPL matches) but not bit-identical on
  every prompt — floating-point reductions run in a different order.
  This was already true of every GPU offload since 0.3.3; now it is
  documented instead of implied. CPU paths remain bit-exact (21 suites +
  token-for-token golden parity).

### Measured, for the record

- llama.cpp head-to-head (Qwen2.5-0.5B q8, M4, interleaved, fresh
  processes): tg128 — theirs 165.5 tok/s at its best `-t 6`, 129.4 at
  its default `-t 4`, 150.9 on Metal; CMF `--core` 151–158. pp512 —
  theirs 1168, CMF 1017–1037; pp1024 CMF 976.
- Dead ends, measured and reverted: an XOR sign-flip in the q1 kernel
  lost 23% to the `select` chain the Metal compiler already emits
  optimally; double-buffering the prefill dequant against the sgemm
  lost ~6% (Accelerate's own threads starve); a hybrid CPU∥GPU lm_head
  split on UMA lost 15% (the runtime probe had it right all along).

## [0.3.3] — 2026-07-16

1-bit models get a real GPU: Bonsai-27B (q1) decode on an Apple M4 goes
from 2.2 to 5.0–5.8 tok/s.

### Added

- **q1 on the native Metal backend**: a two-rows-per-simdgroup matvec
  kernel over the 6-byte tiles (aligned u32 pair loads, activations hot
  in L1), q1 trios in the FFN chain, q1 jobs in the batched matvec
  (QKV / GDN mixers), and a single-matvec route for out_proj/lm_head —
  all no-copy over the mmap (UMA), GPU math in plain f32 (no A8
  activation quantization at all). wgpu refuses q1 jobs honestly until
  its WGSL kernel lands.
- **Whole-block GDN graph**: a run of consecutive GatedDeltaNet layers
  executes in ONE command buffer — rmsnorm (Qwen/Gemma), mixer, causal
  conv + silu, decay/β gates, per-head l2 norms, the delta-rule
  recurrence with gated RMSNorm, out_proj, residuals and the FFN chain —
  hidden state device-resident across the block, one sync per block of
  ~3 layers instead of ~12 per layer. Recurrent states round-trip
  through shared memory, so the CPU stays their owner and prefill / MTP /
  probe paths remain coherent by construction. Anything ineligible
  falls through to the per-layer path unchanged; `CMF_GPU_BLOCK=0`
  opts out.
- q1 ops skip the runtime probe on native Metal (`gpu::q1_force`): the
  CPU q1 kernel is load-port-bound and probe alternation itself cooled
  the device between samples. Other dtypes and backends keep probing.

### Measured, for the record

- A synchronous Metal command-buffer round trip costs ~1.3 ms while
  back-to-back submits pipeline at 0.022 ms — the wall is completion
  latency, which is why the block graph (fewer submissions) is the
  design, not faster waits. A shared-buffer "fast flag" completion
  trick was tried and reverted: flag visibility does not order other
  buffers' write-backs (parity tests passed, real decode corrupted).

## [0.3.2] — 2026-07-16

The 1-bit release: a 27B in 4.8 GB on a 24 GB MacBook.

### Added

- **`q1` (dtype 12)** — 1-bit binary weights for 1-bit-TRAINED models
  (Bonsai / BitNet class): 6-byte tiles `[f16 scale][4B sign bits]` per
  32-group, 1.5 bits/weight; the scale is the group's mean |v| — the
  L2-optimal binary level, which recovers a binary-trained checkpoint's
  stored levels exactly. Explicit opt-in (`--quant q1`): as PTQ of a
  normal model it destroys quality. Fused kernels on all paths; on ARM
  the vtst mask feeds `sdot` directly (0xFF = −1) via
  `dot = −(2·sdot(mask, x) + Σx_group)` — no ±1 expansion at all, with
  per-group activation sums shared across every row. Verified
  end-to-end on prism-ml Bonsai: 1.7B q1 = 334 MB (vs 1653 MB q8) with
  greedy output token-identical to q8; 27B = 4.75 GB, ~3.2 tok/s on an
  M4 with `CMF_THREADS=10`.
- **qwen3_5 hybrid runs from safetensors**: GatedDeltaNet linear layers
  + full attention every 4th (Bonsai-27B class), 248K vocab, MTP head —
  the native converter maps it 1:1; hybrid GGUFs stay refused by
  design (the mixer tensors would be lost).
- Q1 joins `matvec_many` multi-matrix jobs (QKV / gate+up fuse again on
  new-arch models) and the four GDN input projections run under one
  pool dispatch — hybrid 27B: 449 → 353 dispatches/token.

### Changed

- **GDN/linear-core state is f32** (the vendor operator's own dtype —
  `mamba_ssm_dtype: float32`): SIMD-able elementwise state passes
  (read×2/write×1 instead of ×2/×2), heads fan out across the worker
  pool, per-worker scratch from the shared freelists. State memory
  halves; the GDN oracle stays green at 1e-3. `vmf_phase` keeps f64
  math per cell at half the storage.
- **Worker pool defaults to `CMF_POOL_SPIN=4000`** (was 0): at ~39
  dispatches/token, park-immediately paid an unpark syscall per worker
  per dispatch. Measured on M4: q8 decode +14%, q4t +27%, the 50M bench
  model +74%. `CMF_POOL_SPIN=0` remains the share-the-box serving mode.
- q8 4-row interleaved repack ships opt-in (`CMF_REPACK=1`): the
  single-stream hypothesis lost on Apple Silicon (the prefetcher likes
  four adjacent row streams more); kept for x86 experiments,
  bit-identical either way.

## [0.3.1] — 2026-07-16

The GPU release. Field report that triggered it: a 35B model (70 GB bf16 →
35 GB CMF) decoding at 1.9 tok/s on an RTX 4090 — the weights were streaming
through DDR on every token because `CMF_GPU` offload was effectively
unreachable for layer-class matrices and the release binaries shipped
without the backend. Both are fixed; the design principle that emerged is
**measure, don't trust**: enabling the GPU must never make you slower.

### Added

- **Runtime GPU-vs-CPU probe**: per op class (FFN chain, large matvec,
  prefill GEMM, QKV batch) the first calls alternate between the GPU arm
  and the pure-CPU arm, both timed; cold GPU calls (weight upload, cache
  fill) are discarded, and after six clean samples per arm the faster arm
  wins for the rest of the process. Measured on a discrete Radeon Pro 560X,
  where per-op submit+poll costs ~3–4 ms: the old always-GPU path lost 4×
  on decode and 8× on prefill against CPU AVX2 — the probe settles on CPU
  and keeps full speed; on stacks with cheap submissions the same probe
  keeps the GPU. `CMF_GPU_PROBE=0` restores unconditional offload.
- **VRAM-budget weight residency** (`CMF_GPU_VRAM_MB`, default 8192 on
  discrete cards, unlimited on unified memory): tensors become resident in
  first-touch order — decode touches layers in order, so the budget behaves
  like llama.cpp's `-ngl` without a flag. Over budget → the honest CPU path.
- **Device-class thresholds**: discrete cards take FFN/QKV-class matrices
  (≥4096 rows), unified memory only lm_head-class (≥65536) —
  `CMF_GPU_MIN_ROWS` overrides. `WGPU_BACKEND=vulkan|dx12|metal|gl` pins
  the wgpu backend.
- **Fewer polls per token**: Q/K/V projections in one device submission
  (`matvec_batch`, one pooled staging buffer for all readbacks), the dense
  FFN chain gate→silu·up→down in one command buffer with device-resident
  intermediates (the MoE block path, now also covering `q8_row`), pooled
  per-op scratch buffers, and per-tensor scale/col buffers cached across
  tokens. Field path for a dense model: 7 → 3 submissions per layer.
- **Pipeline slot pool in `cortiq serve`** (`CMF_SERVE_SLOTS`): N pipelines
  over one shared mmap check out per request — concurrent requests no
  longer serialize on a mutex.
- **`vbit_ro` (dtype 10)**: v-bit with an in-file row-offset table — readers
  index any row in O(1) instead of scanning bit-lengths; the native
  converter writes it by default, legacy `vbit` (dtype 8) stays readable
  forever, the Python reference reader handles both.
- **`q4_tiled` (dtype 11, `--quant q4t`)**: 18-byte interleaved q4 tiles
  (`[f16 scale][16B nibbles]`) — scale and payload land in one cache line.
  Kernel A/B: ARM ×1.66, x86 ×1.13; end-to-end on Qwen2.5-0.5B: prefill
  +24–32%, decode at parity, bit-identical to `q4_block` (parity-tested).
  The `q4` default stays split-layout until the x86 end-to-end pass.

### Hardened

- `validate_payload` now checks exact payload lengths for every dtype
  (v-bit included: exact bit-length sum, offsets monotonic, bounds before
  slice), and duplicate tensor names are rejected at open and shard merge.

### Fixed

- **Correction of the 0.3.0 performance claim.** The published
  "+70% pp512 / +60% tg128 over llama.cpp" table had unknowingly
  benchmarked an x86-64 build of `llama.cpp` under Rosetta 2 emulation
  (no SIMD). Against native arm64 `llama.cpp` on the same machine, CMF is
  currently **behind**: −67% pp512 / −38% tg128 on CPU (and llama.cpp's
  Metal GPU path is ~9× ahead on prefill). The README table is corrected
  and the correction is kept visible; the file-size (−26%) and
  quant-quality (+0.38% PPL) rows were unaffected.
- Release binaries now build with `--features gpu` on every platform — the
  0.3.0 artifacts shipped CPU-only, so `CMF_GPU=1` did nothing for binary
  users (the root of the field report above).
- The v0.3.0 release was missing `cortiq-aarch64-apple-darwin.tar.gz.sha256`
  (upload interrupted); re-uploaded.
- CLI logs now go to stderr — `bench --json` and piped generation output
  stay machine-parseable under `RUST_LOG`.

## [0.3.0] — 2026-07-16

The performance release: ten waves of engine work guided by the internal
performance roadmap, verified on three machines (Apple Silicon, Intel AVX2,
Xeon Granite Rapids). First like-for-like run against `llama.cpp` (b9310,
Qwen2.5-0.5B, CPU-only, 8 threads, exact attention both): **pp512 +70%,
tg128 +60%, file −26%**, with quantization quality matched (CMF q8 vs own
f16: +0.38% PPL over 12×512 windows). One model on one machine — the full
matrix is still open; reproduce with `cortiq bench --json`.

> **Correction (0.3.1):** the +70%/+60% figures above are wrong — that run
> had benchmarked an x86-64 `llama.cpp` under Rosetta 2 emulation. See the
> 0.3.1 "Fixed" section; the engine-work speedups over CMF's own baseline
> and the file-size/quality rows stand.

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
