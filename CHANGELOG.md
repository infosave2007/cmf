# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
