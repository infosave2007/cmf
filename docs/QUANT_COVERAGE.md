# Quantization format coverage by execution path

*Languages: **English** · [Русский](QUANT_COVERAGE.ru.md) · [中文](QUANT_COVERAGE.zh.md)*

Status: code audit at 0.5.65+wip, 2026-08-10. The unit of truth is the
file:line references in the text; where they disagree with the code, the code
is right and the document gets fixed.

Motivation: "optimization is not for one format". Below is a measured map of
which dtype actually executes on which path, and the full list of gaps of two
kinds: "the kernel exists, the call cuts it off" (cheap to fix) and "there is
no kernel" (deliberate gaps).

## dtype × path matrix

Paths: 1 CPU matvec · 2 CPU matvec_many · 3 CPU fused FFN
(matvec_silu_mul) · 4 CPU matmat (prefill) · 5 Metal per-op ·
6 Metal block/token graph (proj_abs) · 7 Metal chunk prefill ·
8 wgpu token graph (graph_weight) · 9 wgpu per-op · 10 MoE-GPU (moe_parts)

| dtype | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| F32 | ✓ | ✓ | — | ✓ | — | (GDN f32 fields) | — | ✓ | — | — |
| F16/Bf16 | →f32 at load; no quant kernels ||||||||||
| Q8Row | ✓ | ✓ | **—** | ✓ | ✓ range | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q4Block | ✓ | ✓ | ✓ | ✓ | **off** (gpu.rs:1001) | ✓ | **—** | ✓ | ✓ | **—** |
| Mix84/U8/Q4Col | reserved: loading fails honestly (quant.rs:831) ||||||||||
| Vbit/VbitRo | ✓ | ✓ | — | ✓ | CPU-only by construction (portable reference) |||||
| Q8_2f | ✓ | ✓ | **—** | ✓ | ✓ range | ✓ | **—** | **—¹** | ✓ | ✓ |
| Q4Tiled | ✓ | **—** | ✓ | ✓ | **no dispatcher**² | ✓ | ✓ | ✓ | partial | ✓ |
| Q1 | ✓ | ✓ | **—** | ✓ | ✓ | ✓ | **—** | ✓ | ✓ | Metal ✓ / wgpu **—** |
| Q1G | ✓ | ✓ (also mixed with Q1) | ✓ | ✓ | ✓⁵ | ✓⁵ | **—** | ✓ (kind 10) | ✓ | **—** |
| Q1S | →f32: the converter writes it (convert.rs:513), the engine has no kernels ||||||||||
| Q1T | ✓ | ✓ | ✓ | ✓ | ✓ | ✓+overlay | **—** | ✓ | ✓ | **—** |
| Q4TiledP | ✓ | ✓ | ✓ | ✓ | (test wrapper) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q2TiledP | scalar³ | **—** | **—** | ✓ | **—** | **—⁴** | **—** | MoE experts ✓ | matmat | **—** |
| Mxfp4 | ✓⁶ | one by one | one by one | ✓ | CPU⁶ | **—** | **—** | **—** (per-op) | ✓ mv/mm | V4.1 global bank ✓ |

¹ For q8_2f the wgpu graph needs a col-field kernel — not an arm, a kernel.
² q4t_matvec kernels exist on BOTH backends, but there is no `gpu::q4t_matvec`.
³ `q2tp_matvec` is a scalar exact loop, with no SDOT/VNNI variant.
⁴ Fixed in this pass: `q1_parts` no longer advertises Q2TiledP to the
Metal graph, which has no q2tp kernel (the plan was silently truncated).
⁵ Q1G on Metal (2026-10-17): the `q1g_matvec(_h)` kernels live in a separate
library `MSL_Q1G`; compiled and run on an M4 outside cargo
(`tools/metal_q1g_parity.swift`): bit-identical to `q1_matvec(_h)` after
switching to an explicit `fma` in the scale step (otherwise fast math folded
the shared scale of a tile pair). The host path (test
`gpu_q1g_matvec_equals_q1`) was only checked with
`cargo check --tests --target aarch64-apple-darwin` and was not run on a Mac;
on a compile error q1g falls back to the CPU, and other models are not
affected.
All other Q1G paths are bit-identical to Q1 on uniform scales (tests
`q1g_*`, ENGINEERING.md).
⁶ Mxfp4 (dtype 21, 2026-10-19, docs/FP4_NATIVE.md): CPU — the exact contract
is bit-identical to the f32 matvec of the dequantized matrix, A8W8 via
tbl/pshufb + sdot/maddubs; fused jobs (`matvec_many`, SwiGLU, MoE rows) run one
by one; there is no native Metal kernel — Metal answers false and the CPU
computes; wgpu — `mxfp4_mv`/`mxfp4_mm` and the global-bank kernels
`dsv4_moe_frame` (the dynamic pool of DeepSeek-V4.1); the other MoE frames
honestly decline an mxfp4 bank.

## "The kernel exists — the call cuts it off" gaps (by cost of the fix)

1. ~~Q2TiledP in `q1_parts` without a kernel in `proj_abs`~~ — fixed.
2. ~~Q4Tiled decode matvec without a dispatcher~~ — fixed: `gpu::q4t_matvec`
   + a probe arm (Metal; wgpu — an honest refusal until a discrete q4t stand
   appears).
3. Q4Block Metal per-op: `gpu.rs:1001 Metal=>false` while
   `encode_q4b_matvec` is alive — the standalone lm_head of a q4b model is not
   accelerated.
4. Q4t/Q4tp in wgpu `matvec_batch`: `BatchLayout` is modeled, Metal executes
   it, wgpu rejects the whole batch (gpu_wgpu.rs:20134).
5. ~~`gpu_batch_job` knows only Q8/Q1~~ — fixed: Q4t/Q4tp arms (raw
   activations, scales in the payload); probe arbitration with OpClass::Batch
   keeps the winner honestly (on M4 and M1 the probe kept the CPU — the route
   is for configurations where a Metal batch amortizes).
6. ~~q2tp MoE in the batch/prefill graph~~ — fixed (2026-08-10) with the
   Qwen3.6-35B-A3B-Escha-W2 parity oracle: the builder got the token graph's
   q2tp ladder (gu_q2 end to end), the per-position lane branches
   gu_q2 → `moe_gate_up_q2tp` + the exact `moe_down_q4tp` (the chain path's
   pair), the full-chunk q4tp lane EXPLICITLY excludes gu_q2 (a mixed profile
   has q4tp=true — without `!gu_q2` it would grind q2tp bytes with the q4tp
   kernel), and the GU stride uniform follows
   `expected_nbytes(Q2TiledP)`. CPU⇔wgpu output matched byte for byte, no
   bgraph refusals, 151/151 tests. Deliberately NOT wired:
   `bt_moe_gate_up_q2tp(_r4)` — the full-chunk 2-bit variant stays an
   optimization with its own mg layout (the per-position lane lives in the
   same submit), and the draft-grade `moe_down_q2tp_b`
   (DSpark grade, "not required to match bit for bit") is unfit for the main
   path by the accuracy contract.
7. `moe_parts` does not accept Q2TiledP/Q1T/Q4Block; `MoeJob` has no q2tp
   flag — the ordinary MoE path bypasses the existing wgpu kernels.
8. `matvec_silu_mul`: ~~Q1~~ fixed (a fused arm with shared group
   sums; M1 worker bonsai 9.0→7.5-7.9 ms/token, −14%); ~~Q8Row~~
   fixed (a portable `q8_row_dot` with a fast==scalar oracle test;
   the benchmark waits for a q8 checkpoint on the stand). Q8_2f is excluded
   DELIBERATELY: its column field prescales activations per tensor and breaks
   the contract of the shared split_act — it stays two-pass. Q2TiledP has no
   CPU-i8 kernel at all (see "no kernel").
9. `matvec_many` lacks Q4Tiled/Q2TiledP; `matvec2_many` is narrower still.
10. `moe_gate_up_many`/`moe_down_many` — Q4TiledP only.
11. Metal `q8_matvec`/`q8_2f_matvec` — tests only; decode reaches
    Metal q8 only through the hybrid split's `q8_matvec_range`.
12. Q1 MoE: wgpu rejects it (19951), Metal executes it; the comment
    "q2tp down does not exist" (11341) is stale — `moe_down_q2tp_b` exists.
13. The `GraphW` doc lies about the kind codes (gpu.rs:723 "2=q4_tiled", in
    fact 2=Q4Block, 5=Q4Tiled, 6=Q4TiledP) — the very confusion that has
    already led to garbage output (qtensor.rs:568).

## "No kernel" gaps (honest, until a measured need)

- Q1S: the engine does not know it at all — such files run at 4 bytes/weight.
- ~~Q2TiledP: no i8 variant of the CPU matvec and no GPU matvec on either~~ —
  closed (2026-08-10): i8 CPU (NEON shift+mask+vld4 + group sums,
  a fused arm, q2tp in moe_gate_up_many) and Metal `q2tp_matvec_jobs`
  (the mixed trio through MoeJob; on M4 the probes honestly keep the CPU — 5.9
  against 2.8 with the GPU forced; the kernel is for GPU-strong devices).
  Follow-up: a device-vs-dequant unit oracle and the MoE block in the shared
  command buffer.
- Vbit/VbitRo: CPU-only by construction.
- Q1/Q1T/Q4Block in the Metal chunk prefill (`kind_of` knows Q4t/Q4tp/Q8).
- Q8_2f col-field in the wgpu graph.

## Practice gathered on the stand (27B q4tp, M4 24GB + M1 16GB)

- The Metal block graph EXECUTES q4tp/GDN in full (a 32-item plan, layers
  0..64) — "the graph is only for q1" is a myth born of the name `q1_parts`.
- `hd=256` turns device attend off by default → 16 sandwich syncs per
  token; `CMF_GPU_ATTEND=256` enables the wide kernel: the local
  side of the split 159→108 ms/token (−35%).
- Paging masks everything: any measurement is valid only when the span
  fits in residency (see the stand notes).

## Metal MoE in a shared command buffer — IMPLEMENTED (2026-08-10)

The spec below has been carried out: `moe_topk_select` (MSL, a unit oracle
device==`moe_route` with tie-break and both norm modes), `MetalFfn::{Dense,Moe}`
in the token-graph layers, `encode_post_moe_ffn` (addnorm → f32 router →
select → gate/up jobs (q4tp/q2tp by profile) → moe_silu → down jobs →
weighted reduce → axpy — all in a shared command buffer), the builder
`metal_moe_graph_parts` with honest refusals (sigmoid/bias/τ routers,
masks, per-expert scale, router_input_norm, non-Silu — CPU).

The device-attend policy was moved AFTER the plan scan: the hd>128 switch
had been measured on dense models, and an MoE plan inverts it — every
CPU-attend sandwich costs commit+wait. Measurement (W2 q2tp MoE 34.7B, M4,
steady decode, bench --json): sandwich 14.7 tok/s, device attend
**24.8–27.1 tok/s**, pure CPU 18.8. Submits/token 39.8 → 27.4
(with prefill amortization). Dense models do not change the policy.

Regression matrix: bonsai q1 (texts ==, 95.8 tok/s), nanbeige q4t
(texts ==, 17.7), W2 q2tp MoE (graph + CPU attend: texts 48/48 == with the
exact CPU, confidence drift 0.098 inside its own CPU envelope
i8↔exact 0.11; device attend diverges at the 32nd token — the drift class of
the pre-existing knob, not of the MoE branch), 253/253 engine tests.

Open: **35B q4tp (17.8 GB) is larger than the M4's Metal maxBufferLength
(13.6 GB)** — the file's no-copy buffer cannot be created, so the whole Metal
path (both per-op and graph) is unavailable on this file; W2 (12.9 GB) fits.
Follow-up: a buffer window over the layer span instead of the whole file — the
network split needs the same (each side maps only its own span).

## Design: Metal MoE in a shared command buffer (the executed spec)

Motivation, measured on 35B-A3B q4tp (M4): CPU 11.6-12.1 tok/s, Metal
9.8-11.0 — against a bus ceiling of ~30-60 tok/s (~1.9 GB active per token).
The gap is orchestration: per-op MoE does commit+wait_until_completed on
EVERY layer (~30 × ~1.3 ms), because the layer's routing is computed by the
CPU on the host. The cure is the whole token in one command buffer:

1. **MSL `f32_matvec`** (router: n_exp × hidden, f32 kind-4) — if it is
   missing; check the existing simple kernels.
2. **MSL `moe_topk_select`** — from logits[n_exp]: top-k indices + softmax
   weights + shared-gate sigmoid (the twin of wgsl `moe_select`,
   gpu_wgpu.rs:~15864); k ≤ 16, n_exp ≤ 256, one threadgroup.
3. **TokenGraph: `Item::MoeFfn`** in the `q1_graph_gpu` plan (remove the
   scan's `FfnKind::Dense` gate): norm → router → select → jobs kernel
   (`q4tp_matvec_jobs`/`q2tp_matvec_jobs` — exist) → `moe_silu_jobs` →
   down jobs → `moe_reduce` (exists) — all encoded without a commit; sync only
   where it already happens (the CPU-attend sandwich / the final readback).
   Buffer use: sel/act/eo — through io_bufs, as in
   `moe_block_jobs_q4tp`.
4. Parity oracles are ready: W2 (q2tp) and 35B-q4tp, CPU⇔Metal texts.
5. Expectation: −30-40 ms/token of synchronization → Metal ≥ CPU; beyond that
   only kernels move the shared ceiling (the scales_into cache on the device
   already exists — a ladder in threadgroup memory).

Risk: the q1 flagship (35B Qwen3.5 q1) runs through this same plan — a
regression is unacceptable; the MoE branch must not touch the Dense path. Do
it as a separate pass with the full regression matrix (bonsai q1, nanbeige q4t,
27B q4tp dense, W2 q2tp MoE, 35B q4tp MoE).

## Decode profile on a discrete card (2×RTX 5090, 2026-08-11)

Measured with `CMF_GPU_TS=1` (the per-frame GPU profiler) on W2 34.7B,
2048 hidden, 40 layers, top_k 8 + shared:

| phase | µs/token | per layer |
|---|---|---|
| GDN mix (30 layers) | 1949 | 65 |
| MoE FFN (40 layers) | 2717 | 68 |
| attn mix (10 layers) | 767 | 77 |
| tail (norm+lm_head) | 189 | — |
| **total GPU** | **5630** | |

The full token is 8.3 ms at 121 tok/s — i.e., ~2.7 ms is spent off the device.

**Decode is NOT bound by the weights.** `CMF_TOPK_PROBE` 8→4 (half the
expert arithmetic with the same number of dispatches) changes the time by 3%.
The bus ceiling for 1.9 GB of active weights is ≈ 1.1 ms — we are 5× away from
it, and the gap is eaten by launch latency: a 2048-hidden layer is 30-70 µs of
work on a card that wants thousands of workgroups.

**Per-dispatch breakdown (`CMF_GPU_TS=2`, one layer of each kind):**
GDN — proj 39 µs, outp 16, step 11, conv 6, snorm 5; attention — attend
24, qkv 17, wo 14, rope 13, kv 4, gate 4; MoE — router 16, select 13,
down 14, gate/up 13, pre 9. The CHEAPEST dispatches cost 4-6 µs — that is
the floor of launch overhead, and at ~15 dispatches per layer × 40 layers
it eats about a third of the token.

**The knobs were checked — the defaults are already optimal, do not reopen:**

| knob | value | measured (W2, 5090) |
|---|---|---|
| `CMF_MV4` | 1 (default) | 124.0 against 77.7 tok/s at 0 |
| `CMF_MOE_FOLDSEL` | 0 (default) | 120.7 against 100.4 at 1 (the output is byte-for-byte the same — the path is correct, just slower) |
| `CMF_SLOTS_PER_GPU` | 1 (default) | 210 against 185 tok/s at 2 |
| `CMF_BATCH_K` | 0 (default) | 129.9 against 128.2 at 128 — the batch graph does not help this model |
| `CMF_COOP` | 1 (default) | GEMM 51.9 against 25.0 TFLOP/s at 0 |

Hence the next levers, in decreasing order of measured value:

1. **Sequence batching (continuous batching).** The only
   way to fill the SMs: one submit computes B independent
   sequences. It needs per-sequence KV mirrors in the
   graph (today the key is `(kv_id, layer)` — almost what is needed).
   CHECKED AND REJECTED as a substitute: two slots per card give 185 tok/s
   against 210 for one (`CMF_SLOTS_PER_GPU`) — the slots fight over the queue
   instead of overlapping.
2. **A graph bind-group cache.** 12,617 allocations per token against 363 on
   the CPU path — these are wgpu objects recreated every token (uniforms are
   already cached by content, bind groups are not). Design: a key
   `(site_id, layer)` at the 35 call sites of the `bg(...)` helper inside
   `forward_token_graph` + a generation counter that
   `Scratch::ensure` bumps when it recreates a buffer. The cost of a mistake is
   silently wrong bytes, so do it with a parity gate on every model.
3. **Prefill in split mode**: two submits per position against one
   (67.8 against 156.6 tok/s). The cure is pipelining segments over
   positions — segment 1 computes position p while segment 0 takes p+1.

## Why the probe can leave diffusion on the CPU (measured 2026-08-11)

Symptom: in the MiniMax DiT the `out` projection (5376×7168) cost 9.8 s per
step, although the same kernel on the same shape does it in 8.4 ms (kernel test
`wgpu_q4tp_mm_throughput`, 53 TFLOP/s). It did NOT reach the device.

The mechanism is two properties of the probe stacked on top of each other:

1. **Cold GPU samples are discarded.** In diffusion each layer is
   touched once per step, so on the FIRST step every GPU measurement is
   marked cold (weight upload) and does not count. The probe never collects
   a GPU arm and keeps alternating arms.
2. **The alternation counter is shared per class and follows the call order.**
   A block has exactly two arbitrated GEMMs (qkv, out): qkv is always even,
   out is always odd — and for the whole first step `out` lands on the CPU arm.

The result was measured: step 1 — 9.8 s, step 2 — 2.8 s (weights warm, the
verdict reached). So this is a one-off cost of ~7 s per render, not a
systemic one.

The early-verdict threshold was lowered from 3× to 2× (the gap here is 2.8×) —
this did NOT fix the case (item 1 gets in the way), but it removes needless
alternation where the winner is obvious. The real cure: during the probe, if
the weights are not resident yet, prefer the GPU arm (the upload is needed for
the following steps anyway) — `probe_resident_or_upload` already exists for
this.

## The next video lever: attention on tensor cores (calculated)

After the FFN move and the probe fix, the MiniMax DiT step profile (512×288,
RTX 5090, `CMF_MMH3_PROF`) looks like this: **attention 7.5 s (40%)**,
qkv 3.0, out 3.0, fc1 (fused FFN) ~5.

Attention arithmetic per block: n=1859, nh=42, hd=128 →
QK = 2·nh·n²·hd ≈ 37 GFLOP, PV the same. Over 50 blocks that is ~3.7 TFLOP
per step, and it takes 7.5 s — i.e., **0.5 TFLOP/s**, while the q4tp GEMM
kernel test on the same card holds **51.9 TFLOP/s**.
The difference is not in the bus (q/k/v is ~120 MB per block, ~1.6 s over the
whole render) and not in launches (126 dispatches per block ≈ 31 ms per step) —
it is that `dit_qk`/`dit_pv` are scalar f32 kernels, not matrix ones.

Design: the backend ALREADY has a cooperative f32 GEMM (`gemm_nt_coop`,
which `gemm_nt_f32` uses). What is needed is an entry into it over device
buffers:

1. QK: q_h [n×hd] × k_h [n×hd]ᵀ → scores — exactly the NT shape, it fits
   without a transpose.
2. Softmax over rows — as is (`dit_softmax`).
3. PV: scores [n×n] × v_hᵀ [hd×n] — needs vᵀ; either prepare it on the
   host (v is already gathered head-major there, it is the same pass), or
   add a transposing kernel.
4. Gate: frame parity against the current path (threshold 3.4% — the
   CPU↔GPU envelope) plus a kernel test on the attention shape.

Expected from the arithmetic: 7.5 s → 1-2 s per step, i.e., the render
127.8 → ~100 s.

**CHECKED 2026-08-11 — THE PREMISE TURNED OUT WRONG.** The kernel was written
(`DIT_COOP_SRC`/`dit_gemm_coop`, behind `CMF_DIT_ATTN_COOP=1`), QK through the
matrix cores computes CORRECTLY (the frames match, delta 0.000), but the
step did not change: attention 7.5 s against 7.6. So attention is NOT
bound by QK arithmetic, and the "0.5 TFLOP/s" estimate measured something other
than what spends the time. What remained suspect, in decreasing order:

1. **Softmax**: nh·n² = 145 M elements per block, one workgroup
   per row — 78 k groups per block, 3.9 M per step.
2. **Uploading q/k/v and reading back the panel**: 160 MB per block, 8 GB per
   step.
3. **Dispatches**: 42 heads × 3 passes × 50 blocks = 6300 passes per
   step (at a floor of 4-6 µs that is ~30 ms — small, but not zero).

**SEPARATED (2026-08-11).** A timer was put separately on the host repack
and on the device call: of the 7.3 s of the attention phase, **4.4 s is the
repack on the host** (assembling head-major q/k/v from the interleaved qkv),
and only ~2.9 s is the device itself. So the bottleneck was NOT in the kernels
and not in the arithmetic but in data preparation on the CPU — exactly what
the previous attempt (coop QK) could not speed up in principle.

The kernel `dit_qkv_split` + the entry `dit_attention_packed` were written: qkv
goes to the card in one piece and is laid out by head THERE. Checked at
64×32 (the render is correct), not yet measured at full size — so it is
enabled explicitly, `CMF_MMH3_ATTN=split`. To finish: a 512×288 measurement +
frame parity, and if it holds, make it the default.

The earlier list of suspects (softmax, traffic, dispatches) stays for the
remaining ~2.9 s: `CMF_GPU_TS=2` inside dit_attention, or a simple A/B with the
softmax turned off (knowingly wrong output, but an honest time).
The kernel is left behind a flag — it is correct and will come in handy when the
bottleneck turns out to be where I assumed.

## What remained in attention after moving the repack (2026-08-11)

The default now lays qkv out by head ON THE CARD (`dit_qkv_split`):
repack 4.4 → 1.5 s, render 130.0 → 111.8 s, frames bit for bit.

The remaining device part is **5.2 s per step** (29% of the step). Coop QK
was remeasured WITHOUT the repack and is again neutral (5.2 against 5.3) — so
the QK GEMM is not the bottleneck in any arrangement. That means the remainder
sits in the softmax (nh·n² elements, one workgroup per row → 78 k
groups per block) and/or in PV.

**SEPARATED (`CMF_DIT_ATTN_PROF=1`, 512×288):**

| pass | seconds/step | share |
|---|---|---|
| QK | 1.65 | 22% |
| softmax | 0.97 | 13% |
| **PV** | **4.76** | **64%** |

PV is three times as expensive as QK at THE SAME FLOPs (both n²·hd) — so it
is not about arithmetic but about access: its right operand `v` is laid out
[n][hd], and the GEMM reads it by columns, scattered. QK has no such problem,
because k is laid out exactly as the NT product needs it.

The cure (in order of cost): transpose v on the device once per block
(n·hd elements — pennies next to n²·hd) into [hd][n], after which PV becomes an
ordinary NT product like QK and, on top of that, fits the already written coop
kernel `dit_gemm_coop`. Expectation: 4.76 → ~1.5 s, i.e., step 18.9 → ~15.6 and
render 111.8 → ~95 s.

**PV as NT: faster and not yet correct (2026-08-11).** The transpose was
moved into its own module (the auto layout is built PER ENTRY POINT and
reindexes the bindings — a kernel touching a subset cannot be bound).
After that the path worked and gave **PV 4.76 → 2.82 s (1.7×)**,
but the frames diverged by 59%: the NT form takes the reduction width as
`cols4 * 4`, while PV reduces over n = 1859, which is truncated to 1856 and
silently loses three columns of every score row. A gate was added (`n % 4`),
and the path stays behind `CMF_DIT_PV_COOP=1`. To enable it: give the kernel
the true k next to `cols4` (the stride still comes from cols4, the loop bounds
from k), then check the frames and remove the flag. The win is confirmed by
measurement; this is a matter of correctness, not of the idea.

This is exactly an example of the rule "separate first, then write": two
kernels written before the separation turned out correct and useless, while
the separation cost one flag and one run.

## DiT step profile after all the moves (2026-08-11, final)

The accounting was corrected (slots 2 and 3 measured the same work — a 14.1 s
step read as 26.1; the sum of the slots now agrees with the external timer):

| phase | s/step | share |
|---|---|---|
| fc1 (fused FFN) | 5.1 | 36% |
| qkv | 3.2 | 23% |
| out | 2.8 | 20% |
| attention (QK+softmax+PV) | 2.2 | 16% |
| the rest | 0.8 | 5% |

**Attention has stopped being the main item** — the projections were. And they
have a gap against the kernel: qkv is 430 GFLOP per block, 21.5 TFLOP per step,
and takes 3.2 s = **6.7 TFLOP/s**, while the kernel test on the same shape
holds 51-61. The difference is not in the kernel but in the wrapping of EACH
call:

- reading back y: 1859×21504×4 = **160 MB per block**, 8 GB per step;
- uploading x: 40 MB per block;
- unpacking the plane: 115 M elements per block (noticeable, but secondary —
  at b=1859 it is 1/1859 of the GEMM's work).

So granularity pays the price: every GEMM is a separate round trip over the
bus. Caching the unpacked planes is useless (this model has 38 GB of them) and
unnecessary — they are not the problem.

**DESIGN REFINEMENT (checked against the code).** Removing the qkv readback
"by just passing the buffer through" is NOT possible: between the GEMM and
attention the host applies the norm and RoPE to q and k (`norm_rope_w`,
mmh3.rs:866) — the data has to come to the host as long as the CPU does
this. So the order of work is the reverse of what it seems: first a
qk-norm+RoPE kernel on the device (it also removes the 0.4 s of the qknorm
slot), and only then does it make sense not to read back the qkv buffer.
The mechanism for the second step already exists — `q4tp_matmat_dev`
(tp_matmat with an optional readback).

**The `dit_qknorm_rope` kernel is WRITTEN and accepted by the device**
(2026-08-11). The sketch's trap was solved not inside the shader but outside
it: WGSL cannot take a storage binding as a parameter, so one kernel is called
TWICE — q and k differ in the binding of the output plane, the weights and
`src_off`. What remains: feed it the angles and weights from mmh3, take
`norm_rope_w` off the host and keep qkv on the card (the mechanism is
`q4tp_matmat_dev`). Below is the shape it was built to:
one workgroup per (token, head) pair, 64 threads; the norm needs a
reduction over hd, so thread-per-element will not do. Load the head's slice
into workgroup memory, reduce the sum of squares as a tree, multiply by
1/sqrt(ss/hd + eps) and by the norm weights, rotate the first `pairs` against
their partners, write into the plane. v is an ordinary scatter without a norm.
Host semantics (`mmh3::norm_rope_w`): the sum in f64, the rotation
`x[j], x[j+pairs]`, angles per token; on the device f32 is enough (the norm
uses a ratio, hd = 128 values). IMPORTANT: q and k go through the same
transform, but with DIFFERENT weights and into DIFFERENT planes — the helper
has to take both the input and the output, otherwise k will silently stay
unprocessed.

**Design of the next pass: the whole DiT block resident.** x enters the
block once, qkv → attention → out → FFN are computed in device buffers,
and only the block's result comes out. This removes ~200 MB of traffic per
block (10 GB per step) and turns four round trips into one. Expected from the
arithmetic: 11.1 s of projections → 3-4 s, step 14.1 → ~7, render 106.9 → ~70.
The gate is the same: frame parity against the current path (envelope 3.4%).

## Next video step: a resident chain in the VAE decoder

The VAE profile after the host loops were parallelized (`CMF_VAE3D_PROF=1`,
2 steps, 512×288, RTX 5090): 26.1 s in total — ffn 7.1 · qkv gemm 5.3 ·
attention 5.3 · out gemm 4.4 · repack+rope 3.2 · norms 0.5.

This is exactly the structure that the `dit_qkv_attn_out` fusion removed in the
DiT (the chain 13.8 → 7.5 s per step, −46%). Here the comparable piece is
qkv + repack + attention + out = 18.2 s of 26.1.

What prevents reusing `dit_qkv_attn_out` as is — three differences,
each of them checkable:

1. **Panel layout.** In the DiT each token carries `[q(inner) | k | v]`, in the
   VAE the panel is interleaved by head: `p*3*dim + h*3*hd` gives
   `[q_h | k_h | v_h]`. The `dit_qkv_split` kernel needs a layout flag in the
   uniform, not a new kernel.
2. **The qkv bias.** The DiT has none, the VAE adds it after the GEMM.
   The cheapest is to add it in the same `dit_qkv_split`, which is the only
   kernel that reads the whole panel anyway.
3. **A norm without a weight.** `rms_norm_plain` against the DiT's weighted one.
   The `dit_qknorm_rope` kernel computes `x*inv*w`, so a buffer of ones gives an
   identical result — no new kernel is needed, only a buffer.

The order is the one that worked in the DiT: first split the phase into host
and device (done — the profiler above), then write the kernel. And measure
only with `CMF_GPU_PROBE=0` on both arms, otherwise you compare the arbitration
lottery, not the change.

## Qwen3.6 27B: where the token goes (2×RTX 5090, measured with `CMF_GPU_TS=1`)

The baseline with pinned arms (`CMF_GPU_PROBE=0`): prefill 53.6 tok/s,
decode 43.9 — spread ±1%. The whole-token graph is **active**, and the model is
fully resident under the automatic VRAM budget (a forced 8 GB
drops it sixfold — 7.0 tok/s; that is a check, not a working mode).

The token on the device is 17.6 ms; the stack is hybrid (48 GDN layers + 16
full-attention layers):

| phase | time | layers | per layer | share |
|---|---|---|---|---|
| FFN of the GDN layers | 6.69 ms | 48 | 139 µs | 38% |
| GDN mixing | 5.93 ms | 48 | 123 µs | 34% |
| attention | 2.40 ms | 16 | 150 µs | 14% |
| FFN of the attention layers | 2.20 ms | 16 | 138 µs | 12% |
| tail | 0.42 ms | 1 | | 2% |

**The matvec is NOT the lever, and this corrects my own estimate.** At first I
computed the bandwidth from guessed dimensions and got "543 GB/s, a third of the
ceiling". The real dimensions — hidden 5120, 64 layers, a 14.26 GB file —
give about 148 MB of FFN weights per layer, so 139 µs is **~1.07 TB/s**
against a nominal ~1.7. Sixty-three percent of peak bandwidth for a
real kernel that unpacks on the fly is a good level, not
headroom.

The knobs confirm it: `CMF_MV4=0` (the one-row kernel) gives 14.1 tok/s
against 44.2 for the four-row one, and the sixteen-row `q4tp_matvec16w`
is **already selected** by itself for these weights. The area is tuned.

**The lever is host overhead: 20,134 allocations per token** (bind groups) and
the 2.7 ms gap between the device (17.6 ms) and the wall (20.3) — 13% of the
token, spent off the card, on top of a graph that is already submitted in one
piece. That is exactly the bind-group cache task, and now it has a price in
tokens, not just in allocations.
