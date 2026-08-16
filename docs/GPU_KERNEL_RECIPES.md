# GPU kernel recipes — what actually moved the needle, and what did not

Every number here was measured on an RTX PRO 6000 Blackwell (Vulkan/wgpu),
decoding Qwen3.6-35B-A3B in the 2-bit `q2tp` profile, with greedy-parity
gates on every step. The recipes transfer across weight layouts — the
per-layout kernels in `gpu_wgpu.rs` share their structure, so a win in one
is a to-do in the others.

## Measure first: the tools

- `CMF_GPU_TS=1` — GPU timestamps at pass granularity, aggregated per
  (stage, layer kind). `CMF_GPU_TS=2` adds per-dispatch stamps inside the
  passes of the first layer of each kind. **This is the only instrument
  that told the truth all day.** Cost models built on dispatch counts,
  pass counts, occupancy or dependency depth each failed at least once.
- `CMF_SKIP_PROBE`, `CMF_LAYERS_PROBE`, `CMF_TOPK_PROBE` — timing-only
  ablations (output is garbage), for attribution when timestamps are not
  available.
- Compare **equal generation lengths only**: attention grows linearly with
  context, so a 400-token run is not comparable to a 240-token one.
- Never run two benchmarks at once, and check `pgrep cortiq` before
  believing a number. This bit us four times in one day.

## Recipe 1: vec4 loads for weights AND activations (the big one)

The scalar kernels issue one load per weight and one per activation —
LSU-bound long before memory bandwidth. Loading `vec4<u32>` weight words
and `vec4<f32>` activations:

- dense 27B FFN (q4tp matvec): 12.3 → 33.2 tok/s across the two steps
  (`q4tp_matvec4`: +2.3x, then register-blocked row pairs below);
- MoE q4tp `down`: 33 → 16 µs/layer;
- MoE q2tp `gate/up`: part of 73 → 79.

**Parity rule:** keep the ADD ORDER of the scalar kernel. A single WGSL
expression `d0 + d1 + d2 + d3` is left-associative — identical to the
retired k-loop's chain. Loads vectorize; math order stays.

**Binding rule:** declare the vec4 view at the SAME `@binding` slot as the
scalar view it replaces (two module-scope vars may share a slot if no
single entry point uses both). An auto layout lists only the bindings an
entry point actually reads — a view on a NEW slot silently drops the old
one from the layout and the bind group fails at runtime.

Ported so far, with the measured result on a real model:

| layout | kernel | before | after |
|---|---|---|---|
| q4tp | `q4tp_matvec4` / `q4tp_matvec16` | 12.3 (dense 27B) | 39.1 |
| q4tp MoE | `moe_down_q4tp` | 33 µs/layer | 16 µs/layer |
| q2tp MoE | `moe_gate_up_q2tp` | — | part of 73 → 79 |
| **q4t** | **`q4t_matvec8`** | **37.5** (Nanbeige-3B) | **77.6** |
| **q1t** | byte-per-five-codes read | **28.8** (Bonsai-8B) | **37.4** |

q4t keeps its weights u16-assembled — 18-byte tiles are 2-aligned, so only
the activation side vectorizes, and the twin binds FOUR slots (no vec4
weight view). q1t's win came from reading each base-3 code byte once for
its five codes instead of re-reading it per weight; the ternary decode
itself stayed.

**Still scalar (same win waiting): `q4b_matvec`, `q8_matvec`,
`q1_matvec` (binary), `moe_gate_up` (q4t) and `moe_gate_up_q4tp`, the
`*_b` batch twins, and `q4tp_mul_mm` (the dense-prefill GEMM, ~300
GFLOP/s today).**

## Recipe 2: register-blocked row pairs

8 rows per 256-thread workgroup, in pairs: every activation vec4 feeds two
rows' dot chains — the x side of the LSU load halves. Second-largest win
on the dense FFN. Same parity rule: per-row lane layout and add order
unchanged.

## Recipe 3: grid shape — one warp per workgroup is poison

The attention kernel ran 16 workgroups of 32 threads with a 257-stride
shared accumulator and a five-level 256-wide merge: **137 µs per layer at
fifty positions of context**. The decode twin (`gqa_attend_dec`, 256
threads per head: lanes are positions for the score pass, output dims for
the value pass, online softmax across 256-position chunks) does the same
math in 46 µs. If a kernel's grid cannot cover the SMs, its latency is
the grid's, not the math's. The same disease in `gdn_step` (nv=32
workgroups) responded to a (head × column) grid — `gdn_step_par`.

## Recipe 4: match the memory layout's grain

The GDN state `S` is `[nv][dk][dv]` and its rows are dv-contiguous, but the
step kernel gave one workgroup one COLUMN — every lane strided the row at
`dv` floats, so each 4-byte read pulled a 32-byte sector and threw 28 away
(~250 GB/s). Four columns per workgroup, loaded as `vec4`, made every
access 16 bytes and the four column reductions ride together: **99.8 → 105
tok/s in one change**, the single largest step after the attention rewrite.

The general form: before optimizing a kernel's math, check whether its
LANE→ADDRESS map matches the buffer's contiguous axis. Row-major data with
a column-per-lane grid is the same bug in any layout — `q8_row`'s scales,
`q1t`'s overlay and the KV cache all have axes worth re-checking this way.

## Recipe 5: pass fusion — boundaries move, order does not

Dispatches inside one compute pass serialize with memory visibility, so
strictly-serial single-dispatch stages (norms, residuals, gates) can ride
the next pass as a prologue/epilogue instead of opening their own:

- mid-layer norm + layer-tail norm into the FFN pass: a MoE layer is TWO
  passes instead of four (+4.6% on q2tp);
- rope + kv-append + attend + output gate + O-projection in ONE pass
  (+6 tok/s with the attend rewrite).

`CMF_PASSFUSE=0` restores the old shape for A/B.

## Negative results (measured, do not re-try blindly)

| Idea | Result | Why |
|---|---|---|
| Stage x-chunks in workgroup memory | 28.5 → 25.9 | two barriers per chunk cost more than L2 hits |
| 2 groups in flight per lane (unroll) | no change | the scheduler already overlaps iterations |
| Multi-step decode (k frames/submit, on-device argmax) | ~0 at equal length | per-step tail + inter-step drains eat the saved sync; ZML's win does not transfer to this graph shape |
| Fold MoE select into gu/down | 79 → 72.6 | redundant top-k in 7k workgroups outweighs one hop |
| GDN conv inline in the step | −1 tok/s | 128 dv-workgroups amplify conv reads 128-fold |
| Top-k as one lane's serial scan | 96 → 78 | shared-memory reads are ~30 cycles unpipelined in a single thread |
| Subgroup top-k (`moe_select_sg`) | +~2, kept | subgroupMax/Min rounds, two barriers per slot instead of eight |
| Pair the GDN a/b projections in one dispatch | 99.8 → 98.8 | two nv-row dispatches already overlap in-pass; the pair kernel's flat row space serializes them |
| Device argmax at k=1 (4-byte readback vs 1 MB) | 100.0 → 98.5 | two extra dispatches cost more than the PCIe transfer they save |
| vec4 staging in the batched GEMM trio | 18.9 vs 18.6 prefill (noise) | the GEMM already stages through workgroup memory; its loads were never the wall — the ~300 GFLOP/s ceiling is elsewhere (register blocking / K-step shape) |
| Persistent grid for the q4tp matvecs (`CMF_MV_GRID`) | 17.66 → 17.67 ms/token; one WG per SM is 20.81 | launch and teardown are not what a memory-bound matvec pays |
| Halve the batch kernel's live registers by unpacking row PAIRS | FFN 13.40 → 20.39 ms, decode 51 → 42 | it buys occupancy with twice the activation re-reads, and the activation loads are the wall — see below |
| Unpack q4 nibbles into f32 registers before the batch loop | batch FFN 14.64 vs 13.86 packed | fewer ALU ops, but 32 registers of dequantized weight crowd out the occupancy that hides load latency. Keep weights PACKED; unpack one u32 at a time |
| Cache the batched graph's const buffers (norm weights) | verify 52.5-53.6 vs 52.5-53.9 ms | strictly less work and kept for that, but it bought no time — the buffer churn was not on the critical path |
| Lower `par_copy`'s threshold so the per-token logit readback splits | 44.4 → 42.5 tok/s (256 KB), same at 64 KB | `thread::scope` spawns fresh OS threads per call; for ~1 MB once a token that costs more than the copy. The 4 MB default is for the bake's 264 MB planes |

### The q4tp matvec is bus-bound — three proofs, so stop tuning it

Measured on Qwen3.6-27B q4tp (14.26 GB, hidden 5120, FFN 17408) on an
RTX 5090. The token is 17.65 ms of device time and reads as 47% of the
card's 1.79 TB/s, which invites a rewrite. Do not take it.

1. **`CMF_MV_PROBE`** builds the quad-row kernel with its arithmetic
   removed and every load kept, and bits drop the activation loads (+2),
   the code-plane loads (+4) and the cross-lane reduction (+8). Stripped
   to nothing but weight loads the token goes **17.68 → 16.95 ms**. Four
   percent. The unpack is free, the codes are free, the reduction is
   free; the token IS the weight stream.
2. **`CMF_MV_GRID`** makes the kernels persistent. Null at every size.
3. **Two decode processes on one card** (14.26 GB each, 28.5 of 32)
   aggregate **52.8 tok/s against a single process's 48.8** — one stream
   already saturates ~92% of what two can pull.

So 1056 GB/s is the bus for this access pattern, not the kernel's
shortfall. What remains is to read fewer bytes a token, or to amortize
the read over more than one token.

### …but a BATCHED matvec is not

Same kernel, batch > 1, and the budget inverts. At b=1 a dense FFN layer
needs 8.9 ms of weight stream against 6.3 ms of arithmetic, so the
arithmetic hides. At b=3 the stream still needs 6.5 and the arithmetic
needs 13.8 — dead linear in batch (11.83 / 13.83 / 16.03 ms at b=2/3/4)
— because `q4v_dot8` fuses unpack and multiply and every element unpacks
again. Sharing the unpack (`q4tp_matvec4_bku`, `CMF_MV_BK=2`) takes the
batched FFN to 11.15 ms and a k=2 speculative verify from 53.3 to 45.9,
which is what finally makes a drafted token cheaper than a plain one.

What binds it after that is the ACTIVATION loads — 24 vec4 a
group-iteration at b=3 against 4 of weight — not registers, which is
what the row-pair experiment in the table above disproved. Fewer x
loads is the direction: f16 activations halve the count, and they need
their own shader module because `enable f16` would make all of WGSL
require SHADER_F16.

## The failure mode that looks like slowness

An invalid function kills EVERY entry point of its module; `ctx()` goes
None and the whole model silently decodes on the CPU. Symptoms: ~6x slower,
GPU at 0%, and possibly *different text* (CPU tie-breaks differ). Causes we
hit: an inserted kernel orphaning a neighbour's `@compute` attribute; an
`enable subgroups;` directive naga does not accept (subgroup builtins work
WITHOUT it, but the device must carry `Features::SUBGROUP`, so subgroup
kernels live in their own shader module). The 0.5.40 Metal release shipped
this exact class. **If a model got slower after a shader edit, check
`RUST_LOG=warn` for "init failed" before profiling anything.**

## Metal: what the recipes found there (and did not)

The Metal kernels were audited against the same list. Most of it was
already applied — `q4tp_matvec` loads activations as `float4` and weights
as `uint`, runs four rows per simdgroup, expands the scale ladder once,
and reduces with `simd_sum`; `gqa_attend` already splits positions across
simdgroups with per-lane dim slicing (the shape the wgpu attend had to be
rewritten INTO). Nanbeige-3B q4t on an M4: 22.0 tok/s decode, 152 tok/s
prefill.

One candidate looked strong and did not survive measurement: the attend
kernel re-walks every K row a SECOND time to bank Born-importance, whose
only consumer is eviction — which cannot fire until the cache is full.
Gating that pass on a half-full window measured 22.03 vs 21.95 at short
context (noise) and 7.80 vs 8.01 at ctx 3000 — i.e. no gain, possibly a
small loss from the added branch. Reverted. The lesson matches the wgpu
side: on this hardware a second pass over the KV is not what the frame is
made of, and only the profiler gets a vote.

## Metal's actual wall, measured

`CMF_GPU=1 cortiq bench --json` now reports **`metal_submits_per_token`**.
On an M4 decoding Nanbeige-3B q4t: **58 command-buffer round trips per
token**, in a 44 ms frame. Command-buffer completion is the only
system-scope ordering guarantee Metal offers (a "fast flag" variant was
tried years ago and reverted — it corrupted real decodes), so those 58
waits ARE the frame: the kernels themselves are already vectorized and
already run four rows per simdgroup.

So the Metal roadmap is not kernel work — it is the same whole-token
graph the wgpu backend got, worth ~2x there for exactly this reason:
encode the layer stack into ONE command buffer, keep every intermediate
on the device, and wait once per token instead of 58 times. The MoE block
(four dispatches instead of ~1100 encoder pairs) already proved the shape
on this backend.

Until then, the honest statement about Metal is: its kernels are fine,
its submission pattern is not.

## Metal, the batched verify (0.5.82): what measured and what did not

The speculative verify wants a GEMM that streams the weights ONCE for
b ≤ 8 activation rows. Three shapes were built and measured on the M4
(Qwen3.8-27B gate call, 46 MB):

| kernel | b=1 | b=5 | b=8 | note |
|---|---|---|---|---|
| `q4tp_matvec` (one vector) | 0.48 ms (96 GB/s) | — | — | the bandwidth line |
| `q4tp_matvec_bk` (register-blocked, unpack per element) | 0.57 | 1.85 | 2.9 | ALU-bound: 8 FMAs a weight |
| `q4tp_mul_mm` (wide simdgroup GEMM, 32-batch tile) | 2.0 | 2.0 | 2.0 | MAC-bound whatever b |
| **`q4tp_mul_mm_n8`** (64 rows × 8, half unpack, matrix unit) | **0.64** | **0.65** | **0.65** | 68–72 GB/s, flat in b |

n8's memory path alone (MACs removed) is 0.54 — 90 GB/s — and the
matrix-unit MACs add 0.10 that the two barriers an iteration do not
overlap. Five variants tried against it, none faster: 4 lanes a row
(64 B contiguous), NK 32/64/128, 32-/64-/128-row tiles, double-buffered
tiles (one barrier — slower, 18 KB of threadgroup memory), a
device-resident half copy of x for the B tiles (slower, 0.76). Take it
as 0.64 and buy the round elsewhere.

Where the round's time went after that (27B, k=7): verify 242 ms =
199 GEMMs + 14 GDN recurrence + 8 attend + ~20 small kernels; draft 33
(seven MTP steps at bandwidth once the head is a 65536-row shortlist);
commit 18. Fusions that paid: residual add folded into the next norm,
silu+row-scale one pass, a/b projections one dispatch, one attend
dispatch for the b rows (the per-row flash split only for a single
row), the block input folded into the draft submit, zero-copy GDN
states (the memcpy of 300 MB/token was 10% of the plain token).

Two Metal bugs the same campaign found, both invisible in a
"coherent-looking" output: an unbound kernel constant (`q4tp_mul_mm`'s
`wboost`, slot 6 — every batched q4tp prefill was noise) and a head
index derived from `simdgroups_per_threadgroup` under `dispatch_threads`
(the partial last threadgroup made the K heads of every model with
(nh+nkv) % 8 ≠ 0 skip norm+RoPE). Both were caught only by diffing
against the strict CPU (`CMF_SDOT=0 CMF_GPU=0`) — `CMF_LOGIT_DUMP` and
`CMF_ATTN_ORACLE=1` exist for exactly that. Rule: a kernel that indexes
by simdgroup must derive it from `thread_position_in_grid`, or be
launched with `dispatch_thread_groups`.

## Cross-format notes

- `q2tp` shares q4tp's params/codes planes byte-for-byte; only the weight
  plane halves. Anything that speeds the q4tp ladder path speeds q2tp.
- `q2tp`'s rung 0 is an exact zero because the ±0.5/±1.5 grid cannot spell
  it — the q1t ternary layout solves the same problem with an explicit
  zero code; q1 (binary) cannot represent zero at all, which is one reason
  PTQ-to-q1 destroys normal checkpoints.
- The batched GEMM (`q4tp_mul_mm`, shared by dense prefill and the Lumina
  DiT) did NOT respond to the load-shape recipe. Its next candidate is the
  K-step and register-block shape, not the loads.
- Expert-restriction masks (`moe-mask`, patent 2) are a VRAM lever, not a
  decode-speed lever: routing entropy on the models we measured is ~0.94,
  decode always runs top-k experts regardless, and the frame cost lives in
  kernel shape, not expert count.
