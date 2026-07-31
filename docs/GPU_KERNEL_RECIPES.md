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

Ported so far: `q4tp_matvec4`, `moe_gate_up_q2tp`, `moe_down_q4tp`.
**Still scalar (same win waiting): `q4t_matvec`, `q4b_matvec`,
`q1t_matvec`, `q8_matvec`, `moe_gate_up` (q4t), `moe_gate_up_q4tp`, the
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

## Cross-format notes

- `q2tp` shares q4tp's params/codes planes byte-for-byte; only the weight
  plane halves. Anything that speeds the q4tp ladder path speeds q2tp.
- `q2tp`'s rung 0 is an exact zero because the ±0.5/±1.5 grid cannot spell
  it — the q1t ternary layout solves the same problem with an explicit
  zero code; q1 (binary) cannot represent zero at all, which is one reason
  PTQ-to-q1 destroys normal checkpoints.
- Expert-restriction masks (`moe-mask`, patent 2) are a VRAM lever, not a
  decode-speed lever: routing entropy on the models we measured is ~0.94,
  decode always runs top-k experts regardless, and the frame cost lives in
  kernel shape, not expert count.
