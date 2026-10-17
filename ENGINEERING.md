# Engineering notes

## q1g — 1-bit weights with one scale per 128-group (dtype 20)

### What changed

Bonsai-class 1-bit models have a native 128-weight group with one bf16
scale. The CMF `q1` tile holds one f16 scale per 32 weights, so a q1 file of
such a model stores every group scale four times: 99.997 % of the
13,436,752 groups in Bonsai-1.7B are byte-identical 4-runs. `q1g` keeps the
same information at 1.125 bits/weight instead of 1.5.

- **Format** (`cortiq-core`, `TensorDtype::Q1G = 20`, name `q1g`; id 19 is
  left to the LittleBit research codec):
  - Layout: `[sign bits: rows·cols/8][f16 scales: rows·cols/128]`, 2-D,
    `cols % 128 == 0`.
  - These are planes, not tiles, so each group's 16 sign bytes are 16-byte
    aligned.
  - A row's sign plane is byte-identical to its q1 tiles' bit fields.
  - Spec: `docs/CMF_V2_SPEC.md` §3.1/§3.2. Python reader:
    `python/cmf_reader.py`.
- **Repack** — `cortiq requant <in> --quant q1g --output <out>`
  (`--in-place` also works):
  - Lossless by default. Any 128-group whose four tile scales are not
    byte-identical refuses the whole run, and nothing is written. The error
    gives the count and the largest relative spread.
  - `--allow-tol` accepts groups that agree within `--tol` (relative,
    default 2⁻⁸). Each such group stores the f16 nearest the mean of its four
    scales, the L2-optimal level.
  - Tensors with a group beyond `--tol`, or with `cols % 128 != 0`, stay q1
    byte for byte.
  - `requant --quant q1` writes q1 back exactly, for runtimes that predate
    dtype 20.
- **Kernels**: every q1g kernel reproduces the q1 kernel's float sums. Per
  32-tile it computes `acc += d·s`, reusing the group's scale. A q1g repack
  of a uniform-scale q1 tensor therefore gives the same f32 outputs bit for
  bit on every path. Tests assert this on the device and on x86-64 and
  aarch64 hosts.
  - **CPU** (`qtensor.rs`): `BinW` (q1 tiles | q1g planes) drives the binary
    row drivers once for both layouts. They cover matvec, matvec2, matmat
    with the blocked 1×4, the multi-projection job (q1 and q1g may mix), the
    fused gate/up+SiLU and embedding rows.
    - x86 AVX2/VNNI: each tile is selected as in q1. A 128-group's four tiles
      then reduce in one `hadd` tree (`q1g_hsum4`): 4 ops instead of 4×7. The
      group's scale converts once, not per tile. The float epilogue is q1's
      per-tile sequence, so the result is bit-identical and about 2× faster
      (see below).
    - aarch64 SDOT: one aligned 16-byte load per group and one FCVTL per
      group. It mirrors q1's last-row scalar tail.
    - scalar and the exact `CMF_SDOT=0` walk.
  - **wgpu** (`gpu_wgpu.rs`):
    - `q1g_matvec` keeps `q1_matvec`'s lane walk, staging and reduction; the
      scale-plane offset rides in the params. `q1g_mul_mm` is the tiled GEMM.
    - Wired into the per-op entries, batch layout `Q1g` and token-graph kind
      10: resolve, emat, prep, and the shortlisted head, which locates the
      scale plane from the tensor's full row count. Also wired into the
      batched/chunk graph (gemmable, ematb, emat1).
    - `q1g_matvec` fetches its sign words and group scale ahead of the
      staging barrier, which removed a down_proj penalty against q1.
    - `gpu_wgpu/q1g_mv.rs` adds a barrier-free subgroup twin with the same
      bits. It is opt-in (`CMF_Q1G_SG=1`, rows ≥ 4096 columns): it lost on
      lm_head on the 5090 and on every shape on the RTX 2000 Ada (numbers
      below).
  - **Metal** (`gpu_metal.rs`):
    - `q1g_matvec` and `q1g_matvec_h` mirror `q1_matvec(_h)`.
    - They are compiled as their own library (`MSL_Q1G`). If it fails to
      compile, q1g loses its device path and every other model keeps the
      backend.
    - Wired through `ProjKind::Q1g` / `proj_abs` / `encode_proj` (block graph
      and dense fast path), plus the batch layout and the per-op entry.
    - **Kernels checked on an Apple M4, outside cargo**:
      `tools/metal_q1g_parity.swift` compiles `MSL` and `MSL_Q1G` straight
      from `gpu_metal.rs` (MSL 3.0, default options) and runs `q1_matvec(_h)`
      against `q1g_matvec(_h)` with the engine's dispatch shape. The first
      version was NOT bit-identical: written as `acc += s * sum`, fast math
      folds `acc + s·A + s·B` into `acc + s·(A + B)` once one scale serves
      both tiles, and 65 % of rows (f32 kernel) and up to 37 % (half kernel,
      6144-column rows) differed from q1 by float rounding. The scale step is
      now an explicit `fma`, which is what q1's two-scale form compiles to:
      0 differing rows in both kernels on 11 shapes × 2 seeds, up to the
      151669×2048 head.
    - The Rust host path (file arena, `q1g_ready`, per-op entry, batch) has
      the test `gpu_metal::tests::gpu_q1g_matvec_equals_q1`; it is
      type-checked (`cargo check --tests --target aarch64-apple-darwin`) but
      has not been run on a Mac yet.

### How to use

```
cortiq requant bonsai-1.7b-q1.cmf --quant q1g --output bonsai-1.7b-q1g.cmf             # strict: refuses Bonsai (393 groups differ)
cortiq requant bonsai-1.7b-q1.cmf --quant q1g --allow-tol --output bonsai-1.7b-q1g.cmf # those 393 groups take their mean scale
cortiq requant bonsai-1.7b-q1g.cmf --quant q1 --output bonsai-1.7b-q1-again.cmf        # exact inverse
```

### Tests

- **core**: `quant::q1g_tests`.
  - Size and validation.
  - A bit-exact q1 → q1g → q1 round trip.
  - Accounting for differing groups, and zero-vs-nonzero scales.
- **cli**: `requant::tests::q1g_repack_is_lossless_and_refuses_what_it_cannot_keep`
  runs end to end:
  - strict refusal and `--allow-tol`;
  - in-place output equal to `--output`;
  - kept tensors byte-identical;
  - the inverse.

  `skill::tests::every_skill_compatible_codec_round_trips_to_an_encoder`
  checks that a q1g base encodes grafts on the q1 grid.
- **engine CPU**: `qtensor::tests::q1g_cpu_kernels_equal_q1_bit_for_bit` and
  `q1g_mapped_tensor_paths_equal_q1`.
  - The first covers every row kernel and driver, the AVX2/VNNI/1×4 kernels
    directly, gpr = 4 and the last row.
  - The second covers matvec, matvec_many (all-q1g and mixed), fused gate/up,
    matmat, matvec2 and embedding rows.
  - Both pass on x86-64 (5090 pod) and on aarch64 under
    `qemu-aarch64 -cpu max`, which executes the SDOT kernels.
- **engine wgpu**: `gpu_wgpu::tests::wgpu_q1g_kernels_equal_q1_bit_for_bit`.
  - Bit-exact against q1 on the device: staged and subgroup matvec, GEMM and
    the shortlisted head.
  - Also checked against the CPU dequant.
  - Shapes include 3×384: an odd group count, so the payload is not a word
    multiple and the last scale word is half padding.
  - It fails if the subgroup module is rejected on a device that has 32-wide
    subgroups.
- **Metal**: `tools/metal_q1g_parity.swift` (no cargo; kernels, run on an
  M4) and `gpu_metal::tests::gpu_q1g_matvec_equals_q1` (host path,
  type-checked only).
- **speed** (ignored): `gpu_wgpu::tests::q1g_matvec_speed` and
  `qtensor::tests::q1g_cpu_matvec_speed`.

### Measured

Model: Bonsai-1.7B (`infosave/Bonsai-1.7Bcmf/bonsai-1.7b-q1.cmf`, sha256
`977c65ad…a01c`).

Both pods were shared during every measurement:
- 5090 pod: another job held 17 GB and 40–90 % of the GPU, and CPU load was
  15–29 on a 13.6-core quota.
- pod B (RTX 2000 Ada): a co-tenant held the GPU at 100 % and 8–15 GB of
  VRAM, on a 5.1-CPU quota.

Speed numbers therefore come from interleaved runs with medians.

#### Repack

`requant --quant q1g --allow-tol` (binary `1d610580`, 0.6 s):

| | q1 | q1g |
|---|---|---|
| file | 334,185,278 B | 253,782,974 B (−24.1 %) |
| q1 payload | 322.5 MB | 241.9 MB (−25.0 %) |
| wgpu resident upload | 322.5 MB | 241.9 MB |

- All 197 q1 tensors converted. Of 13,436,752 groups, 13,436,359 are exact
  and 393 are within tolerance (largest relative spread 1.848e-3, f16
  rounding). The strict run refuses with exactly that count.
- The output sha256 is `724a34dc…8bac` on both pods (deterministic).

#### Perplexity

Protocol: `/root/bq`, 16 fixed WikiText-2 windows × 511 tokens, exact window
NLL from `CMF_PPL_PROGRESS=1`. Totals are summed over the 16 windows.

| path | q1 | q1g | q1g − q1 | q1u (q1 written back from q1g) |
|---|---|---|---|---|
| GPU wgpu (`CMF_GPU=wgpu CMF_GPU_PROBE=0 CMF_COOP=0 CMF_SDOT=0`), RTX 5090 and RTX 2000 Ada, identical | 27351.593365 nats, PPL 28.3705 | 27351.592607, PPL 28.3705 | −0.000758 nats = −9.3e-8 /token; per window [−0.000592, +0.000567] | == q1g on 16/16 windows |
| CPU strict (`CMF_GPU=0 CMF_SDOT=0 CMF_NLL_SERIAL=1`), 5090 pod, 6 threads | 27351.594075, PPL 28.3705 | 27351.593542, PPL 28.3705 | −0.000533 nats = −6.5e-8 /token; per window [−0.000362, +0.000510] | == q1g on 16/16 windows |
| CPU strict, pod B, windows 00 and 10, 4 threads | 1515.638184, 2111.166327 | 1515.638154, 2111.166160 | as on the 5090 | — |

- **q1 regression check.** The GPU per-window NLL of q1 on this branch
  equals release 0.8.17's recorded values on all 16 windows. CPU-strict
  window 10 = 2111.166327, the release value from the study's independent
  check.
- **Where the q1g − q1 difference comes from.** q1u is q1 tiles written back
  from q1g, so it carries the same 393 adjusted scales. q1g == q1u
  everywhere, so the kernels are exact, and the whole q1g − q1 difference is
  those 393 tolerance groups. The difference is 6.5–9.3e-8 nats per token;
  PPL is equal to 4 decimals.
- **Binaries.** The `fa381ba4`, `1d610580` and final `917b0857` binaries give
  identical GPU NLL for both arms.
- **Independent decode.** `python/cmf_reader.py` only, no cortiq code
  (run-dir `tools/check_reader.py`): 197 tensors and 13,436,752 groups.
  Weights moved in exactly 393 groups (max 1.848e-3 relative). Every sign
  matches, and the hashes verify.

#### Greedy generation

64 tokens, the study's two prompts:
- **GPU:** q1 and q1g give identical text.
- **CPU A8W8:** q1g == q1u. q1 diverges from q1g at token 12 of prompt 1
  ("Augustus" vs "Julius Caesar"), and late in prompt 2. Token 12 is a near
  tie where q1 on CPU and q1 on GPU already disagree; the 393 adjusted
  scales tip it.

#### Kernel speed

In-process, arms interleaved, medians. Ratios are q1g time / q1 time.

**CPU** (x86 A8W8, `q1g_hsum4`):

| | gate/up 6144×2048 | down 2048×6144 | lm_head 151669×2048 |
|---|---|---|---|
| 5090 pod, 8 threads (2 runs) | 0.48 / 0.48 | 0.51 / 0.52 | 0.48 / 0.51 |
| 5090 pod, 1 thread | 0.46 | 0.46 | 0.45 |
| pod B, 4 / 1 threads | — | 0.61 / 0.59 | 0.57 / 0.51 |

Before `q1g_hsum4` the ratio was 0.74 (8 threads) and 0.73 (1 thread). That
gain came from fewer bytes and one f16 decode per group.

**wgpu staged**, final, weights fetched ahead of the barrier. One value per
run:

| | q/o_proj 2048² | gate/up 6144×2048 | down 2048×6144 | lm_head 151669×2048 |
|---|---|---|---|---|
| RTX 5090, 4 runs | 0.82 / 0.94 / 0.99 / 0.98 | 1.16 / 1.00 / 0.99 / 1.02 | 1.00 / 0.98 / 0.93 / 1.00 | 0.98 / 0.97 / 0.97 / 0.96 |
| RTX 2000 Ada, 3 runs | 1.00 / 0.90 / 1.00 | 0.99 / 1.03 / 1.00 | 0.89 / 1.00 / 1.23 | 1.00 / 1.05 / 1.02 |

- **Before the hoist**, down_proj was about 1.16 on the 5090 and 1.19 on
  the 2000 Ada.
- **The subgroup twin** (opt-in) lost on lm_head by 2.0–2.4× on the 5090
  and on every shape on the 2000 Ada. Its down_proj gain on the 5090 was
  2× in a busy window and 1.1× in a quieter one.

#### End-to-end decode

`cortiq bench --json --core --ignore-eos`, `CMF_MAX_SEQ=1024`, medians of 6
interleaved rounds, final binary `917b0857`:

| | q1 | q1g | prefill q1 → q1g |
|---|---|---|---|
| RTX 5090, wgpu whole-token graph, 128 tok | 84.8 tok/s | 89.2 | 47.9 → 61.3 |
| RTX 2000 Ada, wgpu whole-token graph, 128 tok | 40.1 | 40.8 | 42.6 → 55.5 |
| 5090 pod CPU, 8 threads, 64 tok (load avg 30–45) | 10.5 | 18.8 (1.8×) | 19.3 → 27.5 |

- **Earlier round** (binary `1d610580`, before the hoist, other co-tenant
  load):

  | | q1 | q1g staged |
  |---|---|---|
  | 5090 wgpu | 110.5 | 113.8 |
  | 2000 Ada wgpu | 25.0 | 25.9 |
  | 5090 CPU, 8 threads | 20.4 | 41.8 |
  | pod B CPU, 4 threads | 17.5 | 28.0 |

- **Whole-token graph.** Every token ran on it (63 of 63, no misses) for
  both formats.
- **Why the absolute numbers swing between rounds:** co-tenant load. Only
  the interleaved ratios carry over.

#### Independent verification (second build, 2026-10-17)

Rebuilt from this branch and from master `5d6f1830` on the 5090 pod
(binaries `245dbbca` and `18e0fc9a`), tests re-run, Bonsai-1.7B re-measured:
- **Tests:** core `q1g_*` (5), cli `q1g_repack…` + all `requant` (7) +
  the skill codec test, engine `q1` filter (11, wgpu on the device),
  `x86_gemm::q1_blocked_vs_per_row`; `cargo check --tests --target
  aarch64-apple-darwin` clean.
- **Repack** reproduces bit for bit: strict refuses with 393 of 13,436,752
  groups (max rel 1.848e-3) and writes nothing; `--allow-tol` gives sha256
  `724a34dc…8bac`, `--quant q1` gives `f5abb584…2486` (both equal to the
  first build's files); `--in-place` verifies.
- **Window NLL** (`exact_nll`, windows 00 / 10):

  | path | master q1 | branch q1 | q1g | q1u | q1g in place |
  |---|---|---|---|---|---|
  | GPU wgpu | 1515.638304 / 2111.166368 | same | 1515.638440 / 2111.165933 | same as q1g | same as q1g |
  | CPU (ppl, float activations) | 1515.638184 / — | same | 1515.638154 / — | same as q1g | — |

- **CPU A8W8 decode** (ppl never reaches the int8 kernels): 32 greedy steps
  with every step's logits dumped (`CMF_LOGIT_DUMP_ALL`) — master q1 vs
  branch q1 and q1g vs q1u byte-identical on all 32 steps, on the CPU run
  and on the `CMF_GPU=wgpu` run.
- **CPU kernel A/B** (`q1g_cpu_matvec_speed`, quieter window): q1g/q1 time
  0.472 / 0.483 / 0.463 (gate/up, down, lm_head) at 6 threads, 0.458 /
  0.444 / 0.240 at 1 thread.
- A 3-round GPU bench pair was pure noise on the shared card (37–139 tok/s
  within one arm); every token ran on the whole-token graph (127/127) and
  the upload was 241.9 vs 322.5 MB.

### Open issues

- **Metal:** the kernels are verified on an M4 by the Swift harness (see
  above), the host path is not. First thing on a Mac:
  `cargo test -p cortiq-engine --release --lib gpu_q1g_matvec_equals_q1`,
  then a Bonsai q1 vs q1g perplexity pair. Re-run
  `tools/metal_q1g_parity.swift` after any edit to either MSL source.
- **wgpu on macOS (`CMF_GPU=wgpu`) is unchecked.** wgpu-hal compiles Naga's
  MSL with default (fast-math) options, the setting that broke q1g/q1 parity
  in the native kernels; the WGSL `q1g_matvec` may differ from `q1_matvec`
  there by rounding. Vulkan (5090, RTX 2000 Ada) is bit-exact.
- **q1 x86 kernels:** applying `q1g_hsum4` to them would give q1 the same
  CPU speedup, bit for bit. It was not applied, to keep q1 untouched in
  this change.
- **wgpu q1g matvec is still bound by the arithmetic, not the bus.** The
  staged kernel runs at q1's speed now (the hoist removed the down_proj
  penalty). It does not run faster, because the q1 kernel is far from
  memory-bound: the 5090 moves 40–770 GB/s through it. Getting the 25 %
  byte saving to show on the GPU would need a different kernel; one
  candidate is a T-MAC-style nibble LUT for the sign dots. A subgroup
  twin was tried and is opt-in only (numbers above).
- **The speed figures are from shared cards.** Re-measure on a quiet GPU
  before tuning further.
- **Dtype id:** 20 was chosen because 19 is held by the uncommitted
  LittleBit branch. Parallel branches that add dtypes must not collide with
  it.
