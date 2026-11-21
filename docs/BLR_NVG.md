# Binary low-rank weights (`blr`) and the NVG encoder

`blr` is a CMF tensor codec (dtype **19**) that stores a linear layer as a sum of
binary low-rank paths. **NVG** is the encoder that decides what goes into it: the
sign topology Ĉ from the activation-whitened weight, the scale metric Ť solved in
closed form against the layer's real output error, then exact discrete sign
refinement. No gradient step.

The factorization layout `W ≈ Σ_p diag(h_p)·S_U,p·diag(ℓ_p)·S_V,pᵀ·diag(g_p)` is the
one of LittleBit (Lee, Kim, You, Kim — NeurIPS 2025, arXiv 2506.13771). We use it as
the storage container for equal-budget comparisons; LittleBit's own data-free
initializer is kept as the `dual-svid` baseline. Neither the container nor the
baseline is the NVG method.

| part | where |
|---|---|
| codec: header, validation, packing, reference dequant | `crates/cortiq-core/src/blr.rs`, `types.rs` (`TensorDtype::BinaryLowRank`), `quant.rs` |
| CPU kernels (table matvec, GEMM matmat) | `crates/cortiq-engine/src/blr.rs`, `QTensor::Blr` in `qtensor.rs` |
| wgpu kernels | `crates/cortiq-engine/src/gpu_wgpu/blr.rs`, `gpu::blr_matmat` |
| encoders, `convert --quant blr` | `crates/cortiq-cli/src/nvg.rs`, `nvg_linalg.rs` (CPU f64 reference) |
| NVG encoder on the device | `crates/cortiq-cli/src/nvg_gpu.rs` over `crates/cortiq-engine/src/nvg_dev.rs` → `gpu_wgpu/nvg_dev.rs`, or `gpu_cuda/nvg_dev.rs` under `CMF_GPU=cuda` (§8) |
| NVG Ť, the distillation trainer | `crates/cortiq-cli/src/nvgtrain.rs` over `crates/cortiq-engine/src/nvg_train.rs` (wgpu) or `gpu_cuda/nvgt.rs` (CUDA, §8) |
| per-tensor / per-layer bench | `crates/cortiq-cli/src/nvgbench.rs` (`cortiq nvg-bench`) |

## 1. Bits and rank

Per tensor `W ∈ R^{d_out×d_in}`, `P` paths of rank `r`, scales at 16 bits:

```
bits(r) = P · [ r(d_out + d_in) + 16(d_out + d_in) + 16 r ]        b = bits / (d_out·d_in)
P = 2:  r* = [b·d_out·d_in − 32(d_out+d_in)] / [2(d_out+d_in) + 32]
P = 1:  r* = [b·d_out·d_in − 16(d_out+d_in)] / [(d_out+d_in) + 16]
r = max(8, floor(r*·mult / 8)·8),  r ≤ floor(min(d_out, d_in)/8)·8
```

`mult` is the K/V factor (`--blr-kv-factor`, default 1) for `k_proj`/`v_proj` of a GQA
model. The 16-byte payload header is not counted (it is 128/(d_out·d_in) bits per
weight). The converter reports two strict numbers:

* **blr body** — Σ bits / Σ weights over the tensors stored as `blr` (the formula above);
* **all layer projections** — every `model.layers.*.*_proj.weight` whatever its codec:
  `blr` tensors at the formula, the rest (e.g. GDN `in_proj_a/b`, kept f16) at
  8 × payload bytes.

Embeddings and an untied `lm_head` stay `q8_2f` and are reported separately (file size).

## 2. Storage (`blr`, id 19)

The directory entry keeps the logical shape `[d_out, d_in]`; the payload is
self-describing and variable-length (`expected_nbytes` → `None`, `validate_payload`
computes the exact length from the header). Little-endian:

```
header (16 B)  u8 magic 0x4C | u8 version 1 | u8 P ∈ {1,2} | u8 0 | u32 r | u32 d_out | u32 d_in
per path p     h f16[d_out] | g f16[d_in] | ℓ f16[r]
               U: d_out rows × r/8 bytes     bit (i,k) = byte[i·r/8 + k/8] >> (k%8) & 1
               V: r rows × d_in/8 bytes      bit (k,j) = byte[k·d_in/8 + j/8] >> (j%8) & 1
```

Bit 1 → +1, bit 0 → −1, sign(0) → 1. `V` is stored transposed (one row per latent) so
both kernel stages stream contiguous rows. Path size `2(d_out+d_in+r) + r(d_out+d_in)/8`.
Validation checks magic, version, `P`, reserved byte, `r % 8 == 0`, `8 ≤ r ≤ min(d_out,
d_in)`, dims against the shape, `d_in % 8 == 0`, and the exact length, before any slice.
The bytes are identical to the research branch's `lb` files, which stay readable.

Reference dequant: `W[i,:] = Σ_p h_p[i] Σ_k ℓ_p[k] U_p(i,k) (V_p(k,:) ⊙ g_p)`.

## 3. Kernels

Per token and path: `z = x ⊙ g_p`; stage V `t_p[k] = ℓ_p[k]·Σ_j V_p(k,j) z[j]`
(P·r binary dots of length d_in); stage U `y[i] = Σ_p h_p[i]·Σ_k U_p(i,k) t_p[k]`.
`W` is never materialised; signs are read from the mmap.

* **CPU decode** — each binary dot goes through a 16-entry signed partial-sum table per
  group of four inputs (`T[16q+c] = Σ_t ±z[4q+t]`), so a row costs n/4 lookups; four
  independent accumulators; rows split over the pool.
* **CPU batch** (`b ≥ 4`) — signs unpacked once to ±1 f32, two `sgemm`s per path
  (`Z·Vᵀ`, `T·Uᵀ`) split over the pool.
* **wgpu per-op** (`gpu::blr_matmat`, `b = 1` and `b > 1`) — `blr_v`: one workgroup per
  latent row and block of 4 tokens, signs 16 at a time, f16 scales via
  `unpack2x16float`; `blr_u`: one thread per output row, the token's latents staged in
  workgroup memory. One upload, two dispatches and one readback per call. Contract:
  `d_in % 32 == 0`, `d_out % 4 == 0`, `P·r ≤ 2048`; outside it the CPU kernel runs.
  Metal has no `blr` kernel yet (CPU fallback).
* **wgpu whole-token graph** (decode, `b = 1`; `gpu_wgpu/blr.rs`: `blr_gv`, `blr_gu`,
  `blr_gu_act`) — the payload resident in the weight cache, the latents in a graph
  scratch, two dispatches per group of up to three tensors of one input (q|k|v,
  qkv|z; gate|up compute act(gate)·up in stage U and write the FFN activation).
  Stage V: a set of 32 lanes (128 from 2048 columns) owns 8 latent rows, lanes over
  32-bit sign words, four words a lane in flight, `z = x ⊙ g` formed once per word for
  the 8 rows (g resident as a word-transposed f32 copy), subgroup sums. Stage U reads
  U rows where they lie: a row starts `σ = i·r mod 32 ∈ {0, 8, 16, 24}` bits into a
  word, so the latents are stored four times shifted by σ (the neighbouring rows' bits
  meet zeros) and word-transposed (lanes walking words read consecutive vec4s); a
  subgroup owns rows k, k+4, k+8, k+12 (one shift copy) and walks all paths as one
  word sequence. Without 32-wide subgroups both stages sum through workgroup memory.
  Contract: `d_in % 32 == 0`, `d_out % 4 == 0`, no `P·r` limit. `CMF_BLR_GRAPH=0`
  restores the per-op path; `CMF_BLR_LV=32|128|256|512` pins the stage-V lane set.
* **Engine wiring** — `QTensor::Blr`: `matvec`, `matvec2`, `matmat` (GPU first when the
  device is up, `CMF_BLR_GPU=0` keeps it on the CPU); multi-tensor jobs and fused
  gate/up refuse `blr` and fall back to per-tensor calls; rows/columns reconstructed on
  the fly for the rare paths. `graph_weight` and the other device-graph accessors
  answer `None`; only the wgpu token-graph builder asks `graph_weight_or_blr`, which
  hands a `blr` tensor over as kind `GRAPH_KIND_BLR` (11). The batch graph still
  declines `blr` (prefill walks positions through the token graph).
  `CMF_BLR_DENSE=1` dequantizes to f32 at load (A/B parity).

## 4. Encoders

`dual-svid` and the reference `nvg` run per tensor in f64 on the CPU (`nvg_linalg.rs`:
parallel GEMM, blocked Cholesky, triangular inverse, symmetric eigensolver, polar
factor); `nvg` also runs on the wgpu device in f32 (§4.1, `--blr-device`). Scales are
rebalanced before f16 rounding (`h/rms h`, `g/rms g`, magnitude into `ℓ`); a tensor
whose scales do not fit f16 stays `q8_2f`.

**`dual-svid`** (data-free, the LittleBit baseline): rank-r SVD `W ≈ U′V′ᵀ` with
`U′ = U_rΣ^{1/2}`, `V′ = V_rΣ^{1/2}`; signs of the latents; rank-1 power-iteration fits of
`|U′|`, `|V′|` give `h, ℓ_u` and `g, ℓ_v`, `ℓ = ℓ_u ⊙ ℓ_v`; P = 2 fits the same to
`W − Ŵ_pri`. Without training this is far from usable at ≤ 1 bpw — it is the starting
point LittleBit's QAT needs.

**`nvg`** (data-aware). Objective `E(Ŵ) = tr((W − Ŵ) H (W − Ŵ)ᵀ)`, `H = XXᵀ/T + ρ·tr/n·I`,
from calibration inputs `X` of that layer:

1. *Ĉ, topology*: rank-r latents of `W·L` (`LLᵀ = H`), i.e. the best rank-r
   approximation in the H-norm.
2. *Gauge*: an H-weighted ITQ rotation (50 iterations) of the row-normalised `[U′; V′]`
   toward the hypercube (output rows weighted by `‖U′_i‖`, input rows by
   `√H_jj·‖V′_j‖`); signs taken in that frame.
3. *Ť, metric*: exact block least squares for `ℓ` (both paths jointly), `h` (per row,
   P×P) and `g` (per path, n×n), alternated; each solve is a minimum so `E` never
   increases. Path 2 is built the same way on the H-residual of path 1, then joint sweeps.
4. *Discrete knots*: exact single-bit flip deltas per output row (greedy, monotone), and
   a column-sequential sign search on V with Schur-complement error feedback (act order,
   128-column blocks); each step is kept only if `E` drops, with metric refits in between.
5. *Paths*: P = 2 at rank r and P = 1 at the same bits are both built; the one with the
   lower held-out error (SEL slice) is stored (`--blr-paths auto`).

Calibration is sequential: layer `l` is fitted on activations the engine produces with
layers `< l` already `blr` (the not-yet-encoded layers run as `q8_2f`). Inputs are shared
per Gram: q/k/v (attention), gate/up (FFN), in_proj_qkv/in_proj_z (GDN), and o, down,
out_proj each on their own. The capture uses the engine's GPTQ hook: with
`--blr-device cpu` in the CPU batched prefill of the scored forward (the converter sets
`CMF_GPU=0`); with `--blr-device gpu` as quantize-gptq does — per-op device GEMMs
(`CMF_GPU_WGPU_GRAPH=0`), the already-encoded `blr` layers through the wgpu `blr`
kernels, the Grams accumulated on the device, and only layers `0..=l` run
(`calib_forward`) from a work file cut after layer `l` (bit-identical Grams; the same
positions as the scored forward, i.e. a window's last token is not fed).

### 4.1 The NVG encoder on the device (`--blr-device gpu`, f32)

Stage for stage the CPU encoder; what changes is where and how each stage runs.

* *Per input*: `H` (damped), the held-out Gram, act order and `Uc` (the upper Cholesky
  factor of the permuted `H⁻¹`: equilibrated f32 Cholesky + blocked inverse), and the
  lower factor `L` when a tensor reading the input has `m > n`.
* *f32 GEMM*: 128×128×16 tiles, register prefetch, 8×8 outputs per thread, any operand
  transposed or a sub-matrix view, α/β, a Hadamard epilogue, lower-tiles-only mode;
  19–30 TFLOP/s on the RTX PRO 6000. Never the matrix units (their operands are f16/tf32).
* *SPD solves* (ℓ jointly, g per path): Jacobi equilibration and the ridge on the device,
  blocked Cholesky (128-wide diagonal blocks factored and inverted in workgroup memory,
  panel and lower-tile trailing update through the GEMM; n = 17408 in ~0.26 s), blocked
  triangular solves; one readback per factorisation (its pivot flag) — the ridge grows
  ×100 on a failed pivot as on the host. `CMF_NVG_HOST_SOLVE=1` solves ℓ on the host in f64.
* *Latents*: the min(m, n)-sized Gram on the device; its r largest eigenpairs by
  Householder tridiagonalisation on the device, bisection + inverse iteration of the
  tridiagonal on the host in f64, and a compact-WY back-transform on the device
  (`CMF_NVG_HOST_EIG=1`: the host `sym_eig`). Eigenvector signs may differ from the host
  solver's; ITQ and the model are invariant to them.
* *ITQ*: device GEMMs and a Newton–Schulz polar factor (Frobenius scaling on the device,
  ~16 steps on real cross terms) instead of the host's r×r eigendecomposition.
* *Ť*: the ℓ/h/g block least squares from device GEMMs; `H·gs_p` and `WH·gs_p` cached per
  path and recomputed only when that path's g or V-signs change; h, ℓ, g stay on the
  device (per-row 2×2 h solve as a kernel).
* *Output-side knots*: blocks of 32 bits — one thread per row decides the block's bits in
  order with the block's qs entries in registers (the 32×32 corner of `Q` in workgroup
  memory) and records the flips; the rest of every row's qs moves with one GEMM
  `qs −= F·Q[block, :]`. The decisions are the host's sequential pass.
* *Input-side knots*: the column-sequential search carried in the latent space: `Mᵀw_j`
  for every column is one GEMM up front, and a column's error reaches the others as
  `Mᵀe_j = (Mᵀw_j − g_j K s_j)/Uc[j,j]` (no m-length vector in the sequential part). The
  flips of a 128-column block run on the host thread that encodes the tensor
  (`gv_block_host`); the trailing feedback `BV[b1:] −= Uc[b0:b1, b1:]ᵀ·E` is a device GEMM.
  The search is sequential twice over — a column's target carries the errors of the columns
  before it, and a flip changes every candidate of the next one — so a device runs it as one
  block of threads on one multiprocessor, about a microsecond a flip whatever the rank, with
  the rest of the card idle. A flip is one pass over r floats (the update `ks −= 2s·K[row]`,
  every candidate's gain `(−4g²s)·(ks − kd·s) + (4gs)·bv`, the minimum of each 16
  candidates) in vector registers, 0.27 µs at r = 776 and 0.54 µs at r = 2152 with the
  transfers, and every tensor and candidate in flight has a core of its own. The block's rows of `BV`, `S`, `K·S` and its square of `Uc`
  come down in two reads, its signs and error rows go back in two writes. The gain is written
  as the device kernel writes it — the difference first, the last product and sum fused: the
  two terms cancel near the stop test, and a gain summed from them loses the digits the test
  reads. Elementwise IEEE operations and f64 sums: the result does not depend on the vector
  width (x86-64 needs AVX2 and FMA; without them the device's kernel runs).
  `CMF_NVG_GV=device` runs the flips in the device's kernel (one workgroup or block, lanes
  unrolled in registers, argmin and dot products by subgroup operations): the same search
  with its sums in another order, so two files differ in the fourth digit of a tensor's
  error. After the 0.8B's reference training the held-out KL of four conversions is 1.4622
  and 1.4546 with the device's kernel (blocks of 1024 and 256 threads), 1.4669 and 1.4595
  with the host's flips (the gain summed plainly, and fused): one spread.
* *Errors* `tr(E X Eᵀ)` (X = H, the raw fit Gram, the held-out Gram) from the factor-sized
  expansion `tr(WXWᵀ) − 2Σ⟨hsl_p, WX·gs_p⟩ + Σ⟨hsl_pᵀhsl_q, gs_pᵀX gs_q⟩`, read back at once.
* Flip tolerances are 1e-7 relative in f32 and the column scale converges to 1e-8
  between flip rounds (1e-12 / 1e-9 in f64 on the host; `CMF_NVG_FLIP_TOL`,
  `CMF_NVG_GJ_TOL`). 1e-6 / 1e-6 stopped the search early enough to cost ~2 % ppl end to
  end on the 0.8B (686.5 vs 672.5).
* A layer's tensors run `CMF_NVG_JOBS` (default 3) at a time; `CMF_NVG_PROF=1` prints a
  per-stage profile per tensor (with one job: the stage timers synchronise the device).

Covered tensors: `model.layers.{l}.self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`,
`linear_attn.{in_proj_qkv,in_proj_z,out_proj}` with `d_in % 32 == 0`, `d_out % 16 == 0`.
MoE experts are not covered (they stay `q8_2f`); GDN `in_proj_a/b` stay f16.

## 5. CLI

```
cortiq convert --model <hf dir> --quant blr --output m.cmf --bpw 1.0 \
    --blr-method nvg --calib calib.txt [--blr-paths auto|1|2] [--blr-kv-factor 1] \
    [--blr-fit-tokens 14848] [--blr-sel-tokens 3072] [--blr-rho 0.03] [--blr-channel-scale ALPHA] \
    [--blr-threads N] [--blr-keep-src] [--blr-device auto|cpu|gpu] [--blr-plan FILE]
cortiq convert --model <hf dir> --quant blr --output m.cmf --bpw 1.0 --blr-method dual-svid
cortiq nvg-bench --model m.cmf --tensor NAME | --layer L [--jobs J] --bpw 0.55 \
    --device cpu|gpu|both --gram-cache PREFIX --calib calib.txt
```

### 5.1 The budget is a free number, and the plan refines it

`--bpw` is any value in `(0, 16)`: the rank rule turns it into a per-tensor rank.
`0.11`, `0.56`, `1.37` are as valid as `0.55`. The rule can only spend a budget
*approximately*, and the report says what it delivered:

* the rank moves in steps of 8, so the delivered budget sits within one rank
  step **below** the request;
* a request below the container's own floor (rank 8 plus the 16-bit row and
  column scales) cannot be met at all — the floor is delivered and the tensor is
  listed under "could not spend the requested budget";
* the shape cap (`min(d_out, d_in)`) clips the other way: the tensor costs less
  than asked.

`--blr-plan FILE` gives the budget per layer and per tensor — the format the
bit-allocation experiments produce, so a measured allocation can be replayed
verbatim:

```
# a budget plan
layer 0 1.0            # every tensor of layer 0
layer 1 0.55
tensor model.layers.0.mlp.down_proj.weight 0.11   # one tensor, wins over the layer rule
```

The global `--bpw` is the default for everything the plan does not name. A rule
that cannot be honoured (unknown key, bad number, budget out of range) is an
error — a silently ignored rule would spend the wrong bytes. The plan, the
requested budget and the number of clipped tensors are recorded under
`provenance.blr`; the per-tensor report (`<output>.blr.txt`) carries the rank
and the path choice of every tensor.

`cortiq nvg-train --budget-plan FILE` applies the plan **before** the
training: the planned layers are re-encoded (the teacher's weights as the
encode target, the student's own input Grams as the calibration — the same
protocol the budget study measured) and the sign training then optimizes the
metric of the re-planned payloads. The layers the plan does not name keep
their payloads. `--budget-gram-cache DIR` reuses cached Grams
(`L{layer}.{lead}.{kind}.gram`), `--budget-rowweight` applies the per-head row
gauge to the re-encoded tensors.

The budget study behind these knobs — the degradation law, the allocation and
its measured effect — is in [BLR_NVG_BUDGET.md](BLR_NVG_BUDGET.md).

`--blr-device auto` (default) encodes on the device when a wgpu adapter is up (not on
macOS, whose Metal chunk graph runs layer spans out of the capture hook's sight).
`nvg-bench` captures the FIT/SEL Grams of one tensor's (or one layer's) inputs through
layers `0..=l` of the given model once, caches them as f32 files, and encodes from them
with a per-stage profile; `--device both` adds the CPU encoder on the same Grams and
prints the per-tensor comparison.

The converter first writes an f16 source `<output>.src-f16.cmf` (reused if present,
deleted afterwards unless `--blr-keep-src`), prints per-tensor rank, path choice and
errors, writes `<output>.blr.txt`, and records method, budget, both strict bit counts
and the calibration parameters under `provenance.blr`.

## 6. Tests

```
cargo test --release -p cortiq-core blr                       # rank rule, round trip, validation, generic entry points
cargo test --release -p cortiq-engine --features gpu --lib blr   # CPU kernels vs dense, QTensor paths from a CMF, wgpu vs CPU,
                                                                  # graph kernels vs dequant (both reductions, every lane set)
cargo test --release -p cortiq-engine --features gpu --lib wgpu_blr_graph_throughput -- --ignored --nocapture
                                                                  # graph kernels at the 27B shapes, DRAM regime, device timestamps
cargo test --release -p cortiq-cli --bin cortiq -- nvg convert_blr   # monotone ALS/refine, V-search factor, drivers, coverage, CLI parse
bash tests/blr_budget_gate.sh                                     # free --bpw, the plan rules, the budget contract, a bad plan
cargo test --release -p cortiq-engine --features gpu --lib nvg_dev   # GEMM variants, SPD solve + Uc, top-r eigenpairs vs a known spectrum
cargo test --release -p cortiq-cli --bin cortiq -- nvg_gpu            # device vs host per stage: contexts/errors, latents/ITQ/polar,
                                                                      # one ALS sweep, flip_u, gptq_v, the whole-tensor driver
cargo test --release -p cortiq-engine --features gpu --lib nvg_dev -- --ignored --nocapture   # GEMM/Cholesky/flip-rate throughput
cargo test --release -p cortiq-engine --features cuda --lib -- gpu_cuda::nvg_dev::tests gpu_cuda::nvgt::tests --test-threads 1
                                                                      # the CUDA twins: encoder primitives vs f64, blr layer forward /
                                                                      # backward vs f64, the sign update bit for bit vs the host mirror,
                                                                      # GDN and attention cores vs the bake's, the head's top-k and KL
CMF_NVG_ENC=cuda cargo test --release -p cortiq-cli --features cuda --bin cortiq -- nvg_gpu --test-threads 1
                                                                      # the encoder driver vs the f64 host encoder, through the CUDA twin
```

The wgpu tests skip when no adapter is present, the CUDA ones without a card.

## 7. Measured (Qwen3.5-0.8B, 17.10.2026)

GDN hybrid (18 GDN + 6 gated-attention layers, hidden 1024), 150 projections in `blr`,
target 1.0 bpw, P = 2 chosen everywhere by NVG's held-out selection, no training.
`cortiq ppl --windows 16 --window-len 512` on WikiText-2 test; NVG calibration on 14 848
FIT + 3 072 SEL tokens of a mixed calibration text (not the eval split). RTX PRO 6000
(wgpu/Vulkan) and a 27-core CPU quota.

| file | blr body bpw | all projections bpw | file | ppl CPU | ppl wgpu per-op | ppl wgpu default |
|---|---|---|---|---|---|---|
| q8_2f (reference) | — | — | 796 MB | — | — | 24.27 |
| blr, dual-svid | 0.9883 | 1.0061 | 352 MB | 2 719 570 | 2 719 571 | 2 719 571 |
| blr, nvg | 0.9889 | 1.0067 | 352 MB | 656.37 | 656.37 | 656.37 |

"wgpu default" = whole-token/batch graphs allowed; they decline for `blr` and the per-op
path runs. `CMF_BLR_DENSE=1` (dense f32 at load) gives 2 719 569 for the dual-svid file,
equal to the factored kernels. The file is mostly the q8_2f embedding (248 320 × 1024)
and the q8_2f MTP head; the 497 M projection weights take 61.4 MB. Encoding time: dual-svid
7.4 min, NVG 2 h (f64 CPU, sequential calibration ~1.5 min per layer, ~4 min of encoding).

### 7.0 Decode on the wgpu token graph (18.10.2026)

Same card, `bench --core --ignore-eos`, a quiet window (the card is otherwise shared):

| file | decode tok/s | before (per-op `blr`) |
|---|---|---|
| Qwen3.5-0.8B q8_2f | 352.8 | — |
| Qwen3.5-0.8B NVG 1.0 bpw (trained) | 346.5 | 3.9 |
| Qwen3.8-27B NVG 0.55 bpw | 73.7 (32 tokens; prefill 79 tok/s) | not measured (the FFN on the CPU: `P·r > 2048`; ~1 expected) |

Only the token-graph column was measured in the quiet window; the 0.8B per-op 3.9 is the
0.8.20 baseline on the card shared with the 27B training (q8_2f 107.0 tok/s there), a lower bound.
Greedy text of the 0.8B identical to the per-op path, PPL 135.764 on both (16 × 512).
27B on the device: a decoded token ~12.7 ms = the 64 layers ~11.8 ms (~184 µs a layer) plus the
final norm and q8_2f head 0.85 ms. A projection
group costs its V and U dispatches plus their dependency barriers (DRAM regime, V+U pair:
GDN qkv|z ~30 µs, out ~27, gate|up+act ~28, down ~35 against 2–8 µs of bytes at full
bandwidth); stage V of `down` (17408 columns, 272 row sets each re-reading z) is the slowest
single stage.

### 7.1 The device encoder (18.10.2026)

Same card; the CPU side has a 27-core quota. Most device runs shared the card with
another process (NVIDIA time-slices two processes), so device timings are upper bounds.

Per tensor on identical Grams (q8_2f calibration of layers 3 and 17 of the 0.8B, 13
tensors at 1.0 bpw): held-out error within ±0.43 % and fit error within ±0.76 % of the
host encoder (the GPU marginally lower on average). Qwen3.8-27B L0 `gate_proj` at
0.55 bpw: held-out 0.3758 against the host's 0.3759 in 39 s against 2254 s.

Whole 0.8B at 1.0 bpw (sequential calibration, `ppl --windows 16 --window-len 512`):

| encoder | ρ = 0.03 | ρ = 0.0301 | convert |
|---|---|---|---|
| host f64 (`--blr-device cpu`) | 656.37 | 682.16 | 7143 s / 7114 s |
| device, flip tolerance 1e-6 | 686.52 | 710.88 | 1307 s (shared card) |
| device, flip tolerance 1e-7 (default) | 672.52 | 715.80 | 758 s |

The host reference itself moves 3.9 % for a 0.3 % change of ρ: at this compression the
end-to-end perplexity is sensitive to any perturbation of the sequential chain, and
per-tensor errors of two host runs agree within 1 % for only 53 of 150 tensors. Over the
four matched device runs the device sits ~4 % above the host on average (672.5–715.8
against 656.4–682.2); whether that offset comes from the f32 encoder or from the chain's
sensitivity alone is not settled (the run with device calibration and the host encoder
was stopped before the end).

Qwen3.8-27B at 0.55 bpw, one layer encoded on the device from its cached Grams (card
mostly free):

| layer | tensors (encode seconds, one at a time) | layer encode |
|---|---|---|
| 0 (GDN) | down 76.9, gate 23.6, up 23.6, in_proj_qkv 20.0, in_proj_z 17.8, out_proj 14.5 | 176 s (162 s with 3 in flight) |
| 3 (attention) | down 74.5, gate 21.4, up 21.2, q 17.5, o 17.0, k 6.2, v 6.0 | 164 s (155 s with 3 in flight) |

Plus ~3–4.5 s per layer for the input contexts (the 17408-wide `down_proj` input
dominates) and the calibration pass. Where the time goes (layer 0, one at a time): the
column-sequential V search 45 % (34.3 M flips on `down_proj` alone, ~1.3 µs each on one
multiprocessor), the g solves 23 % (n = 17408 Cholesky per g step for `down_proj`), the
eigensolver 9 %, the polar factor 7 %. Tensors in flight overlap little: a Vulkan queue
orders every barrier after all earlier work, so the one-multiprocessor column search of
one tensor holds back the others.

## 8. On CUDA (`CMF_GPU=cuda`)

A build with the `cuda` feature (`cargo build --release -p cortiq-cli --features cuda`; the NVIDIA driver, NVRTC and
cuBLAS at run time) runs the whole NVG path on the card without wgpu: the conversion's calibration, the encoder,
and the trainer. `CMF_GPU=cuda` selects it, as for decoding ([CUDA.md](CUDA.md)):

```bash
CMF_GPU=cuda cortiq convert --model <hf dir> --quant blr --bpw 0.55 --blr-method nvg --calib calib.txt --output m.cmf
CMF_GPU=cuda cortiq nvg-train m.cmf --teacher teacher.cmf --calib text.txt --tokens 1048576 --window 512 --batch 8 \
    --lr 1e-3 --topk 64 --train-signs --sign-opt fact-mom --output m-trained.cmf
CMF_GPU=cuda cortiq nvg-bench --model m.cmf --layer 3 --bpw 0.55 --device gpu --calib calib.txt
```

`--blr-device auto` picks the device encoder when the CUDA backend is up, so a machine without a Vulkan adapter
converts on its card. Files, layouts, snapshots and saved training states are those of the wgpu path: a state
saved by one backend resumes on the other.

| part | where |
|---|---|
| encoder primitives (the API of `gpu_wgpu::nvg_dev`) | `crates/cortiq-engine/src/gpu_cuda/nvg_dev.rs`, kernels `nvg_dev.cu` |
| backend switch of the encoder driver | `crates/cortiq-engine/src/nvg_dev.rs` |
| trainer, calibration forward | `crates/cortiq-engine/src/gpu_cuda/nvgt.rs`, kernels `nvgt.cu`, `nvgt_sign.cu`, and the bake's `bake.cu` |
| trainer backends of `nvg-train` | `crates/cortiq-cli/src/nvgtrain.rs` (`Backend`) |

### 8.1 The encoder

The driver of §4.1 is unchanged: every device call goes through `cortiq_engine::nvg_dev`, which forwards to the
backend the process runs on. The CUDA twin keeps the arithmetic class (f32) and the stage order; what differs:

* *GEMM*: cuBLAS SGEMM, plain f32, over the same sub-matrix views. A row-major operand read column-major is its
  transpose, so `C = op(A)·op(B)` is the call `Cᵀ = op(B)ᵀ·op(A)ᵀ` on the arrays as they lie — no copy. The Hadamard
  epilogue is one more elementwise launch.
* *SPD solves*: an equilibrated right-looking Cholesky — a diagonal block (64 wide up to n = 8192, 128 past it;
  `CMF_NVG_CUDA_CHOL_NB`) factored on the host in f64, its panel by STRSM, the trailing matrix by SSYRK — and
  STRSV / STRSM for the solves and the inverse. A failed pivot grows the ridge ×100, as on the host.
* *Latents*: the Householder tridiagonalisation as kernels (four launches a reflector), the host's f64 bisection +
  inverse iteration (`tridiag_top`, shared with wgpu), the compact-WY back-transform as GEMMs.
* *Knots*: the blocked output-side flips (32 ranks a block, the rest of `qs` by one GEMM); the input-side column
  search's flips on the host (§4.1), its feedback a GEMM. The device kernel of the search (`CMF_NVG_GV=device`)
  takes a 128-column block in one block of 256 threads (`CMF_NVG_GV_WS`; the lanes of a thread in registers,
  argmin and dot products by warp shuffles; ranks up to 4096): a flip is one barrier — the warps' best candidates
  go to shared memory and every thread merges them itself — and a candidate carries its diagonal of K, so the
  winner's acceptance test reads nothing from the matrix.
* *Lanes.* On the backend's one stream every encoder call of every thread waits its turn under the context's
  lock, so tensors in flight overlap only on the host. A job thread queues on a lane of its own instead — a stream
  and a cuBLAS handle (`nvg_dev::private_lane`, pooled and handed on) — and takes no lock: the tensors in flight
  (`CMF_NVG_JOBS`) overlap on the card. The two candidates of a tensor (two paths at rank r, one path at the same bits) are built
  on two threads with a lane each as well (`CMF_NVG_CAND_THREADS=0`: one after the other). The results do not
  depend on any of it: every tensor and candidate is computed from its own matrices and the shared read-only
  weight and Grams. `CMF_NVG_CUDA_LANES=0` puts everything back on the backend's one stream.
* *Buffers*: device matrices come from a pool keyed by size (`CMF_NVG_CUDA_POOL_GB`, default 24 GB of free
  buffers). A buffer a thread frees may still be read by work queued on that thread's stream, so a lane keeps
  what it frees and only it takes it again; when the lane is given up its stream is drained and its buffers go to
  the shared pool.
* A matrix kernel takes its row from the grid's y and its column from x: no 64-bit division per element.

`CMF_NVG_ENC=cuda|wgpu` picks the encoder's backend apart from the engine's (an A/B on identical Grams: wgpu
calibration, CUDA encoder).

### 8.2 The calibration

Under `CMF_GPU=cuda` the conversion does not use the engine's capture hook (its Grams live on the wgpu device or
the host). The calibration forward is the CUDA trainer's (§8.3): for layer `l` a net of the layers `bound..=l` is
loaded from the work file — the layers below `l` already binary low-rank, `l` itself as f16 planes of its q8_2f
weights — each window runs from the cached input of layer `bound` (`nvgt::calib_pass`), and the Grams `H += XᵀX`
of the layer's four inputs (`in`: q/k/v or qkv/z, `mixout`: o_proj / out_proj, `mlp`: gate/up, `down`) accumulate
on the card by plain f32 SGEMM. Positions are the engine calibration's (a window's last token is not fed), each
layer runs once over the conversion. `nvg-bench` and `nvg-train --budget-plan` capture the same way.
`CMF_NVG_CALIB=engine` keeps the engine's hook.

### 8.3 The trainer

`nvg-train` picks its backend at start (`Backend` in `nvgtrain.rs`): the wgpu trainer of `nvg_train.rs`, or
`gpu_cuda::nvgt`. The step is the same — teacher top-k, forward with checkpointed layer inputs, KL through the
frozen head, a reverse sweep that recomputes each layer, Adam on the metric, the sign latents updated right after
each path's backward — and it is laid out for the card's two limits, the tensor cores and the memory bus:

* *Products*: cuBLAS GEMMs on the tensor cores over f16 operands with f32 accumulation (the small GDN a/b
  projections in f32).
* *The scales ride the sign planes.* A bitmap is expanded to the f16 plane `±h_i` (U) or `±g_j` (V), times a
  power of two per path that keeps the largest scale under 1 (`nt_expand_s`). So a layer input is staged in f16
  once for every linear and path that reads it, a path's product lands in the sum — `y`, or the input gradient —
  through the GEMM's β, and nothing the size of an activation is scaled, copied or added per path. h and g are
  f16 in the forward, as they are in the file. The planes of a layer are kept from its forward for its backward.
* *The scales' gradients come out of the sign gradients.* With `G'_U = dyᵀ·(a ⊙ ℓ)` and `G'_V = xᵀ·(t2 ⊙ ℓ)`
  (`t2 = (dy ⊙ h)·S_U`): `dh_i = Σ_k S_U[i][k]·G'_U[i][k]`, `dg_j = Σ_k S_V[j][k]·G'_V[j][k]`, and the signs'
  gradients are `diag(h)·G'_U`, `diag(g)·G'_V` — one row pass over an `m×r` matrix (`nt_g_fold`) instead of a
  column reduction over `tokens×m` activations. Only `dℓ = Σ a ⊙ t2` reduces over tokens, at the rank's width.
* *GDN forward*: the recurrence in blocks of 512 threads (`nt_gdn_fwd`): a thread keeps 32 state rows of one
  column in registers, column sums go through shared memory. A position's q, k and their norms are published a
  position early and its inputs are read ahead, so a position is two barriers and the loads' latency passes under
  the arithmetic. `CMF_NVG_GDN_GC` = 1, 2, 4 picks the block (128, 256, 512 threads).
* *GDN reverse sweep in chunks.* The delta rule over a chunk of C positions is linear algebra in the chunk's start
  state S0 and its rows: with K̂, Q̂ (`[C][128]`), u_t = β_t·(v_t − kv_t) from the forward's `kv`, the decays
  c_τ = Π g inside the chunk and r(τ,σ) = c_τ/c_σ,
  `kv_τ = c_τ(K̂S0)_τ + Σ_{σ<τ} r(τ,σ)(k̂_τ·k̂_σ)u_σ`, `o_τ = c_τ(Q̂S0)_τ + Σ_{σ≤τ} r(τ,σ)(q̂_τ·k̂_σ)u_σ`,
  `S_end = c_end·S0 + K̂ᵀ·diag(c_end/c_σ)·U`. So the adjoints of a chunk are fourteen batched f32 products of
  `[C][128]`, `[C][C]` and `[128][128]` matrices over the (sequence, v-head) units (`cublasSgemmStridedBatched`),
  one back substitution (`X_τ = Ū_τ − Σ_{τ'>τ} β_τ'·r(τ',τ)(k̂_τ'·k̂_τ)·X_τ'`: what the per-position sweep calls du)
  and elementwise kernels (`nt_gdc_*`): `gdn_rev_chunked` in `nvgt.rs`, the derivation at the kernels in
  `nvgt.cu`. Chunks are cut from the window's end (`CMF_NVG_GDN_CHUNK`, default 64: the pairwise products grow
  with the chunk, the launches with their number); the forward keeps only the states at the chunk starts (f16),
  and `kv` and `o` of every position. On an RTX PRO 6000 at 8×512 and 48 v-heads a layer's reverse sweep takes
  7.0 ms where the per-position kernel takes 12.7, the forward that keeps the states 2.6 ms where it took 3.8,
  and the history is 0.1 GB a layer where every second state was 3.2; it is more exact too (a window in one
  chunk needs no stored state: the gradients agree with the skill bake's to 2·10⁻⁷, 1·10⁻⁴ through f16 states).
  `CMF_NVG_GDN_REV=kernel` runs the per-position kernel `nt_gdn_rev` instead: three barriers a position, the
  history every second state (the reverse sweep rebuilds the one between by a rank-one step from `kv`).
  `CMF_NVG_GDN_HIST_GB` (default 16) caps the history by replaying the windows in groups. A v-head writes its
  q̂/k̂ through-gradients, a second launch adds the heads of a k-group in order.
* *The recompute reuses the forward's rank activations.* `a = x·S_V` of every path is kept from the step's first
  forward, per layer and role, and the layer's recompute in the reverse sweep skips that product: the same input
  and the same weights give the same values, so the losses of a run are equal with and without it. The widest
  inputs are kept first (`down_proj` saves the most a byte) within `CMF_NVG_ACACHE_GB` (default 7) and what the
  card has free past a 3 GiB reserve; 0 turns it off.
* *SwiGLU writes its stagings.* In a training net the MLP's activation is read only as `down_proj`'s f16 input,
  and the two gradients of SwiGLU only as the f16 output gradients of `gate_proj` and `up_proj`: the kernels write
  those stagings directly (the values a separate staging pass makes), without the f32 copies.
* *Attention*: batched GEMMs (TF32; `CMF_NVG_CUDA_ATTN=f32` for plain SGEMM) and a causal softmax.
* Every trainable parameter lives in one concatenated buffer with its gradient and Adam moments: the optimizer
  and the gradient norm are one launch each.
* Reductions run in a fixed order and the sign kernels are built without fused multiply-adds, so a step is
  bit-reproducible and a sign update is bit for bit `train_sign`'s host mirror (tested).

`--mask-keep` (DTG-MA before the metric) runs on CUDA too: a gate per MLP neuron scales the activation that
`down_proj` reads (`nt_scale_cols`), its gradient is the column sum of the gated activation's gradient times the
activation, and the gate training, the cut and the refit of ℓ are the wgpu trainer's code over a small trait
(`MaskNet`). `CMF_NVG_PROF=1` prints the time per phase (the stream drained at phase changes), `CMF_NVG_KT=N` the
time of every kernel and GEMM of step N.

### 8.4 Agreement with the wgpu path

The math is the same; the arithmetic differs in its accumulation order (cuBLAS against the wgpu tiles), in the
f16 operands of the trainer's products and its f16 h, g and GDN state history, and in nothing else.
Qwen3.5-0.8B at 1.0 bpw on an RTX PRO 6000 Blackwell:

| what | wgpu | CUDA |
|---|---|---|
| encoder on identical Grams (wgpu calibration, `CMF_NVG_ENC`), layer 0 held-out error: qkv / down / gate / up / z / out | 0.2143 / 0.4883 / 0.3663 / 0.5276 / 0.3371 / 0.3538 | 0.2143 / 0.4880 / 0.3664 / 0.5273 / 0.3377 / 0.3535 |
| the same with the CUDA calibration | — | 0.2145 / 0.4881 / 0.3662 / 0.5279 / 0.3367 / 0.3528 |
| trainer on one file: held-out KL before | 3.47721 | 3.47725 |
| loss of steps 1, 7, 10 | 3.5255, 3.0526, 2.9163 | 3.5255, 3.0533, 2.9170 |
| signs flipped by step 11 | 296 527 | 295 157 |
| held-out KL after 256 steps of 8×512 (signs, `fact-mom`) | 1.44903 | 1.44404 |
| signs flipped in the run | 0.901 % | 0.896 % |
| Qwen3.8-27B at 0.55 bpw, the trainer on one file: held-out KL before | 1.53820 | 1.53821 |
| loss of steps 1, 6, 11, 12 | 1.4572, 1.3403, 1.5225, 1.6106 | 1.4572, 1.3403, 1.5225, 1.6106 |
| signs flipped by step 11 | 4 259 313 | 4 259 718 |
| the 27B's encoder on one capture of layer 3's Grams, held-out error: down / gate / up / q / o / k / v | 0.6899 / 0.5240 / 0.5979 / 0.0825 / 0.0832 / 0.3558 / 0.5079 | 0.6893 / 0.5241 / 0.5978 / 0.0825 / 0.0830 / 0.3560 / 0.5083 |

The 0.8B encoder rows and the step-11 flips were taken with the search in the device's kernel on both backends and
with the per-position GDN reverse sweep; the 27B encoder row is with the flips on the host (§4.1), which the two
backends share. With the device's kernels on CUDA the same capture gives 0.6891 / 0.5239 / 0.5979 / 0.0825 / 0.0829
/ 0.3561 / 0.5083. (An earlier version of this table gave layer 3's MLP tensors the errors of an encode on layer
0's Grams: `nvg-bench` read a cache prefix shared by the two layers. It now refuses such a prefix.)

A conversion repeated on CUDA gives the same bytes, and the training after it the same curve to the last printed
digit. A training stopped at a saved state and resumed gives the losses and the held-out KL of an uninterrupted
run; a state saved by one backend loads on the other.

Kernel tests on the card (`cargo test -p cortiq-engine --features cuda --lib -- gpu_cuda::nvg_dev::tests
gpu_cuda::nvgt::tests`, and the encoder driver's tests with `CMF_NVG_ENC=cuda`): the GEMM views, the elementwise
kernels, the eigensolver, the SPD solve and the whitening factor against f64; a binary low-rank layer's forward
and backward against f64 (4·10⁻⁴: f16 operands and scales); the GDN core against the skill bake's kernels at
odd, even and the shortest lengths, the reverse sweep as the per-position kernel and in chunks (the window as
one chunk, chunks of 2 and 4 with a short first one), the states kept by the forward and replayed a sequence a
launch (output 2·10⁻⁷; gradients 2·10⁻⁷ … 7·10⁻⁴); the attention core against the bake's (equal); the
head's top-k and KL against the host; a sign update against `train_sign::host_step_fact` (no differing latent,
moment or bitmap word); the DTG-MA gates on a real file (`cuda_mask_gradient_check`: open gates leave the loss as
it is, a gate's gradient against a central difference within 0.3 % in the last layer and 4–6 % in the first);
the device encoder against the f64 host encoder (`nvg_gpu` tests; the input-side search with its flips on the
host and in the device's kernel: no sign differs from the f64 encoder's either way).

The trainer's exact changes are checked on a run: 40 steps of the 0.8B give the same losses with and without the
rank-activation cache, and with SwiGLU writing its stagings. A change of the GDN reverse sweep changes its
rounding: the losses stay equal to four digits until the signs move (step 12) and the runs part after it, as two
builds always do (the chunked sweep against the per-position kernel, 40 steps of 4×512: 2.7754 / 2.7754 at step
11, 2.7217 / 2.7216 at 12, held-out KL at 40 2.30670 / 2.30814).

### 8.5 Measured on CUDA (20–21.11.2026)

One RTX PRO 6000 Blackwell (96 GB), 48 host cores; the wgpu column is the same card through Vulkan, the same
binary without `CMF_GPU=cuda`.

Qwen3.5-0.8B at 1.0 bpw (150 projections, calibration 14 848 FIT + 3 072 SEL tokens; the training is the
reference recipe: 1 M tokens, 8 windows of 512, teacher f16, metric and signs, `fact-mom`):

| stage | wgpu | CUDA |
|---|---:|---:|
| calibration of a layer | 2.6 s | 0.4–0.5 s |
| input contexts of a layer | 0.1 s | 0.1 s |
| encode of a layer (three tensors in flight) | 16.6–18.0 s | 1.5–1.7 s |
| whole conversion, 24 layers | 492 s | 64 s |
| teacher top-64 pass | 42.7k tok/s | 55.5k tok/s |
| training step, 4096 tokens | 0.69 s | 0.28 s |
| the reference training, whole run | 227 s | 97 s |

Inside the encoder, layer 0 with one tensor in flight (seconds; `CMF_NVG_PROF=1`):

| tensor | stage | wgpu | CUDA |
|---|---|---:|---:|
| `in_proj_qkv` 6144×1024 | gauge rotation (polar) / g solves / ℓ solves / input-side search / eigenpairs | 0.75 / 0.66 / 0.40 / 0.34 / 0.13 | 0.13 / 0.13 / 0.07 / 0.09 / 0.10 |
| `down_proj` 1024×3584 | g solves / input-side search | 1.59 / 2.11 | 0.42 / 0.48 |

(The wgpu column is the device's search kernel, measured before the flips moved to the host; with them the wgpu
encoder takes layer 0 in 14.2 s with one tensor in flight, 17.8 s with its kernel.)

The linear algebra is 3–6 times faster on CUDA. The column-sequential input-side search was what remained — one
block on one multiprocessor on either backend, 0.9 µs a flip here (1.57 s of `down_proj`) — until its flips
moved to the host thread (§4.1): 0.27 µs a flip, transfers included. Layer 0 on CUDA, warm kernels, a quiet card,
one tensor in flight and four:

| | one | four |
|---|---:|---:|
| the backend's one stream, the search in the device's kernel | 6.0 s | 5.9 s |
| a lane a thread, the two candidates on two threads (§8.1) | 5.2 s | 4.3 s |
| and the flips on the host | 2.7 s | 1.5 s |

The whole conversion: 202 s, 141 s, 64 s. The lanes and the candidate threads do not change the file (one sha256
with and without them); the host's flips give another file of the same quality (§4.1). The q1g repack
(`requant --quant q1g`) is a lossless host pass over bytes and has no device stage.

Qwen3.8-27B at 0.55 bpw. The trainer: the 4.89 GB student against its q8_2f teacher, 8 windows of 512, metric and
signs. The encoder: one layer from the q8_2f weights, 32 768 + 4 096 calibration tokens, each layer's Grams
captured once (on CUDA) and read by both backends.

| stage | wgpu | CUDA |
|---|---:|---:|
| teacher top-64 pass | 2.7k tok/s | 5.0k tok/s |
| training step, 4096 tokens | 5.48 s | 2.97 s |
| 1 M tokens: 256 steps and their teacher pass | 29.5 min | 16.2 min |
| card memory while training | 90.8 GB | 87.9 GB |
| calibration of layer 0 / layer 3 (a forward from the embeddings) | 17.1 s / 36.8 s | 5.6 s / 8.1 s |
| input contexts of a layer | 2.9–5.2 s | 1.3 s |
| encode of layer 0 (GDN, 6 tensors): one tensor in flight / three | 106.6 s | 40.4 s / 29.8 s |
| encode of layer 3 (attention, 7 tensors): one tensor in flight / three | 103.0 s | 38.2 s / 28.2 s |
| of it `down_proj` 5120×17408: whole / input-side search / g solves | 42.2 / 16.6 / 15.2 s | 17.3 / 16.7 / 10.1 s |
| `gate_proj` 17408×5120, `q_proj` 12288×5120, `k_proj` 1024×5120 | 14.7, 12.7, 4.1 s | 5.6, 4.5, 0.8 s |

(CUDA's search and solve times of `down_proj` are summed over its two candidates' threads, which overlap; the
tensor takes 17.3 s.)

What this pass changed, the same stand and the same Grams:

| | before | now |
|---|---:|---:|
| encode of layer 3 on CUDA, one tensor in flight | 90.1 s (one stream, the search in the device's kernel) | 38.2 s |
| three / four tensors in flight | no faster than one | 28.2 s / 29.5 s |
| encode of layer 0 on CUDA, one / three in flight | 96.5 s | 40.4 s / 29.8 s |
| `down_proj`'s input-side search, 31 M flips | 37.8 s (1.22 µs a flip) | 16.7 s over two threads (0.54 µs) |
| encode of a layer on wgpu | 148.9–149.3 s | 103.0–106.6 s |
| training step on CUDA | 3.57 s | 2.97 s (3.29 s with the per-position GDN reverse sweep) |

A training run of 10.5 M tokens of the 27B (2559 steps, their teacher pass, a held-out evaluation every 32 steps
and the saved states) took 3 h 55 min at 3.56 s a step; the next one ran at 3.3–3.4 s a step on an intermediate
build, beside other jobs on the card.

A CUDA step of the 27B (2.97 s; `CMF_NVG_KT`): the f16 GEMMs 1.42 s, 48 % (about 4.3·10¹⁴ flop a step; the six
products of a path run at 280–345 TFLOP/s at the MLP's shapes, `cuda_blr_gemm_shapes_throughput` — cuBLAS's rate on
this card), the GDN recurrence 0.50 s, 17 % (the forward kernel 0.21 s, the reverse sweep's batched products 0.13 s
and its kernels 0.16 s), the sign update's passes over the latents 0.22 s, 7 %, the remaining stagings and plane
expansions 0.23 s, 8 %, SwiGLU 0.11 s, 4 %. The step before this pass, 3.57 s: GEMMs 1.53 s (the recompute's x·S_V
among them), the recurrence 0.94 s (reverse 0.57, forward 0.37), stagings 0.19 s, SwiGLU 0.14 s. And before the
scales rode the planes and the recurrence ran in 512-thread blocks, 4.50 s (the elementwise passes over activations
alone 40 % of it), the 0.8B's 0.54 s. What is left is cuBLAS at its rate and passes that run at the memory bus.

### 8.6 SFq's two ideas at the NVG start

SFq ([Q2TP_SCALE_REFINE.md](Q2TP_SCALE_REFINE.md)) gave q2tp 7–14 % lower perplexity at the same size by a
channel scale on the inputs and a refinement of the codes on calibration rows under a ridge. NVG has both inside:
g is a free scale per input channel, solved exactly, and the discrete knots are a code refinement on the
calibration Gram under the damping ρ. An input rescaling changes the encoder only through that damping and the
column order of the input-side search. Measured on the 0.8B at 1.0 bpw, all on CUDA: perplexity on 16 × 512 of
WikiText-2 test / C4 validation, the reference training (256 steps), its held-out KL. A build's own unchanged run
is the row to compare with: the two builds differ in the last bits of their arithmetic (a build repeats itself bit
for bit).

| variant | no training: ppl | no training: KL | KL at 25 / 100 / 256 steps | after training: ppl |
|---|---:|---:|---:|---:|
| *first build:* NVG as it is | 695.3 / 1014.0 | 3.565 | 2.437 / 1.702 / 1.445 | 80.25 / 154.97 |
| `--blr-channel-scale 0.75` | 592.7 / 921.7 | 3.422 | 2.472 / 1.730 / 1.461 | 82.50 / 155.82 |
| `--blr-rho 0.1` | 823.5 / 1139.9 | 3.699 | 2.474 / 1.703 / 1.437 | 78.02 / 151.47 |
| `--blr-rho 0.3` | 1221.4 / 1621.3 | 4.030 | 2.629 / 1.742 / 1.458 | 79.33 / 156.21 |
| four refinement sweeps (`CMF_NVG_SWEEPS=4`, conversion ×2) | 675.6 / 949.5 | 3.452 | 2.432 / 1.695 / 1.452 | 80.22 / 156.15 |
| *final build:* NVG as it is | 730.1 / 996.1 | 3.525 | 2.448 / 1.721 / 1.461 | 82.01 / 156.87 |
| `--blr-channel-scale 0.5` | 615.8 / 961.9 | 3.432 | 2.435 / 1.714 / 1.466 | 81.37 / 157.35 |
| `--blr-channel-scale 1.0` | 608.5 / 895.6 | 3.396 | 2.391 / 1.697 / 1.435 | 78.91 / 151.92 |
| `--blr-rho 0.01` | 709.6 / 973.8 | 3.491 | 2.481 / 1.736 / 1.483 | 83.61 / 156.25 |
| four times the calibration rows (`--blr-fit-tokens 59392 --blr-sel-tokens 12288`) | 580.8 / 1013.4 | 3.412 | 2.380 / 1.691 / 1.436 | 79.11 / 153.70 |
| the rows and `--blr-rho 0.1` | 746.6 / 1142.9 | 3.644 | 2.404 / 1.680 / 1.424 | 79.52 / 151.72 |
| the rows, `--blr-rho 0.1` and `--blr-channel-scale 1.0` | 674.3 / 1090.2 | 3.571 | 2.397 / 1.666 / 1.420 | 78.87 / 153.29 |

The two unchanged runs differ by 5 % in the untrained perplexity and by 1 % in the trained KL, and one training
seed is one draw: the files of the final build were trained again with two other seeds (a seed changes the order
of the training windows and the held-out windows, so a KL compares within a seed only).

| held-out KL after training | seed 42 | seed 7 | seed 11 | against the unchanged file |
|---|---:|---:|---:|---:|
| NVG as it is | 1.4608 | 1.5142 | 1.5509 | |
| channel scale 0.5 | 1.4657 | 1.5393 | 1.5627 | +0.3 %, +1.7 %, +0.8 % |
| channel scale 1.0 | 1.4350 | 1.5128 | 1.5418 | −1.8 %, −0.1 %, −0.6 % |
| four times the rows | 1.4363 | 1.5254 | 1.5342 | −1.7 %, +0.7 %, −1.1 % |
| the rows and ρ 0.1 | 1.4245 | 1.4904 | 1.5195 | −2.5 %, −1.6 %, −2.0 % |
| the rows, ρ 0.1 and scale 1.0 | 1.4199 | 1.4878 | 1.5160 | −2.8 %, −1.7 %, −2.3 % |

Reading it:

* *The channel scale* makes the untrained file better at every seed — the KL to the teacher 3–4 % lower,
  WikiText-2 perplexity about 15 %, C4 3–10 % — and the training erases it: after 256 steps the scaled files end
  where the unscaled one does.
* *A gentler fit* — a ridge of 0.1 on four times the calibration rows — starts no better (the stronger ridge
  alone makes the untrained file worse) and ends 2 % lower in KL at all three seeds; the scale on top adds
  nothing measurable. The file reaches the unchanged run's final KL at about step 195 of 256.
* More refinement sweeps, a weaker or a much stronger ridge, more rows alone: inside the resolution.

So SFq's channel scale helps a file that will not be trained (`--blr-channel-scale 1.0`), and its ridge with more
calibration rows (`--blr-rho 0.1 --blr-fit-tokens 59392 --blr-sel-tokens 12288`; on CUDA the rows cost nothing:
the conversion takes the same 3–4 minutes) trims the training by about a fifth for the same KL, or leaves the
trained file 2 % better. The defaults are unchanged, and none of this was repeated on the 27B.
