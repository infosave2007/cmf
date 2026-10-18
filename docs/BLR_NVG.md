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
| NVG encoder on the device | `crates/cortiq-cli/src/nvg_gpu.rs` over `crates/cortiq-engine/src/gpu_wgpu/nvg_dev.rs` |
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
  `Mᵀe_j = (Mᵀw_j − g_j K s_j)/Uc[j,j]` (no m-length vector in the sequential part); a
  128-column block runs in one 512-thread workgroup (lanes unrolled in registers, argmin
  and dot products by subgroup operations, two barriers per flip), the trailing feedback
  `BV[b1:] −= Uc[b0:b1, b1:]ᵀ·E` is a GEMM.
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
    [--blr-fit-tokens 14848] [--blr-sel-tokens 3072] [--blr-rho 0.03] \
    [--blr-threads N] [--blr-keep-src] [--blr-device auto|cpu|gpu]
cortiq convert --model <hf dir> --quant blr --output m.cmf --bpw 1.0 --blr-method dual-svid
cortiq nvg-bench --model m.cmf --tensor NAME | --layer L [--jobs J] --bpw 0.55 \
    --device cpu|gpu|both --gram-cache PREFIX --calib calib.txt
```

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
cargo test --release -p cortiq-engine --features gpu --lib nvg_dev   # GEMM variants, SPD solve + Uc, top-r eigenpairs vs a known spectrum
cargo test --release -p cortiq-cli --bin cortiq -- nvg_gpu            # device vs host per stage: contexts/errors, latents/ITQ/polar,
                                                                      # one ALS sweep, flip_u, gptq_v, the whole-tensor driver
cargo test --release -p cortiq-engine --features gpu --lib nvg_dev -- --ignored --nocapture   # GEMM/Cholesky/flip-rate throughput
```

The wgpu tests skip when no adapter is present.

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
