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
| encoders, `convert --quant blr` | `crates/cortiq-cli/src/nvg.rs`, `nvg_linalg.rs` |

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
* **wgpu** (`b = 1` decode and `b > 1` prefill) — `blr_v`: one workgroup per latent row
  and block of 4 tokens, signs 16 at a time, f16 scales via `unpack2x16float`;
  `blr_u`: one thread per output row, the token's latents staged in workgroup memory.
  The payload goes into the resident weight cache once. Contract: `d_in % 32 == 0`,
  `d_out % 4 == 0`, `P·r ≤ 2048`; outside it the CPU kernel runs. Metal has no `blr`
  kernel yet (CPU fallback).
* **Engine wiring** — `QTensor::Blr`: `matvec`, `matvec2`, `matmat` (GPU first when the
  device is up, `CMF_BLR_GPU=0` keeps it on the CPU); multi-tensor jobs and fused
  gate/up refuse `blr` and fall back to per-tensor calls; rows/columns reconstructed on
  the fly for the rare paths. Every device-graph accessor (`graph_weight`,
  `mapped_device_gemm`, `q1_parts`, …) answers `None`, so whole-token and batch graphs
  decline to the per-op path. `CMF_BLR_DENSE=1` dequantizes to f32 at load (A/B parity).

## 4. Encoders

Both run per tensor in f64 on the CPU (`nvg_linalg.rs`: parallel GEMM, blocked
Cholesky, triangular inverse, symmetric eigensolver, polar factor). Scales are
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
out_proj each on their own. The capture uses the engine's GPTQ hook in the CPU batched
prefill (the converter sets `CMF_GPU=0` for `nvg`).

Covered tensors: `model.layers.{l}.self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`,
`linear_attn.{in_proj_qkv,in_proj_z,out_proj}` with `d_in % 32 == 0`, `d_out % 16 == 0`.
MoE experts are not covered (they stay `q8_2f`); GDN `in_proj_a/b` stay f16.

## 5. CLI

```
cortiq convert --model <hf dir> --quant blr --output m.cmf --bpw 1.0 \
    --blr-method nvg --calib calib.txt [--blr-paths auto|1|2] [--blr-kv-factor 1] \
    [--blr-fit-tokens 14848] [--blr-sel-tokens 3072] [--blr-rho 0.03] \
    [--blr-threads N] [--blr-keep-src]
cortiq convert --model <hf dir> --quant blr --output m.cmf --bpw 1.0 --blr-method dual-svid
```

The converter first writes an f16 source `<output>.src-f16.cmf` (reused if present,
deleted afterwards unless `--blr-keep-src`), prints per-tensor rank, path choice and
errors, writes `<output>.blr.txt`, and records method, budget, both strict bit counts
and the calibration parameters under `provenance.blr`.

## 6. Tests

```
cargo test --release -p cortiq-core blr                       # rank rule, round trip, validation, generic entry points
cargo test --release -p cortiq-engine --features gpu --lib blr   # CPU kernels vs dense, QTensor paths from a CMF, wgpu vs CPU
cargo test --release -p cortiq-cli --bin cortiq -- nvg convert_blr   # monotone ALS/refine, V-search factor, drivers, coverage, CLI parse
```

The wgpu parity test skips when no adapter is present.

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
