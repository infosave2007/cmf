# DeepSeek-V4.1-Flash-1017: GPTQ for the experts (wave 1, task B)

Branch `dsv41-moe-gptq` (worktree `/Users/oleg/dev/cmf-dsv41b`), based on
origin/master = cortiq 0.8.23. Mac-only work: builds and CPU tests. No pod
access and no model files. The GPU encode path is compiled but has not been run
(see "Not verified").

Round 2 (branch `dsv41-int-b`, worktree `/Users/oleg/dev/cmf-dsv41ib`, from
`dsv41-int`): the review items B1–B12, see "Round 2" below. The pod commands
in "Pod plan" are the round-2 sequence.

## Goal

`cortiq quantize-moe` handled only Flash-Next (`qwen4_exp`), and only in the q2tp
layout. The owner wants new, more accurate q4tp and q2tp files of
DeepSeek-V4.1-Flash-1017: GPTQ of all 384 routed experts and the shared expert
in every one of the 40 layers, with the exact layouts that the converter writes
and that the V4.1 dynamic MoE bank accepts (`dsv41.rs` `dynamic_moe_gpu`: one
gate/up dtype for every slot, down always q4tp, the shared expert in the same
layout).

## What changed

### Engine

- `crates/cortiq-engine/src/dsv41.rs`
  - `Dsv41MoeCapture` on `Dsv41State` (`moe_capture`). `moe()` records each
    token's MoE input (the RMS-normed mHC fold that both the routed experts and
    the shared expert read), stored as f16, plus its `top_k` routed expert ids.
    The record is taken right after the host route: sqrt-softplus scores, the
    selection bias (`bias_vl` for image tokens), top-k. Both the CPU MoE and the
    dynamic GPU MoE take their route from the host, so one hook covers both.
  - `capture_window(..)`: runs a fresh window from position 0 and records layers
    `lo..hi`. With `stop_after_hi`, each token stops after layer `hi-1` and skips
    the rest of the stack and the head. This leaves the captured rows unchanged
    because no layer reads state written by a later layer: KV, index and
    candidate sources all sit at or below their consumers (checked in
    `attention`: candidates are read only for `li > candidate_source`, and
    `topk` only by layers at or after their index source).
  - The rows are stored as f16. The forward's MoE input is BF16-valued (`rms`
    rounds its output to BF16). A BF16 value in f16's normal range is exact in
    f16, and below 2^-17 only the lowest bits are lost.
- `crates/cortiq-engine/src/pipeline.rs`: `Pipeline::dsv41_moe_capture(ids, lo, hi)`.
- `crates/cortiq-engine/src/gpu_wgpu/train_ops.rs`: `swiglu_lim_fwd` /
  `swiglu_lim(..)` computes `C = s·silu(min(A,L))·clamp(B,−L,L)`. This is the
  DeepSeek `swiglu_limit` (L = 10 in V4.1), with an output scale `s` for the
  layer-prior rows. Test `elt_wgsl_validates` parses and validates the module
  with naga on the CPU: one bad entry would break every kernel in the module,
  `swiglu_fwd` included (used by nvg-train and by Flash-Next quantize-moe).

### CLI: `crates/cortiq-cli/src/moegptq.rs` and `main.rs`

- Two families: `Qwen4` (unchanged behaviour) and `Dsv41`. A file is detected as
  V4.1 when `pipe.dsv41.is_some()`. The expert directory triples are looked up by
  the converter's names (`model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight`,
  `…mlp.shared_expert.…`), with the upstream `ffn`/`w1..w3` spellings as a
  fallback, the same names the runtime loader accepts.
- `--codec q2tp|q4tp` (default q2tp, as before):
  - q2tp: gate/up go through the q2tp sweep (placed rule) plus the output
    debias; down goes through the q4tp sweep.
  - q4tp: all three go through the q4tp sweep, with no debias (same default as
    `quantize-gptq`).
  - The shared expert always gets the same layout as the routed experts. In a
    q2tp file it uses the q2tp slot layout (the Flash-Next trap from 17.10).
- Source weights are the exact dequantized checkpoint, decoded by the
  converter's own unpackers:
  - Routed experts: `I8` nibble pairs `[rows, cols/2]` plus `F8_E8M0` scales
    `[rows, cols/32]`, decoded by `convert::unpack_mxfp4`.
  - Shared expert: `F8_E4M3` plus `F8_E8M0` scales per 32×32 tile, decoded by
    `convert::unpack_fp8_blocks`. The block size comes from the preserved
    `quantization_config.weight_block_size`.
  - Plain float tensors also work.
  - Mapping of source matrices: `w1` = gate, `w3` = up, `w2` = down.
- Down input and SwiGLU clamp. The down input is
  `silu(min(g, L))·clamp(u, −L, L)`, built from the full-precision gate and up
  exactly as the forward builds it (L = 0 means no clamp; Flash-Next is
  unchanged). With a limit, the prior rows are scaled by the kernel's output
  scale, because the clamp does not commute with the `b·W_u` trick. Without a
  limit, the old `b·W_u` path is kept byte for byte.
- The Hessians are unweighted by route weights by default, as in Flash-Next.
  Round 2 adds the route-weighted down Hessian as an option
  (`CMF_MOE_ROUTE_W=1`, see "Round 2").
- `--hf` can be a local directory. Shards are then read in place and never
  deleted. A Hub repo still fetches one layer at a time, prefetches the next
  layer and deletes after use. In the 1017 index every layer's experts (weights
  and scales) sit in one shard, `model-000{L+3}-of-00048` (layer 0 = shard 3,
  layer 39 = shard 42). Scale planes are included in the shard lookup.
- `--in-place`: the experts are written into the calibration file's own slots,
  one layer at a time (`CmfModel::recode_entries_in_place`).
  - All layers are captured in one pass first, so no capture runs over experts
    that were already rewritten. The pass size is forced to the whole range.
  - q4tp payloads fill q4tp slots exactly.
  - q2tp payloads are smaller. The slack is released with
    `fallocate(PUNCH_HOLE|KEEP_SIZE)` on Linux: the file length stays the same
    and the physical size drops.
  - Only a single-file calibration model is accepted.
  - The header provenance stays the calibration file's.
- `CMF_MOE_HOST=1` runs the same steps on the host (f64 Hessians, host fold,
  host sweeps). This is the path the tests use. It is exact but far too slow
  for the full model. Since round 2 there is no silent fallback: without a wgpu
  device, or on one without matrix units, the run stops before the capture.
- `CMF_MOE_RANGE` together with `--output` is now refused, because the other
  layers' experts would be missing from the file. It is allowed with
  `--in-place`, `--patch-shared` or `CMF_MOE_DIAG`.
- After the last capture pass the calibration pipeline is dropped, and its
  device caches are released with `gpu_wgpu::release_model_buffers` (new, a pub
  wrapper over `release_idle_model_buffers`). The V4.1 global expert bank lives
  in the device context, keyed by the model, so dropping the pipeline alone does
  not free it. By default it takes `CMF_DSV41_POOL_PCT=75` % of the card (at
  most 85 %).
- Refactor: `encode_layer` (routing lists → workers → routed and shared
  experts), `Sink` (Stream / InPlace / Patch / Diag), `encode_with_hessians`.
  Flash-Next keeps its behaviour:
  - top-k is computed from the f16-rounded logits at capture time, with the same
    `select_nth` code;
  - the same sweeps, prior, RTN fallback and emission order;
  - the output provenance has new fields (`family`, `codec`, `swiglu_limit`,
    `prior`).
- MTP / dspark tensors (`mtp.*`) are not GPTQ'd. They are copied, or left in
  place, as the calibration file has them.

## Round 2 (branch `dsv41-int-b`)

### Changes, one commit each

- B1, no silent host fallback. Without a wgpu device the run stops:
  `no wgpu GPU (set CMF_MOE_HOST=1 to run on the host)`. It also stops on a
  device without matrix units (cooperative matrices). The device grams and
  GEMMs need them, so before this the first expert failed with `gemm` after
  the whole capture. This happens on the M4 through wgpu/Metal. Tests call
  `encode_layer` with `Engine::Host` and are unaffected.
- B2, slot checks before the capture.
  - `--in-place`: every expert slot of the range must be the codec's dtype
    or q4tp, and hold at least the codec payload (`expected_nbytes`, the
    q4tp/q2tp section formulas). So q4tp into the q2tp slots of a q2tp file is
    refused.
  - `--patch-shared`: the patched file's dtype must equal the codec's. The
    message names the `--codec` to pass.
- B3, hole-punching probe. Before a q2tp `--in-place` run, a 1 MiB file is
  written next to the calibration file and one hole is punched in it
  (`fallocate(PUNCH_HOLE|KEEP_SIZE)`). The allocated blocks must drop. If
  they do not, the run stops and names `CMF_MOE_NO_PUNCH=1`. With that
  variable set, no hole is punched at all. Off Linux the probe always fails.
- B4, V4.1 prior default. When `CMF_MOE_PRIOR` is unset and the mean routed
  rows per expert (`n_cal·top_k/ne`) is below hidden, n0 = hidden (5120). The
  prior used is always printed, and is recorded in the output provenance
  (`prior`, `prior_source`, `prior_tokens`). `CMF_MOE_PRIOR=0` turns it off.
  Flash-Next has no default, as before.
- B5, saved capture (`CMF_MOE_CAPTURE_DIR=<dir>`).
  - Files per layer: `L{li}.x` (f16 rows, raw LE u16), `L{li}.picks` (u16)
    and `L{li}.w` (f32 route weights).
  - `meta.json`: tokens, window, number of windows, hidden, top_k, experts,
    family, SHA-256 of the calibration windows (each window's length, then
    its ids), calibration file path and size, cortiq version, and the layers
    held.
  - Every file goes through a `.tmp` sibling and a rename. A layer is listed
    in `meta.json` only after its files are complete.
  - A pass whose layers are all on disk is loaded instead of captured. When
    every pass is on disk, there is no calibration forward at all: the
    pipeline is opened for the tokenizer and config, then dropped, and its
    device buffers are released.
  - A capture of another run is refused, never overwritten. "Another run"
    means other tokens, hidden, top_k or experts. A different calibration
    file path or size only prints a note, so a resumed `--in-place` run on a
    half-rewritten file reuses the pristine capture.
  - `CMF_MOE_CAPTURE_ONLY=1` captures all layers of the range in one forward,
    saves them and stops. Nothing is encoded or written; the run still needs
    `--output` for clap (`/dev/null`).
  - Size: 40 × (5120·2 + 6·2 + 6·4) = 411 KB per token. That is 13.5 GB at
    32,768 tokens and 26.9 GB at 65,536. The run prints the size before it
    writes.
- B6, route weights in the down Hessian.
  - The V4.1 capture records each pick's route weight (`mix_weights`), next to
    the picks.
  - The reference `Expert.forward` multiplies the SwiGLU activation by the
    route weight before `w2`. So `w2`'s true input Hessian is
    `Σ w_t²·y_t y_tᵀ`.
  - `CMF_MOE_ROUTE_W=1` scales each down-input row by its `w_t`. On the
    device this uses a new ELT entry, `swiglu_lim_rows_fwd`.
  - Gate/up Hessians stay unweighted.
  - With the weighting on, the down prior's pseudo-rows carry the expert's
    mean `w²`, so the prior keeps its share of the Hessian.
  - Default off. `CMF_MOE_ROUTE_W=diag` (with `CMF_MOE_DIAG`) runs every
    prior with both Hessians.
  - DIAG now also reports the held-out error weighted by `w²`, which is the
    error the MoE output sees. Variant labels are `n0` and `n0+w`.
  - Flash-Next is unchanged.
- B7, prefetch failures. A failed background shard prefetch is logged. The
  layer then fetches the shard synchronously, with up to 6 attempts and
  20/40/60/80/100 s pauses.
- B8: `--output` together with `--in-place` is refused, by a clap conflict and
  again in `run`.
- B9: every V4.1 calibration window starts with BOS: the tokenizer's id, or 0.
  Each window then carries `window − 1` text tokens. Flash-Next windows are
  unchanged.
- B10, q2tp debias on the device.
  - With `Engine::Gpu`, `[W_g; W_u]·H` runs as one device GEMM: the NVG
    encoder's exact-f32 FMA GEMM, never the tf32-class matrix units.
  - The host path (`Engine::Host`) keeps its f32 GEMM.
  - Measured on the M4 at 2304×5120 · 5120²:

    | | speed | result |
    |---|---:|---|
    | device | 865–918 GFLOP/s | bit-identical to the host; worst error against f64 9e-8 of Σ\|terms\| |
    | host (Accelerate AMX, 10 threads) | 1032–1385 GFLOP/s | |

    Both run in-order FMA chains, which is why the products are
    bit-identical. 0 of 4608 debiased q2tp rows differ.
  - So the f32 GEMM error does not reach the payload, and no diag measurement
    was needed.
  - On the pod, the host GEMM is matrixmultiply on x86. Its speed was not
    measured; compare the per-layer `gate/up` stage time with round 1.
- B11, tests: see the table below.
- The merged `dsv41-int` lib tests did not compile: `tiny_stack` set
  `candidate_source: 99`, an `Option` since the `-1` fix. Fixed.

### Environment variables

| variable | effect |
|---|---|
| `CMF_MOE_HOST=1` | run the encode on the host; otherwise a wgpu device with matrix units is required |
| `CMF_MOE_PRIOR=n0` | the layer prior (pseudo-rows; down n0/4). V4.1 default: hidden when mean rows per expert < hidden; `0` = none |
| `CMF_MOE_PRIOR_S` | prior sample tokens (8192) |
| `CMF_MOE_ROUTE_W=0\|1\|diag` | route-weighted down Hessian: off (default) / on / both under DIAG |
| `CMF_MOE_CAPTURE_DIR=dir` | save the capture there and reuse it when it matches |
| `CMF_MOE_CAPTURE_ONLY=1` | capture all layers in one forward into the directory, then stop |
| `CMF_MOE_NO_PUNCH=1` | q2tp `--in-place` without hole punching (slack stays allocated) |
| `CMF_MOE_DIAG=n0,…` / `CMF_MOE_HOLDOUT` / `CMF_MOE_RANGE` / `CMF_MOE_WORKERS` / `CMF_MOE_STOP_AFTER` | as in round 1 |
| `CMF_MOE_GPU_TEST=1` | run the opt-in GPU tests (macOS also needs `CMF_GPU=wgpu`) |

### Round 2 tests (Mac)

| test | checks | result |
|---|---|---|
| `moegptq_dsv41_e2e::quantize_moe_on_the_v41_fixture` (cli integration) | The real binary, CPU only, on the engine's 5-layer V4.1 fixture with an F32 source directory (see the list below the table) | ok |
| `dsv41_fixture::moe_capture_early_stop_matches_full_forward` (engine integration) | Early-stop capture equals the full-forward record: rows, routes, route weights. Ranges 1..2, 2..3, 3..4 and 1..3, over the compressed-KV + index source with Engram (layer 1), the candidate source (2) and the index consumer (3). The fixture has these layers; `tiny_stack` would need a CMF file for Engram's raw FP8 rows | ok |
| `dsv41::tests::moe_capture_records_inputs_and_routes` | Extended: the route weights equal the picks' scores normalised to `route_scale` | ok |
| `train_ops::tests::rms_and_elementwise_match_host` (`CMF_GPU=wgpu`) | `swiglu_lim_rows` against f64 on the M4: rel. error 6.7e-8 | ok |
| `moegptq::tests::prior_carries_experts_with_fewer_rows_than_hidden` | hidden 256 with 48/128/192 rows (see the second table below) | ok |
| `moegptq::tests::route_weighted_down_hessian_tracks_weighted_error` | Input-correlated weights (two clusters, w = 1 and 0.05): the route-weighted down Hessian cuts the w²-weighted held-out error of the down projection 0.00798 → 0.00715 | ok |
| `moegptq::tests::capture_dir_round_trip_and_refusal` | Bit-exact round trip. Refusal on other tokens, hidden, top_k or experts. A truncated file is an error. No `.tmp` files are left | ok |
| `moegptq::tests::swiglu_rows_clamp_matches_hand_values` | g=12,u=−15 → silu(10)·(−10); g=−12,u=3 unclamped; no limit → no clamp | ok |
| `moegptq::tests::header_derived_settings`, `passes_cover_the_range`, `production_prior_defaults`, `calib_windows_start_with_bos`, `downloads_are_retried`, `hole_punch_probe_cleans_up`, `tests::moegptq_output_conflicts_with_in_place` | the factored settings, pass plan, prior default, BOS windows, retries, probe, clap conflict | ok |
| `moegptq::tests::gpu_debias_product_matches_host` (`CMF_MOE_GPU_TEST=1 CMF_GPU=wgpu`) | B10 numbers above | ok on the M4 |
| `moegptq::tests::gpu_encode_tracks_the_host_reference` (`CMF_MOE_GPU_TEST=1`) | Now also route-weighted and the 48/128/192-row case. Fails, instead of skipping, without a device or matrix units | panics on the M4 (no matrix units): pod only |

What the end-to-end test runs, in order:

1. No GPU: the run stops.
2. q2tp `--output`, which saves the capture. Checks: BOS, prior n0 = 64 by
   the V4.1 default, `meta.json`, verify, provenance.
3. `CMF_MOE_CAPTURE_ONLY` into a second directory: the same bytes per layer,
   and no file written.
4. A second q2tp run from the saved capture: no forward, and a byte-identical
   output file.
5. Other `--tokens`: refused.
6. DIAG with `CMF_MOE_ROUTE_W=diag` on layers 2..4, from the saved capture.
7. q4tp `--in-place` into the fixture's q2tp slots: refused before any
   capture.
8. q2tp `--in-place`: on macOS the probe stops it; with `CMF_MOE_NO_PUNCH=1`
   it runs from the saved capture, the file verifies, and its experts are
   identical to step 2's.

Held-out relative output error in
`prior_carries_experts_with_fewer_rows_than_hidden` (hidden 256), RTN → GPTQ →
GPTQ + prior:

| rows | q4tp | q2tp |
|---:|---|---|
| 48 | 0.0542 → 0.0248 → 0.0115 | 0.500 → 0.118 → 0.061 |
| 128 | 0.0542 → 0.0194 → 0.0114 | 0.500 → 0.075 → 0.061 |
| 192 | 0.0542 → 0.0140 → 0.0113 | 0.500 → 0.071 → 0.062 |

The same comparison runs in the opt-in GPU test.

The Linux-only code (the probe, `punch_holes`) was type-checked for
`x86_64-unknown-linux-gnu`. It was extracted into one file and checked with
`rustc --emit=metadata` against the target's `libc`; an injected type error is
reported. A full `cargo check --target x86_64-unknown-linux-gnu` fails on the
Mac only because `ring` needs a cross C compiler.

## Tests (Mac, CPU)

| test | checks | result |
|---|---|---|
| `dsv41::tests::moe_capture_records_inputs_and_routes` (engine) | Tiny 3-layer window-only V4.1 stack of f32 tensors, real per-token forward. Capture of layer 1 with early stop is byte-identical to the same record taken through the full stack. The record is per layer. The route recomputed from the captured row (sqrt-softplus + bias, top-k) equals the captured ids. | ok |
| `gpu_wgpu::train_ops::tests::elt_wgsl_validates` (engine, `--features gpu`) | ELT module, including `swiglu_lim_fwd`, parses and validates in naga | ok |
| `moegptq::tests::dsv41_source_decodes_fp4_and_fp8_exactly` | FP4 routed and FP8 (32×32) shared tensors, written by the test, decode bit for bit to an independent f64 decode; a wrong geometry is refused | ok |
| `moegptq::tests::identity_hessian_gptq_is_plain_rounding` | Identity Hessian, production `encode_with_hessians` (see the list below the table) | ok |
| `moegptq::tests::gptq_beats_rounding_on_held_out_rows` | Correlated synthetic inputs, clamp exercised (L = 1), 1024 calibration rows and 256 held-out rows. Relative held-out output error, RTN → GPTQ → GPTQ + prior: q4tp 0.0463 → 0.0104 → 0.0105; q2tp 0.388 → 0.066 → 0.068 | ok |
| `moegptq::tests::dsv41_layers_write_the_converter_layout` | End to end (see the list below the table) | ok |
| `moegptq::tests::qwen4_rows_route_by_f16_logit_topk` | Flash-Next capture rows are converted to f16 inputs plus the top-k of the f16-rounded logits, the routing the Flash-Next path has always used | ok |
| `moegptq::tests::gpu_encode_tracks_the_host_reference` | Device encode vs host reference, by held-out error, with and without the limit, both codecs. Opt-in: `CMF_MOE_GPU_TEST=1` | skipped on the Mac (for the pod) |

What `identity_hessian_gptq_is_plain_rounding` checks:

- q2tp gate/up are byte-identical to the converter's v2 encoder (placed ladder,
  nearest codes, weight-space debias).
- Every q4tp matrix keeps the converter's row ladders and puts every weight on
  the nearest point of its group's grid, so there is no error feedback.
- The rung differs from the converter in 4 of 1024 groups. The cause is
  existing code, not changed here: the host sweep centres its window of 4 rungs
  on the libm `log2(absmax/7)`, the converter on `det_log2(f16_scale(...))`.

What `dsv41_layers_write_the_converter_layout` covers:

- A tiny V4.1 checkpoint (2 layers, 4 FP4 experts, an FP8 shared expert)
  converted twice by `run_convert_multi`: a q4tp calibration file and a q2tp
  reference.
- The layer encode runs on synthetic captured rows. The source is read through
  `Source` on a local directory, and the local shards are checked to survive
  `release`.
- Three outputs: a new q2tp file, the calibration file rewritten in place as
  q4tp, and the calibration file rewritten in place as q2tp.
- Each output must:
  - pass verify;
  - have the expert dtypes, shapes and nbytes of the converter (q2tp reference,
    or q4tp calibration);
  - keep every non-expert tensor byte for byte;
  - actually change the expert payloads.
- The new file's provenance has `weight_quant` and `gptq_moe`.
- The same test also runs the held-out diagnostic (`CMF_MOE_DIAG`) through
  `encode_layer`, with every 4th block of tokens held out. Relative expert output
  error on the held-out rows, RTN → GPTQ → GPTQ + prior: q4tp 0.0324 → 0.0078 →
  0.0076, q2tp 0.342 → 0.038 → 0.035. GPTQ is required to be no worse than RTN
  in every row bin.

Commands:

```
cargo test -p cortiq-engine --features gpu --lib -- dsv41 elt_wgsl_validates   # 22 + 1 ok
cargo test -p cortiq-cli --bin cortiq moegptq                                    # 6 ok (GPU test skips)
```

## Not verified (pod work for the lead)

1. The device encode with the new parts: the `swiglu_lim`/`swiglu_lim_rows`
   kernels, the q4tp gate/up device sweep in `quantize-moe`, the prior with an
   output scale, the route-weighted rows, and the device debias. It cannot run
   on the M4, which has no matrix units through wgpu. On the pod, run:
   `CMF_MOE_GPU_TEST=1 cargo test --release -p cortiq-cli --bin cortiq gpu_ -- --nocapture --test-threads 1`
   and the train_ops GPU test `rms_and_elementwise_match_host`, which checks
   `swiglu_lim` and `swiglu_lim_rows` against f64:
   `cargo test --release -p cortiq-engine --features gpu --lib rms_and_elementwise_match_host -- --nocapture`.
2. The V4.1 capture on the real model: the full forward has not run since 0.6.7
   (task A fixes it). quantize-moe must run with a binary that has task A's
   forward fixes merged (4dd75a8b act_quant(o) and others), because the captured
   rows are whatever that forward produces.
3. Speed and quality on 1017. Nothing has been measured yet.

## Pod plan (RunPod 103.196.86.153:18691)

The stand: 1× RTX PRO 6000 96 GB, 188 GB RAM cgroup, 13.6 CPU quota, 966 GB
disk (900 GB in round 1).

### Sizes (computed from the q4tp/q2tp section formulas, 2304×5120 experts)

| item | q4tp | q2tp |
|---|---:|---:|
| one expert (gate+up+down) | 18,424,832 B | 12,526,592 B |
| one layer (384 routed + shared) | 7.09 GB | 4.82 GB |
| all 40 layers | 283.7 GB | 192.9 GB |

- The plain q4tp calibration file is 499 GB (the scout's figure). About 216 GB
  of it is non-expert: the two Engram byte tables alone are about 197 GB raw.
- A q2tp file is therefore about 499 − 90.8 = 408 GB.
- 499 + 499 (a second q4tp file) does not fit in 900 GB.
- 499 + 408 = 907 GB does not fit either.
- Hence `--in-place` for both passes in round 1. On the 966 GB disk, round 2
  writes the q2tp file to `--output` (499 + 408 + capture + shards fit; see
  the disk table) and keeps `--in-place` for q4tp.

### RAM

- Captured rows: 40 layers × (5120 f16 + 6 ids + 6 f32 route weights) = 411 KB
  per calibration token. That is 13.5 GB at 32,768 tokens, 26.9 GB at 65,536,
  and 53.9 GB at 131,072. The rows of a pass are held until its last layer is
  encoded; in place, and with `--layers-per-pass 40`, that is the whole run.
- Per layer, the encoded payloads of 385 experts are held until the layer is
  written: about 7.1 GB in q4tp.
- Per worker: f32 weights of one expert (141 MB) plus, for q2tp, the gate/up
  Hessian readback (210 MB f64) and its f32 copy.
- The 499 GB file is mmap'd. Its page cache is reclaimable.
- 65,536 tokens with 8 workers stays far below 188 GB.

### VRAM (encode)

- Per worker: about 1 GB of scratch (16,384-row chunks at hidden 5120), plus
  packed Hessians (5120² and 2304²) and device folds.
- 8 workers come to roughly 10 GB.
- The calibration pipeline is dropped and its cached weights and expert bank are
  released (`release_model_buffers`) before the encode starts.
- During the capture, the shared expert's scratch (about 1 GB) is already
  allocated next to the bank.

### Capture time

Capture is the per-token V4.1 forward over the corpus: one pass, 40 layers, no
head.

- capture time = tokens / forward tok/s, paid once with a capture directory.
- The last recorded V4.1 decode speed on a PRO 6000 is from September and an
  older checkpoint: about 11 tok/s warm, with GPU attention and dynamic MoE.
- Measure the actual rate on the smoke run before choosing `--tokens`.

### Inputs on the pod

- Calibration text: a copy of `calib.mix.txt` (21.9 MB, from the 27B/Flash-Next
  pod) is at
  `/Users/oleg/dev/cmf-lb-local-runs/nvg_research_mac_20261018/inputs/data/calib.mix.txt`.
  The code share is `code.txt` in the same folder.
- Copy both to `/root/data/`.
- Flash-Next's best mix was 0.8 text / 0.2 code. A 0.5/0.5 mix was worse on
  every set.

### Commands (round 2)

The plan:

1. Capture once into a capture directory.
2. Take every later step from that directory: the DIAG A/B, GPTQ q2tp into a
   new file (`--output`, the calibration file stays the plain q4tp), and GPTQ
   q4tp `--in-place` on the plain q4tp file.

The capture does not depend on the prior, the route weighting or the codec.
DIAG holds out every 8th window inside the encode, not inside the capture. So
one forward serves all three steps, and a failed step resumes without a new
forward.

The four runs must use the same `--calib` / `--tokens` / `--window`, or the
saved capture is refused.

```bash
# 0. build on the pod (dsv41-int + lane A's forward fixes + this branch merged)
cd /root/cmf && cargo build --release -p cortiq-cli && cp target/release/cortiq /root/cortiq-moe
CALIB=/root/<plain q4tp 1017 file>.cmf           # single file, 499 GB, local disk (not /workspace)
SRC=deepseek-ai/DeepSeek-V4.1-Flash-1017
CAL="--calib /root/data/calib.mix.txt=0.8 --calib /root/data/code.txt=0.2"
TOK="--tokens 32768 --window 2048"
CAP=/root/moecap                                  # 13.5 GB at 32,768 tokens
export XDG_RUNTIME_DIR=/tmp
mkdir -p /root/logs /root/moescratch $CAP

# 0b. device tests (a few minutes): the device encode against the host
#     (with the route-weighted and the 48/128/192-row cases) and the debias
#     product; then swiglu_lim(_rows) against f64
CMF_MOE_GPU_TEST=1 cargo test --release -p cortiq-cli --bin cortiq gpu_ -- --nocapture --test-threads 1 2>&1 | tee /root/logs/moe-gputest.log
cargo test --release -p cortiq-engine --features gpu --lib rms_and_elementwise_match_host -- --nocapture

# 1. smoke: 2048 tokens, no capture directory (other tokens than the
#    production capture); stops after layer 0 is encoded
CMF_MOE_STOP_AFTER=1 CMF_MOE_WORKERS=8 /root/cortiq-moe quantize-moe $CALIB --hf $SRC \
  --calib /root/data/calib.mix.txt --tokens 2048 --window 2048 --layers-per-pass 40 --codec q2tp \
  --scratch /root/moescratch --threads 13 --output /root/moescratch/smoke.cmf > /root/logs/moe-smoke.log 2>&1
rm -f /root/moescratch/smoke.cmf
#   read: capture tok/s, "routed rows per expert", "prior: n0 = …", layer-0 stage
#   times (gate/up now includes the device debias), fetch wait

# 2a. the production capture: one forward over all 40 layers, saved, nothing encoded
CMF_MOE_CAPTURE_DIR=$CAP CMF_MOE_CAPTURE_ONLY=1 /root/cortiq-moe quantize-moe $CALIB --hf $SRC $CAL $TOK \
  --scratch /root/moescratch --threads 13 --output /dev/null > /root/logs/moe-capture.log 2>&1

# 2b. DIAG A/B of the prior and the route weighting on layers 20..22, from the
#     capture: 3 priors x {unweighted, route-weighted} = 6 encodes per expert
CMF_MOE_CAPTURE_DIR=$CAP CMF_MOE_DIAG=2560,5120,10240 CMF_MOE_ROUTE_W=diag CMF_MOE_RANGE=20..22 CMF_MOE_WORKERS=8 \
  /root/cortiq-moe quantize-moe $CALIB --hf $SRC $CAL $TOK --codec q2tp \
  --scratch /root/moescratch --threads 13 --output /dev/null > /root/logs/moe-diag.log 2>&1
#   read "DIAG total": per row bin, RTN and GPTQ for "2560" "2560+w" "5120" …,
#   plain and route-weighted error. Pick N0 (the best prior) and W (1 if "+w"
#   wins in the route-weighted error, else 0)

# 3. GPTQ q2tp into a new file, from the capture (no forward)
CMF_MOE_CAPTURE_DIR=$CAP CMF_MOE_PRIOR=$N0 CMF_MOE_ROUTE_W=$W CMF_MOE_WORKERS=8 \
  /root/cortiq-moe quantize-moe $CALIB --hf $SRC $CAL $TOK --layers-per-pass 40 --min-rows 32 --lambda 0.01 \
  --codec q2tp --scratch /root/moescratch --threads 13 --output /root/<1017>-q2tp-gptq.cmf > /root/logs/moe-q2tp.log 2>&1
/root/cortiq-moe verify /root/<1017>-q2tp-gptq.cmf
#   -> eval + publish the q2tp file before step 4

# 4. GPTQ q4tp in place on the plain q4tp file, from the same capture
CMF_MOE_CAPTURE_DIR=$CAP CMF_MOE_PRIOR=$N0 CMF_MOE_ROUTE_W=$W CMF_MOE_WORKERS=8 \
  /root/cortiq-moe quantize-moe $CALIB --hf $SRC $CAL $TOK --min-rows 32 --lambda 0.01 \
  --codec q4tp --in-place --scratch /root/moescratch --threads 13 > /root/logs/moe-q4tp.log 2>&1
/root/cortiq-moe verify $CALIB
rm -r $CAP                                       # once both files are evaluated
```

Notes on the commands:

- The logs of steps 2b, 3 and 4 must show `every layer's capture is on disk:
  no calibration forward`.
- A failed step 3 or 4 is rerun with the same command. Step 4 then re-encodes
  from layer 0 with the capture of the pristine file. Add
  `CMF_MOE_RANGE=<first layer not written>..40` to skip the layers already
  rewritten; the log of every finished layer says which ones they are.
- Without `CMF_MOE_PRIOR`, the V4.1 default already applies at 32,768 tokens:
  512 mean rows per expert < 5120, so n0 = 5120.
- Each of steps 3 and 4 streams the 40 source shards from the Hub again,
  about 510 GB of downloads per run. `--hf` can point at a local copy instead,
  but a local copy does not fit on this disk next to the other files.
- `--layers-per-pass 40` in step 3 only matters without a saved capture: it
  keeps the forward to one pass, where 24 per pass would run two.
- 65,536 tokens: the same commands with `--tokens 65536`. The capture is then
  26.9 GB, the mean expert gets 1024 rows (still < 5120, so n0 = 5120), and
  the disk margin shrinks to about 17 GB (table below).

### Disk at each step (966 GB)

| step | on disk, 32,768 tokens | on disk, 65,536 tokens |
|---|---:|---:|
| 2a, capture | 499 + 13.5 = 513 GB | 499 + 26.9 = 526 GB |
| 3, q2tp `--output` | 499 + ~408 + 13.5 + ~15 shards = 936 GB | 499 + ~408 + 26.9 + ~15 = 949 GB |
| 4, q4tp in place | 499 + 408 (the q2tp file, if kept) + 13.5 + ~15 = 936 GB | 949 GB |

Recommendation: 32,768 tokens. That leaves about 30 GB free during steps 3
and 4, against about 17 GB at 65,536. Upstream of this, the DIAG at 32,768
tokens can say whether more rows still pay: compare the rows<640 and
rows<2560 bins.

## Open issues and decisions for the lead

- The q2tp debias `[W_g;W_u]·H` (about 242 GFLOP per expert, about 3.7 PFLOP
  for the model) runs on the device since round 2 (B10). Check the per-layer
  `gate/up` stage time on the smoke run.
- The provenance of an in-place file stays the converter's. Its `weight_quant`
  still says q4tp after the q2tp pass, and the file has no `gptq_moe` record.
  The engine reads only per-tensor dtypes. The HF card has to state the recipe.
- The host q4tp GPTQ sweep and the converter use different rung-window centres
  (libm `log2` vs `det_log2`). This is existing code; the effect is 4 of 1024
  groups under an identity Hessian. Left alone: it is shared by `quantize-gptq`
  and by the device sweep.
- Route-weighted down Hessian: `CMF_MOE_ROUTE_W=1`, default off. Decide it
  from the step-2b DIAG.
- Every V4.1 window starts with BOS, so each window contributes one BOS row
  per layer. These rows are identical across windows: 16 copies at 32,768
  tokens. In inference every request starts with BOS, so they are kept.
- The calibration corpus is plain text without the chat template, as for
  Flash-Next. The image-token VL bias is not exercised (text-only calibration).
