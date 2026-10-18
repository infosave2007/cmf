# DeepSeek-V4.1-Flash-1017: GPTQ for the experts (wave 1, task B)

Branch `dsv41-moe-gptq` (worktree `/Users/oleg/dev/cmf-dsv41b`), based on
origin/master = cortiq 0.8.23. Mac-only work: builds and CPU tests. No pod
access and no model files. The GPU encode path is compiled but has not been run
(see "Not verified").

Round 2 (branch `dsv41-int-b`, worktree `/Users/oleg/dev/cmf-dsv41ib`, from
`dsv41-int`): the review items B1–B12, see "Round 2" below. Round 2 review
fixes on the same branch: see "Round 2 review fixes". The pod commands in
"Pod plan" are the current sequence.

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
  the whole capture. This happens on the M4 through wgpu/Metal, and on any
  card with `CMF_COOP=0` exported (the message then says so). A
  `CMF_MOE_CAPTURE_ONLY=1` run encodes nothing and needs no matrix units.
  Tests call `encode_layer` with `Engine::Host` and are unaffected.
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
    means other tokens, hidden, top_k or experts. Since the review fixes, the
    same tokens captured from another calibration file (canonical path or
    size), by another cortiq version, or with another forward setting are
    refused too unless `CMF_MOE_CAPTURE_FORCE=1` (see "Round 2 review
    fixes"). A resumed `--in-place` run still reuses its capture: in place
    keeps both the path and the length of the file.
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
  layer then fetches the shard synchronously. Since the review fixes it keeps
  retrying for up to `CMF_MOE_FETCH_WAIT` minutes (default 240), with pauses
  of 20 s·k capped at 5 minutes; round 2 gave up after 6 attempts (about 10
  minutes).
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
    bit-identical on the M4, and why 0 of 4608 debiased q2tp rows differ
    there. That count is trivial when the products are identical: it does
    not show how a product difference would reach the payload.
  - What is shown: the device product is accurate (worst error against f64
    9e-8 of Σ|terms|). The effect on the payload is measured on the pod by
    `gpu_debias_product_matches_host` in step 0b: the x86 host GEMM
    (matrixmultiply) rounds differently, and the test asserts that at most
    1 % of the debiased rows change.
  - The device product also serves Flash-Next GPU q2tp runs (the item allows
    it). Their gate/up bytes can therefore differ from round-1 outputs at
    the f16 rounding of the row parameters; they are not byte-identical to
    files made before this change.
  - On the pod, the host GEMM speed was not measured; compare the per-layer
    `gate/up` stage time with round 1.
- B11, tests: see the table below.
- The merged `dsv41-int` lib tests did not compile: `tiny_stack` set
  `candidate_source: 99`, an `Option` since the `-1` fix. Fixed.

### Environment variables

| variable | effect |
|---|---|
| `CMF_MOE_HOST=1` | run the encode on the host; otherwise a wgpu device with matrix units is required (`CMF_COOP=0` turns them off: leave it unset for quantize-moe) |
| `CMF_MOE_PRIOR=n0` | the layer prior (pseudo-rows; down n0/4). V4.1 default: hidden when mean rows per expert < hidden; `0` = none |
| `CMF_MOE_PRIOR_S` | prior sample tokens (8192) |
| `CMF_MOE_ROUTE_W=0\|1\|diag` | route-weighted down Hessian: off (default) / on / both under DIAG |
| `CMF_MOE_CAPTURE_DIR=dir` | save the capture there and reuse it when it matches |
| `CMF_MOE_CAPTURE_ONLY=1` | capture all layers in one forward into the directory, then stop (no matrix units needed) |
| `CMF_MOE_CAPTURE_FORCE=1` | reuse a saved capture of the same tokens although its calibration file, cortiq version or forward settings differ |
| `CMF_MOE_NO_SPACE_CHECK=1` | run although the free-space check finds too little room |
| `CMF_MOE_FETCH_WAIT=min` | how long a shard download is retried before the run gives up (240) |
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

## Round 2 review fixes

Two reviews of round 2. Every change below is one commit on `dsv41-int-b`.

- `--output` or `--patch-shared` naming the calibration file is refused
  before the pipeline loads, also through a symlink or a hard link (device
  and inode). `CmfStreamWriter::new` opens the output with truncation, so
  this used to truncate the mapped 499 GB file.
- Free-space check, before the capture and before the output is opened (the
  output is now created only after the checks). Per filesystem it adds up
  what the run will write and compares with `statvfs` `f_bavail`:
  - `--output`: every tensor of the calibration file, the expert slots at the
    codec's payload size, each padded to 4 KiB, plus the head reserve and
    the vocabulary. An earlier output at the same path counts as free space,
    since opening it truncates it (when it has no other hard link);
  - a Hub source: two layers of source shards (the layer encoded and the one
    prefetched), from the shard sizes the Hub reports, minus what is already
    in the scratch directory;
  - the capture layers still to save.
  The run prints one `free space:` line per filesystem and stops on a
  shortfall unless `CMF_MOE_NO_SPACE_CHECK=1`. Not checked: logs, cargo
  output and the smoke run's leftovers, which the pod commands now remove.
- A shard download is retried for up to `CMF_MOE_FETCH_WAIT` minutes
  (default 240), pausing 20 s·k after the k-th failure, at most 5 minutes.
  Step 3 cannot resume, so a Hub outage longer than the old 10 minutes would
  have lost every layer encoded so far.
- Saved-capture provenance (`meta.json`):
  - new fields: `env` (every `CMF_*` variable of the capturing process,
    `CMF_MOE_*` aside), `cortiq_version`, `binary` (path, size, mtime of the
    executable); `calibration_model` is now the canonical path;
  - a reuse compares the calibration file (path and size), the version and
    the forward settings `CMF_GPU`, `CMF_GPU_PROBE`, `CMF_DSV41_O_FP8`,
    `CMF_DSV41_GPU_ATTN`, `CMF_DSV41_DYNAMIC_MOE`, `CMF_DSV41_FUSED_Q`,
    `CMF_DSV41_TAIL_TAP` (set or unset, and the value). A difference is
    refused unless `CMF_MOE_CAPTURE_FORCE=1`; another binary is only a note.
    Other tokens or geometry are refused even with the override;
  - the output provenance (`gptq_moe.calibration_capture`) carries the
    capture's token hash, settings, version, binary, directory, and the
    differences a forced reuse went past (`reused_despite`).
  - The build's git commit is not recorded: the binary has no build script
    that embeds it, and adding one to `cortiq-cli` for this is out of
    proportion. The binary's path, size and mtime identify the build on the
    pod.
- Saving the capture:
  - `meta.json` is rewritten after every saved layer;
  - a layer already listed is not written again;
  - a failed write removes its `.tmp` and the layer's partial files;
  - outside `CMF_MOE_CAPTURE_ONLY` a save failure (a full disk) is a
    warning, and the run goes on from the rows in memory. A capture-only
    run still fails, since saving is all it does.
- `--in-place` writes its recipe to `<file>.gptq_moe.json`: one record per
  run with the codec, the prior, the route weighting, the layer range and
  the capture. The header cannot grow in place (`recode_entries_in_place`
  rewrites only the directory), so the header still carries the converter's
  `weight_quant`. A resumed run adds its own record.
- The hole-punching probe writes pseudo-random bytes. A constant pattern
  takes almost no blocks on a compressing filesystem (zfs lz4, btrfs), so
  the probe could fail where punching works. The Linux-only code (probe,
  `punch_holes`, the `statvfs` call) was type-checked for
  `x86_64-unknown-linux-gnu` with `rustc --emit=metadata -D warnings`
  against that target's `libc`; injected type errors in the new branches are
  reported.
- The q4tp slot-check message says "a q4tp payload fits only a q4tp slot".

Not done:

- A DIAG variant that weights the gate/up Hessians by w² as well
  (`Σ w_t²·x xᵀ`, the weighting consistent with the MoE output error, as
  the reviewer notes). The device grams read the captured rows unscaled, so
  this needs a row-scaling pass for gate/up and the debias Hessian. It is a
  candidate for step 2b if `+w` wins clearly.
- A checkpoint inside a capture pass. The capture-only forward is one pass
  over all 40 layers; if it dies, the next run starts it again. Splitting it
  with `CMF_MOE_RANGE` does not help much, because a later range still runs
  the forward through every layer below it.
- On the M4, a capture-only run with `CMF_GPU=wgpu` now passes the device
  check, but the V4.1 forward sizes the dynamic MoE pool from an unbounded
  budget (98 GB on a 24 GB Mac, the WAVE1_A open issue) and does not finish.
  Lane A's shared pool code; not changed here.

Tests added or changed:

| test | checks | result |
|---|---|---|
| `moegptq::tests::output_naming_the_calibration_file_is_caught` | same file through a symlink and a hard link; another file and a missing one are not | ok |
| `moegptq::tests::free_space_check_refuses_what_does_not_fit` | fits, shortfall, override, needs on one filesystem add up, a file about to be truncated counts as free unless hard-linked | ok |
| `moegptq::tests::capture_dir_round_trip_and_refusal` | also: the meta records env and version; another binary or a non-forward setting loads; another calibration file, size, version or `CMF_DSV41_O_FP8` is refused and loads with the override; other tokens or hidden are refused even with it; a failed write leaves no file and no `.tmp` | ok |
| `moegptq::tests::downloads_are_retried` | retries within a time budget, growing capped pauses, one attempt with a zero budget | ok |
| `moegptq::tests::dsv41_layers_write_the_converter_layout` | also: the free-space estimate of the new q2tp file is at least its size and at most one 4 KiB pad per tensor plus 4 MiB above it | ok |
| `moegptq_dsv41_e2e::quantize_moe_on_the_v41_fixture` | also: `--output` = the calibration file (directly and through a symlink) refused with the file untouched; provenance names the capture; a capture with `CMF_DSV41_O_FP8=1` refused before the output exists; a read-only capture directory warns and writes the same tensors, capture-only fails and leaves nothing; in place from a capture of another path refused, then forced; the sidecar records the run | ok |

On the M4 with `CMF_GPU=wgpu` (binary run by hand): `--output` stops with
the matrix-units message, with `CMF_COOP=0` it adds "CMF_COOP=0 is set and
turns them off: unset it", and capture-only passes the check and starts the
forward.

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
   forward fixes merged, because the captured rows are whatever that forward
   produces. Task A's `act_quant(o)` (4dd75a8b) is behind
   `CMF_DSV41_O_FP8=1`, off by default; the pod plan captures without it (see
   "Commands"), and the capture records which way it ran.
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

### Commands

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
saved capture is refused. They must also use the same calibration file, the
same binary version and the same forward settings (`CMF_GPU`,
`CMF_DSV41_O_FP8`, …), or the capture is refused unless
`CMF_MOE_CAPTURE_FORCE=1`.

`CMF_DSV41_O_FP8`: step 2a runs without it. The capture then holds the rows
of the forward that cortiq runs by default, the one the quantized files are
served with: GPU attention tail, no `act_quant(o)`. The flag gives the
reference semantics but forces the CPU attention tail, so the capture would
be slower. The choice is recorded in `meta.json` (`env`) and in the output
provenance, and the later steps refuse a capture taken the other way. If the
lead makes `act_quant(o)` the runtime default before the capture, export
`CMF_DSV41_O_FP8=1` once for all four steps.

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
#    the binary is copied: the release target directory (build + two test
#    builds) is not needed on the disk any more
rm -rf /root/cmf/target

# 1. smoke: 2048 tokens, no capture directory (other tokens than the
#    production capture); stops after layer 0 is encoded
CMF_MOE_STOP_AFTER=1 CMF_MOE_WORKERS=8 /root/cortiq-moe quantize-moe $CALIB --hf $SRC \
  --calib /root/data/calib.mix.txt --tokens 2048 --window 2048 --layers-per-pass 40 --codec q2tp \
  --scratch /root/moescratch --threads 13 --output /root/moescratch/smoke.cmf > /root/logs/moe-smoke.log 2>&1
rm -f /root/moescratch/smoke.cmf /root/moescratch/*.safetensors /root/moescratch/*.part
#   read: capture tok/s, "routed rows per expert", "prior: n0 = …", layer-0 stage
#   times (gate/up now includes the device debias), fetch wait, "free space:"
#   (the smoke's check counts the whole q2tp output plus two shards, ~423 GB,
#   though it stops after layer 0; about 450 GB are free at that point. If
#   the check stops the smoke, add CMF_MOE_NO_SPACE_CHECK=1 to the smoke only)

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

# 3. GPTQ q2tp into a new file, from the capture (no forward); the run checks
#    the free space itself (needs ~408 GB output + 2 shards), df for the record
df -B1 /root | tee /root/logs/df-step3.txt
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
cat $CALIB.gptq_moe.json                         # the recipe (the header keeps the converter's); keep the log too
rm -r $CAP                                       # once both files are evaluated
```

Notes on the commands:

- The logs of steps 2b, 3 and 4 must show `every layer's capture is on disk:
  no calibration forward`.
- A failed step 3 or 4 is rerun with the same command. Step 3 cannot
  resume: it starts again at layer 0. Step 4 re-encodes from layer 0 with the
  capture of the pristine file (in place keeps the file's path and length,
  so the capture is accepted). Add `CMF_MOE_RANGE=<first layer not
  written>..40` to skip the layers already rewritten; the log of every
  finished layer says which ones they are. Each finished run adds a record
  with its range to `$CALIB.gptq_moe.json`.
- A Hub outage: a shard download is retried for up to 4 hours
  (`CMF_MOE_FETCH_WAIT`, minutes) before the run gives up.
- Without `CMF_MOE_PRIOR`, the V4.1 default already applies at 32,768 tokens:
  512 mean rows per expert < 5120, so n0 = 5120.
- Each of steps 3 and 4 streams the 40 layer shards from the Hub again
  (model-00003 … model-00042, 7.39 GB each): about 296 GB of downloads per
  run. The checkpoint's 510 GB total also counts the two 101.5 GB Engram
  shards (47, 48), MTP and vision, which quantize-moe never fetches. `--hf`
  can point at a local copy of the layer shards instead, but 296 GB does not
  fit on this disk next to the other files.
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

The table counts the model files, the capture and the shards in flight;
logs, the toolchain and the OS come on top. The commands remove the cargo
target directory after step 0b and the smoke's shards after step 1, which
the margin would not cover. Every run prints its own `free space:` line and
stops before the capture when the filesystem is short.

Recommendation: 32,768 tokens. That leaves about 30 GB free during steps 3
and 4, against about 17 GB at 65,536. Upstream of this, the DIAG at 32,768
tokens can say whether more rows still pay: compare the rows<640 and
rows<2560 bins.

## Open issues and decisions for the lead

- The q2tp debias `[W_g;W_u]·H` (about 242 GFLOP per expert, about 3.7 PFLOP
  for the model) runs on the device since round 2 (B10). Check the per-layer
  `gate/up` stage time on the smoke run.
- The header provenance of an in-place file stays the converter's. Its
  `weight_quant` still says q4tp after the q2tp pass, and the header has no
  `gptq_moe` record; the recipe is in `<file>.gptq_moe.json` next to it.
  The engine reads only per-tensor dtypes. The HF card has to state the
  recipe.
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
