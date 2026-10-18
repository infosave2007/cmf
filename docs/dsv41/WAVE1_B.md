# DeepSeek-V4.1-Flash-1017: GPTQ for the experts (wave 1, task B)

Branch `dsv41-moe-gptq` (worktree `/Users/oleg/dev/cmf-dsv41b`), based on
origin/master = cortiq 0.8.23. Mac-only work: builds and CPU tests. No pod
access and no model files. The GPU encode path is compiled but has not been run
(see "Not verified").

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
- The Hessians are unweighted by route weights, as in Flash-Next. The route
  weight multiplies the expert output, so the per-token error weight would be
  w². Not tried.
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
- `CMF_MOE_HOST=1`, or no wgpu adapter, runs the same steps on the host (f64
  Hessians, host fold, host sweeps). This is the path the tests use. It is exact
  but far too slow for the full model.
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

1. The device encode with the new parts: the `swiglu_lim` kernel, the q4tp
   gate/up device sweep in `quantize-moe`, and the prior with an output scale.
   On the pod, run:
   `CMF_MOE_GPU_TEST=1 cargo test --release -p cortiq-cli --bin cortiq gpu_encode_tracks_the_host_reference -- --nocapture`
   and the train_ops GPU test `rms_and_elementwise_match_host`, which now also checks
   `swiglu_lim` against f64:
   `cargo test --release -p cortiq-engine --features gpu --lib rms_and_elementwise_match_host`.
2. The V4.1 capture on the real model: the full forward has not run since 0.6.7
   (task A fixes it). quantize-moe must run with a binary that has task A's
   forward fixes merged (4dd75a8b act_quant(o) and others), because the captured
   rows are whatever that forward produces.
3. Speed and quality on 1017. Nothing has been measured yet.

## Pod plan (RunPod 103.196.86.153:18691)

The stand: 1× RTX PRO 6000 96 GB, 188 GB RAM cgroup, 13.6 CPU quota, 900 GB
disk.

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
- Hence `--in-place` for both passes.

### RAM

- Captured rows: 40 layers × (5120 f16 + 6 ids) = 410 KB per calibration token.
  That is 13.4 GB at 32,768 tokens, 26.9 GB at 65,536, and 53.7 GB at 131,072.
  The rows are held until the end of the run.
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

- capture time = tokens / forward tok/s.
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

```bash
# 0. build on the pod (branch with task A + task B merged)
cd /root/cmf && cargo build --release -p cortiq-cli && cp target/release/cortiq /root/cortiq-moe
CALIB=/root/<plain q4tp 1017 file>.cmf          # the converter's output, single file, 499 GB
export XDG_RUNTIME_DIR=/tmp
mkdir -p /root/logs /root/moescratch

# 1. smoke: capture of all 40 layers on 2048 tokens, then layer 0 only (CMF_MOE_STOP_AFTER bails before the
#    output is finished; delete the partial smoke.cmf afterwards)
CMF_MOE_STOP_AFTER=1 CMF_MOE_WORKERS=8 /root/cortiq-moe quantize-moe $CALIB \
  --hf deepseek-ai/DeepSeek-V4.1-Flash-1017 --calib /root/data/calib.mix.txt \
  --tokens 2048 --window 2048 --layers-per-pass 40 --codec q4tp \
  --scratch /root/moescratch --threads 13 --output /root/moescratch/smoke.cmf \
  > /root/logs/moe-smoke.log 2>&1
#   read: the capture tok/s, "routed rows per expert", layer-0 stage times, fetch wait

# 2. (optional) pick the layer prior on a few layers: held-out expert error, no file written
CMF_MOE_DIAG=0,2560,5120,10240 CMF_MOE_RANGE=20..22 CMF_MOE_WORKERS=8 /root/cortiq-moe quantize-moe $CALIB \
  --hf deepseek-ai/DeepSeek-V4.1-Flash-1017 --calib /root/data/calib.mix.txt=0.8 --calib /root/data/code.txt=0.2 \
  --tokens 32768 --window 2048 --codec q2tp --scratch /root/moescratch --threads 13 --output /dev/null \
  > /root/logs/moe-diag.log 2>&1

# 3. q4tp pass, in place: the plain q4tp file becomes the GPTQ q4tp file
CMF_MOE_PRIOR=5120 CMF_MOE_WORKERS=8 /root/cortiq-moe quantize-moe $CALIB \
  --hf deepseek-ai/DeepSeek-V4.1-Flash-1017 --calib /root/data/calib.mix.txt=0.8 --calib /root/data/code.txt=0.2 \
  --tokens 65536 --window 2048 --min-rows 32 --lambda 0.01 --codec q4tp --in-place \
  --scratch /root/moescratch --threads 13 > /root/logs/moe-q4tp.log 2>&1
/root/cortiq-moe verify $CALIB
#   -> eval + publish the q4tp file (byte-range parts, bounded staging) before step 4

# 4. q2tp pass, in place on the GPTQ q4tp file (better calibration activations), then compact
CMF_MOE_PRIOR=5120 CMF_MOE_WORKERS=8 /root/cortiq-moe quantize-moe $CALIB \
  --hf deepseek-ai/DeepSeek-V4.1-Flash-1017 --calib /root/data/calib.mix.txt=0.8 --calib /root/data/code.txt=0.2 \
  --tokens 65536 --window 2048 --min-rows 32 --lambda 0.01 --codec q2tp --in-place \
  --scratch /root/moescratch --threads 13 > /root/logs/moe-q2tp.log 2>&1
du -h --apparent-size $CALIB; du -h $CALIB     # 499 GB logical, ~408 GB allocated after hole punching
/root/cortiq-moe compact $CALIB --output /root/<1017>-q2tp-gptq.cmf && /root/cortiq-moe verify /root/<1017>-q2tp-gptq.cmf
rm $CALIB                                       # its q4tp version is already on the Hub
```

Notes on the commands:

- The prior n0 = 5120 is one hidden width of pseudo-rows. It is chosen by
  analogy with Flash-Next, where n0 = 2560 = hidden was the measured best
  (held-out diag, 30–40 % below RTN in every row bin). Step 2 measures it for
  V4.1.
- With 65,536 tokens the mean expert gets 65,536·6/384 = 1024 rows for a
  5120-wide input. Most experts would be rank-deficient without the prior.
- `--hf` can point at a local copy of the source shards instead. They are then
  never deleted.
- The 510 GB source cannot stay on disk next to the 499 GB calibration file
  (1009 GB > 900 GB), so the commands stream it again from the Hub. That is
  40 layer shards of about 7.3 GB each, two in flight (one being encoded, one
  prefetched), about 15 GB of scratch.
- Keep `$CALIB` and `--scratch` on the pod's local disk, not on `/workspace`.
  mmap from that network filesystem (MooseFS) runs at about 1 % CPU and looks
  like a hang.
- `/dev/null` as `--output` in step 2 is never opened: diag writes nothing.

### Disk at each step (900 GB)

| step | on disk |
|---|---|
| 3, q4tp in place | 499 GB calibration + about 15 GB of shards in flight |
| 4, q2tp in place | the same 499 GB file, falling to about 408 GB allocated as the slack is punched, + 15 GB of shards |
| compact | 408 + 408 = 816 GB |

If the filesystem refuses hole punching, the log says so. The slack then stays
allocated, and `compact` needs 499 + 408 = 907 GB, which does not fit. In that
case the q2tp file has to be compacted elsewhere, or the HF upload has to go
from a compacted copy streamed directly.

## Open issues and decisions for the lead

- The q2tp debias computes `[W_g;W_u]·H` on the host. At hidden 5120 that is
  about 242 GFLOP per expert, about 3.7 PFLOP for the whole model, shared by 13
  CPU threads. It is visible in the per-layer `gate/up` stage time. The q4tp
  pass has no debias. If it dominates, the fix is to move `W·H` to the GPU.
- The provenance of an in-place file stays the converter's. Its `weight_quant`
  still says q4tp after the q2tp pass, and the file has no `gptq_moe` record.
  The engine reads only per-tensor dtypes. The HF card has to state the recipe.
- The host q4tp GPTQ sweep and the converter use different rung-window centres
  (libm `log2` vs `det_log2`). This is existing code; the effect is 4 of 1024
  groups under an identity Hessian. Left alone: it is shared by `quantize-gptq`
  and by the device sweep.
- Hessians are not weighted by route weights (an option for later).
- The calibration corpus is plain text without the chat template, as for
  Flash-Next. The image-token VL bias is not exercised (text-only calibration).
