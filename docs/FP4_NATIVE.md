# Native FP4 in CMF (`mxfp4`, `--quant fp4`, NVFP4 sources)

Journal of the `fp4-native` branch (worktree `cmf-fp4`, base `origin/master` =
cortiq 0.8.23). What was built, how it behaves, what was tested where, and the
commands still to run on a CUDA/Vulkan pod.

## 1. Format: `mxfp4`, dtype 21

OCP Microscaling MXFP4 (OCP MX spec v1.0): FP4 **E2M1** elements (1 sign, 2
exponent, 1 mantissa bit, bias 1 → magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`)
with one **E8M0** scale byte per 32 consecutive elements of a row (scale
`2^(e−127)`, `e = 255` is NaN). 4.25 bits/weight.

```
[codes : rows · cols/2 bytes]   byte j of row r = elements 2j (LOW nibble) and 2j+1 (high)
[scales: rows · cols/32 bytes]  byte g of row r = exponent of elements 32g .. 32g+32
```

- Two planes (like `q1g`), 2-D only, `cols % 32 == 0`. A group's 16 code bytes
  are 16-byte aligned; the scale plane starts at a 16-byte boundary.
- Code bit 3 is the sign, bits 2..0 index the magnitude table. The nibble order
  is the OCP / DeepSeek-V4 order (`convert.py`'s `FP4_TABLE`, low nibble first),
  so a source checkpoint's FP4 `[rows, cols/2]` bytes and its E8M0
  `[rows, cols/32]` scales **are** the two planes.
- Dequantization is exact: `E2M1(code) · 2^(e−127)` is a ≤2-bit mantissa times
  a power of two; the scale is built from exponent bits, never `exp2`. Only a
  scale byte ≥ 253 with a large code leaves f32's range (±inf); a NaN scale
  yields a NaN group. Code 8 decodes to −0.0 (bit-identical to the converter's
  historical `unpack_mxfp4`).
- OCP quantizer for float sources: shared exponent `floor(log2(amax)) − 2`
  (emax of E2M1), clamped to `[−127, 127]`; elements divided by `2^e` (exact),
  rounded to nearest-even on the E2M1 grid, saturated to ±6. All-zero group →
  scale byte 0; a group holding NaN/±inf → scale 255 (OCP); zero is code 0.
- Lossless re-pack of a source MXFP4 tensor = concatenation of its two planes
  (`cortiq_core::mxfp4::repack`); a source NaN scale is refused.
- Property worth knowing (tested): OCP re-quantization of any decoded MXFP4
  group returns the same values (the exponent can drop by up to 3; the codes
  ×2/×4/×8 stay on the grid). A float detour therefore cannot change an MXFP4
  weight.

Code: `crates/cortiq-core/src/mxfp4.rs` (layout, LUTs, exact decode, OCP
encoder, repack); `quant.rs` (`expected_nbytes`, `validate_payload`,
`dequant_tensor`, `bytes_per_weight`); `types.rs` (`TensorDtype::Mxfp4 = 21`,
serde name `mxfp4`). Spec: `docs/CMF_V2_SPEC.md` §3.1/§3.2. A reader older than
this branch refuses the file at open (`UnknownDtype(21)`).

## 2. Converter

| flag | effect |
|---|---|
| `--quant fp4` | every routed and shared expert matrix (gate/up/down, `.experts.`, `.shared_expert(s).`, `w1/w2/w3`) → `mxfp4`; every other tensor exactly what `--quant q4tp` gives it (skeleton q8_2f rules, f16/f32 force rules, vision f16) |
| `--quant mxfp4` | every quantizable matrix → `mxfp4` |
| `--tensor-quant PATTERN=mxfp4` | per-tensor codec (any profile); `fp4` is refused as a per-tensor choice |
| `--exclude-tensors PATTERNS` | leave tensors out of the file (any profile, every output); see below |

### `--exclude-tensors` (one-file 1017 upload)

Comma-separated globs over the **canonical output** tensor names (repeatable;
`*` is the only wildcard and spans dots — the `--tensor-quant` glob).

- A source tensor whose canonical name matches is skipped before its payload
  is read: never decoded, requantized or written — in every output
  (`--also-output` included) and on every source path of the walk (resident
  local shards, `CMF_LOCAL_READY_DIR` streaming, Hugging Face streaming,
  `--resume`). A name only a split produces (one expert of a packed tensor,
  q/k/v of a fused projection) is checked again at the writer.
- The run prints `--exclude-tensors: N source tensors excluded, B source
  bytes (… GB; 'pattern' ×n, …)` and warns about a pattern that matched
  nothing (patterns are canonical CMF names: `model.mtp.*`, not `mtp.*`).
- The patterns are recorded in `header.provenance.excluded_tensors`.
  `scripts/check_dsv41_cmf.py` reads them: excluded names are no longer
  expected, must be absent, and `--dspark-policy include` refuses such a file.
- Packers outside the walk refuse the flag: `--quant blr`, `--mtp-sidecar`,
  `--dspark-sidecar`, MiMo `--mimo-towers mm-only`, EmbeddingGemma 2.
- `arch.mtp` stays as the source declares it (the header carries the source
  config verbatim anyway); the loader decides by presence, as before.

Why: Hugging Face refuses a file over 500 GB (500e9 bytes). Measured from the
real 1017 tensor list, the `--quant fp4` payload is **504.51 GB** with the
MTP/DSpark tensors (`model.mtp.*`, 7.4 GB in q4tp) and **496.96 GB** without.
The cortiq V4.1 runtime never reads `model.mtp.*`: `dsv41::load` has no MTP
stage, and the generic MTP loader keys on `model.mtp.layers.0.*` /
`model.mtp.eh_proj.weight`, which V4.1's `model.mtp.0.*` names never match
(tested: the forward is bit-identical with and without the stage).

Source handling:

- **DeepSeek-V4 / V4.1 FP4 experts** (`I8 [rows, cols/2]` + `F8_E8M0 .scale
  [rows, cols/32]`, the 1017 checkpoint's routed experts): when an output
  resolves the tensor to `mxfp4`, the payload is the two source planes
  concatenated — **byte-identical, no dequant/requant** (`dsv4_raw_mxfp4` in
  `convert.rs`). In a multi-output pass (`--also-output q4tp=…`) the other
  outputs still decode the tensor to f32 and encode their own codec; the f32
  decode is skipped when every active output took the raw planes. Not taken
  under `--defrag` (pruning needs the floats).
- **V4.1 shared expert** (`F8_E4M3` with 32×32 E8M0 block scales in the 1017
  checkpoint): decoded to f32 as before and OCP-quantized to `mxfp4`, so the
  device expert bank has a single codec. Error ≤ one E2M1 step of the group
  scale (tested). `--tensor-quant '*.shared_expert.*=q4tp'` keeps it q4tp, at
  the price of the dynamic GPU MoE declining that file (one codec per bank).
- **Other OCP MXFP4 sources** (Kimi-K3 `weight_packed`, MiMo-V2 `U8` experts)
  keep their f32 decoders; under `mxfp4`/`fp4` the OCP re-encode is
  value-exact (see §1), though the bytes may be normalized (max code ∈ {4, 6}).
- **NVIDIA ModelOpt NVFP4** (detected from `hf_quant_config.json`
  `quantization.quant_algo` or `config.json` `quantization_config.quant_algo`
  containing `NVFP4`, `group_size` must be 16): `weight` U8 `[rows, cols/2]`
  (low nibble first), `weight_scale` F8_E4M3 `[rows, cols/16]` (linear on
  disk), `weight_scale_2` F32 scalar. Decoded as vLLM's `dequantize_to_dtype`:
  `w = e2m1 · (f32(e4m3) · weight_scale_2)`; an AWQ `pre_quant_scale` (if
  present) multiplies the columns. `weight_scale*`, `input_scale`, `k_scale`,
  `v_scale`, `pre_quant_scale` never reach the output. NaN block scales and a
  non-finite global scale are refused. compressed-tensors
  `nvfp4-pack-quantized` (different names, inverse global scale) is refused
  explicitly instead of being misread as MXFP4. Such a checkpoint converts to
  any profile (`q4tp`, `q2tp`, `mxfp4`, `fp4`, `f16`, …).
- `hf_quant_config.json` is fetched best-effort for hub conversions.

## 3. Engine

### CPU (`crates/cortiq-engine/src/qtensor/mxfp4.rs`)

`QTensor::Mapped` reads the planes straight from the mmap (no f32 copy). Two
contracts, as for every 4-bit codec:

- **exact** (`CMF_SDOT=0`, float-activation scopes, `with_float_activations`,
  no dot ISA): each weight dequantized exactly and summed `sum += w·x[j]` over
  `j` in order — the reference F32 matvec's order (`pool::matvec_rows`), so an
  mxfp4 matrix computes **bit for bit** what its dequantized f32 matrix
  computes. 4 rows per block (independent chains).
- **A8W8** (default on ARM dotprod / x86 AVX2): shared int8 activation split
  (`split_act`, exact outliers); a code becomes the exact int8 `2·E2M1 ∈
  {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}` by one `tbl` (NEON) / `pshufb` (AVX2);
  one 32-wide int dot per group (`sdot` / `maddubs`), scale `2^(k−128)`. SIMD,
  scalar and decoded-row variants are bit-identical.
- `matvec`, `matvec2`, `matmat` (b > 1: the row's codes decoded once per batch;
  each token equals its matvec bit for bit), `row_f32` (embeddings).
  Multi-threaded through the shared `dispatch_rows` pool split.
- Fused multi-tensor jobs (`matvec_many`, `matvec_silu_mul`, `moe_*_many/rows`)
  run mxfp4 per tensor (`per_tensor_only`, alongside `blr`). Device-graph
  accessors (`graph_weight`, `q1_parts`, `mapped_q4tp`, …) answer None, so
  whole-token graphs decline to the per-op path.

### wgpu (`crates/cortiq-engine/src/gpu_wgpu/mxfp4.rs`)

- `mxfp4_mv` / `mxfp4_mm`: one 64-lane workgroup per output row, groups strided
  over lanes, exact E2M1 bit construction (`0x3F000000 + mag<<22`, 0 and 0.5
  patched), exact E8M0 scale, workgroup tree reduction; `mxfp4_mm` carries ≤ 8
  tokens per dispatch (z = token chunk) and equals `mxfp4_mv` per token bit for
  bit. 2-D grid for > 65535 rows (lm_head). Entry `gpu::mxfp4_matvec` /
  `gpu::mxfp4_matmat`; `QTensor` takes them behind the usual probe (matvec ≥
  8.4 M weights, batch b ≥ 2 and ≥ 33.5 M MACs, never for row-exact batches).
- **DeepSeek-V4.1 dynamic expert pool**: the descriptor-indexed segmented bank
  (`dsv4_moe_frame`) learned an mxfp4 bank — `MoeBankCodec::{Q4tp,
  Q2tpGateUp, Mxfp4}`; `Dsv4GlobalMoeBufs.fp4` selects the kernels
  `dsv4_global_gate_up_mxfp4` / `dsv4_global_down_mxfp4` (same bindings,
  route, SwiGLU limit and BF16 boundaries as the q4tp twins; slot word base
  `local · mat16/2`, mxfp4 slot lengths must be word-aligned). Slots hold the
  tensor bytes verbatim (`dsv4_global_slot_fill` unchanged).
  `QwenGpuPool::create_for_dsv41_codec`, `dsv4_global_moe_create_for_dsv41_codec`;
  `dsv41::dynamic_moe_gpu` accepts an all-mxfp4 layer (routed + shared) and
  refuses a pool of another codec. Other bank consumers (the qwen4 device
  frame, the chained MoE encoder, DSpark) refuse an mxfp4 bank, and every
  expert upload (`dsv4_global_slot_fill`, the per-layer slot fill,
  `moe_expert_bufs`) now checks the tensor's dtype as well as its length — an
  mxfp4 matrix can be exactly as long as a q4tp one (e.g. 352 columns), and
  its bytes must never reach a q4tp kernel. So a DeepSeek-V4 (non-.1) or any
  other MoE file with mxfp4 experts runs them on the CPU/per-op path.
- Both shader modules are separate and built lazily; a driver that rejects
  them leaves every other codec untouched and mxfp4 on the CPU.

### Metal

No native Metal mxfp4 kernel: `gpu::mxfp4_*` answer false on the Metal backend
and the CPU kernels run (A8W8 on Apple Silicon). wgpu-over-Metal
(`CMF_GPU=wgpu`) runs the WGSL kernels, which is how they were tested here.

## 4. Tests (MacBook M4, 24 GB; `CARGO_TARGET_DIR` in the worktree)

| command | result |
|---|---|
| `cargo test -p cortiq-core` | all pass (9 mxfp4 tests: tables vs OCP definition, all codes × all scales exact, repack = source planes + NaN refusal, RNE/saturation, OCP exponent rule incl. zero/NaN/inf/clamps, error bound, grid round trip, value-exact re-quantization) |
| `cargo test -p cortiq-engine --features gpu --lib qtensor::mxfp4` | 3 pass: exact matvec/matvec2/matmat == `matvec_rows` on the dequantized matrix bit for bit (± pool); A8W8 SIMD == scalar == decoded-row, == f64 emulation of the A8W8 contract, matmat == matvec per token; mapped CMF tensor == dequantized F32 twin through every QTensor entry point |
| `CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu --lib gpu_wgpu::mxfp4 -- --test-threads=1` | 5 pass on wgpu-over-Metal: WGSL validates (naga, S8 + S16); device matvec within 4e-6·Σ|w·x| of f64 on 300×2304, 70000×64 (2-D grid), 5×5120; batch b ∈ {2, 8, 9, 19} == per-token matvec bit for bit; **global bank frame** (mxfp4 bank, forced + preweighted + BF16, shared slot, one cold winner) matches the host experts at 128×64 (nrmse 1.9–2.5e-3, bound 0.02) and at the **release geometry** gate/up 2304×5120, down 5120×2304 (nrmse 2.1–3.0e-3, cos ≥ 0.9999955, bound 0.01). The fixture's E8M0 scales vary per group, so a wrong scale index fails (see below) |
| `cargo test -p cortiq-cli --bin cortiq fp4_tests` | 11 pass (the GPU one skips without `CMF_GPU`), incl. the end-to-end synthetic V4.1 checkpoint and `--exclude-tensors` (below) |
| `CMF_GPU=wgpu cargo test -p cortiq-cli --bin cortiq fp4_tests -- --test-threads=1 --nocapture` | 11 pass; `dsv41_fp4_dynamic_gpu_moe_matches_the_cpu_forward` runs: 14 of 14 MoE calls on the mxfp4 bank, 9 slot fills, logits nrmse vs CPU **6.37e-3** (bound 0.02) |
| `cargo test -p cortiq-cli --bin cortiq` | 226 pass, 1 ignored (whole bin) |
| `cargo test -p cortiq-engine --features gpu --lib` | 590 pass, 34 ignored (whole lib, 16 min debug); after the upload dtype guards re-run with filters `mxfp4`, `moe`, `dsv4`, `qwen4`, `global`: all pass |
| `cargo check -p cortiq-engine` (no `gpu`) | builds |
| `python3 -I scripts/check_dsv41_cmf.py --self-test` | PASS (mxfp4 sizes) |

End-to-end (`convert_fp4_tests.rs::dsv41_fp4_profile_is_byte_identical_and_forward_matches_f32_experts`):
a 2-layer V4.1 checkpoint written by the test (BF16 skeleton, I8+E8M0 experts,
layer 0 shared expert FP4, layer 1 shared expert FP8) converted in ONE pass to
`fp4` and `q4tp`: all 27 FP4 source matrices land byte-identical; the FP8 shared
expert is mxfp4 within its OCP bound; every non-expert tensor is byte-identical
to the q4tp file; the CPU forward (`Pipeline::forward_ids`, 5 tokens) of the fp4
file equals **bit for bit** the forward of the same file with its mxfp4 tensors
dequantized to f32 (exact contract), and differs from a zero-expert twin; on
the default A8W8 kernels it stays within 25 % of the logit span. NVFP4: a tiny
ModelOpt Qwen3 (all linears NVFP4, AWQ pre-quant scale on down_proj, k/v/input
scalars) converts in one pass to f16 (== f16 of the formula, bit for bit),
q4tp, q2tp and mxfp4, with no companion tensor in any output.

`--exclude-tensors` (`convert_fp4_tests.rs`):
- `dsv41_exclude_mtp_drops_the_stack_and_keeps_the_forward`: the synthetic
  V4.1 checkpoint gains an MTP/DSpark stage in the 1017 spelling (`mtp.0.*`:
  attention, `main_proj`, mHC head, router, FP4 routed + FP8 shared experts,
  Markov and confidence heads; 46 source tensors, config
  `num_nextn_predict_layers: 1`). Without the flag the fp4 file holds its 30
  `model.mtp.*` tensors; with `model.mtp.*` one pass writes fp4 and q4tp
  (`--also-output`) with none, both verify, both record `excluded_tensors`,
  every other tensor is byte-identical, and the V4.1 `Pipeline` forward
  (6 tokens, CPU, exact contract) is bit-identical with and without the
  stage. A NaN scale planted in an MTP expert fails the plain conversion
  ("NaN scale") and passes under the flag — the tensor is never decoded.
- `exclude_tensors_holds_for_split_produced_names`: `up_proj` of a fused
  `gate_up_proj` (dropped at the writer) and `lm_head.weight` (at the source).
- `exclude_tensors_parses_and_counts`: parser (commas, duplicates, empty
  patterns refused) and the tensor/byte counters.
- `scripts/check_dsv41_cmf.py --self-test` covers the glob and the provenance
  read; a synthetic V4.1 fp4 file converted with the flag passes the checker
  (21 source-derived tensors excluded, none present), the HEAD checker failed
  it ("missing 21 source-derived tensors").

GPU (`gpu_wgpu/mxfp4.rs`, wgpu-over-Metal, Apple M4):
- The payload fixture's scale was `119 + ((i*5 + seed) % 5)` — constant per
  tensor, so a kernel reading another group's scale passed. Now
  `119 + ((i*7 + i/3 + seed) % 5)` (adjacent groups always differ). A
  temporary mutation reading the neighbour group's scale (`g ^ 1`) in
  `mxfp4_mv`, `dsv4_global_gate_up_mxfp4` and `dsv4_global_down_mxfp4`:
  old fixture — all 4 tests pass (blind); new fixture — device matvec fails
  (300×2304 row 0: −0.7812 vs 0.4772, tol 1.2e-4) and the bank frame fails
  (nrmse 1.37, cos 0.068); the down-kernel mutation alone: nrmse 0.858,
  cos 0.657.
- The release-geometry bank case is the only one where the lane loops
  iterate (160 groups per gate/up row; the down walk over `slot·72 + g`
  hands one lane groups of different slots). A temporary mutation of both
  lane strides (64 → 128) passes the 128×64 case and fails the release case
  (nrmse 0.850, cos 0.884).
- `dsv41_fp4_dynamic_gpu_moe_matches_the_cpu_forward` (cortiq-cli, needs
  `CMF_GPU`): the synthetic fp4 file through `Pipeline::forward_ids` with the
  dynamic GPU MoE vs the same forward under `gpu::pause_gpu`. It pins the bank
  to 64 slots (`CMF_DSV41_EXPERT_SLOTS` = workspace carve-out / slot + 64) and
  reads `Dsv41State::gpu_moe_stats()` (bank calls, codec label — counted
  without `CMF_DSV41_PROF`).
- Trap found on the way: wgpu-over-Metal reports an "unlimited" weight budget
  (UMA → `u64::MAX`), so an unpinned default V4.1 pool there asks for 75 % of
  that and maps the binding-range maximum, 15.8 M slots (~103 GB) on the
  synthetic model. The CPU-contract test now pauses the device for its
  forwards; unpaused under `CMF_GPU=wgpu` it took 213 s and drove the 24 GB
  Mac into swap. Real-model runs on a Mac with `CMF_GPU=wgpu` should pin
  `CMF_DSV41_EXPERT_SLOTS` (or `CMF_GPU_VRAM_MB`).

Measured CPU speed: see §5. Vulkan behaviour (NVIDIA) is still to be verified
on a pod — §6.

## 5. Measured numbers

`cargo test -p cortiq-engine --release --lib qtensor::mxfp4::tests::cpu_matvec_speed -- --ignored --nocapture`
(M4, 10 threads, best of 7×20; the stand is shared, so ±10 % is noise):

Three runs, µs per matvec (warm: the same 6 MB matrix every call), A8W8
contract (the default on M4):

| shape (V4.1 expert) | mxfp4 A8W8, 10 thr | q4tp A8W8, 10 thr | mxfp4 exact, 10 thr | mxfp4, 1 thr | q4tp, 1 thr |
|---|---:|---:|---:|---:|---:|
| 2304×5120 (gate/up) | 103 – 113 | 117 – 156 | 408 – 456 | 423 – 439 | 500 – 531 |
| 5120×2304 (down) | 90 – 105 | 107 – 117 | 415 – 474 | 412 – 427 | 523 – 529 |

mxfp4 reads 2.1 % more bytes than q4tp and is still 10–20 % faster on the
host: its scale is one byte → one exponent, where q4tp expands a per-row
ladder; the code → int8 step is one instruction for both (`tbl` vs a
subtract). The exact
contract costs ~4× (sequential reduction chains — it exists for parity, not
speed). GPU (wgpu/Vulkan) timings belong to the pod run (§6); on wgpu-over-
Metal only correctness was checked.

## 6. Pod commands (lead)

Build: `cargo build --release -p cortiq-cli` on the pod (Vulkan, `--features
gpu` is the CLI default).

Vulkan kernel tests (NVIDIA; the engine's GPU tests set `CMF_GPU` themselves;
the cortiq-cli end-to-end GPU test needs it from the environment):

```bash
cargo test -p cortiq-engine --features gpu --release --lib gpu_wgpu::mxfp4 -- --test-threads=1 --nocapture
cargo test -p cortiq-engine --features gpu --release --lib qtensor::mxfp4 -- --nocapture
cargo test -p cortiq-engine --features gpu --release --lib qtensor::mxfp4::tests::cpu_matvec_speed -- --ignored --nocapture
CMF_GPU=1 cargo test -p cortiq-cli --release --bin cortiq fp4_tests -- --test-threads=1 --nocapture
# expect "dsv41 fp4 dynamic GPU MoE: 14 bank calls, … logits nrmse vs CPU …" (Metal: 6.37e-3)
```

1017 conversion with FP4 experts, **as one Hugging Face file** (source
streamed one shard at a time; put the output on local disk or `/dev/shm`,
never on a network FS). The MTP/DSpark stack must go — 504.51 GB with it,
496.96 GB without, the limit is 500 GB per file:

```bash
cortiq convert --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --quant fp4 \
  --exclude-tensors "model.mtp.*" --output /dev/shm/dsv41-1017-fp4.cmf
# expect: "--exclude-tensors: N source tensors excluded, B source bytes (… GB; 'model.mtp.*' ×N)"
# check before uploading: stat -c %s /dev/shm/dsv41-1017-fp4.cmf  (must be < 500000000000)

# both files from one streamed pass (the exclusion applies to both):
cortiq convert --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --quant fp4 \
  --exclude-tensors "model.mtp.*" --output /dev/shm/dsv41-1017-fp4.cmf \
  --also-output q4tp=/dev/shm/dsv41-1017-q4tp.cmf
```

Expected: routed experts `mxfp4` (40 layers × 384 × 3 matrices of 2304×5120 /
5120×2304 at 6 266 880 bytes each), shared experts `mxfp4` (OCP from FP8),
skeleton as in the q4tp file. Directory/coverage gate (the checker now knows
dtype 21; it reads headers and the directory only):

```bash
python3 scripts/check_dsv41_cmf.py --model /dev/shm/dsv41-1017-fp4.cmf \
  --source-index /root/dsv41/hf/model.safetensors.index.json \
  --source-config /root/dsv41/hf/config.json
# with --exclude-tensors it also prints
# "excluded by the file's provenance (--exclude-tensors model.mtp.*): N source-derived tensors; none present"
```

Byte identity of one expert against its source shard (layer 3 lives in
`model-00006-of-00048.safetensors`): the CMF payload of
`model.layers.3.mlp.experts.0.gate_proj.weight` must equal the bytes of
`layers.3.ffn.experts.0.w1.weight` followed by those of
`layers.3.ffn.experts.0.w1.scale`.

Runtime on the pod (V4.1 dynamic GPU MoE must take the mxfp4 bank, cold
experts on the CPU mxfp4 kernels):

```bash
CMF_GPU=1 CMF_DSV41_PROF=1 cortiq run --model /dev/shm/dsv41-1017-fp4.cmf \
  --prompt "The capital of France is" --max-tokens 64
# look for: "dynamic-pool capacity=… gu=mxfp4" and "DSV4 unified global pool: … gate/up mxfp4"
CMF_GPU=1 cortiq bench --model /dev/shm/dsv41-1017-fp4.cmf --ignore-eos
```

Open for the pod: decode tok/s of the mxfp4 bank vs the q4tp bank on the same
card (the mxfp4 slot is 2.1 % larger than q4tp's: 6 266 880 vs 6 137 856 B for
2304×5120), and the CUDA/Vulkan parity of `global_bank_frame_runs_mxfp4_experts`.

## 7. CHANGELOG draft (for the lead to merge)

```markdown
## Unreleased

### Added
- **Native FP4: `mxfp4` dtype (id 21).** OCP MX FP4 — E2M1 codes plus one
  E8M0 scale per 32 elements (4.25 bits/weight), two planes. Exact dequant,
  OCP-rule quantizer, lossless re-pack of OCP MXFP4 source planes.
- **`cortiq convert --quant fp4`.** MoE experts (routed and shared) as
  `mxfp4`, the rest as the q4tp profile. DeepSeek-V4/V4.1 FP4 experts
  (I8 + E8M0 1×32) are copied byte for byte — no dequant/requant; an FP8
  shared expert is OCP-quantized so the expert bank has one codec. Also
  `--quant mxfp4` and `--tensor-quant PATTERN=mxfp4`.
- **`cortiq convert --exclude-tensors PATTERNS`.** Comma-separated globs
  over canonical tensor names; matching tensors are never decoded or
  written, in every output; counts printed, patterns recorded in the
  provenance (`excluded_tensors`). `--exclude-tensors "model.mtp.*"` puts
  DeepSeek-V4.1-Flash-1017 `--quant fp4` in one Hugging Face file
  (496.96 GB instead of 504.51 GB); the V4.1 runtime never reads the MTP
  stack.
- **NVIDIA ModelOpt NVFP4 sources.** `hf_quant_config.json` /
  `quantization_config` `quant_algo: NVFP4` checkpoints decode
  (`e2m1 · e4m3 · weight_scale_2`, AWQ pre-quant scale folded) and convert to
  any profile.
- **mxfp4 kernels.** CPU matvec/matmat (exact contract bit-identical to the
  dequantized f32 matrix; A8W8 via tbl/pshufb + sdot/maddubs), wgpu
  matvec/batched matvec, and the DeepSeek-V4.1 dynamic GPU expert pool on an
  mxfp4 bank. Metal falls back to the CPU kernels.
```
