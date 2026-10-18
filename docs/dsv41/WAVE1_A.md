# DeepSeek-V4.1 wave 1, task A: fixture, loader, encoding, act_quant(o)

Branch `dsv41-fixes` (worktree `cmf-dsv41a`), based on `origin/master` =
cortiq 0.8.23 (`336b0e75`); integration pass on `dsv41-int-a` (see "Lane A
integration pass" below). Mac-only work (Apple M4): CPU builds and tests,
no pod, no model files. Reference: `deepseek-ai/DeepSeek-V4.1-Flash-1017`
revision `1bae0e10` (MIT). The 1017 release changes weights only; config,
`model.py`, index and safetensors headers are identical to V4.1-Flash.
Upstream commits after cortiq 0.6.7 that matter (V4.1-Flash repo):

| commit | what | status here |
|---|---|---|
| `8b3cec70` | indexer K cache ownership on incomplete compression steps | cortiq already matches (each owner keeps its own K stream) |
| `4dd75a8b` | `act_quant(o)` (FP8 E4M3, block 32, UE8M0) before the grouped `wo_a`, also in DSpark | part 4: CPU, behind `CMF_DSV41_O_FP8` |
| `314113f8` | encoding: tool image order, current-turn thinking after tool results | part 3 |

`kernel.py` is unchanged since `8b3cec70`.

## Part 1: synthetic V4.1 regression fixture

Files: `crates/cortiq-engine/tests/common/dsv41_fixture.rs` (fixture
builder + independent f64 reference), `tests/dsv41_fixture.rs` (CPU),
`tests/dsv41_fixture_gpu.rs` (WGPU, `--features gpu`, skips without an
adapter). The config parse moved out of the loader into
`Dsv41Cfg::from_source(source, arch) -> (cfg, n_layers)` with no behaviour
change, so tests parse exactly what `Pipeline::from_model` parses;
`Dsv41Cfg` derives `PartialEq`. `qtensor::float_activations_scope` is now
`pub` (doc-hidden) so one test process can run the strict and the default
CPU kernels.

Fixture (written as a CMF by the test, ~0.3 MB, 12 tokens):

| layer | ratio | role | experts gate/up |
|---|---|---|---|
| 0 | 0 | window only | q4tp |
| 1 | 2 | KV source + index source (owns index K) + Engram | q2tp |
| 2 | 1 | second KV source + index source + candidate source | q4tp |
| 3 | 1 | index consumer of L2's keys, uses candidates | q2tp |
| 4 | 0 | window only after the sources; F32 attention linears | q4tp |

dim 64, 2 heads x 64, rope 16, q/o LoRA 32, 2 o-groups, hc 4, 4 experts
top-2, window 4, index 2 heads x 32 top-3, candidate blocks 2 x 2, Engram
max n-gram 4 x 2 heads x 32, 72 rows, release compressed vocabulary 99092
(layer 1 then uses the published hash multipliers, re-derived with NumPy
1.26 for this note: `76632096046245, 4839876093313, 35959672319349,
73987337458391`). Down projections are q4tp; attention linears q4tp except
L4. The residual-writing projections (`wo_b`, `w2`) have gain 0.25: at gain
1 the random network is chaotic and one BF16 flip moves logits by 10-70 %.

The reference (`reference_logits`) is model.py in BF16-checkpoint mode,
decoded token by token, written from `model.py`/`kernel.py`/`engram.py`
without calling cortiq code: f64 accumulation, one BF16 rounding per
materialized tensor, FP8/FP4 quantizers re-implemented from the tilelang
kernels (`fast_log2_ceil` scales, E4M3/E2M1 nearest-even, the 6*2^-9 and
6*2^-126 amax floors), YaRN from `precompute_freqs_cis`, interleaved rope,
Sinkhorn as in `hc_split_sinkhorn_kernel`, `select_candidate_blocks`, the
`NgramHashState` hash. Shared with cortiq only: the q4tp/q2tp dequantizer of
`cortiq-core` (the codec definition) to obtain the same weights.

Measured on the M4 (aarch64):

| comparison | per-position relative logit error | argmax |
|---|---|---|
| strict CPU vs reference | 11/12 positions 4e-8..1.2e-7, one 3.0e-3 | 12/12 |
| default CPU (A8W8) vs reference | worst 7.5e-2 | 12/12 |
| strict CPU vs official kernel semantics | worst 7.9e-2 | |
| strict with one BF16 boundary removed (mutation check) | 0/12 exact, worst 5.0e-2 (test fails) | |

The exact positions mean every BF16 boundary, quantizer, rope pair, hc
transpose, router, index top-k and candidate rule agrees. The one 3e-3
position is position index 4: a local BF16 flip in L4's hc_pre/input-norm
output, absorbed by L4's FP8 window KV (positions 5..11 are exact again),
not carried by the FP4 caches.
Tests assert (`assert_strict_agreement`): on aarch64, where this was
measured, strict >= 10/12 positions within 1e-5 and worst < 2e-2,
candidates-off and O_FP8 12/12; on other targets the median position within
1e-5 and worst < 0.15 (see below). Default worst < 0.25 and argmax >= 10/12;
two runs and pool/no-pool bit-identical (default and strict);
`Pipeline::from_model` parses the same `Dsv41Cfg` and `dsv41_serial_logits`
equals the direct walk bit for bit.

Known differences between cortiq and the official kernels (the reference
can emulate both, `RefOpts`):
- `sparse_attn_kernel` casts the unnormalized probabilities to BF16
  (`acc_s_cast`) before the PV GEMM; cortiq keeps f32.
- `sparse_attn` writes BF16 `o` and the inverse RoPE reads it; cortiq
  rotates the f32 accumulator and rounds once.
Each alone moves the fixture logits by up to ~7.5e-2 (the scale of one
BF16 flip at an unlucky position: the A8W8 CPU run and the device run both
show 7.5e-2 at position 8), so their effect on the real model is unknown;
candidates for a later parity pass, not changed here.

Not covered by the fixture: the multimodal path (vision tower, image spans,
VL routing bias), DSpark/MTP, prefill batching (V4.1 prefill is token
serial), long contexts (> window 4, index top-k 3 only at toy scale).

The CPU tests run in CI on ubuntu (x86 AVX2) and macOS (Apple Silicon); the
bounds were measured on ARM only. The macOS runner executes the same NEON
f32 kernels as the M4 (the only CPU-dispatched aarch64 kernels are the int8
dotprod/i8mm ones, which the strict path does not use), so the measured
counts are asserted there. The x86 strict path regroups f32 sums (AVX2),
which changes where BF16 flips land, not whether the structure agrees; it
could not be measured here (Rosetta on the M4 reports no AVX/AVX2, so an
x86_64 build would run the scalar kernels, not CI's). Two other summation
orders on this fixture, the A8W8 CPU path and the device kernels, each put a
7.5e-2 flip on position 8, so on x86 the strict tests assert the median
per-position error (< 1e-5: a structural regression moves every position)
and keep worst < 0.15. The pod run of the CPU binaries on x86 is the place
to tighten that.

The CPU binaries set `CMF_GPU=0` before any engine call: `cargo test
--workspace` unifies `cortiq-engine/gpu` (the CLI enables it by default),
and on a host with an adapter the V4.1 attention tail and MoE would move to
the device under the strict assertions. Checked by running the CPU binaries
in a `--features gpu` build on the M4 (same numbers).

### GPU (wgpu) fixture run on the M4 (Metal through wgpu)

`CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu --test
dsv41_fixture_gpu -- --nocapture` (on macOS the GPU binaries skip unless
`CMF_GPU=wgpu` is exported; elsewhere they select wgpu themselves). The host
parts run exact (`float_activations_scope`), so the error is the device
kernels'. Two variants in one process: the fixture as built (q4tp gate/up
on L0/L2/L4, so a q4tp bank) and `Fixture::build_q2_even` (q2tp on
L0/L2/L4, so the q2tp bank runs on the device). The test asserts the
`CMF_DSV41_PROF` counters through `dsv41::profile_counters()`: 48 device
attention tails + 12 fallbacks (L4, whose F32 attention linears the q4tp
frame does not take) and device MoE calls where the adapter can hold the
segmented bank (`dsv4_global_moe_supported()`). M4: 48/12, MoE 36 device /
24 host for both variants.

Per-position relative logit error, with two broken device paths for scale
(`CMF_DSV41_TEST_MUTATE=sink` hands the frame zero sinks, `=rope` shifts
its inverse RoPE by one position; both must fail the test):

| run (M4) | median | worst |
|---|---|---|
| q4tp bank, clean | 1.04e-2 | 7.5e-2 |
| q2tp bank, clean | 6.1e-3 | 8.2e-3 |
| O_FP8, q4tp bank, against the act_quant(o) reference | 8.6e-3 | 7.3e-2 |
| zero sinks, q4tp / q2tp / O_FP8 | 4.9e-2 / 3.8e-2 / 4.7e-2 | 9.6e-2 / 8.2e-2 / 1.0e-1 |
| inverse RoPE off by one, q4tp / q2tp / O_FP8 | 5.2e-2 / 4.2e-2 / 5.1e-2 | 1.45e-1 / 1.42e-1 / 1.3e-1 |

The worst position does not separate clean from broken (the old bound,
worst < 0.25, passed both mutations); the median does, by 3.6x or more.
Bounds now: median < 2.5e-2 and worst < 0.12. Other cards round
differently; the median has 2.4x headroom over the M4's clean q4tp run.

Findings for the pod run:
- The dynamic expert bank is created once per state with the first MoE
  layer's gate/up layout (`QwenGpuPool::create_for_dsv41(.., gu_q2)` in
  `dynamic_moe_gpu`). Layers with the other layout (here the q2tp layers
  L1/L3 after the q4tp L0) are refused and run on the host. Correct, but a
  checkpoint mixing q4tp and q2tp gate/up across layers gets only one of
  them on the card. Both layouts are now exercised on the device by the
  two fixture variants.
- On an adapter without a VRAM budget (`device_vram_budget() = u64::MAX`,
  Apple unified memory through wgpu) the bank is sized from that value:
  `resident_bytes` 96 GiB on paper, 18 s of device waits per token
  (`gpu_moe_wait_ms` 215 674 over 60 calls) and still zero device MoE calls.
  `CMF_GPU_VRAM_MB=4096` gives a 1 GiB bank and 36 device calls; 2048 leaves
  no bank (workspace reserve). The GPU test therefore defaults
  `CMF_GPU_VRAM_MB=4096` when unset. The sizing itself is shared Qwen/V4.1
  pool code and was not changed here.

## Part 2: loader hardening

- `text_config.scoring_func` (alias `score_func`, the inference/ModelArgs
  spelling) is read; anything but `"sqrtsoftplus"` is refused with an error
  naming the key and the supported rule. Absent = sqrtsoftplus (release and
  reference default). The CPU router and the preweighted GPU bank implement
  only sqrt-softplus, so a softmax/sigmoid config used to be routed with the
  wrong rule silently.
- `candidate_source_layer_id` (alias `candidate_source_layer`) is read as a
  signed integer; negative = pre-filtering off (`Dsv41Cfg.candidate_source:
  Option<usize>`). Before, `-1` failed `as_u64()` and fell back to 20, so
  every layer after 20 would have masked its index scores with candidates the
  config had disabled. An absent key still means the release layer 20, like
  every other release-shaped default of this parser (the reference default
  is -1; no published config omits the key). A non-integer is refused.
- Router softplus is `exp(x).ln_1p()` (torch `log1p(exp(x))`, linear past
  20): `(1 + e^x).ln()` returned an exact zero score below x ~ -17.
- Router normalization follows `Gate.forward`: divide by `sum + 1e-20` only
  when `norm_topk_prob && top_k > 1` (it was `max(1e-20)` and also for
  top-1).
- Config fixtures: `tests/fixtures/dsv41/config_1017.json` (HF layout) and
  `inference_config_1017.json` (flat ModelArgs keys), byte copies from the
  1017 repo (sha256 `8be45ce0...e21879` for config.json). Both parse to the
  same explicit release `Dsv41Cfg` with an arch whose every fallback is
  wrong; the test passed against the 0.8.23 parser before the changes
  above (candidate source was then the bare `20`).
- The fixture has an end-to-end variant with `candidate_source_layer_id: -1`:
  strict CPU matches the no-candidate reference at 12/12 positions, while the
  candidate run differs from it by up to 0.29 (so the switch is observable).

Not changed: `dsv4::route` (V4 base) still uses `(1 + e^x).ln()`; V4 is out
of scope for this wave. `topk_method` is not validated (only `noaux_tc`
exists for V4.1).

## Part 3: encoding fix 314113f8 ported

`dsv41_encoding.rs` now follows `encoding.py` after `314113f8`:
- `encode_messages` validates image placeholders on the raw context and
  messages first, merges tool messages, sorts tool results by call order
  over context + current turn as one list (the Python dicts are shared, so
  the context is sorted too) and slices the current turn off by the merged
  context length, and only then collects images. Images inside tool results
  therefore come out in prompt order. `encode_messages_text` no longer merges
  or sorts (it receives preprocessed messages, as `_encode_messages_text`).
- `merge_tool_messages`: a tool message with `content_blocks` (and non-string
  `content`) keeps its blocks as the tool_result content; a user message
  whose `content` is a list uses it as `content_blocks` directly.
- `last_user_index` (`find_last_user_index`) skips user/mid-system messages
  whose non-empty `content_blocks` hold no text/image block, i.e. merged
  tool results: the reasoning of the assistant turn that called the tools is
  kept and the generation header after the tool results opens `<think>`.
- `load_cases` ports the reference loader (single case, bare message list,
  case list; case-level `tools` move onto the first message);
  `encode_case` treats an empty `thinking_mode` as absent (`or`).
- Unsupported block placeholders print `None` for a missing type, as the
  Python f-string does.

Tests: the five official goldens (`encoding/tests/test_input_N.json` ->
`test_output_N.txt`, copied to `tests/fixtures/dsv41/encoding/`, kept
byte-exact by a `-text` gitattribute) encode byte-exact through
`load_cases` + `encode_case(case, chat)`; ports of the reference tests
`test_v41_tool_result_images_follow_prompt_order` (4 input formats),
`test_v41_drop_thinking_without_tools`, placeholder rejection for user and
tool, `merge_tool_messages`, task token. Checked: the three behavioural
tests fail against the pre-port encoder; the goldens already passed there
(none of them has a tool result after a thinking turn or tool images).

Remaining divergence (unchanged): a non-string, non-list tool result is
rendered with JSON (`to_json`) where Python's `str.format` prints a Python
repr (`None`, `True`, `{'a': 1}`); OpenAI tool messages carry strings or
block lists, so this was left alone. The V4.1-Flash repo also gained a chat
template (`2cba9e42`, 2026-10-01) which cortiq does not use (it encodes with
this port).

## Part 4: act_quant(o) before the grouped wo_a (CMF_DSV41_O_FP8)

Reference `4dd75a8b`: `o.view(groups, -1)`, then `act_quant(o, 32, "ue8m0",
float8_e8m0fnu, inplace=True)`, then the BF16 `einsum` with `wo_a`. cortiq:
`CMF_DSV41_O_FP8=1` (default off) applies `quantize_attention_output` on
the CPU attention tail after the inverse RoPE and BF16 boundary, before
`o_project`. The group width (release 64*512/8 = 4096, fixture 64) is a
multiple of 32, so the per-group 32-blocks are contiguous runs of the flat
output. The device attention tail applies the same quantizer with its own
kernel (below), so the flag means reference semantics on every backend
without leaving the device frame.

`fp8_activation_inplace` (also the window-KV quantizer) now computes the
scale exactly as the kernel: `amax * f32(1/448)` and the bit-level
`fast_log2_ceil`/`fast_pow2` instead of `(amax/448).log2().ceil()`. For BF16
inputs (everything the runtime feeds it) the two are identical, since amax/448
cannot land within an f32 rounding of a power of two unless it is one; the
fixture logits did not move. The old form picks the smaller scale for an
amax one f32 ulp above 448*2^k, which the new unit test shows (a value at
1.5 subnormal quanta rounds differently).

Tests: `fp8_activation_matches_kernel_act_quant` compares bit for bit with an
independent f64 implementation of the kernel (2000 random BF16 blocks over
2^-30..2^9, all-zero block with the 1e-4 floor, signed zeros, exact 448*2^k
maxima, the f32-ulp-above case, subnormal grid, nearest-even ties 17->16,
19->20, 3.5 quanta->4); `attention_output_quantizes_per_32_block`; and a
separate binary `tests/dsv41_fixture_o_fp8.rs` (the switch is read once per
process): strict CPU with the flag matches the reference with `act_quant(o)`
at 12/12 positions (worst 1.2e-7) and differs from the reference without it
by 8.0e-2.

GPU path (lane A integration): the DSV4 attention frame
(`gpu_wgpu::dsv4_attn_frame`) runs `encode_sparse_attend2` ->
`encode_rope_heads(inverse)` -> `encode_bf16_round(attn)` (cache key 40) ->
`fp8_act_quant32` (new, cache key 47, only when `Dsv4AttnGeom::o_fp8`) ->
`o_lora_a`. The V4.1 caller sets `o_fp8` from `CMF_DSV41_O_FP8`; DSV4 and
the layer frames keep it false, so their encoders are unchanged. The kernel
(`gpu_wgpu/fp8_act.rs`, lazily built module) gives one invocation per
32-value block and works on the f32 bits: the block amax is an integer max
of the magnitudes floored at `1e-4f32`, `t = amax * f32(1/448)` is the one
float operation (a correctly rounded multiply of a normal number),
`e = fast_log2_ceil(t)`, `x / 2^e` is an exponent subtraction, E4M3FN
rounding is nearest-even on the mantissa (3 kept bits for normals, the 2^-9
grid below 2^-6, the carry into the exponent), clamped at 448, `q * 2^e` an
exponent addition, then the BF16 round (exact, kept). It follows the host
sign rule: an exact zero becomes +0, a nonzero value that rounds to zero
keeps its sign unless the host's f32 quotient itself underflows to zero. No
step depends on denormal or fast-math behaviour. A device that rejects the
module declines the frame before encoding, and the host tail quantizes.

Tests: `dsv41::tests::fp8_act_quant32_device_matches_host_bits` (runs where a
wgpu adapter is selected; `CMF_GPU=wgpu` on macOS) compares device and host
bits on the reference set above plus 2000 full-mantissa f32 blocks over the
whole finite range (f32 subnormals included) and the underflow corner of the
sign rule: 0 of 4008 blocks differ on the M4; three kernel mutations (ties
away in the normal branch, `>=` in the subnormal tie, no underflow zero) are
each caught. `tests/dsv41_fixture_gpu_o_fp8.rs` runs the fixture with the
flag on the device: 48 device tails (the frame is no longer skipped), median
8.6e-3 / worst 7.3e-2 against the act_quant(o) reference. A tap at position
5 (`CMF_DSV41_TAIL_TAP=attn CMF_DSV41_TRACE=1 CMF_DSV41_TRACE_FULL=1
CMF_DSV41_TRACE_POS=5`) shows the device frame's attended output on the
E4M3 grid (128/128 values with at most 3 mantissa bits, against 6-9 without
the flag) and equal to the host's quantized output in all 128 values of
L0..L3. The upstream DSpark attention has the same `act_quant(o)`; cortiq's
V4.1 runtime has no DSpark head, so nothing to change there.

Cost. Device tail: one extra compute pass per layer in the frame's existing
encoder (no submit, no readback); for the release output (32768 values,
1024 blocks = 16 workgroups of 64) its arithmetic is negligible against the
grouped `wo_a` that follows, so the cost is the per-pass overhead, a few
microseconds per layer (an estimate; not measured at release size).
Host tail: `fp8_activation_inplace` costs 0.62 ms per 32768-value layer
output on one M4 core (18.9 ns per value, measured with the function copied
into an `-O` microbenchmark), about 25 ms per token over 40 layers, single
threaded; read `attention_other` in the `CMF_DSV41_PROF` report.

## Lane A integration pass (branch `dsv41-int-a`)

- act_quant(o) on the device tail (Part 4 above); `CMF_DSV41_O_FP8=1` no
  longer forces the CPU tail. Still off by default.
- `dsv4_global_moe_create_for_dsv41` picks S16 on its own when the request,
  after the workspace carve-out, exceeds what eight segment buffers address
  (`per_segment` = binding range / max(gu_len, d_len)) and the adapter is
  S16-capable, building the S16 pipelines on first need. The decision is the
  one `dsv4_global_moe_create_slots` already made, now shared
  (`dsv4_global_moe_pick_segments`). `CMF_DSV41_GLOBAL_SEGMENTS=8` forces S8,
  `=16` keeps the eager opt-in. One `tracing` info line per process names
  the family and capacity. Release geometry on a 2 GiB binding range: S8
  addresses 2792 slots; a 3900-slot request gets 3888 under S16. On the M4
  the fixture bank stays S8 (budget-limited); `=16` gives identical logits.
- GPU fixture: device-path counters asserted, both bank layouts, bound from
  a mutation check (GPU section above); `dsv41::profile_counters()`.
- `Dsv41Cfg::from_source`: a null candidate key is skipped (the other
  spelling or the release default applies), an integral float (`20.0`) is
  an integer, and the error names the key it read.
- Fixture containers are written to a temp directory that is removed as
  soon as the model is open (`Fixture::open`).
- Strict CPU bounds: measured counts on aarch64, median per-position on
  other targets (Part 1).
- `cargo test --workspace` on CI: the CPU binaries pin `CMF_GPU=0`. The GPU
  binaries (`dsv41_fixture_gpu`, `dsv41_fixture_gpu_o_fp8`) skip on the
  ubuntu runner (no adapter) and, on macOS, run only with `CMF_GPU=wgpu`
  exported, so the CI's macOS VM (paravirtual Metal, not a reference device:
  the repo's Metal exactness tests are kept off it for the same reason)
  skips them; the M4 and the pod run them. The fp8 device unit test follows
  the lib's convention (runs wherever wgpu is selected; it is bit-exact
  integer arithmetic, so it does not depend on the device's rounding).
- The dsv41-int merge had left the lib tests uncompiled (`tiny_stack` still
  built `candidate_source: 99`); now `Some(99)`, same behaviour.

## Open issues for the lead

- Pod: `cargo test -p cortiq-engine --features gpu --test dsv41_fixture_gpu
  --test dsv41_fixture_gpu_o_fp8 -- --nocapture` on the target card. The
  tests assert 48 device attention tails and device MoE calls themselves;
  read the printed medians against the M4's (bound 2.5e-2). Also the CPU
  binaries on x86, `--test dsv41_fixture --test dsv41_fixture_o_fp8`, and
  the per-position errors they print: if x86 is as exact as ARM, the
  aarch64 bounds of `assert_strict_agreement` can apply there too.
- `CMF_DSV41_O_FP8` is off by default; its effect on the released weights
  (perplexity / answers, with the 1017 checkpoint) is unmeasured. Measure
  `O_FP8=1` against `O_FP8=0` with the SAME attention path in both arms:
  the device tail (default; both arms now keep it, so the comparison is the
  quantizer alone, at a few microseconds per layer), and, for the CPU-tail
  check, `CMF_DSV41_GPU_ATTN=0` in both arms (there the flag also costs
  ~0.6 ms per layer, ~25 ms per token on an M4-class core, so compare
  tokens/s only within one tail). Before this pass `O_FP8=1` silently moved
  attention to the CPU, so an `O_FP8=1` vs default run mixed the quantizer
  with the tail change.
- Two journaled BF16 boundaries in sparse attention differ from the
  official kernel (probabilities before PV, `o` before the inverse RoPE);
  candidates for a parity pass on the real model, not changed.
- Dynamic MoE bank: one gate/up layout per state; budget-less adapters size
  it from u64::MAX (both above). Shared pool code, not changed.
- Engram hash multipliers for layer ids other than 1/14 or a compressed
  vocabulary other than 99092 fall back to SplitMix in cortiq, while the
  reference derives them from NumPy PCG64 (`default_rng(10007 * layer)`).
  Only matters for a non-release checkpoint; the fixture uses layer 1 and
  99092 for that reason.
