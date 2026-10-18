# DeepSeek-V4.1 wave 1, task A: fixture, loader, encoding, act_quant(o)

Branch `dsv41-fixes` (worktree `cmf-dsv41a`), based on `origin/master` =
cortiq 0.8.23 (`336b0e75`). Mac-only work (Apple M4): CPU builds and tests,
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
transpose, router, index top-k and candidate rule agrees; the one 3e-3
position is a single f32-vs-f64 BF16 flip carried by the FP4 caches.
Tests assert: strict >= 2/3 positions within 1e-5 and worst < 0.15;
default worst < 0.25 and argmax >= 10/12; two runs and pool/no-pool
bit-identical (default and strict); `Pipeline::from_model` parses the same
`Dsv41Cfg` and `dsv41_serial_logits` equals the direct walk bit for bit.

Known differences between cortiq and the official kernels (the reference
can emulate both, `RefOpts`):
- `sparse_attn_kernel` casts the unnormalized probabilities to BF16
  (`acc_s_cast`) before the PV GEMM; cortiq keeps f32.
- `sparse_attn` writes BF16 `o` and the inverse RoPE reads it; cortiq
  rotates the f32 accumulator and rounds once.
Each alone moves the fixture logits by up to ~7.5e-2 (same scale as one
BF16 flip propagating through the FP4 caches), so their effect on the real
model is unknown; candidates for a later parity pass, not changed here.

Not covered by the fixture: the multimodal path (vision tower, image spans,
VL routing bias), DSpark/MTP, prefill batching (V4.1 prefill is token
serial), long contexts (> window 4, index top-k 3 only at toy scale).

The CPU tests run in CI on ubuntu (x86 AVX2) and macOS; the bounds were
measured on ARM only. The x86 strict path regroups f32 sums (AVX2), which
changes where BF16 flips land, not whether the structure agrees.

GPU test: see "GPU" below.

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
