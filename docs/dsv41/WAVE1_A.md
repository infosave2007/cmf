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

GPU test: see part 1 GPU section below.
