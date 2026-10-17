# Engineering notes

## q2tp: encoder v2 and GPTQ into q2tp

Two changes to the q2tp tooling. The CMF format, the q2tp reader and the
kernels are unchanged: every file decodes as before.

1. **`encode_q2tp` v2** (`cortiq convert --quant q2tp`, requant, videopack,
   the DSpark sidecar — every caller). This is the zero-bit, data-free
   converter fix "Q0pd" from the q2tp study.
2. **`cortiq quantize-gptq --codec q2tp`**: GPTQ error feedback into the q2tp
   codec for the converter's q2tp profile. It ships with calibration
   plumbing that also serves `--codec q4tp`.

### What changed

**`crates/cortiq-cli/src/convert.rs`**

- `encode_q2tp` now picks an encoder. The default is `encode_q2tp_v2`.
  `CMF_Q2TP_ENCODER=v1` selects `encode_q2tp_v1`, the old code kept
  verbatim. `CMF_Q2TP_SEARCH=0` still selects the plain absmax rule.
- v2 encodes each row in three steps:
  1. `q2tp_place_row`: for each group, choose the scale
     `m·absmax/1.5` with `m ∈ {0.300, 0.325, …, 1.050}` that gives the
     lowest rounding error. The ladder spans those scales. Four ladder
     tops are tried as in v1, and the one whose full rung search gives the
     smallest row error wins.
  2. A full search over all 31 rungs, then round-to-nearest.
  3. `q2tp_unshrink_row`, a weight-space debias. The row slope is
     `b = ⟨W,Ŵ⟩/⟨W,W⟩`, clipped to 0.8…1.25. It is removed by shifting the
     row's f16 `lo` by `−log2 b`. Codes, rungs and steps are unchanged.
- `q2_level` replaces `round_ties_even().clamp(0,3)` by a magic-number
  rounding that gives the same value. It vectorizes on baseline x86-64.
  `q2tp_group_err` sums in eight fixed lanes. As a result, the bytes do not
  depend on the machine or the thread count.

**`crates/cortiq-cli/src/gptq.rs`**

- `run_quantize_gptq_tiled` replaces `run_quantize_gptq_q4tp` and handles
  both profiles. Each tensor gets the codec from `effective_quant(profile)`,
  with `--tensor-quant` overrides applied first.
  - q4tp and q2tp tensors that have a Hessian go through GPTQ in their own
    codec. A q4tp override on a q2tp profile is therefore an escort that
    stays GPTQ.
  - All other tensors get converter round-to-nearest.
  - A plain q4tp run writes the same header as before.
- `gptq_quantize_q2tp` has two rung rules (`Q2tpRule`):
  - `cortiq`: the v1 ladder. When the sweep reaches a group, its rung is the
    best of nominal−2…+1, with nominal taken from the corrected absmax/1.5
    and candidates ranked by `H_ii`-weighted error. This is the study's arm
    N1.
  - `placed`: the ladder is placed with `H_ii` weights, and each group gets
    a full 31-rung search when the sweep reaches it. This is the study's arm
    W0 and the default.
  - With `--act-order`, rungs are fixed up front: v1's rungs for the cortiq
    rule, the placed ladder's own rounding choice for the placed rule.
  - The sweep uses a 128-column lazy block. The update of the remaining
    columns is register-tiled (8 rows × 8 columns, with AVX2 dispatch), and
    work is split into 32-row items. Each row is computed by one thread in a
    fixed order, with multiplies and adds kept separate.
- `q2tp_debias_hessian` (`--q2tp-debias on|off`, default `on`) computes
  the row slope `(ŴHWᵀ)/(WHWᵀ)` on the calibration Hessian and removes it
  through `lo`.
  - Measured −11.5 % on the placed arm at zero bits (see below).
  - `off` reproduces the study's plain GPTQ arms (N1, W0). The N1c, W0c,
    N1c7, W0c7, W0ca and W128c rows below were run with it off.
- `--q4tp-debias` (opt-in) applies the same debias to q4tp tensors through
  `q4tp_debias_hessian`. It is off by default, so the q4tp output stays
  byte-identical to master.
- `--q2tp-rule-tensor PATTERN=RULE` sets the rule per tensor. Example: layer
  7 on the cortiq rule, as in the study's arm W128dh.
- Calibration changes:
  - `--calib` can be repeated as `PATH=SHARE`. Shares are rounded
    cumulatively, so they sum exactly to `--tokens`. Accepted inputs:
    `.txt`, `.jsonl` (`{"text":…}` or a string per line), and `.json`.
  - Text is read in budgeted prefixes (about 8 bytes per token, doubled as
    needed, with a 64-token margin), never the whole corpus.
  - A window never spans two sources.
- The capture keeps only the linears that GPTQ will quantize
  (`gptq_capture::begin_filtered`).
- `--calib-chunk-layers N` captures N layers per calibration pass, with one
  cache file per chunk: `PATH.L{a}-{b}`.
- Hessian groups are deduplicated by exact equality and freed as they are
  grouped.
- The cache is indexed (`index_hessian_cache`) and read one group at a time
  (`read_hessian_entry`). Previously every name got its own full copy.
- `gptq_fold_operator_par` runs both Cholesky factorizations, the
  triangular inverse and the `H⁻¹` product on several threads.
  - Each phase is cache-blocked in 32-column or 32-row blocks.
  - It is bit-identical to the original serial operator because every entry
    is the same `dot64` over the same slices, computed by a single worker.
  - Hessian groups are processed widest first, which shortens the tail.
  - `CMF_GPTQ_MEM_GB` limits how many groups are processed at once.

**`crates/cortiq-engine/src/gptq_capture.rs`**: adds `begin_filtered`.

**`crates/cortiq-cli/src/main.rs`**: new flags `--codec q2tp`,
`--calib PATH[=SHARE]` (repeatable), `--calib-chunk-layers`, `--q2tp-rule`,
`--q2tp-rule-tensor`, `--q2tp-debias on|off`, `--q4tp-debias`.

### How to use

```sh
# data-free (v2 is the default; CMF_Q2TP_ENCODER=v1 for the old encoder)
cortiq convert --model HF_DIR --quant q2tp \
  --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f --output q2tp.cmf

# GPTQ (f16 weights in, q8_2f sibling for the calibration forward; ≥ 32k tokens at 2 bits)
cortiq convert --model HF_DIR --quant f16 --output f16.cmf
cortiq convert --model HF_DIR --quant q8_2f --output q8_2f.cmf
cortiq quantize-gptq f16.cmf --codec q2tp --calib-model q8_2f.cmf \
  --calib wiki.train.raw=0.75 --calib c4.jsonl=0.25 --tokens 32768 --window 2048 \
  --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f \
  --q2tp-rule placed --hessians H.hess --calib-chunk-layers 21 \
  --q2tp-rule-tensor 'model.layers.7.mlp.*=cortiq' --output q2tp-gptq.cmf
# (placed rule + debias are the defaults; the layer-7 rule is MiniCPM5's super-weight layer)
# escort: --tensor-quant model.layers.7.mlp.gate_proj.weight=q4tp (stays GPTQ, in q4tp)
# variants from the same cache: --q2tp-rule cortiq | --q2tp-debias off | --act-order
```

Measurement scripts (pod paths of the q2tp study) are in `tools/q2tp_eval/`:
- `gptq.sh`: one run with peak-RSS timing (`timerun.py`)
- `det.sh`: determinism and chunking checks
- `eval.sh`: raw dump and torch ppl with the study's evaluator
- `paired.py`: paired per-window comparison
- `hess2cmf.py`: the study's Hessians as a cortiq cache
- `par.sh` / `compare_q2.py`: payload parity against the torch arms
- `regress_q4tp.sh`: the q4tp byte-identity check

### Tests

`cargo test --release -p cortiq-cli -- q2tp q2_ gptq calib layer_chunks two_bit four_bit parallel requant fold tiled`
(36 tests) and `cargo test --release -p cortiq-engine --lib gptq_capture`.
The full `cargo test --release -p cortiq-cli` suite passes 191 unit tests.
Two integration tests in `tests/decision_cli.rs` fail on this pod
(`decide_oracle_leaves_a_signal_…` and `decide_oracle_releases_its_lock_…`).
They fail the same way on master 5d6f1830 there, so the failure predates
this work.

- v2 encoder:
  - weight error below v1 on heavy-tailed matrices
  - debias brings each row slope back to 1 and leaves codes and rungs alone
  - `CMF_Q2TP_ENCODER` and `CMF_Q2TP_SEARCH` select the right encoder
  - bytes match between 1 and 8 threads
  - the reader validates the payload; dead rows and groups decode to zero
  - `q2_level` gives the same values as `round_ties_even`
- GPTQ-q2tp:
  - output error beats round-to-nearest with both rules, and `placed` beats
    `cortiq`
  - bytes match for 1, 5 and 7 threads, in natural and act order
  - act order keeps the static rungs
  - with an identity Hessian the result equals placed round-to-nearest
  - Hessian debias brings the slope back to 1
- Building blocks:
  - the trailing update matches a reference loop bit for bit
  - the threaded fold operator and Cholesky match the serial versions bit
    for bit
  - cache round trip through the lazy index
  - calibration share parsing and splitting, and budgeted `.txt` / `.jsonl`
    reading
  - filtered capture

The existing q2tp payload, round-trip, requant and parallel-encoding tests
pass unchanged.

### Measured (MiniCPM5-2B, WikiText-2 test 160×512, the q2tp study's torch evaluator)

Setup:
- Every q2tp file uses the q2tp profile: gate/up are q2tp (2.17 bpw), the
  other 210 linears are q4tp, head and embeddings are q8_2f. That is
  1.31 GB, identical bytes per tensor in every row below.
- The evaluator decodes every payload of the real file exactly: raw dump,
  then f32 on the GPU.
- Δ ± SE compares the 160 windows pairwise, using the study's convention
  (the SE is relative). bf16 = 18.889.
- Parity with cortiq itself (CPU `cortiq ppl`, window 0, 511 tokens):
  W0c 1602.959718 against torch 1602.959229 nats, Q0h_v2 2112.054657
  against 2112.054443. The difference is ≤ 5·10⁻⁴ nats per window.

**Item 1 — `convert --quant q2tp`, no data:**

| file | encoder | ppl | vs the study |
|---|---|---:|---|
| Q0h_v1 | `CMF_Q2TP_ENCODER=v1` | (206.566) | all 381 tensor hashes equal the study's Q0h |
| **Q0h_v2** | v2 (default) | **89.464** | Q0pd 89.459: Δ +0.01 % ± 0.00 |

Payload parity against the study's torch Q0pd, over the 84 gate/up
tensors:

| field | equal |
|---|---:|
| row `lo` | 99.991 % |
| step | 99.996 % |
| rungs | 99.997 % |
| 2-bit codes | 99.99999 % |

The weight SSE ratio is 1.0000001. Converting the whole model takes 33 s
with v2 and 27 s with v1 on 13 cores.

**Item 2 — the GPTQ-q2tp kernel against the study's torch GPTQ, on the
study's own 32k FP-stream gate/up Hessians** (converted with `hess2cmf.py`,
84 tensors):

| rule | vs | rungs equal | codes equal | held-out output error ratio | calibration ratio |
|---|---|---:|---:|---:|---:|
| cortiq | N1 | 99.994 % | 99.999 % | 1.0000014 | 0.9999994 |
| placed | W0 | 99.986 % | 99.998 % | 0.9999995 | 0.9999998 |

**q4tp regression.** The reworked runner rewrote the q4tp study's A1 file
from its Hessian cache (`--codec q4tp`, same command, 3 threads instead of
24). The output is byte-identical to `A1_gptq.cmf` (same sha256 a35761ae…),
with a peak RSS of 8.3 GB. The old loader built a dense copy of the 8.5 GB
upper-triangle cache for every name, about 21 GB by arithmetic; that was not
measured.

**Item 2 — end to end with cortiq's own calibration.**

Setup:
- `quantize-gptq f16.cmf --codec q2tp --calib-model q8_2f.cmf --calib
  wiki.train.raw --tokens 32768 --window 2048`, head and embeddings q8_2f.
  The script is `tools/q2tp_eval/gptq.sh`.
- Every 4-bit tensor is GPTQ'd by cortiq on the same Hessians.
- The study's arms used a bf16 torch forward for their Hessians. Their
  4-bit tensors were C1 (N1) or C2 = C1 + refit (W0).

| file | recipe | ppl | Δ vs study arm |
|---|---|---:|---|
| N1c | cortiq rule, debias off | **37.727** | N1 37.082: +1.74 % ± 0.79 |
| W0c | placed, debias off | **31.344** | W0 31.672: −1.04 % ± 0.47 |
| N1c7 | cortiq rule + layer-7 gate/up q4tp (+0.025 bpw) | **36.451** | N1_7 35.810: +1.79 % ± 0.69 |
| W0c7 | placed + layer-7 gate/up q4tp (+0.025 bpw) | **29.874** | W0_7 30.155: −0.93 % ± 0.46 |
| **W0cd** | **placed + debias (the CLI defaults)** | **27.756** | W0: −12.36 % ± 0.67 |
| W0cdh | W0cd + layer 7 on the cortiq rule (equal size) | **27.753** | W128dh 25.622: +8.32 % ± 0.51 (W128dh adds refit, alternation and 131k tokens) |
| W0cd7 | W0cd + layer-7 gate/up q4tp (+0.025 bpw) | **27.485** | W7d 25.899: +6.12 % ± 0.46 (W7d adds refit and alternation) |
| W0ca | placed + `--act-order`, debias off | 42.220 | W0c: +34.70 % ± 1.20 (static rungs; the runner now warns) |

Paired differences:
- N1c vs Q0h_v2: −57.8 % ± 1.4.
- W0c vs N1c: −16.92 % ± 1.25 (the study's W0_7 vs N1_7 was −15.8 %).
- N1c7 vs N1c: −3.38 % ± 0.38 (the study: −3.43 % ± 0.38).
- W0c7 vs W0c: −4.69 % ± 0.48 (the study's W0_7 vs W0: −4.8 %).
- W0c7 vs N1c7: −18.04 % ± 1.10.
- **W0cd vs W0c: −11.45 % ± 0.50** at zero bits. That beats the layer-7
  escort's +0.025 bpw: W0cd vs W0c7 is −7.09 % ± 0.39. On this evidence
  `--q2tp-debias` became the default.
- W0cdh vs W0cd: −0.01 % ± 0.22. Once the debias is on, putting layer 7 on
  the cortiq rule adds nothing. In the study the rule mattered only because
  the refit clipped layer 7's super-weights.
- W0cd7 vs W0cd: −0.98 % ± 0.17. With the debias on, the layer-7 escort is
  worth only 1 %. The remaining ~6–8 % gap to the study's best arms (W7d,
  W128dh) is its H-metric refit and code/scale alternation, which were not
  ported (see open issues).
- At equal size (1.314 GB), the best cortiq-native q2tp is W0cd at 27.76:
  - −25.2 % ± 1.2 vs the study's 32k GPTQ arm N1;
  - −26.4 % ± 1.2 vs N1c;
  - −69.0 % ± 1.5 vs the data-free v2 converter;
  - −86.6 % ± 2.0 vs the plain converter (Q0h, 206.6);
  - +46.9 % above bf16.

Cost on the 5090 pod (13.6-CPU quota, shared with other jobs):
- The first run calibrates in two 21-layer passes: 2 × ~42 min of CPU
  forward at 16 windows × 2048 tokens. Peak RSS is 17.1 GB, of which
  ≤ 10.6 GB is Hessians per pass. The whole run took 104 min.
- Every later arm reads the cache: 14–20 min of GPTQ, peak RSS 10.4 GB.

**Bit-exactness at full scale.** These pairs of files are byte-identical:
- N1c from the first build vs N1c from the threaded-fold, LPT-order build.
- W0c from that build vs W0c from the cache-blocked fold build.

**Determinism** (`tools/q2tp_eval/det.sh`: 4096 tokens, placed rule,
down_proj left to the converter to keep it short):

| run | how | output |
|---|---|---|
| detA | calibrate in one pass, 13 threads | sha256 5c862b8e… |
| detC | calibrate again from scratch, one pass | **identical file**; Hessian cache identical to detA's (a1096e4c…) |
| detD | GPTQ from detA's cache with 5 threads | **identical file** |
| detB | calibrate in two 21-layer passes | **all 381 tensor hashes identical**; the header records `calib_chunk_layers` |
| detM | `--calib wiki.train.raw=0.75 --calib docs.txt=0.25` | runs; 3072 + 1024 tokens in 3 windows (no window spans two sources), recorded per file in the header |

Peak RSS at 4096 tokens is 8.9–10.8 GB, against 10.4–17.1 GB at 32k tokens.
RAM depends on the captured linears and the chunk size, not on the token
budget.

**q4tp from the same 32k cache, for the Qwen3.8-27B target** (head and
embeddings q8_2f, 1.578 GB):

| file | recipe | ppl | Δ |
|---|---|---:|---|
| Q4c | `--codec q4tp` (cortiq's own 32k calibration) | **19.619** | q4tp study C1 19.622: −0.01 % ± 0.15; shipped A1 recipe (2048 tokens) 20.226: −3.00 % ± 0.21; bf16 +3.87 % |
| Q4cd | + `--q4tp-debias` | **19.620** | Q4c: +0.01 % ± 0.01 (no effect) |

Notes on these two rows:
- The q4tp study's evaluator rounded q8_2f to f16, so its rows carry a
  small evaluator offset.
- For Qwen3.8-27B q4tp, the lever is the calibration budget: 32k tokens
  gave −3.0 % over the shipped 2048 tokens here (20.23 → 19.62, bf16
  18.89), which closes 45 % of that recipe's gap to bf16. The debias adds
  nothing at 4 bits, because the q4tp cortiq rule's absmax scales are
  already unbiased. It stays opt-in.

QWEN06_PLACEHOLDER

### Open issues

- **The calibration forward is CPU-only, and this is the bottleneck for
  Qwen3.8-27B.** The Hessian hook lives in the CPU `QTensor::matmat`.
  - MiniCPM5-2B (q8_2f model, 13 threads): about 155 s per 2048-token
    window, roughly 13 tok/s. 32k tokens take about 40 min per pass; 131k
    tokens take about 2.8 h.
  - At 27B the same path costs about 13× more per token.
  - The tools below are ready for that scale; the forward is not. The next
    step is GPU-side capture: accumulate `XᵀX` for the four distinct
    inputs of each layer from the resident graph (f32 tiles on the device,
    f64 reduction on the host).
- **Hessian RAM** is dense f64 per tensor (8·cols² bytes). q/k/v and
  gate/up are captured separately even though their inputs are identical
  (the cache deduplicates them afterwards).
  - Qwen3.8-27B: about 3.6 GB per layer, so on 128 GB use
    `--calib-chunk-layers 8–16`, at one forward per pass.
  - Two improvements would bring this down to one pass: deduplicating
    identical inputs during capture, and storing a packed upper triangle in
    f32.
- **Fold operator.** In flight, one group holds 3n² f64 + n² f32 (8 GB at
  n = 17408). It is O(n³), about 3.5 T multiply-adds per 27B `down_proj`.
  - The threaded version and `CMF_GPTQ_MEM_GB` handle this.
  - On MiniCPM5 the GPTQ phase is a few minutes, so folding is not the
    bottleneck there.
- **Not ported from the q2tp study:**
  - the H-metric refit and the code/scale alternation (with layer 7 fixed:
    −12 % on the cortiq line, −9 % on the placed line; they need held-out
    Hessians for selection);
  - the automatic super-weight guard (the study chose layer 7 with a
    model-level sensitivity pass);
  - end-to-end metric genesis (R1/R2, 25.6 → 21.9).

  `--q2tp-rule-tensor` and q4tp escorts handle the layer-7 case by hand.
- **Encoder cost.** The v2 encoder evaluates 155 group errors per group
  against 24 in v1. On MiniCPM5 (1.06 G q2tp weights, 13 threads) a
  `convert` takes 33 s instead of 27 s. A bracketed rung search around the
  placed scale would remove most of that cost, but it would need a parity
  check against the full search.
- **Act order with q2tp** uses static rungs, so there is no rung choice at
  sweep time. Measured +35 % ppl against natural order (W0ca vs W0c), so
  the runner warns. Act order would need sweep-time rungs, which a permuted
  sweep cannot pick group by group.
