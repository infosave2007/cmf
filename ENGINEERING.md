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
  - The number of groups processed at once is capped by the RAM left after
    the chunk's Hessians and the tensors already written (85 % of the
    machine's RAM, the cgroup limit in a container); `CMF_GPTQ_MEM_GB`
    overrides it. A calibration pass whose Hessians exceed 90 % of RAM is
    refused up front with a `--calib-chunk-layers` value that fits.

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

**Long, mixed budget on a second model: Qwen3-0.6B.**

Setup:
- `quantize-gptq --codec q2tp --tokens 131072 --calib wiki.train.raw=0.75
  --calib docs.txt=0.25`, defaults (placed rule, debias on), one pass.
- 64 windows of 2048 tokens. The calibration forward took 56 min on the
  shared pod and GPTQ 80 s; peak RSS was 6.4 GB.
- cortiq CPU ppl on WikiText-2 test, the first 10 windows of the 40-window
  512-token placement, the same windows for every file.
- Head and embeddings stay at the converter's q4tp (tied embeddings).

| file | bytes | ppl (10 × 512) |
|---|---:|---:|
| f16 | 1 521 935 294 | 24.98 |
| `convert --quant q2tp`, `CMF_Q2TP_ENCODER=v1` | 367 817 662 | 2201.1 |
| `convert --quant q2tp` (v2) | 367 817 662 | 416.1 |
| GPTQ-q2tp, 131k tokens | 360 821 694 | **39.25** |

The v2 encoder improves the converter 5.3× here as well. GPTQ with the
defaults brings 2-bit gate/up to +57 % over f16 at a quarter of the size.
The 7 MB difference between the files is container padding: the converter's
streaming writer pads more than `CmfModel::write`.

### Independent verification

A second agent re-checked this branch from a fresh copy of the worktree on
the 5090 pod. It built its own binary and treated the study directories as
read-only.

- **Tests.** The 36 targeted tests pass, and the full `cortiq-cli` suite
  passes 193 tests with 1 ignored. With the fixes below the counts are 37 and
  194. The engine's `gptq_capture` and `fcd` tests pass.
- **Converter v2.** `convert --quant q2tp` with head and embeddings q8_2f
  produced a file byte-identical to `Q0h_v2.cmf` (sha256 83d36e66…, torch
  ppl 89.464). It took 30 s.
- **GPTQ-q2tp with the CLI defaults** from the 32k cache produced a file
  byte-identical to the evaluated `W0cd.cmf` (sha256 81f35e49…, ppl 27.756).
  The cache-blocked fold and the row-blocked debias are therefore
  bit-identical at full scale. The run took 510 s with a peak RSS of 10.0 GB.
  cortiq CPU `ppl --windows 12 --window-len 512` on that file gives
  24.9453, the same as the torch evaluator's w12 value of 24.9453.
- **q4tp against master, with no cache.** Master's binary and this branch
  each calibrated from scratch on the same 2048 WikiText-2 train tokens
  (`--codec q4tp`, head and embeddings q8_2f) and wrote byte-identical files
  (sha256 c939b1fd…). This also checks that the budgeted calibration reader
  yields the same windows as master's whole-file read. Peak RSS was 23.2 GB
  on this branch and 23.4 GB on master.
- **Code review.** No correctness bug was found in the q2tp v2 writer
  against the unchanged reader and kernels, in the GPTQ sweep, the blocked
  trailing update, the threaded Cholesky and inverse, the Hessian cache
  reader and writer, or the calibration reader. Three fixes were committed:
  1. RAM guards for 27B-scale runs.
     - At the default `--calib-chunk-layers 0`, a Qwen3.8-27B capture needs
       about 230 GB and was not checked; it would have died with an OOM kill
       during its first CPU window. It is now refused before the forward
       runs, with a chunk size that fits.
     - The widest-first order starts all the widest folds together. Without
       `CMF_GPTQ_MEM_GB` nothing capped them; the RAM left now does.
     - The bytes do not change. The fixed binary rebuilt W0cd from the
       cache on 6 threads, byte-identical (sha256 81f35e49…).
  2. `env_guard` now also clears `CMF_Q2TP_ENCODER`, so a panicking test
     cannot leak the v1 encoder into later tests.
  3. `par.sh` now passes `--q2tp-debias off`. The study's N1 and W0 arms
     have no debias, and the CLI default is now on.
- **Behaviour change outside quantization.** The DSpark draft requantizes its
  gate/up experts from q4tp to q2tp at upload, through `encode_q2tp`. It now
  runs v2, which does about 6× the search work of v1. The draft may encode
  better, but GPU upload of the draft takes longer. `CMF_Q2TP_ENCODER=v1`
  restores the old encoder.

### Open issues

- **The calibration forward is CPU-only, and this is the bottleneck for
  Qwen3.8-27B.** The Hessian hook lives in the CPU `QTensor::matmat`.
  - MiniCPM5-2B (q8_2f model, 13 threads): about 155 s per 2048-token
    window, roughly 13 tok/s. 32k tokens take about 40 min per pass; 131k
    tokens take about 2.8 h. For that reason the MiniCPM5 131k arm was not
    run; the long budget was shown on Qwen3-0.6B instead (52 s per window).
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
  - The threaded version and the RAM cap on concurrent groups handle this.
  - On MiniCPM5 one GPTQ-only arm took 14–20 min on the shared pod, about
    7 min per 21-layer chunk, mostly the 6144-wide folds.
  - The cache blocking made no measurable difference there (W0c_blocked
    879 s against W0c 829 s, same bytes, both under contention). At this
    size the folds are compute-bound in `dot64`. A faster `dot64` with more
    accumulators would change the summation order, and with it the q4tp
    bytes, so it was not done.
- **Mixed calibration** has only been checked functionally (detM, and the
  Qwen3-0.6B run). Its effect on quality against WikiText-only calibration
  was not isolated.
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

## Pipeline "genesis + tooling" (2026-10-17): metric genesis, escort, exact ppl as tools

Goal: turn the q2tp / q4tp studies' Python scripts (MiniCPM5-2B, `cmf-lb-local-runs/
{q4tp_genesis_minicpm5_20261017, q2tp_nvg_minicpm5_20261016T1800Z, q4tp_nvg_minicpm5_20261016T1215Z}`)
into production tools that scale to Qwen3.8-27B (hybrid GDN, `qwen3_5`) on one 32 GB GPU with
≥ 128 GB host RAM, without touching the CMF format or the kernels.

### What changed

- **cortiq CLI** (`crates/cortiq-cli/src/main.rs`): `cortiq dequant --dtype raw` writes a tensor's
  encoded payload verbatim; `cortiq patch-tensor --raw` stores an already encoded payload verbatim
  (the entry's dtype, or `--dtype` as the encoder would emit it for that shape), length-checked by
  the reader's own `validate_payload`, hashes recomputed; in place when the slot fits, else
  `--output`. Unit test `dequant_raw_and_patch_raw_round_trip_payloads_verbatim`.
- **tools/quant/** (new, see its README): `cmfio.py` (read-only CMF reader over mmap + write-back
  via `patch-tensor --raw` + `verify`), `cmfcodec.py` (byte-exact q4tp / q2tp / q8_2f / float
  payloads, field splicing), `qmodel.py` (torch model over a real CMF: llama family + `qwen3_5`
  hybrid; compact weights decoded on the device inside a memory-light custom autograd matmul;
  chunked KL head with an analytic backward; `resident` / `stream` residency with side-stream
  prefetch, bit-identical), `calib.py` (corpus specs, the studies' window placement, the cached
  teacher top-K), `metric_refit.py` (metric genesis), `escort.py` (sensitivity + water-filling,
  greedy re-measure, `--tensor-quant` output, escorted file), `ppl.py` (exact perplexity with
  paired SE), `tests/`.

### How to use

See `tools/quant/README.md` (commands, memory model, 27B sizing, recipes q4tp-v2 / q2tp-v2).

### Tests

- `cargo test -p cortiq-cli --bin cortiq -- dequant_raw` (5090 pod):
  `dequant_raw_and_patch_raw_round_trip_payloads_verbatim` ok.
- `python -m unittest discover -s tools/quant/tests` (5090 pod, CPU + two CUDA tests): 20 tests ok
  in 12.7 s; 21 with the in-place forward test (pod B, 6.1 s) — codecs (round trip, field splice, sequential f32 ladder), device decode == the numpy
  decoder bit for bit, row-blocked == one-block matmul, the custom matmul and the chunked KL head
  against dense autograd, batched head == per-window head (also over streamed, host-parked
  boundaries on CUDA), the chunked gated delta rule against
  the token recurrence (f64), stream == resident bit for bit over 4 training steps on tiny llama
  and qwen3_5 models (CPU and CUDA), escort candidates / cached block inputs / water-fill, the
  studies' window placement verbatim, mixed corpora, snapped write-back touches only f16 fields,
  the in-place forward-only pass == the kept-boundary pass.
- `tests/cortiq_parity.py --refit` (tiny random llama + qwen3_5 checkpoints converted by cortiq to
  f16 / q4tp / q2tp): torch NLL vs `cortiq ppl` (CPU exact) per window, worst relative difference
  5.2e-8; then `metric_refit.py` end to end on the q4tp and q2tp files of both architectures (f16
  teacher, `patch-tensor --raw` write-back + verify): held-out CE falls in all four, 0 bytes changed
  outside the f16 fields, the written files score the same in torch and cortiq (≤ 1.1e-7).

### Measured (MiniCPM5-2B, pod B RTX 2000 Ada shared under a 3 GB GPU cap)

WikiText-2 test, 160 × 512 windows, exact evaluator (`ppl.py`, every payload decoded from the file);
bf16 18.889. Journal and scripts: `cmf-lb-local-runs/genesis_tooling_20261017/`.

| check | result |
|---|---|
| evaluator parity | C2 19.5613 / f16 18.8890 / R2 21.8868 / W128dh 25.6222 = the studies' numbers; cortiq window 0 of G-C2: 1326.206278 vs torch 1326.206228 |
| teacher cache (f16.cmf through qmodel) vs the studies' teacher.pt | same windows, held CE 3.097050 = 3.097050, top-1 100 %, top-64 sets 99.999 % |
| **G-C2 reproduction** (`metric_refit.py --model C2.cmf --teacher f16.cmf --corpus train.ids,train=1024,held=65536:8+200000:16 --step --epochs 2 --mb 2`) | **19.1319** (12 w 18.1454), −2.195 % ± 0.089 vs C2; study 19.1344 (18.1460), −2.18 % ± 0.09. Best held-out CE 3.10660 at step 144 (study 3.10684 at 144); lo changed in 604 333 rows (study 604 286) |
| **streaming == resident** | the streaming file (one layer on the GPU, 0.85 GiB peak) and the resident file (2.00 GiB) are **byte-identical**; all 17 held-out evaluations equal to the last bit (resident run also resumed from a mid-run state) |
| escort inside C2 (`escort.py --hi q8_2f.cmf --corpus train.ids,held=8192:8 --bpw 0.1 0.27`) | +0.1 bpw: 24 v, 17 k, 2 o, 2 q, 1 down (study in A2: 23 v, 18 k, 3 o, 1 q, 1 up); v_proj median gain/MiB 10× other kinds; escorted file 19.3621 (−1.02 % ± 0.05 vs C2; study's C4 19.437) |
| q4tp-v2 chain end to end (C2 → escort → genesis, `--mb 8 --head-windows 8`) | **19.1092**, +1.17 % vs bf16 (study G-C4 19.0983); escort after genesis −0.12 % ± 0.04 |
| q2tp path (`metric_refit.py --model W128dh.cmf ... --step --epochs 2`, the study's R2 recipe) | **21.8833** (12 w 20.1603), −14.59 % ± 0.46 vs W128dh; study R2 21.8867 (20.1584), −14.58 % ± 0.46; best held-out CE 3.22974 at step 176 (study 3.2301 at 176) |

Speed on the shared pod: 28–39 s/step (streaming, mb 2) while three other jobs used the GPU and
~4 of 5.1 CPUs; 13.9 s/step resident mb 2 and 8.5 s/step streaming mb 8 + 8-window head when
alone (the study's script: 6.5 s/step at 7.9 GiB). Qwen3.8-27B geometry (scale probe on the 5090,
1.95 GB cap): 1.62 GiB peak for one 512-token window through a training step, 0.25–0.32 s/layer.

### Independent verification (second agent, same day)

Re-run from the committed branch with a separate build (`cargo build --release` on the 5090 pod):
the CLI test passes; 21 Python tests pass on pod B (CPU + CUDA); `cortiq_parity.py --refit` with
the new binary: worst torch vs cortiq 7.9e-8, all four refit chains ok (0 bytes outside the f16
fields). `cmfcodec` decode == `cortiq dequant --dtype f32` bit for bit (q4tp, q2tp, q8_2f tensors);
`dequant --dtype raw` == the reader's payloads. G-C2s sha256 = the journal's; it differs from C2
only in the 294 q4tp row-param blocks and 85 f16 gains (R2t vs W128dh likewise); C2-esc = C2 with
the 46 listed tensors replaced by q8_2f.cmf's payloads verbatim. `ppl.py` on G-C2s: 19.1319 again.
The engine itself (`cortiq ppl`, Vulkan, probe off, 160 × 512): C2 19.562 → G-C2s 19.132,
−2.199 % ± 0.089 paired over cortiq's own window NLLs (torch: −2.195 % ± 0.089). A fresh resident
`metric_refit.py` run of the G-C2 recipe reproduces the streaming run's step-16 value digest
(`b6d124542cbe8374`). Scale probe at 27B shapes: 1.62 GiB train / 0.92 GiB forward, as above.
Fixed: forward-only passes held the inputs, the current and the next boundary set at once; they
now replace boundaries in place (C2, 160 windows in one streamed block: peak RSS 6.27 → 5.05 GiB,
NLLs byte-identical; ~16 GB of pinned blocks fewer at 27B with `--teacher-block 1024`), and the
README's 27B table counts the power-of-two rounding of pinned blocks.

### Open issues

- GPTQ on 32 k mixed tokens with cortiq (`quantize-gptq --tokens 32768`) was not run here; C1/C2
  came from the q4tp study's torch GPTQ (same rule). The H-metric scale refit (C1 → C2, −0.31 %)
  is still a study script; q2tp GPTQ (placed ladder, layer-7 rule, debias) is not in cortiq.
- The genesis runs here train on WikiText only (to reproduce the studies); at 4 bits the gain is
  mostly in-domain, so a release needs the broad mixed corpus the tool supports (not measured).
- Qwen3.8-27B was not run (no checkpoint on the stands): the qwen3_5 path is validated on tiny
  random checkpoints against cortiq (≤ 1.1e-7) and end to end through the write-back; whether
  `quantize-gptq` captures Hessians for all 400 hybrid linears is unverified.
- Not done: decode caching across micro-batches, bf16 matmuls, CUDA graphs (the shared pod was
  launch-bound).
- Qwen3.8-27B-cmf decodes with MTP speculation by default. The refit leaves the `mtp.*` tensors
  byte-identical but retunes the trunk it drafts from: output stays exact (the trunk verifies), but
  re-measure acceptance / tok/s of the refit file before release.
- The f16 `lo` field moves in steps of 2^-8 log2 (0.27 % row gain) at |lo| in [4, 8) and 2^-7
  (0.54 %) in [8, 16) — C2's rows split 65 / 35 — the same size as the trained gains (mean |a|
  0.0055–0.0072): snapping costs part of the gain (G-C2 held-out CE 3.10660 unsnapped vs 3.10692
  snapped). A format limit, not a tool bug.
