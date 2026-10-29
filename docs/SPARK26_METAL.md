# Spark-X2.6-4B q8_2f on Metal (Mac mini M4, 10-core GPU, 24 GB) — 2026-10-29

Branch `spark26-metal` (worktree `~/dev/cmf-spark26`, from master 4a5b848f = 0.8.40).
Model: `infosave/Spark-X2.6-4B-cmf` default file `spark-x2.6-4b-q8_2f.cmf`,
4,135,951,050 B, sha256 e9331f8f…494f (matches the HF LFS oid), in `~/models/spark-x2.6/`.
Binaries: `cortiq-base` (0.8.40 as is), `cortiq-chunk` (this branch).

Shape: 36 layers (9 full + 27 sliding 512), hidden 2560, FFN 10240 (exact GELU),
16 Q heads × 256, 4 KV heads, tied q8_2f head [131072 × 2560]. Per decode token the
weights read are 36 × 105 MB + 336 MB head ≈ 4.12 GB.

## Measurement conditions (read before the numbers)

- Another session ran a GPU compress A/B (`cmf-compress-runs/fold-dev-ab-20261029`,
  `cortiq-fd compress`) and cargo builds from ~12:04 to ~12:55. Every number taken in
  that window is contaminated (decode fell to 4.5 tok/s, GPU clock includes the other
  job's work) and is NOT in the tables below except where marked.
- Run-to-run drift of the M4 is large even on a quiet machine: the same GEMM process
  gave 2.90 and 2.25 TFLOP/s back to back; decode `bench` gave 14.1 and 18.1 tok/s in
  two runs of the same binary. Only interleaved arms (A,B,A,B) are compared.
- The owner paused testing at ~13:00 (parallel task); the A/B below stopped after two
  full pairs.

## Before / after

`cortiq bench <file> --core --ignore-eos --ctx 1000 --tokens 32`, arms interleaved, quiet GPU:

| pair | base prefill | chunk prefill | base decode | chunk decode |
|---|---:|---:|---:|---:|
| 1 | 117.8 | **215.9** | 17.6 | 18.2 |
| 2 | 112.2 | **197.4** | 16.5 | 16.7 |
| 3 | 87.5 | (stopped) | 13.8 | (stopped) |

Prefill ×1.76–1.83 per pair at 1000 tokens; decode unchanged (no decode code changed).

CPU path, same file (`CMF_GPU=0`, a foreign rustc build was running, so understated):
prefill 19.9 tok/s (41-token prompt), decode 9.3 tok/s.

Not yet measured cleanly (pending the owner's go): the short-prompt (41 tokens) prefill
pair, `cortiq ppl` CPU vs `CMF_GPU=1`, decode at depth with the split-K attend lever.

## What changed

### 1. Dense q8_2f stacks ride the Metal chunk prefill graph (kept)

Diagnosis: `CMF_PREFILL_PROF=1` showed Spark's prompt on the host per-op layer walk
(`prefill-split … zi_mm GEMMs 0`), `CMF_CHUNK_PROF=1` printed nothing. Cause:
`Pipeline::chunk_run_gpu` refused every dense q8_2f layer on purpose — "the chunk FFN
has no x·col staging" — so the default (q8_2f) file never reached the one-submission
chunk graph, while the attention projections already staged x·col there.

Fix (`pipeline.rs` gate + `gpu_metal.rs` chunk FFN):
- `ChunkLayer::ffn_col` carries the gate/up/down input fields.
- gate and up each read `n·col` staged by `col_scale_rows` (as q/k/v already do);
  down's field scales the GELU activation rows in place before the down GEMM
  (`gelu_mul_pre`'s own field indexes one row only, so it cannot be reused for b rows).
- Opened for the exact-GELU dense FFN only (what was measured). A dense SiLU q8_2f
  stack keeps the host walk: its down GEMM fuses SiLU into the X-tile load and has no
  input field. The Metal prep refuses a mismatched field width or a non-GELU layer.

Check: greedy 96 tokens (`run --greedy --no-think`, 140-token prompt) identical for
chunk Metal, base Metal and CPU. Chunk stage profile (one 512-row chunk, contended
run, shares only): gate/up GEMM 36 %, down 23 %, QKV 13 %, O 7 %, attention 15 %,
norms/RoPE/axpy 6 %.

### 2. Kernel bench harness (kept, diagnostics only)

`gpu_metal::q8f_kernel_bench` (decode matvec) and `gpu_metal::q8_mm_kernel_bench`
(prefill GEMM): every tensor of one projection kind across the layers (no tensor warm
in the system cache), all reps in one command buffer on the GPU clock, built-in
pipelines or an entry point compiled from an MSL file at runtime (kernel iteration
without an engine rebuild). Driver: `tests/metal_q8f_bench.rs` (ignored tests):

    CMF_GPU=1 CMF_BENCH_MODEL=<file> cargo test --release -p cortiq-engine \
      --test metal_q8f_bench -- --ignored --nocapture

## Findings that closed levers (measured, not pursued)

- **Decode matvec is at the bandwidth roof.** The four-row `q8f_matvec_r4` the Spark
  plan already opens runs 88–98 GB/s per projection kind (kbench1.txt; contended run,
  so a lower bound): q 92, k 78, o 88, gate 94, down 92, head 98 GB/s. All matvecs of a
  token sum to ~44.5 ms; a quiet decode token is ~51 ms GPU at short context. Wide
  16-byte-load variants (k1.metal, w1/w2/w4/w8/h2/h4) were within ±3 % of r4 on every
  shape. Ceiling for this file on the M4 ≈ 23–24 tok/s; the base already reaches 18.
- **64×64 GEMM tile** (k2.metal, bit-identical output): neutral to −10 % against the
  built-in 64×32 `q8_mul_mm` on a quiet GPU (mmbench_clean1.txt). The built-in runs
  2.3–3.1 TFLOP/s at b=512.
- Decode GPU time grows ~2 ms per 100 positions (51 → 57 ms over 300 tokens): the
  per-head `gqa_attend` (16 threadgroups, K/V re-read per Q head) serves every layer
  under 512 positions, so all 27 sliding layers. The split-K GQA-shared kernel from 64
  positions (`CMF_METAL_AB=mcfr/mcfrb`) read 1.03 (worse) but only under contention —
  open, needs a quiet A/B at depth.

## Files here

- `cortiq-base`, `cortiq-chunk` — the two arms.
- `ab_prefill.sh` — interleaved bench A/B of two binaries.
- `k1.metal` (matvec candidates), `k2.metal` (64×64 GEMM candidate) — rejected kernels.
- `kbench1.txt`, `mmbench_clean1.txt`, `ab_chunk_ctx1000.txt` — raw outputs.
- `out_*.txt`, `prompt1.txt` — the greedy comparison.
