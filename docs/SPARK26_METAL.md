# Spark-X2.6-4B (q8_2f and q4tp) on Metal (Mac mini M4, 10-core GPU, 24 GB) — 2026-10-29

Branch `spark26-metal` (worktree `~/dev/cmf-spark26`, from master 4a5b848f = 0.8.40), 5 commits,
not pushed. Model: `infosave/Spark-X2.6-4B-cmf` default file `spark-x2.6-4b-q8_2f.cmf`,
4,135,951,050 B, sha256 e9331f8f…494f (= HF LFS oid), in `~/models/spark-x2.6/`.

Shape: 36 layers (9 full + 27 sliding 512), hidden 2560, FFN 10240 (exact GELU), 16 Q heads × 256,
4 KV heads, tied q8_2f head [131072 × 2560]. A decode token reads ≈ 4.12 GB of weights.

## Measurement conditions

- Run-to-run drift of the M4 is large: one GEMM process gave 2.90 and 2.25 TFLOP/s back to back;
  one decode bench 14.1 and 18.1 tok/s. Only interleaved arms (A,B,A,B), in-process A/B
  (`CMF_METAL_AB`) or ratios inside one profiled run are compared.
- 12:04–12:55 another session's GPU compress A/B ran; those numbers are excluded.
- From ~14:30 the GPU showed 36–80 % utilization with no model process running (desktop rendering:
  WindowServer 45 % CPU, Comet and Claude renderers). Absolute numbers after that are lower
  (e.g. prefill 4k 152 vs 261 tok/s in the morning) and are only used as ratios.

## Before / after (binaries: base = 0.8.40, new = branch head)

| what | base | new | how |
|---|---:|---:|---|
| prefill, 1000-token prompt | 117.8 / 112.2 tok/s | 215.9 / 197.4 tok/s | interleaved pairs, quiet GPU |
| prefill, 41-token prompt | 91.2 / 85.7 / 99.0 | 203.0 / 202.2 / 204.1 | interleaved pairs, quiet GPU |
| prefill, 20–30-token prompt (`run`) | 0.85–1.29 s | 0.16–0.20 s | same binary, `CMF_CHUNK_MIN_B=32` = old path |
| chat turn 2 via `serve`, 2.3k history, 65 new tokens | 11.3 s (2323 tok) | 0.58 s (65 tok) | same binary, `CMF_KV_ROLLBACK=0` = old path |
| chat turn 2 after a 700-token answer | 16.7 s (2975 tok) | 4.7 s (717 tok) | same |
| `cortiq ppl --tokens 4096` wall (CMF_GPU=1) | 55 s | 19 s | back to back |
| prefill 4000 / 8000 tokens (new only, quiet) | — | 261.3 / 214.7 tok/s | |
| decode | unchanged | 16–19 tok/s short, 17.1 @4k, 15.6 @8k | no decode change kept |

CPU path, same file: prefill 19.9 tok/s (41 tokens, a foreign build running), decode 9.3 tok/s
then; on a quiet machine CPU decode reached 19.9 tok/s on a short prompt — decode is bandwidth-bound
and the CPU and GPU share the same memory.

Quality: `cortiq ppl` 4096 tokens wikitext-2 (BOS in front): CPU 10.717, base Metal 10.717,
new Metal 10.716 (card: 10.71 on Vulkan). Greedy outputs identical new Metal vs base Metal on every
prompt checked (96-token bakery prompt also = CPU; on the 24-token haiku the CPU differs from both
Metal paths at token ~10 — a pre-existing Metal/CPU float difference). Chat turn 2 with rollback
reuse = fresh prefill, character for character (48- and 700-token answers).

## What changed (kept)

1. **Dense q8_2f GELU stacks ride the Metal chunk prefill graph** (d3b48ecb). `chunk_run_gpu`
   refused every dense q8_2f layer ("the chunk FFN has no x·col staging"), so the default file
   prefilled on the host per-op walk. gate/up read `n·col` staged like q/k/v; down's field scales
   the GELU rows in place before its GEMM (`ChunkLayer::ffn_col`). SiLU dense q8_2f stays on the
   host walk (its down GEMM fuses SiLU and has no field).
2. **Chat-turn KV reuse rolls back to the common prefix** (c821c8c8). Reuse was extension-only;
   a chat app re-renders the previous answer through the template, so every turn re-prefilled the
   whole history. `rollback_prefix_len` keeps the longest common prefix when every layer can drop
   its rows exactly (host-owned f32 full attention, no recurrent/O(1)/bounded state, sliding tails
   still holding the window); the sliding tail trim is anchored at the turn's prompt end
   (`trim_window_anchored`) so a long answer stays droppable (costs ~216 KB per answer token in the
   27 sliding layers until the next turn). `CMF_KV_ROLLBACK=0` restores the old behaviour.
3. **Spark prompts of 8..31 tokens take the chunk graph** (9eaa95d4). The graph's 32-row floor
   (from its first commit, measured on Qwen2.5-0.5B) sent short prompts to the host walk.
   `Pipeline::chunk_min_rows` = 8 for spark2_5, 32 for every other model, `CMF_CHUNK_MIN_B`
   overrides. Prompts ≤ 7 tokens keep their faster per-position path (47–102 ms).
4. **Chunk attention P·V GEMM** (773b52a7): K steps of 32 with packed float4 loads instead of 16
   with bounds-checked scalars; same 8x8 blocks in the same k order (ppl and greedy unchanged).
   Deepest 4k chunk, normalized to the gate/up GEMM of the same run: P·V 0.54 → 0.40 (−27 %,
   ~4.5 % of that chunk). End to end not resolvable under the desktop GPU load.
5. Diagnostics: `q8f_kernel_bench` / `q8_mm_kernel_bench` + `tests/metal_q8f_bench.rs` (ignored);
   test `swa_anchored_trim_keeps_the_rollback_window`.

## Measured and rejected

- **Decode matvec** is at the roof: `q8f_matvec_r4` 85–91 GB/s per projection kind on a quiet GPU;
  best variant (`q8f_mv_h4`, 8-byte loads) +1–4 % on most shapes, ~2 % weighted — inside drift.
  Matvecs ≈ 47–48 ms of a 51–62 ms token → decode runs at ~85 % of weight streaming.
- **Fused q8_2f decode** (gate|up|GELU one dispatch, O/down residual epilogues, bit-exact):
  in-process A/B 0.992 (short) / 1.001 (1k) — barriers are not the cost. Reverted.
- **Split-K GQA attend from 64 positions** (`CMF_METAL_AB=mcfr/mcfrb`): 1.001 @1k, 1.002 @4k.
- **64×64 GEMM tile** (k2.metal, bit-identical): neutral to −10 % vs the built-in 64×32
  `q8_mul_mm`, which runs 2.3–3.1 TFLOP/s at b=512.
- **Scores GEMM vector loads** (mul_mm_f32nt): ratio unchanged (0.366 → 0.363). Reverted.

## Where the time goes now

- Prefill (deepest 4k chunk): weight GEMMs 64 % (at 2.3–3 TFLOP/s), attention 34 %
  (scores ~1 TFLOP/s, P·V now ~1 TFLOP/s, softmax 4 %). Next lever for long prompts is a fused
  flash-style causal attention (and scoring only the 512-row band on sliding layers, ~3 %).
- Decode: weights 4.12 GB/token. For a ~2× faster decode on a MacBook the lever is the smaller
  file (`spark-x2.6-4b-q4tp.cmf`, 2.16 GB, card ppl +10.9 %) — not measured here.

## Files here

- `cortiq-base` (0.8.40), `cortiq-chunk`, `cortiq-reuse`, `cortiq-minb`, `cortiq-attmm` — arms in order.
- `ab_prefill.sh` (interleaved bench A/B), `chat2.sh` (two chat turns through `serve`).
- `k1.metal` (matvec candidates), `k2.metal` (64×64 GEMM) — rejected kernels.
- Raw outputs: `kbench1.txt`, `kbench_quiet.txt`, `mmbench_clean1.txt`, `ab_chunk_*.txt`,
  `ab_attmm_*.txt`, `skip_ctx1000.txt`, `ppl_gpu.txt`, `serve*.log`, `r2*_*.json`, `out_*.txt`,
  `short_*.txt`.


## Part 2 (15:15–16:40): q4tp file + decode attention + q4tp GEMM

File `spark-x2.6-4b-q4tp.cmf`, 2,161,419,978 B, sha256 76fcaafe…937d (= HF LFS oid). A decode token
reads ≈ 2.14 GB (36 × 54.7 MB + 175 MB q4tp tied head), roof ≈ 40 tok/s at ~88 GB/s.
Conditions: the desktop kept the GPU at 32–42 % (and a foreign `cortiq compress` at 100 % for
~15 min, excluded); a foreign cargo build ran. Absolute numbers are depressed and noisy; the kept
changes were decided by in-process A/B or by ratios inside one process.

### Stage attribution (new tool: `CMF_METAL_AB=mcfr/mcfrx` + `CMF_METAL_SKIP=<letters>`)

The arm with `x` drops the named stages, tokens alternate, so the pair ratio is the stage cost on a
drifting machine. q4tp decode, GPU ms per token, short context (before the attend change):
gate/up 11.3 · down 6.4 · **attend 6.4** · QKV 4.2 · O 3.0 · head 1.9 · RoPE 0.3 · norms 0.3.
Attend vs depth: 6.4 ms (40–240 positions), 9.6 (≈500), 10.2 (≈1100), 12.7 (≈3100).

### 6. Decode attend: split-K from the second position, 32-row blocks (2a35bcf7, kept)

The Spark plan used the per-head `gqa_attend` up to 64 positions and the split-K kernel in
128-position blocks past that: few long dependent chains for a latency-bound walk. Standalone
(`attend_kernel_bench`, 16 Q / 4 KV heads, width 256): 150 positions per-head 106 µs vs split-K/32
30 µs; 512: split-K/128 247 µs → /32 53 µs; 1000: 194 → 145 µs. `set_attend_blk(1, 32)` for Spark.

| in-process A/B, GPU time per token | short | depth 1000 |
|---|---:|---:|
| q4tp | 0.862 | 0.882 |
| q8_2f | 0.905 | 0.924 |

Greedy output identical to the previous binary on 6 prompts (both files, 96 tokens); q4tp ppl
11.765 before and after.

### 7. q4tp prefill GEMM loads (649d0498, kept)

`q4tp_mul_mm` ran 0.86 of the q8 GEMM rate (layer-0 gate and down, b = 512, alternating rounds in
one process, `metal_gemm_q8_vs_q4tp`). One 8-byte nibble load instead of eight byte loads and the
activation boost folded into the group scale (a power of two — the same products): gate 0.885,
down 0.93 of q8. ppl 11.765 and greedy output unchanged.

### q4tp quality

`ppl --tokens 4096`: Metal 11.765 (base = branch), CPU 11.758 (card: 11.88 on Vulkan, other text
protocol). Greedy: Metal paths identical to each other; the CPU agrees on 2 of 3 prompts.

### Final interleaved runs (desktop GPU load ~41 %)

| file | ctx | base prefill | new prefill | base decode | new decode |
|---|---|---:|---:|---:|---:|
| q4tp | 41 tok | 197.9 / 176.9 | 176.1 / 176.5 | 27.3 / 27.5 | 30.0 / 29.7 |
| q4tp | 1000 | 182.7 / 182.1 | 182.4 / 178.8 | 24.5 / 24.6 | 24.1 / 24.9 |
| q4tp | 4000 | 182.8 | 172.3 | 21.1 | 22.2 |
| q8_2f | 41 tok | 104.1 / 81.0 | 164.7 / 152.2 | 15.1 / 14.6 | 16.2 / 16.4 |
| q8_2f | 1000 | 99.4 / 102.7 | 168.4 / 166.7 | 13.9 / 14.2 | 14.4 / 14.0 |
| q8_2f | 4000 | 90.6 | 171.7 | 14.5 | 14.1 |

(On a quiet GPU in the morning the same q4tp binary decoded 32 tok/s short; process-to-process
noise here is ±10 %, below the in-process A/B resolution.)

### Measured and rejected in part 2

- q4tp matvec with 2/4/8 rows per simdgroup (k3.metal): within noise of `q4tp_matvec_m` on every
  shape once the GPU is warm (the first-kind 43 GB/s was a clock ramp-up artefact).
- Fused flash attention for the chunk prefill: not built — hd 256 leaves 8 query rows per
  simdgroup in registers, K/V reuse caps it near 1.6 TFLOP/s; ~10 % of a 4k prompt for a large
  kernel. Listed as the next step for long prompts.

Files added: `k3.metal`, `q4_skipab*.txt`, `attk_ab.txt`, `ppl_q4tp.txt`, `final_ab.txt`,
binaries `cortiq-attdef` (attend change) and `cortiq-q4mm` (= branch head).
