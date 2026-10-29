# Spark-X2.6-4B on Metal (0.8.43)

Measurements behind the 0.8.43 Metal changes for
[Spark-X2.6-4B](https://huggingface.co/infosave/Spark-X2.6-4B-cmf) (36 layers:
9 full + 27 sliding-window 512, hidden 2560, FFN 10240 with exact GELU, 16 Q
heads × 256, 4 KV heads, tied head). Stand: Mac mini M4, 10-core GPU, 24 GB.

A decode token reads the whole weight set once: 4.12 GB for the q8_2f file,
2.14 GB for q4tp. The M4's matvecs reach 85–91 GB/s here, so the roofs are
~23 tok/s (q8_2f) and ~40 tok/s (q4tp).

## Method

The M4 drifts by ±20 % between processes (one GEMM process measured 2.90 and
2.25 TFLOP/s back to back), and desktop rendering kept its GPU 30–40 % busy
during much of this work. Every kept change was decided by one of:

- interleaved processes, A, B, A, B (`cortiq bench --core --ignore-eos`);
- an in-process A/B whose arms alternate per token (`CMF_METAL_AB=a/b`), the
  pair ratio of the GPU time per token;
- ratios inside one profiled run (`CMF_CHUNK_PROF=1`) or one bench process.

Stage costs of a decode token come from `CMF_METAL_AB=mcfr/mcfrx` with
`CMF_METAL_SKIP=<letters>`: the `x` arm drops the named stages (a attend,
p RoPE, n norms, q QKV, o O, g gate/up, d down, h head), tokens alternate.

## Results

| q8_2f, interleaved pairs | 0.8.42 | 0.8.43 |
|---|---:|---:|
| prefill, 1000-token prompt | 112–118 tok/s | 197–216 tok/s |
| prefill, 41-token prompt | 86–99 tok/s | 202–204 tok/s |
| 20–30-token prompt (`run`) | 0.85–1.29 s | 0.16–0.20 s |
| `serve` chat turn 2, 2.3k history, 65 new tokens | 11.3 s | 0.58 s |
| `serve` turn 2 after a 700-token answer | 16.7 s | 4.7 s |

| decode, in-process A/B, GPU time per token | short | depth 1000 |
|---|---:|---:|
| q4tp | 0.862× | 0.882× |
| q8_2f | 0.905× | 0.924× |

Decode on the quiet machine: q8_2f 16–19 tok/s (17.1 at depth 4k, 15.6 at
8k); q4tp ~30 tok/s with the desktop load, 32 before the attend change on a
quiet GPU. Prefill at depth: q8_2f 261 tok/s at 4000 tokens, 215 at 8000.

Quality, `cortiq ppl --tokens 4096` (wikitext-2, BOS in front):

| file | CPU | Metal 0.8.42 | Metal 0.8.43 |
|---|---:|---:|---:|
| q8_2f | 10.717 | 10.717 | 10.716 |
| q4tp | 11.758 | 11.765 | 11.765 |

Greedy outputs of 0.8.43 equal 0.8.42's on every prompt checked (both files);
the Metal and CPU paths diverge after some tokens on some prompts as before
(float order). A chat turn with prefix rollback equals a fresh prefill
character for character (48- and 700-token answers).

## Changes

1. **Dense q8_2f in the chunk prefill graph.** `chunk_run_gpu` refused every
   dense q8_2f layer: the chunk FFN had no x·col staging. gate/up now read
   `n·col` staged like q/k/v, down's field scales the GELU rows in place
   (`ChunkLayer::ffn_col`). Exact-GELU FFNs only: the SiLU down GEMM fuses its
   activation into the X load and has no field.
2. **Rollback reuse across chat turns.** `rollback_prefix_len` keeps the
   longest common prefix when every layer can drop its rows exactly; the
   sliding tail trim is anchored at the turn's prompt end
   (`trim_window_anchored`), ~216 KB per answer token in the 27 sliding layers
   until the next turn. `CMF_KV_ROLLBACK=0` restores extension-only reuse.
3. **Short prompts.** `Pipeline::chunk_min_rows`: 8 for spark2_5, 32 for every
   other model (`CMF_CHUNK_MIN_B`). Prompts of 2–7 tokens keep their
   per-position path (47–102 ms), from 8 tokens the chunk graph (~105 ms) beats
   the host walk (0.5–1.2 s).
4. **Decode attend.** Stage attribution put the attend at 19 % of a q4tp token
   at short context (6.4 ms) and 12.7 ms at depth 3100. Standalone (16 Q / 4 KV
   heads, width 256): 150 positions per-head 106 µs vs split-K in 32-row blocks
   30 µs; 512 positions split-K 128-row blocks 247 µs → 32-row 53 µs. The Spark
   plan sets `set_attend_blk(1, 32)`; `CMF_GQA_BLK`/`CMF_GQA_BLKSZ` override.
5. **Chunk attention P·V GEMM** (`mul_mm_f32nn`): K steps of 32 with packed
   float4 loads instead of 16 with bounds-checked scalars, the same 8×8 blocks
   in the same k order. Deepest 4k chunk, normalized to the gate/up GEMM of the
   same run: 0.54 → 0.40.
6. **q4tp prefill GEMM** (`q4tp_mul_mm`): one 8-byte nibble load instead of
   eight byte loads, the activation boost (a power of two) folded into the
   group scale. Rate against the q8 GEMM, alternating rounds: gate 0.86 → 0.885,
   down 0.86 → 0.93.

## Measured and rejected

- Wider q8_2f decode matvecs (16- and 8-byte loads, 1–8 rows per simdgroup):
  within ±2 % of `q8f_matvec_r4`.
- Fused q8_2f decode (gate|up|GELU in one dispatch, O and down residual in the
  epilogues, bit-exact): in-process 0.992 / 1.001.
- A 64×64 C tile for the q8 GEMM: neutral to −10 % against the 64×32 tile
  (2.3–3.1 TFLOP/s at b = 512).
- Vector loads in the scores GEMM (`mul_mm_f32nt`): no change.
- q4tp decode matvec with 2/4/8 rows per simdgroup: within noise once the GPU
  clock has ramped (the first kernel of a process measures low).

## Where the time goes now

- Decode: weight matvecs are ~83–85 % of a token. A smaller file is the lever.
- Prefill, deepest 4k chunk: weight GEMMs 64 %, attention 34 % (scores GEMM
  ~1 TFLOP/s). A fused causal flash attend is the next step for long prompts;
  at head width 256 it keeps 8 query rows per simdgroup, which caps it near
  1.6 TFLOP/s.

## Benches

```sh
CMF_GPU=1 CMF_BENCH_MODEL=spark-x2.6-4b-q8_2f.cmf \
  cargo test --release -p cortiq-engine --test metal_q8f_bench -- --ignored --nocapture
```

`metal_q8f_matvec_cost_by_kind`, `metal_q8_mul_mm_cost_by_kind`,
`metal_q4tp_matvec_cost_by_kind` (with a q4tp file), `metal_decode_attend_cost_by_depth`,
`metal_gemm_q8_vs_q4tp` (`CMF_BENCH_MODEL2` = the q4tp file). `CMF_Q8F_MSL` and
the `*_KERNELS` variables add candidate kernels compiled at runtime.
