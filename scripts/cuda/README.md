# CUDA backend: the measurement run

What built, checked and measured the native CUDA backend (`docs/CUDA.md`) on a rented NVIDIA card. The
paths below assume the run's root on the pod, `/root/cuda` (`CUDA_RUN_ROOT` overrides it):
`src/` (this repository), `bin/`, `models/`, `prompts/`, `prompts_long/`, `scripts/` (a copy of `pod/`),
`out/`, `logs/`.

## pod/

| script | what it does |
|---|---|
| `build_cuda.sh fast\|release` | builds `cortiq` with `--features cuda`: `fast` without LTO into its own target directory (iteration), `release` as shipped (measurements) |
| `test_cuda.sh` | the backend's unit tests on the card: kernel parity against the host codecs |
| `bench_one.sh TAG BIN MODEL CTX [ENV=V ...]` | one `cortiq bench --core --json` with a VRAM poller; writes `out/TAG.{json,log,vram,vram_trace}` and a one-line summary (`vram_decode`: the median of the run's last quarter) |
| `chain_vulkan.sh` | the wgpu (Vulkan) baseline at 1K / 4K / 16K, both files |
| `chain_cuda.sh BIN LABEL [q4tp\|q8_2f ...] [-- ENV=V ...]` | the CUDA path at 1K / 4K / 16K |
| `chain_final.sh [reps]` | the table: Vulkan and CUDA taking turns at every prompt length |
| `table.py` | the medians of `chain_final.sh` as a Markdown table |
| `ref_outputs.sh TAG BIN MODEL [FORCE=tag] [ENV=V ...]` | greedy outputs of one backend for every prompt, with the ids and every step's logits; `FORCE=` teacher-forces it along another tag's ids |
| `chain_refs.sh` | the references: Vulkan, Vulkan with `CMF_COOP=0`, the CPU along the Vulkan ids |
| `chain_cuda_check.sh BIN [quants] [-- ENV=V ...]` | the CUDA path against the Vulkan references: a free run and a forced one |
| `compare_refs.py REF OTHER` | ids in common, top-1 agreement and logit differences of two tags |
| `chain_truth.sh`, `compare_truth.py` | every path forced along the same ids against the host with float activations (`CMF_GPU=0 CMF_SDOT=0`): top-1 and logit differences at the first token and at the later ones |
| `chat_check.sh`, `serve_check.sh` | a three-turn chat in one process, and two server requests sharing a prefix (K/V reuse), on both paths: the texts must agree |
| `chain_final_verify.sh` | the final binary: output checks on short and long prompts, the chat and server checks, one more pass of the CUDA rows, the other prompt modes, a 32K prompt |
| `chain_vkfix.sh BIN` | the wgpu path with the first token's head on the token graph: free runs on every prompt against the CUDA path, and the first token's time and the card's memory before and after |
| `chain_release_check.sh [BIN]` | the release binary, both GPU paths: free runs on every prompt (the ids must agree), the CUDA path against the one measured, the wgpu path against the float host, the chat and server checks, one pass of the table's rows |
| `chain_turbo_refs.sh [BIN] [reps]` | the turbo file as the wgpu path runs it (the far field past the seal of 8 192 rows): the rows at 16K / 32K / 64K and greedy outputs on the long prompts |
| `chain_cuda_turbo.sh BIN [check\|bench\|all] [reps]` | the far field on the CUDA path: outputs against the wgpu path taken the same way (`CMF_O1_WHOLE=1`), free and forced, and the rows at 16K / 32K / 64K |
| `chain_cuda_turbo_exact.sh BIN` | the same comparison with both prompts read in f32 (`CMF_COOP=0`, `CMF_CUDA_PREC=exact`) |
| `chain_cuda_turbo_host.sh BIN IDS [quants]` | the far-field token kernels against the host's from one seal: the decode on the card and on the host with float activations, forced along the same ids |
| `turbo_chat_check.sh [BIN]` | a chat that starts past the seal (the long document, then two questions) on both paths |
| `serve_turbo_check.sh [BIN]` | `cortiq serve` on the CUDA path: the health and model endpoints, a chat request and its follow-up, a document past the seal and a second question to it, a streamed answer, a plain completion, two requests at once on two slots |
| `chain_cuda_passkey.sh BIN [chars ...]` | facts planted in long Wikipedia text (`scripts/spark26/passkey_file.py`), eight trials a length: the far field and exact attention on CUDA |
| `chain_final_turbo.sh BIN [reps]` | the turbo table: the wgpu far field, the CUDA far field and CUDA with exact attention taking turns at 16K / 32K / 64K, then one 500-token decode a path |
| `table_turbo.py [--json FILE]` | the medians of `chain_final_turbo.sh` as a Markdown table, and as the data of the page's figures (`figures_turbo.json`) |

`card_figures.py OUT_DIR` draws the figures of the model page (`docs/hf/Spark-X2.6-4B-turbo-CUDA-cmf.md`) in
English, Russian and Chinese: the far field past the seal from `figures_turbo.json`, exact attention from
the medians of `docs/CUDA.md`, and how a token and a prompt run.

## prompts/, prompts_long/

Five prompts of 20 to 2 600 tokens (English, Russian, code, a document to summarize, a paragraph to read)
and one of 20 255 tokens built from this repository's documents; the turbo checks read it once and twice over
(40 494 tokens).

## bench/

Standalone micro-benchmarks, built with `nvcc` on the pod:

- `mv_bench.cu` — the decode matvec (`mv_multi`, the engine's own kernel source) on Spark-X2.6-4B's shapes
  against a load-only kernel; `-DVARIANT=n` adds a variant of the q4tp row loop from `mv_variants.cuh`
  (compute only, no scale lookup, two accumulators, int16 inputs with `__dp2a`, two groups a step, the
  ladder in registers, a compile-time stride).
- `gemm_bench.cu` — cuBLAS GEMM rates for the prompt's shapes in f32, TF32, f16 operands with f32 or f16
  output, and FAST_16F.
