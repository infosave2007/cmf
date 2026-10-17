# tools/quant — post-quantization tools for q4tp / q2tp CMF files

Offline tools that improve an already quantized CMF without changing the format or the
kernels: every file they write is an ordinary CMF that stock cortiq runs. They read the
container directly (`cmfio.py`), decode payloads with the decoder's exact arithmetic
(`cmfcodec.py`), run the model in torch (`qmodel.py`), and write back through
`cortiq patch-tensor --raw` (cortiq recomputes the hashes, `cortiq verify` runs after every
write).

| tool | what it does |
|---|---|
| `ppl.py` | exact perplexity of a real CMF on cortiq's window protocol (per-window NLLs, paired SE vs a reference) |
| `metric_refit.py` | **metric genesis**: trains only the f16 values the file already stores (q4tp/q2tp row `lo`/`step`, q8_2f row scales, RMSNorm gains), codes frozen, KL to a teacher's top-64 tokens; writes a real CMF |
| `escort.py` | per-tensor sensitivity inside the quantized model, water-filled into a bits-per-weight budget; prints `--tensor-quant` overrides and can write the escorted file |
| `calib.py` | corpora, window placement, the cached teacher top-K |
| `tests/` | unit tests (codecs, custom autograd vs dense references, gated delta rule vs the recurrence, stream == resident bit for bit) and `cortiq_parity.py` (torch model vs `cortiq ppl` on tiny random llama / qwen3_5 checkpoints) |

Architectures: the llama family (llama, mistral, MiniCPM5, qwen2 biases, qwen3 q/k norms) and
the `qwen3_5` hybrid of Qwen3.5 / Qwen3.8 dense models (Gated DeltaNet layers + gated full
attention with q/k norms and partial RoPE, zero-centred norms). Geometry comes from the CMF
header; tensor names are cortiq's canonical names. Anything else (MoE, MLA, sliding windows,
soft-capping, YaRN …) is refused at load rather than mis-modelled.

## Requirements

- python 3.10+, torch 2.1+ with CUDA, numpy; `safetensors` only for `tests/cortiq_parity.py`
- cortiq with `patch-tensor --raw` / `dequant --dtype raw` (this branch; 0.8.19 + the raw CLI commit)
- token ids of every corpus, made with the model's own tokenizer:

```
CMF_PPL_IDS_OUT=train.ids cortiq ppl model.cmf --file corpus.txt --windows 1 --window-len 8
```

(`cmfio.tokenize_with_cortiq` does the same from python; `.npy` / raw int32 `.u32` ids work too.)

## Memory model (why a 27B model fits one 32 GB GPU)

The quantized body is kept compact: packed codes (0.5 B/weight q4tp, 0.25 B q2tp, 1 B q8_2f)
plus an f32 scale per 32-group and its rung power — 0.66 B/weight for q4tp. The weights are
decoded to f32 on the device inside a custom autograd function that saves only the input and
the small scale tensor (the backward decodes again), so a layer never holds more than one f32
weight. The KL head scans the vocabulary in chunks in both directions (no `[positions,
vocab]` tensor).

- `--residency resident`: every layer's compact weights live on the GPU.
- `--residency stream`: they live in pinned host RAM and one decoder layer at a time is
  copied to the GPU (the next one is prefetched on a side stream); boundary activations stay
  in host RAM (`--act-device cpu`, the default when streaming).

Both run the same layer-major schedule with the same deterministic kernels, so **stream and
resident produce bit-identical results** (unit test on CPU and CUDA, and the MiniCPM5 run
below). `--mb` splits a batch into micro-batches (same objective, accumulated gradients);
`--head-windows N` lets N windows share one vocabulary scan (fewer, larger kernels — use 8 with
`--mb 8`; 1 keeps the per-window head of the studies op for op). `--gpu-cap-gb` caps this
process's allocator on a shared GPU; a run killed by an out-of-memory error continues with
`--resume`, bit for bit.

Sizing for Qwen3.8-27B q4tp (hidden 5120, 64 layers, vocab 248320), batch 8 x 512:

| | host RAM | GPU |
|---|---:|---:|
| student body, compact (streamed) | ~18 GB pinned | 2 layers, ~0.6 GB |
| boundary activations (65 x 8 x 512 x 5120 f32) | 5.5 GB | — |
| head q4tp (compact, resident) | — | ~0.8 GB |
| per-layer working set, micro-batch 4 | — | ~2–4 GB |
| teacher top-64 cache, 1024 windows | 0.27 GB | — |
| teacher (f16 CMF, 54 GB, streamed once per block from the page cache) | block of 1024 windows: 11 GB | 1 layer + f16 head 2.5 GB |

## metric_refit.py — metric genesis

```
python tools/quant/metric_refit.py --model SRC.cmf --teacher REF.cmf --out RUN \
    --corpus wiki.ids,train=512,held=65536:8 --corpus docs.ids,train=256 --corpus code.ids,train=256 \
    --step --epochs 1 --write SRC-genesis.cmf --cortiq cortiq [--residency stream --mb 4]
```

- Trainable (log2 units, init 0): per output row of every q4tp / q2tp linear `lo += a`
  (an exact row gain; rung scales are 2^(lo + k·step)) and with `--step` `step += b`
  (b ≥ −step/2, lr/16); per row of every q8_2f linear `row_scale *= 2^a`; per channel of every
  RMSNorm gain `(g + o) *= 2^n` (o = 1 for zero-centred norms). Embedding and head frozen.
- Loss: KL(teacher ‖ student) on the teacher's top-K (default 64) plus a tail bucket. The
  teacher is a float CMF of the same model (`cortiq convert --quant f16`), run once in exact
  f32, cached in `--teacher-cache` (validated by file size + corpus digests + window starts).
- Adam (0.9, 0.99), lr 1e-3, 8 warm-up steps, cosine to 0.1×, batch 8 × 512; held-out CE every
  16 steps; the best checkpoint is snapped to f16 (the snapped held-out CE is reported) and
  written as `RUN/raw/*.bin` + `RUN/manifest.json` + `RUN/trained.npz`; `--write` patches them
  into a byte copy of SRC (in place, same sizes: no tensor is held in RAM) and verifies.
- Corpus spec: `PATH[,train=N][,held=OFF:CNT+OFF:CNT][,name=X]`; mixed corpora are pooled,
  the held-out CE is reported per corpus. Placement is the studies' rule (`calib.py`).
- Seeded and deterministic; `--resume` continues from `RUN/state.pt` (saved at every eval).

## escort.py — which tensors get more bits

```
python tools/quant/escort.py --model SRC.cmf --hi SRC-q8_2f.cmf --corpus train.ids,held=8192:8 \
    --bpw 0.1 0.27 --out esc [--metric kl --teacher f16.cmf] [--greedy 3] [--write SRC-esc.cmf]
```

Restores one linear at a time to the version the `--hi` file stores (q8_2f for a q4tp file,
q4tp for the 2-bit tensors of a q2tp file — several `--hi` files are allowed), measures the
held-out NLL (or teacher-KL) drop from cached block inputs (layer l reruns only l..L−1), and
water-fills gain per extra byte into each budget (`--bpw` × body parameters / 8). `--greedy R`
re-measures inside the escorted model after each of R budget slices (use it when one tensor
dominates, e.g. a super-weight layer at 2 bits). Output: `sensitivity.json`, `escort.txt`,
`tensor_quant.txt` (`--tensor-quant NAME=q8_2f` lines), and with `--write` the escorted model
(the hi payloads patched into the measured file — how the q4tp study built C4). The overrides
work for both `cortiq convert` and `cortiq quantize-gptq --codec q4tp`: an overridden linear
skips GPTQ and takes the override (`gptq.rs` routes every tensor whose effective quant is not
q4tp to the converter policy).

## ppl.py

```
python tools/quant/ppl.py --model X.cmf --ids test.ids --windows 160 --also 12 --out pw/X --ref pw/BASE
```

Exact f32 (no TF32), windows placed like `cortiq ppl --windows N --window-len 512`.

## Recipes (measured on MiniCPM5-2B, WikiText-2 test 160 × 512, bf16 = 18.889)

All numbers are perplexities of real files on the same 160 evenly spaced 512-token windows
(81 760 tokens); bf16 = 18.889; Δ = paired change against the line's input (± SE over windows). Head
and embeddings are q8_2f in every q4tp/q2tp line (as shipped). "study" = measured in the
q4tp / q2tp studies with their scripts; "tools" = reproduced here with these tools.

### q4tp-v2

1. Float exports of the checkpoint: `cortiq convert --quant f16` (teacher, GPTQ input) and
   `--quant q8_2f` (GPTQ calibration forward, escort hi file).
2. GPTQ on 32 k tokens of a broad mixed corpus (de-duplicated against every eval set):

   ```
   cortiq quantize-gptq f16.cmf --codec q4tp --calib-model q8_2f.cmf --calib mix.txt \
       --tokens 32768 --window 2048 --lambda 0.01 --hessians gptq.hess --output q4tp-gptq.cmf \
       --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f
   ```
3. (optional) H-metric scale refit of the GPTQ file (the q4tp study's `fpcal.py`; not in this
   tool set — metric genesis subsumes most of it).
4. Escort: `escort.py --model q4tp-gptq.cmf --hi q8_2f.cmf --corpus calib.ids,held=... --bpw 0.1
   --write q4tp-esc.cmf` (+0.1 bpw = +25 MB on MiniCPM5, +1.6 % file).
5. Metric genesis: `metric_refit.py --model q4tp-esc.cmf --teacher f16.cmf --corpus ... --step
   --write q4tp-v2.cmf` (one epoch is probably enough at 4 bits: the second epoch memorises).

| file | body bpw | wiki 160 w | Δ vs input | source |
|---|---:|---:|---:|---|
| bf16 reference | 16 | 18.889 | | tools `ppl.py`: 18.8890 |
| A1 shipped recipe (`quantize-gptq`, 2 048 tokens) | 4.169 | 20.226 | | study |
| C1 GPTQ on 32 768 tokens | 4.169 | 19.622 | −3.0 % | study (torch GPTQ = cortiq's rule) |
| C2 = C1 + H-metric scale refit | 4.169 | 19.561 | −0.31 % | study; tools `ppl.py`: 19.5613 |
| **G-C2 = C2 + metric genesis** | 4.169 | **19.132** | **−2.20 % ± 0.09** | **tools** (study 19.134); stream and resident files byte-identical |
| C2 + escort +0.1 bpw (escort.py list measured in C2) | 4.268 | 19.362 | −1.02 % ± 0.05 | **tools** (study C4 with the A2 list: 19.437) |
| G-C4 (study) = C4 + metric genesis | 4.268 | 19.098 | −1.74 % | study |
| **q4tp-v2 here = C2 + escort + metric genesis** | 4.268 | **19.109** | −1.31 % ± 0.07 | **tools** (−2.31 % ± 0.09 vs C2, +1.17 % vs bf16) |

Paired on the same 160 windows: G-C2 closes 63.9 % of C2's gap to bf16 at identical bytes (study
63.5 %); after genesis the +0.1 bpw escort is worth only −0.12 % ± 0.04 (study −0.19 %). Against
the shipped recipe A1 (20.226) the equal-size G-C2 is −5.4 %.

Escort choice (+0.1 bpw): 24 v_proj, 17 k_proj, 2 o_proj, 2 q_proj, 1 down_proj (study inside A2:
23 v, 18 k, 3 o, 1 q, 1 up); v_proj's median gain per MiB is 10× any other kind. Single-tensor
gains on 8 windows are noisy (per-tensor r = 0.22 between C2 here and A2 in the study), so
the list is recomputed for every base file rather than reused.

At 4 bits the genesis gain is mostly in-domain (study: −0.3 … −0.8 % on docs / code against
−2.2 % on WikiText), so for a release the teacher corpus must be broad and mixed
(`--corpus` repeatable) and de-duplicated against the evaluation sets.

### q2tp-v2 (gate/up 2-bit, everything else q4tp)

1. Exports as above plus a q4tp file (escort hi file for the 2-bit tensors).
2. GPTQ into q2tp on 32–131 k tokens — **not in cortiq yet** (`quantize-gptq` has q1s / q4tp);
   the q2tp study's scripts: the placed ladder (per group the MSE-optimal scale instead of
   absmax/1.5) with the full 31-rung search at sweep time (W0), the super-weight layer's gate/up
   coded by cortiq's absmax rule (MiniCPM5 layer 7, max/rms 94), output debias (a per-row `lo`
   shift that undoes the MSE shrinkage, zero bits).
3. Escort with both hi files and greedy re-measure (one layer dominates at 2 bits):
   `escort.py --model q2tp.cmf --hi q4tp.cmf --hi q8_2f.cmf --greedy 2 --bpw 0.1 --write ...`
4. Metric genesis with `--step` — the largest single step at 2 bits.

| file | body bpw | wiki 160 w | source |
|---|---:|---:|---|
| plain `cortiq convert --quant q2tp` | 3.10 | 206.57 | study |
| zero-bit converter fix Q0pd (placed ladder + full rung search + weight-space debias, no data) | 3.10 | 89.46 | study |
| GPTQ-q2tp 32 k, cortiq rung rule (N1) | 3.10 | 37.08 | study |
| GPTQ on the placed ladder, full rung search (W0) | 3.10 | 31.67 | study |
| + layer-7 gate/up by the cortiq rule + debias, 131 k tokens (W128dh) | 3.10 | 25.62 | study |
| **+ metric genesis with step (R2)** | 3.10 | **21.89** | study 21.887; **tools 21.883** (−14.59 % ± 0.46 vs W128dh, study −14.58 % ± 0.46) |
| + escort +0.1 bpw, then genesis (R3) | 3.20 | 21.56 | study |

At 2 bits the genesis gain holds off-domain (−37 … −44 % on docs / code in the study); at 4
bits it is mostly in-domain (see above), so the 4-bit pass must be trained on a broad corpus.

### Qwen3.8-27B (`infosave/Qwen3.8-27B-cmf`, the planned run)

```
cortiq convert --model Qwen/Qwen3.8-27B --quant f16   --output q27-f16.cmf     # 54 GB teacher
cortiq convert --model Qwen/Qwen3.8-27B --quant q8_2f --output q27-q8_2f.cmf   # 27 GB calib/hi
cortiq quantize-gptq q27-f16.cmf --codec q4tp --calib-model q27-q8_2f.cmf --calib mix.txt \
    --tokens 32768 --window 2048 --hessians q27.hess --output q27-gptq.cmf
python tools/quant/escort.py --model q27-gptq.cmf --hi q27-q8_2f.cmf --corpus mix.ids,held=0:16 \
    --bpw 0.1 --residency stream --write q27-esc.cmf --cortiq cortiq
python tools/quant/metric_refit.py --model q27-esc.cmf --teacher q27-f16.cmf --out run27 \
    --corpus wiki.ids,train=384 --corpus c4.ids,train=384 --corpus code.ids,train=256 \
    --step --epochs 1 --residency stream --mb 8 --head-windows 8 --teacher-block 1024 \
    --write q27-v2.cmf --cortiq cortiq
python tools/quant/ppl.py --model q27-v2.cmf --ids wikitest.ids --windows 160 --also 12 --ref pw/q27-published
```

Check first that `quantize-gptq` reports all 400 linears of the hybrid (64 × 3 FFN + 48 × 3 GDN
in_proj_qkv / in_proj_z / out_proj + 16 × 4 attention; in_proj_a/b stay f16); compare against
the published q4tp (12 × 512: 8.793) on the same windows with `ppl.py --ref`.

Measured at 27B geometry without the model (`tests/scale_probe.py`: a random q4tp stack with
Qwen3.8-27B layer shapes, 3 GDN + 1 attention layers, streaming, on the shared 5090 under a
1.95 GB cap): compact weights 240 MiB per GDN layer / 233 MiB per attention layer; one training
step of one 512-token window peaks at 1.62 GiB of GPU memory with a 32k-row head (the full
248k-row q4tp head adds ~0.7 GiB) and takes 0.25–0.32 s per layer on the shared card; a
forward-only pass peaks at 0.92 GiB. Only one layer's activations are alive during the
backward, so micro-batch 8 stays far inside 32 GB (not measured: the shared card's budget).

## Tests

```
python -m unittest discover -s tools/quant/tests -v
python tools/quant/tests/cortiq_parity.py --cortiq target/release/cortiq --tokenizer-dir HF_MODEL_DIR --work /tmp/parity --refit
python tools/quant/tests/scale_probe.py --layers 4 --vocab 32768 --mb 1   # 27B-geometry memory / time
```
