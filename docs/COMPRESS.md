# Compressing a model for a task: `cortiq compress`

*Languages: **English** · [Русский](COMPRESS.ru.md) · [中文](COMPRESS.zh.md)*

One command turns a general model into a specialist for your tasks:

```bash
cortiq compress model.cmf prompts.jsonl model-task.cmf --ratio 0.75
```

- `model.cmf` is the source model (f16/bf16 or q8; a Hugging Face directory or a
  repo id also works);
- `prompts.jsonl` holds your task's prompts (one object per line with `prompt`
  or `messages`; the `calibration_prompts.jsonl` format works as is);
- `model-task.cmf` is the result;
- `--ratio 0.75` sets the target: the result weighs 75 % of the source (in the
  same codec).

What happens inside: a teacher answers your prompts. By default it is the model
itself; with `--teacher` it is a larger model, including one behind an API (see
below). The answers show which neurons the task uses ("illumination"). The
unused neurons leave, their contribution moves into the kept ones, and the
model then trains further on the answers. Quantization is a separate step if
you need an even smaller file: give a codec as the fourth argument (stage S5
below), or run `cortiq quantize-gptq` or `cortiq convert` afterwards.

While it runs you see what is going on and how much is left: the stage number,
its goal, counters with a rate and a "~N min left" estimate. An interrupted run
resumes with the same command, and finished stages are not recomputed.

The output is a `.cmf` that works right away in `cortiq run` and `cortiq serve`.
Next to it lies a quality report (`model-task.cmf.work/S6.eval/report.md`), and
the file header records its provenance (`provenance.compress`).

Additional options (not needed by default):

| Flag | Default | |
|---|---|---|
| `--teacher` | the model itself | who answers the prompts: another `.cmf` with the same tokenizer, or a model behind an OpenAI-compatible API (`openrouter:…`, `openai:…`, `api:…`) |
| `--api-base`, `--api-extra`, `--api-concurrency` | —, —, 8 | API teacher: the server address, extra request fields, requests in flight at once |
| `--gen-max-tokens` | 2048; 8192 with an API teacher | how many tokens one answer may take (reasoning included) |
| `--gen-samples` | 1 | answers per prompt |
| `--no-think` | | answer without reasoning |
| `--ffn F` / `--size 12B` | | a target as a fraction of the FFN neurons or as a parameter count, instead of `--ratio` |

Instead of prompts you can pass ready teacher data (`.tdata`, see below), for
example the answers of a larger model of the same family.

**MoE models** compress with the same command. Illumination counts how often
the router picks each expert on the task's answers. The file keeps only the
experts that gather the required share of the routes (the threshold is fitted
to `--ratio`), and the router works among the remaining ones as before. Example
(LFM2.5-8B-A1B, 80 coding prompts, `--ratio 0.8`): f16 16.96 GB → 13.57 GB, the
layers kept 20–30 experts of 32, test NLL 0.280 → 0.332 (a test of 2 records is
an indication, not a precise score). MoE models are not healed yet.

The design specification, with the reasoning behind the decisions and the
measured facts (in Russian):
[`TZ_TEACHER_DATA_COMPRESS.ru.md`](TZ_TEACHER_DATA_COMPRESS.ru.md).

## 1. Teacher data

### Generation

```bash
cortiq teacher-data generate TEACHER.cmf PROMPTS.jsonl OUT.tdata \
    [--no-think] [--max-tokens 8192] [--samples 1] [--split 90/5/5] [--seed 42]
```

- `PROMPTS.jsonl`: one object per line with `prompt` or `messages`, and
  optionally `id`, `category` (the split is stratified by it), `samples`,
  `max_tokens`, `meta`. The `calibration_prompts.jsonl` format works unchanged.
- The teacher is any `.cmf` with a chat template. Usually it is the same model
  in `q8_2f` (self-distillation) or a larger model of the same family, that is,
  one with the same tokenizer. The teacher can also be a model behind an API,
  see "A teacher behind an API".
- A reasoning model thinks by default. The boundary between reasoning and
  answer is taken exactly at the `</think>` token, without parsing text.
  `--no-think` switches to direct answers.
- If a generation falls into repetition (fewer than 5 % new 5-grams in the last
  512 tokens), it stops, and the record is marked `usable=false`.
- An interrupted generation resumes with the same command: finished records
  are kept, and the unfinished tail is cut off. The command refuses different
  parameters in the same directory.
- The split is by prompt: all answers to one prompt land in one split. No
  compression stage uses the test split; only the final scoring does.

### A teacher behind an API

A larger teacher does not have to be installed on a rented server: any model
behind an OpenAI-compatible `/chat/completions` will do. The answers arrive as
text and are turned into tokens by the tokenizer of the model you are going to
compress (`--tokenizer`; in `compress` it is the source itself).

```bash
# OpenRouter (key in OPENROUTER_API_KEY)
cortiq teacher-data generate openrouter:qwen/qwen3.8-27b prompts.jsonl task.tdata \
    --tokenizer model.cmf --sampling auto --no-think --api-concurrency 32
# OpenAI (OPENAI_API_KEY)
cortiq teacher-data generate openai:MODEL prompts.jsonl task.tdata --tokenizer model.cmf
# your own server: vLLM, SGLang, llama.cpp, Ollama, LM Studio, `cortiq serve`, …
# (key in CORTIQ_API_KEY if the server needs one)
cortiq teacher-data generate api:Qwen/Qwen3-32B prompts.jsonl task.tdata \
    --api-base http://gpu-box:8000/v1 --tokenizer model.cmf
# the same in one compress command
cortiq compress model.cmf prompts.jsonl model-task.cmf --ratio 0.8 \
    --teacher openrouter:qwen/qwen3.8-27b --no-think --api-concurrency 32
```

- The key is read only from the environment variable and is never written
  anywhere. The `.tdata` manifest records the model, the address and the
  decoding settings.
- Answers are appended to `OUT.tdata.answers.jsonl` as they arrive. An
  interrupted run resumes with the same command: finished answers are neither
  requested nor paid for again. The progress line shows the rate, the amount
  spent (if the server reports it) and the estimated time left.
- Reasoning: with OpenRouter it is switched by the `reasoning` field, with
  vLLM/SGLang/llama.cpp by `chat_template_kwargs.enable_thinking`. The reasoning
  text is taken from `reasoning`/`reasoning_content` or from `<think>…</think>`
  in the answer and placed before `</think>`, the way the student's template
  continues it. An answer cut off by the limit inside the reasoning is stored
  as a truncated record, without the closing tag and without the end token.
- Servers accept different sets of optional fields (`top_k`, `seed`,
  `max_tokens` or `max_completion_tokens`, reasoning switches). A field the
  server rejects is dropped once, with a message, and the work goes on. A wrong
  key or model stops the run at once, without a hundred identical errors.
- By default OpenRouter goes to the cheapest provider that serves the model in
  8 or 16 bits. Prices for one model differ by 2.5×, and some providers serve
  it in fp4. The provider of each answer is recorded in `meta.provider`. Set
  your own choice with `--api-extra`, for example
  `'{"provider":{"sort":"throughput"}}'`.
- `--api-extra '{…}'` adds fields to every request; a field set to `null` is not
  sent.

Measurements (Qwen3.8-27B via OpenRouter, coding prompts):

| | answers | tokens | price | time |
|---|---|---|---|---|
| with reasoning, limit 8192, 8 in parallel | 12 | 87.6k (9 hit the limit) | $0.19 | 4 min |
| `--no-think`, 12 in parallel, any provider | 12 | 22.0k (all complete) | $0.048 | 55 s |
| `--no-think`, the cheapest 8/16-bit provider | 12 | 20.8k (all complete) | $0.032 | 2 min |
| `--no-think`, 32 in parallel, any provider | 160 | 419k | $0.91 | 2 min |
| `--no-think`, 483 prompts × 2, 806 new (160 from the cache) | 966 (52 hit 8192) | 1.35M | $2.38 | 15 min |

**Use the tokenizer of the exact model you compress.** Models of one family
differ: Qwen3.8-27B's `tokenizer.json` has 7 more special tokens than
Qwen3.5-0.8B's (audio/TTS, ids 248070–248076) and its own
`generation_config.json` with `<|im_end|>` as the end of turn. S0 rejects data
tokenized by the smaller model by its vocabulary hash, and the answers would
end with the wrong token. If the model itself is not on the machine, a
tokenizer carrier is enough: a small model of the same family whose directory
holds the target's `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja` and `generation_config.json`. `cortiq convert` turns it
into a CMF that you pass as `--tokenizer`. The cached answers are then
re-imported without new requests.

The reasoning 27B thinks for longer than 8k tokens on three tasks of four. If
the compressed model will run without reasoning, `--no-think` is cheaper and
gives complete answers. If many answers hit the limit, the command suggests at
the end to raise the limit or to turn reasoning off.

### Importing an existing answer cache

```bash
cortiq teacher-data import train_combined.json OUT.tdata --tokenizer MODEL.cmf
```

It accepts an array of `[prompt, rendered ChatML]` pairs (the cache of the
DTG-MA notebooks) or JSONL with `prompt`/`messages` and `response`. The tokens
come from the `--tokenizer`, that is, the model you will compress. Repeats of
one prompt become samples. Keep in mind that such answers were written by
another model: they serve as text targets, not as this model's distribution.

### Checking

```bash
cortiq teacher-data inspect OUT.tdata --verify
```

## 2. Compression in detail

```bash
cortiq compress SOURCE PROMPTS|DATA OUTPUT [QUANT] --ratio R [options]
```

| Argument | What it is |
|---|---|
| `SOURCE` | a `.cmf` (f16/bf16; q8 with a warning), a local HF directory or a repo id |
| `PROMPTS` / `DATA` | a `.jsonl` of prompts (the model answers them itself) or a ready `.tdata` |
| `OUTPUT` | the final `.cmf`, in the source's codec |
| `QUANT` | optional: the output's codec (`q4tp`, `q2tp`, `q8_2f`, …), quantized in S5 |
| `--ratio` | the fraction of the source model's size (its parameters at the same codec) |
| `--ffn` / `--size` | alternatives to `--ratio`: a fraction of the FFN neurons or a target parameter count |

### Stages

Each stage runs as a separate process and is cached in `OUTPUT.work/` under
the key of its inputs. Running the same command again resumes from the first
unfinished stage. A changed parameter recomputes only its own stage and the
stages after it.

| Stage | What it does |
|---|---|
| S0 prepare | source → a working f16 copy + a q8_2f copy for calibration; check of the data's tokenizer |
| S1 illuminate | illumination: the source model runs over the teacher's answers (train and val separately); Grams of each layer's `down_proj` input, neuron energies, refit error curves for any width |
| S2 plan | layer widths: equal (`--allocate uniform`, the default), by the refit error curves (`curve`), or by those curves weighted by the layer's output energy (`abs`); the edge layers can stay at full width (`--protect-first N --protect-last M`, default 0: protection narrows the other layers further and hurts overall — on the 2B the fold gives 0.330 against 0.320 without it); widths are multiples of 128. On the 0.8B the uniform allocation gave NLL 0.621 against 0.629 for `curve` |
| S3 fold | the kept neurons are chosen by importance `‖W_down[:,j]‖²·⟨a_j²⟩`; the dropped ones are regressed onto the kept ones, `W_down' = W_S + W_P·Πᵀ`, `Π = (Σ_SS+εI)⁻¹Σ_SP`; the ridge and the refit itself are chosen on val (`--refit auto`). Computed on the GPU (see below) when the trainer is available |
| S4 heal | KD healing against the uncompressed source on the teacher's answers, with the loss on the generated tokens only and the best checkpoint by val. On NVIDIA (Vulkan, matrix units) and Apple Silicon (Metal, 8×8 simdgroup matrices) it runs on the GPU with the `nvg-train` kernels: the tail layers' weights train whole (f32 + Adam, the f16 planes updated every step), with KL to the source's top-64; the layers below are frozen and the backward pass does not reach them. Otherwise on the CPU: `0.3·CE + 0.7·KL`, windows inside records, generation gates against looping |
| S5 quant | only when `QUANT` is given (by default the output stays in the source's codec). `q4tp`/`q2tp`: GPTQ calibrated on the same data (windows inside records), full-attention k/v in q8_2f (`--escort kv`); other codecs go through the converter's encoder |
| S6 eval | on test: NLL of the teacher's answers, greedy continuations of the test prompts (looping, length relative to the teacher, share of self-terminated answers), speed, size; quality gates; `report.md` |

### Presets (`--recipe`)

| | `fast` | `balanced` (default) | `full` | `quant-only` |
|---|---|---|---|---|
| illumination + fold | ✓ | ✓ | ✓ | — |
| healing | — | FCD, 2 epochs | FCD, 3 epochs | — |
| GPTQ (only with `QUANT`), calibration tokens | 32k | up to 64k | up to 64k | up to 64k |

`quant-only` leaves the structure alone: it is GPTQ on the task data. It is a
convenient reference point for "what the structure gives over plain
quantization".

### Useful flags

| Flag | Default | |
|---|---|---|
| `--plan-only` | | show the plan and stop |
| `--until S3`, `--from S4` | | stop after a stage / recompute starting from a stage |
| `--heal-layers all\|deepN\|auto` | auto | which layers heal (auto: all if memory allows, otherwise the last ones that fit) |
| `--heal-epochs`, `--heal-lr`, `--kd` | see the right column, 0.7 | epochs: CPU 2 (`full` 3), GPU 6 (`full` 12); step: CPU 1e-5; GPU 1e-5 with a tail of up to 30 % of the layers, going down to 3e-6 when all layers train (0.8B, all layers: 1e-5 → 0.331, 5e-6 → 0.310, 3e-6 → 0.307); `--kd` applies to the CPU path only |
| `--heal-loss ce\|kd\|mix=A` | mix=0.5 | GPU healing loss: the teacher's text (CE), the source's distribution (KL), or A·CE + (1−A)·KL (0.8B at 75 % of the FFN: kd 0.303, ce 0.292, mix=0.5 0.281) |
| `--heal-patience`, `--heal-val-windows` | 6, 32 | GPU healing stops after this many held-out checks without improvement; each check reads this many val windows |
| `--heal-device auto\|gpu\|cpu` | auto | where to heal: auto = the GPU if the card has cooperative matrices (NVIDIA Vulkan or Apple Metal), otherwise the CPU |
| `--heal-params ffn\|all`, `--heal-window`, `--heal-topk` | ffn, 2048, 64 | GPU healing: which weights of a layer train (MLP only gives 0.307 against 0.310 for all weights on the 0.8B, at a third of the memory), the window length, how many teacher probabilities to keep per token |
| `--loss-on answer\|generated\|all` | generated | which tokens carry the loss |
| `--keep-grams auto\|on\|off` | auto | keep S1's Grams on disk for the fold (f32, 8·n² bytes per layer — 155 GB for a 27B); auto keeps them when they take less than half of the free space. With them, several targets (`--size 20B`, `17B`, `14B` in different `--work-dir`s sharing S0/S1) fold from one illumination; otherwise S3 repeats the illumination |
| `--ridge` | 0.03,0.1,0.3 | the grid of relative refit ridges (over 1130 layers 0.1 won 86 % of the time, 0.003 never) |
| `--calib-tokens`, `--escort` | 65536, kv | GPTQ calibration and the tensors kept in q8_2f |
| `--mtp-heal auto\|on\|off` | auto | retrain the MTP draft on the finished model (`cortiq mtp-heal`, a 3-step chain) when the source carries a draft and the GPU trainer runs |
| `--general-eval FILE` | | ppl on general text in the report (a check of forgetting) |
| `--compare MODEL.cmf` | | score another model on the same test records (another compression, a plain quant, another teacher); can be repeated |
| `--select importance\|dtgma` | importance | how FFN neurons are chosen: by illumination (one pass) or by a trained DTG-MA mask (Patent 2, see below) |
| `--write-on-fail` | | write the output even when the gates fail (`*.rejected.cmf`) |

### Choosing neurons with a DTG-MA mask (`--select dtgma`)

Instead of ranking by illumination, a mask σ(m) on every FFN neuron is trained
with the task's LM loss through the whole model under growing L1 pressure.
Every 25 steps the hard mask (σ > 0.5) is checked on val, and the first
checkpoint pruned to at least the target share is taken. The trained logits
pick the width of every layer (one common threshold) and the neurons
themselves. Folding with refit and healing then run as usual.

On NVIDIA the mask lives in the GPU trainer: a gate between SwiGLU and
`down_proj`, 0.37 s per step, the whole mask of the 0.8B in ~3 min (through
`skillbake` on the host: 16.7 s per step, 70 min). The pressure grows ×1.5 per
evaluation until the target; a linear growth stalled at 15 % on the 0.8B.

| Qwen3.5-0.8B | fold: illumination / DTG-MA | after healing: illumination / DTG-MA |
|---|---|---|
| FFN 75 % | 0.364 / **0.339** | 0.281 / 0.282 |
| `--ratio 0.9` | 0.394 / **0.372** | 0.305 / 0.310 |
| Qwen3.5-2B, FFN 75 % | 0.320 / 0.320 | **0.255** / 0.261 |

On the 0.8B the mask gives a better fold; after healing the results are equal,
or illumination is slightly better (2B). So illumination stays the default: it
is cheaper.

### GPU: NVIDIA and Apple Silicon

Illumination (S1), the refit curves (S1), DTG-MA masks (S2), the fold (S3),
healing (S4) and test NLL (S6) run on the GPU trainer. On NVIDIA this is Vulkan
with 16×16 tensor cores. On a Mac it is Metal through wgpu with Apple's 8×8
simdgroup matrices. The runtime on a Mac stays on native Metal, and the
trainer gets its own context. `CMF_TRAIN_GPU=0` turns the trainer off on a Mac.

A healing step of Qwen3.5-0.8B (2 windows × 2048 tokens, the MLPs of all
layers) on an M4:

| what is on | s/step |
|---|---|
| f32 fallback kernels for attention and GDN | 47 |
| + attention on 8×8 simdgroup matrices | 27.6 |
| + fast GDN kernels on SIMD groups | 17.4 |
| + a 64×64×32 GEMM tile (3.0 TFLOPS instead of 1.7) | 12.2 |

For comparison, an RTX 5090 takes 0.42 s per step. On a MacBook the step grows
to ~16 s as the machine heats up. The GEMM tops out at ~3 TFLOPS, close to the
practical limit of the M4 GPU, so a full healing of the 0.8B takes hours there.
For a short run use `--heal-epochs 1`.

**The fold on the GPU.** Cholesky in f32 on an equilibrated matrix (a unit
diagonal, as in on-device GPTQ: f32 then sees correlations rather than neuron
energies, and Qwen's outliers are ~10⁴ times the median), with the solves done
as block GEMMs. The correction `Aᵀ = (Σ_SS+εI)⁻¹·Σ_SP·W_Pᵀ` is solved for h
right-hand sides rather than p, as for `Π`. All errors are computed as traces
with `Σ_SP·W_Pᵀ` and `Σ_SS·Aᵀ`. The result matches the f64 host version: on
Qwen3.5-0.8B at 75 % of the FFN all 24 layers have the same val/train errors and
the same ridge. `CMF_FOLD_ON=host` brings back the host path.

| one layer | host (f64) | GPU, M4 |
|---|---|---|
| n 6144 → 4608, h 2048 (2B size) | 29 s | 2.2 s |
| n 17408 → 4352, h 5120 (27B, the 14B point) | ~11 min (scaled) | 12 s |
| n 17408 → 10496 (27B, the 20B point) | — | 33 s |
| refit curve, n 17408 | — | 14 s |

S3 as a whole on the 0.8B: 21 s against 121 s on the host.

**Illuminating a large model.** Grams that do not fit next to the weights are
accumulated over several passes. Each pass resumes from the residual stream
where the previous one stopped, which is kept on the host separately for train
and for val, so every layer is computed once. The room for the accumulators is
what the f16 weights and the working buffers leave. For a 27B on 96 GB that
means passes of about 25 layers.

### Quality gates (S6)

The output is compared with the source model on the same test prompts: no more
loops than the source plus one; a median answer length of at least 0.7 of the
teacher's; no fewer self-terminated answers than the source minus 10 % of the
prompts. If the gates fail, the output file is not written (exit code 5); the
report is always written.

## Examples (RTX 5090, healing on the GPU)

Data: the 483 prompts of `calibration_prompts.jsonl`, with the model itself as
the teacher (80 coding prompts for the MoE). Split 80/10/10. NLL is computed on
test, that is, on teacher answers the compression never saw. Generation uses 16
test prompts. The output is in the source's codec (f16).

| model | `--ratio` | parameters | test NLL: source → fold → healed | gates |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.9 | 0.773B → 0.697B | 0.189 → 0.394 → 0.305 | PASS |
| Qwen3.5-0.8B | 0.8 | 0.773B → 0.622B | 0.189 → 0.698 → 0.524 | FAIL: 3/16 loops |
| Qwen3.5-2B | 0.8 | 1.943B → 1.565B | 0.185 → 0.464 → 0.354 | PASS |
| LFM2.5-8B-A1B (MoE) | 0.9 / 0.8 / 0.65 | 8.47B → 7.62 / 6.77 / 5.49B | 0.280 → 0.289 / 0.332 / 0.399 (no healing) | — |

- A third of the 0.8B's parameters are embeddings, which are not compressed. So
  `--ratio 0.8` takes 30 % of the FFN width, and that is already too much.
- One 0.8B run on a 5090 takes about 6 minutes without data generation:
  illumination 36 s, fold ~1 min, healing ~1.5 min, scoring ~2 min.
- For comparison, on the 2B q4tp quantization alone gives 1.03 GB at NLL 0.225.
  Task compression does not replace quantization; it adds to it.
- A large model: Qwen3.8-27B → 14B on one RTX PRO 6000, with the published
  model, prompts and teacher dataset (see [Further reading](#4-further-reading)).

## 3. Teacher data in other commands

`.tdata` is accepted as a calibration corpus:
`PATH.tdata[:train|val|test|all][=SHARE]`.

```bash
cortiq quantize-gptq model-f16.cmf --codec q4tp --calib-model model-q8_2f.cmf \
    --calib task.tdata:train=0.8 --calib wiki.txt=0.2 --tokens 65536 --output model-q4tp.cmf
```

Windows are built inside records and never join two prompts.

## 4. Further reading

- [Compressing a language model for a task](research/TASK_COMPRESSION_ARTICLE.md):
  the method and the measurements on Qwen3.5-0.8B and 2B.
- [Compressing Qwen3.8-27B to 14B on one RTX PRO 6000, step by step](research/QWEN38_27B_TASK_COMPRESSION.md).
- The resulting model with its prompts and teacher dataset:
  [infosave/Qwen3.8-27B-Code-14B-Experimental-cmf](https://huggingface.co/infosave/Qwen3.8-27B-Code-14B-Experimental-cmf).
