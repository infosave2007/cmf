---
license: apache-2.0
library_name: cortiq
base_model:
  - tencent/Hy-MT2-1.8B
  - tencent/Hy-MT2-7B
  - tencent/Hy-MT2-30B-A3B
base_model_relation: quantized
pipeline_tag: translation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - q1t
  - ternary
  - translation
  - moe
  - hunyuan
  - 4-bit
language:
  - zh
  - en
  - fr
  - pt
  - es
  - ja
  - tr
  - ru
  - ar
  - ko
  - th
  - it
  - de
  - vi
  - ms
  - id
  - tl
  - hi
  - pl
  - cs
  - nl
  - km
  - my
  - fa
  - gu
  - ur
  - te
  - mr
  - he
  - bn
  - ta
  - uk
  - bo
  - kk
  - mn
  - ug
---

# Tencent Hy-MT2 — CMF (1.8B · 7B · 30B-A3B)

Tencent's [Hy-MT2](https://huggingface.co/collections/tencent/hy-mt2)
translation family — 33 languages, instruction-following translation
(terminology, style, placeholders, structured data) — as
[CMF](https://github.com/infosave2007/cmf) files for `cortiq`: one
memory-mapped file per model with the weights, the vendor tokenizer, the
exact chat template and per-tensor hashes. The same file runs on the CPU,
on Vulkan (NVIDIA / AMD / Intel), DX12 and Apple Metal; no Python, no
PyTorch, no CUDA toolkit.

```bash
cargo install cortiq-cli --version 0.7.0        # first release with HunYuan support

hf download infosave/Hy-MT2-cmf hy-mt2-7b-q4tp.cmf --local-dir .
cortiq verify hy-mt2-7b-q4tp.cmf
cortiq run hy-mt2-7b-q4tp.cmf --greedy --prompt "Translate the following text into English. Note that you should only output the translated result without any additional explanation:

Сегодня хорошая погода, и мы пойдём гулять в парк."
```

**Requires cortiq 0.7.0 or newer.** Earlier runtimes do not know the
`hunyuan_v1_dense` / `hy_v3` architectures (and 0.6.9 crashes on any
4-bit embedding — fixed in the same release).

## Files

| model | file | source | bits | size | wikitext-2 ppl¹ | MT ppl² |
|---|---|---|---|---:|---:|---:|
| Hy-MT2-1.8B | `hy-mt2-1.8b-q4tp.cmf` | bf16 checkpoint | 4.17 | 950.5 MB | 27.13 | 14.02 |
| Hy-MT2-1.8B | `hy-mt2-1.8b-q1t.cmf` | Tencent's 1.25-bit STQ (AngelSlim) | 2.25 | 694.9 MB | 4 460³ | 27.88 |
| Hy-MT2-7B | `hy-mt2-7b-q4tp.cmf` | bf16 checkpoint | 4.17 | 3.93 GB | 90.3⁴ | 98.6⁴ |
| Hy-MT2-30B-A3B | `hy-mt2-30b-a3b-q4tp.cmf` | bf16 checkpoint | 4.17 | 15.83 GB | 9.88 | 18.12 |

¹ Twelve 512-token windows of wikitext-2 test, exact attention, the same
yardstick every card on this account uses. The bf16-class reference for
the 1.8B (`q8_2f`, not published) scores 24.77 — the 4-bit file
sits within 10% (27.13 vs 24.77) of it. These are translation models: their
perplexity on English encyclopedia prose is not what they were trained
for, which is why column ² exists.

² Sixteen chat-formatted sentence pairs (RU/EN/ZH/DE/FR/ES/JA/IT/UK/PT,
1 073 tokens) in the model's own prompt format — the number that tracks
what the file is for. The `q1t` file is Tencent's own 1.25-bit
quantization-aware checkpoint carried over **exactly** (see below); it
is a translation-only specialist and its general-text perplexity is not
meaningful, but its translations are — see the samples further down.

³ Not a conversion defect: llama.cpp with the STQ1_0 kernel (PR #22836)
scores the very same GGUF at 1 793 on its own wikitext-2 protocol (first
twelve 512-token chunks, BOS-anchored, second halves scored), our
cold-window protocol at 4 460 — both say the 1.25-bit checkpoint no
longer models free English prose, while it translates correctly on every
backend. On the same Xeon that llama.cpp build decodes it at 2.2 tok/s;
cortiq's `q1t` kernels at 21.9.

⁴ The 7B uses HunYuan's older tokenizer (vocab 128 167, GPT-4-style
splitting, `<|startoftext|>` … `<|extra_0|>` chat markup) and is sensitive
to running without its BOS in the middle of a corpus; its translations are
the best of the three dense files. Compare sizes on column ², not here.

Every q4tp file was quantized straight from the bf16 safetensors,
streamed shard by shard; nothing here is a re-quantization of a
lower-precision file. The tied `lm_head` of the dense models is not
written twice: the embedding serves both ends, as in the source.

## What the runtime had to learn

Two things distinguish HunYuan from the Qwen/Llama block, and both ride
in the file header rather than in flags:

- **Dense 1.8B / 7B (`hunyuan_v1_dense`)** apply the per-head q/k RMSNorm
  *after* RoPE (`query_layernorm` / `key_layernorm` on the rotated
  vectors). That is not a reordering you can fold into the weights — the
  rotation preserves a head's norm but not its elementwise norm weights —
  so the engine carries an explicit `qk_norm_after_rope` flag through the
  CPU, Metal and WGSL rope kernels. Their "dynamic" NTK-alpha RoPE
  (`alpha = 1000`) is one rescaled base, `10000 · 1000^(128/126) =
  11 158 840`, applied at conversion; the full 262 144-token window is
  native, nothing is rescaled per position.
- **30B-A3B (`hy_v3`)** is 48 layers with layer 0 dense (6 912 wide) and
  47 sparse layers of 128 experts, top-8 by *sigmoid* score plus a
  selection bias (`expert_bias`, DeepSeek-V3 style: bias for the choice,
  unbiased scores for the weights), renormalized and multiplied by
  `router_scaling_factor = 2.826`, plus one always-on shared expert of
  768. The converter maps `mlp.router.gate` / `mlp.shared_mlp` onto the
  canonical layout and the header carries the router constants; the wgpu
  whole-token graph learned the two pieces it lacked — the routed scale
  and a shared expert without a gate — so the 30B decodes on the card
  as one graph (12 submits per token) instead of falling to the per-op
  path (145 submits, 1.2 tok/s) the way every sigmoid-scaled MoE did
  before 0.7.0.

## The 1.25-bit file, exactly

Tencent ships `Hy-MT2-1.8B-1.25Bit-GGUF`: a quantization-aware ternary
checkpoint in llama.cpp's `STQ1_0` type (256-weight blocks, one f16
scale, and in every group of four lanes exactly one zero and three ±1 —
5 bits per 4 weights). `cortiq import-gguf` decodes that block layout
natively and re-encodes each 32-group as CMF `q1t` (ternary, base-3
packed, f16 scale, empty outlier overlay): the reconstruction is
**bit-for-bit** the values the GGUF stores — no calibration, no second
quantizer. The cost of the general-purpose container is size: 2.25
bits/weight against 1.31, so the file is 695 MB against 462, with the
token table at `q8_2f` (Tencent stores it at 6.5 bits).

```bash
cortiq import-gguf Hy-MT2-1.8B-1.25Bit.gguf --quant q1t \
    --tokenizer-dir ./Hy-MT2-1.8B --output hy-mt2-1.8b-q1t.cmf
```

`--tokenizer-dir` matters: HunYuan's pre-tokenizer splits digits in runs
of one to three and isolates CJK before the byte-level step, which a
tokenizer rebuilt from GGUF metadata cannot express. The flag embeds the
vendor `tokenizer.json` and chat template verbatim, so the ternary file
tokenizes exactly like the 4-bit ones.

## Measured

Steady-state decode, single stream, `cortiq bench --core --tokens 128
--ignore-eos`, cortiq 0.7.0. The dense files are latency-bound on a
discrete card (a 1 GB model needs ~8 submits per token), so the CPU
matters as much as the GPU there; the MoE row is where the card counts.

| file | RTX PRO 4000 Blackwell 24 GB, Vulkan | Xeon E5-2690 v4 (14C), CPU | Apple M4 24 GB, Metal | M4, CPU (10 cores) |
|---|---:|---:|---:|---:|
| `hy-mt2-1.8b-q4tp.cmf` | 137.7 | 29.1 | 72.8 | 56.9 |
| `hy-mt2-1.8b-q1t.cmf` | 63.7 | 21.9 | 66.9 | 38.1 |
| `hy-mt2-7b-q4tp.cmf` | 77.2 | 9.3 | 23.1 | 16.7 |
| `hy-mt2-30b-a3b-q4tp.cmf` | 52.7 | 11.0 | —⁵ | — |

### The MoE on any card (dynamic loading)

The 30B's whole-token graph is not all-or-nothing: as many leading layers
as the VRAM budget admits stay resident on the card, the host finishes the
rest — one boundary crossing per token, no expert paging. Measured on the
RTX PRO 4000 with `CMF_GPU_VRAM_MB` capped to what each card size would
auto-detect (the CPU alone: 10.5 tok/s on this 14-core Xeon):

| VRAM budget | 4 GB | 6 GB | 8 GB | 12 GB | 16 GB | 24 GB |
|---|---:|---:|---:|---:|---:|---:|
| layers on the card | 7/48 | 14/48 | 20/48 | 33/48 | 45/48 | **48/48** |
| decode, tok/s | 9.9 | 11.1 | 14.5 | 21.9 | 39.2 | **53.7** |

Perplexity and the greedy continuation do not move along the ladder — the
split changes where a layer runs, never what it computes. `CMF_GPU_VRAM_MB`
overrides the auto-detected budget when you want to cap it by hand.

⁵ Not measured: the 15.8 GB file is at the edge of a 24 GB Mac (the 14.3 GB
Qwen3.8-27B decodes at 5.7 tok/s there) and on Metal a sigmoid-routed,
ungated-shared MoE layer still runs its experts on the CPU — the Metal
select kernel is the next port.

Prompt ingest (41-token prompt): on the RTX PRO 4000 the dense files take
180 (1.8B) and 118 (7B) tok/s; the 30B ingests at **8 tok/s** on this card
— its batched prefill re-stages the expert buffers per 32-token chunk
instead of sharing the decode graph's resident copy, so a long source
paragraph costs seconds before the first token. Decode is unaffected;
sharing the buffers is the next item on the MoE list. On the M4: 469 tok/s
for the 1.8B q4tp, 218 for the ternary file, 122 for the 7B.

**First answer vs. the rest.** On a discrete card the weights are uploaded
when the whole-token graph is first built — 18.5 GB for the 30B, ~26 s on
this box — and `cortiq run` folds that into its printed "decode" figure
for a one-shot prompt, which then reads a few tok/s. The per-token cost
after the upload is the table above (`cortiq bench --core` measures it
after an untimed warm-up). `cortiq serve` pays the upload once per
process, so every request after the first decodes at the steady rate.

## Prompting

The models have no system prompt. The chat template is embedded, so
`cortiq run` and `cortiq serve` wrap a plain user message correctly;
what goes into the message is Tencent's instruction, with the language
name written out in the prompt's language:

```
Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:

{source_text}
```

```
将以下文本翻译为{target_lang}，注意只需要输出翻译后的结果，不要额外解释：

{source_text}
```

Terminology, style, personalization, delimiter-preserving and
structured-data prompts are documented on the
[source card](https://huggingface.co/tencent/Hy-MT2-30B-A3B#hy-mt2-translation-task-instruction-examples-chinese-english-comparison).
Recommended sampling (Tencent): 1.8B / 7B — temperature 0.7, top-p 0.6,
top-k 20, repetition penalty 1.05; 30B-A3B — temperature 0.7, top-p 1.0,
no repetition penalty. `--greedy` is the deterministic choice for
evaluation.

Samples from the 1.8B files, greedy, identical on CPU and GPU:

| prompt | q4tp | q1t (1.25-bit) |
|---|---|---|
| RU→EN «Сегодня хорошая погода, и мы пойдём гулять в парк, а вечером посмотрим фильм.» | Today the weather is nice, and we're going to go for a walk in the park. In the evening, we'll watch a movie. | Today, the weather is nice. We'll go for a walk in the park, and in the evening, we'll watch a movie. |
| EN→DE "Please keep the {placeholder} and the number 12345 exactly as they are." | Bitte behalten Sie den {Placeholder} und die Zahl 12345 genau so, wie sie sind. | Bitte lassen Sie {placeholder} und die Zahl 12345 genau so, wie sie sind. |
| ZH→EN «今天天气真好，我们去公园散步吧。» | The weather is really nice today. Let's go for a walk in the park. | The weather is really nice today; let's go for a walk in the park. |

## Server

```bash
cortiq serve hy-mt2-7b-q4tp.cmf --port 8080
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"hy-mt2-7b-q4tp","temperature":0,"messages":[{"role":"user","content":"Translate the following text into French. Note that you should only output the translated result without any additional explanation:\n\nThe meeting is at ten."}]}'
```

`/v1/chat/completions`, `/v1/completions` and `/v1/models` are live;
`--ollama` adds an Ollama-compatible listener.

## Reproduce

```bash
cortiq convert --model tencent/Hy-MT2-1.8B    --quant q4tp --output hy-mt2-1.8b-q4tp.cmf
cortiq convert --model tencent/Hy-MT2-7B      --quant q4tp --output hy-mt2-7b-q4tp.cmf
cortiq convert --model tencent/Hy-MT2-30B-A3B --quant q4tp --output hy-mt2-30b-a3b-q4tp.cmf
```

Streaming from the hub: peak disk is the output file. Every file was
size- and hash-verified after upload (`cortiq verify`).

Weights derive from Tencent's release and remain under its Apache-2.0
terms. The CMF container and the cortiq runtime are Apache-2.0 as well
(see the repository's LICENSE and PATENTS.md).
