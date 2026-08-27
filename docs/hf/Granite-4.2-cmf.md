---
license: apache-2.0
library_name: cortiq
base_model:
  - ibm-granite/granite-4.2-3b
  - ibm-granite/granite-4.2-8b
  - ibm-granite/granite-4.2-30b
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - q8_2f
  - mixed-precision
  - granite
  - reasoning
  - 4-bit
  - 8-bit
language:
  - en
  - de
  - es
  - fr
  - ja
  - pt
  - ar
  - cs
  - it
  - ko
  - nl
  - zh
---

# IBM Granite 4.2 — Q4TP + Q8_2F universal CMF

This repository contains all three dense reasoning models from IBM's
[Granite 4.2 family](https://huggingface.co/blog/ibm-granite/granite-4-2):
3B, 8B and 30B, each converted directly from the original bf16 checkpoint to
Q4TP and Q8_2F CMF. A CMF is one memory-mapped file containing the weights,
tokenizer, exact chat template and integrity hashes. The same artifact runs on
CPU, Vulkan (NVIDIA/AMD/Intel), DX12 and Metal; inference needs no Python,
PyTorch, CUDA toolkit or model-specific GPU build.

```bash
# Granite 4.2 support is newer than cortiq 0.6.3. Until the next packaged
# release, build the checkout containing this support:
git clone --branch codex/granite-4-2 https://github.com/infosave2007/cmf
cd cmf
cargo install --path crates/cortiq-cli --features gpu

hf download infosave/Granite-4.2-cmf granite-4.2-3b-q4tp.cmf --local-dir .
cortiq verify granite-4.2-3b-q4tp.cmf
cortiq run granite-4.2-3b-q4tp.cmf --prompt "Explain why the sky is blue."
```

Use `--no-think` for a direct answer. Without it, the embedded Granite
template opens the model's native `<think>...</think>` reasoning block.

## Files

| model | file | profile | size | A40 Vulkan steady decode |
|---|---|---|---:|---:|
| Granite 4.2 3B | `granite-4.2-3b-q4tp.cmf` | q4tp body; q8_2f vocabulary edges | 2.17 GB | **89.5 tok/s** |
| Granite 4.2 3B | `granite-4.2-3b-q8_2f.cmf` | full q8_2f | 3.68 GB | **60.6 tok/s** |
| Granite 4.2 8B | `granite-4.2-8b-q4tp.cmf` | q4tp body; q8_2f vocabulary edges | 4.98 GB | **49.9 tok/s** |
| Granite 4.2 8B | `granite-4.2-8b-q8_2f.cmf` | full q8_2f | 8.81 GB | **30.9 tok/s** |
| Granite 4.2 30B | `granite-4.2-30b-q4tp.cmf` | q4tp body; q8_2f vocabulary edges | 15.64 GB | **22.9 tok/s** |
| Granite 4.2 30B | `granite-4.2-30b-q8_2f.cmf` | full q8_2f | 29.31 GB | **5.28 tok/s** (55/64-layer dynamic prefix) |

Numbers are single-stream `cortiq bench --tokens 100 --core --ignore-eos`
steady decode on an NVIDIA A40. They exclude model load and prompt prefill.
The 3B Q4TP run prefills at 80.3 tok/s and reaches first token in 0.58 s; its
Q8_2F control prefills at 47.5 tok/s and reaches first token in 0.93 s.
The 30B Q8_2F file is larger than the A40's safe physical graph live set:
Cortiq automatically kept 55 of 64 layers on Vulkan and ran the nine-layer
tail on the host. That 100-token run peaked at 43,036 MiB of the A40's
46,068 MiB and completed without steady-state weight uploads.

SHA-256 (the Hugging Face LFS object id is the same digest):

```text
1a15aa98fe0ef104587f32ceb139cba13820a8c49fb5c28ad9c01181c7c18166  granite-4.2-3b-q4tp.cmf
f35f6536b6f08275638472fb66e265028fd06d47214fd36b37784df4b4595677  granite-4.2-3b-q8_2f.cmf
2f1f9f76f92f3b8b28e6f5e7c00f037c81d6b208f01a00be37f5ae6f4c573bf7  granite-4.2-8b-q4tp.cmf
743180ec51d38a1ae82f44a2dbc550c9dd510050eada387575b752eaec58cb10  granite-4.2-8b-q8_2f.cmf
0a81d4177def5dfa273075fe8df5bdb6b2874665145af03914ead7ad32bff70e  granite-4.2-30b-q4tp.cmf
1c58c47656fd7102cbafaf516993e4e86289a3c4c99bdffd314a1f1fffb13acf  granite-4.2-30b-q8_2f.cmf
```

Q4TP is deliberately a **mixed quality profile**, not a uniform four-bit
dump. Every Transformer projection uses q4tp, while the input embedding and
untied `lm_head` use q8_2f. Granite has a 100,352-token vocabulary; protecting
both vocabulary boundaries costs relatively little and avoids turning rare
token rows or input-channel outliers into the weakest part of the model.
Norms remain f16. The 3B and 8B Q4TP files each contain 280 q4tp, 2 q8_2f and
81 f16 tensors. The Q8_2F variants keep every projection and both vocabulary
edges in the two-field 8-bit layout (282 q8_2f plus 81 f16 tensors).

Q8_2F is the high-quality control and the preferred artifact when memory
allows. Its second scale field protects input channels in addition to the
usual per-output-row scale. Q4TP is the faster, smaller operating point for
cards and machines where bandwidth or capacity matters more.

## Exact Granite 4.2 runtime

Cortiq maps `GraniteForCausalLM` explicitly rather than treating it as an
ordinary Llama approximation:

- the checkpoint's `attention_multiplier` is applied exactly (Granite uses
  `1 / head_dim`, not the common `1 / sqrt(head_dim)`);
- `embedding_multiplier`, `residual_multiplier` and `logits_scaling` are
  validated and mapped into the CMF header;
- GQA geometry, SwiGLU, RMSNorm, RoPE theta and untied embeddings are
  preserved per model size;
- the complete upstream Jinja chat template is embedded, including thinking,
  non-thinking, low-effort reasoning, tools and multi-turn history handling.

The release tests compare exact rendered prompts against the upstream
template. With thinking enabled the assistant prefix ends in
`<|im_start|>assistant\n<think>\n`; `--no-think` produces
`<|im_start|>assistant\n<think></think>` exactly.

The CMF header keeps Granite's native **131,072-token** context. IBM documents
a long-context extension to 512K, but these files do not advertise 512K until
that extension is implemented and measured in Cortiq.

## CPU, constrained VRAM and Metal

The files never require a GPU. `CMF_GPU=0` selects the exact CPU path; on the
96-thread Xeon 6342 test pod the 3B Q4TP control measured 11.9 tok/s steady.
Vulkan uses a whole-token graph when the model fits its detected physical
budget. For a larger dense model it derives a fixed GPU layer prefix from the
actual tensor directory, uses the same boundary during batched prefill, and
runs the remaining layers on CPU. The KV mirror starts at 512 positions and
grows geometrically instead of allocating the advertised 131k context at the
first token. Weights remain mmap-backed in system memory, so a file can run
when it is larger than VRAM instead of failing at load time or cycling the
whole model through the driver's allocation cache.

```bash
cortiq gpu
CMF_GPU=0 cortiq run granite-4.2-8b-q4tp.cmf --prompt "CPU check"
cortiq run granite-4.2-30b-q4tp.cmf --prompt "automatic VRAM split"
CMF_GPU_VRAM_MB=13500 cortiq run granite-4.2-30b-q4tp.cmf --prompt "manual 16 GB ceiling"
CMF_GPU_ADAPTER=0 cortiq run granite-4.2-8b-q8_2f.cmf --prompt "GPU 0"
```

The constrained-card gate was run on the A40 with the weight budget reduced
to emulate a 16 GiB device. Granite 30B Q4TP answered correctly with a 13,500
MiB budget and a measured 15,384 MiB process peak. Granite 30B Q8_2F answered
correctly with a 14,000 MiB budget and a 14,616 MiB peak. On a real 16 GiB
Vulkan adapter the automatic detector reserves 2.5 GiB before choosing the
prefix; `CMF_GPU_VRAM_MB` remains available for containers whose driver does
not report its heap or for a stricter manual ceiling.

The same files and source build for Apple Silicon/Metal. Granite's custom
attention scale is carried explicitly by the single-token, split-context and
batched-attention kernels; no path silently substitutes the usual scale. Both
3B artifacts were downloaded back from this public repository and passed
every payload hash. On a 10-core Apple M4 MacBook Air, Q4TP measured **34.8
tok/s** steady decode, **166.0 tok/s** prefill and **0.57 s** TTFT. Q8_2F
measured **11.2 tok/s**, **95.7 tok/s** prefill and **0.47 s** TTFT in the
first cool 100-token run; the final release correctness run answered `4` at
12.0 tok/s. The optimized native graph keeps all 40 layers on Metal. An
experimental four-row Q8 matvec was neutral-to-slower under alternating runs
on the fanless Air, so the faster stable one-row kernel remains the default
(`CMF_Q8_R4=1` is retained only as an opt-in measurement arm).

On a headless Linux pod, install the complete Vulkan loader set before
probing:

```bash
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y libglvnd0 libgl1 libegl1 libvulkan1 vulkan-tools
export XDG_RUNTIME_DIR=/tmp
vulkaninfo --summary
cortiq gpu
```

## Integrity and conversion

Every published artifact passes both envelope validation and every per-tensor
payload hash:

```bash
cortiq verify granite-4.2-30b-q4tp.cmf
cortiq info granite-4.2-30b-q4tp.cmf
```

The conversion is reproducible and needs no Python:

```bash
cortiq convert --model ibm-granite/granite-4.2-3b \
  --quant q4tp --output granite-4.2-3b-q4tp.cmf
cortiq convert --model ibm-granite/granite-4.2-3b \
  --quant q8_2f --output granite-4.2-3b-q8_2f.cmf
```

Remote conversion streams source shards instead of holding a second full
checkpoint in RAM. The release additionally exercises CPU and Vulkan greedy
answers, a real 100,352-row Q8_2F head parity test, and the cache transition
from a body-only Q8 GEMM to the full packed Q8_2F graph payload.

## Server

```bash
cortiq serve granite-4.2-8b-q4tp.cmf --port 8080
```

The server exposes OpenAI-compatible `/v1/chat/completions`,
`/v1/completions` and `/v1/models` endpoints.

## Кратко по-русски

В репозитории лежат Granite 4.2 3B, 8B и 30B в двух вариантах. Q4TP —
практичный смешанный профиль: тело трансформера четырёхбитное, а embedding и
`lm_head` оставлены в q8_2f для защиты краёв словаря. Q8_2F — полный
восьмибитный контроль качества. Один файл работает на CPU, Vulkan, DX12 и
Metal; при нехватке VRAM веса остаются mmap в системной памяти и runtime
выбирает переносимый fallback.

Поддержка не подменяет Granite обычным Llama: учтён нестандартный масштаб
attention `1/head_dim`, встроен точный официальный chat template с режимами
thinking/`--no-think`, сохранены GQA, RoPE, SwiGLU и раздельные входной и
выходной словари. На A40 вариант 3B Q4TP показал 89,5 ток/с steady, Q8_2F —
60,6 ток/с. Оба варианта 3B скачаны обратно с публичного HF и прошли проверку
всех хешей. На 10-ядерном Apple M4 Q4TP показал 34,8 ток/с decode и 166,0
ток/с prefill, а Q8_2F — 11,2 ток/с decode и 95,7 ток/с prefill; все 40 слоёв
исполняются в нативном Metal-графе. Для больших файлов Vulkan автоматически
выбирает постоянный GPU-префикс и CPU-хвост: проверка, имитирующая 16 ГБ,
дала пики 15 384 MiB для 30B Q4TP и 14 616 MiB для 30B Q8_2F, оба ответа
остались корректными.

Original models and license:
[ibm-granite/granite-4.2-3b](https://huggingface.co/ibm-granite/granite-4.2-3b),
[8b](https://huggingface.co/ibm-granite/granite-4.2-8b),
[30b](https://huggingface.co/ibm-granite/granite-4.2-30b) — Apache-2.0.
CMF runtime and format source:
[infosave2007/cmf](https://github.com/infosave2007/cmf).
