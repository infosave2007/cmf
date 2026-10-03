---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: cortiq
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - q2tp
  - q8_2f
  - mixed-precision
  - moe
  - 4-bit
  - 2-bit
language:
  - en
  - ru
  - zh
---

# Qwen3.8-Flash-Next q4tp + q2tp — universal CPU/GPU CMF

This repository contains the **text tower** of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
converted directly from the original bf16 checkpoint to one memory-mapped CMF
file. It runs with the same file on CPU, Vulkan (NVIDIA/AMD/Intel), DX12 and
Metal. No Python, PyTorch, CUDA toolkit or model-specific GPU build is needed
at inference time.

The upstream language model has 125B parameters with about 6B active per
token, plus the 51B-parameter PLE n-gram table. The sparse/on-demand layout is
why a single artifact can remain usable well below its full file size in VRAM.

```bash
cargo install cortiq-cli --version 0.6.4
hf download infosave/Qwen3.8-Flash-Next-cmf qwen38-flash-next-q4tp.cmf --local-dir .
# Or use qwen38-flash-next-q2tp.cmf for the smaller 2/4-bit expert profile.
cortiq verify qwen38-flash-next-q4tp.cmf
cortiq run qwen38-flash-next-q4tp.cmf --prompt "Explain QSA in plain language."
```

The file requires the dedicated `qwen4_exp` runtime added alongside this
conversion; older packaged binaries reject the new architecture instead of
silently running an approximate Transformer path.

## Device token path (cortiq ≥ 0.8.6)

Starting with cortiq 0.8.6 the same q2tp file runs a different decode path on
Vulkan cards: the whole token stays on the GPU. The four-stream
hyper-connection state, the GDN recurrent state, the QSA key/value and
indexer caches, the PLE history and every skeleton projection (q8_2f and f16)
are resident; routing happens on the card; resident experts come from the
global segmented arena; the only traffic per layer is the list of routed
experts the arena does not hold (admitted immediately and computed on the
card in the next frame) and one hidden vector. A frame carries up to eight
tokens with one dispatch per stage, so prefill runs in chunks of eight and a
speculative window verifies four positions in one pass. The old host path
remains the reference and the fallback (`CMF_QWEN_DEVICE=0`).

Measured on an RTX 4090 (RunPod, EPYC 7702, Vulkan, driver 580.159.04),
`cortiq bench --tokens 120 --core`, 41-token prompt, steady decode, arena
warmed from the routing profile. The host is a container with a 62 GB memory
cgroup, so the 77 GB file is never fully page-cached there and every
cold-expert admission that misses the cache is a disk read; the 12 GB and
16 GB rows are bound by that, not by the card.

| budget | plain decode | MTP decode | prefill (frames of 8) |
|---|---:|---:|---:|
| 21.5 GB (full card) | **41.9 tok/s** (33.9 over 300 tokens) | 41.4 tok/s | 47.9 tok/s |
| `CMF_GPU_VRAM_MB=16000` | 27.2 tok/s | 26.6 tok/s | 25.4 tok/s |
| `CMF_GPU_VRAM_MB=12000` | 16.3 tok/s | 15.2 tok/s | 19.2 tok/s |

On a 200-token story prompt (34 % draft acceptance) the full card gives
28.5 tok/s plain and 26.2 with MTP; on a 200-token Python task (69 %) 25.0
plain and 29.4 with MTP. Steady-state cold experts per token: 11 / 47 / 150 at
the three budgets; admitting them costs 3–4 / 12 / 35 ms per token on this
host (disk-bound under its 62 GB cgroup — the same 12 GB budget measured
24.4 tok/s two days earlier on the same pod with more of the file cached;
a host whose RAM holds the file admits at memcpy speed).

"Plain" is one token per frame; "MTP" is speculative decoding with the
multi-token-prediction head (below). On free-form text the MTP acceptance is
~34 % and the two are even; on code (~69 %) and on repetitive continuations
(~89 %) MTP is faster, because one window shares its expert admissions
between four positions.

The arena holds 8 304 experts at the full budget (all 8 163 of the profile),
5 080 at 16 GB and 2 680 at 12 GB (out of 24 576), plus 80 staging slots for
the in-frame cold passes; the skeleton takes 4.57 GB of it.

### Routing profile

A routing profile warms the arena from the first token:

```bash
# record one (any representative prompts), then start from it
CMF_QWEN_PROFILE_SAVE=flashnext.profile cortiq run qwen38-flash-next-q2tp.cmf --prompt "..."
CMF_QWEN_PROFILE=flashnext.profile cortiq serve qwen38-flash-next-q2tp.cmf --port 8080
```

`flashnext.profile` in this repository was recorded on English chat prompts;
it is a ranking of (layer, expert) hits, not weights, and does not change the
model's output.

### Multi-token prediction (speculative decoding)

The checkpoint ships an MTP head (`model.mtp.*`, one QSA+MoE layer with its
own mixer, 5.2 GB in bf16). cortiq converts it into a sidecar next to the
main file and uses it automatically when present:

```bash
# fetch only the 31 MTP tensors from the HF checkpoint (HTTP range requests)
python tools/qwen4_mtp_fetch.py --out ./mtp-src
# convert them into qwen38-flash-next-q2tp.mtp.cmf (1.08 GB)
cortiq convert --model ./mtp-src --quant q2tp --output qwen38-flash-next-q2tp.cmf --mtp-sidecar
```

Each round drafts `k = 3` tokens (`CMF_QWEN_MTP_K`), verifies them in one
four-position frame of the main model and keeps the longest accepted prefix;
greedy output is identical to plain decoding (checked token by token).
`CMF_QWEN_MTP=0` turns it off.

### Knobs

| variable | default | meaning |
|---|---|---|
| `CMF_GPU_VRAM_MB` | the card | weight budget; the arena takes what the skeleton leaves |
| `CMF_QWEN_PROFILE` / `CMF_QWEN_PROFILE_SAVE` | — | warm start from / record a routing profile |
| `CMF_QWEN_PREFILL_CHUNK` | 8 | tokens per prefill frame (1–8) |
| `CMF_QWEN_MTP` | on when the sidecar exists | speculative decoding with the MTP head |
| `CMF_QWEN_MTP_K` | 3 | drafts per round (≤ 7) |
| `CMF_QWEN_KV_RESERVE_MB` | 768 | VRAM kept free for caches and staging |
| `CMF_QWEN_STAGE_MB` | 256 | pinned staging ring per buffer for expert uploads (0 = plain `write_buffer`) |
| `CMF_MMAP_POPULATE` | 0 | fault the whole file in at open (hosts whose RAM holds it) |
| `CMF_QWEN_PROF` | 0 | one profile line per token (frame time, cold experts, admissions) |
| `CMF_QWEN_DEVICE_CHECK` | 0 | run the host path alongside and print logit cosine / argmax |

Parity: `CMF_QWEN_DEVICE_CHECK=1` runs both paths on every token and prints
the logits' cosine and argmax; over 160 positions of a free-form prompt the
cosine stays at 0.996–0.9995 and the argmax matches at every sampled
position (f32 accumulation on the card against f64 on the host).

## File

| file | quantization | size | tested decode |
|---|---|---:|---:|
| `qwen38-flash-next-q4tp.cmf` | mixed q4tp + q8_2f + f16 | 97.12 GB (90.45 GiB) | 4.71 tok/s core steady with the auto plan at a 16 GB GPU budget |
| `qwen38-flash-next-q2tp.cmf` | mixed q2tp + q4tp + q8_2f + f16 | 76.95 GB (71.66 GiB) | device token path above; 7.13 tok/s at a 16 GB budget on the 0.8.5 host path |
| `qwen38-flash-next-q2tp.mtp.cmf` | MTP head sidecar (same profile, `fc_embedding`/`fc_hidden` q8_2f) | 1.08 GB | speculative decoding; optional, picked up by name |
| `flashnext.profile` | routing profile (layer, expert) hit ranking | 98 KB | warm start of the expert arena, `CMF_QWEN_PROFILE` |

SHA-256 (q4tp): `601474afd6c7144dcfaf8e084cb2d2e786e06b4aeee3f20310ee0cae07224dfd`

SHA-256 (q2tp): `e84cb832124bf4df8b6c9b3e5daa1e8b0caa47187a240f3b45d72173fce9935b`

SHA-256 (q2tp MTP sidecar): `a9cc9760d57db47d7365efcb21bdc13bdfc910947f3a547832d1945f0b56350a`

This is a quality-oriented **q4tp profile**, not a uniform four-bit dump. The
memory wall (routed/shared MoE matrices and the large PLE n-gram embeddings)
uses q4tp. Always-active GDN/QSA projections and both vocabulary edges
(`embed_tokens` and `lm_head`) use q8_2f, whose input-channel field protects
outlier columns. Tiny routers, norms and Hyper-Connection gates remain f16.
This gives the always-active path and vocabulary boundaries an extra quality
margin while retaining the size and sparse-residency advantages of q4tp. The
profile contains 74,000 q4tp tensors, 172 q8_2f tensors and 751 f16 tensors.

The smaller **q2tp profile is not a uniform two-bit dump**. Following CMF's
established dynamic-MoE 2/4 layout, only the two SwiGLU input planes
(`gate_proj` and `up_proj`) of routed and shared experts use q2tp; expert
`down_proj` remains q4tp. The recurrent GDN/QSA/PLE projection skeleton and
both vocabulary edges remain q8_2f, while routers, norms and
Hyper-Connection gates remain f16. Keeping shared and routed expert planes in
the same physical layout also lets the existing per-layer dynamic GPU packs
handle them without a special-case stride. The published q2tp file contains
49,248 q2tp tensors, 24,752 q4tp tensors, 172 q8_2f tensors and 751 f16
tensors; all 74,923 payload hashes pass `cortiq verify`.

The conversion is tensor-streaming: source shards are downloaded, converted
and removed one at a time. The published CMF contains the tokenizer, chat
template, tensor directory and per-tensor hashes with the weights.

The release gate compares the first converted layer against the upstream BF16
operator (GDN cosine 0.99972; full layer/MoE cosine 0.99781), verifies every
one of the 74,923 tensor hashes, and runs deterministic CPU and constrained-
VRAM Vulkan prompts. Reference answers used for the smoke gate include `4`
for `2+2` and `Париж` for the capital of France. The q2tp release passed the
same two greedy CPU checks and produced a coherent free-form Russian answer.
Its 40-token core benchmark measured 4.01 tok/s steady decode; the planner
recorded zero GPU submissions, so that number is the CPU plan running on the
4090 pod rather than GPU-only throughput.

## What is implemented

This is not a generic-Transformer approximation. Cortiq has a dedicated
`qwen4_exp` forward for the release architecture:

- four-stream Gated Residual with groupwise zero-centered RMSNorm;
- 36 Gated DeltaNet and 12 Qwen Sparse Attention layers;
- QSA block indexer (4 query heads, one shared key, 2048-token budget);
- 512-expert MoE, top-10 routed experts plus the shared expert;
- deterministic bigram/trigram PLE lookup, signed-square-root gates and the
  dilated depthwise convolution;
- partial RoPE (64 of 256 dimensions) and the learned final stream mixer.

The checkpoint's vision tower is intentionally omitted, so this repository is
text-generation only. The 4B MTP predictor is also omitted: it is an optional
speculative-speed head and is not part of the trunk logits. The runtime does
not reinterpret it as the older Qwen/DeepSeek MTP format.

## Models larger than VRAM

CMF does not require the whole file to fit the GPU. Weights remain mmap-backed
in host memory; the runtime detects the adapter budget, keeps hot projections
resident, manages routed experts through a global segmented LRU cache, and
completes cold experts exactly on CPU. The cache protects a per-layer floor so
the 48-layer sweep cannot evict every useful expert from the preceding token;
the shared expert has its own pinned slot. The 51B-parameter n-gram table is
ideal for this layout: only the 16 rows selected for the current token are
read, and the table is not uploaded wholesale.

The runtime now selects the residency strategy by expert profile and available
GPU budget. Q2tp gate/up experts automatically use dynamic Vulkan on supported
cards with at least a 14 GB budget. Q4tp remains on the measured CPU/GPU auto
plan unless dynamic mode is explicitly forced: its cold expert payload is
twice as large and blindly using the GPU is slower. Unsupported/smaller GPUs,
Metal systems without this Vulkan kernel, and `CMF_GPU=0` all fall back to the
same exact CPU implementation; no second checkpoint is required.

On an RTX 5090 pod (driver 580.159.04) constrained with
`CMF_GPU_VRAM_MB=16000`, a warmed 100-token core run of q2tp measured **7.13
tok/s steady**. The model-wide cache held 5,952 expert slots / 9.94 GB and made
48 Vulkan submissions per token. The previous 55% reservation reached only
4,040 slots / 6.75 GB and 6.29 tok/s, so the new VRAM-scaled reservation is
about 13% faster at this 16 GB operating point. A/B runs at 70%, 75% and 80%
of the advertised budget measured 6.30, 7.13 and 6.75 tok/s respectively;
75% is therefore the default request, before the allocator's driver/workspace
reserve. First-sighting experts are not admitted immediately, preventing a
512-expert layer from churning the cache. Cache lookup itself uses a dense
layer/expert table and an O(1) free list rather than per-token hash and linear
slot searches.

On the same pod and 16 GB budget, q4tp measured **4.71 tok/s steady** with the
default auto plan versus **1.32 tok/s** when dynamic Vulkan was forced. Pure
CPU q2tp (`CMF_GPU=0`) completed the same inference path at 2.13 tok/s steady.
These are 100-token q2tp/q4tp and 60-token CPU core runs after mmap warm-up;
initial disk page-in is intentionally not reported as decode throughput.

With the same RTX 5090 exposed at its full 32 GB budget, two warmed 100-token
q2tp runs measured **6.58 and 6.62 tok/s steady** (6.60 tok/s representative).
The automatic arena held 12,448 slots / 20.79 GB. Limiting it to 5,744 slots /
9.59 GB measured 6.66 tok/s, within run-to-run noise: the larger arena improves
residency for varied expert traffic but does not accelerate this short fixed
prompt. Exact numbers depend on host CPU, RAM and PCIe bandwidth; more VRAM is
capacity, not a promise of higher single-stream throughput. The selection rule
still avoids the measured q4tp dynamic regression.

Useful controls:

```bash
cortiq gpu                                      # adapters and detected budget
CMF_GPU=0 cortiq run qwen38-flash-next-q4tp.cmf --prompt "CPU check"
CMF_GPU_VRAM_MB=12000 cortiq run qwen38-flash-next-q4tp.cmf --prompt "12 GB budget"
CMF_GPU_ADAPTER=0 cortiq run qwen38-flash-next-q4tp.cmf --prompt "GPU 0"
CMF_QWEN_DYNAMIC_MOE=1 cortiq run qwen38-flash-next-q4tp.cmf --prompt "force dynamic MoE"
CMF_QWEN_DYNAMIC_MOE=0 cortiq run qwen38-flash-next-q2tp.cmf --prompt "disable dynamic MoE"
CMF_QWEN_POOL_PCT=75 cortiq run qwen38-flash-next-q2tp.cmf --prompt "override pool request"
```

On headless Linux, if no Vulkan adapter appears:

```bash
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y libglvnd0 libgl1 libegl1 libvulkan1 vulkan-tools
export XDG_RUNTIME_DIR=/tmp
XDG_RUNTIME_DIR=/tmp vulkaninfo --summary | grep deviceName
XDG_RUNTIME_DIR=/tmp cortiq gpu
```

On RunPod, install that complete five-package set unconditionally before
probing. A partial Vulkan-only install can misleadingly report `Found no
drivers` even though `nvidia-smi` works.

Apple Silicon uses the same mixed q4tp/q8_2f file and Metal backend. Because unified-memory
sizes vary, keep enough free memory for the mapped file, QSA cache and system;
the build is checked on macOS, while exact Metal runtime measurements will be
added after testing this artifact on the target Mac.

## Sampling

Recommended settings from the upstream release:

| mode | temperature | top_p | top_k | min_p | presence penalty | repetition penalty |
|---|---:|---:|---:|---:|---:|---:|
| thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| instruct | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

The text runtime preserves the native 262,144-token context. QSA state stays
in host memory, so long contexts do not permanently consume VRAM; budget about
54 KiB of host RAM per cached token (roughly 1.7 GiB at 32k or 13.5 GiB at the
full context), plus roughly 112 MiB of fixed GDN recurrent state and the
memory-mapped model.

## Server

```bash
cortiq serve qwen38-flash-next-q4tp.cmf --port 8080
```

The server exposes `/v1/chat/completions`, `/v1/completions` and `/v1/models`
with an OpenAI-compatible request shape.

## Кратко по-русски

Это один CMF-файл с текстовой частью Qwen3.8-Flash-Next и встроенными
токенизатором/чат-шаблоном. Один и тот же файл запускается на CPU, Vulkan,
DX12 и Metal. Целиком загружать модель в видеопамять не требуется: Cortiq
сам определяет бюджет карты, держит горячие проекции и экспертов в VRAM, а
остальные веса читает через mmap из системной памяти. Для жёсткого лимита
используйте `CMF_GPU_VRAM_MB`, для CPU-проверки — `CMF_GPU=0`.

Профиль смешанный: GDN/QSA и оба края словаря сохранены в `q8_2f`, маленькие
роутеры/нормализации/Hyper-Connection — в `f16`, а основная масса MoE и PLE —
в `q4tp`. Это даёт запас точности постоянно активному пути, но не раздувает
весь артефакт до q8. Связность ответа обеспечивается точным `qwen4_exp`
оператором; `q4tp` сам по себе не является причиной потери связности.

В уменьшенном `q2tp` варианте два входных поля экспертов (`gate/up`) имеют 2
бита, выходное поле `down` остаётся `q4tp`, а GDN/QSA/PLE-проекции и края
словаря остаются `q8_2f`. То есть это смешанный профиль 2/4/8, а не
равномерное сжатие всей модели до двух бит.

На поддерживаемой карте с бюджетом от 14 ГБ q2tp автоматически включает
динамический Vulkan-кэш; на лимите 16 ГБ он показал 7,13 ток/с и занял 9,94
ГБ VRAM под 5 952 горячих эксперта. Для q4tp быстрее штатный CPU/GPU auto-план
(4,71 против 1,32 ток/с при принудительном dynamic на тестовом поде), поэтому
dynamic для него оставлен ручным. На CPU и неподдерживаемых GPU используется
точный fallback. Управление: `CMF_QWEN_DYNAMIC_MOE=0|1|auto`, жёсткий лимит
VRAM — `CMF_GPU_VRAM_MB`, доля запроса пула — `CMF_QWEN_POOL_PCT`.

При полном бюджете RTX 5090 32 ГБ два прогретых прогона дали 6,58 и 6,62 ток/с
(репрезентативно 6,60), а пул занял 20,79 ГБ под 12 448 слотов. Уменьшение пула
до 9,59 ГБ дало 6,66 ток/с — практически ту же скорость. Дополнительная VRAM
увеличивает резидентность экспертов на разнообразных запросах, но сама по себе
не ускоряет один короткий поток декодирования.

Реализованы именно новые Gated Residual, GDN, QSA, PLE и MoE из `qwen4_exp`.
Vision-башня в этот текстовый релиз не входит.

С cortiq 0.8.6 q2tp-файл на Vulkan-картах исполняется целиком на карте
(device-путь, раздел «Device token path» выше): гипер-состояние, рекуррентное
состояние GDN, кэши QSA и история PLE резидентны, роутинг на карте, кадр
несёт до восьми токенов. MTP-голова (4B) конвертируется в sidecar
`qwen38-flash-next-q2tp.mtp.cmf` и даёт спекулятивный декод с тем же жадным
ответом; выигрыш зависит от доли принятых черновиков (код — около 69 %,
свободный текст — около 34 %). `flashnext.profile` прогревает арену
экспертов с первого токена. Цифры — в таблице выше.

CMF runtime and format source: [infosave2007/cmf](https://github.com/infosave2007/cmf).
The original model remains subject to the
[Qwen Community License](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
