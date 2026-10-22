---
license: mit
library_name: cortiq
base_model:
  - ornith-ai/Ornith-1.5-122B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - moe
  - gdn
  - q4tp
  - q2tp
  - q8_2f
  - mixed-precision
  - text-only
language:
  - en
  - ru
---

# Ornith-1.5-122B — CMF

This repository contains standalone [CMF](https://github.com/infosave2007/cmf)
packages for the **text tower** of Ornith-1.5-122B, executed by the Rust
[`cortiq`](https://github.com/infosave2007/cmf) runtime. The artifacts were
converted from
[ornith-ai/Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B)
revision [`7681ef608624e7532d8164128bcffdbdbaf1b1af`](https://huggingface.co/ornith-ai/Ornith-1.5-122B/tree/7681ef608624e7532d8164128bcffdbdbaf1b1af).

> **Text only.** The upstream 122B checkpoint includes a visual tower; CMF
> intentionally packages only `model.language_model`. These files do not take
> image or video input and do not claim Ornith's vision-language behaviour.

> **Runtime requirement:** use [`cortiq` 0.8.30 or newer](https://github.com/infosave2007/cmf/releases/tag/v0.8.30).
> Earlier versions must not be assumed compatible with this Qwen3.5/GDN MoE
> text profile.

[Quick start](#quick-start) · [122B artifacts](#ornith-15-122b-artifacts) ·
[Measured Vulkan run](#measured-vulkan-run) · [Limits](#limits-and-integrity) ·
[Русская документация](#русская-документация)

## Quick start

The 122B `q2tp-v2` file is the smallest published 122B profile. Download one
model and its matching external checksum, then validate both transport and
CMF integrity:

```bash
REPO=infosave/Ornith-1.5-122B-cmf
MODEL=ornith-1.5-122b-q2tp-v2.cmf

hf download "$REPO" "$MODEL" "$MODEL.sha256" --local-dir .
sha256sum -c "$MODEL.sha256"                 # macOS: shasum -a 256 -c "$MODEL.sha256"
cortiq verify "$MODEL"

cortiq run "$MODEL" \
  --prompt "Explain the difference between a mutex and a semaphore." \
  --no-think --max-tokens 128
```

`cortiq verify` validates the CMF envelope, directory, and per-tensor hashes.
It is not a quality, memory-fit, or performance test. For a deterministic
smoke command use `--greedy`; it cannot be combined with sampling penalties.

## Ornith-1.5-122B artifacts

| file | exact bytes | SHA-256 | intended trade-off |
|---|---:|---|---|
| `ornith-1.5-122b-q2tp-v2.cmf` | 48,097,144,622 | `ebe80b9a6c112a7dd82fe31dca8058c7cc8e7fe03616f49e1431c1b8060ad01a` | lowest storage footprint; fastest of these three on the measured 24 GB Vulkan host |
| `ornith-1.5-122b-q4tp.cmf` | 67,904,220,974 | `fb4706070790e4b3fe1eda81ec7c3fb997a8653334092f0b4e741b6f4d26c13a` | higher-precision direct q4tp profile |
| `ornith-1.5-122b-q8_2f.cmf` | 125,055,889,198 | `44f92bfdbeb0ca94f1097cb85fc1eb6bce1ba55f405b2bd558cab4f778a1c429` | high-precision CMF comparison profile; not FP16/upstream ground truth |

All three are mixed-precision CMF files; their names are not claims that every
stored value has a uniform bit width:

- **`q4tp`**: routed and shared MoE matrices use `Q4TiledP`; the backbone and
  output matrices follow the hybrid `Q8_2f` policy where applicable, while
  norms and controls retain their required floating-point representation.
- **`q2tp-v2`**: routed/shared `gate_proj` and `up_proj` use `Q2TiledP`; the
  routed/shared `down_proj` remains `Q4TiledP`; backbone/head matrices follow
  the `Q8_2f` policy, with required F16/F32 controls retained.
- **`q8_2f`**: model matrices use `Q8_2f` subject to the same required
  floating-point controls.

### Why there is no `q4tp-v2` file

`q2tp-v2` names the current q2tp **encoder revision**. In contrast, a
`q4tp-v2` label is reserved here for a data-aware, calibrated/refit q4
workflow. The published 122B q4 artifact is a direct, non-data-calibrated
conversion, so it is deliberately named **`q4tp`**, not `q4tp-v2`. This avoids
misrepresenting a direct MoE conversion as a calibrated quality result.

## Architecture retained by the text conversion

| property | 122B CMF text target |
|---|---|
| model family | Qwen3.5 MoE / Gated DeltaNet hybrid |
| layers | 48 total: 36 Gated DeltaNet (linear-attention) + 12 full-attention |
| hidden / attention | hidden 3072; 32 query heads, 2 KV heads, head dimension 256 |
| MoE | 256 routed experts, top-8 selected per token, plus one always-on shared expert |
| MTP | one physical MTP block retained |
| CMF structure | 38,384 tensors; embedded tokenizer; CMF v2 |
| modality | text tower only; visual tensors are intentionally omitted |

The architecture table describes what was preserved in the text conversion. It
is not an upstream benchmark claim and does not validate vision, image/video
preprocessing, tool execution, or an external hosted API.

## Measured Vulkan run

This is a narrow, reproducible execution measurement, **not** a general
throughput promise or quality benchmark.

- Host: Ubuntu 24.04, NVIDIA RTX PRO 4000 Blackwell (24,467 MiB reported),
  driver 595.91.07; WGPU/Vulkan.
- Runtime: `cortiq` 0.8.30 release build.
- Command shape: `cortiq bench MODEL --tokens N --ignore-eos --core --json`,
  `CMF_GPU=wgpu`, `WGPU_BACKEND=vulkan`, no MTP; one stream.
- The runtime used a safe **device prefix** and a host tail, rather than
  claiming the full 48-layer model was GPU-resident.

| profile | generated tokens | steady decode tok/s | TTFT | device prefix | resident weight bytes |
|---|---:|---:|---:|---:|---:|
| `q2tp-v2` | 64 | **5.72** | 5.76 s | 13 / 48 layers | 13,075,174,400 |
| `q4tp` | 64 | **4.59** | 5.34 s | 9 / 48 layers | 12,925,678,592 |
| `q8_2f` | 32 | **1.04** | 20.23 s | 5 / 48 layers | 13,353,047,040 |

For every measured token the graph recorded a GPU prefix and no graph miss; it
did **not** record a full-model graph. The q2/q4/q8 files also completed a
bounded GPU-configured generation smoke. The `q2tp-v2` profile is therefore
the practical choice on this particular 24 GB Vulkan machine. Results depend
on driver, free VRAM, host RAM/storage, prompt length, selected experts, and
runtime version.

The same host was set up with the repository's Vulkan prerequisites:

```bash
apt-get update
apt-get install -y libglvnd0 libgl1 libglx0 libgles2 libegl1 libvulkan1 vulkan-tools
export XDG_RUNTIME_DIR=/tmp
vulkaninfo --summary
```

### MTP result

The MTP tensor block is physically present and CMF integrity verification
passed. On a CPU q4tp diagnostic, forcing `CMF_MTP=1` drafted 7 and accepted 5
tokens in a 16-token core bench; the short greedy answer's emitted text matched
the MTP-off run. That is a functional check, not a quality evaluation.

On the measured Vulkan prefix configuration, forced MTP was not beneficial:
its 64-token q2/q4 runs drafted 15 tokens and accepted 0, with 5.36 / 4.12
steady tok/s respectively. Leave MTP disabled for this exact Vulkan profile
unless it is re-measured on the target workload.

## Limits and integrity

- `cortiq verify` succeeded for all three 122B files (38,384 tensors each).
  Each profile also loaded and generated a bounded CPU smoke response; that
  only demonstrates basic execution, not broad answer quality.
- No held-out perplexity, task-suite score, or comparison with the original
  BF16 checkpoint is published here. `q8_2f` is a CMF comparison profile, not
  an FP16 reference.
- The 122B file sizes are about 44.79 GiB (`q2tp-v2`), 63.24 GiB (`q4tp`), and
  116.47 GiB (`q8_2f`) before operating-system and runtime headroom. They must
  not be presented as validated fits for a 24 GB unified-memory Mac.
- No Apple-silicon/Metal performance, capacity, or end-to-end quality result
  is claimed for these 122B files. Check `cortiq gpu`, available memory, and
  the complete target workload rather than extrapolating from Vulkan.
- Large-model runtime performance is route- and context-dependent. Do not
  raise undocumented GPU-memory limits to force a larger resident graph.

## License and attribution

The Ornith 1.5 source checkpoints listed in this card are released under the
[MIT license](https://huggingface.co/ornith-ai/Ornith-1.5-122B/blob/main/LICENSE).
This repository provides quantized, text-only derivatives. CMF and `cortiq`
are available from the [CMF project](https://github.com/infosave2007/cmf).

---

## Русская документация

Здесь находятся самостоятельные CMF-пакеты **текстовой башни**
Ornith-1.5-122B для Rust-runtime [`cortiq`](https://github.com/infosave2007/cmf).
Файлы сконвертированы из
[ornith-ai/Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B),
revision `7681ef608624e7532d8164128bcffdbdbaf1b1af`.

> Это только текстовая модель: visual tower исходного checkpoint намеренно не
> включена. Image/video input и исходное vision-language поведение не
> заявляются.

> Требуется [`cortiq` 0.8.30+](https://github.com/infosave2007/cmf/releases/tag/v0.8.30).

### Файлы 122B

| файл | точный размер | SHA-256 | назначение |
|---|---:|---|---|
| `ornith-1.5-122b-q2tp-v2.cmf` | 48,097,144,622 B | `ebe80b9a6c112a7dd82fe31dca8058c7cc8e7fe03616f49e1431c1b8060ad01a` | наименьший файл; самый быстрый из трёх на измеренном Vulkan-host с 24 GB |
| `ornith-1.5-122b-q4tp.cmf` | 67,904,220,974 B | `fb4706070790e4b3fe1eda81ec7c3fb997a8653334092f0b4e741b6f4d26c13a` | прямой профиль q4tp с более высокой точностью матриц |
| `ornith-1.5-122b-q8_2f.cmf` | 125,055,889,198 B | `44f92bfdbeb0ca94f1097cb85fc1eb6bce1ba55f405b2bd558cab4f778a1c429` | высокоточный CMF-профиль для сравнения, но не FP16/upstream ground truth |

Это mixed-precision файлы, а не равномерно 2-/4-/8-битные веса:

- `q4tp`: routed и shared MoE-матрицы — `Q4TiledP`; backbone/head по
  гибридной политике используют `Q8_2f`, а нормы и управляющие тензоры
  остаются в нужном floating типе.
- `q2tp-v2`: `gate_proj`/`up_proj` routed и shared experts — `Q2TiledP`,
  `down_proj` — `Q4TiledP`, backbone/head — `Q8_2f` с сохранением F16/F32
  controls.
- `q8_2f`: матрицы — `Q8_2f`, кроме обязательных floating controls.

### Почему нет `q4tp-v2`

Суффикс `q2tp-v2` означает текущую ревизию **q2tp encoder**. `q4tp-v2` здесь
зарезервирован для data-aware калибровки/refit. Опубликованный 122B `q4tp`
сконвертирован напрямую и такой MoE-refit пока не проходил, поэтому файл
честно называется `q4tp`, а не `q4tp-v2`.

### Архитектура

Для 122B сохранены: 48 слоёв (36 Gated DeltaNet + 12 full-attention), hidden
3072, 32 Q-head / 2 KV-head, 256 routed experts с top-8 и один shared expert,
а также один физический MTP block. В CMF 38,384 тензора и embedded tokenizer.
Это описание text conversion, а не заявление о vision, качестве или benchmark
upstream-модели.

### Измеренный Vulkan

Ubuntu 24.04, NVIDIA RTX PRO 4000 Blackwell (24,467 MiB), driver 595.91.07,
WGPU/Vulkan, `cortiq` 0.8.30, один stream, `bench --core`, MTP выключен:

| профиль | токенов | steady decode | TTFT | GPU prefix |
|---|---:|---:|---:|---:|
| `q2tp-v2` | 64 | **5.72 tok/s** | 5.76 s | 13 / 48 слоёв |
| `q4tp` | 64 | **4.59 tok/s** | 5.34 s | 9 / 48 слоёв |
| `q8_2f` | 32 | **1.04 tok/s** | 20.23 s | 5 / 48 слоёв |

Это безопасный device prefix с host tail; ни один результат не означает, что
все 48 слоёв целиком resident в GPU. Для всех измеренных токенов был GPU
prefix без graph miss. Скорость зависит от драйвера, VRAM, RAM/накопителя,
длины контекста и выбранных экспертов. На этом host практичнее `q2tp-v2`.

MTP-физически присутствует и прошёл integrity-проверку. CPU-диагностика q4tp
с `CMF_MTP=1` получила 7 drafted / 5 accepted в 16-token bench, но на данном
Vulkan prefix форсированный MTP дал 15 drafted / 0 accepted и снизил скорость.
Для этой конфигурации оставляйте MTP выключенным, пока не появится измерение
на вашей задаче.

### Проверка и ограничения

```bash
REPO=infosave/Ornith-1.5-122B-cmf
MODEL=ornith-1.5-122b-q2tp-v2.cmf
hf download "$REPO" "$MODEL" "$MODEL.sha256" --local-dir .
sha256sum -c "$MODEL.sha256"                 # macOS: shasum -a 256 -c "$MODEL.sha256"
cortiq verify "$MODEL"
```

`verify` успешно прошёл для всех трёх 122B-файлов; ограниченный CPU smoke
также загрузил и сгенерировал ответ. Это не является тестом качества. Пока не
опубликованы PPL, task-suite или сравнение с исходным BF16 checkpoint. Размеры
файлов — примерно 44.79 / 63.24 / 116.47 GiB ещё до headroom ОС и runtime,
поэтому для 24 GB Mac нет заявления о Metal-fit, скорости или end-to-end
качестве. Все перечисленные исходные Ornith checkpoints имеют
[MIT-лицензию](https://huggingface.co/ornith-ai/Ornith-1.5-122B/blob/main/LICENSE).
