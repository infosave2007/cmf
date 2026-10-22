---
license: apache-2.0
library_name: cortiq
base_model: RecursiveSI/flint
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - moe
  - q4tp
  - q2tp
  - vulkan
  - wgpu
language:
  - en
  - ru
---

# Recursive Flint — CMF

[Recursive Flint](https://huggingface.co/RecursiveSI/flint) is Recursive's
text-only mixture-of-experts model for coding agents: 39.8B total parameters
and about 2.5B active parameters per token according to the upstream release.
This repository packages revision
[`2a247be0c0aec895ccb0ac7ca92229ec6df37288`](https://huggingface.co/RecursiveSI/flint/tree/2a247be0c0aec895ccb0ac7ca92229ec6df37288)
as standalone [CMF](https://github.com/infosave2007/cmf) files for the Rust
`cortiq` runtime. Python and a machine-learning framework are not required to
run a downloaded CMF.

> **Use cortiq 0.8.30 or newer.** This is the first release that executes
> Flint's complete attention and routing profile: its mixed sliding/full
> attention schedule, no-RoPE full-attention layers, learned attention sinks,
> and normalized sigmoid routing. An older runtime can open an old CMF but is
> not a compatible Flint executor.

The package is a **text model**, not a general tool-execution environment.
The upstream model card describes tool use and reasoning; validate those
workflows, generated code, permissions, and output limits in the application
where the model will actually run.

[Quick start](#quick-start) · [Artifacts](#artifacts) ·
[Measured Vulkan run](#measured-vulkan-run) ·
[Русская документация](#документация-на-русском)

## Quick start

Install a `cortiq` release at or above 0.8.30, then download one profile and
verify both the transport checksum and CMF's per-tensor hashes:

```bash
REPO=infosave/flint-cmf
MODEL=flint-q2tp.cmf                         # fastest measured profile

hf download "$REPO" "$MODEL" "$MODEL.sha256" --local-dir .
sha256sum -c "$MODEL.sha256"                 # macOS: shasum -a 256 -c "$MODEL.sha256"
cortiq verify "$MODEL"

# Explicit WGPU graph path; Vulkan on the measured Linux host, Metal when WGPU
# selects Metal on Apple silicon.  --greedy is required for CMF_MULTISTEP.
CMF_GPU=wgpu CMF_GPU_WGPU_GRAPH=1 CMF_MULTISTEP=4 \
  cortiq run "$MODEL" \
  --prompt "Write a concise Python function merge_intervals(intervals) that merges overlapping intervals." \
  --no-think --greedy --max-tokens 128
```

For a higher-precision profile, replace `flint-q2tp.cmf` with
`flint-q4tp.cmf` in the download and run commands. `CMF_MULTISTEP` is an
optional greedy-only throughput setting, not a quality mode; it declines to
ordinary one-token stepping when its exactness conditions are not met. Do not
raise undocumented GPU-memory limits to force a larger graph.

The upstream release's sampled-generation settings are temperature **1.0**,
top-p **0.95**, and top-k **20**. They are embedded in the CMF provenance and
can be made explicit per run:

```bash
cortiq run flint-q4tp.cmf --prompt "Explain this repository's test strategy." \
  --temperature 1.0 --top-p 0.95 --top-k 20 --max-tokens 32768
```

For difficult work, the upstream card recommends an output budget of at least
32,768 tokens, including reasoning. That is a source recommendation, not a
claim that every task or context length has been evaluated in this runtime.

## Artifacts

| file | exact bytes | SHA-256 | precision policy | use it when |
|---|---:|---|---|---|
| `flint-q2tp.cmf` | 14,684,032,647 | `194e2d587d6619d344687dabebc6df90ce5ae7e9b3dd21a918d9c414b4d02141` | routed-expert `gate_proj` and `up_proj` are q2tp; routed `down_proj`, the shared expert, and the non-expert backbone retain the q4tp layout | memory/throughput is the priority; evaluate quality on the target workload |
| `flint-q4tp.cmf` | 20,992,265,863 | `086d2d16c510bdd24d8b29c08504bf403322ee1d7be38fc7525094e1f2c10703` | q4tp model-matrix profile; scalar, norm, and other non-matrix controls remain in their required floating representation | fidelity is the priority |

Both files carry the same tokenizer, template, model geometry, and source
revision. The q2tp profile is not a promise of q4tp-equivalent quality: it is
a deliberately lower-precision routed-expert profile. No perplexity score or
broad task evaluation is claimed here.

No mixed Q2/Q4 artifact is published in this release. `cortiq flint-hybrid`
can create a separately named, layer-aligned experiment from an exact paired
q2tp base and q4tp donor; it verifies both catalogues, copies selected routed
`gate_proj`/`up_proj` payloads without re-quantizing them, verifies the output,
and refuses to overwrite a file. It is not a substitute for a quality
benchmark and must not be represented as either published artifact above.

## What the runtime executes

The source configuration is preserved rather than approximated:

| property | Flint CMF execution profile |
|---|---|
| layers | 49 total: 2 dense-FFN layers followed by 47 MoE layers; **all 49 still have attention** |
| attention | 35 sliding-attention layers (window 513: current token + 512 preceding tokens) and 14 full-attention layers |
| position / sink semantics | RoPE only on the 35 sliding layers; the 14 full-attention layers are no-RoPE; every layer has a learned per-head `softmax_offset` attention sink |
| heads | 32 query heads, 4 key/value heads, head dimension 128; per-head QK RMSNorm |
| MoE | 256 routed experts, top-6 selected per token, routed width 512, plus an always-active shared SwiGLU expert of width 1024 |
| routing | sigmoid router scores; choose winners using the source router's bias-adjusted ranking, then normalize the selected unbiased scores with the source's `1e-20` denominator floor |
| vocabulary / configured context | 150,144 tokens / 262,144 tokens |

The configured context length is a model capacity field. It is not a measured
uniform-quality or full-GPU-context guarantee; memory use and performance
change with prompt length and hardware.

## Measured Vulkan run

The following is a reproducible, narrow measurement — **not** a general
throughput promise or a quality evaluation.

- Hardware/backend: RTX PRO 4000 Blackwell (24,467 MiB reported device
  memory), Linux Vulkan through WGPU.
- Build: the Flint-support release candidate containing the execution path
  described above.
- Command: the quick-start code prompt, `--no-think --greedy --max-tokens 128`,
  `CMF_GPU=wgpu CMF_GPU_WGPU_GRAPH=1`.
- Metric: decode tokens/s after the first token; one stream; a cold process.

| profile | `CMF_MULTISTEP` | decode tok/s | first token | observation |
|---|---:|---:|---:|---|
| q2tp | 0 | 82.0 | 11.8 s | one-token greedy baseline |
| q2tp | 4 | **107.5** | 11.9 s | response body was byte-identical to the q2tp `0` run for this prompt |
| q4tp | 0 | 10.5 | 2.5 s | one code-answer smoke run; not a general quality score |

The q2tp multistep result is an opt-in greedy measurement, not speculative
sampling and not a claim that every prompt will reach 107.5 tok/s. The q4tp
row deliberately does **not** claim a universal quality or speed relationship
from one answer. Device, driver, free VRAM, context length, output length,
and route locality all matter.

### Apple silicon / Metal status

The WGPU/Metal route-parity test passed 2/2 and compiled the relevant shaders
on an Apple M4. That is a backend compatibility check only, **not** an
end-to-end Flint qualification. A full q2tp trial on a 24 GB M4 exposed UMA
overcommit when WGPU copies the complete routed-expert banks into device
buffers; it can page heavily. The native-Metal fallback also does not yet
execute Flint's learned attention-sink path. Consequently, 0.8.30 makes no
Apple-silicon throughput or full-decode support claim for Flint. Use the
validated Vulkan configuration for this model today; do not treat the Vulkan
table as a Metal result or try to force a larger resident graph with memory
overrides.

## Integrity and legacy files

The current artifacts include a header-only correction that records Flint's
semantic metadata (sliding window, no-RoPE schedule, learned sinks, and
sigmoid routing). The tensor directory and quantized tensor payloads were
validated unchanged; complete-file checksums changed because the corrected
header was appended.

If you possess an older Flint CMF with the exact released tensor catalogue,
download the current artifact instead. As an auditable migration path,
0.8.30+ also provides:

```bash
cortiq repair-flint flint-q2tp.cmf --dry-run
cortiq repair-flint flint-q2tp.cmf
cortiq verify flint-q2tp.cmf
```

The command validates the exact Flint geometry, complete catalogue, dtype
profile, and source semantics before it appends metadata atomically. It
refuses look-alike, sharded, or semantically incompatible inputs. Recalculate
an external SHA-256 checksum after a local repair.

## Limits and responsible use

- Input and output are text. This card does not claim compatibility with an
  upstream hosted API, tool sandbox, or agent loop.
- Generated code can be insecure or incorrect. Run tests and review changes
  before deployment; enforce permissions outside the model.
- q2tp and q4tp need the complete model file on disk and substantial host/GPU
  memory for runtime state. Check free unified memory on a Mac and leave
  headroom for the OS and KV cache.
- The model's stated 262,144-token context configuration is not an assurance
  of uniform behavior, acceptable latency, or GPU fit at that length.

## License and attribution

The source weights are released by Recursive under
[Apache-2.0](https://huggingface.co/RecursiveSI/flint/blob/main/LICENSE).
This package derives from `RecursiveSI/flint` revision
`2a247be0c0aec895ccb0ac7ca92229ec6df37288`. CMF and `cortiq` are
[Apache-2.0](https://github.com/infosave2007/cmf).

---

## Документация на русском

[Recursive Flint](https://huggingface.co/RecursiveSI/flint) — текстовая
MoE-модель Recursive для coding agents: по данным исходного релиза 39.8B
параметров всего и около 2.5B активных на токен. Здесь revision
`2a247be0c0aec895ccb0ac7ca92229ec6df37288` упакован в самостоятельные
[CMF](https://github.com/infosave2007/cmf)-файлы для Rust-движка `cortiq`.
Для запуска скачанного CMF не нужны Python и ML-фреймворк.

> **Нужен cortiq 0.8.30 или новее.** В этой версии реализован полный
> профиль Flint: чередование sliding/full attention, 14 слоёв без RoPE,
> обученные attention sinks и нормированная sigmoid-маршрутизация. Старый
> runtime не является совместимым исполнителем Flint, даже если открывает
> старый файл.

### Быстрый старт

```bash
REPO=infosave/flint-cmf
MODEL=flint-q2tp.cmf

hf download "$REPO" "$MODEL" "$MODEL.sha256" --local-dir .
sha256sum -c "$MODEL.sha256"                 # macOS: shasum -a 256 -c "$MODEL.sha256"
cortiq verify "$MODEL"

CMF_GPU=wgpu CMF_GPU_WGPU_GRAPH=1 CMF_MULTISTEP=4 \
  cortiq run "$MODEL" \
  --prompt "Напиши компактную Python-функцию merge_intervals(intervals), объединяющую пересекающиеся интервалы." \
  --no-think --greedy --max-tokens 128
```

`CMF_MULTISTEP=4` — опциональное ускорение **только** для greedy-декодирования,
а не режим качества и не speculative sampling. При условиях, где невозможно
сохранить точность такого шага, runtime возвращается к обычной пошаговой
генерации. Не поднимайте недокументированные VRAM-лимиты ради большего graph.

Исходные настройки sampling: temperature **1.0**, top-p **0.95**, top-k
**20**. Для сложной задачи upstream рекомендует не менее 32,768 выходных
токенов, включая reasoning; это рекомендация автора модели, а не обещание
одинакового качества/скорости на любой задаче.

### Файлы

| файл | точный размер | SHA-256 | профиль | когда выбирать |
|---|---:|---|---|---|
| `flint-q2tp.cmf` | 14,684,032,647 B | `194e2d587d6619d344687dabebc6df90ce5ae7e9b3dd21a918d9c414b4d02141` | `gate_proj` и `up_proj` routed-экспертов — q2tp; `down_proj`, shared expert и backbone остаются в q4tp-layout | важнее память/скорость; обязательно проверяйте качество на своей задаче |
| `flint-q4tp.cmf` | 20,992,265,863 B | `086d2d16c510bdd24d8b29c08504bf403322ee1d7be38fc7525094e1f2c10703` | q4tp для основных матриц; скаляры, нормы и другие управляющие данные остаются в нужном floating-типе | важнее точность |

Оба файла содержат одинаковые tokenizer, template, геометрию и revision
исходника. q2tp — сознательный компромисс по точности routed experts, а не
обещание качества q4tp. PPL и широкий набор task-оценок для этих CMF здесь не
заявляются.

### Что исполняет runtime

- 49 слоёв: 2 dense-FFN и 47 MoE; **attention есть во всех 49 слоях**.
- 35 sliding-attention слоёв (окно 513) и 14 full-attention слоёв.
- RoPE применяется только в sliding слоях; 14 full слоёв без RoPE; на каждом
  слое есть learned per-head `softmax_offset` sink.
- 32 Q-head, 4 KV-head, head dim 128, per-head QK RMSNorm.
- 256 routed experts, top-6, routed width 512 и постоянно активный shared
  SwiGLU expert width 1024.
- Router использует sigmoid, bias-adjusted выбор top-6 и нормирование
  выбранных unbiased scores с denominator floor `1e-20` как в исходнике.

Заявленные 262,144 токена контекста — поле конфигурации модели, а не измеренная
гарантия качества, задержки или полного GPU-размещения на такой длине.

### Измерение Vulkan

Это узкий воспроизводимый замер, **не** общее обещание tok/s и не оценка
качества: RTX PRO 4000 Blackwell (24,467 MiB), Linux/Vulkan через WGPU,
один cold-process stream, тот же code-prompt, `--no-think --greedy
--max-tokens 128`, `CMF_GPU=wgpu CMF_GPU_WGPU_GRAPH=1`.

| профиль | `CMF_MULTISTEP` | decode tok/s | first token | наблюдение |
|---|---:|---:|---:|---|
| q2tp | 0 | 82.0 | 11.8 s | базовый greedy run |
| q2tp | 4 | **107.5** | 11.9 s | тело ответа для этого prompt byte-identical с q2tp/0 |
| q4tp | 0 | 10.5 | 2.5 s | один smoke-answer на код; не общий score качества |

Результат q2tp с multistep — opt-in greedy-замер, а не обещание 107.5 tok/s
на каждом prompt. Скорость зависит от GPU, драйвера, свободной VRAM, контекста,
длины вывода и локальности маршрутов.

На Apple M4 WGPU/Metal route-parity test прошёл 2/2 и shaders собраны. Это
лишь проверка совместимости backend, **не** end-to-end квалификация Flint.
Полный q2tp-прогон на M4 с 24 GB показал UMA-overcommit: WGPU копирует целые
банки routed experts в device buffers и может уйти в тяжёлый paging. Native
Metal fallback пока не исполняет путь Flint с learned attention sinks.
Поэтому для 0.8.30 не заявляются ни скорость, ни полная поддержка decode для
Apple silicon. Для этой модели используйте проверенную Vulkan-конфигурацию и
не пытайтесь форсировать больший resident graph memory-overrides; Vulkan-
цифры нельзя считать результатом для Mac.

### Целостность и старые файлы

Нынешние артефакты получили header-only correction семантики Flint; directory
и payload квантизованных тензоров проверены без изменения. Полный SHA-256
изменился из-за добавленного header. Лучше скачать файл из этой таблицы. Для
строго соответствующего старого артефакта доступен аудируемый путь:

```bash
cortiq repair-flint flint-q2tp.cmf --dry-run
cortiq repair-flint flint-q2tp.cmf
cortiq verify flint-q2tp.cmf
```

Команда проверяет точную геометрию, полный каталог, dtype-профиль и семантику
Flint, затем атомарно добавляет metadata; несовместимые/look-alike/sharded
файлы отклоняются. После локального repair пересчитайте внешний SHA-256.

### Ограничения и лицензия

Это text-to-text пакет, а не tool sandbox или готовый агентский контур.
Проверяйте сгенерированный код, запускайте тесты и сами ограничивайте
разрешения. Нужны полный CMF на диске и заметный запас RAM/VRAM; на Mac оставьте
headroom unified memory для ОС и KV cache.

Исходные веса Recursive выпущены под
[Apache-2.0](https://huggingface.co/RecursiveSI/flint/blob/main/LICENSE).
CMF и `cortiq` также [Apache-2.0](https://github.com/infosave2007/cmf).
