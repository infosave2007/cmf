---
license: other
license_name: tencent-hy-license
license_link: LICENSE
library_name: cortiq
base_model: tencent/Hy4-flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - q2tp
  - vulkan
  - moe
  - engram
  - hunyuan
  - hy4-flash
  - cuda
language:
  - en
  - zh
  - ru
---

# Hy4-flash — CMF

[Tencent Hy4-flash](https://huggingface.co/tencent/Hy4-flash) (released 2026-11-17, 220B parameters, 12B active) as
single [CMF](https://github.com/infosave2007/cmf) files for `cortiq`, a Rust inference engine with no Python and
no ML framework. The model routes 16 of 512 experts on a 1536-wide latent, mixes 30 sliding-window and 12 sparse
attention layers, keeps four residual streams, and looks up bigram and trigram tables (Engram) on four layers.

- **RTX PRO 6000 (96 GB), CUDA:** `q2tp` answers at **65.1 tok/s** and `q4tp` at 55.3 tok/s after a
  1 000-token prompt, which the card reads in 4 s.
- **Within 24 GB of the card:** the experts live in host memory and the card keeps the hot ones: `q2tp`
  37.4 tok/s, `q4tp` 22.7 tok/s, 23.7 GB in use at the peak.
- **Vulkan / DX12 / Metal:** the same files decode on the card through wgpu: `q2tp` 45.1 tok/s, `q4tp`
  37.4 tok/s on the same card.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Hy4-flash-cmf hy4-flash-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq run hy4-flash-q4tp.cmf --prompt "Explain in three sentences why the sky is blue." --greedy --no-think
```

These files need cortiq 0.9.6 or newer. The fast path is CUDA (`--features cuda`: CUDA 12.x `libnvrtc` and
`libcublas` at run time, no CUDA toolkit at build time). Vulkan, DX12 and Metal (a plain `cargo install cortiq-cli`,
`CMF_GPU=wgpu`) run on the card too, with two limits for now: the prompt is read in 16-token chunks, and past
2 048 tokens of context the sparse layers' indexer runs on the host. `--no-think` asks for a direct answer.

**Memory.** The file is mapped, not loaded. On the card: the attention, the shared experts and as many routed
experts as fit. In host memory: a copy of the routed experts in the card's layout (up to 86 GB, the RAM
tier) and the Engram tables, read from the page cache (63 GB). With less RAM the engine keeps a smaller tier and
reads the rest from the file — slower on every expert it misses.

## Files

| file | experts (gate/up · down) | the rest | size | use |
|---|---|---|---:|---|
| `hy4-flash-q4tp.cmf` | 4-bit · 4-bit | 8-bit, Engram 8-bit as released | 150.5 GB | **default** |
| `hy4-flash-q2tp.cmf` | 2-bit · 4-bit | 4-bit, Engram 8-bit as released | 122.8 GB | smaller and faster: more experts fit the card |
| `hy4-flash-q2tp-sfq.cmf` | 2-bit · 4-bit, SFq | 4-bit, Engram 8-bit as released | 122.8 GB | code: `q2tp` with the experts calibrated |

`q4tp` and `q2tp` are converted from the FP8 release (e4m3 with 128×128 block scales, decoded exactly) with the bf16
attention shards of `tencent/Hy4-flash` on top. The vision tower travels in the files; cortiq reads text only
for now. `hy4-flash-q2tp-sfq.cmf` is `q2tp` with SFq on its routed experts: a channel scale on the latent and
GPTQ with a code refinement, calibrated on 98 304 tokens of English, Chinese and Rust text (`cortiq hy4-sfq`). It
reads code better and prose worse than the plain `q2tp`.

## Quality

| file | WikiText-2 | Chinese docs | Rust source |
|---|---:|---:|---:|
| `q4tp` | 35.80 | 39.13 | 3.92 |
| `q2tp` | 32.98 | 36.08 | 4.64 |
| `q2tp-sfq` | 42.20 | 41.83 | 4.30 |

Perplexity, windows of 512 tokens on the card's prompt graph: WikiText-2 test (16 windows), Chinese
documentation (8), Rust source (8). The 2-bit experts cost code more than prose; SFq gives code back.

## CUDA · RTX PRO 6000

Stand: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), the CUDA backend (`CMF_GPU=cuda`), cortiq 0.9.6, a
release build. `cortiq run --greedy --no-think`, 256 tokens after a 1 000-token WikiText-2 prompt, the file already
in the page cache (a second run); `CMF_GPU_VRAM_MB` caps the card.

| limit (`CMF_GPU_VRAM_MB`) | `q2tp` decode, tok/s | first token, s | `q4tp` decode, tok/s | first token, s | peak in use `q2tp` / `q4tp`, GB |
|---:|---:|---:|---:|---:|---:|
| 96 GB (whole) | **65.1** | 15.5 | 55.3 | 19.7 | 60.5 / 86.5 |
| 48 GB | **53.1** | 18.0 | 37.8 | 20.8 | 45.6 / 45.8 |
| 24 GB | **37.4** | 19.6 | 22.7 | 23.8 | 23.6 / 23.7 |
| 16 GB | **31.2** | 19.9 | 19.9 | 24.4 | 16.6 / 17.2 |
| 15 GB | **29.9** | 20.0 | 19.0 | 23.1 | 15.7 / 15.8 |
| 12 GB | **26.0** | 19.9 | — | — | 13.1 / — |

![Decode by card size](https://huggingface.co/infosave/Hy4-flash-cmf/resolve/main/assets/perf_cuda_en.png)

The first token is timed from a cold process: it includes building the plan and filling the RAM tier (about
15 s, once a process); reading the 1 000-token prompt itself takes 4 s on the whole card.
`CMF_GPU_VRAM_MB` caps what the engine places; the process peaks up to 1.2 GB above it (the peak column). On a 16 GB card set `CMF_GPU_VRAM_MB=15360`: both files then stay under 16 GB. Under a 12 GB limit `q4tp`'s expert slots do not fit (its prompt falls to the host) and `q2tp` peaks at 13.1 GB, so a 12 GB card is not covered yet.

## Vulkan · RTX PRO 6000

The same card through Vulkan (`CMF_GPU=wgpu`), the same prompt and 256 greedy tokens, first token from a cold
process. The prompt goes through the batch graph in 16-token chunks.

| weights' budget | `q2tp` decode, tok/s | first token, s | `q4tp` decode, tok/s | first token, s | peak in use `q2tp` / `q4tp`, GB |
|---:|---:|---:|---:|---:|---:|
| none (the whole card) | **45.1** | 31.8 | 37.4 | 27.1 | 62.3 / 88.5 |
| 24 GB | **36.4** | 59.4 | 27.9 | 125.5 | 32.4 / 33.7 |

On wgpu `CMF_GPU_VRAM_MB` budgets the weights; the K/V, the prompt's buffers and the experts' staging take the card
beside them (the peak column).

## Server

```bash
CMF_GPU=cuda cortiq serve hy4-flash-q4tp.cmf --port 8080 --host 127.0.0.1
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"hy4-flash","messages":[{"role":"user","content":"Write a haiku about autumn."}],"max_tokens":128}'
```

An OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, streaming) and a web dashboard on the same
port.

## License

The weights are Tencent's, under the [Tencent Hy License](LICENSE) (a copy is in this repository). These files are
a modified form: converted to CMF and quantized as described above.

---

## Документация на русском

[Tencent Hy4-flash](https://huggingface.co/tencent/Hy4-flash) (вышла 17.11.2026, 220 млрд параметров, 12 млрд
активных) в виде одиночных файлов [CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка инференса на
Rust без Python и ML-фреймворков. Модель выбирает 16 из 512 экспертов на латенте шириной 1536, чередует 30 слоёв
скользящего окна и 12 разреженных слоёв внимания, держит четыре остаточных потока и на четырёх слоях читает
таблицы биграмм и триграмм (Engram).

- **RTX PRO 6000 (96 ГБ), CUDA:** `q2tp` отвечает со скоростью **65,1 ток/с**, `q4tp` —
  55,3 ток/с после промпта в 1 000 токенов, который карта читает за 4 с.
- **В 24 ГБ карты:** эксперты живут в памяти хоста, карта держит горячие: `q2tp` 37,4 ток/с, `q4tp`
  22,7 ток/с, пик занятой памяти 23,7 ГБ.
- **Vulkan / DX12 / Metal:** те же файлы декодируются на карте через wgpu: `q2tp` 45,1 ток/с, `q4tp`
  37,4 ток/с на той же карте.

### Быстрый старт

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Hy4-flash-cmf hy4-flash-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq run hy4-flash-q4tp.cmf --prompt "Объясни в трёх предложениях, почему небо голубое." --greedy --no-think
```

Нужен cortiq 0.9.6 или новее. Быстрый путь — CUDA (`--features cuda`: во время работы нужны `libnvrtc` и
`libcublas` из CUDA 12.x, CUDA Toolkit при сборке не нужен). Vulkan, DX12 и Metal (обычный `cargo install cortiq-cli`,
`CMF_GPU=wgpu`) тоже работают на карте, пока с двумя ограничениями: промпт читается кусками по 16 токенов, а после 2 048
токенов контекста индексатор разреженных слоёв работает на хосте. `--no-think` просит ответ без рассуждения.

**Память.** Файл отображается в память, а не загружается. На карте — внимание, общие эксперты и столько
маршрутизируемых экспертов, сколько поместится. В памяти хоста — копия маршрутизируемых экспертов в раскладке
карты (до 86 ГБ, RAM-ярус) и таблицы Engram, которые читаются из кэша страниц (63 ГБ). При меньшем объёме
RAM движок держит ярус меньше и дочитывает остальное из файла — медленнее на каждом промахе.

### Файлы

| файл | эксперты (gate/up · down) | остальное | размер | назначение |
|---|---|---|---:|---|
| `hy4-flash-q4tp.cmf` | 4 бита · 4 бита | 8 бит, Engram 8 бит как в релизе | 150,5 ГБ | **по умолчанию** |
| `hy4-flash-q2tp.cmf` | 2 бита · 4 бита | 4 бита, Engram 8 бит как в релизе | 122,8 ГБ | меньше и быстрее: на карту помещается больше экспертов |
| `hy4-flash-q2tp-sfq.cmf` | 2 бита · 4 бита, SFq | 4 бита, Engram 8 бит как в релизе | 122,8 ГБ | код: `q2tp` с откалиброванными экспертами |

`q4tp` и `q2tp` сконвертированы из FP8-релиза (e4m3 с масштабами блоков 128×128, декодируются точно) с bf16-шардами внимания
из `tencent/Hy4-flash` поверх. Визуальная башня лежит в файлах; cortiq пока читает только текст. `hy4-flash-q2tp-sfq.cmf` — это `q2tp` с SFq на маршрутизируемых экспертах:
масштаб каналов латента и GPTQ с доводкой кодов, откалиброванные на 98 304 токенах английского, китайского текста
и кода на Rust (`cortiq hy4-sfq`). Код он читает лучше, прозу хуже, чем обычный `q2tp`.

### Качество

| файл | WikiText-2 | китайские доки | код на Rust |
|---|---:|---:|---:|
| `q4tp` | 35,80 | 39,13 | 3,92 |
| `q2tp` | 32,98 | 36,08 | 4,64 |
| `q2tp-sfq` | 42,20 | 41,83 | 4,30 |

Перплексия, окна по 512 токенов на prompt-графе карты: WikiText-2 test (16 окон), китайская документация
(8), код на Rust (8). 2-битные эксперты сильнее бьют по коду, чем по прозе; SFq возвращает код.

### CUDA · RTX PRO 6000

Стенд: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 ГБ), бэкенд CUDA (`CMF_GPU=cuda`), cortiq 0.9.6,
release-сборка. `cortiq run --greedy --no-think`, 256 токенов после промпта из 1 000 токенов WikiText-2, файл уже в
кэше страниц (второй запуск); `CMF_GPU_VRAM_MB` ограничивает карту.

| лимит (`CMF_GPU_VRAM_MB`) | `q2tp` декод, ток/с | первый токен, с | `q4tp` декод, ток/с | первый токен, с | пик занятой памяти `q2tp` / `q4tp`, ГБ |
|---:|---:|---:|---:|---:|---:|
| 96 ГБ (вся) | **65,1** | 15,5 | 55,3 | 19,7 | 60,5 / 86,5 |
| 48 ГБ | **53,1** | 18,0 | 37,8 | 20,8 | 45,6 / 45,8 |
| 24 ГБ | **37,4** | 19,6 | 22,7 | 23,8 | 23,6 / 23,7 |
| 16 ГБ | **31,2** | 19,9 | 19,9 | 24,4 | 16,6 / 17,2 |
| 15 ГБ | **29,9** | 20,0 | 19,0 | 23,1 | 15,7 / 15,8 |
| 12 ГБ | **26,0** | 19,9 | — | — | 13,1 / — |

![Декод по размеру карты](https://huggingface.co/infosave/Hy4-flash-cmf/resolve/main/assets/perf_cuda_ru.png)

Первый токен засекается от холодного процесса: в него входят построение плана и заполнение RAM-яруса (около
15 с, один раз на процесс); само чтение промпта из 1 000 токенов на всей карте занимает 4 с.
`CMF_GPU_VRAM_MB` ограничивает то, что размещает движок; пик процесса бывает до 1,2 ГБ выше (столбец пика). На карте 16 ГБ задайте `CMF_GPU_VRAM_MB=15360`: тогда оба файла укладываются в 16 ГБ. Под лимитом 12 ГБ слоты экспертов `q4tp` не помещаются (его промпт уходит на хост), а `q2tp` доходит до 13,1 ГБ, так что карту на 12 ГБ эти файлы пока не покрывают.

### Vulkan · RTX PRO 6000

Та же карта через Vulkan (`CMF_GPU=wgpu`), тот же промпт и 256 жадных токенов, первый токен от холодного
процесса. Промпт идёт через пакетный граф кусками по 16 токенов.

| бюджет весов | `q2tp` декод, ток/с | первый токен, с | `q4tp` декод, ток/с | первый токен, с | пик занятой памяти `q2tp` / `q4tp`, ГБ |
|---:|---:|---:|---:|---:|---:|
| нет (вся карта) | **45,1** | 31,8 | 37,4 | 27,1 | 62,3 / 88,5 |
| 24 ГБ | **36,4** | 59,4 | 27,9 | 125,5 | 32,4 / 33,7 |

В wgpu `CMF_GPU_VRAM_MB` задаёт бюджет весов; K/V, буферы промпта и промежуточная загрузка экспертов занимают карту
рядом с ними (столбец пика).

### Сервер

```bash
CMF_GPU=cuda cortiq serve hy4-flash-q4tp.cmf --port 8080 --host 127.0.0.1
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"hy4-flash","messages":[{"role":"user","content":"Напиши хайку об осени."}],"max_tokens":128}'
```

OpenAI-совместимый API (`/v1/chat/completions`, `/v1/completions`, потоковая выдача) и веб-панель на том же порту.

### Лицензия

Веса принадлежат Tencent и распространяются по [Tencent Hy License](LICENSE) (копия лежит в этом репозитории).
Эти файлы — изменённая форма: сконвертированы в CMF и квантованы, как описано выше.

---

## 中文说明

[腾讯 Hy4-flash](https://huggingface.co/tencent/Hy4-flash)（2026-11-17 发布，总参数 2200 亿，激活 120 亿）转换为
`cortiq` 使用的单文件 [CMF](https://github.com/infosave2007/cmf)。`cortiq` 是一个 Rust 推理引擎，不依赖 Python
和任何机器学习框架。该模型在 1536 维潜空间上从 512 个专家中选出 16 个，交替使用 30 个滑动窗口注意力层和 12 个
稀疏注意力层，保持四条残差流，并在四个层上查询二元与三元哈希表（Engram）。

- **RTX PRO 6000（96 GB），CUDA：** 在 1000 token 的提示（显卡读取用时 4 秒）之后，`q2tp` 以 **65.1 tok/s**、`q4tp` 以 55.3 tok/s 生成。
- **24 GB 显存以内：** 专家放在主机内存，显卡缓存常用专家：`q2tp` 37.4 tok/s，`q4tp` 22.7 tok/s，显存峰值 23.7 GB。
- **Vulkan / DX12 / Metal：** 同样的文件通过 wgpu 在同一张显卡上解码：`q2tp` 45.1 tok/s，`q4tp` 37.4 tok/s。

### 快速开始

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Hy4-flash-cmf hy4-flash-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq run hy4-flash-q4tp.cmf --prompt "用三句话介绍一下长城的历史。" --greedy --no-think
```

需要 cortiq 0.9.6 或更高版本。最快的路径是 CUDA（`--features cuda`：运行时需要 CUDA 12.x 的 `libnvrtc` 与
`libcublas`，编译时不需要 CUDA Toolkit）。Vulkan、DX12 与 Metal（普通的 `cargo install cortiq-cli`，`CMF_GPU=wgpu`）
同样在显卡上运行，目前有两个限制：提示按 16 token 分块读取；上下文超过 2048 token 后，稀疏层的索引器在主机上运行。
`--no-think` 要求直接回答。

**内存。** 文件以内存映射方式使用，而不是整体加载。显卡上放注意力、共享专家，以及能放下的路由专家；主机内存中
放一份按显卡布局排列的路由专家副本（最多 86 GB，即 RAM 层）以及从页缓存读取的 Engram 表（63 GB）。
内存较少时，引擎会缩小 RAM 层，其余部分从文件读取——每次未命中都会更慢。

### 文件

| 文件 | 专家（gate/up · down） | 其余部分 | 大小 | 用途 |
|---|---|---|---:|---|
| `hy4-flash-q4tp.cmf` | 4 位 · 4 位 | 8 位，Engram 保持发布时的 8 位 | 150.5 GB | **默认** |
| `hy4-flash-q2tp.cmf` | 2 位 · 4 位 | 4 位，Engram 保持发布时的 8 位 | 122.8 GB | 更小更快：显卡可放下更多专家 |
| `hy4-flash-q2tp-sfq.cmf` | 2 位 · 4 位，SFq | 4 位，Engram 保持发布时的 8 位 | 122.8 GB | 代码：专家经过校准的 `q2tp` |

`q4tp` 与 `q2tp` 由 FP8 版本（e4m3，128×128 块缩放，精确解码）转换而来，注意力部分使用 `tencent/Hy4-flash` 的 bf16
分片。视觉塔保留在文件中；cortiq 目前只读取文本。`hy4-flash-q2tp-sfq.cmf` 是对路由专家做了 SFq 的 `q2tp`：潜空间的通道缩放
加带代码微调的 GPTQ，在 98 304 个英文、中文与 Rust 代码 token 上校准（`cortiq hy4-sfq`）。与普通 `q2tp` 相比，它读代码更好，
读文本更差。

### 质量

| 文件 | WikiText-2 | 中文文档 | Rust 源码 |
|---|---:|---:|---:|
| `q4tp` | 35.80 | 39.13 | 3.92 |
| `q2tp` | 32.98 | 36.08 | 4.64 |
| `q2tp-sfq` | 42.20 | 41.83 | 4.30 |

困惑度，在显卡的提示图上按 512 token 窗口计算：WikiText-2 测试集（16 个窗口）、中文文档（8 个）、Rust 源码
（8 个）。2 位专家对代码的影响大于对文本的影响；SFq 把代码的损失补了回来。

### CUDA · RTX PRO 6000

测试平台：NVIDIA RTX PRO 6000 Blackwell Server Edition（96 GB），CUDA 后端（`CMF_GPU=cuda`），cortiq 0.9.6，release 构建。
`cortiq run --greedy --no-think`，在 1000 token 的 WikiText-2 提示后生成 256 token，文件已在页缓存中（第二次运行）；用
`CMF_GPU_VRAM_MB` 限制显存。

| 显存上限（`CMF_GPU_VRAM_MB`） | `q2tp` 解码 tok/s | 首 token 秒 | `q4tp` 解码 tok/s | 首 token 秒 | 实际显存峰值 `q2tp` / `q4tp` GB |
|---:|---:|---:|---:|---:|---:|
| 96 GB（整卡） | **65.1** | 15.5 | 55.3 | 19.7 | 60.5 / 86.5 |
| 48 GB | **53.1** | 18.0 | 37.8 | 20.8 | 45.6 / 45.8 |
| 24 GB | **37.4** | 19.6 | 22.7 | 23.8 | 23.6 / 23.7 |
| 16 GB | **31.2** | 19.9 | 19.9 | 24.4 | 16.6 / 17.2 |
| 15 GB | **29.9** | 20.0 | 19.0 | 23.1 | 15.7 / 15.8 |
| 12 GB | **26.0** | 19.9 | — | — | 13.1 / — |

![按显存大小的解码速度](https://huggingface.co/infosave/Hy4-flash-cmf/resolve/main/assets/perf_cuda_zh.png)

首 token 从冷启动的进程开始计时：包括构建计划和填充 RAM 层（约 15 秒，每个进程一次）；在整卡上读取 1000 token
的提示本身用时 4 秒。
`CMF_GPU_VRAM_MB` 限制引擎放置的内容；进程峰值最多比它高 1.2 GB（见峰值列）。16 GB 显卡请设置 `CMF_GPU_VRAM_MB=15360`：两个文件都能控制在 16 GB 以内。在 12 GB 上限下，`q4tp` 的专家槽放不下（提示退回主机），`q2tp` 峰值 13.1 GB，因此暂不支持 12 GB 显卡。

### Vulkan · RTX PRO 6000

同一张显卡通过 Vulkan（`CMF_GPU=wgpu`），同样的提示与 256 个贪心 token，首 token 从冷启动的进程计时。提示通过
批处理图按 16 token 分块读取。

| 权重预算 | `q2tp` 解码 tok/s | 首 token 秒 | `q4tp` 解码 tok/s | 首 token 秒 | 实际显存峰值 `q2tp` / `q4tp` GB |
|---:|---:|---:|---:|---:|---:|
| 无（整卡） | **45.1** | 31.8 | 37.4 | 27.1 | 62.3 / 88.5 |
| 24 GB | **36.4** | 59.4 | 27.9 | 125.5 | 32.4 / 33.7 |

在 wgpu 上 `CMF_GPU_VRAM_MB` 是权重的预算；K/V、提示缓冲区与专家的中转区在其旁边占用显存（见峰值列）。

### 服务器

```bash
CMF_GPU=cuda cortiq serve hy4-flash-q4tp.cmf --port 8080 --host 127.0.0.1
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"hy4-flash","messages":[{"role":"user","content":"写一首关于秋天的俳句。"}],"max_tokens":128}'
```

兼容 OpenAI 的 API（`/v1/chat/completions`、`/v1/completions`、流式输出），同一端口提供网页面板。

### 许可

权重归腾讯所有，按 [Tencent Hy License](LICENSE) 发布（本仓库附有副本）。这些文件是修改后的形式：按上文所述转换为
CMF 并量化。
