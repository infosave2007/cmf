---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - cmf
  - cortiq
  - quantized
language:
  - en
  - ru
  - zh
---

# Qwen3.8-27B → CMF — one file, one Rust binary, no Python

```bash
cargo install cortiq-cli          # 0.5.76+
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q4t.cmf --local-dir .
cortiq run qwen38-27b-q4t.cmf --prompt "Explain quicksort in three sentences."
```

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) is a 27B hybrid:
48 GatedDeltaNet linear-attention layers and 16 full-attention layers,
262k context, thinking mode. This repo is that checkpoint converted to
the [CMF container](https://github.com/infosave2007/cmf) — a single
memory-mapped file read by `cortiq`, a Rust binary with no ML framework
under it. GPU via Vulkan/Metal/DX12 with a CPU fallback; NVIDIA, AMD,
Intel and Apple silicon run the same file.

| file | bits | size | decode, RTX 5090 | wikitext-2 ppl |
|---|---|---|---|---|
| `qwen38-27b-q4tp.cmf` | 4-bit tiled, ladder scales | **14.3 GB** | 48.6 tok/s plain · **60 tok/s** greedy with speculative decode | **8.79** |
| `qwen38-27b-q4t.cmf` | 4-bit tiled | 15.4 GB | 49.4 tok/s plain | 8.86 |
| `qwen38-27b-q8_2f.cmf` | 8-bit | 27.4 GB | 28.9 tok/s | — |

Steady-state decode over Vulkan, single stream, `cortiq bench` on one
harness (medians of three; the q8_2f row is from an earlier session);
perplexity on the same twelve 512-token windows of wikitext-2. The
family is memory-bandwidth-bound, so the 4-bit files are not just
smaller — they decode ~1.6× faster than 8-bit.

**q4tp** keeps q4t's nibbles and stores each 32-weight tile's scale as
a rung on a per-row ladder: 7.5% fewer bytes at the same quality (its
perplexity is a hair *lower*), the file for a 16 GB card. Both 4-bit
files were quantized straight from the bf16 checkpoint, streamed shard
by shard from the hub — never one from the other. The two are the same
weights on two layouts; on this card the q4t decode kernel still
streams a little faster, so q4t is the plain-speed pick and q4tp the
size pick.

**Speculative decode is on by default for greedy** (`--greedy`, or a
server request with `temperature: 0`) on Vulkan graphs since 0.5.80:
the model's own MTP head drafts four tokens, one batched submit
verifies them, and the verify lands on the plain token's bits — the
greedy continuation is identical to the plain one, only faster. It
turns itself off for the rest of a generation if the head stops
agreeing with the trunk. `CMF_GRAPH_SPEC=0` disables it. Sampling
(temperature > 0) stays on the plain path — the speculative
*sampling* arm exists (`CMF_GRAPH_SPEC_SAMPLE=1`, exact by
construction) but measured slower on this card.

## Server and API

```bash
cortiq serve qwen38-27b-q4t.cmf --port 8080
```

The server speaks the OpenAI API, so anything that talks to OpenAI
talks to it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen38-27b-q4t", "messages": [{"role": "user", "content": "Hi!"}]}'
```

`/v1/chat/completions`, `/v1/completions` and `/v1/models` are live;
`--ollama` additionally listens on an Ollama-compatible port for tools
that expect that shape. `--host 127.0.0.1` keeps it local-only. A web
dashboard sits on the same port.

## Which GPUs does Vulkan see?

```bash
cortiq gpu
```

Lists every adapter with its index, name, driver and memory budget, and
measures what a round trip to the device costs (an empty submit vs
submit+dispatch+readback — on a healthy setup that's microseconds).
If the list is empty on Linux, install the loader libraries and retry:

```bash
sudo apt install libvulkan1 libglvnd0 libegl1 libgl1 libglx0
XDG_RUNTIME_DIR=/tmp cortiq gpu     # headless boxes need the env
```

`vulkaninfo --summary` (from `vulkan-tools`) is the system-level cross
check. To pin cortiq to a specific card, set `CMF_GPU_ADAPTER` to the
index from `cortiq gpu` — or to a name substring:

```bash
CMF_GPU_ADAPTER=1 cortiq run qwen38-27b-q4t.cmf --prompt "..."
CMF_GPU_ADAPTER=5090 cortiq run qwen38-27b-q4t.cmf --prompt "..."
```

## Two GPUs

One command:

```bash
cortiq run qwen38-27b-q4t.cmf --prompt "..." --gpus 2
```

It pins the coordinator to adapter 0, spawns a local `cortiq worker`
pinned to adapter 1, and splits the layer stack between them over
loopback (the same machinery as the network split, wire cost ~zero
locally). The log names both pins at startup — check it the first time:
two identical cards otherwise both answer "I am the best adapter" and
land on the same silicon, which runs but crawls.

`--gpus` v1 is exactly two cards. For more, chain explicitly — each
worker takes a layer span:

```bash
CMF_GPU_ADAPTER=1 cortiq worker qwen38-27b-q4t.cmf --listen 127.0.0.1:9911 --token S &
CMF_GPU_ADAPTER=0 cortiq run qwen38-27b-q4t.cmf --prompt "..." \
  --peer 127.0.0.1:9911 --net-token S --peer-split 32
```

For THROUGHPUT (many parallel requests rather than one fast stream),
prefer the server's replica mode instead:

```bash
cortiq serve qwen38-27b-q4t.cmf --gpus 2
```

When the model fits one card, each GPU runs a full replica and requests
decode in parallel; when it does not fit, the server switches to a
layer split. It prints which mode it chose and why.

## Across the network

Machine B (holds the tail layers):

```bash
cortiq worker qwen38-27b-q4t.cmf --listen 0.0.0.0:9911 --token SECRET
```

Machine A (the coordinator; same `.cmf`, verified by directory hash):

```bash
cortiq run qwen38-27b-q4t.cmf \
  --prompt "..." \
  --peer 192.168.1.42:9911 --net-token SECRET --net-dtype f16
```

`--peer-split N` picks the first layer the worker runs (default: half
the stack). `--net-dtype f16` halves the wire bytes; `f32` is
bit-exact. `cortiq peers` lists workers announcing themselves on the
local network — the beacon carries identity and geometry, never the
token.

## macOS — Apple Silicon (Metal)

Since **0.5.79** the 27B runs on the Mac GPU out of the box — earlier
versions silently fell back to the CPU because the 14.4 GiB file
exceeds Metal's single-buffer cap (the engine now maps it as
overlapping windows).

```bash
cortiq run qwen38-27b-q4t.cmf --prompt "..."   # no env vars needed
```

Measured on an M4 Mac mini, 24 GB unified memory:

| | tok/s |
|---|---|
| decode | **5.8** |
| prefill @ 2k context | 20.8 |
| CPU-only (pre-0.5.79) | 3.7 |

For long context on a Mac, add the Metal O(1) mode: decode holds
~4.7 tok/s **independent of depth** and the attention state stays
fixed-size instead of growing with the KV cache:

```bash
CMF_O1_METAL=1 cortiq run qwen38-27b-q4t.cmf --o1 all --prompt "..."
```

`q4t` is the Mac build; `q8_2f` (27.4 GB) does not fit 24 GB machines.

## Sampling

Qwen's recommended parameters for this release:

| mode | temperature | top_p | top_k | presence_penalty | repetition_penalty |
|---|---|---|---|---|---|
| thinking | 1.0 | 0.95 | 20 | 0.0 | 1.0 |
| instruct (`--no-think`) | 0.7 | 0.80 | 20 | 1.5 | 1.0 |

All six knobs are exposed since 0.5.77: `--temperature`, `--top-p`,
`--top-k`, `--min-p`, `--presence-penalty`, `--rep-penalty`. The
aquarium example below was generated with the instruct row verbatim.

One more setting that matters for LONG generations: `CMF_MAX_SEQ`. The
engine sizes its KV cache to 32768 by default (the model itself goes to
262144); when a generation crosses that ceiling the cache evicts half
and quality degrades — the log warns when it happens. Raise it if you
ask for very long outputs and have the memory:

```bash
CMF_MAX_SEQ=65536 cortiq run qwen38-27b-q4t.cmf --prompt "..." --max-tokens 50000
```

## Example

Three one-shot generations from the same 7 KB Russian spec — a Three.js
aquarium with fish, bubbles and click-to-feed — same seed, official
instruct sampling, so the set doubles as a quant comparison. Download
and open in a browser:

- [`examples/aquarium-q4tp.html`](https://huggingface.co/infosave/Qwen3.8-27B-cmf/blob/main/examples/aquarium-q4tp.html) — the ladder-scale file, 9468 tokens in 391 s on an RTX 5090
- [`examples/aquarium-q4t.html`](https://huggingface.co/infosave/Qwen3.8-27B-cmf/blob/main/examples/aquarium-q4t.html) — 288 s on an RTX 5090
- [`examples/aquarium-q8_2f.html`](https://huggingface.co/infosave/Qwen3.8-27B-cmf/blob/main/examples/aquarium-q8_2f.html) — the 8-bit file's take

## O(1) long-context mode

Nyström O(1) attention replaces KV-cache attention on the flagged
layers: memory stays **constant** instead of growing with the context
(the 16 full-attention layers keep a fixed landmark skeleton; the 48
linear layers were O(1) already), and decode speed stays flat at any
depth.

```bash
# Vulkan / discrete GPUs (since 0.5.78), ~25 tok/s on an RTX 5090:
CMF_O1_GPU=1 cortiq run qwen38-27b-q4t.cmf --o1 all --prompt "..." --max-tokens 2000

# Apple Silicon (since 0.5.79), ~4.7 tok/s on an M4 24 GB at any depth:
CMF_O1_METAL=1 cortiq run qwen38-27b-q4t.cmf --o1 all --prompt "..."
```

### Parameters

| flag | default | meaning |
|---|---|---|
| `--o1 all\|deepN\|i,j,k\|off` | file hint | which full-attention layers switch to O(1): `all`, the deepest N (`deep8`), an explicit list, or force off. Overrides `CMF_O1` and the converter hint |
| `--o1-m` | 32 | landmark budget — the far-field's rank. More = better long-range recall, 32 is the validated maximum the GPU kernels accept |
| `--o1-window` | 128 | exact sliding window: the most recent tokens attended exactly |
| `--o1-sink` | 4 | permanent exact keys at the sequence start (attention sinks) |

Practical settings:

- **Defaults are the validated optimum** — start with plain `--o1 all`.
- The GPU kernels accept `sink + window ≤ 196` and `m ≤ 32`; anything
  larger falls back to the CPU step for those layers (it says so in
  the log — run with `RUST_LOG=info` to see refusals).
- Prompts shorter than `window + sink + 8` skip the skeleton entirely
  (exact attention — nothing to approximate yet).
- The **prefill runs on the CPU by design**: it records the query trace
  that seals the landmark skeleton after the prompt. First token of a
  long prompt is slower; every token after is where this mode pays.
- Where it wins: contexts past ~8k, memory-tight machines (24 GB Macs),
  and any workload where decode must not degrade with depth. At short
  contexts plain attention is equal or faster — O(1) already matches it
  at 2k on an M4 (4.7 vs 4.2 tok/s).
- Output is not bit-identical to full attention (it is an approximation
  with an exact window); quality holds while the conversation fits the
  window + landmarks regime the defaults were validated on.

## Verify

```bash
sha256sum -c qwen38-27b-q4tp.cmf.sha256    # or -q4t / -q8_2f
cortiq info qwen38-27b-q4tp.cmf
```

---

## Документация на русском

Одна модель — один файл `.cmf`, один Rust-бинарник `cortiq`, без Python:

```bash
cargo install cortiq-cli          # 0.5.77+
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q4t.cmf --local-dir .
cortiq run qwen38-27b-q4t.cmf --prompt "Объясни квиксорт в трёх предложениях."
```

| файл | биты | размер | декод, RTX 5090 | ppl wikitext-2 |
|---|---|---|---|---|
| `qwen38-27b-q4tp.cmf` | 4, лестница масштабов | **14.3 ГБ** | 48.6 tok/s · **60 tok/s** greedy со спекуляцией | **8.79** |
| `qwen38-27b-q4t.cmf` | 4 | 15.4 ГБ | 49.4 tok/s | 8.86 |
| `qwen38-27b-q8_2f.cmf` | 8 | 27.4 ГБ | 28.9 tok/s | — |

Все числа — один стенд и один бенч (`cortiq bench`, медианы трёх; строка
q8_2f — из более ранней сессии); ppl на одних и тех же 12 окнах по 512
токенов. Семейство упирается в пропускную способность памяти, поэтому
4-битные файлы не только меньше — они и декодируют в ~1.6× быстрее
8-битного.

**q4tp** хранит те же ниблы, что q4t, а масштаб каждой плитки из 32 весов —
как ступень на построчной лестнице: на 7.5% меньше байт при том же качестве
(ppl даже чуть ниже) — файл для карты на 16 ГБ. Оба 4-битных файла
квантованы прямо из bf16-чекпойнта (потоково с HF), не один из другого.
**Спекулятивный декод включён по умолчанию для greedy** (`--greedy` или
`temperature: 0` в запросе к серверу) на Vulkan с 0.5.80: собственная
MTP-голова модели предлагает четыре токена, один батч-сабмит их проверяет,
и проверка ложится на те же биты, что обычный токен, — greedy-продолжение
идентично обычному, только быстрее; при плохом согласии головы с моделью
спекуляция сама выключается до конца генерации. `CMF_GRAPH_SPEC=0`
отключает. Сэмплинг (temperature > 0) идёт обычным путём.

**Сервер с OpenAI-совместимым API:** `cortiq serve qwen38-27b-q4t.cmf
--port 8080` — работают `/v1/chat/completions`, `/v1/completions`,
`/v1/models`; флаг `--ollama` добавляет Ollama-совместимый порт.

**Какие карты видит Vulkan:** `cortiq gpu` — список адаптеров с
индексами и цена круга до устройства. Пиновка: `CMF_GPU_ADAPTER=индекс`
или подстрока имени (`CMF_GPU_ADAPTER=5090`).

**Две карты:** `cortiq run модель.cmf --prompt "..." --gpus 2` —
координатор на адаптере 0, автоматический воркер на адаптере 1, сплит
слоёв через loopback. Для пропускной способности (много параллельных
запросов) — `cortiq serve модель.cmf --gpus 2`: по полной реплике на
карту.

**По сети:** на второй машине `cortiq worker модель.cmf --listen
0.0.0.0:9911 --token СЕКРЕТ`, на первой — те же `run`-флаги плюс
`--peer адрес:9911 --net-token СЕКРЕТ --net-dtype f16`. `cortiq peers`
находит воркеров в локальной сети.

**Сэмплинг (официальные ряды Qwen):** думающий режим — temperature 1.0,
top-p 0.95, top-k 20; инструктный (`--no-think`) — temperature 0.7,
top-p 0.80, top-k 20, presence-penalty 1.5. Все шесть ручек доступны
с 0.5.77. Для очень длинных генераций поднимите `CMF_MAX_SEQ`
(по умолчанию 32768, модель умеет 262144).

**Примеры:** три аквариума в
[`examples/`](https://huggingface.co/infosave/Qwen3.8-27B-cmf/tree/main/examples)
(q4tp, q4t, q8_2f) сгенерированы одним промтом с одним seed — готовое
сравнение квантов.

**O(1) длинный контекст — параметры.** `--o1 all|deepN|i,j,k|off` —
какие attention-слои перевести на O(1) (обычно `all`); `--o1-m 32` —
бюджет ландмарок (валидированный максимум GPU-ядер); `--o1-window 128`
— точное скользящее окно; `--o1-sink 4` — постоянные точные ключи в
начале. Лимиты ядер: sink+window ≤ 196, m ≤ 32 (сверх — слой уходит на
CPU-шаг, с записью в лог при RUST_LOG=info). Префилл в этом режиме
идёт на CPU намеренно — он записывает трассу, запечатывающую скелет.
Память под внимание константна, скорость декода не падает с глубиной;
выгодно от ~8k контекста и на машинах с 24 ГБ. Вывод не бит-в-бит с
полным вниманием (это аппроксимация с точным окном). Vulkan:
`CMF_O1_GPU=1`; Metal: `CMF_O1_METAL=1` (с 0.5.79).

**macOS (Apple Silicon, Metal).** С 0.5.79 модель работает на GPU мака
из коробки (раньше файл не влезал в лимит одного Metal-буфера и всё
тихо уходило на CPU). Замер на M4 mini 24 ГБ: декод **5.8 tok/s**,
префилл на 2k контексте 20.8 tok/s (на CPU было 3.7). Для длинного
контекста — Metal-режим O(1): `CMF_O1_METAL=1 cortiq run … --o1 all` —
декод держит ~4.7 tok/s независимо от глубины, память под внимание
фиксированная. Для мака берите `q4t`; `q8_2f` (27.4 ГБ) в 24 ГБ не
помещается.

---

## 中文文档

一个模型 — 一个 `.cmf` 文件，一个 Rust 可执行文件 `cortiq`，无需 Python：

```bash
cargo install cortiq-cli          # 0.5.77+
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q4t.cmf --local-dir .
cortiq run qwen38-27b-q4t.cmf --prompt "用三句话解释快速排序。"
```

| 文件 | 位宽 | 大小 | 解码速度（RTX 5090） | wikitext-2 困惑度 |
|---|---|---|---|---|
| `qwen38-27b-q4tp.cmf` | 4 位，阶梯缩放 | **14.3 GB** | 48.6 tok/s · 贪心+推测解码 **60 tok/s** | **8.79** |
| `qwen38-27b-q4t.cmf` | 4 位 | 15.4 GB | 49.4 tok/s | 8.86 |
| `qwen38-27b-q8_2f.cmf` | 8 位 | 27.4 GB | 28.9 tok/s | — |

所有数字来自同一台机器、同一基准（`cortiq bench`，三次取中位数；q8_2f
一行来自更早的会话）；困惑度在相同的 12 个 512 token 窗口上测得。该模型
受内存带宽限制，因此 4 位文件不仅更小，解码也比 8 位快约 1.6 倍。
**q4tp** 保留 q4t 的 4 位权重，把每个 32 权重 tile 的缩放系数存为按行
阶梯上的一级：字节少 7.5%，质量不变（困惑度甚至略低），是 16 GB 显卡的
选择。两个 4 位文件都直接由 bf16 检查点量化而来（从 HF 逐分片流式转换），
而非互相转换。**贪心解码默认开启推测解码**（`--greedy` 或服务器请求
`temperature: 0`，0.5.80 起，Vulkan）：模型自带的 MTP 头一次起草四个
token，一次批量提交完成验证，验证结果与普通 token 逐位一致 —— 贪心
输出与普通路径完全相同，只是更快；若草稿头与主干不再一致，本次生成的
其余部分会自动关闭推测。`CMF_GRAPH_SPEC=0` 关闭。采样（temperature > 0）
走普通路径。

**OpenAI 兼容 API 服务器：** `cortiq serve qwen38-27b-q4t.cmf --port
8080` — 支持 `/v1/chat/completions`、`/v1/completions`、`/v1/models`；
`--ollama` 参数额外提供 Ollama 兼容端口。

**查看 Vulkan 显卡：** `cortiq gpu` 列出全部适配器及其索引。用
`CMF_GPU_ADAPTER=索引` 或名称子串（如 `CMF_GPU_ADAPTER=5090`）指定显卡。

**双显卡：** `cortiq run 模型.cmf --prompt "..." --gpus 2` —
协调进程用 0 号卡，自动启动的 worker 用 1 号卡，层间切分走本地回环。
追求吞吐量（多并发请求）用 `cortiq serve 模型.cmf --gpus 2`：每张卡
一个完整副本。

**跨机网络：** 第二台机器运行 `cortiq worker 模型.cmf --listen
0.0.0.0:9911 --token 密钥`，第一台机器加 `--peer 地址:9911 --net-token
密钥 --net-dtype f16`。`cortiq peers` 可发现局域网内的 worker。

**采样参数（Qwen 官方推荐）：** 思考模式 — temperature 1.0、top-p
0.95、top-k 20；指令模式（`--no-think`）— temperature 0.7、top-p
0.80、top-k 20、presence-penalty 1.5。0.5.77 起六个参数全部可用。
超长生成请调高 `CMF_MAX_SEQ`（默认 32768，模型支持 262144）。

**示例：**
[`examples/`](https://huggingface.co/infosave/Qwen3.8-27B-cmf/tree/main/examples)
中的三个水族馆页面（q4tp、q4t、q8_2f）由同一提示词、同一 seed 生成 —
可直接对比三种量化。

**O(1) 长上下文 — 参数。** `--o1 all|deepN|i,j,k|off` 选择切换到
O(1) 的注意力层（通常 `all`）；`--o1-m 32` 地标预算（GPU 内核验证过的
上限）；`--o1-window 128` 精确滑动窗口；`--o1-sink 4` 序列开头的永久
精确键。内核限制：sink+window ≤ 196、m ≤ 32（超出的层回退到 CPU，
RUST_LOG=info 可见）。此模式下预填充有意在 CPU 上运行——它记录封存
地标骨架所需的查询轨迹。注意力内存恒定，解码速度不随深度下降；
在 ~8k 以上上下文和 24 GB 内存的机器上收益最大。输出与完整注意力
并非逐位相同（带精确窗口的近似）。Vulkan 用 `CMF_O1_GPU=1`；
Metal 用 `CMF_O1_METAL=1`（0.5.79 起）。

**macOS（Apple Silicon，Metal）。** 自 **0.5.79** 起，27B 可直接在 Mac
GPU 上运行（此前文件超出单个 Metal 缓冲区上限，会静默回退到 CPU）。
M4 mini 24 GB 实测：解码 **5.8 tok/s**，2k 上下文预填充 20.8 tok/s
（纯 CPU 为 3.7）。长上下文请使用 Metal 版 O(1) 模式：
`CMF_O1_METAL=1 cortiq run … --o1 all` — 解码稳定在 ~4.7 tok/s，
与上下文深度无关，注意力状态恒定大小。Mac 请选 `q4t`；`q8_2f`
（27.4 GB）无法装入 24 GB 内存。
