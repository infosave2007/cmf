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
cargo install cortiq-cli          # 0.6.6+
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
| `qwen38-27b-q4tp.cmf` | 4-bit tiled, ladder scales | **14.3 GB** | 48.7 tok/s plain · **76 tok/s** greedy with speculative decode (66 with the bit-exact f32 verify) | **8.79** |
| `qwen38-27b-q4t.cmf` | 4-bit tiled | 15.4 GB | 49.4 tok/s plain | 8.86 |
| `qwen38-27b-q8_2f.cmf` | 8-bit | 27.4 GB | 33.0 tok/s | — |

Steady-state decode over Vulkan, single stream, `cortiq bench --core`
on one harness (0.5.80, medians of runs), speculative rows on the
bench's own text; on a real prompt the gain depends on how well the
draft head agrees with the trunk — a 2.3k-token code prompt decodes at
56.5 tok/s against 45.7 plain, an essay sits at the plain rate because
the engine's monitor stops speculation where it does not pay.
Perplexity on the same twelve 512-token windows of wikitext-2. The
family is memory-bandwidth-bound, so the 4-bit files are not just
smaller — they decode ~1.5× faster than 8-bit.

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
the model's own MTP head drafts five tokens, one batched submit
verifies them on an int8-activation matvec, and a monitor keeps
averaging tokens-per-round against the plain token — it stops
speculation after four losing rounds (free prose often does not pay)
and retries later, so a prompt that gains keeps the gain and one that
does not sits at the plain rate. `CMF_VERIFY_I8=0` switches the verify
to f32 (66 tok/s on the bench) and makes the greedy continuation
byte-identical to the plain path; the int8 default can resolve a
near-tie differently — a different but equally greedy continuation.
`CMF_GRAPH_SPEC=0` disables speculation. Sampling (temperature > 0)
stays on the plain path — the speculative *sampling* arm exists
(`CMF_GRAPH_SPEC_SAMPLE=1`, exact by construction) but at the instruct
row 60% of drafts are accepted and the round breaks even.

### 0.7.2 on a 24 GB card (RTX PRO 4000 Blackwell, Vulkan)

The 5090 rows above are the 0.5.80 measurement. 0.7.2 was tuned on a
second card, an RTX PRO 4000 Blackwell (24 GB, ~672 GB/s), where the
q4tp file decodes at **27.9 tok/s plain** and the default speculative
round reads:

| text (greedy, 300 tokens) | 0.7.2 default | plain |
|---|---:|---:|
| `bench --core --ignore-eos` (repetitive) | **54.8** | 27.9 |
| a list of primes + explanation | **52.5** | 27.7 |
| Python code prompt | **43.9** | 27.8 |
| English essay | **31.3** | 28.0 |
| Russian essay | **29.7** | 27.7 |

What changed: the draft depth adapts to the accepted fraction (a short
round pays on prose, a long one on code — `CMF_GRAPH_SPEC_K` pins it);
the draft head reads a 65536-row shortlist (`CMF_DRAFT_VOCAB`) that
hands back to the full head for Cyrillic and CJK; the verify runs eight
rows a workgroup; the narrow projections take a persistent grid and the
GDN control projections a vectorized kernel (plain 27.0 → 27.9). Prompt
ingest on this card goes through the batched graph by default: a
2048-token prompt at 53 tok/s (TTFT 38 s) against 28.5 one position at a
time; the batched kernels sum in a different order, so a long prompt's
greedy continuation can resolve a near-tie differently (`CMF_BATCH_K=0`
restores the per-position walk). `bench --ignore-eos` now measures speculation (it used to suppress
EOS through the sampler, which switched the round off — every
`--ignore-eos` number before 0.7.2 is the plain rate).

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

Since **0.5.79** the 27B runs on the Mac GPU out of the box (the 14.3 GB
file is mapped as overlapping windows under Metal's single-buffer cap;
before that it silently fell back to the CPU).

**How to run on a Mac — one command, no flags:**

```bash
cortiq run qwen38-27b-q4tp.cmf --prompt "..."
```

That is the whole fast path. Since **0.7.3** it holds for the CLI's default
sampling too (temperature 0.7, repetition penalty 1.1, top-k 40): before,
Metal speculated only for greedy decoding without penalties — the default
temperature and the repetition penalty each switched it off — so the plain
command decoded at the plain rate. Every Metal lever defaults to its
measured-best setting: the batched verify graph with seven drafts a round,
the 65536-row shortlist of the draft head (Cyrillic and CJK hand back to the
full head), the 4-lane GDN state, the asynchronous state replay, the prefill
graph and the device attend; plain greedy rounds (`--greedy`) also draft the
whole chain in one command buffer and take the verify's argmax on the
device, which is why `--greedy` is still the fastest arm. None of these
levers needs a `CMF_*` variable; each has one only to switch it OFF for
diagnosis. `RUST_LOG=info` prints one line at the first generation naming
the route (`metal native: spec k=7 sampling (batched verify, draft shortlist
65536, trial: proxy), state4 on, async replay on, prefill graph on, …`).
Add `--greedy` for reproducible output, `--no-think` to skip Qwen's
thinking block.

### 0.7.3 on the M4

M4 Mac mini, 24 GB unified memory, `qwen38-27b-q4tp.cmf`, `--no-think`; the
0.7.2 release binary against the 0.7.3 code, alternating in cooled windows
(sampled rows `--seed 42`; plain decode from a separate A/B, three runs an
arm, mean):

| | 0.7.2 | 0.7.3 |
|---|---|---|
| code, default command (sampling, 160 tokens) | 6.5 | **16.1** |
| code, `--greedy` (160 tokens) | 13.6 | **17.3** |
| short answer, `--greedy` (40 tokens) | 10.3 | **17.4** |
| essay, `--greedy` / default command | 6.8 / 6.8 | 8.1 / 6.7 |
| Russian essay, `--greedy` | 7.1 | 8.1 |
| speculative bench (`bench --core --tokens 96 --ignore-eos`) | 15.3 | **20.0** |
| plain decode (`CMF_GRAPH_SPEC=0`) | 6.70 | 6.77 |

tok/s of decode. What moved: speculation on the sampling and the penalized
arms (the Metal verify tile is flat in the batch, so a round of seven drafts
costs about two plain tokens and pays there); no eight plain tokens timed
inside every answer (about 1.1 s at ~143 ms a token) — Metal speculates from
the first token and times the plain path over two tokens only when the
rounds look doubtful; a verify that reads back eight ids from a device argmax
instead of eight 248k-float logit rows, with its scratch sized once — the
400+ ms rounds (two in 34 on a code prompt) were the driver zero-filling
freshly allocated buffers inside the command buffer; the draft chain in one
submit (35.5 → 31.6 ms a round); and, from the work after 0.7.2, a 4-lane
re-tile of the GDN state kernel and the commit's state replay on a second
queue. A speculative round also no longer runs past `max_tokens` (0.7.2
could return a token or two more than asked). Greedy output is
byte-identical to 0.7.2 on code, essay and Russian prompts; greedy with a
repetition penalty (`--temperature 0 --rep-penalty 1.1`) speculates with
text byte-identical to the plain path and to 0.7.2 (code and essay); the
commit oracle (`CMF_METAL_VERIFY_CHECK=2`) reads the appended K/V rows
exactly and the GDN states within 1.2e-3 of the plain path (8e-4 with the
old state kernel over the same 15 rounds).

Where the Mac's ceiling is: a code round is ~268 ms (draft 32, verify about
220) for ~4.8 tokens. The verify's eight-row q4tp GEMM streams the 14.3 GB
at 64-71 GB/s against the plain matvec's 97: on the M4 a half-precision
simdgroup multiply-accumulate issues at the plain FP32 FMA rate (1.7 of
1.88 TMAC/s measured), so at eight rows the matrix work and the weight
stream are nearly equal and cannot fully overlap. Four kernel redesigns
(fewer ALU ops, other threadgroup shapes, contiguous group runs, two rows in
flight a thread) all measured slower in a clean window. More Mac speed now
has to come from more accepted tokens per round.

0.5.82 fixed two silent Metal numerics bugs this file was subject to
(prompts longer than a chunk came back as noise; the device attend ran
15–20% off the CPU) — **use 0.5.82 or later on a Mac.**

For long context on a Mac, add the Metal O(1) mode (it decodes without
speculation): the retained M4 profile measured ~4.7 tok/s after transition
with fixed attention state instead of a growing KV cache; throughput still
depends on backend and profile:

```bash
CMF_O1_METAL=1 cortiq run qwen38-27b-q4t.cmf --o1 all --prompt "..."
```

`q4tp` is the Mac build (`q4t` also runs); `q8_2f` (27.4 GB) does not fit 24 GB machines.

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
layers after an exact prefill: memory stays **bounded** instead of growing
with the context (the 16 full-attention layers keep a fixed landmark
skeleton; the 48 linear layers were O(1) already), and attention work per
token is bounded after the transition. Observed throughput still depends on
the backend and profile. By default, generation runs an exact prefill over
the full prompt and seals once at the end. Set `CMF_O1_PREFILL=256` for the
explicit bounded profile: on prompts at least 256 tokens long, the runtime
keeps the first 256 tokens exact (or raises the prefix to the
skeleton-safe floor `window + sink + 8 + 1`), seals once, and streams the
remaining prompt through the bounded state. A short prompt remains exact
through that floor; if generation crosses it, the runtime seals once.

```bash
# Vulkan / discrete GPUs (since 0.5.78), ~25 tok/s on an RTX 5090:
CMF_O1_GPU=1 cortiq run qwen38-27b-q4t.cmf --o1 all --prompt "..." --max-tokens 2000

# Apple Silicon (since 0.5.79), retained M4 profile ~4.7 tok/s after transition:
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
- Backend limits are separate. Vulkan/wgpu accepts `sink + window ≤ 2052`
  and `m ≤ 32`; with the default sink of 4, `window=2048` is the largest
  extended profile and remains experimental. Native Metal currently keeps
  its own `sink + window ≤ 196` cap. A request beyond the selected backend's
  cap is declined for those layers and uses the CPU step; run with
  `RUST_LOG=info` to see refusals.
- A prompt shorter than `window + sink + 8 + 1` stays exact while it is below
  the deferred boundary. If the request ends before that point, no skeleton
  is built; if generation crosses it, the exact lead-in is sealed once.
- The default full-prompt prefill records the query trace and seals at the
  prompt end. `CMF_O1_PREFILL=256` selects the bounded generation profile
  above, so only the requested exact prefix is retained before the O(1)
  suffix stream. First-token latency includes whichever exact prefix the
  selected backend runs.
- Where it wins: contexts past ~8k, memory-tight machines (24 GB Macs),
  and any workload where decode must not degrade with depth. At short
  contexts plain attention is equal or faster — O(1) already matches it
  at 2k on an M4 (4.7 vs 4.2 tok/s).
- Output is not bit-identical to full attention (it is an approximation
  with an exact window); quality holds while the conversation fits the
  window + landmarks regime the defaults were validated on.

The 0.6.6 Vulkan speed result is bounded and hardware-specific. The measured
profile was `m=32`, `window=128`, `sink=4`, `CMF_O1_PREFILL=256`,
`CMF_BATCH_K=128`, `CMF_BATCH_COOP=1` (K128/COOP1), and `CMF_MTP=0` (MTP
off). On one RTX PRO 4000 Blackwell, three alternating baseline/candidate
pairs at an 8192 token context and 128 output tokens moved median TTFT from
73.8218 s to 68.3111 s (−7.46%) and steady decode from 25.7715 to 26.1725
tok/s (+1.56%). Every run produced 128 tokens, with 16 O(1) device layers and
46,236,672 bytes of O(1) device state. `window=2048` is experimental; the
default remains `window=128`. These measurements do not establish universal
speed, exact far-context recall, or a quality improvement. `convert --o1`
does not train the model; it selects the runtime approximation.

## Verify

```bash
sha256sum -c qwen38-27b-q4tp.cmf.sha256    # or -q4t / -q8_2f
cortiq info qwen38-27b-q4tp.cmf
```

---

## Документация на русском

Одна модель — один файл `.cmf`, один Rust-бинарник `cortiq`, без Python:

```bash
cargo install cortiq-cli          # 0.6.6+
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q4t.cmf --local-dir .
cortiq run qwen38-27b-q4t.cmf --prompt "Объясни квиксорт в трёх предложениях."
```

| файл | биты | размер | декод, RTX 5090 | ppl wikitext-2 |
|---|---|---|---|---|
| `qwen38-27b-q4tp.cmf` | 4, лестница масштабов | **14.3 ГБ** | 48.7 tok/s · **76 tok/s** greedy со спекуляцией (66 с побитово точной f32-проверкой) | **8.79** |
| `qwen38-27b-q4t.cmf` | 4 | 15.4 ГБ | 49.4 tok/s | 8.86 |
| `qwen38-27b-q8_2f.cmf` | 8 | 27.4 ГБ | 33.0 tok/s | — |

Все числа — один стенд и один бенч (`cortiq bench --core`, 0.5.80,
медианы); спекулятивные — на тексте самого бенча, на реальном промпте
выигрыш зависит от согласия черновой головы с моделью: код на 2.3k
контекста — 56.5 против 45.7 tok/s, эссе идёт на скорости plain (монитор
сам останавливает спекуляцию там, где она не окупается). ppl на одних и
тех же 12 окнах по 512 токенов. Семейство упирается в пропускную
способность памяти, поэтому 4-битные файлы не только меньше — они и
декодируют в ~1.5× быстрее 8-битного.

**q4tp** хранит те же ниблы, что q4t, а масштаб каждой плитки из 32 весов —
как ступень на построчной лестнице: на 7.5% меньше байт при том же качестве
(ppl даже чуть ниже) — файл для карты на 16 ГБ. Оба 4-битных файла
квантованы прямо из bf16-чекпойнта (потоково с HF), не один из другого.
**Спекулятивный декод включён по умолчанию для greedy** (`--greedy` или
`temperature: 0` в запросе к серверу) на Vulkan с 0.5.80: собственная
MTP-голова модели предлагает пять токенов, один батч-сабмит проверяет
их матвеком на int8-активациях, а монитор всё время сравнивает токены за
раунд с обычным токеном — после четырёх проигрышных раундов спекуляция
останавливается (проза часто не окупается) и позже пробуется снова, так
что промпт, который выигрывает, выигрыш сохраняет, а который нет — идёт
на скорости plain. `CMF_VERIFY_I8=0` переводит проверку на f32 (66 tok/s
на бенче) и делает greedy-продолжение побитово идентичным обычному; int8
по умолчанию может иначе разрешить близкий тай-брейк — другое, но столь
же greedy продолжение. `CMF_GRAPH_SPEC=0` отключает спекуляцию. Сэмплинг
(temperature > 0) идёт обычным путём (спекулятивный сэмплинг есть за
`CMF_GRAPH_SPEC_SAMPLE=1`, но на instruct-ряду принимается 60% черновиков
и раунд выходит в ноль).

**0.7.2 на карте 24 ГБ (RTX PRO 4000 Blackwell, Vulkan).** Строки про
5090 выше — замер 0.5.80. 0.7.2 настраивался на второй карте, RTX PRO
4000 Blackwell (24 ГБ, ~672 ГБ/с): q4tp декодирует **27.9 tok/s plain**,
спекуляция по умолчанию даёт:

| текст (greedy, 300 токенов) | 0.7.2 по умолчанию | plain |
|---|---:|---:|
| `bench --core --ignore-eos` (повторяющийся) | **54.8** | 27.9 |
| список простых чисел + объяснение | **52.5** | 27.7 |
| промпт на Python-код | **43.9** | 27.8 |
| эссе по-английски | **31.3** | 28.0 |
| эссе по-русски | **29.7** | 27.7 |

Что изменилось: глубина черновика подстраивается под долю принятых
(короткий раунд окупается на прозе, длинный — на коде; `CMF_GRAPH_SPEC_K`
фиксирует); голова черновика читает шортлист из 65536 строк
(`CMF_DRAFT_VOCAB`) и возвращается к полной голове на кириллице и CJK;
верификация идёт по восемь строк на рабочую группу; узкие проекции —
на персистентной сетке, управляющие проекции GDN — на векторном ядре
(plain 27.0 → 27.9). Промпт на этой карте читается батч-графом по
умолчанию: 2048 токенов на 53 tok/s (TTFT 38 с) против 28.5 по одной
позиции; батч-ядра суммируют в другом порядке, поэтому на длинном промпте
greedy-продолжение может по-другому разрешить почти-ничью
(`CMF_BATCH_K=0` возвращает обход по позициям). `bench --ignore-eos` теперь меряет спекуляцию (раньше он
подавлял EOS через сэмплер, что выключало раунд — все числа с
`--ignore-eos` до 0.7.2 были plain).

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

**O(1) длинный контекст — параметры.** Начиная с 0.6.6, `--o1` держит
точный префилл и короткий lead-in до безопасной границы скелета
`window + sink + 8 + 1`, а затем один раз переводит слой в ограниченное
состояние Nyström. Если короткая генерация заканчивается раньше границы,
остаётся точное внимание; веса при этом не обучаются и не меняются.
`--o1 all|deepN|i,j,k|off` —
какие attention-слои перевести на O(1) (обычно `all`); `--o1-m 32` —
бюджет ландмарок (валидированный максимум GPU-ядер); `--o1-window 128`
— точное скользящее окно; `--o1-sink 4` — постоянные точные ключи в
начале. Лимиты зависят от backend: Vulkan/wgpu принимает sink+window ≤ 2052
и m ≤ 32; при sink=4 это допускает `window=2048`, который остаётся
экспериментальным. Нативный Metal сохраняет отдельный предел sink+window ≤
196; более широкая настройка отклоняется этим backend, и слой идёт на
CPU-шаг (отказ виден при RUST_LOG=info). По умолчанию точный префилл идёт по
всему промпту и запечатывает скелет в конце. Для ограниченного профиля
задайте `CMF_O1_PREFILL=256`: точными остаются первые 256 токенов (или
безопасный минимум `window + sink + 8 + 1`), затем остаток промпта проходит
через ограниченное состояние; короткий промпт остаётся точным до этой
границы. После перехода объём работы внимания ограничен, а фактическая
скорость зависит от backend и профиля;
выгодно от ~8k контекста и на машинах с 24 ГБ. Вывод не бит-в-бит с
полным вниманием (это аппроксимация с точным окном). Vulkan:
`CMF_O1_GPU=1`; Metal: `CMF_O1_METAL=1` (с 0.5.79).

Замер 0.6.6 на одной RTX PRO 4000 Blackwell (Vulkan), три чередующиеся
пары control/candidate, контекст 8192 и 128 выходных токенов использовал
профиль `m=32`, `window=128`, `sink=4`, `CMF_O1_PREFILL=256`,
`CMF_BATCH_K=128`, `CMF_BATCH_COOP=1` (K128/COOP1), `CMF_MTP=0` (MTP off):
медианный TTFT 73.8218 → 68.3111 с (−7.46%), steady decode 25.7715 →
26.1725 tok/s (+1.56%). В каждом запуске было 128 токенов, 16 O(1)-слоёв и
46 236 672 байта состояния на устройстве. Это результат одной модели и
одного GPU; `window=2048` остаётся экспериментальным, по умолчанию — 128.
Качество и точное извлечение произвольных дальних фактов этим замером не
утверждаются.

**macOS (Apple Silicon, Metal).** С 0.5.79 модель работает на GPU мака
из коробки (файл 14.3 ГБ отображается перекрывающимися окнами под лимит
одного Metal-буфера; раньше всё тихо уходило на CPU). Запуск — одна
команда, без флагов: `cortiq run qwen38-27b-q4tp.cmf --prompt "..."`. С
**0.7.3** быстрый путь работает и на сэмплировании CLI по умолчанию
(температура 0.7, repetition penalty 1.1, top-k 40): раньше Metal
спекулировал только при greedy без штрафов — и температура по умолчанию, и
штраф за повтор выключали спекуляцию, поэтому команда без флагов шла на
скорости plain. Все рычаги Metal включены по умолчанию: пакетная
верификация на 7 черновиков, шорт-лист головы черновика на 65536 строк
(кириллица и CJK возвращаются к полной голове), 4-канальное GDN-состояние,
асинхронный replay, граф префилла, device-attend; в раундах plain greedy
(`--greedy`) вся цепочка черновика идёт одним командным буфером, а argmax
верификации считается на устройстве — поэтому `--greedy` по-прежнему
быстрее всего. Ни один из этих рычагов не требует переменных `CMF_*` — они
есть только чтобы ВЫКЛЮЧИТЬ рычаг для диагностики; `RUST_LOG=info`
печатает строку `metal native: …`. `--greedy` даёт воспроизводимый вывод,
`--no-think` убирает блок размышлений. M4 mini 24 ГБ, q4tp, `--no-think`,
бинарь релиза 0.7.2 против кода 0.7.3 попеременно в остывших окнах, tok/s
декода: код командой по умолчанию 6.5 → **16.1**, код с `--greedy`
13.6 → **17.3**, короткий ответ с `--greedy` (40 токенов) 10.3 → **17.4**,
эссе с `--greedy` 6.8 → 8.1 (командой по умолчанию 6.8 → 6.7), русское эссе
с `--greedy` 7.1 → 8.1, бенч со спекуляцией 15.3 → **20.0**, plain
6.70 → 6.77 (отдельный остывший A/B, по три прогона). Раунд спекуляции
больше не выходит за `max_tokens`. Greedy-вывод побитово совпадает с 0.7.2 на
промптах кода, эссе и русского текста; greedy со штрафом за повтор
побитово совпадает с plain-путём и с 0.7.2. Раунд на коде ~268 мс (черновик
32, верификация около 220) на ~4.8 токена; потолок мака — восьмистрочный
GEMM верификации (64-71 ГБ/с против 97 у plain-матвека): на M4
simdgroup-MMA половинной точности идёт с темпом обычного FP32 FMA (1.7 из
1.88 TMAC/s), и при восьми строках вычисления и поток весов почти равны;
четыре переделки ядра в чистом замере оказались медленнее, так что дальше
ускорять мак можно только за счёт большего числа принятых токенов за
раунд. С 0.5.82 починены две тихие численные ошибки Metal (промпт длиннее
чанка возвращался шумом; device-attend отклонялся от CPU на 15–20%) — **на
маке нужна 0.5.82 или новее**. Для длинного контекста — Metal-режим O(1)
(спекуляция в нём выключена): `CMF_O1_METAL=1 cortiq run … --o1 all` —
сохранённый профиль даёт ~4.7 tok/s после перехода при фиксированной
памяти внимания; фактическая скорость зависит от backend и профиля. Для
мака берите `q4tp` (`q4t` тоже работает); `q8_2f` (27.4 ГБ) в 24 ГБ не
помещается.

---

## 中文文档

一个模型 — 一个 `.cmf` 文件，一个 Rust 可执行文件 `cortiq`，无需 Python：

```bash
cargo install cortiq-cli          # 0.6.6+
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q4t.cmf --local-dir .
cortiq run qwen38-27b-q4t.cmf --prompt "用三句话解释快速排序。"
```

| 文件 | 位宽 | 大小 | 解码速度（RTX 5090） | wikitext-2 困惑度 |
|---|---|---|---|---|
| `qwen38-27b-q4tp.cmf` | 4 位，阶梯缩放 | **14.3 GB** | 48.7 tok/s · 贪心+推测解码 **76 tok/s**（逐位精确的 f32 验证：66） | **8.79** |
| `qwen38-27b-q4t.cmf` | 4 位 | 15.4 GB | 49.4 tok/s | 8.86 |
| `qwen38-27b-q8_2f.cmf` | 8 位 | 27.4 GB | 33.0 tok/s | — |

所有数字来自同一台机器、同一基准（`cortiq bench --core`，0.5.80，取中位数）；
推测解码一行在基准自身文本上测得，真实提示词上的收益取决于草稿头与主干的
一致程度：2.3k 上下文的代码提示 56.5 对 45.7 tok/s，散文则保持普通速度
（引擎的监视器会在不划算处停止推测）。困惑度在相同的 12 个 512 token 窗口
上测得。该模型受内存带宽限制，因此 4 位文件不仅更小，解码也比 8 位快约 1.5 倍。
**q4tp** 保留 q4t 的 4 位权重，把每个 32 权重 tile 的缩放系数存为按行
阶梯上的一级：字节少 7.5%，质量不变（困惑度甚至略低），是 16 GB 显卡的
选择。两个 4 位文件都直接由 bf16 检查点量化而来（从 HF 逐分片流式转换），
而非互相转换。**贪心解码默认开启推测解码**（`--greedy` 或服务器请求
`temperature: 0`，0.5.80 起，Vulkan）：模型自带的 MTP 头一次起草五个
token，一次批量提交在 int8 激活的矩阵向量核上完成验证，监视器持续把每轮
产出的 token 数与普通 token 比较 —— 连续四轮不划算即停止推测（散文常常
不划算），稍后再试，因此有收益的提示保住收益，没有的按普通速度走。
`CMF_VERIFY_I8=0` 切回 f32 验证（基准 66 tok/s），贪心输出与普通路径
逐位一致；默认的 int8 可能把接近平局的 token 解成另一个同样贪心的续写。
`CMF_GRAPH_SPEC=0` 关闭推测。采样（temperature > 0）
走普通路径。

**0.7.2 在 24 GB 显卡上（RTX PRO 4000 Blackwell，Vulkan）。** 上表的 5090
数据是 0.5.80 的测量。0.7.2 在第二张卡 RTX PRO 4000 Blackwell（24 GB，约
672 GB/s）上调优：q4tp 文件 plain 解码 **27.9 tok/s**，默认推测解码：

| 文本（贪心，300 token） | 0.7.2 默认 | plain |
|---|---:|---:|
| `bench --core --ignore-eos`（重复文本） | **54.8** | 27.9 |
| 素数列表 + 解释 | **52.5** | 27.7 |
| Python 代码提示 | **43.9** | 27.8 |
| 英文作文 | **31.3** | 28.0 |
| 俄文作文 | **29.7** | 27.7 |

变化：草稿深度随接受比例自适应（散文用短回合，代码用长回合；`CMF_GRAPH_SPEC_K`
可固定）；草稿头读取 65536 行的候选表（`CMF_DRAFT_VOCAB`），遇到西里尔文和 CJK
时回退到完整词表头；验证每工作组八行；窄投影使用持久网格，GDN 控制投影使用向量化
内核（plain 27.0 → 27.9）。这张卡上的提示默认走批量图：2048 token 提示 53 tok/s
（TTFT 38 s），逐位置为 28.5；批量内核的累加顺序不同，长提示的贪心续写可能在近似平局处
选出另一个 token（`CMF_BATCH_K=0` 恢复逐位置路径）。`bench --ignore-eos` 现在会测量推测解码（此前它通过
采样器抑制 EOS，从而关闭了回合——0.7.2 之前所有 `--ignore-eos` 数字都是 plain）。

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

**O(1) 长上下文 — 参数。** 从 0.6.6 起，`--o1` 在安全边界
`window + sink + 8 + 1` 之前保留精确预填充和短引导段，然后一次切换到
有界的 Nyström 状态；如果请求在边界前结束，就继续使用精确注意力。
这只是运行时适配，不训练也不修改权重。`--o1 all|deepN|i,j,k|off` 选择切换到
O(1) 的注意力层（通常 `all`）；`--o1-m 32` 地标预算（GPU 内核验证过的
上限）；`--o1-window 128` 精确滑动窗口；`--o1-sink 4` 序列开头的永久
精确键。后端限制分别计算：Vulkan/wgpu 接受 sink+window ≤ 2052、m ≤ 32；
sink=4 时 `window=2048` 可用但仍是实验选项。原生 Metal 仍有独立的
sink+window ≤ 196 上限；超出该后端上限的层会回退到 CPU，
RUST_LOG=info 可见。默认配置对整个提示词做精确预填充并在末尾封存。
设置 `CMF_O1_PREFILL=256` 可选有界配置：对至少 256 token 的提示词，先
精确处理 256 token（或提升到安全下限 `window + sink + 8 + 1`），一次封存，
再用有界状态处理提示词余部；更短提示词在越过该边界前保持精确。切换后
注意力工作量有界，实际速度取决于后端和配置；
在 ~8k 以上上下文和 24 GB 内存的机器上收益最大。输出与完整注意力
并非逐位相同（带精确窗口的近似）。Vulkan 用 `CMF_O1_GPU=1`；
Metal 用 `CMF_O1_METAL=1`（0.5.79 起）。

0.6.6 在一台 RTX PRO 4000 Blackwell（Vulkan）上的三组交替对照中，
上下文 8192、输出 128 token，使用 `m=32`、`window=128`、`sink=4`、
`CMF_O1_PREFILL=256`、`CMF_BATCH_K=128`、`CMF_BATCH_COOP=1`
（K128/COOP1）和 `CMF_MTP=0`（MTP off）：TTFT 中位数 73.8218 →
68.3111 秒（−7.46%），稳定解码 25.7715 → 26.1725 token/s（+1.56%）。
每次生成 128 token，16 个 O(1) 层的设备状态为 46,236,672 字节。这是
单模型单硬件结果；不代表普遍速度、精确的无限上下文记忆或质量提升，
`window=2048` 仍是实验选项，默认值是 128。

**macOS（Apple Silicon，Metal）。** 自 **0.5.79** 起，27B 可直接在 Mac GPU
上运行（14.3 GB 文件以重叠窗口映射，绕过 Metal 单缓冲区上限；此前会静默回退到
CPU）。在 Mac 上只需一条命令、无需额外参数或环境变量：
`cortiq run qwen38-27b-q4tp.cmf --prompt "..."`。自 **0.7.3** 起，CLI 默认的采样
设置（温度 0.7、重复惩罚 1.1、top-k 40）也走快速路径：此前 Metal 只在无惩罚的贪心
解码时启用推测，默认温度和重复惩罚都会关闭推测，因此不带参数的命令只能以普通速度
解码。所有 Metal 优化默认开启：每轮 7 个草稿的批量验证图、65536 行的草稿头短名单
（西里尔文与中日韩文字回退到完整输出头）、4 通道 GDN 状态、异步状态回放、预填充图、
设备端注意力；在普通贪心轮次（`--greedy`）中，整条草稿链只用一个命令缓冲区，验证的
argmax 也在设备端完成——因此 `--greedy` 仍是最快的。这些优化都无需设置 `CMF_*`
变量，相应变量仅用于诊断时关闭某一项；`RUST_LOG=info` 会打印一行 `metal native: …`。
`--greedy` 输出可复现，`--no-think` 跳过思考块。M4 mini 24 GB、q4tp、`--no-think`，
0.7.2 发布版二进制与 0.7.3 代码在冷却窗口内交替测量（解码 tok/s）：默认命令写代码
6.5 → **16.1**，`--greedy` 写代码 13.6 → **17.3**，`--greedy` 40 token 短回答
10.3 → **17.4**，`--greedy` 散文 6.8 → 8.1（默认命令 6.8 → 6.7），`--greedy` 俄语
散文 7.1 → 8.1，推测基准 15.3 → **20.0**，普通解码 6.70 → 6.77（单独冷却 A/B，每组
3 次）。推测轮次不再超出 `max_tokens`。在代码、散文和俄语提示上，贪心输出与 0.7.2
逐字节一致；带重复惩罚的贪心输出与普通路径及 0.7.2 逐字节一致。代码场景每轮约
268 ms（草稿 32、验证约 220）产出约 4.8 个 token；Mac 当前的上限是验证用的八行
GEMM（64-71 GB/s，普通矩阵向量乘为 97）：在 M4 上，半精度 simdgroup MMA 的吞吐与
普通 FP32 FMA 相同（实测 1.7 / 1.88 TMAC/s），八行时计算与权重读取几乎相等；四种内核
重写在干净测量中都更慢，因此 Mac 上的进一步提速只能来自每轮更多被接受的 token。
0.5.82 修复了两处 Metal 静默数值错误（超过一个分块的提示会输出噪声；设备端注意力与
CPU 偏差 15–20%）——**Mac 上请使用 0.5.82 或更新版本**。长上下文请使用 Metal 版
O(1) 模式（该模式下不启用推测解码）：`CMF_O1_METAL=1 cortiq run … --o1 all`——保留的
M4 配置在切换后测得约 4.7 tok/s，注意力状态大小恒定；实际速度取决于后端和配置。Mac 请选
`q4tp`（`q4t` 也可运行）；`q8_2f`（27.4 GB）无法装入 24 GB 内存。
