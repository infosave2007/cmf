---
license: apache-2.0
library_name: cortiq
base_model: google/embeddinggemma-2
base_model_relation: quantized
pipeline_tag: feature-extraction
tags:
  - cmf
  - cortiq
  - embeddings
  - sentence-similarity
  - multimodal-embedding
  - image-feature-extraction
  - audio-feature-extraction
  - video-feature-extraction
  - matryoshka
language:
  - multilingual
---

# EmbeddingGemma 2 — CMF

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) is Google DeepMind's 740M-parameter
embedding model. It maps text in 100+ languages, code, images, video and audio, alone or mixed in one input,
into a single 768-dimensional space. Here the text encoder and the vision and audio towers are packaged as
one [CMF](https://github.com/infosave2007/cmf) file. It runs with `cortiq`, a Rust inference engine with no
Python and no ML framework, on Apple silicon (Metal) and on any CPU, from the command line or behind an
OpenAI-compatible `/v1/embeddings` server.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档) ·
[Demo Space](https://huggingface.co/spaces/infosave/EmbeddingGemma-2-cmf)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/EmbeddingGemma-2-cmf embeddinggemma-2-q8_2f.cmf --local-dir .
cortiq embed --model embeddinggemma-2-q8_2f.cmf --task SearchQuery "a red fox in the snow" --image fox.jpg
```

It prints the start of each unit-length vector and the cosine similarities between the inputs. Use cortiq
0.8.15 or later; prebuilt binaries for Linux, macOS and Windows are on the
[releases page](https://github.com/infosave2007/cmf/releases). For a smaller download take
`embeddinggemma-2-q4tp.cmf` (647 MB: text vectors identical to q8_2f, media slightly less exact), for the
original weights `embeddinggemma-2-bf16.cmf` (1.52 GB); see [Files](#files).

As a server:

```bash
cortiq serve embeddinggemma-2-q8_2f.cmf --host 127.0.0.1 --port 8080

# text with a task prompt, 256-dimension vectors
curl http://127.0.0.1:8080/v1/embeddings -H "Content-Type: application/json" \
  -d '{"input": ["What causes the northern lights?"], "prompt_name": "SearchQuery", "dimensions": 256}'

# an image and a spoken question
curl http://127.0.0.1:8080/v1/embeddings -H "Content-Type: application/json" -d @- <<EOF
{"input": [{"image": "$PWD/fox.jpg"}, {"audio": "$PWD/question.wav"}], "dimensions": 256}
EOF
```

A server bound to 127.0.0.1 reads local file paths. On any other address (the default is 0.0.0.0) media go as
`data:` or `http(s)://` URLs.

## Examples

Real output of `embeddinggemma-2-q8_2f.cmf`. The inputs are in the `examples/` folder of this repo, one JSONL file
per example: the queries first, then the candidates. Run from the folder that holds `examples/`:

```bash
hf download infosave/EmbeddingGemma-2-cmf --include "examples/*" --local-dir .
cortiq embed --model embeddinggemma-2-q8_2f.cmf --jsonl examples/search.jsonl
```

`cortiq embed` prints the cosine matrix of all inputs (up to 8); the tables show the query rows. Queries use the
prompts named below, documents and captions use `Document`, images and audio take no prompt. The pictures were
generated with cortiq (Qwen-Image-2.1, LTX-2.5), the speech synthesized with [piper](https://github.com/rhasspy/piper).

**1. Multilingual search** ([search.jsonl](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/blob/main/examples/search.jsonl)). Query `почему светится полярное сияние` ("why
do the northern lights glow"), `SearchQuery`:

| # | document | language | cosine |
|---:|---|---|---:|
| 1 | Northern lights: solar-wind particles make oxygen and nitrogen glow | en | **0.761** |
| 2 | Где увидеть северное сияние (where and when to see it) | ru | 0.729 |
| 3 | Sonnenzyklus und Polarlichter (auroras and the solar cycle) | de | 0.668 |
| 4 | Light and the body clock | en | 0.656 |
| 5 | La photosynthèse | fr | 0.570 |
| 6 | 蓝鲸 (the blue whale) | zh | 0.555 |
| 7 | `quicksort.py` | code | 0.526 |

**2. Code search** ([code.jsonl](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/blob/main/examples/code.jsonl)). Two queries, `CodeRetrieval`; the snippets carry their
file names as titles:

| snippet | "wait until the user stops typing before calling a function" | "удалить старые логи командой в терминале" |
|---|---:|---:|
| `debounce.js` | **0.745** | 0.645 |
| `cleanup.sh` (`find /var/log/myapp -name '*.log' -mtime +7 -delete`) | 0.592 | **0.769** |
| `longest.rs` | 0.673 | 0.672 |
| `quicksort.py` | 0.665 | 0.665 |
| `binary_search.py` | 0.640 | 0.614 |
| `top_customers.sql` | 0.588 | 0.613 |

**3. Image ↔ text** ([image_to_text.jsonl](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/blob/main/examples/image_to_text.jsonl),
[text_to_image.jsonl](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/blob/main/examples/text_to_image.jsonl)). A photo against captions, then text queries (`SearchQuery`)
against photos:

| <img src="https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/fox_snow.jpg" width="160" alt="fox_snow.jpg"> | cosine |
|---|---:|
| a red fox sitting in the snow | **0.811** |
| рыжая лиса сидит на снегу | 0.795 |
| a red fox on a sandy beach at sunset | 0.766 |
| an arctic fox with white winter fur | 0.741 |
| a corgi in a chef hat cooking | 0.592 |

| image | "a fox on a beach at sunset" | "неоновая вывеска ночью под дождём" | "a dog cooking" |
|---|---:|---:|---:|
| <img src="https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/fox_beach.jpg" width="96" alt="fox_beach.jpg"> | **0.816** | 0.498 | 0.574 |
| <img src="https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/fox_snow.jpg" width="96" alt="fox_snow.jpg"> | 0.720 | 0.482 | 0.561 |
| <img src="https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/neon_sign.jpg" width="96" alt="neon_sign.jpg"> | 0.500 | **0.737** | 0.522 |
| <img src="https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/corgi_chef.jpg" width="96" alt="corgi_chef.jpg"> | 0.630 | 0.514 | **0.788** |
| <img src="https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/baikal_poster.jpg" width="96" alt="baikal_poster.jpg"> | 0.565 | 0.537 | 0.571 |

**4. Speech → text** ([speech.jsonl](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/blob/main/examples/speech.jsonl)). Two spoken questions against documents in five
languages; no transcript is involved:

| document | [ru_baikal.wav](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/ru_baikal.wav): "Какое озеро самое глубокое в мире?" | [es_whales.wav](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/es_whales.wav): "¿Por qué cantan las ballenas jorobadas?" |
|---|---:|---:|
| Озеро Байкал: the deepest lake, 1642 m (ru) | **0.725** | 0.496 |
| Humpback whale song (en) | 0.518 | **0.648** |
| 蓝鲸, the blue whale (zh) | 0.578 | 0.517 |
| Winter on Lake Baikal: ice thickness (en) | 0.559 | 0.501 |
| ザトウクジラの回遊, humpback migration (ja) | 0.477 | 0.509 |
| Der Mars (de) | 0.524 | 0.484 |

**5. Matryoshka.** Example 1 again with `--dim 256` and `--dim 128`:

| `--dim` | 1st | 2nd | 3rd | 4th | ranks 5–7 |
|---:|---|---|---|---|---|
| 768 | Northern lights 0.761 | Где увидеть… 0.729 | Sonnenzyklus… 0.668 | Light and the body clock 0.656 | fr, zh, code |
| 256 | Northern lights 0.780 | Где увидеть… 0.767 | Light and the body clock 0.690 | Sonnenzyklus… 0.669 | zh, fr, code |
| 128 | Northern lights 0.838 | Где увидеть… 0.823 | Sonnenzyklus… 0.774 | Light and the body clock 0.772 | zh, fr, code |

The three text → image queries of example 3 keep their first match at 256 and 128 dimensions (0.882, 0.808 and
0.867 at 128).

## Files

| file | weights | size | sha256 | worst cosine | recommended for |
|---|---|---:|---|---:|---|
| `embeddinggemma-2-q8_2f.cmf` | 8-bit matrices; norms, projection head and audio convolutions exact | 798 MB | `ec4192bf6cac…` | 0.99975 | **default** |
| `embeddinggemma-2-q4tp.cmf` | as q8_2f, but the vision MLPs and audio feed-forward layers 4-bit | 647 MB | `714ac411236e…` | 0.99566 | smaller download; text vectors identical to q8_2f |
| `embeddinggemma-2-bf16.cmf` | the original bf16 weights, byte for byte | 1.52 GB | `96b481acd2cc…` | 0.9999992 | exact reference |

Worst-case cosine similarity to `google/embeddinggemma-2` in float32 (sentence-transformers):

| input | q8_2f | q4tp | bf16 |
|---|---:|---:|---:|
| text: 20 inputs, 11 languages, code, prompts, a 4k-token document | 0.99978 | 0.99978 | 1.000000 |
| images: 8 cases, 70–1120 visual tokens, RGBA, extreme aspect ratios | 0.99975 | 0.99741 | 1.000000 |
| video: 4 frames | 0.99982 | 0.99862 | 1.000000 |
| audio: 10 clips, 0.3–35 s, 8–48 kHz, mono and stereo | 0.99975 | 0.99566 | 0.9999992 |
| text + image + audio in one input | 0.99985 | 0.99910 | 1.000000 |

The video row compares the same decoded frames. Decoding the mp4 with ffmpeg instead of PyAV changes the pixels
slightly, which gives 0.9991 (q8_2f), 0.9981 (q4tp) and 0.9993 (bf16).

On the demo Space corpus `q4tp` returns the same first result as `bf16` for all 41 text queries, 31 captions of
images, videos and audio clips and 8 spoken questions at 768 dimensions, and for 7 of the 8 spoken questions at 256.
Speech moves the most: two spoken questions drop to 0.968 and 0.984 cosine against `bf16` (the first is
`ru_baikal.wav` from the examples) and still find the same document. Plain `--quant q4tp` builds a different
file, with the text encoder at 4 bits (worst text cosine 0.969); this one was built with:

```bash
hf download google/embeddinggemma-2 --local-dir embeddinggemma-2
cortiq convert --model embeddinggemma-2 --output embeddinggemma-2-q4tp.cmf --quant q4tp \
  --tensor-quant '*.embedding_projection.weight=f16' --tensor-quant 'language_model.*=q8_2f' \
  --tensor-quant '*.mlp.*=q4tp' --tensor-quant 'audio_tower.*.feed_forward*=q4tp' --tensor-quant '*=q8_2f'
```

## Task prompts

Text inputs take a short prefix that tells the model what the vector is for. Pass it by name: `--task` on the
command line, `prompt_name` in the API.

| `prompt_name` | prefix | use for |
|---|---|---|
| `SearchQuery` | `task: search result \| query: ` | search queries |
| `QuestionAnswering` | `task: question answering \| query: ` | questions |
| `FactChecking` | `task: fact checking \| query: ` | claims to verify |
| `CodeRetrieval` | `task: code retrieval \| query: ` | natural-language queries for code |
| `Document` | `title: none \| text: ` | everything that is searched (`--title` / `"title"` fills in the title) |
| `Classification` | `task: classification \| query: ` | texts to classify |
| `Clustering` | `task: clustering \| query: ` | texts to cluster |
| `SentenceSimilarity` | `task: sentence similarity \| query: ` | pairwise similarity |

The four query prompts pair with `Document` on the corpus side. The last three are symmetric: give every text
the same one. `cortiq embed --list-prompts` also shows the sentence-transformers aliases (`Retrieval-query`,
`STS`, …). Images, video and audio take no prefix. A request-level prompt applies to text-only inputs, and an
input with media uses only a prompt given inside it.

## Matryoshka dimensions

The first 512, 256 or 128 numbers of a vector are an embedding by themselves. `--dim` (CLI) or `dimensions`
(API) returns the prefix, re-normalized to unit length:

| dimensions | bytes per vector (float32) | MTEB multilingual | MMEB (image, video, documents) |
|---:|---:|---:|---:|
| 768 | 3072 | 61.36 | 59.01 |
| 512 | 2048 | 61.17 | 58.38 |
| 256 | 1024 | 60.41 | 56.24 |
| 128 | 512 | 57.89 | 45.65 |

Scores are Google's, for the full-precision model. Queries and documents must use the same size. 128 is best
kept for text-only workloads.

## Images, video, audio and mixed inputs

```bash
F=embeddinggemma-2-q8_2f.cmf
cortiq embed --model $F --image photo.jpg                     # 280 visual tokens
cortiq embed --model $F --image scan.png --image-tokens 1120  # finer detail: 70 | 140 | 280 | 560 | 1120
cortiq embed --model $F --video clip.mp4                      # 1 frame per second, at most 32 frames
cortiq embed --model $F --audio question.wav                  # speech or sounds
cortiq embed --model $F --interleave --image shoe.jpg --video test.mp4 \
  "Waterproof running shoes. <|image|> Grip test on wet rock: <|video|>"
```

`--interleave` makes one vector of the text and its media: each `<|image|>`, `<|video|>` or `<|audio|>` takes
the next file of that kind. Every vector, whatever went in, is comparable with every other. A text query finds
images, a spoken question finds the documents that answer it, a photo finds its caption.

`--jsonl inputs.jsonl` embeds many inputs in one run, each line a string or an object:

```json
{"text": "What causes the northern lights?", "prompt_name": "SearchQuery"}
{"text": "Aurora borealis", "prompt_name": "Document", "title": "Northern lights"}
{"image": "aurora.jpg", "image_tokens": 560}
{"text": "Photo: <|image|> Narration: <|audio|>", "image": "aurora.jpg", "audio": "tour.wav"}
```

Add `--json` for OpenAI-style output or `--npy out.npy` for a float32 matrix.

In the API, an input is a string, an object like the JSONL lines above, or a list of OpenAI content parts. Media
go as `data:` URLs, `http(s)://` URLs or, when the server listens on localhost, file paths:

```json
{"input": [
  {"image": "data:image/jpeg;base64,/9j/4AAQ…"},
  [{"type": "text", "text": "Narration:"},
   {"type": "input_audio", "input_audio": {"data": "UklGR…", "format": "wav"}}]
], "dimensions": 512}
```

`GET /v1/embeddings/prompts` lists the prompts and limits. `encoding_format: "base64"` is supported, so the
OpenAI SDKs work unchanged.

**Formats.** Images: PNG, JPEG, WebP, GIF. Audio: WAV natively, anything else through `ffmpeg`; any sample rate
or channel count is mixed to mono 16 kHz. Video: anything `ffmpeg` decodes (it must be on `PATH`), a `.y4m`
file, or a directory of frames.

## Performance

`embeddinggemma-2-q8_2f.cmf`, cortiq 0.8.15, `cortiq embed --repeat`, median of in-process runs:

| hardware | backend | one query, ms | short texts, texts/s | long documents, tok/s | image (280 tokens), s | video (4 frames), s | audio (30 s), s |
|---|---|---:|---:|---:|---:|---:|---:|
| Mac mini M4 (24 GB) | Metal | 15 | 232 | 3600 | 0.84 | 1.54 | 0.71 |
| Mac mini M4 (24 GB) | CPU, 10 cores | 28 | 128 | 2300 | 1.46 | 2.06 | 1.02 |
| AMD EPYC 7H12, Linux (shared host) | CPU, 30 threads | 51 | 43 | 1170 | 3.2 | 5.1 | 4.2 |

Texts go without a task prompt: one query is 8 tokens, short texts are 256 texts of 32 tokens in one call, long
documents 64 of 512 tokens. The image is 512×512, the video has 4 frames at 140 tokens each, and the audio
clip is 35 s long, cut at 30 s. The `bf16` file runs at about the same speed, except batches of text on the CPU,
which take about 1.3× longer. Memory: about 3.3 GB with all three towers loaded.

`cortiq serve` batches concurrent requests into shared forward passes. On the M4 (Metal) it answers 59
single-text requests per second to one client and 185 to 64 concurrent clients; one request with 256 texts runs
at 233 texts per second. On the EPYC host (CPU) one request with 256 texts runs at 38 texts per second, and a
single text takes about 80 ms.

On Apple silicon the text and image encoders run on the GPU by default and the audio encoder on the CPU;
`CMF_EGEMMA2_GPU=0` keeps everything on the CPU. On Linux and Windows everything runs on the CPU, except the
vision attention of images with 4096 or more patches (`--image-tokens` 560 and 1120), which runs on the GPU when
one is present. The CPU path uses all cores but one (at most 32); `CMF_THREADS=N` sets the thread count.

## Limits

- One input holds at most 8192 tokens, shared by everything in it. Text is cut at that point; an input whose
  media do not fit is rejected.
- Each audio clip is cut at 30 seconds (750 tokens), as in the reference processor. For a longer recording,
  pass 30-second pieces as several `<|audio|>` clips of one input.
- Video uses the frames only: 1 per second, at most 32, spread evenly over longer clips. To include the
  soundtrack, add it as audio to the same input.
- `cortiq convert` builds `bf16`, `f32`, `q8_2f` and `q4tp` files from the original checkpoint. It refuses
  `f16`, because the model's activations overflow half precision.

## Verify the download

```bash
sha256sum -c embeddinggemma-2-q8_2f.cmf.sha256
cortiq verify embeddinggemma-2-q8_2f.cmf
cortiq info embeddinggemma-2-q8_2f.cmf
```

EmbeddingGemma 2 is made by Google DeepMind. The weights derive from Google's release of
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) and remain under its Apache 2.0
license; deployments must follow the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

---

## Документация на русском

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) — модель эмбеддингов Google DeepMind на 740M
параметров. Она переводит текст на 100+ языках, код, изображения, видео и аудио, по отдельности или вперемешку в
одном входе, в общее пространство размерности 768. Здесь текстовый энкодер и башни зрения и звука собраны в один
файл [CMF](https://github.com/infosave2007/cmf). Его запускает `cortiq`, движок на Rust без Python и
ML-фреймворков, на Apple silicon (Metal) и на любом CPU, из командной строки или как сервер с
OpenAI-совместимым `/v1/embeddings`.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/EmbeddingGemma-2-cmf embeddinggemma-2-q8_2f.cmf --local-dir .
cortiq embed --model embeddinggemma-2-q8_2f.cmf --task SearchQuery "рыжая лиса на снегу" --image fox.jpg

cortiq serve embeddinggemma-2-q8_2f.cmf --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/v1/embeddings -H "Content-Type: application/json" \
  -d '{"input": ["Почему светится полярное сияние?"], "prompt_name": "SearchQuery", "dimensions": 256}'
curl http://127.0.0.1:8080/v1/embeddings -H "Content-Type: application/json" -d @- <<EOF
{"input": [{"image": "$PWD/fox.jpg"}, {"audio": "$PWD/vopros.wav"}], "dimensions": 256}
EOF
```

Нужен cortiq 0.8.15 или новее; готовые сборки — на
[странице релизов](https://github.com/infosave2007/cmf/releases). Сервер на 127.0.0.1 читает локальные файлы; на
другом адресе (по умолчанию 0.0.0.0) медиа передаются как `data:`- или `http(s)://`-URL. Файл поменьше —
`embeddinggemma-2-q4tp.cmf` (647 МБ: текстовые векторы те же, что у q8_2f, медиа чуть менее точные), исходные
веса — `embeddinggemma-2-bf16.cmf` (1.52 ГБ); см. [Файлы](#файлы).

### Примеры

Настоящий вывод `embeddinggemma-2-q8_2f.cmf` на файлах из папки `examples/` этого репозитория (подробные таблицы — в
[английском разделе](#examples)):

```bash
hf download infosave/EmbeddingGemma-2-cmf --include "examples/*" --local-dir .
cortiq embed --model embeddinggemma-2-q8_2f.cmf --jsonl examples/search.jsonl   # code, image_to_text, text_to_image, speech
```

| пример | запрос | первое место | второе место |
|---|---|---|---|
| поиск на разных языках (`SearchQuery`) | «почему светится полярное сияние» | Northern lights (en) **0.761** | Где увидеть северное сияние (ru) 0.729 |
| поиск кода (`CodeRetrieval`) | «удалить старые логи командой в терминале» | `cleanup.sh` **0.769** | `longest.rs` 0.672 |
| картинка → текст | [fox_snow.jpg](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/fox_snow.jpg) | «a red fox sitting in the snow» **0.811** | «рыжая лиса сидит на снегу» 0.795 |
| текст → картинка | «неоновая вывеска ночью под дождём» | [neon_sign.jpg](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/neon_sign.jpg) **0.737** | [baikal_poster.jpg](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/baikal_poster.jpg) 0.537 |
| речь → текст | [ru_baikal.wav](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/ru_baikal.wav) «Какое озеро самое глубокое в мире?» | Озеро Байкал (ru) **0.725** | 蓝鲸 (zh) 0.578 |
| матрёшка, `--dim 256` / `128` | первый пример | Northern lights 0.780 / 0.838 | Где увидеть северное сияние 0.767 / 0.823 |

### Файлы

| файл | веса | размер | sha256 | худший косинус | назначение |
|---|---|---:|---|---:|---|
| `embeddinggemma-2-q8_2f.cmf` | матрицы 8 бит; нормы, проекционная голова и свёртки аудио точные | 798 МБ | `ec4192bf6cac…` | 0.99975 | **по умолчанию** |
| `embeddinggemma-2-q4tp.cmf` | как q8_2f, но MLP зрения и feed-forward аудио 4 бита | 647 МБ | `714ac411236e…` | 0.99566 | загрузка поменьше; текстовые векторы те же, что у q8_2f |
| `embeddinggemma-2-bf16.cmf` | исходные веса bf16 байт в байт | 1.52 ГБ | `96b481acd2cc…` | 0.9999992 | точная копия |

Худшее косинусное сходство с `google/embeddinggemma-2` в float32: q8_2f — текст 0.99978, изображения 0.99975,
видео 0.99982, аудио 0.99975, текст + изображение + аудио 0.99985; q4tp — 0.99978, 0.99741, 0.99862, 0.99566,
0.99910; bf16 — 1.000000 (аудио 0.9999992).

На корпусе демо-Space `q4tp` даёт тот же первый результат, что `bf16`, для всех 41 текстового запроса, 31 подписи к
картинкам, видео и аудио и 8 голосовых вопросов при 768 измерениях и для 7 из 8 голосовых вопросов при 256. Сильнее
всего сдвигается речь: у двух голосовых вопросов косинус с `bf16` падает до 0.968 и 0.984 (первый — `ru_baikal.wav`
из примеров), но документ находится тот же. Простой `--quant q4tp` собирает другой файл, с 4-битным текстовым
энкодером (худший косинус текста 0.969); этот собран так:

```bash
hf download google/embeddinggemma-2 --local-dir embeddinggemma-2
cortiq convert --model embeddinggemma-2 --output embeddinggemma-2-q4tp.cmf --quant q4tp \
  --tensor-quant '*.embedding_projection.weight=f16' --tensor-quant 'language_model.*=q8_2f' \
  --tensor-quant '*.mlp.*=q4tp' --tensor-quant 'audio_tower.*.feed_forward*=q4tp' --tensor-quant '*=q8_2f'
```

### Промпты задач

Текст получает короткий префикс, который говорит модели, для чего вектор: `--task` в CLI, `prompt_name` в API.
Запросы: `SearchQuery`, `QuestionAnswering`, `FactChecking`, `CodeRetrieval`, а то, по чему ищут, — `Document`
(`--title` / `"title"` подставляет заголовок). Симметричные задачи: `Classification`, `Clustering`,
`SentenceSimilarity`, один и тот же промпт для всех текстов. Изображения, видео и аудио идут без префикса.
`cortiq embed --list-prompts` показывает все имена.

### Матрёшка

Первые 512, 256 или 128 чисел вектора — самостоятельный эмбеддинг: `--dim` / `dimensions` возвращает префикс,
заново нормированный. По данным Google, MTEB multilingual 61.36 / 61.17 / 60.41 / 57.89 и MMEB 59.01 / 58.38 /
56.24 / 45.65 при 768 / 512 / 256 / 128. 128 лучше оставлять для чисто текстовых задач.

### Изображения, видео, аудио и смешанные входы

```bash
F=embeddinggemma-2-q8_2f.cmf
cortiq embed --model $F --image photo.jpg --image-tokens 560   # 70 | 140 | 280 | 560 | 1120
cortiq embed --model $F --video clip.mp4                       # 1 кадр в секунду, не больше 32
cortiq embed --model $F --audio vopros.wav                     # речь или звуки
cortiq embed --model $F --interleave --image shoe.jpg "Кроссовки для бега: <|image|>"
```

`--interleave` делает один вектор из текста и медиа: каждый `<|image|>`, `<|video|>`, `<|audio|>` берёт
следующий файл своего типа. Все векторы сравнимы между собой: текстовый запрос находит картинки, вопрос голосом —
документы с ответом. В API медиа передаются как `data:`-URL, `http(s)://`-URL или путь к файлу (если сервер слушает
localhost); `--jsonl` принимает пакет входов. Для видео и не-WAV аудио нужен `ffmpeg` в `PATH`.

### Скорость

`embeddinggemma-2-q8_2f.cmf`, cortiq 0.8.15, `cortiq embed --repeat`, медиана замеров внутри процесса:

| железо | бэкенд | один запрос, мс | короткие тексты, текстов/с | длинные документы, ток/с | изображение (280 токенов), с | видео (4 кадра), с | аудио (30 с), с |
|---|---|---:|---:|---:|---:|---:|---:|
| Mac mini M4 (24 ГБ) | Metal | 15 | 232 | 3600 | 0.84 | 1.54 | 0.71 |
| Mac mini M4 (24 ГБ) | CPU, 10 ядер | 28 | 128 | 2300 | 1.46 | 2.06 | 1.02 |
| AMD EPYC 7H12, Linux (общий хост) | CPU, 30 потоков | 51 | 43 | 1170 | 3.2 | 5.1 | 4.2 |

Тексты без промпта задачи: запрос — 8 токенов, короткие тексты — 256 текстов по 32 токена одним вызовом, длинные —
64 по 512 токенов; изображение 512×512, видео из 4 кадров по 140 токенов, аудиоклип 35 с, обрезанный до 30 с.
Файл `bf16` работает примерно с той же скоростью, кроме пакетов текста на CPU: они в 1.3 раза медленнее. Памяти нужно около 3.3 ГБ со
всеми тремя башнями. Сервер объединяет параллельные запросы в общие проходы: на M4 (Metal) это 59 запросов по
одному тексту в секунду от одного клиента и 185 от 64 клиентов; один запрос с 256 текстами — 233 текста в секунду.
На хосте EPYC (CPU) один запрос с 256 текстами идёт со скоростью 38 текстов в секунду, один текст — около 80 мс.

На Apple silicon текстовый энкодер и зрение по умолчанию работают на GPU, аудио — на CPU; `CMF_EGEMMA2_GPU=0`
оставляет всё на процессоре. На Linux и Windows всё считается на CPU, кроме внимания зрения для изображений с 4096
и более патчами (`--image-tokens` 560 и 1120): оно идёт на видеокарте, если она есть. На CPU по
умолчанию заняты все ядра, кроме одного (не больше 32), `CMF_THREADS=N` задаёт число потоков.

### Ограничения

- Один вход — не больше 8192 токенов на всё содержимое. Текст обрезается, вход с медиа сверх лимита отклоняется.
- Аудиоклип обрезается до 30 секунд, как в эталонном процессоре; длинную запись подавайте кусками по 30 с
  несколькими `<|audio|>` в одном входе.
- Видео — только кадры (1 в секунду, не больше 32); звуковую дорожку добавляйте в тот же вход как аудио.
- `cortiq convert` собирает `bf16`, `f32`, `q8_2f` и `q4tp`, но не `f16`: активации модели выходят за его диапазон.

### Проверка загрузки

```bash
sha256sum -c embeddinggemma-2-q8_2f.cmf.sha256
cortiq verify embeddinggemma-2-q8_2f.cmf
```

EmbeddingGemma 2 создана Google DeepMind. Веса получены из релиза
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) и распространяются на условиях Apache 2.0
с соблюдением [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

---

## 中文文档

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) 是 Google DeepMind 的 7.4 亿参数嵌入模型，
可将 100 多种语言的文本、代码、图像、视频和音频（单独或在同一输入中混合）映射到同一个 768 维空间。
本仓库把文本编码器以及视觉、音频编码器打包为一个 [CMF](https://github.com/infosave2007/cmf) 文件，由 `cortiq`
运行——一个不依赖 Python 和机器学习框架的 Rust 推理引擎，支持 Apple silicon（Metal）和任意 CPU，
可在命令行使用，也可作为 OpenAI 兼容的 `/v1/embeddings` 服务器。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/EmbeddingGemma-2-cmf embeddinggemma-2-q8_2f.cmf --local-dir .
cortiq embed --model embeddinggemma-2-q8_2f.cmf --task SearchQuery "雪地里的红狐狸" --image fox.jpg

cortiq serve embeddinggemma-2-q8_2f.cmf --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/v1/embeddings -H "Content-Type: application/json" \
  -d '{"input": ["极光是怎么形成的？"], "prompt_name": "SearchQuery", "dimensions": 256}'
curl http://127.0.0.1:8080/v1/embeddings -H "Content-Type: application/json" -d @- <<EOF
{"input": [{"image": "$PWD/fox.jpg"}, {"audio": "$PWD/question.wav"}], "dimensions": 256}
EOF
```

需要 cortiq 0.8.15 或更新版本；预编译二进制见[发布页面](https://github.com/infosave2007/cmf/releases)。服务器绑定
127.0.0.1 时可读取本地文件；绑定其他地址（默认 0.0.0.0）时，媒体须以 `data:` 或 `http(s)://` URL 传入。
更小的下载：`embeddinggemma-2-q4tp.cmf`（647 MB；文本向量与 q8_2f 完全相同，媒体精度略低）；原始权重：
`embeddinggemma-2-bf16.cmf`（1.52 GB）。见[文件](#文件)。

### 示例

`embeddinggemma-2-q8_2f.cmf` 在本仓库 `examples/` 文件夹中文件上的真实输出（完整表格见[英文部分](#examples)）：

```bash
hf download infosave/EmbeddingGemma-2-cmf --include "examples/*" --local-dir .
cortiq embed --model embeddinggemma-2-q8_2f.cmf --jsonl examples/search.jsonl   # code, image_to_text, text_to_image, speech
```

| 示例 | 查询 | 第一名 | 第二名 |
|---|---|---|---|
| 跨语言检索（`SearchQuery`） | 「почему светится полярное сияние」（极光为什么会发光） | Northern lights (en) **0.761** | Где увидеть северное сияние (ru) 0.729 |
| 代码检索（`CodeRetrieval`） | 「wait until the user stops typing before calling a function」 | `debounce.js` **0.745** | `longest.rs` 0.673 |
| 图像 → 文本 | [fox_snow.jpg](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/fox_snow.jpg) | 「a red fox sitting in the snow」 **0.811** | 「рыжая лиса сидит на снегу」 0.795 |
| 文本 → 图像 | 「a fox on a beach at sunset」 | [fox_beach.jpg](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/fox_beach.jpg) **0.816** | [fox_snow.jpg](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/fox_snow.jpg) 0.720 |
| 语音 → 文本 | [es_whales.wav](https://huggingface.co/infosave/EmbeddingGemma-2-cmf/resolve/main/examples/es_whales.wav)「¿Por qué cantan las ballenas jorobadas?」 | Humpback whale song (en) **0.648** | 蓝鲸 (zh) 0.517 |
| 套娃维度，`--dim 256` / `128` | 第一个示例 | Northern lights 0.780 / 0.838 | Где увидеть северное сияние 0.767 / 0.823 |

### 文件

| 文件 | 权重 | 大小 | sha256 | 最差余弦 | 适用场景 |
|---|---|---:|---|---:|---|
| `embeddinggemma-2-q8_2f.cmf` | 矩阵 8 位；归一化层、投影头和音频卷积保持精确 | 798 MB | `ec4192bf6cac…` | 0.99975 | **默认** |
| `embeddinggemma-2-q4tp.cmf` | 同 q8_2f，但视觉 MLP 和音频前馈层为 4 位 | 647 MB | `714ac411236e…` | 0.99566 | 更小的下载；文本向量与 q8_2f 相同 |
| `embeddinggemma-2-bf16.cmf` | 原始 bf16 权重，逐字节一致 | 1.52 GB | `96b481acd2cc…` | 0.9999992 | 精确参考 |

与 float32 的 `google/embeddinggemma-2` 相比的最差余弦相似度：q8_2f 文本 0.99978、图像 0.99975、视频 0.99982、
音频 0.99975、文本 + 图像 + 音频 0.99985；q4tp 依次为 0.99978、0.99741、0.99862、0.99566、0.99910；
bf16 为 1.000000（音频 0.9999992）。

在演示 Space 的语料上，768 维时 `q4tp` 对全部 41 条文本查询、31 条图像/视频/音频描述和 8 个语音提问给出的第一名
与 `bf16` 相同，256 维时 8 个语音提问中 7 个相同。变化最大的是语音：两个语音提问与 `bf16` 的余弦降至 0.968 和
0.984（前者是示例中的 `ru_baikal.wav`），但找到的文档不变。直接用 `--quant q4tp` 会生成另一种文件
（文本编码器为 4 位，文本最差余弦 0.969）；本文件的生成命令：

```bash
hf download google/embeddinggemma-2 --local-dir embeddinggemma-2
cortiq convert --model embeddinggemma-2 --output embeddinggemma-2-q4tp.cmf --quant q4tp \
  --tensor-quant '*.embedding_projection.weight=f16' --tensor-quant 'language_model.*=q8_2f' \
  --tensor-quant '*.mlp.*=q4tp' --tensor-quant 'audio_tower.*.feed_forward*=q4tp' --tensor-quant '*=q8_2f'
```

### 任务提示词

文本输入需要一个说明用途的短前缀：命令行用 `--task`，API 用 `prompt_name`。查询使用 `SearchQuery`、
`QuestionAnswering`、`FactChecking`、`CodeRetrieval`，被检索的内容使用 `Document`（`--title` / `"title"` 填入标题）。
对称任务 `Classification`、`Clustering`、`SentenceSimilarity` 对所有文本使用同一个提示词。图像、视频和音频不加前缀。
`cortiq embed --list-prompts` 列出全部名称。

### 套娃维度（Matryoshka）

向量的前 512、256 或 128 维本身就是可用的嵌入：`--dim` / `dimensions` 返回重新归一化的前缀。按 Google 的数据，
768 / 512 / 256 / 128 维下 MTEB multilingual 为 61.36 / 61.17 / 60.41 / 57.89，MMEB 为 59.01 / 58.38 / 56.24 / 45.65。
128 维更适合纯文本任务。

### 图像、视频、音频与混合输入

```bash
F=embeddinggemma-2-q8_2f.cmf
cortiq embed --model $F --image photo.jpg --image-tokens 560   # 70 | 140 | 280 | 560 | 1120
cortiq embed --model $F --video clip.mp4                       # 每秒 1 帧，最多 32 帧
cortiq embed --model $F --audio question.wav                   # 语音或声音
cortiq embed --model $F --interleave --image shoe.jpg "跑鞋：<|image|>"
```

`--interleave` 把文本和媒体合成一个向量：每个 `<|image|>`、`<|video|>`、`<|audio|>` 依次取对应类型的下一个文件。
所有向量彼此可比：文本查询可以找到图片，语音提问可以找到回答它的文档。API 中媒体以 `data:` URL、`http(s)://` URL
或本地路径（服务器监听 localhost 时）传入；`--jsonl` 支持批量输入。视频和非 WAV 音频需要 `PATH` 中有 `ffmpeg`。

### 性能

`embeddinggemma-2-q8_2f.cmf`，cortiq 0.8.15，`cortiq embed --repeat`，进程内多次运行的中位数：

| 硬件 | 后端 | 单条查询，ms | 短文本，条/秒 | 长文档，tok/s | 图像（280 token），秒 | 视频（4 帧），秒 | 音频（30 秒），秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| Mac mini M4（24 GB） | Metal | 15 | 232 | 3600 | 0.84 | 1.54 | 0.71 |
| Mac mini M4（24 GB） | CPU，10 核 | 28 | 128 | 2300 | 1.46 | 2.06 | 1.02 |
| AMD EPYC 7H12，Linux（共享主机） | CPU，30 线程 | 51 | 43 | 1170 | 3.2 | 5.1 | 4.2 |

文本不加任务提示词：查询为 8 个 token，短文本为一次调用中的 256 条 32 token 的文本，长文档为 64 篇 512 token；图像 512×512，
视频 4 帧、每帧 140 token，音频片段 35 秒、截取前 30 秒。`bf16` 文件速度基本相同，只有 CPU 上的批量文本慢约 1.3 倍。
加载全部三个编码器约需 3.3 GB 内存。服务器会把并发请求合并计算：在 M4（Metal）上，单客户端每秒 59 个单文本请求，64 个并发客户端时每秒 185 个；单个含 256 条文本的请求每秒处理 233 条。
在 EPYC 主机（CPU）上，单个含 256 条文本的请求每秒处理 38 条，单条文本约 80 ms。

在 Apple silicon 上，文本和视觉编码器默认在 GPU 上运行，音频编码器在 CPU 上；`CMF_EGEMMA2_GPU=0` 让全部计算留在 CPU。
在 Linux 和 Windows 上，全部计算在 CPU 上进行，只有 4096 个及以上 patch 的图像（`--image-tokens` 560 和 1120）的视觉注意力在有显卡时在 GPU 上运行。
CPU 默认使用除一个核心外的全部核心（最多 32 个），`CMF_THREADS=N` 设置线程数。

### 限制

- 每个输入最多 8192 个 token，由其中所有内容共享。文本超出部分被截断，媒体超出限制的输入会被拒绝。
- 每段音频截取前 30 秒，与参考处理器一致；更长的录音请切成 30 秒的片段，作为同一输入中的多个 `<|audio|>`。
- 视频只使用画面（每秒 1 帧，最多 32 帧）；如需包含声音，请把音轨作为音频加入同一输入。
- `cortiq convert` 可生成 `bf16`、`f32`、`q8_2f` 和 `q4tp` 文件，但不支持 `f16`：模型的激活值超出半精度范围。

### 校验下载

```bash
sha256sum -c embeddinggemma-2-q8_2f.cmf.sha256
cortiq verify embeddinggemma-2-q8_2f.cmf
```

EmbeddingGemma 2 由 Google DeepMind 开发。权重来自 Google 发布的
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)，遵循其 Apache 2.0 许可，使用时须遵守
[Gemma 禁止使用政策](https://ai.google.dev/gemma/prohibited_use_policy)。
