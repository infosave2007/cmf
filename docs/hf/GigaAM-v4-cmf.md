---
license: mit
library_name: cortiq
base_model: ai-sage/GigaAM-v4
base_model_relation: quantized
pipeline_tag: automatic-speech-recognition
tags:
  - cmf
  - cortiq
  - quantized
  - speech-recognition
  - asr
  - rnnt
  - ctc
  - cuda
  - vulkan
  - metal
language:
  - ru
  - en
---

# GigaAM-v4 — CMF

[GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4) is a speech recognition model for Russian and English: a
238 M-parameter Zapformer encoder with two recognition heads on top — formatted text with case, punctuation and
digits, and plain lower-case words — as RNN-T or CTC. Here it is packaged as single
[CMF](https://github.com/infosave2007/cmf) files for `cortiq`, a Rust inference engine with no Python and no ML
framework: NVIDIA GPUs through CUDA or Vulkan, AMD and Intel GPUs through Vulkan/DX12, Apple silicon through Metal,
and the CPU. A recording of any length goes in: each file carries the model's own voice-activity network, which
cuts long audio into segments of up to 30 s. Measured:

- **NVIDIA L4 (24 GB), CUDA:** 30 min of speech in 2.73 s — 660 times faster than real
  time; a 30 s clip in 95 ms, in 1.6 GB of the card.
- **The same card over Vulkan:** 30 min in 4.64 s (388× real time), a 30 s clip in 117 ms.
- **MacBook Air M4 (24 GB):** a 30 s clip in 154 ms through Metal and 269 ms on the CPU alone; 30 min in 10.6 s and 16.7 s.
- **Accuracy:** the word error rate of the 8-bit file is the reference implementation's: 4.93 % against 4.96 % on Golos Farfield, 14.15 % against 14.19 % on Golos Crowd, 4.29 % against 4.27 % on Russian LibriSpeech.
  With the released half-precision weights 13 261 of 13 262 transcripts are the reference's, character for character.
- **Against the authors' PyTorch code on the same L4:** the same 30 min take it 7.71 s in float32 and 4.78 s with
  its half-precision encoder; `cortiq` takes 2.73 s.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/GigaAM-v4-cmf GigaAM-v4-rnnt-q8_2f.cmf --local-dir .
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf speech.wav
```

Use cortiq 0.9.21 or later.

Prebuilt binaries for Linux, macOS and Windows are on the [releases page](https://github.com/infosave2007/cmf/releases).
Add `--features cuda` to the install for the CUDA backend (`CMF_GPU=cuda`). WAV is read directly; mp3, flac, ogg,
m4a and the rest go through `ffmpeg` when it is on the PATH. Any sample rate and channel count: the audio is
mixed down to mono at 16 kHz.

## Files

| file | heads | weights | size | WER, Golos Farfield | WER, Golos Crowd | WER, Russian LibriSpeech |
|---|---|---|---:|---:|---:|---:|
| `GigaAM-v4-rnnt-q8_2f.cmf` | RNN-T | 8-bit, **default** | 260 MB | 4.93 | 14.15 | 4.29 |
| `GigaAM-v4-ctc-q8_2f.cmf` | CTC | 8-bit | 246 MB | 5.29 | 14.80 | 4.65 |
| reference: the authors' PyTorch code, RNN-T |  | float32 | — | 4.96 | 14.19 | 4.27 |
| reference: the authors' PyTorch code, CTC |  | float32 | — | 5.27 | 14.81 | 4.72 |

Word error rate of the formatted transcript (`e2e` head) on the test sets' own clips, the reference run on the
same clips, both sides through the reference package's text normalization (lower case, ё → е, punctuation
removed; numbers the model writes as digits count against a reference that spells them out, for every row
alike). RNN-T is the more accurate of the two and the upstream default; CTC decodes in one pass and is the
faster one. Every file holds both heads, the mel front end's filter bank and the voice-activity network.
Not here: the `ssl` revision (an encoder without a recognition head) and the streaming modes (160 / 320 / 640 ms
of look-ahead) — these files run the full-context mode. A 4-bit file is not published: 4-bit rounding without retraining costs accuracy (RNN-T on Russian LibriSpeech: 4.90 % against 4.27 %).

## Performance

One recording a run, `cortiq transcribe --stats`, the second pass in one process (weights already on the device):
the time from samples to transcript — voice-activity cut, features, encoder, decoding — and the recording's
length divided by it. The recordings are test-set clips laid end to end. Card memory is the peak `nvidia-smi`
shows above the idle card. Read each backend's section on its own.

### CUDA · NVIDIA L4

Stand: NVIDIA L4 (24 GB), Google Colab (12 vCPU), the CUDA backend (`CMF_GPU=cuda`). The voice-activity network, the convolutional front end and the encoder run on the card (TF32 products); the host computes the mel features and the greedy decoding.

![The files on CUDA](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_cuda_en.png)

| recording | time, s (RNN-T `q8_2f`) | time, s (CTC `q8_2f`) | × real time (RNN-T `q8_2f`) | × real time (CTC `q8_2f`) | peak card memory, GB |
|---:|---:|---:|---:|---:|---:|
| 10 s | 0.037 | 0.011 | 270 | 909 | 1.3 |
| 30 s | 0.095 | 0.025 | 316 | 1200 | 1.4 |
| 1 min | 0.114 | 0.056 | 526 | 1071 | 1.4 |
| 5 min | 0.487 | 0.301 | 616 | 997 | 1.5 |
| 30 min | 2.73 | 1.87 | 660 | 963 | 1.6 |
| 2 h | 12.3 | 8.72 | 588 | 825 | 1.6 |

### Vulkan · NVIDIA L4

Stand: the same NVIDIA L4 (24 GB), over Vulkan (the default backend off macOS).

![The files on Vulkan](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_vulkan_en.png)

| recording | time, s (RNN-T `q8_2f`) | time, s (CTC `q8_2f`) | × real time (RNN-T `q8_2f`) | × real time (CTC `q8_2f`) | peak card memory, GB |
|---:|---:|---:|---:|---:|---:|
| 10 s | 0.052 | 0.026 | 192 | 385 | 1.2 |
| 30 s | 0.117 | 0.049 | 256 | 612 | 1.2 |
| 1 min | 0.178 | 0.108 | 337 | 556 | 1.2 |
| 5 min | 0.823 | 0.637 | 364 | 471 | 1.4 |
| 30 min | 4.64 | 3.80 | 388 | 474 | 1.4 |
| 2 h | 22.7 | 18.9 | 318 | 381 | 1.5 |

### Metal · MacBook Air M4

Stand: MacBook Air M4 (24 GB), fanless, Metal (the default backend on macOS). The encoder runs on the GPU; memory is the process's (the Mac's memory is shared with the GPU).

![The files on Metal](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_metal_en.png)

| recording | time, s (RNN-T `q8_2f`) | time, s (CTC `q8_2f`) | × real time (RNN-T `q8_2f`) | × real time (CTC `q8_2f`) | peak process memory, GB |
|---:|---:|---:|---:|---:|---:|
| 10 s | 0.051 | 0.053 | 196 | 189 | 1.2 |
| 30 s | 0.154 | 0.149 | 195 | 201 | 1.2 |
| 1 min | 0.298 | 0.297 | 201 | 202 | 1.3 |
| 5 min | 1.64 | 1.67 | 182 | 179 | 1.4 |
| 30 min | 10.6 | 10.0 | 170 | 179 | 1.8 |

### CPU · MacBook Air M4

Stand: MacBook Air M4 (24 GB), fanless, CPU only (`CMF_GPU=0`): Accelerate's matrix units do the products.

![The files on the M4's CPU](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_cpu_m4_en.png)

| recording | time, s (RNN-T `q8_2f`) | time, s (CTC `q8_2f`) | × real time (RNN-T `q8_2f`) | × real time (CTC `q8_2f`) | peak process memory, GB |
|---:|---:|---:|---:|---:|---:|
| 10 s | 0.122 | 0.117 | 82 | 86 | 0.3 |
| 30 s | 0.269 | 0.289 | 112 | 104 | 0.4 |
| 1 min | 0.471 | 0.475 | 127 | 126 | 0.4 |
| 5 min | 2.70 | 2.47 | 111 | 121 | 0.6 |
| 30 min | 16.7 | 15.3 | 108 | 118 | 0.9 |

### CPU · Xeon, 12 vCPU

Stand: the L4 instance's host, Intel Xeon 2.2 GHz, 12 vCPU, no GPU (`CMF_GPU=0`).

![The files on a server CPU](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_cpu_en.png)

| recording | time, s (RNN-T `q8_2f`) | time, s (CTC `q8_2f`) | × real time (RNN-T `q8_2f`) | × real time (CTC `q8_2f`) | peak process memory, GB |
|---:|---:|---:|---:|---:|---:|
| 10 s | 0.517 | 0.572 | 19 | 18 | 0.3 |
| 30 s | 1.58 | 1.50 | 19 | 20 | 0.3 |
| 1 min | 2.68 | 2.64 | 22 | 23 | 0.4 |
| 5 min | 13.0 | 13.1 | 23 | 23 | 0.5 |
| 30 min | 78.6 | 78.3 | 23 | 23 | 0.6 |

## Usage

```bash
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf speech.wav                 # the formatted transcript
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf speech.wav --head raw      # lower-case words, no punctuation
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf speech.wav --timestamps    # every word with its start and end
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf a.wav b.mp3 c.ogg          # several files, one load
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf talk.wav --json --head both  # text, segments, word times, timings
```

A recording of up to 30 s is one segment. A longer one is cut where the voice-activity network finds pauses
(segments of at most 30 s, as the reference package's `transcribe_longform` cuts them) and the segments'
transcripts are joined; `--vad` applies the cut to a short clip too. `--stats` prints the time of each stage.

### OpenAI-compatible server

```bash
cortiq serve GigaAM-v4-rnnt-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/audio/transcriptions -F file=@speech.wav
curl http://localhost:8080/v1/audio/transcriptions -F file=@talk.mp3 \
  -F response_format=verbose_json -F "timestamp_granularities[]=word"
curl http://localhost:8080/v1/audio/transcriptions -F file=@talk.mp3 -F response_format=srt
```

`POST /v1/audio/transcriptions` takes the multipart form the OpenAI clients send (`file`, `response_format`:
`json`, `text`, `verbose_json`, `srt`, `vtt`; `timestamp_granularities[]`: `segment`, `word`), or JSON with
`file` as a `data:` / `http(s)` URL or `{"data": <base64>}`. `head=raw` asks for the plain-words transcript.
`/v1/models` and `/healthz` describe the loaded file.

### Settings

| variable | effect |
|---|---|
| `CMF_GPU=cuda` / `1` / `0` | the CUDA backend / the default GPU backend (Vulkan, Metal) / the CPU |
| `CMF_CUDA_PREC=exact` | plain float32 products on CUDA instead of TF32: the CPU's transcript, character for character, at about half the encoder speed |
| `CMF_GIGAAM_GEMM=f32` | the same on Vulkan |
| `CMF_GIGAAM_DENSE=1` | CPU: keep float copies of the weights (about 0.95 GB) — a short clip then pays no unpacking |
| `CMF_THREADS=N` | CPU threads |

## Accuracy

Word error rate, %, of each file and of the reference implementation on the same clips with the same text
normalization; the last column counts the clips whose normalized transcript is the reference's exactly.
The published files ran on CUDA at its default precision.

Formatted transcript (`e2e` head):

**RNN-T**

| file | Golos Farfield, 1 916 | Golos Crowd, 9 994 | Russian LibriSpeech, 1 352 | transcripts identical to the reference |
|---|---:|---:|---:|---:|
| reference: the authors' PyTorch code, float32 | 4.96 | 14.19 | 4.27 | — |
| CMF, the released half-precision weights, float32 products (not uploaded) | 4.96 | 14.19 | 4.27 | 13 261 / 13 262 |
| `GigaAM-v4-rnnt-q8_2f.cmf` | 4.93 | 14.15 | 4.29 | 13 138 / 13 262 |

**CTC**

| file | Golos Farfield, 1 916 | Golos Crowd, 9 994 | Russian LibriSpeech, 1 352 | transcripts identical to the reference |
|---|---:|---:|---:|---:|
| reference: the authors' PyTorch code, float32 | 5.27 | 14.81 | 4.72 | — |
| CMF, the released half-precision weights, float32 products (not uploaded) | 5.29 | 14.81 | 4.72 | 13 260 / 13 262 |
| `GigaAM-v4-ctc-q8_2f.cmf` | 5.29 | 14.80 | 4.65 | 13 011 / 13 262 |

Plain words (`raw` head):

**RNN-T**

| file | Golos Farfield, 1 916 | Golos Crowd, 9 994 | Russian LibriSpeech, 1 352 | transcripts identical to the reference |
|---|---:|---:|---:|---:|
| reference: the authors' PyTorch code, float32 | 3.73 | 4.69 | 4.17 | — |
| CMF, the released half-precision weights, float32 products (not uploaded) | 3.72 | 4.69 | 4.17 | 13 261 / 13 262 |
| `GigaAM-v4-rnnt-q8_2f.cmf` | 3.70 | 4.71 | 4.21 | 13 164 / 13 262 |

**CTC**

| file | Golos Farfield, 1 916 | Golos Crowd, 9 994 | Russian LibriSpeech, 1 352 | transcripts identical to the reference |
|---|---:|---:|---:|---:|
| reference: the authors' PyTorch code, float32 | 4.29 | 5.40 | 4.60 | — |
| CMF, the released half-precision weights, float32 products (not uploaded) | 4.29 | 5.40 | 4.60 | 13 261 / 13 262 |
| `GigaAM-v4-ctc-q8_2f.cmf` | 4.32 | 5.39 | 4.54 | 13 026 / 13 262 |

## Verify the download

```bash
cortiq verify GigaAM-v4-rnnt-q8_2f.cmf
cortiq info GigaAM-v4-rnnt-q8_2f.cmf
```

| file | sha256 |
|---|---|
| `GigaAM-v4-rnnt-q8_2f.cmf` | `caf9653d0010e17a3d37cc6436765c741762d0049894712a2dce2c87a1d9d4f2` |
| `GigaAM-v4-ctc-q8_2f.cmf` | `372aaf0f566eb39184eaded5d37884014d3a39cb9cac989dbcb9cb0474daa8d6` |

Weights: [MIT](https://huggingface.co/ai-sage/GigaAM-v4/blob/main/LICENSE), from
[ai-sage/GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4). The model card's notes on intended use and
limitations apply to these files as well.

---

## Документация на русском

[GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4) — модель распознавания речи для русского и английского:
Zapformer-энкодер на 238 млн параметров и две головы распознавания над ним — оформленный текст с регистром,
пунктуацией и цифрами и простой текст строчными словами — в вариантах RNN-T и CTC. Здесь она упакована в
отдельные файлы [CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка на Rust без Python и
ML-фреймворков: GPU NVIDIA через CUDA или Vulkan, AMD и Intel через Vulkan/DX12, Apple silicon через Metal, а
также CPU. На вход идёт запись любой длины: в каждом файле лежит собственная сеть голосовой активности модели,
которая режет длинное аудио на сегменты до 30 с. Измерено:

- **NVIDIA L4 (24 ГБ), CUDA:** 30 мин речи за 2,73 с — в 660 раз быстрее реального
  времени; клип 30 с — за 95 мс, в 1,6 ГБ карты.
- **Та же карта через Vulkan:** 30 мин за 4,64 с (388× реального времени), клип 30 с — за 117 мс.
- **MacBook Air M4 (24 ГБ):** клип 30 с за 154 мс через Metal и за 269 мс на одном CPU; 30 мин — за 10,6 с и 16,7 с.
- **Точность:** доля ошибок по словам у 8-битного файла та же, что у эталонной реализации: 4,93 % против 4,96 % на Golos Farfield, 14,15 % против 14,19 % на Golos Crowd, 4,29 % против 4,27 % на Russian LibriSpeech.
  С исходными весами половинной точности 13 261 из 13 262 транскриптов совпадают с эталоном символ в символ.
- **Против кода авторов на PyTorch на той же L4:** те же 30 мин занимают у него 7,71 с во float32 и 4,78 с с
  энкодером половинной точности; у `cortiq` — 2,73 с.

Быстрый старт:

```bash
cargo install cortiq-cli
hf download infosave/GigaAM-v4-cmf GigaAM-v4-rnnt-q8_2f.cmf --local-dir .
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf speech.wav
```

WAV читается напрямую; mp3, flac, ogg, m4a и прочее — через `ffmpeg`, если он есть в PATH. Частота и число
каналов любые: звук сводится в моно 16 кГц. Таблица файлов с WER — в разделе [Files](#files): RNN-T точнее и
стоит у авторов по умолчанию, CTC декодирует за один проход и быстрее. В каждом файле обе головы и сеть
голосовой активности. Не выложены ревизия `ssl` (энкодер без головы распознавания) и потоковые режимы
(160 / 320 / 640 мс заглядывания вперёд): файлы работают в режиме полного контекста. 4-битный файл не выложен: 4-битное округление без дообучения стоит точности (RNN-T на Russian LibriSpeech: 4,90 % против 4,27 %).

### Скорость

Одна запись за прогон, `cortiq transcribe --stats`, второй проход в одном процессе (веса уже на устройстве):
время от отсчётов до транскрипта — нарезка по голосовой активности, признаки, энкодер, декодирование — и длина
записи, делённая на него. Записи — клипы тестовых наборов, уложенные подряд. Память карты — пик по `nvidia-smi`
сверх простаивающей карты. Каждый раздел читается сам по себе.

**CUDA · NVIDIA L4**

Стенд: NVIDIA L4 (24 ГБ), Google Colab (12 vCPU), бэкенд CUDA (`CMF_GPU=cuda`). Сеть голосовой активности, свёрточный вход и энкодер считаются на карте (произведения TF32); на хосте — mel-признаки и жадное декодирование.

![CUDA · NVIDIA L4](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_cuda_ru.png)

| запись | время, с (RNN-T `q8_2f`) | время, с (CTC `q8_2f`) | × реального времени (RNN-T `q8_2f`) | × реального времени (CTC `q8_2f`) | пик памяти карты, ГБ |
|---:|---:|---:|---:|---:|---:|
| 10 с | 0,037 | 0,011 | 270 | 909 | 1,3 |
| 30 с | 0,095 | 0,025 | 316 | 1200 | 1,4 |
| 1 мин | 0,114 | 0,056 | 526 | 1071 | 1,4 |
| 5 мин | 0,487 | 0,301 | 616 | 997 | 1,5 |
| 30 мин | 2,73 | 1,87 | 660 | 963 | 1,6 |
| 2 ч | 12,3 | 8,72 | 588 | 825 | 1,6 |

**Vulkan · NVIDIA L4**

Стенд: та же NVIDIA L4 (24 ГБ), Vulkan (бэкенд по умолчанию вне macOS).

![Vulkan · NVIDIA L4](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_vulkan_ru.png)

| запись | время, с (RNN-T `q8_2f`) | время, с (CTC `q8_2f`) | × реального времени (RNN-T `q8_2f`) | × реального времени (CTC `q8_2f`) | пик памяти карты, ГБ |
|---:|---:|---:|---:|---:|---:|
| 10 с | 0,052 | 0,026 | 192 | 385 | 1,2 |
| 30 с | 0,117 | 0,049 | 256 | 612 | 1,2 |
| 1 мин | 0,178 | 0,108 | 337 | 556 | 1,2 |
| 5 мин | 0,823 | 0,637 | 364 | 471 | 1,4 |
| 30 мин | 4,64 | 3,80 | 388 | 474 | 1,4 |
| 2 ч | 22,7 | 18,9 | 318 | 381 | 1,5 |

**Metal · MacBook Air M4**

Стенд: MacBook Air M4 (24 ГБ), без вентилятора, Metal (бэкенд по умолчанию на macOS). Энкодер считается на GPU; память — всего процесса (память Mac общая с GPU).

![Metal · MacBook Air M4](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_metal_ru.png)

| запись | время, с (RNN-T `q8_2f`) | время, с (CTC `q8_2f`) | × реального времени (RNN-T `q8_2f`) | × реального времени (CTC `q8_2f`) | пик памяти процесса, ГБ |
|---:|---:|---:|---:|---:|---:|
| 10 с | 0,051 | 0,053 | 196 | 189 | 1,2 |
| 30 с | 0,154 | 0,149 | 195 | 201 | 1,2 |
| 1 мин | 0,298 | 0,297 | 201 | 202 | 1,3 |
| 5 мин | 1,64 | 1,67 | 182 | 179 | 1,4 |
| 30 мин | 10,6 | 10,0 | 170 | 179 | 1,8 |

**CPU · MacBook Air M4**

Стенд: MacBook Air M4 (24 ГБ), без вентилятора, только CPU (`CMF_GPU=0`): произведения считают матричные блоки Accelerate.

![CPU · MacBook Air M4](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_cpu_m4_ru.png)

| запись | время, с (RNN-T `q8_2f`) | время, с (CTC `q8_2f`) | × реального времени (RNN-T `q8_2f`) | × реального времени (CTC `q8_2f`) | пик памяти процесса, ГБ |
|---:|---:|---:|---:|---:|---:|
| 10 с | 0,122 | 0,117 | 82 | 86 | 0,3 |
| 30 с | 0,269 | 0,289 | 112 | 104 | 0,4 |
| 1 мин | 0,471 | 0,475 | 127 | 126 | 0,4 |
| 5 мин | 2,70 | 2,47 | 111 | 121 | 0,6 |
| 30 мин | 16,7 | 15,3 | 108 | 118 | 0,9 |

**CPU · Xeon, 12 vCPU**

Стенд: хост инстанса с L4, Intel Xeon 2,2 ГГц, 12 vCPU, без GPU (`CMF_GPU=0`).

![CPU · Xeon, 12 vCPU](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_cpu_ru.png)

| запись | время, с (RNN-T `q8_2f`) | время, с (CTC `q8_2f`) | × реального времени (RNN-T `q8_2f`) | × реального времени (CTC `q8_2f`) | пик памяти процесса, ГБ |
|---:|---:|---:|---:|---:|---:|
| 10 с | 0,517 | 0,572 | 19 | 18 | 0,3 |
| 30 с | 1,58 | 1,50 | 19 | 20 | 0,3 |
| 1 мин | 2,68 | 2,64 | 22 | 23 | 0,4 |
| 5 мин | 13,0 | 13,1 | 23 | 23 | 0,5 |
| 30 мин | 78,6 | 78,3 | 23 | 23 | 0,6 |

### Использование

```bash
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf speech.wav                 # оформленный транскрипт
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf speech.wav --head raw      # строчные слова без пунктуации
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf speech.wav --timestamps    # каждое слово с началом и концом
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf a.wav b.mp3 c.ogg          # несколько файлов за одну загрузку
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf talk.wav --json --head both  # текст, сегменты, времена слов, тайминги
```

Запись до 30 с — один сегмент. Более длинная режется там, где сеть голосовой активности находит паузы
(сегменты не длиннее 30 с, как режет `transcribe_longform` из пакета авторов), транскрипты сегментов
соединяются; `--vad` включает нарезку и для короткого клипа. `--stats` печатает время каждой стадии.

Сервер с API в формате OpenAI:

```bash
cortiq serve GigaAM-v4-rnnt-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/audio/transcriptions -F file=@speech.wav
curl http://localhost:8080/v1/audio/transcriptions -F file=@talk.mp3 \
  -F response_format=verbose_json -F "timestamp_granularities[]=word"
```

`POST /v1/audio/transcriptions` принимает multipart-форму клиентов OpenAI (`file`, `response_format`: `json`,
`text`, `verbose_json`, `srt`, `vtt`; `timestamp_granularities[]`: `segment`, `word`) или JSON, где `file` —
URL `data:` / `http(s)` либо `{"data": <base64>}`. `head=raw` запрашивает транскрипт простыми словами.

Настройки: `CMF_GPU=cuda` / `1` / `0` — бэкенд CUDA / GPU по умолчанию (Vulkan, Metal) / CPU;
`CMF_CUDA_PREC=exact` (на Vulkan `CMF_GIGAAM_GEMM=f32`) — обычные float32-произведения вместо TF32: транскрипт
CPU символ в символ при примерно вдвое меньшей скорости энкодера; `CMF_GIGAAM_DENSE=1` — на CPU держать
веса во float (около 0,95 ГБ), короткий клип тогда не платит за распаковку; `CMF_THREADS=N` — потоки CPU.

### Точность

Доля ошибок по словам, %, у каждого файла и у эталонной реализации на тех же клипах с той же нормализацией
текста (нижний регистр, ё → е, без пунктуации; числа, которые модель пишет цифрами, считаются ошибкой против
разметки прописью — одинаково для всех строк). Последний столбец — клипы, чей нормализованный транскрипт
совпал с эталоном полностью. Выложенные файлы сняты на CUDA с точностью по умолчанию.

Оформленный транскрипт (голова `e2e`):

**RNN-T**

| файл | Golos Farfield, 1 916 | Golos Crowd, 9 994 | Russian LibriSpeech, 1 352 | транскрипты, совпавшие с эталоном |
|---|---:|---:|---:|---:|
| эталон: код авторов на PyTorch, float32 | 4,96 | 14,19 | 4,27 | — |
| CMF, исходные веса половинной точности, float32-произведения (не выложен) | 4,96 | 14,19 | 4,27 | 13 261 / 13 262 |
| `GigaAM-v4-rnnt-q8_2f.cmf` | 4,93 | 14,15 | 4,29 | 13 138 / 13 262 |

**CTC**

| файл | Golos Farfield, 1 916 | Golos Crowd, 9 994 | Russian LibriSpeech, 1 352 | транскрипты, совпавшие с эталоном |
|---|---:|---:|---:|---:|
| эталон: код авторов на PyTorch, float32 | 5,27 | 14,81 | 4,72 | — |
| CMF, исходные веса половинной точности, float32-произведения (не выложен) | 5,29 | 14,81 | 4,72 | 13 260 / 13 262 |
| `GigaAM-v4-ctc-q8_2f.cmf` | 5,29 | 14,80 | 4,65 | 13 011 / 13 262 |

Простые слова (голова `raw`):

**RNN-T**

| файл | Golos Farfield, 1 916 | Golos Crowd, 9 994 | Russian LibriSpeech, 1 352 | транскрипты, совпавшие с эталоном |
|---|---:|---:|---:|---:|
| эталон: код авторов на PyTorch, float32 | 3,73 | 4,69 | 4,17 | — |
| CMF, исходные веса половинной точности, float32-произведения (не выложен) | 3,72 | 4,69 | 4,17 | 13 261 / 13 262 |
| `GigaAM-v4-rnnt-q8_2f.cmf` | 3,70 | 4,71 | 4,21 | 13 164 / 13 262 |

**CTC**

| файл | Golos Farfield, 1 916 | Golos Crowd, 9 994 | Russian LibriSpeech, 1 352 | транскрипты, совпавшие с эталоном |
|---|---:|---:|---:|---:|
| эталон: код авторов на PyTorch, float32 | 4,29 | 5,40 | 4,60 | — |
| CMF, исходные веса половинной точности, float32-произведения (не выложен) | 4,29 | 5,40 | 4,60 | 13 261 / 13 262 |
| `GigaAM-v4-ctc-q8_2f.cmf` | 4,32 | 5,39 | 4,54 | 13 026 / 13 262 |

### Проверка загрузки

`cortiq verify GigaAM-v4-rnnt-q8_2f.cmf` сверяет хеши секций файла; sha256 файлов — в таблице
[Verify the download](#verify-the-download). Веса: [MIT](https://huggingface.co/ai-sage/GigaAM-v4/blob/main/LICENSE),
из [ai-sage/GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4); замечания карточки модели о назначении и
ограничениях относятся и к этим файлам.

---

## 中文文档

[GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4) 是面向俄语和英语的语音识别模型：2.38 亿参数的 Zapformer
编码器，上面有两个识别头——带大小写、标点和数字的规范文本，以及纯小写词——分为 RNN-T 和 CTC 两种版本。这里把它
打包成单个 [CMF](https://github.com/infosave2007/cmf) 文件，由 `cortiq` 运行：这是一个用 Rust 写的推理引擎，不依赖
Python 和任何机器学习框架，NVIDIA 显卡走 CUDA 或 Vulkan，AMD 和 Intel 显卡走 Vulkan/DX12，Apple 芯片走 Metal，也可以
只用 CPU。任意长度的录音都可以直接输入：每个文件都带有模型自己的语音活动检测网络，把长音频切成不超过 30 秒的片段。
实测：

- **NVIDIA L4（24 GB），CUDA：** 30 分钟语音用时 2.73 秒，是实时的 660 倍；30 秒的片段用时 95 毫秒，占用显存 1.6 GB。
- **同一张卡走 Vulkan：** 30 分钟用时 4.64 秒（388 倍实时），30 秒的片段用时 117 毫秒。
- **MacBook Air M4（24 GB）：** 30 秒的片段，Metal 用时 154 毫秒，只用 CPU 用时 269 毫秒；30 分钟分别用时 10.6 秒和 16.7 秒。
- **准确率：** 8 位文件的词错误率与参考实现相同：Golos Farfield 上 4.93 % 对 4.96 %，Golos Crowd 上 14.15 % 对 14.19 %，Russian LibriSpeech 上 4.29 % 对 4.27 %。
  使用原始半精度权重时，13 262 条转写中有 13 261 条与参考实现逐字符一致。
- **与同一张 L4 上作者的 PyTorch 代码相比：** 同样的30 分钟，它用 float32 需要 7.71 秒，用半精度编码器需要 4.78 秒；`cortiq` 用时 2.73 秒。

快速开始：

```bash
cargo install cortiq-cli
hf download infosave/GigaAM-v4-cmf GigaAM-v4-rnnt-q8_2f.cmf --local-dir .
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf speech.wav
```

WAV 直接读取；mp3、flac、ogg、m4a 等在 PATH 中有 `ffmpeg` 时通过它解码。采样率和声道数不限：音频会混成 16 kHz 单声道。
带 WER 的文件表见 [Files](#files)：RNN-T 更准确，也是上游的默认版本；CTC 一遍解码，速度更快。每个文件都包含两个识别头
和语音活动检测网络。未提供的有：`ssl` 版本（没有识别头的编码器）和流式模式（160 / 320 / 640 毫秒前瞻）——这些文件以
全上下文模式运行。 未发布 4 位文件：不经再训练的 4 位量化会损失准确率（RNN-T 在 Russian LibriSpeech 上为 4.90 %，参考为 4.27 %）。

### 性能

每次运行一段录音，`cortiq transcribe --stats`，同一进程内的第二遍（权重已在设备上）：从采样到转写的时间——语音活动
切分、特征、编码器、解码——以及录音长度与它之比。录音由测试集的片段首尾相接而成。显存为 `nvidia-smi` 显示的、高于
空闲状态的峰值。各后端的小节请分别阅读。

**CUDA · NVIDIA L4**

测试平台：NVIDIA L4（24 GB），Google Colab（12 vCPU），CUDA 后端（`CMF_GPU=cuda`）。语音活动检测网络、卷积前端和编码器在显卡上运行（TF32 乘法）；主机计算 mel 特征和贪心解码。

![CUDA · NVIDIA L4](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_cuda_zh.png)

| 录音 | 耗时，秒 (RNN-T `q8_2f`) | 耗时，秒 (CTC `q8_2f`) | 倍实时 (RNN-T `q8_2f`) | 倍实时 (CTC `q8_2f`) | 显存峰值，GB |
|---:|---:|---:|---:|---:|---:|
| 10 秒 | 0.037 | 0.011 | 270 | 909 | 1.3 |
| 30 秒 | 0.095 | 0.025 | 316 | 1200 | 1.4 |
| 1 分钟 | 0.114 | 0.056 | 526 | 1071 | 1.4 |
| 5 分钟 | 0.487 | 0.301 | 616 | 997 | 1.5 |
| 30 分钟 | 2.73 | 1.87 | 660 | 963 | 1.6 |
| 2 小时 | 12.3 | 8.72 | 588 | 825 | 1.6 |

**Vulkan · NVIDIA L4**

测试平台：同一张 NVIDIA L4（24 GB），走 Vulkan（macOS 之外的默认后端）。

![Vulkan · NVIDIA L4](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_vulkan_zh.png)

| 录音 | 耗时，秒 (RNN-T `q8_2f`) | 耗时，秒 (CTC `q8_2f`) | 倍实时 (RNN-T `q8_2f`) | 倍实时 (CTC `q8_2f`) | 显存峰值，GB |
|---:|---:|---:|---:|---:|---:|
| 10 秒 | 0.052 | 0.026 | 192 | 385 | 1.2 |
| 30 秒 | 0.117 | 0.049 | 256 | 612 | 1.2 |
| 1 分钟 | 0.178 | 0.108 | 337 | 556 | 1.2 |
| 5 分钟 | 0.823 | 0.637 | 364 | 471 | 1.4 |
| 30 分钟 | 4.64 | 3.80 | 388 | 474 | 1.4 |
| 2 小时 | 22.7 | 18.9 | 318 | 381 | 1.5 |

**Metal · MacBook Air M4**

测试平台：MacBook Air M4（24 GB），无风扇，Metal（macOS 上的默认后端）。编码器在 GPU 上运行；内存为整个进程的占用（Mac 的内存与 GPU 共享）。

![Metal · MacBook Air M4](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_metal_zh.png)

| 录音 | 耗时，秒 (RNN-T `q8_2f`) | 耗时，秒 (CTC `q8_2f`) | 倍实时 (RNN-T `q8_2f`) | 倍实时 (CTC `q8_2f`) | 进程内存峰值，GB |
|---:|---:|---:|---:|---:|---:|
| 10 秒 | 0.051 | 0.053 | 196 | 189 | 1.2 |
| 30 秒 | 0.154 | 0.149 | 195 | 201 | 1.2 |
| 1 分钟 | 0.298 | 0.297 | 201 | 202 | 1.3 |
| 5 分钟 | 1.64 | 1.67 | 182 | 179 | 1.4 |
| 30 分钟 | 10.6 | 10.0 | 170 | 179 | 1.8 |

**CPU · MacBook Air M4**

测试平台：MacBook Air M4（24 GB），无风扇，仅 CPU（`CMF_GPU=0`）：乘法由 Accelerate 的矩阵单元完成。

![CPU · MacBook Air M4](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_cpu_m4_zh.png)

| 录音 | 耗时，秒 (RNN-T `q8_2f`) | 耗时，秒 (CTC `q8_2f`) | 倍实时 (RNN-T `q8_2f`) | 倍实时 (CTC `q8_2f`) | 进程内存峰值，GB |
|---:|---:|---:|---:|---:|---:|
| 10 秒 | 0.122 | 0.117 | 82 | 86 | 0.3 |
| 30 秒 | 0.269 | 0.289 | 112 | 104 | 0.4 |
| 1 分钟 | 0.471 | 0.475 | 127 | 126 | 0.4 |
| 5 分钟 | 2.70 | 2.47 | 111 | 121 | 0.6 |
| 30 分钟 | 16.7 | 15.3 | 108 | 118 | 0.9 |

**CPU · Xeon，12 vCPU**

测试平台：L4 实例的主机，Intel Xeon 2.2 GHz，12 vCPU，不使用 GPU（`CMF_GPU=0`）。

![CPU · Xeon，12 vCPU](https://huggingface.co/infosave/GigaAM-v4-cmf/resolve/main/assets/speed_cpu_zh.png)

| 录音 | 耗时，秒 (RNN-T `q8_2f`) | 耗时，秒 (CTC `q8_2f`) | 倍实时 (RNN-T `q8_2f`) | 倍实时 (CTC `q8_2f`) | 进程内存峰值，GB |
|---:|---:|---:|---:|---:|---:|
| 10 秒 | 0.517 | 0.572 | 19 | 18 | 0.3 |
| 30 秒 | 1.58 | 1.50 | 19 | 20 | 0.3 |
| 1 分钟 | 2.68 | 2.64 | 22 | 23 | 0.4 |
| 5 分钟 | 13.0 | 13.1 | 23 | 23 | 0.5 |
| 30 分钟 | 78.6 | 78.3 | 23 | 23 | 0.6 |

### 使用

```bash
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf speech.wav                 # 规范文本
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf speech.wav --head raw      # 小写词，无标点
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf speech.wav --timestamps    # 每个词的起止时间
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf a.wav b.mp3 c.ogg          # 多个文件，只加载一次
cortiq transcribe GigaAM-v4-rnnt-q8_2f.cmf talk.wav --json --head both  # 文本、片段、词时间、各阶段耗时
```

不超过 30 秒的录音是一个片段。更长的录音在语音活动检测网络找到停顿的位置切开（片段不超过 30 秒，与作者软件包的
`transcribe_longform` 的切法相同），各片段的转写再拼接起来；`--vad` 让短片段也走切分。`--stats` 打印各阶段的耗时。

OpenAI 兼容服务：

```bash
cortiq serve GigaAM-v4-rnnt-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/audio/transcriptions -F file=@speech.wav
curl http://localhost:8080/v1/audio/transcriptions -F file=@talk.mp3 \
  -F response_format=verbose_json -F "timestamp_granularities[]=word"
```

`POST /v1/audio/transcriptions` 接受 OpenAI 客户端发送的 multipart 表单（`file`；`response_format`：`json`、`text`、
`verbose_json`、`srt`、`vtt`；`timestamp_granularities[]`：`segment`、`word`），也接受 JSON，其中 `file` 为 `data:` /
`http(s)` URL 或 `{"data": <base64>}`。`head=raw` 请求纯词转写。

设置：`CMF_GPU=cuda` / `1` / `0` —— CUDA 后端 / 默认 GPU 后端（Vulkan、Metal）/ CPU；`CMF_CUDA_PREC=exact`（Vulkan 上为
`CMF_GIGAAM_GEMM=f32`）—— 用普通 float32 乘法代替 TF32：得到与 CPU 逐字符相同的转写，编码器速度约减半；
`CMF_GIGAAM_DENSE=1` —— 在 CPU 上保留浮点权重副本（约 0.95 GB），短片段不再为解包付出时间；`CMF_THREADS=N` —— CPU 线程数。

### 准确率

各文件与参考实现在相同片段上、使用相同文本规范化（小写、ё → е、去掉标点；模型写成数字的数，相对于用文字标注的
参考文本计为错误，对每一行都一样）得到的词错误率（%）。最后一列是规范化后的转写与参考实现完全一致的片段数。
已发布的文件在 CUDA 上以默认精度测得。

规范文本（`e2e` 头）：

**RNN-T**

| 文件 | Golos Farfield, 1 916 | Golos Crowd, 9 994 | Russian LibriSpeech, 1 352 | 与参考实现完全一致的转写 |
|---|---:|---:|---:|---:|
| 参考：作者的 PyTorch 代码，float32 | 4.96 | 14.19 | 4.27 | — |
| CMF，原始半精度权重，float32 乘法（未上传） | 4.96 | 14.19 | 4.27 | 13 261 / 13 262 |
| `GigaAM-v4-rnnt-q8_2f.cmf` | 4.93 | 14.15 | 4.29 | 13 138 / 13 262 |

**CTC**

| 文件 | Golos Farfield, 1 916 | Golos Crowd, 9 994 | Russian LibriSpeech, 1 352 | 与参考实现完全一致的转写 |
|---|---:|---:|---:|---:|
| 参考：作者的 PyTorch 代码，float32 | 5.27 | 14.81 | 4.72 | — |
| CMF，原始半精度权重，float32 乘法（未上传） | 5.29 | 14.81 | 4.72 | 13 260 / 13 262 |
| `GigaAM-v4-ctc-q8_2f.cmf` | 5.29 | 14.80 | 4.65 | 13 011 / 13 262 |

纯词（`raw` 头）：

**RNN-T**

| 文件 | Golos Farfield, 1 916 | Golos Crowd, 9 994 | Russian LibriSpeech, 1 352 | 与参考实现完全一致的转写 |
|---|---:|---:|---:|---:|
| 参考：作者的 PyTorch 代码，float32 | 3.73 | 4.69 | 4.17 | — |
| CMF，原始半精度权重，float32 乘法（未上传） | 3.72 | 4.69 | 4.17 | 13 261 / 13 262 |
| `GigaAM-v4-rnnt-q8_2f.cmf` | 3.70 | 4.71 | 4.21 | 13 164 / 13 262 |

**CTC**

| 文件 | Golos Farfield, 1 916 | Golos Crowd, 9 994 | Russian LibriSpeech, 1 352 | 与参考实现完全一致的转写 |
|---|---:|---:|---:|---:|
| 参考：作者的 PyTorch 代码，float32 | 4.29 | 5.40 | 4.60 | — |
| CMF，原始半精度权重，float32 乘法（未上传） | 4.29 | 5.40 | 4.60 | 13 261 / 13 262 |
| `GigaAM-v4-ctc-q8_2f.cmf` | 4.32 | 5.39 | 4.54 | 13 026 / 13 262 |

### 校验下载

`cortiq verify GigaAM-v4-rnnt-q8_2f.cmf` 校验文件各部分的哈希；各文件的 sha256 见
[Verify the download](#verify-the-download) 表。权重：[MIT](https://huggingface.co/ai-sage/GigaAM-v4/blob/main/LICENSE)，
来自 [ai-sage/GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4)；模型卡片中关于用途和局限的说明同样适用于这些文件。
