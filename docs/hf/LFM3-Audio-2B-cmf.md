---
license: other
license_name: lfm1.0
license_link: LICENSE
library_name: cortiq
base_model: LiquidAI/LFM3-Audio-2B
base_model_relation: quantized
pipeline_tag: audio-text-to-text
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - liquid
  - lfm3
  - speech
  - asr
  - tts
  - vulkan
  - metal
language:
  - en
  - ar
  - da
  - de
  - el
  - fi
  - fr
  - he
  - it
  - ja
  - ko
  - nl
  - pl
  - pt
  - es
  - sv
  - tr
  - zh
---

# LFM3-Audio-2B — CMF

[Liquid AI LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) — a 2B
speech model on LFM3: a causal audio encoder (16 kHz) feeding the LFM3-1.7B
text model, which answers in text or in 24 kHz speech through a depth decoder
and the LiquidCodec tokenizer (12 codes per 80 ms) — speech recognition,
speech synthesis in 19 languages, voice chat and audio understanding,
packaged as single [CMF](https://github.com/infosave2007/cmf) files with the
codec inside. They run with `cortiq`, a Rust inference engine with no Python
and no ML framework: NVIDIA, AMD and Intel GPUs through Vulkan/DX12, Apple
silicon through Metal, and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/LFM3-Audio-2B-cmf lfm3-audio-2b-q8_2f.cmf --local-dir .

# speech recognition
cortiq run lfm3-audio-2b-q8_2f.cmf --system "Perform ASR in English." --audio clip.wav --prompt ""

# speech synthesis (24 kHz WAV)
cortiq run lfm3-audio-2b-q8_2f.cmf --system "Perform TTS in American English using a female voice." \
  --prompt "The quick brown fox jumps over the lazy dog." --speech-out tts.wav --max-tokens 512
```

Use cortiq 0.8.50 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). `--audio`
takes a WAV file (other formats through `ffmpeg`), a data URL or an HTTP(S) URL
and may be repeated; clips are resampled to 16 kHz mono. `--system` sets the
task, as in Liquid's examples. `--speech-out` writes the spoken answer;
`--max-tokens` counts its frames (12.5 a second).

## Files

| file | language model | audio encoder, depth decoder, codec | size | recommended for |
|---|---|---|---:|---|
| `lfm3-audio-2b-q8_2f.cmf` | 8-bit | 16-bit | 2,451,370,906 B (2.45 GB) | **default** |
| `lfm3-audio-2b-q4tp.cmf` | 4-bit | 16-bit | 1,621,656,474 B (1.62 GB) | the smallest memory footprint |

Against the bf16 checkpoint run in float32 by Transformers:

- **Speech input.** The log-mel features, the encoder rows (cosine 1.000000
  for every row) and the rows handed to the language model (largest
  difference 4e-6) match; both files transcribe the test clips word for word
  as Transformers does.
- **Speech output.** With greedy codes in a fixed codebook order, an f16
  conversion reproduces Transformers' speech frames exactly (42 of 42 frames,
  every code). The quantized files take other greedy codes after the first
  frames — a near tie over 2,017 codes — and their speech transcribes back to
  the input sentence word for word.

## Performance

`cortiq run`, cortiq 0.8.50, `q4tp`:

| task | NVIDIA A40 (Vulkan) | Apple M4 (Metal) |
|---|---:|---:|
| speech synthesis, 7.0 s of speech | 5.6 s | 3.2 s |
| text decode (`cortiq bench --core`) | 204 tok/s (`q8_2f`: 172) | 79 tok/s |
| transcription of a 4.4 s clip, loading included | — | 1.5 s |

A speech frame is one step of the language model (on the GPU) plus three passes
of the depth decoder (on the CPU, 16–23 ms a frame); the codec decodes the
frames at the end (0.1 s on the M4, 0.6 s on the A40 host's CPU for 7 s of
speech).

## Usage

```bash
# voice chat: interleaved text and speech
cortiq run lfm3-audio-2b-q8_2f.cmf --system "Respond with interleaved text and audio." \
  --audio question.wav --prompt "" --speech-out reply.wav --speech-mode mixed --max-tokens 1024

# questions about a recording
cortiq run lfm3-audio-2b-q8_2f.cmf --audio street.wav --prompt "What can be heard in this clip?" --no-think

# speech in another language and voice
cortiq run lfm3-audio-2b-q8_2f.cmf --system "Perform TTS in German using a male voice." \
  --prompt "Guten Morgen! Wie geht es dir?" --speech-out de.wav
```

TTS languages: Arabic, Danish, German, Greek, British English, American
English, Finnish, French, Hebrew, Italian, Japanese, Korean, Dutch, Polish,
Portuguese, Spanish, Swedish, Turkish, Chinese; `female` or `male` voice.
Speech is sampled as Liquid recommends (temperature 1.0, top-k 128, repetition
penalty 1.1 over 4096 frames); `--greedy` takes the likeliest codes. Text
answers never open a speech turn.

### OpenAI-compatible server

```bash
cortiq serve lfm3-audio-2b-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "lfm3-audio", "messages": [
         {"role": "system", "content": "Perform ASR in English."},
         {"role": "user", "content": [{"type": "input_audio",
           "input_audio": {"data": "<base64 WAV>", "format": "wav"}}]}]}'
```

Audio parts take `input_audio` (base64), an HTTP(S) or `data:` URL, or a local
path; the server answers in text. Speech output (`--speech-out`) is a `run`
feature for now, as are live (streaming) recognition and speech in the server.

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12).** Both files fit any card with 3 GB of
free memory; on headless Linux install `libvulkan1 libglvnd0 libegl1 libgl1
libglx0` and set `XDG_RUNTIME_DIR=/tmp`. **Apple silicon:** the Metal backend
is the default. **CPU only:** `CMF_GPU=0`.

## Verify the download

```bash
sha256sum -c lfm3-audio-2b-q8_2f.cmf.sha256
cortiq verify lfm3-audio-2b-q8_2f.cmf
```

| file | sha256 |
|---|---|
| `lfm3-audio-2b-q8_2f.cmf` | `2b9ac37d52ac256f64028af86b07583badb86a995a32d0eb9194a536f74b207e` |
| `lfm3-audio-2b-q4tp.cmf` | `23a8d85061adcc830c87c1cc7e8eb2b00a388a558c334087575e446a7219b37f` |

## Reproduce

```bash
cortiq convert --model LiquidAI/LFM3-Audio-2B --quant q8_2f \
  --output lfm3-audio-2b-q8_2f.cmf --also-output q4tp=lfm3-audio-2b-q4tp.cmf
```

The converter reads the codec from the checkpoint's `audio_tokenizer/` folder
into the same file. Scripts (Transformers reference, speech parity):
[scripts/lfm3](https://github.com/infosave2007/cmf/tree/master/scripts/lfm3).

## License

These files are a modified form of Liquid AI's LFM3-Audio-2B: the weights were
converted to CMF and the language model quantized (8-bit `q8_2f`, 4-bit `q4tp`)
by cortiq. They remain under the LFM Open License v1.0, a copy of which is in
this repository ([LICENSE](LICENSE)); among its terms, commercial use by an
entity with annual revenue of US$10 million or more is not licensed. See the
[upstream card](https://huggingface.co/LiquidAI/LFM3-Audio-2B).

---

## Документация на русском

[LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) от Liquid AI —
речевая модель на 2 млрд параметров поверх LFM3: причинный аудиоэнкодер (16 кГц)
и текстовая LFM3-1.7B, отвечающая текстом или речью 24 кГц через depth-декодер
и кодек LiquidCodec (12 кодов на 80 мс): распознавание речи, синтез на 19 языках,
голосовой чат и понимание аудио. Файлы [CMF](https://github.com/infosave2007/cmf)
включают кодек и запускаются движком `cortiq` на Rust без Python.

```bash
cargo install cortiq-cli
hf download infosave/LFM3-Audio-2B-cmf lfm3-audio-2b-q8_2f.cmf --local-dir .
cortiq run lfm3-audio-2b-q8_2f.cmf --system "Perform ASR in Russian." --audio clip.wav --prompt ""
cortiq run lfm3-audio-2b-q8_2f.cmf --system "Perform TTS in American English using a female voice." \
  --prompt "Hello! How are you?" --speech-out tts.wav
```

| файл | языковая модель | энкодер, depth-декодер, кодек | размер | назначение |
|---|---|---|---:|---|
| `lfm3-audio-2b-q8_2f.cmf` | 8 бит | 16 бит | 2.45 ГБ | **по умолчанию** |
| `lfm3-audio-2b-q4tp.cmf` | 4 бита | 16 бит | 1.62 ГБ | минимум памяти |

Нужен cortiq 0.8.50 или новее. Распознавание совпадает с Transformers дословно
(строки энкодера — косинус 1.000000); синтез в f16 воспроизводит кадры речи
Transformers точно, речь квантованных файлов распознаётся обратно в исходную
фразу. 7 с речи синтезируются за 5.6 с на NVIDIA A40 и за 3.2 с на Apple M4.
`--speech-mode mixed` — голосовой чат (текст и речь вперемешку); `cortiq serve`
принимает аудио в формате OpenAI (`input_audio`) и отвечает текстом. Файлы —
изменённая форма LFM3-Audio-2B от Liquid AI по LFM Open License v1.0 (копия —
[LICENSE](LICENSE)); коммерческое использование организацией с годовой выручкой
от 10 млн долларов США лицензией не покрывается.

## 中文说明

[LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) 是 Liquid AI 基于
LFM3 的 2B 语音模型：因果音频编码器（16 kHz）接 LFM3-1.7B 文本模型，可用文本或
通过 depth 解码器与 LiquidCodec 编解码器（每 80 ms 12 个码）输出 24 kHz 语音——
语音识别、19 种语言的语音合成、语音对话与音频理解。本仓库的 CMF 文件内含编解码器，
可由 Rust 推理引擎 `cortiq`（0.8.50 及以上）加载。

```bash
cargo install cortiq-cli
hf download infosave/LFM3-Audio-2B-cmf lfm3-audio-2b-q8_2f.cmf --local-dir .
cortiq run lfm3-audio-2b-q8_2f.cmf --system "Perform ASR in Chinese." --audio clip.wav --prompt ""
cortiq run lfm3-audio-2b-q8_2f.cmf --system "Perform TTS in Chinese using a female voice." \
  --prompt "你好，今天天气很好。" --speech-out tts.wav
```

| 文件 | 语言模型 | 编码器、depth 解码器、编解码器 | 大小 | 适用 |
|---|---|---|---:|---|
| `lfm3-audio-2b-q8_2f.cmf` | 8 位 | 16 位 | 2.45 GB | **默认** |
| `lfm3-audio-2b-q4tp.cmf` | 4 位 | 16 位 | 1.62 GB | 内存最小 |

识别结果与 Transformers 逐字一致；f16 转换的语音帧与 Transformers 完全相同，量化文件
合成的语音可被逐字识别回原句。7 秒语音在 NVIDIA A40 上 5.6 s、在 Apple M4 上 3.2 s
合成完毕。这些文件是 Liquid AI LFM3-Audio-2B 的修改形式，仍受 LFM Open License
v1.0 约束（副本见 [LICENSE](LICENSE)）；年收入 1000 万美元及以上的实体进行商业使用
不在许可范围内。
