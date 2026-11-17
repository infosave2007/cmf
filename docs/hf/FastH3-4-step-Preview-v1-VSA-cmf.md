---
library_name: cortiq
license: other
license_name: minimax-h3-community
license_link: LICENSE
base_model:
- FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree
base_model_relation: quantized
pipeline_tag: text-to-video
tags:
- cmf
- cortiq
- video
- audio
- text-to-audio-video
- few-step
- sparse-attention
- 4-bit
- cuda
- vulkan
- metal
---

# FastH3 4-step (VSA) — one file, no Python

[FastVideo's FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
is MiniMax-H3 distilled to **four transformer passes** and trained with **video sparse attention**: each
4×4×4 block of video tokens attends only the 10 % of video blocks it scores highest, plus all text and
audio. This repository packages it as one [CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a
Rust engine with no Python and no ML framework: the 4-bit DiT with its trained sparse-attention gates, the
Qwen3-VL prompt encoder and both decoders (video and stereo audio).

- **One RTX A5000 (24 GB), CUDA:** a 5-second 1344×768 clip with sound in **194 s**; a 512×288 preview of
  39 frames in **22 s**.
- The sparse attention runs on the card the way the student was trained — on CUDA, Vulkan and Metal.
- The upstream release targets four B200 GPUs; here it is one file and one card.

![frames](https://huggingface.co/infosave/FastH3-4-step-Preview-v1-VSA-cmf/resolve/main/samples/beach_full.gif)

*"A corgi runs along a sunny beach, waves rolling in behind it, the camera tracking alongside; the dog barks
twice."* — 1344×768, 124 frames, seed 7, four steps. With sound:
[mp4](https://huggingface.co/infosave/FastH3-4-step-Preview-v1-VSA-cmf/resolve/main/samples/beach_full.mp4).

## Quick start

```bash
cargo install cortiq-cli --features cuda   # NVIDIA: the CUDA backend (driver + CUDA 12.x libraries)
hf download infosave/FastH3-4-step-Preview-v1-VSA-cmf fasth3-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq animate fasth3-q4tp.cmf \
  --prompt "A corgi runs along a sunny beach, waves rolling in behind it; the dog barks twice." \
  --width 1344 --height 768 --frames 124 --out clip.avi
```

The output is an MJPEG AVI with the stereo soundtrack inside (and `clip.wav`). Four steps on the student's
own schedule are the default; `--seed` picks the noise. Width and height are multiples of 32, frames play
at 24 fps. Without `--features cuda` the same command runs over Vulkan (`CMF_GPU=1`); on a Mac it runs on
Metal.

## Which file

| file | size | prompt encoder | peak VRAM 512×288 / 1344×768 (CUDA) |
|---|---:|---|---|
| `fasth3-q4tp.cmf` | 26.05 GB | Qwen3-VL-32B, 4-bit | 13.7 / 17.3 GB |
| `fasth3-clipproj4b-q4tp.cmf` | 14.98 GB | Qwen3-VL-4B + ClipProj, 4-bit | 13.7 / 17.3 GB |

Both carry the same DiT and decoders, so they need the same card memory: the stages run one after another.
The small file is 11 GB less to download and to keep in RAM, and it renders the same scenes
([same prompt, both files](https://huggingface.co/infosave/FastH3-4-step-Preview-v1-VSA-cmf/resolve/main/samples/ab_full.mp4) ·
[small file](https://huggingface.co/infosave/FastH3-4-step-Preview-v1-VSA-cmf/resolve/main/samples/ab_clip.mp4)).

## Speed

**RTX A5000 (24 GB), CUDA**, one clip, four steps, `fasth3-q4tp.cmf`:

| clip | prompt encoder | denoising | video decoder | audio decoder | total |
|---|---:|---:|---:|---:|---:|
| 512×288, 39 frames | 2.7 s | 10.1 s | 3.7 s | 4.2 s | **22 s** |
| 1344×768, 124 frames (5 s) | 2.7 s | 115.6 s | 52.6 s | 11.8 s | **194 s** |

Full attention instead of the trained sparse one (`CMF_VSA=0`) takes 204 s of denoising for the 5-second
clip, and the picture degrades toward its end
([side by side](https://huggingface.co/infosave/FastH3-4-step-Preview-v1-VSA-cmf/resolve/main/samples/vsa_vs_full_attention.png)):
the student was trained with sparse attention.

**RTX A5000, Vulkan**, `fasth3-q4tp.cmf`:

| clip | prompt encoder | denoising | video decoder | audio decoder | total | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| 512×288, 39 frames | 3.5 s | 10.4 s | 5.6 s | 4.3 s | **26 s** | 12.6 GB |
| 1344×768, 124 frames | 5.8 s | 182.8 s | 78.7 s | 14.1 s | **294 s** | 20.6 GB |

**Apple M4 (24 GB), Metal**, `fasth3-clipproj4b-q4tp.cmf`, 512×288, 39 frames: prompt 0.6 s, denoising
128.5 s, video decoder 47.2 s, audio decoder 2.2 s — **179 s**.

## Settings

| variable | effect |
|---|---|
| `CMF_VSA=0` | full attention instead of the trained sparse attention (slower at large sizes, not what the student learned) |
| `CMF_VSA_SPARSITY=0.8` | keep 20 % of the video blocks instead of 10 % |

## License and attribution

The weights derive from [FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
(FastVideo) and [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and are distributed under
the **MiniMax H3 Community License** (`LICENSE`, `NOTICE` in this repository), including its territory
restrictions and acceptable-use policy. The files here are modified: the DiT is re-quantized to 4 bits and its
timestep modulation stored as a rank-9 curve; the prompt encoders are 4-bit Qwen3-VL (Apache-2.0). This
repository is not made or endorsed by FastVideo or MiniMax.

---

## На русском

[FastH3 от FastVideo](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) —
MiniMax-H3, дистиллированная до **четырёх проходов** трансформера и обученная с **разреженным вниманием по
видео**: каждый блок 4×4×4 видеотокенов смотрит только на 10 % видеоблоков с наибольшей оценкой, а также на
весь текст и звук. Здесь она упакована в один файл [CMF](https://github.com/infosave2007/cmf) для `cortiq` —
движка на Rust без Python: DiT в 4 битах с обученными гейтами, энкодер промпта Qwen3-VL и оба декодера
(видео и стереозвук).

- **Одна RTX A5000 (24 ГБ), CUDA:** 5-секундный клип 1344×768 со звуком за **194 с**, превью 512×288 на 39
  кадров — за **22 с**. Vulkan: 26 и 294 с. Apple M4 (Metal, малый файл): 512×288 за 179 с.
- Разреженное внимание считается на видеокарте так, как студента обучали, — на CUDA, Vulkan и Metal.

```bash
cargo install cortiq-cli --features cuda
hf download infosave/FastH3-4-step-Preview-v1-VSA-cmf fasth3-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq animate fasth3-q4tp.cmf --prompt "..." --width 1344 --height 768 --frames 124 --out clip.avi
```

Файлы: `fasth3-q4tp.cmf` (26,05 ГБ, энкодер Qwen3-VL-32B) и `fasth3-clipproj4b-q4tp.cmf` (14,98 ГБ, Qwen3-VL-4B +
ClipProj); видеопамяти обоим нужно одинаково — 13,7 ГБ на 512×288 и 17,3 ГБ на 1344×768 (CUDA). `CMF_VSA=0`
включает полное внимание (медленнее на больших размерах и не то, чему учился студент). Лицензия — MiniMax H3
Community License (`LICENSE`, `NOTICE`) с её территориальными ограничениями; файлы изменены (DiT переквантован
в 4 бита). Репозиторий не сделан и не одобрен FastVideo или MiniMax.

## 中文说明

[FastVideo 的 FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) 是把
MiniMax-H3 蒸馏到 **四次 Transformer 前向** 并以 **视频稀疏注意力** 训练的模型：每个 4×4×4 视频 token 块只关注
得分最高的 10% 视频块以及全部文本和音频。本仓库把它打包为一个 [CMF](https://github.com/infosave2007/cmf)
文件，由 `cortiq`（不依赖 Python 的 Rust 引擎）运行：4 位 DiT（含训练好的门控）、Qwen3-VL 提示编码器以及视频与
立体声音频解码器。

- **单张 RTX A5000（24 GB），CUDA：** 5 秒 1344×768 带声音的视频 **194 秒**，512×288、39 帧预览 **22 秒**。
  Vulkan：26 秒与 294 秒。Apple M4（Metal，小文件）：512×288 用 179 秒。
- 稀疏注意力按训练方式在显卡上运行——CUDA、Vulkan 与 Metal 均支持。

```bash
cargo install cortiq-cli --features cuda
hf download infosave/FastH3-4-step-Preview-v1-VSA-cmf fasth3-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq animate fasth3-q4tp.cmf --prompt "..." --width 1344 --height 768 --frames 124 --out clip.avi
```

文件：`fasth3-q4tp.cmf`（26.05 GB，Qwen3-VL-32B 编码器）与 `fasth3-clipproj4b-q4tp.cmf`（14.98 GB，Qwen3-VL-4B +
ClipProj）；两者显存需求相同——512×288 为 13.7 GB，1344×768 为 17.3 GB（CUDA）。`CMF_VSA=0` 改用全注意力（大尺寸下
更慢，也不是学生模型所学）。许可证为 MiniMax H3 Community License（`LICENSE`、`NOTICE`），含地域限制；文件已修改
（DiT 重新量化为 4 位）。本仓库并非由 FastVideo 或 MiniMax 制作或认可。
