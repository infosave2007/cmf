---
library_name: cortiq
license: other
license_name: minimax-h3-community
license_link: LICENSE
base_model:
- H3-Tiny/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1
base_model_relation: quantized
pipeline_tag: text-to-video
tags:
- cmf
- cortiq
- video
- audio
- text-to-audio-video
- reference-to-video
- few-step
- h3-tiny
- 4-bit
- cuda
- vulkan
- metal
---

# MiniMax-H3-Tiny FewStep — one file, a 12 GB card

[H3-Tiny's FewStep student](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1) is MiniMax-H3
pruned to 17 B (42 blocks, a narrower FFN) and distilled to **six steps** (four or more work) with MeanFlow: the model
predicts the average velocity over each step and reads the step's endpoint as well as its start. This repository
packages it as one [CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust engine with no Python and no ML
framework: the 4-bit DiT, a 4-bit Qwen3-VL-4B prompt encoder with its vision tower and ClipProj, the video VAE and the
stereo audio decoder — **13.0 GB in all**.

- **The whole DiT fits a 12 GB card.** RTX 3060, CUDA: a 512×288 clip of 39 frames with sound in **24 s** (4 steps)
  or 30 s (6 steps); 768×448, 73 frames in 145 s. Our MiniMax-H3 Turbo file on the same card takes 408 s and 2 201 s:
  its DiT does not fit and streams.
- **Text to video, or reference pictures to video (Ref2VA).** Name the pictures in the prompt as `<Picture 1>`,
  `<Picture 2>`; without `--ref` it is text-to-video.
- CUDA, Vulkan and Metal from the same file.

![frames](https://huggingface.co/infosave/MiniMax-H3-Tiny-FewStep-cmf/resolve/main/samples/frog_ref2va_768x448.gif)

![the two reference pictures](https://huggingface.co/infosave/MiniMax-H3-Tiny-FewStep-cmf/resolve/main/samples/refs.jpg)

*"`<Picture 1>` walks into the rainy backyard garden of `<Picture 2>`, splashing through a puddle in her red rain
boots. She spots the little green frog sitting on the stone, crouches down, her eyes go wide, and she laughs with
delight: "Hello, little frog!" …"* — the two pictures above, 768×448, 73 frames, seed 42, six steps, RTX 3060.
With sound: [mp4](https://huggingface.co/infosave/MiniMax-H3-Tiny-FewStep-cmf/resolve/main/samples/frog_ref2va_768x448.mp4).
Text only, four steps, 512×288 in 24 s:
[mp4](https://huggingface.co/infosave/MiniMax-H3-Tiny-FewStep-cmf/resolve/main/samples/frog_t2va_4step_512x288.mp4).

## Quick start

H3-Tiny support (the MeanFlow time embedding, `--ref`) is in cortiq's `h3tiny-fewstep` branch until the next release:

```bash
cargo install --git https://github.com/infosave2007/cmf --branch h3tiny-fewstep cortiq-cli --features cuda
hf download infosave/MiniMax-H3-Tiny-FewStep-cmf h3tiny-17b-fewstep-clipproj4b-q4tp.cmf --local-dir .

# text to video with sound
CMF_GPU=cuda cortiq animate h3tiny-17b-fewstep-clipproj4b-q4tp.cmf \
  --prompt "A little girl in a yellow raincoat finds a small green frog in a garden after the rain and laughs." \
  --width 512 --height 288 --frames 39 --out clip.avi

# reference pictures (binary PPM: ffmpeg -i girl.jpg girl.ppm)
CMF_GPU=cuda cortiq animate h3tiny-17b-fewstep-clipproj4b-q4tp.cmf \
  --ref girl.ppm --ref garden.ppm \
  --prompt "<Picture 1> walks into the garden of <Picture 2> and finds the frog on the stone." \
  --width 768 --height 448 --frames 73 --out clip.avi
```

The output is an MJPEG AVI with the stereo soundtrack inside (and `clip.wav`). Six steps are the default (`--steps 4`
is faster, `--steps 8` adds detail); width and height are multiples of 32, frames play at 24 fps. Without
`--features cuda` the same command runs over Vulkan (`CMF_GPU=1`); on a Mac it runs on Metal. A reference keeps its own
aspect ratio at the clip's area.

## Speed

**RTX 3060 (12 GB, limited to 120 W), CUDA**, Core i9-11900K, `h3tiny-17b-fewstep-clipproj4b-q4tp.cmf`:

| clip | steps | references | prompt + pictures | denoising | video decoder | audio decoder | total | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512×288, 39 frames | 4 | — | 1.0 s | 13.6 s | 7.1 s | 1.7 s | **24.1 s** | 10.5 GiB |
| 512×288, 39 frames | 6 | — | 1.0 s | 20.1 s | 7.1 s | 1.7 s | **30.4 s** | 10.5 GiB |
| 512×288, 39 frames | 4 | 2 | 5.2 s | 17.9 s | 7.0 s | 1.7 s | **32.5 s** | 10.7 GiB |
| 512×288, 39 frames | 6 | 2 | 5.1 s | 26.3 s | 7.0 s | 1.7 s | **40.9 s** | 10.7 GiB |
| 768×448, 73 frames (3 s) | 6 | 2 | 12.7 s | 109.0 s | 18.8 s | 3.0 s | **144.7 s** | 11.4 GiB |

The same card with our [MiniMax-H3 Turbo](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf) small file
(`mmh3-turbo-clipproj4b-fl2va-v2-q4tp.cmf`, 14.5 GB, four steps): 512×288 in 408 s, 768×448 in 2 201 s — its
20 B DiT does not fit 12 GB.

**RTX 3060, Vulkan**: 512×288, 39 frames — 4 steps, text only: **34.7 s**; 6 steps with two references: **63.2 s**;
peak 9.4 GiB.

**Apple silicon, Metal:** not measured with this file yet.

## How it was made

```bash
cortiq animate-pack --in mmh3-turbo-clipproj4b-fl2va-v2-q4tp.cmf \
  --dit minimax_h3_tiny_17b_fewstep_ref2va_bf16.safetensors --quant q4tp \
  --out h3tiny-17b-fewstep-clipproj4b-q4tp.cmf
```

The DiT comes from [H3-Tiny/MiniMax-H3-Tiny-ComfyUI](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-ComfyUI)
(`minimax_h3_tiny_17b_fewstep_ref2va_bf16.safetensors`); its attention and FFN projections go to 4 bits (q4tp), its
rank-8 adaLN and both time embedders stay as released, and the runtime evaluates `silu(emb_t(t) + emb_r(r − t))` for
every row: the next point of `linspace(1, 0, steps + 1)` on the video shift (12) for the target, `r = t` for the
reference rows. The prompt encoder, the vision tower and the decoders are carried over from our Turbo small file.

## License and attribution

The weights derive from [H3-Tiny/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1)
(the H3-Tiny project) and [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and are distributed under
the **MiniMax H3 Community License** (`LICENSE`, `NOTICE` in this repository), including its territory restrictions
and acceptable-use policy. The files here are modified: the DiT is re-quantized to 4 bits; the prompt encoder is a 4-bit
Qwen3-VL-4B (Apache-2.0) with a ClipProj adapter. The sample reference pictures are from the
[H3-Tiny project page](https://github.com/fuyi25/MiniMax-H3-Tiny). This repository is not made or endorsed by the
H3-Tiny project, MiniMax or Comfy Org.

---

## На русском

[FewStep-ученик H3-Tiny](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1) — MiniMax-H3,
обрезанная до 17 млрд параметров (42 блока, более узкий FFN) и дистиллированная до **шести шагов** (работает от
четырёх) методом MeanFlow. Здесь она упакована в один файл [CMF](https://github.com/infosave2007/cmf) для `cortiq`:
DiT в 4 битах, кодировщик Qwen3-VL-4B с vision-башней и ClipProj, видео-VAE и стереозвук — **13,0 ГБ**.

- **DiT целиком помещается в карту на 12 ГБ.** RTX 3060, CUDA: 512×288 на 39 кадров со звуком за **24 с** (4 шага)
  или 30 с (6 шагов), 768×448 на 73 кадра — за 145 с. Наш файл MiniMax-H3 Turbo на той же карте — 408 и 2 201 с.
  Vulkan на той же карте: 34,7 с (4 шага) и 63,2 с (6 шагов с двумя опорными картинками).
- **Текст → видео или опорные картинки → видео (Ref2VA):** картинки передаются через `--ref` и упоминаются в промпте
  как `<Picture 1>`, `<Picture 2>`.

```bash
cargo install --git https://github.com/infosave2007/cmf --branch h3tiny-fewstep cortiq-cli --features cuda
hf download infosave/MiniMax-H3-Tiny-FewStep-cmf h3tiny-17b-fewstep-clipproj4b-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq animate h3tiny-17b-fewstep-clipproj4b-q4tp.cmf --prompt "..." --width 512 --height 288 --frames 39 --out clip.avi
```

По умолчанию 6 шагов (`--steps 4` быстрее, `--steps 8` детальнее). Лицензия — MiniMax H3 Community License (`LICENSE`,
`NOTICE`) с её территориальными ограничениями; файлы изменены (DiT переквантован в 4 бита). Репозиторий не сделан и
не одобрен проектом H3-Tiny, MiniMax или Comfy Org.

## 中文说明

[H3-Tiny 的 FewStep 学生模型](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1)是将 MiniMax-H3
剪枝到 17B（42 个块、更窄的 FFN）并用 MeanFlow 蒸馏到 **6 步**（4 步及以上可用）的模型。本仓库把它打包成一个
[CMF](https://github.com/infosave2007/cmf) 文件供 `cortiq` 使用：4 位 DiT、带视觉塔和 ClipProj 的 4 位 Qwen3-VL-4B
提示词编码器、视频 VAE 和立体声音频解码器，共 **13.0 GB**。RTX 3060（12 GB，CUDA）上 512×288、39 帧带声音的视频
**24 秒**（4 步）生成；用 `--ref` 传入参考图，并在提示词中写 `<Picture 1>`、`<Picture 2>` 即为 Ref2VA。许可证为
MiniMax H3 Community License（`LICENSE`、`NOTICE`），含地域限制；文件已修改（DiT 重新量化为 4 位）。本仓库并非由
H3-Tiny 项目、MiniMax 或 Comfy Org 制作或认可。
