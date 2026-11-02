---
license: apache-2.0
library_name: cortiq
base_model: ostris/primal_vae
base_model_relation: quantized
tags:
  - cmf
  - cortiq
  - vae
  - autoencoder
  - latent-diffusion
  - vulkan
  - metal
language:
  - en
  - ru
  - zh
---

# Primal VAE — CMF

[Ostris Primal VAE](https://huggingface.co/ostris/primal_vae) — a deterministic
convolutional autoencoder that compresses an image 2× spatially into 1 or 2
channels whose values are already unit normal (no scaling or shift factor;
patchify by 4 for the 16- or 32-channel 8× layout of the Flux.1 / Flux.2 VAEs) —
packaged as single [CMF](https://github.com/infosave2007/cmf) files. They run
with `cortiq`, a Rust inference engine with no Python and no ML framework: the
whole encode or decode runs on the GPU (Vulkan/DX12 on NVIDIA, AMD and Intel,
Metal on Apple silicon) in one submission, or on the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/primal_vae-cmf primal_vae_2x2.cmf --local-dir .
cortiq vae --model primal_vae_2x2.cmf --mode roundtrip --input photo.png --output recon.png
```

Use cortiq 0.8.51 or later; prebuilt binaries for Linux, macOS and Windows
are on the [releases page](https://github.com/infosave2007/cmf/releases). Image
sides must be multiples of 16.

## Files

| file | latent | patch 4 = | size | PSNR on the card's three images |
|---|---|---|---:|---|
| `primal_vae_2x2.cmf` | 2 channels at 2× | 32 ch at 8× (Flux.2 layout) | 160,948,224 B (161 MB) | 37.5 / 27.1 / 27.4 dB |
| `primal_vae_2x1.cmf` | 1 channel at 2× | 16 ch at 8× (Flux.1 layout) | 160,944,128 B (161 MB) | 32.1 / 22.1 / 21.3 dB |

Both are the same 80 M-parameter network with 16-bit kernels (biases and norm
weights exact). Against the original float32 model in PyTorch on the same
images, the reconstructions have the same PSNR to 0.01 dB and the latents differ
by at most 0.14 % of their range.

## Usage

```bash
cortiq vae --model primal_vae_2x2.cmf --mode encode --input photo.png --output latent.npy   # [1, 2, H/2, W/2]
cortiq vae --model primal_vae_2x2.cmf --mode decode --input latent.npy --output image.png
cortiq vae --model primal_vae_2x2.cmf --mode roundtrip --input photo.png --output recon.png  # prints the PSNR
```

The latent is a float32 `.npy` of shape `[1, C, H/2, W/2]`, ~N(0, 1) per image
and channel — feed it to a diffusion model as it is, or patchify by 4
(`pixel_unshuffle`) for the 8× token grid. `decode` accepts `[1, C, h, w]` or
`[C, h, w]`; the image size is the latent's ×2. `CMF_GPU=0` forces the CPU.

## Performance

cortiq 0.8.51, `cortiq vae` (weights loaded once, the GPU run is one submission):

| device | 512×512 encode / decode | 1024×1024 encode / decode |
|---|---:|---:|
| NVIDIA A40 (Vulkan) | 0.20 s / 0.08 s | 0.40 s / 0.29 s |
| Apple M4 (Metal) | 0.34 s / 0.32 s | 1.27 s / 1.23 s |
| CPU (A40 host) | 3.9 s / 3.6 s | 13.3 s / 12.6 s |

## Verify

```bash
sha256sum -c primal_vae_2x2.cmf.sha256
```

```
2e81d119d8985bab6ed03f707992d44c0ce23e5e12d096766160adc6d5b4dc8d  primal_vae_2x2.cmf
979db68b6902aba6906d442f2034b0b8846e9e08f848f85c1d5cd065b5157eb9  primal_vae_2x1.cmf
```

## Reproduce

```bash
hf download ostris/primal_vae --local-dir primal_vae
cortiq convert --model primal_vae/primal_vae_2x2.safetensors --output primal_vae_2x2.cmf
cortiq convert --model primal_vae/primal_vae_2x1.safetensors --output primal_vae_2x1.cmf
```

## License

Apache-2.0, as the original release; the weights were converted to CMF by cortiq.

---

## Документация на русском

[Primal VAE от Ostris](https://huggingface.co/ostris/primal_vae) — детерминированный
свёрточный автоэнкодер: сжимает изображение в 2 раза по сторонам до 1 или 2
каналов, распределённых как N(0, 1) (без scaling/shift; patchify 4 даёт
16- или 32-канальную раскладку 8× VAE Flux.1 / Flux.2). Здесь он упакован в
одиночные файлы [CMF](https://github.com/infosave2007/cmf) для `cortiq` —
движка на Rust без Python: кодирование и декодирование целиком идут на GPU
(Vulkan/DX12, Metal) одной отправкой, либо на CPU.

```bash
cargo install cortiq-cli
hf download infosave/primal_vae-cmf primal_vae_2x2.cmf --local-dir .
cortiq vae --model primal_vae_2x2.cmf --mode roundtrip --input photo.png --output recon.png
```

Нужен cortiq 0.8.51 или новее. Стороны картинки кратны 16.

| файл | латент | размер | PSNR на трёх картинках карточки |
|---|---|---:|---|
| `primal_vae_2x2.cmf` | 2 канала, 2× | 160,948,224 B (161 MB) | 37,5 / 27,1 / 27,4 дБ |
| `primal_vae_2x1.cmf` | 1 канал, 2× | 160,944,128 B (161 MB) | 32,1 / 22,1 / 21,3 дБ |

Ядра свёрток 16-битные, смещения и веса норм точные; против исходной модели
float32 в PyTorch PSNR совпадает до 0,01 дБ, латенты отличаются не более чем
на 0,14 % диапазона. `--mode encode` пишет латент `[1, C, H/2, W/2]` в `.npy`,
`--mode decode` собирает картинку обратно, `roundtrip` печатает PSNR.

Скорость (cortiq 0.8.51): кодирование / декодирование 1024×1024 — 0.40 / 0.29 с
на NVIDIA A40 (Vulkan), 1.27 / 1.23 с на Apple M4 (Metal), около 13 с на CPU.

---

## 中文说明

[Ostris Primal VAE](https://huggingface.co/ostris/primal_vae) 是一个确定性卷积自编码器：
把图像在空间上压缩 2 倍，得到 1 或 2 个通道、本身即服从 N(0, 1) 的潜变量（无需
scaling/shift；按 4 做 patchify 即得到 Flux.1 / Flux.2 VAE 的 16 / 32 通道 8× 布局）。
这里打包为单个 [CMF](https://github.com/infosave2007/cmf) 文件，由 Rust 推理引擎
`cortiq` 运行（无 Python）：整个编码或解码在 GPU 上一次提交完成（Vulkan/DX12、Metal），
也可在 CPU 上运行。

```bash
cargo install cortiq-cli
hf download infosave/primal_vae-cmf primal_vae_2x2.cmf --local-dir .
cortiq vae --model primal_vae_2x2.cmf --mode roundtrip --input photo.png --output recon.png
```

需要 cortiq 0.8.51 或更高版本，图像边长须为 16 的倍数。

| 文件 | 潜变量 | 大小 | 卡片三张图的 PSNR |
|---|---|---:|---|
| `primal_vae_2x2.cmf` | 2 通道，2× | 160,948,224 B (161 MB) | 37.5 / 27.1 / 27.4 dB |
| `primal_vae_2x1.cmf` | 1 通道，2× | 160,944,128 B (161 MB) | 32.1 / 22.1 / 21.3 dB |

卷积核为 16 位，偏置与归一化权重保持精确；与 PyTorch 中的原始 float32 模型相比，
PSNR 相差不超过 0.01 dB，潜变量差异不超过其范围的 0.14%。

速度（cortiq 0.8.51）：1024×1024 编码 / 解码在 NVIDIA A40（Vulkan）上为 0.40 / 0.29 s，
在 Apple M4（Metal）上为 1.27 / 1.23 s，CPU 约 13 s。
