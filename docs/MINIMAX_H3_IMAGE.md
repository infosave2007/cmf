# MiniMax-H3-Image

*Languages: **English** · [Русский](MINIMAX_H3_IMAGE.ru.md) · [中文](MINIMAX_H3_IMAGE.zh.md)*

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) generates and edits images of 4 to 16
megapixels in six sampling steps. cortiq runs it from one CMF file of 24.6 GB that holds the diffusion
transformer (20 B parameters in 4 bits), the Qwen3-VL prompt encoder with its vision tower (4 bits), and the
image VAE (8 bits). On an NVIDIA card the whole pipeline runs through the native CUDA backend; on any other GPU
it runs through Vulkan or Metal. A ceiling on the card's memory lets it run on a small card.

```bash
cargo install cortiq-cli --features cuda
# text to image
CMF_GPU=cuda cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
# image editing: <Picture 1> in the prompt is the file given with --image
CMF_GPU=cuda cortiq imagine minimax-h3-image-q4tp.cmf --image photo.png --prompt @edit.txt --out edit.png
# a card with 6 GB
CMF_GPU=cuda CMF_GPU_VRAM_MB=6000 cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
# Vulkan (any vendor) and Metal: the default backend
cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
```

The file is on Hugging Face: [infosave/MiniMax-H3-Image-cmf](https://huggingface.co/infosave/MiniMax-H3-Image-cmf). The model is under the MiniMax-H3-Image
Community License, which limits where it may be used; see the page.

## Text to image

| option | meaning |
|---|---|
| `--prompt TEXT` or `--prompt @FILE` | the description; a long one reads better from a file |
| `--width`, `--height` | multiples of 32, 2048 x 2048 by default |
| `--min-area PIXELS` | the smallest canvas the model is run at, 4194304 (2048 x 2048) by default; a smaller size is generated at the same aspect ratio on such a canvas and resized back, as the reference server does. `--min-area 1024` turns this off |
| `--seed N` | 42 by default; `--num-images K` renders seeds N .. N + K - 1 from one prompt encode |
| `--steps N` | fewer than the model's six: the same sigma grid in longer strides |
| `--out FILE` | PNG, JPEG or PPM by extension |

The model is distilled: six steps on its own sigma grid, no classifier-free guidance and no negative prompt.
It works from about 4 megapixels (2048 x 2048) to 20 and is at its best near 16 (4096 x 4096). It expects a
detailed English description of the whole image. Its authors produce one from a short request with a prompt
expander (the H3-Image-Context-IR API, or the local MiniMax-H3-Image-PE model); cortiq does not include one, so
write the description yourself or expand it with any language model.

## Image editing

`--image FILE` makes the run an edit; repeat it for several pictures. The prompt refers to them as
`<Picture 1>`, `<Picture 2>` and so on, in the order given, and `<Picture 1>` is the picture to edit. Without
`--width` and `--height` the result takes the size of the first picture, brought into a short edge of 1024 to
2048 pixels.

The prompt encoder sees each picture through the Qwen3-VL vision tower at a short edge of 1024 pixels, and the
transformer also receives each picture's VAE latent as rows of the sequence ahead of the target. An edit of one
1024 x 1024 picture at 2048 x 2048 therefore runs over 4096 target rows, 1024 reference rows and the prompt.

The model's own edit prompts are briefs that name every subject and say what stays (`The man on the left in
<Picture 1> is <Subject 1>. ... The target image shows ...`). A short instruction also works, with less of the
picture preserved: text that the prompt does not name tends to change.

## Small cards

`CMF_GPU_VRAM_MB` is a ceiling on what the process holds on the card. The blocks of the transformer that do
not fit under it are not kept there: their weights pass through the card every step. What grows with the image
is small: panels are produced a chunk of rows at a time and attention keeps no score matrix.

2048 x 2048, CUDA, NVIDIA RTX 2000 Ada, 16 GB:

| ceiling | total, s | a step, s | peak memory of the card, MiB |
|---|---:|---:|---:|
| none | 50.3 | 7.38 | 12 654 |
| `CMF_GPU_VRAM_MB=8000` | 53.5 | 8.05 | 7 634 |
| `CMF_GPU_VRAM_MB=6000` | 54.4 | 8.18 | 5 720 |
| `CMF_GPU_VRAM_MB=4000` | 60.2 | 9.15 | 3 710 |

The picture under the 4 GB ceiling is the same picture: 49.6 dB PSNR against the run without one.

## Speed

NVIDIA RTX 2000 Ada, 16 GB, one image, seed 42, single runs (2048 x 2048: the median of three).

| run | prompt, s | six steps, s | a step, s | decode, s | total, s | peak memory of the card, MiB |
|---|---:|---:|---:|---:|---:|---:|
| CUDA, 1024 x 1024 (`--min-area 1024`) | 2.4 | 11.7 | 1.95 | 0.5 | **16.1** | 12 116 |
| CUDA, 2048 x 2048 | 2.6 | 44.3 | 7.38 | 1.7 | **50.3** | 12 654 |
| CUDA, 4096 x 4096 | 9.2 | 264.5 | 43.90 | 8.3 | **283.7** | 13 414 |
| CUDA, edit at 2048 x 2048, one 1024 x 1024 picture | 9.8 | 67.9 | 11.37 | 1.7 | **80.8** | 12 854 |
| Vulkan, 1024 x 1024 (`--min-area 1024`) | 13.0 | 70.0 | 10.99 | 4.0 | **88.7** | 12 529 |

## How it runs on CUDA

- **The transformer is one chain on the card.** The residual stream goes up once a step and comes back once.
  Between them every block runs there: an adaLN-modulated RMS norm into f16, the qkv product, per-head q/k norm
  and the rotation, attention, the output projection, the gated residual, then the SwiGLU MLP. The products are
  cuBLAS on the tensor cores (f16 operands, f32 sums).
- **Attention materializes nothing.** A warp owns 16 query rows of one head and walks the keys 16 at a time: two
  tensor-core products (`mma.sync`) with the softmax between them in registers. Keys and values are stored in
  the order the lanes read them. Measured by leaving parts of the kernel out, its time was the loads of its
  operands, not the products; the reordering halved it.
- **Weights stay in their codec.** A q4tp block is 222 MB on the card and becomes a dense f16 plane only while its
  product runs; one plane buffer serves the stack.
- **The prompt encoder** (50 layers of Qwen3-VL-32B) runs the same way with a causal attention kernel. Every
  weight is used once, so none is kept: the encoder never holds more than one layer's planes.
- **The VAE** decodes 256-pixel tiles in batches through the same chain; an edit's reference pictures are
  encoded by a chain of convolutions (im2col panels and one product each).

On Vulkan and Metal the pipeline uses the per-operation kernels of the MiniMax-H3 video path: each block's
activations cross to the host and back.

## Packing

```bash
cortiq animate-pack --out minimax-h3-image-q4tp.cmf \
  --in mmh3-turbo-fl2va-q4tp.cmf \
  --dit minimax_h3_image_pruned_bf16.safetensors \
  --image-vae minimax_h3_image_vae_fp16.safetensors --image-vae-quant q8_2f \
  --drop avae --quant q4tp
```

`--dit` is ComfyUI's pruned checkpoint (`Comfy-Org/MiniMax-H3-Image`): its adaLN is already a rank-8 curve and
its output layer is the bank of six per-step heads. The prompt encoder is MiniMax-H3's, unchanged, so `--in`
carries it and the vision tower over from a MiniMax-H3 container; `--te` packs it from the bf16 source instead.

## What was checked

- The chain against the per-operation path (every product a separate f32 GEMM) on a 2048 x 2048 edit:
  54.0 dB PSNR between the pictures.
- CUDA against Vulkan, 1024 x 1024, the same seed: 47.0 dB.
- A 4 GB ceiling against none: 49.6 dB.
- The VAE decoder's chain against its per-operation path: 47.5 dB; against f16 weights: 48.5 dB.
- `CMF_H3_SAT=1` counts the halves a panel stores at the saturation value: none on text-to-image and edit runs.
- A prompt encoder in 8 bits under the same transformer: the conditioning differs by 3 % (relative RMS), the
  pictures by 17.7 to 25.4 dB, with the same level of detail. The file keeps 4 bits.
- Not checked: the reference implementation was not run on the same card, so there is no picture-to-picture
  comparison with it.

## Options

| variable | effect |
|---|---|
| `CMF_GPU=cuda` | the native CUDA backend (a build with `--features cuda`) |
| `CMF_GPU_VRAM_MB=N` | the ceiling on the card's memory |
| `CMF_CUDA_PROF=1` | stage times of the chains (synchronous, slower) |
| `CMF_H3_SAT=1` | count clamped halves and the largest magnitudes of the chain's panels |
| `CMF_MMH3_CHAIN=0`, `CMF_TE_CHAIN=0`, `CMF_VAE3D_CHAIN=0` | the per-operation path for the transformer, the encoder, the VAE |
| `CMF_H3_FLASH=0` | attention as two batched products a chunk instead of the flash kernel |
| `CMF_LATENT_OUT=FILE`, `CMF_INIT_LATENT=FILE` | write the final latent; start from a given noise |
| `CMF_TE_DUMP=FILE`, `CMF_TE_LOAD=FILE`, `CMF_TE_FROM=FILE.cmf` | write the prompt's conditioning; use another run's; take the encoder from another container |

## Limits

- **Edits of a densely detailed, high-contrast picture come out over-processed at 2048 x 2048** (oversaturated,
  posterized). The same edit at 1024 x 1024 (`--width 1024 --height 1024 --min-area 1024`) is clean, and so are
  edits of softer pictures at 2048 x 2048. An 8-bit encoder, an 8-bit transformer and the per-operation f32
  path give the same result, so the codec and the chain are ruled out; the cause is not found.
- Vulkan runs per operation and is 5.5 times slower than CUDA on the same card. Metal uses the same
  kernels and has not been measured.
- The attention kernel needs tensor cores with `mma.sync` (compute capability 8.0 and up); older cards take
  two batched products instead.
- The far-field (O(1)) attention of the turbo language models does not apply: priced on the model's own
  operands at 2048 x 2048, it changes the attention output by 12 to 65 %.
