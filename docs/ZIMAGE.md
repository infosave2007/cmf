# Z-Image and Z-Image-Turbo

Tongyi-MAI's text-to-image models. Both use the same 6.15B single-stream
`ZImageTransformer2DModel`, the same Qwen3-4B text encoder and the same Flux
VAE. cortiq ships each one as a single `.cmf` file:

| model | recipe stored in the file | source DiT |
|---|---|---|
| Z-Image-Turbo | 1024², 8 steps, guidance 0 (one DiT forward per step), shift 3 | fp32 |
| Z-Image (base) | 1024², 28 steps, guidance 4 with CFG, negative prompt "", shift 6 | bf16 |

## Run

```sh
cortiq imagine z-image-turbo.cmf --prompt "A cat sitting on a windowsill at sunset"
cortiq imagine z-image.cmf --prompt "…" --negative-prompt "blurry, low quality"
```

Every option falls back to the recipe stored in the file. Available options:

| option | meaning |
|---|---|
| `--height`, `--width` | image size in pixels; each must be a multiple of 16 |
| `--steps` | number of DiT forwards per image |
| `--cfg` (alias `--guidance`) | 0 turns CFG off. Above 0 the prediction is `pos + g·(pos − neg)`, the diffusers Z-Image formula, which is not Lumina's |
| `--negative-prompt` | the CFG negative prompt (default ""); it goes through the same chat template as the prompt |
| `--cfg-normalization [C]` | clips ‖pred‖ to C·‖pos‖, where the norm is taken over the whole tensor. A bare flag means 1.0, which is diffusers `True` |
| `--cfg-truncation T` | turns CFG off on every step whose `t_norm = 1 − σ` is greater than T |
| `--shift` | scheduler shift override |
| `--max-sequence-length` | token cap applied after the chat template (default 512) |
| `--num-images N` | image i uses seed + i. The prompt is encoded once, and each latent is finished before the VAE loads. The files are named `<stem>_<i>.<ext>` |
| `--seed` | random seed |
| `--out` | output path: `.png` (default `out.png`), `.jpg`, or `.ppm` |

Environment variables:

| variable | effect |
|---|---|
| `CMF_ZIMAGE_PROF=1` | prints stage times and the median step time |
| `CMF_INIT_LATENT=<raw f32 [1,16,H/8,W/8]>` | injects a noise latent, for example one of the oracle `noise_*.f32` files |
| `CMF_ZIMAGE_TRACE=<dir>` | dumps `v_i`, `lat_i`, `vpos_i` and `vneg_i` for every step |
| `CMF_ZIMAGE_DIT_DIR=<diffusers transformer dir>` | runs the DiT straight from the source weights (parity work only) |
| `CMF_GPU=0` | the whole pipeline on the CPU (the bit-level reference; minutes per image) |
| `CMF_ZIMAGE_GPU=0` | the DiT on the CPU, everything else unchanged |
| `CMF_ZI_VAE=0` | the older per-conv VAE instead of the resident one |
| `CMF_ZIMAGE_TE_EXACT=0` | the text encoder's q8 projections through the default int8-activation kernel (3.3× less accurate, see below) |
| `CMF_ZIMAGE_OVERLAP=0` | uploads the device weights after the text encoder instead of beside it |
| `CMF_ZI_AMAX=1` | prints the per-block f16 maxima of the device chain |

## Where it runs

On Linux and Windows with a Vulkan GPU (tensor cores via cooperative
matrices, f16, 32-wide subgroups — NVIDIA), everything but the text encoder
runs on the device with no flags:

- **DiT**: every block's weights are expanded once per process to f16
  planes (11.6 GB for either model), and a step is one resident chain —
  tensor-core GEMMs, flash attention, fused row kernels, one 1 MB upload and
  one 1 MB readback. CFG runs the cond/uncond pair as one batch-2 forward;
  steps past `--cfg-truncation` switch to a batch-1 program.
- **VAE**: resident decoder — implicit-GEMM 3×3 convs on the tensor cores
  (the 2× upsample folded into the gather), device GroupNorm, the mid-block
  attention in query chunks; one upload, one readback.
- **Text encoder**: on the CPU, with weight-only exact q8 projections, while
  a helper thread uploads the DiT weights; the device context and the kernel
  compiles start on another helper at the beginning of the run, and the VAE
  weights upload while the steps run.

Other devices and backends fall back to the CPU path piece by piece.

## Pack

```sh
cortiq imagine-pack /path/Z-Image-Turbo --out z-image-turbo.cmf      # DiT q8_2f, TE q8_2f (defaults)
cortiq imagine-pack /path/Z-Image --out z-image.cmf
```

The packer finds the model through `model_index.json` (`ZImagePipeline`) and
streams the shards one tensor at a time. It writes `<out>.sha256` next to the
file and records the sha256 of every source shard in the provenance.

The file holds four parts:

- `dit.*` uses the diffusers names. The block projections and `cap_embedder.1` use `--quant`. adaLN is kept at 16 bit. The embedders, the final layer, the pad tokens and every norm and bias are stored as f32.
- `te.*` holds Qwen3 layers 0..34, because `hidden_states[-2]` is the raw output of layer 34. The layer-35 weights and the final norm are dropped. The projections use `--te-quant`, `embed_tokens` is q8_row, and the `--te-keep` projections stay bf16.
- `vae.*` holds the decoder only, in f32.
- `zimage.config_json` holds the recipe, and the tokenizer is in VOCAB.

Why the defaults are q8 (measured, Turbo unless noted, device path, injected
oracle noise, PSNR in dB of the u8 image against the fp32 diffusers image):

| codec | size | text encoder `h_m2` rel vs fp32 | images |
|---|---|---|---|
| DiT q8_2f, TE q8_2f (shipped) | 10.46 GB | 4.1e-3 (p0), 5.2e-3 (p1) | 6-seed sweep 512² p0: median 28.6, min 27.1 |
| DiT q8_2f, TE q4tp (layer-6 down_proj kept bf16) | 8.78 GB | 5.7e-2, 6.6e-2 | sweep median 22.2, min 16.4; 6 to 10 dB below the q8 TE on every case below |
| diffusers bf16 (reference) | 20.5 GB | 8.7e-3 | sweep median 32.9, min 16.0 |

| case (PSNR vs fp32; vs bf16 where no fp32 exists) | TE q8 | TE q4tp |
|---|---|---|
| Turbo 512² p0 / p1 | 26.1 / 26.1 | 19.4 / 17.0 |
| Turbo 512² p2 (vs bf16) | 32.9 | 23.1 |
| Turbo 1024² p0 / p1 (p1 vs bf16) | 33.3 / 23.6 | 23.2 / 21.5 |
| base 512², 3 steps, CFG + negative | 35.4 | 24.2 |
| base 1024², 28 steps, CFG (vs bf16) | 26.6 | 18.4 |

The q4tp text encoder keeps the composition and legible text but moves
details and colours visibly; it does not hold quality, so the files stay at
10.46 GB. (The DiT codec was settled earlier on the CPU path: q4tp costs 3 to
15 dB against q8.)

Dev flags:

| flag | effect |
|---|---|
| `--quant raw --te-quant raw` | keeps the source dtype, which gives an exact container |
| `--dit-layers N` | builds a truncated DiT for kernel tests |
| `--variant turbo\|base` | overrides the recipe written into the file |

## Correctness

Both models were checked against diffusers 0.40 with fp32 CPU oracles
(`python/zimage_oracle.py`, which uses a CPU `torch.Generator(42)` noise that
is then injected). The engine side ran through `tests/zimage_parity.rs`
(`CMF_GPU=0`) and through the CLI, with `python/zimage_compare.py` comparing
the outputs.

| stage | result |
|---|---|
| tokenizer | ids exact on 12 prompts, including one truncated to 512 tokens |
| text encoder `h_m2` | rel 2.1e-7 to 3.0e-7 against fp32 |
| DiT forward (Turbo r512 i0/i5, r400x592 i3 with image pad rows; base r512 c3 i0/i2) | `v` rel 6e-7 to 2.3e-6. RoPE tables are bit-exact. For comparison, diffusers bf16 is 1.2e-2 to 2.8e-2 away from fp32 |
| VAE decoder (r512, r400x592) | rel 6.6e-7 |
| full CLI run, Turbo 512² and 400×592, 8 steps | `lat_8` rel 5.2e-5 and 5.7e-5; PNG PSNR 78.3 and 78.4 dB |
| full CLI run, base 512², CFG 4 plus negative prompt, 3 steps | PSNR 78.4 dB |
| the same with `--cfg-normalization --cfg-truncation 0.1` | PSNR 84.3 dB (the last step correctly runs without CFG) |

## Device parity (RTX 3090, Vulkan)

| check | result |
|---|---|
| one DiT forward on the oracle inputs, device vs the CPU path on the same container (Turbo r512 i0 / i5, r1024 i0) | `v` rel 8.8e-4 / 5.7e-4 / 1.7e-3 |
| the same, base (r512 c3 i0 / i2) | 3.0e-4 / 1.8e-4 |
| CFG pair (one batch-2 forward) against the two items stepped one by one (base r512 i0) | cond 2.2e-4, uncond 0 (bit-identical); uncond single vs CPU 1.7e-4 |
| VAE on the oracle latent (r512, r400x592, r1024) | `img` rel 1.4e-4 to 1.6e-4, u8 PSNR 69.3 to 69.9 dB vs the fp32 decoder |
| Turbo 512², whole CLI run, device vs the CPU pipeline (same file, same exact text encoder), p0 / p1 | `v_0` 8.9e-4 / 1.1e-3, `lat_8` 8.8e-3 / 1.9e-2, PNG PSNR 53.4 / 45.5 dB |
| Turbo, 6 seeds 512², device vs the CPU pipeline, both with the older int8-activation text encoder | PSNR median 52.8 dB, min 49.1 dB |
| Turbo images vs fp32 (512² p0, p1, 400×592 p0, 1024² p0) | 26.1, 26.1, 23.6, 33.3 dB; diffusers bf16 vs fp32: 24.2, 30.0, 33.3, 35.2 |
| base, 3 steps CFG + negative (512² p0, 400×592 p1) vs fp32 | 35.4, 29.5 dB; diffusers bf16: 25.5 |
| base with `--cfg-normalization --cfg-truncation 0.1` vs fp32 | 37.9 dB; diffusers bf16: 33.1 |

Turbo's trajectory is chaotic in the first steps: a 1.4e-4 change of
`lat_1` moves `v_1` by 3.5e-2 on the same program, and diffusers bf16 itself
ends at `lat_8` rel 0.23 from fp32 on p0. Per-step error is therefore
checked on identical inputs (first rows above), and whole images over seeds.

## Speed (RTX 3090, driver 610.43, in-process timers)

`cortiq imagine <file> --prompt P [--height/--width]`, nothing else, one
image per process (the cold command line: model open, device upload and
kernel compile included). Medians of 3 alternating runs; peak = device
memory (nvidia-smi).

| model, size | total | text encoder (beside the upload) | upload + prepare | steps (median step) | VAE | peak |
|---|---|---|---|---|---|---|
| Turbo 512² | 4.55 s | 0.66 s | 1.79 + 0.11 s | 2.05 s (0.255) | 0.13 s | 13.8 GB |
| Turbo 1024² | 11.37 s | 0.67 s | 1.77 + 0.12 s | 8.33 s (1.039) | 0.50 s | 14.3 GB |
| base 512² | 16.20 s | 0.85 s | 1.75 + 0.30 s | 13.60 s (0.486) | 0.12 s | 13.8 GB |
| base 1024² | 59.96 s | 0.84 s | 1.81 + 0.33 s | 56.83 s (2.029) | 0.49 s | 15.3 GB |

diffusers 0.40 bf16 on the same card (all resident, SDPA, `torch 2.8`),
measured the same day, medians of 3:

| model | warm image 512² / 1024² | DiT step 512² / 1024² | cold process → first 512² image | peak |
|---|---|---|---|---|
| Turbo | 2.25 s / 8.33 s | 0.261 / 0.997 s | 17.3 s | 19.9 / 21.7 GB |
| base | 14.26 s / 53.57 s | 0.501 / 1.90 s | 24.0 s | 19.9 / 21.7 GB |

Ratios (cortiq / diffusers):

- cold command line against a cold diffusers process: Turbo 512² 0.26×,
  base 512² 0.68×;
- warm, per image in one process (4 images of one prompt; steps + VAE, plus
  the 0.64–0.85 s CPU text encoding once per prompt): Turbo 2.15 s at 512²
  (0.95×; 1.24× if the prompt is re-encoded every image), 8.81 s at 1024²
  (1.06×; 1.13×); base 13.8 s at 512² (0.97×; 1.03×), 57.4 s at 1024²
  (1.07×; 1.09×);
- DiT step: 0.98× / 1.04× (Turbo 512² / 1024²), 0.97× / 1.07× (base).

## Oracles

On a CUDA machine, `python/zimage_oracle.py` holds the full recipe: noise,
tokenizer, TE taps, fp32 CPU runs, DiT taps, the VAE, bf16 runs, the real
pipeline image and the timing bench. Run the fp32 phases with
`CUDA_VISIBLE_DEVICES=` and the GPU phases under the stand lock.
