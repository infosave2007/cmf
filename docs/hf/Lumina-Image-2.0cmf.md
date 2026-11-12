---
library_name: cortiq
license: apache-2.0
base_model: Alpha-VLLM/Lumina-Image-2.0
pipeline_tag: text-to-image
tags:
- cmf
- cortiq
- lumina
- text-to-image
- diffusion
- quantized
- cuda
- vulkan
- metal
base_model_relation: quantized
---

# Lumina-Image 2.0 — one 3.27 GB file, no Python

[Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0)
— text encoder, DiT and VAE — packed into a single
[CMF](https://github.com/infosave2007/cmf) file and run by a pure-Rust
engine. No torch, no diffusers, no CUDA install, no C++ toolchain: one
binary, one file, and the GPU it finds. On an NVIDIA card the engine's own
CUDA backend renders 512 × 512 in **40.5 s** on an RTX A5000 (30 steps,
CFG 4).

**What it buys you** is deployment, not speed: 3.27 GB against the
reference checkout's 19.8, 3.25 GB of VRAM against 10.5, and nothing to
install. **What it costs** is time on hardware that can run the
reference — the numbers are below, unedited.

## The same image, both ways

![cortiq q4tp against the diffusers reference](compare-512.png)

*Left: this file, 4-bit, rendered by cortiq on Vulkan. Right: the
diffusers reference in bf16 on CUDA. Same prompt, same seed, same 30
steps at CFG 4, and — through `CMF_INIT_LATENT` — the same starting
noise, without which the two sample different noise and no pixel
comparison means anything.*

**22.3 dB PSNR, mean difference 11.8 of 255.** The composition, pose and
light are the same; the fur, the branches and the snow texture are not
identical. Four-bit weights cost detail, not the picture.

## Measured

One machine, both arms, 2026-08-06: **NVIDIA RTX PRO 6000 Blackwell** +
**48-thread AMD EPYC 9B45**, cortiq 0.5.58 against diffusers 0.39 /
torch 2.11. 512 × 512, 30 steps, CFG 4, seed 7, end to end.

| | this file (cortiq) | reference (diffusers) |
|---|---|---|
| GPU render | 14.7 s (Vulkan) | **2.2 s** (CUDA, bf16) |
| GPU, first image incl. load | 14.7 s | 11.8 s |
| GPU peak VRAM | **3.25 GB** | 10.5 GB |
| CPU render, 48 threads | 468 s | **158 s** (fp32) |
| CPU peak RSS | **3.16 GB** | 20.6 GB |
| On disk | **3.27 GB** | 19.8 GB |
| Runtime dependencies | **none** | torch + diffusers + CUDA |

The reference is still 6.7× faster on a card this size and 3× on the CPU.
If your machine can hold 20 GB of weights and a torch stack, **run the
reference** — it is faster, and it is the thing this file was made from.

The gap was 36× two releases ago. What closed most of it, in order of
size: the fused block stopped allocating its 300 MB of scratch on every
one of 1560 calls per image; both GEMM tiles were laid out 16 floats to a
row and read down the column, so the sixteen threads of a row shared one
memory bank of 32 and every read serialised; attention ran one head at a
time behind a shared score buffer, at 34 workgroups on a 150-SM card; and
the GEMM now runs on the card's matrix units — tensor cores on NVIDIA —
through cooperative matrices at f16 16×16 with an f32 accumulator, which
took the isolated kernel from 5 605 to 50 730 GFLOP/s.

That last one moves the image in its last bits, since the multiplies are
f16 where they were f32. It does not move it away from the reference: the
tensor path measures 22.5 dB against it and the scalar path 22.3. The
matrix units are used only where the adapter reports that exact shape, and
`CMF_COOP=0` turns them off.

Where this file is the better answer: machines that cannot. Apple
silicon with no CUDA, a laptop with 13 GB free, a shipped application
that cannot carry a Python environment, an offline box, an image
pipeline that has to start in one second rather than twelve.

### The CUDA backend

`cortiq` 0.8.79 runs a whole Next-DiT block on the card in one call over
CUDA: the sandwich norms, the q/k/v products, grouped-query attention with
the interleaved rotation, the gated residuals and the SwiGLU feed-forward,
the weights kept on the card in their 4-bit codec. NVIDIA RTX A5000 (24 GB),
seed 7, 30 steps, CFG 4, single runs, the same binary on both backends:

| | CUDA | Vulkan |
|---|---|---|
| 512 × 512 | **40.5 s** | 55.4 s |
| 1024 × 1024 | **180.0 s** | 252.7 s |

Before 0.8.79 the same card took 332.2 s on CUDA at 512 × 512: every
product went to the card on its own and attention, whose 24 query heads
share 8 key heads, ran on the host. The CUDA picture is 62.6 dB from that
per-operation render and 40.7 dB from the Vulkan one. The CUDA backend is a
build feature; the prebuilt release binaries do not include it:

```bash
cargo install cortiq-cli --features cuda     # needs the driver and the CUDA 12.x libraries
CMF_GPU=cuda cortiq imagine lumina-q4tp.cmf --prompt "…" --height 512 --width 512 --out fox.ppm
```

### Apple silicon

MacBook Air M4 (fanless, 24 GB), Metal, same settings, cortiq 0.5.57:

| | 30 steps |
|---|---|
| 512 × 512 | 212 s |
| 256 × 256 | 67 s |

Under 0.5.57 a render also leaves the CPU alone — the wait for each
command buffer used to spin, which cost a performance core and, on a
fanless machine, some of the GPU's power budget. It is 2% of one core
now, down from 100%.

The reference is not measured on this machine: its 19.8 GB does not fit
the free disk, which is the point of the comparison rather than a gap in
it.

## Run it

```bash
cargo install cortiq-cli          # pure Rust; needs >= 0.5.57
hf download infosave/Lumina-Image-2.0cmf lumina-q4tp.cmf --local-dir .

cortiq imagine lumina-q4tp.cmf \
  --prompt "a red fox sitting in snow at sunset, photorealistic, detailed fur" \
  --height 512 --width 512 --steps 30 --seed 7 --out fox.ppm
```

The GPU is used when one is visible — Metal on macOS, Vulkan or DX12
elsewhere. `CMF_GPU=0` forces the CPU. `cortiq gpu` prints what the
engine sees; if it names `llvmpipe` rather than your card, the Vulkan
loader has found a software driver and the run will crawl.

Output is P6 PPM — no image library is linked into the engine. Convert
with `sips -s format png fox.ppm --out fox.png` on macOS, `convert
fox.ppm fox.png` with ImageMagick, or `ffmpeg -i fox.ppm fox.png`.

Flags that matter: `--steps` (30 is the default and a good trade),
`--cfg` (4.0 default; **≤ 1 disables classifier-free guidance and halves
the work** at some prompt-adherence cost), `--seed`, `--height/--width`
(multiples of 16).

**Vulkan and DX12 need ≥ 0.5.54.** Before it, 512 × 512 died on the first
step: one SwiGLU dispatch asks for 75060 workgroups where a wgpu
dimension holds 65535. Do not set `CMF_DIT_FUSED=0` there either — the
per-op path renders a visibly wrong image at 512 × 512 (11.0 dB against
the CPU reference, where the default fused path is at 28.2). Both are
fixed or defaulted correctly in 0.5.54 and later.

## Reproduce the comparison

```bash
pip install torch diffusers transformers accelerate
hf download Alpha-VLLM/Lumina-Image-2.0 --local-dir lumina-ref

python3 tools/bench_lumina.py --ref lumina-ref --cmf lumina-q4tp.cmf \
    --cortiq ./target/release/cortiq                 # the timings above
python3 tools/bench_lumina.py --ref lumina-ref --cmf lumina-q4tp.cmf \
    --cortiq ./target/release/cortiq --same-noise    # the image pair
```

The script is in the [engine repository](https://github.com/infosave2007/cmf).
`--same-noise` writes the reference's starting latent and hands it to
cortiq through `CMF_INIT_LATENT` (needs ≥ 0.5.56).

## Which file

Two layouts of the same weights — same 4-bit grid, same images.

| file | size | notes |
|---|---|---|
| `lumina-q4tp.cmf` | **3.27 GB** | **take this one.** Needs cortiq ≥ 0.5.40 |
| `lumina-q4t.cmf` | 3.45 GB | the older layout, for cortiq < 0.5.40 |

`q4tp` keeps the nibbles byte-identical and stores each tile's scale as a
5-bit rung on a per-row ladder instead of a standalone f16 — the scale
field was 11% of a q4t file. Rendering speed is unchanged (interleaved at
256px/8 steps on an M4: 26 and 27 s against 27 and 26), and so is the
image: 43.6 dB against the q4t render, 0.31% mean pixel difference.

Any existing `.cmf` converts in place, no checkpoint needed:

```sh
cortiq requant lumina-q4t.cmf --output lumina-q4tp.cmf --quant q4tp
```

## What is in the file

The whole pipeline, not just the transformer: **Gemma-2 text encoder +
Next-DiT + FLUX VAE + the tokenizer**, 888 tensors, 5.27B parameters
across the three stages, in one mmap-served file that checks its own
integrity. Projections are `q4_tiled`; modulation and embeddings stay q8,
the VAE f16, norms f32.

## Fidelity

The engine mirrors the diffusers `Lumina2Pipeline`: the system-prompt
template, Gemma `hidden_states[-2]` as caption features,
FlowMatchEulerDiscrete with static shift 6, the model called at
t = 1 − σ, CFG with per-row norm rescaling and the sign flip before the
Euler step. The 22.3 dB above is what remains after that: 4-bit weights,
int8 activations on the CPU arm, and a device working in f32 where the
reference works in bf16.

CPU and GPU renders of the same seed also differ from each other — same
cause, and a flow-matching sampler grows small differences over its
steps. On the card above, 4 steps at 512 × 512: 28.2 dB between them.

## Reproduce the packing

```bash
cortiq imagine-pack <diffusers-root> --quant q4t --out lumina-q4t.cmf
```

`<diffusers-root>` is the Lumina-Image-2.0 checkout with `tokenizer/`,
`text_encoder/`, `transformer/`, `vae/`. Takes ~9 minutes and streams
shard by shard.

## License

Apache-2.0, inherited from the base model. Weights © Alpha-VLLM; this
repackaging only changes the storage format.

The engine, converter and format spec: https://github.com/infosave2007/cmf
