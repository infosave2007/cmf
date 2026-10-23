---
library_name: cortiq
license: apache-2.0
base_model:
- OpenMOSS-Team/MOSS-Image-Base
base_model_relation: quantized
pipeline_tag: text-to-image
language:
- en
- zh
tags:
- cmf
- cortiq
- text-to-image
- moss-image
- q8_2f
- q4tp
- rust
---

# MOSS-Image-Base — CMF

> Native MOSS-Image support is available in **cortiq v0.8.30**. This is a
> separate CMF publication for MOSS-Image-Base; it does not replace any other
> model card or repository.

This repository packages
[OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)
at source revision
[`85975793d8b3d432f1b6e27f352abaa4c2a4c0d3`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/85975793d8b3d432f1b6e27f352abaa4c2a4c0d3)
as a single portable [CMF](https://github.com/infosave2007/cmf) file for the
Rust `cortiq` runtime. The intended destination is the **separate** repository
[`infosave/MOSS-Image-Base-cmf`](https://huggingface.co/infosave/MOSS-Image-Base-cmf);
it must never replace an existing model card or a different model repository.

MOSS-Image is not a generic Z-Image alias. Its source pipeline combines:

- a dense hybrid **Qwen3.5** text encoder and its exact tokenizer/chat template;
- a 42-layer **MOSSImageTransformer** (width 3072, 24 heads, FFN width 8192);
- a **FLUX.2 AutoencoderKLFlux2** decoder.

The CMF embeds the tokenizer, template, scheduler and component configuration.
It retains only the inference path: no unused Qwen visual tower, no Flux2 VAE
encoder and no Flux2 `quant_conv` encoder-side weights. Generation runs in
three explicit residency stages — text encoder, DiT, then VAE — so a component
is released before the next one claims GPU memory. This is important on 24 GB
Vulkan cards and unified-memory Metal systems.

## Release files and validation

| artifact | exact bytes | SHA-256 | profile |
|---|---:|---|---|
| [`moss-image-base-q8_2f.cmf`](moss-image-base-q8_2f.cmf) | 13,256,239,565 (12.346 GiB) | `f4dd15a29b58bf6185f5018b40614a65b7ade33e2617339d327f5dc1c6f38e11` | quality-first default |
| [`moss-image-base-q4tp.cmf`](moss-image-base-q4tp.cmf) | 10,856,278,477 (10.111 GiB) | `c6250b2bab153a0a0b9f6c774ea77fc4a80d472e12b9a48d3d498a825b94f8dd` | lower DiT footprint |

A matching `.sha256` file is published beside each CMF. Both final files passed
`cortiq verify`, an external SHA-256 check, a Vulkan CFG render, and the
source-pinned template/Flux2-decoder contract tests. The latter checks the
stored template token contract and the BatchNorm → 2×2 unpatch →
post-quant-convolution → decoder path on a fixed source oracle latent.

### Measured Vulkan reference

These are release-hardware measurements, not estimates: NVIDIA GeForce RTX
3090 (24 GiB), NVIDIA Vulkan driver 595.91.07, `cortiq v0.8.30`, 512×512,
8 steps, CFG 3, shift 5, seed 42, with the exact commands below. `median step`
includes the two DiT passes required by CFG; text preparation, model loading
and VAE decoding are reported separately by the CLI.

| file | total | median DiT step | text / DiT load / VAE |
|---|---:|---:|---:|
| q8_2f | 348.8 s | 41.66 s | 2.6 s / 0.3 s / 6.7 s |
| q4tp | 333.7 s | 39.80 s | 2.7 s / 0.3 s / 7.1 s |

No Metal performance number is claimed here: the release uses the same staged
GPU-residency design and Metal-capable projection/attention paths, but this
specific release was not benchmarked on Apple hardware. A successful render is
a functional smoke test, not a claim of numerical parity or broad image-quality
equivalence with upstream PyTorch.

## Quick start

Use the `cortiq` build that contains native MOSS-Image support (the release
notes must name the exact minimum version before publication):

```bash
REPO=infosave/MOSS-Image-Base-cmf
MODEL=moss-image-base-q8_2f.cmf       # or moss-image-base-q4tp.cmf

hf download "$REPO" "$MODEL" "$MODEL.sha256" --local-dir .
sha256sum -c "$MODEL.sha256"          # macOS: shasum -a 256 -c "$MODEL.sha256"
cortiq verify "$MODEL"

cortiq imagine "$MODEL" \
  --prompt "A small red fox reading a book in a sunlit library, highly detailed." \
  --out fox.png
```

The embedded source recipe is **1024×1024**, **50** flow-matching Euler steps,
**CFG 3**, scheduler **shift 5**, an empty negative prompt, and a negative-pass
skip at zero-based DiT layer **22**. Dimensions must be positive multiples of
16. The source performs two DiT forwards per step at its default CFG setting;
this is expected rather than a hidden slowdown.

For a fast functional smoke test before a full render:

```bash
WGPU_BACKEND=vulkan CMF_GPU=wgpu cortiq imagine "$MODEL" \
  --prompt "A small red cube on a white table." \
  --width 256 --height 256 --steps 2 --cfg 3 --seed 42 \
  --out smoke.png
```

On headless Linux, set `XDG_RUNTIME_DIR=/tmp` if the graphics stack requires
it. `CMF_GPU=0` is a CPU diagnostic fallback, not a practical default for this
model. Leave `CMF_MOSS_TE_GPU` unset for release-quality runs: GPU text
encoding remains an opt-in experiment rather than the default path.

## Profiles and what is packed

| profile | DiT matrices | text encoder | VAE / control tensors | intended use |
|---|---|---|---|---|
| `q8_2f` | q8_2f | q8_2f projections; q8-row embeddings | Flux2 decoder, norms, biases and time/modulation controls stay floating-point | quality-first default |
| `q4tp` | q4tp | q8_2f projections; q8-row embeddings | same floating-point decoder and controls | lower DiT footprint; evaluate output on the target prompts |

`q4tp` is an explicit lower-precision profile, not a statement that it is
visually interchangeable with q8. The release must publish fixed-seed q8 and
q4 samples and their exact commands before making a quality comparison.

The prompt path is source-sensitive: the stored Qwen template is rendered with
`add_generation_prompt=true` and `enable_thinking=true`, producing the required
assistant `<think>` suffix. The scheduler is stored with the model and uses the
source nearest-grid flow schedule; it is not substituted with a generic Euler
ladder.

## Reproducible examples and inspected samples

The two images below were generated and visually inspected on the Vulkan
reference system using the exact 512×512/8-step commands shown here. They are
intentionally a fast, reproducible demonstration rather than a claim that
8 steps replaces the upstream 1024×1024/50-step recipe.

| q8_2f | q4tp |
|---|---|
| ![q8_2f fox reading in a library](samples/fox-512-q8.png) | ![q4tp fox reading in a library](samples/fox-512-q4tp.png) |

```bash
PROMPT='A small red fox reading a book in a sunlit library, highly detailed, cinematic composition'

# samples/fox-512-q8.png
WGPU_BACKEND=vulkan CMF_GPU=wgpu cortiq imagine moss-image-base-q8_2f.cmf \
  --prompt "$PROMPT" --width 512 --height 512 --steps 8 --cfg 3 --shift 5 \
  --seed 42 --out samples/fox-512-q8.png

# samples/fox-512-q4tp.png
WGPU_BACKEND=vulkan CMF_GPU=wgpu cortiq imagine moss-image-base-q4tp.cmf \
  --prompt "$PROMPT" --width 512 --height 512 --steps 8 --cfg 3 --shift 5 \
  --seed 42 --out samples/fox-512-q4tp.png
```

For the source-quality recipe, use 1024×1024 and 50 steps as described above;
expect significantly higher runtime. The CMF-native random generator is
deterministic for a given CMF, command and seed, but it is not presented as a
bit-identical replacement for PyTorch's random-generator sequence. Do not
claim that the same integer seed recreates the upstream image without a direct
parity measurement.

## Release validation record

The source contract is pinned to the revision above. The following checks were
run against the final release files (or, where noted, against the pinned source
oracle):

1. `cortiq verify` and external SHA-256 checks for q8_2f and q4tp;
2. exact embedded-template token IDs for the pinned English, Chinese and empty
   prompt fixtures;
3. the Flux2 BN → 2×2 unpatch → post-quant-convolution → decoder oracle on a
   fixed source latent;
4. Vulkan renders for q8_2f and q4tp through the CFG / negative-layer-skip
   path, plus inspected fixed-seed q8_2f and q4tp samples above;
5. the measured 512×512 timing record above on the named RTX 3090/Vulkan
   reference system.

The test suite validates critical component contracts. It does **not** replace a
full end-to-end numerical-parity oracle for the upstream DiT, so the card makes
no bit-exact PyTorch-equivalence claim.

## Options

| option | meaning |
|---|---|
| `--width`, `--height` | positive multiples of 16; source default 1024×1024 |
| `--steps` | requested scheduler steps; default 50 |
| `--cfg G` | CFG runs when `G != 1`; source default 3 |
| `--negative-prompt` | source default is empty; used by the CFG branch |
| `--shift` | flow-scheduler shift; source default 5 |
| `--max-sequence-length` | prompt cap after the stored template; source default 4096 |
| `--seed`, `--num-images N` | image *i* uses `seed + i` |
| `--out` | PNG/JPEG/PPM output path; PNG is the default |

## License and provenance

The upstream MOSS-Image-Base release is licensed under
[Apache-2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/LICENSE).
CMF artifacts are quantized/repackaged derivatives and retain the upstream
license and attribution. Consult the upstream repository and model card for
model behavior, dataset disclosures and any use restrictions. This card does
not claim support for LoRA adapters, image editing, layout control, hosted API
compatibility, or source-level numerical parity unless a future release adds
and validates those capabilities.
