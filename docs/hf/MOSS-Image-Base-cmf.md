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

> **Publication draft — do not upload this card until the artifact table and
> sample links below have been filled from the release-validation log.** It is
> intentionally explicit about what has and has not been measured.

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

## Release status

The following values are deliberately **pending** until the final q8/q4 packs
are generated, verified, and rendered on the release hardware. Do not replace
them with estimates.

| artifact | bytes / SHA-256 | validation | measured Vulkan / Metal speed |
|---|---:|---|---|
| `moss-image-base-q8_2f.cmf` | pending | pending | pending |
| `moss-image-base-q4tp.cmf` | pending | pending | pending |

The release must include a `.sha256` file beside every CMF and must report only
measurements from the exact uploaded files. A successful render is a functional
smoke test; it is not by itself a claim of numerical parity or broad image
quality equivalence with the upstream PyTorch pipeline.

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

## Reproducible examples and samples

The following commands define the samples that should be uploaded under
`samples/` together with this card. **The image links are intentionally absent
until those files have actually been generated and inspected.**

```bash
# samples/fox-1024-q8.png
cortiq imagine moss-image-base-q8_2f.cmf \
  --prompt "A small red fox reading a book in a sunlit library, highly detailed." \
  --width 1024 --height 1024 --steps 50 --cfg 3 --shift 5 --seed 42 \
  --out samples/fox-1024-q8.png

# samples/refrigerator-1024-q8.png
cortiq imagine moss-image-base-q8_2f.cmf \
  --prompt "A stainless-steel refrigerator with the Chinese text '新鲜每一天' on its door." \
  --width 1024 --height 1024 --steps 50 --cfg 3 --shift 5 --seed 7 \
  --out samples/refrigerator-1024-q8.png

# samples/fox-1024-q4tp.png — same recipe, lower-precision DiT
cortiq imagine moss-image-base-q4tp.cmf \
  --prompt "A small red fox reading a book in a sunlit library, highly detailed." \
  --width 1024 --height 1024 --steps 50 --cfg 3 --shift 5 --seed 42 \
  --out samples/fox-1024-q4tp.png
```

The CMF-native random generator is deterministic for a given CMF, command and
seed, but it is not presented as a bit-identical replacement for PyTorch's
random-generator sequence. Do not claim that the same integer seed recreates
the upstream image without a direct parity measurement.

## Validation required before publishing

The source contract is pinned to the revision above. Release validation must
record all of the following against the final artifact, not a temporary smoke
file:

1. `cortiq verify` and the external SHA-256 check for both profiles;
2. exact embedded-template token IDs for the source English, Chinese and empty
   prompt fixtures;
3. the Flux2 BN → 2×2 unpatch → post-quant-convolution → decoder oracle on a
   fixed latent;
4. a q8 render with CFG disabled and one with the source CFG/negative-layer-skip
   path enabled, then the same two functional paths for q4tp;
5. inspected fixed-seed samples and measured stage timings/peak memory on each
   hardware/backend that is listed on this card.

Only after those checks may this card replace the `pending` cells in the
release-status table with artifact sizes, checksums, sample links and measured
performance.

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
