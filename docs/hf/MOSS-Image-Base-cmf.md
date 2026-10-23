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

# MOSS-Image-Base CMF

Native, single-file CMF packaging of
[OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)
at source revision
[`85975793d8b3d432f1b6e27f352abaa4c2a4c0d3`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/85975793d8b3d432f1b6e27f352abaa4c2a4c0d3),
for the Rust [`cortiq`](https://github.com/infosave2007/cmf) runtime.

This is the standalone repository for the MOSS-Image-Base CMF artifacts. It
neither replaces nor changes cards or artifacts for other models. Native MOSS
text-to-image support, including the source layout and guidance controls below,
requires Cortiq **v0.8.31 or newer**.

## Downloads and integrity

| Artifact | Size | SHA-256 | Use |
|---|---:|---|---|
| [`moss-image-base-q8_2f.cmf`](moss-image-base-q8_2f.cmf) | 13,256,239,565 bytes (12.346 GiB) | `f4dd15a29b58bf6185f5018b40614a65b7ade33e2617339d327f5dc1c6f38e11` | Recommended, quality-first profile |
| [`moss-image-base-q4tp.cmf`](moss-image-base-q4tp.cmf) | 10,856,278,477 bytes (10.111 GiB) | `c6250b2bab153a0a0b9f6c774ea77fc4a80d472e12b9a48d3d498a825b94f8dd` | Lower DiT-memory-footprint profile |

A matching `.sha256` file is published beside each CMF. Verify a download
before use:

```bash
REPO=infosave/MOSS-Image-Base-cmf
MODEL=moss-image-base-q8_2f.cmf       # or moss-image-base-q4tp.cmf

hf download "$REPO" "$MODEL" "$MODEL.sha256" --local-dir .
sha256sum -c "$MODEL.sha256"          # macOS: shasum -a 256 -c "$MODEL.sha256"
cortiq verify "$MODEL"
```

## What the CMF runs

The packaged Base text-to-image path is:

1. the dense hybrid **Qwen3.5** text encoder with its tokenizer and stored chat
   template;
2. the 42-layer **MOSSImageTransformer** (width 3072, 24 heads, FFN width
   8192); and
3. the **FLUX.2 AutoencoderKLFlux2** decode path.

The tokenizer, template, scheduler, and component configuration are embedded.
The runtime uses staged residency—text encoder, then DiT, then decoder—so the
previous stage can be released before the next is resident.

The CMF deliberately packs components reached by this observed Base,
string-prompt text-to-image route. The Qwen visual tower and the encoder-side
Flux2/VAE path (including encoder-side `quant_conv` weights) are not included
because this route does not call them. This is a scope decision, **not** a
claim that those source components are unnecessary in every upstream workflow.
Image-conditioned, editing, video, and reference-image workflows are not
advertised by this Base CMF.

## Prompting and JSON layouts

MOSS-Image-Base supports the upstream prompt forms:

- ordinary natural-language descriptions;
- tag sequences; and
- a raw JSON layout prompt for text placement and object arrangement.

Use natural language or tags with `--prompt`. In a build whose `cortiq
imagine --help` exposes `--layout` / `--layout-file` (v0.8.31+), use exactly
one of them instead. The CLI validates that the input is a
JSON object and passes its text unchanged through the stored official MOSS
prompt template; it does not reinterpret it as a separate image-conditioning
format. The upstream layout convention documents `summary.caption`, `items`,
`bbox_format: "yxyx"`, and `bbox_coord_space: "abs_centralized"`; use the
upstream schema and matching output dimensions.

```bash
# Requires Cortiq v0.8.31+.
cortiq imagine moss-image-base-q8_2f.cmf \
  --layout-file layout.json \
  --width 1024 --height 1024 --out layout.png
```

For example, this is a direct upstream-format prompt for a 1024×1024 canvas;
save it as `layout.json` without reformatting it through another control-image
tool:

```json
{
  "summary": {
    "meta": ["poster"],
    "canvas": "single",
    "caption": "A minimalist travel poster with a cream background, a blue mountain, and EXPLORE at the top."
  },
  "items": [
    {"label": "text", "caption": "Large dark-blue uppercase title at the top.", "text": "EXPLORE", "bbox": [-416, -384, -288, 384]},
    {"label": "object", "caption": "A flat blue mountain illustration in the lower center.", "bbox": [-192, -288, 384, 288]}
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

Layout editing means editing the JSON caption, items, or bounding boxes and
then generating a new image. It is **not** pixel-space editing or img2img. In
particular, this Base CMF has no MOSS `--image` input.

The upstream project also provides a prompt-enhancer workflow that can produce
detailed captions and spatial-layout JSON. It remains an upstream Python/tool
workflow rather than a CMF CLI feature.

## Recommended quality recipe

The embedded source recipe is **1024 × 1024**, **50** Flow-Matching Euler
steps, **CFG 3**, scheduler **shift 5**, an empty negative prompt, and the
source negative-branch skip at zero-based DiT layer **22**. Width and height
must be positive multiples of 16.

For quality output, follow the upstream guidance: use **at least 50 steps** and
about **1–2 million pixels**. The default 1024 × 1024 setting is about 1.05
million pixels. Faster, lower-resolution settings are useful for functional
smokes but are not a substitute for this quality recipe.

```bash
cortiq imagine moss-image-base-q8_2f.cmf \
  --prompt "A small red fox reading a book in a sunlit library, highly detailed." \
  --width 1024 --height 1024 --steps 50 --cfg 3 --shift 5 \
  --out fox.png
```

Use **q8_2f** when output fidelity is the priority. Choose **q4tp** when a
smaller DiT footprint is more important, and evaluate it on the intended
prompts rather than assuming it is visually interchangeable with q8_2f.

## Precision profiles

CMF permits precision to be chosen independently for individual components and
tensor groups. These are deliberately mixed-precision profiles, rather than a
single uniform precision applied to every tensor.

| Profile | DiT matrices | Text conditioning | Decoder and control tensors |
|---|---|---|---|
| `q8_2f` | q8_2f | q8_2f projections; q8-row embeddings | Flux2 decoder, norms, biases, and time/modulation controls remain floating-point |
| `q4tp` | q4tp | q8_2f projections; q8-row embeddings | Same floating-point decoder and controls |

Thus the two published files differ in the large DiT matrix codec while keeping
the text-conditioning and decoder/control choices above. This separation is
intentional: it preserves the source text path and decode path while offering a
lower-footprint DiT alternative.

`q4tp-v2` is intentionally **not** used as a filename for this image profile:
in this project that name is reserved for a data-aware, calibrated/refit q4
conversion. No such image-DiT calibration has been measured here. For an
explicit, inspectable compromise, `imagine-pack --dit-keep` can lift selected
MOSS DiT blocks or projections to q8_2f over a q4tp base:

```bash
cortiq imagine-pack /path/to/MOSS-Image-Base \
  --quant q4tp --dit-keep layers.0,layers.1,layers.40,layers.41 \
  --out moss-image-base-q4tp-guarded.cmf
```

The selectors and their q8 lifts are recorded in CMF provenance. This is a
mixed-precision packaging option, not an assertion that every selector is
quality-optimal; compare the result on the actual target prompts before
publishing it.

## MOSS controls in Cortiq v0.8.31+

Confirm the installed binary with `cortiq imagine --help`. Older v0.8.30
binaries do not expose these flags.

- `--uncond-skip-layers LAYERS` selects zero-based negative-branch layers to
  skip during the active interval. The packaged source default is layer `22`.
  Use a bare flag, `none`, or `off` to disable that default.
- `--slg-layers LAYERS` and `--slg-scale SCALE` enable standard
  skip-layer-guidance (SLG) correction on selected positive-branch layers.
  When active, SLG adds a third DiT forward, so it can increase render time.
- `--slg-start F` and `--slg-stop F` set the active denoising-fraction interval
  from 0 to 1 for the source skip/SLG behavior.

```bash
# Disable the packaged negative-branch skip.
cortiq imagine moss-image-base-q8_2f.cmf \
  --prompt "A lacquered red teapot on a wooden table" \
  --uncond-skip-layers none --out no-skip.png

# Example SLG configuration; choose layer indices and scale for the target prompt.
cortiq imagine moss-image-base-q8_2f.cmf \
  --prompt "A lacquered red teapot on a wooden table" \
  --slg-layers 24,26 --slg-scale 3 --slg-start 0 --slg-stop 1 \
  --out slg.png
```

## Examples

Only visually inspected source-step examples belong in this section. The
earlier 512×512 / 8-step smoke images are deliberately not presented as
quality showcases.

### Quality-first q8_2f wildlife photograph

![A red fox on a moss-covered stone beside a forest stream at dawn](samples/fox-wildlife-512-q8-slg50.png)

This image was visually inspected after a full 50-step, q8_2f MOSS render on
the named Vulkan reference system. It uses the source negative-branch skip and
standard SLG; 512×512 is the source-supported lower resolution boundary, so
use the 1–2 MP guidance above when the target hardware budget permits it.

```bash
PROMPT='best quality, highres, RAW editorial wildlife photograph of an adult red fox standing on a moss-covered stone beside a clear forest stream at dawn, detailed natural fur, amber eyes, soft mist, autumn ferns, golden backlight, 85mm lens, shallow depth of field, realistic cinematic color grading'
NEGATIVE='nsfw, worst quality, bad anatomy, watermark, jpeg artifact'

WGPU_BACKEND=vulkan CMF_GPU=wgpu cortiq imagine moss-image-base-q8_2f.cmf \
  --prompt "$PROMPT" --negative-prompt "$NEGATIVE" \
  --width 512 --height 512 --steps 50 --cfg 3 --shift 5 \
  --slg-layers 24 --slg-scale 3 --slg-start 0.01 --slg-stop 1 \
  --seed 20261023 --out samples/fox-wildlife-512-q8-slg50.png
```

PNG SHA-256:
`f79d807e8b892bcf0c6def4b1c1f4060075d7236444ea4c930b6d4a2b6a317b5`.
The render completed 50 actual scheduler steps; it is an inspected quality
example, not a claim of PyTorch bit parity or a generic throughput benchmark.

## Validation and measured baselines

Both final files passed `cortiq verify`, an external SHA-256 check, a Vulkan
CFG render, and source-pinned template/Flux2-decoder contract tests. The
contract tests cover the stored template token contract and the BatchNorm →
2×2 unpatch → post-quant-convolution → decoder path on a fixed source-oracle
latent. They validate critical component contracts, but are not a full
end-to-end numerical-parity oracle for the upstream DiT.

### Vulkan baseline

Measured on an NVIDIA GeForce RTX 3090 (24 GiB), NVIDIA Vulkan driver
595.91.07, with Cortiq v0.8.30: 512 × 512, 8 steps, CFG 3, shift 5, seed 42.
`median DiT step` includes the two DiT passes required by CFG. Text preparation,
DiT loading, and VAE decoding are reported separately by the CLI.

| File | Total | Median DiT step | Text / DiT load / VAE |
|---|---:|---:|---:|
| q8_2f | 348.8 s | 41.66 s | 2.6 s / 0.3 s / 6.7 s |
| q4tp | 333.7 s | 39.80 s | 2.7 s / 0.3 s / 7.1 s |

This is a named, reproducible baseline, not a general performance guarantee
for other drivers, hardware, resolution, precision profiles, or settings.

### Metal smoke

A functional Metal smoke was run on Apple M4 / 24 GB: native DiT plus Flux2
VAE, 256 × 256, one step, CFG 1. The VAE recorded 178.5 ms GPU time / 0.183 s
wall time. Its CPU decoder comparison was PSNR 70.13 dB with a maximum 1-LSB
RGB difference. This isolates decoder correctness/Metal execution; it is **not**
an end-to-end throughput benchmark and must not be compared to the Vulkan
table.

## Compatibility and scope

This publication supports the Base **text-to-image** path, including
natural-language and tag-sequence prompts; serialized JSON-layout prompts
require a v0.8.31+ build exposing the layout flags described above.
Its scope ends there. It does not claim support for:

- image input, img2img, or pixel-space image editing;
- video or reference-image workflows;
- LoRA adapter loading or training;
- the upstream Gradio application, ComfyUI integration, prompt-enhancer UI,
  CPU-offload/compilation options, or other Python-only surfaces; or
- bit-exact numerical parity with the upstream PyTorch implementation.

Those upstream tools and integrations may expose additional capabilities, but
they are not CMF CLI feature parity and should not be inferred from this Base
artifact.

## License and provenance

The upstream MOSS-Image-Base release is licensed under
[Apache-2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/LICENSE).
These CMF artifacts are quantized/repackaged derivatives and retain upstream
license and attribution. Consult the upstream repository and model card for
model behavior, dataset disclosures, and any use restrictions.
