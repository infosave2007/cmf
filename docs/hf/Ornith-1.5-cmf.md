---
library_name: cortiq
pipeline_tag: text-generation
license: apache-2.0
base_model:
  - ornith-ai/Ornith-1.5-9B
  - ornith-ai/Ornith-1.5-35B-A3B
tags:
  - cmf
  - cortiq
  - quantized
language:
  - en
  - ru
base_model_relation: quantized
---

# Ornith 1.5 → CMF — one file, one Rust binary, no Python

```bash
cargo install cortiq-cli
hf download infosave/Ornith-1.5-cmf ornith-1.5-9b-q4tp.cmf --local-dir .
cortiq run ornith-1.5-9b-q4tp.cmf --prompt "Explain quicksort in three sentences."
```

[Ornith 1.5](https://huggingface.co/ornith-ai) is a Qwen3.5-family pair: a
9B dense hybrid and a 35B mixture with 3B active per token, both mostly
linear-attention layers with full attention every fourth. These are the text
towers converted to the [CMF container](https://github.com/infosave2007/cmf)
— a single memory-mapped file read by `cortiq`, a Rust binary with no ML
framework under it. GPU via Vulkan/Metal/DX12 with a CPU fallback.

| file | params | layers | size |
|---|---|---|---|
| `ornith-1.5-9b-q4tp.cmf` | 9B | 32 (8 attention / 24 linear) | **4.37 GB** |
| `ornith-1.5-35b-a3b-q4tp.cmf` | 35B total, 3B active | 40 (10 attention / 30 linear), 256 experts, 8 per token + shared | **17.4 GB** |

Both 4-bit tiled with ladder scales (`q4tp`), quantized straight from the
bf16 checkpoints, MTP draft head included. The upstream repos are
vision-language; these files carry the text tower — image input is not
part of the container.

## Speed

Steady-state decode, `cortiq bench --core`, single stream, A100 80GB over
Vulkan, cortiq 0.5.99. The whole token runs as one submitted graph —
linear-attention state, expert routing and the draft head included.

| | A100 80GB (Vulkan) | Apple M4 (Metal) |
|---|---:|---:|
| 9B | **73 tok/s** | 20 tok/s |
| 35B-A3B | **65 tok/s** (first token pays a one-off ~10 s expert upload) | — |

The 9B on the M4 was verified against the published file itself —
downloaded back from this repo, checksum matched, generation coherent.

For scale: this stack decodes the same class of files at ~55–65% of an
RTX 5090's rate, so consumer-card numbers run higher.

## Server and API

```bash
cortiq serve ornith-1.5-9b-q4tp.cmf --port 8080
```

Speaks the OpenAI API, so anything that talks to OpenAI talks to it.

## Checksums

Each `.cmf` ships a `.sha256` beside it.
