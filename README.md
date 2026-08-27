Русский: [README.ru.md](README.ru.md) · 中文: [README.zh.md](README.zh.md)

# CMF — Cortiq Model Format

**One file that carries the weights, the tokenizer and the chat template, checks its own integrity, and runs without an ML framework.**

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![downloads](https://img.shields.io/crates/d/cortiq-cli.svg)](https://crates.io/crates/cortiq-cli)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

No torch, no BLAS, no ONNX, no CUDA install, no C++ toolchain. A small Rust
core, CPU everywhere, GPU via native Metal and wgpu (Vulkan / DX12). Weights are
memory-mapped and read in place. One flag turns a model's attention into a
constant-memory operator, without retraining and without changing a weight.

## Try it

```sh
cargo install cortiq-cli          # or a prebuilt binary from the releases page

cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --output qwen.cmf
cortiq run qwen.cmf --prompt "What is the capital of France?" --greedy --no-think
```

```console
Ready: qwen3 | Task: general | Sparsity: 0%
The capital of France is **Paris**.
[10 tokens, 40.1 tok/s, finish: stop]
```

Already have a GGUF? `cortiq import-gguf <file-or-repo> --output model.cmf`.

Without installing anything: [convert a model](https://huggingface.co/spaces/infosave/cmf-converter) ·
[render an image](https://huggingface.co/spaces/infosave/cmf-imagine) ·
[watch the clips](https://huggingface.co/spaces/infosave/cmf-animate)

## Scored against other formats

Eight criteria, weighted, each format scored 0–100. Full matrix and the
reasoning behind every cell: [docs/COMPARISON.md](docs/COMPARISON.md).

| | CMF | GGUF | safetensors | ONNX | PyTorch | GGML | TensorRT |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Weighted total /100** | **80** | **86** | 53 | 56 | 45 | 55 | 52 |
| *without the ecosystem weight* | **97** | 83 | 41 | 48 | 33 | 63 | 50 |

GGUF wins the total, and it should: 20 of the 100 points are ecosystem, and
there CMF scores 15 against GGUF's 100 — one author, first release July 2026.
On the container's own properties CMF leads 97 to 83, on per-tensor integrity
hashing, in-file specialists and a quantization ladder that reaches ternary.

Both numbers are true. Need a model running tonight on hardware someone has
already tested — that is GGUF. Need one auditable file carrying N specialists
that proves its own integrity — that is this.

## What is in the file

A fixed 128-byte envelope, then sections addressed only through it, never by
assumed order:

| section | what it holds |
|---|---|
| header JSON | arch, quant defaults, chat bundle, skill registry, provenance |
| tensor directory | 56-byte records: name, dtype, shape, offset, nbytes, `hash64` |
| weight blob | page-aligned, mapped and read in place |
| skills | task masks and per-skill replacement tensors |
| tokenizer | the verbatim Hugging Face file |

```sh
cortiq verify model.cmf     # envelope, sections, every tensor hash
cortiq info   model.cmf     # arch, tensors, quantization, skills
cortiq sign   model.cmf     # detached Ed25519 over the SHA-256
```

A `.cmf` is either valid or `open()` fails loudly — it catches truncation and
bit-rot. Also carried: MTP heads, MoE layers, mixed global/sliding attention,
dual RoPE/YaRN, append-only skill growth, and sharding into N standalone-valid
files.

**You are not locked in.** `python/cmf_reader.py` is a complete reader in ~300
lines of stdlib + numpy, written from the spec and sharing no code with the
Rust runtime:

```python
from cmf_reader import CmfReader
r = CmfReader("model.cmf")
w = r.tensor("model.layers.0.mlp.gate_proj.weight")   # np.ndarray, dequantized
assert r.verify() == []                               # every tensor hash checks
```

Normative spec: [docs/CMF_V2_SPEC.md](docs/CMF_V2_SPEC.md).

## Quantization

Per tensor and mixable — keep attention at q8 and push the FFN to q4 in the
same file.

| quant | bits/param | notes |
|---|---|---|
| `f16` | 16 | no quantization |
| `q8` | 8 | per-row scale |
| `q8_2f` | 8 | per-row **and** per-column scale — better quality, same size |
| `q4` · `q4t` | 4.5 | block / interleaved tiles |
| `q4tp` | **4.17** | `q4t` with predicted scales — 7% smaller, +0.1% error |
| `q2tp` | ~2 | 2/4 mixed MoE profile |
| `vbit` | ~4.25 | variable 3–8 bit |
| `q1t` | 2.25–3.5 | training-free ternary + sparse outlier overlay ([docs](docs/Q1T_PTQ.md)) |
| `q1` | 1.5 | for checkpoints **trained** binary (Bonsai / BitNet) |

`q4tp` in one line: a `q4t` tile spends 16 bits on an f16 scale for 32 weights —
11% of the file. Making that scale a 5-bit rung on a per-row geometric ladder
costs **+0.1% relative error** at the median within-row spread. Existing files
convert in place, no checkpoint needed:

```sh
cortiq requant model.cmf --output model-q4tp.cmf --quant q4tp
# KAT-Coder-V2.5: 12.65 → 11.80 GB (19254 tensors) in 2 min
```

## O(1) attention

`--o1` replaces a layer's softmax attention with a fixed-size state: a few exact
anchor keys, an exact recent window, and a landmark sketch of everything older,
under one shared denominator. **Weights never change** — the flag records a
header hint.

Qwen3.5-4B (8 softmax layers converted), Apple M4:

| context | `--o1 off` | `--o1 all` | decode |
|---:|---:|---:|---:|
| 543 | 141.0 MB | **124.1 MB** | 15.7 → 16.5 tok/s |
| 1055 | 174.5 MB | **124.1 MB** | 15.5 → 16.5 tok/s |
| 4127 | 380.3 MB | **124.1 MB** — 3.1× less | 8.2 → 10.7 tok/s |

Constant at every length. The KV it replaces grows ~64 KiB/token, so the curves
cross near 290 tokens: below that `--o1` costs a few MB, above it only saves.

**What it costs.** Perplexity rises **1.13×** on Qwen3.5-4B and **1.30×** on
Qwen3-0.6B (28/28 layers) on held-out wikitext. The more of the model is softmax
attention, the more it costs. A memory/quality dial, not a free win — measure
your own model:

```sh
cortiq ppl model.cmf --file wiki.txt --o1 all   # prints the exact baseline next to it
cortiq run model.cmf --o1 all|deep12|off        # or decide at load time
```

`cortiq fcd` recovers part of the cost with a bounded native training pass, with
no Python and no ML framework.

## Against llama.cpp

Qwen2.5-0.5B-Instruct, Apple M4, exact attention both sides, interleaved runs
from fresh processes, each side at its best thread count. `cortiq bench --core`
matches `llama-bench`'s contract.

| | `llama.cpp` (q8_0) | CMF (q8) | Δ |
|---|---|---|---|
| tg128, CPU, their best `-t 6` | 165.5 tok/s | 151–158 | −5% |
| tg128, CPU, their default `-t 4` | 129.4 tok/s | 151–158 | **+18%** |
| tg128, their Metal `-ngl 99` | 150.9 tok/s | 151–158 (CPU) | **CMF CPU ≥ their GPU** |
| pp512, CPU | 1168 tok/s | 1017–1051 | −12% |
| pp512, GPU | 3333–3396 | 2742–3215 | −5% best-vs-best |
| PPL vs own f16 | near-lossless | +0.38% | matched |
| File size | 644 MB | **479 MB** | **−26%** |

Reproduce with `cortiq bench --json --core`.

## Many specialists, one backbone

Shipping N fine-tunes normally means N full copies. CMF keeps one backbone plus
one small skill each: a skill stores only the tensors it replaces, the runtime
reads those in place of the backbone's, and an unused skill costs **zero RAM**.
Storage is `|backbone| + Σ|skills|`, not `N × |model|`.

On its own task a skill cuts perplexity by **24.9%** against the backbone it
sits on (held-out, [spec §9](docs/CMF_V2_SPEC.md)).

```sh
cortiq skill add ...                        # bake from a donor checkpoint
cortiq run model.cmf --prompt "SELECT ..." --skill sql
cortiq route model.cmf --prompt "..."       # or let it pick, and `explain` why
```

Three real skills from public fine-tunes baked into one 0.5B file, with the
failure modes: [docs/SKILLS.md](docs/SKILLS.md).

**MoE specialists.** Expert usage is strongly task-conditional — code and prose
route to near-disjoint sets (top-64 Jaccard 0.25). `cortiq moe-defrag` physically
drops the experts a task never uses: a 34.7B coder goes **19.6 → 12.7 GB (−35%)**
at +2.8% code perplexity, and on a 24 GB MacBook where the full model paged, the
specialist fits and decodes **1.8× faster**. `cortiq moe-mask` bakes the same
restriction as a switchable task mask instead — one file, `run --task coder`,
token-identical to the physical cut. [docs/KAT_CODER.md](docs/KAT_CODER.md)

## Speculative decode

A model that ships an MTP head drafts with it and verifies the chain in one
batched submit. **On by default for greedy decoding of q4tp files.** A monitor
compares tokens-per-round against the plain token and stops speculation after
four losing rounds, retrying later — so a prompt that gains keeps the gain and
one that does not sits at the plain rate.

- Qwen3.8-27B q4tp, RTX 5090: **76 tok/s against a plain 48.5**, 90% of drafts accepted
- The same on an M4 mini (24 GB): plain 6.7, a code body **12.2**; 447-token prompt **11.6 s** to first token
- `CMF_VERIFY_I8=0` makes the stream bit-identical to the plain path; `CMF_GRAPH_SPEC=0` turns it off

That decode is **bus-bound**: two processes on one card aggregate 52.8 tok/s
against a single process's 48.8, and the matvec stripped of its arithmetic runs
the same token to within 4%. The levers are reading fewer bytes or amortizing
the read — not a faster kernel. [docs/GPU_KERNEL_RECIPES.md](docs/GPU_KERNEL_RECIPES.md)

## It renders, too

Same container, same binary, no Python at inference.

| `cortiq animate` — video **and its soundtrack** | `--first-frame` — continue from a picture |
|---|---|
| ![a corgi in a chef hat flipping a pancake](docs/media/corgi.gif) | ![the same clip continued from a still frame](docs/media/keyframe.gif) |

The audio is not dubbed on afterwards: it is denoised in the same packed
sequence as the video, on its own flow schedule, so it arrives in sync.

- **`cortiq animate`** — MiniMax-H3 + Turbo LoRA. 512×288, 39 frames, four steps, out of one 23.9 GB file (124.4 GB of reference tree). One RTX 5090: **60.2 s**.
- **`cortiq ltx-video`** — LTX-2.5, a 21B DiT with a joint audio stream. Eight steps, or `--two-stage` for detail. `--lora` applies an adapter at runtime, `--ref` conditions on up to five reference stills. [docs/LTX.md](docs/LTX.md)
- **`cortiq imagine`** — Lumina-Image 2.0, a 19 GB diffusers tree in a **3.2 GB** `.cmf`. M4: 256×256/30 steps in ~37 s.

<img src="docs/media/fox-512.png" width="340" alt="a red fox in a snowy forest">

## On a phone

The same format on Android and iOS: **[Cortiq: Local AI Models](https://play.google.com/store/apps/details?id=ai.cortiq.cmf_mobile)**
carries this runtime as a native library — chat against a `.cmf` on the device,
convert a Hugging Face repo to CMF on the handset itself, and serve what is
loaded to your LAN over the same OpenAI-compatible API.

- What it does — **[huggingface.co/spaces/infosave/cortiq-mobile](https://huggingface.co/spaces/infosave/cortiq-mobile)**
- Source — [github.com/infosave2007/cmfmobile](https://github.com/infosave2007/cmfmobile) (Apache-2.0)

Paired with `cortiq worker` on a desktop the phone runs a model larger than its
own memory: a 34.7B MoE at **16.3 tok/s** on a handset with 2 GB free.

## Serve

`cortiq serve` speaks the OpenAI API, so existing clients work unchanged.

```sh
cortiq serve model.cmf --port 8080     # + a web dashboard on /
```

`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/healthz`, streaming.
Scope it honestly: **requests are serialized** and **there is no
authentication** — local-first, not a multi-tenant gateway.

## Commands

| command | what it does |
|---|---|
| `convert --model <hf-repo\|dir>` | HF checkpoint → `.cmf` (native Rust) |
| `import-gguf <file\|hf-repo>` | GGUF → `.cmf`, every common ggml quant |
| `run` · `serve` | chat / one-shot; OpenAI-compatible server |
| `info` · `verify` · `masks` · `diff` | inspect, check integrity, compare versions |
| `bench` · `ppl` | tok/s and memory; teacher-forced perplexity |
| `requant` · `compact` | change quantization in place; tighten the container |
| `skill add` · `list` · `route` · `explain` | bake, list and route specialists |
| `moe-defrag` · `moe-mask` | physically drop unused experts, or make them switchable |
| `fcd` | restoration trainer for `--o1` models |
| `sign` | detached Ed25519 signature |
| `imagine` · `animate` · `ltx-video` | image, video-with-sound, LTX-2.5 |
| `imagine-pack` · `animate-pack` · `ltx-pack` | pack a reference tree into one `.cmf` |
| `worker` · `peers` | serve layers to another machine; find one on the LAN |

`cortiq <command> --help` documents every flag.

## GPU

```sh
CMF_GPU=1 cortiq run model.cmf
```

The backend is picked automatically — Vulkan on Linux/Windows, DX12 as a
fallback, Metal on macOS. **Enabling the GPU never makes you slower**: for each
op class the engine measures both arms at startup and keeps the faster one
(`CMF_GPU_PROBE=0` trusts the device unconditionally).

On Metal and on discrete cards alike, a whole token executes as one graph with
one readback: hidden state stays on the device across every layer, attention
attends there, and routed MoE runs its router, top-k and every selected expert
in the same submit. Measured:

| | |
|---|---|
| Bonsai-27B q1, RTX 4090 | **40 tok/s** (7.7× its CPU path), 38 at ctx 4K |
| KAT-Coder 34.7B-A3B, RTX 5090 | **32.8 tok/s** vs 14.4 on its 32-core host |
| Nanbeige4.2-3B, fanless MacBook Air M4 | **22.4 tok/s**, prompt ingest 181 tok/s |

Output is distribution-equivalent to the CPU path, not bit-identical —
floating-point reductions run in a different order, as with any GPU offload.

**More than one GPU** — two flags for two problems. `serve --gpus N` puts a full
replica on each card (2×RTX 5090, 34.7B MoE: 115.3 tok/s for one request,
**218.5 aggregate** for two). `run --gpus N` splits the layer stack across cards
for models bigger than one card — it buys room, not speed, and costs a few
percent on a model that already fits. `--peer` does the same split over the
network. [docs/MULTI_GPU.md](docs/MULTI_GPU.md)

## Does it run your model?

Native conversion: qwen2 · qwen3 · qwen3.5 (incl. fused qwen3_next) · llama ·
mistral · qwen-moe · gemma / gemma-2 / gemma-3 · gemma-4 dense 12B/31B and MoE
26B-A4B · gemma-3n E4B · phi-3 / phi-4 · DeepSeek-R1 distills · DeepSeek-V2 MLA ·
Kimi Linear 48B-A3B · MiniCPM3 · MXFP4-packed checkpoints. Not yet: the
Kimi-K3-only extras, until the modeling code is public.

Anything else — try `import-gguf`. If it refuses, that is a bug worth filing.

## Install and build

```sh
cargo install cortiq-cli                 # the CLI
cargo add cortiq-core                    # or the format from your own Rust code

cargo build --release --workspace                            # wgpu backend on by default
cargo build --release -p cortiq-cli --no-default-features    # CPU-only
```

Prebuilt binaries for Linux x86-64, macOS (Apple Silicon and Intel), Windows
(x86-64 and ARM64) and `aarch64-linux-android`, each with a `.sha256`:
[latest release](https://github.com/infosave2007/cmf/releases/latest).

```
crates/cortiq-core     format reader: envelope, directory, quant, masks, mmap
crates/cortiq-engine   portable CPU/GPU runtime, tokenizer, chat, skills
crates/cortiq-server   OpenAI-compatible HTTP serving
crates/cortiq-cli      the `cortiq` command
python/                reference reader — stdlib plus numpy, nothing else
docs/                  specification, comparison, model walkthroughs
```

## Status

**The format is the settled part.** It is v2: readers navigate only through the
envelope, unknown header fields are ignored, and a breaking change costs a
feature bit or a version bump — never a silent reinterpretation. A `.cmf`
written today stays readable, and `cortiq verify` is the contract.

The crate APIs may still move before 1.0. First public release July 2026, one
author. Every change is in [CHANGELOG.md](CHANGELOG.md).

Bugs and features: [open an issue](https://github.com/infosave2007/cmf/issues).
Security: **do not** open a public issue — see [SECURITY.md](SECURITY.md).
A model that won't convert is a bug report, not a user error.

> **Hub integration.** The upstream `.cmf` download-counting and Cortiq
> snippet change was merged on August 26, 2026:
> [huggingface.js#2354](https://github.com/huggingface/huggingface.js/pull/2354).
> It is present in the published `@huggingface/tasks` package, but the Hub's
> production UI and statistics workers have not rolled it out yet: CMF repos
> still show `0` downloads and no “Use with Cortiq” entry as of August 27.

## License

**Apache-2.0** ([LICENSE](LICENSE)) — use it, modify it, ship it commercially.

This software practices methods claimed in four pending US patent applications
([PATENTS.md](PATENTS.md)). Apache-2.0 §3 grants you a perpetual, worldwide,
royalty-free license to the claims necessarily infringed by this software as
distributed: **running, forking and shipping it is covered**, and the grant
lapses only if you sue the project over patents. That grant is scoped to this
Work; for an independent reimplementation of the container, email
urevich55@gmail.com — an implementer's grant is available.

Design origins, with a hard line between what is measured and what stays a
metaphor: [VMF/NVG principles behind CMF](VMF_principles_in_CMF.md)
([Русский](VMF_principles_in_CMF.ru.md) · [中文](VMF_principles_in_CMF.zh.md)).
