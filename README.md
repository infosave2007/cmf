Русский: [README.ru.md](README.ru.md) · 中文: [README.zh.md](README.zh.md)

# CMF — Cortiq Model Format

**A single-file LLM format whose attention memory stops growing with the context.**

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![downloads](https://img.shields.io/crates/d/cortiq-cli.svg)](https://crates.io/crates/cortiq-cli)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/infosave2007/cmf/blob/master/LICENSE)

A `.cmf` file carries the weights, the tokenizer and the chat template together,
checks its own integrity, and memory-maps straight off disk. The runtime is a
small Rust core with no ML framework under it — no torch, no BLAS, no ONNX, no
CUDA install, no C++ toolchain — running on CPU everywhere, and on GPU via wgpu
(Vulkan / DX12 / Metal) in a source build. Converting a model takes one command
and no Python.

What makes it different: **you can convert a model's attention into a
constant-memory streaming operator with one flag** — no retraining, weights
byte-identical — so a long conversation stops costing more memory than a short
one.

## Try it

```sh
# prebuilt binary: github.com/infosave2007/cmf/releases/latest
# or, with a Rust toolchain:
cargo install cortiq-cli

cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --output qwen.cmf
cortiq run qwen.cmf --prompt "What is the capital of France?" --greedy --no-think
```

```console
Loading model: qwen.cmf
Ready: qwen3 | Task: general | Sparsity: 0%

Prompt: What is the capital of France?

The capital of France is **Paris**.
[10 tokens, 40.1 tok/s, finish: stop]
```

`convert` pulls the checkpoint from Hugging Face (shards in parallel), quantizes
it and writes one self-contained file — native Rust, no torch, no numpy. Already
have a GGUF? `cortiq import-gguf <file-or-repo-id> --output model.cmf` reads it
natively too.

`run` applies the chat template stored in the file, so this is a real chat turn
and the model stops on its own. Qwen3 is a reasoning model — drop `--no-think`
and it shows its `<think>` reasoning first. `--raw` skips the template entirely
(completion mode). `Task` and `Sparsity` report the skill overlay; with no skill
selected they read `general` / `0%` — more on [skills](#many-specialists-one-backbone) below.

**Does it run your model?** Native conversion today: qwen2 · qwen3 · qwen3.5
(including the fused qwen3_next / AgentWorld layout) · llama · mistral ·
qwen-moe — dense, MoE and GatedDeltaNet. Not yet: gemma, phi, deepseek. Anything
else, try `import-gguf` — and if it refuses, that is a bug worth filing.

## Plug it into what you already use

`cortiq serve` speaks the OpenAI API, so existing clients and SDKs work unchanged
— just point them at it:

```sh
cortiq serve qwen.cmf --port 8080        # + a web dashboard on /
```

```sh
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "cmf",
  "messages": [{"role": "user", "content": "Explain mmap in one sentence."}]
}'
```

`/v1/models`, `/v1/completions` and `/healthz` are there too, and streaming
(`"stream": true`) works. The `model` field is required by the schema but is not
matched against anything — send whatever your client sends.

Scope it honestly before you deploy: **requests are serialized** (one at a time
per model) and **there is no authentication** — this is a local-first server, not
a multi-tenant gateway. Don't expose it to a network you don't trust.

## Why CMF

### Attention that stops growing with the context

Normally every token you add to a conversation adds to the KV cache, forever.
`--o1` replaces a layer's softmax attention with a streaming operator that keeps
a **fixed-size state** instead: a few exact anchor keys, an exact recent window,
and a landmark sketch of everything older, all under one shared softmax
denominator. Conversion is instant and **the weights never change** — the flag
only records a hint in the header.

Measured on **Qwen3.5-4B** (24 GatedDeltaNet + 8 softmax layers; `--o1 all`
converts the 8; 16 query heads / 4 KV heads, head_dim 256; q8_2f). Apple M4, the
machine allowed to cool between runs:

| context | attention memory, `--o1 off` | `--o1 all` | decode, `off` → `all` |
|---:|---:|---:|---:|
| 543 | 141.0 MB | **124.1 MB** | 15.7 → 16.5 tok/s |
| 1055 | 174.5 MB | **124.1 MB** | 15.5 → 16.5 tok/s |
| 4127 | 380.3 MB | **124.1 MB** — 3.1× less | 8.2 → 10.7 tok/s |

**124.1 MB at every context length** — that is the whole point. It breaks down as
a constant recurrent-layer floor plus a fixed **18.8 MB** stand-in for the softmax
layers' KV cache. That KV would otherwise grow at ~64 KiB/token, so the two curves
cross at about **290 tokens**: below that, `--o1` costs you a few MB; above it, it
only saves — 3.1× less at 4k, and ~17× at 32k by extrapolation (the state is
constant, so the ratio keeps climbing; we have benchmarked to 4k — run
`cortiq bench model.cmf --ctx 32768` on your own box).

**What it costs.** The sketch is an approximation, and you pay for it in quality:
perplexity rises **1.13×** on Qwen3.5-4B and **1.30×** on Qwen3-0.6B (28/28 layers
converted) — measured on held-out wikitext through the real streaming kernel on
the harshest region (landmarks sealed from a 256-token prefill, scoring only the
drift rows). The more of the model is softmax attention, the more `--o1` costs: a
hybrid has recurrent layers to carry long-range state, a pure-attention model
makes the sketch do all the work. Treat `--o1` as a memory/quality dial, not a
free win. The cost doesn't grow with context — the state doesn't either. Don't
take our word for any of it; measure your own model:

```sh
cortiq ppl model.cmf --file wiki.txt --o1 all
```

It scores the converted model through the real streaming kernel and prints the
exact-attention baseline over the identical tokens next to it, so the ratio is a
like-for-like measurement rather than a claim.

If that cost is too high for your use case, `cortiq fcd` recovers part of it with
a bounded native training pass — see [O(1) in depth](#o1-in-depth). We haven't
published a clean before/after figure for it yet.

To be clear about the axis: `llama.cpp` is the yardstick we measure against,
and today it is ahead on raw single-stream speed. One like-for-like run
(2026-07-16: Qwen2.5-0.5B-Instruct, Apple Silicon M4, exact attention for
both, native arm64 `llama.cpp` master vs CMF 0.3.1, 2+ series each from
fresh processes):

| Apple M4 | `llama.cpp` (q8_0) | CMF (q8) | Δ |
|---|---|---|---|
| pp512, CPU 8 threads | 1156 ± 14 tok/s | 375–383 tok/s | **−67%** |
| tg128, CPU 8 threads | 162.0 ± 1.3 tok/s | 95–101 tok/s | **−38%** |
| pp512, GPU (Metal `-ngl 99` / `CMF_GPU=1`) | 3339 ± 50 tok/s | 323–325 tok/s\* | **−90%** |
| tg128, GPU (Metal `-ngl 99` / `CMF_GPU=1`) | 150.0 ± 0.4 tok/s | 100–101 tok/s\* | **−33%** |
| Quant quality (PPL vs own f16, 12×512 windows) | near-lossless | +0.38% | matched |
| File size | 644 MB | 479 MB | **−26%** |

\* CMF has no on-device graph yet, so `CMF_GPU=1` means: the runtime probe
measures per-op offload against the CPU path and keeps the winner — on
Apple-silicon unified memory that is (correctly) the CPU. Decode therefore
lands at CPU parity, and prefill pays a one-time ~15% probing tax on the
first prompt of a process; later prompts run at full CPU speed.

An earlier version of this table (0.3.0) claimed +70%/+60% over `llama.cpp`;
that run had unknowingly benchmarked an x86-64 build of `llama.cpp` under
Rosetta 2 emulation, which strips its SIMD. We keep the correction visible
because the numbers only mean something if you can trust how they were taken.

The honest gap decomposes cleanly: `llama.cpp` decode reaches ~85% of memory
bandwidth through its repacked interleaved Q8 kernels (ours reach ~53% —
row-repack on load is queued), its CPU prefill rides Apple's AMX units via
Accelerate GEMM, and its Metal path is a full on-device graph, which the
wgpu backend does not have yet. Where CMF stands apart is not the drag race:
the file is 26% smaller at matched quality, attention memory can be O(1)
(`--o1` holds ~16.5 tok/s at contexts where exact attention decays from 15.7
to 8.2), and the whole engine is portable Rust with no BLAS or C++ toolchain.
Reproduce everything with `cortiq bench --json` — it reports tok/s alongside
allocations/token and scheduler dispatches/token.

### One file, nothing on the side

The tokenizer (HF byte-level BPE) and the chat template (Jinja) travel **inside**
the model — GGUF does this too, and it was right to: the file, not your runtime
binary, defines chat behavior, and there are no sidecars to lose or let drift out
of sync. What a `.cmf` adds on top is integrity: a fixed 128-byte envelope plus a
64-bit hash per tensor means a `.cmf` is either valid or `open()` fails loudly. It
detects truncation and bit-rot; it is not a signature.

```sh
cortiq verify model.cmf     # envelope, sections, every tensor hash
cortiq info   model.cmf     # arch, tensors, quantization, skills
```

Weights are memory-mapped and read in place, so startup is instant and unused
weights never touch RAM. Quantization is per tensor and mixable — `q8`
(1 byte/param) · `q8_2f` (int8 with both a per-row and a per-column scale — better
quality at the same byte count) · `q4` (0.5) · `f16` · `vbit` (variable 3–8 bit,
~4.25 avg ≈ 0.53) — so you can keep attention at q8 and push the FFN to q4 in the
same file.

### Many specialists, one backbone

Shipping *N* fine-tunes normally means *N* full copies on disk and in RAM. CMF
keeps **one backbone plus one small skill per specialist**: a skill stores only
the tensors it actually replaces, and at inference the runtime reads those *in
place of* the backbone's — no separate model is ever assembled. Storage is
`|backbone| + Σ|skills|`, not `N × |model|`, and a skill you don't use costs
**zero RAM**.

A skill isn't just cheaper to ship — on its own task it beats the backbone it sits
on: on held-out data, a skill overlay cuts task perplexity by **24.9%**
([spec §9](docs/CMF_V2_SPEC.md)). Skills pay off most where the backbone is
weakest; on domains it already handles well, expect less.

```sh
cortiq run model.cmf --prompt "SELECT ..." --skill sql
```

Don't want to pick by hand? `cortiq route` chooses a skill from the prompt, and
`cortiq explain` shows you why.

Serving *N* task-specialists:

| | N full fine-tunes | base + N external LoRAs | **CMF** |
|---|---|---|---|
| On disk | N × full model | base + N adapters (sidecars) | one backbone + N small skills, **one file** |
| Tokenizer + chat template | per copy / sidecar | embedded if the base is GGUF, else sidecar | **embedded** |
| Per-tensor integrity hash | — | — | **yes** |
| Unused skill in RAM | loaded | 0 with an adapter-paging server (S-LoRA / vLLM); loaded otherwise | **0**, paged on use, no serving stack required |
| Skill ships inside the model file | — | no (separate adapter files) | **yes, under the same hash chain** |

A full format-by-format comparison — GGUF, safetensors, ONNX, PyTorch, GGML,
TensorRT, with the trade-offs spelled out — is in
[docs/COMPARISON.md](docs/COMPARISON.md).

## Install

```sh
cargo install cortiq-cli                 # the `cortiq` command-line tool
cargo add cortiq-core                    # or use the format from your own Rust code
```

Prebuilt binaries are on the [latest release](https://github.com/infosave2007/cmf/releases/latest)
— Linux x86-64, macOS (Apple Silicon and Intel), Windows (x86-64 and ARM64); every
archive ships a `.sha256`. Since 0.3.1 they include the wgpu GPU backend —
set `CMF_GPU=1` to use it (see [GPU](#gpu)).

## Commands

| command | what it does |
|---|---|
| `cortiq convert --model <hf-repo\|dir>` | Hugging Face checkpoint → `.cmf` (native Rust) |
| `cortiq import-gguf <file\|hf-repo>` | GGUF → `.cmf`, every common ggml quant |
| `cortiq run model.cmf` | chat, or `--prompt` for one shot |
| `cortiq serve model.cmf` | OpenAI-compatible HTTP server + dashboard |
| `cortiq info` · `masks` · `verify` | inspect arch, tensors, skills; check integrity |
| `cortiq bench --ctx 4096` | tok/s and memory at a given context |
| `cortiq ppl --file f.txt` | teacher-forced perplexity — the quality gate |
| `cortiq fcd` | restoration trainer for `--o1` models (KL-anchored, generation-gated) |
| `cortiq diff a.cmf b.cmf` | what changed between two model versions |
| `cortiq route` · `explain` | which skill the router picks, and why |

`cortiq <command> --help` documents every flag.

### Converting

```sh
cortiq convert --model Qwen/Qwen2.5-0.5B-Instruct --quant q8    --output model.cmf
cortiq convert --model ./my-hf-checkpoint         --quant q8_2f --output model.cmf
cortiq import-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF --output model.cmf --quant q8
```

GGUF import covers `Q4_0/1`, `Q5_0/1`, `Q8_0`, `Q2_K`…`Q6_K`, `IQ4_NL/XS` and
`BF16`.

The native converter writes **backbones**. The Python tooling in `converter/` is
still what produces the per-skill replacement tensors and task masks described
above, and the GPTQ-calibrated v-bit variant, which needs an activation Hessian.
The weight-only v-bit path is native.

## O(1) in depth

Record the hint at convert time, or decide at load time — the runtime picks the
header hint up automatically:

```sh
# at convert time: all softmax layers, the deepest N, or an explicit list
cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --o1 all    --output model.cmf
cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --o1 deep12 --output model.cmf

# or override at load time, without reconverting
cortiq run   model.cmf --o1 all      # force-convert every softmax layer
cortiq run   model.cmf --o1 off      # back to exact attention
cortiq bench model.cmf --ctx 4096    # memory + tok/s, with and without
CMF_O1=deep6 cortiq serve model.cmf  # env override, same syntax

# tuning (validated defaults: 32 landmarks, window 128, 4 anchor keys)
cortiq run model.cmf --o1 all --o1-m 32 --o1-window 128 --o1-sink 4
```

On hybrid models (e.g. qwen3.5: GatedDeltaNet layers with softmax islands)
`--o1 all` converts just the softmax layers, which makes the whole model's
attention state constant in context length.

**Restoration.** `cortiq fcd` is a bounded native training pass — no Python, no ML
framework — that tunes only the converted layers' norm/FFN tensors against the
same model running exact attention (KL-anchored), and keeps a checkpoint only if
long-context generation stays loop-free:

```sh
cortiq fcd model.cmf --corpus corpus.txt --gen-check --gen-gate --out model.fcd.cmf
# knobs: --steps 300 --eval-every 25 --kl 0.7 --lr 5e-5 --o1 all|deepN|i,j,k
#        --val-corpus val.txt --gate-threshold 0.35 --gate-slack 0.10
```

## The format

A `.cmf` is a fixed 128-byte envelope followed by sections that a reader addresses
**only** through that envelope, never by assuming order:

- **header JSON** — arch, quant defaults, chat bundle, skill registry, provenance
- **tensor directory** — 56-byte binary records (name, dtype, shape, offset, nbytes, hash64), readable without touching the JSON
- **weight blob** — page-aligned, mapped and read in place
- **skills** — bit-packed task masks and per-skill replacement tensors
- **tokenizer** — the verbatim Hugging Face file
- **sparse index** — precomputed

Also supported: multi-token-prediction (MTP) heads, MoE FFN layers, append-only
skill growth with compaction, and sharding a model across `N` standalone-valid
files.

**You are not locked in.** `python/cmf_reader.py` is a complete reader in ~300
lines of stdlib + numpy that shares no code with the Rust runtime — it was written
from the spec, on purpose, to prove the format outlives this implementation:

```python
from cmf_reader import CmfReader
r = CmfReader("model.cmf")
w = r.tensor("model.layers.0.mlp.gate_proj.weight")   # np.ndarray, dequantized
assert r.verify() == []                               # every tensor hash checks
```

If this project disappeared tomorrow, your weights are still readable from the
spec alone. The complete normative specification is in
[docs/CMF_V2_SPEC.md](docs/CMF_V2_SPEC.md).

## Status

CMF is **0.2.x** and young — first public release July 2026, one author. The crate
APIs may still move before 1.0. The **format** is the settled part: it is v2,
readers navigate only through the envelope, unknown header fields are ignored
(additive evolution), and a breaking change costs a feature bit or a `version`
bump — never a silent reinterpretation. A `.cmf` written today stays readable;
`cortiq verify` is the contract. Every change is in [CHANGELOG.md](CHANGELOG.md).

Bugs and feature requests: [open an issue](https://github.com/infosave2007/cmf/issues).
Security problems: **do not** open a public issue — see [SECURITY.md](SECURITY.md).
A model that won't convert is a bug report, not a user error.

## Build from source

```sh
cargo build --release --workspace
cargo build --release --workspace --features gpu   # + wgpu → Vulkan / DX12 / Metal
```

```
crates/
  cortiq-core     format reader: envelope, directory, quant, masks, mmap
  cortiq-engine   portable CPU/GPU inference runtime, tokenizer, chat, skills
  cortiq-server   OpenAI-compatible HTTP serving
  cortiq-cli      the `cortiq` command-line tool
converter/        Python: DTG-MA skills/masks + the GPTQ-calibrated v-bit path
python/           reference reader — stdlib plus numpy, nothing else
docs/             format specification and comparison
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## GPU

```sh
CMF_GPU=1 cortiq run model.cmf
```

The backend is picked automatically: wgpu chooses Vulkan on Linux/Windows,
DX12 on Windows if Vulkan is absent, Metal on macOS — nothing to configure
(`WGPU_BACKEND=vulkan|dx12|metal|gl` overrides). Weights stay in VRAM up to
a budget (`CMF_GPU_VRAM_MB`, default 8192 on discrete cards); layers are made
resident in first-touch order, so the budget behaves like llama.cpp's `-ngl`
without a flag: first N layers on the GPU, the rest on the CPU.

Enabling the GPU never makes you slower. Per-op offload pays a fixed
submit+poll latency that differs by an order of magnitude between driver
stacks, so at startup the engine *measures* instead of guessing: for each op
class (FFN chain, large matvec, prefill GEMM, QKV batch) the first calls
alternate between the GPU and the CPU path, both timed, and the faster arm is
kept for the rest of the run — per machine, per model. Run with
`RUST_LOG=cortiq_engine=info` to see the verdicts; `CMF_GPU_PROBE=0` skips
the probe and trusts the GPU unconditionally.

## License

**Apache-2.0** ([LICENSE](LICENSE)) — use it, modify it, ship it commercially.

This software practices methods claimed in four pending US patent applications by
the author, listed in [PATENTS.md](PATENTS.md). Apache-2.0 Section 3 grants you a
perpetual, worldwide, royalty-free patent license to those applications' claims
that are necessarily infringed by this software as distributed: **running, forking
and shipping this software is covered**, and the grant lapses only if you sue the
project over patents.

That grant is scoped to this Work, as Apache-2.0 §3 always is — it does not by
itself extend to an independent reimplementation of the container. If you want to
implement CMF in another language or embed it in your own runtime, email
urevich55@gmail.com: an implementer's grant is available, and the format is meant
to be implemented widely.

## Where this came from

The design ideas came out of the author's separate work on a physics theory — the
Vacuum Mass Fraction (VMF) within Null-Vector Gravity (NVG): the
shared-backbone-plus-perturbations model, the two-field quantization. Nothing in
the format depends on that theory being right; it stands on the spec and the
numbers above. The mapping, with a hard line drawn between what is *measured* and
what stays a metaphor: [the VMF/NVG principles behind CMF](VMF_principles_in_CMF.md)
([Русский](VMF_principles_in_CMF.ru.md) · [中文](VMF_principles_in_CMF.zh.md)).
The physics itself lives in [its own repository](https://github.com/infosave2007/vmf).
