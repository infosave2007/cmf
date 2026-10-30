# CMF vs. other model formats

This is an honest, factual comparison of **CMF** (Cortiq Model Format,
see [CMF_V2_SPEC.md](CMF_V2_SPEC.md)) against the model container and
engine formats it is most often weighed against:

- **GGUF** — the single-file container used by `llama.cpp`/`ggml`.
- **safetensors** — Hugging Face's zero-copy tensor container.
- **ONNX** — the cross-framework graph + weights interchange format.
- **PyTorch `.pt` / `.pth`** — pickled state dicts / checkpoints.
- **MLX** — Apple's MLX layout: safetensors shards plus `config.json` with
  group-wise quantization.
- **EXL3** — the ExLlamaV3 quantization format (a QTIP-style trellis codec):
  safetensors plus a quantization config, CUDA only.
- **GGML (legacy)** — the pre-GGUF `ggml` file format.
- **TensorRT engines** — NVIDIA's serialized, hardware-specific plan files.

A recurring caveat: these are not all the same *kind* of thing. GGUF,
safetensors, ONNX, `.pt`, and GGML are primarily **storage/interchange
formats**; TensorRT engines are a **compiled runtime artifact**; CMF is
a storage format that also ships a reference CPU and GPU runtime. The table
compares them on the axes people actually ask about, but read the prose
for the shape each one is really meant for.

## Score out of 100

Nine criteria, each scored 0–100 per format and then weighted. The weights are
ours and printed, so you can recompute the totals for your own needs.

What is scored:

- The score covers the file format and its reference runtime.
- Ecosystem size (how many models are already published, how many third-party
  tools exist) is not a property of the file, so it is not scored.
- GGML is superseded by GGUF, and a TensorRT engine is compiled for one GPU and
  one TensorRT version, so neither is a distributable model file. Both stay in
  the feature table below.

| criterion | weight | CMF | GGUF | safetensors | ONNX | MLX | EXL3 | PyTorch `.pt` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Quantization range**: encodings the format itself defines, from 16 bits to below 1 bit | 15 | **100** | 90 | 30 | 50 | 60 | 70 | 25 |
| **Mixed codecs in one file**: any codec on any tensor, re-quantization in place | 10 | **100** | 90 | 50 | 60 | 70 | 75 | 30 |
| **Empty weights take no space**: pruned neurons and experts are not stored, every layer keeps its own width | 15 | **100** | 45 | 30 | 60 | 25 | 25 | 30 |
| **Flexibility, many tasks on one base**: task masks, skills and expert routing inside one file | 15 | **100** | 40 | 35 | 20 | 30 | 20 | 35 |
| **Self-contained file**: tokenizer, chat template, multi-component pipelines | 10 | **100** | 80 | 30 | 45 | 30 | 25 | 25 |
| **Integrity and authenticity**: content hashes, verification, signature | 10 | **100** | 30 | 55 | 35 | 50 | 50 | 10 |
| **Zero-copy, lazy loading**: mmap, the GPU reads the mapped file, experts on demand | 10 | **100** | 90 | 85 | 40 | 80 | 60 | 40 |
| **Safe evolution**: an old reader refuses what it cannot run | 5 | **100** | 70 | 60 | 85 | 50 | 40 | 40 |
| **Standalone runtime**: no Python or ML framework, CPU and GPU | 10 | 90 | **100** | 30 | 70 | 50 | 15 | 0 |
| **Weighted total** | **100** | **99** | 69 | 42 | 49 | 48 | 42 | 26 |

### How the scores were assigned

**Quantization range** (encodings the format itself defines):

| format | score | reason |
|---|---:|---|
| CMF | 100 | 17 encodings, listed below. It is the only format here with a variable-bit codec, and the only one that goes below 1 bit/weight. |
| GGUF | 90 | A broad ladder: Q8_0 down to IQ1_S, ternary TQ1_0/TQ2_0, and MXFP4. Every type has a fixed width, and none goes below 1.5 bits/weight. |
| EXL3 | 70 | Any average bitrate, but from one trellis codec family. |
| MLX | 60 | Group-wise affine 2–8 bit, plus MXFP4/NVFP4 modes. |
| ONNX | 50 | int8, int4 and float8 types, and QDQ operators. |
| safetensors | 30 | Stores plain dtypes only. GPTQ, AWQ and bitsandbytes layouts are conventions read from `config.json`, not part of the format. |
| PyTorch | 25 | `qint8`, `quint8` and `quint4x2` tensors. |

CMF's 17 encodings:

- Float: f32, f16, bf16.
- 8-bit: `q8_row`, `q8_2f`.
- 4-bit: `q4_block`, `q4t`, `q4tp`, `mxfp4`.
- Variable 3–8 bit: `vbit`, `vbit_ro`.
- 2-bit: `q2tp`.
- Ternary: `q1t`.
- Binary: `q1`, `q1g`, `q1s`.
- Below 1 bit/weight: `blr`.

**Mixed codecs in one file:**

| format | score | reason |
|---|---:|---|
| CMF | 100 | Each tensor has its own dtype. `--tensor-quant PATTERN=CODEC` sets it at conversion, and `cortiq requant` changes it in an existing file. |
| GGUF | 90 | Per-tensor types, chosen at quantization time with `llama-quantize --tensor-type`. |
| EXL3 | 75 | The quantizer allocates bits per layer. |
| MLX | 70 | Per-layer bits through quantization predicates. |
| ONNX | 60 | Quantization per graph node. |
| safetensors | 50 | Per-tensor dtypes, but by convention one quantization config per model. |
| PyTorch | 30 | Per-tensor dtypes inside the pickle. |

**Empty weights take no space:**

| format | score | reason |
|---|---:|---|
| CMF | 100 | Pruned FFN neurons and MoE experts are neither stored nor computed. Each layer keeps its own width and expert count, because the directory is the only authority on shapes (spec §11). Unused experts and skills are never paged in. |
| ONNX | 60 | The graph carries exact shapes, so a pruned network can be represented. There are no task masks and no per-task subset. |
| GGUF | 45 | Per-layer FFN and expert sizes exist as metadata arrays, but only for architectures llama.cpp hard-codes (the DeciLM/Nemotron-NAS class). Everything else uses one global size. |
| safetensors | 30 | Any shape can be stored, but model code reads one `intermediate_size` from `config.json`. |
| PyTorch | 30 | Same as safetensors. |
| MLX | 25 | Same, with no per-layer override in the config. |
| EXL3 | 25 | Same as MLX. |

**Flexibility, many tasks on one base:**

| format | score | reason |
|---|---:|---|
| CMF | 100 | Task masks cost 1 bit per neuron, head, layer and expert. Skills are full-shape replacement tensors in the same file and cost no memory until used. A skill can also ship as a standalone file bound to its base by hash. |
| GGUF | 40 | LoRA adapters and control vectors, as separate files. |
| safetensors | 35 | PEFT adapters, as separate files. |
| PyTorch | 35 | PEFT adapters, as separate files. |
| MLX | 30 | LoRA adapters, as separate files. |
| ONNX | 20 | No task or adapter structure in the file. Where adapters exist, they are a runtime feature. |
| EXL3 | 20 | Same as ONNX. |

**Self-contained file:**

| format | score | reason |
|---|---:|---|
| CMF | 100 | Weights, tokenizer and chat template are in one file. So are multi-component pipelines: image editing (transformer, text encoder, VAE, scheduler), video and speech. |
| GGUF | 80 | Weights, tokenizer and chat template are in one file. Vision and audio encoders ship as a second `mmproj` file. |
| ONNX | 45 | Graph and weights in one file, but models over 2 GB spill weights into external data files. The tokenizer is included only through onnxruntime-extensions. |
| safetensors | 30 | Weights only. Config, tokenizer and chat template are sidecar files. |
| MLX | 30 | Same as safetensors. |
| EXL3 | 25 | Weights only, plus a quantization config sidecar. |
| PyTorch | 25 | Weights only, and loading needs the model class. |

**Integrity and authenticity:**

| format | score | reason |
|---|---:|---|
| CMF | 100 | Mandatory `hash64` for every tensor and section. Header and directory hashes sit in the envelope. `cortiq verify` checks the whole chain. `cortiq sign` adds an optional Ed25519 signature. |
| safetensors | 55 | The JSON header is bounded and nothing executes on load. There are no content hashes. |
| MLX | 50 | Built on safetensors. |
| EXL3 | 50 | Built on safetensors. |
| ONNX | 35 | Protobuf, with no content hashes. |
| GGUF | 30 | Magic and version only. Nothing in the format detects corruption. |
| PyTorch | 10 | Pickle. `weights_only=True` narrows what loading can execute, but nothing checks the content. |

**Zero-copy, lazy loading:**

| format | score | reason |
|---|---:|---|
| CMF | 100 | The weight blob is page-aligned and tensors are 64-byte aligned, all memory-mapped. On Apple silicon the GPU reads the mapping without a copy. MoE experts are paged in, and cached on the GPU, when the router picks them. |
| GGUF | 90 | Memory-mapped aligned tensors, zero-copy on Apple silicon. |
| safetensors | 85 | Memory-mapped, with per-tensor lazy reads. |
| MLX | 80 | Lazy safetensors loading into unified memory. |
| EXL3 | 60 | Weights are loaded into CUDA memory. |
| ONNX | 40 | The protobuf is parsed. |
| PyTorch | 40 | A zip archive, with an optional `mmap=True`. |

**Safe evolution:**

| format | score | reason |
|---|---:|---|
| CMF | 100 | Every capability that changes execution has a required-feature bit. A reader refuses unknown bits, and checks "bit ⇔ content" both ways. |
| ONNX | 85 | Per-operator opset versioning. |
| GGUF | 70 | A versioned file with open key-value metadata. Unknown tensor types fail to load, but meaning added through new keys is invisible to an old reader. |
| safetensors | 60 | Minimal and stable, but quantization semantics live outside the file. |
| MLX | 50 | Semantics live in code and sidecar configs. |
| EXL3 | 40 | Same as MLX. |
| PyTorch | 40 | Same as MLX. |

**Standalone runtime:**

| format | score | reason |
|---|---:|---|
| GGUF | 100 | llama.cpp is C/C++ and runs on CPU, CUDA, ROCm, Metal, Vulkan and SYCL. |
| CMF | 90 | Rust, with no Python. Runs on CPU, Metal, Vulkan and DX12, plus an independent stdlib + numpy Python reader. There is no CUDA- or ROCm-specific path. |
| ONNX | 70 | ONNX Runtime: C++, with many execution providers, but an ML runtime of its own. |
| MLX | 50 | The MLX framework, Apple silicon first. |
| safetensors | 30 | A framework is needed to interpret the model. |
| EXL3 | 15 | Python, PyTorch and CUDA. |
| PyTorch | 0 | PyTorch itself. |

## Feature table

Legend: **Yes** = supported by the format itself; **Partial** = possible
but conditional, external, or incomplete; **No** = not provided by the
format; **N/A** = not applicable to this kind of artifact.

| Capability | CMF | GGUF | safetensors | ONNX | PyTorch `.pt`/`.pth` | GGML (legacy) | TensorRT engine |
|---|---|---|---|---|---|---|---|
| Single self-describing file | Yes | Yes | Partial | Partial | Partial | Partial | Yes |
| Memory-mappable / zero-copy load | Yes | Yes | Yes | Partial | Partial | Yes | No |
| Quantization built into the format | Yes | Yes | No | Partial | Partial | Yes | Yes |
| Embedded tokenizer | Yes | Yes | No | No | No | Partial | No |
| Embedded chat template | Yes | Yes | No | No | No | No | No |
| Multi-model / adapter / skill overlay from one base | Yes | Partial | No | No | No | No | Partial |
| Runs without an ML framework | Yes | Yes | N/A | Partial | No | Yes | No |
| GPU-accelerated reference runtime | Yes | Yes | N/A | Yes | Yes | Partial | Yes |
| Streaming / partial load | Partial | Partial | Yes | Partial | Partial | Partial | No |
| Integrity hashing | Yes | No | No | No | No | No | Partial |
| Primary language / runtime | Rust (+ numpy Python reader) | C/C++ (llama.cpp) | Rust lib (framework-agnostic) | Protobuf spec / ONNX Runtime (C++) | Python / PyTorch | C (ggml) | C++ / CUDA (TensorRT) |
| License | Apache-2.0 | MIT | Apache-2.0 | Apache-2.0 | BSD-3-Clause | MIT | Proprietary (NVIDIA) |

### Notes on specific cells (so the marks are defensible)

- **CMF — self-describing / integrity:** the fixed 128-byte envelope,
  binary tensor directory, and per-tensor + per-section `hash64` are
  mandatory; `cortiq verify` checks the whole chain. Tokenizer and chat
  template are *supported and embedded when present* (optional sections),
  not forced into every file.
- **CMF — streaming = Partial:** CMF does lazy `mmap` paging (cold
  weights cost no RSS) and multi-file sharding, but it is not a
  network-streaming/progressive-download protocol. We mark that honestly
  as Partial rather than Yes.
- **CMF — skill overlay:** the one place CMF is genuinely differentiated
  — many skills share one backbone via full-shape replacement tensors
  addressed from a single directory (see spec §9). GGUF's Partial is for
  *external* LoRA adapters applied at load time; TensorRT-LLM's Partial
  is runtime multi-LoRA, not a property of the engine file.
- **CMF — GPU = Yes:** the reference runtime ships two GPU backends —
  native Metal (Apple Silicon) and wgpu (Vulkan / DX12 / Metal → NVIDIA /
  AMD / Intel) — with kernels for every built-in quant (`q8`, `q4`, `q1`,
  and the ternary `q1t`), a whole-token decode graph on Metal, and a
  register-blocked prefill GEMM on both. A runtime probe uses the GPU only
  where it beats the CPU, so it degrades cleanly to the CPU path. Marked
  Partial for GGML because its GPU support was early/limited before GGUF.
- **GGUF / GGML — integrity = No:** the containers carry magic +
  version, but no built-in per-tensor content hash. Corruption is not
  caught by the format itself.
- **safetensors — runtime = N/A:** it is a storage format with a small
  loader library, not an inference runtime; "Runs without an ML framework
  runtime" does not apply.
- **ONNX / PyTorch — single file = Partial:** both routinely spill large
  weights to external data files (ONNX external data; sharded / `zip`
  checkpoints), so "one self-contained file" is not guaranteed.

## Format-by-format

### GGUF (llama.cpp)
GGUF is the closest peer to CMF in intent: one self-describing file with
weights, a rich key/value metadata block, an embedded tokenizer and chat
template, and a wide menu of built-in quantizations (the k-quants and
i-quants). It is battle-tested, `mmap`-friendly, and backed by the
large `llama.cpp` CPU/GPU ecosystem. Its main gaps versus CMF are the
absence of built-in content hashing and of a first-class one-base /
many-skill overlay (LoRA is applied as a separate adapter). Choose GGUF
when you want maximum tooling/hardware reach and a proven local-inference
stack today.

### safetensors
safetensors solves exactly one problem extremely well: safe, fast,
zero-copy tensor storage with no arbitrary-code-execution risk (unlike
pickle). It is the de-facto standard for distributing raw weights in the
Hugging Face ecosystem. It deliberately stores *only* tensors plus a
small JSON metadata header — no architecture, tokenizer, chat template,
or quantization scheme — so a model is really a directory of files
(`config.json`, `tokenizer.json`, one or more `.safetensors`). Choose
safetensors when you want a secure, framework-agnostic weight container
and are happy to carry configuration and tokenizer as sidecar files.

### ONNX
ONNX is a graph-plus-weights interchange format: it captures the
computation, not just the parameters, which makes it strong for
cross-framework portability and for running the same model across many
runtimes and accelerators via ONNX Runtime. It supports quantized
operators (QDQ / int8), but it has no embedded tokenizer or chat
template, is not designed for `mmap` zero-copy, and large models split
weights into external data files. Choose ONNX when your priority is
interoperability across frameworks and deployment targets, or when you
need the full computational graph rather than a fixed LLM architecture.

### PyTorch `.pt` / `.pth`
These are pickled Python objects — usually a `state_dict`, sometimes a
whole model or training checkpoint. They are the native, most flexible
format inside PyTorch and are ideal during training and research. The
downsides for distribution are real: pickle can execute arbitrary code
on load (a genuine security concern), the file is not self-describing
without the model class, there is no embedded tokenizer/template, and
loading requires a full PyTorch install. Choose `.pt`/`.pth` for
training, checkpointing, and PyTorch-internal workflows; convert to a
safer container for shipping.

### GGML (legacy)
GGML was the original single-file format behind `llama.cpp` before GGUF
superseded it. It offered `mmap`, built-in quantization, and an embedded
vocab, and it proved the "one small file, CPU inference" idea. But its
metadata was limited and often architecture-assumed, with no extensible
key/value block, no chat template, and no integrity hashing — which is
precisely why GGUF replaced it. It is listed here for context; new work
should target GGUF (or CMF), not GGML.

### TensorRT engines
A TensorRT engine (`.plan` / `.engine`) is not a portable model file —
it is a compiled, optimized artifact for a specific GPU, TensorRT
version, and often batch/shape profile, with quantization (INT8/FP8)
baked in during the build. Within those constraints it delivers
top-tier NVIDIA GPU throughput and latency, and it guards against
version/hardware mismatch at deserialization. It is opaque, non-portable,
not `mmap`/CPU-oriented, and carries no tokenizer or chat template.
Choose TensorRT engines when you are deploying on NVIDIA GPUs and want
maximum inference performance, and you can rebuild per target
environment.

## When to choose CMF
Choose CMF when you want a single, self-describing, integrity-checked
file for **memory-mapped** inference — on the CPU *or* the GPU (native
Metal on Apple Silicon, wgpu/Vulkan/DX12 on NVIDIA/AMD/Intel) — with the
tokenizer and chat template travelling inside the model, and — the
distinguishing feature — when you want **many task-specialized skills to
share one backbone** in one file (or one sharded set), overlaying
full-shape replacement tensors without assembling a separate model per
skill. Its built-in two-field, variable-bit and training-free ternary
(`q1t`) quantization, mandatory per-tensor and per-section hashing, and
reference runtimes with no ML framework (Rust, plus a stdlib-plus-numpy
Python reader) make it a good fit for distributing and serving compact,
routed, verifiable models on commodity hardware — CPU or GPU. If you instead need the broadest existing tooling
today, pick GGUF; if you need a raw secure weight container for the HF
ecosystem, pick safetensors; if you need cross-framework graph
portability, pick ONNX; and if you need peak NVIDIA-GPU latency, compile
a TensorRT engine. CMF is deliberately narrower than a general graph
format and, unlike GGUF, is newer with a smaller ecosystem — that is the
honest trade for its skill-overlay and integrity guarantees.
