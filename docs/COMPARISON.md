# CMF vs. other model formats

This is an honest, factual comparison of **CMF** (Cortiq Model Format,
see [CMF_V2_SPEC.md](CMF_V2_SPEC.md)) against the model container and
engine formats it is most often weighed against:

- **GGUF** — the single-file container used by `llama.cpp`/`ggml`.
- **safetensors** — Hugging Face's zero-copy tensor container.
- **ONNX** — the cross-framework graph + weights interchange format.
- **PyTorch `.pt` / `.pth`** — pickled state dicts / checkpoints.
- **GGML (legacy)** — the pre-GGUF `ggml` file format.
- **TensorRT engines** — NVIDIA's serialized, hardware-specific plan files.

A recurring caveat: these are not all the same *kind* of thing. GGUF,
safetensors, ONNX, `.pt`, and GGML are primarily **storage/interchange
formats**; TensorRT engines are a **compiled runtime artifact**; CMF is
a storage format that also ships a reference CPU runtime. The table
compares them on the axes people actually ask about, but read the prose
for the shape each one is really meant for.

## Score out of 100

Eight criteria, each scored 0–100 per format, then weighted. The weights are
our choice and they are printed so you can change them: the ranking is a
function of what *you* need, not a property of the formats.

| criterion | weight | CMF | GGUF | safetensors | ONNX | PyTorch | GGML | TensorRT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Self-contained file (weights + tokenizer + chat template) | 15 | **100** | **100** | 40 | 50 | 30 | 60 | 70 |
| Integrity & load safety | 10 | **100** | 20 | 60 | 30 | 0 | 20 | 40 |
| Quantization in the format | 15 | 95 | 95 | 0 | 50 | 30 | 80 | 85 |
| Runs without an ML framework | 10 | **100** | **100** | 20 | 60 | 0 | **100** | 30 |
| Hardware reach of the reference runtime | 10 | 80 | **100** | 50 | 85 | 80 | 60 | 30 |
| Zero-copy mmap load | 10 | **100** | **100** | **100** | 50 | 40 | 90 | 30 |
| Many specialists from one base | 10 | **100** | 50 | 40 | 10 | 50 | 20 | 40 |
| Ecosystem, tooling, model availability | 20 | 15 | **100** | **100** | 85 | 95 | 25 | 60 |
| **Weighted total** | **100** | **80** | **86** | 53 | 56 | 45 | 55 | 52 |
| *the same without the ecosystem row* | | **97** | 83 | 41 | 48 | 33 | 63 | 50 |

**GGUF wins on the total, and it should.** Twenty of the hundred points are
ecosystem, and there CMF scores 15 against GGUF's 100: one author, a first
public release in July 2026, a few dozen published models against thousands.
That is the honest state of it.

**On the properties of the container itself, CMF leads 97 to 83.** Drop the
ecosystem row and what remains is per-tensor integrity hashing that no other
single-file format has, skills that live inside the file under the same hash
chain, and a quantization ladder that reaches ternary and variable-bit.

Both numbers are true. Pick the one that matches your problem: if you need a
model to run tonight on hardware someone else has already tested, that is
GGUF. If you need one auditable file that carries N specialists and proves its
own integrity, that is what CMF is for.

### How the scores were assigned

- **Self-contained** — 100 requires weights, tokenizer and chat template in
  one file with no sidecars. ONNX and PyTorch routinely spill weights to
  external data; safetensors carries tensors only.
- **Integrity & load safety** — CMF's 100 is the mandatory per-tensor and
  per-section `hash64` plus `cortiq verify`. safetensors gets 60 for being
  parse-safe (no pickle execution) without content hashing. PyTorch gets 0:
  `.pt` is pickle, and loading one can execute arbitrary code.
- **Quantization** — CMF and GGUF are level at 95: different ladders, both
  broad and both mixable per tensor. Neither is 100 because both keep adding.
- **Hardware reach** — GGUF's 100 is CUDA, ROCm, Metal, Vulkan, SYCL and CPU.
  CMF's 80 is CPU everywhere plus native Metal and wgpu (Vulkan / DX12); there
  is no CUDA-specific path.
- **Many specialists** — CMF's 100 is skills stored inside the file, addressed
  from the same directory, costing zero RAM when unused. GGUF's 50 is external
  LoRA applied at load. PyTorch's 50 is the PEFT ecosystem, which is large but
  is not a property of the container.
- **Ecosystem** — model availability, third-party tools, how likely it is that
  your problem is already solved by someone else.

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
