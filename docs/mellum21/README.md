---
license: apache-2.0
base_model: JetBrains/Mellum2.1-12B-A2.5B-Thinking
base_model_relation: quantized
library_name: cortiq
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - local-inference
  - code-generation
  - moe
  - quantized
  - reasoning
  - thinking
  - vulkan
  - metal
---

# Mellum2.1 12B-A2.5B Thinking · CMF

A ready-to-run CMF conversion of
[JetBrains/Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking)
for local inference with [Cortiq](https://github.com/infosave2007/cmf).
It keeps the upstream tokenizer and chat template in one memory-mapped `.cmf`
file. This is a quantized conversion, not a fine-tune or a new foundation
model.

## At a glance

| | |
|---|---|
| Base checkpoint | Mellum2.1 12B-A2.5B Thinking (Apache-2.0) |
| Architecture | 28-layer MoE; 64 experts, top-8 routing; 12.15B total / 2.5B active parameters |
| Attention | 21 sliding-window layers (1,024 tokens) + 7 full-attention YaRN layers; declared 131,072-token context |
| CMF profile | Expert matrices: Q4TP · always-active attention, embedding and output: Q8_2f · router and norms: F16 |
| Artifact | `mellum2.1-12b-a2.5b-thinking-q4tp.cmf` · 6,884,556,163 bytes (6.41 GiB) |
| Verified with | Cortiq CLI 0.8.9 |
| Artifact SHA-256 | `1734d8c585134547efa6eb60f092fd741135fe0f398fe9dc77ab4b97df6a7175` |

The mixed profile spends precision where every token passes and compresses the
sparse expert bank. It is intentionally not described as lossless: quantization
can change logits and completions.

## Run locally

Install Cortiq, fetch the single file, and verify it before execution:

```bash
cargo install cortiq-cli --locked

REPO='infosave/Mellum2.1-12B-A2.5B-Thinking-CMF'
MODEL='mellum2.1-12b-a2.5b-thinking-q4tp.cmf'
curl --fail --location --output "$MODEL" \
  "https://huggingface.co/${REPO}/resolve/main/${MODEL}"

printf '%s  %s\n' \
  '1734d8c585134547efa6eb60f092fd741135fe0f398fe9dc77ab4b97df6a7175' \
  "$MODEL" | sha256sum --check
cortiq verify "$MODEL"
```

Generate code or an answer with the embedded chat template:

```bash
cortiq run "$MODEL" --greedy --max-tokens 512 \
  --prompt 'Write a small, well-tested Python function that parses an ISO-8601 date.'
```

Mellum is a thinking model. Leave enough output budget for its reasoning and
answer. To request the template's direct-answer mode:

```bash
cortiq run "$MODEL" --no-think --greedy --max-tokens 160 \
  --prompt 'Explain binary search in three sentences.'
```

## Local API

```bash
cortiq serve "$MODEL" --host 127.0.0.1 --port 8080
```

```bash
curl --fail http://127.0.0.1:8080/healthz
curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mellum-cortiq",
    "temperature": 0,
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Implement stable merge sort in Python."}]
  }'
```

The server provides `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/healthz`, and `/v1/cortiq/status`. Keep it on loopback unless an authenticated
reverse proxy protects it.

## Measured CPU performance

Five independent CPU runs of the same artifact on a shared RunPod host
(AMD EPYC 7663, 112 logical CPUs, approximately 251 GiB RAM), using
`CMF_GPU=0 cortiq bench ... --ctx 512 --tokens 256 --core --ignore-eos --json`:

| Context / generation | Prefill, median | Steady decode, median (range) | TTFT, median (range) |
|---:|---:|---:|---:|
| 512 / 256 tokens | 30.81 tok/s | 40.75 tok/s (40.60–41.27) | 16.53 s (15.84–16.65) |

These are CPU-core measurements, not an end-to-end service SLA. The observed
KV state at sequence 767 was 87,965,696 bytes; it is not process RSS.
On this pod the automatic worker policy selected 22 threads; a bounded sweep
of 12, 16, 20, 22, 24, and 28 workers found 22 fastest for steady decode.
The CMF file can also use eligible Metal and Vulkan runtime paths, but this
release intentionally publishes no end-to-end GPU throughput number until it
has a complete, reproducible hardware record. Check the selected adapter with
`cortiq gpu` before treating a run as GPU-backed.

See [BENCHMARKS.md](BENCHMARKS.md) and
[measurements.json](measurements.json) for the command, all five samples, and
scope.

## Provenance and reproducibility

| Input | Pinned value |
|---|---|
| Upstream revision | [`92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking/tree/92ddae9fc7665e9f801d141d2e5a6b2caf2460c4) |
| `config.json` SHA-256 | `f43d018246094dee6dca455b3072766b6acbefd1ac8e7fe664df34710606a541` |
| `tokenizer.json` SHA-256 | `58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a` |
| `chat_template.jinja` SHA-256 | `4593a34d52f3364ac13ce57f4bea5688924e8942da6df8201e411db17e729f48` |
| Conversion | `CMF_ENCODE_THREADS=112 cortiq convert --model /workspace/mellum-cmf/upstream --quant q4tp --output /workspace/mellum-cmf/out/mellum2.1-12b-a2.5b-thinking-q4tp.cmf` |

`cortiq verify` passed after conversion: it validates the CMF envelope,
sections, directory, and all 5,631 tensor hashes. A successful integrity check
does not establish task quality, safety, or byte-for-byte equivalence with the
BF16 checkpoint.

The conversion path has regression coverage for Mellum's dual RoPE schedule,
sliding-attention metadata, top-8 MoE contract, mixed quantization profile, and
a tiny real forward pass. `/healthz`, `/v1/models`, and a local chat completion
also passed. In a fixed five-prompt greedy smoke suite, CMF CPU matched four of
five source BF16 CUDA completions after the upstream end marker was normalized.
That small fixed set is an implementation check—not a general code benchmark or
a substitute for reviewing generated code.

## Limits

- The upstream architecture declares 131,072 tokens; no full-context quality or
  throughput claim is made here.
- This file is intended for local code-oriented generation. Review output before
  running it, especially when it can alter code, infrastructure, or data.
- `cortiq verify` proves file integrity, not capability, harmlessness, or
  suitability for a particular project.

## License and attribution

The original Mellum checkpoint is released by JetBrains under Apache-2.0. This
repository distributes a derived, quantized representation and retains that
attribution. Cortiq and the CMF container implementation are Apache-2.0.
Refer to the [upstream model card](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking)
for the original model's intended use and limitations.
