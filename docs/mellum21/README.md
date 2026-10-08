---
license: apache-2.0
library_name: cortiq
base_model: JetBrains/Mellum2.1-12B-A2.5B-Thinking
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - moe
  - mellum
  - code
  - reasoning
  - thinking
  - vulkan
  - metal
language:
  - en
---

<!--
RELEASE GATE — complete the artifact table and provenance fields from the
actual build before publishing. Do not replace them with estimates, and do not
add a throughput or quality claim without the raw command output described in
BENCHMARKS.md.
-->

# Mellum2.1 Thinking · CMF

A single-file CMF release of
[JetBrains/Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking)
for the Cortiq runtime. It is a conversion and quantization of the upstream
checkpoint, **not** a new fine-tune or a claim that quantization preserves every
output exactly.

CMF stores the model payload, tokenizer and chat metadata in one memory-mapped
file and records structural and per-tensor integrity metadata. Use
`cortiq verify` after every download; verification establishes file integrity,
not model quality or safety.

## Release manifest

| Field | Release value |
|---|---|
| CMF artifact | `mellum2.1-12b-a2.5b-thinking-q4tp.cmf` <!-- FILL: change only if the final artifact name differs. --> |
| Cortiq CLI minimum | **FILL FROM THE RELEASE BUILD** |
| Quantization policy | **FILL FROM THE CONVERSION MANIFEST** |
| File size | **FILL FROM THE UPLOADED FILE** |
| SHA-256 | **FILL FROM THE UPLOADED FILE** |
| CMF directory hash | **FILL FROM `cortiq info` / release record** |
| Source revision | [`92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking/tree/92ddae9fc7665e9f801d141d2e5a6b2caf2460c4) |
| Conversion command | **FILL WITH THE EXACT RELEASE COMMAND** |

The pinned upstream configuration identifies Mellum as a 28-layer MoE language
model with 64 experts and top-8 routing (12B total / 2.5B active parameters),
with a mixture of sliding-window and full-attention layers. The conversion must
preserve that declared architecture, the tokenizer and the upstream chat
template; the release provenance record is the authority for what was actually
included.

## Quick start

Install the current Cortiq CLI, download the published artifact, then verify it
before running it:

```bash
cargo install cortiq-cli --locked

REPO='infosave/Mellum2.1-12B-A2.5B-Thinking-CMF'
MODEL='mellum2.1-12b-a2.5b-thinking-q4tp.cmf'
curl --fail --location --output "$MODEL" \
  "https://huggingface.co/${REPO}/resolve/main/${MODEL}"

cortiq verify "$MODEL"
cortiq run "$MODEL" \
  --prompt 'Write a small, well-tested Python function that parses an ISO-8601 date.' \
  --greedy --max-tokens 512
```

`cortiq run` applies the chat template embedded in the CMF file. Mellum is a
thinking model: keep enough generation budget for a reasoning block plus the
visible answer. For a direct-answer template mode, use `--no-think` after the
release's template-parity check has passed:

```bash
cortiq run "$MODEL" --prompt 'Explain binary search in three sentences.' \
  --no-think --greedy --max-tokens 160
```

## Local API server

The same file can stay loaded behind Cortiq's local OpenAI-compatible server:

```bash
cortiq serve "$MODEL" --host 127.0.0.1 --port 8080
```

```bash
curl --fail http://127.0.0.1:8080/healthz
curl --fail http://127.0.0.1:8080/v1/models
curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mellum2.1-thinking-cmf",
    "temperature": 0,
    "messages": [{"role": "user", "content": "Implement a stable merge sort in Python."}]
  }'
```

The server exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/healthz` and `/v1/cortiq/status`. Bind to loopback unless you intentionally
place authentication and transport security in front of it.

## CPU, Vulkan and Metal

One CMF artifact is used on all backends. Select the path explicitly when
checking a deployment:

```bash
# Portable CPU reference
CMF_GPU=0 cortiq run "$MODEL" --prompt 'Return the sum of 21 and 21.' --greedy

# Linux with a working Vulkan/wgpu adapter
CMF_GPU=wgpu XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" \
  cortiq run "$MODEL" --prompt 'Return the sum of 21 and 21.' --greedy

# macOS: native Metal when the installed Cortiq build and device support it
CMF_GPU=1 cortiq run "$MODEL" --prompt 'Return the sum of 21 and 21.' --greedy
```

Use `cortiq gpu` to inspect adapters before treating a run as GPU-backed. GPU
speed, memory use and even the selected execution path depend on the artifact,
Cortiq build, driver and available memory; the published benchmark table must
name all of them. If no compatible GPU path is available, use `CMF_GPU=0` for
the CPU reference path.

## Reproduce the conversion

The release must retain a pinned source snapshot and an exact conversion
command. The following is the expected shape for the Q4TP artifact above;
replace it only with the command recorded in the release manifest:

```bash
# Download the pinned upstream snapshot to ./mellum-upstream with a tool that
# honors revision 92ddae9fc7665e9f801d141d2e5a6b2caf2460c4, then:
cortiq convert --model ./mellum-upstream --quant q4tp \
  --output mellum2.1-12b-a2.5b-thinking-q4tp.cmf
cortiq verify mellum2.1-12b-a2.5b-thinking-q4tp.cmf
```

A proper release records the source revision; hashes of `config.json`,
`tokenizer.json` and `chat_template.jinja`; the resulting CMF SHA-256; and the
conversion/runtime version. `BENCHMARKS.md` defines the quality and speed gate.

## What is and is not measured

This card intentionally contains no invented tokens-per-second, VRAM or quality
numbers. Add only measurements produced by the documented commands in
[`BENCHMARKS.md`](BENCHMARKS.md), with the raw JSON retained in the repository.
A quantized model can change logits and generated text. A passing `cortiq
verify` result detects malformed or changed file payloads; it does not prove
benchmark parity, suitability for a particular codebase, or safe use.

## Attribution and license

- Original model: [JetBrains/Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking),
  Apache-2.0. The upstream source and its model card define the original
  capabilities, intended use and limitations.
- CMF container and Cortiq runtime: [infosave2007/cmf](https://github.com/infosave2007/cmf),
  Apache-2.0.
- This repository distributes a derived, quantized representation. Preserve
  the upstream license and attribution with every redistribution.
