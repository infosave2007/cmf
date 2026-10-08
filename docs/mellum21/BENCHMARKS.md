# Mellum2.1 CMF — measurement record

This document records only measured results for the release artifact. It does
not compare CPU and GPU figures, or CMF and the upstream checkpoint, as though
they were the same experiment.

## Artifact under test

| Field | Value |
|---|---|
| File | `mellum2.1-12b-a2.5b-thinking-q4tp.cmf` |
| Size | 6,884,556,163 bytes (6.41 GiB) |
| SHA-256 | `1734d8c585134547efa6eb60f092fd741135fe0f398fe9dc77ab4b97df6a7175` |
| Cortiq | 0.8.14 |
| Integrity | `cortiq verify` passed after conversion: envelope, sections, 5,631 directory entries, and per-tensor hashes |
| Quantization | 5,376 Q4TP tensors; 114 Q8_2f tensors; 141 F16 tensors |

## CPU core benchmark

**Host:** shared RunPod machine with AMD EPYC 7663, 112 logical CPUs, and
approximately 251 GiB RAM. The attached RTX PRO 4500 Blackwell GPU (32,623 MiB)
was not used for these CPU measurements. The container had a 23.8-core CPU
quota; the runtime's automatic policy selected a 22-worker pool.

**Command:**

```bash
for i in 1 2 3 4 5; do
  CMF_GPU=0 cortiq bench "$MODEL" \
    --ctx 512 --tokens 256 --core --ignore-eos --json \
    > "raw/bench/cpu-${i}.json"
done
```

Each sample was a separate process. `--core` reports model-core timing; it is
not a request latency or service-throughput SLA.

| Sample | Prefill tok/s | Steady decode tok/s | TTFT (s) |
|---:|---:|---:|---:|
| 1 | 36.6873 | 40.8387 | 13.7703 |
| 2 | 38.4731 | 40.6155 | 13.4867 |
| 3 | 38.4055 | 40.6176 | 13.5313 |
| 4 | 37.2785 | 40.7339 | 13.4317 |
| 5 | 38.5184 | 41.2311 | 13.5708 |
| **median** | **38.4055** | **40.7339** | **13.5313** |
| **range** | **36.6873–38.5184** | **40.6155–41.2311** | **13.4317–13.7703** |

The model's observed KV state at sequence 767 was 87,965,696 bytes. That is a
KV-state observation, not total process memory or a 131K-context measurement.

### Worker-pool calibration

Before the five-sample record above, one bounded 512-context / 128-generation
core run was made for each worker count. The automatic 22-worker setting gave
the highest steady decode of the tested values (41.75 tok/s); forcing more
workers did not help on this quota (24: 36.58; 28: 37.40 tok/s). This is a
host-specific tuning check, not a cross-machine performance claim.

## Scope and validation

- The release artifact passed `cortiq verify` on the conversion host.
- The converter/runtime regression tests cover Mellum's full-vs-sliding RoPE
  schedule, top-8 MoE, mixed tensor profile, and a tiny forward pass.
- A local OpenAI-compatible API smoke request completed against the converted
  artifact.
- A fixed five-prompt greedy smoke suite paired the source BF16 CUDA model and
  this CMF CPU artifact. Four of five completions matched exactly after
  normalizing the source end marker. This is a functional smoke check, **not**
  a task-quality score or a claim of byte-identical upstream parity.
- A 1,152-token CPU prefill/generation run completed across the 1,024-token
  sliding window (`seq_len=1,155`); the raw observation is
  `benchmarks/cpu-swa-boundary-1152.json`.

No end-to-end Metal or Vulkan model throughput number is published in this
release. On the benchmark pod, Vulkan exposed llvmpipe rather than the NVIDIA
adapter because the NVIDIA ICD was unavailable in the container. The runtime's
generic selected-top-8 Q4TP MoE kernel has component coverage on Metal and
wgpu; that is deliberately not presented as a full-model benchmark.

The structured copy of this record is [measurements.json](measurements.json).
The 1,152-token boundary probe completed through the local sliding-attention
window; its JSON record is included in the `benchmarks/` directory.
For a new benchmark, retain the emitted JSON, exact environment, selected
adapter, artifact SHA-256, context, generation budget, and whether a model was
reloaded between samples.
