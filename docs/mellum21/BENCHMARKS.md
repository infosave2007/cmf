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
| Cortiq | 0.8.9 |
| Integrity | `cortiq verify` passed after conversion: envelope, sections, 5,631 directory entries, and per-tensor hashes |
| Quantization | 5,376 Q4TP tensors; 114 Q8_2f tensors; 141 F16 tensors |

## CPU core benchmark

**Host:** shared RunPod machine with AMD EPYC 7663, 112 logical CPUs, and
approximately 251 GiB RAM. The attached RTX PRO 4500 Blackwell GPU (32,623 MiB) was not used
for these CPU measurements.

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
| 1 | 30.5451 | 41.2580 | 16.5323 |
| 2 | 30.8812 | 40.7382 | 16.6234 |
| 3 | 30.6842 | 41.2662 | 16.6462 |
| 4 | 30.8070 | 40.5975 | 15.8445 |
| 5 | 31.4649 | 40.7521 | 16.3741 |
| **median** | **30.8070** | **40.7521** | **16.5323** |
| **range** | **30.5451–31.4649** | **40.5975–41.2662** | **15.8445–16.6462** |

The model's observed KV state at sequence 767 was 87,965,696 bytes. That is a
KV-state observation, not total process memory or a 131K-context measurement.

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

No end-to-end Metal or Vulkan model throughput number is published in this
release. On the benchmark pod, Vulkan exposed llvmpipe rather than the NVIDIA
adapter because the NVIDIA ICD was unavailable in the container. The runtime's
generic selected-top-8 Q4TP MoE kernel has component coverage on Metal and
wgpu; that is deliberately not presented as a full-model benchmark.

The structured copy of this record is [measurements.json](measurements.json).
For a new benchmark, retain the emitted JSON, exact environment, selected
adapter, artifact SHA-256, context, generation budget, and whether a model was
reloaded between samples.
