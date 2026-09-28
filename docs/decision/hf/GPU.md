# Metal and Vulkan: the same CMF, accelerated

The model file, skills and gates stay unchanged. The GPU prepares text features
and computes reconstruction errors; it does not generate an answer.

**Available in Cortiq 0.8.0 and newer.** Install or update from crates.io:

```bash
cargo install cortiq-cli --locked
```

Requires Rust 1.88 or newer. GPU support is included; CPU remains the runtime default.

![GPU latency, within-host comparisons; different request schedules and independent scales.](figures/gpu.svg)

## RTX PRO 4000 Blackwell · Vulkan

| Dataset | CPU p50, ms | GPU p50, ms | GPU p95, ms | GPU p99, ms |
|---|---:|---:|---:|---:|
| BANKING77 | 24.05–24.30 | 1.24–1.24 | 1.77–1.78 | 2.23–2.30 |
| CLINC150 | 21.07–21.11 | 1.18–1.20 | 1.41–1.43 | 1.56–1.59 |
| MASSIVE | 16.23–16.26 | 1.10–1.15 | 1.33–1.37 | 1.50–1.56 |

Two full runs, each with 3,080 BANKING77, 4,500 CLINC150 and 2,974 MASSIVE rows.
CPU baseline: native portable-release path on the same Xeon E5-2690 v4 server.
CPU references are computed first; after GPU warmup, GPU requests run serially
without a CPU reference calculation between requests. Ranges span both runs.
This is warm local latency, not HTTP, cold startup, concurrent throughput or a Jev test.

**Idle behavior matters:** a final-build diagnostic alternates CPU/GPU requests
on 400 deterministically sampled rows per dataset. The GPU waits while the CPU
computes its reference; these gaps allow the RTX to lower its memory clock.

| Dataset | Rows | CPU p50, ms | Vulkan p50 / p95, ms |
|---|---:|---:|---:|
| BANKING77 | 400 | 25.29 | 8.57 / 9.16 |
| CLINC150 | 400 | 21.50 | 5.02 / 8.84 |
| MASSIVE | 400 | 16.56 | 3.40 / 7.50 |

Do not substitute continuous-stream latency for sparse-traffic latency.
No clocks, power limits or background keep-alive loads were forced. These
diagnostic samples are separate from the full-corpus quality checks above.

## Apple M4 · Metal

Default alternating CPU/GPU protocol, two full runs:

| Dataset | CPU p50, ms | GPU p50, ms | GPU p95, ms | GPU p99, ms |
|---|---:|---:|---:|---:|
| BANKING77 | 3.45–3.46 | 2.57–2.58 | 3.75–4.43 | 6.02–7.54 |
| CLINC150 | 3.26–3.27 | 2.52–2.55 | 3.38–3.54 | 4.09–6.37 |
| MASSIVE | 2.66–2.68 | 2.28–2.31 | 3.07–3.86 | 3.37–7.17 |

Separate serial streams, also two full runs:

| Dataset | CPU p50, ms | GPU p50, ms | GPU p95, ms | GPU p99, ms |
|---|---:|---:|---:|---:|
| BANKING77 | 3.43–3.43 | 2.08–2.09 | 3.13–3.15 | 3.75–3.80 |
| CLINC150 | 3.24–3.25 | 2.06–2.13 | 2.93–4.11 | 3.15–5.33 |
| MASSIVE | 2.63–2.64 | 1.81–3.21 | 2.70–5.44 | 2.92–5.97 |

The Mac had an active desktop. Metal shares the GPU with other applications;
one MASSIVE serial-stream run was slower than CPU, and some p95/p99 tails also
regressed. Acceleration is not a guarantee for every request or workload.
CPU remains the default; GPU selection is explicit.

## Correctness and resources

Across the two full runs per mode, winners, abstentions and hash features match
CPU exactly. This preserves the existing model's quality; it is not a new accuracy
improvement. Embedding tolerance remains `1e-5`; scaled reconstruction-error
difference is limited to `2e-5`. No retraining, oracle calls or lower precision.
[All per-dataset measurements and numerical checks](evidence/gpu.json).

The working path is used by CLI, `/v1/decisions` and `/v1/route`. Hardware tests cover
empty/long/Unicode input, non-aligned dimensions, non-orthogonal bases, multiple
skills and concurrent requests. GPU failures return errors, not a hidden CPU fallback.
Resident GPU weights use additional memory; model file size is not RAM or VRAM.
No energy comparison or hardware-normalized Jev resource comparison was measured.

| Host | CPU RSS, MiB | GPU-mode RSS, MiB |
|---|---:|---:|
| Apple M4 | 478.5 | 808.1 |
| Xeon + RTX | 437.6 | 694.8 |

RSS snapshots after 11 successful HTTP requests per backend, all three skills
loaded; not peak memory. The Vulkan process also reported **522 MiB of VRAM**.
Four concurrent clients exercised routing; both decision API aliases were
checked. Temporary servers were stopped. CPU/GPU timing excludes model loading,
shader compilation, response serialization, network and external oracles.

## Choose a GPU

```bash
# Apple Silicon
CORTIQ_DECISION_DEVICE=metal cortiq decide cortiq-decision.cmf \
  --skill banking77 -p "I still have not received my new card"

# Linux with a hardware Vulkan driver
CORTIQ_DECISION_DEVICE=vulkan CORTIQ_DECISION_VULKAN_ADAPTER="RTX PRO 4000" \
  cortiq decide cortiq-decision.cmf \
  --skill banking77 -p "I still have not received my new card"
```

The same environment variables apply to `cortiq serve`. `/healthz` reports the
actual adapter and completed GPU submissions. `gpu` timing includes the joint
encoder and reconstruction pass plus its wait, not a fictional kernel-only time.
Other GPUs require their own validation; a software Vulkan adapter is rejected.

[Back to model card](README.md) · [Existing Jev comparison](BENCHMARKS.md)
