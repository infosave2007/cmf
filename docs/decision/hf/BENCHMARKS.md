# Benchmark details

[Back to the model card](README.md)

The quality and Jev/API comparisons below retain the original measurements of this CMF. New GPU measurements are documented separately in [Metal and Vulkan](GPU.md). The charts are generated from [comparison.json](evidence/comparison.json); it includes the source report names and SHA-256 hashes.

## Accuracy and coverage

**All examples** means the best-ranked label before applying the abstention gate. **Accepted accuracy** counts only answers the gate accepts; coverage tells how many examples receive those answers.

| Dataset | Test rows | Cortiq correct / all | Jev correct / all | Cortiq accepted correct / accepted | Coverage |
|---|---:|---:|---:|---:|---:|
| BANKING77 | 3,080 | 2,875 / 3,080 · 93.34% | 2,636 / 3,080 · 85.58% | 2,714 / 2,791 · 97.24% | 90.62% |
| CLINC150 | 4,500 | 4,328 / 4,500 · 96.18% | 4,354 / 4,500 · 96.76% | 4,091 / 4,145 · 98.70% | 92.11% |
| MASSIVE | 2,974 | 2,562 / 2,974 · 86.15% | 2,551 / 2,974 · 85.78% | 1,581 / 1,615 · 97.89% | 54.30% |

| Dataset | Cortiq 95% Wilson interval | Jev 95% Wilson interval | Paired McNemar p |
|---|---:|---:|---:|
| BANKING77 | 92.41–94.17% | 84.30–86.78% | 9.72e-33 |
| CLINC150 | 95.58–96.70% | 96.20–97.23% | 0.099 |
| MASSIVE | 84.86–87.34% | 84.48–86.99% | 0.607 |

Cortiq has higher BANKING77 accuracy in this experiment. Jev is 0.58 percentage points higher on CLINC150; the MASSIVE point estimates differ by 0.37 points. Non-significant p-values do not prove equivalence. These are separate task-level comparisons, not proof of universal superiority.

## Latency

| Dataset | Local CPU p50 / p95, ms | Cortiq HTTPS p50 / p95, ms | Jev API p50 / p95, ms |
|---|---:|---:|---:|
| BANKING77 | 3.92 / 5.27 | 78.40 / 141.60 | 348.33 / 426.03 |
| CLINC150 | 3.79 / 4.57 | 78.50 / 134.60 | 391.51 / 528.49 |
| MASSIVE | 3.00 / 3.58 | 70.40 / 117.90 | 346.37 / 492.93 |

- **Local:** Apple M4, 24 GB, one CPU thread; `cortiq decide --bench`, 50 warm-up rows, then every test row. Includes tokenization, encoder, hashing and resonance. Not HTTP and not kernel-only latency.
- **Cortiq HTTPS:** sequential requests over one persistent connection from the same client Mac, every test row, oracle explicitly disabled. The deployed server used a 2-vCPU AMD EPYC 9655P VM behind nginx. All 10,554 responses were HTTP 200; accuracy and gate totals match the local run.
- **Jev:** `typesafe/jev-1.13` through OpenRouter. BANKING77 uses per-row round trips from a four-worker run over 3,080 rows. CLINC150 and MASSIVE use serial samples of 200 rows each. Serving hardware is unknown; these runs were not simultaneous with Cortiq HTTPS.
- API latency includes network and TLS. Hardware, load, sample size and concurrency differ. The charts describe recorded client experience, not a controlled hardware speedup or throughput benchmark.

## Optional oracle: quality and API fees

Local-only decisions have no external API charge. The hybrid experiment uses **DeepSeek V4.1 Flash**, not Jev, for questions the local gate rejects. Cache and self-learning are enabled. This is a two-component system: its quality and fees must not be attributed to the standalone CMF.

| Dataset | Hybrid correct / all | Hybrid accuracy | Oracle calls | Hybrid API $ / 1M | Jev API $ / 1M |
|---|---:|---:|---:|---:|---:|
| BANKING77 | 2,893 / 3,080 | 93.93% | 282 | 3.01 | 183.69 |
| CLINC150 | 4,386 / 4,500 | 97.47% | 351 | 3.53 | 271.61 |
| MASSIVE | 2,617 / 2,974 | 88.00% | 1,335 | 8.48 | 110.86 |

The hybrid run replayed recorded oracle answers, with 24 additional live MASSIVE calls. The shown cost is recorded oracle spend divided by all test examples, multiplied by one million. It is not a million-call run, a future price guarantee, or total cost of ownership. Hardware, electricity and operations are excluded. The pure-local latency chart does not describe oracle-assisted requests. [Full oracle experiment and setup](ORACLE.md#measured-effect).

## What the comparison does and does not establish

- Public test sets were reused across development iterations. This is not a new blind holdout.
- Supervision is unequal: Cortiq was fitted on train + dev, while Jev received two training examples per label. Neither this result nor a confidence interval removes that difference.
- English inputs, up to 512 tokens. The downloadable model uses a Cortiq NVG-modified text encoder (75% MLP compression) and trained task topologies; it is distinct from the separate Embryo research model.
- The confidence gate is calibrated in-domain. It is not a universal correctness or out-of-distribution guarantee.
- Incremental fitting preserves other labels’ parameters; that alone does not guarantee unchanged decisions because labels can compete for the same input.
- No hardware-normalized RAM, VRAM or energy comparison with Jev was measured. The file is 304,520,292 bytes; model-file size is not process memory.

## Model identity and attribution

File SHA-256: `ed9b8ec2bbfe9e9fd30f14a5eaf82314f38bc7e7510a39772baa2de3801d79b1`.

- Cortiq engine and CMF: Apache-2.0, [source](https://github.com/infosave2007/cmf).
- BANKING77: PolyAI, CC-BY-4.0. CLINC150: Larson et al., CC-BY-3.0. MASSIVE en-US: Amazon, CC-BY-4.0.
- Resonance Routing — US Patent Application 19/452,440.

The original Jev/API charts are unchanged. The new GPU work adds local timing and CPU/GPU parity tests; it makes no paid oracle calls and does not retrain the model.

## Execution and language scope

The original local measurements above use CPU. Cortiq 0.8.0 and newer
run this same CMF on Metal and Vulkan. See [GPU measurements and usage](GPU.md)
for request schedules, full-corpus parity, latency tails and release availability.
Those GPU results do not replace the original Jev/API experiment.

The included seed skills and their reported scores were evaluated in English.
The CMF format itself does not impose a language; local language coverage
depends on the encoder and skill training data. An optional oracle can handle
cases outside the local skills, within that oracle's language capabilities.
Enabling an oracle does not by itself certify multilingual local accuracy.
