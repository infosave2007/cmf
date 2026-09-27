---
license: apache-2.0
library_name: cortiq
pipeline_tag: text-classification
base_model: BAAI/bge-small-en-v1.5
language:
  - en
tags:
  - cmf
  - cortiq
  - intent-classification
  - selective-prediction
  - resonance-routing
---

# Cortiq Decision

Cortiq Decision is a typed-decision model in one CMF file (304.5 MB,
304520292 bytes) that runs on a CPU with the Rust engine `cortiq`. It decides
by **Resonance Routing**: every label is an isolated affine subspace fitted to
the signal `[φ_P ; 0.5·φ_H]`, where φ_P is the model's own router encoder —
bge-small-en-v1.5 with its MLP compressed by 75 % with Cortiq NVG (12 layers,
hidden 384) — and φ_H are 4096 hashed lexical features; the answer is the
label with the smallest reconstruction error, with a calibrated confidence, a
novelty score and a gate certified on held-out calibration rows, so the model
abstains instead of guessing. Three skills are included: `banking77` (77
labels), `clinc150` (150) and `massive` (60, MASSIVE en-US). It serves the
request/response shape of the Jev decisions API on OpenRouter (Cortiq
Decision itself is not listed on OpenRouter) and the cortiq-router API, and
it can escalate the questions it abstains on to an LLM through OpenRouter,
whose answers can teach the model without changing any other label's
parameters (checked by sha256).

## Results on the test sets

Jev 1.13 is `typesafe/jev-1.13` through OpenRouter's decisions API: its stored
answers on the same test rows. Wilson 95 % intervals; McNemar is the exact
two-sided test on the paired rows (on the gate rows: the answered rows only).

| Test set | Row | Correct / n | Accuracy [Wilson 95 %] | Jev 1.13, same rows | McNemar p |
|---|---|---|---|---|---|
| BANKING77 | CMF, all rows | 2875 / 3080 | 93.34 % [92.41, 94.17] | 2636 / 3080 (85.58 %) | 9.72e-33 |
| | CMF, certified gate (answered 2791 / 3080, coverage 90.62 %) | 2714 / 2791 | 97.24 % [96.57, 97.79] | 2456 / 2791 (88.00 %) | 1.63e-56 |
| | CMF + DeepSeek V4.1 Flash, cascade with self-learning (282 calls) | 2893 / 3080 | 93.93 % [93.03, 94.72] | 2636 / 3080 (85.58 %) | 3.23e-48 |
| CLINC150 | CMF, all rows | 4328 / 4500 | 96.18 % [95.58, 96.70] | 4354 / 4500 (96.76 %) | 0.099 |
| | CMF, certified gate (answered 4145 / 4500, coverage 92.11 %) | 4091 / 4145 | 98.70 % [98.30, 99.00] | 4055 / 4145 (97.83 %) | 0.000769 |
| | CMF + DeepSeek V4.1 Flash, cascade with self-learning (351 calls) | 4386 / 4500 | 97.47 % [96.97, 97.89] | 4354 / 4500 (96.76 %) | 0.0117 |
| MASSIVE | CMF, all rows | 2562 / 2974 | 86.15 % [84.86, 87.34] | 2551 / 2974 (85.78 %) | 0.607 |
| | CMF, certified gate (answered 1615 / 2974, coverage 54.30 %) | 1581 / 1615 | 97.89 % [97.07, 98.49] | 1530 / 1615 (94.74 %) | 4.29e-13 |
| | CMF + DeepSeek V4.1 Flash, cascade with self-learning (1335 calls) | 2617 / 2974 | 88.00 % [86.78, 89.12] | 2551 / 2974 (85.78 %) | 7.57e-06 |

* On CLINC150 all rows Jev is 0.58 points higher; the difference is not
  significant (p = 0.099). On MASSIVE all rows the two are level (p = 0.607).
* The cascade row is `cortiq serve` with the cache and self-learning on, in
  one pass over the test rows (no ground truth, no feedback): the certified
  gate answers what it accepts, and the rest is answered by
  `deepseek/deepseek-v4.1-flash` through OpenRouter with Jev's rubric
  (temperature 0, reasoning off; stored answers replayed, 24 MASSIVE calls
  live) or by the semantic cache (21 hits). The static cascade — every abstention answered, no cache, no
  learning — got 2894, 4386 and 2617 right with 289, 355 and 1359 calls
  (McNemar p against Jev 8.2e-49, 0.0117 and 8.37e-06), one more right on
  BANKING77 than the served cascade ([ORACLE.md](ORACLE.md#measured-effect)).

**Speed.** Text → decision (tokenize, encode, hash, resonance) on one thread of
an Apple M4, `cortiq decide --bench` over every test row: p50 / p95 =
3.92 / 5.27 ms (BANKING77), 3.79 / 4.57 ms (CLINC150), 3.00 / 3.58 ms
(MASSIVE). Jev 1.13, stored hosted round trips from the same Mac:
348.33 / 426.03 ms (per-row times of a 4-worker run), 391.51 / 528.49 ms and
346.37 / 492.93 ms (serial, 200 rows each). Jev's times include the network;
they are not hardware-normalized.

**$ per 1M decisions.** CMF alone: $0 in API fees (hardware and electricity
not counted). CMF + DeepSeek, the served cascade with self-learning: $3.01
(BANKING77), $3.53 (CLINC150), $8.48 (MASSIVE); the static cascade $3.09,
$3.56 and $8.52 — the stored OpenRouter cost of the oracle calls divided by
all test rows. Jev 1.13: $183.69, $271.61, $110.86 (mean stored cost per
call).

## What you get that Jev does not

* **Abstention and novelty.** Every answer carries a gate verdict, a novelty
  score and a margin. On the 1000 out-of-scope CLINC150 test queries the same
  certified gate, never tuned on them, rejected 864 (86.4 %). The stored Jev
  runs offered only the 150 in-scope labels, so Jev picked one of them for
  every out-of-scope query.
* **Certified selective accuracy.** Each skill's gate is chosen on held-out
  calibration rows so that the Clopper–Pearson lower bound of the accepted
  answers' accuracy is at least 0.95; on the test sets the accepted answers
  were 97.24 %, 98.70 % and 97.89 % right. `certified: true` marks them.
* **Complexity and routing.** Each decision has a complexity score and tier
  (low / medium / high); with `routing_tiers` the response names your own
  model for that tier.
* **Explanations.** Reconstruction errors of the candidates, the margin, the
  decision path (`router:certified`, `escalate→oracle`, …).
* **Oracle only where needed, with a cache.** Questions the gate accepts never
  leave the machine; only undetermined ones (abstentions, and questions no
  skill covers) can go to an LLM through OpenRouter, under a budget with stop
  rules; paraphrases of an answered question (cosine ≥ 0.97) are served from
  a semantic cache at no cost.
* **Self-learning with isolated labels.** Oracle answers and the feedback of
  keys allowed to teach become examples; a label is refitted after 25 new
  examples, kept only if it is at least as good on a holdout, and the gate is
  re-certified. Every other label's parameters stay byte for byte the same
  (0 isolation violations in the measured runs), though a refitted label can
  win rows others used to win. Each promotion is a generation you can roll
  back.
* **On-premises and private.** One file, CPU only, no Python, runs offline.
  With the oracle on, e-mail addresses, secret-like tokens and long numbers
  are redacted by default before a question leaves (a heuristic, see
  [ORACLE.md](ORACLE.md#what-leaves-the-machine)).
* **Drop-in for cortiq-router clients.** The same server answers the
  cortiq-router API (schema 1.1), imports its API keys (sha256) and can run in
  shadow mode next to the old router before the switch.

## Quick start

```bash
cargo install cortiq-cli --version 0.7.8
hf download infosave/cortiq-decision cortiq-decision.cmf --local-dir .
cortiq decide cortiq-decision.cmf --skill banking77 -p "I still have not received my new card"
```

Output of the third command (cortiq 0.7.8, Apple M4):

```text
choice:     card_arrival
action:     local (accepted by the gate), certified true
skill:      banking77 (exact match, 77 candidates)
gate:       p_top 0.9980738 (tau 0.7), novelty 0.40156537 (theta 0.804234), margin 0.116146445, confidence 0.9980485
errors:     card_arrival 0.097457424, card_linking 0.25778145, activate_my_card 0.30103046, get_physical_card 0.31008253, order_physical_card 0.31063738
model:      cortiq/decision@386b6e43fd35 (generation 0), 3489 input tokens, 4335 µs
```

`cortiq serve cortiq-decision.cmf` starts the HTTP server: see
[API.md](API.md) for both protocols, keys, your own skills and the migration
from cortiq-router, and [ORACLE.md](ORACLE.md) for the oracle cascade,
self-learning and the OpenRouter setup.

## Limits

* The benchmarks are public and were reused; their test splits had been read
  before. K per skill was chosen by cross-validation on train ∪ dev only, and
  test rows were never used for training or calibration. Jev (and DeepSeek)
  may well have seen these public datasets during training.
* Supervision is not equal: CMF was trained on the train and dev splits, Jev
  was given the label names and two training examples per label.
* The encoder is English (bge-small-en-v1.5 base, English WordPiece
  vocabulary); texts are read up to 512 tokens.
* The gate is certified in-domain, for traffic like the skill's calibration
  split; shifted traffic can break the guarantee, and coverage can be low
  (54.30 % on MASSIVE). Self-learning picks each challenger on a holdout that
  is part of the calibration rows the gate is then re-certified on, so the
  bound is nominal after any promotion.
* Every speed number was measured on macOS arm64 (Apple M4, 24 GB), one
  thread; other platforms were not measured.
* The cascade numbers use stored DeepSeek V4.1 Flash answers; a provider can
  change its answers and prices.

## Files and licenses

`cortiq-decision.cmf` (sha256
`ed9b8ec2bbfe9e9fd30f14a5eaf82314f38bc7e7510a39772baa2de3801d79b1`),
`README.md`, `API.md`, `ORACLE.md` and `SHA256SUMS`
(`shasum -a 256 -c SHA256SUMS`).

* Cortiq engine and CMF format: Apache-2.0,
  [github.com/infosave2007/cmf](https://github.com/infosave2007/cmf).
* Encoder base: [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
  (BAAI), MIT; its MLP was compressed by 75 % with Cortiq NVG.
* Training data: BANKING77 (PolyAI), CC-BY-4.0; CLINC150 (Larson et al.,
  2019), CC-BY-3.0; MASSIVE (Amazon, en-US), CC-BY-4.0.
* Resonance Routing — US Patent Application 19/452,440.
