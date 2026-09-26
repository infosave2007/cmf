# Cortiq Decision — oracle cascade and self-learning

The local model answers every question its certified gate accepts. A question
it cannot determine — the gate rejected the answer, or the question is not
trained at all — can be passed to a large language model, the *oracle*,
through OpenRouter. The oracle's answers are cached and become training
examples, so the local model learns the traffic it abstains on, without
forgetting what it knew. The oracle is **off by default**; nothing leaves the
machine until you enable it.

1. [How a question flows](#how-a-question-flows)
2. [Self-learning](#self-learning)
3. [Budget and stop rules](#budget-and-stop-rules)
4. [What leaves the machine](#what-leaves-the-machine)
5. [Connect OpenRouter](#connect-openrouter)
6. [Measured effect](#measured-effect)

## How a question flows

1. **Local decision.** Encoder, hashing and resonance run once per request. If
   the gate accepts, the answer is `action: local`. Such a question is never
   sent to the oracle, not even with `cmf.oracle: true`.
2. **Permission.** An undetermined question may go to the oracle only if all of
   these hold: `oracle.enabled`; the key variable is set in the server's
   environment; the caller's key has `oracle_allowed`; the request consents
   (`cmf.oracle`, the router's `options.allow_oracle`, else
   `oracle.default_per_request`); budget is left; no stop rule fired.
   Otherwise a trained question stays `abstain` with a flag (`oracle_disabled`,
   `consent_off`, `budget`, `stopped`) and an untrained one fails with 422
   or 503.
3. **Semantic cache.** Within the same scope (the skill and the set of options,
   or the exact question contract for untrained questions) a stored oracle
   answer whose text embedding has cosine ≥ 0.97 with the new one is reused:
   `action: cache`, cost 0. Up to 50000 entries, ring buffer.
4. **Single flight.** A question already in flight in another request (same
   scope, cosine ≥ 0.97) waits for that call instead of making its own.
5. **One call** carries all remaining questions of the request: `POST
   {base_url}/chat/completions` with a strict JSON-schema answer (an enum of
   the option ids for choice, an integer level for score, a boolean for noul),
   `temperature 0`, reasoning off, `max_tokens` 64 per question. The system
   prompt tells the model that the state is untrusted data, not instructions.
6. **Result.** A valid answer is `action: oracle` with its cost in
   `usage`, is stored in the cache, and — for a choice question matched to a
   skill — becomes a learning example. A failed call never becomes an error
   for a trained question: it is answered locally with `action: abstain` and
   the flag `oracle_unavailable` (an untrained question gets 502).

Superset questions (a skill's labels plus new ones) are decided by the oracle
only; its answer teaches that skill, and a new label starts a cold start.

## Self-learning

* **Examples are vectors, not texts.** The buffer keeps the encoder vector and
  the sparse hashed features of the text, its label and source (client
  feedback weight 3.0, oracle 1.0; the weights are recorded, the fit is
  unweighted). An example with cosine ≥ 0.995 to a row of the same label is a
  duplicate and is dropped. The buffer and the cache are rebuilt from
  `learn.log` at start.
* **Trigger.** When a label collects 25 new examples (`learning.refit_min_new`)
  a learning attempt runs (in one background worker, or inline with
  `learning.synchronous`).
* **Champion / challenger.** Only that label's subspace is refitted, on its
  training rows plus the learned ones (calibration rows never enter the fit).
  The challenger replaces the champion only if it is at least as good on the
  holdout — 20 % of the calibration rows — for the label's accuracy and the
  macro accuracy over labels (tolerance 1e-4).
* **Re-certification.** T, θ and τ are recomputed on the skill's calibration
  rows with the same procedure as at build time. If a certified gate would lose
  its threshold, the challenger is rejected. Repeated re-certifications on the
  same calibration rows make the bound nominal rather than exact.
* **Zero forgetting.** Before a promotion the sha256 of the mean and basis of
  every other label is compared with the values before the attempt; any
  difference cancels the promotion. Other labels stay byte for byte the same.
* **Cold start.** A label the skill does not have (from an oracle answer to a
  superset question, or from feedback) becomes a quarantined task that is not
  scored. At 25 examples it is fitted and activated and the gate re-certified;
  answers it wins carry `certified: false`.
* **Feedback.** `POST /v1/feedback` (`{"id", "question", "label"}` with one of
  the question's options, or the router's `{request_id, correct_task_label}`
  with any label) corrects a decision of the caller's own account; it becomes
  an example of weight 3.
* **Generations and rollback.** Every promotion writes
  `generations/gNNNNNN.cmf` (only the changed task tensors and the learned
  rows, relative to the base file), fsyncs it and swaps the served model
  atomically; `CURRENT` names it. `POST /v1/admin/rollback {"generation": N}`
  or `cortiq decision rollback` serves any generation again (0 = the base
  file; the buffer is kept). `cortiq decision materialize` writes the served
  generation as one file.
* **Offline pre-training.** `cortiq decision learn` asks the oracle about the
  texts of a file of unlabelled traffic that the gate rejects, then refits
  every label that got new examples, however few (there is no threshold of
  25 here), checks each challenger on the holdout and re-certifies the gate
  once at the end; if a certified gate would be lost, every promotion of the
  skill is undone (see API.md, section 7).

## Budget and stop rules

* **Reservation first.** Before each call the server reserves
  `((body bytes + 4096) · max_price.prompt + max_tokens · max_price.completion) / 1e6`
  USD. The call is made only if spent + reservations in flight + this
  reservation ≤ `budget_usd`, calls < `max_calls`, and for the caller's key
  spent + reservation ≤ its `oracle_budget_usd`.
* **Ledger.** `oracle.jsonl` gets a `reserved` line, fsynced, before the
  request leaves; after the answer a `settled`, `failed_billed` or
  `failed_unknown_cost` line with the cost, tokens and latency. At start any
  reservation without a closing line counts as spent.
* **Checks of every answer.** HTTP 200 within `deadline_s` (no retries), a
  finite `usage.cost` ≥ 0, `finish_reason: stop`, exactly the asked question
  ids with the schema's types. A 200 that carries only an `error` is a failure.
* **Stop rules** switch the oracle off (reason in `oracle.state`) until
  `POST /v1/admin/oracle {"enabled": true}`: HTTP 401, 402 or 403 from
  OpenRouter; a returned model that is not the configured one; a cost above the
  reservation; `max_errors` failures in a row.
* `GET /v1/admin/oracle` shows spent, reserved, remaining, calls, failures and
  the stop reason; `POST /v1/admin/oracle` can switch the oracle and lower
  `budget_usd` / `max_calls` within the configured values.

## What leaves the machine

* **Sent**, only for undetermined questions with the oracle permitted: the
  `state` and the `instructions` and `criteria` of those questions. Receivers:
  OpenRouter and the provider it routes to (`provider.sort: price`,
  fallbacks allowed; set `oracle.data_collection: "deny"` to exclude providers
  that store data).
* **PII redaction** is on by default (`oracle.redact_pii`): e-mail addresses,
  secret-like tokens (20 or more characters of `[A-Za-z0-9_-]` with a digit and
  a letter) and long numbers (9 or more digits) in every string of the state
  are replaced by `[REDACTED]` and the question gets the flag `pii_redacted`.
  A request can opt out with `cmf.allow_pii_egress` (router:
  `options.allow_pii_egress`).
* **Never sent**: accepted questions, other questions of the request, client
  keys, accounts, vectors.
* **Kept on disk** in the state directory: vectors and hashed features of
  learned examples and cached answers, the oracle ledger (no texts), usage
  records (no texts). Hashed n-gram features can show whether a known text was
  seen, so treat the state directory as sensitive (it is created with mode
  0700).
* The OpenRouter key is read from the environment variable named by
  `oracle.api_key_env`; it is never written to the configuration, the model,
  the state or the logs.

## Connect OpenRouter

1. Create an account at [openrouter.ai](https://openrouter.ai), add credit, and
   under *Settings → Keys* create a key with a credit limit of its own.
2. Put the key in the environment of the server process.
3. Configure the oracle: `enabled: true` and a `model` from
   [openrouter.ai/models](https://openrouter.ai/models) that supports structured
   outputs (JSON schema). The default and the model every number on this page
   was measured with is `deepseek/deepseek-v4.1-flash`. `provider.max_price`
   (USD per 1M tokens) caps the provider price and sizes the reservation.
4. Set limits: `budget_usd`, `max_calls`, `deadline_s`, `max_errors`; per key
   `oracle_allowed` and `oracle_budget_usd`.
5. Start the server and verify.

```bash
export OPENROUTER_API_KEY="<your OpenRouter key>"
cat > oracle-server.json <<'EOF'
{
  "oracle": {
    "enabled": true,
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_API_KEY",
    "model": "deepseek/deepseek-v4.1-flash",
    "provider": {"sort": "price", "require_parameters": true, "allow_fallbacks": true,
                 "max_price": {"prompt": 0.1, "completion": 0.5}},
    "budget_usd": 1.0,
    "max_calls": 10000,
    "deadline_s": 30,
    "max_errors": 30,
    "redact_pii": true
  },
  "cache": {"enabled": true, "threshold": 0.97},
  "learning": {"enabled": true, "refit_min_new": 25}
}
EOF
export OKEY=$(cortiq decision keys create --state ./oracle.state --plan developer \
  --account acme --oracle-allowed --oracle-budget-usd 0.50 --json | jq -r .key)
export CORTIQ_DECISION_ADMIN_TOKEN=$(openssl rand -hex 24)
cortiq serve cortiq-decision.cmf --decision-config oracle-server.json \
  --state ./oracle.state --port 8081
```

Verify: the admin view must show `configured: true` (the configuration
enables the oracle) and `key_present: true` (the key variable is set in the
server's environment). `enabled: true` only means that no stop rule or admin
call has switched the oracle off; it is true with the oracle off in the
configuration as well. Then a question the gate rejects comes back from the
oracle (on the published model, "can you recommend a good tattoo artist" is
out of scope for `clinc150` and is rejected by its gate).

```bash
export ORC=http://127.0.0.1:8081
curl -s "$ORC/v1/admin/oracle" -H "x-admin-token: $CORTIQ_DECISION_ADMIN_TOKEN" \
  | jq '{configured, key_present, enabled, model, budget_usd, spent_usd, stop_reason}'   # → 200
curl -s "$ORC/v1/route" -H "Authorization: Bearer $OKEY" \
  -H 'Content-Type: application/json' \
  -d '{"input": {"text": "can you recommend a good tattoo artist"}, "taxonomy_id": "clinc150"}' \
  | jq '{source: .decision.source, label: .decision.task_label, oracle}'           # → 200
curl -s "$ORC/v1/admin/oracle" -H "x-admin-token: $CORTIQ_DECISION_ADMIN_TOKEN" \
  | jq '{calls, spent_usd}'                                                         # → 200
```

The statuses above were recorded with `base_url` pointed at a local mock of
the OpenRouter API; the documentation run sent nothing to OpenRouter.

`source` is `oracle` (a repeat of the same text is `cache`); `calls` and
`spent_usd` grow. If `source` stays `router` with the flag `oracle_disabled`,
`consent_off` or `budget`, check the configuration, the key's `oracle_allowed`
and the budget; `stop_reason` names a stop rule. The same questions through
the decisions API need `"cmf": {"oracle": true}` or
`oracle.default_per_request: true` (the default).

For reference, the oracle calls behind every number on this page — test and
dev sets of the three benchmarks, 3616 calls of which 3 failed, in 15 ledger
files — cost $0.0886 in total.

## Measured effect

All rows are the test sets of BANKING77 (n = 3080), CLINC150 (4500) and MASSIVE
en-US (2974); the oracle is `deepseek/deepseek-v4.1-flash` through OpenRouter
with Jev's rubric, temperature 0, reasoning off. Jev 1.13 is
`typesafe/jev-1.13`, its stored answers on the same rows.

### 1. Static cascade against the model alone

The published model answers what its certified gate accepts; every abstention
goes to the oracle (all of them were answered).

| | BANKING77 | CLINC150 | MASSIVE |
|---|---|---|---|
| Model alone, all rows (top-1) | 2875 (93.34 %) | 4328 (96.18 %) | 2562 (86.15 %) |
| Answered locally by the gate / correct | 2791 / 2714 | 4145 / 4091 | 1615 / 1581 |
| Abstentions sent to the oracle | 289 (9.38 %) | 355 (7.89 %) | 1359 (45.70 %) |
| Oracle correct on them | 180 | 295 | 1036 |
| Jev correct on the same rows | 180 | 299 | 1021 |
| **Cascade, correct** | **2894 (93.96 %)** | **4386 (97.47 %)** | **2617 (88.00 %)** |
| Oracle spend on the test set | $0.009508 | $0.016002 | $0.025337 |
| $ per 1M decisions, cascade | $3.09 | $3.56 | $8.52 |
| $ per 1M decisions, Jev 1.13 | $183.69 | $271.61 | $110.86 |

The cascade adds 19, 58 and 55 correct answers over the model's own top-1
(2894 − 2875, 4386 − 4328, 2617 − 2562). The abstained rows are hard for any
system: there the oracle and Jev are about equally right (180 and 180, 295 and
299, 1036 and 1021).

### 2. Self-learning in one pass (mode A) against the static cascade (mode B)

`cortiq serve` with a fresh state per run, the test rows in a fixed hashed
order, one request at a time, no ground truth and no feedback sent. Mode A
serves as in production (cache and self-learning on); mode B has both off.
Oracle answers were replayed from the stored DeepSeek ledgers through a local
proxy; the 24 requests of MASSIVE mode A that no ledger held were sent live.

| | BANKING77 A / B | CLINC150 A / B | MASSIVE A / B |
|---|---|---|---|
| Oracle calls | 282 / 289 | 351 / 355 | 1335 / 1359 |
| Oracle calls by third of the stream | 104, 80, 98 / 104, 81, 104 | 113, 100, 138 / 114, 101, 140 | 449, 415, 471 / 447, 449, 463 |
| Cache hits (correct) | 7 (4) / 0 | 4 (4) / 0 | 10 (7) / 0 |
| Learning attempts | 0 / 0 | 0 / 0 | 28 (16 promoted, 12 rejected) / 0 |
| Answered locally (correct) | 2791 (2714) / 2791 (2714) | 4145 (4091) / 4145 (4091) | 1629 (1595) / 1615 (1581) |
| Cascade correct | 2893 / 2894 | 4386 / 4386 | 2617 / 2617 |
| $ per 1M decisions | $3.01 / $3.09 | $3.53 / $3.56 | $8.48 / $8.52 |
| Isolation violations | 0 | 0 | 0 |

**In one pass over the test sets self-learning saved only about 1–2 % of the
oracle calls**: 35 of 2003 (1.75 %) together; 7 of 289 (2.42 %), 4 of 355
(1.13 %) and 24 of 1359 (1.77 %) per set, with no change in accuracy (−1, 0
and 0 correct).
The reason is the refit threshold. A label is refitted only after 25 new
examples, and a benchmark test set has few abstentions per label: 282
examples spread over the 77 BANKING77 labels and 351 over the 150 CLINC150
labels, so no label reached 25 and the only savings were cache hits. MASSIVE
gave 1333 examples over 60 labels: 28 attempts, 16 promoted (14 more questions
answered locally) and 12 rejected by the holdout, and the calls per third did
not fall steadily. Self-learning pays off on traffic that keeps coming back to
the same labels, not on one pass over a benchmark; these numbers do not show a
larger effect.

### 3. Pre-training on unlabelled dev traffic

A base model trained on the train split only (not the published model) was
pre-trained with `cortiq decision learn` on the dev texts without labels
(1498, 2998 and 2025 texts). Its gate rejected 220, 280 and 727 of them; all
were answered from stored DeepSeek ledgers. 59, 94 and 50 labels were promoted
and 2, 3 and 6 rejected; every gate stayed certified with the same τ. On the
test sets, pre-trained minus base:

| Δ (pre-trained − base) | BANKING77 | CLINC150 | MASSIVE |
|---|---|---|---|
| All rows correct | −10 | +23 | +4 |
| Answered by the gate | +17 | −10 | +15 |
| Correct among them | +11 | −7 | +10 |
| Static cascade correct | −5 | +1 | −6 |
| Oracle calls | −17 | +10 | −15 |

The effect is about neutral. The published model uses the labelled dev rows as
training data instead, so it is not pre-trained through the oracle; it keeps
learning from the oracle and from feedback on your own traffic.
