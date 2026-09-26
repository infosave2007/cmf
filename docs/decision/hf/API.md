# Cortiq Decision — API

`cortiq serve cortiq-decision.cmf` starts one HTTP server that speaks two
protocols at the same time:

* the **decisions protocol** of Jev / OpenRouter (`POST /api/alpha/decisions`),
  with an optional `cmf` extension;
* the **cortiq-router API**, schema `1.1` (`POST /v1/route` and the rest), so
  that existing router clients switch by changing the backend address only.

Both run the same local model and the same oracle cascade ([ORACLE.md](ORACLE.md)).
There is no web interface and no CORS layer. Request bodies are never logged:
a log line holds the request id, status, latency and account.

The command and `curl` blocks below were run in order, as written, against
local servers of the published `cortiq-decision.cmf` with cortiq 0.7.8; the
HTTP status each call returned is written after it as `# → 200`. Three parts
used stand-ins or were not run: the `mysql` export of section 8.1 was not run
(the import read a hand-written export of one test key and its counters); the
old router of section 8.3 was a local stand-in, a `cortiq serve` of the same
file holding that imported key at `127.0.0.1:8080`; and `cortiq decision
learn` in section 7 called a local mock of the OpenRouter API (`base_url`
pointed at it) instead of OpenRouter. The examples use `curl` and `jq`.

1. [Start a server](#1-start-a-server)
2. [Keys, plans and limits](#2-keys-plans-and-limits)
3. [Decisions API (Jev / OpenRouter protocol)](#3-decisions-api-jev--openrouter-protocol)
4. [cortiq-router API (schema 1.1)](#4-cortiq-router-api-schema-11)
5. [Administration](#5-administration)
6. [Configuration reference](#6-configuration-reference)
7. [Your own model](#7-your-own-model)
8. [Migrating from cortiq-router](#8-migrating-from-cortiq-router)

## 1. Start a server

Create an API key first. It is printed once; the state directory keeps only
its sha256.

```bash
export KEY=$(cortiq decision keys create --state ./decision.state \
  --plan developer --account acme --json | jq -r .key)
```

An optional configuration file (all fields are listed in
[section 6](#6-configuration-reference)):

```bash
cat > decision.json <<'EOF'
{
  "default_skill": "banking77",
  "task_complexity": {"card_arrival": 0.2, "compromised_card": 0.8},
  "routing_tiers": {"low": "my-small-model", "medium": "my-mid-model", "high": "my-large-model"}
}
EOF
```

Start the server. It stays in the foreground until Ctrl-C; the admin API
needs the token in the server's environment and in the shell that calls it.

```bash
export CORTIQ_DECISION_ADMIN_TOKEN=$(openssl rand -hex 24)
cortiq serve cortiq-decision.cmf --decision-config decision.json \
  --state ./decision.state --port 8080
```

A decision server listens on `127.0.0.1:8080` unless `--host` / `--port` say
otherwise. Other `cortiq serve` flags for decision files: `--break-lock`
(remove a stale `LOCK` of a dead process), `--shadow-of URL` and
`--shadow-timeout-s N` ([section 8](#8-migrating-from-cortiq-router)).
Language-model flags (`--task`, `--gpus`, …) are refused for a decision file.

```bash
export CORTIQ=http://127.0.0.1:8080
curl -s "$CORTIQ/healthz"                                    # → 200
```

The state directory (mode 0700) holds `LOCK`, `keys.json`, `usage/`,
`oracle.jsonl`, `oracle.state`, `learn.log`, `generations/` and `CURRENT`. One
server process per state directory.

## 2. Keys, plans and limits

* A key is `cortiq_` followed by 40 hex characters from the OS random
  generator. Send it as `Authorization: Bearer <key>` or `x-api-key: <key>`.
* The server re-reads `keys.json` when it changes (checked at most every 15 s),
  so keys created with the CLI while the server runs start working without a
  restart. Keys created through the admin API work at once.
* **Open mode** (no key needed) holds only while `keys.json` has no key and
  `auth.require` is false; `auth.require: null` (the default) means "required
  unless the server listens on loopback".
* A rate window is one fixed minute per account; quotas are checked before any
  work is done. A *decision* is one answered question.

| Plan | Requests per minute | Decision quota | Key lifetime |
|---|---|---|---|
| starter | 60 | none | 30 days |
| developer | 120 | 100000 | none |
| pro | 600 | 1000000 | none |
| scale | 3000 | 10000000 | none |

`auth.plans` in the configuration replaces entries of this table by name.
`cortiq decision keys create` also takes `--days`, `--rate-per-min`,
`--decision-quota`, `--token-quota`, `--credit-usd`, `--oracle-budget-usd`
and `--oracle-allowed` (0 means unlimited for the counters).

```bash
cortiq decision keys list --state ./decision.state
```

Keys through the admin API (router-compatible body; `x-admin-token`):

```bash
curl -s "$CORTIQ/v1/admin/keys" -H "x-admin-token: $CORTIQ_DECISION_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"account": "beta", "plan": "starter", "label": "trial"}' \
  | jq '{account, plan, rate_per_min, expires_at}'           # → 200
curl -s "$CORTIQ/v1/admin/keys" -H "x-admin-token: $CORTIQ_DECISION_ADMIN_TOKEN" \
  | jq '.keys[] | {account, plan, key_hash_prefix}'          # → 200
curl -s -X DELETE "$CORTIQ/v1/admin/keys/beta" \
  -H "x-admin-token: $CORTIQ_DECISION_ADMIN_TOKEN"           # → 200
```

The response of `POST /v1/admin/keys` is the only place the raw key appears.

## 3. Decisions API (Jev / OpenRouter protocol)

`POST /api/alpha/decisions` (also `POST /v1/decisions`), key required.

### 3.1 Request

| Field | Rules |
|---|---|
| `model` | `"cortiq/decision"` or `"cortiq/decision@<12 hex>"` of the served generation; anything else, a Jev name included, is 404 `MODEL_NOT_FOUND` |
| `state` | non-empty string, object or array; an object or array is used as its canonical JSON and the answers are then not certified; at most 32 KiB |
| `questions` | object of 1–32 questions in request order; ids up to 128 characters |
| question | `{type, instructions, criteria}`; `type` is `choice`, `score` or `noul`; `instructions` is required |
| choice `criteria` | object of 2–255 options; option ids 1–256 bytes; each description a string, object, array or `null` up to 24000 bytes; key order is kept |
| score `criteria` | array of 2–10 levels, lowest first |
| noul `criteria` | optional; an object with only `true` and/or `false` |
| `provider`, `user`, `session_id`, `trace` | OpenRouter's optional fields: accepted and ignored |
| `cmf` | extension, all optional: `skill`, `oracle` (bool), `allow_pii_egress` (bool), `round` (`2` or `null`), `explain` (bool), `profile` (`balanced`, `quality-first`, `cost-saver`) |

Any other key at the top level or inside `cmf`, or a key repeated inside any
object, is 400. The body is at most 1 MiB (413) and must be
`Content-Type: application/json`.

```bash
curl -s "$CORTIQ/api/alpha/decisions" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{
    "model": "cortiq/decision",
    "state": "I still have not received my new card",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "Classify the banking customer message.",
        "criteria": {
          "card_arrival": "The customer is waiting for a card to arrive",
          "card_delivery_estimate": "The customer asks when a card will be delivered",
          "lost_or_stolen_card": "The card was lost or stolen"
        }
      }
    }
  }' | jq                                                    # → 200
```

```json
{
  "id": "cmf-dec-1790462653-d5agU2C4o0BM69enIx7G",
  "created": 1790462653,
  "model": "cortiq/decision@386b6e43fd35",
  "provider": "Cortiq",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "card_arrival",
      "probabilities": {
        "card_arrival": 0.9999062,
        "card_delivery_estimate": 0.00003233644,
        "lost_or_stolen_card": 0.00006148422
      },
      "confidence": 0.99985933
    }
  },
  "usage": {"input_tokens": 59, "output_tokens": 3, "cost": 0.0},
  "cmf": {
    "generation": 0,
    "model_sha": "386b6e43fd35d5aa9e9710529dfdf30aa868df98ab4434d8e45dae9829898f85",
    "representation_id": "5e83d2345be08aaa526b63addd243bb511f877e1cbab3caab9749a53e9519c25",
    "timings_us": {"tokenize": 8, "encode": 2000, "hash": 5, "resonance": 2222, "oracle": 0, "total": 4287},
    "questions": {
      "intent": {
        "action": "local",
        "source": "local",
        "skill": "banking77",
        "match": "subset",
        "certified": false,
        "gate": {
          "accepted": true, "p_top": 0.9999062, "tau": 0.7, "novelty": 0.38166195,
          "theta": 0.804234, "is_novel": false, "margin": 0.1587196, "profile": "balanced"
        },
        "errors": {"card_arrival": 0.097457424, "lost_or_stolen_card": 0.32894346, "card_delivery_estimate": 0.3442838},
        "flags": [],
        "confident": true,
        "complexity": {
          "score": 0.15727276, "tier": "low",
          "factors": {"base": 0.2, "ambiguity": 0.00009381771, "novelty": 0.38166195, "margin": 0.0, "length": 0.2}
        },
        "routing": {"target": "my-small-model", "reason": "card_arrival @ complexity 0.16 (low)"},
        "decision_path": "router:uncertified_subset"
      }
    },
    "usage": {
      "local": {"input_tokens": 59, "output_tokens": 3, "processed_tokens": 10, "cost": 0.0},
      "oracle": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "billed": 0.0, "passthrough": true}
    }
  }
}
```

### 3.2 Skill matching and `certified`

The local model looks only at the option ids (instructions and descriptions
matter to the oracle alone). With L = the option ids of a choice question:

| Match | When | Decided by |
|---|---|---|
| `exact` | L equals the active labels of exactly one skill | that skill's certified gate |
| `subset` | L is a strict subset (at least 2) of one skill's labels | argmin, softmax, margin and novelty over L only; the same T, θ, τ; never certified |
| `superset` | one skill's labels plus labels it does not know | the oracle only; its answer teaches that skill |
| `untrained` | anything else, and every `score` / `noul` question | the oracle only; without it the request is 422 |

`cmf.skill` names the skill and skips the search (an unknown id is 400). An
answer is `certified: true` only for an exact match on a string `state`, the
`balanced` or `quality-first` profile, a skill whose gate is certified, and a
winning label that comes from the training data. The guarantee covers the
answers with `action: local`; oracle and cache answers are never certified.

An exact question is easiest to build from the skill's own rubric:

```bash
curl -s "$CORTIQ/v1/skills/banking77" -H "Authorization: Bearer $KEY" \
| jq '{model: "cortiq/decision", state: "I still have not received my new card",
       questions: {intent: {type: "choice", instructions: .rubric.instructions,
                            criteria: .rubric.criteria}}}' \
| curl -s "$CORTIQ/api/alpha/decisions" -H "Authorization: Bearer $KEY" \
    -H 'Content-Type: application/json' --data-binary @- \
| jq '{choice: .answers.intent.choice, confidence: .answers.intent.confidence,
       local: (.cmf.questions.intent | {action, match, certified, decision_path})}'
# → 200, 200
```

### 3.3 Response

`200` with `{id, created, model, provider, answers, usage, cmf}` and the header
`x-request-id` equal to `id`.

* **choice, `action: local` or `abstain`** — `{type, choice, probabilities,
  confidence}`: `probabilities` has every option in request order,
  `confidence = (N·p_max − 1)/(N − 1)` (Jev's formula). Numbers are the
  shortest f32 form; `cmf.round: 2` (or `response.round: 2`) rounds to
  hundredths and prints 0 and 1 as integers, as Jev does.
* **choice from the oracle or the cache** — `{type, choice}`.
* **score** (oracle only) — `{type, score, legend}` with the level index.
* **noul** (oracle only) — `{type, noul: 1 | 0, value_semantics:
  "boolean_verdict_not_probability"}`.

`cmf.questions.<id>` explains each answer:

| Field | Meaning |
|---|---|
| `action` | `local` (gate accepted), `abstain` (gate rejected and no oracle answer), `cache`, `oracle` |
| `source`, `skill`, `match`, `certified` | where the answer came from; section 3.2 |
| `gate` | `accepted`, `p_top` and `tau`, `novelty` and `theta`, `is_novel`, `margin`, `profile` |
| `errors` | reconstruction errors of the 5 best labels (all of them with `cmf.explain`) |
| `flags` | e.g. `oracle_disabled`, `consent_off`, `budget`, `stopped`, `oracle_unavailable`, `pii_redacted` |
| `confident` | the answer can be used as is (gate accepted and not novel, or a valid oracle answer) |
| `complexity` | `{score, tier, factors: {base, ambiguity, novelty, margin, length}}`, the cortiq-router formula |
| `routing` | `{target, reason}` when `routing_tiers` maps the tier |
| `decision_path` | `router:certified`, `router:uncertified`, `router:uncertified_subset`, `escalate→cache`, `escalate→oracle`, `escalate→oracle_unavailable`, `escalate→disabled` |
| `explanation` | with `cmf.explain`: `{top1_vs_top2, decision_path}` |

Complexity: `score = Σ weight·factor` with base = `task_complexity[label]`
(default 0.4), ambiguity = 1 − p_top, novelty, margin = 1 − clamp(8·margin, 0, 1),
length = clamp(words/40, 0, 1); weights 0.40 / 0.25 / 0.15 / 0.10 / 0.10;
tiers low ≤ 0.33 < medium ≤ 0.66 < high.

**Profiles** (`cmf.profile`): `balanced` (default) is the certified gate
`p_top ≥ τ and novelty ≤ θ`; `quality-first` also needs `margin ≥ 0.08` and
`novelty ≤ min(θ, 0.50)` (more abstentions, fewer errors); `cost-saver` uses
`novelty ≤ θ` only and is never certified.

```bash
curl -s "$CORTIQ/api/alpha/decisions" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' -d '{
    "model": "cortiq/decision",
    "state": "someone used my card without my permission",
    "questions": {"intent": {"type": "choice", "instructions": "Classify the message.",
      "criteria": {"compromised_card": null, "card_payment_not_recognised": null, "lost_or_stolen_card": null}}},
    "cmf": {"explain": true, "profile": "quality-first", "round": 2}
  }' | jq '{answer: .answers.intent, explanation: .cmf.questions.intent.explanation,
            complexity: .cmf.questions.intent.complexity.tier, routing: .cmf.questions.intent.routing}'
# → 200
```

### 3.4 Usage and price

* `usage.input_tokens` counts WordPiece tokens of everything the model was
  given, as Jev counts them: the `state` plus each question's `instructions`
  and `criteria` (keys and values). An exact question over a 77-label rubric
  therefore counts thousands of tokens although the encoder reads only the
  state; `cmf.usage.local.processed_tokens` is what the encoder read.
* `usage.output_tokens` is the number of values in `probabilities` (1 for an
  oracle, cache or noul answer).
* `usage.cost = input·input_usd_per_1m/1e6 + output·output_usd_per_1m/1e6 +
  request_usd + [oracle passthrough] oracle cost × markup`. **All prices are
  `"0"` by default**, so local answers cost 0 and oracle answers cost what
  OpenRouter charged. Prices are decimal strings in `pricing`.
* Errors are not billed. `GET /v1/usage` shows the caller's own account.

### 3.5 Errors

`{"error": {"code": <HTTP>, "message": "…", "metadata": {"reason": "<CODE>",
"retriable": bool, "request_id": "…", "details": {…}}}}` — OpenRouter's shape
with cortiq-router's codes.

| HTTP | reason | When |
|---|---|---|
| 400 | `INVALID_REQUEST` | schema, limits, duplicate or unknown keys |
| 401 | `UNAUTHORIZED` | key missing, wrong, expired or revoked |
| 402 | `QUOTA_EXCEEDED` | decision or token quota, credit |
| 404 | `MODEL_NOT_FOUND`, `INVALID_REQUEST`, `ADMIN_DISABLED` | unknown model; feedback target not found; admin token not configured |
| 413 | `PAYLOAD_TOO_LARGE` | body over `limits.body_bytes` |
| 422 | `UNSUPPORTED_QUESTION` | untrained question and the oracle not allowed |
| 429 | `RATE_LIMITED`, `OVERLOADED` | the minute window; more than `max_inflight` requests (`Retry-After`) |
| 500 | `INTERNAL` | |
| 502 | `ORACLE_UNAVAILABLE` | untrained question, the oracle call failed |
| 503 | `ORACLE_BUDGET_EXHAUSTED`, `ORACLE_DISABLED` | untrained question, no budget or a stop rule |

A trained question whose oracle call fails is never an error: it is answered
locally with `action: abstain` and the flag `oracle_unavailable`.

```bash
# Jev's model id is not served here
curl -s "$CORTIQ/api/alpha/decisions" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "typesafe/jev-1.13", "state": "x",
       "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": null, "b": null}}}}' \
  | jq .error.metadata.reason                                # → 404
# score and noul questions need the oracle
curl -s "$CORTIQ/api/alpha/decisions" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "cortiq/decision", "state": "The parcel arrived two weeks late.",
       "questions": {"urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
                     "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}}}' \
  | jq .error.metadata                                       # → 422
# no key
curl -s "$CORTIQ/api/alpha/decisions" -H 'Content-Type: application/json' \
  -d '{"model": "cortiq/decision", "state": "x",
       "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": null, "b": null}}}}' \
  | jq .error.metadata.reason                                # → 401
```

### 3.6 Feedback, listings, health

`POST /v1/feedback {"id", "question", "label"}` corrects a decision of the
caller's own account (a decision of another account is not found). The label
must be one of the question's options; it becomes a training example of
weight 3, and an option the skill does not have starts a cold start
([ORACLE.md](ORACLE.md#self-learning)). The router form of feedback
(section 4) accepts any label.

```bash
ID=$(curl -s "$CORTIQ/api/alpha/decisions" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "cortiq/decision", "state": "where is the card you sent me",
       "questions": {"intent": {"type": "choice", "instructions": "Classify the message.",
         "criteria": {"card_arrival": null, "card_delivery_estimate": null}}}}' | jq -r .id)   # → 200
curl -s "$CORTIQ/v1/feedback" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"id\": \"$ID\", \"question\": \"intent\", \"label\": \"card_arrival\"}"             # → 200
```

| Path | Access | Returns |
|---|---|---|
| `GET /v1/models` | open | the model in the shape of an OpenRouter provider listing (`id`, `pricing` as USD-per-token strings, `context_length` 512, `max_output_length` 255) plus `cmf.skills` with each gate |
| `GET /v1/skills`, `GET /v1/skills/{id}` | key | labels, gate, rubric (`instructions`, `criteria`) |
| `GET /v1/usage` | key | the caller's account (router format; `x-cmf-extensions: 1` adds token and cost totals) |
| `GET /healthz` | open | status, model, generation, skills, oracle on/off |

`/v1/models` only has the listing's shape; Cortiq Decision is not listed on
OpenRouter.

```bash
curl -s "$CORTIQ/v1/models" \
  | jq '.data[0] | {id, pricing, skills: [.cmf.skills[] | {id, labels, certified, tau}]}'   # → 200
curl -s "$CORTIQ/v1/skills" -H "Authorization: Bearer $KEY" \
  | jq '[.skills[] | {id, labels: (.labels | length), certified}]'                          # → 200
curl -s "$CORTIQ/v1/usage" -H "Authorization: Bearer $KEY" -H 'x-cmf-extensions: 1' \
  | jq '{account, totals: .cmf.totals}'                                                     # → 200
```

## 4. cortiq-router API (schema 1.1)

The same server answers the router's API with exactly its keys, JSON types and
error envelope, so existing clients need no change:

| Path | Access |
|---|---|
| `POST /v1/route`, `POST /v1/route:batch` (up to 1024 inputs) | key |
| `POST /v1/feedback` `{request_id, correct_task_label}` | key |
| `GET /v1/taxonomies`, `GET /v1/taxonomies/{id}` | key |
| `GET /v1/usage`, `GET /v1/escalations?limit=N` | key (own account only) |
| `GET /v1/healthz`, `GET /v1/readyz`, `GET /metrics` (Prometheus) | open |
| `POST /v1/admin/keys`, `GET /v1/admin/keys`, `DELETE /v1/admin/keys/{account}` | `x-admin-token` |

* `taxonomy_id` is a skill id (`banking77`, `clinc150`, `massive`, or your
  own); without it the server uses `default_skill`, else the file's only skill.
* `options`: `policy_profile` (`balanced`; also `cost-saver`, `quality-first`),
  `allow_oracle` (`true`: consent to the oracle for this input if it is
  undetermined), `allow_pii_egress` (`false`), `top_k` (3, 1–64),
  `return_explanation` (`false`), `routing_table_id` (adds `routing` when
  `routing_tiers` is configured).
* `decision.confidence` is the calibrated `p_top` (1 for an oracle or cache
  answer), `raw_confidence` the winner's `1/(1+E)`; `scores[]` carry
  `probability`, `score` and `reconstruction_error`. A rejected answer that is
  not escalated has the flag `low_confidence` and `confident: false`.
* `input.embedding` (bring your own vector) must have the signal's dimension
  (4480) and `embedding_model`; it is decided locally only.
* The header `x-cmf-extensions: 1` adds a `cmf` object (the `cmf-dec-…` id,
  action, `certified`, gate) to results, listings and errors. Without it the
  responses have the router's keys only.

```bash
curl -s "$CORTIQ/v1/route" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input": {"text": "I still have not received my new card"},
       "taxonomy_id": "banking77", "options": {"return_explanation": true}}' | jq   # → 200
```

```json
{
  "schema_version": "1.1",
  "request_id": "req_18d9004c29a85ce8000005",
  "decision": {
    "task_id": 12,
    "task_label": "card_arrival",
    "taxonomy_id": "banking77",
    "confidence": 0.9980738,
    "confident": true,
    "raw_confidence": 0.91119707,
    "margin": 0.116146445,
    "is_novel": false,
    "novelty_score": 0.40156537,
    "complexity": {
      "score": 0.1677992,
      "tier": "low",
      "factors": {"base": 0.2, "ambiguity": 0.0019261837, "novelty": 0.40156537, "margin": 0.07082844, "length": 0.2}
    },
    "source": "router",
    "flags": []
  },
  "scores": [
    {"task_id": 12, "task_label": "card_arrival", "probability": 0.9980738, "score": 0.91119707, "reconstruction_error": 0.097457424},
    {"task_id": 14, "task_label": "card_linking", "probability": 0.001209337, "score": 0.7950506, "reconstruction_error": 0.25778145},
    {"task_id": 1, "task_label": "activate_my_card", "probability": 0.00019758825, "score": 0.76862156, "reconstruction_error": 0.30103046}
  ],
  "explanation": {"top1_vs_top2": "card_arrival leads card_linking by 0.116 score", "decision_path": "router:certified"},
  "usage": {"billable_decisions": 1, "oracle_calls": 0},
  "meta": {
    "model_version": "cortiq/decision@386b6e43fd35",
    "taxonomy_version": "banking77@1",
    "latency_ms": 1.234917,
    "embedding_latency_ms": 1.986042,
    "served_by": "cortiq/0.7.8"
  }
}
```

With the configuration of section 1 (`default_skill`, `task_complexity`,
`routing_tiers`), a route without `taxonomy_id` goes to `banking77` and names
the client's model for the complexity tier:

```bash
curl -s "$CORTIQ/v1/route" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input": {"text": "someone used my card without my permission"},
       "options": {"routing_table_id": "default"}}' \
  | jq '{label: .decision.task_label, tier: .decision.complexity.tier, routing}'   # → 200
curl -s "$CORTIQ/v1/route:batch" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"taxonomy_id": "clinc150",
       "inputs": [{"text": "what will the weather be tomorrow"}, {"text": "set an alarm for 7 am"}]}' \
  | jq '[.results[].decision | {task_label, confident}]'                           # → 200
```

Feedback, listings and probes:

```bash
RID=$(curl -s "$CORTIQ/v1/route" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input": {"text": "can I get a card for my teenager"}, "taxonomy_id": "banking77"}' \
  | jq -r .request_id)                                                             # → 200
curl -s "$CORTIQ/v1/feedback" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"request_id\": \"$RID\", \"correct_task_label\": \"age_limit\"}"           # → 200
curl -s "$CORTIQ/v1/taxonomies" -H "Authorization: Bearer $KEY" \
  | jq '[.taxonomies[] | {taxonomy_id, taxonomy_version, labels: (.labels | length)}]'   # → 200
curl -s "$CORTIQ/v1/usage" -H "Authorization: Bearer $KEY" | jq .usage          # → 200
curl -s "$CORTIQ/v1/escalations?limit=5" -H "Authorization: Bearer $KEY" | jq .summary   # → 200
curl -s "$CORTIQ/v1/healthz"                                                     # → 200
curl -s "$CORTIQ/v1/readyz"                                                      # → 200
curl -s "$CORTIQ/metrics" | grep '^cortiq_decisions_total'                       # → 200
curl -s "$CORTIQ/v1/route" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input": {"text": "hello"}, "taxonomy_id": "data-assistant"}' | jq .error.code   # → 404
```

Router errors keep the router's envelope: `{"schema_version": "1.1",
"request_id": "req_…", "error": {"code": "<CODE>", "message": "…",
"retriable": bool, "details": null}}` with `INVALID_REQUEST` /
`EMBEDDING_REQUIRED` (400), `UNAUTHORIZED` (401), `QUOTA_EXCEEDED` (402),
`TAXONOMY_NOT_FOUND` (404, `details: {"taxonomy_id"}`), `RATE_LIMITED` (429);
`retriable` is true only for 429 and 500. A body the router's own framework
rejects is answered as the router answers it, in plain text: a non-JSON
content type 415, a body over `limits.body_bytes` 413, malformed JSON 400,
wrong field types 422.

## 5. Administration

Every admin path needs `x-admin-token` equal to the environment variable named
by `auth.admin_token_env` (`CORTIQ_DECISION_ADMIN_TOKEN`); when the variable is
unset the admin API answers 404 `ADMIN_DISABLED`.

| Path | Does |
|---|---|
| `POST / GET /v1/admin/keys`, `DELETE /v1/admin/keys/{account}`, `DELETE /v1/admin/keys/hash/{hash12}` | create (raw key returned once), list, revoke |
| `GET /v1/admin/usage` | usage of every account |
| `GET /v1/admin/oracle`, `POST /v1/admin/oracle {"enabled", "budget_usd", "max_calls"}` | oracle status (`configured` = `oracle.enabled` of the configuration, `enabled` = not switched off by a stop rule or the admin, `key_present`, spent, calls, stop reason); switch it and lower limits within the configuration |
| `GET /v1/admin/learning` | buffer, cache, quarantine, attempts, promotions, recent events |
| `GET /v1/admin/generations`, `POST /v1/admin/rollback {"generation": N}` | generations; serve generation N (0 = the base file) |
| `GET /v1/admin/shadow` | agreement statistics in shadow mode (section 8) |

```bash
A="x-admin-token: $CORTIQ_DECISION_ADMIN_TOKEN"
curl -s "$CORTIQ/v1/admin/usage" -H "$A" | jq 'keys'                          # → 200
curl -s "$CORTIQ/v1/admin/oracle" -H "$A" | jq '{enabled, configured, key_present, budget_usd, spent_usd}'   # → 200
curl -s "$CORTIQ/v1/admin/learning" -H "$A" | jq '{generation, buffer: .buffer.examples, promotions}'       # → 200
curl -s "$CORTIQ/v1/admin/generations" -H "$A" | jq '{current, model}'        # → 200
curl -s "$CORTIQ/v1/admin/rollback" -H "$A" -H 'Content-Type: application/json' \
  -d '{"generation": 0}' | jq                                                   # → 200
```

With the server stopped, the same is available from the CLI:
`cortiq decision rollback --state DIR --to N --model FILE`, and
`cortiq decision materialize FILE --state DIR -o OUT.cmf` writes the served
generation as one self-contained file (this also works while a server runs).

## 6. Configuration reference

`--decision-config FILE` takes a JSON object; every field is optional and an
unknown key is an error. The defaults:

```json
{
  "state_dir": null,
  "default_skill": null,
  "auth": {"require": null, "admin_token_env": "CORTIQ_DECISION_ADMIN_TOKEN", "key_prefix": "cortiq_"},
  "limits": {"body_bytes": 1048576, "state_bytes": 32768, "questions": 32, "max_inflight": 64},
  "pricing": {"input_usd_per_1m": "0", "output_usd_per_1m": "0", "request_usd": "0",
              "oracle_passthrough": true, "oracle_markup": 1.0},
  "response": {"round": null},
  "oracle": {"enabled": false, "default_per_request": true,
             "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY",
             "model": "deepseek/deepseek-v4.1-flash",
             "provider": {"sort": "price", "require_parameters": true, "allow_fallbacks": true,
                          "max_price": {"prompt": 0.1, "completion": 0.5}},
             "max_tokens_per_question": 64, "deadline_s": 30, "budget_usd": 1.0, "max_calls": 10000,
             "max_errors": 30, "redact_pii": true, "title": "cortiq-decision", "data_collection": null},
  "cache": {"enabled": true, "threshold": 0.97, "cap": 50000},
  "learning": {"enabled": true, "refit_min_new": 25, "dedup": 0.995, "cold_start": true, "synchronous": false},
  "feedback": {"pending_cap": 50000},
  "complexity_weights": {"base": 0.4, "ambiguity": 0.25, "novelty": 0.15, "margin": 0.1, "length": 0.1},
  "complexity_tiers": [{"tier": "low", "max": 0.33}, {"tier": "medium", "max": 0.66}, {"tier": "high", "max": 1.0}],
  "task_complexity": {},
  "routing_tiers": {}
}
```

`state_dir: null` means `<FILE>.state` next to the model. Secrets are never
part of the file: the admin token and the OpenRouter key are read from the
environment variables it names. `auth.plans` may override the plan table.
The oracle, cache and learning sections are explained in [ORACLE.md](ORACLE.md).

## 7. Your own model

A skill is trained from JSONL rows `{"text", "label"}`: a non-empty text up to
32 KiB, a label of 1–256 bytes, no other keys. Duplicates and conflicting
labels are kept (conflicts are reported). A label with a single training row
is kept inactive.

```bash
# the router's datasets_dir layout: one directory per label, *.txt lines
for d in datasets/*/; do
  label=$(basename "$d")
  cat "$d"*.txt | jq -Rc --arg l "$label" 'select(length > 0) | {text: ., label: $l}'
done > train.jsonl
cat > question.json <<'EOF'
{
  "instructions": "Classify the task type of the user request.",
  "criteria": {
    "chitchat": "Small talk and greetings",
    "code": "Writing, fixing or explaining code",
    "creative-writing": "Stories, poems and other creative text",
    "extraction": "Pulling structured facts out of a text",
    "math": "Solving a math problem",
    "qa": "A factual question",
    "summarization": "Shortening a text",
    "translation": "Translating between languages"
  }
}
EOF
```

`train` takes the encoder of any decision file byte for byte and adds one
skill; `add-skill` copies every existing skill byte for byte and adds a new one
(an existing id is refused). Outputs are never overwritten.

```bash
cortiq decision train --encoder cortiq-decision.cmf --skill data-assistant \
  --train train.jsonl --question question.json -o data-assistant.cmf
cortiq decision add-skill cortiq-decision.cmf --skill data-assistant \
  --train train.jsonl --question question.json -o cortiq-decision-plus.cmf
cortiq decision info data-assistant.cmf
cortiq decision verify data-assistant.cmf
cortiq decide data-assistant.cmf -p "Write a Python function that merges two sorted lists"
```

* **Skill id**: `[a-z0-9][a-z0-9_-]{0,63}`; it is the `taxonomy_id` of the
  router API.
* **Several training files**: repeat `--train` (the published file was
  trained on train and dev).
* **Calibration**: `--calibration FILE`, or carved out of the training rows
  (inside each label, rows in sha256 order, every fifth row). The calibration
  rows are split by sha256 order into halves: the even half sets the
  temperature T and the novelty threshold θ (95th percentile), the odd half
  picks τ from a fixed grid of 14 thresholds. The gate is **certified** when a
  threshold keeps at least 100 accepted odd-half rows with a Clopper–Pearson
  lower bound ≥ 0.95 at α = 0.05/14; the one accepting the most rows wins.
  Otherwise the skill is served with θ only and `certified: false` (the small
  example above has 160 rows and is not certified). 20 % of the calibration
  rows are held out for self-learning (champion/challenger).
* **K** (`--k`, default 16) is the most directions per label (a label with n
  rows gets min(K, n − 1)). The published skills use K chosen by 5-fold
  cross-validation on train ∪ dev: 32 (banking77), 16 (clinc150),
  24 (massive).
* `--dev FILE` records dev accuracy in the skill; `--threads N` changes speed,
  never the result; `--json` prints the build report.
* After writing, the file is re-opened and the calibration errors, confidence
  and novelty are recomputed and compared bit for bit.

**Pre-training through the oracle** on unlabelled traffic: only the texts the
gate rejects are asked; answers already in driver ledgers are reused by
request sha256 (`--answers`). Each label with new examples is refitted,
checked on the holdout, and the gate is re-certified; if the certified gate
would be lost every promotion of the skill is undone. The effect measured on
the three public sets was neutral ([ORACLE.md](ORACLE.md#measured-effect)).

```bash
cat > traffic.jsonl <<'EOF'
{"text": "Could you tidy up this paragraph so it reads better"}
{"text": "What is the capital of Australia"}
{"text": "my flight got cancelled what now"}
{"text": "the invoice total looks wrong"}
{"text": "Book a table for two at 8 pm"}
EOF
cat > oracle.json <<'EOF'
{
  "oracle": {
    "enabled": true,
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_API_KEY",
    "model": "deepseek/deepseek-v4.1-flash",
    "budget_usd": 0.05,
    "max_calls": 100
  }
}
EOF
export OPENROUTER_API_KEY="<your OpenRouter key>"
cortiq decision learn data-assistant.cmf --traffic traffic.jsonl \
  --oracle-config oracle.json -o data-assistant-learned.cmf
```

The key is read from the variable `api_key_env` names
([ORACLE.md](ORACLE.md#connect-openrouter)); every live call is reserved in
`<OUTPUT>.oracle.jsonl` before it is sent.

Batch evaluation and timing (`--input` rows `{"text", "label"?}`; one JSON
result per row, never the text; totals on stderr):

```bash
cortiq decide data-assistant.cmf --input train.jsonl --bench --out rows.jsonl
```

## 8. Migrating from cortiq-router

### 8.1 Keys

Existing keys keep working without reissue: the router stores `sha256(raw)`
and so does this server. Export the router's `api_keys` table (and, for
continuous quotas, `usage_counters`) as JSON lines:

```text
mysql -N -B -r -e "SELECT JSON_OBJECT('key_hash',key_hash,'account',account,'plan',plan,
  'label',label,'active',active,'rate_per_min',rate_per_min,'decision_quota',decision_quota,
  'expires_at',expires_at,'created_at',created_at) FROM api_keys" cortiq > api_keys.jsonl
mysql -N -B -r -e "SELECT JSON_OBJECT('account',account,'decisions',decisions,
  'oracle_calls',oracle_calls) FROM usage_counters" cortiq > usage_counters.jsonl
```

```bash
cortiq decision keys import --state ./router.state --from api_keys.jsonl \
  --usage usage_counters.jsonl
```

* Also accepted: MySQL Shell, Workbench and phpMyAdmin JSON exports, and the
  `[[api_keys]]` of the router's TOML configuration (`--format router-toml`;
  raw keys are hashed as they are read).
* The import is idempotent, checks every row before writing anything, never
  overwrites or re-activates a stored key, and prints no key or hash.
* Imported keys may use the oracle (`oracle_allowed: true`, as every key could
  in the router); `--oracle-allowed=false` imports them without it.
* `--usage` writes the usage ledger: run it while no server holds the state
  directory.

### 8.2 The taxonomy

A router taxonomy becomes a skill with the same id, trained from the router's
datasets (section 7 shows the conversion of a `datasets_dir`). The router's
self-learned examples are vectors of its own encoder (the router keeps no
texts), so they cannot be carried over; the new skill learns again from the
oracle and from feedback on live traffic.

### 8.3 Shadow mode, switch and rollback

Run the new server next to the old router with `--shadow-of URL`. Every request
to a router path is forwarded unchanged to the old router, with the client's
own `Authorization` header, and its answer goes back to the client byte for
byte (errors included), so clients see no change. `/v1/route` and
`/v1/route:batch` are also decided locally (no oracle, no learning, no
billing) and compared line by line in `<state>/shadow.jsonl` (labels,
confidence, latency and the text's sha256, never the text).

`--shadow-of` reaches the old router over **https**, or over plain http **only
at a loopback address** (`127.0.0.1`, `::1`, `localhost`), because the
clients' keys pass through it. cortiq-router itself speaks plain http, with
TLS at nginx, so run the new server on the router's host and point it at the
router's own port there. The admin token of section 1 must be in its
environment as well.

```bash
# on the router's host; the old router listens on port 8080
cortiq serve cortiq-decision-plus.cmf --state ./router.state --port 8090 \
  --shadow-of http://127.0.0.1:8080
```

A router on another machine is named by an https address in front of it
(`--shadow-of https://router-a.internal`); `--shadow-of http://router-a:8080`
is refused. `$ROUTER_KEY` below is the key of one of your router clients: it
was imported in 8.1, so both the old router and the new server accept it.

```bash
export SHADOW=http://127.0.0.1:8090
curl -s "$SHADOW/v1/route" -H "Authorization: Bearer $ROUTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input": {"text": "Write a haiku about autumn"}, "taxonomy_id": "data-assistant"}' \
  | jq .decision.task_label                                                     # → 200
curl -s "$SHADOW/v1/admin/shadow" -H "x-admin-token: $CORTIQ_DECISION_ADMIN_TOKEN" \
  | jq '{lines, compared, agree, agreement, confident, latency_ms}'            # → 200
```

nginx on the same host, in place of the `upstream cortiq` block of the
router's `deploy/nginx/nginx.conf`:

```nginx
upstream cortiq {
    least_conn;
    server 127.0.0.1:8090;           # the new server: in shadow mode, then after the switch
    server 127.0.0.1:8080 backup;    # the old router: only while the new server restarts
    keepalive 64;
}
```

With nginx on another machine, as in the router's own `deploy/nginx`
(`server router-a:8080`), start the new server with `--host 0.0.0.0` as well
(it listens on `127.0.0.1` by default) and write `router-a:8090` and
`router-a:8080 backup` in the upstream; `--shadow-of` stays
`http://127.0.0.1:8080`. One state directory serves one process, so quotas
and usage are counted per server.

**1. Shadow.** Start the shadow server, point the upstream at it as above and
reload nginx. Clients are still answered by the old router. Watch
`GET /v1/admin/shadow` (overall agreement, agreement when both sides are
confident, per-label agreement, latency).

**2. Switch** only when the agreement and your own spot checks are good
enough. Stop the shadow server (Ctrl-C) and start it again on the same state
directory and port without `--shadow-of`. Keys and usage are already there,
and nginx sends requests to the old router (`backup`) while the new server
loads; nginx itself needs no change.

```bash
cortiq serve cortiq-decision-plus.cmf --state ./router.state --port 8090
```

```bash
curl -s "$SHADOW/v1/route" -H "Authorization: Bearer $ROUTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input": {"text": "Translate good morning into French"}, "taxonomy_id": "data-assistant"}' \
  | jq '{label: .decision.task_label, source: .decision.source}'                 # → 200
curl -s "$SHADOW/v1/usage" -H "Authorization: Bearer $ROUTER_KEY" | jq .usage   # → 200
```

**3. Rollback.** Leave `server 127.0.0.1:8080;` as the only line of the
upstream and reload nginx; the old router was never changed. Decisions the
new server counted after the switch stay in its own usage ledger, not in the
router's `usage_counters`.
