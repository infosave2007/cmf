# Cortiq Decision — API

`cortiq serve cortiq-decision.cmf` starts one HTTP server with two native
protocols and, when explicitly enabled, a third request adapter:

* the **Cortiq decisions protocol** in the Jev / OpenRouter-shaped form
  (`POST /api/alpha/decisions`), with an optional `cmf` extension;
* the **cortiq-router API**, schema `1.1` (`POST /v1/route` and the rest), so
  that existing router clients switch by changing the backend address only;
* `POST /v1/systemone`, the opt-in TypeSafe System One / Jev-compatible
  request format (`--jev-compatible`). It always identifies its result as a
  local CMF model, not as Jev.

Both run the same local model and the same oracle cascade ([ORACLE.md](ORACLE.md)).
There is no web interface and no CORS layer. Request bodies are never logged:
a log line holds the request id, status, latency and account.

The command and `curl` blocks below were run in order, as written, against
local servers of the published `cortiq-decision.cmf` with cortiq 0.7.8; the
HTTP status each call returned is written after it as `# → 200`. Four parts
used stand-ins or were not run: the `mysql` export of section 8.1 was not run
(the import read a hand-written export of one test key and its counters); the
old router of section 8.3 was a local stand-in, a `cortiq serve` of the same
file holding that imported key at `127.0.0.1:8080`; `cortiq decision
learn` in section 7 called a local mock of the OpenRouter API (`base_url`
pointed at it) instead of OpenRouter; and so did the oracle commands of
[Check your setup](#check-your-setup) in section 1 (`--base-url`, for
`decide` `--oracle-base-url`, pointed at the mock, with a test key). The
examples use `curl` and `jq`.

1. [Start a server](#1-start-a-server)
2. [Keys, plans and limits](#2-keys-plans-and-limits)
3. [Decisions API (Jev / OpenRouter protocol)](#3-decisions-api-jev--openrouter-protocol)
3a. [System One request adapter (Jev-compatible)](#3a-system-one-request-adapter-jev-compatible)
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

### CPU, Metal or Vulkan

From Cortiq 0.8.0, the default crates.io CLI includes GPU decision support.
Set the device **before starting the process**:

```bash
# Apple Silicon
CORTIQ_DECISION_DEVICE=metal cortiq serve cortiq-decision.cmf \
  --state ./decision.state --port 8080

# A hardware Vulkan device and driver
CORTIQ_DECISION_DEVICE=vulkan cortiq serve cortiq-decision.cmf \
  --state ./decision.state --port 8080
```

Use only one server per state directory. CPU remains the default. On multi-GPU
hosts set `CORTIQ_DECISION_VULKAN_ADAPTER` to a unique part of the adapter name.
`GET /healthz` identifies the encoder device and counts completed GPU submissions.
The decisions endpoints and `/v1/route` use the same accelerated path; a device
error is reported, not silently sent to CPU. [Measurements and limits](GPU.md).

A decision server listens on `127.0.0.1:8080` unless `--host` / `--port` say
otherwise. Other `cortiq serve` flags for decision files: `--break-lock`
(only where the state directory's filesystem has no advisory locks: remove a
stale `LOCK` of a dead process), `--shadow-of URL` and
`--shadow-timeout-s N` ([section 8](#8-migrating-from-cortiq-router)), and
`--oracle MODEL` with its companions `--oracle-budget`, `--oracle-max-calls`,
`--oracle-max-price`, `--oracle-key-env`, `--oracle-base-url` and
`--no-oracle-learning`: the oracle in two steps, the OpenRouter key in
`OPENROUTER_API_KEY` and then this flag
([ORACLE.md](ORACLE.md#connect-openrouter)). Add `--jev-compatible` to expose
the separate System One adapter described in [section 3a](#3a-system-one-request-adapter-jev-compatible).
Language-model flags (`--task`, `--gpus`, …) are refused for a decision file.

```bash
export CORTIQ=http://127.0.0.1:8080
curl -s "$CORTIQ/healthz"                                    # → 200
```

The state directory (mode 0700) holds `LOCK`, `keys.json`, `usage/`,
`oracle.jsonl`, `oracle.state`, `learn.log`, `generations/` and `CURRENT`
(and in shadow mode `shadow.jsonl` with its key `shadow.key`). One server
process per state directory; the CLI may change `keys.json` while it runs
(both take `keys.json.lock` for each change). The server holds an advisory
lock (`flock`) on `LOCK` while it runs, and the file names its pid; the
lock ends with the process however it ends, so a `LOCK` left by a server
that was killed (SIGKILL after a stop timeout, an out-of-memory kill, a
crash of the host) is taken over by the next start, with a warning naming
that pid: a restart policy such as `restart: unless-stopped` needs no
`--break-lock`, also in a container where the server is always pid 1. The
lock of a running process is never broken. Versions up to 0.7.8 do not hold
the lock: do not run one on the same directory at the same time.

<a id="check-your-setup"></a>**Check your setup.** Whether the oracle is
ready can be checked before a server with `--oracle MODEL` is started or
anything is spent:

```bash
cortiq decision oracle check               # free: the key, the account, the model
cortiq decision oracle check --test-call   # and one tiny structured call, a small fraction of a cent
```

Each check prints ✓ or ✗ with what to do; the exit code is 0 only when the
oracle is ready (`--json`: `ready` and `problems[]`). A server reports its
oracle as `oracle_status` of `GET /healthz` and `status` of `GET
/v1/admin/oracle` ([section 5](#5-administration)), `cortiq decide …
--json` as `cmf.oracle.status`:

| Status | What to do |
|---|---|
| `ready` | nothing: what the gate rejects goes to the oracle |
| `no_key` | `export OPENROUTER_API_KEY=…` where the server (then restart it) or `decide` runs |
| `bad_key` | fix the variable as the message says (a quote, `Bearer `, a `NAME=…` line, whitespace or a non-ASCII byte in it); nothing was sent |
| `disabled` | start the server with `--oracle MODEL`; if the admin switched it off, `POST /v1/admin/oracle {"enabled": true}` |
| `budget_too_small` | `--oracle-budget` of at least the figure the message names (and `--oracle-max-calls` of 1 or more) |
| `budget_exhausted` | a larger `--oracle-budget` or `--oracle-max-calls` (a server counts its whole ledger; each `decide` run has its own budget) |
| `stopped: <reason>` | fix the cause (`http_401` / `http_403` the key was refused, `http_402` no credit, …), then `POST /v1/admin/oracle {"enabled": true}`, or one `decide` run with `--oracle-resume` |

An admin limit kept in `oracle.state` is lifted by the `POST
/v1/admin/oracle` the message names, not by a restart. One text or a batch
from the command line (only what the gate rejects is sent; the ledger and
the cache are in `<FILE>.state`, or `--state DIR`):

```bash
cortiq decide cortiq-decision.cmf --skill banking77 -p "the exchange rate you gave me looks wrong" \
  --oracle deepseek/deepseek-v4.1-flash
printf '%s\n' '{"text": "I still have not received my new card", "label": "card_arrival"}' \
  '{"text": "can I pay my rent with a virtual card"}' > rows.jsonl
cortiq decide cortiq-decision.cmf --skill banking77 --input rows.jsonl --out results.jsonl \
  --oracle deepseek/deepseek-v4.1-flash --oracle-budget 0.05
jq -c '{answer, action, source, oracle_cost_usd, flags}' results.jsonl
```

Each batch row adds `answer`, `action` (`local`, `oracle`, `cache` or
`abstain`), `source`, `oracle_cost_usd` and `flags` to the local columns;
`--oracle-resume` turns the oracle on again after a stop rule is fixed, and
a `LOCK` a killed run left is taken over by the next one
([ORACLE.md](ORACLE.md#one-text-or-a-batch)).

## 2. Keys, plans and limits

* A key is `cortiq_` followed by 40 hex characters from the OS random
  generator. Send it as `Authorization: Bearer <key>` or `x-api-key: <key>`.
* The server re-reads `keys.json` when it changes (checked at most every 15 s),
  so keys created with the CLI while the server runs start working without a
  restart. Keys created through the admin API work at once.
* **Open mode** (no key needed) holds only while `keys.json` has no key and
  `auth.require` is false; `auth.require: null` (the default) means "required
  unless the server listens on loopback". The open caller may reach the
  oracle and teach the model when the configuration says
  `auth.require: false`. On a loopback address without that setting it may
  reach the oracle only when the server was started with `--oracle MODEL`
  (the operator enabled it explicitly), and it never teaches the model: a
  loopback address says nothing about the client (a reverse proxy on the
  same host forwards anyone), so its feedback is not learned and the
  oracle's answers to it are cached, not learned. Behind a reverse proxy
  set `auth.require: true`.
* A rate window is one fixed minute per account; quotas and credit are
  checked before any work is done, and again before every input of a
  `/v1/route:batch` after the first (402 at the input that finds them used
  up). A *decision* is one answered question: a `/v1/decisions` request
  with more questions than the decision quota has left is refused (402,
  `requested` in the details) before any work. An oracle call is made only
  when its reserved worst-case cost (times `oracle_markup`) fits in the
  credit left; otherwise the questions get the `budget` flag, as past the
  key's `oracle_budget_usd`. The token quota and the local prices are
  checked before the request only, so one request can still go past them
  by its own tokens and price; parallel requests of one account are each
  checked against the usage recorded before them.
* **Teaching the model** is a permission of its own, `learning_allowed`
  (default false for every key, created or imported): the key's feedback is
  learned (a label the skill does not have starts a cold start), and the
  oracle's answers to its questions become training examples. Without it,
  feedback is answered and consumed but not learned, and an oracle answer
  teaches the skill only when the question is exactly the skill's own (its
  rubric's instructions and criteria over all its labels, as `/v1/route`
  asks it). The skills are shared by every account.

| Plan | Requests per minute | Decision quota | Key lifetime |
|---|---|---|---|
| starter | 60 | none | 30 days |
| developer | 120 | 100000 | 30 days |
| pro | 600 | 1000000 | 30 days |
| scale | 3000 | 10000000 | 30 days |

This is cortiq-router's plan table. `auth.plans` in the configuration
replaces entries by name (`{"rate_per_min", "decision_quota", "days"}`,
`days: null` = no expiry); a plan it does not name is refused (400), where
the router minted a key without limits. `cortiq decision keys create` also
takes `--days` (0 = never expires), `--rate-per-min`, `--decision-quota`,
`--token-quota`, `--credit-usd`, `--oracle-budget-usd`, `--oracle-allowed`
and `--learning-allowed` (0 means unlimited for the counters); the admin
API takes the same fields. A key created with the CLI may use the oracle
(`oracle_allowed: true`, like an imported router key) unless it is created
with `--oracle-allowed=false`; the server's oracle switch, budget, the key's
`--oracle-budget-usd` and the stop rules still apply. A key created through
`POST /v1/admin/keys` has `oracle_allowed` only when the body sets it, as
in the router.

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
| `state` | string, object or array; an object or array is used as its canonical JSON and the answers are then not certified; at most 32 KiB. Empty (`""`, `{}`, `[]`, `null`): a *state-less* request, each question's `instructions` are its input ([section 3.2](#32-skill-matching-and-certified)) |
| `questions` | object of 1–32 questions in request order; ids up to 128 characters |
| question | `{type, instructions, criteria}`; `type` is `choice`, `score` or `noul`; `instructions` is required |
| choice `criteria` | object of 2–255 options; option ids 1–256 bytes; each description a string, object, array or `null` up to 24000 bytes; key order is kept |
| score `criteria` | array of 2–10 levels, lowest first |
| noul `criteria` | optional; an object with only `true` and/or `false` |
| `provider`, `user`, `session_id`, `trace` | OpenRouter's optional fields: accepted and ignored |
| `cmf` | extension, all optional: `skill`, `oracle` (bool), `allow_pii_egress` (bool), `round` (`2` or `null`), `explain` (bool), `profile` (`balanced`, `quality-first`, `cost-saver`) |

Any other key at the top level or inside `cmf`, or a key repeated inside any
object, is 400. A request over a size limit — the body, the state, a
state-less question's instructions, a description, the options of a choice,
the questions — is a *capacity* error: its status and reason stay (413 / 400)
and its message contains `maximum context length` (with
`details.capacity: true`), the phrase clients such as the Decision Index kit
read as "the input does not fit". The body is at most 1 MiB (413) and must be
`Content-Type: application/json` (parameters such as `charset` allowed):
any other content type, or none, is 400 `INVALID_REQUEST` on this API (the
router paths of [section 4](#4-cortiq-router-api-schema-11) answer it with
415, as the router does).

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
| `superset` | one skill's labels plus labels it does not know | the oracle only; its answer teaches that skill only for a key with `learning_allowed` |
| `untrained` | anything else, and every `score` / `noul` question | the oracle only; without it the request is 422 |

**Auto-skills (0.8.6).** A choice question no skill fits (`untrained` because
of its options, not an ambiguous one and not one forced with `cmf.skill`) is
learned from the oracle's answers when the caller's key has `learning_allowed`
and `learning.auto_skills` is on (the default). Its *contract* is the whole
question — `instructions` and `criteria` with their descriptions, not the id
set: the key is `auto-<12 hex of the sha256 of the canonical JSON of {type,
instructions, criteria}>` with the criteria keys sorted (so the criteria in
any order are one contract, while other instructions, a changed description,
a dropped or an added option are each another contract — `{yes, no}` under
"Is this spam?" and under "Is this urgent?" are two skills that never answer
each other's question, and positional ids `{A, B, C, D}` whose descriptions
change with every request are never learned as one). The answers accumulate
as examples of that skill, and once enough evidence is there (2 labels with ≥
`auto_min_rows` fit rows, the active labels holding ≥ `auto_min_coverage` of
the examples, a macro agreement with the oracle on a held-out fifth of the
rows ≥ `auto_min_agreement`) the skill is fitted, written as a generation and
served: the same contract is then `exact`, `action: local`, `skill: auto-…`,
always `certified: false` (`decision_path` `router:uncertified`). An
auto-skill is matched by its contract alone — never as a `subset` or
`superset` — so to be learned and then answered locally a client keeps the
question stable: the same `instructions` text, the same option ids with the
same descriptions (`null` descriptions, and absent instructions, count as
such and stay stable). Labels the oracle rarely picks stay *quarantined*: they
count as known within the contract, get probability 0, and a text of such a
label is expected to abstain on novelty and keep teaching it. A
young auto-skill's gate comes from a handful of rows, so a local answer also
needs `p_top ≥ learning.auto_tau` (0.90; `balanced` and `quality-first`) and
its temperature is floored at `learning.auto_temperature_min` (0.02 — a clean
calibration subset fits T at its lower bound, where every `p_top` is 1);
whatever abstains escalates and teaches, and later attempts refit the whole
skill under a regression gate. *Exploration* is the one exception to the hard
rule that a gate-accepted question never reaches the oracle: while a label of
an auto-skill is quarantined, a question the gate accepted is escalated anyway
when the text's hash says so (`u64le(sha256(φ_P))` ≡ 0 modulo
`learning.auto_explore_every`, 8 — one text in eight; 0 turns it off), because
the gate confidently names a quarantined label's texts as a neighbour and they
would otherwise never teach it. The oracle's answer is served (`action:
oracle` / `cache`, `decision_path` `escalate→…`, the flag `explore`, the
`gate` block still `accepted: true`) and learned as usual, so the rare label
collects examples at a quarter of its traffic until it activates; a refused
or failed call leaves the local answer. Only auto-skills explore, only while
a label is quarantined, and only when the answer could teach (the oracle
consented, learning on, a key with `learning_allowed`); `cortiq decide`,
shadow mode and `/v1/route` never do. Its rubric is the contract — the first
caller's instructions and criteria verbatim — visible through
`/v1/skills/{id}` to every key (as cached answers are shared across accounts).
Limits, who teaches and the state directory's compatibility:
[ORACLE.md](ORACLE.md#auto-skills). When a data skill and auto-skills fit a
question equally, the data skill answers.

**State-less requests (0.8.7).** Many benchmark and batch clients send
`state: {}` and put the item's text inside the question's instructions
(`"Classify the banking intent of this user request:\n<text>"`). A request
whose `state` is empty (`""`, `{}`, `[]`, `null`) is *state-less*: for each
question the local model reads that question's `instructions` (a string as
is, an object or array as its canonical JSON; a question without
instructions reads the state's text as before), the encoder running once
per distinct instructions text. Data skills are matched by their ids as
always. The auto-skill contract of a state-less question is `{type, input:
"instructions", criteria}` — the criteria with their descriptions, without
the instructions, which are the data: every item of a classification set
under the same options is one contract, learned and then answered locally,
and never the same contract as a request with a non-empty state over the
same criteria (whose contract includes its instructions). Its cache scope is
that contract too, with the φ of the instructions text, so a near-identical
text under the same criteria is a cache hit. The instructions of a
state-less question are its input: they leave for the oracle PII-redacted
like a state (unless `allow_pii_egress`), the oracle request otherwise
unchanged (`state` sent as `{}`). Each one is limited to `limits.state_bytes`
(a capacity error past it); state-less answers are never `certified`. A
state-less contract is registered and learned only from its fifth sighting
(`learning.auto_min_sightings`; sightings counted in memory, at most
`learning.auto_sightings_cap` contracts, lost on restart): a one-off contract
— a multiple-choice item whose option descriptions change with every
question — is answered by the oracle and never written to the state
directory nor counted against the registry; the answers before that
sighting are not learned. State-less contracts are capped by
`learning.auto_max_stateless_skills`, apart from the stateful ones'
`auto_max_skills`. Requests with a non-empty state behave exactly as in
0.8.6 (their contracts register at the first sighting). The same rules
hold on `/v1/systemone` ([section 3a](#3a-system-one-request-adapter-jev-compatible)).
`/v1/skills/{id}` of a state-less auto-skill shows `rubric.instructions:
null` and `rubric.input: "instructions"`; `GET /v1/admin/learning` lists it
with `stateless: true` and counts `auto_sightings` (contracts seen, not
registered) and `auto_registered` (contracts this process registered).

`cmf.skill` names the skill and skips the search (an unknown id is 400). An
answer is `certified: true` only for an exact match on a string `state`, the
`balanced` or `quality-first` profile, a skill whose gate is certified, and a
winning label that comes from the training data. The bound (Clopper–Pearson
lower bound ≥ 0.95 on the odd calibration half) covers only `certified: true`
answers, all of them `action: local`, and only for traffic like the skill's
calibration split; subset matches, `cost-saver`, object or array states and
uncertified skills also answer `action: local` with `certified: false`.
`quality-first` answers are a sub-selection of the certified acceptance set,
for which no bound of their own was computed. Oracle and cache answers are
never certified.

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
* **choice from the oracle or the cache** — `{type, choice}`; on
  `/v1/systemone` it carries the one-hot distribution too (`probabilities`:
  the chosen option 1, every other 0, in request order; `confidence: 1`),
  since Jev's schema requires one.
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
| `flags` | e.g. `oracle_disabled`, `no_key` (with `oracle_disabled`: the key variable is not set), `bad_key` (with `oracle_disabled`: it holds something that is not a key), `consent_off`, `budget`, `stopped`, `oracle_unavailable`, `pii_redacted`, `explore` (a gate-accepted question of an auto-skill answered by the oracle for exploration, section 3.2) |
| `confident` | the answer can be used as is (gate accepted and not novel, or a valid oracle answer) |
| `complexity` | `{score, tier, factors: {base, ambiguity, novelty, margin, length}}`, the cortiq-router formula |
| `routing` | `{target, reason}` when `routing_tiers` maps the tier |
| `decision_path` | `router:certified`, `router:uncertified`, `router:uncertified_subset`, `escalate→cache`, `escalate→oracle`, `escalate→oracle_unavailable`, `escalate→disabled` |
| `explanation` | with `cmf.explain`: `{top1_vs_top2, decision_path}` |

When a trained question abstains because the oracle is not ready (flags
`oracle_disabled`, `no_key`, `bad_key`, `budget` or `stopped`, not `consent_off`), the
answer also has `cmf.hint`, one line that says what to do, e.g. `"the oracle
key is not set: set OPENROUTER_API_KEY in the server's environment and
restart it"` or `"no oracle answers the questions the local model cannot
decide: start the server with --oracle MODEL and set OPENROUTER_API_KEY"`.
The router API (section 4) has no hint; the server logs it (each hint at most
once a minute; once, at INFO, on a server started without an oracle).

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
| 400 | `INVALID_REQUEST` | schema, limits, duplicate or unknown keys, malformed JSON, a content type other than `application/json` |
| 401 | `UNAUTHORIZED` | key missing, wrong, expired or revoked |
| 402 | `QUOTA_EXCEEDED` | decision or token quota, credit |
| 404 | `MODEL_NOT_FOUND`, `INVALID_REQUEST`, `ADMIN_DISABLED` | unknown model; feedback target not found; admin token not configured |
| 413 | `PAYLOAD_TOO_LARGE` | body over `limits.body_bytes` |
| 422 | `UNSUPPORTED_QUESTION` | untrained question and the oracle not allowed; untrained question whose oracle call the upstream refused as over its context (message with `maximum context length`, `details.capacity`) |
| 429 | `RATE_LIMITED`, `OVERLOADED` | the minute window; more than `max_inflight` requests (`Retry-After`) |
| 500 | `INTERNAL` | |
| 502 | `ORACLE_UNAVAILABLE` | untrained question, the oracle call failed |
| 503 | `ORACLE_BUDGET_EXHAUSTED`, `ORACLE_DISABLED` | untrained question, no budget or a stop rule |

A trained question whose oracle call fails is never an error: it is answered
locally with `action: abstain` and the flag `oracle_unavailable`. Capacity
errors (a size limit, or the oracle's context) say `maximum context length`
in their message; an oracle refusal of that kind is the code `context_length`
in `oracle.jsonl` and does not count toward `oracle.max_errors`.

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
must be one of the question's options. For a key with `learning_allowed` it
becomes a training example of weight 3, and an option the skill does not
have starts a cold start ([ORACLE.md](ORACLE.md#self-learning)); for any
other key it is consumed and answered 200 with `accepted: false` and
`refused: "learning_not_allowed"`. The router form of feedback (section 4)
accepts any label.

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
| `GET /v1/skills` | key | every skill: `id`, `taxonomy_version`, `labels` (the active ones), `certified`, `tau`, `theta`, `temperature`, `has_rubric` (whether it has a rubric; the rubric itself is per skill) and, since 0.8.6, `auto` (an auto-skill), `active_labels` (their count), `quarantined_labels` (the labels not scored yet) and `examples` (learned rows); auto-skills are listed after the file's skills |
| `GET /v1/skills/{id}` | key | the same plus `tasks`, the whole `gate` and `rubric` (`instructions`, `criteria`; `null` without one) |
| `GET /v1/usage` | key | the caller's account (router format; `x-cmf-extensions: 1` adds token and cost totals) |
| `GET /healthz` | open | status, model, generation, skills (every served skill), `auto_skills` (0.8.6), `oracle` (on/off) and `oracle_status` (`ready`, `no_key`, `bad_key`, `disabled`, `budget_exhausted`, `budget_too_small`, `stopped: <reason>`; section 5) |

Without `--jev-compatible`, `/v1/models` has only the native OpenRouter-style
listing shape; Cortiq Decision is not listed on OpenRouter. With the adapter
enabled it instead returns the System One discovery `models` array described in
[section 3a](#3a-system-one-request-adapter-jev-compatible).

```bash
curl -s "$CORTIQ/v1/models" \
  | jq '.data[0] | {id, pricing, skills: [.cmf.skills[] | {id, labels, certified, tau}]}'   # → 200
curl -s "$CORTIQ/v1/skills" -H "Authorization: Bearer $KEY" \
  | jq '[.skills[] | {id, labels: (.labels | length), certified}]'                          # → 200
curl -s "$CORTIQ/v1/usage" -H "Authorization: Bearer $KEY" -H 'x-cmf-extensions: 1' \
  | jq '{account, totals: .cmf.totals}'                                                     # → 200
```

## 3a. System One request adapter (Jev-compatible)

Enable this endpoint explicitly; it is off by default so native clients retain
their existing contracts:

```bash
cortiq serve cortiq-decision.cmf --state ./decision.state \
  --jev-compatible --port 8080
```

`POST /v1/systemone` accepts the TypeSafe System One request shape. It uses the
same local CMF skills, gate, key policy, limits and optional oracle as the
native server. It does **not** load Jev weights or present itself as Jev.

| Field | Adapter rule |
|---|---|
| `model` | optional. Omitted, `jev-latest`, `jev-preview`, `jev-1.13.0`, the `typesafe/jev-1.13` selector series, `cmf-decision-<version>` and `default` (the Decision Index kit's placeholder; not listed by discovery) select the current local CMF decision model at this endpoint only. |
| `state` | a string, object, array or `null`; empty (`""`, `{}`, `[]`, `null`) makes the request state-less ([section 3.2](#32-skill-matching-and-certified)). |
| `questions` | an object of `choice`, `score` or `noul` questions. `instructions` may be omitted or `null`, matching System One clients. |

For example, a client that omits `model` can send a small local choice:

```bash
curl -s "$CORTIQ/v1/systemone" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{
    "state": "I still have not received my new card",
    "questions": {
      "intent": {
        "type": "choice",
        "criteria": {
          "card_arrival": null,
          "card_delivery_estimate": null
        }
      }
    }
  }' | jq
```

A successful response is intentionally small and identifies the local model:

```json
{
  "model": "cmf-decision-0.8.5",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "card_arrival",
      "confidence": 0.99,
      "probabilities": {"card_arrival": 0.99, "card_delivery_estimate": 0.01}
    }
  },
  "usage": {"input_tokens": 42, "output_tokens": 2}
}
```

The token figures use the same request metering as the native endpoint; their
values vary with the request. `GET /v1/models` changes to System One discovery
format in adapter mode. Its `models` array contains the canonical
`cmf-decision-0.8.5` entry and transport aliases such as `jev-latest`; each
alias describes itself as a route to the local CMF model. Requests use the same
Cortiq key policy as the rest of the server. System One errors use
`{"error":{"type", "message", "code", "request_id"}}`. Schema errors and
capacity errors (a size limit, the body included, or the oracle's context)
are 422 here, the latter with `maximum context length` in the message — the
marker the Decision Index `http` engine treats as an unsupported item.

The adapter is deliberately isolated: `/api/alpha/decisions` and
`/v1/decisions` keep their stricter native model rules and continue to reject
Jev model identifiers. Compatibility means the request/response wire format,
not a claim of Jev identity, weights, affiliation or benchmark equivalence.

## 4. cortiq-router API (schema 1.1)

The same server answers the router's API with exactly its keys, JSON types and
error envelope, so existing clients need no change:

| Path | Access |
|---|---|
| `POST /v1/route`, `POST /v1/route:batch` (up to 1024 inputs) | key |
| `POST /v1/feedback` `{request_id, correct_task_label}` (`accepted: false` for a key without `learning_allowed`) | key |
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
  action, `certified`, gate) to results, listings and errors, and
  `cmf.oracle_status` to `/v1/healthz`. Without it the responses have the
  router's keys only.
* The router's flags keep their vocabulary: an escalation the oracle does
  not answer is `low_confidence` plus `oracle_disabled` (also when the key
  variable is not set or not a key — `no_key` and `bad_key` appear only on
  the decisions API),
  `consent_off`, `budget`, `stopped` or `oracle_unavailable`, in the answer
  and in `/v1/escalations`; the hint of the decisions API is only logged.

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
| `GET /v1/admin/oracle`, `POST /v1/admin/oracle {"enabled", "budget_usd", "max_calls"}` | oracle status: `status` (`ready`; `no_key` — the key variable is unset or empty; `bad_key` — it holds something that is not a key, never sent; `disabled` — not configured or switched off by the admin; `budget_exhausted` — something was spent and the budget (what is left cannot hold the smallest possible call, or a call it refused) or `max_calls` is used up; `budget_too_small` — nothing spent, and the budget cannot hold one call, `min_call_usd`, or `max_calls` is 0; `stopped: <reason>` — a stop rule), `configured` = `oracle.enabled` of the configuration, `enabled` = not switched off by a stop rule or the admin, `key_present`, `key_ok`, `key_problem` (by position and length, never a byte of the key), `key_trimmed` (surrounding whitespace was trimmed), `key_env` (the variable's name, never its value), `model`, `max_price`, `min_call_usd` (the least budget a call needs: the smallest possible call's reservation, or, once the budget refused a longer call — a budget below the smallest call included — that call's, the larger), spent, calls, stop reason, `last_error` (the code of the last failed call; both from a closed code set, another value read back from `oracle.state` is `unknown_code`); switch it and lower limits within the configuration (kept in `oracle.state` across restarts: when one of them refuses the next call, `cmf.hint` and the startup line name it and the `POST /v1/admin/oracle` that lifts it) |
| `GET /v1/admin/learning` | buffer, cache, quarantine, attempts, promotions, recent events; since 0.8.6 `auto_contracts`, `auto_skipped` and `auto_skills` (per contract: `id`, `labels`, `examples` per label — the rows an attempt fits: the served learned rows plus the pending buffer examples, each row once — `created_unix`, `served`) |
| `GET /v1/admin/generations`, `POST /v1/admin/rollback {"generation": N}` | generations; serve generation N (0 = the base file) |
| `GET /v1/admin/shadow` | agreement statistics in shadow mode, and the routed requests not compared since the start (section 8) |

```bash
A="x-admin-token: $CORTIQ_DECISION_ADMIN_TOKEN"
curl -s "$CORTIQ/v1/admin/usage" -H "$A" | jq 'keys'                          # → 200
curl -s "$CORTIQ/v1/admin/oracle" -H "$A" | jq '{status, enabled, configured, key_present, budget_usd, spent_usd}'   # → 200
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
  "learning": {"enabled": true, "refit_min_new": 25, "dedup": 0.995, "cold_start": true, "synchronous": false,
               "auto_skills": true, "auto_min_rows": 10, "auto_k": 8, "auto_tau": 0.9,
               "auto_min_agreement": 0.8, "auto_min_coverage": 0.8, "auto_max_skills": 256,
               "auto_max_labels": 64, "auto_max_examples_per_label": 1000,
               "auto_temperature_min": 0.02, "auto_explore_every": 8,
               "auto_min_sightings": 5, "auto_sightings_cap": 100000,
               "auto_max_stateless_skills": 256},
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
`oracle.base_url` must be https, or plain http to a loopback address only (a
local proxy). The oracle, cache and learning sections are explained in
[ORACLE.md](ORACLE.md). The `learning.auto_*` keys govern auto-skills
(section 3.2, [ORACLE.md](ORACLE.md#auto-skills)): `auto_skills` learns
untrained choice contracts at all; `auto_min_rows` fit rows (≥ 2) a label
needs to be active; `auto_k` the rank of its topologies; `auto_tau` the
confidence floor of a local answer (0..1; a serving parameter — `cortiq
decide` uses the default); `auto_min_agreement` the macro agreement with the
oracle required to activate and `auto_min_coverage` the share of the
contract's examples its active labels must hold (both 0..1); `auto_max_skills`
how many stateful contracts are learned at most and `auto_max_labels`
(2..255) how many options one may have — a contract past either is answered
by the oracle and not learned (intent suites need more than 64: BANKING77
has 77 ids, CLINC150 151); `auto_max_examples_per_label` (≤ 5000) replaces the
per-label buffer cap for auto-skills; `auto_temperature_min` (0..1) floors
the gate temperature an attempt records (0: the fitted T, whose lower bound
makes every `p_top` 1); `auto_explore_every` (an integer, 0 = off) explores
one text in that many while a label is quarantined — the one case in which a
gate-accepted question reaches the oracle, auto-skills only;
`auto_min_sightings` (≥ 1) is the sighting of a state-less contract from
which it is registered and learned (1: at once, like a stateful one),
`auto_sightings_cap` (≥ 1) how many unregistered state-less contracts the
in-memory sightings LRU tracks and `auto_max_stateless_skills` (≥ 1) how
many state-less contracts are learned at most. `cortiq serve --oracle MODEL` and its companions
override the `oracle` section (and `--no-oracle-learning` sets
`learning.enabled` false); a file that sets `oracle.provider` keeps its
`max_price` unless `--oracle-max-price` is given.

## 7. Your own model

A skill is trained from JSONL rows `{"text", "label"}`: a non-empty text up to
32 KiB, a label of 1–256 bytes, no other keys. Duplicates and conflicting
labels are kept (conflicts are reported). A label with a single training row
is kept inactive.

```bash
# the router's datasets_dir layout, read as its read_label_prompts does: one
# directory per label, every file in it, one text per line (trimmed, blank
# lines skipped); in a .jsonl file the line's "text" (else "prompt") string,
# else the line itself; a file that is not UTF-8 is skipped
python3 - datasets > train.jsonl <<'PY'
import json, os, sys
root = sys.argv[1]
for label in sorted(os.listdir(root)):
    d = os.path.join(root, label)
    if not os.path.isdir(d):
        continue
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        try:
            lines = open(p, encoding="utf-8").read().split("\n")
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            text = line.strip()
            if not text:
                continue
            if name.endswith(".jsonl"):
                try:
                    v = json.loads(text)
                except ValueError:
                    v = None
                if isinstance(v, dict):
                    t = v["text"] if "text" in v else v.get("prompt")
                    if isinstance(t, str):
                        text = t
            print(json.dumps({"text": text, "label": label}, ensure_ascii=False))
PY
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
* Imported keys do not teach the model (`learning_allowed: false`): their
  `/v1/feedback` is answered and consumed but not learned (`accepted:
  false`), where the router learned from every key; `--learning-allowed`
  imports them with it. The oracle's answers to their `/v1/route` still
  teach the skill (the route question is the skill's own).
* `--usage` writes the usage ledger: run it while no server holds the state
  directory. Importing a newer export again adds only the new keys, the
  revocations and the growth of the counters.

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
byte (errors included), so clients see no change. A `/v1/route` or
`/v1/route:batch` that the old router answered 200 is also decided locally
(no oracle, no learning, no billing) and compared line by line in
`<state>/shadow.jsonl`: labels, confidence, latency and a keyed digest of the
text (HMAC-SHA256 under a random key kept in `<state>/shadow.key`), never the
text. An answer that is not a 200 — a missing or wrong key, a quota, an error
— is neither decided nor written. The comparisons have 4 slots of their own,
apart from the decisions API's `limits.max_inflight` (64 more may wait; one
past that is dropped), and at most 256 requests are forwarded at a time;
`GET /v1/admin/shadow` counts the requests not compared since the start
(`skipped`).

**Configuration.** The router's own settings do not come over by themselves;
write them into a decision configuration from the router's TOML. For the
router's `deploy/config.prod.toml`:

```bash
cat > router.json <<'EOF'
{
  "default_skill": "data-assistant",
  "auth": {"admin_token_env": "CORTIQ_ADMIN_TOKEN"},
  "complexity_tiers": [{"tier": "low", "max": 0.33}, {"tier": "medium", "max": 0.48},
                       {"tier": "high", "max": 1.0}]
}
EOF
export CORTIQ_ADMIN_TOKEN="<the router's admin token>"
```

* `default_skill` is the router's `taxonomy_id`. Router clients may leave
  `taxonomy_id` out; with several skills in the file and no `default_skill`
  such a request is 400. A skill id is `[a-z0-9][a-z0-9_-]{0,63}`: a router
  `taxonomy_id` outside that pattern (the default `general.task-type`) cannot
  be kept, and its clients have to send the new id.
* `auth.admin_token_env` names the router's variable, so that the portal's
  `x-admin-token` keeps working when this server answers `/v1/admin/keys`
  after the switch (the same token also reads `/v1/admin/shadow`).
* Copy `[complexity_weights]`, `[[complexity_tiers]]` (the default here is
  medium ≤ 0.66), `[task_complexity]` and `[routing_tiers]` (without them no
  response has a `routing` block) as JSON, and `[auth.plans]` if the router
  changes them (`duration_days` is `days` here, 0 is `null`).
* A router that escalated to an oracle: add the `oracle` section of
  [ORACLE.md](ORACLE.md#connect-openrouter).

`--shadow-of` reaches the old router over **https**, or over plain http **only
at a loopback address** (`127.0.0.1`, `::1`, `localhost`), because the
clients' keys pass through it. cortiq-router itself speaks plain http, with
TLS at nginx, so run the new server on the router's host and point it at the
router's own port there.

```bash
# on the router's host; the old router listens on port 8080
cortiq serve cortiq-decision-plus.cmf --decision-config router.json \
  --state ./router.state --port 8090 --shadow-of http://127.0.0.1:8080
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
curl -s "$SHADOW/v1/admin/shadow" -H "x-admin-token: $CORTIQ_ADMIN_TOKEN" \
  | jq '{lines, compared, agree, agreement, confident, latency_ms, skipped}'  # → 200
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
enough. In shadow mode `/v1/admin/keys` went to the old router: keys the
portal minted or revoked since 8.1, and the decisions the router counted,
are only there. Hold the portal's admin calls, stop the shadow server
(Ctrl-C), export `api_keys` and `usage_counters` again as in 8.1 and import
them once more (new keys are added, keys the router marks inactive are
revoked, and only the growth of the counters is added), then start the
server on the same state directory and port without `--shadow-of`. nginx
sends requests to the old router (`backup`) while the new server is down;
nginx itself needs no change.

```bash
cortiq decision keys import --state ./router.state --from api_keys.jsonl \
  --usage usage_counters.jsonl
cortiq serve cortiq-decision-plus.cmf --decision-config router.json \
  --state ./router.state --port 8090
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
