# Cortiq Decision — oracle cascade and self-learning

The local model answers every question its certified gate accepts. A question
it cannot determine — the gate rejected the answer, or the question is not
trained at all — can be passed to a large language model, the *oracle*,
through OpenRouter. The oracle's answers are cached and can become training
examples, so the local model learns the traffic it abstains on; a refit
changes no other label's parameters (checked by sha256), although a refitted
label can win rows another label used to win. The oracle is **off by
default**; nothing leaves the machine until you enable it — in two steps:
the key in `OPENROUTER_API_KEY`, then `cortiq serve FILE --oracle MODEL`
(or `cortiq decide FILE … --oracle MODEL`; `cortiq decision oracle check`
says first whether it is ready — [Check your setup](#check-your-setup)).

1. [How a question flows](#how-a-question-flows)
2. [Self-learning](#self-learning)
3. [Budget and stop rules](#budget-and-stop-rules)
4. [What leaves the machine](#what-leaves-the-machine)
5. [Connect OpenRouter](#connect-openrouter) —
   [check your setup](#check-your-setup),
   [one text or a batch](#one-text-or-a-batch)
6. [Measured effect](#measured-effect)

## How a question flows

1. **Local decision.** Encoder, hashing and resonance run once per request. If
   the gate accepts, the answer is `action: local`. Such a question is never
   sent to the oracle, not even with `cmf.oracle: true`.
2. **Permission.** An undetermined question may go to the oracle only if all of
   these hold: `oracle.enabled` (or `serve --oracle MODEL`); the key variable
   is set in the server's environment; the caller's key has `oracle_allowed`
   (in open mode, without keys: `auth.require: false` in the configuration,
   or `serve --oracle` on a loopback address — see
   [Who may use it](#who-may-use-it-and-who-teaches)); the request consents
   (`cmf.oracle`, the router's `options.allow_oracle`, else
   `oracle.default_per_request`); budget is left; no stop rule fired.
   Otherwise the question is still looked up in the cache (step 3; since
   0.8.11): a hit is answered `action: cache`, `decision_path
   escalate→cache`, with no call and nothing sent. Only a miss is refused:
   a trained question stays `abstain` with a flag (`oracle_disabled`,
   `no_key` beside `oracle_disabled` when the key variable is not set,
   `bad_key` beside it when the variable holds something that is not a key,
   `consent_off`, `budget`, `stopped`) and, on the decisions API, a one-line
   `cmf.hint` that says what to do (the router API keeps its answer and its
   flags as they were — a missing key is `oracle_disabled` there — and the
   server logs the hint); an untrained question fails with 422 or 503, and
   the error names every untrained question of the request — also those
   the cache answered — so it never tells which of them the cache holds.
3. **Cache, exact first.** Within the same scope — the question's contract
   (type, instructions and criteria), and for a question matched to a skill
   also the skill and the set of options — a stored oracle answer to the
   *same input* is reused: `action: cache`, cost 0. The input is the sha256
   of the whole state as asked (canonical JSON, before PII redaction; for a
   state-less question its whole instructions, the lead-in line included),
   not the text embedding, which reads only the first 512 tokens and barely
   moves when one number of a JSON state changes. Up to 50000 entries, ring
   buffer. The cache is shared by all accounts: an answer given under one
   caller's instructions is never served to a question with other
   instructions or criteria, and a `cache` answer tells its caller that some
   account asked the same text under the same contract.
   * **Near reuse is opt-in** (since 0.8.11): with `cache.threshold` below
     1, an answer whose text embedding has cosine ≥ the threshold with the
     new one is reused too (0.97 was the default before 0.8.11). The default
     1 turns it off: decision states that look alike often differ in the
     detail that decides (a note in a JSON score, a number in a causal
     question, one address in an e-mail), and at 0.97 a Decision Index run
     through the gateway answered ~60k of 282k questions with another row's
     oracle answer (index 58 → 51.9). Turn it on only for traffic whose
     paraphrases share their answer (intent routing of short texts).
   * **Entries written before 0.8.11** carry no input digest: they are
     reused for an embedding cosine ≥ `cache.legacy_cos` (default 0.9999,
     the repeat of a text as far as the encoder reads it), or at the
     threshold with near reuse on. That is not exact: two states that
     differ only past the encoder's 512 tokens, or two state-less questions
     that differ only in their lead-in line, have cosine 1. On such a state
     directory set `cache.legacy_cos: 1`: the old entries are not loaded and
     answer nothing (their records stay in `learn.log` beside the examples
     and contracts), so the next oracle pass asks each question once and
     stores its answer with its digest. Re-running the pass without it
     changes nothing: an old entry still answers first.
   * **Without the oracle.** The cache is consulted for every escalated
     question, also when the oracle may not be called for it: a key with
     `oracle_allowed: false`, `cmf.oracle: false` (the router's
     `allow_oracle: false`), `oracle.enabled: false`, the admin switch off,
     a stop rule or an exhausted budget (and `cortiq decide --oracle`
     without a usable key, from its state directory's cache, read only). A
     hit costs nothing and sends nothing, so after an oracle pass the same
     questions are answered without it; a miss is refused as before (the
     flag, 422 or 503; the error names every untrained question). Like any
     hit, it tells such a caller that some account asked the same text
     under the same contract. Neither the admin switch nor a stop rule
     stops cached answers; only `cache.enabled: false` (a restart) does,
     with the rest of the cache.
4. **Single flight.** A question already in flight in another request (same
   scope and the same input; with near reuse on, cosine ≥ `cache.threshold`)
   waits for that call instead of making its own. A follower of the same
   input is answered by the leader's entry, cached and kept in `learn.log`,
   so its repeat is a hit after a restart as well. A near duplicate's
   follower (near reuse on) gets the leader's answer but nothing is stored
   under its own input — the oracle never read it; while near reuse is on
   its repeat hits the leader's entry, and with it off it is asked.
5. **One call** carries all remaining questions of the request: `POST
   {base_url}/chat/completions` with a strict JSON-schema answer (an enum of
   the option ids for choice, an integer level for score, a boolean for noul)
   and, since 0.8.8, a distribution beside each verdict ([Probabilities](#probabilities)),
   `temperature 0`, reasoning off unless `oracle.reasoning` sets an effort
   ([Reasoning](#reasoning)), `max_tokens` 64 per question plus 128 per
   question for the distribution. The system prompt tells the model that the
   state is untrusted data, not instructions.
6. **Result.** A valid answer is `action: oracle` with its cost in
   `usage` and is stored in the cache (with its distribution). For a choice question matched to a
   skill it becomes a learning example of that skill — which every account
   is served — when the caller's key has `learning_allowed`, or when the
   question is exactly the skill's own (its rubric's instructions and
   criteria over all its active labels, as `/v1/route` asks it), so that
   the answer is the skill's rubric applied to the text and not a caller's
   instructions. The one exception is the open mode of `serve --oracle` on a
   loopback address without keys: nobody is identified there, so its
   answers are cached but never learned. A failed call never becomes an error for a trained
   question: it is answered locally with `action: abstain` and the flag
   `oracle_unavailable` (an untrained question gets 502).

A question whose ids no skill has but whose option descriptions name one
data skill's labels (positional ids `option_0…`, the label names as
descriptions — the Decision Index's BANKING77 and CLINC150 rows) is decided
by that skill ([API.md §3.2](API.md#32-skill-matching-and-certified),
"Matching by descriptions"): locally when its gate accepts; a rejected text
is answered locally with the question's none option ("out of scope…",
"none of…") when it has one, else it goes to the oracle, whose answer
teaches the skill the option's label. In a state-less request the text after
a lead-in line ("Classify the intent of this request:\n<text>") is what the
local model reads; the oracle gets the instructions verbatim.

Superset questions (a skill's labels plus new ones) are decided by the oracle
only; for a caller with `learning_allowed` the answer teaches that skill, and
a new label starts a cold start. A choice question no skill fits is learned
into an *auto-skill* of its own ([Auto-skills](#auto-skills)), for such a
caller only.

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
  its threshold, the challenger is rejected. The holdout that picks the
  challenger is part of those calibration rows, so from the first promotion
  the rows that certify τ also chose the model: the bound is nominal after
  any promotion, while answers keep `certified: true`.
* **Isolation.** Before a promotion the sha256 of the mean and basis of every
  other label is compared with the values before the attempt; any difference
  cancels the promotion. Other labels' parameters stay byte for byte the same.
  This is not a promise about accuracy: the refitted label still competes in
  the argmin and can take rows from the others (after the pre-training below,
  BANKING77 had 10 fewer correct rows).
* **Cold start.** A label the skill does not have (from an oracle answer to a
  superset question, or from feedback, of a caller with `learning_allowed`)
  becomes a quarantined task that is not scored. At 25 examples it is fitted
  and activated and the gate re-certified; answers it wins carry
  `certified: false`.
* **Feedback.** `POST /v1/feedback` (`{"id", "question", "label"}` with one of
  the question's options, or the router's `{request_id, correct_task_label}`
  with any label) corrects a decision of the caller's own account. From a key
  with `learning_allowed` it becomes an example of weight 3; from any other
  key it is consumed and answered (`accepted: false`) but not learned. No key
  has the permission unless it was created or imported with it.
* **Limits.** At most 5000 examples per label of a skill and 32 labels per
  skill waiting for a cold start; an example past either is refused.
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
  skill is undone (see API.md, section 7). An auto-skill (below) learns
  online only.

### Auto-skills

A choice question whose option ids match no skill (`match: untrained`, not
an ambiguous one — several skills fit — and not one forced with `cmf.skill`)
goes to the oracle on every new text and, until 0.8.5, was never learned. Since
0.8.6 its *contract* becomes a skill of its own:

* **Key.** The contract is the question itself: `auto-` + the first 12 hex
  characters of the sha256 of the canonical JSON of `{type, instructions,
  criteria}` — the instructions as sent (`null` when absent), the criteria
  object with its keys sorted and every description kept. The criteria in
  any order are one contract; other instructions, a changed description, a
  dropped or an added option are each another contract, learned separately.
  The id set alone is not the key: two clients using `{yes, no}` under
  different questions get two skills, and a multiple-choice benchmark whose
  positional ids `{A, B, C, D}` carry a new description per request is never
  learned as one skill (every request is its own contract, bounded by
  `auto_max_skills`). To be learned, a client keeps its question stable —
  the same instructions text, the same ids with the same descriptions. The
  first request's instructions and criteria are stored verbatim as the
  skill's rubric (the id is recomputed from it), visible through
  `/v1/skills/{id}` to every key. The contract is written to `learn.log`
  before its first example, so a restart rebuilds it.
* **State-less requests (0.8.8).** A request with an empty `state` (`""`,
  `{}`, `[]`, `null`) — a benchmark kit that writes `state: {}` and the item
  into the instructions — is read through each question's instructions
  (canonical JSON for an object or array). The contract of such a question
  is `{type, input: "instructions", criteria}`: the instructions are its
  data, not part of the key, so every item under the same options and
  descriptions teaches one auto-skill, and the same criteria under a
  non-empty state are another contract. The cache and single flight use the
  φ of the instructions text under that contract. The rubric of such a skill
  stores `instructions: null` and `input: "instructions"`; `learn.log` tells
  the contract apart by its id (the record format is unchanged). Because a
  contract that is seen only once is useless to learn — a multiple-choice
  item whose option descriptions change with every question is its own
  contract — a state-less contract is registered only at its
  `learning.auto_min_sightings`-th sighting (5; an escalation of one of its
  learnable questions, whatever answers it), counted in an in-memory LRU of
  `learning.auto_sightings_cap` (100000) contracts that a restart clears;
  before that its answers are served and cached but not learned, nothing is
  written to `learn.log` and it takes no slot of the registry. 5, because a
  contract cannot activate before `auto_min_rows` (10) examples of each of
  its labels anyway, while items seen two or three times (WinoGrande twin
  sentences, a repeated MuSR question) are not worth a skill: on the
  Decision Index suite's state-less rows 2 registered 756 contracts, 5
  registered 20. Stateful contracts register at their first sighting, as in
  0.8.6 (from 0.8.8 at their `learning.auto_min_sightings_stateful`-th, 1 by
  default, counted in the same LRU), against `auto_max_skills`; state-less
  ones have their own cap
  `auto_max_stateless_skills` (256), so stateful one-offs (per-row
  instructions, a tool catalogue per request) never take their slots. State-less
  answers are never `certified`. A state-less auto-skill answers state-less
  requests only: its rubric has no instructions, so `/v1/route` and
  `cortiq decide --skill` (which read a text as the state) do not reach it.
* **Who teaches.** Only a caller whose key has `learning_allowed` (the explicit
  open mode `auth.require: false` has it; the implicit open mode of a loopback
  address never teaches). The rule "a question that is exactly the skill's own
  teaches it" does not apply to auto-skills: their rubric *is* the contract
  every caller sends. `learning.auto_skills: false` turns the learning off;
  the oracle answers as before.
* **Examples.** Each answer is an example of the auto-skill under the
  contract's label (the dedup of 0.995 applies). The labels are closed: a
  feedback with a label outside the contract is refused (a request with one
  more id is another contract, learned on its own).
  At most `learning.auto_max_examples_per_label` (1000) examples per label.
* **Activation.** When a label collects 25 new examples (the usual trigger)
  the whole skill is attempted: one row in five — by a hash of the row itself,
  so a row never moves between the fit and the calibration subset as more
  arrive — is held out; a label with ≥ `auto_min_rows` (10) fit rows is
  *eligible* and fitted (rank `auto_k`, 8), the others stay quarantined; the
  attempt needs two eligible labels including the triggering one and, the
  first time, the eligible labels must hold ≥ `auto_min_coverage` (0.8) of
  the contract's rows. T and θ are certified on the held-out rows as at build
  time (τ certifies only with 100 accepted odd rows, so the skill reports
  `certified: false` for a long time, and its winners are `cold_start` ones —
  never certified). The first activation (`auto_start`) requires a macro
  agreement of the fitted skill with the oracle's labels on the held-out rows
  ≥ `auto_min_agreement` (0.8) and ≥ 0.5 per label; a later attempt
  (`auto_refit`, the whole skill refitted) must not regress the champion on
  the same rows (the holdout rule). A promotion writes a generation that
  carries the auto-skill entirely; every other skill is byte for byte the
  same.
* **Serving.** The contract is then decided locally, `match: exact`,
  `skill: auto-…`, `certified: false`. An auto-skill is matched by its
  contract alone: a part of its ids, a superset, other instructions or another
  description is another contract (`untrained`, learned on its own), never a
  `subset` or `superset` of the auto-skill. A quarantined label still counts
  as known within the contract: it gets probability 0, and a text of it is
  expected to abstain on novelty and keep teaching it. Because θ and T of a
  young auto-skill come from a handful of rows (θ is close to the largest
  novelty of the even half, T may sit at a bound), a local answer also needs
  `p_top ≥ learning.auto_tau` (0.9) under `balanced` and `quality-first`
  (`cost-saver` keeps its θ-only rule), and the T an attempt records is
  floored at `learning.auto_temperature_min` (0.02): a clean held-out subset
  fits T at its lower bound, where every `p_top` is 1 and the `auto_tau`
  floor never bites. An abstention escalates and teaches as any other. The
  agreement gate is the real guard while the subset is small; the gate
  improves with every refit. Rejected examples are never forgotten: a
  contract polluted by a noisy oracle needs more consistent examples, not a
  reset (or a rollback to a generation before it).
* **Exploration.** The hard rule — a question the gate accepted never
  reaches the oracle — has exactly one exception, for auto-skills only and
  only while one of their labels is quarantined: the gate names such a
  label's texts as a neighbour with full confidence (measured on a stand: all
  21 texts of a quarantined label answered locally as another label at
  `p_top` 1), so without it the label could never learn. One text in
  `learning.auto_explore_every` (8; `u64le(sha256(φ_P))` modulo it is 0, a
  property of the text, never of the order; 0 turns it off) is escalated
  although accepted: the oracle's answer is served (`action: oracle` /
  `cache`, the flag `explore`, the `gate` block still reports `accepted:
  true`) and learned as usual, so the rare label collects examples at a
  quarter of its traffic until the next attempt activates it, after which
  the contract explores no more. A refused or failed call leaves the local
  answer. Exploration draws only when the answer could teach: the oracle
  consented, learning and `auto_skills` on, a key with `learning_allowed`;
  data skills, `cortiq decide`, shadow mode and `/v1/route` never explore.
* **Limits.** A contract is learned only with 2..`auto_max_labels` (64) option
  ids and while fewer than `auto_max_skills` (256) stateful, or
  `auto_max_stateless_skills` (256) state-less, contracts are registered;
  otherwise the oracle answers and nothing is recorded (`auto_skipped` in
  `GET /v1/admin/learning`, one warning an hour, never the ids). Intent
  suites have more options than the default: BANKING77 sends 77 ids and
  CLINC150 151 (with out-of-scope), so a gateway meant to learn them sets
  `learning.auto_max_labels` to 255. Every
  generation re-carries every auto-skill's rows: with many contracts run
  `cortiq decision materialize` from time to time and serve the materialised
  file (it holds the auto-skills as ordinary skills; `decide --labels` and
  `verify` read it; such a file needs cortiq ≥ 0.8.6). A served auto-skill
  keeps learning wherever its contract comes from: the contract record of
  `learn.log`, or — a materialised file on a fresh state directory, a
  `learn.log` lost while the generations were kept — the skill's own labels
  and rubric, registered (and written to `learn.log`) at start and after a
  rollback.
* **Routing.** `/v1/route` without `taxonomy_id` still means the file's only
  data skill; an auto-skill is routed by its id (its question is then the
  whole contract from the rubric). `cortiq decide --labels` never names an
  auto-skill by labels alone: `--skill auto-…` names it and its rubric
  supplies the contract. When a data skill and auto-skills fit a question
  equally, the data skill answers; two auto-skills of one contract cannot
  exist (the contract is the id).
* **Admin.** `GET /v1/admin/learning` lists `auto_skills` (id, labels,
  `examples` per label — the rows the next attempt fits: the served learned
  rows of the label plus the buffer examples not among them, each row once,
  so their sum is the attempt's `rows.total` — what is served, `stateless`),
  `auto_contracts`, `auto_skipped`, `auto_sightings` (state-less contracts
  seen and not registered yet) and `auto_registered` (contracts this process
  registered);
  attempts carry `kind: auto_start | auto_refit` and an `auto` block with the
  eligible and quarantined labels, the rows and the agreement. `/healthz`
  adds `auto_skills`. Rollback to an earlier generation drops the skill from
  the served model (the contract is untrained again; its examples stay and
  start it over at the next trigger).
* **Compatibility.** A 0.8.5 binary opening a 0.8.6 state directory truncates
  `learn.log` at the first contract record and refuses a generation that
  carries an auto-skill (the same rule as the `LOCK` of 0.7.9: never run an
  older binary on a newer state directory). The cache scope of a contract
  changes from the contract to the skill at its activation, so its first
  requests after that miss the cache (they are answered locally anyway).
  A 0.8.7 or older binary truncates a 0.8.8 `learn.log` at the first state-less
  contract record and refuses a manifest whose rubric carries `input`.

## Probabilities

Since 0.8.8 (`oracle.probabilities`, on by default) the oracle answers each
question with its verdict and a distribution: for a choice the at most 5 most
likely option ids with their probabilities, for a score one probability per
level, for a noul p(true). The server checks it (finite numbers in [0, 1],
which the schema also states, ids of the question, each once, at most one
per level; a malformed distribution is dropped and the valid verdict kept as
one-hot, so the call neither fails nor counts toward `max_errors`) and
normalizes it: the listed options keep their mass (renormalized when it
is above 1), the rest is spread uniformly over the unlisted ones, and the
verdict becomes the argmax (a tie goes to the stated verdict; a noul is true
above 0.5). Every surface answers with it — `probabilities` and `confidence`
= p(choice) on `/v1/decisions` and `/v1/systemone` (a native noul keeps its
verdict and adds `probability`; System One's noul is p(true)) — and the
cache keeps it: a cache answer carries the stored distribution, also after a
restart. The learning example is the argmax label, as before. A verdict
without a distribution (a bare one, or `probabilities: false`, which sends
the 0.8.7 request) is one-hot. Each question adds
`oracle.probability_tokens_per_question` (128) to `max_tokens`: measured with
the o200k tokenizer, five listed ids cost 57–87 tokens more than the bare
verdict, a 10-level score 39, a noul 10.

`learn.log` keeps a cache entry with a distribution as a new `CachePutP`
record; an entry without one keeps the 0.8.7 `CachePut` record, and every
0.8.7 `CachePut` replays as one-hot. A 0.8.7 binary stops replaying at the
first `CachePutP`: never run an older binary on a newer state directory.
Since 0.8.11 a cache entry's input digest rides in the scope field of the
same two records (`exact:<sha256>|<scope>`): records of 0.8.9 and older
replay unchanged (as entries without a digest), and a 0.8.9 binary reads
the new records as entries of a scope it never looks up (they never hit)
instead of stopping its replay.

## Reasoning

`oracle.reasoning` (`off` by default; `low`, `medium`, `high`) lets the
oracle model reason before it answers, at that OpenRouter effort
(`reasoning: {effort, exclude: true}`: the reasoning text is not returned,
the verdicts are still the final message). It is a trade of accuracy against
latency and cost: every call's `max_tokens` grows by
`oracle.reasoning_max_tokens` (4096) — which the reservation, and so the
budget, accounts for — its deadline by `oracle.reasoning_deadline_s` (60 s,
on top of `deadline_s`), and the reasoning tokens are billed: OpenRouter's
`usage.cost` includes them, the ledger's `settled` line and
`cmf.usage.oracle.reasoning_tokens` show how many there were. Measure it on
your own traffic before turning it on; the provider must support reasoning
(`provider.require_parameters: true` keeps OpenRouter from routing to one
that does not, but not from one that accepts the parameter and ignores it:
check that the ledger shows reasoning tokens, and list such a provider in
`provider.ignore`).

A reasoning call that outgrows its token allowance (`finish_length`) or its
deadline (`read_timeout`, `transport_timeout`) is asked once more without
reasoning, so the question still gets the oracle's direct answer. Both calls
are in the ledger (the first one is billed when the provider bills it), the
stop rules count the second one's outcome, and a follower waiting for the
same question waits for both.

## Budget and stop rules

* **Reservation first.** Before each call the server reserves
  `((body bytes + 4096) · max_price.prompt + max_tokens · max_price.completion) / 1e6`
  USD. The call is made only if spent + reservations in flight + this
  reservation ≤ `budget_usd`, calls < `max_calls`, for the caller's key
  spent + reservation ≤ its `oracle_budget_usd`, and, for a key with
  `credit_usd` when the oracle is billed (`oracle_passthrough`), reservation
  × `oracle_markup` ≤ the credit left.
* **Ledger.** `oracle.jsonl` gets a `reserved` line, fsynced, before the
  request leaves; after the answer a `settled`, `failed_billed` or
  `failed_unknown_cost` line with the cost, tokens and latency. At start any
  reservation without a closing line counts as spent.
* **Checks of every answer.** HTTP 200 within `deadline_s` (no retries), a
  finite `usage.cost` ≥ 0, `finish_reason: stop`, exactly the asked question
  ids with the schema's types. A 200 that carries only an `error` is a failure.
* **Stop rules** switch the oracle off (reason in `oracle.state`) until
  `POST /v1/admin/oracle {"enabled": true}` (or `cortiq decide … --oracle
  --oracle-resume` on a state directory no server holds): HTTP 401, 402 or
  403 from OpenRouter; a returned model that is not the configured one; a
  cost above the reservation; `max_errors` failures in a row (counted across
  restarts and `cortiq decide` runs). An upstream refusal that the prompt
  does not fit the model's context (a 400/413/422 whose body names the
  context length) is the failure `context_length`: the question gets a 422
  whose message says `maximum context length` (a trained one abstains), it
  does not count toward `max_errors` — a run of long items must not stop a
  working oracle — and its reservation counts as likely unbilled.
* `GET /v1/admin/oracle` shows spent, reserved, remaining, calls, failures and
  the stop reason; `POST /v1/admin/oracle` can switch the oracle and lower
  `budget_usd` / `max_calls` within the configured values. Since 0.8.11 the
  switch and the stop rules stop calls only: answers already in the cache
  are still served (step 3); `cache.enabled: false` and a restart stop
  those.

## What leaves the machine

* **Sent**, only for undetermined questions with the oracle permitted: the
  `state` and the `instructions` and `criteria` of those questions. In a
  state-less request (empty `state`) the instructions are the input; since
  0.8.8 every question's instructions and its criteria's descriptions are
  redacted like a state when PII redaction is on (below), the option ids
  never. Receivers:
  OpenRouter and the provider it routes to (`provider.sort: price`,
  fallbacks allowed; set `oracle.data_collection: "deny"` to exclude providers
  that store data).
* **PII redaction** is off by default: the text is sent as asked. It was on
  through 0.8.11, and its secret-like pattern also rewrote tool names, slugs
  and chemical names (10,225 questions of one Decision Index run). Turn it
  on with `"oracle": {"redact_pii": true}`; then e-mail addresses,
  secret-like tokens (20 or more characters of `[A-Za-z0-9_-]` with a digit and
  a letter) and numbers of 9 or more digits — also when their digit groups
  are separated by spaces, dashes, dots, slashes or parentheses, as in
  `4111 1111 1111 1111`, `+1 (555) 123-4567` or a spaced IBAN — in every
  string of the state and, since 0.8.8, of each question's instructions and
  criteria descriptions (object keys — the option ids — and a score level's
  position are kept) are replaced by `[REDACTED]` and the question gets the
  flag `pii_redacted`. Only the copy sent is redacted: the cache scope and
  the auto-skill contract are those of the question as asked, so caching and
  learning do not change. It is a heuristic: names, postal addresses, numbers
  written in words and identifiers with letters between short digit groups
  are not detected. With it on, a request can opt out with
  `cmf.allow_pii_egress` (router: `options.allow_pii_egress`).
* **Never sent**: accepted questions, other questions of the request, client
  keys, accounts, vectors.
* **Kept on disk** in the state directory: vectors and hashed features of
  learned examples and cached answers, since 0.8.11 the sha256 of each cached
  question's input as asked (the whole state, or a state-less question's
  instructions, before PII redaction), the oracle ledger (no texts), usage
  records (no texts). Hashed n-gram features can show whether a known text was
  seen, and an input's sha256 confirms a known text exactly (a short e-mail,
  a phone number, a card number guessed and hashed), so treat the state
  directory as sensitive (it is created with mode 0700).
* The OpenRouter key is read from the environment variable named by
  `oracle.api_key_env`; it is never written to the configuration, the model,
  the state or the logs.

## Connect OpenRouter

Create an account at [openrouter.ai](https://openrouter.ai), add credit, and
under *Settings → Keys* create a key with a credit limit of its own. Then:

```bash
export OPENROUTER_API_KEY="<your OpenRouter key>"                     # step 1
cortiq serve cortiq-decision.cmf --oracle deepseek/deepseek-v4.1-flash  # step 2
```

That is the whole setup. The model is any
[openrouter.ai/models](https://openrouter.ai/models) id with structured
outputs (JSON schema); `deepseek/deepseek-v4.1-flash` is the one every
number on this page was measured with. At start the server:

* makes **one public request** without the key, `GET
  https://openrouter.ai/api/v1/models/<MODEL>/endpoints`, and sets the max
  price (which caps the provider price and sizes every reservation) to
  twice the prompt and completion prices of the model's cheapest endpoint
  with structured outputs;
* **refuses to start** with a message that names the problem — and 2–3
  cheap models that fit — when OpenRouter does not list the model, when
  none of its endpoints supports structured outputs, or when it has only
  variable prices (as `openrouter/auto`: give `--oracle-max-price` or pick a
  concrete model); it also refuses a given max price below every
  structured-output endpoint, since OpenRouter would refuse every call;
* falls back to a max price of $0.10/$0.50 per 1M in/out, with a warning,
  when the listing cannot be fetched;
* logs one line about the oracle, never the key:

```text
oracle: ready — deepseek/deepseek-v4.1-flash via openrouter.ai, budget $1.00, max price in/out $0.06/$0.58 per 1M (2× the cheapest structured-output endpoint, … at $0.03/$0.29)
oracle: NOT ready — OPENROUTER_API_KEY is not set (set it to your OpenRouter key and restart; …)
```

Everything else keeps its safe default: a budget of $1.00, provider
routing `{sort: price, require_parameters: true, allow_fallbacks: true}`,
the stop rules, and the oracle only for questions the local model cannot
decide. The text is sent as asked: PII redaction is an opt-in
(`oracle.redact_pii` in `--decision-config`). Optional companions of `--oracle` (each
overrides `--decision-config`):

| Flag | Default | Meaning |
|---|---|---|
| `--oracle-budget USD` | 1.0 | the most this server spends on the oracle |
| `--oracle-max-calls N` | 10000 | the most oracle calls |
| `--oracle-max-price IN,OUT` | 2× the cheapest structured-output endpoint | max price, USD per 1M prompt / completion tokens |
| `--oracle-key-env VAR` | `OPENROUTER_API_KEY` | the NAME of the variable that holds the key: 1–128 bytes of `[A-Za-z0-9_]` starting with a letter or `_`. A value that looks like a key (see [below](#key-like-values)) is refused and never shown, only its length |
| `--oracle-base-url URL` | `https://openrouter.ai/api/v1` | https; plain http only to a loopback address (a local proxy or a test mock); no credentials, query or key in it (a URL holding `sk-or-` is refused unshown) |
| `--no-oracle-learning` | off | answers are cached but the served model never changes |

A value the numeric flags (`--oracle-budget`, `--oracle-max-calls`,
`--oracle-max-price`, and `--max-price` of `oracle check`) refuse is never
shown either — only the flag, the value's length and why — and neither is a
string a `--decision-config` gives where a number belongs.

<a id="key-like-values"></a>**Key-like values.** A value *looks like a
key* when, checked in this order, it holds `sk-or-` anywhere or starts with
`sk-` (ASCII letters in any case), starts with `Bearer ` (any case, a space
after the word), has leading or trailing ASCII whitespace, is 41 bytes or
longer without a `/`, or holds 32 or more hexadecimal digits in a row. The
test applies as it is to `--oracle MODEL` and `oracle check --model`
(which must also be 1–256 bytes without whitespace or control characters,
and `author/slug`: a `/`, no empty, `.` or `..` segment); to the value of a
numeric flag, its surrounding whitespace trimmed; and to every command-line
argument, or the value of a `--flag=VALUE`, that a usage error would quote
(shown only by its length). For a variable's name (`--oracle-key-env`,
`oracle check --key-env`) a value of only `[A-Z0-9_]` starting with an
upper-case letter or `_` (such as
`MY_COMPANY_PRODUCTION_OPENROUTER_API_KEY_V2`) is a name at any length,
unless it holds 32 hexadecimal digits in a row; any other value is judged
by the test above and, besides, taken for a random token when it is 16
bytes or longer with letters and digits and no `_`. A base URL holding
`sk-or-` in any case is refused unshown.

Check it — `/healthz` needs no token, the admin view needs
`CORTIQ_DECISION_ADMIN_TOKEN` in the server's environment:

```bash
curl -s http://127.0.0.1:8080/healthz | jq -r .oracle_status                  # → ready
curl -s http://127.0.0.1:8080/v1/admin/oracle -H "x-admin-token: $CORTIQ_DECISION_ADMIN_TOKEN" \
  | jq '{status, model, max_price, budget_usd, spent_usd, calls}'
```

`status` is one of `ready`, `no_key` (the variable is unset or empty),
`bad_key` (the variable holds something that is not a key — see below;
`key_problem` says what, by position and length), `disabled` (not
configured, or switched off by `POST /v1/admin/oracle`),
`budget_exhausted` (something was spent and the budget or `max_calls` is
used up: what is left cannot hold the smallest possible call, or a call it
refused; restart with a larger `--oracle-budget` / `--oracle-max-calls`),
`budget_too_small` (nothing spent, and the budget
cannot hold even one call, or `max_calls` is 0: `min_call_usd` is the least
budget a call needs — the smallest possible call's reservation, one short
question, until the budget refuses a real, longer call; from then on the
larger of the two, that call's reservation, whether the budget was below
even the smallest call (0 included) or only below that one, so the status
is `budget_too_small` then, not `ready`; the hint of the refused request
names it rounded up to the micro-dollar, so passing that figure admits the
call; the startup line, before any request, can only give the smallest
call's, "more for longer questions"; with `max_calls` 0 the hint names the
call limit, and the budget too only when it is short as well) or `stopped: <reason>` (a stop rule; `last_error` names
the code of the last failed call, which `max_errors` counts; `POST
/v1/admin/oracle {"enabled": true}` resumes after the fix). An **admin
limit** (`budget_usd` or `max_calls` set by `POST /v1/admin/oracle`) is
kept in `oracle.state` and outlives a restart, so a larger
`--oracle-budget` or `--oracle-max-calls` alone cannot lift it: when it is
among what refuses the next call, the startup line (with the file's path)
and the hint (`the server's oracle.state`) name it and the admin request
that lifts it, e.g. `the admin limit budget_usd $0.00 in
STATE/oracle.state binds (…; the next call needs a budget of $0.00178924):
raise it with POST /v1/admin/oracle {"budget_usd": 0.00179} (at most the
configured $5.00) or lift it with {"budget_usd": null}` — a figure rounded
up to the micro-dollar that the admin API takes (never above the
configured budget; when rounding would put it above, only `null` is
offered); when the configured limit refuses the call as well, the hint
says to lift the admin limit and restart with the flag of at least the
figure. On the router
API, `/v1/healthz` carries the status as `cmf.oracle_status` only with
`x-cmf-extensions: 1`, so the router's own shape stays exact.

The key is read as it is in the variable, less surrounding spaces, tabs, CR
and LF (a `.env` file's line ending): the server, `decide` and `oracle
check` warn `the key had surrounding whitespace, trimmed` and go on. A value
that still holds whitespace, a control byte or a byte outside ASCII, that
starts with `Bearer ` (the request adds it), that is a whole `.env` line
(`NAME=…`) or starts or ends with a quote is `bad_key`: it is never sent, and the message names only the position
and the length of the problem, both counted among the variable's raw bytes
(the whitespace around the key included, as an editor shows the `.env`
line). No message, log line (at any `RUST_LOG` level), status, ledger or
state file ever holds a byte of the key: a failed request is a fixed code
such as `transport_connect`, `transport_timeout` or
`transport_bad_header`; an answer that did not finish is `finish_length`,
`finish_content_filter`, `finish_tool_calls`, `finish_error` or
`finish_other`, never the upstream's `finish_reason` text; every code
shown comes from that closed set (with `http_` and a three-digit status),
and a `stop_reason` or `last_error` read back from `oracle.state` outside
it (an older version's file, or one edited by hand) is shown as
`unknown_code` — still a stop. A name an upstream answers (a response's
`provider` and `model`, a listing's provider names and model ids) is kept
only as it is — at most 64 bytes of `[A-Za-z0-9 ._:/()-]` — and written or
printed as `[redacted]` otherwise, never filtered or cut; also when it
holds `Bearer` (any case), looks like a key, or holds 8 bytes in a row of
the configured key, and when its letters and digits alone (an echo
interleaved with dots, dashes or spaces) hold `bearer`, `skorv` or 8 in a
row of the key's letters and digits. The public listings are fetched
without the key, but their names are checked against it too (the value of
its variable); a suggested model id must also be one `--oracle` takes. A
key pasted where the command line takes no value (`cortiq decision oracle
check sk-or-…`, `--test-call=sk-or-…`) is shown in the usage error only by
its length.

Out of scope: an upstream at the configured base URL (OpenRouter, or your
own proxy) that has already received the key and deliberately echoes it,
or pieces of it, in its own answers holds the key already — Cortiq still
copies none of its free text into another client's answer, and every code
it shows comes from a closed set. So is a key typed into an argument that
is not about the key or the oracle (a file path, `--host`, a skill id).

### Check your setup

`cortiq decision oracle check` tells whether the oracle is ready before
anything is started or spent. It reads the same flags as `--oracle`
(`--model`, default `deepseek/deepseek-v4.1-flash`; `--key-env`;
`--base-url`; `--max-price IN,OUT`, which a model listed only with variable
pricing such as `openrouter/auto` needs, as `--oracle-max-price` does):

```bash
cortiq decision oracle check               # free: the key, the account, the model
cortiq decision oracle check --test-call   # and one tiny structured call (a 2-option choice, max_tokens 16)
```

```text
Oracle check: deepseek/deepseek-v4.1-flash via openrouter.ai
  ✓ key        OPENROUTER_API_KEY is set
  ✓ account    the key is valid (credit limit $10.00, $1.25 used, $8.75 left)
  ✓ model      3 endpoints, 2 with structured outputs; the cheapest is MockProvider at $0.03/$0.29 per 1M in/out, so --oracle sets the max price to $0.06/$0.58
  ✓ test call  answered 'yes' for $0.00001 (reserved $0.000317, provider MockProvider, 0 ms)
ready: cortiq serve FILE --oracle deepseek/deepseek-v4.1-flash   (or: cortiq decide FILE -p TEXT --oracle deepseek/deepseek-v4.1-flash)
```

Each line is one check: **key** — the variable holds a usable key (only its
presence is shown, never the key; surrounding whitespace trimmed is said,
anything else wrong is `bad_key`); **account** — `GET /auth/key` accepts the key
(free) and shows its credit limit and usage; **model** — OpenRouter's
public endpoint listing (free, no key) has the model with structured
outputs, and its cheapest price; **test call** — with `--test-call` only,
one call through the same reservation ledger and stop rules as a server,
with its answer and cost. `✓` passed, `✗` failed (the line says why and
what to do, and for a model it names 2–3 cheap ones that fit), `–` not
checked. The exit code is 0 only when every check passed (and the test
call answered, when asked); `--json` gives the same as one object with
`ready` and `problems[]` (`no_key`, `bad_key`, `key_refused`, `no_credit`,
`account_unreachable`, `unknown_model`, `no_structured_outputs`,
`variable_price`, `max_price_too_low`, `listing_unreachable`,
`test_call_failed`).

The commands of this section and the next were run as written against a
local mock of the OpenRouter API with a test key: `--base-url` (for `serve`
and `decide`, `--oracle-base-url`) pointed at the mock, so the outputs here
show `openrouter.ai`, and no `--oracle-base-url`, where the run showed the
mock's address. Endpoints, provider, prices, credit, costs and the oracle's
answers are the mock's; nothing was sent to OpenRouter.

**Statuses.** A server reports its oracle in the startup line (`oracle:
ready — …`, `oracle: NOT ready — <what to do>`, `oracle: off — …`), in `GET
/healthz` (`oracle_status`) and in `GET /v1/admin/oracle` (`status`);
`cortiq decide … --oracle MODEL` in the start line of a batch, in its
`hint` and in `cmf.oracle.status` of `--json`. Each message says what to do;
in short:

```bash
curl -s http://127.0.0.1:8080/healthz | jq -r .oracle_status     # the server of step 2 → ready
```

| Status | Means | What to do |
|---|---|---|
| `ready` | questions the gate rejects go to the oracle | nothing |
| `no_key` | the key variable (`OPENROUTER_API_KEY`, or `--oracle-key-env`) is unset or empty | `export OPENROUTER_API_KEY=…` where the server (then restart it) or `decide` runs |
| `bad_key` | the variable holds something that is not a key; nothing was sent | fix it as the message says — it names the problem (a quote, `Bearer `, a whole `NAME=…` line, whitespace, a control or non-ASCII byte) by position and length, never the key — then restart the server or run `decide` again |
| `disabled` | the server has no oracle (no `--oracle MODEL`, or `oracle.enabled` false), or the admin switched it off | start the server with `--oracle MODEL`; after the admin's switch, `POST /v1/admin/oracle {"enabled": true}` |
| `budget_too_small` | nothing spent, and the budget cannot hold one call (`min_call_usd`), or `max_calls` is 0 | `--oracle-budget` of at least the figure the message names (and `--oracle-max-calls` of 1 or more); for an admin limit in `oracle.state`, the `POST /v1/admin/oracle` the message names |
| `budget_exhausted` | the budget or `max_calls` is used up | a server: restart with a larger `--oracle-budget` / `--oracle-max-calls` (the budget counts the state directory's whole ledger, across restarts); `decide`: each run has its own budget, pass a larger `--oracle-budget`; for an admin limit, the `POST` the message names |
| `stopped: <reason>` | a stop rule: `http_401` or `http_403` (the key was refused), `http_402` (no credit), `unexpected_model`, `cost_above_reservation`, `max_errors` (failures in a row; `last_error` names the last one) | fix the cause (`cortiq decision oracle check` tests the key, the credit and the model), then `POST /v1/admin/oracle {"enabled": true}` on a server, or one `decide` run with `--oracle-resume` |

### One text or a batch

`cortiq decide` takes the same `--oracle MODEL` (and the `--oracle-*`
flags of the table above) for one text or a batch. The text is decided
locally first; only when the gate rejects it — or no skill has the asked
labels — is one call made, with the same reservation, stop rules and key
handling as a server; the text is sent as written (no PII redaction). A text
the gate accepts sends nothing.

```bash
cortiq decide cortiq-decision.cmf --skill banking77 -p "the exchange rate you gave me looks wrong" \
  --oracle deepseek/deepseek-v4.1-flash
```

```text
choice:     card_payment_wrong_exchange_rate (from oracle deepseek/deepseek-v4.1-flash, $0.00014)
action:     oracle (the gate rejected the local choice card_payment_wrong_exchange_rate), certified false
skill:      banking77 (exact match, 77 candidates)
gate:       p_top 0.5678609 (tau 0.7), novelty 0.84277564 (theta 0.804234), margin 0.0045858026, confidence 0.56217486
errors:     card_payment_wrong_exchange_rate 0.19301137, wrong_exchange_rate_for_cash_withdrawal 0.19957425, exchange_rate 0.35151693, wrong_amount_of_cash_received 0.43852428, extra_charge_on_statement 0.48889443
oracle:     deepseek/deepseek-v4.1-flash via openrouter.ai: $0.00014 spent in this run (1 call), budget $1.00; ledger cortiq-decision.cmf.state/oracle.jsonl: $0.00014 over 1 call in all
model:      cortiq/decision@386b6e43fd35 (generation 0), 3489 input tokens, 15828 µs
```

A batch reads one `{"text", "label"?}` object per line and writes one JSON
row per input, never the text:

```bash
printf '%s\n' '{"text": "I still have not received my new card", "label": "card_arrival"}' \
  '{"text": "can I pay my rent with a virtual card"}' > rows.jsonl
cortiq decide cortiq-decision.cmf --skill banking77 --input rows.jsonl --out results.jsonl \
  --oracle deepseek/deepseek-v4.1-flash --oracle-budget 0.05
jq -c '{answer, action, source, oracle_cost_usd, flags}' results.jsonl
```

```text
{"answer":"card_arrival","action":"local","source":"local","oracle_cost_usd":0.0,"flags":[]}
{"answer":"get_disposable_virtual_card","action":"oracle","source":"oracle","oracle_cost_usd":0.0001397,"flags":[]}
```

* **Columns.** A batch row keeps the columns of a run without `--oracle`
  (`choice`, `p_top`, `accepted`, …) and adds `answer` — the final answer:
  the oracle's or its cache's for `action` `oracle` / `cache`, else the
  local choice (for an `abstain` row that is the local model's choice the
  gate rejected, so read `action` before trusting `answer`) — `action`
  (`local`, `oracle`, `cache` or `abstain`), `source`, `oracle_cost_usd`
  (this row's call; 0 for a local or cached answer), `flags`
  (`pii_redacted`, or why a row abstained: `budget`, `no_key`, `bad_key`,
  `stopped`, `oracle_unavailable`, …) and, for a labelled row, `answer_correct`
  (`answer` equals the label). The oracle's totals and a `hint` are in the
  summary on stderr. For one text, `--json` carries `action` and `source`
  in `cmf.questions.task`, the call's cost in `cmf.usage.oracle` and the
  run's status, spend and budget in `cmf.oracle`.
* **Without a usable key** a rejected text the state directory's cache
  holds (the oracle answered the same text in an earlier run, or a server
  on that directory did) is answered from it, `action: cache` (since
  0.8.11; the directory is only read — no `LOCK`, nothing written); any
  other rejected text abstains with `no_key` (or `bad_key`) and the hint
  `OPENROUTER_API_KEY is not set (decide --oracle reads the key from the
  environment): export …`. Nothing is sent and no state directory is made.
  Labels no skill has can only be answered by the oracle or its cache, so
  there the same words end the run with an error on a miss.
* `--oracle-budget USD` (default $1.00) and `--oracle-max-calls N` cap each
  run; the rows past the cap abstain with the flag `budget`. A budget that
  cannot hold the run's first call — 0 included — is `budget_too_small`
  (the hint names that call's reservation and the least `--oracle-budget`
  that holds it, rounded up to the micro-dollar: rerun with that figure and
  the call is made; a batch's start line, before any row, gives the
  smallest call's, "more for longer questions"); one the run spent is
  `budget_exhausted`. Without
  `--oracle`, `decide` never calls the oracle.
* **State.** The reservation ledger, the answer cache and `oracle.state`
  live in the state directory, `<FILE>.state` next to the file (or
  `--state DIR`), under its `LOCK`: one process per directory — a running
  `cortiq serve` on it is asked through its API instead. A repeated text
  comes from the cache for free. Admin limits a server set there
  (`budget_usd`, `max_calls` in `oracle.state`) count the whole ledger and
  hold for `decide` too, whatever `--oracle-budget` or `--oracle-max-calls`
  say; its messages name the one that binds, with the file's path and the
  figure the ledger needs, and how a server of that directory lifts it
  (`POST /v1/admin/oracle {"budget_usd": null}`, or that figure rounded up
  when it is within the server's configured budget).

**After a stop rule** (a refused key, no credit, another model, a cost
above the reservation, `max_errors` failures in a row) the oracle of that
directory stays off for every later run: a rejected text abstains with the
flag `stopped` and a hint that names the rule. After the fix, add
`--oracle-resume` once, as `POST /v1/admin/oracle {"enabled": true}` does
(here after an `http_402`, credit added):

```bash
cortiq decide cortiq-decision.cmf --skill banking77 -p "hello there" --oracle deepseek/deepseek-v4.1-flash --oracle-resume
```

```text
oracle: resumed — the oracle of state directory cortiq-decision.cmf.state was stopped by the stop rule http_402: OpenRouter answered HTTP 402: the key in OPENROUTER_API_KEY has no credits left (add credits at https://openrouter.ai/settings/credits, or raise the key's own limit); it may be called again
…
```

**After an interrupted run** (Ctrl-C, SIGTERM, a closed terminal) the
`LOCK` is released; a signal the run inherited as ignored (`nohup`, a
background job of a script) stays ignored. A `LOCK` left by a crash is
named with its pid (`state directory cortiq-decision.cmf.state has a LOCK
left by pid 42341, which is no longer running (an interrupted run): pass
--break-lock to remove it, or give this run a directory of its own with
--state DIR`); `--break-lock` removes it only when that process is gone:

```bash
cortiq decide cortiq-decision.cmf --skill banking77 -p "thanks for your help" --oracle deepseek/deepseek-v4.1-flash --break-lock
```

```text
warning: removed the LOCK of state directory cortiq-decision.cmf.state left by pid 42341, which is not running (--break-lock)
…
```

### Who may use it, and who teaches

* **No keys, loopback address** (the default `127.0.0.1:8080`): with
  `--oracle`, callers without a key may use the oracle within its budget —
  the operator enabled it on the command line — but never teach the model:
  their feedback is not learned and the oracle's answers to them are cached,
  not learned. A reverse proxy on the same host forwards anyone to a
  loopback address, so behind one set `auth.require: true` and use keys.
* **No keys, `auth.require: false`** in `--decision-config`: the explicit
  open mode, which may use the oracle and teach.
* **No keys, another address**: keys are required (`auth.require: null`
  means "required unless loopback").
* **Keys**: `cortiq decision keys create` makes keys that may use the oracle
  (`oracle_allowed: true`, like imported router keys); `--oracle-allowed=false`
  makes one that never calls the oracle — since 0.8.11 its undetermined
  questions are still answered from the shared cache when it holds the same
  question (other accounts' oracle answers; only `cache.enabled: false`
  turns that off) — and `--oracle-budget-usd` caps one key's
  oracle spending. Teaching the shared skills is a separate permission,
  `--learning-allowed` (off by default); it alone lets a caller's untrained
  contracts become [auto-skills](#auto-skills). Keys made through `POST
  /v1/admin/keys` keep the router's default (`oracle_allowed` only when the
  body says so).

### With a configuration file

For every oracle setting — deadline, error limit, provider options,
`data_collection: "deny"`, per-request consent — use `--decision-config`
(the `--oracle*` flags, when given, override its values):

1. Configure the oracle: `enabled: true` and a `model` from
   [openrouter.ai/models](https://openrouter.ai/models) that supports structured
   outputs (JSON schema). `provider.max_price` (USD per 1M tokens) caps the
   provider price and sizes the reservation. `base_url` must be https (plain
   http only to a loopback address).
2. Set limits: `budget_usd`, `max_calls`, `deadline_s`, `max_errors`; per key
   `oracle_allowed`, `oracle_budget_usd` and, for keys whose own questions
   and feedback may teach the model, `learning_allowed`.
3. Put the key in the environment of the server process, start the server
   and verify.

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
    "redact_pii": false
  },
  "cache": {"enabled": true, "threshold": 1.0, "legacy_cos": 0.9999},
  "learning": {"enabled": true, "refit_min_new": 25}
}
EOF
export OKEY=$(cortiq decision keys create --state ./oracle.state --plan developer \
  --account acme --oracle-allowed --oracle-budget-usd 0.50 --json | jq -r .key)
export CORTIQ_DECISION_ADMIN_TOKEN=$(openssl rand -hex 24)
cortiq serve cortiq-decision.cmf --decision-config oracle-server.json \
  --state ./oracle.state --port 8081
```

Verify: the admin view must show `status: "ready"` (`configured: true`: the
configuration enables the oracle; `key_present: true`: the key variable is
set in the server's environment). `enabled: true` only means that no stop
rule or admin call has switched the oracle off; it is true with the oracle
off in the configuration as well. Then a question the gate rejects comes back from the
oracle (on the published model, "can you recommend a good tattoo artist" is
out of scope for `clinc150` and is rejected by its gate).

```bash
export ORC=http://127.0.0.1:8081
curl -s "$ORC/v1/admin/oracle" -H "x-admin-token: $CORTIQ_DECISION_ADMIN_TOKEN" \
  | jq '{status, configured, key_present, enabled, model, budget_usd, spent_usd, stop_reason}'   # → 200
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
`consent_off` or `budget`, check `status`, the key's `oracle_allowed`
(without keys: `auth.require: false`, or `--oracle` on a loopback address)
and the budget; `stop_reason` names a stop rule, and the decisions API's
`cmf.hint` says what to do. The same questions through
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
has the cache and self-learning on, with learning inline
(`learning.synchronous`) and PII redaction off so that the request bodies
match the stored ledgers, in open mode (`auth.require: false`), with
`deadline_s` 200, `budget_usd` 10 and `max_calls` 1000000; mode B has the
cache and learning off. Oracle answers were replayed from the stored
DeepSeek ledgers through a local proxy; the 24 requests of MASSIVE mode A
that no ledger held were sent live.

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

**In one pass over the test sets mode A (cache and self-learning) saved only
35 of 2003 oracle calls (1.75 %)**: 21 were cache hits (1.05 %) and 14 were
questions answered locally after the 16 promotions on MASSIVE (0.70 %). Per
set that is 7 of 289 (2.42 %), 4 of 355 (1.13 %) and 24 of 1359 (1.77 %),
with cascade correct −1, 0 and 0.
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
