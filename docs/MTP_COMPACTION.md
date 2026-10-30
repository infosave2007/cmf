# Independent FFN compaction of the MTP draft

*Languages: **English** · [Русский](MTP_COMPACTION.ru.md) · [中文](MTP_COMPACTION.zh.md)*

`cortiq mtp-compact` shrinks **only** the dense FFN layers of the MTP draft. The
main backbone is not repacked: the plan explicitly records
`trunk.mode = preserve`. This lets you size the draft layers on their own and
then measure whether the lighter weights pay for the drop in acceptance. The
command promises neither quality nor a speedup: both have to be confirmed by
the test described below.

> MTP here is the draft for speculative decode, not the main MoE/trunk. Physical
> compaction changes the neurons of the draft FFN and can therefore change the
> number of accepted draft tokens, but it must not change the verified greedy
> output of the main model.

## What is supported

Two placements of a canonical **dense** MTP are supported:

1. `model.mtp.*` lives in the main CMF: the result is a new main `.cmf`.
2. The MTP lives in a sidecar `model.mtp.cmf` (typical for MiMo): the command
   finds it automatically and creates a **ready pair** for the candidate.

Preferably, `--output` names the path of the new runnable **main** CMF. For
MiMo the command creates the matching `<output-stem>.mtp.cmf` next to it by
itself: an unsharded backbone is first hard-linked from the source (no extra
copy), and copied if the filesystem does not support hard links. A sharded
backbone is instead coalesced into one candidate in a stream. A hard link is an
alias of the same bytes that the command never modifies: `mtp-compact` writes
neither the source path nor the alias. Do not modify the source while the job
runs, and after a successful hard-link run do not edit the candidate main "in
place" — it is the same inode as the source. If you need a main you can modify
independently, make a separate copy/reflink after validation. So one command
yields a working pair, not a lone sidecar:

```text
base/model.cmf              # source backbone, unchanged
base/model.mtp.cmf          # source draft, unchanged

candidate/model.cmf         # new runnable main (hard link/copy/coalesce)
candidate/model.mtp.cmf     # new compact draft
```

For compatibility with early scripts, an external MiMo draft can be named
explicitly as `--output candidate/model.mtp.cmf`. In this **legacy
sidecar-only** mode only the new sidecar and the plan are published;
`candidate/model.cmf` is not created, and such a result is not a runnable model
on its own. It has to be placed by hand next to a compatible main CMF. For new
or automated runs, always give the main path without the `.mtp.cmf` suffix.

Do not replace the source. The command refuses to overwrite an existing main,
sidecar or plan, including a dangling symlink. A sharded **main** is supported:
the streaming writer reads the merged shards through mmap and creates one new,
self-contained runnable unsharded candidate (`candidate/model.cmf`). For MiMo
this also works with a sharded main and one ordinary `.mtp.cmf` sidecar: the
main is coalesced and the draft is compacted next to it. A sharded external
`.mtp.cmf` is still rejected fail-closed — the MiMo loader deliberately looks
for a single sidecar with a derived name.

## Calibration and the plan

Supply a finalized, verifiable `.tdata` selector. Use only the
train/calibration split for channel selection; keep validation/test for later
evaluation. The command itself verifies the main CMF, the discovered MTP
sidecar and the `.tdata`. Before the runtime is loaded, the preflight parses
**all** `records.jsonl`/`tokens.idx` lines (including unselected val/test) and
authenticates the payload by size/SHA from the manifest. No `Vec<u32>` is
allocated for unselected records; a legacy v1 without a manifest descriptor
goes through the same read fence with a fixed 64-KiB buffer. The selected train
records are then read by an ordinary streaming pass for calibration.

After this preflight the `.tdata` must stay immutable until the job ends. The
source CMF gets a start/end identity fence, while a second full SHA pass over
the dataset after hours of calibration is deliberately not done: it would be
one more full I/O cost. Do not replace `records.jsonl`, `tokens.idx` or the
manifest under a running command.

This is O(1) specifically in the **token payload** of unselected records. For
the native `input.format=jsonl-v1`, the strict prompt-ledger check deliberately
keeps `id → split` and the set of seen IDs, so it needs memory that grows with
the number of prompts. With millions of prompts do not treat this ledger as
free: plan the RAM, or prepare a separate finalized calibration selector. This
is a conscious price for detecting a duplicated or substituted prompt outside
the selected split; the command does not quietly weaken it.

If the manifest carries a native `tokenizer` contract, the command checks
`vocab_size`, `vocab_sha256` and `config_sha256` against the main CMF,
fail-closed, before the first forward. This prevents silently calibrating on
token IDs of another model whose ID range overlaps. An old generic `.tdata`
without a contract is still supported, but it prints a warning and is marked in
the plan as `tokenizer_contract: "absent-legacy"`; a matching native contract
gets `"verified-v1"`.

Calibration is currently canonically **CPU-only**: `probe_mtp_ffn` deliberately
runs the teacher-forced trunk and the pre-down draft activation on the CPU,
with deterministic `f64` accumulation. For the duration of this probe the
runtime forcibly turns off `CMF_FFN_GATE_TOPK` and `CMF_FFN_ORACLE_TOPK`, so the
plan always describes the full dense draft rather than an accidental serving
experiment. `CMF_GPU=1` does not speed this stage up and must not be read as a
promise of GPU conversion; Metal/Vulkan are checked on the finished candidate.
This protects one and the same plan from differences in FMA/SIMD and in
reduction order between backends.

Physical writing is built for large models: preserved trunk tensors go from
mmap to `CmfStreamWriter` in 16 MiB pieces, with no `Vec<u8>` for the whole
model. In the private `mtp-compact` stage, the already verified hash of an
unchanged payload is reused at write time, so the source is not reread just to
get the same hash; before publishing, the candidate is still fully verified
again. The MTP FFN is rewritten projection by projection: only the source and
the compacted f32 matrix of one projection are alive at a time (for
`down_proj.t`, also its temporary transpose); the encoded `down_proj` is
explicitly freed before the transpose is built, instead of holding all of
gate/up/down at once. Canonical MTP calibration also does not store the whole
trunk hidden history; MiMo replay keeps 128 trunk rows inside one probe. The
main memory bound is now explicit: every source record is split into contiguous
fresh-causal chunks no longer than `--max-record-tokens` (default 4096), and
the trunk/draft KV is reset at each chunk boundary. So one abnormally long
`.tdata` record cannot grow the KV to the length of the whole record; memory
for causal-state calibration is bounded by the cap and the model geometry, not
by the record's context size. The `.tdata` reader itself still materializes the
selected record as a `Vec<u32>`, so do not store one pathological
multi-million-token record in place of normal calibration records. This does
**not** preserve attention across the boundary: for a quality-critical
long-context model, choose a cap comparable to the working context and the
available RAM rather than lowering it for speed. The disk must hold the source
and the new candidate at the same time. Do not run two rewrites of the same
model into the same output directory at once.

For source `q4tp`/`q2tp` gate/up an additional exact fast path applies: a kept
neuron is a whole row with an unchanged hidden axis, so its weight plane, row
ladder and scale code are copied byte for byte in the new order. This keeps the
decoded selected row **exactly**, runs no second PTQ pass and allocates no f32
gate/up panel. `down_proj` and `down_proj.t` cannot be copied this way:
compaction selects and reorders their columns, which changes the codec groups;
`down_proj` is repacked, and the transpose is rebuilt from it. But for
`q4tp`/`q2tp` the writer decodes for `down_proj` only the source columns that
are actually selected, straight from the row/group planes, without
materializing the full source f32 panel, and then encodes the compact matrix
the usual way. The selected columns are first grouped by their source 32-wide
tile, so the ladder/code is read once per group/row, while the result keeps the
ranking order. For a codec without one of the direct paths described here and
below, gate/up and down stay on the correct dequant → select → requant path.

For raw `f32`/`bf16`/`f16`, `q8_row` and `q8_2f` the fast path is exact too,
for a different reason: gate/up copy the selected rows, and `down_proj` selects
and reorders raw values or q8 codes together with their source row/column
scales. So narrowing runs no extra PTQ pass and adds no error on top of the
source CMF; `bf16` keeps its dtype and is not silently downgraded to f16. If
`down_proj.t` exists, only the already compacted selected `down_proj` is briefly
decoded to build the transpose; the full source `down_proj` is not
materialized. Other codecs remain on the fail-safe fallback.

The practical RAM budget is set by the **single** largest draft projection, not
by the trunk. The score pass now computes `down_proj_column_l2_squared` straight
over the mmap bytes, holding only a `Vec<f64>(I)`, for `f32`, `f16`, `bf16`,
`q8_row`, `q8_2f`, `q4tp` and `q2tp`; the order of f32 dequantization and f64
summation matches the historical full-panel ranker. So this stage creates no
temporary `[H,I]` f32 buffer for the usual production codecs. A rare unknown or
complex codec stays on the fail-safe `dequant → score` fallback and may need the
full source panel. For raw `f32`/`bf16`/`f16`, `q8_row`, `q8_2f`, `q4tp` and
`q2tp` the bake also does not allocate the full f32 source `down_proj`; during
the `down_proj.t` stage the compacted `down_proj`, its f32 transpose and the
encoded transpose are needed at the same time; the already written encoded
`down_proj` has been freed by then. Other codecs may still briefly hold the
source and the compacted panel during the bake itself.

Before the score pass of each draft layer the command prints the layer index,
the `down_proj` shape, the codec and the elapsed time; this is a deliberate
separate progress fence for very wide matrices, not a sign of a hang. The
plan's layer report keeps `down_proj_dtype`, so after the job you can see which
storage path was scored.

This is a deliberate trade-off: it does not need a full `Vec<TensorSpec>` or the
full model in RAM, but a very wide single MTP layer still needs enough RAM.
When in doubt, first run one layer or a small model and watch RSS; do not run
parallel conversion jobs.

Integrity is a conscious price of safety: the source is fully verified before
and after calibration; after the bake the source MTP draft is verified once
more, and before publishing the candidate is fully verified. For 100B+ that is
several sequential passes over the disk; it is not a "hang" and not a promise of
instant conversion. Every full-CMF scan, the external MiMo sidecar included,
prints its start/end and wall time; calibration progress is printed after the
first and then every eighth bounded chunk. Do not modify the source until the
command finishes: the repeated check catches an ordinary in-place
substitution, and the command does not publish a mixed candidate.

On the real teacher-forced MTP path it collects, for each channel,

```text
sum_teacher_forced((activation(gate, up))^2) * down_proj_column_l2_squared
```

and keeps the channels with the highest score. This is not a trunk statistic
and not a guess from the backbone's `intermediate_size`. `activation(gate, up)`
is exactly the MLP function of the given draft from the runtime (for example
`silu(gate) * up`), not always SiLU; its contract is also recorded in the plan.
`--keep-ratio` applies to each draft layer, after which the width is rounded
down to a group of 32 channels; the minimum is one 32-channel group. The
resulting JSON plan records the selector, the number of selected records read
and, separately, the records that actually contributed, the actual
chunks/tokens, the cap and the splitting policy, the tokenizer contract status,
the separate directory hashes and the merged `trunk_hash` of main/draft, the
channel order and the share of energy kept — store it next to the artifact.
`trunk_hash` covers all merged shard tensor entries and their payload hashes.
With `mtp.mode=compact`, `cortiq tube-bake` checks the required
`source_draft_directory_hash` and, in new plans, `source_draft_trunk_hash`
against the input draft; a calibrated plan cannot be applied by accident to
another same-shape sidecar or to a late shard that was swapped.

Short records/chunks still contribute to the statistics of the draft layers
they reach, but the admission prefix is taken from the **longest** suitable
bounded chunk (at most 256 tokens and no shorter than `draft_layers + 2`). This
gives the small runtime gate more covered positions without holding the whole
record. If there is no such chunk, or some layer has zero activation energy
after aggregation, the command refuses to pick channels by an arbitrary
tie-break rather than emit a formally valid but unverifiable compact draft.

## A simple command for your own prompts

The common case needs neither a separate teacher nor a hand-made data
directory:

```bash
cortiq compact source.cmf prompts.jsonl task-model.cmf
```

`prompts.jsonl` is plain JSONL: one task per line. A simple `prompt` is enough,
and for a chat model you can pass a full `messages` array:

```jsonl
{"id":"sql","prompt":"Write a safe SQL query that looks up an order by id."}
{"id":"review","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"Check this Rust code for off-by-one errors."}]}
```

This form uses `source.cmf` as the teacher, creates `task-model.tdata` next to
the candidate, generates deterministic direct-answer records from
`prompts.jsonl`, finalizes and verifies the set, then calibrates/physically
compacts the MTP draft and publishes only the verified runnable
`task-model.cmf`. On a rerun with the same output, the already finalized
`task-model.tdata` is **not overwritten** but reused; for a different prompt
set choose a new output/cache (or deliberately delete the old cache before the
run).

This is not generic fine-tuning: the trunk stays byte-identical to the source,
and only the dense FFN of the existing MTP draft changes. The speculative
verifier is still the source of the emitted tokens, so greedy output must not
change because of the accepted draft. A model without a supported dense MTP is
rejected fail-closed instead of receiving a fictitious "adaptation". To control
the ratio, the split and the limits, the simple form accepts `--keep-ratio`,
`--max-tokens`, `--samples`, `--seed`, `--split`, `--limit`, `--max-records`,
`--max-record-tokens`, `--keep-truncated` and `--disable-mtp`. For example, a
quick smoke run that narrows the draft by 25%:

```bash
cortiq compact source.cmf prompts.jsonl task-model.cmf \
  --keep-ratio 0.75 --max-tokens 512 --limit 24 --split 100/0/0
```

For a separate teacher, an independent `.tdata` path, a split seed or an audit
plan, use the extended `mtp-compact` below.

## One run: create or reuse teacher data

`mtp-compact` can prepare a missing calibration set by itself before the
physical compaction. This orchestrator is **only for the dense MTP draft FFN**;
it is not a general `compress` for the trunk, MoE or DSpark.

```bash
BIN=target/release/cortiq

# If calibration.tdata does not exist yet, the command creates it from teacher/prompts,
# finalizes it, then calibrates right away and writes the candidate.
"$BIN" mtp-compact base/model.cmf \
  --data calibration.tdata \
  --teacher /models/teacher.cmf \
  --prompts calibration_prompts.jsonl \
  --teacher-max-tokens 512 \
  --teacher-samples 1 \
  --teacher-seed 42 \
  --teacher-split 90/5/5 \
  --keep-ratio 0.90 \
  --max-record-tokens 4096 \
  --output candidate/model.cmf
```

`--data` stays an explicit path so that the generated corpus can be inspected,
kept and reused. `PATH.tdata[:train][=SHARE]` is accepted: compaction uses only
the train split (and, if requested, a deterministic share of it), leaving
val/test for separate evaluation. When the set is created automatically, the
directory root is always the `PATH.tdata` part before the selector/suffix. When
the directory does not exist, `--teacher` and `--prompts` are both required;
the teacher generates deterministic greedy
direct-answer records (`--no-think`, `top-k=0`, no `--raw`), and the writer
first finalizes an immutable `.tdata`. The teacher may be the source itself, but
the set's token IDs and tokenizer contract are still checked against the source
before the first calibration forward. Even **before** the long generation, a
metadata preflight compares the exact bytes of the source's and the teacher's
vocabulary and tokenizer config: an incompatible pair is rejected without
wasting pod time. For a short smoke set that is compacted right away, set
`--teacher-split 100/0/0` so that the usable records land in train.

When the directory already exists, the command **neither overwrites nor
regenerates it**: it reuses only a finalized/readable `.tdata`, prints this
decision and continues with the usual full integrity/tokenizer preflight. In
that case `--teacher`, `--prompts` and the teacher-generation flags are
deliberately ignored. If you pass them on reuse anyway, the CLI requires
`--teacher` and `--prompts` together. An incomplete or damaged directory fails
closed; choose a new path instead of trying to continue it silently.

For generation, `--teacher-limit`, `--teacher-keep-truncated` and
`--teacher-disable-mtp` are available, as well as `--teacher-split-seed`.
`--max-records` and `--max-record-tokens` belong to compaction, not to answer
generation.

## Manual mode: ready-made teacher data

One command for an embedded MTP **and** a MiMo sidecar with a set prepared in
advance:

```sh
BIN=target/release/cortiq

"$BIN" mtp-compact base/model.cmf \
  --data calibration.tdata:train \
  --keep-ratio 0.90 \
  --max-record-tokens 4096 \
  --output candidate/model.cmf \
  --plan-out candidate/model.mtp-plan.json
```

For an embedded MTP it writes `candidate/model.cmf`; if the source was sharded,
this is one valid unsharded CMF. For MiMo it writes `candidate/model.cmf` and
`candidate/model.mtp.cmf` without a manual `ln`/`cp`; with a sharded main it
coalesces the main in a stream. The path `--output candidate/model.mtp.cmf`
stays a compatible legacy variant only for an external MiMo sidecar: it
publishes one compact draft and the plan but creates no main CMF. That is not a
ready standalone model; for new work use `--output candidate/model.cmf` so that
the command emits a verifiable runnable pair.

Before publishing the final names, the command uses nonce-private staging
directories, reopens and verifies the candidate's hashes, builds a `Pipeline`,
confirms that the MTP draft itself is what got loaded, and runs a
teacher-forced draft probe at the new FFN width on the stored calibration
sequence. For this, admission keeps the longest suitable calibration prefix of
up to 256 tokens (or `draft_layers + 2` if the draft is deeper); its length is
recorded in the plan as `admission_probe_tokens`. It runs the teacher-forced
probe of the compact draft **and** a separate CPU-only plain greedy gate: the
source and the staged candidate get the same first ≤16 prefix tokens and must
produce the same next 2 tokens with the speculative branch and the GPU turned
off. After the bake, the source main/draft are verified again by directory
**and** merged `trunk_hash`; a changed source causes a refusal. For MiMo the
`trunk_hash` of the staged backbone is also checked against the one the plan
was built from. If the admission gate fails, the final candidate is not
published. This short CPU gate does not certify a Metal/Vulkan kernel or the
speculative verifier — those need the separate exact-ID test below.

Before the full source hash scan, the command checks the topology of the
explicit `--output`/`--plan-out` (including aliases, parent/child and the
`.mtp.cmf` sidecar name) and makes sure that these final leaves are free. So a
mistake like `--plan-out candidate.cmf/plan.json` does not waste hours on a
100B scan. The real race-safe no-clobber protection is still done by the late
atomic publish; new output parents/stages are not created until the tokenizer
and full `.tdata` preflights succeed. So a selector error or a foreign
tokenizer does not leave an empty nested directory next to the future
candidate. After the source is opened, but before the `.tdata`, the canonical
dense-MTP geometry, a materializable codec and `--keep-ratio` are checked
separately, so that these errors do not start calibration either.

Before publishing, atomically created nonce-private stage **directories** are
used; the internal writers open files with `create_new`, so a collision cannot
overwrite someone else's stage. The payload and the parent directory entries of
stage/final are synced (`sync_data`/`fsync`) before success is reported.
Publishing every final name is no-clobber too; for MiMo the sidecar is
published before the main, so a runnable main without a draft never becomes
visible. This is not a promise of one filesystem
transaction for the whole pair/plan on a filesystem failure: if the second
final link fails, the already published plan/sidecar is deliberately **not
deleted**, so that a race cannot delete someone else's file. Inspect and delete
such an incomplete artifact by hand before retrying; the command will refuse to
overwrite it. So keep the source and the plan. The gate proves that the CMF/MTP
is intact and runnable, but it does not replace hold-out quality and a
benchmark on the target backend.

`--max-records N` caps the number of **selected records read**, and only after
the mandatory full `.tdata` integrity preflight; it makes the plan depend on a
truncated sample and is not a way to skip checking a large dataset. In the
plan, `calibration_selected_records` records this actual capped pass, and
`calibration_records` only the records that yielded ≥2 tokens and contributed
statistics. For quick debugging, use a small separate finalized selector.
`--max-record-tokens N` (default 4096) is a production safety cap, not a
quality knob: `N` must be at least `draft_layers + 2`, and too small an `N`
changes the long-context activation distribution because of the fresh-causal
boundaries. For a release candidate,
use a full calibration selector fixed in advance and a context cap chosen and
documented in advance.

## Independent draft width

The new header stores `arch.mtp.ffn_intermediate_sizes` — an optional
`Vec<usize>` with the local FFN width of each MTP layer. After compaction it can
be smaller than the trunk's `arch.intermediate_size`. Old files lack the field;
the loader takes the width from the verified triple of tensors `gate_proj`,
`up_proj`, `down_proj`. If the field is present, it must match the actual
tensor shapes. MiMo likewise derives the width of each draft layer from its
tensors, not from the trunk geometry.

`eh_proj`, `enorm`, `hnorm`, attention and the layer/final norms work on the
hidden width, not on the FFN axis: this operation keeps them byte for byte;
only the matching `gate/up/down` axes (and `down_proj.t`, if present) and
`arch.mtp.ffn_intermediate_sizes` change. So the compactor does not try to
"shrink" a projection or norm along the wrong axis.

Consequence: do not edit the stored directory/trunk hashes or widths in the
JSON, and do not apply a plan from another model. Missing or mismatched
hashes, as well as a mismatch of the header, the layer count or the shapes, are
a reason to refuse, not a reason to "try running it". Several draft layers are allowed only for
`mimo_v2`, because only it has a multi-layer executor in the runtime.

## Mandatory correctness check

First qualify **ordinary CPU greedy**, then, separately, the same on the Metal
backend. In all runs pin one binary, template, `--raw`/prompt, `--greedy`,
token count and set of environment variables. Compare **token ids**, not the
rendered text. `--trace-json` writes JSONL to stderr.

The CPU is the mandatory baseline check of compaction: it does not enable the
MTP pair, so it cannot "successfully" accept a faulty draft. Turn the GPU off
explicitly so that no stray backend is inherited from the shell:

```bash
set -euo pipefail

BIN="${BIN:-target/release/cortiq}"
PROMPT="${PROMPT:-Explain in one paragraph why deterministic decoding is useful.}"
mkdir -p verify

extract_trace_ids() {
  local trace_stderr="$1" trace_ids="$2"
  if ! grep '^{"t":' "$trace_stderr" | \
    sed -E 's/.*"token_id":([0-9]+).*/\1/' >"$trace_ids"; then
    echo "cannot extract token IDs from $trace_stderr" >&2
    return 1
  fi
  if ! test -s "$trace_ids"; then
    echo "trace contains no token IDs: $trace_stderr" >&2
    return 1
  fi
}

trace_cpu_plain() {
  model="$1"; out="$2"
  CMF_GPU=0 CMF_MTP=0 CMF_MIMO_MTP=0 "$BIN" run "$model" \
    --raw --prompt "$PROMPT" --greedy -n 128 --trace --trace-json \
    >/dev/null 2>"$out.stderr"
  extract_trace_ids "$out.stderr" "$out.ids"
}

trace_cpu_plain base/model.cmf      verify/base-cpu-plain
trace_cpu_plain candidate/model.cmf verify/candidate-cpu-plain
cmp verify/base-cpu-plain.ids verify/candidate-cpu-plain.ids
```

For an embedded MTP check one `.cmf`; for MiMo also run `"$BIN" verify` on the
main and the `.mtp.cmf` of both models. The admission command already does a
structural check, but this trace is mandatory on several hold-out prompts and
at a sufficient length.

### The host fused pair is not a release default

`CMF_MTP_PAIR=1` enables the experimental CPU fused pair. It optimizes the
arithmetic of two positions but is not a verifier. On q4tp Qwen3.5 a near-tie
has already been found where the fused pair gave a different greedy ID than two
serial target forwards. So the standard CPU `CMF_MTP=1` does **not** enable this
path; do not take a run with zero `mtp_drafted` as evidence of speculative
speed.

If a particular model and CPU backend are to use the host pair, first run a
separate long exact-ID gate with `CMF_GPU=0 CMF_MTP=1 CMF_MTP_PAIR=1` for the
source and the candidate against their own plain traces. At the first mismatch, leave the variable off. This experiment does not
replace the Metal qualification below.

For GDN hybrids the host pair additionally requires an explicit `CMF_MTP=1`,
because their recurrent part does not parallelize and is excluded from this
experiment by default; this is not a recommended speed preset.

The trace is also part of the verification contract: it must hold exactly one
JSON line per emitted token, including the extra draft tokens accepted by one
batched graph-speculative round. For such accepted rows the runtime does not
run a separate full softmax just for telemetry, so `confidence` is honestly
`0.0`; this does not mean zero probability and must not be used as a quality
metric.

## Acceptance and speed on Metal/Vulkan

First make sure the backend is visible:

```sh
"$BIN" gpu
```

### Native Metal: the safe default

For the standard canonical MTP, the native-Metal **batched graph verifier is
off by default**. This is a deliberate fail-closed default: on a real model one
near-tie row of the batched verifier produced a different *emitted* greedy ID
than the ordinary serial Metal path. So `CMF_GPU=1 CMF_MTP=1` with
`CMF_GRAPH_SPEC` and `CMF_MTP_PAIR` unset keeps the ordinary serial target
decode; this is not a draft-loading error and not a reason to convert the model
again. In such a run `mtp_drafted=0` for canonical MTP is the expected safe
result.

`CMF_GRAPH_SPEC=1` is kept only as an explicit experiment for kernel
diagnostics. It is **not a production speed preset**: a finite set of prompts
does not prove that there is no next near-tie row. Do not add this variable to
a model's instructions until the batch and serial kernels are fixed and widely
stress-tested again. The separate `CMF_MTP_PAIR=1` is likewise the
experimental host fused pair from the previous section; it neither enables nor
fixes the Metal graph verifier. MiMo speculation is controlled separately by
`CMF_MIMO_MTP`, so this restriction applies specifically to the canonical MTP
graph verifier, not to all MTP drafts.

Run a safe Metal smoke test/benchmark with the experimental flags explicitly
removed:

```sh
env -u CMF_GRAPH_SPEC -u CMF_MTP_PAIR \
  CMF_GPU=1 CMF_MTP=1 CMF_MIMO_MTP=0 \
  "$BIN" bench candidate/model.cmf --tokens 256 --ignore-eos --json \
  > verify/candidate-metal-serial.json
```

This is a correct measurement of ordinary Metal generation of the compact
candidate, but not proof of an MTP speedup. Keep `decode_tok_s_steady`, the
backend, the device and the binary version from the JSON; do not call this
result a speculative benchmark.

### Native Metal: a diagnostic exact-ID gate only

Below is a gate **only for investigating** `CMF_GRAPH_SPEC=1`. CPU/Metal
arithmetic differences are not an error in themselves, but within one Metal
backend the plain and the experimental graph IDs must match. Use several
hold-out prompts and a sufficient length. `CMF_METAL_VERIFY_CHECK=1` is a
native-Metal diagnostic: it prints the batched-vs-serial comparison, but it does
not change a decision the verifier has already made and is not a fallback.

```bash
set -euo pipefail

BIN="${BIN:-target/release/cortiq}"
PROMPT="${PROMPT:-Explain in one paragraph why deterministic decoding is useful.}"
mkdir -p verify

extract_trace_ids() {
  local trace_stderr="$1" trace_ids="$2"
  if ! grep '^{"t":' "$trace_stderr" | \
    sed -E 's/.*"token_id":([0-9]+).*/\1/' >"$trace_ids"; then
    echo "cannot extract token IDs from $trace_stderr" >&2
    return 1
  fi
  if ! test -s "$trace_ids"; then
    echo "trace contains no token IDs: $trace_stderr" >&2
    return 1
  fi
}

trace_metal_graph_experiment() {
  mode="$1"; model="$2"; out="$3"
  if [ "$mode" = plain ]; then
    CMF_GPU=1 CMF_MTP=0 CMF_MIMO_MTP=0 CMF_METAL_VERIFY_CHECK=1 \
      "$BIN" run "$model" --raw --prompt "$PROMPT" --greedy -n 128 \
      --trace --trace-json >/dev/null 2>"$out.stderr"
  else
    CMF_GPU=1 CMF_MTP=1 CMF_MIMO_MTP=0 CMF_GRAPH_SPEC=1 \
      CMF_METAL_VERIFY_CHECK=1 \
      "$BIN" run "$model" --raw --prompt "$PROMPT" --greedy -n 128 \
      --trace --trace-json >/dev/null 2>"$out.stderr"
  fi
  extract_trace_ids "$out.stderr" "$out.ids"
}

trace_metal_graph_experiment plain base/model.cmf      verify/base-metal-plain
trace_metal_graph_experiment graph base/model.cmf      verify/base-metal-graph
trace_metal_graph_experiment plain candidate/model.cmf verify/candidate-metal-plain
trace_metal_graph_experiment graph candidate/model.cmf verify/candidate-metal-graph

cmp verify/base-metal-plain.ids verify/candidate-metal-plain.ids
cmp verify/base-metal-plain.ids verify/base-metal-graph.ids
cmp verify/candidate-metal-plain.ids verify/candidate-metal-graph.ids

# Any MISMATCH means the batch verifier is no longer exact for
# this run. It does not fall back automatically.
if grep -H ' MISMATCH ' verify/*-metal-graph.stderr; then
  echo 'Metal graph verifier mismatch: leave CMF_GRAPH_SPEC unset.' >&2
  exit 1
fi
```

A successful short gate is useful for hunting kernel bugs, but it does not
cancel the safe default above. If `mtp_drafted` is zero, the graph was not
admitted at all, and such a trace does not check the speculative route.

### Vulkan/WGPU

For Vulkan the existing graph policy is kept, but a speedup still cannot be
claimed without an exact-ID gate on the **same** device. Select Vulkan
explicitly, pin `XDG_RUNTIME_DIR` on the pod and compare the plain/speculative
traces within one backend. This is an example for canonical MTP; check MiMo with
a separate run with `CMF_MIMO_MTP=1`, because its round is not controlled by
`CMF_GRAPH_SPEC`.

```bash
set -euo pipefail

BIN="${BIN:-target/release/cortiq}"
PROMPT="${PROMPT:-Explain in one paragraph why deterministic decoding is useful.}"
mkdir -p verify

extract_trace_ids() {
  local trace_stderr="$1" trace_ids="$2"
  if ! grep '^{"t":' "$trace_stderr" | \
    sed -E 's/.*"token_id":([0-9]+).*/\1/' >"$trace_ids"; then
    echo "cannot extract token IDs from $trace_stderr" >&2
    return 1
  fi
  if ! test -s "$trace_ids"; then
    echo "trace contains no token IDs: $trace_stderr" >&2
    return 1
  fi
}

trace_vulkan() {
  mode="$1"; model="$2"; out="$3"
  if [ "$mode" = plain ]; then
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" WGPU_BACKEND=vulkan \
      CMF_GPU=1 CMF_MTP=0 CMF_MIMO_MTP=0 \
      "$BIN" run "$model" --raw --prompt "$PROMPT" --greedy -n 128 \
      --trace --trace-json >/dev/null 2>"$out.stderr"
  else
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" WGPU_BACKEND=vulkan \
      CMF_GPU=1 CMF_MTP=1 CMF_MIMO_MTP=0 \
      "$BIN" run "$model" --raw --prompt "$PROMPT" --greedy -n 128 \
      --trace --trace-json >/dev/null 2>"$out.stderr"
  fi
  extract_trace_ids "$out.stderr" "$out.ids"
}

trace_vulkan plain base/model.cmf      verify/base-vulkan-plain
trace_vulkan spec  base/model.cmf      verify/base-vulkan-spec
trace_vulkan plain candidate/model.cmf verify/candidate-vulkan-plain
trace_vulkan spec  candidate/model.cmf verify/candidate-vulkan-spec
cmp verify/base-vulkan-plain.ids verify/base-vulkan-spec.ids
cmp verify/candidate-vulkan-plain.ids verify/candidate-vulkan-spec.ids
```

On a new Vulkan device, first run this gate with `CMF_MTP_GRAPH=0`, then repeat
it with the usual graph setting. A late `Declined` after admission cancels the
speculative round fail-closed rather than being bound to fall through
transparently to the host; watch `mtp_drafted` and the logs. Only if
`mtp_drafted > 0`, count acceptance and compare several interleaved
`decode_tok_s_steady` runs of base/candidate with the same binary, context,
device and power. A smaller draft may lower acceptance and make the candidate
slower — that is a valid result.

q4tp uses the local width of `gate_proj`, not the backbone's
`intermediate_size`; the 32-channel alignment keeps it eligible for the graph
path. q8_2f needs its own alignment/`graph_x` conditions. q2tp for
canonical MTP is deliberately not accepted by the native-Metal draft graph and
stays on the ordinary path. A MiMo sidecar with independent draft widths loads
correctly, but its small draft batches do not yet have a dedicated fused
native-Metal MTP graph. So `mtp-compact` reduces its draft storage and CPU
work, but does not promise a Metal speedup without separate exact-ID and
benchmark evidence.

## Limits and refusals

The command is deliberately fail-closed. It is designed for a complete
canonical dense MTP FFN triple with the shapes `gate=[I,H]`, `up=[I,H]`,
`down=[H,I]`, where `I` is a multiple of 32. Not supported without a separate
implementation/validation:

* routed/MoE or segmented MTP FFN, incomplete tensor triples and mismatched
  shapes;
* an MTP whose width does not allow a 32-channel group, an invalid keep ratio
  or an empty teacher-forced calibration;
* special MTP formats not represented by such a dense triple, and external
  DSpark/draft sidecars that are not an MTP FFN;
* a frozen genome when the MiMo main has to be physically coalesced from
  shards: a full rewrite would break the append-only layout, so such input is
  rejected;
* a plan without `selection.source_draft_directory_hash`, with a mismatching
  (in new plans) `source_draft_trunk_hash` or with a hash that belongs to a
  different draft; editing the plan does not cancel the source tensor/config
  checks.

Keep the source main, sidecar and plan until all checks are done. A compact
draft can be accepted only after file integrity, the exact greedy comparison
and a benchmark on the target backend under the matching exact-ID policy above.
