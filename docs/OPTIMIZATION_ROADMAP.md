# Where the next speed is — a roadmap with numbers

Companion to `GPU_KERNEL_RECIPES.md`, which records what worked and what
did not. This one says what to do next, in the order the measurements
justify. Every item names its evidence; nothing here is a hunch.

State as of 0.5.43 (RTX PRO 6000 Blackwell / Vulkan unless noted):

| model | layout | decode | note |
|---|---|---|---|
| Qwen3.6-35B-A3B-Escha-W2 | q2tp | 105.0 | 2-bit experts |
| Qwen3.6-35B-A3B | q4tp | 99.6 | was 67.8 |
| KAT-Coder-V2.5 | q4tp MoE | 93.1 | was 69.7 |
| Nanbeige4.2-3B | q4t | 77.6 | was 37.5 |
| Bonsai-27B | q1 | 45.3 | recipe not applicable |
| Qwen3.6-27B dense | q4tp | 39.1 | was 32.3 |
| Bonsai-8B | q1t | 37.4 | was 28.8 |
| Nanbeige4.2-3B (M4, Metal) | q4t | 22.3 | see item 1 |

---

## 1. Metal: one command buffer per token (the biggest single item)

**Evidence:** `bench --json` reports `metal_submits_per_token`. An M4
decoding Nanbeige-3B does **58 round trips in a 44 ms frame**. Command
buffer completion is the only system-scope ordering Metal offers — a
"fast flag" shortcut was tried and reverted because it corrupted real
decodes — so those waits *are* the frame. The kernels are not the
problem: they already load `float4`, run four rows per simdgroup, expand
the scale ladder once and reduce with `simd_sum`.

**Do:** port the wgpu whole-token graph shape to Metal — encode the layer
stack into ONE command buffer, keep every intermediate on the device,
wait once per token. The MoE block already proved the shape on this
backend (four dispatches instead of ~1100 encoder/dispatch pairs).

**Expect:** ~2x, the same lever and the same reason it paid on wgpu.

**Watch for:** the CPU currently computes norms/softmax between GPU ops on
this path; each of those is a reason a wait exists. They have to move to
the device in the same change, exactly as the wgpu graph did.

## 2. The batched GEMM's shape (dense prefill, and the Lumina DiT)

**Evidence:** `q4tp_mul_mm` runs at roughly 300 GFLOP/s. The load-shape
recipe that doubled the matvecs did **nothing** here: a true A/B of two
builds measured 18.81/18.97 against 18.59/18.91 tok/s prefill — 0.7%,
noise. Its loads were never the wall; it already stages through
workgroup memory.

**Do:** attack the register block and K-step instead — the 4x4 block with
a 16-wide K step is small for a modern card. Try 8x8 per thread and a
32-wide K step, and check the staging arrays' bank conflicts.

**Serves:** dense prompt ingest for every q4tp model *and* Lumina-Image
2.0's DiT, which is the same kernel.

## 3. The layouts still on scalar kernels

`GPU_KERNEL_RECIPES.md` has the recipe and the parity rules. Unported:
`q4b_matvec`, `q8_matvec`, `moe_gate_up` (q4t) and `moe_gate_up_q4tp`,
and the `*_b` batch twins. The q4t and q1t ports returned 2.1x and 1.3x
on real models, so this is known-value work, not speculation.

**Do not bother with binary `q1`:** measured 152.8 vs 152.3 tok/s — it
reads 4 bytes per 32 weights and has nothing to vectorize.

## 4. A batched GEMM twin for `q2tp`

The 2-bit layout has no batched kernel, so prompt ingest for
Escha-W2-class files runs per position on the CPU. Decode is 105 tok/s;
long prompts are slow. The q4tp twin is the template.

## 5. Kimi-Linear-48B (KDA) never enters the graph

**Evidence:** `RUST_LOG=info` prints `wgpu whole-token graph refused —
per-op path` for it, so none of the 0.5.43 kernel work reaches this
model. Parity CPU↔GPU is fine; it is simply on the slow road.

**Do:** find the refusal reason (the graph logs one), then decide whether
KDA attention gets graph kernels or an explicit, documented exclusion.

## 6. Long-context decode

`--o1` (Nyström) holds a flat 53.8 tok/s at ctx 16 384 on the 35B where
exact attention falls to 37.8, but its calibration still requires an
O(n²) CPU prefill (~30 min at 16K). That prefill is the next long-context
target, not the decode.

---

## 7. DeepSeek-V4-Flash: ported, unproven

The converter reads this model end to end — FP8 E4M3 with 128×128 block
scales (our decoder matched `torch.float8_e4m3fn` bit for bit on its own
weights), MXFP4 experts, the integer routing table, and the full name map.
The engine now carries all five of its blocks (`crates/cortiq-engine/src/dsv4.rs`):
double-LoRA attention with a 512-wide compressed KV, the per-layer KV
compressor including its overlapping variant, the sparse indexer,
hyper-connections with Sinkhorn normalization, and hash routing.

**What is established:** seventeen tests, of which one decodes ten tokens
through a two-layer toy model and one pins `hc_split_sinkhorn` to the
reference's own numbers. **What is not:** numerical parity against a
reference run. The first coherent generation is the gate.

Reading the release's `config.json` and `inference/model.py` against the
port found six real bugs that no unit test would have caught, and they are
worth listing because the same classes recur:

| what | why it was invisible |
|---|---|
| `swiglu_limit` (10.0) not applied | only fires on saturating activations |
| RoPE built over `head_dim` 512, not the 64-wide tail | every angle wrong, output still finite |
| YaRN dropped at conversion (`rope_scaling.type` vs `rope_type`) | a missing field reads as "no scaling" |
| the sliding window never slid | correct under 128 tokens |
| hash layers weighted the wrong experts | both lists the right length |
| `overlap` stored but never read | entry twice the expected width, filed elsewhere |
| the 2-bit profile reached 1 expert per layer, not 256 | a merely large file |
| only the FIRST Split of a pre-tokenizer Sequence was applied | ids the model never saw; the port looks guilty |
| the generic weight loader demanded `q_proj` | fails before the arch's own loader runs |
| batched prefill and pair-decode walked an empty layer stack | panic on index 0 of nothing |

**The lesson worth carrying:** for an architecture with no sibling in the
tree, read the reference's config and forward pass line by line, write a test
per operator AND one that decodes — then build a toy checkpoint with the
release's real names and run the whole path on it (`tools/mk_dsv4_toy.py`).
The reading found the config-level bugs, the decode test found the shape
bugs, and only the toy found the three that live outside the architecture
file entirely, in a pipeline that assumes every model keeps its layers where
the generic loader put them.

**And check the tokenizer against the reference before blaming the model.**
The single most expensive bug of the night was `find_split_pattern` taking
one rule out of a Sequence of three: the prompt never met a word boundary,
so the ids were ones the checkpoint had never seen. Nothing about that looks
like a tokenizer problem from the outside — it looks like a broken port.
Ten lines of `tokenizers` in Python settle it.

**On size:** the model is already 4-bit, so q4tp does not shrink it. The
q2tp expert profile (2-bit gate/up, 4-bit down and skeleton) takes it to
112 GB — past what a Hub repo accepts in one file, so it ships as 8 GB
slices that `cat` back into the file. CMF reads sharded models natively
(`open_sharded`, spec §10) but nothing WRITES them yet; a `cortiq shard`
command is the tidier answer and wants a box with room for two copies. Expert defrag does NOT help on the hash layers: their table reaches
all 256 experts within the first 8 000 vocabulary ids, measured.

**Deliberately absent:** the indexer's Hadamard rotation and FP4 simulation.
The rotation is orthogonal and the reference applies it to both sides of the
same dot product, so it cancels — it exists to condition the FP4 quantization,
which we also skip by keeping f32. Omitting the pair is exact, not an
approximation. (An earlier revision of this file called it approximate; that
was wrong.)

**Still open:** there is no GPU graph for any of this, so decode is CPU-only —
which for a ~10B-active MoE is the whole performance story, not a detail.

## 8. The scale is the 2-bit quantizer

`q2tp`'s four levels are `(c−1.5)·s`, so the group scale IS the quantizer,
and the rule was `absmax/1.5` — which pins the outer level to the largest
weight in the group. That minimises the worst error; with four levels what
matters is the mean one, and a single outlier coarsens the other 31 values
behind it.

Trying the neighbouring rungs of the row's ladder and keeping whichever
reconstructs the group best costs 1% of conversion time and nothing else —
same bytes, same layout, same decoder and GPU kernels. Measured on weights
shaped like a real expert plane (Gaussian, periodic outliers):

The ladder itself had the same flaw one level up: it was built to SPAN the
row (lo..hi over the group scales), so one loud group stretched it and
coarsened the step for everything quiet behind it. Trying shorter ladders
and keeping the best is the same argument again.

| layout | absmax rule | + rung search | + ladder search |
|---|---|---|---|
| q2tp (4 levels) | 0.631 | 0.476 | **0.380** |
| q4tp (16 levels) | 0.1179 | 0.1103 | — |

Sixteen levels forgive an outlier; four do not. Cost: 9x the encoder,
single-threaded — about 30 s across 48 cores for a 300B model, against a
conversion measured in hours. Probing the WHOLE ladder per candidate cost
42x and was less accurate (0.402): the tighter ±2 probe changes which
ladder wins.

Worth re-converting any 2-bit file that predates this. The DeepSeek-V4 q2tp
variant answers correctly and then loses the thread, which is what expert
noise looks like, and it carries 40% more of it than it needs to.

## 9. Converters write once now

`CmfStreamWriter` (`cortiq-core/src/format.rs`) reserves a gap at the head
of the output, appends each payload as it is encoded, and patches the
envelope, header and directory into that gap at the end. Before this, a
conversion held every payload — first in RAM (+1.8 GB/min), then in a spill
file it copied into the output, so peak disk was twice the model. On the
300B DeepSeek that was 240 GB on a 236 GB box: it would have died at ~60%,
hours in. Any new converter path should use it rather than collecting a
`Vec<TensorSpec>`.

An append-only manifest rides alongside, with a checkpoint per source shard.
`cortiq convert --resume` rolls the output back to the last checkpoint and
skips the shards behind it, download included. Two orderings make that safe
rather than plausible: the file is flushed BEFORE a checkpoint claims its
bytes are durable (a buffered writer otherwise puts the manifest ahead of
the disk — measured, six bytes), and the rollback goes to the checkpoint
rather than the last tensor, since tensors after it belong to a shard that
will be redone. Verified by killing an eight-shard conversion after three
and resuming: byte-identical to a straight-through run.

For a conversion long enough that the machine may not outlive it, ship the
output in slices as they complete (the first Colab died at 75% and took
80 GB with it). `tools/manifest_add_marks.py` reconstructs checkpoints for a
manifest written before they existed.

## How to work on any of this

1. **Profile first.** `CMF_GPU_TS=1` (pass boundaries) and `CMF_GPU_TS=2`
   (every dispatch inside the first layer of each kind) on wgpu;
   `metal_submits_per_token` on Metal. Four cost models were tried by
   hand during 0.5.43 — dispatch count, pass count, occupancy, dependency
   depth — and each was wrong at least once. The profiler was not.
2. **A/B honestly.** An env flag that does not actually disable the change
   produces a null result that looks like a null finding. Verify the arm
   really is off (the GEMM A/B above needed two builds, not a flag).
3. **Equal lengths only.** Attention grows with context; a 400-token run
   is not comparable to a 240-token one.
4. **One benchmark at a time.** `pgrep cortiq` before trusting a number.
   Stale runs poisoned measurements four times in one day.
5. **Gate on real models.** Greedy text from a real prompt, plus an
   arithmetic prompt, against the previous kernel — not just unit tests.
   Twice during 0.5.43 a kernel change killed the shader, the whole model
   silently fell back to the CPU, and the only symptom was "slower".
6. **Record the failures with their numbers.** The negative-results table
   in `GPU_KERNEL_RECIPES.md` is what keeps the next attempt from
   re-walking the same seven dead ends.
