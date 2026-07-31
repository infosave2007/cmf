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

## 7. DeepSeek-V4-Flash: the converter is done, the runtime is not

The converter reads this model end to end — FP8 E4M3 with 128×128 block
scales (our decoder matched `torch.float8_e4m3fn` bit for bit on its own
weights), MXFP4 experts, the integer routing table, and the full name map.
Conversion itself forced a fix worth having anyway: the converter used to
hold every encoded tensor in RAM (+1.8 GB/min, a 176 GB box exhausted a
third of the way into a 300B model), and now spills each shard's payloads
to disk while the writer streams them back through an mmap.

**Five blocks still have no runtime**, and the file cannot decode without
them. In dependency order, with the reference at
`inference/model.py` in the upstream repo:

1. **Hyper-connections first** (`Block`, `kernel.py::hc_split_sinkhorn`) —
   they change the SHAPE of the hidden state: it is `hc_mult=4` copies, not
   a vector. Each block folds 4→1 through learned per-copy weights and
   expands 1→4 through a mixing matrix normalized by 20 Sinkhorn iterations
   (row-softmax, then alternating row/column normalization). There is no
   ordinary residual. This touches the KV cache, prefill and the GPU graph,
   so it cannot be bolted on later.
2. **Attention + compressor** on the layers without an indexer — double
   LoRA q (`wq_a`→norm→`wq_b`, then a SECOND normalization on the heads),
   RoPE on the last 64 dims only, 512-wide compressed KV, grouped low-rank
   output (8×1024), a learned per-head sink, and gated pooling of four
   consecutive tokens through a softmax over overlapping windows.
3. **The sparse indexer** — its own compressor with a Hadamard rotation and
   FP4 simulation, scoring which `index_topk=512` compressed positions each
   query reads.
4. **The gate's three forks** — `sqrt(softplus(x))` scoring, a selection
   bias that shifts top-k but NOT the weights (they come from the
   pre-bias scores), and hash layers whose experts come from a token-id
   table instead of the router.
5. **The GPU graph** for all of the above.

**On size**: the model is already 4-bit, so q4tp does not shrink it (167 →
~175 GB). Only lower bit-depth helps — hence the q2tp expert profile. And
expert defrag does NOT help on the hash layers: their table reaches all 256
experts within the first 8 000 vocabulary ids, measured. Routing-statistics
defrag on the ordinary layers needs a working runtime first.

---

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
