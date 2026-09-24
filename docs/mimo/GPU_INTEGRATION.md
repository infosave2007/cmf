# MiMo GPU + dynamic MoE + MTP integration

Integration branch: `codex/mimo-gpu-full`.

Inputs: `mimo-gpu-attn` 00e37a6a, `mimo-dyn-moe` 26933e34,
`mimo-mtp` 39beb4b8. The source branches and the user's main worktree
remain untouched. This is not a release qualification report.

## Changes under verification

- Choose MiMo placement before prompt graphs allocate weights. The token and
  batch builders stop at the bank-owned layer boundary, including burst and
  full-stack-only callers. Placement now recognizes wgpu's MiMo attention.
- Keep backbone hidden history for MTP on both GPU prefill paths.
- Verify through the batch graph prefix plus the dynamic MoE tail. Short q8
  verification panels use the wgpu scalar batched kernel rather than falling
  through the wide-prefill CPU threshold.
- A bank frame handles 1–4 token routes in two dispatches and one readback.
  Resolve/pin the union of experts before admission. Cold CPU experts run
  concurrently and share weight reads across rows where the exact kernels
  support it. Fallback reuses the already computed route.
- Roll back host and device KV to an absolute accepted position. Do not
  truncate stale host mirrors by a relative draft count. SWA rollback uses
  the backend's retained-window check; a refusal terminates and clears the
  sequence instead of silently proceeding with corrupt KV.
- The in-process MTP benchmark emits token IDs and the first mismatch and
  exits nonzero on token-parity failure.

## Validation protocol

On Vulkan, run GPU and timing work under the pod's GPU lock and full-model
CPU work under its CPU lock. Do not overlap with another benchmark.

```sh
cargo build --release -p cortiq-cli --features gpu --bin cortiq --example mimo_mtp_bench -j 8
CMF_GPU=0 cargo test --release -p cortiq-engine --features gpu --lib mimo -- --test-threads=1
CMF_SDOT=0 CMF_COOP=0 CMF_GPU_PROBE=0 \
  cargo test --release -p cortiq-engine --features gpu --lib bank_tests -- --test-threads=1
```

Then compare plain/spec in one process on the converted toy with the MTP
sidecar, explicitly exercising prefix/dynamic/hybrid placement. A CPU-only
unit test is not evidence that the device path was admitted. The bank frame
unit test checks each 1–4-row batch against separate GPU frames bit for bit,
including cold slots and repeated experts.

Real-model gates remain separate: natural EN, RU and code token identity;
128-token `bench --core --ignore-eos`; PPL; VRAM ladder (three repetitions
per point); then multimodal and release qualification. Acceptance rate is
not token-parity evidence and is not a decode speed measurement.

## First integrated run (2026-09-24)

- macOS: `cargo check -p cortiq-engine --tests -j 4` and CLI example check pass.
- Vulkan release build and engine test build pass.
- CPU MiMo filter: 18 passed; bank GPU filter: 5 passed.
- Toy plain/spec: 64 token IDs equal in each placement override. Random toy
  drafts all reject; this does not cover GPU partial/full acceptance.
- Real EN, 64 tokens, one loaded model: plain 10.75 tok/s, MTP 17.56 tok/s,
  identical token IDs. 23 rounds, 39/68 drafts accepted (2.696 tokens/round),
  30.2 ms draft and 123.9 ms verify per round. This is not the final
  128-token benchmark, and it does not satisfy the 40 tok/s target.
- Note: the inherited toy prefix/hybrid graph builder declines on its small
  f16 router; those placement labels alone do **not** prove full-graph use.

## Precision and cost follow-up

The first default-mode EN speed result is **not a release gate**: the next
code prompt exposed a plain/spec token mismatch. Two independently variable
precision choices had to be removed from the banked target:

1. Cold experts used CPU A8W8 activation rounding while hot experts used
   device f32. Cache fills could therefore change the function being decoded.
   MiMo cold dispatches now select float activation kernels in a nested,
   thread-local scope. Other pipelines retain their activation policy.
2. Plain O/head q8 matvecs defaulted to a 50/50 CPU/GPU row split; the short
   verification GEMM ran all rows on the GPU. The CPU half rounded activations.
   Banked MiMo now uses full-device q8 O/head projections in both arms. This
   scoped contract supersedes `CMF_GPU_SPLIT` for those calls but does not
   enable a disabled GPU or bypass device refusal.

The diagnostic with `CMF_GPU_SPLIT=1` plus exact cold experts passed both
code and EN (64 tokens); globally disabling A8W8 also passed code and RU, but
was slower. The final integrated default-scope gate is recorded below.

To recover the precision cost, float q4tp MoE now shares pool dispatches and
weight/scale rows across tokens, rather than reverting to serial expert
calls. AVX2 vectorizes unpack/conversion/products **without changing the
scalar pair/group sum order**. A standalone test on the pod checked 450
rows bit for bit and measured 11.10 ms scalar versus 5.53 ms AVX2 (median of
three small kernel trials); this is a kernel microbenchmark, not model tok/s.

Short MTP draft q8 panels also stay GPU eligible. The natural-prompt harness
accepts repeated `--extra-prompt PATH` arguments, so EN/RU/code share one
model upload. `--modes plain,plain,spec` additionally checks cold/warm plain
stability and exposes cache-warmup timing instead of claiming it as MTP gain.


## Final checkpoint (2026-09-24, v3)

The hybrid bank had an independent eviction bug: it advanced its epoch at
model layer 1, which is inside the graph prefix and never visits the bank.
Consequently a filled hybrid bank could not evict old slots. Both single-row
and verification paths now advance at `dyn_from`; the admission floor uses
only the layers served by the bank. A GPU regression test checks both paths.

Final default placement: hybrid, 14 MoE prefix layers, dynamic tail from
layer 15, 3096 bank slots. One loaded model; chat prompts; 128 tokens each;
`--modes plain,plain,spec`; `CMF_GPU_PROBE=0`; no SDOT/COOP/SPLIT overrides.
Rates exclude prefill; the second plain arm exposes the warm-cache baseline.

| Prompt | First plain tok/s | Warm plain tok/s | MTP tok/s | IDs equal |
|---|---:|---:|---:|---|
| Code | 16.31 | 25.99 | 24.55 | yes, all 128 |
| RU | 23.89 | 29.75 | 29.26 | yes, all 128 |
| EN | 23.70 | 27.93 | 27.84 | yes, all 128 |

These runs cross the 128-token SWA boundary and exercise partial and full
MTP acceptance. An EN forced-dynamic comparison also preserved all 128 IDs
within its own plain/spec arms: 27.88 warm plain, 23.41 MTP tok/s. This is
not an assertion that different placement modes produce identical tokens.

Checks: macOS engine/tests and CLI-example `cargo check -j4`; Vulkan release
example and lib-test builds; MiMo test filter 19 pass (device tests skip in
this CPU-only invocation), dedicated GPU bank filter 6 pass; float/A8W8
multi-token MoE tests 2 pass; float-scope, q8-scope and AVX2 bitwise tests
1 pass each; `git diff --check`. The bank tests include a real GPU hybrid
boundary regression, not just a placement-label check.

Raw evidence on the pod: `/root/mimo/out/codex-mimo-gpu-full/v3/`.
Local copy: `/Users/oleg/dev/mimo-handoff/results-codex-gpu-full/`.
Earlier v1/v2 logs are diagnostic, not the final release measurements.

### Remaining gates and next optimization

- **40 tok/s is not reached.** MTP has no consistent gain over warm plain
  here; do not market cold-plain versus warm-MTP differences as speedup.
- Profile/fuse the dynamic tail's short QKV/O dispatches and attention;
  current verification is still about 90–124 ms/round versus ~8–9 ms draft
  after warm-up. Attention-only batch graphs need per-layer GPU KV ownership
  and absolute rollback; do not introduce stale-host fallback after mutation.
- Optimize device expert weight reuse across the verification token axis;
  the current frame batches submissions but does not explicitly broadcast
  one expert's dequantized weights across all routed token rows.
- Re-run integrated CPU-reference/PPL gates before publishing. The earlier
  GPU-attention branch's CPU comparison is not proof for this integration.
- The VRAM ladder/three-run medians, multimodal merge and release/HF/card
  work remain pending. The auto-placement cost model has not been qualified
  as fastest across all budgets. No release or public upload was made.
