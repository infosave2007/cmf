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
