# Embryo bounded anchor `swa_sink_v1` — operator contract (trainer ⇔ runtime)

Status: implementation contract, 2026-09-23. Format side is done
(`cortiq-core`: `LayerType::BoundedAttention`, `ModelArch.anchor_core`,
`features::BOUNDED_STATE`, validation in open/write — see
`docs/CMF_V2_SPEC.ru.md` §1.1 bit 12 — bit 7 before 0.8.1 — and §2.3). This file fixes the math and
the byte layout both sides must implement identically (trainer, runtime and probe).

## 1. Operator

Per anchor layer, GQA geometry as today (Embryo-0: `qh=8, kvh=2, hd=128`,
`group = qh/kvh`). Let `q̂_t, k̂_j, v_j` be the projected, UNROTATED vectors
(`W_q x`, `W_k x`, `W_v x`), `S` = number of trained sink vectors per KV
head, `W` = exact-window width (serving window), `R(δ)` = the RoPE rotation
of the layer (`rope_base`, `hd`, same `inv_freq` as today) by `δ` positions.

```
window keys:  j ∈ (t − W, t]           (W keys INCLUDING the current token)
window score: s_j = ( R(t − j) q̂_t ) · k̂_j / √hd
sink score:   s_s = q̂_t · k̂ˢ_s / √hd,   s ∈ [0, S)      (NoPE: nothing rotated)
one softmax over {s_s} ∪ {s_j};   out_t = Σ_s p_s v̂ˢ_s + Σ_j p_j v_j
```

Identity that makes train and serve the same function: with absolute RoPE
`q_rot(t) = R(t) q̂`, `k_rot(j) = R(j) k̂`, one has
`q_rot(t)·k_rot(j) = q̂ᵀ R(t)ᵀ R(j) k̂ = (R(t−j) q̂)·k̂`. So the trainer may keep
rotating q and k by their absolute positions (exactly what
`crates/cortiq-embryo/src/model.rs` does now around the `cmd.rope(q, …)` /
`cmd.rope(k, …)` calls) and only MASK the scores to the band, while the
runtime stores raw `k̂` in a ring and rotates `q̂` by `Δ = t − j ∈ [0, W)`.
No absolute position appears anywhere in the served operator.

Sink vectors are WEIGHTS (`self_attn.sink_k.weight`, `self_attn.sink_v.weight`
`[kvh, S, hd]`, f32, never quantized), not positions 0..S-1 of the sequence.

Stochastic training window (SWAX): each training step samples
`W_t ∈ anchor_core.train_windows` (e.g. {64, 128}); the served window is
`anchor_core.window` (128). The last ~10% of a birth uses the served window.

Defaults for Embryo-O1 variant A: `W = 128`, `S = 4`, `train_windows = [64, 128]`.
Constraint: `S + W ≤ 160` (GPU scratch), enforced by the format validator.

## 2. Trainer (`crates/cortiq-embryo`) — what changes

- `EmbryoCfg` gets `anchor_window: usize` (0 = legacy full causal, serde
  default → old checkpoints bit-identical), `anchor_sink: usize` (default 0),
  `anchor_train_windows: Vec<usize>` (empty = fixed `anchor_window`),
  `anchor_layers: Option<Vec<usize>>` (None = `(l+1) % anchor_every == 0`).
- `Layout` allocates `sink_k`, `sink_v` `[kvh·S·hd]` per anchor layer (after
  the matrices; keep 4-float alignment). Init: `sink_k ~ N(0, 0.02)`,
  `sink_v = 0`.
- Scores matrix per (b, head): `p[T, S + T]` — sink columns `0..S` first,
  then the causal T×T block. Forward: sink block = `q̂ · sink_kᵀ · scale`
  (GEMM on UNROTATED q, before `cmd.rope(q)`, or on a saved `q_raw` copy);
  window block = today's `Q_rot K_rotᵀ · scale`; band+sink softmax:
  `valid(row, col) = col < S || (S ≤ col ≤ S + row && row − (col − S) < W_t)`;
  `O = P[:, S..] · V + P[:, 0..S] · sink_v` (second GEMM with beta = 1).
- Backward: `softmax_bwd_rows` over `n_cols = S + T` (P = 0 outside the band
  ⇒ dS = 0 there, no other change); `dq = dS_win · K_rot · scale` (then the
  existing inverse rope) **+** `dS_sink · sink_k · scale` (raw space, added
  AFTER the inverse rope); `dsink_k = dS_sinkᵀ · q̂ · scale`; `dsink_v = P_sinkᵀ · dO`.
- Kernels: Metal `causal_softmax_rows_f32` / `softmax_bwd_rows_f32`
  (`shaders.metal` ~2242-2296) and WGSL op 19 / op 20 (`vulkan.rs` ~2817-2821)
  currently index blocks as `block·n·n` with `len = row+1`. Both need the
  `[T, S+T]` row layout, a `(sink, window)` pair in params and the band
  predicate. `anchor_window = 0, anchor_sink = 0` must reproduce today's
  numbers bit-for-bit (regression gate).
- Export (`export.rs`): anchor layers → `LayerType::BoundedAttention`,
  `arch.anchor_core = {kind:"swa_sink_v1", window, sink, rope:"relative_in_window",
  sink_scores:"nope", train_windows}`, tensors `self_attn.sink_k.weight`,
  `self_attn.sink_v.weight` `[kvh, S, hd]` f32; legacy configs
  (`anchor_window == 0`) export exactly as before (FullAttention, no record).
- f64 reference for gradchecks: `crates/cortiq-engine/src/fcd_ops.rs`
  `attn_head_fwd/bwd` (with the band mask + sink block added in the test),
  or a small standalone f64 reference in the test crate. Tolerances: forward
  ≤ 1e-5 rel vs f64; gradients (dq, dk, dv, dsink_k, dsink_v, dW_o) ≤ 1e-5
  rel on `EmbryoCfg::tiny()` (repo precedent: `hk_gradcheck.rs` 1e-6..3e-6).

## 3. Runtime (`crates/cortiq-engine`, `crates/cortiq-cli`) — what changes

- Loader: `LayerType::BoundedAttention` + `arch.anchor_core` →
  `AttnKind::Bounded { wq, wk, wv, wo, sink_k, sink_v, cfg }`; refuse a file
  whose anchor tensors are missing; refuse `--o1` / `CMF_O1` / `CMF_O1_*`
  on such a file with an explicit error ("anchor is native bounded").
- `LayerKvCache` gets `bounded: Option<BoundedState>` created from the
  header in `KvCache::new` (not per prompt):
  `ring_k [kvh][W][hd]` (raw, unrotated), `ring_v [kvh][W][hd]`, `len`, `head`.
  `memory_bytes()` counts it; `clear()` zeroes it; `evict*` skip it;
  `truncate_last` rolls it back (keep a small snapshot API for speculation).
- Attend (per token, per KV group, all its Q heads): `n = S + min(len, W)`
  keys; sink scores on raw q̂; window scores with q̂ rotated by `Δ` via a
  `[W][hd/2]` cos/sin table built once from the layer's `inv_freq`;
  one softmax; write `k̂_t, v_t` into slot `(t mod W)` (insert BEFORE read so
  the current token is in its own window, matching the trainer's band that
  includes `col == S + row`).
- Batched prefill (`prefill_batch` → `qwen_attention_batch`) must route
  bounded layers through the same operator (chunk scores only against
  ring + chunk, never a growing KV). Per-token path: `qwen_attention_core`
  must not `cache.append` for bounded layers.
- Pipeline: no `o1_begin/o1_seal` for bounded layers; `reuse_from` allowed
  (drop the `o1_cfg.is_none()` requirement for files without o1);
  `kv_history` bounded (last W ids + rolling hash of the consumed prefix +
  length) so nothing in the pipeline grows with the dialogue; repetition
  penalty window bounded; `ppl` scores through the same operator (no
  `nll_ids_o1`).
- State wire v2 (`export_wire/import_wire`): versioned header
  `{magic, version=2, operator identity hash64, layer, kind, position}` +
  fixed-size record per kind (linear: S + conv ring; bounded: len/head +
  ring_k/ring_v; full: as today). Old unversioned wire stays readable for
  old kinds.
- Telemetry: `attention_state_bytes/recurrent_state_bytes` include the ring.

## 4. Gates (both sides)

- Trainer regression: `anchor_window=0` ⇒ loss bit-identical to master on 3
  fixed batches; Metal ↔ WGSL op 19/20 `max|Δ| ≤ 1e-6`.
- Trainer gradcheck ≤ 1e-5 rel vs f64 on `tiny()` with `S=2, W=3, T=8`.
- Runtime `bounded_runtime.rs` (real geometry, synthetic weights):
  state bytes identical after 64 / 4096 / 32768 generated tokens from a
  23-token prompt; ms/token(32768)/ms/token(64) ≤ 1.10; 20 turns × 512
  tokens: RSS(turn 20) ≤ 1.01·RSS(turn 2) and prefill_tokens per turn = new
  tokens only; `import(export(state))` → next 64 logits Δ = 0; `--o1` on a
  bounded file → error; old-style files unchanged (ppl of the legacy
  Embryo file reproduces).
- Trainer ⇔ runtime parity on an exported tiny genome: `max|Δ logp| ≤ 5e-5`
  over 512 + 32 tokens (precedent `tests/runtime_parity.rs`).
