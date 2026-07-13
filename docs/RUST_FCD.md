# RUST_FCD — native FCD polish trainer for O(1)-converted models

Status: implemented (`cortiq fcd`). This document records the design,
the exact math the hand-rolled backward differentiates, the deliberate
deviations from the torch reference probes, and what a future
GatedDeltaNet (GDN) BPTT extension needs.

## 1. Why

Zero-shot O(1) conversion (`cortiq convert --o1`, the Nyström joint
kernel of `nystrom.rs`) keeps teacher-forced PPL close to the exact
model but LOOPS in autoregressive generation (measured on three
architectures). The certified cure (torch reference
`nystrom_fcd_full2_06b.py`, Qwen3-0.6B, full 28/28 conversion):

- train ONLY the LN gains + FFN (gate/up/down) of the CONVERTED layers;
- loss = 0.3·CE + 0.7·KL(teacher‖student) on log-softmax logits,
  mean per position;
- AdamW lr 5e-5 (torch defaults: betas 0.9/0.999, eps 1e-8,
  decoupled weight_decay 0.01), global grad-norm clip 1.0;
- batch 2×512 fresh random windows per step; quick val every 25 steps
  on FIXED deterministic evenly-spaced windows; RESTORE THE BEST
  checkpoint at the end (best was step 150 of 300);
- attention stays FROZEN closed-form; the Nyström mixing matrix M is
  CONSTANT in backward (gradients do not flow through the pinv).

Torch result to match in direction: ppl 25.29 → 20.16 AND generation
loops → 3/3 clean. `cortiq fcd` is the native Rust replacement — the
last Python dependency of the O(1) pipeline.

## 2. Architecture

Three pieces, all in `cortiq-engine`:

- `fcd_ops.rs` — the op set with hand-rolled backwards, generic over a
  minimal `Fp` float trait (f32/f64). Every op has a central-finite-
  difference gradcheck in f64 (`tests/fcd_gradcheck.rs`), rel-err
  < 1e-3. No autograd, no tape: the graph is FIXED, so each block's
  backward is written out by hand (llm.c style).
- `fcd.rs` — `FcdModel` (whole model dequantized to f32 once at
  startup), layer-checkpointed forward/backward, AdamW, the training
  loop, eval, best-checkpoint restore, and the `.cmf` write-back.
- CLI `cortiq fcd <model.cmf> --o1 all --corpus text [...]` →
  `<model>.fcd.cmf`.

### 2.1 Memory/precision plan

- All weights are dequantized to f32 once at startup (≤ 1B targets:
  Qwen3-0.6B ≈ 2.4 GB). The teacher and the student SHARE this frozen
  set; trainable tensors additionally get f32 master copies + grad +
  AdamW m/v. Teacher forward always reads the frozen originals, so the
  KL anchor never drifts.
- Activations use layer-level checkpointing, mirroring the torch
  reference's `torch.utils.checkpoint`: the forward stores only each
  layer's INPUT hidden `[B·T, H]`; the backward re-runs one layer's
  forward to rebuild its intermediates, then differentiates. Peak
  memory ≈ one layer's graph + 29 hidden snapshots.
- f32 everywhere except: attention weight matrices (logits/exp/skeleton)
  in f64 — the same precision the certified CPU probe used, which also
  removes any need for flash-style shift bookkeeping in training
  (`exp(±40)` is trivially safe in f64); RMSNorm sum-of-squares in f64
  (runtime discipline); loss reductions in f64.
- lm_head (tied embeddings) is processed in position chunks so the
  full `[B·T, vocab]` logits of teacher and student are never
  materialized together.

### 2.2 The student attention form (what backward differentiates)

Training is teacher-forced over a whole window, so the student uses the
MATRIX form of the joint kernel (the certified probe form,
`nystrom_ppl3_06b.py::head_out`), per head, T×T in f64:

```
lg[t,j] = (q_t·k_j)·scale                 (causal)
T ≤ W+8      → plain causal softmax (no skeleton)
near[t,j]   = causal ∧ ((t−j < W) ∨ (j < sink))
far          = causal ∧ ¬near
c[t]         = max_j causal lg[t,j]
w_near       = exp(lg − c)  on near
Q̃,K̃          = contiguous segment means of q,k (m_eff = clamp(T/8, 4, m))
M            = ridge_pinv(exp(Q̃·K̃ᵀ·scale))          ← CONSTANT in backward
A            = exp(q·K̃ᵀ·scale) · M · exp(Q̃·kᵀ·scale)
w_far        = max(A, 0) · exp(−c)  on far
out[t]       = Σ_j w[t,j]·v_j / Σ_j w[t,j]
```

Backward facts used:

- The per-row shift `c` multiplies numerator and denominator
  identically, so treating it as a constant is EXACT for ∂out (not an
  approximation).
- `M` is constant (validated in the torch runs — `pinv` under
  `no_grad`). The landmarks Q̃/K̃ themselves ARE differentiated: they
  are linear (segment means), so their backward is a scatter of the
  averaged gradient.
- The `max(A,0)` clamp backpropagates as a mask (`A>0`).
- Everything else is matmuls + exp: standard chain rule, written per
  head.

Deliberate deviation from the torch probe: `M` is computed with the
RUNTIME's `ridge_pinv` (Cholesky of AᵀA+λI, λ = 1e-6·mean diag, f64 —
`nystrom.rs`) instead of `torch.linalg.pinv(rtol=1e-6)`. The polish
adapts the model to the kernel it will RUN, and M is constant in
backward either way, so runtime-consistency wins over probe parity.

Deliberate deviation #2: the streaming runtime kernel guards the far
field with an AGGREGATE `far_den ≥ 0` test per row, while the matrix
form clamps per (t,j). Training uses the per-element clamp (the form
the certified polish was trained through); generation gates run the
real streaming kernel. Both are the same skeleton to first order; the
measured transfer (torch: loops → 3/3 clean under the streaming-
equivalent evaluation) is the evidence this mismatch is benign.

### 2.3 Op inventory (each with a gradcheck)

| op | forward | backward |
|---|---|---|
| `matmul_nt` | y[n,m] = x[n,k]·W[m,k]ᵀ (runtime weight layout) | dX = dY·W, dW = dYᵀ·X |
| SiLU | x·σ(x) | σ(x)(1 + x(1−σ(x))) |
| mul (SwiGLU) | a⊙b | dA = dY⊙b, dB = dY⊙a |
| RMSNorm ×2 | x̂·w (Qwen) and x̂·(1+w) (Gemma) | gain grad + through-grad |
| residual add | y = a+b | pass-through |
| tied lm_head | logits = x·Embᵀ | dX only (embeddings frozen) |
| softmax-CE | mean NLL | (softmax − onehot)/N |
| KL(t‖s) | Σ p_t(log p_t − log p_s)/N | (softmax_s − p_t)/N |
| RoPE | half-split rotation | inverse rotation (through only) |
| exact causal attention | per-head softmax(QKᵀ/√d)V | dq,dk,dv (probs recomputed) |
| Nyström joint | §2.2 | dq,dk,dv with M constant |
| GQA repeat_interleave | k,v broadcast to Q heads | sum-reduce over the group |
| seg means | landmark averages | scatter of mean grad |

### 2.4 Trainer loop

Per step: sample `bs` random windows of `seq+1` tokens from the train
ids → teacher forward (frozen weights, exact attention, no grad) →
student forward (checkpointed, converted layers use §2.2) → chunked
CE+KL loss and dlogits → backward through final norm and layers
(recompute per layer) → global-norm clip 1.0 → AdamW. Every
`eval_every` steps: student CE-ppl on deterministic evenly-spaced val
windows (`heal_hybridk_06b.py::val_ppl` discipline — random windows
made all gate comparisons ride ±15% sampling noise); track best; at
the end restore the best parameter snapshot.

Corpus: `--corpus` text is tokenized with the model's embedded
tokenizer. With `--val-corpus` absent, the LAST 10% of tokens is held
out for validation (never sampled for training windows).

### 2.5 Write-back

`<model>.fcd.cmf` = byte-copy of every source tensor except the
converted layers' `input_layernorm.weight`,
`post_attention_layernorm.weight`, `mlp.{gate,up,down}_proj.weight`,
which are written as F32 tensors (the directory is per-tensor dtyped;
mixed files are first-class — the honest simple path, no requant
noise on freshly trained weights). The header gains
`provenance.o1_attn` (so `cortiq run` picks the kernel up
automatically) and `provenance.fcd` (steps, lr, kl weight, val ppl
before/after, best step — measured provenance, `cortiq story`
material).

## 3. GDN (Qwen3.5 hybrids) through-backward — BUILT

Implemented (`fcd_ops::gdn_*`, dispatched via the `FcdAttn` enum in
`fcd.rs`): GDN layers of a hybrid run FROZEN in both teacher and
student, with a true BPTT through-backward so converted layers BELOW
them still learn. Through-grad ONLY — every GDN weight stays frozen
(the FCD policy). `vmf_phase` linear cores are still refused loudly.

What was built, mirroring the original plan:

1. **State chain (BPTT).** Exact reverse of `gdn_step`'s S ← g·S;
   kv = Sᵀk̂; S += k̂⊗β(v−kv); o = Sᵀq̂ per step:
   dS ← q̂⊗do; du = dSᵀk̂; dk̂ += dS·u + S_pre·dkv; dβ = du·(v−kv);
   dv = β·du; dkv = −β·du; dS_pre = dS + k̂⊗dkv;
   dg = ⟨dS_pre, S_{t−1}⟩; dS_{t−1} = g·dS_pre.
   The backward stores the FULL per-head state history (one head at a
   time, freed between heads): T·dk·dv f64 ≈ 67 MB at Qwen3.5-0.8B
   geometry — affordable; larger models would switch to the segment
   checkpointing described in the original plan (entry points don't
   change).
2. **Conv ring.** `gdn_conv_fwd/bwd`: the ring is unrolled into a
   causal FIR over the window (positions < 0 are zeros — fresh-state
   semantics); backward is the flipped-tap correlation through the
   SiLU derivative at the pre-activation.
3. **Gates.** da = dg·g·(−e^{A_log})·σ(a+dt_bias) (softplus threshold
   20 matches `linear_core` exactly), db = dβ·β(1−β).
4. **l2-normalization** with the 1e-6 floor inside the sqrt and the
   1/√dk on q: dq = invq·dq̂ − q·(dq̂·q)·invq/(Σq²+1e-6).
5. **Gated per-head RMSNorm output** x̂·w·silu(z), norm-before-gate:
   RMSNorm through-grad with the input-dependent effective gain
   w⊙silu(z) (z independent of o, so the gain is constant w.r.t. the
   normalized input), plus dz = dout·x̂·w·silu′(z).
6. **GQA mapping**: rep = nv/nk v-heads share one k-head's q/k
   channels; the backward accumulates the shared-channel grads across
   the group (`gdn_group_bwd` is the parallel unit — one (sequence,
   k-head) per worker, exclusively-owned output channels).

Validation:
- op-level f64 gradcheck over a T=10 window, FD on all four projection
  streams: dQKV 1.6e-9, dZ 6.1e-12, dA 8.0e-11, dB 8.7e-11;
- forward parity vs the runtime operator `linear_core::gdn_forward`:
  max |Δ| 3.4e-8 over 10 positions (the hybrid teacher computes the
  same function the runtime serves);
- block-level FD through the WHOLE training graph with a trainable
  Full layer BELOW a frozen GDN layer: worst rel err 8.9e-4.

The Qwen3.5 full-attention **output gate** (per-head [q; gate] rows in
q_proj, σ(gate) on the head outputs) is also implemented — forward
split + gate backward + grad re-interleave — block-FD-checked at
6.9e-4. Converted Qwen3.5 full-attention layers are therefore
polishable.

## 4. Honest limits

- Only dense FFN layers are trainable (MoE FFN polish would need
  routed dW accumulation — out of scope until an O(1) MoE target
  exists).
- `vmf_phase` linear cores have no backward — refused loudly.
- GDN layers are never converted and never trained (through-grad
  only); their LN/FFN stay frozen because the certified recipe trains
  converted layers only.
- Training assumes `tie_word_embeddings` or an explicit `lm_head`
  tensor; both paths keep the head frozen.
