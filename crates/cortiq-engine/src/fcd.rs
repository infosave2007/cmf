//! FCD polish trainer — native Rust quality-polish for O(1)-converted
//! models (docs/RUST_FCD.md). Removes the last Python dependency from
//! the `cortiq convert --o1` pipeline.
//!
//! Certified recipe (torch reference `nystrom_fcd_full2_06b.py`,
//! Qwen3-0.6B 28/28): train ONLY the LN gains + FFN of the converted
//! layers, loss = (1−0.7)·CE + 0.7·KL(teacher‖student), AdamW 5e-5
//! (torch defaults), grad clip 1.0, batch 2×512 fresh random windows,
//! quick deterministic val every 25 steps, restore the best checkpoint.
//!
//! Structure:
//! - the whole model is dequantized to f32 once; teacher and student
//!   SHARE the frozen set, trainables get separate master copies (the
//!   KL anchor never drifts);
//! - layer-level activation checkpointing: the forward keeps only each
//!   layer's input hidden; the backward re-runs one layer at a time;
//! - converted layers use the certified matrix form of the Nyström
//!   joint kernel in f64 (fcd_ops), M constant in backward;
//! - GDN (GatedDeltaNet) layers of Qwen3.5-class hybrids run frozen in
//!   BOTH teacher and student, with a true BPTT through-backward
//!   (fcd_ops::gdn_*) so trainable layers BELOW them still learn;
//!   vmf_phase linear layers are refused (no backward yet).

use crate::fcd_ops::{self as ops, NysCfg};
use crate::nystrom::{O1Cfg, O1Layers};
use crate::pipeline::{DenseFfn, FfnKind, Pipeline};
use crate::pool::Pool;
use crate::qtensor::QTensor;
use crate::sampler::{SamplerConfig, SplitMix64};
use cortiq_core::{CmfModel, LayerType, NormStyle, TensorDtype};
use std::sync::Arc;

/// Position-chunk size for the tied-head loss (never materializes the
/// full [B·T, vocab] logits of teacher AND student together).
const LM_CHUNK: usize = 32;

/// AdamW hyper-parameters — torch defaults, part of the certified recipe.
const ADAM_B1: f64 = 0.9;
const ADAM_B2: f64 = 0.999;
const ADAM_EPS: f64 = 1e-8;
const ADAM_WD: f64 = 0.01;

/// Training hyper-parameters (defaults = the certified recipe).
#[derive(Clone, Debug)]
pub struct FcdHyper {
    pub steps: usize,
    pub lr: f64,
    pub kl_w: f64,
    pub eval_every: usize,
    pub bs: usize,
    pub seq: usize,
    pub seed: u64,
}

impl Default for FcdHyper {
    fn default() -> Self {
        Self { steps: 300, lr: 5e-5, kl_w: 0.7, eval_every: 25, bs: 2, seq: 512, seed: 0 }
    }
}

/// What the polish measured — written into `provenance.fcd` and
/// reported by the CLI.
#[derive(Clone, Debug)]
pub struct FcdReport {
    pub converted: Vec<usize>,
    /// Teacher (exact attention) quick-val ppl — the anchor.
    pub teacher_ppl: f64,
    /// Student quick-val ppl BEFORE training (zero-shot o1 damage).
    pub ppl_start: f64,
    /// Best quick-val ppl during training (the restored checkpoint).
    pub ppl_best: f64,
    pub best_step: usize,
    /// Final val ppl of the restored checkpoint on the wider window set.
    pub ppl_final: f64,
    pub steps_run: usize,
    pub sec_per_step: f64,
    /// Per-step (ce, kl) — the training trajectory, unweighted.
    pub losses: Vec<(f64, f64)>,
    /// Generation-gate record (None = ppl-only selection).
    pub gate: Option<GateReport>,
}

/// What the generation gate saw and decided.
#[derive(Clone, Debug)]
pub struct GateReport {
    /// Zero-shot (step-0) loop scores per prompt — the baseline.
    pub baseline: Vec<f64>,
    /// Per eval checkpoint: (step, val ppl, loop scores, passed).
    pub evals: Vec<(usize, f64, Vec<f64>, bool)>,
    /// Step whose params were restored (None = identity: the polish
    /// was rejected, the artifact carries the zero-shot state).
    pub chosen: Option<usize>,
}

// ───────────────────── generation gate (claim 13) ─────────────────────

/// Loopiness of a generated id sequence: 1 − unique 4-grams / total.
/// 0 = no repeated 4-gram; near 1 = a tight loop.
pub fn loop_score(ids: &[u32]) -> f64 {
    if ids.len() < 5 {
        return 0.0;
    }
    let grams: std::collections::HashSet<&[u32]> = ids.windows(4).collect();
    1.0 - grams.len() as f64 / ids.windows(4).count() as f64
}

/// Generation-gate configuration (Patent 16 draft, claim 13:
/// checkpoint selection gated on generation-behavior metrics measured
/// through the SERVED kernel, not on the training objective alone).
#[derive(Clone, Debug)]
pub struct GenGateCfg {
    /// Fixed long-context prompts (token ids), greedy-decoded at every
    /// eval checkpoint.
    pub prompts: Vec<Vec<u32>>,
    pub gen_tokens: usize,
    /// A checkpoint fails if ANY prompt's loop score exceeds this.
    pub threshold: f64,
    /// …or exceeds its zero-shot baseline by more than this.
    pub baseline_slack: f64,
}

impl GenGateCfg {
    /// The standard 3-prompt probe of the torch reference: 400-token
    /// windows at L/10, L/2, 8L/10 of the val stream, greedy 60.
    pub fn standard(va: &[u32]) -> Option<Self> {
        let l = va.len().saturating_sub(500);
        if l < 400 {
            return None;
        }
        let prompts = [l / 10, l / 2, 8 * l / 10]
            .iter()
            .map(|&off| va[off..off + 400].to_vec())
            .collect();
        Some(Self { prompts, gen_tokens: 60, threshold: 0.35, baseline_slack: 0.10 })
    }
}

/// Gate predicate (Patent 16 draft, claim 13): a checkpoint PASSES iff
/// no prompt's loop score exceeds `threshold` AND none exceeds its
/// zero-shot baseline by more than `slack` — boundary values pass.
pub fn gate_pass(scores: &[f64], baseline: &[f64], threshold: f64, slack: f64) -> bool {
    scores.iter().zip(baseline).all(|(&s, &b)| s <= threshold && s <= b + slack)
}

/// Checkpoint selection: lowest val ppl AMONG GATE-PASSING checkpoints
/// (ties → earliest). None = nothing passed → the caller must restore
/// the zero-shot state (identity polish): the stage must never make
/// generation worse than conversion alone. (Patent 16 draft, claim 13.)
pub fn select_checkpoint(
    evals: &[(usize, f64, Vec<f64>)],
    baseline: &[f64],
    threshold: f64,
    slack: f64,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, (_, ppl, scores)) in evals.iter().enumerate() {
        if !gate_pass(scores, baseline, threshold, slack) {
            continue;
        }
        if best.map(|b| *ppl < evals[b].1).unwrap_or(true) {
            best = Some(i);
        }
    }
    best
}

// ───────────────────────── model container ─────────────────────────

/// Frozen attention operator of one layer — the per-layer dispatch
/// point for through-backwards (docs/RUST_FCD.md §3).
enum FcdAttn {
    Full {
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        q_norm: Option<Vec<f32>>,
        k_norm: Option<Vec<f32>>,
        bias: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
        /// Qwen3.5: wq rows = 2·nh·hd, per-head [q; gate]; the head
        /// outputs are multiplied by σ(gate) before o_proj.
        output_gate: bool,
    },
    /// GatedDeltaNet (Qwen3.5 hybrids): never converted, never trained;
    /// through-backward = BPTT over the window (fcd_ops::gdn_*).
    Gdn {
        wqkv: Vec<f32>,
        wz: Vec<f32>,
        wa: Vec<f32>,
        wb: Vec<f32>,
        conv: Vec<f32>,
        a_log: Vec<f32>,
        dt_bias: Vec<f32>,
        norm: Vec<f32>,
        wout: Vec<f32>,
    },
}

struct FcdLayer {
    attn: FcdAttn,
    inter: usize,
    // Frozen originals: the teacher's LN/FFN and the student's init.
    iln: Vec<f32>,
    pln: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

/// GDN geometry shared by every linear layer (arch.linear_* fields).
#[derive(Clone, Copy)]
struct GdnDims {
    nv: usize,
    nk: usize,
    dk: usize,
    dv: usize,
    kk: usize,
}

impl GdnDims {
    fn c_dim(&self) -> usize {
        2 * self.nk * self.dk + self.nv * self.dv
    }
    fn vd(&self) -> usize {
        self.nv * self.dv
    }
}

/// The f32 training replica of a .cmf model (≤ 1B targets).
pub struct FcdModel {
    pub hidden: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub nl: usize,
    pub vocab: usize,
    eps: f64,
    gemma: bool,
    rotary_dim: usize,
    inv_freq: Vec<f64>,
    /// [vocab, hidden]; also the tied head when `lm_head` is None.
    embed: Vec<f32>,
    lm_head: Option<Vec<f32>>,
    final_norm: Vec<f32>,
    layers: Vec<FcdLayer>,
    /// Which layers run the Nyström kernel in the student forward.
    o1_flags: Vec<bool>,
    nys: NysCfg,
    /// GDN geometry (present when the model has linear layers).
    gdn: Option<GdnDims>,
    pool: Option<Arc<Pool>>,
}

fn deq(model: &CmfModel, name: &str) -> Result<Vec<f32>, String> {
    let e = model
        .tensor(name)
        .ok_or_else(|| format!("tensor '{name}' not found"))?;
    let mut out = vec![0f32; e.n_elems()];
    cortiq_core::quant::dequant_tensor(e, model.entry_bytes(e), &mut out)?;
    Ok(out)
}

impl FcdModel {
    /// Dequantize a model into the f32 training replica. Refuses what
    /// the backward cannot honestly differentiate yet (loud, not silent).
    pub fn from_cmf(model: &CmfModel, o1: &O1Cfg) -> Result<Self, String> {
        let arch = model.arch().clone();
        let has_linear = arch
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::LinearAttention));
        let gdn = if has_linear {
            let lc = arch.linear_core.as_ref().ok_or_else(|| {
                "model has linear layers but no arch.linear_core".to_string()
            })?;
            if lc.kind != "gated_delta_net" {
                return Err(format!(
                    "linear core '{}' has no FCD backward (only gated_delta_net)",
                    lc.kind
                ));
            }
            Some(GdnDims {
                nv: lc.num_heads,
                nk: arch
                    .linear_num_key_heads
                    .ok_or("linear core needs arch.linear_num_key_heads")?,
                dk: arch
                    .linear_key_head_dim
                    .ok_or("linear core needs arch.linear_key_head_dim")?,
                dv: lc.value_head_dim,
                kk: arch
                    .linear_conv_kernel_dim
                    .ok_or("linear core needs arch.linear_conv_kernel_dim")?,
            })
        } else {
            None
        };
        let (nh, nkv, hd, h) = (
            arch.num_attention_heads,
            arch.num_kv_heads,
            arch.head_dim,
            arch.hidden_size,
        );
        let embed = deq(model, "model.embed_tokens.weight")?;
        let lm_head = if model.tensor("lm_head.weight").is_some() {
            Some(deq(model, "lm_head.weight")?)
        } else if arch.tie_word_embeddings {
            None
        } else {
            return Err("no lm_head.weight and tie_word_embeddings is false".into());
        };
        let final_norm = deq(model, "model.norm.weight")?;

        let mut layers = Vec::with_capacity(arch.num_layers);
        for li in 0..arch.num_layers {
            let p = format!("model.layers.{li}.");
            if model.tensor(&format!("{p}mlp.gate.weight")).is_some() {
                return Err(format!(
                    "layer {li} is MoE — FCD polish supports dense FFN only"
                ));
            }
            let attn = match arch.layer_types.get(li) {
                Some(LayerType::LinearAttention) => {
                    let la = |n: &str| deq(model, &format!("{p}linear_attn.{n}"));
                    FcdAttn::Gdn {
                        wqkv: la("in_proj_qkv.weight")?,
                        wz: la("in_proj_z.weight")?,
                        wa: la("in_proj_a.weight")?,
                        wb: la("in_proj_b.weight")?,
                        conv: la("conv1d.weight")?,
                        a_log: la("A_log")?,
                        dt_bias: la("dt_bias")?,
                        norm: la("norm.weight")?,
                        wout: la("out_proj.weight")?,
                    }
                }
                _ => {
                    let wq = deq(model, &format!("{p}self_attn.q_proj.weight"))?;
                    let output_gate = wq.len() == 2 * nh * hd * h;
                    let opt = |n: &str| -> Option<Vec<f32>> {
                        model
                            .tensor(&format!("{p}self_attn.{n}"))
                            .and_then(|_| deq(model, &format!("{p}self_attn.{n}")).ok())
                    };
                    let bias = match (
                        opt("q_proj.bias"),
                        opt("k_proj.bias"),
                        opt("v_proj.bias"),
                    ) {
                        (Some(a), Some(b), Some(c)) => Some((a, b, c)),
                        _ => None,
                    };
                    FcdAttn::Full {
                        wq,
                        wk: deq(model, &format!("{p}self_attn.k_proj.weight"))?,
                        wv: deq(model, &format!("{p}self_attn.v_proj.weight"))?,
                        wo: deq(model, &format!("{p}self_attn.o_proj.weight"))?,
                        q_norm: opt("q_norm.weight"),
                        k_norm: opt("k_norm.weight"),
                        bias,
                        output_gate,
                    }
                }
            };
            let gate = deq(model, &format!("{p}mlp.gate_proj.weight"))?;
            let inter = gate.len() / h;
            layers.push(FcdLayer {
                attn,
                inter,
                iln: deq(model, &format!("{p}input_layernorm.weight"))?,
                pln: deq(model, &format!("{p}post_attention_layernorm.weight"))?,
                gate,
                up: deq(model, &format!("{p}mlp.up_proj.weight"))?,
                down: deq(model, &format!("{p}mlp.down_proj.weight"))?,
            });
        }

        let rotary_dim = ((hd as f32 * arch.partial_rotary_factor) as usize).max(2).min(hd);
        let base = arch.rope_theta;
        let inv_freq: Vec<f64> = (0..rotary_dim / 2)
            .map(|i| 1.0 / base.powf(2.0 * i as f64 / rotary_dim as f64))
            .collect();
        let mut flags = o1.layer_flags(arch.num_layers);
        flags.resize(arch.num_layers, false);
        // Only full-attention layers are o1-convertible (same rule as
        // Pipeline::set_o1) — a GDN layer keeps its own operator.
        for (li, f) in flags.iter_mut().enumerate() {
            if *f && !matches!(layers[li].attn, FcdAttn::Full { .. }) {
                *f = false;
            }
        }
        Ok(Self {
            hidden: h,
            nh,
            nkv,
            hd,
            nl: arch.num_layers,
            vocab: arch.vocab_size.min(embed.len() / h),
            eps: arch.rms_norm_eps,
            gemma: matches!(arch.norm_style, NormStyle::Gemma),
            rotary_dim,
            inv_freq,
            embed,
            lm_head,
            final_norm,
            layers,
            o1_flags: flags,
            // prefill: None = half the window, the same seal point
            // `cortiq ppl --o1` defaults to (see NysCfg::prefill).
            nys: NysCfg { m: o1.m, w: o1.w, sink: o1.sink, prefill: None },
            gdn,
            pool: Pool::from_env(),
        })
    }

    /// Converted (trainable) layer indices.
    pub fn converted(&self) -> Vec<usize> {
        (0..self.nl).filter(|&i| self.o1_flags[i]).collect()
    }

    fn head_weight(&self) -> &[f32] {
        self.lm_head.as_deref().unwrap_or(&self.embed)
    }
}

// ───────────────────────── trainable state ─────────────────────────

/// Per converted layer, in this fixed order.
const PARAMS_PER_LAYER: usize = 5; // iln, pln, gate, up, down

/// Master copies + grads + AdamW moments of the trainable tensors.
pub struct TrainState {
    pub layers: Vec<usize>,
    /// layers.len()·5 tensors, layer-major, [iln, pln, gate, up, down].
    pub data: Vec<Vec<f32>>,
    grad: Vec<Vec<f32>>,
    m1: Vec<Vec<f32>>,
    m2: Vec<Vec<f32>>,
    step_t: u64,
}

impl TrainState {
    pub fn new(fm: &FcdModel) -> Self {
        let layers = fm.converted();
        let mut data = Vec::with_capacity(layers.len() * PARAMS_PER_LAYER);
        for &li in &layers {
            let l = &fm.layers[li];
            data.push(l.iln.clone());
            data.push(l.pln.clone());
            data.push(l.gate.clone());
            data.push(l.up.clone());
            data.push(l.down.clone());
        }
        let zeros: Vec<Vec<f32>> = data.iter().map(|d| vec![0f32; d.len()]).collect();
        Self {
            layers,
            grad: zeros.clone(),
            m1: zeros.clone(),
            m2: zeros,
            data,
            step_t: 0,
        }
    }

    fn slot(&self, li: usize) -> Option<usize> {
        self.layers.iter().position(|&x| x == li)
    }

    /// Read access to the accumulated gradients (gradcheck harness).
    #[doc(hidden)]
    pub fn grads(&self) -> &[Vec<f32>] {
        &self.grad
    }

    fn zero_grad(&mut self) {
        for g in &mut self.grad {
            for v in g.iter_mut() {
                *v = 0.0;
            }
        }
    }

    /// Global-norm clip (1.0) + one AdamW step (torch defaults,
    /// decoupled weight decay).
    fn clip_and_step(&mut self, lr: f64) -> f64 {
        let mut sq = 0f64;
        for g in &self.grad {
            for &v in g {
                sq += (v as f64) * (v as f64);
            }
        }
        let gn = sq.sqrt();
        let scale = if gn > 1.0 { 1.0 / (gn + 1e-6) } else { 1.0 };
        self.step_t += 1;
        let bc1 = 1.0 - ADAM_B1.powi(self.step_t as i32);
        let bc2 = 1.0 - ADAM_B2.powi(self.step_t as i32);
        for p in 0..self.data.len() {
            let (d, g, m, v) = (
                &mut self.data[p],
                &self.grad[p],
                &mut self.m1[p],
                &mut self.m2[p],
            );
            for i in 0..d.len() {
                let gi = g[i] as f64 * scale;
                let mi = ADAM_B1 * m[i] as f64 + (1.0 - ADAM_B1) * gi;
                let vi = ADAM_B2 * v[i] as f64 + (1.0 - ADAM_B2) * gi * gi;
                m[i] = mi as f32;
                v[i] = vi as f32;
                let upd = (mi / bc1) / ((vi / bc2).sqrt() + ADAM_EPS) + ADAM_WD * d[i] as f64;
                d[i] = (d[i] as f64 - lr * upd) as f32;
            }
        }
        gn
    }
}

/// LN/FFN weight view of one layer — frozen originals for the teacher
/// (and non-converted student layers), master copies for trainables.
#[derive(Clone, Copy)]
struct LnFfn<'a> {
    iln: &'a [f32],
    pln: &'a [f32],
    gate: &'a [f32],
    up: &'a [f32],
    down: &'a [f32],
}

fn ln_ffn<'a>(fm: &'a FcdModel, ts: Option<&'a TrainState>, li: usize) -> LnFfn<'a> {
    if let Some(t) = ts {
        if let Some(s) = t.slot(li) {
            let b = s * PARAMS_PER_LAYER;
            return LnFfn {
                iln: &t.data[b],
                pln: &t.data[b + 1],
                gate: &t.data[b + 2],
                up: &t.data[b + 3],
                down: &t.data[b + 4],
            };
        }
    }
    let l = &fm.layers[li];
    LnFfn { iln: &l.iln, pln: &l.pln, gate: &l.gate, up: &l.up, down: &l.down }
}

// ───────────────────── layer forward (+ recompute) ─────────────────────

/// Intra-layer activations rebuilt during the checkpointed backward.
enum AttnActs {
    Full {
        qpre: Vec<f32>,
        kpre: Vec<f32>,
        vproj: Vec<f32>,
        qrot: Vec<f32>,
        krot: Vec<f32>,
        qinv: Vec<f32>,
        kinv: Vec<f32>,
        /// Pre-gate per-head attention outputs (needed for the output
        /// gate's backward); always kept — transient per layer.
        ao: Vec<f32>,
        /// Raw gate half of q_proj (empty without an output gate).
        gate_pre: Vec<f32>,
    },
    /// Raw projection streams — the GDN backward replays conv + the
    /// recurrence from these.
    Gdn { qkv: Vec<f32>, z: Vec<f32>, a: Vec<f32>, b: Vec<f32> },
}

struct LayerActs {
    inv1: Vec<f32>,
    attn: AttnActs,
    h1: Vec<f32>,
    n2: Vec<f32>,
    inv2: Vec<f32>,
    gpre: Vec<f32>,
    upre: Vec<f32>,
    act: Vec<f32>,
}

/// Disjoint-write pointer for pooled per-head scatter (pipeline pattern).
struct SendMut<T>(*mut T);
unsafe impl<T> Send for SendMut<T> {}
unsafe impl<T> Sync for SendMut<T> {}
impl<T> SendMut<T> {
    #[inline]
    unsafe fn at(&self, i: usize) -> *mut T {
        unsafe { self.0.add(i) }
    }
}

impl FcdModel {
    /// Per-head RMS-norm (qk-norm) + partial RoPE for all rows of a
    /// projection buffer. `heads` per row, `x` is `[n, heads·hd]`.
    /// Saves the per-(row, head) rms inv when a norm gain is present.
    fn qk_norm_rope(
        &self,
        x: &mut [f32],
        norm: Option<&[f32]>,
        heads: usize,
        t: usize,
        inv_out: &mut [f32],
    ) {
        let hd = self.hd;
        let n = x.len() / (heads * hd);
        for r in 0..n {
            let pos = r % t;
            for hh in 0..heads {
                let s = (r * heads + hh) * hd;
                let head = &mut x[s..s + hd];
                if let Some(w) = norm {
                    let mut inv = [0f32; 1];
                    let mut y = [0f32; 256];
                    debug_assert!(hd <= 256);
                    ops::rmsnorm_fwd(head, w, self.eps, self.gemma, &mut y[..hd], &mut inv);
                    head.copy_from_slice(&y[..hd]);
                    inv_out[r * heads + hh] = inv[0];
                }
                ops::rope_fwd(&mut head[..self.rotary_dim], pos, &self.inv_freq);
            }
        }
    }

    /// One layer forward over `b` sequences of length `t` (rows are
    /// b-major). `nystrom` switches converted (Full) student layers to
    /// the certified matrix kernel (f64 per head); exact heads run in
    /// f32; GDN layers run the frozen BPTT-capable operator. Returns
    /// (h_out, intra-layer activations when `want_acts`).
    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        li: usize,
        h_in: &[f32],
        b: usize,
        t: usize,
        wts: &LnFfn,
        nystrom: bool,
        want_acts: bool,
    ) -> (Vec<f32>, Option<LayerActs>) {
        let hsz = self.hidden;
        let n = b * t;
        let l = &self.layers[li];
        let pool = self.pool.as_deref();

        let mut n1 = vec![0f32; n * hsz];
        let mut inv1 = vec![0f32; n];
        ops::rmsnorm_fwd(h_in, wts.iln, self.eps, self.gemma, &mut n1, &mut inv1);

        let (attn_out, attn_acts) = match &l.attn {
            FcdAttn::Full { .. } => self.full_attn_fwd(&l.attn, &n1, b, t, nystrom),
            FcdAttn::Gdn { .. } => self.gdn_attn_fwd(&l.attn, &n1, b, t),
        };

        let mut h1 = h_in.to_vec();
        for (a, &x) in h1.iter_mut().zip(&attn_out) {
            *a += x;
        }

        let mut n2 = vec![0f32; n * hsz];
        let mut inv2 = vec![0f32; n];
        ops::rmsnorm_fwd(&h1, wts.pln, self.eps, self.gemma, &mut n2, &mut inv2);

        let inter = l.inter;
        let mut gpre = vec![0f32; n * inter];
        ops::gemm_nt(&n2, wts.gate, &mut gpre, n, hsz, inter, pool);
        let mut upre = vec![0f32; n * inter];
        ops::gemm_nt(&n2, wts.up, &mut upre, n, hsz, inter, pool);
        let mut act = vec![0f32; n * inter];
        for i in 0..n * inter {
            act[i] = ops::silu(gpre[i]) * upre[i];
        }
        let mut ffn = vec![0f32; n * hsz];
        ops::gemm_nt(&act, wts.down, &mut ffn, n, inter, hsz, pool);
        let mut h2 = h1.clone();
        for (a, &x) in h2.iter_mut().zip(&ffn) {
            *a += x;
        }

        let acts = want_acts.then_some(LayerActs {
            inv1,
            attn: attn_acts,
            h1,
            n2,
            inv2,
            gpre,
            upre,
            act,
        });
        (h2, acts)
    }

    /// Full-attention forward: projections (+optional biases), optional
    /// per-head [q; gate] split (Qwen3.5 output gate), qk-norm + RoPE,
    /// per-head exact-or-Nyström attention, σ(gate) multiply, o_proj.
    fn full_attn_fwd(
        &self,
        attn: &FcdAttn,
        n1: &[f32],
        b: usize,
        t: usize,
        nystrom: bool,
    ) -> (Vec<f32>, AttnActs) {
        let FcdAttn::Full { wq, wk, wv, wo, q_norm, k_norm, bias, output_gate } = attn else {
            unreachable!("full_attn_fwd on a non-Full layer");
        };
        let (hsz, nh, nkv, hd) = (self.hidden, self.nh, self.nkv, self.hd);
        let n = b * t;
        let pool = self.pool.as_deref();
        let qdim = nh * hd;
        let kvdim = nkv * hd;
        let rep = nh / nkv;
        let qrows = if *output_gate { 2 * qdim } else { qdim };

        let mut qraw = vec![0f32; n * qrows];
        ops::gemm_nt(n1, wq, &mut qraw, n, hsz, qrows, pool);
        let mut kpre = vec![0f32; n * kvdim];
        ops::gemm_nt(n1, wk, &mut kpre, n, hsz, kvdim, pool);
        let mut vproj = vec![0f32; n * kvdim];
        ops::gemm_nt(n1, wv, &mut vproj, n, hsz, kvdim, pool);
        if let Some((bq, bk, bv)) = bias {
            for r in 0..n {
                for (x, bb) in qraw[r * qrows..(r + 1) * qrows].iter_mut().zip(bq) {
                    *x += bb;
                }
                for (x, bb) in kpre[r * kvdim..(r + 1) * kvdim].iter_mut().zip(bk) {
                    *x += bb;
                }
                for (x, bb) in vproj[r * kvdim..(r + 1) * kvdim].iter_mut().zip(bv) {
                    *x += bb;
                }
            }
        }
        // Gate split: per-head [q(hd); gate(hd)] (runtime convention).
        let (qpre, gate_pre) = if *output_gate {
            let mut qh = vec![0f32; n * qdim];
            let mut gp = vec![0f32; n * qdim];
            for r in 0..n {
                for h in 0..nh {
                    let src = r * qrows + 2 * h * hd;
                    let dst = r * qdim + h * hd;
                    qh[dst..dst + hd].copy_from_slice(&qraw[src..src + hd]);
                    gp[dst..dst + hd].copy_from_slice(&qraw[src + hd..src + 2 * hd]);
                }
            }
            (qh, gp)
        } else {
            (qraw, Vec::new())
        };

        let mut qrot = qpre.clone();
        let mut krot = kpre.clone();
        let mut qinv = vec![0f32; n * nh];
        let mut kinv = vec![0f32; n * nkv];
        self.qk_norm_rope(&mut qrot, q_norm.as_deref(), nh, t, &mut qinv);
        self.qk_norm_rope(&mut krot, k_norm.as_deref(), nkv, t, &mut kinv);

        // ── attention heads: parallel over (sequence, head) ──
        let mut ao = vec![0f32; n * qdim];
        {
            let units = b * nh;
            let aop = SendMut(ao.as_mut_ptr());
            let qr = &qrot;
            let kr = &krot;
            let vr = &vproj;
            let nys = self.nys;
            let run_unit = |u: usize| {
                let (bi, h) = (u / nh, u % nh);
                let g = h / rep;
                if nystrom {
                    // Certified matrix kernel in f64 (docs/RUST_FCD.md §2.2).
                    let mut q64 = vec![0f64; t * hd];
                    let mut k64 = vec![0f64; t * hd];
                    let mut v64 = vec![0f64; t * hd];
                    for p in 0..t {
                        let r = bi * t + p;
                        for c in 0..hd {
                            q64[p * hd + c] = qr[r * qdim + h * hd + c] as f64;
                            k64[p * hd + c] = kr[r * kvdim + g * hd + c] as f64;
                            v64[p * hd + c] = vr[r * kvdim + g * hd + c] as f64;
                        }
                    }
                    let mut o64 = vec![0f64; t * hd];
                    ops::nystrom_head_fwd(&q64, &k64, &v64, t, hd, hd, &nys, &mut o64);
                    for p in 0..t {
                        let r = bi * t + p;
                        for c in 0..hd {
                            // SAFETY: (row, head) slices are disjoint per unit.
                            unsafe {
                                *aop.at(r * qdim + h * hd + c) = o64[p * hd + c] as f32;
                            }
                        }
                    }
                } else {
                    let mut q32 = vec![0f32; t * hd];
                    let mut k32 = vec![0f32; t * hd];
                    let mut v32 = vec![0f32; t * hd];
                    for p in 0..t {
                        let r = bi * t + p;
                        q32[p * hd..(p + 1) * hd]
                            .copy_from_slice(&qr[r * qdim + h * hd..r * qdim + (h + 1) * hd]);
                        k32[p * hd..(p + 1) * hd]
                            .copy_from_slice(&kr[r * kvdim + g * hd..r * kvdim + (g + 1) * hd]);
                        v32[p * hd..(p + 1) * hd]
                            .copy_from_slice(&vr[r * kvdim + g * hd..r * kvdim + (g + 1) * hd]);
                    }
                    let mut o32 = vec![0f32; t * hd];
                    ops::attn_head_fwd(&q32, &k32, &v32, t, hd, hd, &mut o32);
                    for p in 0..t {
                        let r = bi * t + p;
                        for c in 0..hd {
                            // SAFETY: disjoint (row, head) slices per unit.
                            unsafe {
                                *aop.at(r * qdim + h * hd + c) = o32[p * hd + c];
                            }
                        }
                    }
                }
            };
            match pool {
                Some(p) if units > 1 => p.run(&|widx, nw| {
                    for u in (widx..units).step_by(nw) {
                        run_unit(u);
                    }
                }),
                _ => {
                    for u in 0..units {
                        run_unit(u);
                    }
                }
            }
        }

        // Output gate: multiply the head outputs by σ(gate) before o_proj.
        let ao_eff: Vec<f32> = if *output_gate {
            ao.iter()
                .zip(&gate_pre)
                .map(|(&a, &g)| a * (1.0 / (1.0 + (-g).exp())))
                .collect()
        } else {
            ao.clone()
        };
        let mut attn_out = vec![0f32; n * hsz];
        ops::gemm_nt(&ao_eff, wo, &mut attn_out, n, qdim, hsz, pool);
        (
            attn_out,
            AttnActs::Full { qpre, kpre, vproj, qrot, krot, qinv, kinv, ao, gate_pre },
        )
    }

    /// GDN forward (frozen operator, teacher AND student): batched
    /// projections → f64 conv+SiLU per sequence → pooled per-(seq,
    /// k-head) delta-rule recurrence → out_proj. Matches the runtime
    /// `gdn_forward` (parity-tested in fcd_gradcheck).
    fn gdn_attn_fwd(
        &self,
        attn: &FcdAttn,
        n1: &[f32],
        b: usize,
        t: usize,
    ) -> (Vec<f32>, AttnActs) {
        let FcdAttn::Gdn { wqkv, wz, wa, wb, conv, a_log, dt_bias, norm, wout } = attn else {
            unreachable!("gdn_attn_fwd on a non-GDN layer");
        };
        let d = self.gdn.expect("gdn layer without gdn dims");
        let (hsz, n) = (self.hidden, b * t);
        let pool = self.pool.as_deref();
        let (c_dim, vd, nv) = (d.c_dim(), d.vd(), d.nv);

        let mut qkv = vec![0f32; n * c_dim];
        ops::gemm_nt(n1, wqkv, &mut qkv, n, hsz, c_dim, pool);
        let mut z = vec![0f32; n * vd];
        ops::gemm_nt(n1, wz, &mut z, n, hsz, vd, pool);
        let mut a = vec![0f32; n * nv];
        ops::gemm_nt(n1, wa, &mut a, n, hsz, nv, pool);
        let mut bstr = vec![0f32; n * nv];
        ops::gemm_nt(n1, wb, &mut bstr, n, hsz, nv, pool);

        let cfg = ops::GdnSeqCfg {
            nv: d.nv,
            nk: d.nk,
            dk: d.dk,
            dv: d.dv,
            kk: d.kk,
            rms_eps: self.eps,
            conv,
            a_log,
            dt_bias,
            norm,
        };
        // f64 streams (runtime-precision recurrence) + per-seq conv.
        let qkv64: Vec<f64> = qkv.iter().map(|&v| v as f64).collect();
        let z64: Vec<f64> = z.iter().map(|&v| v as f64).collect();
        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let b64: Vec<f64> = bstr.iter().map(|&v| v as f64).collect();
        let mut pre64 = vec![0f64; n * c_dim];
        let mut cq64 = vec![0f64; n * c_dim];
        for bi in 0..b {
            let r = bi * t * c_dim..(bi + 1) * t * c_dim;
            ops::gdn_conv_fwd(
                &qkv64[r.clone()],
                t,
                c_dim,
                d.kk,
                conv,
                &mut pre64[r.clone()],
                &mut cq64[r],
            );
        }
        let mut of = vec![0f32; n * vd];
        {
            let units = b * d.nk;
            let rep_v = d.nv / d.nk;
            let ofp = SendMut(of.as_mut_ptr());
            let (cqr, zr, ar, br) = (&cq64, &z64, &a64, &b64);
            let cfg_ref = &cfg;
            let run_unit = |u: usize| {
                let (bi, ko) = (u / d.nk, u % d.nk);
                let mut local = vec![0f64; t * vd];
                ops::gdn_group_fwd(
                    &cqr[bi * t * c_dim..(bi + 1) * t * c_dim],
                    &zr[bi * t * vd..(bi + 1) * t * vd],
                    &ar[bi * t * nv..(bi + 1) * t * nv],
                    &br[bi * t * nv..(bi + 1) * t * nv],
                    t,
                    cfg_ref,
                    ko,
                    &mut local,
                );
                for hh in 0..rep_v {
                    let h = ko * rep_v + hh;
                    for p in 0..t {
                        for dj in 0..d.dv {
                            // SAFETY: v-head columns are exclusive per unit.
                            unsafe {
                                *ofp.at((bi * t + p) * vd + h * d.dv + dj) =
                                    local[p * vd + h * d.dv + dj] as f32;
                            }
                        }
                    }
                }
            };
            match pool {
                Some(p) if units > 1 => p.run(&|widx, nw| {
                    for u in (widx..units).step_by(nw) {
                        run_unit(u);
                    }
                }),
                _ => {
                    for u in 0..units {
                        run_unit(u);
                    }
                }
            }
        }
        let mut attn_out = vec![0f32; n * hsz];
        ops::gemm_nt(&of, wout, &mut attn_out, n, vd, hsz, pool);
        (attn_out, AttnActs::Gdn { qkv, z, a, b: bstr })
    }

    /// One layer backward (docs/RUST_FCD.md §2.3 chain), given the
    /// recomputed `acts`. Accumulates trainable grads when `grads` is
    /// Some; always produces the through-grad dh_in.
    #[allow(clippy::too_many_arguments)]
    fn layer_backward(
        &self,
        li: usize,
        h_in: &[f32],
        b: usize,
        t: usize,
        wts: &LnFfn,
        nystrom: bool,
        acts: &LayerActs,
        dh2: &[f32],
        mut grads: Option<&mut [Vec<f32>]>,
    ) -> Vec<f32> {
        let hsz = self.hidden;
        let n = b * t;
        let l = &self.layers[li];
        let pool = self.pool.as_deref();
        let inter = l.inter;

        // ── FFN backward ──
        let mut dact = vec![0f32; n * inter];
        ops::gemm_dx(dh2, wts.down, &mut dact, n, inter, hsz, pool);
        if let Some(g) = grads.as_deref_mut() {
            ops::gemm_dw(dh2, &acts.act, &mut g[4], n, inter, hsz, pool);
        }
        let mut dg = vec![0f32; n * inter];
        let mut du = vec![0f32; n * inter];
        for i in 0..n * inter {
            dg[i] = dact[i] * acts.upre[i] * ops::silu_bwd(acts.gpre[i]);
            du[i] = dact[i] * ops::silu(acts.gpre[i]);
        }
        let mut dn2 = vec![0f32; n * hsz];
        ops::gemm_dx(&dg, wts.gate, &mut dn2, n, hsz, inter, pool);
        ops::gemm_dx(&du, wts.up, &mut dn2, n, hsz, inter, pool);
        if let Some(g) = grads.as_deref_mut() {
            ops::gemm_dw(&dg, &acts.n2, &mut g[2], n, hsz, inter, pool);
            ops::gemm_dw(&du, &acts.n2, &mut g[3], n, hsz, inter, pool);
        }

        let mut dh1 = dh2.to_vec();
        ops::rmsnorm_bwd(
            &acts.h1,
            wts.pln,
            &acts.inv2,
            &dn2,
            self.gemma,
            &mut dh1,
            grads.as_deref_mut().map(|g| &mut g[1][..]),
        );

        // ── attention backward (dispatch) → dn1 ──
        let dn1 = match &l.attn {
            FcdAttn::Full { .. } => {
                self.full_attn_bwd(&l.attn, &acts.attn, &dh1, b, t, nystrom)
            }
            FcdAttn::Gdn { .. } => self.gdn_attn_bwd(&l.attn, &acts.attn, &dh1, b, t),
        };

        let mut dh_in = dh1.clone();
        ops::rmsnorm_bwd(
            h_in,
            wts.iln,
            &acts.inv1,
            &dn1,
            self.gemma,
            &mut dh_in,
            grads.map(|g| &mut g[0][..]),
        );
        dh_in
    }

    /// Full-attention through-backward: o_proj → output gate →
    /// per-head attention (exact / Nyström-frozen-M) → RoPE → qk-norm →
    /// projections. Frozen weights: dX only.
    fn full_attn_bwd(
        &self,
        attn: &FcdAttn,
        acts: &AttnActs,
        dattn: &[f32],
        b: usize,
        t: usize,
        nystrom: bool,
    ) -> Vec<f32> {
        let FcdAttn::Full { wq, wk, wv, wo, q_norm, k_norm, output_gate, .. } = attn else {
            unreachable!("full_attn_bwd on a non-Full layer");
        };
        let AttnActs::Full { qpre, kpre, vproj, qrot, krot, qinv, kinv, ao, gate_pre } = acts
        else {
            unreachable!("acts mismatch");
        };
        let (hsz, nh, nkv, hd) = (self.hidden, self.nh, self.nkv, self.hd);
        let n = b * t;
        let pool = self.pool.as_deref();
        let qdim = nh * hd;
        let kvdim = nkv * hd;
        let rep = nh / nkv;
        let qrows = if *output_gate { 2 * qdim } else { qdim };

        let mut dao_eff = vec![0f32; n * qdim];
        ops::gemm_dx(dattn, wo, &mut dao_eff, n, qdim, hsz, pool);
        // Output gate: ao_eff = ao·σ(g) → dao = d·σ(g), dg = d·ao·σ′(g).
        let (dao, dgate) = if *output_gate {
            let mut dao = vec![0f32; n * qdim];
            let mut dgp = vec![0f32; n * qdim];
            for i in 0..n * qdim {
                let sig = 1.0 / (1.0 + (-gate_pre[i]).exp());
                dao[i] = dao_eff[i] * sig;
                dgp[i] = dao_eff[i] * ao[i] * sig * (1.0 - sig);
            }
            (dao, dgp)
        } else {
            (dao_eff, Vec::new())
        };

        let mut dqrot = vec![0f32; n * qdim];
        let mut dkrot = vec![0f32; n * kvdim];
        let mut dvproj = vec![0f32; n * kvdim];
        {
            // Parallel over (sequence, kv-group): a unit owns the dk/dv
            // slices of its group and the dq slices of its rep Q heads.
            let units = b * nkv;
            let dqp = SendMut(dqrot.as_mut_ptr());
            let dkp = SendMut(dkrot.as_mut_ptr());
            let dvp = SendMut(dvproj.as_mut_ptr());
            let (qr, kr, vr) = (qrot, krot, vproj);
            let daor = &dao;
            let nys = self.nys;
            let run_unit = |u: usize| {
                let (bi, g) = (u / nkv, u % nkv);
                let mut k64 = vec![0f64; t * hd];
                let mut v64 = vec![0f64; t * hd];
                for p in 0..t {
                    let r = bi * t + p;
                    for c in 0..hd {
                        k64[p * hd + c] = kr[r * kvdim + g * hd + c] as f64;
                        v64[p * hd + c] = vr[r * kvdim + g * hd + c] as f64;
                    }
                }
                let mut dk64 = vec![0f64; t * hd];
                let mut dv64 = vec![0f64; t * hd];
                let mut q64 = vec![0f64; t * hd];
                let mut do64 = vec![0f64; t * hd];
                let mut dq64 = vec![0f64; t * hd];
                for hh in 0..rep {
                    let h = g * rep + hh;
                    for p in 0..t {
                        let r = bi * t + p;
                        for c in 0..hd {
                            q64[p * hd + c] = qr[r * qdim + h * hd + c] as f64;
                            do64[p * hd + c] = daor[r * qdim + h * hd + c] as f64;
                        }
                    }
                    for v in dq64.iter_mut() {
                        *v = 0.0;
                    }
                    if nystrom {
                        ops::nystrom_head_bwd(
                            &q64, &k64, &v64, &do64, t, hd, hd, &nys, &mut dq64, &mut dk64,
                            &mut dv64,
                        );
                    } else {
                        ops::attn_head_bwd(
                            &q64, &k64, &v64, &do64, t, hd, hd, &mut dq64, &mut dk64, &mut dv64,
                        );
                    }
                    for p in 0..t {
                        let r = bi * t + p;
                        for c in 0..hd {
                            // SAFETY: disjoint (row, head) slices per unit.
                            unsafe {
                                *dqp.at(r * qdim + h * hd + c) = dq64[p * hd + c] as f32;
                            }
                        }
                    }
                }
                for p in 0..t {
                    let r = bi * t + p;
                    for c in 0..hd {
                        // SAFETY: disjoint (row, group) slices per unit.
                        unsafe {
                            *dkp.at(r * kvdim + g * hd + c) = dk64[p * hd + c] as f32;
                            *dvp.at(r * kvdim + g * hd + c) = dv64[p * hd + c] as f32;
                        }
                    }
                }
            };
            match pool {
                Some(p) if units > 1 => p.run(&|widx, nw| {
                    for u in (widx..units).step_by(nw) {
                        run_unit(u);
                    }
                }),
                _ => {
                    for u in 0..units {
                        run_unit(u);
                    }
                }
            }
        }

        // qk-norm + RoPE through-grads (frozen gains → no dw).
        let mut dqpre = vec![0f32; n * qdim];
        let mut dkpre = vec![0f32; n * kvdim];
        for r in 0..n {
            let pos = r % t;
            for h in 0..nh {
                let s = r * qdim + h * hd;
                ops::rope_bwd(&mut dqrot[s..s + self.rotary_dim], pos, &self.inv_freq);
                match q_norm {
                    Some(w) => ops::rmsnorm_bwd(
                        &qpre[s..s + hd],
                        w,
                        &qinv[r * nh + h..r * nh + h + 1],
                        &dqrot[s..s + hd],
                        self.gemma,
                        &mut dqpre[s..s + hd],
                        None,
                    ),
                    None => dqpre[s..s + hd].copy_from_slice(&dqrot[s..s + hd]),
                }
            }
            for g in 0..nkv {
                let s = r * kvdim + g * hd;
                ops::rope_bwd(&mut dkrot[s..s + self.rotary_dim], pos, &self.inv_freq);
                match k_norm {
                    Some(w) => ops::rmsnorm_bwd(
                        &kpre[s..s + hd],
                        w,
                        &kinv[r * nkv + g..r * nkv + g + 1],
                        &dkrot[s..s + hd],
                        self.gemma,
                        &mut dkpre[s..s + hd],
                        None,
                    ),
                    None => dkpre[s..s + hd].copy_from_slice(&dkrot[s..s + hd]),
                }
            }
        }

        // Re-interleave [dq; dgate] per head for gated projections.
        let dqraw: Vec<f32> = if *output_gate {
            let mut dq = vec![0f32; n * qrows];
            for r in 0..n {
                for h in 0..nh {
                    let dst = r * qrows + 2 * h * hd;
                    let src = r * qdim + h * hd;
                    dq[dst..dst + hd].copy_from_slice(&dqpre[src..src + hd]);
                    dq[dst + hd..dst + 2 * hd].copy_from_slice(&dgate[src..src + hd]);
                }
            }
            dq
        } else {
            dqpre
        };

        // Projections (frozen weights → dX only; bias add is identity).
        let mut dn1 = vec![0f32; n * hsz];
        ops::gemm_dx(&dqraw, wq, &mut dn1, n, hsz, qrows, pool);
        ops::gemm_dx(&dkpre, wk, &mut dn1, n, hsz, kvdim, pool);
        ops::gemm_dx(&dvproj, wv, &mut dn1, n, hsz, kvdim, pool);
        dn1
    }

    /// GDN through-backward: out_proj → pooled per-(seq, k-head) BPTT
    /// (fcd_ops::gdn_group_bwd) → conv backward → projections. Frozen
    /// weights: dX only.
    fn gdn_attn_bwd(
        &self,
        attn: &FcdAttn,
        acts: &AttnActs,
        dattn: &[f32],
        b: usize,
        t: usize,
    ) -> Vec<f32> {
        let FcdAttn::Gdn { wqkv, wz, wa, wb, conv, a_log, dt_bias, norm, wout } = attn else {
            unreachable!("gdn_attn_bwd on a non-GDN layer");
        };
        let AttnActs::Gdn { qkv, z, a, b: bstr } = acts else {
            unreachable!("acts mismatch");
        };
        let d = self.gdn.expect("gdn layer without gdn dims");
        let (hsz, n) = (self.hidden, b * t);
        let pool = self.pool.as_deref();
        let (c_dim, vd, nv) = (d.c_dim(), d.vd(), d.nv);

        let mut dof = vec![0f32; n * vd];
        ops::gemm_dx(dattn, wout, &mut dof, n, vd, hsz, pool);

        let cfg = ops::GdnSeqCfg {
            nv: d.nv,
            nk: d.nk,
            dk: d.dk,
            dv: d.dv,
            kk: d.kk,
            rms_eps: self.eps,
            conv,
            a_log,
            dt_bias,
            norm,
        };
        let qkv64: Vec<f64> = qkv.iter().map(|&v| v as f64).collect();
        let z64: Vec<f64> = z.iter().map(|&v| v as f64).collect();
        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let b64: Vec<f64> = bstr.iter().map(|&v| v as f64).collect();
        let dof64: Vec<f64> = dof.iter().map(|&v| v as f64).collect();
        let mut pre64 = vec![0f64; n * c_dim];
        let mut cq64 = vec![0f64; n * c_dim];
        for bi in 0..b {
            let r = bi * t * c_dim..(bi + 1) * t * c_dim;
            ops::gdn_conv_fwd(
                &qkv64[r.clone()],
                t,
                c_dim,
                d.kk,
                conv,
                &mut pre64[r.clone()],
                &mut cq64[r],
            );
        }

        let mut dcq64 = vec![0f64; n * c_dim];
        let mut dz64 = vec![0f64; n * vd];
        let mut da64 = vec![0f64; n * nv];
        let mut db64 = vec![0f64; n * nv];
        {
            let units = b * d.nk;
            let rep_v = d.nv / d.nk;
            let kd = d.nk * d.dk;
            let dcqp = SendMut(dcq64.as_mut_ptr());
            let dzp = SendMut(dz64.as_mut_ptr());
            let dap = SendMut(da64.as_mut_ptr());
            let dbp = SendMut(db64.as_mut_ptr());
            let (cqr, zr, ar, br, dor) = (&cq64, &z64, &a64, &b64, &dof64);
            let cfg_ref = &cfg;
            let run_unit = |u: usize| {
                let (bi, ko) = (u / d.nk, u % d.nk);
                // Full-width locals — the group only fills its own
                // channels; the scatter below copies exactly those.
                let mut dcq_l = vec![0f64; t * c_dim];
                let mut dz_l = vec![0f64; t * vd];
                let mut da_l = vec![0f64; t * nv];
                let mut db_l = vec![0f64; t * nv];
                ops::gdn_group_bwd(
                    &cqr[bi * t * c_dim..(bi + 1) * t * c_dim],
                    &zr[bi * t * vd..(bi + 1) * t * vd],
                    &ar[bi * t * nv..(bi + 1) * t * nv],
                    &br[bi * t * nv..(bi + 1) * t * nv],
                    t,
                    cfg_ref,
                    ko,
                    &dor[bi * t * vd..(bi + 1) * t * vd],
                    &mut dcq_l,
                    &mut dz_l,
                    &mut da_l,
                    &mut db_l,
                );
                // SAFETY of every store below: the written channel /
                // column ranges are exclusively owned by (bi, ko).
                for p in 0..t {
                    let row = (bi * t + p) * c_dim;
                    for c in ko * d.dk..(ko + 1) * d.dk {
                        unsafe {
                            *dcqp.at(row + c) = dcq_l[p * c_dim + c];
                            *dcqp.at(row + kd + c) = dcq_l[p * c_dim + kd + c];
                        }
                    }
                    for hh in 0..rep_v {
                        let h = ko * rep_v + hh;
                        for dj in 0..d.dv {
                            unsafe {
                                *dcqp.at(row + 2 * kd + h * d.dv + dj) =
                                    dcq_l[p * c_dim + 2 * kd + h * d.dv + dj];
                                *dzp.at((bi * t + p) * vd + h * d.dv + dj) =
                                    dz_l[p * vd + h * d.dv + dj];
                            }
                        }
                        unsafe {
                            *dap.at((bi * t + p) * nv + h) = da_l[p * nv + h];
                            *dbp.at((bi * t + p) * nv + h) = db_l[p * nv + h];
                        }
                    }
                }
            };
            match pool {
                Some(p) if units > 1 => p.run(&|widx, nw| {
                    for u in (widx..units).step_by(nw) {
                        run_unit(u);
                    }
                }),
                _ => {
                    for u in 0..units {
                        run_unit(u);
                    }
                }
            }
        }

        let mut dqkv64 = vec![0f64; n * c_dim];
        for bi in 0..b {
            let r = bi * t * c_dim..(bi + 1) * t * c_dim;
            ops::gdn_conv_bwd(
                &pre64[r.clone()],
                t,
                c_dim,
                d.kk,
                conv,
                &dcq64[r.clone()],
                &mut dqkv64[r],
            );
        }
        let to32 = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let (dqkv, dz, da, db) = (to32(&dqkv64), to32(&dz64), to32(&da64), to32(&db64));

        let mut dn1 = vec![0f32; n * hsz];
        ops::gemm_dx(&dqkv, wqkv, &mut dn1, n, hsz, c_dim, pool);
        ops::gemm_dx(&dz, wz, &mut dn1, n, hsz, vd, pool);
        ops::gemm_dx(&da, wa, &mut dn1, n, hsz, nv, pool);
        ops::gemm_dx(&db, wb, &mut dn1, n, hsz, nv, pool);
        dn1
    }

    /// Full forward: embeddings → layers → final hidden [b·t, hidden].
    /// `student` switches converted layers to the Nyström kernel and
    /// reads trainable weights from `ts`; `keep` collects each layer's
    /// input hidden for the checkpointed backward.
    fn forward_hidden(
        &self,
        ids: &[u32],
        b: usize,
        t: usize,
        ts: Option<&TrainState>,
        student: bool,
        mut keep: Option<&mut Vec<Vec<f32>>>,
    ) -> Vec<f32> {
        let hsz = self.hidden;
        let mut h = vec![0f32; b * t * hsz];
        for (r, &id) in ids.iter().enumerate() {
            let src = (id as usize).min(self.embed.len() / hsz - 1) * hsz;
            h[r * hsz..(r + 1) * hsz].copy_from_slice(&self.embed[src..src + hsz]);
        }
        for li in 0..self.nl {
            if let Some(k) = keep.as_deref_mut() {
                k.push(h.clone());
            }
            let wts = ln_ffn(self, if student { ts } else { None }, li);
            let nys = student && self.o1_flags[li];
            h = self.layer_forward(li, &h, b, t, &wts, nys, false).0;
        }
        h
    }

    /// Loss head: chunked tied-lm_head CE+KL against the teacher hidden,
    /// returning (ce_mean, kl_mean, dHidden_student).
    fn loss_and_dhidden(
        &self,
        hs: &[f32],
        ht: &[f32],
        targets: &[u32],
        kl_w: f64,
    ) -> (f64, f64, Vec<f32>) {
        let hsz = self.hidden;
        let n = targets.len();
        let pool = self.pool.as_deref();
        let wh = self.head_weight();
        let vs = self.vocab;

        let mut ns = vec![0f32; n * hsz];
        let mut invs = vec![0f32; n];
        ops::rmsnorm_fwd(hs, &self.final_norm, self.eps, self.gemma, &mut ns, &mut invs);
        let mut nt = vec![0f32; n * hsz];
        let mut invt = vec![0f32; n];
        ops::rmsnorm_fwd(ht, &self.final_norm, self.eps, self.gemma, &mut nt, &mut invt);

        let inv_n = 1.0 / n as f64;
        let mut ce_sum = 0f64;
        let mut kl_sum = 0f64;
        let mut dns = vec![0f32; n * hsz];
        let mut ls = vec![0f32; LM_CHUNK * vs];
        let mut lt = vec![0f32; LM_CHUNK * vs];
        let mut dlg = vec![0f32; LM_CHUNK * vs];
        let mut r0 = 0usize;
        while r0 < n {
            let r1 = (r0 + LM_CHUNK).min(n);
            let c = r1 - r0;
            ops::gemm_nt(&ns[r0 * hsz..r1 * hsz], wh, &mut ls[..c * vs], c, hsz, vs, pool);
            ops::gemm_nt(&nt[r0 * hsz..r1 * hsz], wh, &mut lt[..c * vs], c, hsz, vs, pool);
            for r in 0..c {
                let (ce, kl) = ops::ce_kl_position(
                    &ls[r * vs..(r + 1) * vs],
                    &lt[r * vs..(r + 1) * vs],
                    targets[r0 + r] as usize,
                    kl_w,
                    inv_n,
                    &mut dlg[r * vs..(r + 1) * vs],
                );
                ce_sum += ce;
                kl_sum += kl;
            }
            ops::gemm_dx(
                &dlg[..c * vs],
                wh,
                &mut dns[r0 * hsz..r1 * hsz],
                c,
                hsz,
                vs,
                pool,
            );
            r0 = r1;
        }

        let mut dhs = vec![0f32; n * hsz];
        ops::rmsnorm_bwd(hs, &self.final_norm, &invs, &dns, self.gemma, &mut dhs, None);
        (ce_sum * inv_n, kl_sum * inv_n, dhs)
    }

    /// Checkpointed backward: per layer, recompute the intra-layer
    /// activations and differentiate.
    fn backward(
        &self,
        b: usize,
        t: usize,
        keep: &[Vec<f32>],
        dh_last: Vec<f32>,
        ts: &mut TrainState,
    ) {
        // Split-borrow: the weight view reads `data`, the grads write
        // `grad` — disjoint fields of TrainState.
        let TrainState { layers, data, grad, .. } = ts;
        let mut dh = dh_last;
        for li in (0..self.nl).rev() {
            let h_in = &keep[li];
            let nys = self.o1_flags[li];
            let slot = layers.iter().position(|&x| x == li);
            let wts = match slot {
                Some(s) => {
                    let bi = s * PARAMS_PER_LAYER;
                    LnFfn {
                        iln: &data[bi],
                        pln: &data[bi + 1],
                        gate: &data[bi + 2],
                        up: &data[bi + 3],
                        down: &data[bi + 4],
                    }
                }
                None => {
                    let l = &self.layers[li];
                    LnFfn { iln: &l.iln, pln: &l.pln, gate: &l.gate, up: &l.up, down: &l.down }
                }
            };
            let (_, acts) = self.layer_forward(li, h_in, b, t, &wts, nys, true);
            let acts = acts.expect("want_acts");
            dh = match slot {
                Some(s) => {
                    let gb = s * PARAMS_PER_LAYER;
                    let gr = &mut grad[gb..gb + PARAMS_PER_LAYER];
                    self.layer_backward(li, h_in, b, t, &wts, nys, &acts, &dh, Some(gr))
                }
                None => self.layer_backward(li, h_in, b, t, &wts, nys, &acts, &dh, None),
            };
        }
    }

    /// Test-only: one full training-graph evaluation — teacher forward,
    /// student forward, CE+KL loss, checkpointed backward into the
    /// grads. Returns the weighted total loss. The block-level
    /// gradcheck runs finite differences over trainable weights through
    /// this, which exercises EVERY through-grad in the graph (layer-0
    /// gains flow through all attention/rope/qk-norm/GQA paths above).
    #[doc(hidden)]
    pub fn loss_and_grads_for_test(
        &self,
        ids: &[u32],
        tgt: &[u32],
        b: usize,
        t: usize,
        ts: &mut TrainState,
        kl_w: f64,
    ) -> f64 {
        let ht = self.forward_hidden(ids, b, t, None, false, None);
        let mut keep = Vec::with_capacity(self.nl);
        let hs = self.forward_hidden(ids, b, t, Some(ts), true, Some(&mut keep));
        let (ce, kl, dhs) = self.loss_and_dhidden(&hs, &ht, tgt, kl_w);
        ts.zero_grad();
        self.backward(b, t, &keep, dhs, ts);
        (1.0 - kl_w) * ce + kl_w * kl
    }

    /// Teacher-forced CE perplexity on deterministic evenly-spaced val
    /// windows (`heal_hybridk_06b.py::val_ppl` discipline — random
    /// windows made gate comparisons ride ±15% noise).
    pub fn val_ppl(
        &self,
        va: &[u32],
        ts: Option<&TrainState>,
        student: bool,
        bs: usize,
        nrounds: usize,
        seq: usize,
    ) -> f64 {
        let nwin = nrounds * bs;
        if va.len() < seq + 2 || nwin == 0 {
            return f64::NAN;
        }
        let stride = (va.len() - seq - 1) / nwin;
        let hsz = self.hidden;
        let wh = self.head_weight();
        let vs = self.vocab;
        let pool = self.pool.as_deref();
        let mut nll = 0f64;
        let mut cnt = 0usize;
        for j in 0..nrounds {
            let mut ids = Vec::with_capacity(bs * seq);
            let mut tgt = Vec::with_capacity(bs * seq);
            for bi in 0..bs {
                let off = ((j * bs + bi) * stride.max(1)).min(va.len() - seq - 1);
                ids.extend_from_slice(&va[off..off + seq]);
                tgt.extend_from_slice(&va[off + 1..off + seq + 1]);
            }
            let h = self.forward_hidden(&ids, bs, seq, ts, student, None);
            let n = bs * seq;
            let mut ns = vec![0f32; n * hsz];
            let mut inv = vec![0f32; n];
            ops::rmsnorm_fwd(&h, &self.final_norm, self.eps, self.gemma, &mut ns, &mut inv);
            let mut lg = vec![0f32; LM_CHUNK * vs];
            let mut r0 = 0usize;
            while r0 < n {
                let r1 = (r0 + LM_CHUNK).min(n);
                let c = r1 - r0;
                ops::gemm_nt(&ns[r0 * hsz..r1 * hsz], wh, &mut lg[..c * vs], c, hsz, vs, pool);
                for r in 0..c {
                    let row = &lg[r * vs..(r + 1) * vs];
                    let target = tgt[r0 + r] as usize;
                    let mut mx = f64::NEG_INFINITY;
                    for &v in row {
                        mx = mx.max(v as f64);
                    }
                    let mut s = 0f64;
                    for &v in row {
                        s += (v as f64 - mx).exp();
                    }
                    nll += mx + s.ln() - row[target.min(vs - 1)] as f64;
                    cnt += 1;
                }
                r0 = r1;
            }
        }
        (nll / cnt.max(1) as f64).exp()
    }
}

// ─────────────────────────── training loop ───────────────────────────

/// Run the full certified polish: train, early-stop/restore-best, and
/// write `<out>` (source tensors byte-copied, polished LN/FFN as f32).
///
/// With `gate` (Patent 16 draft, claim 13), every eval checkpoint is
/// additionally scored by greedy generation through the REAL streaming
/// O(1) runtime, and the restored checkpoint is the lowest-ppl one
/// AMONG GATE-PASSERS; if none passes, the zero-shot state is restored
/// (identity polish) — the stage never makes generation worse than
/// conversion alone.
pub fn run_polish(
    model: &Arc<CmfModel>,
    o1: &O1Cfg,
    hp: &FcdHyper,
    tr: &[u32],
    va: &[u32],
    out: &std::path::Path,
    gate: Option<&GenGateCfg>,
) -> Result<FcdReport, String> {
    if tr.len() < hp.seq + 2 {
        return Err(format!(
            "train corpus too small: {} tokens < seq+2 = {}",
            tr.len(),
            hp.seq + 2
        ));
    }
    let fm = FcdModel::from_cmf(model, o1)?;
    let converted = fm.converted();
    if converted.is_empty() {
        return Err("no converted layers under this --o1 spec (nothing to polish)".into());
    }
    tracing::info!(
        "fcd: {} layers converted ({} trainable tensors), m={} w={} sink={}, \
         corpus train {} / val {} tokens",
        converted.len(),
        converted.len() * PARAMS_PER_LAYER,
        fm.nys.m,
        fm.nys.w,
        fm.nys.sink,
        tr.len(),
        va.len()
    );

    let mut ts = TrainState::new(&fm);
    let teacher_ppl = fm.val_ppl(va, None, false, hp.bs, 2, hp.seq);
    let ppl_start = fm.val_ppl(va, Some(&ts), true, hp.bs, 2, hp.seq);
    tracing::info!(
        "fcd: quick-val teacher ppl {teacher_ppl:.2} | zero-shot o1 student ppl {ppl_start:.2}"
    );

    // ── generation gate (claim 13): baseline at step 0 ──
    let mut gate_state: Option<(Pipeline, Vec<f64>)> = match gate {
        Some(g) if !g.prompts.is_empty() => {
            let greedy = SamplerConfig {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                min_p: 0.0,
                seed: Some(0),
            };
            let mut pipe = Pipeline::from_model(model, greedy)
                .map_err(|e| format!("gen-gate pipeline: {e}"))?;
            pipe.set_o1(Some(o1.clone()));
            apply_trainables(&mut pipe, &fm, &ts);
            let base = gate_gen_scores(&mut pipe, g)?;
            tracing::info!("fcd gen-gate baseline loop-scores: {base:?}");
            Some((pipe, base))
        }
        Some(_) => {
            tracing::warn!("fcd gen-gate requested but val stream too short — gate off");
            None
        }
        None => None,
    };
    // Identity fallback: the pre-training master copies.
    let init_snapshot: Option<Vec<Vec<f32>>> =
        gate_state.is_some().then(|| ts.data.clone());
    let mut gate_evals: Vec<(usize, f64, Vec<f64>, bool)> = Vec::new();

    let mut rng = SplitMix64::new(hp.seed);
    let mut best: (f64, Option<Vec<Vec<f32>>>, usize) = (f64::INFINITY, None, 0);
    let mut losses: Vec<(f64, f64)> = Vec::with_capacity(hp.steps);
    let t0 = std::time::Instant::now();
    let n_per_step = hp.bs * hp.seq;
    for st in 1..=hp.steps {
        // Fresh random windows each step (the recipe; indices need not
        // match the torch RNG — the distribution does).
        let mut ids = Vec::with_capacity(n_per_step);
        let mut tgt = Vec::with_capacity(n_per_step);
        for _ in 0..hp.bs {
            let off = (rng.next_u64() as usize) % (tr.len() - hp.seq - 1);
            ids.extend_from_slice(&tr[off..off + hp.seq]);
            tgt.extend_from_slice(&tr[off + 1..off + hp.seq + 1]);
        }

        let ht = fm.forward_hidden(&ids, hp.bs, hp.seq, None, false, None);
        let mut keep: Vec<Vec<f32>> = Vec::with_capacity(fm.nl);
        let hs = fm.forward_hidden(&ids, hp.bs, hp.seq, Some(&ts), true, Some(&mut keep));
        let (ce, kl, dhs) = fm.loss_and_dhidden(&hs, &ht, &tgt, hp.kl_w);
        ts.zero_grad();
        fm.backward(hp.bs, hp.seq, &keep, dhs, &mut ts);
        let gn = ts.clip_and_step(hp.lr);
        losses.push((ce, kl));

        let el = t0.elapsed().as_secs_f64();
        tracing::info!(
            "fcd step {st}/{}: ce {ce:.3} kl {kl:.3} |g| {gn:.3} ({:.1}s/step)",
            hp.steps,
            el / st as f64
        );
        if hp.eval_every > 0 && st % hp.eval_every == 0 {
            let p = fm.val_ppl(va, Some(&ts), true, hp.bs, 2, hp.seq);
            match (&mut gate_state, gate) {
                (Some((pipe, base)), Some(g)) => {
                    apply_trainables(pipe, &fm, &ts);
                    let scores = gate_gen_scores(pipe, g)?;
                    let pass = gate_pass(&scores, base, g.threshold, g.baseline_slack);
                    let tag = if pass && p < best.0 {
                        best = (p, Some(ts.data.clone()), st);
                        " *best*"
                    } else {
                        ""
                    };
                    tracing::info!(
                        "fcd eval step {st}: val ppl {p:.2} | gen-gate {}                          (loop-scores {scores:?}){tag}",
                        if pass { "PASS" } else { "FAIL" }
                    );
                    gate_evals.push((st, p, scores, pass));
                }
                _ => {
                    let tag = if p < best.0 {
                        best = (p, Some(ts.data.clone()), st);
                        " *best*"
                    } else {
                        ""
                    };
                    tracing::info!("fcd eval step {st}: val ppl {p:.2}{tag}");
                }
            }
        }
    }

    // Early stop: restore the best checkpoint (certified: best was step
    // 150 of 300 in the torch run). Under the gate, `best` only ever
    // held GATE-PASSING checkpoints; none passing → identity restore.
    let mut gate_chosen: Option<usize> = None;
    if let Some(snap) = best.1.take() {
        ts.data = snap;
        gate_chosen = Some(best.2);
        tracing::info!(
            "fcd: restored best checkpoint from step {} (val ppl {:.2})",
            best.2,
            best.0
        );
    } else if let Some(init) = init_snapshot {
        ts.data = init;
        tracing::info!(
            "fcd: polish rejected by generation gate — identity artifact              (zero-shot state written; claim 13 floor)"
        );
    }
    let ppl_final = fm.val_ppl(va, Some(&ts), true, hp.bs, 6, hp.seq);
    let report = FcdReport {
        converted: converted.clone(),
        teacher_ppl,
        ppl_start,
        ppl_best: best.0.min(ppl_final),
        best_step: best.2,
        ppl_final,
        steps_run: hp.steps,
        sec_per_step: t0.elapsed().as_secs_f64() / hp.steps.max(1) as f64,
        losses,
        gate: gate_state.map(|(_, base)| GateReport {
            baseline: base,
            evals: gate_evals,
            chosen: gate_chosen,
        }),
    };
    save_polished(model, out, &fm, &ts, o1, hp, &report)?;
    Ok(report)
}

/// Hot-swap the trainable LN/FFN master copies into a runtime Pipeline
/// (frozen tensors stay mmap-backed — this reproduces the artifact the
/// polish would write, without writing it).
fn apply_trainables(pipe: &mut Pipeline, fm: &FcdModel, ts: &TrainState) {
    let hidden = fm.hidden;
    for (slot, &li) in ts.layers.iter().enumerate() {
        let b = slot * PARAMS_PER_LAYER;
        let inter = fm.layers[li].inter;
        let lw = &mut pipe.weights.layers[li];
        lw.input_norm = ts.data[b].clone();
        lw.post_norm = ts.data[b + 1].clone();
        lw.ffn = FfnKind::Dense(DenseFfn {
            gate_proj: QTensor::from_f32(ts.data[b + 2].clone(), inter, hidden),
            up_proj: QTensor::from_f32(ts.data[b + 3].clone(), inter, hidden),
            down_proj: QTensor::from_f32(ts.data[b + 4].clone(), hidden, inter),
        });
    }
}

/// Greedy loop-score probe through the streaming runtime.
fn gate_gen_scores(pipe: &mut Pipeline, g: &GenGateCfg) -> Result<Vec<f64>, String> {
    g.prompts
        .iter()
        .map(|p| {
            pipe.generate_from_ids(p, g.gen_tokens, None, None)
                .map(|r| loop_score(&r.token_ids))
        })
        .collect()
}

/// Write the polished container: every source tensor byte-copied except
/// the converted layers' LN/FFN, which become f32 (per-tensor dtypes
/// are first-class in the directory — no requant noise on fresh
/// weights). Adds `provenance.o1_attn` + `provenance.fcd`.
fn save_polished(
    model: &CmfModel,
    out: &std::path::Path,
    fm: &FcdModel,
    ts: &TrainState,
    o1: &O1Cfg,
    hp: &FcdHyper,
    report: &FcdReport,
) -> Result<(), String> {
    use cortiq_core::format::TensorSpec;
    let mut replace: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new(); // name → (slot, param idx)
    for (s, &li) in ts.layers.iter().enumerate() {
        let p = format!("model.layers.{li}.");
        for (k, suffix) in [
            (0usize, "input_layernorm.weight"),
            (1, "post_attention_layernorm.weight"),
            (2, "mlp.gate_proj.weight"),
            (3, "mlp.up_proj.weight"),
            (4, "mlp.down_proj.weight"),
        ] {
            replace.insert(format!("{p}{suffix}"), (s, k));
        }
    }
    let mut specs = Vec::with_capacity(model.tensors.len());
    for t in &model.tensors {
        if let Some(&(s, k)) = replace.get(&t.name) {
            let data = &ts.data[s * PARAMS_PER_LAYER + k];
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            specs.push(TensorSpec {
                name: t.name.clone(),
                dtype: TensorDtype::F32,
                shape: t.shape.clone(),
                data: bytes,
            });
        } else {
            specs.push(TensorSpec {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                data: model.entry_bytes(t).to_vec(),
            });
        }
    }

    let mut header = model.header.clone();
    let mut prov = match header.provenance.take() {
        Some(serde_json::Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    let layers_json = match &o1.layers {
        O1Layers::All => serde_json::json!("all"),
        O1Layers::Deep(n) => serde_json::json!(format!("deep{n}")),
        O1Layers::List(v) => serde_json::json!(v),
    };
    prov.insert(
        "o1_attn".into(),
        serde_json::json!({
            "layers": layers_json, "m": o1.m, "w": o1.w, "sink": o1.sink
        }),
    );
    prov.insert(
        "fcd".into(),
        serde_json::json!({
            "steps": hp.steps, "lr": hp.lr, "kl_w": hp.kl_w,
            "bs": hp.bs, "seq": hp.seq,
            "teacher_ppl": report.teacher_ppl,
            "ppl_start": report.ppl_start,
            "ppl_final": report.ppl_final,
            "best_step": report.best_step,
            "converted_layers": report.converted,
        }),
    );
    header.provenance = Some(serde_json::Value::Object(prov));
    let _ = fm; // geometry only used for validation today

    let masks = if model.masks.masks.is_empty() {
        None
    } else {
        Some(&model.masks)
    };
    CmfModel::write(out, &header, &specs, masks, model.vocab.as_deref())
        .map_err(|e| format!("writing polished cmf: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Claim-13 selection: lowest ppl AMONG PASSING, not global lowest.
    #[test]
    fn gate_selects_lowest_ppl_among_passing() {
        let base = vec![0.10, 0.00, 0.20];
        let evals = vec![
            (25usize, 21.0, vec![0.10, 0.05, 0.20]), // pass
            (50, 18.0, vec![0.40, 0.00, 0.10]),      // fail: 0.40 > threshold
            (75, 19.0, vec![0.15, 0.05, 0.25]),      // pass — best passing
            (100, 18.5, vec![0.20, 0.30, 0.20]),     // fail: 0.30 > base+0.10
        ];
        let sel = select_checkpoint(&evals, &base, 0.35, 0.10);
        assert_eq!(sel, Some(2), "step 75 is the lowest-ppl PASSING checkpoint");
    }

    /// All checkpoints fail → identity (None): the polish must never
    /// make generation worse than conversion alone.
    #[test]
    fn gate_all_fail_is_identity() {
        let base = vec![0.0, 0.0, 0.0];
        let evals = vec![
            (25usize, 15.0, vec![0.50, 0.0, 0.0]),
            (50, 14.0, vec![0.0, 0.36, 0.0]),
            (75, 13.0, vec![0.0, 0.0, 0.11]), // 0.11 > 0 + 0.10 slack
        ];
        assert_eq!(select_checkpoint(&evals, &base, 0.35, 0.10), None);
    }

    /// Boundary discipline: scores AT the threshold / AT base+slack pass
    /// ("exceeds" is strict); ties in ppl resolve to the earliest step.
    #[test]
    fn gate_boundaries_and_tie_break() {
        let base = vec![0.25];
        assert!(gate_pass(&[0.35], &base, 0.35, 0.10), "== threshold passes");
        assert!(gate_pass(&[0.35], &[0.25], 0.35, 0.10), "== base+slack passes");
        assert!(!gate_pass(&[0.351], &base, 0.35, 0.10));
        assert!(!gate_pass(&[0.30], &[0.10], 0.35, 0.10), "0.30 > 0.10+0.10");
        let evals = vec![
            (25usize, 20.0, vec![0.10]),
            (50, 20.0, vec![0.10]),
        ];
        assert_eq!(
            select_checkpoint(&evals, &base, 0.35, 0.10),
            Some(0),
            "equal ppl → earliest checkpoint"
        );
    }
}
