//! Native DTG-MA skill bake (Patent 2) — no Python, no torch.
//!
//! The certified recipe of `converter/make_skill_l1fcd.py`, in Rust on
//! the `FcdModel` f32 replica:
//!
//! - **Phase A** — a trainable L1 mask over FFN neurons (one logit per
//!   neuron, applied to the input of down_proj as σ(m)): pure LM loss
//!   on the task corpus + a progressive L1 penalty. Every 30 steps the
//!   binarized mask (σ>τ) is scored on held-out chunks; the best
//!   checkpoint — the *denoising bottom* — is restored at the end.
//!   Pruning noise neurons IMPROVES the model before it starts to hurt.
//! - **Phase B** — FCD: the FFN of the last N layers trains against the
//!   same LM loss with the hard mask active (cosine LR), held-out
//!   gated, best checkpoint restored.
//!
//! Attention (softmax and GDN alike) is FROZEN and carries no gradient
//! — exactly like the reference recipe (`torch.no_grad()` around the
//! attention branch): the backward walks the residual stream through
//! the FFN chain only, which is what makes a pure-Rust backward small.

use crate::fcd::{FcdModel, LnFfn};
use crate::fcd_ops as ops;
use crate::sampler::SplitMix64;
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Hyper-parameters — defaults are the certified recipe.
#[derive(Clone, Debug)]
pub struct BakeHyper {
    pub steps_a: usize,
    pub steps_b: usize,
    pub l1_init: f64,
    pub l1_step: f64,
    pub eval_every: usize,
    pub lr_a: f64,
    pub lr_b: f64,
    pub tau: f32,
    pub fcd_layers: usize,
    /// Independent fixed-length records per optimizer step.  The loss is
    /// normalized over all focused targets in the batch.
    pub batch: usize,
    /// Focused records per cached final-FFN optimizer step.  This is separate
    /// from `batch` because Phase A stores every layer's activations while a
    /// one-layer FCD cache stores only two hidden vectors per record.
    pub fcd_batch: usize,
    pub seed: u64,
    /// Target sparsity (0..1). When >0, the best checkpoint must have
    /// at least this fraction of neurons pruned; if none qualifies the
    /// highest-sparsity checkpoint is used.
    pub target_sparsity: f64,
    /// L1 aggression multiplier: scales both l1_init and l1_step.
    /// >1.0 = harder pruning push, <1.0 = softer.
    pub l1_mult: f64,
    /// Effective unlooped mask logit at step zero. The per-visit value is
    /// solved so that the product over loop visits equals sigmoid(init).
    /// 2.0 preserves the native recipe; 4.0 reproduces the older DTG-MA
    /// trading notebooks' near-identity start.
    pub mask_init: f32,
    /// Penalize softplus(logit), whose derivative is sigmoid(logit), instead
    /// of penalizing sigmoid(logit) itself. This reproduces the older DTG-MA
    /// recipe and avoids an extra (1-sigmoid) attenuation near an open gate.
    pub softplus_l1: bool,
    /// Select Phase-A checkpoints by held-out hard balanced accuracy when
    /// focused class tokens are configured. Otherwise held-out PPL remains
    /// the checkpoint metric.
    pub checkpoint_accuracy: bool,
    /// Optional strict lower bound for a focused checkpoint's raw accuracy.
    /// A checkpoint is eligible only when its measured accuracy is greater
    /// than this value. This lets callers impose a natural-distribution
    /// majority guard instead of selecting from a balanced holdout.
    pub checkpoint_min_accuracy: Option<f64>,
    /// Optional strict lower bound for focused balanced accuracy. Combined
    /// with `checkpoint_min_accuracy`, this prevents a majority-only mask
    /// from becoming the shipped specialist.
    pub checkpoint_min_balanced_accuracy: Option<f64>,
    /// When a joint accuracy guard is configured, rank eligible checkpoints
    /// by raw accuracy first, then balanced accuracy and PPL. The historical
    /// balanced-first selector remains the default for compatibility.
    pub checkpoint_raw_priority: bool,
    /// Round each layer's kept-neuron count UP to a multiple of this
    /// (0/1 = off). 32 keeps the defragged FFN on grouped codecs
    /// (in % 32 == 0) and SIMD kernels off their scalar tails.
    pub align: usize,
    /// Force one FFN width across all layers (the max aligned count) —
    /// the whole-token GPU graphs require a uniform intermediate size.
    pub uniform_inter: bool,
    /// When non-empty, LM loss is accumulated only where the next token
    /// is one of these ids. The whole chunk is still forwarded as context.
    /// This is useful for supervised corpora with a long input and a
    /// one-token answer, where ordinary all-token LM loss would drown the
    /// task signal in prompt reconstruction.
    pub focus_tokens: Vec<u32>,
    /// Optional token(s) that must immediately follow a focused target.
    /// Supervised ChatML uses the one-token label followed by `<|im_end|>`;
    /// this prevents label names mentioned inside the user instruction from
    /// being mistaken for answer positions.
    pub focus_follow_tokens: Vec<u32>,
}

impl Default for BakeHyper {
    fn default() -> Self {
        Self {
            steps_a: 240,
            steps_b: 120,
            l1_init: 0.01,
            l1_step: 0.005,
            eval_every: 30,
            lr_a: 0.1,
            lr_b: 1e-5,
            tau: 0.5,
            fcd_layers: 4,
            batch: 1,
            fcd_batch: 128,
            seed: 0,
            target_sparsity: 0.0,
            l1_mult: 1.0,
            mask_init: 2.0,
            softplus_l1: false,
            checkpoint_accuracy: false,
            checkpoint_min_accuracy: None,
            checkpoint_min_balanced_accuracy: None,
            checkpoint_raw_priority: false,
            align: 32,
            uniform_inter: false,
            focus_tokens: Vec::new(),
            focus_follow_tokens: Vec::new(),
        }
    }
}

/// What the bake measured and produced.
pub struct BakeReport {
    /// Held-out PPL of the untouched backbone.
    pub backbone: f64,
    /// Held-out PPL with the best hard mask (the denoising bottom).
    pub masked: f64,
    /// Held-out PPL after FCD (the final specialist).
    pub overlaid: f64,
    pub pruned_ratio: f64,
    pub kept_per_layer: Vec<usize>,
    /// Hard focused-label accuracy, present when focus tokens were supplied.
    pub backbone_accuracy: Option<f64>,
    pub masked_accuracy: Option<f64>,
    pub overlaid_accuracy: Option<f64>,
    /// Macro recall over focused labels. Unlike raw accuracy this cannot be
    /// improved by collapsing to the majority UP/DOWN class.
    pub backbone_balanced_accuracy: Option<f64>,
    pub masked_balanced_accuracy: Option<f64>,
    pub overlaid_balanced_accuracy: Option<f64>,
    pub selected_step: usize,
    pub sec: f64,
}

pub struct BakeCheckpoint {
    pub step: usize,
    pub l1: f64,
    pub ppl: f64,
    pub sparsity: f64,
    pub accuracy: Option<f64>,
    pub balanced_accuracy: Option<f64>,
}

/// The trained artifacts: everything the defrag writer needs, f32.
pub struct BakeArtifacts {
    /// Per-PHYSICAL-layer live flags: the union over visits — a weight
    /// row is removable from disk only when no visit keeps it.
    pub keep: Vec<Vec<bool>>,
    /// Per-VIRTUAL-layer live flags (physical × loops, pass-major): the
    /// mask the file ships and the runtime applies per visit.
    pub keep_visits: Vec<Vec<bool>>,
    /// Per-layer down_proj `[hidden, inter]` with dead columns zeroed
    /// (FCD layers: the trained weights; others: the backbone's).
    pub down: Vec<Vec<f32>>,
    /// Trained gate/up for the FCD layers (`None` elsewhere).
    pub gate_up: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    /// Which layers went through Phase B.
    pub fcd_layers: Vec<usize>,
    /// The trained mask logits, per virtual layer — a CONTINUOUS
    /// per-neuron importance the hard keep flags throw away. The tube
    /// planner ranks and orders neurons by these, not by raw
    /// activation mass.
    pub logits: Vec<Vec<f32>>,
    /// Phase-A logits after the last requested optimization step, before
    /// restoring the selected hard-validation checkpoint.
    pub final_logits: Vec<Vec<f32>>,
    pub checkpoints: Vec<BakeCheckpoint>,
}

const CLIP: f64 = 1.0;
const B1: f64 = 0.9;
const B2: f64 = 0.999;
const EPS: f64 = 1e-8;

/// Plain Adam over a set of f32 tensors (masks are tiny, FFN mid-size).
struct Adam {
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    t: i32,
    lr: f64,
}

impl Adam {
    fn new(sizes: &[usize], lr: f64) -> Self {
        Self {
            m: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            v: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            t: 0,
            lr,
        }
    }

    /// Global-norm clip + Adam step. `params[i].len() == grads[i].len()`.
    fn step(&mut self, params: &mut [&mut [f32]], grads: &[Vec<f64>], lr_scale: f64) {
        let gn: f64 = grads
            .iter()
            .flat_map(|g| g.iter().map(|x| x * x))
            .sum::<f64>()
            .sqrt();
        let clip = if gn > CLIP { CLIP / gn } else { 1.0 };
        self.t += 1;
        let (bc1, bc2) = (1.0 - B1.powi(self.t), 1.0 - B2.powi(self.t));
        for (pi, p) in params.iter_mut().enumerate() {
            for j in 0..p.len() {
                let g = grads[pi][j] * clip;
                let m = &mut self.m[pi][j];
                let v = &mut self.v[pi][j];
                *m = B1 * *m + (1.0 - B1) * g;
                *v = B2 * *v + (1.0 - B2) * g * g;
                let upd = (*m / bc1) / ((*v / bc2).sqrt() + EPS);
                p[j] -= (self.lr * lr_scale * upd) as f32;
            }
        }
    }
}

/// Mask logit at step zero, solved for the loop depth.
///
/// The gate multiplies the FFN once per VISIT, so a Looped Transformer
/// applies it `loops` times per token and the factor compounds. What
/// must be held constant across depths is the EFFECTIVE start — the
/// product the stack actually sees — at the value the recipe was
/// validated with on ordinary models, σ(2.0) = 0.881:
///
/// ```text
/// σ(m0)^loops = σ(2.0)   →   m0 = logit( σ(2.0)^(1/loops) )
/// ```
///
/// `loops = 1` returns 2.0 exactly, so nothing regresses. Two known
/// wrong answers this replaces: the old hardcoded 2.0, which at two
/// visits compounds to 0.776 and took Nanbeige 4.2 from a baseline of
/// 4.187 to 278.4 at step 30; and a start pushed to identity, which
/// cannot learn because the update carries σ'(m) = σ(1−σ), worth 5e-4
/// at σ = 0.9995 against 0.105 at 2.0.
pub fn mask_init_logit_for(loops: usize, effective_logit: f32) -> f32 {
    let base = 1.0f32 / (1.0 + (-effective_logit).exp());
    let per_visit = base.powf(1.0 / loops.max(1) as f32);
    (per_visit / (1.0 - per_visit)).ln()
}

pub fn mask_init_logit(loops: usize) -> f32 {
    mask_init_logit_for(loops, 2.0)
}

/// Learning-rate scale for the mask step, given the loop depth.
///
/// The backward accumulates every visit of a physical layer into the
/// same mask gradient, so an unscaled step is `loops` times the tuned
/// one. One step should mean one token's worth of movement at any depth.
pub fn mask_step_scale(loops: usize) -> f64 {
    1.0 / loops.max(1) as f64
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sparsity_grad(logit: f32, softplus_l1: bool) -> f64 {
    let s = sigmoid(logit) as f64;
    if softplus_l1 { s } else { s * (1.0 - s) }
}

fn is_scored_target(
    ids: &[u32],
    target_index: usize,
    sequence_end: usize,
    focus: &[u32],
    follow: &[u32],
) -> bool {
    if focus.is_empty() {
        return true;
    }
    focus.contains(&ids[target_index])
        && (follow.is_empty()
            || (target_index + 1 < sequence_end && follow.contains(&ids[target_index + 1])))
}

/// One forward + CE(+optionally backward through the FFN chain).
/// Returns (nll_sum, tokens). `dmask`/`dffn` accumulate when given.
struct Pass<'a> {
    fm: &'a FcdModel,
    tau: f32,
    /// σ(m) per layer when soft; binarized when `hard`.
    logits: &'a [Vec<f32>],
    hard: bool,
    /// Phase-B replacement FFN weights per layer (trained copies).
    ffn: &'a [Option<(Vec<f32>, Vec<f32>, Vec<f32>)>],
    /// Empty means ordinary all-token LM loss.
    focus_tokens: &'a [u32],
    /// Empty means no right-context constraint on focused targets.
    focus_follow_tokens: &'a [u32],
}

#[derive(Clone, Debug, Default)]
struct FocusStats {
    total: usize,
    correct: usize,
    class_total: Vec<usize>,
    class_correct: Vec<usize>,
}

impl FocusStats {
    fn new(classes: usize) -> Self {
        Self {
            class_total: vec![0; classes],
            class_correct: vec![0; classes],
            ..Self::default()
        }
    }

    fn accuracy(&self) -> Option<f64> {
        (self.total > 0).then(|| self.correct as f64 / self.total as f64)
    }

    fn balanced_accuracy(&self) -> Option<f64> {
        let recalls: Vec<f64> = self
            .class_total
            .iter()
            .zip(&self.class_correct)
            .filter_map(|(&n, &ok)| (n > 0).then(|| ok as f64 / n as f64))
            .collect();
        (!recalls.is_empty()).then(|| recalls.iter().sum::<f64>() / recalls.len() as f64)
    }

    fn merge(&mut self, other: &Self) {
        self.total += other.total;
        self.correct += other.correct;
        if self.class_total.len() < other.class_total.len() {
            self.class_total.resize(other.class_total.len(), 0);
            self.class_correct.resize(other.class_correct.len(), 0);
        }
        for (dst, src) in self.class_total.iter_mut().zip(&other.class_total) {
            *dst += src;
        }
        for (dst, src) in self.class_correct.iter_mut().zip(&other.class_correct) {
            *dst += src;
        }
    }
}

#[derive(Clone, Debug)]
struct HeldScore {
    ppl: f64,
    accuracy: Option<f64>,
    balanced_accuracy: Option<f64>,
}

/// Frozen boundary immediately before one trainable final FFN.  This is not
/// an adapter or a donor checkpoint: both vectors are extracted directly from
/// the opened CMF, kept in RAM, and discarded when the native bake ends.
#[derive(Default)]
struct FocusedFcdCache {
    h1: Vec<f32>,
    n2: Vec<f32>,
    targets: Vec<usize>,
}

impl FocusedFcdCache {
    fn len(&self) -> usize {
        self.targets.len()
    }

    fn append(&mut self, mut other: Self) {
        self.h1.append(&mut other.h1);
        self.n2.append(&mut other.n2);
        self.targets.append(&mut other.targets);
    }
}

fn gather_rows(values: &[f32], rows: &[usize], width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * width);
    for &row in rows {
        out.extend_from_slice(&values[row * width..(row + 1) * width]);
    }
    out
}

impl Pass<'_> {
    fn gates(&self, li: usize) -> Vec<f32> {
        self.logits[li]
            .iter()
            .map(|&l| {
                let s = sigmoid(l);
                if self.hard {
                    if s > self.tau { 1.0 } else { 0.0 }
                } else {
                    s
                }
            })
            .collect()
    }

    fn wts<'b>(&'b self, li: usize, mats: &'b crate::fcd::LayerMats) -> LnFfn<'b> {
        let l = &self.fm.layers[li];
        match &self.ffn[li] {
            Some((g, u, d)) => LnFfn {
                iln: &l.iln,
                pln: &l.pln,
                gate: g,
                up: u,
                down: d,
                // Trained copies move every Adam step — no prebuilt concat.
                gu: None,
            },
            None => LnFfn {
                iln: &l.iln,
                pln: &l.pln,
                gate: &[],
                up: &[],
                down: &mats.down,
                gu: Some(&mats.gu),
            },
        }
    }

    /// Extract the exact frozen boundary of the final FFN for focused answer
    /// positions.  Only the ordinary one-pass/final-layer case is cacheable:
    /// looped stacks revisit the same FFN after its own changed output and
    /// correctly fall back to the full Phase-B path.
    fn cache_final_ffn_batch(&self, ids: &[u32], batch: usize) -> Result<FocusedFcdCache, String> {
        let fm = self.fm;
        if fm.loops.max(1) != 1 || fm.layers.is_empty() || self.focus_tokens.is_empty() {
            return Err("focused final-FFN cache needs a one-pass stack and focus tokens".into());
        }
        if ids.len() % batch.max(1) != 0 {
            return Err("focused final-FFN cache received a ragged batch".into());
        }
        let t = ids.len() / batch.max(1);
        let hsz = fm.hidden;
        let last = fm.layers.len() - 1;
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        for bi in 0..batch {
            let base = bi * t;
            for target_index in base + 1..base + t {
                if !is_scored_target(
                    ids,
                    target_index,
                    base + t,
                    self.focus_tokens,
                    self.focus_follow_tokens,
                ) {
                    continue;
                }
                sources.push(target_index - 1);
                targets.push(
                    self.focus_tokens
                        .iter()
                        .position(|&id| id == ids[target_index])
                        .expect("focused target belongs to focus_tokens"),
                );
            }
        }
        if sources.is_empty() {
            return Ok(FocusedFcdCache::default());
        }

        let mut hidden = vec![0f32; ids.len() * hsz];
        for (row, &id) in ids.iter().enumerate() {
            hidden[row * hsz..(row + 1) * hsz]
                .copy_from_slice(&fm.embed[id as usize * hsz..(id as usize + 1) * hsz]);
        }
        for layer in 0..=last {
            let gate = self.gates(layer);
            let mats = fm.mats(layer)?;
            let weights = self.wts(layer, &mats);
            let (next, acts) = fm.layer_forward_scaled(
                layer,
                &hidden,
                batch,
                t,
                &weights,
                false,
                layer == last,
                Some(&gate),
            );
            if layer == last {
                let acts = acts.expect("last FFN boundary requested");
                return Ok(FocusedFcdCache {
                    h1: gather_rows(&acts.h1, &sources, hsz),
                    n2: gather_rows(&acts.n2, &sources, hsz),
                    targets,
                });
            }
            hidden = next;
        }
        unreachable!("non-empty stack has a final layer")
    }

    /// Teacher-forced NLL over one chunk; when `grad` is set, backprop
    /// through the FFN chain into the mask grads (and FFN grads for
    /// Phase-B layers).
    #[allow(clippy::too_many_arguments)]
    fn chunk(
        &self,
        ids: &[u32],
        grad: Option<(
            &mut [Vec<f64>],
            &mut [Option<(Vec<f64>, Vec<f64>, Vec<f64>)>],
        )>,
    ) -> (f64, usize) {
        self.chunk_batch(ids, 1, grad)
    }

    /// `chunk` over `b` equal-length sequences flattened into `ids`.
    ///
    /// Everything under this level was batch-aware all along
    /// (`layer_forward_scaled` and every attention fwd/bwd take `b`);
    /// only this wrapper hardcoded 1. Evaluation is where it pays: the
    /// held set is 12 chunks scored one at a time, which on a 4 B model
    /// meant 12× the GEMM submits for the same arithmetic.
    fn chunk_batch(
        &self,
        ids: &[u32],
        b: usize,
        grad: Option<(
            &mut [Vec<f64>],
            &mut [Option<(Vec<f64>, Vec<f64>, Vec<f64>)>],
        )>,
    ) -> (f64, usize) {
        self.chunk_batch_scored(ids, b, grad, None)
    }

    fn chunk_batch_scored(
        &self,
        ids: &[u32],
        b: usize,
        grad: Option<(
            &mut [Vec<f64>],
            &mut [Option<(Vec<f64>, Vec<f64>, Vec<f64>)>],
        )>,
        mut focus_stats: Option<&mut FocusStats>,
    ) -> (f64, usize) {
        let fm = self.fm;
        let hsz = fm.hidden;
        debug_assert!(ids.len() % b.max(1) == 0, "ragged batch");
        let t = ids.len() / b.max(1);
        let n = b * t;
        let nl = fm.layers.len();
        // Embed.
        let mut h = vec![0f32; n * hsz];
        for (r, &id) in ids.iter().enumerate() {
            h[r * hsz..(r + 1) * hsz]
                .copy_from_slice(&fm.embed[id as usize * hsz..(id as usize + 1) * hsz]);
        }
        // Forward over VIRTUAL layers: a Looped Transformer runs the
        // stack `fm.loops` times, with a final_norm at each loop
        // boundary when the file says so. Everything below indexes
        // activations by the virtual step and weights/grads by the
        // physical layer `vl % nl` — so a physical layer visited twice
        // accumulates both visits' gradients, which is what the loop
        // means mathematically.
        let loops = fm.loops.max(1);
        let vn = nl * loops;
        let mut h_ins = Vec::with_capacity(vn);
        let mut acts = Vec::with_capacity(vn);
        let mut masks = Vec::with_capacity(vn);
        // Loop-boundary norms, saved for the backward: (input, inv).
        let mut lnorms: Vec<Option<(Vec<f32>, Vec<f32>)>> = vec![None; vn];
        for vl in 0..vn {
            let li = vl % nl;
            // The gate is PER VISIT: the two passes of a loop are
            // different computations sharing one set of weights, so the
            // mask must be allowed to differ between them. Weights stay
            // indexed by the physical layer.
            let g = self.gates(vl);
            let mats_hold = fm.mats(li).expect("layer mats");
            let wts = self.wts(li, &mats_hold);
            let want = grad.is_some();
            let (h2, a) = fm.layer_forward_scaled(li, &h, b, t, &wts, false, want, Some(&g));
            h_ins.push(if want { h } else { Vec::new() });
            acts.push(a);
            masks.push(g);
            h = h2;
            // Mid-stack norm at every loop boundary except the last —
            // the final one folds into the head below.
            if fm.loop_norm && li + 1 == nl && vl + 1 < vn {
                let mut hn = vec![0f32; n * hsz];
                let mut inv = vec![0f32; n];
                ops::rmsnorm_fwd(&h, &fm.final_norm, fm.eps, fm.gemma, &mut hn, &mut inv);
                if want {
                    lnorms[vl] = Some((h, inv));
                }
                h = hn;
            }
        }
        // Final norm + tied LM head, CE summed over positions 1..t.
        let mut hn = vec![0f32; n * hsz];
        let mut inv = vec![0f32; n];
        ops::rmsnorm_fwd(&h, &fm.final_norm, fm.eps, fm.gemma, &mut hn, &mut inv);
        let lm: &[f32] = fm.lm_head.as_deref().unwrap_or(&fm.embed);
        let vocab = lm.len() / hsz;
        let pool = fm.pool.as_deref();
        let mut nll = 0f64;
        let mut dh_n = vec![0f32; n * hsz]; // dL/d hn
        // Chunk the vocab matmul over positions to bound the logits buf.
        // Positions are walked PER SEQUENCE: the last position of chunk
        // i must not be scored against the first token of chunk i+1.
        const POS_CHUNK: usize = 64;
        let scored = (0..b)
            .map(|bi| {
                let base = bi * t;
                (base + 1..base + t)
                    .filter(|&target_index| {
                        is_scored_target(
                            ids,
                            target_index,
                            base + t,
                            self.focus_tokens,
                            self.focus_follow_tokens,
                        )
                    })
                    .count()
            })
            .sum::<usize>();
        if scored == 0 {
            return (0.0, 0);
        }
        if self.focus_tokens.is_empty() {
            // Ordinary language-model mode still needs the complete
            // vocabulary distribution at every target position.
            for bi in 0..b {
                let base = bi * t;
                let mut p0 = 0usize;
                while p0 < t - 1 {
                    let pc = POS_CHUNK.min(t - 1 - p0);
                    let mut logits = vec![0f32; pc * vocab];
                    ops::gemm_nt(
                        &hn[(base + p0) * hsz..(base + p0 + pc) * hsz],
                        lm,
                        &mut logits,
                        pc,
                        hsz,
                        vocab,
                        pool,
                    );
                    for r in 0..pc {
                        let target_index = base + p0 + r + 1;
                        let target = ids[target_index] as usize;
                        let row = &mut logits[r * vocab..(r + 1) * vocab];
                        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                        let mut sum = 0f64;
                        for v in row.iter() {
                            sum += ((*v as f64) - mx).exp();
                        }
                        nll += mx + sum.ln() - row[target] as f64;
                        if grad.is_some() {
                            // dCE/dlogit = softmax − onehot, scaled by 1/scored.
                            let inv_n = 1.0 / scored as f64;
                            for v in row.iter_mut() {
                                *v = ((((*v as f64) - mx).exp() / sum) * inv_n) as f32;
                            }
                            row[target] -= inv_n as f32;
                        }
                    }
                    if grad.is_some() {
                        ops::gemm_dx(
                            &logits,
                            lm,
                            &mut dh_n[(base + p0) * hsz..(base + p0 + pc) * hsz],
                            pc,
                            hsz,
                            vocab,
                            pool,
                        );
                    }
                    p0 += pc;
                }
            }
        } else {
            // Exact classifier mode. Only the declared label rows can enter
            // either the normalizer or its gradient, so multiplying every
            // hidden by the full ~250k-row LM head is pure wasted work. It
            // made a two-label archaeology run spend more time in the head
            // than in mask Adam and limited experiments to a tiny fraction
            // of the updates used by the original notebooks.
            let inv_n = 1.0 / scored as f64;
            for bi in 0..b {
                let base = bi * t;
                for target_index in base + 1..base + t {
                    if !is_scored_target(
                        ids,
                        target_index,
                        base + t,
                        self.focus_tokens,
                        self.focus_follow_tokens,
                    ) {
                        continue;
                    }
                    let target_class = self
                        .focus_tokens
                        .iter()
                        .position(|&id| id == ids[target_index])
                        .expect("focused target belongs to focus_tokens");
                    let source = target_index - 1;
                    let hidden = &hn[source * hsz..(source + 1) * hsz];
                    let class_logits: Vec<f32> = self
                        .focus_tokens
                        .iter()
                        .map(|&id| {
                            let row = &lm[id as usize * hsz..(id as usize + 1) * hsz];
                            hidden.iter().zip(row).map(|(&x, &w)| x * w).sum()
                        })
                        .collect();
                    let mx = class_logits
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max) as f64;
                    let probs: Vec<f64> = class_logits
                        .iter()
                        .map(|&value| ((value as f64) - mx).exp())
                        .collect();
                    let sum: f64 = probs.iter().sum();
                    nll += mx + sum.ln() - class_logits[target_class] as f64;
                    if let Some(stats) = focus_stats.as_deref_mut() {
                        let predicted_class = class_logits
                            .iter()
                            .enumerate()
                            .max_by(|(_, left), (_, right)| left.total_cmp(right))
                            .map(|(index, _)| index)
                            .expect("focus_tokens is non-empty");
                        stats.total += 1;
                        stats.class_total[target_class] += 1;
                        if predicted_class == target_class {
                            stats.correct += 1;
                            stats.class_correct[target_class] += 1;
                        }
                    }
                    if grad.is_some() {
                        let dh = &mut dh_n[source * hsz..(source + 1) * hsz];
                        for (class, (&id, probability)) in
                            self.focus_tokens.iter().zip(probs).enumerate()
                        {
                            let coefficient = (probability / sum
                                - usize::from(class == target_class) as f64)
                                * inv_n;
                            let row = &lm[id as usize * hsz..(id as usize + 1) * hsz];
                            for (value, &weight) in dh.iter_mut().zip(row) {
                                *value += (coefficient * weight as f64) as f32;
                            }
                        }
                    }
                }
            }
        }
        let Some((dmask, dffn)) = grad else {
            return (nll, scored);
        };
        // Backward: final norm, then the FFN chain layer by layer.
        let t_bwd = std::time::Instant::now();
        let mut dh = vec![0f32; n * hsz];
        ops::rmsnorm_bwd(&h, &fm.final_norm, &inv, &dh_n, fm.gemma, &mut dh, None);
        for vl in (0..vn).rev() {
            let li = vl % nl;
            // Undo the loop-boundary norm this step fed into.
            if let Some((hb, inv)) = lnorms[vl].as_ref() {
                let mut dprev = vec![0f32; n * hsz];
                ops::rmsnorm_bwd(hb, &fm.final_norm, inv, &dh, fm.gemma, &mut dprev, None);
                dh = dprev;
            }
            let a = acts[vl].as_ref().expect("acts saved in grad mode");
            let g = &masks[vl];
            let inter = fm.layers[li].inter;
            let mats_hold = fm.mats(li).expect("layer mats");
            let wts = self.wts(li, &mats_hold);
            // h2 = h1 + act2 @ downᵀ  →  dact2 = dh @ down.
            let mut dact2 = vec![0f32; n * inter];
            ops::gemm_dx(&dh, wts.down, &mut dact2, n, inter, hsz, fm.pool.as_deref());
            if let Some((_, _, dd)) = dffn[li].as_mut() {
                // dW_down += dhᵀ · act2 (act2 = act·g).
                let mut act2 = a.act.clone();
                for r in 0..n {
                    for (x, &gv) in act2[r * inter..(r + 1) * inter].iter_mut().zip(g) {
                        *x *= gv;
                    }
                }
                let mut dw = vec![0f32; hsz * inter];
                ops::gemm_dw(&dh, &act2, &mut dw, n, inter, hsz, fm.pool.as_deref());
                for (o, &x) in dd.iter_mut().zip(&dw) {
                    *o += x as f64;
                }
            }
            // Mask grad: dm = Σ_t dact2·act · σ'(m)  (soft; STE-equal).
            // Indexed by the VIRTUAL layer: each visit's mask row gets
            // exactly its own visit's gradient, no cross-visit sum.
            {
                let dm = &mut dmask[vl];
                for r in 0..n {
                    let da = &dact2[r * inter..(r + 1) * inter];
                    let aa = &a.act[r * inter..(r + 1) * inter];
                    for j in 0..inter {
                        dm[j] += da[j] as f64 * aa[j] as f64;
                    }
                }
                // σ'(m) folded in once per chunk (constant per neuron).
                for (j, d) in dm.iter_mut().enumerate() {
                    let _ = j;
                    let _ = d;
                }
            }
            // dact = dact2 · g;  silu·mul backward.
            let mut dg_pre = vec![0f32; n * inter];
            let mut du_pre = vec![0f32; n * inter];
            for r in 0..n {
                for j in 0..inter {
                    let i = r * inter + j;
                    let da = dact2[i] * g[j];
                    let sg = ops::silu(a.gpre[i]);
                    dg_pre[i] = da * a.upre[i] * ops::silu_bwd(a.gpre[i]);
                    du_pre[i] = da * sg;
                }
            }
            // dn2 = dg_pre @ gate + du_pre @ up — one fused submit when
            // the frozen concat exists; the trained-copy path keeps two.
            let mut dn2 = vec![0f32; n * hsz];
            if let Some(gu) = wts.gu {
                let mut dgu = vec![0f32; n * 2 * inter];
                for r in 0..n {
                    let row = &mut dgu[r * 2 * inter..(r + 1) * 2 * inter];
                    row[..inter].copy_from_slice(&dg_pre[r * inter..(r + 1) * inter]);
                    row[inter..].copy_from_slice(&du_pre[r * inter..(r + 1) * inter]);
                }
                ops::gemm_dx(&dgu, gu, &mut dn2, n, hsz, 2 * inter, fm.pool.as_deref());
            } else {
                ops::gemm_dx(
                    &dg_pre,
                    wts.gate,
                    &mut dn2,
                    n,
                    hsz,
                    inter,
                    fm.pool.as_deref(),
                );
                let mut dn2b = vec![0f32; n * hsz];
                ops::gemm_dx(
                    &du_pre,
                    wts.up,
                    &mut dn2b,
                    n,
                    hsz,
                    inter,
                    fm.pool.as_deref(),
                );
                for (x, &y) in dn2.iter_mut().zip(&dn2b) {
                    *x += y;
                }
            }
            if let Some((dgw, duw, _)) = dffn[li].as_mut() {
                let mut dw = vec![0f32; inter * hsz];
                ops::gemm_dw(&dg_pre, &a.n2, &mut dw, n, hsz, inter, fm.pool.as_deref());
                for (o, &x) in dgw.iter_mut().zip(&dw) {
                    *o += x as f64;
                }
                dw.fill(0.0);
                ops::gemm_dw(&du_pre, &a.n2, &mut dw, n, hsz, inter, fm.pool.as_deref());
                for (o, &x) in duw.iter_mut().zip(&dw) {
                    *o += x as f64;
                }
            }
            // Post-norm backward into h1; the attention branch carries
            // no gradient (frozen), so dh1 flows straight to dh_in.
            let mut dh1 = dh.clone(); // residual h2 = h1 + ffn
            ops::rmsnorm_bwd(&a.h1, wts.pln, &a.inv2, &dn2, fm.gemma, &mut dh1, None);
            dh = dh1;
            let _ = &h_ins[vl];
        }
        crate::fcd::prof::add(&crate::fcd::prof::BWD, t_bwd);
        (nll, scored)
    }
}

/// Held-out PPL with the hard mask (and Phase-B weights when present).
fn held_ppl(pass: &Pass, held: &[Vec<u32>]) -> f64 {
    held_score(pass, held).ppl
}

fn held_score(pass: &Pass, held: &[Vec<u32>]) -> HeldScore {
    // Bounded batched passes over the held set: equal-length records share
    // one GEMM per weight, while the cap keeps activation/GPU scratch memory
    // predictable for a full validation sweep. Aggregation is exact because
    // NLL and focused-class counts are additive across groups.
    if held.is_empty() {
        return HeldScore {
            ppl: f64::NAN,
            accuracy: None,
            balanced_accuracy: None,
        };
    }
    let mut nll = 0f64;
    let mut n = 0usize;
    let mut stats = FocusStats::new(pass.focus_tokens.len());
    const GROUP: usize = 32;
    for group in held.chunks(GROUP) {
        let t = group[0].len();
        if group.iter().all(|c| c.len() == t) {
            let flat: Vec<u32> = group.iter().flatten().copied().collect();
            let mut part = FocusStats::new(pass.focus_tokens.len());
            let (l, k) = pass.chunk_batch_scored(&flat, group.len(), None, Some(&mut part));
            nll += l;
            n += k;
            stats.merge(&part);
        } else {
            for c in group {
                let mut part = FocusStats::new(pass.focus_tokens.len());
                let (l, k) = pass.chunk_batch_scored(c, 1, None, Some(&mut part));
                nll += l;
                n += k;
                stats.merge(&part);
            }
        }
    }
    HeldScore {
        ppl: (nll / n.max(1) as f64).exp(),
        accuracy: stats.accuracy(),
        balanced_accuracy: stats.balanced_accuracy(),
    }
}

fn calibration_batch(
    calib: &[Vec<u32>],
    step: usize,
    requested: usize,
) -> Result<(Vec<u32>, usize), String> {
    let batch = requested.max(1).min(calib.len());
    let width = calib[0].len();
    let mut flat = Vec::with_capacity(batch * width);
    for offset in 0..batch {
        let record = &calib[(step * batch + offset) % calib.len()];
        if record.len() != width {
            return Err(format!(
                "skill bake: --batch needs equal-length records ({} != {width})",
                record.len()
            ));
        }
        flat.extend_from_slice(record);
    }
    Ok((flat, batch))
}

fn build_focused_fcd_cache(
    pass: &Pass<'_>,
    records: &[Vec<u32>],
    extraction_batch: usize,
) -> Result<FocusedFcdCache, String> {
    let mut cache = FocusedFcdCache::default();
    for group in records.chunks(extraction_batch.max(1)) {
        let width = group[0].len();
        if group.iter().any(|record| record.len() != width) {
            return Err("focused final-FFN cache needs equal-length records".into());
        }
        let flat: Vec<u32> = group.iter().flatten().copied().collect();
        cache.append(pass.cache_final_ffn_batch(&flat, group.len())?);
    }
    Ok(cache)
}

/// Exact final-FFN focused loss, optionally with gradients for the three full
/// FCD projections.  The upstream representation came from the same CMF and
/// mask in `build_focused_fcd_cache`; no external checkpoint format is used.
fn cached_fcd_run(
    fm: &FcdModel,
    cache: &FocusedFcdCache,
    indices: &[usize],
    weights: (&[f32], &[f32], &[f32]),
    gate: &[f32],
    focus_tokens: &[u32],
    want_grad: bool,
) -> (HeldScore, Option<(Vec<f64>, Vec<f64>, Vec<f64>)>) {
    let (gate_w, up_w, down_w) = weights;
    let rows = indices.len();
    let hidden = fm.hidden;
    let inter = gate.len();
    if rows == 0 {
        return (
            HeldScore {
                ppl: f64::NAN,
                accuracy: None,
                balanced_accuracy: None,
            },
            None,
        );
    }
    let h1 = gather_rows(&cache.h1, indices, hidden);
    let n2 = gather_rows(&cache.n2, indices, hidden);
    let mut gate_pre = vec![0f32; rows * inter];
    let mut up_pre = vec![0f32; rows * inter];
    ops::gemm_nt(
        &n2,
        gate_w,
        &mut gate_pre,
        rows,
        hidden,
        inter,
        fm.pool.as_deref(),
    );
    ops::gemm_nt(
        &n2,
        up_w,
        &mut up_pre,
        rows,
        hidden,
        inter,
        fm.pool.as_deref(),
    );
    let mut act = vec![0f32; rows * inter];
    for row in 0..rows {
        for column in 0..inter {
            let at = row * inter + column;
            act[at] = ops::silu(gate_pre[at]) * up_pre[at] * gate[column];
        }
    }
    let mut ffn = vec![0f32; rows * hidden];
    ops::gemm_nt(
        &act,
        down_w,
        &mut ffn,
        rows,
        inter,
        hidden,
        fm.pool.as_deref(),
    );
    let mut h2 = h1;
    for (value, &delta) in h2.iter_mut().zip(&ffn) {
        *value += delta;
    }
    let mut normed = vec![0f32; rows * hidden];
    let mut inv = vec![0f32; rows];
    ops::rmsnorm_fwd(&h2, &fm.final_norm, fm.eps, fm.gemma, &mut normed, &mut inv);

    let lm: &[f32] = fm.lm_head.as_deref().unwrap_or(&fm.embed);
    let mut stats = FocusStats::new(focus_tokens.len());
    let mut nll = 0f64;
    let mut dh_normed = want_grad.then(|| vec![0f32; rows * hidden]);
    for (local_row, &cache_row) in indices.iter().enumerate() {
        let target = cache.targets[cache_row];
        let state = &normed[local_row * hidden..(local_row + 1) * hidden];
        let logits: Vec<f32> = focus_tokens
            .iter()
            .map(|&id| {
                state
                    .iter()
                    .zip(&lm[id as usize * hidden..(id as usize + 1) * hidden])
                    .map(|(&left, &right)| left * right)
                    .sum()
            })
            .collect();
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = logits
            .iter()
            .map(|&value| ((value as f64) - mx).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        nll += mx + sum.ln() - logits[target] as f64;
        let predicted = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(class, _)| class)
            .expect("focused classes are non-empty");
        stats.total += 1;
        stats.class_total[target] += 1;
        if predicted == target {
            stats.correct += 1;
            stats.class_correct[target] += 1;
        }
        if let Some(gradient) = dh_normed.as_mut() {
            let row = &mut gradient[local_row * hidden..(local_row + 1) * hidden];
            for (class, (&id, probability)) in focus_tokens.iter().zip(exps).enumerate() {
                let coefficient =
                    (probability / sum - usize::from(class == target) as f64) / rows as f64;
                let head = &lm[id as usize * hidden..(id as usize + 1) * hidden];
                for (value, &weight) in row.iter_mut().zip(head) {
                    *value += (coefficient * weight as f64) as f32;
                }
            }
        }
    }
    let score = HeldScore {
        ppl: (nll / rows as f64).exp(),
        accuracy: stats.accuracy(),
        balanced_accuracy: stats.balanced_accuracy(),
    };
    let Some(dh_normed) = dh_normed else {
        return (score, None);
    };

    let mut dh = vec![0f32; rows * hidden];
    ops::rmsnorm_bwd(
        &h2,
        &fm.final_norm,
        &inv,
        &dh_normed,
        fm.gemma,
        &mut dh,
        None,
    );
    let mut dact = vec![0f32; rows * inter];
    ops::gemm_dx(
        &dh,
        down_w,
        &mut dact,
        rows,
        inter,
        hidden,
        fm.pool.as_deref(),
    );
    let mut dg_pre = vec![0f32; rows * inter];
    let mut du_pre = vec![0f32; rows * inter];
    for row in 0..rows {
        for column in 0..inter {
            let at = row * inter + column;
            let da = dact[at] * gate[column];
            dg_pre[at] = da * up_pre[at] * ops::silu_bwd(gate_pre[at]);
            du_pre[at] = da * ops::silu(gate_pre[at]);
        }
    }
    let mut dg = vec![0f32; inter * hidden];
    let mut du = vec![0f32; inter * hidden];
    let mut dd = vec![0f32; hidden * inter];
    ops::gemm_dw(
        &dg_pre,
        &n2,
        &mut dg,
        rows,
        hidden,
        inter,
        fm.pool.as_deref(),
    );
    ops::gemm_dw(
        &du_pre,
        &n2,
        &mut du,
        rows,
        hidden,
        inter,
        fm.pool.as_deref(),
    );
    ops::gemm_dw(&dh, &act, &mut dd, rows, inter, hidden, fm.pool.as_deref());
    (
        score,
        Some((
            dg.into_iter().map(f64::from).collect(),
            du.into_iter().map(f64::from).collect(),
            dd.into_iter().map(f64::from).collect(),
        )),
    )
}

/// Score a WRITTEN specialist through the replica's own math (f32
/// dequant of whatever the file carries) with the file's binary mask
/// held hard — the decomposition probe that tells "the requant at write
/// cost the quality" from "the runtime applies the mask differently".
/// Returns (bare, masked) held-PPL over the chunks.
pub fn replica_score_file_mask(
    model: &Arc<CmfModel>,
    chunks: &[Vec<u32>],
) -> Result<(f64, f64), String> {
    let o1_off = crate::nystrom::O1Cfg {
        layers: crate::nystrom::O1Layers::List(Vec::new()),
        m: 4,
        w: 8,
        sink: 1,
        rect: crate::nystrom::O1_DEFAULT_RECT,
    };
    let fm = FcdModel::from_cmf(model, &o1_off, false)?;
    let nl = fm.layers.len();
    let loops = fm.loops.max(1);
    let vn = nl * loops;
    let inter = fm.layers[0].inter;
    let ffn: Vec<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> = vec![None; nl];
    // Binary mask → logits at ±50: σ crosses any τ exactly as the bit says.
    let task = &model.masks.default_task;
    let mask = model
        .masks
        .masks
        .iter()
        .find(|m| &m.name == task)
        .or_else(|| model.masks.masks.first());
    let open: Vec<Vec<f32>> = vec![vec![50.0; inter]; vn];
    let masked_logits: Vec<Vec<f32>> = match mask {
        Some(m) => (0..vn)
            .map(|vl| {
                let row = m.ffn_masks.get(vl).map(|v| v.as_slice()).unwrap_or(&[]);
                (0..inter)
                    .map(|j| {
                        if (row.get(j >> 3).copied().unwrap_or(0) >> (j & 7)) & 1 != 0 {
                            50.0
                        } else {
                            -50.0
                        }
                    })
                    .collect()
            })
            .collect(),
        None => open.clone(),
    };
    let score = |logits: &[Vec<f32>]| -> f64 {
        let pass = Pass {
            fm: &fm,
            tau: 0.5,
            logits,
            hard: true,
            ffn: &ffn,
            focus_tokens: &[],
            focus_follow_tokens: &[],
        };
        held_ppl(&pass, chunks)
    };
    Ok((score(&open), score(&masked_logits)))
}

/// The whole recipe. `log` receives progress lines.
pub fn skill_bake(
    model: &Arc<CmfModel>,
    chunks: &[Vec<u32>],
    held_n: usize,
    hy: &BakeHyper,
    mut log: impl FnMut(&str),
) -> Result<(BakeReport, BakeArtifacts), String> {
    let t0 = std::time::Instant::now();
    let o1_off = crate::nystrom::O1Cfg {
        layers: crate::nystrom::O1Layers::List(Vec::new()),
        m: 4,
        w: 8,
        sink: 1,
        rect: crate::nystrom::O1_DEFAULT_RECT,
    };
    let fm = FcdModel::from_cmf(model, &o1_off, false)?;
    let nl = fm.layers.len();
    let inter = fm.layers.iter().map(|l| l.inter).max().unwrap_or(0);
    let held: Vec<Vec<u32>> = chunks[..held_n.min(chunks.len())].to_vec();
    let calib: Vec<Vec<u32>> = chunks[held_n.min(chunks.len())..].to_vec();
    if calib.len() < 12 {
        return Err(format!(
            "skill bake: corpus too small ({} calib chunks)",
            calib.len()
        ));
    }
    // Analysis-only callers can request a native, batched focused score over
    // a validation corpus without training or writing a specialist.  The
    // scoring path already handles per-layer FFN widths; the optimizer and
    // defragment writer below still require one common width, so return the
    // identity checkpoint before that training-only constraint is applied.
    if hy.steps_a == 0 && hy.steps_b == 0 && hy.fcd_layers == 0 {
        let logits: Vec<Vec<f32>> = fm
            .layers
            .iter()
            .map(|layer| vec![100.0; layer.inter])
            .collect();
        let ffn = vec![None; nl];
        let pass = Pass {
            fm: &fm,
            tau: hy.tau,
            logits: &logits,
            hard: true,
            ffn: &ffn,
            focus_tokens: &hy.focus_tokens,
            focus_follow_tokens: &hy.focus_follow_tokens,
        };
        let score = held_score(&pass, &held);
        let keep: Vec<Vec<bool>> = fm
            .layers
            .iter()
            .map(|layer| vec![true; layer.inter])
            .collect();
        let loops = fm.loops.max(1);
        let keep_visits = (0..loops).flat_map(|_| keep.iter().cloned()).collect();
        let report = BakeReport {
            backbone: score.ppl,
            masked: score.ppl,
            overlaid: score.ppl,
            pruned_ratio: 0.0,
            kept_per_layer: keep.iter().map(Vec::len).collect(),
            backbone_accuracy: score.accuracy,
            masked_accuracy: score.accuracy,
            overlaid_accuracy: score.accuracy,
            backbone_balanced_accuracy: score.balanced_accuracy,
            masked_balanced_accuracy: score.balanced_accuracy,
            overlaid_balanced_accuracy: score.balanced_accuracy,
            selected_step: 0,
            sec: t0.elapsed().as_secs_f64(),
        };
        let arts = BakeArtifacts {
            keep,
            keep_visits,
            down: vec![Vec::new(); nl],
            gate_up: vec![None; nl],
            fcd_layers: Vec::new(),
            logits: logits.clone(),
            final_logits: logits,
            checkpoints: Vec::new(),
        };
        return Ok((report, arts));
    }
    if fm.layers.iter().any(|l| l.inter != inter) {
        return Err("skill bake: non-uniform FFN widths".into());
    }
    if hy.batch == 0 {
        return Err("skill bake: batch must be positive".into());
    }
    let fcd: Vec<usize> = (nl.saturating_sub(hy.fcd_layers)..nl).collect();
    let _rng = SplitMix64::new(hy.seed);

    // Trainables. The gate starts as close to OPEN as the arithmetic
    // allows, because step zero must be the backbone and nothing else.
    //
    // It used to start at 2.0, and σ(2.0) = 0.881 — every FFN neuron
    // scaled to seven eighths before a single gradient. On an ordinary
    // stack that costs a few percent of perplexity and hides. On a
    // LOOPED Transformer it does not: Nanbeige 4.2 runs its 22 layers
    // twice, so each physical FFN is visited twice and the factor
    // compounds to 0.881² = 0.776 per layer over 44 visits. Measured:
    // baseline 4.187 → 278.4 at step 30 with 0% pruned. Nothing had been
    // pruned; the mask had simply turned the model down.
    //
    // So solve for the init instead of hardcoding it — but solve for the
    // right target. Pushing σ(m0)^loops to 0.999 starts at the backbone
    // and cannot move: the gradient carries σ'(m) = σ(1−σ), which at
    // σ = 0.9995 is 5e-4 against 0.105 at the old 2.0, and BOTH the data
    // term and the L1 term are scaled by it (see the update below). That
    // was tried: 60 steps, 0% pruned, hard-PPL equal to the baseline to
    // three digits. Identity that cannot learn is not an improvement.
    //
    // The quantity to preserve is the EFFECTIVE start — what the stack
    // actually multiplies by, once per visit compounded over the loop —
    // at the value the recipe was validated with on ordinary models:
    // σ(m0)^loops = σ(2.0) = 0.881. One loop reproduces the old constant
    // exactly, so nothing regresses; two loops open the per-visit gate to
    // 0.9385 so the compounded factor is again 0.881, with σ' = 0.058
    // rather than 0.0005.
    let loops = fm.loops.max(1);
    let m0 = mask_init_logit_for(loops, hy.mask_init);
    // One mask row per VIRTUAL layer: nl × loops. Unlooped: vn == nl.
    let vn = nl * loops;
    let mut logits: Vec<Vec<f32>> = vec![vec![m0; inter]; vn];
    let mut ffn: Vec<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> = vec![None; nl];

    // Baseline (no mask): even σ(m0) is not exactly 1, so measure with
    // gates forced open via hard mask over +∞… simplest: logits +50.
    let open: Vec<Vec<f32>> = vec![vec![50.0; inter]; vn];
    let base_pass = Pass {
        fm: &fm,
        tau: hy.tau,
        logits: &open,
        hard: true,
        ffn: &ffn,
        focus_tokens: &hy.focus_tokens,
        focus_follow_tokens: &hy.focus_follow_tokens,
    };
    let backbone_score = held_score(&base_pass, &held);
    let backbone = backbone_score.ppl;
    log(&format!(
        "baseline (full): {backbone:.3}{}",
        backbone_score
            .accuracy
            .zip(backbone_score.balanced_accuracy)
            .map(|(a, b)| format!(" | acc {:.2}% bal {:.2}%", a * 100.0, b * 100.0))
            .unwrap_or_default()
    ));

    // ── Phase A: mask training ──
    let mut adam_a = Adam::new(&vec![inter; vn], hy.lr_a);
    let mut l1 = hy.l1_init * hy.l1_mult;
    let l1_step_eff = hy.l1_step * hy.l1_mult;
    // best = (ppl, logits_snapshot, sparsity)
    let mut best: (f64, Option<Vec<Vec<f32>>>, f64) = (f64::MAX, None, 0.0);
    let mut best_accuracy = f64::NEG_INFINITY;
    let mut best_balanced = f64::NEG_INFINITY;
    let mut best_step = 0usize;
    let mut checkpoints = Vec::new();
    // Track the highest-sparsity checkpoint as fallback.
    let mut max_sp: (f64, Option<Vec<Vec<f32>>>, f64) = (f64::MAX, None, 0.0);
    let mut max_sp_step = 0usize;
    let mut prev_alive: Option<Vec<Vec<bool>>> = None;
    // In-process phase timers: this loop was estimated three different
    // ways and every estimate came out under a fifth of the measured
    // step time. Measure, then optimize the top line, not the guess.
    let mut acc_chunk = 0f64;
    let mut acc_adam = 0f64;
    // Phase A trains in strict f32: the mask SELECTS neurons by its
    // gradient, and f16 operand rounding on that signal — fine for every
    // forward and eval in this file — compounds over ~90 steps into
    // closing the wrong neurons (measured: hard-PPL 5.207 vs 4.293 at the
    // same 2.56% sparsity; the f32 run retraces the reference trajectory
    // to the third decimal). Evals inside the loop lift the restriction —
    // their tensor-core numbers match f32 at print precision.
    crate::gpu::bake_precision_strict(true);
    for step in 0..hy.steps_a {
        let t_step = std::time::Instant::now();
        let (batch_ids, batch) = calibration_batch(&calib, step, hy.batch)?;
        let mut dmask: Vec<Vec<f64>> = vec![vec![0.0; inter]; vn];
        let mut dffn: Vec<Option<(Vec<f64>, Vec<f64>, Vec<f64>)>> = vec![None; nl];
        let pass = Pass {
            fm: &fm,
            tau: hy.tau,
            logits: &logits,
            hard: false,
            ffn: &ffn,
            focus_tokens: &hy.focus_tokens,
            focus_follow_tokens: &hy.focus_follow_tokens,
        };
        let _ = pass.chunk_batch(&batch_ids, batch, Some((&mut dmask, &mut dffn)));
        // Fold σ'(m) into the mask grads + add the L1 term.
        let l1_per = l1 / (inter as f64 * nl as f64);
        for li in 0..vn {
            for j in 0..inter {
                let s = sigmoid(logits[li][j]) as f64;
                let sparse_grad = sparsity_grad(logits[li][j], hy.softplus_l1);
                dmask[li][j] = dmask[li][j] * s * (1.0 - s) + l1_per * sparse_grad;
            }
        }
        // One gradient per VISIT, one step per token. A Looped
        // Transformer visits each physical layer `loops` times and the
        // backward accumulates every visit into the same mask, so an
        // unnormalised step is `loops` times the one the recipe was
        // tuned with — the mask overshoots, neurons cross tau within the
        // first evaluation window, and each of them is then missing from
        // both passes. Dividing by the visit count makes a step mean the
        // same thing at any loop depth.
        // Per-visit rows: each mask logit receives exactly one visit's
        // gradient, so the step needs no visit normalisation here — that
        // scale now belongs to Phase B alone, where the FFN weights ARE
        // shared across visits.
        let t_chunk = t_step.elapsed().as_secs_f64();
        let mut params: Vec<&mut [f32]> = logits.iter_mut().map(|v| v.as_mut_slice()).collect();
        adam_a.step(&mut params, &dmask, 1.0);
        acc_chunk += t_chunk;
        acc_adam += t_step.elapsed().as_secs_f64() - t_chunk;
        if (step + 1) % hy.eval_every == 0 {
            l1 += l1_step_eff;
            let pass = Pass {
                fm: &fm,
                tau: hy.tau,
                logits: &logits,
                hard: true,
                ffn: &ffn,
                focus_tokens: &hy.focus_tokens,
                focus_follow_tokens: &hy.focus_follow_tokens,
            };
            crate::gpu::bake_precision_strict(false);
            let hs = held_score(&pass, &held);
            let hp = hs.ppl;
            crate::gpu::bake_precision_strict(true);
            // Name the neurons that crossed τ since the last eval. At
            // 0.01% pruned = ~24 neurons for a 135 held-PPL, WHICH 24 is
            // the whole diagnosis: it decides between "this model has no
            // noise neurons" and "a shared mask cannot spare a neuron
            // that only one visit of the loop needs".
            let cur: Vec<Vec<bool>> = logits
                .iter()
                .map(|l| l.iter().map(|&x| sigmoid(x) > hy.tau).collect())
                .collect();
            if let Some(prev) = &prev_alive {
                let died: Vec<String> = cur
                    .iter()
                    .zip(prev)
                    .enumerate()
                    .flat_map(|(li, (c, p))| {
                        c.iter()
                            .zip(p.iter())
                            .enumerate()
                            .filter(|&(_, (&cj, &pj))| pj && !cj)
                            .map(move |(j, _)| format!("L{li}:{j}"))
                    })
                    .collect();
                if !died.is_empty() {
                    log(&format!(
                        "    closed since last eval: {}: {}{}",
                        died.len(),
                        died.iter().take(32).cloned().collect::<Vec<_>>().join(" "),
                        if died.len() > 32 { " …" } else { "" }
                    ));
                }
            }
            let alive: usize = cur.iter().map(|l| l.iter().filter(|&&b| b).count()).sum();
            prev_alive = Some(cur);
            let sp = 1.0 - alive as f64 / (vn * inter) as f64;
            // Track highest-sparsity checkpoint.
            if sp > max_sp.2 {
                max_sp = (hp, Some(logits.clone()), sp);
                max_sp_step = step + 1;
            }
            checkpoints.push(BakeCheckpoint {
                step: step + 1,
                l1,
                ppl: hp,
                sparsity: sp,
                accuracy: hs.accuracy,
                balanced_accuracy: hs.balanced_accuracy,
            });
            // Best checkpoint selection: respect target_sparsity and any
            // caller-declared natural-distribution quality guards. The
            // guards are strict (`>`, not `>=`) so a majority baseline cannot
            // sneak through on an exactly tied checkpoint.
            let eligible_sparsity = hy.target_sparsity <= 0.0 || sp >= hy.target_sparsity;
            let eligible_accuracy = hy
                .checkpoint_min_accuracy
                .map_or(true, |min| hs.accuracy.is_some_and(|value| value > min));
            let eligible_balanced = hy.checkpoint_min_balanced_accuracy.map_or(true, |min| {
                hs.balanced_accuracy.is_some_and(|value| value > min)
            });
            let eligible = eligible_sparsity && eligible_accuracy && eligible_balanced;
            if eligible && hy.checkpoint_raw_priority && !hy.focus_tokens.is_empty() {
                let acc = hs.accuracy.unwrap_or(f64::NEG_INFINITY);
                let bal = hs.balanced_accuracy.unwrap_or(f64::NEG_INFINITY);
                if acc > best_accuracy
                    || (acc == best_accuracy && bal > best_balanced)
                    || (acc == best_accuracy && bal == best_balanced && hp < best.0)
                {
                    best_balanced = bal;
                    best_accuracy = acc;
                    best = (hp, Some(logits.clone()), sp);
                    best_step = step + 1;
                }
            } else if eligible && hy.checkpoint_accuracy && !hy.focus_tokens.is_empty() {
                let acc = hs.accuracy.unwrap_or(f64::NEG_INFINITY);
                let bal = hs.balanced_accuracy.unwrap_or(f64::NEG_INFINITY);
                if bal > best_balanced
                    || (bal == best_balanced && acc > best_accuracy)
                    || (bal == best_balanced && acc == best_accuracy && hp < best.0)
                {
                    best_balanced = bal;
                    best_accuracy = acc;
                    best = (hp, Some(logits.clone()), sp);
                    best_step = step + 1;
                }
            } else if eligible && hp < best.0 {
                best = (hp, Some(logits.clone()), sp);
                best_step = step + 1;
            }
            log(&format!(
                "  [A] step {}: L1={l1:.3} pruned={:.2}% hard-PPL={hp:.3}{} (bottom {}@{:.2}%) [fwd+bwd {:.1}s, adam {:.2}s per step]",
                step + 1,
                sp * 100.0,
                hs.accuracy
                    .zip(hs.balanced_accuracy)
                    .map(|(a, b)| format!(" acc={:.2}% bal={:.2}%", a * 100.0, b * 100.0))
                    .unwrap_or_default(),
                if best.0 == f64::MAX {
                    "—".to_string()
                } else {
                    format!("{:.3}", best.0)
                },
                best.2 * 100.0,
                acc_chunk / (step + 1) as f64,
                acc_adam / (step + 1) as f64
            ));
        }
    }
    // If target_sparsity was set but no checkpoint qualified, fall back
    // to the highest-sparsity checkpoint.
    // Phase A is over — phase B and every eval after run on the fast arms.
    crate::gpu::bake_precision_strict(false);
    if (hy.checkpoint_min_accuracy.is_some() || hy.checkpoint_min_balanced_accuracy.is_some())
        && best.1.is_none()
    {
        return Err(
            "skill bake: no Phase-A checkpoint met the configured focused accuracy guards".into(),
        );
    }
    if hy.target_sparsity > 0.0 && best.1.is_none() {
        log(&format!(
            "[A] target sparsity {:.0}% not reached; using max-sparsity checkpoint ({:.0}%)",
            hy.target_sparsity * 100.0,
            max_sp.2 * 100.0
        ));
        best = max_sp;
        best_step = max_sp_step;
    }
    // Phase totals, printed unconditionally — a 5-step measurement run
    // must report even though no eval fired.
    {
        use crate::fcd::prof;
        let (a, f, bw, g, gc) = (
            prof::take(&prof::ATTN_FWD),
            prof::take(&prof::FFN_FWD),
            prof::take(&prof::BWD),
            prof::take(&prof::GEMM),
            prof::GEMM_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed),
        );
        log(&format!(
            "[prof] phase A over {} step(s): attn-fwd {a:.1}s | ffn-fwd {f:.1}s | bwd {bw:.1}s |              gemm total {g:.1}s in {gc} calls ({:.1} ms/call)",
            hy.steps_a,
            if gc > 0 { g * 1000.0 / gc as f64 } else { 0.0 }
        ));
        log(&format!("[prof] gemm shapes:\n{}", prof::shape_report(6)));
    }
    let final_logits = logits.clone();
    if let Some(b) = best.1.take() {
        logits = b;
    }
    let pass = Pass {
        fm: &fm,
        tau: hy.tau,
        logits: &logits,
        hard: true,
        ffn: &ffn,
        focus_tokens: &hy.focus_tokens,
        focus_follow_tokens: &hy.focus_follow_tokens,
    };
    // With no mask-training steps the hard mask is still exactly all-open,
    // so rescoring the same validation batch is pure duplicate work.  This
    // matters for focused 512-token records where one gate can take minutes
    // on a laptop, and keeps short FCD sweeps practical without changing a
    // single number.
    let masked_score = if hy.steps_a == 0 {
        backbone_score.clone()
    } else {
        held_score(&pass, &held)
    };
    let masked = masked_score.ppl;
    log(&format!(
        "[A] {:.0}s: masked-PPL {masked:.3}",
        t0.elapsed().as_secs_f64()
    ));

    // ── Phase B: FCD of the last N layers' FFN (hard mask active) ──
    for &li in &fcd {
        let p = format!("model.layers.{li}.");
        ffn[li] = Some((
            crate::fcd::deq_pub(&fm.src, &format!("{p}mlp.gate_proj.weight"))
                .map_err(|e| format!("phase-B gate: {e}"))?,
            crate::fcd::deq_pub(&fm.src, &format!("{p}mlp.up_proj.weight"))
                .map_err(|e| format!("phase-B up: {e}"))?,
            crate::fcd::deq_pub(&fm.src, &format!("{p}mlp.down_proj.weight"))
                .map_err(|e| format!("phase-B down: {e}"))?,
        ));
    }
    let sizes: Vec<usize> = fcd
        .iter()
        .flat_map(|&li| {
            let (g, u, d) = ffn[li].as_ref().expect("phase-B masters");
            [g.len(), u.len(), d.len()]
        })
        .collect();
    let mut adam_b = Adam::new(&sizes, hy.lr_b);
    // The mask-only model is a real checkpoint too. If every FCD eval is
    // worse, restore `None` overlays rather than accidentally writing the
    // final (rejected) training step while reporting the mask-only PPL.
    let mut best_b: (
        HeldScore,
        Option<Vec<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>>>,
    ) = (masked_score.clone(), Some(vec![None; nl]));
    let cached_phase_b =
        hy.steps_b > 0 && fcd.len() == 1 && loops == 1 && !hy.focus_tokens.is_empty();
    if cached_phase_b {
        let last = fcd[0];
        let pass = Pass {
            fm: &fm,
            tau: hy.tau,
            logits: &logits,
            hard: true,
            ffn: &ffn,
            focus_tokens: &hy.focus_tokens,
            focus_follow_tokens: &hy.focus_follow_tokens,
        };
        log(&format!(
            "[B-cache] extracting native CMF boundaries: {} train + {} held records",
            calib.len(),
            held.len()
        ));
        let train_cache = build_focused_fcd_cache(&pass, &calib, 32)?;
        let held_cache = build_focused_fcd_cache(&pass, &held, 32)?;
        if train_cache.len() < 12 || held_cache.len() == 0 {
            return Err(format!(
                "focused final-FFN cache is too small: {} train, {} held answers",
                train_cache.len(),
                held_cache.len()
            ));
        }
        let gate = pass.gates(last);
        let held_indices: Vec<usize> = (0..held_cache.len()).collect();
        let (initial_cached, _) = {
            let (g, u, d) = ffn[last].as_ref().expect("cached FCD master");
            cached_fcd_run(
                &fm,
                &held_cache,
                &held_indices,
                (g, u, d),
                &gate,
                &hy.focus_tokens,
                false,
            )
        };
        log(&format!(
            "[B-cache] ready: {} train / {} held | parity PPL {:.3} vs full {:.3}{}",
            train_cache.len(),
            held_cache.len(),
            initial_cached.ppl,
            masked_score.ppl,
            initial_cached
                .accuracy
                .map(|value| format!(" | acc {:.2}%", value * 100.0))
                .unwrap_or_default()
        ));
        if (initial_cached.ppl - masked_score.ppl).abs() > 5e-3
            || initial_cached.accuracy != masked_score.accuracy
        {
            return Err(format!(
                "focused final-FFN cache parity failed: PPL {:.6} vs {:.6}, accuracy {:?} vs {:?}",
                initial_cached.ppl,
                masked_score.ppl,
                initial_cached.accuracy,
                masked_score.accuracy
            ));
        }
        for step in 0..hy.steps_b {
            let count = hy.fcd_batch.min(train_cache.len());
            let indices: Vec<usize> = (0..count)
                .map(|offset| (step * count + offset) % train_cache.len())
                .collect();
            let (_, gradients) = {
                let (g, u, d) = ffn[last].as_ref().expect("cached FCD master");
                cached_fcd_run(
                    &fm,
                    &train_cache,
                    &indices,
                    (g, u, d),
                    &gate,
                    &hy.focus_tokens,
                    true,
                )
            };
            let (dg, du, dd) = gradients.expect("cached FCD requested gradients");
            let lr_scale =
                0.5 * (1.0 + (std::f64::consts::PI * step as f64 / hy.steps_b as f64).cos());
            let (g, u, d) = ffn[last].as_mut().expect("cached FCD master");
            let mut params = vec![g.as_mut_slice(), u.as_mut_slice(), d.as_mut_slice()];
            let grads = vec![dg, du, dd];
            adam_b.step(&mut params, &grads, lr_scale);
            if (step + 1) % hy.eval_every == 0 {
                let cur = {
                    let (g, u, d) = ffn[last].as_ref().expect("cached FCD master");
                    cached_fcd_run(
                        &fm,
                        &held_cache,
                        &held_indices,
                        (g, u, d),
                        &gate,
                        &hy.focus_tokens,
                        false,
                    )
                    .0
                };
                let cur_bal = cur.balanced_accuracy.unwrap_or(f64::NEG_INFINITY);
                let best_bal = best_b.0.balanced_accuracy.unwrap_or(f64::NEG_INFINITY);
                let cur_acc = cur.accuracy.unwrap_or(f64::NEG_INFINITY);
                let best_acc = best_b.0.accuracy.unwrap_or(f64::NEG_INFINITY);
                let better = if hy.checkpoint_accuracy {
                    cur_bal > best_bal
                        || (cur_bal == best_bal && cur_acc > best_acc)
                        || (cur_bal == best_bal && cur_acc == best_acc && cur.ppl < best_b.0.ppl)
                } else {
                    cur.ppl < best_b.0.ppl
                };
                if better {
                    best_b = (cur.clone(), Some(ffn.clone()));
                }
                log(&format!(
                    "  [B-cache] step {}: held-PPL {:.3} acc={:.2}% bal={:.2}% (best {:.3})",
                    step + 1,
                    cur.ppl,
                    cur_acc * 100.0,
                    cur_bal * 100.0,
                    best_b.0.ppl
                ));
            }
        }
    } else {
        for step in 0..hy.steps_b {
            let (batch_ids, batch) = calibration_batch(&calib, step, hy.batch)?;
            let mut dmask: Vec<Vec<f64>> = vec![vec![0.0; inter]; vn];
            let mut dffn: Vec<Option<(Vec<f64>, Vec<f64>, Vec<f64>)>> = (0..nl)
                .map(|li| {
                    ffn[li].as_ref().map(|(g, u, d)| {
                        (vec![0.0; g.len()], vec![0.0; u.len()], vec![0.0; d.len()])
                    })
                })
                .collect();
            let pass = Pass {
                fm: &fm,
                tau: hy.tau,
                logits: &logits,
                hard: true,
                ffn: &ffn,
                focus_tokens: &hy.focus_tokens,
                focus_follow_tokens: &hy.focus_follow_tokens,
            };
            let _ = pass.chunk_batch(&batch_ids, batch, Some((&mut dmask, &mut dffn)));
            // Cosine LR.
            let lr_scale =
                0.5 * (1.0 + (std::f64::consts::PI * step as f64 / hy.steps_b as f64).cos());
            let first_fcd = fcd[0];
            let mut params: Vec<&mut [f32]> = Vec::new();
            let mut grads: Vec<Vec<f64>> = Vec::new();
            for (off, slot) in ffn[first_fcd..].iter_mut().enumerate() {
                let li = first_fcd + off;
                let Some((g, u, d)) = slot.as_mut() else {
                    continue;
                };
                let (dg, du, dd) = dffn[li].take().unwrap();
                params.push(g.as_mut_slice());
                grads.push(dg);
                params.push(u.as_mut_slice());
                grads.push(du);
                params.push(d.as_mut_slice());
                grads.push(dd);
            }
            // Same visit normalisation as Phase A: dffn accumulates every
            // visit of a physical layer, and an FFN update perturbs BOTH
            // passes of the loop, so per-step damage is `loops` times what
            // lr_b was tuned for on ordinary stacks.
            adam_b.step(&mut params, &grads, lr_scale * mask_step_scale(loops));
            if (step + 1) % hy.eval_every == 0 {
                let pass = Pass {
                    fm: &fm,
                    tau: hy.tau,
                    logits: &logits,
                    hard: true,
                    ffn: &ffn,
                    focus_tokens: &hy.focus_tokens,
                    focus_follow_tokens: &hy.focus_follow_tokens,
                };
                let cur = held_score(&pass, &held);
                let better = if hy.checkpoint_accuracy && !hy.focus_tokens.is_empty() {
                    let cur_bal = cur.balanced_accuracy.unwrap_or(f64::NEG_INFINITY);
                    let best_bal = best_b.0.balanced_accuracy.unwrap_or(f64::NEG_INFINITY);
                    let cur_acc = cur.accuracy.unwrap_or(f64::NEG_INFINITY);
                    let best_acc = best_b.0.accuracy.unwrap_or(f64::NEG_INFINITY);
                    cur_bal > best_bal
                        || (cur_bal == best_bal && cur_acc > best_acc)
                        || (cur_bal == best_bal && cur_acc == best_acc && cur.ppl < best_b.0.ppl)
                } else {
                    cur.ppl < best_b.0.ppl
                };
                if better {
                    best_b = (cur.clone(), Some(ffn.clone()));
                }
                log(&format!(
                    "  [B] step {}: held-PPL {:.3}{} (best {:.3})",
                    step + 1,
                    cur.ppl,
                    cur.accuracy
                        .zip(cur.balanced_accuracy)
                        .map(|(a, b)| format!(" acc={:.2}% bal={:.2}%", a * 100.0, b * 100.0))
                        .unwrap_or_default(),
                    best_b.0.ppl
                ));
            }
        }
    }
    ffn = best_b.1.take().expect("phase-B always has a checkpoint");
    let overlaid_score = best_b.0;
    let overlaid = overlaid_score.ppl;

    // ── Export artifacts ──
    // Per-visit keep flags are the mask that ships; the PHYSICAL keep is
    // their union, because a weight row can only be removed from disk if
    // no visit needs it.
    let keep_visits = keep_masks(&logits, hy.tau, hy.align, hy.uniform_inter);
    let keep: Vec<Vec<bool>> = (0..nl)
        .map(|li| {
            (0..inter)
                .map(|j| (0..loops).any(|v| keep_visits[v * nl + li][j]))
                .collect()
        })
        .collect();
    if hy.align > 1 || hy.uniform_inter {
        let raw: usize = logits
            .iter()
            .map(|l| l.iter().filter(|&&x| sigmoid(x) > hy.tau).count())
            .sum();
        // Compare like with like: raw σ-counts are over the VIRTUAL
        // rows, so the padded count must be too — the union rows are
        // fewer and the subtraction would underflow.
        let padded: usize = keep_visits
            .iter()
            .map(|a| a.iter().filter(|&&x| x).count())
            .sum::<usize>()
            .saturating_sub(raw);
        log(&format!(
            "align: +{padded} neurons resurrected (align {}, uniform {})",
            hy.align, hy.uniform_inter
        ));
    }
    let mut down_out = Vec::with_capacity(nl);
    let mut gate_up = Vec::with_capacity(nl);
    let mut kept_per_layer = Vec::with_capacity(nl);
    for li in 0..nl {
        let alive = &keep[li];
        kept_per_layer.push(alive.iter().filter(|&&a| a).count());
        let mut down = match &ffn[li] {
            Some((_, _, d)) => d.clone(),
            None => fm.mats(li).expect("layer mats").down.clone(),
        };
        let hsz = fm.hidden;
        for r in 0..hsz {
            for (c, &a) in alive.iter().enumerate() {
                if !a {
                    down[r * inter + c] = 0.0;
                }
            }
        }
        gate_up.push(ffn[li].as_ref().map(|(g, u, _)| (g.clone(), u.clone())));
        down_out.push(down);
    }
    let total: usize = keep_visits
        .iter()
        .map(|a| a.iter().filter(|&&x| x).count())
        .sum();
    let report = BakeReport {
        backbone,
        masked,
        overlaid,
        pruned_ratio: 1.0 - total as f64 / (vn * inter) as f64,
        kept_per_layer,
        backbone_accuracy: backbone_score.accuracy,
        masked_accuracy: masked_score.accuracy,
        overlaid_accuracy: overlaid_score.accuracy,
        backbone_balanced_accuracy: backbone_score.balanced_accuracy,
        masked_balanced_accuracy: masked_score.balanced_accuracy,
        overlaid_balanced_accuracy: overlaid_score.balanced_accuracy,
        selected_step: best_step,
        sec: t0.elapsed().as_secs_f64(),
    };
    let arts = BakeArtifacts {
        keep,
        keep_visits,
        down: down_out,
        gate_up,
        fcd_layers: fcd,
        logits: logits.clone(),
        final_logits,
        checkpoints,
    };
    Ok((report, arts))
}

/// Hard-threshold keep masks from the trained logits, then resurrect
/// the highest-logit pruned neurons until each layer's kept count is a
/// multiple of `align` (rounding UP — the resurrected neurons are the
/// ones the mask ranked closest to the threshold, so this only moves
/// toward the full backbone). `uniform` additionally raises every layer
/// to the max layer's aligned count. A layer with 0 live neurons gets
/// `align.max(1)` — the defrag writer rejects empty layers.
fn keep_masks(logits: &[Vec<f32>], tau: f32, align: usize, uniform: bool) -> Vec<Vec<bool>> {
    let inter = logits[0].len();
    let round = |n: usize| -> usize {
        let n = n.max(1);
        if align <= 1 {
            n.min(inter)
        } else {
            (n.div_ceil(align) * align).min(inter)
        }
    };
    let mut want: Vec<usize> = logits
        .iter()
        .map(|l| round(l.iter().filter(|&&x| sigmoid(x) > tau).count()))
        .collect();
    if uniform {
        let k = want.iter().copied().max().unwrap_or(inter);
        want = vec![k; logits.len()];
    }
    logits
        .iter()
        .zip(&want)
        .map(|(l, &k)| {
            let mut idx: Vec<usize> = (0..inter).collect();
            idx.sort_unstable_by(|&a, &b| l[b].total_cmp(&l[a]));
            let mut alive = vec![false; inter];
            for &i in idx.iter().take(k) {
                alive[i] = true;
            }
            alive
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kept(masks: &[Vec<bool>]) -> Vec<usize> {
        masks
            .iter()
            .map(|m| m.iter().filter(|&&a| a).count())
            .collect()
    }

    #[test]
    fn terminal_focus_ignores_label_names_inside_the_prompt() {
        // DOWN and UP occur in the instruction, but only the final UP is an
        // assistant answer because it is immediately followed by im_end.
        let down = 10;
        let up = 11;
        let im_end = 99;
        let ids = [1, down, 2, up, 3, up, im_end, 4];
        let focus = [down, up];
        let follow = [im_end];
        assert!(!is_scored_target(&ids, 1, ids.len(), &focus, &follow));
        assert!(!is_scored_target(&ids, 3, ids.len(), &focus, &follow));
        assert!(is_scored_target(&ids, 5, ids.len(), &focus, &follow));
    }

    #[test]
    fn configurable_mask_init_preserves_effective_gate_across_loops() {
        for effective_logit in [2.0, 4.0] {
            let target = sigmoid(effective_logit);
            for loops in [1usize, 2, 4] {
                let per_visit = sigmoid(mask_init_logit_for(loops, effective_logit));
                assert!((per_visit.powi(loops as i32) - target).abs() < 2e-6);
            }
        }
    }

    #[test]
    fn softplus_penalty_keeps_a_gradient_near_an_open_gate() {
        let gate_penalty = sparsity_grad(4.0, false);
        let softplus_penalty = sparsity_grad(4.0, true);
        assert!(softplus_penalty > gate_penalty * 50.0);
        assert!((softplus_penalty - sigmoid(4.0) as f64).abs() < 1e-7);
    }

    #[test]
    fn balanced_accuracy_exposes_majority_class_collapse() {
        let stats = FocusStats {
            total: 100,
            correct: 90,
            class_total: vec![90, 10],
            class_correct: vec![90, 0],
        };
        assert_eq!(stats.accuracy(), Some(0.9));
        assert_eq!(stats.balanced_accuracy(), Some(0.5));
    }

    #[test]
    fn grouped_focus_stats_merge_is_additive() {
        let mut all = FocusStats {
            total: 3,
            correct: 2,
            class_total: vec![2, 1],
            class_correct: vec![1, 1],
        };
        let second = FocusStats {
            total: 4,
            correct: 3,
            class_total: vec![1, 3],
            class_correct: vec![1, 2],
        };
        all.merge(&second);
        assert_eq!(all.total, 7);
        assert_eq!(all.correct, 5);
        assert_eq!(all.class_total, vec![3, 4]);
        assert_eq!(all.class_correct, vec![2, 3]);
        assert_eq!(all.accuracy(), Some(5.0 / 7.0));
        assert_eq!(all.balanced_accuracy(), Some((2.0 / 3.0 + 3.0 / 4.0) / 2.0));
    }

    #[test]
    fn calibration_batches_keep_records_independent_and_wrap_deterministically() {
        let records = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        assert_eq!(
            calibration_batch(&records, 0, 2).unwrap(),
            (vec![1, 2, 3, 4], 2)
        );
        assert_eq!(
            calibration_batch(&records, 1, 2).unwrap(),
            (vec![5, 6, 1, 2], 2)
        );
        assert!(calibration_batch(&[vec![1, 2], vec![3]], 0, 2).is_err());
    }

    /// align=32 rounds each layer UP by resurrecting the largest
    /// pruned logits; the originally-alive set stays alive.
    #[test]
    fn keep_masks_aligns_up_and_preserves_alive() {
        let inter = 96;
        // Layer 0: 40 alive (logits > 0 → σ > 0.5), the rest ramp
        // below threshold so resurrection order is deterministic.
        let l0: Vec<f32> = (0..inter)
            .map(|i| if i < 40 { 1.0 } else { -1.0 - i as f32 * 0.01 })
            .collect();
        // Layer 1: 64 alive — already aligned, must stay exactly 64.
        let l1: Vec<f32> = (0..inter)
            .map(|i| if i < 64 { 2.0 } else { -3.0 })
            .collect();
        let masks = keep_masks(&[l0.clone(), l1], 0.5, 32, false);
        assert_eq!(kept(&masks), vec![64, 64]);
        // The 40 originally-alive stay; resurrected are the top pruned
        // logits (indices 40..64 — the least-negative of the ramp).
        for i in 0..64 {
            assert!(masks[0][i], "neuron {i} should be kept");
        }
        for i in 64..inter {
            assert!(!masks[0][i], "neuron {i} should stay pruned");
        }
    }

    /// uniform=true raises every layer to the max aligned count.
    #[test]
    fn keep_masks_uniform_takes_max() {
        let inter = 96;
        let l0: Vec<f32> = (0..inter)
            .map(|i| if i < 10 { 1.0 } else { -2.0 })
            .collect();
        let l1: Vec<f32> = (0..inter)
            .map(|i| if i < 70 { 1.0 } else { -2.0 })
            .collect();
        let masks = keep_masks(&[l0, l1], 0.5, 32, true);
        assert_eq!(kept(&masks), vec![96, 96]);
    }

    /// align capped at inter; align=1 (off) keeps the raw threshold
    /// count; an all-pruned layer still keeps at least one neuron.
    #[test]
    fn keep_masks_edges() {
        let inter = 48;
        let l: Vec<f32> = (0..inter)
            .map(|i| if i < 47 { 1.0 } else { -2.0 })
            .collect();
        let masks = keep_masks(&[l.clone()], 0.5, 32, false);
        assert_eq!(kept(&masks), vec![48]); // 47 → 64 capped to 48
        let masks = keep_masks(&[l], 0.5, 1, false);
        assert_eq!(kept(&masks), vec![47]);
        let dead: Vec<f32> = vec![-5.0; inter];
        let masks = keep_masks(&[dead], 0.5, 32, false);
        assert_eq!(kept(&masks), vec![32]); // max(1) → rounded to 32
    }
}
