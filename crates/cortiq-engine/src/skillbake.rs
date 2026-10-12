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

use crate::fcd::{AttnW, FcdModel, LnFfn, act_bwd_rows, add_into_pooled, pooled_rows, rmsnorm_bwd_pooled, rmsnorm_fwd_pooled};
use crate::fcd_ops as ops;
use crate::sampler::SplitMix64;
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Supervised chat records scored on the assistant's answer only: every
/// target from the first token after `start` (the rendered generation
/// prompt, e.g. `<|Bot|></think>`) through the first `end` token (the
/// turn terminator) carries the LM loss; the prompt is context. This is
/// the multi-token sibling of `focus_tokens` — a JSON verdict, a number,
/// a sentence — where a one-token label does not fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnswerSpan {
    /// Token ids of the generation prompt; the answer starts right after
    /// its LAST occurrence in the record.
    pub start: Vec<u32>,
    /// The turn terminator (one token); the last scored target.
    pub end: u32,
    /// Decoded text of every vocabulary id. When present, the answer's
    /// top-level JSON values are located in it and the held score grades
    /// whole verdicts (`answer_values`): a question counts as correct only
    /// when every token of its value is the argmax. Token accuracy over
    /// the span alone is dominated by the braces, quotes and question ids
    /// the model copies from the prompt.
    pub values: Option<TokenText>,
}

/// Decoded text per token id (see `AnswerSpan::values`).
#[derive(Clone, PartialEq, Eq)]
pub struct TokenText(pub Arc<Vec<String>>);

impl std::fmt::Debug for TokenText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TokenText({} ids)", self.0.len())
    }
}

/// The target positions of each top-level value of the JSON object an
/// answer span holds (`{"q1": "option_24", "q2": 3}` → two groups), in
/// order. A token overlapping a value's characters belongs to it, so a
/// token that merges a quote with the label's first letters is graded with
/// the label. Positions are indices into `ids`, within `first..=last`.
pub fn answer_values(ids: &[u32], first: usize, last: usize, text: &TokenText) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut cur: Option<Vec<usize>> = None;
    let (mut depth, mut in_str, mut esc, mut expect_value) = (0i32, false, false, false);
    let mark = |cur: &mut Option<Vec<usize>>, p: usize| {
        if let Some(g) = cur.as_mut() {
            if g.last() != Some(&p) {
                g.push(p);
            }
        }
    };
    for p in first..=last.min(ids.len().saturating_sub(1)) {
        let piece = text.0.get(ids[p] as usize).map(String::as_str).unwrap_or("");
        for ch in piece.chars() {
            if in_str {
                if esc {
                    esc = false;
                } else if ch == '\\' {
                    esc = true;
                } else if ch == '"' {
                    in_str = false;
                }
                mark(&mut cur, p);
                continue;
            }
            if expect_value && !ch.is_whitespace() {
                expect_value = false;
                cur = Some(Vec::new());
            }
            match ch {
                '"' => {
                    in_str = true;
                    mark(&mut cur, p);
                }
                '{' | '[' => {
                    depth += 1;
                    mark(&mut cur, p);
                }
                '}' | ']' => {
                    if depth == 1 {
                        if let Some(g) = cur.take() {
                            if !g.is_empty() {
                                groups.push(g);
                            }
                        }
                    } else {
                        mark(&mut cur, p);
                    }
                    depth -= 1;
                }
                ',' if depth == 1 => {
                    if let Some(g) = cur.take() {
                        if !g.is_empty() {
                            groups.push(g);
                        }
                    }
                }
                ':' if depth == 1 && cur.is_none() => expect_value = true,
                c if c.is_whitespace() && depth == 1 => {}
                _ => mark(&mut cur, p),
            }
        }
    }
    if let Some(g) = cur {
        if !g.is_empty() {
            groups.push(g);
        }
    }
    groups
}

/// The scored target range `(first, last)` (inclusive, indices into
/// `ids`) of one record, or `None` when the record carries no answer.
pub fn answer_targets(ids: &[u32], span: &AnswerSpan) -> Option<(usize, usize)> {
    let m = span.start.len();
    if m == 0 || ids.len() <= m {
        return None;
    }
    let at = (0..=ids.len() - m).rev().find(|&i| ids[i..i + m] == span.start[..])?;
    let first = at + m;
    if first >= ids.len() {
        return None;
    }
    let last = (first..ids.len())
        .find(|&i| ids[i] == span.end)
        .unwrap_or(ids.len() - 1);
    Some((first, last))
}

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
    /// Score the assistant's answer span of each chat record instead of
    /// every token (or one-token labels). Exclusive with `focus_tokens`.
    pub answer_span: Option<AnswerSpan>,
    /// Start Phase A from these mask logits (one row per virtual layer,
    /// e.g. a previous run's archaeology `selected_logits`) instead of
    /// the uniform init — so a mask trained once can be FCD-polished
    /// again, or Phase A resumed, in a separate run (`--steps-a 0` keeps it).
    pub mask_logits: Option<Vec<Vec<f32>>>,
    /// Patent 2 Soft-Fold compilation: the neurons that survive τ keep
    /// their LEARNED gain σ(m) (folded into down_proj by the writer) instead
    /// of being snapped to 1. Evaluation and Phase B then run that exact
    /// function (σ(m) above τ, 0 below), so what is scored is what ships.
    pub soft_fold: bool,
    /// Full-layer FCD: Phase B also trains the attention projections
    /// (q/k/v/o, the projected head gate) and both norm gains of the FCD
    /// layers — the notebook's "unfreeze the last N layers entirely".
    pub fcd_attn: bool,
    /// Write every Phase-A evaluation's mask logits to
    /// `<dir>/phase-a-step-<n>.json` as it happens (a `--mask-from` input),
    /// so a long run survives being stopped.
    pub checkpoint_dir: Option<String>,
    /// Tags of the held records (in held order before the length sort):
    /// held scores then report per-tag verdict accuracy and gate on its
    /// macro average.
    pub held_tags: Option<HeldTags>,
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
            answer_span: None,
            mask_logits: None,
            soft_fold: false,
            fcd_attn: false,
            checkpoint_dir: None,
            held_tags: None,
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
    /// The mask logits at this evaluation — so any iteration's mask can
    /// be picked after the run (`--mask-from` with a chosen checkpoint).
    pub logits: Vec<Vec<f32>>,
}

/// Trained attention + norm masters of a full-layer FCD layer (f32).
#[derive(Clone)]
pub struct AttnMasters {
    /// Fused `[qrows + 2·kvdim, hidden]` (q rows first, then k, then v).
    pub wqkv: Vec<f32>,
    /// `[hidden, nh·hd]`.
    pub wo: Vec<f32>,
    /// Per-head gate projection `[nh, hidden]` (Spark-X2.5), if the layer has one.
    pub gproj: Option<Vec<f32>>,
    pub iln: Vec<f32>,
    pub pln: Vec<f32>,
}

/// Their gradients (f64 accumulators), same shapes.
pub struct AttnGrads {
    pub wqkv: Vec<f64>,
    pub wo: Vec<f64>,
    pub gproj: Option<Vec<f64>>,
    pub iln: Vec<f64>,
    pub pln: Vec<f64>,
}

impl AttnGrads {
    fn zeros_like(m: &AttnMasters) -> Self {
        Self {
            wqkv: vec![0.0; m.wqkv.len()],
            wo: vec![0.0; m.wo.len()],
            gproj: m.gproj.as_ref().map(|g| vec![0.0; g.len()]),
            iln: vec![0.0; m.iln.len()],
            pln: vec![0.0; m.pln.len()],
        }
    }
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
    /// Trained attention/norm masters (full-layer FCD), per layer.
    pub attn: Vec<Option<AttnMasters>>,
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

/// The scored target indices (into the flat batch) of the sequence at
/// `base..base + t`: its answer span, its focused labels, or every
/// position.
fn scored_targets(
    ids: &[u32],
    base: usize,
    t: usize,
    focus: &[u32],
    follow: &[u32],
    span: Option<&AnswerSpan>,
) -> Vec<usize> {
    match span {
        Some(sp) => match answer_targets(&ids[base..base + t], sp) {
            Some((first, last)) => (base + first.max(1)..=base + last).collect(),
            None => Vec::new(),
        },
        None => (base + 1..base + t)
            .filter(|&target_index| is_scored_target(ids, target_index, base + t, focus, follow))
            .collect(),
    }
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
    /// Answer-span scoring (see `AnswerSpan`); None = focus/all-token.
    span: Option<&'a AnswerSpan>,
    /// `hard` gates keep σ(m) above τ instead of snapping to 1 (Soft-Fold).
    soft_fold: bool,
    /// Full-layer FCD masters per layer (attention + norms); None = frozen.
    attn_tr: &'a [Option<AttnMasters>],
}

/// Raw-pointer carriers for pooled row writes (disjoint ranges per worker).
struct SendMutF(*mut f32);
unsafe impl Send for SendMutF {}
unsafe impl Sync for SendMutF {}
impl SendMutF {
    #[inline]
    unsafe fn at(&self, i: usize) -> *mut f32 {
        unsafe { self.0.add(i) }
    }
}
struct SendMutF64(*mut f64);
unsafe impl Send for SendMutF64 {}
unsafe impl Sync for SendMutF64 {}
impl SendMutF64 {
    #[inline]
    unsafe fn at(&self, i: usize) -> *mut f64 {
        unsafe { self.0.add(i) }
    }
}

#[derive(Clone, Debug, Default)]
struct FocusStats {
    total: usize,
    correct: usize,
    class_total: Vec<usize>,
    class_correct: Vec<usize>,
    /// Answer-span verdicts (`answer_values`) and those predicted whole.
    questions: usize,
    questions_correct: usize,
    /// (verdicts, right) per sequence of the LAST scored call — the held
    /// scorer attributes them to the records' tags; never merged.
    seq: Vec<(usize, usize)>,
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

    fn question_accuracy(&self) -> Option<f64> {
        (self.questions > 0).then(|| self.questions_correct as f64 / self.questions as f64)
    }

    /// The held score's accuracy pair: whole-verdict accuracy when the span
    /// located values (then the token accuracy rides along for the log),
    /// otherwise the token/class accuracy alone.
    fn accuracies(&self) -> (Option<f64>, Option<f64>) {
        match self.question_accuracy() {
            Some(q) => (Some(q), self.accuracy()),
            None => (self.accuracy(), None),
        }
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
        self.questions += other.questions;
        self.questions_correct += other.questions_correct;
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
    /// Whole-verdict accuracy in answer-span mode with located values,
    /// else token (span) or class (focus) accuracy.
    accuracy: Option<f64>,
    balanced_accuracy: Option<f64>,
    /// The span's token accuracy when `accuracy` is the verdict accuracy.
    token_accuracy: Option<f64>,
    /// Verdict accuracy over all questions when `accuracy` is the macro
    /// over held tags (families); None otherwise.
    micro_accuracy: Option<f64>,
    /// (tag, verdicts, right) per held tag, in tag order.
    per_tag: Vec<(String, usize, usize)>,
}

/// Held records' tags (e.g. the benchmark family of each validation
/// record): the names, and one name index per held record.
#[derive(Clone, Debug, Default)]
pub struct HeldTags {
    pub names: Vec<String>,
    pub of: Vec<usize>,
}

/// " | <tag> 61.0 (41)" per tag — the macro gate's ingredients.
fn tag_line(score: &HeldScore) -> Option<String> {
    (!score.per_tag.is_empty()).then(|| {
        score
            .per_tag
            .iter()
            .map(|(name, n, ok)| {
                format!(
                    "{name} {:.1} ({n})",
                    if *n > 0 { *ok as f64 * 100.0 / *n as f64 } else { f64::NAN }
                )
            })
            .collect::<Vec<_>>()
            .join(" | ")
    })
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
                    if s > self.tau {
                        if self.soft_fold { s } else { 1.0 }
                    } else {
                        0.0
                    }
                } else {
                    s
                }
            })
            .collect()
    }

    fn wts<'b>(&'b self, li: usize, mats: &'b crate::fcd::LayerMats) -> LnFfn<'b> {
        let l = &self.fm.layers[li];
        let (iln, pln): (&[f32], &[f32]) = match &self.attn_tr[li] {
            Some(m) => (&m.iln, &m.pln),
            None => (&l.iln, &l.pln),
        };
        match &self.ffn[li] {
            Some((g, u, d)) => LnFfn {
                iln,
                pln,
                gate: g,
                up: u,
                down: d,
                // Trained copies move every Adam step — no prebuilt concat.
                gu: None,
            },
            None => LnFfn {
                iln,
                pln,
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
                true,
                None,
                false,
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
            &mut [Option<AttnGrads>],
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
            &mut [Option<AttnGrads>],
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
            &mut [Option<AttnGrads>],
        )>,
        mut focus_stats: Option<&mut FocusStats>,
    ) -> (f64, usize) {
        let fm = self.fm;
        let hsz = fm.hidden;
        let pool = fm.pool.as_deref();
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
            // A frozen layer's backward runs on the card from the plane the
            // forward parks there; only a trained (Phase-B) layer needs its
            // gate/up/act planes on the host.
            let planes_host = !(want && self.ffn[li].is_none());
            let attn_w = self.attn_tr[li].as_ref().map(|m| AttnW {
                wqkv: &m.wqkv,
                wo: &m.wo,
                head_gate: m.gproj.as_deref(),
            });
            let (h2, a) = fm.layer_forward_scaled(
                li,
                &h,
                b,
                t,
                &wts,
                false,
                want,
                Some(&g),
                planes_host,
                attn_w,
                want && attn_w.is_some(),
            );
            h_ins.push(if want { h } else { Vec::new() });
            acts.push(a);
            masks.push(g);
            h = h2;
            // Mid-stack norm at every loop boundary except the last —
            // the final one folds into the head below.
            if fm.loop_norm && li + 1 == nl && vl + 1 < vn {
                let mut hn = vec![0f32; n * hsz];
                let mut inv = vec![0f32; n];
                rmsnorm_fwd_pooled(&h, &fm.final_norm, fm.eps, fm.gemma, &mut hn, &mut inv, pool);
                if want {
                    lnorms[vl] = Some((h, inv));
                }
                h = hn;
            }
        }
        // Final norm + tied LM head, CE summed over positions 1..t.
        let mut hn = vec![0f32; n * hsz];
        let mut inv = vec![0f32; n];
        rmsnorm_fwd_pooled(&h, &fm.final_norm, fm.eps, fm.gemma, &mut hn, &mut inv, pool);
        let lm: &[f32] = fm.lm_head.as_deref().unwrap_or(&fm.embed);
        let vocab = lm.len() / hsz;
        let mut nll = 0f64;
        let mut dh_n = vec![0f32; n * hsz]; // dL/d hn
        // Chunk the vocab matmul over positions to bound the logits buf.
        // Positions are walked PER SEQUENCE: the last position of chunk
        // i must not be scored against the first token of chunk i+1.
        const POS_CHUNK: usize = 64;
        let targets: Vec<usize> = (0..b)
            .flat_map(|bi| {
                scored_targets(
                    ids,
                    bi * t,
                    t,
                    self.focus_tokens,
                    self.focus_follow_tokens,
                    self.span,
                )
            })
            .collect();
        let scored = targets.len();
        if scored == 0 {
            return (0.0, 0);
        }
        if self.focus_tokens.is_empty() {
            // Ordinary language-model mode: the complete vocabulary
            // distribution at every scored target — every position, or
            // the answer span of each record. The scored rows are gathered
            // so a span pays for its own positions only, not the prompt's.
            let inv_n = 1.0 / scored as f64;
            // Argmax hit per scored target, for the whole-verdict count.
            let mut hits: Option<Vec<bool>> = (focus_stats.is_some()
                && self.span.is_some_and(|sp| sp.values.is_some()))
            .then(|| vec![false; scored]);
            // CMF_BAKE_QDEBUG=N prints gold vs argmax of the first N verdicts
            // of every held call (diagnosing the verdict locator).
            let qdebug: usize = std::env::var("CMF_BAKE_QDEBUG").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let mut preds: Vec<u32> = if qdebug > 0 && hits.is_some() { vec![0; scored] } else { Vec::new() };
            let mut p0 = 0usize;
            while p0 < scored {
                let pc = POS_CHUNK.min(scored - p0);
                let rows = &targets[p0..p0 + pc];
                let src: Vec<usize> = rows.iter().map(|&target_index| target_index - 1).collect();
                let hsrc = gather_rows(&hn, &src, hsz);
                let mut logits = vec![0f32; pc * vocab];
                ops::gemm_nt(&hsrc, lm, &mut logits, pc, hsz, vocab, pool);
                for (r, &target_index) in rows.iter().enumerate() {
                    let target = ids[target_index] as usize;
                    let row = &mut logits[r * vocab..(r + 1) * vocab];
                    let (mut mx, mut arg) = (f32::NEG_INFINITY, 0usize);
                    for (j, &v) in row.iter().enumerate() {
                        if v > mx {
                            mx = v;
                            arg = j;
                        }
                    }
                    let mx = mx as f64;
                    let mut sum = 0f64;
                    for v in row.iter() {
                        sum += ((*v as f64) - mx).exp();
                    }
                    nll += mx + sum.ln() - row[target] as f64;
                    // Teacher-forced token accuracy of the answer span — the
                    // one number a verdict corpus is really graded on.
                    if let (Some(stats), Some(_)) = (focus_stats.as_deref_mut(), self.span) {
                        stats.total += 1;
                        if arg == target {
                            stats.correct += 1;
                        }
                    }
                    if let Some(h) = hits.as_mut() {
                        h[p0 + r] = arg == target;
                    }
                    if !preds.is_empty() {
                        preds[p0 + r] = arg as u32;
                    }
                    if grad.is_some() {
                        // dCE/dlogit = softmax − onehot, scaled by 1/scored.
                        for v in row.iter_mut() {
                            *v = ((((*v as f64) - mx).exp() / sum) * inv_n) as f32;
                        }
                        row[target] -= inv_n as f32;
                    }
                }
                if grad.is_some() {
                    let mut dsrc = vec![0f32; pc * hsz];
                    ops::gemm_dx(&logits, lm, &mut dsrc, pc, hsz, vocab, pool);
                    for (r, &s) in src.iter().enumerate() {
                        for (d, &x) in dh_n[s * hsz..(s + 1) * hsz]
                            .iter_mut()
                            .zip(&dsrc[r * hsz..(r + 1) * hsz])
                        {
                            *d += x;
                        }
                    }
                }
                p0 += pc;
            }
            // Whole verdicts: a question is right when every token of its
            // value is the argmax (what greedy decoding emits after the
            // same prefix).
            if let (Some(h), Some(stats), Some(sp)) = (&hits, focus_stats.as_deref_mut(), self.span) {
                if let Some(text) = &sp.values {
                    let at: std::collections::HashMap<usize, bool> =
                        targets.iter().copied().zip(h.iter().copied()).collect();
                    let pred_at: std::collections::HashMap<usize, u32> =
                        targets.iter().copied().zip(preds.iter().copied()).collect();
                    let mut shown = 0usize;
                    stats.seq = vec![(0, 0); b];
                    for bi in 0..b {
                        let seq = &ids[bi * t..(bi + 1) * t];
                        let Some((first, last)) = answer_targets(seq, sp) else {
                            continue;
                        };
                        if shown < qdebug {
                            let gold: Vec<String> = (first.max(1)..=last).map(|q| format!("{:?}", text.0.get(seq[q] as usize).map(String::as_str).unwrap_or("?"))).collect();
                            let got: Vec<String> = (first.max(1)..=last).map(|q| format!("{:?}", pred_at.get(&(bi * t + q)).and_then(|&id| text.0.get(id as usize)).map(String::as_str).unwrap_or("?"))).collect();
                            let groups = answer_values(seq, first.max(1), last, text);
                            eprintln!("[qdebug] gold {} | argmax {} | value groups {:?} (span starts at {})", gold.join(" "), got.join(" "), groups, first.max(1));
                            shown += 1;
                        }
                        for group in answer_values(seq, first.max(1), last, text) {
                            stats.questions += 1;
                            stats.seq[bi].0 += 1;
                            if group.iter().all(|&q| at.get(&(bi * t + q)).copied().unwrap_or(false)) {
                                stats.questions_correct += 1;
                                stats.seq[bi].1 += 1;
                            }
                        }
                    }
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
        let Some((dmask, dffn, dattn)) = grad else {
            return (nll, scored);
        };
        // Backward: final norm, then the FFN chain layer by layer.
        let t_bwd = std::time::Instant::now();
        let mut dh = vec![0f32; n * hsz];
        rmsnorm_bwd_pooled(&h, &fm.final_norm, &inv, &dh_n, fm.gemma, &mut dh, pool);
        for vl in (0..vn).rev() {
            let li = vl % nl;
            // Undo the loop-boundary norm this step fed into.
            if let Some((hb, inv)) = lnorms[vl].as_ref() {
                let mut dprev = vec![0f32; n * hsz];
                rmsnorm_bwd_pooled(hb, &fm.final_norm, inv, &dh, fm.gemma, &mut dprev, pool);
                dh = dprev;
            }
            let a = acts[vl].as_ref().expect("acts saved in grad mode");
            let g = &masks[vl];
            let inter = fm.layers[li].inter;
            let mats_hold = fm.mats(li).expect("layer mats");
            let wts = self.wts(li, &mats_hold);
            let mut dn2 = vec![0f32; n * hsz];
            let mut device_done = false;
            #[cfg(feature = "gpu")]
            if a.act.is_empty() && dffn[li].is_none() {
                if let Some(gu) = wts.gu {
                    // Phase A on the card: dact = dh·down, the mask gradient
                    // from the resident plane, dgu through the act link
                    // with the gate folded in, dn2 = dgu·gu. Only dn2 and
                    // dm[inter] come back.
                    let mut dm_dev = vec![0f32; inter];
                    if crate::gpu_wgpu::ffn_bwd_chain_f32(
                        &dh, wts.down, gu, li, &mut dn2, n, hsz, inter, fm.act, Some(g), Some(&mut dm_dev),
                    ) {
                        for (d, &x) in dmask[vl].iter_mut().zip(&dm_dev) {
                            *d += x as f64;
                        }
                        device_done = true;
                    }
                }
            }
            if !device_done {
                assert!(
                    !a.act.is_empty(),
                    "layer {li}: the forward left its FFN plane on the device but the device backward declined"
                );
                // h2 = h1 + act2 @ downᵀ  →  dact2 = dh @ down.
                let mut dact2 = vec![0f32; n * inter];
                ops::gemm_dx(&dh, wts.down, &mut dact2, n, inter, hsz, fm.pool.as_deref());
                if let Some((_, _, dd)) = dffn[li].as_mut() {
                    // dW_down += dhᵀ · act2 (act2 = act·g).
                    let mut act2 = a.act.clone();
                    {
                        let ap = SendMutF(act2.as_mut_ptr());
                        pooled_rows(n, pool, &|lo, hi| {
                            // SAFETY: disjoint row ranges per worker.
                            let rows = unsafe { std::slice::from_raw_parts_mut(ap.at(lo * inter), (hi - lo) * inter) };
                            for row in rows.chunks_exact_mut(inter) {
                                for (x, &gv) in row.iter_mut().zip(g) {
                                    *x *= gv;
                                }
                            }
                        });
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
                    // Parallel over neuron ranges: each worker owns a column
                    // span of dm and walks every row for it.
                    let dmp = SendMutF64(dm.as_mut_ptr());
                    let (dact2_r, act_r) = (&dact2, &a.act);
                    pooled_rows(inter.div_ceil(256), pool, &|lo, hi| {
                        let (js, je) = (lo * 256, (hi * 256).min(inter));
                        // SAFETY: disjoint column ranges per worker.
                        let dmj = unsafe { std::slice::from_raw_parts_mut(dmp.at(js), je - js) };
                        for r in 0..n {
                            let da = &dact2_r[r * inter + js..r * inter + je];
                            let aa = &act_r[r * inter + js..r * inter + je];
                            for ((d, &x), &y) in dmj.iter_mut().zip(da).zip(aa) {
                                *d += x as f64 * y as f64;
                            }
                        }
                    });
                    // σ'(m) folded in once per chunk (constant per neuron).
                    for (j, d) in dm.iter_mut().enumerate() {
                        let _ = j;
                        let _ = d;
                    }
                }
                // dact = dact2 · g;  act·mul backward (pooled).
                let mut dg_pre = vec![0f32; n * inter];
                let mut du_pre = vec![0f32; n * inter];
                act_bwd_rows(
                    fm.act,
                    &a.gpre,
                    &a.upre,
                    &dact2,
                    Some(g),
                    &mut dg_pre,
                    &mut du_pre,
                    n,
                    inter,
                    fm.pool.as_deref(),
                );
                // dn2 = dg_pre @ gate + du_pre @ up — one fused submit when
                // the frozen concat exists; the trained-copy path keeps two.
                if let Some(gu) = wts.gu {
                    let mut dgu = vec![0f32; n * 2 * inter];
                    {
                        let gp = SendMutF(dgu.as_mut_ptr());
                        let (dgr, dur) = (&dg_pre, &du_pre);
                        pooled_rows(n, pool, &|lo, hi| {
                            for r in lo..hi {
                                // SAFETY: disjoint rows per worker.
                                let row = unsafe { std::slice::from_raw_parts_mut(gp.at(r * 2 * inter), 2 * inter) };
                                row[..inter].copy_from_slice(&dgr[r * inter..(r + 1) * inter]);
                                row[inter..].copy_from_slice(&dur[r * inter..(r + 1) * inter]);
                            }
                        });
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
            }
            // Post-norm backward into h1; a frozen attention branch carries
            // no gradient, so dh1 flows straight to dh_in.
            let mut dh1 = dh.clone(); // residual h2 = h1 + ffn
            rmsnorm_bwd_pooled(&a.h1, wts.pln, &a.inv2, &dn2, fm.gemma, &mut dh1, pool);
            if let (Some(m), Some(dg)) = (&self.attn_tr[li], dattn[li].as_mut()) {
                // Full-layer FCD: through the trained attention — post-norm
                // gain, o_proj, the head gate, the attention core (device),
                // RoPE, q/k/v projections, the input norm — into dh_in.
                norm_gain_grad(&a.h1, &a.inv2, &dn2, &mut dg.pln, pool);
                let dh_attn = self.attn_backward(li, vl, &h_ins[vl], a, &dh1, m, dg, b, t);
                add_into_pooled(&mut dh1, &dh_attn, pool);
            }
            dh = dh1;
        }
        crate::fcd::prof::add(&crate::fcd::prof::BWD, t_bwd);
        (nll, scored)
    }
}


/// dw[j] = Σ_r dy[r,j]·x[r,j]·inv[r] — the RMSNorm gain gradient (Qwen
/// style), pooled over column spans.
fn norm_gain_grad(x: &[f32], inv: &[f32], dy: &[f32], dw: &mut [f64], pool: Option<&crate::pool::Pool>) {
    let d = dw.len();
    let n = x.len() / d;
    let dp = SendMutF64(dw.as_mut_ptr());
    pooled_rows(d.div_ceil(64), pool, &|lo, hi| {
        let (js, je) = (lo * 64, (hi * 64).min(d));
        // SAFETY: disjoint column spans per worker.
        let out = unsafe { std::slice::from_raw_parts_mut(dp.at(js), je - js) };
        for r in 0..n {
            let inv_r = inv[r] as f64;
            for (o, j) in out.iter_mut().zip(js..je) {
                *o += dy[r * d + j] as f64 * x[r * d + j] as f64 * inv_r;
            }
        }
    });
}

impl Pass<'_> {
    /// The attention branch's backward for a full-layer FCD layer: from
    /// `dattn` (the gradient at attn_out) to the gradient at h_in through
    /// the attention path, accumulating the trained weights' gradients.
    #[allow(clippy::too_many_arguments)]
    fn attn_backward(
        &self,
        li: usize,
        _vl: usize,
        h_in: &[f32],
        a: &crate::fcd::LayerActs,
        dattn: &[f32],
        m: &AttnMasters,
        dg: &mut AttnGrads,
        b: usize,
        t: usize,
    ) -> Vec<f32> {
        let fm = self.fm;
        let pool = fm.pool.as_deref();
        let (hsz, nh, nkv, hd) = (fm.hidden, fm.nh, fm.nkv, fm.hd);
        let n = b * t;
        let (qdim, kvdim) = (nh * hd, nkv * hd);
        let (window, local) = fm.attn_geometry(li).expect("full-layer FCD on a non-attention layer");
        let (qrot, krot, vproj, ao, head_gain) = a
            .attn_full()
            .expect("full-layer FCD needs the attention activations");
        // n1: the normed input the projections read (recomputed from h_in).
        let mut n1 = vec![0f32; n * hsz];
        let mut inv1 = vec![0f32; n];
        rmsnorm_fwd_pooled(h_in, &m.iln, fm.eps, fm.gemma, &mut n1, &mut inv1, pool);
        // ao_eff = ao ⊙ gain; dWo += dattnᵀ·ao_eff; dao_eff = dattn·Wo.
        let mut ao_eff = ao.to_vec();
        if !head_gain.is_empty() {
            for r in 0..n {
                for h in 0..nh {
                    let gain = head_gain[r * nh + h];
                    for x in &mut ao_eff[r * qdim + h * hd..r * qdim + (h + 1) * hd] {
                        *x *= gain;
                    }
                }
            }
        }
        {
            let mut dw = vec![0f32; hsz * qdim];
            ops::gemm_dw(dattn, &ao_eff, &mut dw, n, qdim, hsz, pool);
            for (o, &x) in dg.wo.iter_mut().zip(&dw) {
                *o += x as f64;
            }
        }
        let mut dao_eff = vec![0f32; n * qdim];
        ops::gemm_dx(dattn, &m.wo, &mut dao_eff, n, qdim, hsz, pool);
        // Head gate: dao = dao_eff·gain; dgain = Σ_c dao_eff·ao; dgraw = dgain·σ′.
        let mut dn1 = vec![0f32; n * hsz];
        let mut dao = dao_eff;
        if let (Some(gw), Some(dgw)) = (&m.gproj, dg.gproj.as_mut()) {
            let mut dgraw = vec![0f32; n * nh];
            for r in 0..n {
                for h in 0..nh {
                    let gain = head_gain[r * nh + h];
                    let s = r * qdim + h * hd;
                    let mut dgain = 0f32;
                    for c in 0..hd {
                        dgain += dao[s + c] * ao[s + c];
                        dao[s + c] *= gain;
                    }
                    dgraw[r * nh + h] = dgain * gain * (1.0 - gain);
                }
            }
            let mut dw = vec![0f32; nh * hsz];
            ops::gemm_dw(&dgraw, &n1, &mut dw, n, hsz, nh, pool);
            for (o, &x) in dgw.iter_mut().zip(&dw) {
                *o += x as f64;
            }
            ops::gemm_dx(&dgraw, gw, &mut dn1, n, hsz, nh, pool);
        }
        // Attention core: dq/dk (rotated space), dv — device chain, host fallback.
        let mut dqrot = vec![0f32; n * qdim];
        let mut dkrot = vec![0f32; n * kvdim];
        let mut dvproj = vec![0f32; n * kvdim];
        let mut done = false;
        #[cfg(feature = "gpu")]
        {
            done = crate::gpu_wgpu::attn_bwd_chain_f32(
                qrot, krot, vproj, &dao, b, t, nh, nkv, hd, window, &mut dqrot, &mut dkrot, &mut dvproj,
            );
        }
        if !done {
            let rep = nh / nkv;
            for bi in 0..b {
                for h in 0..nh {
                    let g = h / rep;
                    let mut q64 = vec![0f64; t * hd];
                    let mut k64 = vec![0f64; t * hd];
                    let mut v64 = vec![0f64; t * hd];
                    let mut do64 = vec![0f64; t * hd];
                    for p in 0..t {
                        let r = bi * t + p;
                        for c in 0..hd {
                            q64[p * hd + c] = qrot[r * qdim + h * hd + c] as f64;
                            k64[p * hd + c] = krot[r * kvdim + g * hd + c] as f64;
                            v64[p * hd + c] = vproj[r * kvdim + g * hd + c] as f64;
                            do64[p * hd + c] = dao[r * qdim + h * hd + c] as f64;
                        }
                    }
                    let mut dq = vec![0f64; t * hd];
                    let mut dk = vec![0f64; t * hd];
                    let mut dv = vec![0f64; t * hd];
                    ops::attn_head_bwd_window(&q64, &k64, &v64, &do64, t, hd, hd, window, &mut dq, &mut dk, &mut dv);
                    for p in 0..t {
                        let r = bi * t + p;
                        for c in 0..hd {
                            dqrot[r * qdim + h * hd + c] += dq[p * hd + c] as f32;
                            dkrot[r * kvdim + g * hd + c] += dk[p * hd + c] as f32;
                            dvproj[r * kvdim + g * hd + c] += dv[p * hd + c] as f32;
                        }
                    }
                }
            }
        }
        // RoPE backward (no qk-norm on these layers), per head.
        let (rotary_dim, inv_freq) = fm.layer_rope(local);
        for r in 0..n {
            let pos = r % t;
            for h in 0..nh {
                let s = r * qdim + h * hd;
                ops::rope_bwd(&mut dqrot[s..s + rotary_dim], pos, inv_freq);
            }
            for g in 0..nkv {
                let s = r * kvdim + g * hd;
                ops::rope_bwd(&mut dkrot[s..s + rotary_dim], pos, inv_freq);
            }
        }
        // dqkv = [dq | dk | dv]; dWqkv += dqkvᵀ·n1; dn1 += dqkv·Wqkv.
        let fused = qdim + 2 * kvdim;
        let mut dqkv = vec![0f32; n * fused];
        for r in 0..n {
            let row = &mut dqkv[r * fused..(r + 1) * fused];
            row[..qdim].copy_from_slice(&dqrot[r * qdim..(r + 1) * qdim]);
            row[qdim..qdim + kvdim].copy_from_slice(&dkrot[r * kvdim..(r + 1) * kvdim]);
            row[qdim + kvdim..].copy_from_slice(&dvproj[r * kvdim..(r + 1) * kvdim]);
        }
        {
            let mut dw = vec![0f32; fused * hsz];
            ops::gemm_dw(&dqkv, &n1, &mut dw, n, hsz, fused, pool);
            for (o, &x) in dg.wqkv.iter_mut().zip(&dw) {
                *o += x as f64;
            }
        }
        {
            let mut dx = vec![0f32; n * hsz];
            ops::gemm_dx(&dqkv, &m.wqkv, &mut dx, n, hsz, fused, pool);
            add_into_pooled(&mut dn1, &dx, pool);
        }
        // Input norm: gain gradient + through-grad into h_in.
        norm_gain_grad(h_in, &inv1, &dn1, &mut dg.iln, pool);
        let mut dh_in = vec![0f32; n * hsz];
        rmsnorm_bwd_pooled(h_in, &m.iln, &inv1, &dn1, fm.gemma, &mut dh_in, pool);
        dh_in
    }
}

/// Held-out PPL with the hard mask (and Phase-B weights when present).
/// The device chains' phase timers (0 weights, 1 submit→poll, 2 copies);
/// zero without the gpu feature.
fn chain_prof(i: usize) -> f64 {
    #[cfg(feature = "gpu")]
    {
        crate::gpu_wgpu::chain_prof_take(i)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = i;
        0.0
    }
}

/// " | acc X% bal Y%" (focused classes), " | tok-acc X%" (answer span),
/// or nothing (all-token LM).
fn score_suffix(score: &HeldScore) -> String {
    match (score.accuracy, score.balanced_accuracy) {
        (Some(a), Some(b)) => format!(" | acc {:.2}% bal {:.2}%", a * 100.0, b * 100.0),
        (Some(a), None) => match (score.micro_accuracy, score.token_accuracy) {
            (Some(m), Some(t)) => format!(
                " | q-acc macro {:.2}% micro {:.2}% tok {:.2}%",
                a * 100.0,
                m * 100.0,
                t * 100.0
            ),
            (None, Some(t)) => format!(" | q-acc {:.2}% tok-acc {:.2}%", a * 100.0, t * 100.0),
            _ => format!(" | tok-acc {:.2}%", a * 100.0),
        },
        _ => String::new(),
    }
}

fn held_ppl(pass: &Pass, held: &[Vec<u32>]) -> f64 {
    held_score(pass, held).ppl
}

fn held_score(pass: &Pass, held: &[Vec<u32>]) -> HeldScore {
    held_score_tagged(pass, held, None)
}

/// `held_score` with per-tag verdict counts: when the records carry tags
/// and the span locates values, the score's accuracy is the MACRO verdict
/// accuracy over tags (each family weighs the same, as in a benchmark
/// index), the micro one rides along.
fn held_score_tagged(pass: &Pass, held: &[Vec<u32>], tags: Option<&HeldTags>) -> HeldScore {
    // Bounded batched passes over the held set: equal-length records share
    // one GEMM per weight, while the cap keeps activation/GPU scratch memory
    // predictable for a full validation sweep. Aggregation is exact because
    // NLL and focused-class counts are additive across groups.
    if held.is_empty() {
        return HeldScore {
            ppl: f64::NAN,
            accuracy: None,
            balanced_accuracy: None,
            token_accuracy: None,
            micro_accuracy: None,
            per_tag: Vec::new(),
        };
    }
    let mut nll = 0f64;
    let mut n = 0usize;
    let mut stats = FocusStats::new(pass.focus_tokens.len());
    // Up to 32 records per pass, bounded by tokens: 32 x 3072-token chat
    // records made a 98k-row pass whose gate+up plane (n x 2*inter f32)
    // crossed the 2 GiB device binding limit and panicked wgpu.
    // 8k tokens keeps a 4 B model's gate+up plane (n × 20480 f32) under
    // the 2 GiB binding ceiling once the scratch rounds up; CMF_BAKE_EVAL_TOKENS
    // overrides.
    const GROUP_MAX: usize = 32;
    let group_tokens: usize = std::env::var("CMF_BAKE_EVAL_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8_192);
    let groups: Vec<&[Vec<u32>]> = {
        let mut v = Vec::new();
        let mut i = 0;
        while i < held.len() {
            let len = held[i].len();
            let cap = (group_tokens / len.max(1)).clamp(1, GROUP_MAX);
            let mut j = i;
            while j < held.len() && held[j].len() == len && j - i < cap {
                j += 1;
            }
            v.push(&held[i..j]);
            i = j;
        }
        v
    };
    let mut per_tag: Vec<(usize, usize)> = vec![(0, 0); tags.map_or(0, |t| t.names.len())];
    let mut credit = |at: usize, part: &FocusStats| {
        if let Some(tg) = tags {
            for (j, &(q, ok)) in part.seq.iter().enumerate() {
                if let Some(&ti) = tg.of.get(at + j) {
                    per_tag[ti].0 += q;
                    per_tag[ti].1 += ok;
                }
            }
        }
    };
    let mut at = 0usize;
    for group in groups {
        let t = group[0].len();
        if group.iter().all(|c| c.len() == t) {
            let flat: Vec<u32> = group.iter().flatten().copied().collect();
            let mut part = FocusStats::new(pass.focus_tokens.len());
            let (l, k) = pass.chunk_batch_scored(&flat, group.len(), None, Some(&mut part));
            nll += l;
            n += k;
            credit(at, &part);
            stats.merge(&part);
        } else {
            for (j, c) in group.iter().enumerate() {
                let mut part = FocusStats::new(pass.focus_tokens.len());
                let (l, k) = pass.chunk_batch_scored(c, 1, None, Some(&mut part));
                nll += l;
                n += k;
                credit(at + j, &part);
                stats.merge(&part);
            }
        }
        at += group.len();
    }
    let (mut accuracy, token_accuracy) = stats.accuracies();
    let mut micro_accuracy = None;
    let mut tagged = Vec::new();
    if let Some(tg) = tags
        && token_accuracy.is_some()
    {
        let rates: Vec<f64> =
            per_tag.iter().filter(|(q, _)| *q > 0).map(|&(q, ok)| ok as f64 / q as f64).collect();
        if !rates.is_empty() {
            micro_accuracy = accuracy;
            accuracy = Some(rates.iter().sum::<f64>() / rates.len() as f64);
            tagged = tg.names.iter().cloned().zip(per_tag).map(|(name, (q, ok))| (name, q, ok)).collect();
        }
    }
    HeldScore {
        ppl: (nll / n.max(1) as f64).exp(),
        accuracy,
        balanced_accuracy: stats.balanced_accuracy(),
        token_accuracy,
        micro_accuracy,
        per_tag: tagged,
    }
}

/// The calibration records as runs of equal length, each cut into blocks
/// of at most `batch` records: `(start, count)`. Records are expected
/// sorted by length (stable, so the shuffle within a length survives);
/// the batch at `step` is `blocks[step % blocks.len()]`.
fn length_blocks(calib: &[Vec<u32>], batch: usize) -> Vec<(usize, usize)> {
    let batch = batch.max(1);
    let mut blocks = Vec::new();
    let mut i = 0;
    while i < calib.len() {
        let len = calib[i].len();
        let mut j = i;
        while j < calib.len() && calib[j].len() == len && j - i < batch {
            j += 1;
        }
        blocks.push((i, j - i));
        i = j;
    }
    blocks
}

fn calibration_batch(
    calib: &[Vec<u32>],
    step: usize,
    requested: usize,
) -> Result<(Vec<u32>, usize), String> {
    let batch = requested.max(1).min(calib.len());
    // Equal-length records only: with length buckets the records are
    // sorted by length and the blocks never mix widths.
    let blocks = length_blocks(calib, batch);
    let (start, count) = blocks[step % blocks.len()];
    let width = calib[start].len();
    let mut flat = Vec::with_capacity(count * width);
    for record in &calib[start..start + count] {
        if record.len() != width {
            return Err(format!(
                "skill bake: --batch needs equal-length records ({} != {width})",
                record.len()
            ));
        }
        flat.extend_from_slice(record);
    }
    Ok((flat, count))
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
                token_accuracy: None,
                micro_accuracy: None,
                per_tag: Vec::new(),
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
            act[at] = fm.act.fwd(gate_pre[at]) * up_pre[at] * gate[column];
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
        token_accuracy: None,
        micro_accuracy: None,
        per_tag: Vec::new(),
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
            dg_pre[at] = da * up_pre[at] * fm.act.bwd(gate_pre[at]);
            du_pre[at] = da * fm.act.fwd(gate_pre[at]);
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
    let no_attn: Vec<Option<AttnMasters>> = vec![None; ffn.len()];
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
            span: None,
            soft_fold: false,
            attn_tr: &no_attn,
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
    let mut fm = FcdModel::from_cmf(model, &o1_off, false)?;
    // The bake's backward never enters attention (frozen, residual
    // through-path only), so the attention activations are dead weight:
    // on the device chain they were ~700 MB of PCIe readback per layer.
    fm.attn_acts = false;
    let nl = fm.layers.len();
    // Full-layer FCD masters (attention + norms), filled before Phase B;
    // every Pass carries the slice so a trained layer's forward reads them.
    let mut attn_tr: Vec<Option<AttnMasters>> = vec![None; nl];
    let inter = fm.layers.iter().map(|l| l.inter).max().unwrap_or(0);
    // Stable sorts by length: with --length-buckets the batches and the
    // held-out groups stay homogeneous while the shuffle within a length
    // survives; without buckets every record has the chunk width and
    // nothing moves.
    let mut held: Vec<Vec<u32>> = chunks[..held_n.min(chunks.len())].to_vec();
    // The tags follow their records through the stable length sort.
    let mut order: Vec<usize> = (0..held.len()).collect();
    order.sort_by_key(|&i| held[i].len());
    let held_tags: Option<HeldTags> = hy.held_tags.as_ref().map(|t| HeldTags {
        names: t.names.clone(),
        of: order.iter().map(|&i| t.of.get(i).copied().unwrap_or(0)).collect(),
    });
    held = order.iter().map(|&i| std::mem::take(&mut held[i])).collect();
    let mut calib: Vec<Vec<u32>> = chunks[held_n.min(chunks.len())..].to_vec();
    calib.sort_by_key(Vec::len);
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
    // (A supplied mask is not an identity run: it goes through the scored
    // path below so the mask is evaluated and written.)
    if hy.steps_a == 0 && hy.steps_b == 0 && hy.fcd_layers == 0 && hy.mask_logits.is_none() {
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
            span: hy.answer_span.as_ref(),
            soft_fold: hy.soft_fold,
            attn_tr: &attn_tr,
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
            attn: (0..nl).map(|_| None).collect(),
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
    let mut logits: Vec<Vec<f32>> = match &hy.mask_logits {
        Some(m) => {
            if m.len() != vn || m.iter().any(|row| row.len() != inter) {
                return Err(format!(
                    "skill bake: --mask-from has {} rows of {} logits; this model needs {vn} × {inter}",
                    m.len(),
                    m.first().map(Vec::len).unwrap_or(0)
                ));
            }
            log(&format!("mask: resuming from {} supplied logit rows", m.len()));
            m.clone()
        }
        None => vec![vec![m0; inter]; vn],
    };
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
        span: hy.answer_span.as_ref(),
            soft_fold: hy.soft_fold,
            attn_tr: &attn_tr,
    };
    // CMF_BAKE_SKIP_BASELINE=1 with a supplied mask: a checkpoint
    // measurement that already knows the backbone's score skips its pass.
    let skip_baseline = hy.mask_logits.is_some()
        && std::env::var("CMF_BAKE_SKIP_BASELINE").is_ok_and(|v| v == "1");
    let backbone_score = if skip_baseline {
        HeldScore {
            ppl: f64::NAN,
            accuracy: None,
            balanced_accuracy: None,
            token_accuracy: None,
            micro_accuracy: None,
            per_tag: Vec::new(),
        }
    } else {
        held_score_tagged(&base_pass, &held, held_tags.as_ref())
    };
    let backbone = backbone_score.ppl;
    log(&format!(
        "baseline (full): {backbone:.3}{}",
        score_suffix(&backbone_score)
    ));
    if let Some(line) = tag_line(&backbone_score) {
        log(&format!("    per tag: {line}"));
    }
    // A resumed mask is scored before it trains on: the run's own start,
    // and with --steps-a 0 a held-out measurement of any checkpoint.
    if hy.mask_logits.is_some() {
        let start_pass = Pass {
            fm: &fm,
            tau: hy.tau,
            logits: &logits,
            hard: true,
            ffn: &ffn,
            focus_tokens: &hy.focus_tokens,
            focus_follow_tokens: &hy.focus_follow_tokens,
            span: hy.answer_span.as_ref(),
            soft_fold: hy.soft_fold,
            attn_tr: &attn_tr,
        };
        let s0 = held_score_tagged(&start_pass, &held, held_tags.as_ref());
        log(&format!("supplied mask: {:.3}{}", s0.ppl, score_suffix(&s0)));
        if let Some(line) = tag_line(&s0) {
            log(&format!("    per tag: {line}"));
        }
    }

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
    // CMF_BAKE_A_STRICT=0 lifts it: Phase A then rides the tensor-core
    // chains like Phase B (tf32-class operands) — the choice of a bake
    // that measures its mask on the held-out set anyway and cannot
    // afford a CPU-only phase on a 4 B model.
    let strict_a = std::env::var("CMF_BAKE_A_STRICT")
        .map(|v| v != "0")
        .unwrap_or(true);
    crate::gpu::bake_precision_strict(strict_a);
    for step in 0..hy.steps_a {
        let t_step = std::time::Instant::now();
        let (batch_ids, batch) = calibration_batch(&calib, step, hy.batch)?;
        let mut dmask: Vec<Vec<f64>> = vec![vec![0.0; inter]; vn];
        let mut dffn: Vec<Option<(Vec<f64>, Vec<f64>, Vec<f64>)>> = vec![None; nl];
        let mut dattn: Vec<Option<AttnGrads>> = (0..nl).map(|_| None).collect();
        let pass = Pass {
            fm: &fm,
            tau: hy.tau,
            logits: &logits,
            hard: false,
            ffn: &ffn,
            focus_tokens: &hy.focus_tokens,
            focus_follow_tokens: &hy.focus_follow_tokens,
            span: hy.answer_span.as_ref(),
            soft_fold: hy.soft_fold,
            attn_tr: &attn_tr,
        };
        let _ = pass.chunk_batch(&batch_ids, batch, Some((&mut dmask, &mut dffn, &mut dattn)));
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
                span: hy.answer_span.as_ref(),
            soft_fold: hy.soft_fold,
            attn_tr: &attn_tr,
            };
            crate::gpu::bake_precision_strict(false);
            let hs = held_score_tagged(&pass, &held, held_tags.as_ref());
            let hp = hs.ppl;
            crate::gpu::bake_precision_strict(strict_a);
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
                logits: logits.clone(),
            });
            if let Some(dir) = &hy.checkpoint_dir {
                let doc = serde_json::json!({
                    "schema": "cmf-bake-checkpoint-v1",
                    "step": step + 1,
                    "l1": l1,
                    "ppl": hp,
                    "sparsity": sp,
                    "accuracy": hs.accuracy,
                    "micro_accuracy": hs.micro_accuracy,
                    "token_accuracy": hs.token_accuracy,
                    "per_tag": hs.per_tag,
                    "selected_logits": &logits,
                });
                let path = format!("{dir}/phase-a-step-{}.json", step + 1);
                match serde_json::to_vec(&doc) {
                    Ok(bytes) => match std::fs::write(&path, bytes) {
                        Ok(()) => log(&format!("  [A] checkpoint written: {path}")),
                        Err(e) => log(&format!("  [A] checkpoint NOT written ({path}): {e}")),
                    },
                    Err(e) => log(&format!("  [A] checkpoint NOT serialised: {e}")),
                }
            }
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
            } else if eligible && hy.checkpoint_accuracy && hy.answer_span.is_some() {
                // Answer-span mode: the held-out accuracy of the hard mask
                // decides — whole verdicts when the span locates values,
                // else span tokens (the experiments' accuracy gate); PPL
                // breaks ties.
                let acc = hs.accuracy.unwrap_or(f64::NEG_INFINITY);
                if acc > best_accuracy || (acc == best_accuracy && hp < best.0) {
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
                score_suffix(&hs),
                if best.0 == f64::MAX {
                    "—".to_string()
                } else {
                    format!("{:.3}", best.0)
                },
                best.2 * 100.0,
                acc_chunk / (step + 1) as f64,
                acc_adam / (step + 1) as f64
            ));
            if let Some(line) = tag_line(&hs) {
                log(&format!("    per tag: {line}"));
            }
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
        log(&format!(
            "[prof] inside: attn-chain {:.1}s | ffn-pre {:.1}s | ffn-chain {:.1}s | ffn-act(split+act) {:.1}s | ffn-host {:.1}s | ffn-post {:.1}s | chain-w {:.1}s gpu {:.1}s copy {:.1}s",
            prof::take(&prof::ATTN_CHAIN),
            prof::take(&prof::FFN_PRE),
            prof::take(&prof::FFN_CHAIN),
            prof::take(&prof::FFN_ACT),
            prof::take(&prof::FFN_HOST),
            prof::take(&prof::FFN_POST),
            chain_prof(0),
            chain_prof(1),
            chain_prof(2),
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
        span: hy.answer_span.as_ref(),
            soft_fold: hy.soft_fold,
            attn_tr: &attn_tr,
    };
    // With no mask-training steps the hard mask is still exactly all-open,
    // so rescoring the same validation batch is pure duplicate work.  This
    // matters for focused 512-token records where one gate can take minutes
    // on a laptop, and keeps short FCD sweeps practical without changing a
    // single number.
    let masked_score = if hy.steps_a == 0 && hy.mask_logits.is_none() {
        backbone_score.clone()
    } else {
        held_score_tagged(&pass, &held, held_tags.as_ref())
    };
    let masked = masked_score.ppl;
    log(&format!(
        "[A] {:.0}s: masked-PPL {masked:.3}{}",
        t0.elapsed().as_secs_f64(),
        score_suffix(&masked_score)
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
    if hy.fcd_attn {
        for &li in &fcd {
            if fm.attn_geometry(li).is_none() {
                return Err(format!("--fcd-attn: layer {li} is not a softmax-attention layer"));
            }
            let mats = fm.mats(li)?;
            attn_tr[li] = Some(AttnMasters {
                wqkv: mats.wqkv.clone(),
                wo: mats.wo.clone(),
                gproj: fm.head_gate_weights(li).map(<[f32]>::to_vec),
                iln: fm.layers[li].iln.clone(),
                pln: fm.layers[li].pln.clone(),
            });
        }
        log(&format!("[B] full-layer FCD: attention + norms of {} layer(s) train too", fcd.len()));
    }
    let sizes: Vec<usize> = fcd
        .iter()
        .flat_map(|&li| {
            let (g, u, d) = ffn[li].as_ref().expect("phase-B masters");
            let mut v = vec![g.len(), u.len(), d.len()];
            if let Some(m) = &attn_tr[li] {
                v.extend([m.wqkv.len(), m.wo.len(), m.gproj.as_ref().map_or(0, Vec::len), m.iln.len(), m.pln.len()]);
            }
            v
        })
        .filter(|&n| n > 0)
        .collect();
    let mut adam_b = Adam::new(&sizes, hy.lr_b);
    // The mask-only model is a real checkpoint too. If every FCD eval is
    // worse, restore `None` overlays rather than accidentally writing the
    // final (rejected) training step while reporting the mask-only PPL.
    let mut best_b: (
        HeldScore,
        Option<Vec<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>>>,
    ) = (masked_score.clone(), Some(vec![None; nl]));
    let mut best_attn: Vec<Option<AttnMasters>> = attn_tr.clone();
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
            span: hy.answer_span.as_ref(),
            soft_fold: hy.soft_fold,
            attn_tr: &attn_tr,
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
                    best_attn = attn_tr.clone();
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
            let mut dattn: Vec<Option<AttnGrads>> =
                attn_tr.iter().map(|m| m.as_ref().map(AttnGrads::zeros_like)).collect();
            let pass = Pass {
                fm: &fm,
                tau: hy.tau,
                logits: &logits,
                hard: true,
                ffn: &ffn,
                focus_tokens: &hy.focus_tokens,
                focus_follow_tokens: &hy.focus_follow_tokens,
                span: hy.answer_span.as_ref(),
            soft_fold: hy.soft_fold,
            attn_tr: &attn_tr,
            };
            let _ = pass.chunk_batch(&batch_ids, batch, Some((&mut dmask, &mut dffn, &mut dattn)));
            // Cosine LR.
            let lr_scale =
                0.5 * (1.0 + (std::f64::consts::PI * step as f64 / hy.steps_b as f64).cos());
            let first_fcd = fcd[0];
            let mut params: Vec<&mut [f32]> = Vec::new();
            let mut grads: Vec<Vec<f64>> = Vec::new();
            let attn_tr_slots = &mut attn_tr[first_fcd..];
            for (off, (slot, aslot)) in ffn[first_fcd..]
                .iter_mut()
                .zip(attn_tr_slots.iter_mut())
                .enumerate()
            {
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
                if let (Some(m), Some(ag)) = (aslot.as_mut(), dattn[li].take()) {
                    params.push(m.wqkv.as_mut_slice());
                    grads.push(ag.wqkv);
                    params.push(m.wo.as_mut_slice());
                    grads.push(ag.wo);
                    if let (Some(gp), Some(dgp)) = (m.gproj.as_mut(), ag.gproj) {
                        params.push(gp.as_mut_slice());
                        grads.push(dgp);
                    }
                    params.push(m.iln.as_mut_slice());
                    grads.push(ag.iln);
                    params.push(m.pln.as_mut_slice());
                    grads.push(ag.pln);
                }
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
                    span: hy.answer_span.as_ref(),
            soft_fold: hy.soft_fold,
            attn_tr: &attn_tr,
                };
                let cur = held_score_tagged(&pass, &held, held_tags.as_ref());
                let better = if hy.checkpoint_accuracy && hy.answer_span.is_some() {
                    let cur_acc = cur.accuracy.unwrap_or(f64::NEG_INFINITY);
                    let best_acc = best_b.0.accuracy.unwrap_or(f64::NEG_INFINITY);
                    cur_acc > best_acc || (cur_acc == best_acc && cur.ppl < best_b.0.ppl)
                } else if hy.checkpoint_accuracy && !hy.focus_tokens.is_empty() {
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
                // CMF_BAKE_B_TAKE_LAST=1: the correction the recipe prescribes
                // (Patent 2 §4.5: one light FCD step) ships as trained — a
                // held set of a few verdicts per family cannot veto it on noise;
                // its score is still logged.
                let take_last = std::env::var("CMF_BAKE_B_TAKE_LAST").is_ok_and(|v| v == "1");
                if better || (take_last && step + 1 == hy.steps_b) {
                    best_b = (cur.clone(), Some(ffn.clone()));
                    best_attn = attn_tr.clone();
                }
                log(&format!(
                    "  [B] step {}: held-PPL {:.3}{} (best {:.3})",
                    step + 1,
                    cur.ppl,
                    score_suffix(&cur),
                    best_b.0.ppl
                ));
                if let Some(line) = tag_line(&cur) {
                    log(&format!("    per tag: {line}"));
                }
            }
        }
    }
    ffn = best_b.1.take().expect("phase-B always has a checkpoint");
    attn_tr = best_attn;
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
        attn: attn_tr,
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
    fn answer_values_group_each_top_level_value() {
        // {"q1": "option_24", "q2": [ "a" ], "q3": 3}<end>, split the way a
        // byte-level BPE might: quotes merged with keys and labels.
        let pieces = [
            "{\"", "q1", "\":", " \"", "option", "_", "24", "\",", " \"q2", "\":", " [", " \"a\"", " ]", ",", " \"q3\":",
            " 3", "}", "<end>",
        ];
        let text = TokenText(Arc::new(pieces.iter().map(|p| p.to_string()).collect()));
        let ids: Vec<u32> = (0..pieces.len() as u32).collect();
        let groups = answer_values(&ids, 0, ids.len() - 1, &text);
        assert_eq!(groups, vec![vec![3, 4, 5, 6, 7], vec![10, 11, 12], vec![15]]);
    }

    #[test]
    fn answer_values_keep_escaped_quotes_inside_the_value() {
        let pieces = ["{\"q\": \"a", "\\\"", "b\"}", "<end>"];
        let text = TokenText(Arc::new(pieces.iter().map(|p| p.to_string()).collect()));
        let ids: Vec<u32> = (0..pieces.len() as u32).collect();
        assert_eq!(answer_values(&ids, 0, 3, &text), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn balanced_accuracy_exposes_majority_class_collapse() {
        let stats = FocusStats {
            total: 100,
            correct: 90,
            class_total: vec![90, 10],
            class_correct: vec![90, 0],
            ..FocusStats::default()
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
            questions: 2,
            questions_correct: 1,
            seq: Vec::new(),
        };
        let second = FocusStats {
            total: 4,
            correct: 3,
            class_total: vec![1, 3],
            class_correct: vec![1, 2],
            questions: 3,
            questions_correct: 3,
            seq: Vec::new(),
        };
        all.merge(&second);
        assert_eq!(all.question_accuracy(), Some(4.0 / 5.0));
        assert_eq!(all.accuracies(), (Some(4.0 / 5.0), Some(5.0 / 7.0)));
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
        // A run of equal lengths is cut into blocks of at most `batch`;
        // the tail block is smaller, and the schedule wraps.
        assert_eq!(calibration_batch(&records, 1, 2).unwrap(), (vec![5, 6], 1));
        assert_eq!(
            calibration_batch(&records, 2, 2).unwrap(),
            (vec![1, 2, 3, 4], 2)
        );
        // Mixed lengths never share a block (length buckets).
        assert_eq!(calibration_batch(&[vec![1, 2], vec![3]], 0, 2).unwrap(), (vec![1, 2], 1));
        assert_eq!(calibration_batch(&[vec![1, 2], vec![3]], 1, 2).unwrap(), (vec![3], 1));
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

#[cfg(test)]
mod answer_span_tests {
    use super::*;

    #[test]
    fn answer_targets_follow_the_last_prompt_to_the_terminator() {
        let sp = AnswerSpan {
            start: vec![7, 8],
            end: 1,
            values: None,
        };
        // prompt … [7 8] answer 20 21 [1] pad 1 1
        let ids = [5, 7, 8, 9, 7, 8, 20, 21, 1, 1, 1];
        assert_eq!(answer_targets(&ids, &sp), Some((6, 8)));
        // No terminator: the span runs to the end of the record.
        let ids = [5, 7, 8, 20, 21];
        assert_eq!(answer_targets(&ids, &sp), Some((3, 4)));
        // No prompt, or a prompt with nothing after it: unscored.
        assert_eq!(answer_targets(&[5, 6, 1], &sp), None);
        assert_eq!(answer_targets(&[5, 7, 8], &sp), None);
    }

    #[test]
    fn scored_targets_cover_every_position_without_focus_or_span() {
        let ids = [1u32, 2, 3, 4, 5, 6];
        assert_eq!(scored_targets(&ids, 0, 3, &[], &[], None), vec![1, 2]);
        assert_eq!(scored_targets(&ids, 3, 3, &[], &[], None), vec![4, 5]);
        let sp = AnswerSpan {
            start: vec![2],
            end: 6,
            values: None,
        };
        assert_eq!(scored_targets(&ids, 0, 6, &[], &[], Some(&sp)), vec![2, 3, 4, 5]);
    }
}
