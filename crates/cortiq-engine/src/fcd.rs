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

use crate::fcd_ops::{self as ops, FfnAct, NysCfg};
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
    /// Polish WITHOUT converting attention: the `--o1` spec then only
    /// names which layers are trainable, and both teacher and student
    /// keep exact attention. This is the shape the notebook's FCD
    /// correction has — restore accuracy a compression already cost by
    /// training the last layers on real text — and it is what a
    /// narrowed FFN needs, where the damage is in the FFN, not in the
    /// attention kernel.
    pub polish_only: bool,
}

impl Default for FcdHyper {
    fn default() -> Self {
        Self {
            steps: 300,
            lr: 5e-5,
            kl_w: 0.7,
            eval_every: 25,
            bs: 2,
            seq: 512,
            seed: 0,
            polish_only: false,
        }
    }
}

/// A training or validation token stream for the polish.
///
/// `weight[i]` scales the loss of predicting `ids[i]` (0 = context only),
/// so an answer-only loss is a stream whose prompt tokens weigh 0.
/// `starts` marks record boundaries (ascending, `starts[0] == 0`): a window
/// then begins inside one record and nothing past that record's end is
/// scored, so one prompt's answer is never trained on another's context.
/// Without either, the stream is plain text and every position counts —
/// exactly the original corpus behaviour.
#[derive(Clone, Copy, Debug)]
pub struct FcdStream<'a> {
    pub ids: &'a [u32],
    pub weight: Option<&'a [f32]>,
    pub starts: Option<&'a [usize]>,
}

impl<'a> FcdStream<'a> {
    pub fn plain(ids: &'a [u32]) -> Self {
        Self { ids, weight: None, starts: None }
    }

    fn is_plain(&self) -> bool {
        self.weight.is_none() && self.starts.is_none()
    }

    fn w(&self, i: usize) -> f32 {
        self.weight.map_or(1.0, |w| w[i])
    }

    /// Records that carry any loss: `(start, end, first scored target,
    /// last scored target + 1)`. A plain stream is one record.
    fn scored_records(&self) -> Vec<(usize, usize, usize, usize)> {
        let n = self.ids.len();
        let whole = [0usize];
        let starts = self.starts.unwrap_or(&whole);
        let mut out = Vec::new();
        for (r, &st) in starts.iter().enumerate() {
            let en = starts.get(r + 1).copied().unwrap_or(n).min(n);
            let mut first = None;
            let mut last = 0;
            for i in st + 1..en {
                if self.w(i) > 0.0 {
                    first.get_or_insert(i);
                    last = i + 1;
                }
            }
            if let Some(f) = first {
                out.push((st, en, f, last));
            }
        }
        out
    }

    /// Window start for record `rec` that keeps target `p` in view:
    /// a record that fits starts at its own first token (the whole prompt
    /// is context), a longer one at a uniform offset around `p`.
    fn window_start(&self, rec: (usize, usize, usize, usize), p: usize, seq: usize, jitter: u64) -> usize {
        let (st, en, _, _) = rec;
        let max_off = self.ids.len() - seq - 1;
        if en - st <= seq + 1 {
            return st.min(max_off);
        }
        let lo = st.max(p.saturating_sub(seq));
        let hi = (p - 1).min(max_off).max(lo);
        (lo + (jitter as usize) % (hi - lo + 1)).min(max_off)
    }

    /// Loss weights of the `seq` targets of the window at `off` (targets
    /// outside the window's own record weigh 0).
    fn target_weights(&self, off: usize, seq: usize, rec: (usize, usize, usize, usize)) -> Vec<f32> {
        let (st, en, _, _) = rec;
        (0..seq)
            .map(|t| {
                let i = off + 1 + t;
                if i > st && i < en { self.w(i) } else { 0.0 }
            })
            .collect()
    }
}

/// Input of [`FcdModel::forward_span`].
enum SpanIn<'a> {
    Ids(&'a [u32]),
    Hidden(Vec<f32>),
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
        Some(Self {
            prompts,
            gen_tokens: 60,
            threshold: 0.35,
            baseline_slack: 0.10,
        })
    }
}

/// Gate predicate (Patent 16 draft, claim 13): a checkpoint PASSES iff
/// no prompt's loop score exceeds `threshold` AND none exceeds its
/// zero-shot baseline by more than `slack` — boundary values pass.
pub fn gate_pass(scores: &[f64], baseline: &[f64], threshold: f64, slack: f64) -> bool {
    scores
        .iter()
        .zip(baseline)
        .all(|(&s, &b)| s <= threshold && s <= b + slack)
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

/// Wall-clock phase counters for the bake profiler (CMF_BAKE_PROF=1
/// prints them). Written from worker threads, so atomics; nanoseconds.
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static ATTN_FWD: AtomicU64 = AtomicU64::new(0);
    pub static FFN_FWD: AtomicU64 = AtomicU64::new(0);
    /// Inside FFN_FWD: the device chain call, the plane split + host
    /// activation, and the host (unfused) GEMM path.
    pub static FFN_CHAIN: AtomicU64 = AtomicU64::new(0);
    pub static FFN_ACT: AtomicU64 = AtomicU64::new(0);
    pub static FFN_HOST: AtomicU64 = AtomicU64::new(0);
    /// Before the chain (residual add, post-norm, buffers) and after it
    /// (residual add, acts struct).
    pub static FFN_PRE: AtomicU64 = AtomicU64::new(0);
    pub static FFN_POST: AtomicU64 = AtomicU64::new(0);
    /// Inside ATTN_FWD: the device chain call.
    pub static ATTN_CHAIN: AtomicU64 = AtomicU64::new(0);
    pub static BWD: AtomicU64 = AtomicU64::new(0);
    pub static GEMM: AtomicU64 = AtomicU64::new(0);
    pub static GEMM_CALLS: AtomicU64 = AtomicU64::new(0);
    #[inline]
    pub fn add(c: &AtomicU64, t: std::time::Instant) {
        c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    /// Read and reset, in seconds.
    pub fn take(c: &AtomicU64) -> f64 {
        c.swap(0, Ordering::Relaxed) as f64 / 1e9
    }

    use std::sync::Mutex;
    /// (n, k, m) → (calls, ns). A handful of shapes dominate a training
    /// step; naming them decides which ones are worth a GPU submit and
    /// which are cheaper computed in place.
    pub static SHAPES: Mutex<Vec<((usize, usize, usize), (u64, u64))>> = Mutex::new(Vec::new());
    pub fn gemm_shape(n: usize, k: usize, m: usize, t0: std::time::Instant) {
        let ns = t0.elapsed().as_nanos() as u64;
        let mut g = SHAPES.lock().unwrap();
        match g.iter_mut().find(|(s, _)| *s == (n, k, m)) {
            Some((_, (c, tt))) => {
                *c += 1;
                *tt += ns;
            }
            None => g.push(((n, k, m), (1, ns))),
        }
    }
    pub fn shape_report(top: usize) -> String {
        let mut g = SHAPES.lock().unwrap();
        g.sort_by_key(|(_, (_, ns))| std::cmp::Reverse(*ns));
        let out = g
            .iter()
            .take(top)
            .map(|((n, k, m), (c, ns))| {
                format!(
                    "    [{n}x{k}x{m}] {c} calls, {:.1}s total, {:.1} ms/call",
                    *ns as f64 / 1e9,
                    *ns as f64 / 1e6 / *c as f64
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        g.clear();
        out
    }
}

/// Frozen attention operator of one layer — the per-layer dispatch
/// point for through-backwards (docs/RUST_FCD.md §3).
enum FcdAttn {
    Full {
        /// The projections live in the streamed `LayerMats`, not here:
        /// eagerly resident weights are what made a 4 B bake swap a
        /// laptop. Only the small per-head metadata stays.
        qrows: usize,
        q_norm: Option<Vec<f32>>,
        k_norm: Option<Vec<f32>>,
        bias: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
        /// (bq|bk|bv) concatenated once at build: the device chain binds
        /// it by a STABLE address, which is what keeps the fingerprinted
        /// const cache from minting an entry per call.
        bias_cat: Option<Vec<f32>>,
        /// Qwen3.5: wq rows = 2·nh·hd, per-head [q; gate]; the head
        /// outputs are multiplied by σ(gate) before o_proj.
        output_gate: bool,
        /// Spark-X2.5: `self_attn.g_proj` [nh, hidden] — one sigmoid gain
        /// per Q head from the layer's normed input, multiplied into that
        /// head's output before o_proj (`attention::apply_projected_gate`).
        head_gate: Option<Vec<f32>>,
        /// Sliding-window layer: a query sees its last `w` keys only.
        window: Option<usize>,
        /// Local layers rotate with their own table (`inv_freq_local`,
        /// `rotary_dim_local`) — Gemma-3 / Spark-X2.5 geometry.
        local_rope: bool,
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

pub(crate) struct FcdLayer {
    attn: FcdAttn,
    pub(crate) inter: usize,
    // Frozen originals: the teacher's LN (small, eager). The big FFN
    // matrices stream through `LayerMats`.
    pub(crate) iln: Vec<f32>,
    pub(crate) pln: Vec<f32>,
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
    pub(crate) eps: f64,
    pub(crate) gemma: bool,
    rotary_dim: usize,
    inv_freq: Vec<f64>,
    /// Sliding (local) layers' own RoPE geometry when the arch declares
    /// one (`rope_local_base_freq` / `local_partial_rotary_factor`).
    rotary_dim_local: Option<usize>,
    inv_freq_local: Option<Vec<f64>>,
    /// The FFN gate activation every forward and backward in this file
    /// and in `skillbake` differentiate.
    pub(crate) act: FfnAct,
    /// Keep the per-layer attention activations (needed only by the
    /// through-attention backward of `cortiq fcd`); the skill bake turns
    /// this off and the device chain then reads back attn_out alone.
    pub(crate) attn_acts: bool,
    /// [vocab, hidden]; also the tied head when `lm_head` is None.
    pub(crate) embed: Vec<f32>,
    pub(crate) lm_head: Option<Vec<f32>>,
    pub(crate) final_norm: Vec<f32>,
    pub(crate) layers: Vec<FcdLayer>,
    /// The container the replica came from — the streaming mats
    /// dequantize from its mmap on demand.
    pub(crate) src: std::sync::Arc<CmfModel>,
    /// Per-layer big-matrix cache: filled on touch, evicted outside the
    /// window. The f32 replica used to hold every matrix of every layer
    /// eagerly — 16.7 GB for a 4 B model before the fused concats, ~25
    /// after — which a 24 GB laptop answered with ten minutes of swap.
    mats_cache: Vec<std::sync::Mutex<Option<std::sync::Arc<LayerMats>>>>,
    /// cos/sin table for the device attention chain, keyed by t. Arc so
    /// the underlying address stays STABLE across calls — the GPU const
    /// cache dedups uploads by (pointer, fingerprint).
    rope_tab: std::sync::Mutex<Option<((usize, bool), std::sync::Arc<Vec<f32>>)>>,
    /// How many layers stay resident (0 = all, big machines).
    mats_window: usize,
    /// Which layers run the Nyström kernel in the student forward.
    o1_flags: Vec<bool>,
    nys: NysCfg,
    /// GDN geometry (present when the model has linear layers).
    gdn: Option<GdnDims>,
    /// Looped Transformer: how many times the layer stack runs per
    /// token, and whether a final_norm is inserted at each loop
    /// boundary. A replica that ignores these computes a DIFFERENT
    /// model — on Nanbeige 4.2 (22 layers x 2 loops) the single-pass
    /// forward measured perplexity 4773 where the engine reads 16.5,
    /// and every gradient taken through it was fitted to that fiction.
    pub(crate) loops: usize,
    pub(crate) loop_norm: bool,
    pub(crate) pool: Option<Arc<Pool>>,
    /// Train the flagged layers WITHOUT switching their attention to the
    /// Nyström kernel — see `FcdHyper::polish_only`.
    pub polish_only: bool,
}

fn deq(model: &CmfModel, name: &str) -> Result<Vec<f32>, String> {
    let e = model
        .tensor(name)
        .ok_or_else(|| format!("tensor '{name}' not found"))?;
    let mut out = vec![0f32; e.n_elems()];
    cortiq_core::quant::dequant_tensor(e, model.entry_bytes(e), &mut out)?;
    Ok(out)
}

/// Physical RAM total, for the hard replica ceiling.
/// The container's own memory ceiling, when it has one.
///
/// `/proc/meminfo` reports the HOST inside a container. On a rented box
/// that is the difference between the 944 GB `free` prints and the
/// 117 GB the cgroup will actually hand out, and believing the larger
/// number is an OOM kill one allocation later — which is exactly how a
/// 27B bake died: it read 944 GB, concluded the f32 replica fit with
/// room to spare, and never streamed a single layer.
#[cfg(target_os = "linux")]
fn cgroup_limit_bytes() -> Option<u64> {
    let read =
        |p: &str| -> Option<u64> { std::fs::read_to_string(p).ok()?.trim().parse::<u64>().ok() };
    // v2 first, then v1. An unlimited v2 cgroup holds the word "max",
    // which fails to parse; an unlimited v1 holds a huge sentinel.
    read("/sys/fs/cgroup/memory.max")
        .or_else(|| read("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        .filter(|&v| v < u64::MAX / 2)
}

/// Which flagged layers survive.
///
/// For an O(1) conversion only full-attention layers are eligible — a
/// GDN layer keeps its own operator. Under `polish_only` the flag means
/// "train this layer's FFN and norms", which any layer supports, so the
/// filter must not run: on a hybrid it silently shrank `--o1 deep8` to
/// the two or three full-attention layers that happened to fall in the
/// window, and 48 of Qwen3.8-27B's 64 layers are GDN.
pub(crate) fn apply_layer_eligibility(flags: &mut [bool], is_full: &[bool], polish_only: bool) {
    if polish_only {
        return;
    }
    for (li, f) in flags.iter_mut().enumerate() {
        if *f && !is_full.get(li).copied().unwrap_or(false) {
            *f = false;
        }
    }
}

/// The lower of what the host reports and what the container will hand
/// out. Pure so the decision can be tested without a cgroup to hand.
pub(crate) fn effective_total(host: u64, cgroup_limit: Option<u64>) -> u64 {
    match cgroup_limit {
        Some(l) => host.min(l),
        None => host,
    }
}

/// Headroom: inside a cgroup it is the limit minus what the cgroup
/// already holds, never whatever the host happens to have free.
pub(crate) fn effective_available(host: u64, limit: Option<u64>, used: Option<u64>) -> u64 {
    match (limit, used) {
        (Some(l), Some(u)) => host.min(l.saturating_sub(u)),
        (Some(l), None) => host.min(l),
        _ => host,
    }
}

/// Bytes this cgroup is already holding, for the same reason.
#[cfg(target_os = "linux")]
fn cgroup_used_bytes() -> Option<u64> {
    let read =
        |p: &str| -> Option<u64> { std::fs::read_to_string(p).ok()?.trim().parse::<u64>().ok() };
    read("/sys/fs/cgroup/memory.current")
        .or_else(|| read("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
}

/// Physical RAM total — the cgroup's limit when the container has a lower
/// one. Also the ceiling `quantize-gptq` plans its Hessian RAM against.
pub fn physical_total_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        return String::from_utf8_lossy(&out.stdout).trim().parse().ok();
    }
    #[cfg(target_os = "linux")]
    {
        let mem = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = mem
            .lines()
            .find(|l| l.starts_with("MemTotal:"))?
            .split_whitespace()
            .nth(1)?
            .parse()
            .ok()?;
        return Some(effective_total(kb * 1024, cgroup_limit_bytes()));
    }
    #[allow(unreachable_code)]
    None
}

/// AVAILABLE RAM in bytes, best effort — used only to refuse a bake
/// whose f32 replica would swap instead of run.
pub(crate) fn available_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // free% × total from the same counters the memory-pressure
        // subsystem uses; hw.memsize alone lied by whatever the other
        // apps were holding.
        let total: u64 = {
            let out = std::process::Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
                .ok()?;
            String::from_utf8_lossy(&out.stdout).trim().parse().ok()?
        };
        let out = std::process::Command::new("memory_pressure")
            .arg("-Q")
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let pct: u64 = text
            .lines()
            .find(|l| l.contains("free percentage"))?
            .split(':')
            .nth(1)?
            .trim()
            .trim_end_matches('%')
            .parse()
            .ok()?;
        return Some(total / 100 * pct);
    }
    #[cfg(target_os = "linux")]
    {
        let mem = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = mem
            .lines()
            .find(|l| l.starts_with("MemAvailable:"))?
            .split_whitespace()
            .nth(1)?
            .parse()
            .ok()?;
        // Inside a cgroup, headroom is the limit minus what it already
        // holds, not whatever the host happens to have free.
        return Some(effective_available(
            kb * 1024,
            cgroup_limit_bytes(),
            cgroup_used_bytes(),
        ));
    }
    #[allow(unreachable_code)]
    None
}

/// `deq` for sibling modules (the bake's Phase-B masters dequant once,
/// straight from the source container).
pub(crate) fn deq_pub(model: &CmfModel, name: &str) -> Result<Vec<f32>, String> {
    deq(model, name)
}

/// The four big matrices the hot path reads for one layer, dequantized
/// together: the fused attention projection, the attention output, the
/// fused FFN gate+up, and the FFN down. Everything else a layer owns —
/// norms, biases, GDN cores — is small and stays eagerly resident.
pub(crate) struct LayerMats {
    pub(crate) wqkv: Vec<f32>,
    pub(crate) wo: Vec<f32>,
    pub(crate) gu: Vec<f32>,
    pub(crate) down: Vec<f32>,
}

impl FcdModel {
    /// The layer's big matrices, dequantized on first touch. Sequential
    /// sweeps (forward 0..nl, backward nl..0) hit the same physical
    /// layer at most twice per virtual pass, so a small window streams
    /// a model far bigger than RAM without swapping.
    pub(crate) fn mats(&self, li: usize) -> Result<std::sync::Arc<LayerMats>, String> {
        {
            let slot = self.mats_cache[li].lock().unwrap();
            if let Some(m) = slot.as_ref() {
                return Ok(m.clone());
            }
        }
        let p = format!("model.layers.{li}.");
        let d = |name: String| -> Result<Vec<f32>, String> { deq(&self.src, &name) };
        let built = if matches!(self.layers[li].attn, FcdAttn::Full { .. }) {
            let wq = d(format!("{p}self_attn.q_proj.weight"))?;
            let wk = d(format!("{p}self_attn.k_proj.weight"))?;
            let wv = d(format!("{p}self_attn.v_proj.weight"))?;
            let mut wqkv = Vec::with_capacity(wq.len() + wk.len() + wv.len());
            wqkv.extend_from_slice(&wq);
            wqkv.extend_from_slice(&wk);
            wqkv.extend_from_slice(&wv);
            let gate = d(format!("{p}mlp.gate_proj.weight"))?;
            let up = d(format!("{p}mlp.up_proj.weight"))?;
            let mut gu = Vec::with_capacity(gate.len() + up.len());
            gu.extend_from_slice(&gate);
            gu.extend_from_slice(&up);
            LayerMats {
                wqkv,
                wo: d(format!("{p}self_attn.o_proj.weight"))?,
                gu,
                down: d(format!("{p}mlp.down_proj.weight"))?,
            }
        } else {
            // GDN layers keep their attention core eagerly; only the FFN
            // streams.
            let gate = d(format!("{p}mlp.gate_proj.weight"))?;
            let up = d(format!("{p}mlp.up_proj.weight"))?;
            let mut gu = Vec::with_capacity(gate.len() + up.len());
            gu.extend_from_slice(&gate);
            gu.extend_from_slice(&up);
            LayerMats {
                wqkv: Vec::new(),
                wo: Vec::new(),
                gu,
                down: d(format!("{p}mlp.down_proj.weight"))?,
            }
        };
        let arc = std::sync::Arc::new(built);
        *self.mats_cache[li].lock().unwrap() = Some(arc.clone());
        // Evict outside the window — cheap, and the access pattern is
        // sequential, so the evicted slots are exactly the cold ones.
        if self.mats_window > 0 {
            let w = self.mats_window;
            for (j, slot) in self.mats_cache.iter().enumerate() {
                let dist = li.abs_diff(j).min(self.layers.len() - li.abs_diff(j));
                if dist > w {
                    *slot.lock().unwrap() = None;
                }
            }
        }
        Ok(arc)
    }
}

impl FcdModel {
    /// Dequantize a model into the f32 training replica. Refuses what
    /// the backward cannot honestly differentiate yet (loud, not silent).
    pub fn from_cmf(
        model: &std::sync::Arc<CmfModel>,
        o1: &O1Cfg,
        polish_only: bool,
    ) -> Result<Self, String> {
        let arch = model.arch().clone();
        let act = FfnAct::from_arch(&arch.hidden_act).ok_or_else(|| {
            format!(
                "fcd/skill-bake: hidden_act '{}' not supported yet (silu | gelu)",
                arch.hidden_act
            )
        })?;
        let has_linear = arch
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::LinearAttention));
        let gdn = if has_linear {
            let lc = arch
                .linear_core
                .as_ref()
                .ok_or_else(|| "model has linear layers but no arch.linear_core".to_string())?;
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
                    let bias = match (opt("q_proj.bias"), opt("k_proj.bias"), opt("v_proj.bias")) {
                        (Some(a), Some(b), Some(c)) => Some((a, b, c)),
                        _ => None,
                    };
                    let qrows = wq.len() / h;
                    drop(wq);
                    let bias_cat = bias.as_ref().map(|(bq, bk, bv)| {
                        let mut v = Vec::with_capacity(bq.len() + bk.len() + bv.len());
                        v.extend_from_slice(bq);
                        v.extend_from_slice(bk);
                        v.extend_from_slice(bv);
                        v
                    });
                    // A projected output gate rides in one form: one row
                    // per Q head with a sigmoid (Spark-X2.5). A per-element
                    // or softplus gate has no trainer path yet — loud.
                    let head_gate = match opt("g_proj.weight") {
                        None => None,
                        Some(g) if g.len() == nh * h => Some(g),
                        Some(g) => {
                            return Err(format!(
                                "layer {li}: self_attn.g_proj has {} values; only a per-head gate (nh·hidden = {}) is supported",
                                g.len(),
                                nh * h
                            ));
                        }
                    };
                    let sliding = matches!(arch.layer_types.get(li), Some(LayerType::SlidingAttention));
                    let window = if sliding {
                        Some(arch.sliding_window.ok_or_else(|| {
                            format!("layer {li} is SlidingAttention but arch.sliding_window is unset")
                        })?)
                    } else {
                        None
                    };
                    FcdAttn::Full {
                        qrows,
                        q_norm: opt("q_norm.weight"),
                        k_norm: opt("k_norm.weight"),
                        bias,
                        bias_cat,
                        output_gate,
                        head_gate,
                        window,
                        local_rope: sliding,
                    }
                }
            };
            let gate = deq(model, &format!("{p}mlp.gate_proj.weight"))?;
            let inter = gate.len() / h;
            drop(gate);
            layers.push(FcdLayer {
                attn,
                inter,
                iln: deq(model, &format!("{p}input_layernorm.weight"))?,
                pln: deq(model, &format!("{p}post_attention_layernorm.weight"))?,
            });
        }

        let rotary_dim = ((hd as f32 * arch.partial_rotary_factor) as usize)
            .max(2)
            .min(hd);
        let base = arch.rope_theta;
        let inv_freq: Vec<f64> = (0..rotary_dim / 2)
            .map(|i| 1.0 / base.powf(2.0 * i as f64 / rotary_dim as f64))
            .collect();
        // Local (sliding) layers: the runtime's rule (`loader.rs`) — their
        // rotary width follows `local_partial_rotary_factor` (else the
        // global factor) and their table `rope_local_base_freq` when set.
        let any_sliding = arch
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::SlidingAttention));
        let rotary_dim_local = any_sliding.then(|| {
            ((hd as f32 * arch.local_partial_rotary_factor.unwrap_or(arch.partial_rotary_factor))
                as usize)
                .max(2)
                .min(hd)
        });
        let inv_freq_local = match (rotary_dim_local, arch.rope_local_base_freq) {
            (Some(rd), Some(lb)) => Some(
                (0..rd / 2)
                    .map(|i| 1.0 / lb.powf(2.0 * i as f64 / rd as f64))
                    .collect::<Vec<f64>>(),
            ),
            _ => None,
        };
        let loops = arch.num_loops.max(1);
        let loop_norm = arch.loop_final_norm;
        let mut flags = o1.layer_flags(arch.num_layers);
        flags.resize(arch.num_layers, false);
        // Only full-attention layers are o1-convertible (same rule as
        // Pipeline::set_o1) — a GDN layer keeps its own operator. Under
        // `polish_only` the flag means "train this layer's FFN and
        // norms", which a GDN layer supports like any other: its own
        // operator stays frozen and the backward already carries the
        // through-gradient. Excluding them there would silently shrink
        // `--o1 deep8` on a hybrid to the two or three full-attention
        // layers that happen to fall in the window.
        {
            let is_full: Vec<bool> = layers
                .iter()
                .map(|l| matches!(l.attn, FcdAttn::Full { .. }))
                .collect();
            apply_layer_eligibility(&mut flags, &is_full, polish_only);
        }
        // Streaming window: how many layers' big matrices stay resident.
        // Sized from available memory; CMF_BAKE_MATS_LAYERS overrides.
        // 0 = everything resident (big machines keep today's speed).
        let per_layer: u64 = layers
            .first()
            .map(|_| {
                let qrows = match layers[0].attn {
                    FcdAttn::Full { qrows, .. } => qrows,
                    _ => 0,
                };
                ((qrows + 2 * (nkv * hd)) as u64 * h as u64      // wqkv
                    + (h as u64 * nh as u64 * hd as u64)          // wo
                    + 3 * (layers[0].inter as u64 * h as u64))    // gu + down
                    * 4
            })
            .unwrap_or(0);
        let mats_window = if let Ok(v) = std::env::var("CMF_BAKE_MATS_LAYERS") {
            v.parse().unwrap_or(0)
        } else {
            let avail = available_ram_bytes().unwrap_or(u64::MAX);
            let cap = physical_total_bytes()
                .map(|t| t / 5 * 3)
                .unwrap_or(u64::MAX)
                .min(avail.saturating_sub(avail / 10));
            let all = per_layer.saturating_mul(arch.num_layers as u64);
            if all + 4 * 1024 * 1024 * 1024 <= cap {
                0 // everything fits with headroom — stay eager-equivalent
            } else {
                let w = (cap.saturating_sub(3 * 1024 * 1024 * 1024) / per_layer.max(1))
                    .clamp(2, arch.num_layers as u64) as usize;
                tracing::info!(
                    "fcd replica: streaming {w} of {} layers (~{:.1} GB resident of ~{:.1} GB total)",
                    arch.num_layers,
                    (w as u64 * per_layer) as f64 / 1e9,
                    all as f64 / 1e9,
                );
                w
            }
        };
        let mats_cache = (0..arch.num_layers)
            .map(|_| std::sync::Mutex::new(None))
            .collect();
        Ok(Self {
            src: model.clone(),
            mats_cache,
            rope_tab: std::sync::Mutex::new(None),
            mats_window,
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
            rotary_dim_local,
            inv_freq_local,
            act,
            attn_acts: true,
            embed,
            lm_head,
            final_norm,
            layers,
            o1_flags: flags,
            // prefill: None = half the window, the same seal point
            // `cortiq ppl --o1` defaults to (see NysCfg::prefill).
            nys: NysCfg {
                m: o1.m,
                w: o1.w,
                sink: o1.sink,
                prefill: None,
            },
            gdn,
            loops,
            loop_norm,
            pool: Pool::from_env(),
            polish_only: false,
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
            let p = format!("model.layers.{li}.");
            data.push(l.iln.clone());
            data.push(l.pln.clone());
            // Trained masters dequant once, straight from the source —
            // the streamed cache holds only the fused hot-path concats.
            data.push(deq(&fm.src, &format!("{p}mlp.gate_proj.weight")).expect("gate"));
            data.push(deq(&fm.src, &format!("{p}mlp.up_proj.weight")).expect("up"));
            data.push(deq(&fm.src, &format!("{p}mlp.down_proj.weight")).expect("down"));
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
pub(crate) struct LnFfn<'a> {
    pub(crate) iln: &'a [f32],
    pub(crate) pln: &'a [f32],
    pub(crate) gate: &'a [f32],
    pub(crate) up: &'a [f32],
    pub(crate) down: &'a [f32],
    /// `[gate; up]` rows concatenated — one GEMM submit instead of two.
    /// `None` for Phase-B trained copies, whose rows move every step;
    /// the frozen majority carries the concat prebuilt at load.
    pub(crate) gu: Option<&'a [f32]>,
}

fn ln_ffn<'a>(
    fm: &'a FcdModel,
    ts: Option<&'a TrainState>,
    li: usize,
    mats: &'a LayerMats,
) -> LnFfn<'a> {
    if let Some(t) = ts {
        if let Some(s) = t.slot(li) {
            let b = s * PARAMS_PER_LAYER;
            return LnFfn {
                iln: &t.data[b],
                pln: &t.data[b + 1],
                gate: &t.data[b + 2],
                up: &t.data[b + 3],
                down: &t.data[b + 4],
                gu: None,
            };
        }
    }
    let l = &fm.layers[li];
    LnFfn {
        iln: &l.iln,
        pln: &l.pln,
        // The fused concat carries both projections; the fallback slots
        // are never read while `gu` is Some.
        gate: &[],
        up: &[],
        down: &mats.down,
        gu: Some(&mats.gu),
    }
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
        /// Per-(row, head) sigmoid gains of a projected head gate (empty
        /// without one). Frozen weights: the backward scales by the gain
        /// and does not chase the gate's own input.
        head_gain: Vec<f32>,
    },
    /// Raw projection streams — the GDN backward replays conv + the
    /// recurrence from these.
    Gdn {
        qkv: Vec<f32>,
        z: Vec<f32>,
        a: Vec<f32>,
        b: Vec<f32>,
    },
}

/// Trained attention weights of a full-layer FCD layer, in place of the
/// frozen `LayerMats`/head gate for one forward.
#[derive(Clone, Copy)]
pub(crate) struct AttnW<'a> {
    pub(crate) wqkv: &'a [f32],
    pub(crate) wo: &'a [f32],
    pub(crate) head_gate: Option<&'a [f32]>,
}

pub(crate) struct LayerActs {
    pub(crate) inv1: Vec<f32>,
    attn: AttnActs,
    pub(crate) h1: Vec<f32>,
    pub(crate) n2: Vec<f32>,
    pub(crate) inv2: Vec<f32>,
    pub(crate) gpre: Vec<f32>,
    pub(crate) upre: Vec<f32>,
    pub(crate) act: Vec<f32>,
}

impl LayerActs {
    /// The full-attention activations a full-layer FCD backward needs:
    /// (qrot, krot, vproj, ao, head_gain) — None for GDN layers or when
    /// the forward did not keep them.
    pub(crate) fn attn_full(&self) -> Option<(&[f32], &[f32], &[f32], &[f32], &[f32])> {
        match &self.attn {
            AttnActs::Full {
                qrot,
                krot,
                vproj,
                ao,
                head_gain,
                ..
            } if !qrot.is_empty() => Some((qrot, krot, vproj, ao, head_gain)),
            _ => None,
        }
    }
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


/// Split `[0, n)` into one contiguous range per pool worker and run `f(lo, hi)`
/// on each — the shape of every row-wise host op below (norms, residual adds,
/// plane copies), which were single-threaded memory walks over hundreds of MB.
pub(crate) fn pooled_rows(n: usize, pool: Option<&Pool>, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) if n >= 64 => p.run(&|w, nw| {
            let lo = n * w / nw;
            let hi = n * (w + 1) / nw;
            if hi > lo {
                f(lo, hi);
            }
        }),
        _ => f(0, n),
    }
}

/// `ops::rmsnorm_fwd` over row ranges in parallel.
pub(crate) fn rmsnorm_fwd_pooled(
    x: &[f32],
    w: &[f32],
    eps: f64,
    gemma: bool,
    y: &mut [f32],
    inv: &mut [f32],
    pool: Option<&Pool>,
) {
    let d = w.len();
    let n = x.len() / d;
    let yp = SendMut(y.as_mut_ptr());
    let ip = SendMut(inv.as_mut_ptr());
    pooled_rows(n, pool, &|lo, hi| {
        // SAFETY: disjoint row ranges per worker.
        let (yr, ir) = unsafe {
            (
                std::slice::from_raw_parts_mut(yp.at(lo * d), (hi - lo) * d),
                std::slice::from_raw_parts_mut(ip.at(lo), hi - lo),
            )
        };
        ops::rmsnorm_fwd(&x[lo * d..hi * d], w, eps, gemma, yr, ir);
    });
}

/// `ops::rmsnorm_bwd` (through-grad only, frozen gain) over row ranges in parallel.
pub(crate) fn rmsnorm_bwd_pooled(
    x: &[f32],
    w: &[f32],
    inv: &[f32],
    dy: &[f32],
    gemma: bool,
    dx: &mut [f32],
    pool: Option<&Pool>,
) {
    let d = w.len();
    let n = x.len() / d;
    let dp = SendMut(dx.as_mut_ptr());
    pooled_rows(n, pool, &|lo, hi| {
        // SAFETY: disjoint row ranges per worker.
        let dxr = unsafe { std::slice::from_raw_parts_mut(dp.at(lo * d), (hi - lo) * d) };
        ops::rmsnorm_bwd(&x[lo * d..hi * d], w, &inv[lo..hi], &dy[lo * d..hi * d], gemma, dxr, None);
    });
}

/// `a += b`, elementwise, in parallel.
pub(crate) fn add_into_pooled(a: &mut [f32], b: &[f32], pool: Option<&Pool>) {
    let n = a.len().min(b.len());
    let ap = SendMut(a.as_mut_ptr());
    const CHUNK: usize = 4096;
    let chunks = n.div_ceil(CHUNK);
    pooled_rows(chunks, pool, &|lo, hi| {
        let (s, e) = (lo * CHUNK, (hi * CHUNK).min(n));
        // SAFETY: disjoint element ranges per worker.
        let ar = unsafe { std::slice::from_raw_parts_mut(ap.at(s), e - s) };
        for (x, &y) in ar.iter_mut().zip(&b[s..e]) {
            *x += y;
        }
    });
}

/// `act[r] = f(gpre[r]) ⊙ upre[r]` over `n` rows of `inter`, pooled: the
/// erf GELU is not free (63 M elements per 4 B training step), and one
/// thread walking it was the slowest link of a Spark bake.
pub(crate) fn act_mul_rows(
    f: FfnAct,
    gpre: &[f32],
    upre: &[f32],
    act: &mut [f32],
    n: usize,
    inter: usize,
    pool: Option<&Pool>,
) {
    let row = |r: usize, out: &mut [f32]| {
        let g = &gpre[r * inter..(r + 1) * inter];
        let u = &upre[r * inter..(r + 1) * inter];
        for ((o, &gv), &uv) in out.iter_mut().zip(g).zip(u) {
            *o = f.fwd(gv) * uv;
        }
    };
    match pool {
        Some(p) if n > 1 => {
            let ap = SendMut(act.as_mut_ptr());
            p.run(&|w, nw| {
                for r in (w..n).step_by(nw) {
                    // SAFETY: disjoint row slices per worker.
                    let out = unsafe { std::slice::from_raw_parts_mut(ap.at(r * inter), inter) };
                    row(r, out);
                }
            });
        }
        _ => {
            for r in 0..n {
                row(r, &mut act[r * inter..(r + 1) * inter]);
            }
        }
    }
}

/// The backward of `act_mul_rows` with an optional per-neuron scale `s`
/// (the DTG-MA gate the forward multiplied in after the activation):
/// `dg = dact·s·u·f′(g)`, `du = dact·s·f(g)`. Pooled like the forward.
#[allow(clippy::too_many_arguments)]
pub(crate) fn act_bwd_rows(
    f: FfnAct,
    gpre: &[f32],
    upre: &[f32],
    dact: &[f32],
    scale: Option<&[f32]>,
    dg: &mut [f32],
    du: &mut [f32],
    n: usize,
    inter: usize,
    pool: Option<&Pool>,
) {
    let row = |r: usize, dg: &mut [f32], du: &mut [f32]| {
        let base = r * inter;
        for j in 0..inter {
            let i = base + j;
            let da = dact[i] * scale.map(|s| s[j]).unwrap_or(1.0);
            dg[j] = da * upre[i] * f.bwd(gpre[i]);
            du[j] = da * f.fwd(gpre[i]);
        }
    };
    match pool {
        Some(p) if n > 1 => {
            let gp = SendMut(dg.as_mut_ptr());
            let up = SendMut(du.as_mut_ptr());
            p.run(&|w, nw| {
                for r in (w..n).step_by(nw) {
                    // SAFETY: disjoint row slices per worker.
                    let (dgr, dur) = unsafe {
                        (
                            std::slice::from_raw_parts_mut(gp.at(r * inter), inter),
                            std::slice::from_raw_parts_mut(up.at(r * inter), inter),
                        )
                    };
                    row(r, dgr, dur);
                }
            });
        }
        _ => {
            for r in 0..n {
                let (dgr, dur) = (&mut dg[r * inter..(r + 1) * inter], &mut du[r * inter..(r + 1) * inter]);
                row(r, dgr, dur);
            }
        }
    }
}

impl FcdModel {
    /// Per-head RMS-norm (qk-norm) + partial RoPE for all rows of a
    /// projection buffer. `heads` per row, `x` is `[n, heads·hd]`.
    /// Saves the per-(row, head) rms inv when a norm gain is present.
    /// (window, local-RoPE) of a full-attention layer — None for GDN.
    pub(crate) fn attn_geometry(&self, li: usize) -> Option<(Option<usize>, bool)> {
        match &self.layers[li].attn {
            FcdAttn::Full {
                window, local_rope, ..
            } => Some((*window, *local_rope)),
            _ => None,
        }
    }

    /// The frozen per-head gate projection of a layer (Spark-X2.5), if any.
    pub(crate) fn head_gate_weights(&self, li: usize) -> Option<&[f32]> {
        match &self.layers[li].attn {
            FcdAttn::Full { head_gate, .. } => head_gate.as_deref(),
            _ => None,
        }
    }

    /// The RoPE geometry of a layer: local layers may own a table.
    pub(crate) fn layer_rope(&self, local: bool) -> (usize, &[f64]) {
        if local {
            (
                self.rotary_dim_local.unwrap_or(self.rotary_dim),
                self.inv_freq_local.as_deref().unwrap_or(&self.inv_freq),
            )
        } else {
            (self.rotary_dim, &self.inv_freq)
        }
    }

    fn qk_norm_rope(
        &self,
        x: &mut [f32],
        norm: Option<&[f32]>,
        heads: usize,
        t: usize,
        local: bool,
        inv_out: &mut [f32],
    ) {
        let hd = self.hd;
        let (rotary_dim, inv_freq) = self.layer_rope(local);
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
                ops::rope_fwd(&mut head[..rotary_dim], pos, inv_freq);
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
        self.layer_forward_scaled(li, h_in, b, t, wts, nystrom, want_acts, None, true, None, false)
    }

    /// `layer_forward` with an optional per-neuron FFN activation scale
    /// (the DTG-MA mask σ(m), Patent 2): `act·scale` feeds down_proj.
    /// `LayerActs.act` stays PRE-scale so the mask backward can read it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn layer_forward_scaled(
        &self,
        li: usize,
        h_in: &[f32],
        b: usize,
        t: usize,
        wts: &LnFfn,
        nystrom: bool,
        want_acts: bool,
        ffn_scale: Option<&[f32]>,
        ffn_planes_to_host: bool,
        attn_w: Option<AttnW<'_>>,
        want_attn_acts: bool,
    ) -> (Vec<f32>, Option<LayerActs>) {
        // Polish-only: the flags select what to TRAIN, not what to convert.
        let nystrom = nystrom && !self.polish_only;
        let hsz = self.hidden;
        let n = b * t;
        let l = &self.layers[li];
        let pool = self.pool.as_deref();

        let mut n1 = vec![0f32; n * hsz];
        let mut inv1 = vec![0f32; n];
        rmsnorm_fwd_pooled(h_in, wts.iln, self.eps, self.gemma, &mut n1, &mut inv1, pool);

        let mats = self.mats(li).expect("layer mats");
        let t_attn = std::time::Instant::now();
        let (attn_out, attn_acts) = match &l.attn {
            FcdAttn::Full { .. } => {
                self.full_attn_fwd(&l.attn, &mats, &n1, b, t, nystrom, attn_w, want_attn_acts)
            }
            FcdAttn::Gdn { .. } => self.gdn_attn_fwd(&l.attn, &n1, b, t),
        };
        prof::add(&prof::ATTN_FWD, t_attn);
        let t_ffn = std::time::Instant::now();

        let mut h1 = h_in.to_vec();
        add_into_pooled(&mut h1, &attn_out, pool);

        let mut n2 = vec![0f32; n * hsz];
        let mut inv2 = vec![0f32; n];
        rmsnorm_fwd_pooled(&h1, wts.pln, self.eps, self.gemma, &mut n2, &mut inv2, pool);

        let inter = l.inter;
        // The gate/up/act planes are allocated only by the path that fills
        // them: zero-filling three n×inter buffers per layer that the device
        // chain never touches cost more than the chain itself (48 s of a
        // 73 s FFN forward on a 1.7 B smoke bake).
        let mut gpre: Vec<f32> = Vec::new();
        let mut upre: Vec<f32> = Vec::new();
        let mut act: Vec<f32> = Vec::new();
        let mut ffn = vec![0f32; n * hsz];
        prof::add(&prof::FFN_PRE, t_ffn);
        // The frozen fused-gu FFN rides the whole chain on the device when
        // the tensor-core arm is up: one submit, and the gate+up plane
        // only crosses PCIe when the backward pass will need it. An eval
        // call moves three matrices and reads back one.
        let mut fused = false;
        #[cfg(feature = "gpu")]
        if let Some(gu) = wts.gu {
            if crate::gpu::enabled_here() {
                // A training call whose backward runs on the card leaves the
                // plane there: nothing of n×2·inter crosses PCIe.
                let mut both = (want_acts && ffn_planes_to_host).then(|| vec![0f32; n * 2 * inter]);
                let t_chain = std::time::Instant::now();
                let chained = crate::gpu_wgpu::ffn_chain_f32(
                    &n2,
                    gu,
                    wts.down,
                    ffn_scale,
                    both.as_deref_mut(),
                    want_acts.then_some(li),
                    &mut ffn,
                    n,
                    hsz,
                    inter,
                    self.act,
                );
                prof::add(&prof::FFN_CHAIN, t_chain);
                if chained {
                    fused = true;
                    if let Some(b) = &both {
                        let t_act = std::time::Instant::now();
                        gpre.reserve_exact(n * inter);
                        upre.reserve_exact(n * inter);
                        for r in 0..n {
                            let row = &b[r * 2 * inter..(r + 1) * 2 * inter];
                            gpre.extend_from_slice(&row[..inter]);
                            upre.extend_from_slice(&row[inter..]);
                        }
                        act = vec![0f32; n * inter];
                        // PRE-scale, as the mask backward expects.
                        act_mul_rows(self.act, &gpre, &upre, &mut act, n, inter, pool);
                        prof::add(&prof::FFN_ACT, t_act);
                    }
                }
            }
        }
        let t_host = std::time::Instant::now();
        if !fused {
            gpre = vec![0f32; n * inter];
            upre = vec![0f32; n * inter];
            act = vec![0f32; n * inter];
            if let Some(gu) = wts.gu {
                // One submit; split back so everything downstream is
                // byte-identical to the two-call path.
                let mut both = vec![0f32; n * 2 * inter];
                ops::gemm_nt(&n2, gu, &mut both, n, hsz, 2 * inter, pool);
                for r in 0..n {
                    let row = &both[r * 2 * inter..(r + 1) * 2 * inter];
                    gpre[r * inter..(r + 1) * inter].copy_from_slice(&row[..inter]);
                    upre[r * inter..(r + 1) * inter].copy_from_slice(&row[inter..]);
                }
            } else {
                ops::gemm_nt(&n2, wts.gate, &mut gpre, n, hsz, inter, pool);
                ops::gemm_nt(&n2, wts.up, &mut upre, n, hsz, inter, pool);
            }
            act_mul_rows(self.act, &gpre, &upre, &mut act, n, inter, pool);
            match ffn_scale {
                Some(g) => {
                    debug_assert_eq!(g.len(), inter);
                    let mut act2 = act.clone();
                    for r in 0..n {
                        for (a, &gv) in act2[r * inter..(r + 1) * inter].iter_mut().zip(g) {
                            *a *= gv;
                        }
                    }
                    ops::gemm_nt(&act2, wts.down, &mut ffn, n, inter, hsz, pool);
                }
                None => ops::gemm_nt(&act, wts.down, &mut ffn, n, inter, hsz, pool),
            }
        }
        prof::add(&prof::FFN_HOST, t_host);
        let t_post = std::time::Instant::now();
        // The acts keep h1; without them the residual add runs in place.
        let mut h2 = if want_acts { h1.clone() } else { std::mem::take(&mut h1) };
        add_into_pooled(&mut h2, &ffn, pool);
        prof::add(&prof::FFN_POST, t_post);

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
        prof::add(&prof::FFN_FWD, t_ffn);
        (h2, acts)
    }

    /// Full-attention forward: projections (+optional biases), optional
    /// per-head [q; gate] split (Qwen3.5 output gate), qk-norm + RoPE,
    /// per-head exact-or-Nyström attention, σ(gate) multiply, o_proj.
    #[allow(clippy::too_many_arguments)]
    fn full_attn_fwd(
        &self,
        attn: &FcdAttn,
        mats: &LayerMats,
        n1: &[f32],
        b: usize,
        t: usize,
        nystrom: bool,
        attn_w: Option<AttnW<'_>>,
        want_attn_acts: bool,
    ) -> (Vec<f32>, AttnActs) {
        // Trained copies (full-layer FCD) stand in for the frozen mats.
        let (wqkv, wo) = match attn_w {
            Some(w) => (w.wqkv, w.wo),
            None => (&mats.wqkv[..], &mats.wo[..]),
        };
        let FcdAttn::Full {
            qrows: _,
            q_norm,
            k_norm,
            bias,
            bias_cat,
            output_gate,
            head_gate,
            window,
            local_rope,
        } = attn
        else {
            unreachable!("full_attn_fwd on a non-Full layer");
        };

        let (hsz, nh, nkv, hd) = (self.hidden, self.nh, self.nkv, self.hd);
        let n = b * t;
        let pool = self.pool.as_deref();
        let qdim = nh * hd;
        let kvdim = nkv * hd;
        let rep = nh / nkv;
        let qrows = if *output_gate { 2 * qdim } else { qdim };

        let fused = qrows + 2 * kvdim;
        // Projected head gate (Spark-X2.5): gain_h = σ(g_proj_h · n1) per
        // row, applied to the whole head — the runtime's
        // `apply_projected_gate(per_head = true, sigmoid = true)`. Computed
        // once here; both the device chain and the host path consume it.
        let head_gate_w: Option<&[f32]> = attn_w.and_then(|w| w.head_gate).or(head_gate.as_deref());
        let head_gain: Vec<f32> = match head_gate_w {
            Some(gw) => {
                let mut graw = vec![0f32; n * nh];
                ops::gemm_nt(n1, gw, &mut graw, n, hsz, nh, pool);
                for v in graw.iter_mut() {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
                graw
            }
            None => Vec::new(),
        };
        let (rotary_dim, inv_freq) = self.layer_rope(*local_rope);
        // ── device chain: the whole attention forward in one submit ──
        // Declines in strict-f32 mode, on Nyström layers (their kernel is
        // certified f64 and stays host), and wherever a shape steps
        // outside the kernels; the host path below is the fallback and
        // the reference.
        #[cfg(feature = "gpu")]
        if !nystrom && crate::gpu::enabled_here() {
            let want_acts = self.attn_acts || want_attn_acts;
            let rope = {
                let mut rt = self.rope_tab.lock().unwrap();
                match rt.as_ref() {
                    Some((key, arc)) if *key == (t, *local_rope) => arc.clone(),
                    _ => {
                        let half = rotary_dim / 2;
                        let mut tab = Vec::with_capacity(t * half * 2);
                        for pos in 0..t {
                            for &freq in inv_freq.iter() {
                                let a = pos as f64 * freq;
                                tab.push(a.cos() as f32);
                                tab.push(a.sin() as f32);
                            }
                        }
                        let arc = std::sync::Arc::new(tab);
                        *rt = Some(((t, *local_rope), arc.clone()));
                        arc
                    }
                }
            };
            let cfg = crate::gpu_wgpu::AttnChainCfg {
                wqkv,
                wo,
                q_norm: q_norm.as_deref(),
                k_norm: k_norm.as_deref(),
                bias: bias_cat.as_deref(),
                output_gate: *output_gate,
                head_gain: (!head_gain.is_empty()).then_some(&head_gain[..]),
                window: *window,
                gemma: self.gemma,
                eps: self.eps as f32,
                rotary_half: rotary_dim / 2,
                rope: &rope,
                b,
                t,
                nh,
                nkv,
                hd,
                hsz: self.hidden,
            };
            let mut attn_out = vec![0f32; n * self.hidden];
            let t_chain = std::time::Instant::now();
            let chained = crate::gpu_wgpu::attn_chain_f32(n1, &cfg, &mut attn_out, want_acts);
            prof::add(&prof::ATTN_CHAIN, t_chain);
            if let Some(ch) = chained {
                if !want_acts {
                    return (
                        attn_out,
                        AttnActs::Full {
                            qpre: Vec::new(),
                            kpre: Vec::new(),
                            vproj: Vec::new(),
                            qrot: Vec::new(),
                            krot: Vec::new(),
                            qinv: Vec::new(),
                            kinv: Vec::new(),
                            ao: Vec::new(),
                            gate_pre: Vec::new(),
                            head_gain,
                        },
                    );
                }
                // Rebuild the host-visible acts from the raw plane, with
                // exactly the host split's bias/gate arithmetic.
                let mut qraw = vec![0f32; n * qrows];
                let mut kpre = vec![0f32; n * kvdim];
                let mut vproj = vec![0f32; n * kvdim];
                for r in 0..n {
                    let row = &ch.qkv_plane[r * fused..(r + 1) * fused];
                    qraw[r * qrows..(r + 1) * qrows].copy_from_slice(&row[..qrows]);
                    kpre[r * kvdim..(r + 1) * kvdim].copy_from_slice(&row[qrows..qrows + kvdim]);
                    vproj[r * kvdim..(r + 1) * kvdim].copy_from_slice(&row[qrows + kvdim..]);
                }
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
                return (
                    attn_out,
                    AttnActs::Full {
                        qpre,
                        kpre,
                        vproj,
                        qrot: ch.qrot,
                        krot: ch.krot,
                        qinv: ch.qinv,
                        kinv: ch.kinv,
                        ao: ch.ao,
                        gate_pre,
                        head_gain,
                    },
                );
            }
        }

        // One fused submit; the split back into three buffers is a
        // memcpy that costs microseconds and keeps everything below
        // this line byte-identical to the unfused path.
        let mut qkv = vec![0f32; n * fused];
        ops::gemm_nt(n1, wqkv, &mut qkv, n, hsz, fused, pool);
        let mut qraw = vec![0f32; n * qrows];
        let mut kpre = vec![0f32; n * kvdim];
        let mut vproj = vec![0f32; n * kvdim];
        for r in 0..n {
            let row = &qkv[r * fused..(r + 1) * fused];
            qraw[r * qrows..(r + 1) * qrows].copy_from_slice(&row[..qrows]);
            kpre[r * kvdim..(r + 1) * kvdim].copy_from_slice(&row[qrows..qrows + kvdim]);
            vproj[r * kvdim..(r + 1) * kvdim].copy_from_slice(&row[qrows + kvdim..]);
        }
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
        self.qk_norm_rope(&mut qrot, q_norm.as_deref(), nh, t, *local_rope, &mut qinv);
        self.qk_norm_rope(&mut krot, k_norm.as_deref(), nkv, t, *local_rope, &mut kinv);

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
                    ops::attn_head_fwd_window(&q32, &k32, &v32, t, hd, hd, *window, &mut o32);
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
        let mut ao_eff: Vec<f32> = if *output_gate {
            ao.iter()
                .zip(&gate_pre)
                .map(|(&a, &g)| a * (1.0 / (1.0 + (-g).exp())))
                .collect()
        } else {
            ao.clone()
        };
        // The projected head gate's gains (computed above) scale each head.
        if !head_gain.is_empty() {
            for r in 0..n {
                for h in 0..nh {
                    let gain = head_gain[r * nh + h];
                    for a in &mut ao_eff[r * qdim + h * hd..r * qdim + (h + 1) * hd] {
                        *a *= gain;
                    }
                }
            }
        }
        let mut attn_out = vec![0f32; n * hsz];
        ops::gemm_nt(&ao_eff, wo, &mut attn_out, n, qdim, hsz, pool);
        (
            attn_out,
            AttnActs::Full {
                qpre,
                kpre,
                vproj,
                qrot,
                krot,
                qinv,
                kinv,
                ao,
                gate_pre,
                head_gain,
            },
        )
    }

    /// GDN forward (frozen operator, teacher AND student): batched
    /// projections → f64 conv+SiLU per sequence → pooled per-(seq,
    /// k-head) delta-rule recurrence → out_proj. Matches the runtime
    /// `gdn_forward` (parity-tested in fcd_gradcheck).
    fn gdn_attn_fwd(&self, attn: &FcdAttn, n1: &[f32], b: usize, t: usize) -> (Vec<f32>, AttnActs) {
        let FcdAttn::Gdn {
            wqkv,
            wz,
            wa,
            wb,
            conv,
            a_log,
            dt_bias,
            norm,
            wout,
        } = attn
        else {
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
        let nystrom = nystrom && !self.polish_only;
        let hsz = self.hidden;
        let n = b * t;
        let l = &self.layers[li];
        let pool = self.pool.as_deref();
        let inter = l.inter;

        // ── FFN backward ──
        let mut dn2 = vec![0f32; n * hsz];
        // A frozen layer (no weight grads wanted) rides the device chain
        // fed by the plane its forward parked: only dh2 goes down and dn2
        // comes back, where the host path uploads the 80 MB dgu concat.
        let mut bwd_fused = false;
        #[cfg(feature = "gpu")]
        if grads.is_none() {
            if let Some(gu) = wts.gu {
                if crate::gpu::enabled_here()
                    && crate::gpu_wgpu::ffn_bwd_chain_f32(
                        dh2, wts.down, gu, li, &mut dn2, n, hsz, inter, self.act, None, None,
                    )
                {
                    bwd_fused = true;
                }
            }
        }
        if !bwd_fused {
            let mut dact = vec![0f32; n * inter];
            ops::gemm_dx(dh2, wts.down, &mut dact, n, inter, hsz, pool);
            if let Some(g) = grads.as_deref_mut() {
                ops::gemm_dw(dh2, &acts.act, &mut g[4], n, inter, hsz, pool);
            }
            let mut dg = vec![0f32; n * inter];
            let mut du = vec![0f32; n * inter];
            act_bwd_rows(self.act, &acts.gpre, &acts.upre, &dact, None, &mut dg, &mut du, n, inter, pool);
            // Fused when the frozen concat exists (its gate/up slots are
            // empty by design); trained masters keep the two-call path.
            if let Some(gu) = wts.gu {
                let mut dgu = vec![0f32; n * 2 * inter];
                for r in 0..n {
                    let row = &mut dgu[r * 2 * inter..(r + 1) * 2 * inter];
                    row[..inter].copy_from_slice(&dg[r * inter..(r + 1) * inter]);
                    row[inter..].copy_from_slice(&du[r * inter..(r + 1) * inter]);
                }
                ops::gemm_dx(&dgu, gu, &mut dn2, n, hsz, 2 * inter, pool);
            } else {
                ops::gemm_dx(&dg, wts.gate, &mut dn2, n, hsz, inter, pool);
                ops::gemm_dx(&du, wts.up, &mut dn2, n, hsz, inter, pool);
            }
            if let Some(g) = grads.as_deref_mut() {
                ops::gemm_dw(&dg, &acts.n2, &mut g[2], n, hsz, inter, pool);
                ops::gemm_dw(&du, &acts.n2, &mut g[3], n, hsz, inter, pool);
            }
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
        let mats = self.mats(li).expect("layer mats");
        let dn1 = match &l.attn {
            FcdAttn::Full { .. } => {
                self.full_attn_bwd(&l.attn, &mats, &acts.attn, &dh1, b, t, nystrom)
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
        mats: &LayerMats,
        acts: &AttnActs,
        dattn: &[f32],
        b: usize,
        t: usize,
        nystrom: bool,
    ) -> Vec<f32> {
        let (wqkv, wo) = (&mats.wqkv[..], &mats.wo[..]);
        let FcdAttn::Full {
            qrows: _,
            q_norm,
            k_norm,
            output_gate,
            local_rope,
            window,
            ..
        } = attn
        else {
            unreachable!("full_attn_bwd on a non-Full layer");
        };
        let AttnActs::Full {
            qpre,
            kpre,
            vproj,
            qrot,
            krot,
            qinv,
            kinv,
            ao,
            gate_pre,
            head_gain,
        } = acts
        else {
            unreachable!("acts mismatch");
        };
        let (rotary_dim, inv_freq) = self.layer_rope(*local_rope);
        let (hsz, nh, nkv, hd) = (self.hidden, self.nh, self.nkv, self.hd);
        let n = b * t;
        let pool = self.pool.as_deref();
        let qdim = nh * hd;
        let kvdim = nkv * hd;
        let rep = nh / nkv;
        let qrows = if *output_gate { 2 * qdim } else { qdim };

        let mut dao_eff = vec![0f32; n * qdim];
        ops::gemm_dx(dattn, wo, &mut dao_eff, n, qdim, hsz, pool);
        // Projected head gate: the gain multiplies the whole head, so the
        // gradient scales by it; the gate's weights are frozen and its
        // input path is not chased (FCD policy: attention is closed-form).
        if !head_gain.is_empty() {
            for r in 0..n {
                for h in 0..nh {
                    let gain = head_gain[r * nh + h];
                    for d in &mut dao_eff[r * qdim + h * hd..r * qdim + (h + 1) * hd] {
                        *d *= gain;
                    }
                }
            }
        }
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
                        ops::attn_head_bwd_window(
                            &q64, &k64, &v64, &do64, t, hd, hd, *window, &mut dq64, &mut dk64,
                            &mut dv64,
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
                ops::rope_bwd(&mut dqrot[s..s + rotary_dim], pos, inv_freq);
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
                ops::rope_bwd(&mut dkrot[s..s + rotary_dim], pos, inv_freq);
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
        // One fused submit: wqkv's row layout is exactly what gemm_dx
        // wants, so the sum dq·Wq + dk·Wk + dv·Wv is a concat of the
        // gradients and a single call.
        let fused = qrows + 2 * kvdim;
        let mut dqkv = vec![0f32; n * fused];
        for r in 0..n {
            let row = &mut dqkv[r * fused..(r + 1) * fused];
            row[..qrows].copy_from_slice(&dqraw[r * qrows..(r + 1) * qrows]);
            row[qrows..qrows + kvdim].copy_from_slice(&dkpre[r * kvdim..(r + 1) * kvdim]);
            row[qrows + kvdim..].copy_from_slice(&dvproj[r * kvdim..(r + 1) * kvdim]);
        }
        let mut dn1 = vec![0f32; n * hsz];
        ops::gemm_dx(&dqkv, wqkv, &mut dn1, n, hsz, fused, pool);
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
        let FcdAttn::Gdn {
            wqkv,
            wz,
            wa,
            wb,
            conv,
            a_log,
            dt_bias,
            norm,
            wout,
        } = attn
        else {
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
        keep: Option<&mut Vec<Vec<f32>>>,
    ) -> Vec<f32> {
        self.forward_span(SpanIn::Ids(ids), 0, self.nl, b, t, ts, student, keep)
    }

    /// Layers `from..to` on the token embeddings of `ids` or on a hidden
    /// state entering layer `from`; `keep` collects each layer's input.
    #[allow(clippy::too_many_arguments)]
    fn forward_span(
        &self,
        input: SpanIn,
        from: usize,
        to: usize,
        b: usize,
        t: usize,
        ts: Option<&TrainState>,
        student: bool,
        mut keep: Option<&mut Vec<Vec<f32>>>,
    ) -> Vec<f32> {
        let hsz = self.hidden;
        let mut h = match input {
            SpanIn::Hidden(h) => h,
            SpanIn::Ids(ids) => {
                let mut h = vec![0f32; b * t * hsz];
                for (r, &id) in ids.iter().enumerate() {
                    let src = (id as usize).min(self.embed.len() / hsz - 1) * hsz;
                    h[r * hsz..(r + 1) * hsz].copy_from_slice(&self.embed[src..src + hsz]);
                }
                h
            }
        };
        for li in from..to {
            if let Some(k) = keep.as_deref_mut() {
                k.push(h.clone());
            }
            let mats = self.mats(li).expect("layer mats");
            let wts = ln_ffn(self, if student { ts } else { None }, li, &mats);
            let nys = student && self.o1_flags[li];
            h = self.layer_forward(li, &h, b, t, &wts, nys, false).0;
        }
        h
    }

    /// The vocabulary head of [`Self::loss_and_dhidden`] over normed rows:
    /// (Σ w·ce, Σ w·kl, dNormed). `inv_w` is 1/Σw of the whole batch.
    #[allow(clippy::too_many_arguments)]
    fn head_core(
        &self,
        ns: &[f32],
        nt: &[f32],
        targets: &[u32],
        kl_w: f64,
        weights: Option<&[f32]>,
        inv_w: f64,
    ) -> (f64, f64, Vec<f32>) {
        let hsz = self.hidden;
        let n = targets.len();
        let pos_w = |i: usize| weights.map_or(1.0, |w| w[i] as f64);
        let pool = self.pool.as_deref();
        let wh = self.head_weight();
        let vs = self.vocab;
        let mut ce_sum = 0f64;
        let mut kl_sum = 0f64;
        let mut dns = vec![0f32; n * hsz];
        let mut ls = vec![0f32; LM_CHUNK * vs];
        let mut lt = vec![0f32; LM_CHUNK * vs];
        let mut dlg = vec![0f32; LM_CHUNK * vs];
        let mut r0 = 0usize;
        let nth = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        while r0 < n {
            let r1 = (r0 + LM_CHUNK).min(n);
            let c = r1 - r0;
            // a chunk with nothing scored adds no loss and no gradient
            if weights.is_some_and(|w| w[r0..r1].iter().all(|&x| x == 0.0)) {
                r0 = r1;
                continue;
            }
            ops::gemm_nt(
                &ns[r0 * hsz..r1 * hsz],
                wh,
                &mut ls[..c * vs],
                c,
                hsz,
                vs,
                pool,
            );
            ops::gemm_nt(
                &nt[r0 * hsz..r1 * hsz],
                wh,
                &mut lt[..c * vs],
                c,
                hsz,
                vs,
                pool,
            );
            // Rows are independent (each writes only its own gradient row):
            // the vocabulary-wide softmaxes run on all cores.
            let per = c.div_ceil(nth.min(c)).max(1);
            let parts: Vec<(f64, f64)> = std::thread::scope(|sc| {
                let (ls, lt, pos_w) = (&ls, &lt, &pos_w);
                let hs: Vec<_> = dlg[..c * vs]
                    .chunks_mut(per * vs)
                    .enumerate()
                    .map(|(ti, dchunk)| {
                        sc.spawn(move || {
                            let mut acc = (0f64, 0f64);
                            for (k, drow) in dchunk.chunks_mut(vs).enumerate() {
                                let r = ti * per + k;
                                let w = pos_w(r0 + r);
                                if w == 0.0 {
                                    drow.fill(0.0);
                                    continue;
                                }
                                let (ce, kl) = ops::ce_kl_position(
                                    &ls[r * vs..(r + 1) * vs],
                                    &lt[r * vs..(r + 1) * vs],
                                    targets[r0 + r] as usize,
                                    kl_w,
                                    w * inv_w,
                                    drow,
                                );
                                acc.0 += w * ce;
                                acc.1 += w * kl;
                            }
                            acc
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().expect("ce/kl worker")).collect()
            });
            for (ce, kl) in parts {
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

        (ce_sum, kl_sum, dns)
    }

    /// Loss head: chunked tied-lm_head CE+KL against the teacher hidden,
    /// returning (ce_mean, kl_mean, dHidden_student). `weights` scales each
    /// position (the means are then weighted means; a 0 position adds
    /// nothing and gets no gradient).
    fn loss_and_dhidden(
        &self,
        hs: &[f32],
        ht: &[f32],
        targets: &[u32],
        kl_w: f64,
        weights: Option<&[f32]>,
    ) -> (f64, f64, Vec<f32>) {
        let hsz = self.hidden;
        let n = targets.len();
        let wsum: f64 = weights.map_or(n as f64, |w| w.iter().map(|&v| v as f64).sum());
        let inv_w = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };

        let mut ns = vec![0f32; n * hsz];
        let mut invs = vec![0f32; n];
        ops::rmsnorm_fwd(
            hs,
            &self.final_norm,
            self.eps,
            self.gemma,
            &mut ns,
            &mut invs,
        );
        let mut nt = vec![0f32; n * hsz];
        let mut invt = vec![0f32; n];
        ops::rmsnorm_fwd(
            ht,
            &self.final_norm,
            self.eps,
            self.gemma,
            &mut nt,
            &mut invt,
        );

        // Only scored rows reach the head (vocabulary GEMMs + softmaxes, the
        // dearest part of a step): an answer-only loss scores about half of
        // a window's positions.
        let (ce_sum, kl_sum, dns) = match weights {
            Some(w) if w.iter().any(|&x| x == 0.0) => {
                let rows: Vec<usize> = (0..n).filter(|&i| w[i] > 0.0).collect();
                let gather = |src: &[f32]| -> Vec<f32> {
                    rows.iter().flat_map(|&r| src[r * hsz..(r + 1) * hsz].iter().copied()).collect()
                };
                let (nsc, ntc) = (gather(&ns), gather(&nt));
                let tgc: Vec<u32> = rows.iter().map(|&r| targets[r]).collect();
                let wc: Vec<f32> = rows.iter().map(|&r| w[r]).collect();
                let (ce, kl, dnc) = self.head_core(&nsc, &ntc, &tgc, kl_w, Some(&wc), inv_w);
                let mut dns = vec![0f32; n * hsz];
                for (k, &r) in rows.iter().enumerate() {
                    dns[r * hsz..(r + 1) * hsz].copy_from_slice(&dnc[k * hsz..(k + 1) * hsz]);
                }
                (ce, kl, dns)
            }
            _ => self.head_core(&ns, &nt, targets, kl_w, weights, inv_w),
        };
        let mut dhs = vec![0f32; n * hsz];
        ops::rmsnorm_bwd(
            hs,
            &self.final_norm,
            &invs,
            &dns,
            self.gemma,
            &mut dhs,
            None,
        );
        (ce_sum * inv_w, kl_sum * inv_w, dhs)
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
        self.backward_from(0, b, t, keep, dh_last, ts)
    }

    /// [`Self::backward`] down to layer `from` only (`keep[i]` is the input
    /// of layer `from + i`): below the lowest trainable layer there is
    /// nothing to differentiate.
    fn backward_from(
        &self,
        from: usize,
        b: usize,
        t: usize,
        keep: &[Vec<f32>],
        dh_last: Vec<f32>,
        ts: &mut TrainState,
    ) {
        // Split-borrow: the weight view reads `data`, the grads write
        // `grad` — disjoint fields of TrainState.
        let TrainState {
            layers, data, grad, ..
        } = ts;
        let mut dh = dh_last;
        for li in (from..self.nl).rev() {
            let h_in = &keep[li - from];
            let nys = self.o1_flags[li];
            let slot = layers.iter().position(|&x| x == li);
            let mats_hold = self.mats(li).expect("layer mats");
            let wts = match slot {
                Some(s) => {
                    let bi = s * PARAMS_PER_LAYER;
                    LnFfn {
                        iln: &data[bi],
                        pln: &data[bi + 1],
                        gate: &data[bi + 2],
                        up: &data[bi + 3],
                        down: &data[bi + 4],
                        gu: None,
                    }
                }
                None => {
                    let l = &self.layers[li];
                    LnFfn {
                        iln: &l.iln,
                        pln: &l.pln,
                        gate: &[],
                        up: &[],
                        down: &mats_hold.down,
                        gu: Some(&mats_hold.gu),
                    }
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
        let (ce, kl, dhs) = self.loss_and_dhidden(&hs, &ht, tgt, kl_w, None);
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
            ops::rmsnorm_fwd(
                &h,
                &self.final_norm,
                self.eps,
                self.gemma,
                &mut ns,
                &mut inv,
            );
            let mut lg = vec![0f32; LM_CHUNK * vs];
            let mut r0 = 0usize;
            while r0 < n {
                let r1 = (r0 + LM_CHUNK).min(n);
                let c = r1 - r0;
                ops::gemm_nt(
                    &ns[r0 * hsz..r1 * hsz],
                    wh,
                    &mut lg[..c * vs],
                    c,
                    hsz,
                    vs,
                    pool,
                );
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

    /// [`Self::val_ppl`] over an [`FcdStream`]: a plain stream takes the
    /// original evenly-spaced windows; a record stream scores up to
    /// `4·nrounds·bs` records, evenly spaced, each window at its record's
    /// start (or centred on its first scored token when the prompt is
    /// longer than half a window), weighted by the stream's loss weights.
    pub fn val_ppl_stream(
        &self,
        va: &FcdStream,
        ts: Option<&TrainState>,
        student: bool,
        bs: usize,
        nrounds: usize,
        seq: usize,
    ) -> f64 {
        if va.is_plain() {
            return self.val_ppl(va.ids, ts, student, bs, nrounds, seq);
        }
        if va.ids.len() < seq + 2 || bs == 0 {
            return f64::NAN;
        }
        let recs = va.scored_records();
        let want = (4 * nrounds * bs).min(recs.len());
        if want == 0 {
            return f64::NAN;
        }
        let picks: Vec<(usize, Vec<f32>)> = (0..want)
            .map(|j| {
                let rec = recs[j * recs.len() / want];
                let (st, _, f, _) = rec;
                let max_off = va.ids.len() - seq - 1;
                let off = if f - st < seq / 2 { st } else { f - seq / 2 }.min(max_off);
                (off, va.target_weights(off, seq, rec))
            })
            .collect();
        let hsz = self.hidden;
        let wh = self.head_weight();
        let vs = self.vocab;
        let pool = self.pool.as_deref();
        let (mut nll, mut wsum) = (0f64, 0f64);
        for group in picks.chunks(bs) {
            let b = group.len();
            let mut ids = Vec::with_capacity(b * seq);
            let mut tgt = Vec::with_capacity(b * seq);
            let mut tw = Vec::with_capacity(b * seq);
            for (off, w) in group {
                ids.extend_from_slice(&va.ids[*off..*off + seq]);
                tgt.extend_from_slice(&va.ids[*off + 1..*off + seq + 1]);
                tw.extend_from_slice(w);
            }
            let h = self.forward_hidden(&ids, b, seq, ts, student, None);
            let n = b * seq;
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
                    let w = tw[r0 + r] as f64;
                    if w == 0.0 {
                        continue;
                    }
                    let row = &lg[r * vs..(r + 1) * vs];
                    let target = tgt[r0 + r] as usize;
                    let mx = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
                    let s: f64 = row.iter().map(|&v| (v as f64 - mx).exp()).sum();
                    nll += w * (mx + s.ln() - row[target.min(vs - 1)] as f64);
                    wsum += w;
                }
                r0 = r1;
            }
        }
        if wsum == 0.0 { f64::NAN } else { (nll / wsum).exp() }
    }

    /// Weighted NLL of `tgt` under final hidden `h` (rows of `tgt.len()`):
    /// (Σ w·nll, Σ w).
    fn nll_on(&self, h: &[f32], tgt: &[u32], w: &[f32]) -> (f64, f64) {
        let hsz = self.hidden;
        let wh = self.head_weight();
        let vs = self.vocab;
        let pool = self.pool.as_deref();
        let n = tgt.len();
        let mut ns = vec![0f32; n * hsz];
        let mut inv = vec![0f32; n];
        ops::rmsnorm_fwd(h, &self.final_norm, self.eps, self.gemma, &mut ns, &mut inv);
        let mut lg = vec![0f32; LM_CHUNK * vs];
        let (mut nll, mut wsum) = (0f64, 0f64);
        let mut r0 = 0usize;
        while r0 < n {
            let r1 = (r0 + LM_CHUNK).min(n);
            let c = r1 - r0;
            if w[r0..r1].iter().any(|&x| x > 0.0) {
                ops::gemm_nt(&ns[r0 * hsz..r1 * hsz], wh, &mut lg[..c * vs], c, hsz, vs, pool);
                let lg = &lg;
                let parts: Vec<(f64, f64)> = std::thread::scope(|sc| {
                    let hs: Vec<_> = (0..c)
                        .map(|r| {
                            sc.spawn(move || {
                                let wr = w[r0 + r] as f64;
                                if wr == 0.0 {
                                    return (0.0, 0.0);
                                }
                                let row = &lg[r * vs..(r + 1) * vs];
                                let mx = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
                                let se: f64 = row.iter().map(|&v| (v as f64 - mx).exp()).sum();
                                (wr * (mx + se.ln() - row[(tgt[r0 + r] as usize).min(vs - 1)] as f64), wr)
                            })
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().expect("nll worker")).collect()
                });
                for (a, b) in parts {
                    nll += a;
                    wsum += b;
                }
            }
            r0 = r1;
        }
        (nll, wsum)
    }
}

/// One fixed training/validation window of a record stream.
struct Slot {
    off: usize,
    w: Vec<f32>,
    wsum: f64,
}

/// Fixed windows over a record stream: a record that fits is one window
/// at its start; a longer one is covered by windows at half-window steps
/// (the last flush with its end). Windows without scored targets are left
/// out. Fixed windows are what lets the teacher and the frozen prefix be
/// computed once.
fn record_slots(s: &FcdStream, seq: usize, val: bool) -> Vec<Slot> {
    let max_off = s.ids.len() - seq - 1;
    let mut out = Vec::new();
    for rec in s.scored_records() {
        let (st, en, f, _) = rec;
        let offs: Vec<usize> = if en - st <= seq + 1 {
            vec![st.min(max_off)]
        } else if val {
            // one window per record, centred on the start of the answer
            vec![if f - st < seq / 2 { st } else { f - seq / 2 }.min(en - seq - 1).min(max_off)]
        } else {
            let last = (en - seq - 1).min(max_off);
            let mut v: Vec<usize> = (st..=last).step_by((seq / 2).max(1)).collect();
            if v.last() != Some(&last) {
                v.push(last);
            }
            v
        };
        for off in offs {
            let w = s.target_weights(off, seq, rec);
            let wsum: f64 = w.iter().map(|&x| x as f64).sum();
            if wsum > 0.0 {
                out.push(Slot { off, w, wsum });
            }
        }
    }
    out
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
/// Polish `model`, optionally distilling from a SEPARATE `teacher` file.
///
/// Without a teacher the KL anchor is the model's own frozen state,
/// which is right for an O(1) conversion (the weights are the same, only
/// the attention kernel moved) and useless for a compression: the anchor
/// is then the damage itself, and `--kl 0` is the only sane setting.
/// Pointing `teacher` at the UNCOMPRESSED file makes the anchor what the
/// model is supposed to do, everywhere — not just on the training text.
/// The two files share embeddings and head (a narrowing rewrites FFN
/// tensors only), so one loss head serves both exactly.
pub fn run_polish_distilled(
    model: &Arc<CmfModel>,
    teacher: Option<&Arc<CmfModel>>,
    o1: &O1Cfg,
    hp: &FcdHyper,
    tr: &[u32],
    va: &[u32],
    out: &std::path::Path,
    gate: Option<&GenGateCfg>,
) -> Result<FcdReport, String> {
    run_polish_stream(
        model,
        teacher,
        o1,
        hp,
        FcdStream::plain(tr),
        FcdStream::plain(va),
        out,
        gate,
        None,
    )
}

/// Files the record path runs through the runtime to build its window
/// cache, when they differ from the training files: e.g. q8_2f copies of
/// the teacher and the student, which the device executes, where an f16
/// file is expanded to f32 and runs on the host.
#[derive(Clone, Copy)]
pub struct CacheModels<'a> {
    pub teacher: &'a Arc<CmfModel>,
    pub student: &'a Arc<CmfModel>,
}

/// [`run_polish_distilled`] over [`FcdStream`]s: per-token loss weights
/// (answer-only distillation) and record-bounded windows. Plain streams
/// reproduce the corpus behaviour exactly.
#[allow(clippy::too_many_arguments)]
pub fn run_polish_stream(
    model: &Arc<CmfModel>,
    teacher: Option<&Arc<CmfModel>>,
    o1: &O1Cfg,
    hp: &FcdHyper,
    tr: FcdStream,
    va: FcdStream,
    out: &std::path::Path,
    gate: Option<&GenGateCfg>,
    cache_models: Option<CacheModels>,
) -> Result<FcdReport, String> {
    if tr.ids.len() < hp.seq + 2 {
        return Err(format!(
            "train corpus too small: {} tokens < seq+2 = {}",
            tr.ids.len(),
            hp.seq + 2
        ));
    }
    let tr_recs = tr.scored_records();
    if tr_recs.is_empty() {
        return Err("train stream has no scored positions".into());
    }
    // Records drawn in proportion to their scored targets, so every scored
    // token is equally likely to be trained on.
    let tr_cum: Vec<f64> = tr_recs
        .iter()
        .scan(0f64, |acc, &(_, _, f, l)| {
            *acc += (l - f) as f64;
            Some(*acc)
        })
        .collect();
    let mut fm = FcdModel::from_cmf(model, o1, hp.polish_only)?;
    fm.polish_only = hp.polish_only;
    let fm = fm;
    if !tr.is_plain() {
        // the record path never builds an f32 teacher replica: the runtime
        // computes the teacher's hidden states once per window
        let converted = fm.converted();
        if converted.is_empty() {
            return Err("no converted layers under this --o1 spec (nothing to polish)".into());
        }
        return polish_records(model, fm, teacher, converted, o1, hp, tr, va, out, gate, cache_models);
    }
    let tfm: Option<FcdModel> = match teacher {
        Some(t) => {
            let off = O1Cfg {
                layers: O1Layers::List(Vec::new()),
                m: o1.m,
                w: o1.w,
                sink: o1.sink,
                rect: o1.rect,
            };
            let mut t = FcdModel::from_cmf(t, &off, true)?;
            t.polish_only = true;
            if t.hidden != fm.hidden || t.nl != fm.nl {
                return Err(format!(
                    "teacher geometry differs: hidden {}/{} layers {}/{}",
                    t.hidden, fm.hidden, t.nl, fm.nl
                ));
            }
            tracing::info!("fcd: distilling from a separate teacher file");
            Some(t)
        }
        None => None,
    };
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
        tr.ids.len(),
        va.ids.len()
    );

    let mut ts = TrainState::new(&fm);
    let teacher_ppl = match &tfm {
        Some(t) => t.val_ppl_stream(&va, None, false, hp.bs, 2, hp.seq),
        None => fm.val_ppl_stream(&va, None, false, hp.bs, 2, hp.seq),
    };
    let ppl_start = fm.val_ppl_stream(&va, Some(&ts), true, hp.bs, 2, hp.seq);
    tracing::info!(
        "fcd: quick-val teacher ppl {teacher_ppl:.2} | zero-shot o1 student ppl {ppl_start:.2}"
    );

    // ── generation gate (claim 13): baseline at step 0 ──
    let mut gate_state: Option<(Pipeline, Vec<f64>)> = match gate {
        Some(g) if !g.prompts.is_empty() => {
            let greedy = SamplerConfig {
                temperature: 0.0,
                top_p: 1.0,
                presence_penalty: 0.0,
                top_k: 0,
                repetition_penalty: 1.0,
                min_p: 0.0,
                seed: Some(0),
                suppress_tokens: Vec::new(),
                penalty_window: 0,
            };
            let mut pipe = Pipeline::from_model(model, greedy)
                .map_err(|e| format!("gen-gate pipeline: {e}"))?;
            if !hp.polish_only {
                pipe.set_o1(Some(o1.clone()));
            }
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
    let init_snapshot: Option<Vec<Vec<f32>>> = gate_state.is_some().then(|| ts.data.clone());
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
        let mut tw: Vec<f32> = Vec::with_capacity(n_per_step);
        for _ in 0..hp.bs {
            let off = if tr.is_plain() {
                (rng.next_u64() as usize) % (tr.ids.len() - hp.seq - 1)
            } else {
                let total = *tr_cum.last().unwrap();
                let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
                let r = tr_cum.partition_point(|&c| c <= u).min(tr_recs.len() - 1);
                let rec = tr_recs[r];
                let p = rec.2 + (rng.next_u64() as usize) % (rec.3 - rec.2);
                let off = tr.window_start(rec, p, hp.seq, rng.next_u64());
                tw.extend(tr.target_weights(off, hp.seq, rec));
                off
            };
            ids.extend_from_slice(&tr.ids[off..off + hp.seq]);
            tgt.extend_from_slice(&tr.ids[off + 1..off + hp.seq + 1]);
        }

        let ht = match &tfm {
            Some(t) => t.forward_hidden(&ids, hp.bs, hp.seq, None, false, None),
            None => fm.forward_hidden(&ids, hp.bs, hp.seq, None, false, None),
        };
        let mut keep: Vec<Vec<f32>> = Vec::with_capacity(fm.nl);
        let hs = fm.forward_hidden(&ids, hp.bs, hp.seq, Some(&ts), true, Some(&mut keep));
        let (ce, kl, dhs) =
            fm.loss_and_dhidden(&hs, &ht, &tgt, hp.kl_w, (!tr.is_plain()).then_some(&tw[..]));
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
            let p = fm.val_ppl_stream(&va, Some(&ts), true, hp.bs, 2, hp.seq);
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
    let ppl_final = fm.val_ppl_stream(&va, Some(&ts), true, hp.bs, 6, hp.seq);
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

/// The polish over a record stream (`FcdStream` with record starts): fixed
/// windows, so the teacher's final hidden state and the student's hidden
/// state entering its lowest trainable layer are computed once per window
/// (the frozen prefix and the teacher never change) — the teacher is freed
/// before training — and every step runs and differentiates only the
/// trainable tail.
#[allow(clippy::too_many_arguments)]
fn polish_records(
    model: &Arc<CmfModel>,
    fm: FcdModel,
    teacher: Option<&Arc<CmfModel>>,
    converted: Vec<usize>,
    o1: &O1Cfg,
    hp: &FcdHyper,
    tr: FcdStream,
    va: FcdStream,
    out: &std::path::Path,
    gate: Option<&GenGateCfg>,
    cache_models: Option<CacheModels>,
) -> Result<FcdReport, String> {
    let (t, hsz) = (hp.seq, fm.hidden);
    let mut ts = TrainState::new(&fm);
    let from = *ts.layers.iter().min().ok_or("no trainable layers")?;
    let tr_slots = record_slots(&tr, t, false);
    let mut va_slots = record_slots(&va, t, true);
    if va_slots.len() > 64 {
        // 64 val windows, evenly spaced over the records
        let k = va_slots.len();
        let keep: std::collections::HashSet<usize> = (0..64).map(|j| j * k / 64).collect();
        va_slots = va_slots.into_iter().enumerate().filter(|(i, _)| keep.contains(i)).map(|(_, s)| s).collect();
    }
    if tr_slots.is_empty() || va_slots.is_empty() {
        return Err("record streams have no scored windows".into());
    }
    let n_slots = tr_slots.len() + va_slots.len();
    let cache_bytes = n_slots as f64 * t as f64 * hsz as f64 * 4.0 * if from > 0 { 2.0 } else { 1.0 };
    tracing::info!(
        "fcd: {} train + {} val windows of {t}; trainable layers {}..{} — caching the teacher and the frozen prefix ({:.2} GB)",
        tr_slots.len(),
        va_slots.len(),
        from,
        fm.nl - 1,
        cache_bytes / 1e9
    );
    // ── the fixed parts of every window, computed once ──
    let t_cache = std::time::Instant::now();
    // The fixed parts come from the runtime's batched prefill (one window at
    // a time, the device where there is one): the teacher's final hidden
    // state and the student's state entering layer `from`. The training
    // replica's own forward is the fallback for an architecture without a
    // batched prefill.
    let runtime = |m: &Arc<CmfModel>| -> Option<Pipeline> {
        let mut p = Pipeline::from_model(m, SamplerConfig::default()).ok()?;
        p.set_confidence(false);
        p.can_prefill_batched().then_some(p)
    };
    let (t_file, s_file) = match cache_models {
        Some(c) => (c.teacher, c.student),
        None => (teacher.unwrap_or(model), model),
    };
    let mut t_rt = runtime(t_file);
    let mut s_rt = if from > 0 { runtime(s_file) } else { None };
    let fill = |s: &FcdStream, slots: &[Slot], label: &str, t_rt: &mut Option<Pipeline>, s_rt: &mut Option<Pipeline>| -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
        let (mut pre, mut th) = (Vec::with_capacity(slots.len()), Vec::with_capacity(slots.len()));
        let tc = std::time::Instant::now();
        for (k, sl) in slots.iter().enumerate() {
            let ids = &s.ids[sl.off..sl.off + t];
            let h = match t_rt.as_mut() {
                Some(p) => p.final_hidden_rows(ids)?,
                None => fm.forward_hidden(ids, 1, t, None, false, None),
            };
            let p = if from == 0 {
                Vec::new()
            } else {
                match s_rt.as_mut() {
                    Some(p) => p.calib_window_span(ids, None, 0, from, from)?,
                    None => fm.forward_span(SpanIn::Ids(ids), 0, from, 1, t, None, true, None),
                }
            };
            if h.len() != t * hsz || (from > 0 && p.len() != t * hsz) {
                return Err(format!("window cache: hidden of {} / {} values, expected {}", h.len(), p.len(), t * hsz));
            }
            pre.push(p);
            th.push(h);
            if k % 64 == 63 {
                tracing::info!("fcd: {label} cache {}/{} windows ({:.0}s)", k + 1, slots.len(), tc.elapsed().as_secs_f64());
            }
        }
        Ok((pre, th))
    };
    // the whole-stack prefill must reproduce the span walk it replaces
    if let (Some(p), Some(sl)) = (t_rt.as_mut(), tr_slots.first()) {
        let ids = &tr.ids[sl.off..sl.off + t];
        let a = p.final_hidden_rows(ids)?;
        let b = p.calib_window_span(ids, None, 0, fm.nl, fm.nl)?;
        let num: f64 = a.iter().zip(&b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
        tracing::info!("fcd: teacher prefill vs span walk, relative difference {:.2e}", (num / den.max(1e-30)).sqrt());
    }
    // Val windows are scored at every evaluation: all of them up front.
    // Train windows are drawn at random, steps×bs of them with repeats: each
    // is computed when first drawn and kept, so the cache holds only what
    // the run uses (a fraction of a long-record data set).
    let (va_pre, va_th) = fill(&va, &va_slots, "val", &mut t_rt, &mut s_rt)?;
    let mut tr_pre: Vec<Option<Vec<f32>>> = vec![None; tr_slots.len()];
    let mut tr_th: Vec<Option<Vec<f32>>> = vec![None; tr_slots.len()];
    tracing::info!("fcd: val window cache built in {:.0}s; train windows are cached as they are drawn", t_cache.elapsed().as_secs_f64());

    let gather = |s: &FcdStream, slots: &[Slot], pre: &[Vec<f32>], th: &[Vec<f32>], pick: &[usize]| {
        let mut ids = Vec::with_capacity(pick.len() * t);
        let mut tgt = Vec::with_capacity(pick.len() * t);
        let mut w = Vec::with_capacity(pick.len() * t);
        let mut h0 = Vec::with_capacity(if from > 0 { pick.len() * t * hsz } else { 0 });
        let mut ht = Vec::with_capacity(pick.len() * t * hsz);
        for &i in pick {
            let sl = &slots[i];
            ids.extend_from_slice(&s.ids[sl.off..sl.off + t]);
            tgt.extend_from_slice(&s.ids[sl.off + 1..sl.off + t + 1]);
            w.extend_from_slice(&sl.w);
            h0.extend_from_slice(&pre[i]);
            ht.extend_from_slice(&th[i]);
        }
        (ids, tgt, w, h0, ht)
    };
    let student_hidden = |ids: &[u32], h0: Vec<f32>, b: usize, ts: &TrainState, keep: Option<&mut Vec<Vec<f32>>>| {
        if from > 0 {
            fm.forward_span(SpanIn::Hidden(h0), from, fm.nl, b, t, Some(ts), true, keep)
        } else {
            fm.forward_span(SpanIn::Ids(ids), 0, fm.nl, b, t, Some(ts), true, keep)
        }
    };
    // weighted val ppl over every val window (teacher: cached hiddens)
    let val_ppl = |ts: Option<&TrainState>| -> f64 {
        let (mut nll, mut ws) = (0f64, 0f64);
        let idx: Vec<usize> = (0..va_slots.len()).collect();
        for chunk in idx.chunks(hp.bs.max(1)) {
            let (ids, tgt, w, h0, ht) = gather(&va, &va_slots, &va_pre, &va_th, chunk);
            let h = match ts {
                Some(ts) => student_hidden(&ids, h0, chunk.len(), ts, None),
                None => ht,
            };
            let (a, b) = fm.nll_on(&h, &tgt, &w);
            nll += a;
            ws += b;
        }
        if ws == 0.0 { f64::NAN } else { (nll / ws).exp() }
    };
    let teacher_ppl = val_ppl(None);
    let ppl_start = val_ppl(Some(&ts));
    tracing::info!("fcd: val ppl (scored tokens) teacher {teacher_ppl:.3} | student before training {ppl_start:.3}");

    let mut gate_state: Option<(Pipeline, Vec<f64>)> = match gate {
        Some(g) if !g.prompts.is_empty() => {
            let greedy = SamplerConfig {
                temperature: 0.0,
                top_p: 1.0,
                presence_penalty: 0.0,
                top_k: 0,
                repetition_penalty: 1.0,
                min_p: 0.0,
                seed: Some(0),
                suppress_tokens: Vec::new(),
                penalty_window: 0,
            };
            let mut pipe = Pipeline::from_model(model, greedy).map_err(|e| format!("gen-gate pipeline: {e}"))?;
            if !hp.polish_only {
                pipe.set_o1(Some(o1.clone()));
            }
            apply_trainables(&mut pipe, &fm, &ts);
            let base = gate_gen_scores(&mut pipe, g)?;
            tracing::info!("fcd gen-gate baseline loop-scores: {base:?}");
            Some((pipe, base))
        }
        _ => None,
    };
    let init_snapshot: Vec<Vec<f32>> = ts.data.clone();
    let mut gate_evals: Vec<(usize, f64, Vec<f64>, bool)> = Vec::new();
    // the zero-shot state is a candidate: training never makes val worse
    let mut best: (f64, Option<Vec<Vec<f32>>>, usize) = (ppl_start, None, 0);
    let cum: Vec<f64> = tr_slots
        .iter()
        .scan(0f64, |a, s| {
            *a += s.wsum;
            Some(*a)
        })
        .collect();
    let total = *cum.last().unwrap();
    let mut rng = SplitMix64::new(hp.seed);
    let mut losses: Vec<(f64, f64)> = Vec::with_capacity(hp.steps);
    let t0 = std::time::Instant::now();
    for st in 1..=hp.steps {
        let pick: Vec<usize> = (0..hp.bs)
            .map(|_| {
                let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
                cum.partition_point(|&c| c <= u).min(tr_slots.len() - 1)
            })
            .collect();
        for &i in &pick {
            if tr_th[i].is_none() {
                let (mut p, mut h) = fill(&tr, &tr_slots[i..i + 1], "train", &mut t_rt, &mut s_rt)?;
                tr_pre[i] = p.pop();
                tr_th[i] = h.pop();
            }
        }
        let pre_v: Vec<Vec<f32>> = pick.iter().map(|&i| tr_pre[i].clone().unwrap_or_default()).collect();
        let th_v: Vec<Vec<f32>> = pick.iter().map(|&i| tr_th[i].clone().unwrap_or_default()).collect();
        let local: Vec<usize> = (0..pick.len()).collect();
        let picked: Vec<Slot> = pick.iter().map(|&i| Slot { off: tr_slots[i].off, w: tr_slots[i].w.clone(), wsum: tr_slots[i].wsum }).collect();
        let (ids, tgt, w, h0, ht) = gather(&tr, &picked, &pre_v, &th_v, &local);
        let mut keep: Vec<Vec<f32>> = Vec::with_capacity(fm.nl - from);
        let t_step = std::time::Instant::now();
        let hs = student_hidden(&ids, h0, hp.bs, &ts, Some(&mut keep));
        let t_fwd = t_step.elapsed().as_secs_f64();
        let (ce, kl, dhs) = fm.loss_and_dhidden(&hs, &ht, &tgt, hp.kl_w, Some(&w));
        let t_loss = t_step.elapsed().as_secs_f64() - t_fwd;
        ts.zero_grad();
        fm.backward_from(from, hp.bs, t, &keep, dhs, &mut ts);
        let t_bwd = t_step.elapsed().as_secs_f64() - t_fwd - t_loss;
        let gn = ts.clip_and_step(hp.lr);
        if std::env::var("CMF_BAKE_PROF").is_ok() && st % 10 == 0 {
            tracing::info!(
                "fcd prof step {st}: forward {t_fwd:.2}s (attn {:.2} ffn {:.2}) · loss head {t_loss:.2}s · backward {t_bwd:.2}s · adam {:.2}s",
                prof::take(&prof::ATTN_FWD),
                prof::take(&prof::FFN_FWD),
                t_step.elapsed().as_secs_f64() - t_fwd - t_loss - t_bwd
            );
            let _ = prof::take(&prof::BWD);
        }
        losses.push((ce, kl));
        tracing::info!(
            "fcd step {st}/{}: ce {ce:.3} kl {kl:.3} |g| {gn:.3} ({:.1}s/step)",
            hp.steps,
            t0.elapsed().as_secs_f64() / st as f64
        );
        if hp.eval_every > 0 && (st % hp.eval_every == 0 || st == hp.steps) {
            let p = val_ppl(Some(&ts));
            let pass = match (&mut gate_state, gate) {
                (Some((pipe, base)), Some(g)) => {
                    apply_trainables(pipe, &fm, &ts);
                    let scores = gate_gen_scores(pipe, g)?;
                    let pass = gate_pass(&scores, base, g.threshold, g.baseline_slack);
                    gate_evals.push((st, p, scores, pass));
                    pass
                }
                _ => true,
            };
            let tag = if pass && p < best.0 {
                best = (p, Some(ts.data.clone()), st);
                " *best*"
            } else {
                ""
            };
            tracing::info!("fcd eval step {st}: val ppl {p:.3}{}{tag}", if pass { "" } else { " (gen-gate FAIL)" });
        }
    }
    drop((t_rt, s_rt));
    tracing::info!(
        "fcd: {} of {} train windows were drawn and cached",
        tr_th.iter().filter(|x| x.is_some()).count(),
        tr_slots.len()
    );
    let chosen = match best.1.take() {
        Some(snap) => {
            ts.data = snap;
            tracing::info!("fcd: restored best checkpoint from step {} (val ppl {:.3})", best.2, best.0);
            Some(best.2)
        }
        None => {
            ts.data = init_snapshot;
            tracing::info!("fcd: no checkpoint beat the untrained state on val — identity written");
            None
        }
    };
    let ppl_final = val_ppl(Some(&ts));
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
        gate: gate_state.map(|(_, base)| GateReport { baseline: base, evals: gate_evals, chosen }),
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
            act: crate::pipeline::Act::Silu,
            down_t: None,
            segs: Vec::new(),
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
    // A polish-only run names TRAINABLE layers with the o1 spec; its
    // attention stayed exact. Writing that spec as the runtime's O(1) hint
    // would switch those layers to Nyström attention at every later load.
    if !hp.polish_only {
        prov.insert(
            "o1_attn".into(),
            serde_json::json!({
                "layers": layers_json, "m": o1.m, "w": o1.w, "sink": o1.sink
            }),
        );
    }
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
            "polish_only": hp.polish_only,
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

    /// The bug this pins: on a hybrid, `--o1 deep8` under `--polish-only`
    /// used to select two layers instead of eight, because the eligibility
    /// filter for the O(1) CONVERSION also ran for a plain polish.
    #[test]
    fn polish_only_keeps_recurrent_layers_trainable() {
        // A Qwen3.5-shaped stretch: mostly GDN, a full-attention layer
        // every fourth. Ask for the deepest eight.
        let is_full: Vec<bool> = (0..8).map(|i| i % 4 == 3).collect();

        let mut converting = vec![true; 8];
        apply_layer_eligibility(&mut converting, &is_full, false);
        assert_eq!(
            converting.iter().filter(|&&b| b).count(),
            2,
            "an O(1) conversion may only take the full-attention layers"
        );

        let mut polishing = vec![true; 8];
        apply_layer_eligibility(&mut polishing, &is_full, true);
        assert_eq!(
            polishing.iter().filter(|&&b| b).count(),
            8,
            "a polish trains FFN and norms, which every layer has"
        );

        // An unflagged layer is never raised by either path.
        let mut none = vec![false; 8];
        apply_layer_eligibility(&mut none, &is_full, true);
        assert!(none.iter().all(|&b| !b));
    }

    /// `/proc/meminfo` reports the HOST inside a container. Believing it
    /// is what killed a 27B bake: 944 GB read, 117 GB actually available,
    /// the f32 replica declared to fit, SIGKILL a minute later.
    #[test]
    fn the_container_ceiling_wins_over_the_host_figure() {
        let host = 944 * 1_000_000_000u64;
        let cgroup = 117 * 1_000_000_000u64;
        assert_eq!(effective_total(host, Some(cgroup)), cgroup);
        // No cgroup (bare metal): the host figure is the truth.
        assert_eq!(effective_total(host, None), host);
        // A cgroup wider than the machine cannot conjure memory.
        assert_eq!(effective_total(host, Some(host * 2)), host);
    }

    /// Headroom inside a cgroup is limit-minus-held, not host-free: the
    /// host can have hundreds of gigabytes free that this process will
    /// never be allowed to touch.
    #[test]
    fn container_headroom_is_the_limit_minus_what_it_holds() {
        let host_free = 800 * 1_000_000_000u64;
        let limit = 117 * 1_000_000_000u64;
        let used = 79 * 1_000_000_000u64;
        assert_eq!(
            effective_available(host_free, Some(limit), Some(used)),
            38 * 1_000_000_000u64
        );
        // Over the limit already — report nothing left, never underflow.
        assert_eq!(
            effective_available(host_free, Some(limit), Some(limit + 1)),
            0
        );
        // Limit known, usage not: fall back to the limit itself.
        assert_eq!(effective_available(host_free, Some(limit), None), limit);
        // Neither known: the host figure stands.
        assert_eq!(effective_available(host_free, None, None), host_free);
    }

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
        assert!(
            gate_pass(&[0.35], &[0.25], 0.35, 0.10),
            "== base+slack passes"
        );
        assert!(!gate_pass(&[0.351], &base, 0.35, 0.10));
        assert!(!gate_pass(&[0.30], &[0.10], 0.35, 0.10), "0.30 > 0.10+0.10");
        let evals = vec![(25usize, 20.0, vec![0.10]), (50, 20.0, vec![0.10])];
        assert_eq!(
            select_checkpoint(&evals, &base, 0.35, 0.10),
            Some(0),
            "equal ppl → earliest checkpoint"
        );
    }
}

/// Backwards-compatible entry point: polish with the model as its own
/// KL anchor.
pub fn run_polish(
    model: &Arc<CmfModel>,
    o1: &O1Cfg,
    hp: &FcdHyper,
    tr: &[u32],
    va: &[u32],
    out: &std::path::Path,
    gate: Option<&GenGateCfg>,
) -> Result<FcdReport, String> {
    run_polish_distilled(model, None, o1, hp, tr, va, out, gate)
}
