//! Block-diffusion generation (Clover, `arch.block_diffusion`).
//!
//! The trunk is a causal Qwen3.5-MoE: the prompt and every resolved block
//! are committed through the ordinary batched prefill, which advances the
//! KV cache and the GDN states. Generation itself runs on blocks of
//! `block_size` rows that start as mask tokens: each denoising pass is a
//! PREVIEW of the whole block over the committed state, read-only —
//!
//! * full attention sees the prefix and every row of the block;
//! * a GDN layer replaces its recurrence by the release's block operator
//!   (right conv taps + a softmax over the block with one history slot,
//!   `linear_core::gdn_preview_batch`);
//! * the MoE keeps, for the whole block, the smallest expert set that
//!   holds `coreset_top_p` of the block's routed mass, and every row routes
//!   inside it.
//!
//! LowConfidence: after each pass every masked row whose greedy token is
//! more probable than `threshold` is fixed (at least the single most
//! confident one), until the block is resolved; then the block is
//! committed causally and the next one starts. Presence / frequency /
//! repetition penalties count committed response tokens only, and two
//! ramps bias the stop tokens and `</think>` late in long answers — the
//! vendor's semantics, `sampling_clover_moe.py`.
//!
//! Self-speculation (`CMF_CLOVER_MODE=draft`, the release's `draft-self`):
//! one preview of `[pending, mask…]` proposes a block, the causal path
//! verifies it in one batched forward, and the accepted prefix plus one
//! boundary token go out; the state rewinds to what was accepted.
//!
//! On wgpu the prompt, every pass and every commit are one batched-graph
//! submission each (`gpu::BlockPreview`, kernels in `gpu_wgpu::clover`),
//! the greedy pick included. On Apple silicon the passes run on the native
//! Metal executor (`gpu_metal::clover`) and the prompt and commits on the
//! CPU; elsewhere the request runs on the host CPU.
//! docs/CLOVER.md has the format record and the measurements.

use super::{dense_ffn_batch, moe_ffn_batch, AttnKind, FfnKind, GenerateResult, MoeFfn, Pipeline, PrefillIn, TokenCallback};
use crate::attention::{self, QwenAttnCfg};
use crate::inference;
use crate::linear_core::{gdn_preview_batch, GdnCfg, GdnPreviewWeights};
use crate::loader::{load_f32, Overlay};
use crate::pool::Pool;
use cortiq_core::{BlockDiffusionConfig, CmfError, CmfModel, LayerType, ModelArch};

pub struct Clover {
    pub cfg: BlockDiffusionConfig,
    /// One entry per layer; `Some` on the GDN layers.
    gdn: Vec<Option<GdnPreviewWeights>>,
}

/// Generation statistics of the last block-diffusion request.
#[derive(Clone, Copy, Debug, Default)]
pub struct CloverStats {
    /// Denoising passes (one block-wide forward each).
    pub passes: usize,
    /// Blocks resolved.
    pub blocks: usize,
    /// Wall time of the prompt ingest (first-use weight upload included),
    /// of the passes and of the block commits, seconds.
    pub prompt_s: f64,
    pub preview_s: f64,
    pub commit_s: f64,
    /// Inside the passes: GDN mixers, full attention, MoE (coreset
    /// included) and the head.
    pub gdn_s: f64,
    pub attn_s: f64,
    pub moe_s: f64,
    pub head_s: f64,
    /// Layers the device graph ran (the rest walked the host); 0 = host.
    pub device_layers: usize,
    /// Host time choosing the tokens of every pass (penalties, pick).
    pub pick_s: f64,
    /// Self-speculative rounds, their drafts and the accepted ones, and the
    /// causal verifies' time (0 = the request denoised).
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub verify_s: f64,
    /// Passes on the native Metal executor (`gpu_metal::clover`): its waits
    /// on the device, the host routing between layers and the expert-slot
    /// fills, seconds; the expert cache's hits and misses.
    pub metal_passes: usize,
    pub metal_gpu_s: f64,
    pub metal_exec_s: f64,
    pub metal_route_s: f64,
    pub metal_fill_s: f64,
    pub expert_hits: u64,
    pub expert_misses: u64,
    /// `CMF_CLOVER_METAL_PROF=1`: device seconds per stage (see
    /// `gpu_metal::clover::PreviewStats::stage_s`).
    pub metal_stage_s: [f64; 9],
}

/// What a pass on the Metal executor does (`clover_metal_pass`).
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq)]
enum MetalMode {
    /// A denoising pass: logits, nothing advanced.
    Preview,
    /// A causal commit: the state advances, no logits.
    Commit,
    /// A causal commit with the logits of every row (perplexity).
    Score,
}

/// Rows a causal pass on the Metal executor takes at once (the prompt,
/// perplexity): `CMF_CLOVER_METAL_CHUNK`, 128.
#[cfg(target_os = "macos")]
fn metal_chunk() -> usize {
    std::env::var("CMF_CLOVER_METAL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&c| (1..=256).contains(&c))
        .unwrap_or(128)
}

/// Block-diffusion requests started in this process: with the committed
/// length, the Metal executor's key for the state it has uploaded.
#[cfg(target_os = "macos")]
static REQUESTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Clover {
    /// Layer `li`'s preview weights for the batched device graph.
    pub(crate) fn graph_preview(&self, li: usize) -> Option<crate::gpu::GdnPreviewW<'_>> {
        self.gdn.get(li)?.as_ref().map(|p| crate::gpu::GdnPreviewW {
            conv_right: &p.conv_right,
            logit_scale: &p.logit_scale,
            history_bias: &p.history_bias,
            history_c: &p.history_c,
        })
    }
}

pub(crate) fn load(
    model: &CmfModel,
    arch: &ModelArch,
    gdn: Option<&GdnCfg>,
    ov: &Overlay,
) -> Result<Clover, CmfError> {
    let cfg = arch
        .block_diffusion
        .clone()
        .ok_or_else(|| CmfError::Parse("block diffusion: no record".into()))?;
    if cfg.algorithm != "low_confidence" {
        return Err(CmfError::Parse(format!(
            "block diffusion: algorithm '{}' is not supported (low_confidence only)",
            cfg.algorithm
        )));
    }
    if cfg.block_size == 0 || !(0.0..=1.0).contains(&cfg.threshold) {
        return Err(CmfError::Parse(format!(
            "block diffusion: block size {} / threshold {} out of range",
            cfg.block_size, cfg.threshold
        )));
    }
    let err = |e: String| CmfError::Parse(e);
    let mut layers = Vec::with_capacity(arch.num_layers);
    for li in 0..arch.num_layers {
        if arch.layer_types.get(li) != Some(&LayerType::LinearAttention) {
            layers.push(None);
            continue;
        }
        let g = gdn.ok_or_else(|| err("block diffusion: GDN layer without a GDN geometry".into()))?;
        let f = |s: &str| load_f32(model, &format!("model.layers.{li}.linear_attn.{s}"), ov).map_err(err);
        let pw = GdnPreviewWeights {
            conv_right: f("conv_right_weight")?,
            logit_scale: f("bd_softmax_logit_scale")?,
            history_bias: f("bd_softmax_history_bias")?,
            history_c: f("bd_softmax_history_c")?,
        };
        let nv = g.num_v_heads;
        if pw.conv_right.len() != g.conv_dim() * (g.conv_kernel - 1)
            || [&pw.logit_scale, &pw.history_bias, &pw.history_c].iter().any(|v| v.len() != nv)
        {
            return Err(err(format!("block diffusion: layer {li} preview weights have the wrong shape")));
        }
        layers.push(Some(pw));
    }
    Ok(Clover { cfg, gdn: layers })
}

/// The block's expert coreset (`coreset_torch`): every row's top-k
/// experts add their softmax probabilities, the totals renormalize, and
/// experts are kept in descending order while the mass BEFORE them is
/// below `top_p` (the first always).
fn coreset(m: &MoeFfn, xs: &[f32], b: usize, top_p: f32, pool: Option<&Pool>) -> Vec<bool> {
    let ne = m.experts.len();
    let mut logits = vec![0.0f32; b * ne];
    m.router.matmat(xs, b, &mut logits, pool);
    coreset_from_logits(m, &logits, top_p)
}

/// `coreset` over router logits already computed, `[rows × experts]`.
fn coreset_from_logits(m: &MoeFfn, logits: &[f32], top_p: f32) -> Vec<bool> {
    let ne = m.experts.len();
    let mut totals = vec![0f64; ne];
    let mut ids: Vec<usize> = Vec::with_capacity(ne);
    let k = m.top_k.min(ne);
    for row in logits.chunks_exact(ne) {
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = row.iter().map(|&l| (l as f64 - mx).exp()).sum();
        ids.clear();
        ids.extend(0..ne);
        // The top-k set — descending, lower index first on ties (a stable
        // argsort's prefix); its order inside does not matter.
        if k > 0 && k < ne {
            ids.select_nth_unstable_by(k - 1, |&x, &y| row[y].partial_cmp(&row[x]).unwrap().then(x.cmp(&y)));
        }
        for &e in &ids[..k] {
            totals[e] += (row[e] as f64 - mx).exp() / sum;
        }
    }
    coreset_from_totals(&totals, top_p)
}

/// The kept experts given each expert's summed top-k probability.
fn coreset_from_totals(totals: &[f64], top_p: f32) -> Vec<bool> {
    let total: f64 = totals.iter().sum::<f64>().max(1e-9);
    let mut order: Vec<usize> = (0..totals.len()).collect();
    order.sort_by(|&x, &y| totals[y].partial_cmp(&totals[x]).unwrap().then(x.cmp(&y)));
    let mut allowed = vec![false; totals.len()];
    let mut before = 0f64;
    for (i, &e) in order.iter().enumerate() {
        if i == 0 || before < top_p as f64 {
            allowed[e] = true;
        }
        before += totals[e] / total;
    }
    allowed
}

/// `CMF_CLOVER_DUMP=<path>`: the prompt ids and every row's top-10 raw
/// logits of the first denoising pass, as JSON — the parity probe against
/// the reference implementation's `model(block, preview=True)`.
fn dump_first_preview(prompt: &[u32], logits: &[f32], vocab: usize) {
    let Ok(path) = std::env::var("CMF_CLOVER_DUMP") else {
        return;
    };
    let rows: Vec<serde_json::Value> = logits
        .chunks_exact(vocab)
        .map(|row| {
            let mut ids: Vec<usize> = (0..vocab).collect();
            ids.sort_by(|&x, &y| row[y].partial_cmp(&row[x]).unwrap().then(x.cmp(&y)));
            serde_json::json!({
                "ids": &ids[..10],
                "logits": ids[..10].iter().map(|&i| row[i]).collect::<Vec<_>>(),
            })
        })
        .collect();
    let j = serde_json::json!({ "prompt_ids": prompt, "rows": rows });
    if let Err(e) = std::fs::write(&path, j.to_string()) {
        tracing::warn!("CMF_CLOVER_DUMP {path}: {e}");
    }
}

/// The first maximum (torch.argmax) and its softmax probability.
fn greedy_pick(row: &[f32]) -> (u32, f32) {
    let mut id = 0usize;
    for (i, &l) in row.iter().enumerate() {
        if l > row[id] {
            id = i;
        }
    }
    let mx = row[id];
    let sum: f64 = row.iter().map(|&l| (l - mx).exp() as f64).sum();
    (id as u32, (1.0 / sum) as f32)
}

/// How a block-diffusion request generates (`CMF_CLOVER_MODE`).
#[derive(Clone, Copy, PartialEq, Debug)]
enum CloverMode {
    /// LowConfidence denoising, the release's `generate`.
    Denoise,
    /// Previews draft, the causal path verifies (`draft-self`).
    DraftSelf,
    /// The ordinary causal decode.
    Causal,
}

/// The default is the causal decode wherever it was measured against the
/// other two: RTX 5090 8 ms a token against a 90 ms self-speculative round
/// (3.4 tokens on prose, 10 on code) and a 35 ms denoising pass (1.1, 2.6);
/// a 16 GB budget 53 tok/s against 5.6–14; the CPU 11–14 against 0.7–1.3.
fn clover_mode() -> CloverMode {
    match std::env::var("CMF_CLOVER_MODE").as_deref() {
        Ok("denoise") => CloverMode::Denoise,
        Ok("draft") => CloverMode::DraftSelf,
        Ok("ar") => CloverMode::Causal,
        _ if cfg!(target_os = "macos") => CloverMode::Denoise,
        _ => CloverMode::Causal,
    }
}

/// Which drafts a verify keeps (`clover_correct_drafts`).
#[derive(Clone, Copy)]
struct AcceptRule {
    /// Only drafts equal to the penalized row's argmax.
    exact: bool,
    ratio: f32,
    delta: f32,
}

impl AcceptRule {
    /// `CMF_CLOVER_ACCEPT=exact`, `CMF_CLOVER_ACCEPT_RATIO`,
    /// `CMF_CLOVER_ACCEPT_DELTA`; the defaults are the vendor's llama.cpp
    /// settings (0.9, 0.6).
    fn from_env() -> Self {
        let num = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        Self {
            exact: std::env::var("CMF_CLOVER_ACCEPT").as_deref() == Ok("exact"),
            ratio: num("CMF_CLOVER_ACCEPT_RATIO", 0.9),
            delta: num("CMF_CLOVER_ACCEPT_DELTA", 0.6),
        }
    }

    /// Whether the verify row `logits` (penalized) keeps the draft `cand`,
    /// and the draft's probability: `p ≥ min(ratio, delta·e^(-H))`, p > 0.
    fn accepts(&self, row: &[f32], cand: u32) -> (bool, f32) {
        let (mut mx, mut arg) = (f32::NEG_INFINITY, 0usize);
        for (i, &l) in row.iter().enumerate() {
            if l > mx {
                (mx, arg) = (l, i);
            }
        }
        let (mut z, mut w) = (0f64, 0f64);
        for &l in row {
            if l > f32::NEG_INFINITY {
                let d = (l - mx) as f64;
                let e = d.exp();
                z += e;
                w += e * d;
            }
        }
        let p = (((row[cand as usize] - mx) as f64).exp() / z) as f32;
        if self.exact {
            return (arg == cand as usize, p);
        }
        // H = ln z − Σ pᵢ·(lᵢ − max): no per-element log.
        let h = (z.ln() - w / z) as f32;
        (p >= self.ratio.min(self.delta * (-h).exp()) && p > 0.0, p)
    }
}

/// What a denoising pass hands the token choice.
enum PassOut {
    /// `[block × vocab]` logits, chosen on the host.
    Logits(Vec<f32>),
    /// Each row's greedy token and its probability, chosen on the device.
    Picks(Vec<(u32, f32)>),
}

/// Response-only penalty state of one request (`CloverPenaltyState`).
#[derive(Default)]
pub(super) struct Penalties {
    counts: Vec<u32>,
    seen: Vec<u32>,
    length: usize,
}

impl Penalties {
    fn commit(&mut self, ids: &[u32]) {
        self.length += ids.len();
        for &t in ids {
            let c = &mut self.counts[t as usize];
            if *c == 0 {
                self.seen.push(t);
            }
            *c += 1;
        }
    }

    /// Mutate one row's logits: frequency, presence, repetition, then the
    /// length ramp on the stop ids and the `</think>` ramp. `row_pos` is
    /// the 1-based response position this row would take.
    fn apply(&self, cfg: &BlockDiffusionConfig, logits: &mut [f32], row_pos: usize, stop: &[u32], close_open: bool) {
        for &t in &self.seen {
            let l = &mut logits[t as usize];
            *l -= self.counts[t as usize] as f32 * cfg.frequency_penalty;
            *l -= cfg.presence_penalty;
            if cfg.repetition_penalty != 1.0 {
                *l = if *l < 0.0 { *l * cfg.repetition_penalty } else { *l / cfg.repetition_penalty };
            }
        }
        let ramp = |start: usize, window: usize, strength: f32| {
            ((row_pos as f32 - start as f32) / window as f32).clamp(0.0, 1.0) * strength
        };
        if cfg.length_penalty != 0.0 {
            let add = ramp(cfg.length_penalty_start, cfg.length_penalty_window, cfg.length_penalty);
            for &t in stop {
                logits[t as usize] += add;
            }
        }
        if let (true, Some(close)) = (close_open && cfg.think_end_bias != 0.0, cfg.think_end_token_id) {
            logits[close as usize] += ramp(cfg.think_end_start, cfg.think_end_window, cfg.think_end_bias);
        }
    }
}

impl CloverStats {
    /// Generation rate after the prompt: tokens over the passes, the token
    /// picks and the block commits.
    pub fn denoise_tok_s(&self, tokens: usize) -> f64 {
        tokens as f64 / (self.preview_s + self.pick_s + self.commit_s + self.verify_s).max(1e-9)
    }
}

impl Pipeline {
    pub fn is_block_diffusion(&self) -> bool {
        self.clover.is_some()
    }

    /// Statistics of the last block-diffusion request.
    pub fn clover_stats(&self) -> CloverStats {
        self.clover_stats
    }

    /// Whether this request's blocks go through the batched device graph
    /// (`CMF_CLOVER_GRAPH=0` keeps them on the host).
    fn clover_graph_ready(&self) -> bool {
        crate::gpu::enabled_here()
            && crate::gpu::wgpu_graph_on(crate::gpu::GraphPhase::Prefill)
            && std::env::var("CMF_CLOVER_GRAPH").as_deref() != Ok("0")
    }

    /// `ids` at `pos` through the batched wgpu graph: a causal commit
    /// (`preview` None: the KV and GDN state advance) or a block preview
    /// (logits `[ids × vocab]`, nothing advanced). A stack the card cannot
    /// hold whole runs its leading layers on the device and the rest on the
    /// host — the same device prefix the prefill takes, so each layer's
    /// state stays on one side. `whole` remembers that the whole stack plus
    /// the head did not fit, so later previews go straight to the prefix.
    /// Ok(None) = the graph declined before touching any device state.
    fn clover_graph_pass(
        &mut self,
        clover: &Clover,
        ids: &[u32],
        pos: usize,
        preview: Option<crate::gpu::BlockPreview>,
        pick: Option<crate::gpu::PreviewPick<'_>>,
        whole: &mut bool,
        stats: &mut CloverStats,
    ) -> Result<Option<PassOut>, String> {
        let (hs, k, nl) = (self.hidden_size, ids.len(), self.num_layers);
        let embed = |me: &Self| {
            let mut h = Vec::with_capacity(k * hs);
            for &id in ids {
                h.extend_from_slice(&me.embed_single(id));
            }
            h
        };
        let positions: Vec<usize> = (pos..pos + k).collect();
        let fail = || Err("block diffusion: device graph failed after admission".to_string());
        let Some(p) = preview else {
            let mut hiddens = embed(self);
            let mut run = 0usize;
            match self.try_batch_graph_wgpu_prefix(&mut hiddens, &positions, k, None, Some(&mut run)) {
                crate::gpu::BatchGraphOutcome::Completed => {}
                crate::gpu::BatchGraphOutcome::Declined => return Ok(None),
                crate::gpu::BatchGraphOutcome::Failed => return fail(),
            }
            if run < nl {
                let _ = self.prefill_batch_span(PrefillIn::Hidden(&hiddens), pos, None, run, nl);
            }
            stats.device_layers = run;
            return Ok(Some(PassOut::Logits(Vec::new())));
        };
        let _scope = crate::gpu::enter_block_preview(p);
        if *whole {
            let (_, idx, kind, row_scale) = self
                .weights
                .lm_head
                .graph_weight()
                .ok_or("block preview: the head has no device layout")?;
            let lm = crate::gpu::GraphW {
                idx,
                kind,
                row_scale,
                data: &[],
                prism: crate::gpu::GraphPrismOp::None,
                affine: false,
            };
            let final_norm = self.weights.final_norm.clone();
            let mut logits = Vec::new();
            let mut picks = Vec::new();
            let mut hiddens = embed(self);
            let outcome = self.try_batch_graph_wgpu(
                &mut hiddens,
                &positions,
                k,
                Some(crate::gpu::SpecTail {
                    lm,
                    lm_rows: self.weights.lm_head.rows(),
                    final_norm: &final_norm,
                    logits_out: &mut logits,
                    pick: pick.map(|pk| (pk, &mut picks)),
                }),
            );
            match outcome {
                crate::gpu::BatchGraphOutcome::Completed if !picks.is_empty() => {
                    stats.device_layers = nl;
                    return Ok(Some(PassOut::Picks(picks)));
                }
                crate::gpu::BatchGraphOutcome::Completed => {
                    let (rows, v) = (self.weights.lm_head.rows(), self.vocab_size);
                    if rows != v {
                        let mut out = vec![f32::NEG_INFINITY; k * v];
                        for (src, dst) in logits.chunks_exact(rows).zip(out.chunks_exact_mut(v)) {
                            let n = rows.min(v);
                            dst[..n].copy_from_slice(&src[..n]);
                        }
                        logits = out;
                    }
                    stats.device_layers = nl;
                    return Ok(Some(PassOut::Logits(logits)));
                }
                crate::gpu::BatchGraphOutcome::Declined => *whole = false,
                crate::gpu::BatchGraphOutcome::Failed => return fail(),
            }
        }
        let mut hiddens = embed(self);
        let mut run = 0usize;
        match self.try_batch_graph_wgpu_prefix(&mut hiddens, &positions, k, None, Some(&mut run)) {
            crate::gpu::BatchGraphOutcome::Completed => {}
            crate::gpu::BatchGraphOutcome::Declined => return Ok(None),
            crate::gpu::BatchGraphOutcome::Failed => return fail(),
        }
        drop(_scope);
        stats.device_layers = run;
        self.clover_preview_layers(clover, &mut hiddens, run, pos, stats)?;
        Ok(Some(PassOut::Logits(self.clover_head(&hiddens, stats))))
    }

    /// The prompt through the device graph in prefill-sized chunks; false
    /// when the graph declined the first chunk (nothing committed).
    fn clover_graph_prompt(
        &mut self,
        clover: &Clover,
        ids: &[u32],
        stats: &mut CloverStats,
    ) -> Result<bool, String> {
        let chunk = self.prefill_chunk().max(1);
        let mut whole = true;
        for (i, part) in ids.chunks(chunk).enumerate() {
            match self.clover_graph_pass(clover, part, i * chunk, None, None, &mut whole, stats)? {
                Some(_) => {}
                None if i == 0 => return Ok(false),
                None => return Err("block diffusion: device graph declined a prompt chunk".into()),
            }
        }
        Ok(true)
    }

    /// Commit `ids` at absolute position `pos` (causal, state-advancing).
    fn clover_commit(&mut self, ids: &[u32], pos: usize) {
        let chunk = self.prefill_chunk().max(1);
        for (i, part) in ids.chunks(chunk).enumerate() {
            let _ = self.prefill_batch_span(PrefillIn::Ids(part), pos + i * chunk, None, 0, usize::MAX);
        }
    }

    /// One denoising pass: the block at absolute position `pos` over the
    /// committed state, logits `[block × vocab]`. Leaves every cache as
    /// it found it.
    fn clover_preview(
        &mut self,
        clover: &Clover,
        block: &[u32],
        pos: usize,
        stats: &mut CloverStats,
    ) -> Result<Vec<f32>, String> {
        let hs = self.hidden_size;
        let mut h: Vec<f32> = Vec::with_capacity(block.len() * hs);
        for &id in block {
            h.extend_from_slice(&self.embed_single(id));
        }
        self.clover_preview_layers(clover, &mut h, 0, pos, stats)?;
        Ok(self.clover_head(&h, stats))
    }

    /// A pass off the wgpu graph: the Metal executor while it holds, the
    /// host walk otherwise. A Metal failure sends the rest of the request
    /// to the host — the executor never touches the host's state.
    fn clover_preview_off_graph(
        &mut self,
        clover: &Clover,
        block: &[u32],
        pos: usize,
        metal: &mut bool,
        req: u64,
        stats: &mut CloverStats,
    ) -> Result<Vec<f32>, String> {
        #[cfg(target_os = "macos")]
        if *metal {
            match self.clover_metal_pass(clover, block, pos, req, MetalMode::Preview, stats) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    tracing::warn!("{e} — the passes go to the CPU");
                    *metal = false;
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (metal, req);
        self.clover_preview(clover, block, pos, stats)
    }

    /// Teacher-forced NLL of `ids` (targets from `start`) on the Metal
    /// executor's causal path — the path every Clover commit takes on Apple
    /// silicon, scored in chunks of 128 rows. None = the executor is off or
    /// declined before scoring anything.
    #[cfg(target_os = "macos")]
    pub(super) fn clover_metal_nll(&mut self, ids: &[u32], start: usize) -> Option<Result<(f64, usize), String>> {
        let clover = self.clover.clone()?;
        if std::env::var("CMF_CLOVER_METAL").as_deref() == Ok("0") || crate::gpu_metal::clover::ready().is_err() {
            return None;
        }
        self.clear_sequence_state();
        let req = REQUESTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let v = self.vocab_size;
        let (mut nll, mut cnt) = (0f64, 0usize);
        let mut stats = CloverStats::default();
        let chunk = metal_chunk();
        for (ci, part) in ids.chunks(chunk).enumerate() {
            let p0 = ci * chunk;
            let logits = match self.clover_metal_pass(&clover, part, p0, req, MetalMode::Score, &mut stats) {
                Ok(l) => l,
                Err(e) if ci == 0 => {
                    tracing::warn!("{e} — scoring on the CPU");
                    self.clear_sequence_state();
                    return None;
                }
                Err(e) => return Some(Err(e)),
            };
            for (r, row) in logits.chunks_exact(v).enumerate() {
                let t = p0 + r + 1;
                if t >= ids.len() || t < start.max(1) {
                    continue;
                }
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = mx + row.iter().map(|&l| (l as f64 - mx).exp()).sum::<f64>().ln();
                nll += lse - row[ids[t] as usize] as f64;
                cnt += 1;
            }
        }
        Some(Ok((nll, cnt)))
    }

    /// `CMF_CLOVER_METAL_CHECK=1` (diagnostic): the prompt just committed on
    /// the Metal executor is committed again on the CPU, and every layer's
    /// state (conv ring, S) or first K head is compared, the CPU's kept.
    #[cfg(target_os = "macos")]
    fn clover_metal_check(&mut self, ids: &[u32]) {
        let snap: Vec<(Vec<f32>, Vec<f32>)> = self
            .kv_cache
            .layers
            .iter()
            .map(|l| (l.linear_state.clone(), if l.num_kv_heads > 0 { l.head_keys(0).to_vec() } else { Vec::new() }))
            .collect();
        self.clear_sequence_state();
        self.clover_commit(ids, 0);
        let Some(gc) = self.gdn_cfg else { return };
        let ring = (gc.conv_kernel - 1) * gc.conv_dim();
        let md = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let mx = |a: &[f32]| a.iter().map(|x| x.abs()).fold(0f32, f32::max);
        for (li, (l, (st, k0))) in self.kv_cache.layers.iter().zip(&snap).enumerate() {
            if l.linear_state.len() == st.len() && st.len() > ring {
                let ((cr, cs), (mr, ms)) = (l.linear_state.split_at(ring), st.split_at(ring));
                eprintln!(
                    "metal check layer {li}: ring |d| {:.3e} (max {:.3e}) · S |d| {:.3e} (max {:.3e})",
                    md(cr, mr),
                    mx(cr),
                    md(cs, ms),
                    mx(cs)
                );
            } else if !k0.is_empty() {
                let ck = l.head_keys(0);
                eprintln!("metal check layer {li}: K head 0 |d| {:.3e} (max {:.3e})", md(ck, k0), mx(ck));
            }
        }
    }

    /// A commit off the wgpu graph: the Metal executor while it holds, the
    /// CPU otherwise. A Metal failure hands the rest of the request to the
    /// CPU, which resumes after the last chunk the host already holds.
    fn clover_commit_any(&mut self, clover: &Clover, ids: &[u32], pos: usize, metal: &mut bool, req: u64) {
        #[cfg(target_os = "macos")]
        if *metal {
            match self.clover_metal_commit(clover, ids, pos, req) {
                Ok(()) if pos == 0 && std::env::var("CMF_CLOVER_METAL_CHECK").is_ok_and(|v| v == "1") => {
                    self.clover_metal_check(ids);
                    return;
                }
                Ok(()) => return,
                Err(e) => {
                    tracing::warn!("{e} — the rest of the request runs on the CPU");
                    *metal = false;
                    let done = self.kv_cache.seq_len().saturating_sub(pos).min(ids.len());
                    self.clover_commit(&ids[done..], pos + done);
                    return;
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (clover, metal, req);
        self.clover_commit(ids, pos);
    }

    /// Whether this request's passes go to the native Metal executor
    /// (`CMF_CLOVER_METAL=0` keeps them on the CPU).
    #[cfg(target_os = "macos")]
    fn clover_metal_ready(&self) -> bool {
        if std::env::var("CMF_CLOVER_METAL").as_deref() == Ok("0") {
            return false;
        }
        match crate::gpu_metal::clover::ready() {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!("{e} — the passes go to the CPU");
                false
            }
        }
    }

    /// The Metal executor's description of every layer (`gpu_metal::clover`):
    /// tensor indices, host slices, the committed state; plus each layer's
    /// MoE (for the host routing) and the experts' width.
    #[cfg(target_os = "macos")]
    #[allow(clippy::type_complexity)]
    fn clover_metal_layers<'a>(
        &'a self,
        clover: &'a Clover,
        inv: &'a [std::sync::Arc<Vec<f32>>],
    ) -> Result<(Vec<crate::gpu_metal::clover::Layer<'a>>, Vec<&'a MoeFfn>, usize), String> {
        use crate::gpu_metal::clover as cm;
        use crate::qtensor::QTensor;
        let q8 = |t: &QTensor, what: &str| {
            t.q8_2f_parts().map(|p| p.0).ok_or_else(|| format!("clover metal: {what} is not q8_2f"))
        };
        let q4 = |t: &QTensor| t.q4tp_parts().map(|p| p.0).ok_or_else(|| "clover metal: an expert is not q4tp".to_string());
        fn f32p<'t>(t: &'t QTensor, what: &str) -> Result<&'t [f32], String> {
            t.f32_parts().map(|p| p.0).ok_or_else(|| format!("clover metal: {what} is not f32 on the host"))
        }
        let nl = self.num_layers;
        let mut moes: Vec<&MoeFfn> = Vec::with_capacity(nl);
        let mut layers = Vec::with_capacity(nl);
        let mut inter = 0usize;
        for li in 0..nl {
            let lw = &self.weights.layers[self.phys_layer(li)];
            if lw.attn_out_norm.is_some() || lw.ffn_out_norm.is_some() || lw.layer_scale.is_some() {
                return Err(format!("clover metal: layer {li} has sandwich norms"));
            }
            let mixer = match &lw.attn {
                AttnKind::LinearGdn(w) => {
                    let pw = clover.gdn[li].as_ref().ok_or("block diffusion: GDN layer without preview weights")?;
                    cm::Mixer::Gdn(cm::GdnLayer {
                        qkv: q8(&w.in_proj_qkv, "in_proj_qkv")?,
                        z: q8(&w.in_proj_z, "in_proj_z")?,
                        out: q8(&w.out_proj, "out_proj")?,
                        a: f32p(&w.in_proj_a, "in_proj_a")?,
                        bproj: f32p(&w.in_proj_b, "in_proj_b")?,
                        conv1d: &w.conv1d,
                        a_log: &w.a_log,
                        dt_bias: &w.dt_bias,
                        norm: &w.norm,
                        conv_right: &pw.conv_right,
                        logit_scale: &pw.logit_scale,
                        history_bias: &pw.history_bias,
                        history_c: &pw.history_c,
                        state: &self.kv_cache.layers[li].linear_state,
                    })
                }
                AttnKind::Full { wq, wk, wv, wo, q_norm, k_norm, output_gate, softplus_gate: None, bias: None } => {
                    let cache = &self.kv_cache.layers[li];
                    let (nkv, hd, rd) = self.layer_geom(li);
                    if cache.o1.is_some() || cache.sinks.is_some() || self.layer_v_dim(li) != hd || self.qk_norm_after_rope {
                        return Err(format!("clover metal: attention layer {li} is not plain full context"));
                    }
                    cm::Mixer::Attn(cm::AttnLayer {
                        wq: q8(wq, "q_proj")?,
                        wk: q8(wk, "k_proj")?,
                        wv: q8(wv, "v_proj")?,
                        wo: q8(wo, "o_proj")?,
                        q_norm: q_norm.as_deref(),
                        k_norm: k_norm.as_deref(),
                        output_gate: *output_gate,
                        nh: self.layer_num_heads(li),
                        nkv,
                        hd,
                        rd,
                        scale: self.attn_scale,
                        rope_scale: self.layer_rope_scale(li),
                        inv_freq: &inv[li],
                        k: (0..nkv).map(|g| cache.head_keys(g)).collect(),
                        v: (0..nkv).map(|g| cache.head_values(g)).collect(),
                    })
                }
                _ => return Err(format!("clover metal: layer {li} mixer has no device path")),
            };
            let FfnKind::Moe(m) = &lw.ffn else {
                return Err(format!("clover metal: layer {li} is not a MoE layer"));
            };
            let Some((se, Some(sg))) = &m.shared else {
                return Err("clover metal: the MoE has no gated shared expert".into());
            };
            if m.resonance.is_some() || m.experts.iter().chain(std::iter::once(se)).any(|e| e.act != super::Act::Silu || !e.segs.is_empty()) {
                return Err(format!("clover metal: layer {li} experts are not plain SwiGLU"));
            }
            inter = m.experts[0].gate_proj.rows();
            if se.gate_proj.rows() != inter || m.experts.iter().any(|e| e.gate_proj.rows() != inter) {
                return Err("clover metal: experts of unequal width".into());
            }
            let experts = m
                .experts
                .iter()
                .map(|e| Ok((q4(&e.gate_proj)?, q4(&e.up_proj)?, q4(&e.down_proj)?)))
                .collect::<Result<Vec<_>, String>>()?;
            layers.push(cm::Layer {
                input_norm: &lw.input_norm,
                post_norm: &lw.post_norm,
                mixer,
                moe: cm::MoeLayer {
                    router: f32p(&m.router, "the router")?,
                    sgate: f32p(sg, "the shared expert gate")?,
                    experts,
                    shared: (q4(&se.gate_proj)?, q4(&se.up_proj)?, q4(&se.down_proj)?),
                },
            });
            moes.push(m);
        }
        Ok((layers, moes, inter))
    }

    /// One pass (`causal` false: a preview of the block, logits
    /// `[block × vocab]`) or commit (`causal`: the rows advance the device
    /// state, then the host owners) on the native Metal executor, rows
    /// `ids` at `pos` of request `req`. Err = a layer the executor does not
    /// cover or a device failure; a preview never touches the host.
    #[cfg(target_os = "macos")]
    fn clover_metal_pass(
        &mut self,
        clover: &Clover,
        ids: &[u32],
        pos: usize,
        req: u64,
        mode: MetalMode,
        stats: &mut CloverStats,
    ) -> Result<Vec<f32>, String> {
        let causal = mode != MetalMode::Preview;
        use crate::gpu_metal::clover as cm;
        let (hs, b) = (self.hidden_size, ids.len());
        let gc = self.gdn_cfg.ok_or("clover metal: no GDN geometry")?;
        let mut h = Vec::with_capacity(b * hs);
        for &id in ids {
            h.extend_from_slice(&self.embed_single(id));
        }
        let mut logits = Vec::new();
        let mut ps = cm::PreviewStats::default();
        {
            let inv: Vec<std::sync::Arc<Vec<f32>>> = (0..self.num_layers).map(|li| self.layer_inv_freq(li)).collect();
            let (layers, moes, inter) = self.clover_metal_layers(clover, &inv)?;
            let model = self.weights.lm_head.graph_weight().ok_or("clover metal: the head is not mapped")?.0;
            let pass = cm::Pass {
                model,
                layers,
                final_norm: &self.weights.final_norm,
                lm_head: self.weights.lm_head.q8_2f_parts().ok_or("clover metal: the head is not q8_2f")?.0,
                lm_rows: self.weights.lm_head.rows(),
                hidden: hs,
                inter,
                eps: self.rms_eps as f32,
                gemma: self.norm_style == cortiq_core::types::NormStyle::Gemma,
                gdn: cm::GdnGeom {
                    nv: gc.num_v_heads,
                    nk: gc.num_k_heads,
                    dk: gc.key_head_dim,
                    dv: gc.value_head_dim,
                    kk: gc.conv_kernel,
                    eps: gc.rms_eps as f32,
                    gate_sigmoid: gc.output_gate_sigmoid,
                },
                epoch: (req, pos),
            };
            // The routing of the CPU path, on the device's router logits: a
            // preview inside the block's expert coreset, a commit plain.
            let top_p = (!causal && clover.cfg.coreset_enabled).then_some(clover.cfg.coreset_top_p);
            let mut route = |li: usize, logits: &[f32], rows: usize| -> Option<Vec<Vec<(usize, f32)>>> {
                let m = moes[li];
                let ne = m.experts.len();
                let allowed = top_p.map(|p| coreset_from_logits(m, logits, p));
                Some(
                    (0..rows)
                        .map(|r| {
                            let (idx, p, wsum) = super::moe_route(&logits[r * ne..(r + 1) * ne], m, allowed.as_deref());
                            idx.iter().map(|&e| (e, p[e] / wsum)).collect()
                        })
                        .collect(),
                )
            };
            match mode {
                MetalMode::Commit => cm::commit(&pass, &h, b, pos, &mut route, &mut ps)?,
                MetalMode::Score => cm::commit_scored(&pass, &h, b, pos, &mut route, &mut logits, &mut ps)?,
                MetalMode::Preview => cm::preview(&pass, &h, b, pos, &mut route, &mut logits, &mut ps)?,
            }
            if !logits.is_empty() {
                let (rows, v) = (pass.lm_rows, self.vocab_size);
                if rows != v {
                    let mut out = vec![f32::NEG_INFINITY; b * v];
                    for (src, dst) in logits.chunks_exact(rows).zip(out.chunks_exact_mut(v)) {
                        let n = rows.min(v);
                        dst[..n].copy_from_slice(&src[..n]);
                    }
                    logits = out;
                }
            }
        }
        if causal {
            let layers = &mut self.kv_cache.layers;
            cm::commit_results(&mut |li, part| match part {
                cm::CommitPart::State(st) => {
                    let ls = &mut layers[li].linear_state;
                    ls.clear();
                    ls.extend_from_slice(st);
                }
                cm::CommitPart::Kv(k, v) => {
                    let row = k.len() / b;
                    for r in 0..b {
                        layers[li].append(&k[r * row..(r + 1) * row], &v[r * row..(r + 1) * row], &[]);
                    }
                }
            })?;
        } else {
            stats.metal_passes += 1;
            stats.metal_gpu_s += ps.gpu_s;
            stats.metal_exec_s += ps.exec_s;
            stats.metal_route_s += ps.route_s;
            stats.metal_fill_s += ps.fill_s;
            stats.expert_hits += ps.hits;
            stats.expert_misses += ps.misses;
            for (t, x) in stats.metal_stage_s.iter_mut().zip(ps.stage_s) {
                *t += x;
            }
        }
        Ok(logits)
    }

    /// Commit `ids` at `pos` on the Metal executor in chunks of up to 128
    /// rows (the prompt, a resolved block). Err leaves the host state at the
    /// last whole chunk; the caller restarts the request on the CPU.
    #[cfg(target_os = "macos")]
    fn clover_metal_commit(&mut self, clover: &Clover, ids: &[u32], pos: usize, req: u64) -> Result<(), String> {
        let mut stats = CloverStats::default();
        let chunk = metal_chunk();
        for (i, part) in ids.chunks(chunk).enumerate() {
            self.clover_metal_pass(clover, part, pos + i * chunk, req, MetalMode::Commit, &mut stats)?;
        }
        Ok(())
    }

    /// Layers `from..` of a preview on the host, rows `h` in place.
    fn clover_preview_layers(
        &mut self,
        clover: &Clover,
        h: &mut [f32],
        from: usize,
        pos: usize,
        stats: &mut CloverStats,
    ) -> Result<(), String> {
        let hs = self.hidden_size;
        let b = h.len() / hs;
        let (eps, ns) = (self.rms_eps, self.norm_style);
        let pool = self.pool.clone();
        let gdn_cfg = self.gdn_cfg;
        // On the host the layers keep to the CPU: behind a device prefix the
        // per-op uploads would stage the very weights that did not fit, and
        // without the batched graph (Apple silicon) the per-op device calls
        // of a 32-row block measured 10x slower than the CPU (M4: 12.4 s
        // against 1.25 s a pass).
        let _cpu = crate::gpu::enter_cpu_scope();
        let mut normed = vec![0.0f32; b * hs];
        for li in from..self.num_layers {
            let pl = self.phys_layer(li);
            let inv_freq = self.layer_inv_freq(li);
            let (nkv_l, hd_l, rd_l) = self.layer_geom(li);
            let (nh_l, rope_scale, v_dim) = (self.layer_num_heads(li), self.layer_rope_scale(li), self.layer_v_dim(li));
            let lw = &self.weights.layers[pl];
            for (src, dst) in h.chunks_exact(hs).zip(normed.chunks_exact_mut(hs)) {
                inference::rms_norm_into(src, &lw.input_norm, eps, ns, dst);
            }
            let t_attn = std::time::Instant::now();
            let mut attn = match &lw.attn {
                AttnKind::LinearGdn(w) => {
                    let pw = clover.gdn[li].as_ref().ok_or("block diffusion: GDN layer without preview weights")?;
                    let cfg = gdn_cfg.ok_or("gdn layer without gdn_cfg")?;
                    gdn_preview_batch(&normed, b, w, pw, &cfg, &self.kv_cache.layers[li].linear_state, pool.as_deref())
                }
                AttnKind::Full { wq, wk, wv, wo, q_norm, k_norm, output_gate, softplus_gate, bias } => {
                    if softplus_gate.is_some() {
                        return Err("block diffusion: projected attention gates are not supported".into());
                    }
                    let cfg = QwenAttnCfg {
                        mrope: None,
                        num_heads: nh_l,
                        num_kv_heads: nkv_l,
                        head_dim: hd_l,
                        hidden_size: hs,
                        position: pos,
                        inv_freq: &inv_freq,
                        rotary_dim: rd_l,
                        scale: self.attn_scale,
                        softcap: self.attn_softcap,
                        window: None,
                        v_norm: self.attn_v_norm,
                        qk_norm_after_rope: self.qk_norm_after_rope,
                        gate_sigmoid: self.proj_gate_sigmoid,
                        q_norm: q_norm.as_deref(),
                        k_norm: k_norm.as_deref(),
                        output_gate: *output_gate,
                        softplus_gate: None,
                        rope_scale,
                        bias: bias.as_ref().map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice())),
                        rms_eps: eps,
                        norm_style: ns,
                        pool: pool.as_deref(),
                        v_head_dim: v_dim,
                    };
                    attention::qwen_attention_preview(&normed, b, wq, wk, wv, wo, &mut self.kv_cache.layers[li], &cfg)
                }
                _ => return Err(format!("block diffusion: layer {li} kind has no preview")),
            };
            let attn_s = t_attn.elapsed().as_secs_f64();
            if matches!(lw.attn, AttnKind::LinearGdn(_)) {
                stats.gdn_s += attn_s;
            } else {
                stats.attn_s += attn_s;
            }
            if let Some(w) = &lw.attn_out_norm {
                for (src, dst) in attn.chunks_exact(hs).zip(normed.chunks_exact_mut(hs)) {
                    inference::rms_norm_into(src, w, eps, ns, dst);
                }
                attn.copy_from_slice(&normed);
            }
            for (x, a) in h.iter_mut().zip(&attn) {
                *x += a;
            }
            attention::recycle_buf(&mut attn);
            for (src, dst) in h.chunks_exact(hs).zip(normed.chunks_exact_mut(hs)) {
                inference::rms_norm_into(src, &lw.post_norm, eps, ns, dst);
            }
            let t_ffn = std::time::Instant::now();
            let ffn = match &lw.ffn {
                FfnKind::Moe(m) => {
                    let allowed = clover
                        .cfg
                        .coreset_enabled
                        .then(|| coreset(m, &normed, b, clover.cfg.coreset_top_p, pool.as_deref()));
                    moe_ffn_batch(m, &normed, b, hs, pool.as_deref(), allowed.as_deref())
                }
                FfnKind::Dense(d) => dense_ffn_batch(d, &normed, b, pool.as_deref(), None),
                FfnKind::DenseMoe(_) => return Err(format!("block diffusion: layer {li} FFN kind has no preview")),
            };
            stats.moe_s += t_ffn.elapsed().as_secs_f64();
            for (x, f) in h.iter_mut().zip(&ffn) {
                *x += f;
            }
        }
        Ok(())
    }

    /// Final norm and the head over every row: logits `[rows × vocab]`.
    fn clover_head(&self, h: &[f32], stats: &mut CloverStats) -> Vec<f32> {
        let hs = self.hidden_size;
        let b = h.len() / hs;
        let pool = self.pool.clone();
        let t_head = std::time::Instant::now();
        let mut normed = vec![0.0f32; b * hs];
        for (src, dst) in h.chunks_exact(hs).zip(normed.chunks_exact_mut(hs)) {
            inference::rms_norm_into(src, &self.weights.final_norm, self.rms_eps, self.norm_style, dst);
        }
        let rows = self.weights.lm_head.rows();
        let mut logits = vec![0.0f32; b * rows];
        self.weights.lm_head.matmat(&normed, b, &mut logits, pool.as_deref());
        if rows != self.vocab_size {
            let v = self.vocab_size;
            let mut out = vec![f32::NEG_INFINITY; b * v];
            for (src, dst) in logits.chunks_exact(rows).zip(out.chunks_exact_mut(v)) {
                let n = rows.min(v);
                dst[..n].copy_from_slice(&src[..n]);
            }
            logits = out;
        }
        stats.head_s += t_head.elapsed().as_secs_f64();
        logits
    }

    /// The host's token choice for the pending `rows` of a pass's logits:
    /// mask and suppressed ids out, the response penalties, then greedy
    /// rows across the pool (one vocab-wide softmax a row) or the sampler.
    #[allow(clippy::too_many_arguments)]
    fn clover_host_picks(
        &mut self,
        cfg: &BlockDiffusionConfig,
        logits: &mut [f32],
        rows: &[usize],
        pen: &Penalties,
        stop: &[u32],
        suppress: &[u32],
        close_open: bool,
    ) -> Vec<(u32, f32)> {
        let vocab = self.vocab_size;
        let mask = cfg.mask_token_id;
        let prepare = |r: usize, row: &mut [f32]| {
            row[mask as usize] = f32::NEG_INFINITY;
            for &s in suppress {
                if let Some(l) = row.get_mut(s as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
            pen.apply(cfg, row, pen.length + r + 1, stop, close_open);
        };
        let mut picks = vec![(0u32, 0f32); rows.len()];
        match self.pool.clone() {
            Some(pool) if self.sampler_config.temperature < 1e-6 => {
                let lp = crate::pool::SendMut::new(logits.as_mut_ptr());
                let pp = crate::pool::SendMutT::new(picks.as_mut_ptr());
                let prepare = &prepare;
                pool.run_rows(rows.len(), &|i0, i1| {
                    for i in i0..i1 {
                        let r = rows[i];
                        // SAFETY: row r and pick i belong to this job.
                        let row = unsafe { std::slice::from_raw_parts_mut(lp.at(r * vocab), vocab) };
                        prepare(r, row);
                        unsafe { *pp.at(i) = greedy_pick(row) };
                    }
                });
            }
            _ => {
                for (i, &r) in rows.iter().enumerate() {
                    let row = &mut logits[r * vocab..(r + 1) * vocab];
                    prepare(r, row);
                    picks[i] = self.clover_pick(row);
                }
            }
        }
        picks
    }

    /// Greedy (temperature 0) or the sampler's temperature / top-k /
    /// top-p draw; the confidence is the T=1 probability of the pick.
    fn clover_pick(&mut self, row: &[f32]) -> (u32, f32) {
        let sc = &self.sampler_config;
        if sc.temperature < 1e-6 {
            return greedy_pick(row);
        }
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = row.iter().map(|&l| ((l - mx) as f64).exp()).sum();
        let conf = |id: usize| (((row[id] - mx) as f64).exp() / sum) as f32;
        let t = sc.temperature;
        let mut cand: Vec<(usize, f32)> = row
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_finite())
            .map(|(i, &l)| (i, l / t))
            .collect();
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        if sc.top_k > 0 && (sc.top_k as usize) < cand.len() {
            // torch: keep every logit >= the k-th (ties at the cutoff stay).
            let cut = cand[sc.top_k as usize - 1].1;
            cand.retain(|c| c.1 >= cut);
        }
        let m0 = cand[0].1;
        let mut p: Vec<f64> = cand.iter().map(|c| ((c.1 - m0) as f64).exp()).collect();
        let z: f64 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= z);
        if sc.top_p > 0.0 && sc.top_p < 1.0 {
            let mut before = 0f64;
            let keep = p
                .iter()
                .position(|&x| {
                    let drop = before > sc.top_p as f64;
                    before += x;
                    drop
                })
                .unwrap_or(p.len());
            cand.truncate(keep.max(1));
            p.truncate(keep.max(1));
            let z: f64 = p.iter().sum();
            p.iter_mut().for_each(|x| *x /= z);
        }
        let mut u = self.rng.next_f32() as f64;
        let mut id = cand[cand.len() - 1].0;
        for (c, &pi) in cand.iter().zip(&p) {
            if u < pi {
                id = c.0;
                break;
            }
            u -= pi;
        }
        (id as u32, conf(id))
    }

    /// A self-speculative verify: `ids` at `pos` through the causal batched
    /// graph, the penalty-free logits of every row back. The state advances
    /// over all of them, with a GDN snapshot per row for
    /// `clover_keep_verified`. Ok(None) = declined, nothing moved.
    fn clover_verify_pass(&mut self, ids: &[u32], pos: usize) -> Result<Option<Vec<f32>>, String> {
        let (hs, k) = (self.hidden_size, ids.len());
        let (_, idx, kind, row_scale) = self
            .weights
            .lm_head
            .graph_weight()
            .ok_or("self-speculation: the head has no device layout")?;
        let lm = crate::gpu::GraphW {
            idx,
            kind,
            row_scale,
            data: &[],
            prism: crate::gpu::GraphPrismOp::None,
            affine: false,
        };
        let final_norm = self.weights.final_norm.clone();
        let lm_rows = self.weights.lm_head.rows();
        let mut h = Vec::with_capacity(k * hs);
        for &id in ids {
            h.extend_from_slice(&self.embed_single(id));
        }
        let positions: Vec<usize> = (pos..pos + k).collect();
        let mut logits = Vec::new();
        let outcome = self.try_batch_graph_wgpu(
            &mut h,
            &positions,
            k,
            Some(crate::gpu::SpecTail {
                lm,
                lm_rows,
                final_norm: &final_norm,
                logits_out: &mut logits,
                pick: None,
            }),
        );
        match outcome {
            crate::gpu::BatchGraphOutcome::Completed => {}
            crate::gpu::BatchGraphOutcome::Declined => return Ok(None),
            crate::gpu::BatchGraphOutcome::Failed => {
                return Err("self-speculation: device graph failed after admission".into());
            }
        }
        let v = self.vocab_size;
        if lm_rows != v {
            let mut out = vec![f32::NEG_INFINITY; k * v];
            for (src, dst) in logits.chunks_exact(lm_rows).zip(out.chunks_exact_mut(v)) {
                let n = lm_rows.min(v);
                dst[..n].copy_from_slice(&src[..n]);
            }
            logits = out;
        }
        Ok(Some(logits))
    }

    /// After a verify of `rows` rows at `pos`: keep the first `keep`, the
    /// GDN states and the KV mirrors back to them.
    #[cfg(feature = "gpu")]
    fn clover_keep_verified(&self, pos: usize, keep: usize, rows: usize) -> Result<(), String> {
        let gdn = self.graph_gdn_layer_count();
        if keep < rows && gdn > 0 && !crate::gpu::gdn_spec_restore(self.graph_kv_id, keep - 1, pos, gdn) {
            return Err("self-speculation: GDN restore failed".into());
        }
        if !self.rewind_trunk_graph_mirrors(pos + keep) {
            return Err("self-speculation: KV rewind failed".into());
        }
        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    fn clover_keep_verified(&self, _pos: usize, _keep: usize, _rows: usize) -> Result<(), String> {
        Err("self-speculation needs the device graph".into())
    }

    /// Self-speculative decoding on the device graph (`CloverWorker`): the
    /// pending token and masks through one preview propose the next rows,
    /// the causal verify scores them, the accepted prefix plus a token from
    /// the first rejected row go out. Ok(None) = the graph cannot run this
    /// (a stack that does not fit whole); the caller denoises instead.
    fn clover_draft_self(
        &mut self,
        clover: &Clover,
        cfg: &BlockDiffusionConfig,
        input_ids: &[u32],
        max_tokens: usize,
        on_token: &mut Option<TokenCallback>,
    ) -> Result<Option<GenerateResult>, String> {
        let (vocab, mask) = (self.vocab_size, cfg.mask_token_id);
        let width = std::env::var("CMF_CLOVER_DRAFT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&w| w >= 2 && cfg.block_size % w == 0)
            .unwrap_or(16.min(cfg.block_size));
        let rule = AcceptRule::from_env();
        let n = input_ids.len();
        if n == 0 {
            return Ok(None);
        }
        let mut stats = CloverStats::default();
        self.clear_sequence_state();
        self.last_prefill_tokens = n;
        // The prompt but its last token, then that token as the first verify:
        // its row samples the first output token.
        let t0 = std::time::Instant::now();
        if n > 1 && !self.clover_graph_prompt(clover, &input_ids[..n - 1], &mut stats)? {
            self.clear_sequence_state();
            return Ok(None);
        }
        let Some(mut first) = self.clover_verify_pass(&input_ids[n - 1..], n - 1)? else {
            self.clear_sequence_state();
            return Ok(None);
        };
        stats.prompt_s = t0.elapsed().as_secs_f64();
        let stop = self.clover_stop_ids(cfg);
        let is_stop = |t: u32| stop.binary_search(&t).is_ok();
        let suppress: Vec<u32> = self.sampler_config.suppress_tokens.clone();
        let forbid: Vec<u32> = std::iter::once(mask).chain(suppress.iter().copied()).collect();
        let device_pick = std::env::var_os("CMF_CLOVER_DUMP").is_none()
            && std::env::var("CMF_CLOVER_DEVICE_PICK").as_deref() != Ok("0");
        let preview_mode = crate::gpu::BlockPreview {
            coreset_top_p: cfg.coreset_enabled.then_some(cfg.coreset_top_p),
        };
        let mut pen = Penalties { counts: vec![0; vocab], seen: Vec::new(), length: 0 };
        let prepare = |row: &mut [f32], row_pos: usize, pen: &Penalties, close_open: bool| {
            row[mask as usize] = f32::NEG_INFINITY;
            for &s in &suppress {
                if let Some(l) = row.get_mut(s as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
            pen.apply(cfg, row, row_pos, &stop, close_open);
        };
        let open = |pen: &Penalties| {
            cfg.think_end_token_id.is_some_and(|c| pen.counts.get(c as usize).is_none_or(|&n| n == 0))
        };
        let mut committed = input_ids.to_vec();
        let mut out: Vec<u32> = Vec::new();
        let mut confidence: Vec<f32> = Vec::new();
        let mut finish = "length".to_string();
        prepare(&mut first[..vocab], 1, &pen, open(&pen));
        let (mut pending, conf) = self.clover_pick(&first[..vocab]);
        // The next tokens to hand out, in order; false ends the request.
        let mut emit = |me: &Self, ids: &[(u32, f32)], pen: &mut Penalties, finish: &mut String, out: &mut Vec<u32>| {
            for &(id, c) in ids {
                out.push(id);
                confidence.push(c);
                pen.commit(&[id]);
                if is_stop(id) && !me.ignore_eos {
                    *finish = "stop".into();
                    return false;
                }
                if let Some(cb) = on_token.as_mut() {
                    if !cb(&me.tokenizer.decode_token(id)) {
                        *finish = "cancelled".into();
                        return false;
                    }
                }
            }
            true
        };
        let mut going = max_tokens > 0 && emit(self, &[(pending, conf)], &mut pen, &mut finish, &mut out);
        let mut whole = true;
        while going && out.len() < max_tokens {
            if self.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                finish = "cancelled".into();
                break;
            }
            // Accepted drafts and the boundary token are `keep` new tokens,
            // so the round never commits past `max_tokens`.
            let room = max_tokens - out.len();
            let mut w = width;
            while w > room {
                w /= 2;
            }
            let w = w.max(1);
            let pos = committed.len();
            let mut cand = vec![mask; w];
            cand[0] = pending;
            if w > 1 {
                let t = std::time::Instant::now();
                let pick_spec = device_pick.then(|| crate::gpu::PreviewPick {
                    vocab,
                    counts: &pen.counts,
                    stop: &stop,
                    forbid: &forbid,
                    think_end: cfg.think_end_token_id,
                    close_open: open(&pen),
                    // Row 0 is the pending token, row r the response position
                    // `length + r` (`pen` already counts the pending one).
                    len0: pen.length - 1,
                    frequency: cfg.frequency_penalty,
                    presence: cfg.presence_penalty,
                    repetition: cfg.repetition_penalty,
                    length_ramp: (cfg.length_penalty, cfg.length_penalty_start as f32, cfg.length_penalty_window as f32),
                    think_ramp: (cfg.think_end_bias, cfg.think_end_start as f32, cfg.think_end_window as f32),
                });
                let Some(pass) = self.clover_graph_pass(clover, &cand, pos, Some(preview_mode), pick_spec, &mut whole, &mut stats)? else {
                    return Err("self-speculation: device graph declined a preview".into());
                };
                if !whole {
                    // Only the prefix of the stack fits: no verify graph.
                    self.clear_sequence_state();
                    return Ok(None);
                }
                stats.preview_s += t.elapsed().as_secs_f64();
                stats.passes += 1;
                let t = std::time::Instant::now();
                match pass {
                    PassOut::Picks(p) => {
                        for r in 1..w {
                            cand[r] = p[r].0;
                        }
                    }
                    PassOut::Logits(mut l) => {
                        for r in 1..w {
                            let row = &mut l[r * vocab..(r + 1) * vocab];
                            prepare(row, pen.length + r, &pen, open(&pen));
                            cand[r] = greedy_pick(row).0;
                        }
                    }
                }
                stats.pick_s += t.elapsed().as_secs_f64();
            }
            let t = std::time::Instant::now();
            let Some(mut logits) = self.clover_verify_pass(&cand, pos)? else {
                return Err("self-speculation: device graph declined a verify mid-sequence".into());
            };
            stats.verify_s += t.elapsed().as_secs_f64();
            // logits[j] verifies cand[j + 1]; the first row that rejects (or
            // the last) gives the boundary token.
            let t = std::time::Instant::now();
            let close_open = open(&pen);
            let mut kept: Vec<(u32, f32)> = Vec::with_capacity(w);
            for j in 0..w {
                let row = &mut logits[j * vocab..(j + 1) * vocab];
                prepare(row, pen.length + j + 1, &pen, close_open);
                if j + 1 == w {
                    break;
                }
                let (ok, p) = rule.accepts(row, cand[j + 1]);
                if !ok {
                    break;
                }
                kept.push((cand[j + 1], p));
            }
            let r = kept.len();
            let (bonus, c) = self.clover_pick(&logits[r * vocab..(r + 1) * vocab]);
            kept.push((bonus, c));
            stats.pick_s += t.elapsed().as_secs_f64();
            self.clover_keep_verified(pos, r + 1, w)?;
            committed.extend_from_slice(&cand[..=r]);
            stats.rounds += 1;
            stats.drafted += w - 1;
            stats.accepted += r;
            pending = bonus;
            going = emit(self, &kept, &mut pen, &mut finish, &mut out);
        }
        if finish == "cancelled" {
            self.clear_history();
        } else {
            self.record_consumed_prefix(&committed, 0);
        }
        stats.device_layers = self.num_layers;
        self.clover_stats = stats;
        Ok(Some(GenerateResult {
            text: self.tokenizer.decode(&out),
            token_ids: out.clone(),
            prompt_tokens: n,
            tokens_generated: out.len(),
            finish_reason: finish,
            mtp_drafted: 0,
            mtp_accepted: 0,
            token_confidence: confidence,
            traces: Vec::new(),
        }))
    }

    /// The ids that end an answer: the record's, the file's EOS ids.
    fn clover_stop_ids(&self, cfg: &BlockDiffusionConfig) -> Vec<u32> {
        let mut stop: Vec<u32> = cfg.stop_token_ids.clone();
        let tk = &self.tokenizer;
        stop.extend(tk.eos_token_id.into_iter().chain(tk.im_end_id).chain(tk.extra_eos.iter().copied()));
        stop.retain(|&t| (t as usize) < self.vocab_size);
        stop.sort_unstable();
        stop.dedup();
        stop
    }

    /// Whether this request takes the ordinary causal decode.
    pub(super) fn clover_causal_request(&mut self) -> bool {
        if self.clover.is_none() || clover_mode() != CloverMode::Causal {
            return false;
        }
        self.clover_stats = CloverStats::default();
        true
    }

    /// The causal decode's share of the release's response processing, on
    /// the row in place: the mask token never comes out, then the response
    /// penalties and the two ramps, the same `Penalties::apply` a preview
    /// row takes. Only the ids seen so far change, so the sampler's greedy
    /// is a plain argmax instead of a penalized pass over the 248k vocab
    /// (RTX 5090: 1 ms of an 8 ms token). `response` = the ids generated so far.
    pub(super) fn clover_causal_logits(&mut self, logits: &mut [f32], generated: usize, response: &[u32]) {
        let Some(clover) = self.clover.clone() else {
            return;
        };
        let (cfg, v) = (&clover.cfg, self.vocab_size);
        if logits.len() < v {
            return;
        }
        logits[cfg.mask_token_id as usize] = f32::NEG_INFINITY;
        if generated == 0 || self.clover_pen.counts.len() != v {
            self.clover_pen = Penalties { counts: vec![0; v], seen: Vec::new(), length: 0 };
        }
        let done = self.clover_pen.length;
        self.clover_pen.commit(&response[done..]);
        let close_open = cfg.think_end_token_id.is_some_and(|c| self.clover_pen.counts[c as usize] == 0);
        // The stop ids matter only past the length ramp's start.
        let stop = if cfg.length_penalty != 0.0 && generated >= cfg.length_penalty_start {
            self.clover_stop_ids(cfg)
        } else {
            Vec::new()
        };
        self.clover_pen.apply(cfg, logits, generated + 1, &stop, close_open);
    }

    /// Block LowConfidence generation (the `generate` of the release).
    pub(super) fn clover_generate(
        &mut self,
        input_ids: &[u32],
        max_tokens: usize,
        mut on_token: Option<TokenCallback>,
    ) -> Result<GenerateResult, String> {
        // Shared, not taken: the device graph's builder reads the layers'
        // preview weights through `self.clover` while this runs.
        let clover = self.clover.clone().ok_or("not a block-diffusion model")?;
        let r = self.clover_generate_with(&clover, input_ids, max_tokens, &mut on_token);
        if r.is_err() {
            self.clear_sequence_state();
        }
        r
    }

    fn clover_generate_with(
        &mut self,
        clover: &Clover,
        input_ids: &[u32],
        max_tokens: usize,
        on_token: &mut Option<TokenCallback>,
    ) -> Result<GenerateResult, String> {
        // `CMF_CLOVER_THRESHOLD` / `CMF_CLOVER_BLOCK`: the release's own
        // generate overrides (a lower threshold fixes more rows a pass; a
        // block must divide the trained one).
        let mut cfg_own = clover.cfg.clone();
        if let Some(t) = std::env::var("CMF_CLOVER_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|t| (0.0..=1.0).contains(t))
        {
            cfg_own.threshold = t;
        }
        if let Some(b) = std::env::var("CMF_CLOVER_BLOCK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&b| b > 0 && clover.cfg.block_size % b == 0)
        {
            cfg_own.block_size = b;
        }
        let cfg = &cfg_own;
        let (bs, vocab, mask) = (cfg.block_size, self.vocab_size, cfg.mask_token_id);
        self.cancel.store(false, std::sync::atomic::Ordering::Relaxed);
        let mut stats = CloverStats::default();
        // Device graph: the prompt, every preview and every commit in one
        // submission each; the host walk otherwise.
        let mut graph = self.clover_graph_ready();
        if graph && clover_mode() == CloverMode::DraftSelf {
            if let Some(r) = self.clover_draft_self(clover, cfg, input_ids, max_tokens, on_token)? {
                return Ok(r);
            }
            // Nothing committed and the state clear: denoise instead.
            graph = self.clover_graph_ready();
        }
        // Without the batched graph the whole request keeps to the CPU: the
        // per-op device path of a 32-row block lost to it everywhere it was
        // measured (Apple M4, 24 GB: passes 1.43 against 1.25 s, the prompt
        // 10.7 against 2.0 s — Metal kept the 19 GB of weights resident
        // where the CPU touches only the pages a pass reads).
        let _cpu = (!graph).then(crate::gpu::enter_cpu_scope);
        // Apple silicon: the passes on the native Metal executor instead.
        #[cfg(target_os = "macos")]
        let mut metal = !graph && self.clover_metal_ready();
        #[cfg(not(target_os = "macos"))]
        let mut metal = false;
        #[cfg(target_os = "macos")]
        let req = REQUESTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        #[cfg(not(target_os = "macos"))]
        let req = 0u64;
        let preview_mode = crate::gpu::BlockPreview {
            coreset_top_p: cfg.coreset_enabled.then_some(cfg.coreset_top_p),
        };
        // Cross-turn reuse (host path): a prompt that extends the committed
        // sequence prefills its tail only (the commits are causal, so exact).
        let mut reuse = if graph || std::env::var("CMF_KV_REUSE").is_ok_and(|v| v == "0") {
            0
        } else {
            self.cached_prefix_len(input_ids)
        };
        if reuse > 0 && !self.prepare_kv_reuse(reuse) {
            reuse = 0;
        }
        if reuse == 0 {
            self.clear_sequence_state();
        }
        self.last_prefill_tokens = input_ids.len() - reuse;
        let t0 = std::time::Instant::now();
        let mut whole = true;
        if graph && !self.clover_graph_prompt(clover, input_ids, &mut stats)? {
            graph = false;
        }
        if !graph {
            self.clover_commit_any(clover, &input_ids[reuse..], reuse, &mut metal, req);
        }
        stats.prompt_s = t0.elapsed().as_secs_f64();
        let mut committed: Vec<u32> = input_ids.to_vec();

        // The length ramp's stop ids; the file's EOS ids end the answer too.
        let stop = self.clover_stop_ids(cfg);
        let is_stop = |t: u32| stop.binary_search(&t).is_ok();
        let suppress: Vec<u32> = self.sampler_config.suppress_tokens.clone();
        let forbid: Vec<u32> = std::iter::once(mask).chain(suppress.iter().copied()).collect();
        // The first-pass parity dump needs the logits on the host.
        let device_pick_ok = std::env::var_os("CMF_CLOVER_DUMP").is_none()
            && std::env::var("CMF_CLOVER_DEVICE_PICK").as_deref() != Ok("0");
        let mut pen = Penalties { counts: vec![0; vocab], seen: Vec::new(), length: 0 };

        let mut out: Vec<u32> = Vec::new();
        let mut confidence: Vec<f32> = Vec::new();
        let mut finish = "length".to_string();
        'generation: while out.len() < max_tokens {
            let remaining = bs.min(max_tokens - out.len());
            let mut block = vec![mask; bs];
            let mut pending = vec![true; bs];
            let mut conf_of = vec![0f32; bs];
            let pos = committed.len();
            let mut passes = 0;
            while pending.iter().any(|&p| p) {
                if passes == bs {
                    return Err("block diffusion: denoising step limit reached with unresolved masks".into());
                }
                if self.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    finish = "cancelled".into();
                    break 'generation;
                }
                let t = std::time::Instant::now();
                let close_open = cfg.think_end_token_id.is_some_and(|c| {
                    pen.counts.get(c as usize).is_none_or(|&n| n == 0) && !block.contains(&c)
                });
                let rows: Vec<usize> = (0..bs).filter(|&r| pending[r]).collect();
                // Greedy on the whole-stack graph: penalties and the pick run
                // on the device (`CMF_CLOVER_DEVICE_PICK=0` reads the logits).
                let device_pick = device_pick_ok && self.sampler_config.temperature < 1e-6;
                let pick_spec = device_pick.then(|| crate::gpu::PreviewPick {
                    vocab,
                    counts: &pen.counts,
                    stop: &stop,
                    forbid: &forbid,
                    think_end: cfg.think_end_token_id,
                    close_open,
                    len0: pen.length,
                    frequency: cfg.frequency_penalty,
                    presence: cfg.presence_penalty,
                    repetition: cfg.repetition_penalty,
                    length_ramp: (
                        cfg.length_penalty,
                        cfg.length_penalty_start as f32,
                        cfg.length_penalty_window as f32,
                    ),
                    think_ramp: (
                        cfg.think_end_bias,
                        cfg.think_end_start as f32,
                        cfg.think_end_window as f32,
                    ),
                });
                let graph_out = if graph {
                    self.clover_graph_pass(clover, &block, pos, Some(preview_mode), pick_spec, &mut whole, &mut stats)?
                } else {
                    None
                };
                let mut out = match graph_out {
                    Some(o) => o,
                    // The first preview declined: the device holds only the
                    // prompt — redo it on the host and stay there.
                    None if graph && stats.blocks == 0 && passes == 0 => {
                        tracing::warn!("block preview declined by the device graph: host walk");
                        graph = false;
                        self.clear_sequence_state();
                        self.clover_commit(input_ids, 0);
                        PassOut::Logits(self.clover_preview(clover, &block, pos, &mut stats)?)
                    }
                    None if graph => {
                        return Err("block diffusion: device graph declined a preview mid-sequence".into());
                    }
                    None => PassOut::Logits(self.clover_preview_off_graph(clover, &block, pos, &mut metal, req, &mut stats)?),
                };
                stats.preview_s += t.elapsed().as_secs_f64();
                if let (0, PassOut::Logits(l)) = (stats.passes + passes, &out) {
                    dump_first_preview(&committed, l, vocab);
                }
                passes += 1;
                let t_pick = std::time::Instant::now();
                let picks: Vec<(u32, f32)> = match &mut out {
                    PassOut::Picks(p) => rows.iter().map(|&r| p[r]).collect(),
                    PassOut::Logits(logits) => self.clover_host_picks(cfg, logits, &rows, &pen, &stop, &suppress, close_open),
                };
                let mut fixed: Vec<(usize, u32, f32)> = Vec::new();
                let mut best: Option<(usize, u32, f32)> = None;
                for (&r, &(id, c)) in rows.iter().zip(&picks) {
                    if c > cfg.threshold {
                        fixed.push((r, id, c));
                    }
                    if best.is_none_or(|b| c > b.2) {
                        best = Some((r, id, c));
                    }
                }
                if fixed.is_empty() {
                    fixed.extend(best);
                }
                stats.pick_s += t_pick.elapsed().as_secs_f64();
                for (r, id, c) in fixed {
                    block[r] = id;
                    pending[r] = false;
                    conf_of[r] = c;
                }
            }
            stats.passes += passes;
            stats.blocks += 1;
            // Emit in position order; stop only after the whole block is
            // resolved (an EOS may be fixed before the rows ahead of it).
            let new = &block[..remaining];
            let mut take = new.len();
            let mut stopped = false;
            for (i, &id) in new.iter().enumerate() {
                out.push(id);
                confidence.push(conf_of[i]);
                if is_stop(id) && !self.ignore_eos {
                    finish = "stop".into();
                    (take, stopped) = (i + 1, true);
                    break;
                }
                if let Some(cb) = on_token.as_mut() {
                    if !cb(&self.tokenizer.decode_token(id)) {
                        finish = "cancelled".into();
                        (take, stopped) = (i + 1, true);
                        break;
                    }
                }
            }
            pen.commit(&new[..take]);
            if stopped || out.len() >= max_tokens {
                break;
            }
            let t = std::time::Instant::now();
            if graph {
                if self.clover_graph_pass(clover, new, pos, None, None, &mut whole, &mut stats)?.is_none() {
                    return Err("block diffusion: device graph declined a commit mid-sequence".into());
                }
            } else {
                self.clover_commit_any(clover, new, pos, &mut metal, req);
            }
            stats.commit_s += t.elapsed().as_secs_f64();
            committed.extend_from_slice(new);
        }
        if finish == "cancelled" {
            self.clear_history();
        } else {
            self.record_consumed_prefix(&committed, reuse);
        }
        self.clover_stats = stats;
        Ok(GenerateResult {
            text: self.tokenizer.decode(&out),
            token_ids: out.clone(),
            prompt_tokens: input_ids.len(),
            tokens_generated: out.len(),
            finish_reason: finish,
            mtp_drafted: 0,
            mtp_accepted: 0,
            token_confidence: confidence,
            traces: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `coreset_torch` on a hand-checked block: two rows, top-2 of four
    /// experts. Row 0 puts .5/.3 on experts 0/1, row 1 .6/.2 on 2/0, so the
    /// totals are e0 .7, e1 .3, e2 .6, e3 0 — normalized .4375/.1875/.375/0.
    /// Descending: e0 (mass before 0), e2 (.4375), e1 (.8125). At top_p 0.8
    /// e0 and e2 are kept (.8125 ≥ .8 stops e1); at 0.9 e1 joins.
    #[test]
    fn coreset_keeps_the_prefix_that_holds_top_p() {
        let ln = |p: &[f32]| p.iter().map(|x| x.ln()).collect::<Vec<f32>>();
        let mut l = ln(&[0.5, 0.3, 0.1, 0.1]);
        l.extend(ln(&[0.2, 0.1, 0.6, 0.1]));
        let pick = |top_p: f32| {
            let mut totals = vec![0f64; 4];
            for row in l.chunks_exact(4) {
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let sum: f64 = row.iter().map(|&x| (x as f64 - mx).exp()).sum();
                let mut ids: Vec<usize> = (0..4).collect();
                ids.sort_by(|&x, &y| row[y].partial_cmp(&row[x]).unwrap().then(x.cmp(&y)));
                for &e in &ids[..2] {
                    totals[e] += (row[e] as f64 - mx).exp() / sum;
                }
            }
            coreset_from_totals(&totals, top_p)
        };
        assert_eq!(pick(0.8), vec![true, false, true, false]);
        assert_eq!(pick(0.9), vec![true, true, true, false]);
        assert_eq!(pick(0.0), vec![true, false, false, false]);
    }

    /// `clover_correct_drafts` by hand: probabilities .7 / .2 / .1 / 0, so
    /// H = 0.8018 and the threshold min(.9, .6·e^-H) = 0.2692.
    #[test]
    fn accept_rule_follows_the_vendor_threshold() {
        let row = [0.7f32.ln(), 0.2f32.ln(), 0.1f32.ln(), f32::NEG_INFINITY];
        let rule = AcceptRule { exact: false, ratio: 0.9, delta: 0.6 };
        let (ok, p) = rule.accepts(&row, 0);
        assert!(ok);
        assert!((p - 0.7).abs() < 1e-5);
        assert!(!rule.accepts(&row, 1).0);
        // A blocked token never passes, however low the threshold.
        let open = AcceptRule { exact: false, ratio: 0.0, delta: 0.0 };
        assert!(!open.accepts(&row, 3).0);
        assert!(open.accepts(&row, 2).0);
        let exact = AcceptRule { exact: true, ratio: 0.9, delta: 0.6 };
        assert!(exact.accepts(&row, 0).0);
        assert!(!exact.accepts(&row, 1).0);
    }

    #[test]
    fn penalties_follow_the_vendor_order() {
        let cfg = BlockDiffusionConfig {
            presence_penalty: 0.5,
            frequency_penalty: 0.25,
            repetition_penalty: 2.0,
            length_penalty: 4.0,
            length_penalty_start: 2,
            length_penalty_window: 4,
            think_end_bias: 8.0,
            think_end_start: 0,
            think_end_window: 2,
            think_end_token_id: Some(3),
            ..BlockDiffusionConfig::default()
        };
        let mut p = Penalties { counts: vec![0; 5], seen: Vec::new(), length: 0 };
        p.commit(&[1, 1, 2]);
        let mut l = vec![1.0f32, 2.0, -1.0, 0.0, 0.0];
        // row position 4: length ramp (4−2)/4·4 = 2 on stop id 4; think
        // ramp min(4/2,1)·8 = 8 on id 3.
        p.apply(&cfg, &mut l, 4, &[4], true);
        // id 1: 2 − 2·0.25 − 0.5 = 1.0, positive → /2 = 0.5
        // id 2: −1 − 0.25 − 0.5 = −1.75, negative → ×2 = −3.5
        assert_eq!(l, vec![1.0, 0.5, -3.5, 8.0, 2.0]);
    }
}
