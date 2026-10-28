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
//! On wgpu the prompt, every pass and every commit are one batched-graph
//! submission each (`gpu::BlockPreview`, kernels in `gpu_wgpu::clover`),
//! the greedy pick included; elsewhere the request runs on the host CPU.
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
}

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
    let mut totals = vec![0f64; ne];
    let mut ids: Vec<usize> = Vec::with_capacity(ne);
    for row in logits.chunks_exact(ne) {
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = row.iter().map(|&l| (l as f64 - mx).exp()).sum();
        ids.clear();
        ids.extend(0..ne);
        // Descending, lower index first on ties (a stable argsort).
        ids.sort_by(|&x, &y| row[y].partial_cmp(&row[x]).unwrap().then(x.cmp(&y)));
        for &e in &ids[..m.top_k.min(ne)] {
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

/// What a denoising pass hands the token choice.
enum PassOut {
    /// `[block × vocab]` logits, chosen on the host.
    Logits(Vec<f32>),
    /// Each row's greedy token and its probability, chosen on the device.
    Picks(Vec<(u32, f32)>),
}

/// Response-only penalty state of one request (`CloverPenaltyState`).
struct Penalties {
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
        tokens as f64 / (self.preview_s + self.pick_s + self.commit_s).max(1e-9)
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
        // Without the batched graph the whole request keeps to the CPU: the
        // per-op device path of a 32-row block lost to it everywhere it was
        // measured (Apple M4, 24 GB: passes 1.43 against 1.25 s, the prompt
        // 10.7 against 2.0 s — Metal kept the 19 GB of weights resident
        // where the CPU touches only the pages a pass reads).
        let _cpu = (!graph).then(crate::gpu::enter_cpu_scope);
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
            self.clover_commit(&input_ids[reuse..], reuse);
        }
        stats.prompt_s = t0.elapsed().as_secs_f64();
        let mut committed: Vec<u32> = input_ids.to_vec();

        // The length ramp's stop ids; the file's EOS ids end the answer too.
        let mut stop: Vec<u32> = cfg.stop_token_ids.clone();
        let tk = &self.tokenizer;
        stop.extend(tk.eos_token_id.into_iter().chain(tk.im_end_id).chain(tk.extra_eos.iter().copied()));
        stop.retain(|&t| (t as usize) < vocab);
        stop.sort_unstable();
        stop.dedup();
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
                    None => PassOut::Logits(self.clover_preview(clover, &block, pos, &mut stats)?),
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
                self.clover_commit(new, pos);
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
