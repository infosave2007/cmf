//! MiMo-V2 multi-token prediction: the release's draft layers
//! (`model.mtp.layers.{0,1,2}.*`, `num_nextn_predict_layers` = 3) and the
//! speculative greedy round that drafts with them and verifies the drafts
//! in ONE batched backbone forward.
//!
//! # The draft forward (one layer)
//!
//! HF `modeling_mimo_v2.py` ignores MTP. vLLM
//! (`vllm/model_executor/models/mimo_v2_mtp.py`, `MiMoV2MTPLayer.forward`)
//! and SGLang (`sglang/srt/models/mimo_v2_nextn.py`,
//! `MiMoV2ModelNextN.forward`) agree on one layer:
//!
//! ```text
//! u  = eh_proj · [enorm(embed(tok)) ; hnorm(hid)]      embedding FIRST
//! u += o_proj(swa_attn(input_layernorm(u)))            the SWA geometry of the
//!                                                      backbone's sliding layers:
//!                                                      8 KV heads, head 192 / V 128,
//!                                                      window 128, sinks, swa theta
//! u += mlp(pre_mlp_layernorm(u))                       dense SiLU MLP
//! logits = lm_head(final_layernorm(u))                 lm_head shared with the trunk
//! ```
//!
//! `hid` is the backbone's last-layer output BEFORE its final norm. Both
//! servers feed `model.norm(h)` instead (their target returns the normed
//! hidden), but the exact oracle (`tools/mimo_ref.py mtp`, 3 × 320 natural
//! tokens) measures the pre-norm hidden better at every depth: teacher-forced
//! chain acceptance d1/d2/d3 wikitext .589/.313/.152 vs .551/.263/.098,
//! HTML/JS .943/.864/.801 vs .937/.854/.772, Russian .845/.623/.424 vs
//! .813/.576/.383. `CMF_MIMO_MTP_HIDDEN=post` selects the servers' reading.
//!
//! # How the three layers chain (SGLang multi-layer MTP)
//!
//! SGLang's `multi_layer_eagle_worker_v2.py` runs MiMoV2MTP with
//! `draft_model_idx = step` (layer k drafts step k) and — MiMoV2MTP is NOT in
//! its `chain_mtp_hidden_states` list — every layer reads the TARGET hidden,
//! never the previous draft layer's output. Its input ids are rotated one
//! place per step. For a round that starts after the backbone processed
//! position `t` (hidden `h_t`) with `x_{t+1}` sampled:
//!
//! ```text
//! layer k, row j:  (embed(x_{j+k+1}), h_j)        at RoPE position j
//! draft d_{k+1} = argmax of layer k's row t        (the token for t+k+2)
//! ```
//!
//! where `x_{j+k+1}` beyond `x_{t+1}` is this round's own earlier draft.
//! Every layer keeps its OWN KV cache. Row `j` of layer `k` is final once
//! `x_{j+k+1}` is a committed token, i.e. for `j <= t-k`; later rows are
//! provisional and are recomputed next round. (vLLM instead runs layer 0
//! recursively on its own output; `tools/mimo_ref.py mtp` measures both.)
//!
//! # Exactness
//!
//! The drafts only choose WHICH tokens the backbone verifies. The verify is
//! `prefill_batch` over `[x_{t+1}, d_1..d_K]` — for MiMo bit-identical to the
//! single-token walk (`mimo_shaped_decode_matches_prefill_batch_bitwise`) —
//! each row's token is chosen by the plain sampler over the same history, the
//! longest matching prefix is kept, and the rejected rows' KV is truncated.
//! The emitted stream is the plain greedy stream.

use std::collections::VecDeque;
use std::sync::Arc;

use cortiq_core::{CmfError, CmfModel};

use super::{AttnKind, DenseFfn, FfnKind, LayerWeights, MtpModule, Pipeline};
use crate::attention::{self, QwenAttnCfg};
use crate::inference;
use crate::kv_cache::LayerKvCache;
use crate::qtensor::QTensor;
use crate::sampler;

/// Backbone hiddens kept for the draft layers. A round needs the rows from
/// the oldest provisional one (t - K + 1) on, so a few dozen would do; the
/// first round after a prompt starts every draft cache at the oldest kept
/// position, and a draft row sees `window` (128) rows, so keeping more than
/// the window makes the first rounds' drafts exact too.
pub(crate) const HIST_CAP: usize = 320;

/// Speculation counters of one generation.
#[derive(Clone, Debug, Default)]
pub struct MtpStats {
    /// Speculative rounds (one batched verify each).
    pub rounds: u64,
    /// Draft tokens proposed / accepted.
    pub drafted: u64,
    pub accepted: u64,
    /// `accept_hist[a]` = rounds that accepted exactly `a` drafts.
    pub accept_hist: Vec<u64>,
    /// Per depth d (0-based): rounds that drafted depth d, and rounds whose
    /// accepted prefix reached depth d.
    pub depth_drafted: Vec<u64>,
    pub depth_accepted: Vec<u64>,
    /// Wall time in the drafts and in the verify (forward + heads).
    pub draft_ns: u128,
    pub verify_ns: u128,
}

impl MtpStats {
    /// Tokens a round emits on average: accepted drafts + the verify's own.
    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        (self.accepted + self.rounds) as f64 / self.rounds as f64
    }

    pub fn line(&self) -> String {
        let depth: Vec<String> = self
            .depth_drafted
            .iter()
            .zip(&self.depth_accepted)
            .enumerate()
            .map(|(d, (&n, &a))| {
                format!(
                    "d{}={:.1}%({a}/{n})",
                    d + 1,
                    100.0 * a as f64 / n.max(1) as f64
                )
            })
            .collect();
        format!(
            "mimo-mtp: {} rounds, {:.3} tokens/round, accepted {}/{} drafts, per depth {}, \
             accept hist {:?}, draft {:.1} ms/round, verify {:.1} ms/round",
            self.rounds,
            self.tokens_per_round(),
            self.accepted,
            self.drafted,
            depth.join(" "),
            self.accept_hist,
            self.draft_ns as f64 / 1e6 / self.rounds.max(1) as f64,
            self.verify_ns as f64 / 1e6 / self.rounds.max(1) as f64,
        )
    }
}

/// The MiMo draft stack and its per-sequence state.
pub struct MimoMtp {
    /// One module per draft layer; each owns its KV cache (sinks attached).
    pub layers: Vec<MtpModule>,
    /// Drafts per round, ≤ `layers.len()` (`CMF_MIMO_MTP_K`, default all).
    pub depth: usize,
    /// Per layer: rows at positions `< committed[k]` are final.
    committed: Vec<usize>,
    /// Per layer: absolute position of the cache's first row.
    base: Vec<usize>,
    /// Backbone hiddens the draft layers read (pre-final-norm unless
    /// `post_norm_hidden`), positions `hist_start..`.
    hist: VecDeque<Vec<f32>>,
    hist_start: usize,
    hist_cap: usize,
    pub stats: MtpStats,
    /// The draft layers read the backbone hidden AFTER the final norm
    /// (vLLM / SGLang) instead of before it (the default, measured better:
    /// see the module doc). `CMF_MIMO_MTP_HIDDEN=post`.
    pub post_norm_hidden: bool,
    /// Test hook: drafts are read from this sequence (by position) instead
    /// of the layers — to drive exact partial acceptance through the round.
    #[cfg(test)]
    pub(crate) draft_override: Option<Vec<u32>>,
}

/// One round's outcome: the accepted drafts (to commit), the backbone
/// hidden (pre-norm) of the last accepted row and that row's logits, which
/// choose the round's own token at the loop top.
pub(crate) struct MimoRound {
    pub accepted: Vec<u32>,
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub drafted: usize,
}

impl MimoMtp {
    /// Build from already-loaded modules (tests; the loader uses `load`).
    pub fn from_layers(layers: Vec<MtpModule>) -> Self {
        let n = layers.len();
        let depth = std::env::var("CMF_MIMO_MTP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&k| k >= 1)
            .unwrap_or(n)
            .min(n);
        Self {
            layers,
            depth,
            committed: vec![0; n],
            base: vec![0; n],
            hist: VecDeque::new(),
            hist_start: 0,
            hist_cap: HIST_CAP,
            stats: MtpStats::default(),
            post_norm_hidden: std::env::var("CMF_MIMO_MTP_HIDDEN").as_deref() == Ok("post"),
            #[cfg(test)]
            draft_override: None,
        }
    }

    /// Load `n` draft layers `model.mtp.layers.{0..n}` from `model` (the
    /// sidecar, or a main file that carries them) with the backbone's SWA
    /// geometry: `nh` query heads, `nkv` KV heads, `hd` Q/K head width,
    /// `vd` V head width, `inter` MLP width. Shapes are checked here so a
    /// mismatched sidecar fails at load, not as garbage drafts.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        model: &Arc<CmfModel>,
        n: usize,
        h: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        vd: usize,
        inter: usize,
    ) -> Result<Self, CmfError> {
        let err = |e: String| CmfError::Parse(format!("MiMo MTP: {e}"));
        let ov = crate::loader::Overlay::None;
        let mut layers = Vec::with_capacity(n);
        for k in 0..n {
            let p = format!("model.mtp.layers.{k}.");
            let vec = |name: &str, len: usize| -> Result<Vec<f32>, CmfError> {
                let v = crate::loader::load_f32(model, &format!("{p}{name}"), &ov).map_err(err)?;
                if v.len() != len {
                    return Err(err(format!(
                        "{p}{name}: {} values, expected {len}",
                        v.len()
                    )));
                }
                Ok(v)
            };
            let mat = |name: &str, rows: usize, cols: usize| -> Result<QTensor, CmfError> {
                let t = QTensor::from_model(model, &format!("{p}{name}")).map_err(err)?;
                if t.rows() != rows || t.cols() != cols {
                    return Err(err(format!(
                        "{p}{name}: [{}, {}], expected [{rows}, {cols}]",
                        t.rows(),
                        t.cols()
                    )));
                }
                Ok(t)
            };
            let sinks = vec("self_attn.sinks", nh)?;
            if let Some(bad) = sinks.iter().find(|s| !s.is_finite()) {
                return Err(err(format!("{p}self_attn.sinks: non-finite {bad}")));
            }
            let mut kv = LayerKvCache::new(nkv, hd);
            kv.sinks = Some(sinks);
            layers.push(MtpModule {
                enorm: vec("enorm.weight", h)?,
                hnorm: vec("hnorm.weight", h)?,
                eh_proj: mat("eh_proj.weight", h, 2 * h)?,
                layer: LayerWeights {
                    input_norm: vec("input_layernorm.weight", h)?,
                    post_norm: vec("post_attention_layernorm.weight", h)?,
                    attn_out_norm: None,
                    ffn_out_norm: None,
                    layer_scale: None,
                    attn: AttnKind::Full {
                        wq: mat("self_attn.q_proj.weight", nh * hd, h)?,
                        wk: mat("self_attn.k_proj.weight", nkv * hd, h)?,
                        wv: mat("self_attn.v_proj.weight", nkv * vd, h)?,
                        wo: mat("self_attn.o_proj.weight", h, nh * vd)?,
                        q_norm: None,
                        k_norm: None,
                        output_gate: false,
                        softplus_gate: None,
                        bias: None,
                    },
                    ffn: FfnKind::Dense(DenseFfn {
                        gate_proj: mat("mlp.gate_proj.weight", inter, h)?,
                        up_proj: mat("mlp.up_proj.weight", inter, h)?,
                        down_proj: mat("mlp.down_proj.weight", h, inter)?,
                        act: super::Act::Silu,
                        down_t: None,
                        segs: Vec::new(),
                    }),
                },
                final_norm: vec("final_layernorm.weight", h)?,
                kv,
            });
        }
        Ok(Self::from_layers(layers))
    }

    /// Fresh sequence: empty caches and history, zeroed counters.
    pub fn reset(&mut self) {
        for m in &mut self.layers {
            m.kv.clear();
        }
        self.committed.iter_mut().for_each(|c| *c = 0);
        self.base.iter_mut().for_each(|c| *c = 0);
        self.hist.clear();
        self.hist_start = 0;
        self.hist_cap = HIST_CAP;
        self.stats = MtpStats::default();
    }

    /// Drop the draft caches but keep the history (the next draft rebuilds
    /// every layer from the oldest kept position).
    fn reset_caches(&mut self) {
        for m in &mut self.layers {
            m.kv.clear();
        }
        self.committed.iter_mut().for_each(|c| *c = 0);
        self.base.iter_mut().for_each(|c| *c = 0);
    }

    fn hist_end(&self) -> usize {
        self.hist_start + self.hist.len()
    }
}

/// The MiMo draft stack for a loaded `mimo_v2` backbone, if the file has
/// one: `model.mtp.layers.*` inside the main file (a conversion that kept
/// them), else the sidecar `<stem>.mtp.cmf` beside it. `CMF_MIMO_MTP=0`
/// skips it. A sidecar that does not match the backbone is an ERROR, not a
/// silent plain decode.
pub fn load_for(
    main: &Arc<CmfModel>,
    arch: &cortiq_core::ModelArch,
) -> Result<Option<MimoMtp>, CmfError> {
    if std::env::var("CMF_MIMO_MTP").as_deref() == Ok("0") {
        return Ok(None);
    }
    let count = |m: &CmfModel| {
        (0..)
            .take_while(|k| {
                m.tensor(&format!("model.mtp.layers.{k}.eh_proj.weight"))
                    .is_some()
            })
            .count()
    };
    let (src, n, from) = if main.tensor("model.mtp.layers.0.eh_proj.weight").is_some() {
        (main.clone(), count(main), main.path.display().to_string())
    } else {
        let path = cortiq_core::mtp_sidecar_path(&main.path);
        if path == main.path || !path.exists() {
            return Ok(None);
        }
        let side = Arc::new(CmfModel::open(&path)?);
        let a = side.arch();
        let bad = |what: &str| {
            CmfError::Parse(format!(
                "MiMo MTP sidecar {}: {what} differs from the backbone                  (CMF_MIMO_MTP=0 runs without it)",
                path.display()
            ))
        };
        if a.arch_name != arch.arch_name {
            return Err(bad("arch"));
        }
        if a.hidden_size != arch.hidden_size
            || a.num_attention_heads != arch.num_attention_heads
            || a.head_dim != arch.head_dim
            || a.v_head_dim != arch.v_head_dim
            || a.vocab_size != arch.vocab_size
        {
            return Err(bad("geometry"));
        }
        let n = a.mtp.as_ref().map(|m| m.num_layers).unwrap_or(0);
        if n == 0 || count(&side) < n {
            return Err(bad("MTP layer count"));
        }
        (side, n, path.display().to_string())
    };
    if n == 0 {
        return Ok(None);
    }
    let li = arch
        .layer_types
        .iter()
        .position(|t| matches!(t, cortiq_core::LayerType::SlidingAttention))
        .unwrap_or(0);
    let nkv = arch
        .kv_heads_per_layer
        .as_ref()
        .and_then(|v| v.get(li).copied())
        .unwrap_or(arch.num_kv_heads);
    let hd = arch.head_dim;
    let st = MimoMtp::load(
        &src,
        n,
        arch.hidden_size,
        arch.num_attention_heads,
        nkv,
        hd,
        arch.v_head_dim.unwrap_or(hd),
        arch.intermediate_size,
    )?;
    tracing::info!(
        "MiMo MTP: {n} draft layers from {from} (draft depth {})",
        st.depth
    );
    Ok(Some(st))
}

impl Pipeline {
    /// Record backbone hiddens (pre-final-norm, `hidden_size` each) of
    /// consecutive positions from `first_pos` for the draft stack.
    pub(crate) fn mimo_note_rows(&mut self, rows: &[f32], first_pos: usize) {
        let Some(mut st) = self.mimo_mtp.take() else {
            return;
        };
        for (i, r) in rows.chunks_exact(self.hidden_size).enumerate() {
            self.mimo_mtp_note(&mut st, first_pos + i, r);
        }
        self.mimo_mtp = Some(st);
    }

    /// The backbone layer whose attention geometry the draft blocks share:
    /// the first sliding-window layer.
    fn mimo_mtp_geom_layer(&self) -> usize {
        (0..self.num_layers)
            .find(|&li| self.layer_is_local(li))
            .unwrap_or(0)
    }

    /// Record the backbone hidden (pre-final-norm) of position `pos`. Rows
    /// arrive in position order; an already-recorded position is skipped,
    /// a gap restarts the history there.
    pub(crate) fn mimo_mtp_note(&self, st: &mut MimoMtp, pos: usize, hidden: &[f32]) {
        let end = st.hist_end();
        if !st.hist.is_empty() && pos < end && pos >= st.hist_start {
            return;
        }
        if st.hist.is_empty() || pos != end {
            st.hist.clear();
            st.hist_start = pos;
        }
        let g = if st.post_norm_hidden {
            let mut g = vec![0.0f32; self.hidden_size];
            inference::rms_norm_into(
                hidden,
                &self.weights.final_norm,
                self.rms_eps,
                self.norm_style,
                &mut g,
            );
            g
        } else {
            hidden.to_vec()
        };
        st.hist.push_back(g);
        while st.hist.len() > st.hist_cap {
            st.hist.pop_front();
            st.hist_start += 1;
        }
    }

    /// Run `x` (n rows at positions first_pos.., after eh_proj) through one
    /// draft block in place, appending its K/V rows to the block's cache.
    pub(crate) fn mimo_mtp_block(
        &self,
        m: &mut MtpModule,
        x: &mut [f32],
        n: usize,
        first_pos: usize,
    ) {
        let hs = self.hidden_size;
        let li = self.mimo_mtp_geom_layer();
        let MtpModule { layer, kv, .. } = m;
        let mut normed = vec![0.0f32; n * hs];
        for i in 0..n {
            inference::rms_norm_into(
                &x[i * hs..(i + 1) * hs],
                &layer.input_norm,
                self.rms_eps,
                self.norm_style,
                &mut normed[i * hs..(i + 1) * hs],
            );
        }
        let AttnKind::Full { wq, wk, wv, wo, .. } = &layer.attn else {
            unreachable!("MiMo MTP blocks are softmax attention")
        };
        let inv_freq_l = self.layer_inv_freq(li);
        let (nkv_l, hd_l, rd_l) = self.layer_geom(li);
        let cfg = QwenAttnCfg {
            num_heads: self.layer_num_heads(li),
            num_kv_heads: nkv_l,
            head_dim: hd_l,
            hidden_size: hs,
            position: first_pos,
            inv_freq: &inv_freq_l,
            rotary_dim: rd_l,
            scale: self.attn_scale,
            softcap: self.attn_softcap,
            window: self.layer_window(li),
            v_norm: self.attn_v_norm,
            qk_norm_after_rope: self.qk_norm_after_rope,
            q_norm: None,
            k_norm: None,
            output_gate: false,
            softplus_gate: None,
            rope_scale: self.layer_rope_scale(li),
            bias: None,
            rms_eps: self.rms_eps,
            norm_style: self.norm_style,
            pool: self.pool.as_deref(),
            v_head_dim: self.layer_v_dim(li),
        };
        let attn = attention::qwen_attention_batch(&normed, n, wq, wk, wv, wo, kv, &cfg);
        for (a, b) in x.iter_mut().zip(&attn) {
            *a += b;
        }
        for i in 0..n {
            inference::rms_norm_into(
                &x[i * hs..(i + 1) * hs],
                &layer.post_norm,
                self.rms_eps,
                self.norm_style,
                &mut normed[i * hs..(i + 1) * hs],
            );
        }
        let FfnKind::Dense(d) = &layer.ffn else {
            unreachable!("MiMo MTP blocks have a dense MLP")
        };
        let f = super::dense_ffn_batch(d, &normed, n, self.pool.as_deref(), None);
        for (a, b) in x.iter_mut().zip(&f) {
            *a += b;
        }
    }

    /// Draft up to `k` tokens for the round that starts after the backbone
    /// processed position `t` (`ids[..=t+1]` known: the prompt, the
    /// committed tokens and the pending `x_{t+1}`). Returns the drafts for
    /// positions t+2, t+3, …; empty when the history lacks position `t`.
    /// `ids` longer than t+2 (the teacher-forced probe) supplies the later
    /// tokens instead of the drafts.
    pub(crate) fn mimo_mtp_draft(
        &self,
        st: &mut MimoMtp,
        t: usize,
        ids: &[u32],
        k: usize,
    ) -> Vec<u32> {
        let hs = self.hidden_size;
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        #[cfg(test)]
        if let Some(ov) = &st.draft_override {
            for i in 0..k.min(st.layers.len()) {
                drafts.push(ov.get(t + 2 + i).copied().unwrap_or(0));
            }
            return drafts;
        }
        if st.hist.is_empty() || t < st.hist_start || t >= st.hist_end() || ids.len() < t + 2 {
            return drafts;
        }
        let lo = st.hist_start;
        let MimoMtp {
            layers,
            committed,
            base,
            hist,
            ..
        } = st;
        for layer in 0..k.min(layers.len()) {
            let m = &mut layers[layer];
            if committed[layer] < lo || committed[layer] > t + 1 {
                // A history gap (first round after a long prompt) or a
                // stale state: the cache restarts at the oldest kept row.
                m.kv.clear();
                base[layer] = lo;
                committed[layer] = lo;
            }
            // Drop last round's provisional rows.
            let keep = committed[layer] - base[layer];
            if m.kv.seq_len > keep {
                let extra = m.kv.seq_len - keep;
                m.kv.truncate_last(extra);
            }
            let start = committed[layer];
            let n = t + 1 - start;
            let mut cats = vec![0.0f32; n * 2 * hs];
            for (r, j) in (start..=t).enumerate() {
                let idx = j + layer + 1;
                let tok = if idx < ids.len() {
                    ids[idx]
                } else {
                    drafts[idx - ids.len()]
                };
                let e = self.embed_single(tok);
                let (ce, ch) = cats[r * 2 * hs..(r + 1) * 2 * hs].split_at_mut(hs);
                inference::rms_norm_into(&e, &m.enorm, self.rms_eps, self.norm_style, ce);
                inference::rms_norm_into(
                    &hist[j - lo],
                    &m.hnorm,
                    self.rms_eps,
                    self.norm_style,
                    ch,
                );
            }
            let mut x = vec![0.0f32; n * hs];
            if n == 1 {
                m.eh_proj.matvec(&cats, &mut x, self.pool.as_deref());
            } else {
                m.eh_proj.matmat(&cats, n, &mut x, self.pool.as_deref());
            }
            self.mimo_mtp_block(m, &mut x, n, start);
            let mut y = vec![0.0f32; hs];
            inference::rms_norm_into(
                &x[(n - 1) * hs..],
                &m.final_norm,
                self.rms_eps,
                self.norm_style,
                &mut y,
            );
            let mut lg = self.lm_head_forward(&y);
            drafts.push(sampler::argmax(&lg));
            attention::recycle_buf(&mut lg);
            // Rows whose token is known are final.
            let real_end = (t + 1).min(ids.len().saturating_sub(layer + 1));
            committed[layer] = committed[layer].max(real_end);
        }
        drafts
    }

    /// One speculative greedy round. On entry the backbone has processed
    /// positions `..next_pos` and `all_ids[next_pos]` is the pending token
    /// (committed, not yet forwarded). Drafts `k`, verifies `[pending,
    /// drafts]` in one batched forward, keeps the longest prefix the plain
    /// sampler agrees with and truncates the rest of the backbone KV. None
    /// = no drafts (nothing was forwarded).
    pub(crate) fn mimo_spec_round(
        &mut self,
        st: &mut MimoMtp,
        next_pos: usize,
        all_ids: &[u32],
        k: usize,
    ) -> Result<Option<MimoRound>, String> {
        if next_pos == 0 || all_ids.len() != next_pos + 1 || k == 0 {
            return Ok(None);
        }
        let t = next_pos - 1;
        let t0 = std::time::Instant::now();
        let drafts = self.mimo_mtp_draft(st, t, all_ids, k);
        st.stats.draft_ns += t0.elapsed().as_nanos();
        if drafts.is_empty() {
            return Ok(None);
        }
        let k = drafts.len();
        let t1 = std::time::Instant::now();
        let mut ids = Vec::with_capacity(k + 1);
        ids.push(all_ids[next_pos]);
        ids.extend_from_slice(&drafts);
        // Decode-exact batched verify: row-exact kernels and the exact
        // multi-row MoE (`CMF_MIMO_MTP_VERIFY=fast` = the prompt path's
        // blocked kernels / expert-order sums: close to decode, not equal).
        let exact = std::env::var("CMF_MIMO_MTP_VERIFY").as_deref() != Ok("fast");
        let hb = if exact {
            self.verify_exact_moe = true;
            let hb = crate::qtensor::row_exact_scope(|| self.prefill_rows(&ids, next_pos, None));
            self.verify_exact_moe = false;
            hb
        } else {
            self.prefill_rows(&ids, next_pos, None)
        }?;
        let hs = self.hidden_size;
        let mut history = all_ids.to_vec();
        let mut a = 0usize;
        let mut normed = vec![0.0f32; hs];
        let mut logits = Vec::new();
        for i in 0..=k {
            inference::rms_norm_into(
                &hb[i * hs..(i + 1) * hs],
                &self.weights.final_norm,
                self.rms_eps,
                self.norm_style,
                &mut normed,
            );
            let mut lg = self.lm_head_forward(&normed);
            let tok = sampler::sample_with_scratch_pool(
                &lg,
                &self.sampler_config,
                &history,
                &mut self.rng,
                &mut self.sampler_scratch,
                self.pool.as_deref(),
            );
            if i < k && tok == drafts[i] {
                a += 1;
                history.push(tok);
                attention::recycle_buf(&mut lg);
            } else {
                logits = lg;
                break;
            }
        }
        // Host caches can lag graph-owned layers: rewind to an absolute
        // position, not by K-a from a possibly stale host cursor. SWA rings
        // retain two windows; the backend refuses if the needed old rows
        // have already been overwritten. Such a failure is terminal.
        self.mimo_verify_rewind(next_pos + a + 1)?;
        for i in 0..=a {
            self.mimo_mtp_note(st, next_pos + i, &hb[i * hs..(i + 1) * hs]);
        }
        st.stats.verify_ns += t1.elapsed().as_nanos();
        let s = &mut st.stats;
        s.rounds += 1;
        s.drafted += k as u64;
        s.accepted += a as u64;
        if s.accept_hist.len() <= k {
            s.accept_hist.resize(k + 1, 0);
        }
        s.accept_hist[a] += 1;
        if s.depth_drafted.len() < k {
            s.depth_drafted.resize(k, 0);
            s.depth_accepted.resize(k, 0);
        }
        for d in 0..k {
            s.depth_drafted[d] += 1;
            if d < a {
                s.depth_accepted[d] += 1;
            }
        }
        Ok(Some(MimoRound {
            accepted: drafts[..a].to_vec(),
            hidden: hb[a * hs..(a + 1) * hs].to_vec(),
            logits,
            drafted: k,
        }))
    }

    /// Reconcile both KV owners after a speculative suffix is rejected.
    pub(super) fn mimo_verify_rewind(&mut self, keep: usize) -> Result<(), String> {
        for li in 0..self.num_layers {
            if crate::gpu::graph_kv_stored(self.graph_kv_id, li).is_some_and(|n| n > keep)
                && !crate::gpu::graph_kv_set_stored(self.graph_kv_id, li, keep)
            {
                self.clear_sequence_state();
                return Err(format!("MiMo MTP: GPU KV rollback refused at layer {li}"));
            }
            let layer = &mut self.kv_cache.layers[li];
            if layer.seq_len > keep {
                layer.truncate_last(layer.seq_len - keep);
            }
        }
        // No out-of-band logits from a discarded speculative row survive.
        self.graph_logits = None;
        Ok(())
    }

    /// `CMF_MIMO_MTP_PROBE=<file>`: teacher-forced drafts of every prompt
    /// position through the SAME incremental draft path as decode (later
    /// tokens come from the prompt instead of the drafts), one JSON line
    /// per round start: `{"t", "drafts": [d1, d2, d3]}` — the table
    /// `tools/mimo_ref.py mtp` writes for variant A.
    pub(crate) fn mimo_mtp_probe(&self, st: &mut MimoMtp, ids: &[u32], path: &str) {
        let k = st.depth;
        let mut out = String::new();
        let first = st.hist_start;
        for t in first..ids.len().saturating_sub(k + 1) {
            let d = self.mimo_mtp_draft(st, t, ids, k);
            out.push_str(&format!("{{\"t\":{t},\"drafts\":{d:?}}}\n"));
        }
        if let Err(e) = std::fs::write(path, out) {
            tracing::error!("CMF_MIMO_MTP_PROBE: cannot write {path}: {e}");
        }
        st.reset_caches();
    }

    /// Widen the history so a probe sees every prompt position.
    pub(crate) fn mimo_mtp_hist_cap(st: &mut MimoMtp, cap: usize) {
        st.hist_cap = st.hist_cap.max(cap);
    }
}
