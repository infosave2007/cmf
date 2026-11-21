//! Maxwell-V1-Flash (`maxwell_moe`) text runtime.
//!
//! Transcribed from the reference `modeling_maxwell_moe.py`, not inferred
//! from tensor names. The stack:
//!
//! * four residual streams with manifold-constrained hyper-connections: each
//!   sub-layer folds the streams with `pre = σ(·) + eps`, returns its output
//!   to every stream with `post = 2·σ(·)` and mixes the streams with a
//!   doubly stochastic matrix (a row softmax and 20 Sinkhorn rounds) — the
//!   DeepSeek-V4 form (`dsv4::hc_split_sinkhorn`), its router's input scaled
//!   by `1 / (rms + eps)`; a learned sigmoid fold closes the stack;
//! * GQA (32 heads over 4 KV groups, 128 wide) with a per-head RMSNorm on q
//!   and k before the rotation (base 2e7), and block-sparse support: a query
//!   attends to the first 8 blocks of 128 positions, to its own block and
//!   the 7 before it, and to at most 16 blocks in between that the layer's
//!   indexer picks for each KV group by the best token score of the block.
//!   Below 4096 positions that is every position: plain causal attention;
//! * a dense SwiGLU FFN in the first three layers, then 42 MoE blocks: a
//!   sigmoid router, 12 of 256 experts chosen with a selection bias, their
//!   weights the unbiased scores renormalized, plus an ungated shared expert
//!   twice as wide.
//!
//! The converter is `cortiq-cli/src/maxwellpack.rs`; the MTP layer of the
//! checkpoint is not converted.

use crate::dsv4::{hc_expand, hc_fold, hc_split_sinkhorn};
use crate::loader::{Overlay, load_f32, load_matrix};
use crate::pool::Pool;
use crate::qtensor::QTensor;
use cortiq_core::{CmfError, CmfModel, ModelArch, NormStyle};
use std::sync::Arc;

pub const ARCH_NAME: &str = "maxwell_moe";

fn err(s: impl Into<String>) -> CmfError {
    CmfError::Parse(format!("maxwell_moe: {}", s.into()))
}

fn f(model: &CmfModel, name: &str) -> Result<Vec<f32>, CmfError> {
    load_f32(model, name, &Overlay::None).map_err(err)
}

fn t(model: &Arc<CmfModel>, name: &str) -> Result<QTensor, CmfError> {
    load_matrix(model, name, false, &Overlay::None)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn rms_w(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    crate::inference::rms_norm(x, w, eps, NormStyle::Qwen)
}

/// One hyper-connection: `[2·hc + hc², hc·dim]` mapping, its bias and the
/// three gating factors (pre, post, mix).
pub struct Hc {
    pub function: Vec<f32>,
    pub base: Vec<f32>,
    pub scale: [f32; 3],
}

impl Hc {
    /// The mapping's rows over the flattened streams, scaled by
    /// `1 / (‖x‖ / √n + eps)` — the reference adds its epsilon to the RMS,
    /// not under the root.
    fn mixes(&self, flat: &[f32], eps: f32, pool: Option<&Pool>) -> Vec<f32> {
        let n = flat.len();
        let rows = self.base.len();
        let ms = flat.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
        let r = (1.0 / (ms.sqrt() + eps as f64)) as f32;
        let mut out = vec![0f32; rows];
        let w = &self.function;
        let work = |i: usize| -> f32 { w[i * n..(i + 1) * n].iter().zip(flat).map(|(a, b)| a * b).sum::<f32>() * r };
        match pool {
            Some(p) => {
                let addr = crate::pool::SendMut::new(out.as_mut_ptr());
                p.run_rows(rows, &|a, b| {
                    for i in a..b {
                        unsafe { *addr.at(i) = work(i) };
                    }
                });
            }
            None => {
                for (i, o) in out.iter_mut().enumerate() {
                    *o = work(i);
                }
            }
        }
        out
    }

    /// (pre, post, mix) of the sub-layer.
    fn split(&self, flat: &[f32], cfg: &Cfg, pool: Option<&Pool>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let hc = cfg.hc;
        let m = self.mixes(flat, cfg.hc_eps, pool);
        let (mut pre, mut post, mut comb) = (vec![0f32; hc], vec![0f32; hc], vec![0f32; hc * hc]);
        hc_split_sinkhorn(&m, &self.scale, &self.base, hc, cfg.sinkhorn_iters, cfg.hc_eps, &mut pre, &mut post, &mut comb);
        (pre, post, comb)
    }
}

/// The block router: `[groups·id, hidden]` index queries, one `[id, hidden]`
/// index key shared by the groups, their per-head norms (f32).
pub struct Indexer {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

pub struct Attn {
    pub q: QTensor,
    pub k: QTensor,
    pub v: QTensor,
    pub o: QTensor,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub indexer: Indexer,
}

pub struct Moe {
    /// `[E, hidden]` f32 router and the selection bias.
    pub router: Vec<f32>,
    pub bias: Vec<f32>,
    /// (gate, up, down) of every routed expert and of the shared one.
    pub experts: Vec<[QTensor; 3]>,
    pub shared: [QTensor; 3],
}

pub enum Ffn {
    Dense { gate: QTensor, up: QTensor, down: QTensor },
    Moe(Moe),
}

pub struct Layer {
    pub attn_hc: Hc,
    pub ffn_hc: Hc,
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub attn: Attn,
    pub ffn: Ffn,
}

pub struct Globals {
    pub model: Arc<CmfModel>,
    pub embed: QTensor,
    pub lm_head: QTensor,
    pub norm: Vec<f32>,
    pub head_fn: Vec<f32>,
    pub head_base: Vec<f32>,
    pub head_scale: f32,
}

pub struct Cfg {
    pub dim: usize,
    pub hc: usize,
    pub eps: f64,
    pub hc_eps: f32,
    pub sinkhorn_iters: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub scale: f32,
    pub invf: Vec<f32>,
    /// Positions a block holds, the blocks every query sees at the start of
    /// the context and before itself (its own included), and the blocks the
    /// indexer adds a KV group.
    pub block: usize,
    pub prefix_blocks: usize,
    pub local_blocks: usize,
    pub remote_blocks: usize,
    pub top_k: usize,
}

impl Cfg {
    /// Positions below which every query attends to its whole past.
    pub fn dense_range(&self) -> usize {
        (self.prefix_blocks + self.local_blocks + self.remote_blocks) * self.block
    }
}

/// One layer's cache: the rotated keys and the values of every position
/// (`nkv·hd` each) and the indexer's rotated key (`hd`).
#[derive(Default)]
struct LayerState {
    k: Vec<f32>,
    v: Vec<f32>,
    ik: Vec<f32>,
}

pub struct State {
    pub pos: usize,
    pub tokens: Vec<u32>,
    layers: Vec<LayerState>,
    /// The sequence's device key (process-unique: a pipeline reuses one model
    /// for many prompts).
    sequence_id: u64,
    /// The sequence runs on the card's graphs: None not tried yet, Some(false)
    /// the host runs it, Some(true) the card holds its states.
    graph: Option<bool>,
}

static NEXT_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl State {
    pub fn reset(&mut self, n_layers: usize) {
        if self.graph == Some(true) {
            crate::gpu::graph_kv_reset(self.kv_id());
        }
        self.graph = None;
        self.sequence_id = NEXT_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.pos = 0;
        self.tokens.clear();
        self.layers = (0..n_layers).map(|_| LayerState::default()).collect();
    }

    fn kv_id(&self) -> u64 {
        (1u64 << 60) | self.sequence_id
    }
}

impl Drop for State {
    fn drop(&mut self) {
        if self.graph == Some(true) {
            crate::gpu::graph_kv_reset(self.kv_id());
        }
    }
}

pub fn load(model: &Arc<CmfModel>, arch: &ModelArch) -> Result<(Globals, Vec<Layer>, Cfg, State), CmfError> {
    let prov = model.header.provenance.as_ref().and_then(|p| p.get("maxwell_moe")).ok_or_else(|| err("header has no maxwell_moe record"))?;
    let c = prov.get("config").ok_or_else(|| err("no config"))?;
    let u = |k: &str| -> Result<usize, CmfError> { c.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| err(format!("config.{k}"))) };
    let (nh, nkv, hd) = (arch.num_attention_heads, arch.num_kv_heads, arch.head_dim);
    if nkv == 0 || nh % nkv != 0 || hd % 2 != 0 {
        return Err(err(format!("attention geometry {nh}/{nkv} x {hd}")));
    }
    let block = u("block_sparse_gqa_block_size")?;
    let moe = arch.moe.as_ref().ok_or_else(|| err("missing MoE geometry"))?;
    let cfg = Cfg {
        dim: arch.hidden_size,
        hc: u("num_residual_streams")?,
        eps: arch.rms_norm_eps,
        hc_eps: 1e-6,
        sinkhorn_iters: u("mhc_sinkhorn_iterations")?,
        nh,
        nkv,
        hd,
        scale: (hd as f32).sqrt().recip(),
        invf: (0..hd / 2).map(|i| (1.0 / arch.rope_theta.powf(2.0 * i as f64 / hd as f64)) as f32).collect(),
        block,
        prefix_blocks: u("block_sparse_gqa_prefix_window_size")? / block.max(1),
        local_blocks: u("block_sparse_gqa_local_window_size")? / block.max(1),
        remote_blocks: u("block_sparse_gqa_num_remote_blocks")?,
        top_k: moe.top_k,
    };
    if block == 0 || cfg.prefix_blocks == 0 || cfg.local_blocks == 0 || !(1..=4).contains(&cfg.hc) {
        return Err(err("block-sparse geometry"));
    }
    let hs = f(model, "model.hc_head.scale")?;
    let globals = Globals {
        model: model.clone(),
        embed: t(model, "model.embed_tokens.weight")?,
        lm_head: t(model, "lm_head.weight")?,
        norm: f(model, "model.norm.weight")?,
        head_fn: f(model, "model.hc_head.fn")?,
        head_base: f(model, "model.hc_head.base")?,
        head_scale: *hs.first().ok_or_else(|| err("hc_head.scale"))?,
    };
    let mix = 2 * cfg.hc + cfg.hc * cfg.hc;
    let mut layers = Vec::with_capacity(arch.num_layers);
    for li in 0..arch.num_layers {
        let p = format!("model.layers.{li}.");
        let hc_of = |kind: &str| -> Result<Hc, CmfError> {
            let s = f(model, &format!("{p}{kind}.scale"))?;
            let h = Hc { function: f(model, &format!("{p}{kind}.fn"))?, base: f(model, &format!("{p}{kind}.base"))?, scale: [s.first().copied().unwrap_or(0.0), s.get(1).copied().unwrap_or(0.0), s.get(2).copied().unwrap_or(0.0)] };
            if s.len() != 3 || h.base.len() != mix || h.function.len() != mix * cfg.hc * cfg.dim {
                return Err(err(format!("{p}{kind}: shape")));
            }
            Ok(h)
        };
        let a = format!("{p}self_attn.");
        let attn = Attn {
            q: t(model, &format!("{a}q_proj.weight"))?,
            k: t(model, &format!("{a}k_proj.weight"))?,
            v: t(model, &format!("{a}v_proj.weight"))?,
            o: t(model, &format!("{a}o_proj.weight"))?,
            q_norm: f(model, &format!("{a}q_norm.weight"))?,
            k_norm: f(model, &format!("{a}k_norm.weight"))?,
            indexer: Indexer {
                q: f(model, &format!("{a}indexer.q_proj.weight"))?,
                k: f(model, &format!("{a}indexer.k_proj.weight"))?,
                q_norm: f(model, &format!("{a}indexer.q_norm.weight"))?,
                k_norm: f(model, &format!("{a}indexer.k_norm.weight"))?,
            },
        };
        if attn.q.rows() != nh * hd || attn.k.rows() != nkv * hd || attn.v.rows() != nkv * hd || attn.indexer.q.len() != nkv * hd * cfg.dim || attn.indexer.k.len() != hd * cfg.dim {
            return Err(err(format!("layer {li}: attention shapes")));
        }
        let m = format!("{p}mlp.");
        let ffn = if model.tensor(&format!("{m}gate.weight")).is_none() {
            Ffn::Dense { gate: t(model, &format!("{m}gate_proj.weight"))?, up: t(model, &format!("{m}up_proj.weight"))?, down: t(model, &format!("{m}down_proj.weight"))? }
        } else {
            let mut experts = Vec::with_capacity(moe.num_experts);
            for e in 0..moe.num_experts {
                let x = format!("{m}experts.{e}.");
                experts.push([t(model, &format!("{x}gate_proj.weight"))?, t(model, &format!("{x}up_proj.weight"))?, t(model, &format!("{x}down_proj.weight"))?]);
            }
            let mo = Moe {
                router: f(model, &format!("{m}gate.weight"))?,
                bias: f(model, &format!("{m}gate.e_score_correction_bias"))?,
                experts,
                shared: [
                    t(model, &format!("{m}shared_experts.gate_proj.weight"))?,
                    t(model, &format!("{m}shared_experts.up_proj.weight"))?,
                    t(model, &format!("{m}shared_experts.down_proj.weight"))?,
                ],
            };
            if mo.bias.len() != moe.num_experts || mo.router.len() != moe.num_experts * cfg.dim {
                return Err(err(format!("layer {li}: router shapes")));
            }
            Ffn::Moe(mo)
        };
        layers.push(Layer {
            attn_hc: hc_of("attn_hc")?,
            ffn_hc: hc_of("ffn_hc")?,
            input_norm: f(model, &format!("{p}input_layernorm.weight"))?,
            post_norm: f(model, &format!("{p}post_attention_layernorm.weight"))?,
            attn,
            ffn,
        });
    }
    let mut state = State { pos: 0, tokens: Vec::new(), layers: Vec::new(), sequence_id: 0, graph: None };
    state.reset(layers.len());
    Ok((globals, layers, cfg, state))
}

/// Per-head RMS norm with one weight vector shared by the heads.
fn norm_heads(x: &mut [f32], w: &[f32], hd: usize, eps: f64) {
    for h in x.chunks_mut(hd) {
        let ms = h.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hd as f64;
        let r = (1.0 / (ms + eps).sqrt()) as f32;
        for (v, &g) in h.iter_mut().zip(w) {
            *v = *v * r * g;
        }
    }
}

fn rope_heads(x: &mut [f32], invf: &[f32], hd: usize, pos: usize) {
    for h in x.chunks_mut(hd) {
        crate::hy4::rope_half(h, invf, pos);
    }
}

/// The blocks position `pos` attends to in every KV group, ascending: the
/// prefix, its own block and the ones before it, and — once the blocks in
/// between outnumber `remote_blocks` — the best of those by `score(group,
/// block)`, the block's largest token score (ties to the lower block).
pub fn route_blocks(pos: usize, cfg: &Cfg, groups: usize, mut score: impl FnMut(usize, usize) -> f32) -> Vec<Vec<usize>> {
    let cur = pos / cfg.block;
    let local0 = (cur + 1).saturating_sub(cfg.local_blocks);
    let mut fixed: Vec<usize> = (0..cfg.prefix_blocks.min(cur + 1)).collect();
    fixed.extend((local0.max(cfg.prefix_blocks)..=cur).filter(|b| *b >= cfg.prefix_blocks));
    // the blocks in between: wholly past the prefix and wholly before the local window
    let between: Vec<usize> = (cfg.prefix_blocks..local0).collect();
    (0..groups)
        .map(|g| {
            let mut sel = fixed.clone();
            if between.len() <= cfg.remote_blocks {
                sel.extend(&between);
            } else {
                let mut scored: Vec<(usize, f32)> = between.iter().map(|&b| (b, score(g, b))).collect();
                scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
                sel.extend(scored[..cfg.remote_blocks].iter().map(|s| s.0));
            }
            sel.sort_unstable();
            sel
        })
        .collect()
}

impl Attn {
    /// One position: its rows into the cache, the block route, the attention
    /// over the routed positions, the O projection.
    fn forward(&self, x: &[f32], ls: &mut LayerState, pos: usize, cfg: &Cfg, pool: Option<&Pool>) -> Vec<f32> {
        let (nh, nkv, hd, dim) = (cfg.nh, cfg.nkv, cfg.hd, cfg.dim);
        let per = nh / nkv;
        let mut q = vec![0f32; nh * hd];
        let mut k = vec![0f32; nkv * hd];
        let mut v = vec![0f32; nkv * hd];
        self.q.matvec(x, &mut q, pool);
        self.k.matvec(x, &mut k, pool);
        self.v.matvec(x, &mut v, pool);
        norm_heads(&mut q, &self.q_norm, hd, cfg.eps);
        norm_heads(&mut k, &self.k_norm, hd, cfg.eps);
        rope_heads(&mut q, &cfg.invf, hd, pos);
        rope_heads(&mut k, &cfg.invf, hd, pos);
        ls.k.extend_from_slice(&k);
        ls.v.extend_from_slice(&v);
        // the indexer's key of this position (every position keeps one: a later query may need it)
        let ix = &self.indexer;
        let mut ik = vec![0f32; hd];
        crate::pool::matvec_rows(pool, &ix.k, x, &mut ik);
        norm_heads(&mut ik, &ix.k_norm, hd, cfg.eps);
        rope_heads(&mut ik, &cfg.invf, hd, pos);
        ls.ik.extend_from_slice(&ik);
        debug_assert_eq!(ls.k.len(), (pos + 1) * nkv * hd);

        let sparse = pos >= cfg.dense_range();
        let routes: Vec<Vec<usize>> = if sparse {
            let mut iq = vec![0f32; nkv * hd];
            crate::pool::matvec_rows(pool, &ix.q, x, &mut iq);
            norm_heads(&mut iq, &ix.q_norm, hd, cfg.eps);
            rope_heads(&mut iq, &cfg.invf, hd, pos);
            // the best token score of every candidate block, a group at a time
            let cur = pos / cfg.block;
            let local0 = (cur + 1).saturating_sub(cfg.local_blocks);
            let (b0, b1) = (cfg.prefix_blocks, local0);
            let nb = b1 - b0;
            let mut best = vec![f32::NEG_INFINITY; nkv * nb];
            let keys = &ls.ik;
            let block = cfg.block;
            let work = |i: usize, out: &mut f32| {
                let (g, b) = (i / nb, b0 + i % nb);
                let qg = &iq[g * hd..(g + 1) * hd];
                let mut mx = f32::NEG_INFINITY;
                for tpos in b * block..(b + 1) * block {
                    let s: f32 = keys[tpos * hd..(tpos + 1) * hd].iter().zip(qg).map(|(a, b)| a * b).sum();
                    mx = mx.max(s);
                }
                *out = mx;
            };
            match pool {
                Some(p) => {
                    let addr = crate::pool::SendMut::new(best.as_mut_ptr());
                    p.run_rows(nkv * nb, &|a, b| {
                        for i in a..b {
                            work(i, unsafe { &mut *addr.at(i) });
                        }
                    });
                }
                None => {
                    for (i, o) in best.iter_mut().enumerate() {
                        work(i, o);
                    }
                }
            }
            route_blocks(pos, cfg, nkv, |g, b| best[g * nb + (b - b0)])
        } else {
            Vec::new()
        };

        let mut out = vec![0f32; nh * hd];
        let (kc, vc) = (&ls.k, &ls.v);
        let scale = cfg.scale;
        let block = cfg.block;
        let head = |h: usize, o: &mut [f32]| {
            let g = h / per;
            let qh = &q[h * hd..(h + 1) * hd];
            let mut scores: Vec<(usize, f32)> = Vec::new();
            let mut push = |tpos: usize| {
                let kr = &kc[(tpos * nkv + g) * hd..(tpos * nkv + g + 1) * hd];
                scores.push((tpos, kr.iter().zip(qh).map(|(a, b)| a * b).sum::<f32>() * scale));
            };
            if sparse {
                for &b in &routes[g] {
                    for tpos in b * block..((b + 1) * block).min(pos + 1) {
                        push(tpos);
                    }
                }
            } else {
                for tpos in 0..=pos {
                    push(tpos);
                }
            }
            let mx = scores.iter().fold(f32::NEG_INFINITY, |a, s| a.max(s.1));
            let mut den = 0f64;
            for s in scores.iter_mut() {
                s.1 = (s.1 - mx).exp();
                den += s.1 as f64;
            }
            let inv = (1.0 / den) as f32;
            for &(tpos, w) in &scores {
                let vr = &vc[(tpos * nkv + g) * hd..(tpos * nkv + g + 1) * hd];
                let w = w * inv;
                for (a, b) in o.iter_mut().zip(vr) {
                    *a += w * b;
                }
            }
        };
        match pool {
            Some(p) => {
                let addr = crate::pool::SendMut::new(out.as_mut_ptr());
                p.run_rows(nh, &|a, b| {
                    for h in a..b {
                        head(h, unsafe { std::slice::from_raw_parts_mut(addr.at(h * hd), hd) });
                    }
                });
            }
            None => {
                for (h, o) in out.chunks_mut(hd).enumerate() {
                    head(h, o);
                }
            }
        }
        let mut y = vec![0f32; dim];
        self.o.matvec(&out, &mut y, pool);
        y
    }
}

fn swiglu(g: &mut [f32], u: &[f32]) {
    for (a, &b) in g.iter_mut().zip(u) {
        *a = *a / (1.0 + (-*a).exp()) * b;
    }
}

fn dense(gate: &QTensor, up: &QTensor, down: &QTensor, x: &[f32], pool: Option<&Pool>) -> Vec<f32> {
    let mut g = vec![0f32; gate.rows()];
    let mut u = vec![0f32; up.rows()];
    gate.matvec(x, &mut g, pool);
    up.matvec(x, &mut u, pool);
    swiglu(&mut g, &u);
    let mut out = vec![0f32; down.rows()];
    down.matvec(&g, &mut out, pool);
    out
}

impl Moe {
    /// The selected experts and their mixing weights: the choice by
    /// `σ(logit) + bias`, the weights the chosen scores over their sum.
    pub fn route(&self, x: &[f32], cfg: &Cfg, pool: Option<&Pool>) -> Vec<(usize, f32)> {
        let ne = self.bias.len();
        let mut logits = vec![0f32; ne];
        crate::pool::matvec_rows(pool, &self.router, x, &mut logits);
        route_logits(&logits, &self.bias, cfg.top_k)
    }

    fn forward(&self, x: &[f32], cfg: &Cfg, pool: Option<&Pool>) -> Vec<f32> {
        let mut y = dense(&self.shared[0], &self.shared[1], &self.shared[2], x, pool);
        for (e, w) in self.route(x, cfg, pool) {
            let [g, u, d] = &self.experts[e];
            let ye = dense(g, u, d, x, pool);
            for (a, v) in y.iter_mut().zip(&ye) {
                *a += w * v;
            }
        }
        y
    }
}

/// Top-k by `σ(logit) + bias` (ties to the lower expert), weights
/// `σ / (Σσ + 1e-20)` over the chosen ones.
pub fn route_logits(logits: &[f32], bias: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|&a, &b| (scores[b] + bias[b]).partial_cmp(&(scores[a] + bias[a])).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    order.truncate(top_k);
    let sum: f32 = order.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20;
    order.into_iter().map(|e| (e, scores[e] / sum)).collect()
}

/// `CMF_MAXWELL_DUMP=<dir>,<pos>,<pos>…`: the host stack writes every layer's
/// attention output and streams at those positions
/// (`<dir>/<kind>_l<li>_p<pos>.f32`), for a layer-by-layer diff against the
/// reference.
fn dump(li: usize, pos: usize, kind: &str, v: &[f32]) {
    static CFG: std::sync::OnceLock<Option<(String, Vec<usize>)>> = std::sync::OnceLock::new();
    let cfg = CFG.get_or_init(|| {
        let s = std::env::var("CMF_MAXWELL_DUMP").ok()?;
        let mut it = s.split(',');
        let dir = it.next()?.to_string();
        Some((dir, it.filter_map(|p| p.parse().ok()).collect()))
    });
    if let Some((dir, ps)) = cfg {
        if ps.contains(&pos) {
            let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/{kind}_l{li}_p{pos}.f32"), b);
        }
    }
}

const GRAPH_CAP: usize = 1 << 20;

fn graph_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_MAXWELL_GRAPH").as_deref() != Ok("0"))
}

fn gw(t: &QTensor) -> Option<crate::gpu::GraphW<'_>> {
    if let Some((_, i, kind, rs)) = t.graph_weight_or_blr() {
        return Some(crate::gpu::GraphW { idx: i, kind, row_scale: rs, data: &[], prism: crate::gpu::GraphPrismOp::None, affine: false });
    }
    t.as_f32().map(|d| crate::gpu::GraphW { idx: 0, kind: 4, row_scale: &[], data: d, prism: crate::gpu::GraphPrismOp::None, affine: false })
}

fn gf(d: &[f32]) -> crate::gpu::GraphW<'_> {
    crate::gpu::GraphW { idx: 0, kind: 4, row_scale: &[], data: d, prism: crate::gpu::GraphPrismOp::None, affine: false }
}

/// The stack as graph layers for a call ending at `end` positions (None: a
/// weight the graphs cannot address, or a call past the dense range on a
/// backend whose attention has no block router).
fn graph_layers<'a>(globals: &'a Globals, layers: &'a [Layer], cfg: &'a Cfg, end: usize) -> Option<Vec<crate::gpu::GraphLayer<'a>>> {
    use crate::gpu::{GraphAct, GraphAttn, GraphAttnGeom, GraphBlockRoute, GraphFfn, GraphLayer, GraphMhc, GraphMhcDsv4, GraphMhcHead, GraphMoeDetachedShared};
    let routed = end > cfg.dense_range();
    if routed && !crate::gpu::block_route_on() {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!(
                "maxwell_moe: past {} positions the attention picks its blocks, which this backend's graphs do not do; \
                 the sequence continues on the host (the CUDA backend, CMF_GPU=cuda, keeps it on the card)",
                cfg.dense_range()
            );
        }
        return None;
    }
    let last = layers.len().checked_sub(1)?;
    let mut out = Vec::with_capacity(layers.len());
    for (li, l) in layers.iter().enumerate() {
        let a = &l.attn;
        let ix = &a.indexer;
        let attn = GraphAttn::Full {
            wq: gw(&a.q)?,
            wk: gw(&a.k)?,
            wv: gw(&a.v)?,
            wo: gw(&a.o)?,
            q_norm: Some(&a.q_norm),
            k_norm: Some(&a.k_norm),
            late_qk_norm: false,
            bias: None,
            output_gate: false,
            cpu_k: &[],
            cpu_v: &[],
            cpu_base: 0,
            geom: Some(GraphAttnGeom {
                nkv: cfg.nkv,
                hd: cfg.hd,
                dv: cfg.hd,
                rd: cfg.hd,
                invf: &cfg.invf,
                rope_scale: 1.0,
                window: None,
                sink: None,
                v_norm: false,
                dash: None,
                vres: None,
                route: Some(GraphBlockRoute {
                    q: &ix.q,
                    k: &ix.k,
                    q_norm: &ix.q_norm,
                    k_norm: &ix.k_norm,
                    block: cfg.block,
                    prefix: cfg.prefix_blocks,
                    local: cfg.local_blocks,
                    remote: cfg.remote_blocks,
                }),
            }),
            head_gate: None,
        };
        let ffn = match &l.ffn {
            Ffn::Dense { gate, up, down } => GraphFfn::Dense { gate: gw(gate)?, up: gw(up)?, down: gw(down)?, act: GraphAct::Silu, limit: 0.0, hot: None },
            Ffn::Moe(m) => {
                let e0 = m.experts.first()?;
                let q82 = e0[0].mapped_q8_2f().is_some();
                let gu_q2 = e0[0].mapped_q2tp().is_some();
                let uniform = m.experts.iter().all(|[g, u, d]| {
                    if q82 {
                        g.mapped_q8_2f().is_some() && u.mapped_q8_2f().is_some() && d.mapped_q8_2f().is_some()
                    } else if gu_q2 {
                        g.mapped_q2tp().is_some() && u.mapped_q2tp().is_some() && d.mapped_q4tp().is_some()
                    } else {
                        g.mapped_q4tp().is_some() && u.mapped_q4tp().is_some() && d.mapped_q4tp().is_some()
                    }
                });
                if !uniform {
                    return None;
                }
                let experts = m.experts.iter().map(|[g, u, d]| Some((g.model_idx()?, u.model_idx()?, d.model_idx()?))).collect::<Option<Vec<_>>>()?;
                GraphFfn::Moe {
                    router: gf(&m.router),
                    shared_gate: gf(&m.router),
                    experts,
                    n_exp: m.experts.len(),
                    top_k: cfg.top_k,
                    inter: e0[0].rows(),
                    act: GraphAct::Silu,
                    norm_topk: true,
                    q4tp: !q82,
                    gu_q2,
                    q82,
                    sigmoid: true,
                    sigmoid_denom_eps: 1e-20,
                    bias: Some(&m.bias),
                    has_shared: false,
                    detached_shared: Some(GraphMoeDetachedShared { gate: gw(&m.shared[0])?, up: gw(&m.shared[1])?, down: gw(&m.shared[2])?, width: m.shared[0].rows() }),
                    shared_gated: false,
                    route_scale: 1.0,
                    limit: 0.0,
                    shared_limit: None,
                    latent: None,
                }
            }
        };
        out.push(GraphLayer {
            mhc: Some(GraphMhc {
                attn_w: &l.attn_hc.function,
                attn_b: &l.attn_hc.base,
                ffn_w: &l.ffn_hc.function,
                ffn_b: &l.ffn_hc.base,
                enter: li == 0,
                leave: li == last,
                iters: cfg.sinkhorn_iters,
                pre_scale: 1.0,
                // a negative RMS epsilon: added to the RMS itself (`Hc::mixes`)
                dsv4: Some(GraphMhcDsv4 { attn_scale: l.attn_hc.scale, ffn_scale: l.ffn_hc.scale, eps: cfg.hc_eps, rms_eps: -cfg.hc_eps, ihc: None }),
                head: (li == last).then_some(GraphMhcHead { w: &globals.head_fn, base: &globals.head_base, scale: globals.head_scale }),
            }),
            input_norm: &l.input_norm,
            attn,
            post_norm: &l.post_norm,
            ffn,
            attn_out_norm: None,
            ffn_out_norm: None,
            layer_scale: None,
            engram: None,
        });
    }
    Some(out)
}

/// The widest FFN of the stack (the graphs' scratch width).
fn graph_inter(layers: &[Layer]) -> usize {
    layers
        .iter()
        .map(|l| match &l.ffn {
            Ffn::Dense { gate, .. } => gate.rows(),
            Ffn::Moe(m) => m.experts.first().map_or(0, |e| e[0].rows()).max(m.shared[0].rows()),
        })
        .max()
        .unwrap_or(0)
}

/// The host takes the sequence over from the card: its tokens replayed
/// through the host stack.
fn host_replay(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, pool: Option<&Pool>) {
    let toks = std::mem::take(&mut state.tokens);
    tracing::warn!("maxwell_moe: the host replays {} tokens the card held", toks.len());
    state.reset(layers.len());
    state.graph = Some(false);
    let mut scratch = Vec::new();
    for (p, &tk) in toks.iter().enumerate() {
        host_token(globals, layers, cfg, state, tk, p, pool, &mut scratch, false);
    }
}

/// A prompt's positions `start..` through the card's batch graph (the rows in
/// chunks), its last one through the token graph with the logits. Returns
/// the positions taken (0: none, the caller goes a token at a time).
#[allow(clippy::too_many_arguments)]
pub fn forward_prompt(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, ids: &[u32], start: usize, pool: Option<&Pool>, logits: &mut Vec<f32>) -> usize {
    if ids.len() < 2 {
        return 0;
    }
    let body = ids.len() - 1;
    let done = prompt_rows(globals, layers, cfg, state, &ids[..body], start, pool, None);
    if done < body {
        return done;
    }
    forward_token(globals, layers, cfg, state, ids[body], start + body, pool, logits, true);
    ids.len()
}

/// `ids` from `start` on through the card's prompt graph, a chunk a call
/// (`CMF_MAXWELL_PROMPT_CHUNK`, 512; halved down to 16 when the card
/// declines a chunk); with `keep`, every row's hidden state after the stack (the head's fold,
/// before the final norm) appended to it. Returns the rows the card took.
#[allow(clippy::too_many_arguments)]
fn prompt_rows(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, ids: &[u32], start: usize, pool: Option<&Pool>, mut keep: Option<&mut Vec<f32>>) -> usize {
    use crate::gpu::BatchGraphOutcome;
    if ids.is_empty() || !graph_on() || !(crate::gpu::cuda_active() || crate::gpu::wgpu_active()) {
        return 0;
    }
    if start == 0 || state.pos != start {
        state.reset(layers.len());
    }
    if state.graph == Some(false) || (start > 0 && state.graph != Some(true)) {
        return 0;
    }
    let model = globals.model.clone();
    let inter = graph_inter(layers);
    let dim = cfg.dim;
    let mut chunk = std::env::var("CMF_MAXWELL_PROMPT_CHUNK").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(512).max(1);
    let body = ids.len();
    let mut done = 0usize;
    let timing = std::env::var_os("CMF_CUDA_PROF").is_some();
    while done < body {
        let end = (done + chunk).min(body);
        let k = end - done;
        let t0 = std::time::Instant::now();
        let Some(gl) = graph_layers(globals, layers, cfg, start + end) else { return done };
        let o1: Vec<Option<Vec<crate::nystrom::O1DeviceView<'_>>>> = (0..gl.len()).map(|_| None).collect();
        let mut h = vec![0.0f32; k * dim];
        for (j, &tk) in ids[done..end].iter().enumerate() {
            if (tk as usize) < globals.embed.rows() {
                globals.embed.row_f32(tk as usize, &mut h[j * dim..(j + 1) * dim]);
            }
        }
        let positions: Vec<usize> = (start + done..start + end).collect();
        let outcome = crate::gpu::forward_batch_graph_at(
            &model, state.kv_id(), 0, &gl, &cfg.invf, &mut h, cfg.nh, cfg.nkv, cfg.hd, cfg.hd, dim, inter, &positions, GRAPH_CAP, false, cfg.eps as f32, cfg.scale, k, &o1, 0, None, None,
        );
        if timing {
            tracing::info!("maxwell_moe prompt chunk of {k} rows: the card {:.2} s", t0.elapsed().as_secs_f64());
        }
        match outcome {
            BatchGraphOutcome::Completed => {
                if state.graph.is_none() {
                    tracing::info!("maxwell_moe: the prompt runs on the card's batch graph");
                }
                state.graph = Some(true);
                state.tokens.extend_from_slice(&ids[done..end]);
                state.pos = start + end;
                done = end;
                if let Some(kp) = keep.as_deref_mut() {
                    kp.extend_from_slice(&h);
                }
            }
            BatchGraphOutcome::Declined => {
                // a chunk's rows may not fit beside the weights (a card whose budget the
                // experts fill): fewer rows a call before the rows go a token at a time
                if chunk > 16 {
                    chunk = (chunk / 2).max(16);
                    continue;
                }
                if done > 0 || state.graph == Some(true) {
                    // the card holds the rows before this chunk: the host takes them over
                    host_replay(globals, layers, cfg, state, pool);
                }
                return done;
            }
            BatchGraphOutcome::Failed => {
                host_replay(globals, layers, cfg, state, pool);
                return done;
            }
        }
    }
    done
}

/// Perplexity on the card's prompt graph (`CMF_MAXWELL_NLL_BATCH=0`: the
/// token path): the rows of `ids[..n−1]` through the stack in long calls, the
/// final norm and the head on the host, `nll` of every next token from
/// `from` on. None when the card does not take the window.
pub fn nll(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, ids: &[u32], from: usize, pool: Option<&Pool>) -> Option<(f64, usize)> {
    if std::env::var("CMF_MAXWELL_NLL_BATCH").as_deref() == Ok("0") || ids.len() < 2 || !graph_on() || !(crate::gpu::cuda_active() || crate::gpu::wgpu_active()) {
        return None;
    }
    let n = ids.len() - 1;
    let dim = cfg.dim;
    let t0 = std::time::Instant::now();
    state.reset(layers.len());
    // a window the card cannot take whole is scored the token way: no host replay here
    graph_layers(globals, layers, cfg, n)?;
    let mut rows = Vec::with_capacity(n * dim);
    if prompt_rows(globals, layers, cfg, state, &ids[..n], 0, pool, Some(&mut rows)) != n || rows.len() != n * dim {
        state.reset(layers.len());
        return None;
    }
    state.reset(layers.len());
    let t_rows = t0.elapsed().as_secs_f64();
    let vocab = globals.lm_head.rows();
    let (mut nll, mut cnt) = (0f64, 0usize);
    const B: usize = 64;
    let mut out = vec![0f32; B * vocab];
    let mut xs = vec![0f32; B * dim];
    let mut r0 = from.min(n);
    while r0 < n {
        let b = B.min(n - r0);
        for j in 0..b {
            let x = rms_w(&rows[(r0 + j) * dim..(r0 + j + 1) * dim], &globals.norm, cfg.eps);
            xs[j * dim..(j + 1) * dim].copy_from_slice(&x);
        }
        globals.lm_head.matmat(&xs[..b * dim], b, &mut out[..b * vocab], pool);
        for j in 0..b {
            let row = &out[j * vocab..(j + 1) * vocab];
            let target = ids[r0 + j + 1] as usize;
            let best = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v)) as f64;
            let den: f64 = row.iter().map(|&v| ((v as f64) - best).exp()).sum();
            nll += den.ln() - (row[target.min(vocab - 1)] as f64 - best);
            cnt += 1;
        }
        r0 += b;
    }
    tracing::info!("maxwell_moe nll: {n} rows through the card in {t_rows:.1} s ({:.1} rows/s), the head on the host {:.1} s", n as f64 / t_rows.max(1e-9), t0.elapsed().as_secs_f64() - t_rows);
    Some((nll, cnt))
}

/// One token through the card's graph: the embedding in, the logits out.
#[allow(clippy::too_many_arguments)]
fn graph_token(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &State, token_id: u32, position: usize, logits: &mut Vec<f32>, want_logits: bool) -> crate::gpu::TokenGraphOutcome {
    use crate::gpu::TokenGraphOutcome;
    if !graph_on() || !(crate::gpu::cuda_active() || crate::gpu::wgpu_active()) {
        return TokenGraphOutcome::Declined;
    }
    let Some(head) = gw(&globals.lm_head) else { return TokenGraphOutcome::Declined };
    let Some(gl) = graph_layers(globals, layers, cfg, position + 1) else { return TokenGraphOutcome::Declined };
    let mut h = vec![0.0f32; cfg.dim];
    if (token_id as usize) < globals.embed.rows() {
        globals.embed.row_f32(token_id as usize, &mut h);
    }
    let o1: Vec<Option<Vec<crate::nystrom::O1DeviceView<'_>>>> = (0..gl.len()).map(|_| None).collect();
    let vocab = globals.lm_head.rows();
    crate::gpu::forward_token_graph(
        &globals.model, state.kv_id(), &gl, &o1, 0, &cfg.invf, &mut h, cfg.nh, cfg.nkv, cfg.hd, cfg.scale, cfg.hd, cfg.dim, graph_inter(layers), position, GRAPH_CAP, false, cfg.eps as f32,
        want_logits.then_some((&head, vocab)), &globals.norm, logits, &[], 1, None, None, None, 0, false,
    )
}

/// One token: on the card's graph while the card holds the sequence (or it
/// takes it at position 0), else through the host stack.
#[allow(clippy::too_many_arguments)]
pub fn forward_token(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, token_id: u32, position: usize, pool: Option<&Pool>, logits: &mut Vec<f32>, want_logits: bool) {
    use crate::gpu::TokenGraphOutcome;
    if position == 0 || state.pos != position {
        if position != 0 && state.pos != position {
            tracing::warn!("maxwell_moe: position {position} after {}; the sequence restarts", state.pos);
        }
        state.reset(layers.len());
    }
    let try_graph = match state.graph {
        Some(true) => true,
        None => position == 0,
        Some(false) => false,
    };
    if try_graph {
        match graph_token(globals, layers, cfg, state, token_id, position, logits, want_logits) {
            TokenGraphOutcome::Completed => {
                if state.graph.is_none() {
                    tracing::info!("maxwell_moe: decode runs on the card's token graph");
                }
                state.graph = Some(true);
                state.tokens.push(token_id);
                state.pos = position + 1;
                return;
            }
            TokenGraphOutcome::Declined if state.graph == Some(true) => {
                // the card held the sequence and cannot go on: the host replays it
                host_replay(globals, layers, cfg, state, pool);
            }
            TokenGraphOutcome::Declined => state.graph = Some(false),
            TokenGraphOutcome::Failed => host_replay(globals, layers, cfg, state, pool),
        }
    }
    host_token(globals, layers, cfg, state, token_id, position, pool, logits, want_logits);
}

/// The streams of one position after the whole stack (host).
fn host_streams(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, token_id: u32, position: usize, pool: Option<&Pool>) -> Vec<f32> {
    let (dim, hc) = (cfg.dim, cfg.hc);
    let mut h = vec![0f32; dim];
    if (token_id as usize) < globals.embed.rows() {
        globals.embed.row_f32(token_id as usize, &mut h);
    }
    let mut streams: Vec<f32> = (0..hc).flat_map(|_| h.iter().copied()).collect();
    let mut folded = vec![0f32; dim];
    let mut next = vec![0f32; hc * dim];
    for (li, l) in layers.iter().enumerate() {
        let (pre, post, comb) = l.attn_hc.split(&streams, cfg, pool);
        hc_fold(&streams, &pre, hc, dim, &mut folded);
        let x = rms_w(&folded, &l.input_norm, cfg.eps);
        let y = l.attn.forward(&x, &mut state.layers[li], position, cfg, pool);
        dump(li, position, "attn", &y);
        hc_expand(&y, &streams, &post, &comb, hc, dim, &mut next);
        std::mem::swap(&mut streams, &mut next);
        let (pre, post, comb) = l.ffn_hc.split(&streams, cfg, pool);
        hc_fold(&streams, &pre, hc, dim, &mut folded);
        let x = rms_w(&folded, &l.post_norm, cfg.eps);
        let y = match &l.ffn {
            Ffn::Dense { gate, up, down } => dense(gate, up, down, &x, pool),
            Ffn::Moe(m) => m.forward(&x, cfg, pool),
        };
        dump(li, position, "ffn", &y);
        hc_expand(&y, &streams, &post, &comb, hc, dim, &mut next);
        std::mem::swap(&mut streams, &mut next);
        dump(li, position, "streams", &streams);
    }
    streams
}

/// The closing fold of the streams: `σ(mixes · scale + base) + eps` weights.
fn head_fold(globals: &Globals, cfg: &Cfg, streams: &[f32]) -> Vec<f32> {
    let (dim, hc) = (cfg.dim, cfg.hc);
    let n = streams.len();
    let ms = streams.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
    let r = (1.0 / (ms + cfg.hc_eps as f64).sqrt()) as f32;
    let pre: Vec<f32> = (0..hc)
        .map(|j| {
            let m = globals.head_fn[j * n..(j + 1) * n].iter().zip(streams).map(|(a, b)| a * b).sum::<f32>() * r;
            sigmoid(m * globals.head_scale + globals.head_base[j]) + cfg.hc_eps
        })
        .collect();
    let mut out = vec![0f32; dim];
    hc_fold(streams, &pre, hc, dim, &mut out);
    out
}

/// One token through the host stack; the logits when asked.
#[allow(clippy::too_many_arguments)]
fn host_token(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, token_id: u32, position: usize, pool: Option<&Pool>, logits: &mut Vec<f32>, want_logits: bool) {
    if position == 0 {
        state.layers = (0..layers.len()).map(|_| LayerState::default()).collect();
        state.tokens.clear();
    }
    state.tokens.push(token_id);
    let streams = host_streams(globals, layers, cfg, state, token_id, position, pool);
    state.pos = position + 1;
    if !want_logits {
        return;
    }
    let x = rms_w(&head_fold(globals, cfg, &streams), &globals.norm, cfg.eps);
    dump(layers.len(), position, "final", &x);
    logits.resize(globals.lm_head.rows(), 0.0);
    globals.lm_head.matvec(&x, logits, pool);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Cfg {
        Cfg { dim: 8, hc: 4, eps: 1e-6, hc_eps: 1e-6, sinkhorn_iters: 20, nh: 4, nkv: 2, hd: 4, scale: 0.5, invf: vec![1.0, 0.1], block: 128, prefix_blocks: 8, local_blocks: 8, remote_blocks: 16, top_k: 2 }
    }

    #[test]
    fn below_the_dense_range_every_block_is_routed() {
        let c = cfg();
        assert_eq!(c.dense_range(), 4096);
        for pos in [0usize, 127, 128, 1023, 1024, 2047, 2048, 4095] {
            let r = route_blocks(pos, &c, 2, |_, _| panic!("no scoring below the dense range"));
            let want: Vec<usize> = (0..=pos / 128).collect();
            assert_eq!(r[0], want, "position {pos}");
            assert_eq!(r[1], want);
        }
    }

    #[test]
    fn past_the_dense_range_the_best_blocks_join_the_fixed_ones() {
        let c = cfg();
        // position 5000: block 39; local 32..=39, prefix 0..8, candidates 8..32 (24 blocks), 16 picked
        let r = route_blocks(5000, &c, 2, |g, b| if g == 0 { b as f32 } else { -(b as f32) });
        let mut want0: Vec<usize> = (0..8).collect();
        want0.extend(16..32);
        want0.extend(32..=39);
        assert_eq!(r[0], want0);
        let mut want1: Vec<usize> = (0..8).collect();
        want1.extend(8..24);
        want1.extend(32..=39);
        assert_eq!(r[1], want1);
        // a tie keeps the lower block
        let r = route_blocks(5000, &c, 1, |_, _| 1.0);
        let mut want: Vec<usize> = (0..24).collect();
        want.extend(32..=39);
        assert_eq!(r[0], want);
        // position 4096: block 32, local 25..=32, 17 candidates 8..25 of which the tie drops the last
        let r = route_blocks(4096, &c, 1, |_, _| 1.0);
        let mut want: Vec<usize> = (0..24).collect();
        want.extend(25..=32);
        assert_eq!(r[0], want);
    }

    #[test]
    fn the_router_picks_by_the_biased_score_and_weighs_by_the_plain_one() {
        // scores σ(0)=0.5, σ(2)≈0.881, σ(-2)≈0.119; the bias lifts expert 2 over expert 0
        let picks = route_logits(&[0.0, 2.0, -2.0], &[0.0, 0.0, 0.5], 2);
        assert_eq!(picks.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 2]);
        let (s1, s2) = (sigmoid(2.0), sigmoid(-2.0));
        assert!((picks[0].1 - s1 / (s1 + s2)).abs() < 1e-6);
        assert!((picks[1].1 - s2 / (s1 + s2)).abs() < 1e-6);
        assert!((picks.iter().map(|p| p.1).sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_mapping_is_scaled_by_the_rms_plus_epsilon() {
        // 2 streams of 2: rms of [3, 4, 0, 0] is 2.5; one mapping row of ones → 7 / (2.5 + eps)
        let hc = Hc { function: vec![1.0; 4], base: vec![0.0], scale: [1.0; 3] };
        let m = hc.mixes(&[3.0, 4.0, 0.0, 0.0], 1e-6, None);
        assert!((m[0] - 7.0 / (2.5 + 1e-6)).abs() < 1e-5, "{m:?}");
    }
}
