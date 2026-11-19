//! Hy4-flash (`hy_v4_flash`) text runtime.
//!
//! Transcribed from the reference `modeling_hy_v4_flash.py`
//! (transformers PR #50903), not inferred from tensor names. The stack:
//!
//! * four residual streams with identity hyper-connections (iHC): each
//!   sub-layer folds the streams with `pre = σ(·) + eps` and returns its
//!   output to every stream with `post = 2·σ(·) + eps`; the streams are never
//!   mixed (no Sinkhorn matrix);
//! * gated MLA (q through a 1024 rank, a 512 latent, 192+64 wide heads, V 256)
//!   with learned sinks and a sandwich norm on the O projection; 30 sliding
//!   layers (64 heads, a 512-position window, RoPE base 1e4) and 12 sparse
//!   ones (32 heads, base 1e7) that attend to everything up to 2048 positions
//!   and past that to the 2048 their compressed indexer recalls (groups of 4);
//! * a dense first FFN, then 41 latent MoE blocks: the router and the shared
//!   expert read the 3072-wide token, the 16-of-512 routed experts a
//!   1536-wide projection of it, their sum projected back; a norm closes the
//!   block;
//! * Engram memory before the attention of layers 1, 11, 21 and 31: bigram
//!   and trigram hashes index a 165M-row table, the rows gate into every
//!   stream.
//!
//! The converter (`cortiq-cli/src/hy4pack.rs`) stores the rotary rows
//! half-split, MLA heads rope first, and the Engram rows already through the
//! model's fp8 round trip.

use crate::loader::{Overlay, load_f32, load_matrix};
use crate::pool::Pool;
use crate::qtensor::QTensor;
use cortiq_core::{CmfError, CmfModel, ModelArch, NormStyle};
use std::sync::Arc;

pub const ARCH_NAME: &str = "hy_v4_flash";

/// The nn.RMSNorm epsilon of the Engram block. `eps=None` there, and torch's
/// fused RMSNorm then takes the epsilon of its accumulation type (f32) even
/// for bf16 activations — not `finfo(bfloat16).eps`. It matters: the streams
/// the gate's query norms have an RMS near 0.03, so 2^-7 would shrink every
/// query (and with it the gate) about threefold.
const ENGRAM_EPS: f64 = f32::EPSILON as f64;

fn err(s: impl Into<String>) -> CmfError {
    CmfError::Parse(format!("hy_v4_flash: {}", s.into()))
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

fn rms_plain(x: &mut [f32], eps: f64) {
    let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len().max(1) as f64;
    let r = 1.0 / (ms + eps).sqrt() as f32;
    for v in x.iter_mut() {
        *v *= r;
    }
}

fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    if x.is_nan() {
        return x;
    }
    f32::from_bits((b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) << 16)
}

fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as f32;
    let mag = if exp == 0 { man * (2f32).powi(-9) } else { (1.0 + man / 8.0) * (2f32).powi(exp - 7) };
    sign * mag
}

/// Half-split rotation of `v[..rd]` at `pos` (`rd / 2` frequencies).
pub(crate) fn rope_half(v: &mut [f32], invf: &[f32], pos: usize) {
    let half = invf.len();
    for i in 0..half {
        let a = pos as f64 * invf[i] as f64;
        let (s, c) = (a.sin() as f32, a.cos() as f32);
        let (x0, x1) = (v[i], v[i + half]);
        v[i] = x0 * c - x1 * s;
        v[i + half] = x1 * c + x0 * s;
    }
}

/// One identity hyper-connection: `[2·hc, hc·dim]` mixes, their bases and
/// the two group scales.
pub struct Hc {
    pub function: Vec<f32>,
    pub base: Vec<f32>,
    pub scale: [f32; 2],
}

impl Hc {
    /// The mixes `function · flat · rsqrt(mean(flat²) + eps)`.
    fn mixes(&self, flat: &[f32], rows: usize, eps: f64, pool: Option<&Pool>) -> Vec<f32> {
        let n = flat.len();
        let ms = flat.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
        let r = 1.0 / (ms + eps).sqrt() as f32;
        let mut out = vec![0f32; rows];
        let f = &self.function;
        let work = |i: usize| -> f32 { f[i * n..(i + 1) * n].iter().zip(flat).map(|(a, b)| a * b).sum::<f32>() * r };
        match pool {
            Some(p) => {
                let cells: Vec<std::sync::atomic::AtomicU32> = (0..rows).map(|_| std::sync::atomic::AtomicU32::new(0)).collect();
                p.run_rows(rows, &|a, b| {
                    for i in a..b {
                        cells[i].store(work(i).to_bits(), std::sync::atomic::Ordering::Relaxed);
                    }
                });
                for (o, c) in out.iter_mut().zip(&cells) {
                    *o = f32::from_bits(c.load(std::sync::atomic::Ordering::Relaxed));
                }
            }
            None => {
                for (i, o) in out.iter_mut().enumerate() {
                    *o = work(i);
                }
            }
        }
        out
    }

    /// (pre, post) of the sub-layer: `σ(m·s₀ + b) + eps`, `mag·σ(m·s₁ + b) + eps`.
    fn split(&self, flat: &[f32], hc: usize, eps: f64, hc_eps: f32, mag: f32, pool: Option<&Pool>) -> (Vec<f32>, Vec<f32>) {
        let m = self.mixes(flat, 2 * hc, eps, pool);
        let pre = (0..hc).map(|j| sigmoid(m[j] * self.scale[0] + self.base[j]) + hc_eps).collect();
        let post = (0..hc).map(|j| mag * sigmoid(m[hc + j] * self.scale[1] + self.base[hc + j]) + hc_eps).collect();
        (pre, post)
    }
}

pub struct Indexer {
    /// `[ih·id, q_rank]`, each head `[id - rope | rope]` (rope half-split).
    pub q: QTensor,
    /// `[ih, hidden]` head weights.
    pub hw: Vec<f32>,
    /// `[id, hidden]` token key, then its norm.
    pub tk: Vec<f32>,
    pub tk_norm: Vec<f32>,
    /// `[id, id]` pooling value and gate projections (f32), the pooled key's
    /// norm and the intra-group position embedding `[ratio, id]`.
    pub kv: Vec<f32>,
    pub kg: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub ape: Vec<f32>,
}

pub struct Attn {
    pub q_a: QTensor,
    pub q_a_norm: Vec<f32>,
    /// `[nh·(rope + nope), q_rank]`, every head rope first.
    pub q_b: QTensor,
    /// `[lora + rope, hidden]`.
    pub kv_a: QTensor,
    pub kv_a_norm: Vec<f32>,
    /// `[nh·(nope + v), lora]`, per head `[k_nope | v]`.
    pub kv_b: QTensor,
    /// `[nh·v, hidden]` elementwise output gate.
    pub gate: QTensor,
    pub o: QTensor,
    pub post_norm: Vec<f32>,
    pub sinks: Vec<f32>,
    pub nh: usize,
    pub sliding: bool,
    pub indexer: Option<Indexer>,
    kvb_f32: std::sync::OnceLock<Vec<f32>>,
}

pub struct Moe {
    /// `[E, hidden]` f32 router, the selection bias.
    pub router: Vec<f32>,
    pub bias: Vec<f32>,
    /// `[latent, hidden]`, `[hidden, latent]`.
    pub down: QTensor,
    pub up: QTensor,
    pub shared: [QTensor; 3],
    pub out_norm: Vec<f32>,
    /// (gate, up, down) of every routed expert.
    pub experts: Vec<[QTensor; 3]>,
}

pub enum Ffn {
    Dense { gate: QTensor, up: QTensor, down: QTensor, out_norm: Vec<f32> },
    Moe(Moe),
}

pub struct Engram {
    pub table: String,
    pub table_scale: String,
    pub hd: usize,
    pub lookup: Vec<u32>,
    pub mult: Vec<u64>,
    /// Hash modulus and table row offset of every head (orders major).
    pub moduli: Vec<u64>,
    pub offsets: Vec<u64>,
    pub head_norms: Vec<f32>,
    pub key: Vec<QTensor>,
    pub key_b: Vec<Vec<f32>>,
    pub value: QTensor,
    pub value_b: Vec<f32>,
    pub norm_key: Vec<Vec<f32>>,
    pub norm_query: Vec<Vec<f32>>,
    pub gamma: Vec<f32>,
    /// `[hc, hidden]`.
    pub hc_scales: Vec<f32>,
}

pub struct Layer {
    pub attn_hc: Hc,
    pub ffn_hc: Hc,
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub attn: Attn,
    pub ffn: Ffn,
    pub engram: Option<Engram>,
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
    pub mag: f32,
    pub rope: usize,
    pub nope: usize,
    pub v: usize,
    pub lora: usize,
    pub scale: f32,
    pub window: usize,
    pub invf_swa: Vec<f32>,
    pub invf_dsa: Vec<f32>,
    pub ix_nh: usize,
    pub ix_hd: usize,
    pub ratio: usize,
    pub group_topk: usize,
    /// Positions past which a sparse layer selects (group_topk · ratio).
    pub candidate: usize,
    pub top_k: usize,
    pub route_scale: f32,
    pub limit: f32,
    pub latent: usize,
    pub pad_id: u32,
    pub image_id: u32,
    pub video_id: u32,
    pub max_ngram: usize,
    pub n_per_ngram: usize,
}

/// One layer's cache: the normalized latent and the rotated shared key per
/// position (a sliding layer keeps its window from `base` on), and on the
/// sparse layers the indexer's token keys and pooled group keys.
#[derive(Default)]
struct LayerState {
    lat: Vec<f32>,
    kr: Vec<f32>,
    base: usize,
    tkeys: Vec<f32>,
    gkeys: Vec<f32>,
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
        (1u64 << 61) | self.sequence_id
    }
}

impl Drop for State {
    fn drop(&mut self) {
        if self.graph == Some(true) {
            crate::gpu::graph_kv_reset(self.kv_id());
        }
    }
}

fn inv_freq(theta: f64, rope: usize) -> Vec<f32> {
    (0..rope / 2).map(|i| (1.0 / theta.powf(2.0 * i as f64 / rope as f64)) as f32).collect()
}

pub fn load(model: &Arc<CmfModel>, arch: &ModelArch) -> Result<(Globals, Vec<Layer>, Cfg, State), CmfError> {
    let prov = model.header.provenance.as_ref().and_then(|p| p.get("hy4_flash")).ok_or_else(|| err("header has no hy4_flash record"))?;
    let tc = prov.get("text_config").ok_or_else(|| err("no text_config"))?;
    let u = |k: &str| -> Result<usize, CmfError> {
        tc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| err(format!("text_config.{k}")))
    };
    let fl = |k: &str, d: f64| tc.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
    if prov.get("rope_layout").and_then(|v| v.as_str()) != Some("half_split") {
        return Err(err("the file's rotary layout is not the converter's half-split one"));
    }
    let mla = arch.mla.as_ref().ok_or_else(|| err("missing MLA geometry"))?;
    let (rope, nope, v, lora) = (mla.qk_rope_head_dim, mla.qk_nope_head_dim, mla.v_head_dim, mla.kv_lora_rank);
    let theta = |lt: &str, d: f64| tc["rope_parameters"][lt]["rope_theta"].as_f64().unwrap_or(d);
    let cfg = Cfg {
        dim: arch.hidden_size,
        hc: u("hc_mult")?,
        eps: arch.rms_norm_eps,
        hc_eps: fl("hc_eps", 1e-6) as f32,
        mag: fl("hc_magnitude", 2.0) as f32,
        rope,
        nope,
        v,
        lora,
        scale: ((rope + nope) as f32).sqrt().recip(),
        // `CMF_HY4_WINDOW=<n>`: another sliding window for both paths (a check)
        window: std::env::var("CMF_HY4_WINDOW").ok().and_then(|v| v.parse().ok()).unwrap_or(u("sliding_window")?),
        invf_swa: inv_freq(theta("sliding_attention", 1e4), rope),
        invf_dsa: inv_freq(theta("deepseek_sparse_attention", 1e7), rope),
        ix_nh: u("compressed_indexer_num_attention_heads")?,
        ix_hd: u("compressed_indexer_head_dim")?,
        ratio: u("compressed_indexer_compress_ratio")?,
        group_topk: u("compressed_indexer_group_topk")?,
        candidate: u("compressed_indexer_group_topk")? * u("compressed_indexer_compress_ratio")?,
        top_k: u("num_experts_per_tok")?,
        route_scale: fl("routed_scaling_factor", 1.0) as f32,
        limit: fl("swiglu_limit", 0.0) as f32,
        latent: u("routed_expert_hidden_size")?,
        pad_id: u("engram_pad_id")? as u32,
        image_id: prov.get("image_token_id").and_then(|v| v.as_u64()).unwrap_or(120120) as u32,
        video_id: prov.get("video_token_id").and_then(|v| v.as_u64()).unwrap_or(120124) as u32,
        max_ngram: u("engram_max_ngram_size")?,
        n_per_ngram: u("engram_n_head_per_ngram")?,
    };
    let layer_types: Vec<&str> = tc["layer_types"].as_array().ok_or_else(|| err("layer_types"))?.iter().map(|v| v.as_str().unwrap_or("")).collect();
    let mlp_types: Vec<&str> = tc["mlp_layer_types"].as_array().ok_or_else(|| err("mlp_layer_types"))?.iter().map(|v| v.as_str().unwrap_or("")).collect();
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
    let n_exp = arch.moe.as_ref().map_or(0, |m| m.num_experts);
    let hc = cfg.hc;
    let mut layers = Vec::with_capacity(arch.num_layers);
    for li in 0..arch.num_layers {
        let p = format!("model.layers.{li}.");
        let hc_of = |kind: &str| -> Result<Hc, CmfError> {
            let s = f(model, &format!("{p}{kind}.scale"))?;
            if s.len() != 2 {
                return Err(err(format!("{p}{kind}.scale has {} values", s.len())));
            }
            Ok(Hc { function: f(model, &format!("{p}{kind}.fn"))?, base: f(model, &format!("{p}{kind}.base"))?, scale: [s[0], s[1]] })
        };
        let a = format!("{p}self_attn.");
        let sliding = layer_types.get(li).copied() == Some("sliding_attention");
        let q_b = t(model, &format!("{a}q_b_proj.weight"))?;
        let nh = q_b.rows() / (rope + nope);
        let indexer = if sliding {
            None
        } else {
            let i = format!("{a}indexer.");
            Some(Indexer {
                q: t(model, &format!("{i}query_projection.weight"))?,
                hw: f(model, &format!("{i}head_weight_projection.weight"))?,
                tk: f(model, &format!("{i}token_key_projection.weight"))?,
                tk_norm: f(model, &format!("{i}token_key_norm.weight"))?,
                kv: f(model, &format!("{i}key_value_projection.weight"))?,
                kg: f(model, &format!("{i}key_gate_projection.weight"))?,
                k_norm: f(model, &format!("{i}key_norm.weight"))?,
                ape: f(model, &format!("{i}intra_group_ape"))?,
            })
        };
        let attn = Attn {
            q_a: t(model, &format!("{a}q_a_proj.weight"))?,
            q_a_norm: f(model, &format!("{a}q_a_layernorm.weight"))?,
            q_b,
            kv_a: t(model, &format!("{a}kv_a_proj_with_mqa.weight"))?,
            kv_a_norm: f(model, &format!("{a}kv_a_layernorm.weight"))?,
            kv_b: t(model, &format!("{a}kv_b_proj.weight"))?,
            gate: t(model, &format!("{a}gate_proj.weight"))?,
            o: t(model, &format!("{a}o_proj.weight"))?,
            post_norm: f(model, &format!("{a}post_proj_norm.weight"))?,
            sinks: f(model, &format!("{a}sinks"))?,
            nh,
            sliding,
            indexer,
            kvb_f32: std::sync::OnceLock::new(),
        };
        let m = format!("{p}mlp.");
        let ffn = if mlp_types.get(li).copied() == Some("dense") {
            Ffn::Dense {
                gate: t(model, &format!("{m}gate_proj.weight"))?,
                up: t(model, &format!("{m}up_proj.weight"))?,
                down: t(model, &format!("{m}down_proj.weight"))?,
                out_norm: f(model, &format!("{m}post_fc2_norm.weight"))?,
            }
        } else {
            let mut experts = Vec::with_capacity(n_exp);
            for e in 0..n_exp {
                let x = format!("{m}experts.{e}.");
                experts.push([
                    t(model, &format!("{x}gate_proj.weight"))?,
                    t(model, &format!("{x}up_proj.weight"))?,
                    t(model, &format!("{x}down_proj.weight"))?,
                ]);
            }
            Ffn::Moe(Moe {
                router: f(model, &format!("{m}gate.weight"))?,
                bias: f(model, &format!("{m}gate.e_score_correction_bias"))?,
                down: t(model, &format!("{m}latent_down.weight"))?,
                up: t(model, &format!("{m}latent_up.weight"))?,
                shared: [
                    t(model, &format!("{m}shared_experts.gate_proj.weight"))?,
                    t(model, &format!("{m}shared_experts.up_proj.weight"))?,
                    t(model, &format!("{m}shared_experts.down_proj.weight"))?,
                ],
                out_norm: f(model, &format!("{m}combine_output_norm.weight"))?,
                experts,
            })
        };
        let engram = match tc.get("engram").and_then(|e| e.get(format!("{li}"))) {
            None => None,
            Some(meta) => {
                let d = format!("{p}engram.");
                let lid = meta["layer_id"].as_u64().ok_or_else(|| err("engram layer_id"))? as usize;
                let mult: Vec<u64> = meta["multipliers"].as_array().ok_or_else(|| err("engram multipliers"))?.iter().filter_map(|v| v.as_u64()).collect();
                let moduli: Vec<u64> = tc["engram_layer_head_vocab_sizes"][format!("{lid}")]
                    .as_array()
                    .ok_or_else(|| err("engram head vocab sizes"))?
                    .iter()
                    .flat_map(|o| o.as_array().into_iter().flatten().filter_map(|v| v.as_u64()))
                    .collect();
                let mut offsets = Vec::with_capacity(moduli.len());
                let mut acc = 0u64;
                for &mo in &moduli {
                    offsets.push(acc);
                    acc += mo;
                }
                let te = model.tensor(&format!("{d}table")).ok_or_else(|| err(format!("{d}table")))?;
                if te.shape.len() != 2 || te.shape[0] as u64 != acc {
                    return Err(err(format!("{d}table rows {:?} != {acc}", te.shape)));
                }
                let hd = te.shape[1];
                let lb = model.tensor_bytes(&format!("{d}lookup"))?;
                let lookup = lb.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
                Some(Engram {
                    table: format!("{d}table"),
                    table_scale: format!("{d}table_scale"),
                    hd,
                    lookup,
                    mult,
                    moduli,
                    offsets,
                    head_norms: f(model, &format!("{d}head_norms"))?,
                    key: (0..hc).map(|j| t(model, &format!("{d}key_projs.{j}.weight"))).collect::<Result<_, _>>()?,
                    key_b: (0..hc).map(|j| f(model, &format!("{d}key_projs.{j}.bias"))).collect::<Result<_, _>>()?,
                    value: t(model, &format!("{d}value_proj.weight"))?,
                    value_b: f(model, &format!("{d}value_proj.bias"))?,
                    norm_key: (0..hc).map(|j| f(model, &format!("{d}norm_key.{j}.weight"))).collect::<Result<_, _>>()?,
                    norm_query: (0..hc).map(|j| f(model, &format!("{d}norm_query.{j}.weight"))).collect::<Result<_, _>>()?,
                    gamma: f(model, &format!("{d}pre_gamma"))?,
                    hc_scales: f(model, &format!("{d}hc_scales"))?,
                })
            }
        };
        layers.push(Layer {
            attn_hc: hc_of("attn_hc")?,
            ffn_hc: hc_of("ffn_hc")?,
            input_norm: f(model, &format!("{p}input_layernorm.weight"))?,
            post_norm: f(model, &format!("{p}post_attention_layernorm.weight"))?,
            attn,
            ffn,
            engram,
        });
    }
    let mut state = State { pos: 0, tokens: Vec::new(), layers: Vec::new(), sequence_id: 0, graph: None };
    state.reset(layers.len());
    Ok((globals, layers, cfg, state))
}

impl Engram {
    /// The table rows (one per hash head) of the n-gram ending at `pos`.
    pub fn rows(&self, tokens: &[u32], pos: usize, cfg: &Cfg) -> Vec<u64> {
        let table = self.lookup.len();
        let comp = |p: Option<usize>| -> u64 {
            let t = p.map_or(cfg.pad_id, |p| tokens[p]) as usize;
            let t = if t < table { t } else { cfg.pad_id as usize };
            self.lookup[t] as u64
        };
        let mixed: Vec<u64> = (0..cfg.max_ngram).map(|k| comp(pos.checked_sub(k)).wrapping_mul(self.mult[k])).collect();
        let mut out = Vec::with_capacity(self.moduli.len());
        for n in 2..=cfg.max_ngram {
            let mut mix = mixed[0];
            for m in &mixed[1..n] {
                mix ^= *m;
            }
            let o0 = (n - 2) * cfg.n_per_ngram;
            for h in 0..cfg.n_per_ngram {
                out.push(self.offsets[o0 + h] + mix % self.moduli[o0 + h]);
            }
        }
        out
    }

    /// The looked-up memory `[heads · hd]`: every row decoded (bf16 of
    /// e4m3 times its scale, the reference's round trip) and RMS-normed.
    pub fn memory(&self, model: &CmfModel, rows: &[u64]) -> Result<Vec<f32>, CmfError> {
        let tab = model.tensor_bytes(&self.table)?;
        let sc = model.tensor_bytes(&self.table_scale)?;
        let hd = self.hd;
        let mut out = Vec::with_capacity(rows.len() * hd);
        for (h, &r) in rows.iter().enumerate() {
            let r = r as usize;
            let s = f32::from_bits((u16::from_le_bytes([sc[2 * r], sc[2 * r + 1]]) as u32) << 16);
            let row: Vec<f32> = tab[r * hd..(r + 1) * hd].iter().map(|&b| bf16_round(e4m3(b) * s)).collect();
            out.extend(rms_w(&row, &self.head_norms[h * hd..(h + 1) * hd], ENGRAM_EPS));
        }
        Ok(out)
    }

    /// Add the Engram residual to the `hc` streams.
    pub fn apply(&self, streams: &mut [f32], mem: &[f32], cfg: &Cfg, pool: Option<&Pool>) {
        let dim = cfg.dim;
        let mut value = vec![0f32; dim];
        self.value.matvec(mem, &mut value, pool);
        for (v, b) in value.iter_mut().zip(&self.value_b) {
            *v += b;
        }
        rms_plain(&mut value, ENGRAM_EPS);
        for (v, g) in value.iter_mut().zip(&self.gamma) {
            *v *= g;
        }
        let inv = (dim as f32).sqrt().recip();
        for j in 0..cfg.hc {
            let mut k = vec![0f32; dim];
            self.key[j].matvec(mem, &mut k, pool);
            for (v, b) in k.iter_mut().zip(&self.key_b[j]) {
                *v += b;
            }
            let k = rms_w(&k, &self.norm_key[j], ENGRAM_EPS);
            let s = &mut streams[j * dim..(j + 1) * dim];
            let q = rms_w(s, &self.norm_query[j], ENGRAM_EPS);
            let g = k.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() * inv;
            let g = g.abs().max(1e-6).sqrt() * g.signum() * (g != 0.0) as u8 as f32;
            let gate = sigmoid(g);
            let hs = &self.hc_scales[j * dim..(j + 1) * dim];
            for ((x, &m), &c) in s.iter_mut().zip(&value).zip(hs) {
                *x += gate * m + c * m;
            }
        }
    }
}

/// The groups a sparse query recalls past `cfg.candidate` positions:
/// the `group_topk − 1` best complete groups before its own, then its own.
fn select_groups(ix: &Indexer, ls: &LayerState, x: &[f32], q_resid: &[f32], pos: usize, cfg: &Cfg, pool: Option<&Pool>) -> Vec<usize> {
    let (ih, id) = (cfg.ix_nh, cfg.ix_hd);
    let own = pos / cfg.ratio;
    let mut iq = vec![0f32; ih * id];
    ix.q.matvec(q_resid, &mut iq, pool);
    for h in 0..ih {
        rope_half(&mut iq[h * id + id - cfg.rope..(h + 1) * id], &cfg.invf_dsa, pos);
    }
    let hscale = (ih as f32).sqrt().recip() * (id as f32).sqrt().recip();
    let hw: Vec<f32> = (0..ih).map(|h| ix.hw[h * cfg.dim..(h + 1) * cfg.dim].iter().zip(x).map(|(a, b)| a * b).sum::<f32>() * hscale).collect();
    let n = own.min(ls.gkeys.len() / id);
    let mut scored: Vec<(usize, f32)> = (0..n)
        .map(|g| {
            let k = &ls.gkeys[g * id..(g + 1) * id];
            let s = (0..ih)
                .map(|h| {
                    let d = iq[h * id..(h + 1) * id].iter().zip(k).map(|(a, b)| a * b).sum::<f32>();
                    hw[h] * d.max(0.0)
                })
                .sum::<f32>();
            (g, s)
        })
        .collect();
    let keep = (cfg.group_topk - 1).min(scored.len());
    scored.sort_unstable_by(|(ga, sa), (gb, sb)| sb.partial_cmp(sa).unwrap_or(std::cmp::Ordering::Equal).then(ga.cmp(gb)));
    scored.truncate(keep);
    let mut out: Vec<usize> = scored.into_iter().map(|(g, _)| g).collect();
    out.push(own);
    out
}

/// Pool the just-completed group `g` of a sparse layer's token keys into its
/// key: `Σ_slot softmax_slot(kg·k + ape) ⊙ (kv·k)`, normed, rotated at the
/// group's first position.
fn pool_group(ix: &Indexer, ls: &mut LayerState, g: usize, cfg: &Cfg) {
    let id = cfg.ix_hd;
    let r = cfg.ratio;
    let mut vals = vec![0f32; r * id];
    let mut gates = vec![0f32; r * id];
    for s in 0..r {
        let k = &ls.tkeys[(g * r + s) * id..(g * r + s + 1) * id];
        for d in 0..id {
            vals[s * id + d] = ix.kv[d * id..(d + 1) * id].iter().zip(k).map(|(a, b)| a * b).sum();
            gates[s * id + d] = ix.kg[d * id..(d + 1) * id].iter().zip(k).map(|(a, b)| a * b).sum::<f32>() + ix.ape[s * id + d];
        }
    }
    let mut pooled = vec![0f32; id];
    for d in 0..id {
        let mx = (0..r).map(|s| gates[s * id + d]).fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = (0..r).map(|s| (gates[s * id + d] - mx).exp()).sum();
        pooled[d] = (0..r).map(|s| (gates[s * id + d] - mx).exp() / z * vals[s * id + d]).sum();
    }
    let mut key = rms_w(&pooled, &ix.k_norm, cfg.eps);
    rope_half(&mut key[id - cfg.rope..], &cfg.invf_dsa, g * r);
    ls.gkeys.extend_from_slice(&key);
}

impl Attn {
    fn kvb(&self) -> &[f32] {
        self.kvb_f32.get_or_init(|| {
            let (rows, cols) = (self.kv_b.rows(), self.kv_b.cols());
            let mut v = vec![0f32; rows * cols];
            for (r, row) in v.chunks_exact_mut(cols).enumerate() {
                self.kv_b.row_f32(r, row);
            }
            v
        })
    }

    fn forward(&self, x: &[f32], ls: &mut LayerState, pos: usize, cfg: &Cfg, pool: Option<&Pool>) -> Vec<f32> {
        let (nh, rope, nope, vd, lora) = (self.nh, cfg.rope, cfg.nope, cfg.v, cfg.lora);
        let hd = rope + nope;
        let invf = if self.sliding { &cfg.invf_swa } else { &cfg.invf_dsa };
        let mut qa = vec![0f32; self.q_a.rows()];
        self.q_a.matvec(x, &mut qa, pool);
        let q_resid = rms_w(&qa, &self.q_a_norm, cfg.eps);
        let mut q = vec![0f32; nh * hd];
        self.q_b.matvec(&q_resid, &mut q, pool);
        let mut kv = vec![0f32; lora + rope];
        self.kv_a.matvec(x, &mut kv, pool);
        let lat = rms_w(&kv[..lora], &self.kv_a_norm, cfg.eps);
        let mut kr = kv[lora..].to_vec();
        rope_half(&mut kr, invf, pos);
        for h in 0..nh {
            rope_half(&mut q[h * hd..h * hd + rope], invf, pos);
        }
        ls.lat.extend_from_slice(&lat);
        ls.kr.extend_from_slice(&kr);
        let mut gate = vec![0f32; nh * vd];
        self.gate.matvec(x, &mut gate, pool);

        // the positions this query attends to
        let n_have = ls.lat.len() / lora;
        let positions: Vec<usize> = if self.sliding {
            let lo = (pos + 1).saturating_sub(cfg.window);
            (lo..=pos).collect()
        } else {
            let ix = self.indexer.as_ref().expect("a sparse layer has its indexer");
            let mut tk = vec![0f32; cfg.ix_hd];
            for (d, o) in tk.iter_mut().enumerate() {
                *o = ix.tk[d * cfg.dim..(d + 1) * cfg.dim].iter().zip(x).map(|(a, b)| a * b).sum();
            }
            let tk = rms_w(&tk, &ix.tk_norm, cfg.eps);
            ls.tkeys.extend_from_slice(&tk);
            if (pos + 1) % cfg.ratio == 0 {
                pool_group(ix, ls, pos / cfg.ratio, cfg);
            }
            if pos < cfg.candidate {
                (0..=pos).collect()
            } else {
                let mut p = Vec::with_capacity(cfg.candidate);
                for g in select_groups(ix, ls, x, &q_resid, pos, cfg, pool) {
                    for s in g * cfg.ratio..((g + 1) * cfg.ratio).min(pos + 1) {
                        p.push(s);
                    }
                }
                p
            }
        };
        let kvb = self.kvb();
        let mut merged = vec![0f32; nh * vd];
        let base = ls.base;
        debug_assert!(n_have + base > pos);
        let head = |h: usize, out: &mut [f32]| {
            let qh = &q[h * hd..(h + 1) * hd];
            let wk = &kvb[h * (nope + vd) * lora..(h * (nope + vd) + nope) * lora];
            let wv = &kvb[(h * (nope + vd) + nope) * lora..(h + 1) * (nope + vd) * lora];
            // absorbed query over the latent
            let mut ql = vec![0f32; lora];
            for r in 0..nope {
                let qv = qh[rope + r];
                for (o, &w) in ql.iter_mut().zip(&wk[r * lora..(r + 1) * lora]) {
                    *o += qv * w;
                }
            }
            let scores: Vec<f32> = positions
                .iter()
                .map(|&p| {
                    let i = p - base;
                    let l = &ls.lat[i * lora..(i + 1) * lora];
                    let k = &ls.kr[i * rope..(i + 1) * rope];
                    (ql.iter().zip(l).map(|(a, b)| a * b).sum::<f32>() + qh[..rope].iter().zip(k).map(|(a, b)| a * b).sum::<f32>()) * cfg.scale
                })
                .collect();
            let sink = self.sinks[h];
            let mx = scores.iter().copied().fold(sink, f32::max);
            let z = scores.iter().map(|&s| (s - mx).exp()).sum::<f32>() + (sink - mx).exp();
            let mut ol = vec![0f32; lora];
            for (&p, &s) in positions.iter().zip(&scores) {
                let w = (s - mx).exp() / z;
                let i = p - base;
                for (o, &l) in ol.iter_mut().zip(&ls.lat[i * lora..(i + 1) * lora]) {
                    *o += w * l;
                }
            }
            for (d, o) in out.iter_mut().enumerate() {
                *o = wv[d * lora..(d + 1) * lora].iter().zip(&ol).map(|(a, b)| a * b).sum();
            }
        };
        match pool {
            Some(p) => {
                let ptr = merged.as_mut_ptr() as usize;
                p.run_rows(nh, &|a, b| {
                    for h in a..b {
                        // SAFETY: each head writes its own disjoint vd-wide slice
                        let out = unsafe { std::slice::from_raw_parts_mut((ptr as *mut f32).add(h * vd), vd) };
                        head(h, out);
                    }
                });
            }
            None => {
                for (h, out) in merged.chunks_mut(vd).enumerate() {
                    head(h, out);
                }
            }
        }
        for (m, g) in merged.iter_mut().zip(&gate) {
            *m *= sigmoid(*g);
        }
        let mut out = vec![0f32; cfg.dim];
        self.o.matvec(&merged, &mut out, pool);
        // a sliding layer forgets what left its window
        if self.sliding && ls.lat.len() / lora > 2 * cfg.window {
            let drop = ls.lat.len() / lora - cfg.window;
            ls.lat.drain(..drop * lora);
            ls.kr.drain(..drop * rope);
            ls.base += drop;
        }
        rms_w(&out, &self.post_norm, cfg.eps)
    }
}

fn swiglu(g: &mut [f32], u: &[f32], limit: f32) {
    for (a, &b) in g.iter_mut().zip(u) {
        let (ga, ub) = if limit > 0.0 { (a.min(limit), b.clamp(-limit, limit)) } else { (*a, b) };
        *a = ga / (1.0 + (-ga).exp()) * ub;
    }
}

fn dense(gate: &QTensor, up: &QTensor, down: &QTensor, x: &[f32], limit: f32, pool: Option<&Pool>) -> Vec<f32> {
    let mut g = vec![0f32; gate.rows()];
    let mut u = vec![0f32; up.rows()];
    gate.matvec(x, &mut g, pool);
    up.matvec(x, &mut u, pool);
    swiglu(&mut g, &u, limit);
    let mut out = vec![0f32; down.rows()];
    down.matvec(&g, &mut out, pool);
    out
}

impl Moe {
    /// The selected experts and their mixing weights.
    pub fn route(&self, x: &[f32], cfg: &Cfg) -> Vec<(usize, f32)> {
        let dim = cfg.dim;
        let ne = self.bias.len();
        let scores: Vec<f32> = (0..ne).map(|e| sigmoid(self.router[e * dim..(e + 1) * dim].iter().zip(x).map(|(a, b)| a * b).sum())).collect();
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_unstable_by(|&a, &b| (scores[b] + self.bias[b]).partial_cmp(&(scores[a] + self.bias[a])).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        order.truncate(cfg.top_k);
        let sum: f32 = order.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20;
        order.into_iter().map(|e| (e, scores[e] / sum * cfg.route_scale)).collect()
    }

    fn forward(&self, x: &[f32], cfg: &Cfg, pool: Option<&Pool>) -> Vec<f32> {
        let picks = self.route(x, cfg);
        let mut lat = vec![0f32; cfg.latent];
        self.down.matvec(x, &mut lat, pool);
        let mut acc = vec![0f32; cfg.latent];
        for &(e, w) in &picks {
            let [g, u, d] = &self.experts[e];
            let y = dense(g, u, d, &lat, cfg.limit, pool);
            for (a, v) in acc.iter_mut().zip(&y) {
                *a += w * v;
            }
        }
        let mut routed = vec![0f32; cfg.dim];
        self.up.matvec(&acc, &mut routed, pool);
        let sh = dense(&self.shared[0], &self.shared[1], &self.shared[2], x, 0.0, pool);
        for (r, s) in routed.iter_mut().zip(&sh) {
            *r += s;
        }
        rms_w(&routed, &self.out_norm, cfg.eps)
    }
}

/// `CMF_HY4_DUMP=<dir>,<pos>,<pos>…`: the host stack writes every layer's
/// attention output and streams at those positions (`<dir>/<kind>_l<li>_p<pos>.f32`),
/// for a layer-by-layer diff against the reference.
fn dump(li: usize, pos: usize, kind: &str, v: &[f32]) {
    static CFG: std::sync::OnceLock<Option<(String, Vec<usize>)>> = std::sync::OnceLock::new();
    let cfg = CFG.get_or_init(|| {
        let s = std::env::var("CMF_HY4_DUMP").ok()?;
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

/// One routed FFN's SFq calibration rows (`capture`): the latent input of
/// every captured token (f16, `latent` wide), its `top_k` picks and their
/// mixing weights, as the forward routes them.
pub struct CapLayer {
    pub li: usize,
    pub x: Vec<u16>,
    pub picks: Vec<u16>,
    pub w: Vec<f32>,
}

impl Moe {
    /// The picks and mixing weights from the router's logits (`route`'s rule).
    fn route_logits(&self, logits: &[f32], cfg: &Cfg) -> Vec<(usize, f32)> {
        let ne = self.bias.len();
        let scores: Vec<f32> = logits[..ne].iter().map(|&l| sigmoid(l)).collect();
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_unstable_by(|&a, &b| (scores[b] + self.bias[b]).partial_cmp(&(scores[a] + self.bias[a])).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        order.truncate(cfg.top_k);
        let sum: f32 = order.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20;
        order.into_iter().map(|e| (e, scores[e] / sum * cfg.route_scale)).collect()
    }
}

/// SFq calibration: every window through the card's prompt graph (CUDA
/// hybrid) with the capture armed; per routed FFN, the latent rows of the
/// window's tokens and the router's picks. A window is followed by one more
/// token so all of its own rows take the batch path. Each routed FFN's rows
/// of a window go to `sink` as they come; returns the tokens captured.
pub fn capture(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    windows: &[Vec<u32>],
    sink: &mut dyn FnMut(&CapLayer) -> Result<(), String>,
) -> Result<usize, String> {
    let moe: Vec<usize> = layers.iter().enumerate().filter(|(_, l)| matches!(l.ffn, Ffn::Moe(_))).map(|(i, _)| i).collect();
    let ne = layers.iter().find_map(|l| match &l.ffn {
        Ffn::Moe(m) => Some(m.bias.len()),
        _ => None,
    });
    let ne = ne.ok_or("no routed FFN")?;
    let dim = cfg.latent;
    let t0 = std::time::Instant::now();
    let mut done = 0usize;
    for (wi, w) in windows.iter().enumerate() {
        if w.is_empty() {
            continue;
        }
        if !crate::gpu::hy4_capture_begin(moe.len(), dim, ne) {
            return Err("the capture needs the CUDA backend (CMF_GPU=cuda)".into());
        }
        state.reset(layers.len());
        let mut ids = w.clone();
        ids.push(*w.last().unwrap());
        let mut lg = Vec::new();
        let taken = forward_prompt(globals, layers, cfg, state, &ids, 0, None, &mut lg);
        let rows = crate::gpu::hy4_capture_take().ok_or("the capture lost its rows")?;
        if taken != ids.len() || rows.len() != moe.len() {
            return Err(format!("window {wi}: the card's prompt graph took {taken} of {} rows ({} routed FFNs captured)", ids.len(), rows.len()));
        }
        for (k, (lat, logits)) in rows.iter().enumerate() {
            let n = lat.len() / dim;
            if n < w.len() || logits.len() < n * ne {
                return Err(format!("window {wi}: layer {} captured {n} rows of {}", moe[k], w.len()));
            }
            let Ffn::Moe(m) = &layers[moe[k]].ffn else { unreachable!() };
            let mut o = CapLayer { li: moe[k], x: Vec::with_capacity(w.len() * dim), picks: Vec::with_capacity(w.len() * cfg.top_k), w: Vec::with_capacity(w.len() * cfg.top_k) };
            o.x.extend(lat[..w.len() * dim].iter().map(|&v| cortiq_core::quant::f32_to_f16(v)));
            for r in 0..w.len() {
                for (e, wt) in m.route_logits(&logits[r * ne..(r + 1) * ne], cfg) {
                    o.picks.push(e as u16);
                    o.w.push(wt);
                }
            }
            sink(&o)?;
        }
        done += w.len();
        tracing::info!("hy_v4_flash capture: window {}/{} ({done} tokens, {:.0} s)", wi + 1, windows.len(), t0.elapsed().as_secs_f64());
    }
    state.reset(layers.len());
    Ok(done)
}

/// Fold the streams with `pre`.
fn fold(streams: &[f32], pre: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    for (j, &p) in pre.iter().enumerate() {
        for (o, &s) in out.iter_mut().zip(&streams[j * dim..(j + 1) * dim]) {
            *o += p * s;
        }
    }
    out
}

/// Return `y` to every stream with its `post` weight.
fn expand(streams: &mut [f32], post: &[f32], y: &[f32], dim: usize) {
    for (j, &p) in post.iter().enumerate() {
        for (s, &v) in streams[j * dim..(j + 1) * dim].iter_mut().zip(y) {
            *s += p * v;
        }
    }
}

const GRAPH_CAP: usize = 1 << 20;

fn graph_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_HY4_GRAPH").as_deref() != Ok("0"))
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

/// The call's Engram memory rows of every Engram layer (the host's hash and
/// table read), for the tokens at `positions` of `tokens`.
fn engram_rows(globals: &Globals, layers: &[Layer], cfg: &Cfg, tokens: &[u32], positions: std::ops::Range<usize>) -> Option<Vec<(Vec<f32>, Vec<bool>)>> {
    // the tables' rows are reads of a large mapping, a page each: they fault
    // in in parallel (`CMF_HY4_ENGRAM_THREADS`, 32; on a pod whose page cache
    // cannot hold the 63 GB of tables, one thread took 5-15 s a 512-row chunk)
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let threads = *THREADS.get_or_init(|| std::env::var("CMF_HY4_ENGRAM_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(32).max(1));
    let pos: Vec<usize> = positions.collect();
    let mut out = Vec::new();
    for l in layers {
        let Some(e) = &l.engram else { continue };
        let width = e.hd * e.moduli.len();
        let per = pos.len().div_ceil(threads).max(16);
        let parts: Vec<Option<Vec<f32>>> = std::thread::scope(|sc| {
            let hs: Vec<_> = pos
                .chunks(per)
                .map(|ch| {
                    sc.spawn(move || {
                        let mut mem = Vec::with_capacity(ch.len() * width);
                        for &p in ch {
                            let rows = e.rows(tokens, p, cfg);
                            mem.extend(e.memory(&globals.model, &rows).ok()?);
                        }
                        Some(mem)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().ok().flatten()).collect()
        });
        let mut mem = Vec::with_capacity(pos.len() * width);
        for p in parts {
            mem.extend(p?);
        }
        let on: Vec<bool> = pos
            .iter()
            .map(|&p| {
                let t = tokens[p];
                t != cfg.image_id && t != cfg.video_id && t != cfg.pad_id
            })
            .collect();
        out.push((mem, on));
    }
    Some(out)
}

/// The stack as graph layers for a call ending at `end` positions (None: a
/// weight the graphs cannot address, or a sparse layer past its dense range,
/// whose indexer has no device arm yet).
fn graph_layers<'a>(globals: &'a Globals, layers: &'a [Layer], cfg: &'a Cfg, eng: &'a [(Vec<f32>, Vec<bool>)], end: usize) -> Option<Vec<crate::gpu::GraphLayer<'a>>> {
    use crate::gpu::{GraphAct, GraphAttn, GraphEngram, GraphFfn, GraphHyIndexer, GraphLayer, GraphMhc, GraphMhcDsv4, GraphMhcHead, GraphMlaHy, GraphMoeDetachedShared, GraphMoeLatent};
    // `CMF_HY4_DEVICE_INDEXER=0`: the card stops at the dense range of the sparse layers
    if end > cfg.candidate && std::env::var("CMF_HY4_DEVICE_INDEXER").as_deref() == Ok("0") {
        return None;
    }
    let last = layers.len().checked_sub(1)?;
    let mut ei = 0usize;
    let mut out = Vec::with_capacity(layers.len());
    for (li, l) in layers.iter().enumerate() {
        let a = &l.attn;
        let attn = GraphAttn::Mla {
            q_a: Some(gw(&a.q_a)?),
            q_a_norm: Some(&a.q_a_norm),
            q: gw(&a.q_b)?,
            kv_a: gw(&a.kv_a)?,
            kv_a_norm: &a.kv_a_norm,
            kv_b: gw(&a.kv_b)?,
            o: gw(&a.o)?,
            nh: a.nh,
            rope: cfg.rope,
            nope: cfg.nope,
            v_dim: cfg.v,
            lora: cfg.lora,
            scale: cfg.scale,
            no_rope: false,
            cpu_k: &[],
            cpu_v: &[],
            cpu_base: 0,
            dsa: None,
            hy: Some(GraphMlaHy {
                gate: gw(&a.gate)?,
                sinks: &a.sinks,
                window: a.sliding.then_some(cfg.window),
                invf: if a.sliding { &cfg.invf_swa } else { &cfg.invf_dsa },
                indexer: match &a.indexer {
                    Some(ix) => Some(GraphHyIndexer {
                        q: gw(&ix.q)?,
                        hw: &ix.hw,
                        tk: &ix.tk,
                        tk_norm: &ix.tk_norm,
                        kv: &ix.kv,
                        kg: &ix.kg,
                        k_norm: &ix.k_norm,
                        ape: &ix.ape,
                        ih: cfg.ix_nh,
                        id: cfg.ix_hd,
                        ratio: cfg.ratio,
                        group_topk: cfg.group_topk,
                        kv_b: a.kvb(),
                    }),
                    None => None,
                },
            }),
        };
        let (ffn, ffn_out) = match &l.ffn {
            Ffn::Dense { gate, up, down, out_norm } => (
                GraphFfn::Dense { gate: gw(gate)?, up: gw(up)?, down: gw(down)?, act: GraphAct::Silu, limit: 0.0, hot: None },
                &out_norm[..],
            ),
            Ffn::Moe(m) => {
                let e0 = m.experts.first()?;
                let gu_q2 = e0[0].mapped_q2tp().is_some();
                if !(gu_q2 || e0[0].mapped_q4tp().is_some()) || e0[2].mapped_q4tp().is_none() {
                    return None;
                }
                let experts = m
                    .experts
                    .iter()
                    .map(|[g, u, d]| Some((g.model_idx()?, u.model_idx()?, d.model_idx()?)))
                    .collect::<Option<Vec<_>>>()?;
                (
                    GraphFfn::Moe {
                        router: gf(&m.router),
                        shared_gate: gf(&m.router),
                        experts,
                        n_exp: m.experts.len(),
                        top_k: cfg.top_k,
                        inter: e0[0].rows(),
                        act: GraphAct::Silu,
                        norm_topk: true,
                        q4tp: true,
                        gu_q2,
                        q82: false,
                        sigmoid: true,
                        sigmoid_denom_eps: 1e-20,
                        bias: Some(&m.bias),
                        has_shared: false,
                        detached_shared: Some(GraphMoeDetachedShared {
                            gate: gw(&m.shared[0])?,
                            up: gw(&m.shared[1])?,
                            down: gw(&m.shared[2])?,
                            width: m.shared[0].rows(),
                        }),
                        shared_gated: false,
                        route_scale: cfg.route_scale,
                        limit: cfg.limit,
                        shared_limit: Some(0.0),
                        latent: Some(GraphMoeLatent { down: gw(&m.down)?, up: gw(&m.up)?, dim: cfg.latent }),
                    },
                    &m.out_norm[..],
                )
            }
        };
        let engram = match &l.engram {
            Some(e) => {
                let (mem, on) = eng.get(ei)?;
                ei += 1;
                Some(GraphEngram {
                    mem,
                    on,
                    mem_dim: e.hd * e.moduli.len(),
                    value: gw(&e.value)?,
                    value_b: &e.value_b,
                    keys: e.key.iter().map(gw).collect::<Option<Vec<_>>>()?,
                    key_b: e.key_b.iter().map(|v| &v[..]).collect(),
                    norm_key: e.norm_key.iter().map(|v| &v[..]).collect(),
                    norm_query: e.norm_query.iter().map(|v| &v[..]).collect(),
                    gamma: &e.gamma,
                    hc_scales: &e.hc_scales,
                    eps: ENGRAM_EPS as f32,
                })
            }
            None => None,
        };
        out.push(GraphLayer {
            mhc: Some(GraphMhc {
                attn_w: &l.attn_hc.function,
                attn_b: &l.attn_hc.base,
                ffn_w: &l.ffn_hc.function,
                ffn_b: &l.ffn_hc.base,
                enter: li == 0,
                leave: li == last,
                iters: 0,
                pre_scale: 1.0,
                dsv4: Some(GraphMhcDsv4 {
                    attn_scale: [l.attn_hc.scale[0], l.attn_hc.scale[1], 0.0],
                    ffn_scale: [l.ffn_hc.scale[0], l.ffn_hc.scale[1], 0.0],
                    eps: cfg.hc_eps,
                    rms_eps: cfg.eps as f32,
                    ihc: Some(cfg.mag),
                }),
                head: (li == last).then_some(GraphMhcHead { w: &globals.head_fn, base: &globals.head_base, scale: globals.head_scale }),
            }),
            input_norm: &l.input_norm,
            attn,
            post_norm: &l.post_norm,
            ffn,
            attn_out_norm: Some(&a.post_norm),
            ffn_out_norm: Some(ffn_out),
            layer_scale: None,
            engram,
        });
    }
    Some(out)
}

/// The call-wide geometry: the widest query heads, head width, softmax
/// scale and the widest FFN.
fn graph_geom(layers: &[Layer], cfg: &Cfg) -> (usize, usize, f32, usize) {
    let nh = layers.iter().map(|l| l.attn.nh).max().unwrap_or(1);
    let inter = layers
        .iter()
        .map(|l| match &l.ffn {
            Ffn::Dense { gate, .. } => gate.rows(),
            Ffn::Moe(m) => m.experts.first().map_or(0, |e| e[0].rows()).max(m.shared[0].rows()),
        })
        .max()
        .unwrap_or(0);
    (nh, cfg.rope + cfg.nope, cfg.scale, inter)
}

/// The host takes the sequence over from the card: its tokens replayed
/// through the host stack.
fn host_replay(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, pool: Option<&Pool>) {
    let toks = std::mem::take(&mut state.tokens);
    tracing::warn!("hy_v4_flash: the host replays {} tokens the card held", toks.len());
    state.reset(layers.len());
    state.graph = Some(false);
    let mut scratch = Vec::new();
    for (p, &t) in toks.iter().enumerate() {
        host_token(globals, layers, cfg, state, t, p, pool, &mut scratch, false);
    }
}

/// A prompt's positions `start..` through the card's batch graph (the rows in
/// chunks), its last one through the token graph with the logits. Returns
/// the positions taken (0: none, the caller goes a token at a time).
#[allow(clippy::too_many_arguments)]
pub fn forward_prompt(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    ids: &[u32],
    start: usize,
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
) -> usize {
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
/// (`CMF_HY4_PROMPT_CHUNK`, 512); with `keep`, every row's hidden state
/// after the stack (the head's fold, before the final norm) appended to it.
/// Returns the rows the card took.
#[allow(clippy::too_many_arguments)]
fn prompt_rows(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    ids: &[u32],
    start: usize,
    pool: Option<&Pool>,
    mut keep: Option<&mut Vec<f32>>,
) -> usize {
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
    let (nh, hd, scale, inter) = graph_geom(layers, cfg);
    let dim = cfg.dim;
    // long calls: a streamed layer's experts cross the bus once a call (the
    // CUDA hybrid runs a long prompt a layer at a time), and the Engram rows
    // of a call fault in together — 2048 rows on a card of 32 GB and more
    // (wgpu: chunks of 16 rows, `gpu_wgpu::hy4_wgpu_rows`)
    let big = crate::gpu::vram_budget() >= 32 << 30;
    #[cfg(feature = "gpu")]
    let small = if crate::gpu::wgpu_active() { Some(crate::gpu_wgpu::hy4_wgpu_rows()) } else { None };
    #[cfg(not(feature = "gpu"))]
    let small: Option<usize> = None;
    let chunk = std::env::var("CMF_HY4_PROMPT_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(small.unwrap_or(if big { 2048 } else { 512 }))
        .max(1);
    let body = ids.len();
    let mut done = 0usize;
    let mut all: Vec<u32> = state.tokens.clone();
    all.extend_from_slice(ids);
    let timing = std::env::var_os("CMF_CUDA_PROF").is_some();
    while done < body {
        let end = (done + chunk).min(body);
        let k = end - done;
        let t0 = std::time::Instant::now();
        let Some(eng) = engram_rows(globals, layers, cfg, &all, start + done..start + end) else { return done };
        let t1 = std::time::Instant::now();
        let Some(gl) = graph_layers(globals, layers, cfg, &eng, start + end) else { return done };
        let t2 = std::time::Instant::now();
        let o1: Vec<Option<Vec<crate::nystrom::O1DeviceView<'_>>>> = (0..gl.len()).map(|_| None).collect();
        let mut h = vec![0.0f32; k * dim];
        for (j, &t) in ids[done..end].iter().enumerate() {
            if (t as usize) < globals.embed.rows() {
                globals.embed.row_f32(t as usize, &mut h[j * dim..(j + 1) * dim]);
            }
        }
        let positions: Vec<usize> = (start + done..start + end).collect();
        let outcome = crate::gpu::forward_batch_graph_at(
            &model, state.kv_id(), 0, &gl, &cfg.invf_swa, &mut h, nh, nh, hd, cfg.rope, dim, inter, &positions, GRAPH_CAP, false, cfg.eps as f32, scale, k, &o1, 0, None, None,
        );
        if timing {
            tracing::info!(
                "hy_v4_flash prompt chunk of {k} rows: engram rows {:.2} s, layer views {:.2} s, the card {:.2} s",
                (t1 - t0).as_secs_f64(),
                (t2 - t1).as_secs_f64(),
                t2.elapsed().as_secs_f64()
            );
        }
        match outcome {
            BatchGraphOutcome::Completed => {
                if state.graph.is_none() {
                    tracing::info!("hy_v4_flash: the prompt runs on the card's batch graph");
                }
                state.graph = Some(true);
                state.tokens.extend_from_slice(&ids[done..end]);
                state.pos = start + end;
                done = end;
                if let Some(kp) = keep.as_deref_mut() {
                    kp.extend_from_slice(&h);
                }
            }
            BatchGraphOutcome::Declined => return done,
            BatchGraphOutcome::Failed => {
                host_replay(globals, layers, cfg, state, pool);
                return done;
            }
        }
    }
    done
}

/// Perplexity on the card's prompt graph (`CMF_HY4_NLL_BATCH=0`: the token
/// path): the rows of `ids[..n−1]` through the stack in long calls, the final
/// norm and the head on the host, `nll` of every next token from `from` on.
/// None when the card does not take the window (the caller scores it the
/// token way). The same function the token path computes, chunked the way a
/// prompt is.
pub fn nll(globals: &Globals, layers: &[Layer], cfg: &Cfg, state: &mut State, ids: &[u32], from: usize, pool: Option<&Pool>) -> Option<(f64, usize)> {
    if std::env::var("CMF_HY4_NLL_BATCH").as_deref() == Ok("0") || ids.len() < 2 {
        return None;
    }
    let n = ids.len() - 1;
    let dim = cfg.dim;
    let t0 = std::time::Instant::now();
    state.reset(layers.len());
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
    tracing::info!("hy_v4_flash nll: {n} rows through the card in {t_rows:.1} s ({:.1} rows/s), the head on the host {:.1} s", n as f64 / t_rows.max(1e-9), t0.elapsed().as_secs_f64() - t_rows);
    #[cfg(feature = "cuda")]
    if std::env::var_os("CMF_CUDA_PROF").is_some() {
        crate::gpu_cuda::perf_report();
    }
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
    let mut toks = state.tokens.clone();
    toks.push(token_id);
    let Some(eng) = engram_rows(globals, layers, cfg, &toks, position..position + 1) else { return TokenGraphOutcome::Declined };
    let Some(gl) = graph_layers(globals, layers, cfg, &eng, position + 1) else { return TokenGraphOutcome::Declined };
    let (nh, hd, scale, inter) = graph_geom(layers, cfg);
    let mut h = vec![0.0f32; cfg.dim];
    if (token_id as usize) < globals.embed.rows() {
        globals.embed.row_f32(token_id as usize, &mut h);
    }
    let o1: Vec<Option<Vec<crate::nystrom::O1DeviceView<'_>>>> = (0..gl.len()).map(|_| None).collect();
    let vocab = globals.lm_head.rows();
    crate::gpu::forward_token_graph(
        &globals.model, state.kv_id(), &gl, &o1, 0, &cfg.invf_swa, &mut h, nh, nh, hd, scale, cfg.rope, cfg.dim, inter, position, GRAPH_CAP, false, cfg.eps as f32,
        want_logits.then_some((&head, vocab)), &globals.norm, logits, &[], 1, None, None, None, 0, false,
    )
}

/// One token: on the card's graph while the card holds the sequence (or it
/// takes it at position 0), else through the host stack.
#[allow(clippy::too_many_arguments)]
pub fn forward_token(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    token_id: u32,
    position: usize,
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
    want_logits: bool,
) {
    use crate::gpu::TokenGraphOutcome;
    if position == 0 || state.pos != position {
        if position != 0 && state.pos != position {
            tracing::warn!("hy_v4_flash: position {position} after {}; the sequence restarts", state.pos);
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
                    tracing::info!("hy_v4_flash: decode runs on the card's token graph");
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

/// One token through the host stack; the logits when asked.
#[allow(clippy::too_many_arguments)]
fn host_token(
    globals: &Globals,
    layers: &[Layer],
    cfg: &Cfg,
    state: &mut State,
    token_id: u32,
    position: usize,
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
    want_logits: bool,
) {
    if position == 0 {
        state.layers = (0..layers.len()).map(|_| LayerState::default()).collect();
        state.tokens.clear();
    }
    state.tokens.push(token_id);
    let dim = cfg.dim;
    let hc = cfg.hc;
    let mut h = vec![0f32; dim];
    if (token_id as usize) < globals.embed.rows() {
        globals.embed.row_f32(token_id as usize, &mut h);
    }
    let mut streams: Vec<f32> = (0..hc).flat_map(|_| h.iter().copied()).collect();
    let engram_on = token_id != cfg.image_id && token_id != cfg.video_id && token_id != cfg.pad_id;
    for (li, l) in layers.iter().enumerate() {
        if let (Some(e), true) = (&l.engram, engram_on) {
            let rows = e.rows(&state.tokens, position, cfg);
            match e.memory(&globals.model, &rows) {
                Ok(mem) => {
                    dump(li, position, "engmem", &mem);
                    let before = streams.clone();
                    e.apply(&mut streams, &mem, cfg, pool);
                    let d: Vec<f32> = streams.iter().zip(&before).map(|(a, b)| a - b).collect();
                    dump(li, position, "engout", &d);
                }
                Err(er) => tracing::error!("hy_v4_flash: layer {li} Engram: {er}"),
            }
        }
        let (pre, post) = l.attn_hc.split(&streams, hc, cfg.eps, cfg.hc_eps, cfg.mag, pool);
        let x = rms_w(&fold(&streams, &pre, dim), &l.input_norm, cfg.eps);
        let y = l.attn.forward(&x, &mut state.layers[li], position, cfg, pool);
        dump(li, position, "attn", &y);
        expand(&mut streams, &post, &y, dim);
        let (pre, post) = l.ffn_hc.split(&streams, hc, cfg.eps, cfg.hc_eps, cfg.mag, pool);
        let x = rms_w(&fold(&streams, &pre, dim), &l.post_norm, cfg.eps);
        let y = match &l.ffn {
            Ffn::Dense { gate, up, down, out_norm } => rms_w(&dense(gate, up, down, &x, 0.0, pool), out_norm, cfg.eps),
            Ffn::Moe(m) => m.forward(&x, cfg, pool),
        };
        expand(&mut streams, &post, &y, dim);
        dump(li, position, "streams", &streams);
    }
    state.pos = position + 1;
    if !want_logits {
        return;
    }
    // the head's fold: σ(mixes · scale + base) + eps
    let n = streams.len();
    let ms = streams.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
    let r = 1.0 / (ms + cfg.eps).sqrt() as f32;
    let pre: Vec<f32> = (0..hc)
        .map(|j| {
            let m = globals.head_fn[j * n..(j + 1) * n].iter().zip(&streams).map(|(a, b)| a * b).sum::<f32>() * r;
            sigmoid(m * globals.head_scale + globals.head_base[j]) + cfg.hc_eps
        })
        .collect();
    let x = rms_w(&fold(&streams, &pre, dim), &globals.norm, cfg.eps);
    logits.resize(globals.lm_head.rows(), 0.0);
    globals.lm_head.matvec(&x, logits, pool);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_split_rotation_matches_the_interleaved_reference() {
        // interleaved pairs (x0, y0, x1, y1) rotate pair i by pos·f_i;
        // de-interleaved (x0, x1, y0, y1) must give the same numbers
        let invf = [0.5f32, 0.25];
        let pos = 3;
        let inter = [1.0f32, 2.0, 3.0, 4.0];
        let mut want = [0f32; 4];
        for i in 0..2 {
            let a = pos as f32 * invf[i];
            let (x, y) = (inter[2 * i], inter[2 * i + 1]);
            want[2 * i] = x * a.cos() - y * a.sin();
            want[2 * i + 1] = y * a.cos() + x * a.sin();
        }
        let mut half = [inter[0], inter[2], inter[1], inter[3]];
        rope_half(&mut half, &invf, pos);
        let got = [half[0], half[2], half[1], half[3]];
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).abs() < 1e-5, "{want:?} vs {got:?}");
        }
    }

    #[test]
    fn e4m3_decodes_the_ocp_values() {
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0x7e), 448.0);
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert_eq!(e4m3(0xb8), -1.0);
    }
}
