//! Qwen3-VL as MiniMax-H3's prompt encoder: token ids → the
//! unnormalized hidden state after layer 50.
//!
//! The conditioning checkpoint is the 32 B model truncated at layer 50,
//! and the DiT consumes that stream directly — no final norm, no LM
//! head, and no chat template either: the H3 presentation is raw prompt
//! text with no special tokens at all.
//!
//! Same shape as the Gemma encoder next door and deliberately as plain:
//! a full-sequence causal forward, no KV cache, no sampling. Qwen3
//! specifics carried exactly — per-head RMSNorm on q and k BEFORE the
//! rotation, GQA 64/8, split-half RoPE at θ=5e6, SwiGLU, plain-w
//! RMSNorm (not Gemma's 1+w) and no embedding scale.
//!
//! The rotation is Qwen3-VL's interleaved MRoPE. A text token's three
//! axis positions are equal, so every frequency slot sees the same
//! angle and the interleave is invisible; an image span pins the time
//! axis and lays its merged grid out on the other two, and the tokens
//! after it resume from the grid's larger side rather than from the
//! span's length. `encode` is the text-only entry point and
//! `encode_with_images` the general one.
//!
//! Deepstack features go in at LM layers 0, 1, 2 — the FIRST layers.
//! `deepstack_visual_indexes` (8, 16, 24) names the vision layers the
//! features are taken FROM, which is a different list and an easy one
//! to conflate.

use crate::dit::Proj;
use crate::pool::Pool;
use crate::qtensor::QTensor;
use cortiq_core::CmfModel;
use std::sync::Arc;

struct Layer {
    input_norm: Vec<f32>,
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    q_norm: Vec<f32>, // [head_dim]
    k_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    gate: Proj,
    up: Proj,
    down: Proj,
}

pub struct Qwen3Encoder {
    embed: QTensor,
    layers: Vec<Layer>,
    final_norm: Option<Vec<f32>>,
    proj: Option<ClipProj>,
    /// Optional vision-row twin (`te.proj.vis.*`): a projection fitted
    /// on VISION activations, routed to image-span rows only. The base
    /// projection was fitted on text and holds R^2 0.65 there while
    /// actively wrong on vision (-0.57); one affine map cannot serve
    /// both distributions — measured, 2026-08-15.
    proj_vis: Option<ClipProj>,
    pool: Option<Arc<Pool>>,
    pub hidden: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    theta: f32,
    eps: f64,
}

fn rms_norm_into(x: &[f32], w: &[f32], eps: f64, dst: &mut [f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for ((d, &v), &g) in dst.iter_mut().zip(x).zip(w) {
        *d = (v as f64 * inv) as f32 * g;
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// SAFETY: caller guarantees disjoint `[off, off+len)` per worker.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

/// Gauss error function, Abramowitz & Stegun 7.1.26 (|ε| < 1.5e-7).
/// `nn.GELU()` with no `approximate=` is the erf form, and the residual
/// this feeds was fitted through exactly that.
fn erf(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    s * y
}

fn gelu(v: f32) -> f32 {
    (0.5 * v as f64 * (1.0 + erf(v as f64 / std::f64::consts::SQRT_2))) as f32
}

/// The GELU residual of a `-mlp` ClipProj: `d_in → hidden → d_out`,
/// added to the ridge matrix's output in the STANDARDIZED space.
struct ProjMlp {
    w0: Vec<f32>, // [hidden, d_in], torch Linear layout
    b0: Vec<f32>,
    w2: Vec<f32>, // [d_out, hidden]
    b2: Vec<f32>,
    hidden: usize,
}

/// ClipProj: the fitted map that lets a SMALL Qwen3-VL stand in for the
/// 32 B prompt encoder.
///
/// ```text
/// cond = ((h - mean_in) / std_in) @ W [+ mlp(...)] * std_out + mean_out
/// ```
///
/// Token 0 is not projected but OVERWRITTEN with `sink_out`: it is the
/// attention sink, an outlier the ridge fit cannot represent and would
/// otherwise smear across the whole conditioning.
struct ClipProj {
    w: Vec<f32>, // [d_in, d_out], so the product is a plain row sweep
    mean_in: Vec<f32>,
    std_in: Vec<f32>,
    mean_out: Vec<f32>,
    std_out: Vec<f32>,
    sink_out: Vec<f32>,
    mlp: Option<ProjMlp>,
    d_in: usize,
    d_out: usize,
}

impl ClipProj {
    /// `[n, d_in]` tapped hidden state → `[n, d_out]` conditioning.
    fn apply(&self, h: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        self.apply_inner(h, n, pool, true)
    }

    /// `sink=false` skips the token-0 overwrite — for span-gathered
    /// rows, whose first row is NOT the sequence's attention sink.
    fn apply_inner(&self, h: &[f32], n: usize, pool: Option<&Pool>, sink: bool) -> Vec<f32> {
        let (di, dout) = (self.d_in, self.d_out);
        let mut out = vec![0f32; n * dout];
        let ptr = SendPtr(out.as_mut_ptr());
        let work = |lo: usize, hi: usize| {
            let mut xn = vec![0f32; di];
            let mut g = vec![0f32; self.mlp.as_ref().map_or(0, |m| m.hidden)];
            for p in lo..hi {
                let hp = &h[p * di..(p + 1) * di];
                for i in 0..di {
                    xn[i] = (hp[i] - self.mean_in[i]) / self.std_in[i];
                }
                // SAFETY: workers own disjoint token ranges.
                let o = unsafe { ptr.row(p * dout, dout) };
                o.fill(0.0);
                for i in 0..di {
                    let x = xn[i];
                    let row = &self.w[i * dout..(i + 1) * dout];
                    for (d, &wv) in o.iter_mut().zip(row) {
                        *d += x * wv;
                    }
                }
                if let Some(m) = &self.mlp {
                    for (oi, gv) in g.iter_mut().enumerate() {
                        let row = &m.w0[oi * di..(oi + 1) * di];
                        let mut s = m.b0[oi];
                        for (&r, &x) in row.iter().zip(&xn) {
                            s += r * x;
                        }
                        *gv = gelu(s);
                    }
                    for (j, d) in o.iter_mut().enumerate() {
                        let row = &m.w2[j * m.hidden..(j + 1) * m.hidden];
                        let mut s = m.b2[j];
                        for (&r, &gv) in row.iter().zip(&g) {
                            s += r * gv;
                        }
                        *d += s;
                    }
                }
                for (j, d) in o.iter_mut().enumerate() {
                    *d = *d * self.std_out[j] + self.mean_out[j];
                }
            }
        };
        match pool {
            Some(pl) => pl.run_rows(n, &work),
            None => work(0, n),
        }
        if sink && n > 0 {
            out[..dout].copy_from_slice(&self.sink_out);
        }
        out
    }
}

/// One image spliced into the prompt: where its tokens start, how many
/// there are, and the merged patch grid they came from.
pub struct ImageSpan {
    pub start: usize,
    pub len: usize,
    pub merged_h: usize,
    pub merged_w: usize,
}

impl Qwen3Encoder {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model
                .tensor_bytes("te.config_json")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("te.config_json: {e}"))?;
        let u = |k: &str, d: usize| cfg[k].as_u64().map(|v| v as usize).unwrap_or(d);
        // `CMF_TE_TAP` runs FEWER layers than the file carries. The tap
        // a ClipProj was fitted on is an index, and index conventions
        // differ by one between frameworks; packing one layer spare and
        // calibrating against the teacher beats repacking to find out.
        let nl = match std::env::var("CMF_TE_TAP")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            Some(t) if t > 0 && t <= u("num_hidden_layers", 0) => t,
            _ => u("num_hidden_layers", 0),
        };
        let f32v = |n: &str| crate::dit::cmf_f32(model, n);
        let mut layers = Vec::with_capacity(nl);
        for l in 0..nl {
            let p = format!("te.layers.{l}");
            layers.push(Layer {
                input_norm: f32v(&format!("{p}.input_layernorm.weight"))?,
                q: Proj::from_model(model, &format!("{p}.self_attn.q_proj.weight"))?,
                k: Proj::from_model(model, &format!("{p}.self_attn.k_proj.weight"))?,
                v: Proj::from_model(model, &format!("{p}.self_attn.v_proj.weight"))?,
                o: Proj::from_model(model, &format!("{p}.self_attn.o_proj.weight"))?,
                q_norm: f32v(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: f32v(&format!("{p}.self_attn.k_norm.weight"))?,
                post_attn_norm: f32v(&format!("{p}.post_attention_layernorm.weight"))?,
                gate: Proj::from_model(model, &format!("{p}.mlp.gate_proj.weight"))?,
                up: Proj::from_model(model, &format!("{p}.mlp.up_proj.weight"))?,
                down: Proj::from_model(model, &format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        let hidden = u("hidden_size", 0);
        let proj = match model.tensor_bytes("te.proj.config_json") {
            Ok(b) => {
                let pc: serde_json::Value =
                    serde_json::from_slice(b).map_err(|e| format!("te.proj.config_json: {e}"))?;
                let pu = |k: &str| pc[k].as_u64().unwrap_or(0) as usize;
                let (d_in, d_out) = (pu("d_in"), pu("d_out"));
                if d_in != hidden {
                    return Err(format!(
                        "te.proj expects a {d_in}-wide encoder, the packed one is {hidden}: \
                         the projection and the encoder come from different models"
                    ));
                }
                let mlp = match pc["mlp"].as_bool() {
                    Some(true) => Some(ProjMlp {
                        w0: f32v("te.proj.mlp.0.weight")?,
                        b0: f32v("te.proj.mlp.0.bias")?,
                        w2: f32v("te.proj.mlp.2.weight")?,
                        b2: f32v("te.proj.mlp.2.bias")?,
                        hidden: pu("mlp_hidden"),
                    }),
                    _ => None,
                };
                Some(ClipProj {
                    w: f32v("te.proj.W")?,
                    mean_in: f32v("te.proj.mean_in")?,
                    std_in: f32v("te.proj.std_in")?,
                    mean_out: f32v("te.proj.mean_out")?,
                    std_out: f32v("te.proj.std_out")?,
                    sink_out: f32v("te.proj.sink_out")?,
                    mlp,
                    d_in,
                    d_out,
                })
            }
            Err(_) => None,
        };
        let proj_vis = match (&proj, model.tensor_bytes("te.proj.vis.W").is_ok()) {
            (Some(base), true) => {
                let g = |n: &str| crate::dit::cmf_f32(model, n);
                Some(ClipProj {
                    w: g("te.proj.vis.W")?,
                    mean_in: g("te.proj.vis.mean_in")?,
                    std_in: g("te.proj.vis.std_in")?,
                    mean_out: g("te.proj.vis.mean_out")?,
                    std_out: g("te.proj.vis.std_out")?,
                    // vision rows never include token 0; the base sink is
                    // authoritative. Kept for struct uniformity.
                    sink_out: base.sink_out.clone(),
                    mlp: None,
                    d_in: base.d_in,
                    d_out: base.d_out,
                })
            }
            _ => None,
        };
        Ok(Self {
            embed: QTensor::from_model(model, "te.embed_tokens.weight")?,
            layers,
            final_norm: match cfg["final_norm"].as_bool() {
                Some(false) | None => None,
                Some(true) => Some(f32v("te.norm.weight")?),
            },
            proj,
            proj_vis,
            pool: Pool::from_env(),
            hidden,
            nh: u("num_attention_heads", 0),
            nkv: u("num_key_value_heads", 0),
            hd: u("head_dim", 128),
            theta: cfg["rope_theta"].as_f64().unwrap_or(5e6) as f32,
            eps: cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        })
    }

    /// Per-head RMSNorm then split-half RoPE over the whole head, in
    /// place across a `[n, heads·hd]` buffer.
    fn norm_rope(&self, all: &mut [f32], n: usize, heads: usize, w: &[f32], pos: &[[f32; 3]]) {
        let hd = self.hd;
        let half = hd / 2;
        let pool = self.pool.as_deref();
        let ptr = SendPtr(all.as_mut_ptr());
        let work = |lo: usize, hi: usize| {
            for p in lo..hi {
                for h in 0..heads {
                    // SAFETY: workers own disjoint token ranges.
                    let x = unsafe { ptr.row((p * heads + h) * hd, hd) };
                    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hd as f64;
                    let inv = 1.0 / (ss + self.eps).sqrt();
                    for (d, &g) in x.iter_mut().zip(w) {
                        *d = (*d as f64 * inv) as f32 * g;
                    }
                    for i in 0..half {
                        let freq = 1.0 / self.theta.powf(2.0 * i as f32 / hd as f32);
                        // Qwen3-VL's interleaved MRoPE: the time axis by
                        // default, with height and width taking every
                        // third slot below 3·rope_dims. All three axes
                        // carry the same value on a text token, so this
                        // is `p` there whatever the slot.
                        let axis = if i < 60 && i % 3 == 1 {
                            1
                        } else if i < 60 && i % 3 == 2 {
                            2
                        } else {
                            0
                        };
                        let (s, c) = (pos[p][axis] * freq).sin_cos();
                        let (a, b) = (x[i], x[i + half]);
                        x[i] = a * c - b * s;
                        x[i + half] = a * s + b * c;
                    }
                }
            }
        };
        match pool {
            Some(pl) => pl.run_rows(n, &work),
            None => work(0, n),
        }
    }

    /// Causal full-sequence forward. Returns `[n, hidden]` — the
    /// residual stream leaving the last layer, normed only if the
    /// config says the checkpoint has a final norm (H3's does not).
    pub fn encode(&self, ids: &[u32]) -> Vec<f32> {
        self.encode_with_images(ids, &[], &[], &[])
    }

    /// The 3-axis MRoPE position of every token.
    ///
    /// Text runs sequentially on all three axes; an image span pins the
    /// time axis and lays its merged grid out on the other two, and
    /// everything after it resumes from the grid's larger side rather
    /// than from the span's length. With no image the three axes are
    /// equal and this reduces to `0..n` — which is why the text-only
    /// path can ignore the interleave entirely.
    fn mrope_positions(&self, n: usize, spans: &[ImageSpan]) -> Vec<[f32; 3]> {
        let mut pos: Vec<[f32; 3]> = (0..n).map(|i| [i as f32; 3]).collect();
        let mut offset = 0i64;
        for sp in spans {
            let (start, end) = (sp.start, sp.start + sp.len);
            let len_max = sp.merged_h.max(sp.merged_w) as i64;
            let base = start as i64 + offset;
            for i in start..end {
                let k = i - start;
                pos[i] = [
                    base as f32,
                    (base + (k / sp.merged_w) as i64) as f32,
                    (base + (k % sp.merged_w) as i64) as f32,
                ];
            }
            let next = len_max + start as i64;
            for (j, p) in pos.iter_mut().enumerate().skip(end) {
                let v = (next + offset + (j - end) as i64) as f32;
                *p = [v; 3];
            }
            offset += len_max - sp.len as i64;
        }
        pos
    }

    /// Full forward with images. `embeds` replaces the token embedding
    /// at `[span.start, span.start + span.len)` with the vision tower's
    /// merged tokens; `deepstack[k]` is added at the visual positions
    /// after LM layer k — the first layers, not the vision layers the
    /// features were taken from.
    pub fn encode_with_images(
        &self,
        ids: &[u32],
        spans: &[ImageSpan],
        embeds: &[Vec<f32>],
        deepstack: &[Vec<f32>],
    ) -> Vec<f32> {
        let n = ids.len();
        let hs = self.hidden;
        let (nh, nkv, hd) = (self.nh, self.nkv, self.hd);
        let hpk = nh / nkv;
        let pool = self.pool.as_deref();
        let scale = 1.0 / (hd as f32).sqrt();

        let mut h = vec![0f32; n * hs];
        for (i, &id) in ids.iter().enumerate() {
            self.embed
                .row_f32(id as usize, &mut h[i * hs..(i + 1) * hs]);
        }
        for (sp, e) in spans.iter().zip(embeds) {
            h[sp.start * hs..(sp.start + sp.len) * hs].copy_from_slice(&e[..sp.len * hs]);
        }
        let pos = self.mrope_positions(n, spans);
        // Which rows a deepstack feature lands on, in span order.
        let visual: Vec<usize> = spans
            .iter()
            .flat_map(|s| s.start..s.start + s.len)
            .collect();
        let mut xn = vec![0f32; n * hs];
        let mut q_all = vec![0f32; n * nh * hd];
        let mut k_all = vec![0f32; n * nkv * hd];
        let mut v_all = vec![0f32; n * nkv * hd];
        let mut attn = vec![0f32; n * nh * hd];
        let mut proj = vec![0f32; n * hs];

        for (li, layer) in self.layers.iter().enumerate() {
            for (o, src) in xn.chunks_exact_mut(hs).zip(h.chunks_exact(hs)) {
                rms_norm_into(src, &layer.input_norm, self.eps, o);
            }
            layer.q.matmat(&xn, n, &mut q_all, pool);
            layer.k.matmat(&xn, n, &mut k_all, pool);
            layer.v.matmat(&xn, n, &mut v_all, pool);
            self.norm_rope(&mut q_all, n, nh, &layer.q_norm, &pos);
            self.norm_rope(&mut k_all, n, nkv, &layer.k_norm, &pos);

            attn.fill(0.0);
            let ap = SendPtr(attn.as_mut_ptr());
            let heads = |lo: usize, hi: usize| {
                let mut row = vec![0f32; n];
                for hh in lo..hi {
                    let kv = hh / hpk;
                    for p in 0..n {
                        let qv = &q_all[(p * nh + hh) * hd..(p * nh + hh + 1) * hd];
                        for (j, r) in row[..=p].iter_mut().enumerate() {
                            let kvv = &k_all[(j * nkv + kv) * hd..(j * nkv + kv + 1) * hd];
                            *r = qv.iter().zip(kvv).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                        }
                        let mx = row[..=p].iter().cloned().fold(f32::MIN, f32::max);
                        let mut den = 0f32;
                        for r in row[..=p].iter_mut() {
                            *r = (*r - mx).exp();
                            den += *r;
                        }
                        let inv = 1.0 / den;
                        // SAFETY: workers own disjoint head ranges, and
                        // one head's slice is disjoint per token.
                        let out = unsafe { ap.row((p * nh + hh) * hd, hd) };
                        for (j, &rw) in row[..=p].iter().enumerate() {
                            let vv = &v_all[(j * nkv + kv) * hd..(j * nkv + kv + 1) * hd];
                            let wgt = rw * inv;
                            for (o, &s) in out.iter_mut().zip(vv) {
                                *o += wgt * s;
                            }
                        }
                    }
                }
            };
            match pool {
                Some(pl) => pl.run_rows(nh, &heads),
                None => heads(0, nh),
            }

            layer.o.matmat(&attn, n, &mut proj, pool);
            for (d, &v) in h.iter_mut().zip(&proj) {
                *d += v;
            }

            for (o, src) in xn.chunks_exact_mut(hs).zip(h.chunks_exact(hs)) {
                rms_norm_into(src, &layer.post_attn_norm, self.eps, o);
            }
            let inter = layer.gate.rows();
            let mut g = vec![0f32; n * inter];
            let mut u = vec![0f32; n * inter];
            layer.gate.matmat(&xn, n, &mut g, pool);
            layer.up.matmat(&xn, n, &mut u, pool);
            for (a, &b) in g.iter_mut().zip(&u) {
                *a = silu(*a) * b;
            }
            layer.down.matmat(&g, n, &mut proj, pool);
            for (d, &v) in h.iter_mut().zip(&proj) {
                *d += v;
            }
            if let Some(f) = deepstack.get(li) {
                for (k, &row) in visual.iter().enumerate() {
                    for c in 0..hs {
                        h[row * hs + c] += f[k * hs + c];
                    }
                }
            }
        }
        if let Some(w) = &self.final_norm {
            let mut out = vec![0f32; n * hs];
            for (o, src) in out.chunks_exact_mut(hs).zip(h.chunks_exact(hs)) {
                rms_norm_into(src, w, self.eps, o);
            }
            return out;
        }
        // A ClipProj file makes this a STAND-IN encoder: the stream
        // leaving the tap is in the small model's space and the DiT
        // never sees it raw.
        // `CMF_CP_DUMP=<path>`: the RAW tapped hidden state, before the
        // projection — one side of a (student, teacher) pair a refit
        // regresses on. Same [u64 n][u64 width] + f32 rows framing as
        // CMF_TE_DUMP, so one reader serves both.
        if let (Ok(path), Some(_)) = (std::env::var("CMF_CP_DUMP"), &self.proj) {
            let mut b = Vec::with_capacity(16 + h.len() * 4);
            b.extend_from_slice(&(n as u64).to_le_bytes());
            b.extend_from_slice(&(self.hidden as u64).to_le_bytes());
            for v in &h {
                b.extend_from_slice(&v.to_le_bytes());
            }
            if let Err(e) = std::fs::write(&path, &b) {
                tracing::warn!("CMF_CP_DUMP {path}: {e}");
            } else {
                eprintln!("cp dump: {n} tokens x {} -> {path}", self.hidden);
            }
        }
        match &self.proj {
            Some(p) => {
                let mut out = p.apply(&h, n, self.pool.as_deref());
                // Vision rows go through their own map when the file
                // ships one: gather span rows, project, scatter back.
                if let Some(pv) = &self.proj_vis {
                    if !visual.is_empty() {
                        let (din, dout) = (pv.d_in, pv.d_out);
                        let mut hv = Vec::with_capacity(visual.len() * din);
                        for &r in &visual {
                            hv.extend_from_slice(&h[r * din..(r + 1) * din]);
                        }
                        let ov = pv.apply_inner(&hv, visual.len(), self.pool.as_deref(), false);
                        for (i, &r) in visual.iter().enumerate() {
                            out[r * dout..(r + 1) * dout]
                                .copy_from_slice(&ov[i * dout..(i + 1) * dout]);
                        }
                    }
                }
                out
            }
            None => h,
        }
    }

    /// Width of what `encode` returns — the DiT's conditioning width,
    /// which is the projection's output when one is packed and the
    /// encoder's own hidden size otherwise.
    pub fn out_hidden(&self) -> usize {
        self.proj.as_ref().map_or(self.hidden, |p| p.d_out)
    }
}
