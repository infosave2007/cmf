//! MiniMax-Music-3's flow-matching DiT: latent + conditioning → velocity.
//!
//! Ported against ComfyUI's `comfy/ldm/minimax_music/dit.py`. Four of its
//! conventions are worth naming here, because each is invisible in a
//! tensor name and wrong in a way that still produces plausible output:
//!
//! 1. The input is `[x | zeros_like(x) | condition]` on the CHANNEL axis
//!    — 128 + 128 + 2048 = 2304, which is what `preprocess_conv`'s width
//!    is telling you. The zero plane is not padding, it is a slot the
//!    reference leaves empty.
//! 2. Both 1×1 convs are RESIDUAL: `conv(x) + x`, not `conv(x)`.
//! 3. The timestep embedding is prepended as an extra TOKEN, carried
//!    through all 36 blocks, and dropped before `project_out`. It also
//!    shifts every latent frame's rotary position by one.
//! 4. The transformer's output is NEGATED. A flow-matching sampler that
//!    steps the wrong way still moves, and still decodes to sound.
//!
//! RoPE covers the first 32 of each head's 64 dims, split-half: for
//! `i < 16` the pair is `(x[i], x[i+16])`, rotated by `pos·inv_freq[i]`.

use crate::dit::Proj;
use crate::qtensor::QTensor;
use crate::audiovae::SendPtr;
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// LayerNorm with learned scale AND shift — not the RMSNorm the rest of
/// this engine's transformers use.
struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl LayerNorm {
    fn load(model: &Arc<CmfModel>, p: &str) -> Result<Self, String> {
        Ok(Self {
            gamma: crate::dit::cmf_f32(model, &format!("{p}.gamma"))?,
            beta: crate::dit::cmf_f32(model, &format!("{p}.beta"))?,
        })
    }

    fn apply(&self, x: &mut [f32], d: usize) {
        for row in x.chunks_exact_mut(d) {
            let mean = row.iter().map(|v| *v as f64).sum::<f64>() / d as f64;
            let var = row.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for ((v, &g), &b) in row.iter_mut().zip(&self.gamma).zip(&self.beta) {
                *v = ((*v as f64 - mean) * inv) as f32 * g + b;
            }
        }
    }
}

struct Block {
    pre_norm: LayerNorm,
    qkv: Proj,
    out: Proj,
    ff_norm: LayerNorm,
    ff_in: Proj,
    ff_in_b: Vec<f32>,
    ff_out: Proj,
    ff_out_b: Vec<f32>,
}

impl Block {
    fn load(model: &Arc<CmfModel>, p: &str) -> Result<Self, String> {
        Ok(Self {
            pre_norm: LayerNorm::load(model, &format!("{p}.pre_norm"))?,
            qkv: Proj::from_model(model, &format!("{p}.self_attn.to_qkv.weight"))?,
            out: Proj::from_model(model, &format!("{p}.self_attn.to_out.weight"))?,
            ff_norm: LayerNorm::load(model, &format!("{p}.ff_norm"))?,
            ff_in: Proj::from_model(model, &format!("{p}.ff.ff.0.proj.weight"))?,
            ff_in_b: crate::dit::cmf_f32(model, &format!("{p}.ff.ff.0.proj.bias"))?,
            ff_out: Proj::from_model(model, &format!("{p}.ff.ff.2.weight"))?,
            ff_out_b: crate::dit::cmf_f32(model, &format!("{p}.ff.ff.2.bias"))?,
        })
    }
}

pub struct Music3Dit {
    pre_conv: Vec<f32>,  // [2304, 2304] 1x1
    post_conv: Vec<f32>, // [128, 128] 1x1
    fourier: Vec<f32>,   // [128]
    t0: Proj,
    t0_b: Vec<f32>,
    t2: Proj,
    t2_b: Vec<f32>,
    project_in: Proj,
    project_out: Proj,
    inv_freq: Vec<f32>,
    blocks: Vec<Block>,
    pool: Option<Arc<Pool>>,
    hidden: usize,
    heads: usize,
    hd: usize,
    rot: usize,
    inter: usize,
    /// The learned mix over the AR stack's eight codebook levels, and
    /// the 3-tap conv that turns the mixture into the DiT's condition.
    cond_logits: Vec<f32>,
    cond_scale: f32,
    lc_w: Vec<f32>,
    lc_b: Vec<f32>,
}

impl Music3Dit {
    pub const IN_CH: usize = 128;
    pub const COND_CH: usize = 2048;
    pub const CONCAT_CH: usize = 2304;

    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value =
            serde_json::from_slice(model.tensor_bytes("mdit.config_json").map_err(|e| e.to_string())?)
                .map_err(|e| format!("mdit.config_json: {e}"))?;
        let u = |k: &str, d: usize| cfg[k].as_u64().map(|v| v as usize).unwrap_or(d);
        let nl = u("num_layers", 36);
        let dt = "mdit.diffusion_transformer";
        Ok(Self {
            pre_conv: crate::dit::cmf_f32(model, &format!("{dt}.preprocess_conv.weight"))?,
            post_conv: crate::dit::cmf_f32(model, &format!("{dt}.postprocess_conv.weight"))?,
            fourier: crate::dit::cmf_f32(model, &format!("{dt}.timestep_features.weight"))?,
            t0: Proj::from_model(model, &format!("{dt}.to_timestep_embed.0.weight"))?,
            t0_b: crate::dit::cmf_f32(model, &format!("{dt}.to_timestep_embed.0.bias"))?,
            t2: Proj::from_model(model, &format!("{dt}.to_timestep_embed.2.weight"))?,
            t2_b: crate::dit::cmf_f32(model, &format!("{dt}.to_timestep_embed.2.bias"))?,
            project_in: Proj::from_model(model, &format!("{dt}.transformer.project_in.weight"))?,
            project_out: Proj::from_model(model, &format!("{dt}.transformer.project_out.weight"))?,
            inv_freq: crate::dit::cmf_f32(model, &format!("{dt}.transformer.rotary_pos_emb.inv_freq"))?,
            blocks: (0..nl)
                .map(|i| Block::load(model, &format!("{dt}.transformer.layers.{i}")))
                .collect::<Result<_, _>>()?,
            pool: Pool::from_env(),
            hidden: u("hidden", 2048),
            heads: u("num_heads", 32),
            hd: u("head_dim", 64),
            rot: u("rotary_dim", 32),
            inter: u("ff_inner", 8192),
            cond_logits: crate::dit::cmf_f32(model, "mdit.cond_layer_logits")?,
            cond_scale: crate::dit::cmf_f32(model, "mdit.cond_layer_scale")?[0],
            lc_w: crate::dit::cmf_f32(model, "mdit.latent_conditioners.0.weight")?,
            lc_b: crate::dit::cmf_f32(model, "mdit.latent_conditioners.0.bias")?,
        })
    }

    /// Latent frames for `audio_frames` of AR output — the reference's
    /// `latent_length`: 44100/24000 · 960/512 = 3.4453125 per frame.
    pub fn latent_length(audio_frames: usize) -> usize {
        ((audio_frames as f64 * 44100.0 / 24000.0 * 960.0 / 512.0) as usize).max(1)
    }

    /// AR hidden `[frames, 8·4096]` → the DiT's condition `[2048, L]`.
    ///
    /// The eight are RVQ CODEBOOK levels, softmax-mixed by
    /// `cond_layer_logits`, scaled, passed through one 3-tap conv and
    /// then NEAREST-resampled to the latent rate. Nearest, not linear:
    /// the reference interpolates that way and a smoother resample
    /// smears the onset of every note.
    pub fn aligned_condition(&self, hidden: &[f32], frames: usize) -> (Vec<f32>, usize) {
        let levels = self.cond_logits.len();
        let ar = hidden.len() / (frames * levels);
        let mx = self.cond_logits.iter().cloned().fold(f32::MIN, f32::max);
        let ex: Vec<f32> = self.cond_logits.iter().map(|v| (v - mx).exp()).collect();
        let sum: f32 = ex.iter().sum();
        // [ar, frames], channel-major, mixed and scaled.
        let mut mixed = vec![0f32; ar * frames];
        for t in 0..frames {
            for l in 0..levels {
                let w = ex[l] / sum * self.cond_scale;
                let src = &hidden[(t * levels + l) * ar..(t * levels + l + 1) * ar];
                for (c, &v) in src.iter().enumerate() {
                    mixed[c * frames + t] += w * v;
                }
            }
        }
        // Conv1d(ar -> 2048, k=3, pad=1).
        let out_ch = self.lc_b.len();
        let mut conv = vec![0f32; out_ch * frames];
        for o in 0..out_ch {
            let dst = &mut conv[o * frames..(o + 1) * frames];
            dst.fill(self.lc_b[o]);
            for i in 0..ar {
                let k = &self.lc_w[(o * ar + i) * 3..(o * ar + i + 1) * 3];
                let src = &mixed[i * frames..(i + 1) * frames];
                for t in 0..frames {
                    for (j, &kv) in k.iter().enumerate() {
                        let p = t as isize + j as isize - 1;
                        if p >= 0 && (p as usize) < frames {
                            dst[t] += kv * src[p as usize];
                        }
                    }
                }
            }
        }
        let l = Self::latent_length(frames);
        let mut out = vec![0f32; out_ch * l];
        for o in 0..out_ch {
            for t in 0..l {
                let s = (t * frames) / l.max(1);
                out[o * l + t] = conv[o * frames + s.min(frames - 1)];
            }
        }
        (out, l)
    }

    /// `[out, in]` 1×1 conv over the channel axis of a `[in, n]` panel,
    /// added back to its input — both convs in this model are residual.
    fn conv1x1_residual(w: &[f32], x: &mut [f32], ch: usize, n: usize) {
        let mut acc = vec![0f32; ch * n];
        for o in 0..ch {
            let row = &w[o * ch..(o + 1) * ch];
            let dst = &mut acc[o * n..(o + 1) * n];
            for (i, &wv) in row.iter().enumerate() {
                if wv == 0.0 {
                    continue;
                }
                for (d, &s) in dst.iter_mut().zip(&x[i * n..(i + 1) * n]) {
                    *d += wv * s;
                }
            }
        }
        for (d, a) in x.iter_mut().zip(&acc) {
            *d += a;
        }
    }

    /// `cat(cos, sin)` of `2π·t·w`, then the two-layer SiLU head.
    fn timestep_embedding(&self, t: f32) -> Vec<f32> {
        let half = self.fourier.len();
        let mut feats = vec![0f32; half * 2];
        for (i, &w) in self.fourier.iter().enumerate() {
            let a = std::f32::consts::TAU * t * w;
            feats[i] = a.cos();
            feats[half + i] = a.sin();
        }
        let mut h = vec![0f32; self.t0_b.len()];
        self.t0.matmat(&feats, 1, &mut h, self.pool.as_deref());
        for (v, &b) in h.iter_mut().zip(&self.t0_b) {
            *v = silu(*v + b);
        }
        let mut out = vec![0f32; self.t2_b.len()];
        self.t2.matmat(&h, 1, &mut out, self.pool.as_deref());
        for (v, &b) in out.iter_mut().zip(&self.t2_b) {
            *v += b;
        }
        out
    }

    /// Split-half RoPE over the first `rot` dims of every head.
    fn rope(&self, q: &mut [f32], k: &mut [f32], n: usize) {
        let half = self.rot / 2;
        for p in 0..n {
            for h in 0..self.heads {
                let off = p * self.heads * self.hd + h * self.hd;
                for i in 0..half {
                    let (s, c) = (p as f32 * self.inv_freq[i]).sin_cos();
                    for x in [&mut *q, &mut *k] {
                        let (a, b) = (x[off + i], x[off + i + half]);
                        x[off + i] = a * c - b * s;
                        x[off + i + half] = a * s + b * c;
                    }
                }
            }
        }
    }

    /// One block over `[n, hidden]`, in place.
    fn block(&self, blk: &Block, x: &mut [f32], n: usize) {
        let (hs, nh, hd) = (self.hidden, self.heads, self.hd);
        let pool = self.pool.as_deref();
        let mut h = x.to_vec();
        blk.pre_norm.apply(&mut h, hs);
        let mut qkv = vec![0f32; n * 3 * hs];
        blk.qkv.matmat(&h, n, &mut qkv, pool);
        // to_qkv emits q|k|v concatenated along the FEATURE axis, so the
        // three live at column offsets, not row offsets.
        let mut q = vec![0f32; n * hs];
        let mut k = vec![0f32; n * hs];
        let mut v = vec![0f32; n * hs];
        for p in 0..n {
            let s = &qkv[p * 3 * hs..(p + 1) * 3 * hs];
            q[p * hs..(p + 1) * hs].copy_from_slice(&s[..hs]);
            k[p * hs..(p + 1) * hs].copy_from_slice(&s[hs..2 * hs]);
            v[p * hs..(p + 1) * hs].copy_from_slice(&s[2 * hs..]);
        }
        self.rope(&mut q, &mut k, n);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut attn = vec![0f32; n * hs];
        // The device first. This is the quadratic part and the reason a
        // CPU-only run was beating a GPU one: everything around it went
        // to the card and the biggest single stage stayed home. The
        // engine's kernel wants HEAD-major panels where the projections
        // leave them token-major, and three transposes of n x hs are
        // noise against n^2 work.
        if crate::gpu::enabled_here() {
            let mut qh = vec![0f32; n * hs];
            let mut kh = vec![0f32; n * hs];
            let mut vh = vec![0f32; n * hs];
            for h in 0..nh {
                for i in 0..n {
                    let (s, d) = (i * hs + h * hd, (h * n + i) * hd);
                    qh[d..d + hd].copy_from_slice(&q[s..s + hd]);
                    kh[d..d + hd].copy_from_slice(&k[s..s + hd]);
                    vh[d..d + hd].copy_from_slice(&v[s..s + hd]);
                }
            }
            if crate::gpu::dit_attention(&qh, &kh, &vh, nh, nh, n, hd, scale, &mut attn) {
                let mut proj = vec![0f32; n * hs];
                blk.out.matmat(&attn, n, &mut proj, pool);
                for (a, b) in x.iter_mut().zip(&proj) {
                    *a += b;
                }
                return self.block_ffn(blk, x, n);
            }
            attn.fill(0.0);
        }
        // Heads are independent and write disjoint columns, so the host
        // arm splits without a lock.
        {
            let ptr = SendPtr(attn.as_mut_ptr());
            let work = |lo: usize, hi: usize| {
                let mut scores = vec![0f32; n];
                for hh in lo..hi {
                    for i in 0..n {
                        let qi = &q[i * hs + hh * hd..i * hs + hh * hd + hd];
                        let mut mx = f32::NEG_INFINITY;
                        for (j, sc) in scores.iter_mut().enumerate() {
                            let kj = &k[j * hs + hh * hd..j * hs + hh * hd + hd];
                            *sc = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                            mx = mx.max(*sc);
                        }
                        let mut sum = 0.0;
                        for sc in scores.iter_mut() {
                            *sc = (*sc - mx).exp();
                            sum += *sc;
                        }
                        let inv = 1.0 / sum;
                        // SAFETY: head `hh` owns these columns of every row.
                        let dst = unsafe { ptr.row(i * hs + hh * hd, hd) };
                        for (j, &sc) in scores.iter().enumerate() {
                            let w = sc * inv;
                            let vj = &v[j * hs + hh * hd..j * hs + hh * hd + hd];
                            for (d, &vv) in dst.iter_mut().zip(vj) {
                                *d += w * vv;
                            }
                        }
                    }
                }
            };
            match pool {
                Some(p) => p.run_rows(nh, &work),
                None => work(0, nh),
            }
        }
        let mut proj = vec![0f32; n * hs];
        blk.out.matmat(&attn, n, &mut proj, pool);
        for (a, b) in x.iter_mut().zip(&proj) {
            *a += b;
        }
        self.block_ffn(blk, x, n)
    }

    /// The second half of a block: norm, GEGLU, residual.
    fn block_ffn(&self, blk: &Block, x: &mut [f32], n: usize) {
        let hs = self.hidden;
        let pool = self.pool.as_deref();
        let mut h = x.to_vec();
        blk.ff_norm.apply(&mut h, hs);
        let mut gu = vec![0f32; n * 2 * self.inter];
        blk.ff_in.matmat(&h, n, &mut gu, pool);
        // GLU here is `value * silu(gate)` with VALUE first, and the
        // projection's bias belongs to BOTH halves before they meet.
        // Swapping the halves still makes sound, which is why this is
        // spelled out rather than inferred.
        let inter = self.inter;
        let (vb, gb) = blk.ff_in_b.split_at(inter);
        let mut act = vec![0f32; n * inter];
        for p in 0..n {
            let row = &gu[p * 2 * inter..(p + 1) * 2 * inter];
            let (val, gate) = row.split_at(inter);
            let dst = &mut act[p * inter..(p + 1) * inter];
            for i in 0..inter {
                dst[i] = (val[i] + vb[i]) * silu(gate[i] + gb[i]);
            }
        }
        let mut ffo = vec![0f32; n * hs];
        blk.ff_out.matmat(&act, n, &mut ffo, pool);
        for p in 0..n {
            for j in 0..hs {
                x[p * hs + j] += ffo[p * hs + j] + blk.ff_out_b[j];
            }
        }
    }

    /// Latent frames the transformer will attend across in one go, and
    /// the stride it advances by — `latent_length(200)` and
    /// `latent_length(100)` in the reference. Attention is quadratic, so
    /// a whole song in one pass is not merely slow, it is not what the
    /// model was run as.
    pub const WINDOW: usize = 689;
    pub const HOP: usize = 344;

    /// The velocity over any length, windowed like the reference:
    /// overlapping passes averaged by how many covered each frame.
    pub fn forward_windowed(&self, x: &[f32], condition: &[f32], n: usize, t: f32) -> Vec<f32> {
        if n <= Self::WINDOW {
            return self.forward(x, condition, n, t);
        }
        let ch = Self::IN_CH;
        let cc = Self::COND_CH;
        let mut out = vec![0f32; ch * n];
        let mut count = vec![0f32; n];
        let mut start = 0usize;
        loop {
            let end = (start + Self::WINDOW).min(n);
            let w = end - start;
            let mut xw = vec![0f32; ch * w];
            for c in 0..ch {
                xw[c * w..(c + 1) * w].copy_from_slice(&x[c * n + start..c * n + end]);
            }
            let mut cw = vec![0f32; cc * w];
            for c in 0..cc {
                cw[c * w..(c + 1) * w].copy_from_slice(&condition[c * n + start..c * n + end]);
            }
            let v = self.forward(&xw, &cw, w, t);
            for c in 0..ch {
                for i in 0..w {
                    out[c * n + start + i] += v[c * w + i];
                }
            }
            for i in 0..w {
                count[start + i] += 1.0;
            }
            if end == n {
                break;
            }
            start += Self::HOP;
        }
        for c in 0..ch {
            for i in 0..n {
                out[c * n + i] /= count[i];
            }
        }
        out
    }

    /// `x` is `[128, n]` and `condition` `[2048, n]`; the result is the
    /// velocity at `[128, n]`.
    pub fn forward(&self, x: &[f32], condition: &[f32], n: usize, t: f32) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let mut full = vec![0f32; Self::CONCAT_CH * n];
        full[..Self::IN_CH * n].copy_from_slice(x);
        // rows 128..256 stay zero: the reference's `zeros_like(x)` plane
        full[2 * Self::IN_CH * n..].copy_from_slice(condition);
        Self::conv1x1_residual(&self.pre_conv, &mut full, Self::CONCAT_CH, n);

        // channel-major -> token-major for the transformer
        let mut toks = vec![0f32; n * Self::CONCAT_CH];
        for c in 0..Self::CONCAT_CH {
            for p in 0..n {
                toks[p * Self::CONCAT_CH + c] = full[c * n + p];
            }
        }
        let mut h = vec![0f32; n * self.hidden];
        self.project_in.matmat(&toks, n, &mut h, pool);

        // The timestep rides as token 0 and shifts every rotary position.
        let temb = self.timestep_embedding(t);
        let mut seq = vec![0f32; (n + 1) * self.hidden];
        seq[..self.hidden].copy_from_slice(&temb);
        seq[self.hidden..].copy_from_slice(&h);
        for blk in &self.blocks {
            self.block(blk, &mut seq, n + 1);
        }

        let mut out = vec![0f32; n * Self::IN_CH];
        self.project_out
            .matmat(&seq[self.hidden..], n, &mut out, pool);
        let mut ch = vec![0f32; Self::IN_CH * n];
        for c in 0..Self::IN_CH {
            for p in 0..n {
                ch[c * n + p] = out[p * Self::IN_CH + c];
            }
        }
        Self::conv1x1_residual(&self.post_conv, &mut ch, Self::IN_CH, n);
        for v in ch.iter_mut() {
            *v = -*v;
        }
        ch
    }

    /// Denoise `[128, n]` from noise to a latent, `steps` Euler steps
    /// along σ: 1 → 0. `progress` is called with (step, total).
    pub fn sample(
        &self,
        noise: &[f32],
        condition: &[f32],
        n: usize,
        steps: usize,
        mut progress: impl FnMut(usize, usize),
    ) -> Vec<f32> {
        let sigmas = flow_sigmas(steps);
        let mut x = noise.to_vec();
        for i in 0..steps {
            let (s, s_next) = (sigmas[i], sigmas[i + 1]);
            // ComfyUI's process_timestep for this model.
            let v = self.forward_windowed(&x, condition, n, 1.0 - s);
            let dt = s_next - s;
            for (a, b) in x.iter_mut().zip(&v) {
                *a += dt * b;
            }
            progress(i + 1, steps);
        }
        x
    }
}

/// Euler flow-matching sampler for Music-3.
///
/// ComfyUI registers this model as a plain `ModelType.FLOW` with
/// `multiplier: 1.0` and `process_timestep(t) = 1.0 - t`, so the sampler
/// is the ordinary one and NOT the `FlowMatchEulerDiscreteScheduler`
/// named in MiniMax's own `scheduler_config.json` — that belongs to
/// their diffusers pipeline. Worth stating because the config is the
/// first thing you find and it sends you somewhere else: with
/// `num_train_timesteps: 1` its schedule degenerates to a constant,
/// which is the tell that the caller supplies the sigmas.
///
/// σ walks 1 → 0, the DiT is asked at `1 − σ`, and the step is
/// `x += (σ_next − σ)·v`. The DiT already negates its own output, so
/// the sign lives there rather than here.
///
/// The walk is NOT uniform, and that detail is audible. ComfyUI's
/// `normal_scheduler` evaluates at `linspace(σ_max, σ_min, steps)` and
/// only THEN appends zero, and this model's `ModelSamplingDiscreteFlow`
/// has `σ_min = 1/1000` — so the last velocity is measured essentially
/// at the end of the trajectory. A uniform `1 → 0` in `steps` stops at
/// `1/steps` and integrates the whole remaining tail from a velocity
/// sampled well before it, which is a smeared, mushy final approach.
pub fn flow_sigmas(steps: usize) -> Vec<f32> {
    const SIGMA_MIN: f32 = 0.001;
    let n = steps.max(1);
    let mut s: Vec<f32> = (0..n)
        .map(|i| {
            if n == 1 {
                1.0
            } else {
                1.0 + (SIGMA_MIN - 1.0) * i as f32 / (n - 1) as f32
            }
        })
        .collect();
    s.push(0.0);
    s
}

/// RMSNorm with a weight and no bias, eps 1e-6.
struct RmsNorm {
    w: Vec<f32>,
}

impl RmsNorm {
    fn load(model: &Arc<CmfModel>, n: &str) -> Result<Self, String> {
        Ok(Self {
            w: crate::dit::cmf_f32(model, n)?,
        })
    }

    fn apply(&self, x: &mut [f32], d: usize) {
        for row in x.chunks_exact_mut(d) {
            let ss = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / d as f64;
            let inv = 1.0 / (ss + 1e-6).sqrt();
            for (v, &g) in row.iter_mut().zip(&self.w) {
                *v = (*v as f64 * inv) as f32 * g;
            }
        }
    }
}

struct RvqBlock {
    n1: RmsNorm,
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    n2: RmsNorm,
    gate: Proj,
    up: Proj,
    down: Proj,
}

impl RvqBlock {
    fn load(model: &Arc<CmfModel>, p: &str) -> Result<Self, String> {
        Ok(Self {
            n1: RmsNorm::load(model, &format!("{p}.input_layernorm.weight"))?,
            q: Proj::from_model(model, &format!("{p}.self_attn.q_proj.weight"))?,
            k: Proj::from_model(model, &format!("{p}.self_attn.k_proj.weight"))?,
            v: Proj::from_model(model, &format!("{p}.self_attn.v_proj.weight"))?,
            o: Proj::from_model(model, &format!("{p}.self_attn.o_proj.weight"))?,
            n2: RmsNorm::load(model, &format!("{p}.post_attention_layernorm.weight"))?,
            gate: Proj::from_model(model, &format!("{p}.mlp.gate_proj.weight"))?,
            up: Proj::from_model(model, &format!("{p}.mlp.up_proj.weight"))?,
            down: Proj::from_model(model, &format!("{p}.mlp.down_proj.weight"))?,
        })
    }
}

/// The RVQ depth decoder: given the frame's hidden state and the codes
/// chosen so far, it predicts the next codebook level.
///
/// It is a small CAUSAL transformer over a sequence that never exceeds
/// the codebook count — the positional table is 16 rows for 8 levels —
/// with no rotary embedding at all. Attention that forgets the mask here
/// leaks a level's own answer backwards and the model still samples.
pub struct RvqDepthDecoder {
    projection: Proj,
    pos: Vec<f32>,
    blocks: Vec<RvqBlock>,
    norm: RmsNorm,
    heads: Vec<Proj>,
    pool: Option<Arc<Pool>>,
    hidden: usize,
    nh: usize,
    hd: usize,
    inter: usize,
}

impl RvqDepthDecoder {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value =
            serde_json::from_slice(model.tensor_bytes("mte.config_json").map_err(|e| e.to_string())?)
                .map_err(|e| format!("mte.config_json: {e}"))?;
        let u = |k: &str, d: usize| cfg[k].as_u64().map(|v| v as usize).unwrap_or(d);
        let hidden = u("hidden_size", 4096);
        let nl = u("decoder_num_layers", 4);
        let nh = u("decoder_num_heads", 16);
        let cb = u("audio_num_codebooks", 8);
        Ok(Self {
            projection: Proj::from_model(model, "mte.audio_decoder.projection.weight")?,
            pos: crate::dit::cmf_f32(model, "mte.audio_decoder.pos_embedding.weight")?,
            blocks: (0..nl)
                .map(|i| RvqBlock::load(model, &format!("mte.audio_decoder.layers.{i}")))
                .collect::<Result<_, _>>()?,
            norm: RmsNorm::load(model, "mte.audio_decoder.norm.weight")?,
            heads: (0..cb - 1)
                .map(|i| {
                    Proj::from_model(model, &format!("mte.audio_decoder.audio_heads.{i}.weight"))
                })
                .collect::<Result<_, _>>()?,
            pool: Pool::from_env(),
            hidden,
            nh,
            hd: hidden / nh,
            inter: u("decoder_intermediate_size", 6144),
        })
    }

    pub fn codebooks(&self) -> usize {
        self.heads.len() + 1
    }

    /// `projection` is applied to every element entering the sequence —
    /// the frame hidden, the c0 embedding and each extra embedding.
    pub fn project(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() / self.hidden;
        let mut out = vec![0f32; n * self.hidden];
        self.projection
            .matmat(x, n, &mut out, self.pool.as_deref());
        out
    }

    /// Run the stack over `[n, hidden]` and return the LAST position's
    /// normed hidden — the only one the caller reads.
    pub fn forward_last(&self, seq: &[f32], n: usize) -> Vec<f32> {
        let (hs, nh, hd) = (self.hidden, self.nh, self.hd);
        let pool = self.pool.as_deref();
        let mut x = seq.to_vec();
        for p in 0..n {
            for (v, &pe) in x[p * hs..(p + 1) * hs].iter_mut().zip(&self.pos[p * hs..(p + 1) * hs]) {
                *v += pe;
            }
        }
        for blk in &self.blocks {
            let mut h = x.clone();
            blk.n1.apply(&mut h, hs);
            let (mut q, mut k, mut v) = (vec![0f32; n * hs], vec![0f32; n * hs], vec![0f32; n * hs]);
            blk.q.matmat(&h, n, &mut q, pool);
            blk.k.matmat(&h, n, &mut k, pool);
            blk.v.matmat(&h, n, &mut v, pool);
            let scale = 1.0 / (hd as f32).sqrt();
            let mut attn = vec![0f32; n * hs];
            for hh in 0..nh {
                for i in 0..n {
                    let qi = &q[i * hs + hh * hd..i * hs + hh * hd + hd];
                    // Causal: position i sees 0..=i and nothing after.
                    let mut sc = vec![0f32; i + 1];
                    let mut mx = f32::NEG_INFINITY;
                    for (j, s) in sc.iter_mut().enumerate() {
                        let kj = &k[j * hs + hh * hd..j * hs + hh * hd + hd];
                        *s = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                        mx = mx.max(*s);
                    }
                    let mut sum = 0.0;
                    for s in sc.iter_mut() {
                        *s = (*s - mx).exp();
                        sum += *s;
                    }
                    let inv = 1.0 / sum;
                    let dst = &mut attn[i * hs + hh * hd..i * hs + hh * hd + hd];
                    for (j, &s) in sc.iter().enumerate() {
                        let w = s * inv;
                        let vj = &v[j * hs + hh * hd..j * hs + hh * hd + hd];
                        for (d, &vv) in dst.iter_mut().zip(vj) {
                            *d += w * vv;
                        }
                    }
                }
            }
            let mut proj = vec![0f32; n * hs];
            blk.o.matmat(&attn, n, &mut proj, pool);
            for (a, b) in x.iter_mut().zip(&proj) {
                *a += b;
            }

            let mut h = x.clone();
            blk.n2.apply(&mut h, hs);
            let (mut g, mut u) = (vec![0f32; n * self.inter], vec![0f32; n * self.inter]);
            blk.gate.matmat(&h, n, &mut g, pool);
            blk.up.matmat(&h, n, &mut u, pool);
            for (a, b) in g.iter_mut().zip(&u) {
                *a = silu(*a) * b;
            }
            let mut ffo = vec![0f32; n * hs];
            blk.down.matmat(&g, n, &mut ffo, pool);
            for (a, b) in x.iter_mut().zip(&ffo) {
                *a += b;
            }
        }
        let mut last = x[(n - 1) * hs..].to_vec();
        self.norm.apply(&mut last, hs);
        last
    }

    /// Logits for codebook `level` (1-based; level 1 uses head 0).
    pub fn head(&self, level: usize, hidden: &[f32]) -> Vec<f32> {
        let h = &self.heads[level - 1];
        let rows = match h {
            Proj::F32 { rows, .. } => *rows,
            Proj::Q(q) => q.rows(),
        };
        let mut out = vec![0f32; rows];
        h.matmat(hidden, 1, &mut out, self.pool.as_deref());
        out
    }
}

// ── the autoregressive stack ────────────────────────────────────────

/// One Qwen3 block: RMSNorm → GQA attention with per-head q/k norms and
/// split-half RoPE → residual → RMSNorm → SwiGLU → residual.
struct ArBlock {
    n1: RmsNorm,
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    qn: RmsNorm,
    kn: RmsNorm,
    n2: RmsNorm,
    gate: Proj,
    up: Proj,
    down: Proj,
}

impl ArBlock {
    fn load(model: &Arc<CmfModel>, p: &str) -> Result<Self, String> {
        Ok(Self {
            n1: RmsNorm::load(model, &format!("{p}.input_layernorm.weight"))?,
            q: Proj::from_model(model, &format!("{p}.self_attn.q_proj.weight"))?,
            k: Proj::from_model(model, &format!("{p}.self_attn.k_proj.weight"))?,
            v: Proj::from_model(model, &format!("{p}.self_attn.v_proj.weight"))?,
            o: Proj::from_model(model, &format!("{p}.self_attn.o_proj.weight"))?,
            qn: RmsNorm::load(model, &format!("{p}.self_attn.q_norm.weight"))?,
            kn: RmsNorm::load(model, &format!("{p}.self_attn.k_norm.weight"))?,
            n2: RmsNorm::load(model, &format!("{p}.post_attention_layernorm.weight"))?,
            gate: Proj::from_model(model, &format!("{p}.mlp.gate_proj.weight"))?,
            up: Proj::from_model(model, &format!("{p}.mlp.up_proj.weight"))?,
            down: Proj::from_model(model, &format!("{p}.mlp.down_proj.weight"))?,
        })
    }
}

/// Keys and values for one layer, one CFG branch.
///
/// Per HEAD, because that is the shape the token graph mirrors from:
/// it uploads `cpu_k[h][synced..position]` into its device cache each
/// step. A flat `[pos][nkv·hd]` buffer would have to be transposed on
/// every token to hand it over.
#[derive(Default, Clone)]
struct KvRun {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

impl KvRun {
    fn init(nkv: usize) -> Self {
        Self {
            k: vec![Vec::new(); nkv],
            v: vec![Vec::new(); nkv],
            len: 0,
        }
    }
}

/// Deterministic top-k sampler.
///
/// NOT torch's: reproducing `torch.multinomial` under a seeded
/// `Generator` bit-for-bit is its own project, and nothing downstream
/// needs the same seed to mean the same song — only that one seed here
/// always means one song.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as f32) / (1u32 << 24) as f32
    }
}

fn sample_topk(logits: &[f32], top_k: usize, rng: &mut Rng) -> usize {
    let mut idx: Vec<usize> = (0..logits.len()).filter(|&i| logits[i].is_finite()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(top_k.max(1));
    let mx = idx.iter().map(|&i| logits[i]).fold(f32::MIN, f32::max);
    let exps: Vec<f32> = idx.iter().map(|&i| (logits[i] - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut r = rng.next_f32() * sum;
    for (j, &e) in exps.iter().enumerate() {
        r -= e;
        if r <= 0.0 {
            return idx[j];
        }
    }
    *idx.last().unwrap()
}

/// MiniMax-Music-3's AR stack: it does not encode a prompt, it GENERATES
/// the conditioning — audio tokens sampled frame by frame, whose hidden
/// states become what the DiT is conditioned on.
pub struct Music3Ar {
    blocks: Vec<ArBlock>,
    norm: RmsNorm,
    embed_prefill: QTensor,
    embed_audio: QTensor,
    embed_extra: QTensor,
    lm_head: Proj,
    pub depth: RvqDepthDecoder,
    inv_freq: Vec<f32>,
    pool: Option<Arc<Pool>>,
    hidden: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    inter: usize,
    audio_vocab: usize,
    codebooks: usize,
    pub cfg_scale: f32,
    pub top_k: usize,
    pub fps: usize,
    pub max_frames: usize,
}

/// Token ids the prompt is built from — `comfy/ldm/minimax_music/prompt.py`.
pub mod tokens {
    pub const IM_START: u32 = 151644;
    pub const IM_END: u32 = 151645;
    pub const AUDIO_CFG: u32 = 151654;
    pub const AUDIO_START: u32 = 151669;
    pub const CAPTION_START: u32 = 151671;
    pub const CAPTION_END: u32 = 151672;
    pub const LYRICS_START: u32 = 151673;
    pub const LYRICS_END: u32 = 151674;
}

impl Music3Ar {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value =
            serde_json::from_slice(model.tensor_bytes("mte.config_json").map_err(|e| e.to_string())?)
                .map_err(|e| format!("mte.config_json: {e}"))?;
        let u = |k: &str, d: usize| cfg[k].as_u64().map(|v| v as usize).unwrap_or(d);
        let f = |k: &str, d: f64| cfg[k].as_f64().unwrap_or(d);
        let hidden = u("hidden_size", 4096);
        let hd = u("head_dim", 128);
        let theta = f("rope_theta", 1_000_000.0) as f32;
        Ok(Self {
            blocks: (0..u("num_hidden_layers", 36))
                .map(|i| ArBlock::load(model, &format!("mte.layers.{i}")))
                .collect::<Result<_, _>>()?,
            norm: RmsNorm::load(model, "mte.norm.weight")?,
            embed_prefill: QTensor::from_model(model, "mte.embed_tokens_prefill.weight")?,
            embed_audio: QTensor::from_model(model, "mte.embed_tokens_audio.weight")?,
            embed_extra: QTensor::from_model(model, "mte.audio_extra_embedding.weight")?,
            lm_head: Proj::from_model(model, "mte.lm_head_pruned.weight")?,
            depth: RvqDepthDecoder::from_cmf(model)?,
            inv_freq: (0..hd / 2)
                .map(|i| 1.0 / theta.powf(2.0 * i as f32 / hd as f32))
                .collect(),
            pool: Pool::from_env(),
            hidden,
            nh: u("num_attention_heads", 32),
            nkv: u("num_key_value_heads", 8),
            hd,
            inter: u("intermediate_size", 12288),
            audio_vocab: u("audio_vocab_size", 1024),
            codebooks: u("audio_num_codebooks", 8),
            cfg_scale: f("cfg_scale", 1.5) as f32,
            top_k: u("top_k", 50),
            fps: u("audio_frames_per_second", 25),
            max_frames: u("max_audio_frames", 9000),
        })
    }

    /// One embedding row. A table lookup, not a matmul: these are
    /// `[vocab, hidden]` and only ever read one row at a time.
    fn embed_row(p: &QTensor, row: usize, hidden: usize) -> Vec<f32> {
        let mut out = vec![0f32; hidden];
        p.row_f32(row, &mut out);
        out
    }

    /// One decode position through the WHOLE stack in a single device
    /// submission, per CFG branch.
    ///
    /// The op-by-op path spends ~9 µs of arithmetic behind each of five
    /// dispatches a layer — 360 round trips a token — and the card idles
    /// between them however cheap the trip is. `forward_token_graph`
    /// already solves exactly this for text models: it keeps K/V on the
    /// device per `(kv_id, layer)`, mirrors the newly appended positions
    /// from the host cache, and folds the final norm and lm_head into
    /// the same submit. Qwen3 is its native shape, q/k norms included,
    /// so no kernel is new here — only the wiring.
    ///
    /// Returns the logits per branch when the graph took the token.
    #[allow(clippy::type_complexity)]
    fn graph_step(
        &self,
        x: &mut [f32],
        pos: usize,
        cap: usize,
        cache: &mut [Vec<KvRun>],
        logits: &mut [Vec<f32>; 2],
    ) -> bool {
        let hs = self.hidden;
        let Some((model, _)) = self.blocks.first().and_then(|b| b.q.graph_w()) else {
            return false;
        };
        let model = model.clone();
        let Some((_, lm)) = self.lm_head.graph_w() else {
            return false;
        };
        let lm_rows = self.lm_head_rows();
        let mut staged: [(Vec<f32>, Vec<f32>); 2] = Default::default();
        for bi in 0..2 {
            let mut layers: Vec<crate::gpu::GraphLayer> = Vec::with_capacity(self.blocks.len());
            for (li, blk) in self.blocks.iter().enumerate() {
                let (Some((_, wq)), Some((_, wk)), Some((_, wv)), Some((_, wo))) = (
                    blk.q.graph_w(),
                    blk.k.graph_w(),
                    blk.v.graph_w(),
                    blk.o.graph_w(),
                ) else {
                    return false;
                };
                let (Some((_, g)), Some((_, u)), Some((_, d))) = (
                    blk.gate.graph_w(),
                    blk.up.graph_w(),
                    blk.down.graph_w(),
                ) else {
                    return false;
                };
                layers.push(crate::gpu::GraphLayer {
                    input_norm: &blk.n1.w,
                    attn: crate::gpu::GraphAttn::Full {
                        wq,
                        wk,
                        wv,
                        wo,
                        q_norm: Some(&blk.qn.w),
                        k_norm: Some(&blk.kn.w),
                        bias: None,
                        output_gate: false,
                        cpu_k: &cache[li][bi].k,
                        cpu_v: &cache[li][bi].v,
                    },
                    post_norm: &blk.n2.w,
                    ffn: crate::gpu::GraphFfn::Dense {
                        gate: g,
                        up: u,
                        down: d,
                    },
                });
            }
            let mut h = x[bi * hs..(bi + 1) * hs].to_vec();
            let mut out = Vec::new();
            let ok = crate::gpu::forward_token_graph(
                &model,
                bi as u64,
                &layers,
                &[],
                0,
                &self.inv_freq,
                &mut h,
                self.nh,
                self.nkv,
                self.hd,
                self.hd,
                hs,
                self.inter,
                pos,
                cap,
                false,
                1e-6,
                Some((&lm, lm_rows)),
                &self.norm.w,
                &mut out,
                &[],
                1,
                None,
                None,
                None,
                0,
            );
            if !ok || out.len() < lm_rows {
                return false;
            }
            staged[bi] = (h, out);
        }
        // Both branches or neither: a half-applied token would leave the
        // two CFG streams a position apart, which reads as the model
        // losing the plot rather than as an error.
        for (bi, (h, out)) in staged.into_iter().enumerate() {
            x[bi * hs..(bi + 1) * hs].copy_from_slice(&h);
            logits[bi] = out;
        }
        true
    }

    /// Run one position through every block, appending to the cache.
    /// `x` is `[b, hidden]` for the CFG pair; returns the normed hidden.
    fn step_blocks(&self, x: &mut [f32], b: usize, pos: usize, cache: &mut [Vec<KvRun>]) {
        let (hs, nh, nkv, hd) = (self.hidden, self.nh, self.nkv, self.hd);
        let pool = self.pool.as_deref();
        let kvw = nkv * hd;
        for (li, blk) in self.blocks.iter().enumerate() {
            let mut h = x.to_vec();
            blk.n1.apply(&mut h, hs);
            let mut q = vec![0f32; b * nh * hd];
            let mut k = vec![0f32; b * kvw];
            let mut v = vec![0f32; b * kvw];
            blk.q.matmat(&h, b, &mut q, pool);
            blk.k.matmat(&h, b, &mut k, pool);
            blk.v.matmat(&h, b, &mut v, pool);
            // Per-head RMSNorm on q and k BEFORE the rotation, then
            // split-half RoPE — Qwen3's order, not the other way round.
            for bi in 0..b {
                for hh in 0..nh {
                    let s = bi * nh * hd + hh * hd;
                    blk.qn.apply(&mut q[s..s + hd], hd);
                    rope_half(&mut q[s..s + hd], pos, &self.inv_freq);
                }
                for hh in 0..nkv {
                    let s = bi * kvw + hh * hd;
                    blk.kn.apply(&mut k[s..s + hd], hd);
                    rope_half(&mut k[s..s + hd], pos, &self.inv_freq);
                }
            }
            let mut attn = vec![0f32; b * nh * hd];
            let per_kv = nh / nkv;
            for bi in 0..b {
                let run = &mut cache[li][bi];
                for g in 0..nkv {
                    run.k[g].extend_from_slice(&k[bi * kvw + g * hd..bi * kvw + (g + 1) * hd]);
                    run.v[g].extend_from_slice(&v[bi * kvw + g * hd..bi * kvw + (g + 1) * hd]);
                }
                run.len += 1;
                let n = run.len;
                let scale = 1.0 / (hd as f32).sqrt();
                for hh in 0..nh {
                    let g = hh / per_kv;
                    let qi = &q[bi * nh * hd + hh * hd..bi * nh * hd + hh * hd + hd];
                    let mut sc = vec![0f32; n];
                    let mut mx = f32::NEG_INFINITY;
                    for (j, s) in sc.iter_mut().enumerate() {
                        let kj = &run.k[g][j * hd..(j + 1) * hd];
                        *s = qi.iter().zip(kj).map(|(a, c)| a * c).sum::<f32>() * scale;
                        mx = mx.max(*s);
                    }
                    let mut sum = 0.0;
                    for s in sc.iter_mut() {
                        *s = (*s - mx).exp();
                        sum += *s;
                    }
                    let inv = 1.0 / sum;
                    let dst = &mut attn[bi * nh * hd + hh * hd..bi * nh * hd + hh * hd + hd];
                    for (j, &s) in sc.iter().enumerate() {
                        let w = s * inv;
                        let vj = &run.v[g][j * hd..(j + 1) * hd];
                        for (d, &vv) in dst.iter_mut().zip(vj) {
                            *d += w * vv;
                        }
                    }
                }
            }
            let mut proj = vec![0f32; b * hs];
            blk.o.matmat(&attn, b, &mut proj, pool);
            for (a, c) in x.iter_mut().zip(&proj) {
                *a += c;
            }
            let mut h = x.to_vec();
            blk.n2.apply(&mut h, hs);
            let (mut g, mut u2) = (vec![0f32; b * self.inter], vec![0f32; b * self.inter]);
            blk.gate.matmat(&h, b, &mut g, pool);
            blk.up.matmat(&h, b, &mut u2, pool);
            for (a, c) in g.iter_mut().zip(&u2) {
                *a = silu(*a) * c;
            }
            let mut ffo = vec![0f32; b * hs];
            blk.down.matmat(&g, b, &mut ffo, pool);
            for (a, c) in x.iter_mut().zip(&ffo) {
                *a += c;
            }
        }
    }

    /// Generate `frames` of conditioning: `[frames, 8·hidden]`.
    ///
    /// The CFG pair runs as batch 2 — the conditioned prompt and one
    /// whose middle is replaced by `<|audio_cfg|>` — and every sampled
    /// code is shared by both branches, which is what makes them stay
    /// in step.
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        frames: usize,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<(Vec<f32>, usize), String> {
        let hs = self.hidden;
        let pool = self.pool.as_deref();
        let want = frames.min(self.max_frames);
        let mut cache: Vec<Vec<KvRun>> = (0..self.blocks.len())
            .map(|_| vec![KvRun::init(self.nkv); 2])
            .collect();
        // The unconditioned branch keeps the frame but not the words.
        let mut uncond = prompt_ids.to_vec();
        if uncond.len() > 3 {
            let n = uncond.len();
            for t in uncond[1..n - 2].iter_mut() {
                *t = tokens::AUDIO_CFG;
            }
        }
        let cap = prompt_ids.len() + want + 2;
        // One path for the whole generation. The graph keeps K/V on the
        // device and the host loop keeps it here; alternating between
        // them mid-song would leave the two caches disagreeing.
        // Opt-in, not opt-out. Measured on a 3090 next to a 256-core EPYC:
        // the whole-token graph costs 136.7 s of AR against 47.8 s host,
        // 2.9x SLOWER. The AR is batch-1/2 matvec, so every layer is a
        // bandwidth errand too small to amortise a submit, while the host
        // arm has 256 cores to spread it over. The graph is the right
        // shape for a thin CPU; it is the wrong one here, so it ships off.
        // (Gating this on `enabled_here()` was also wrong — the AR runs
        // before the backend is up, so that read false and the graph was
        // never once attempted. Keep it un-gated so the A/B stays honest.)
        let mut graph = std::env::var("CMF_MUSIC3_GRAPH").as_deref() == Ok("1");
        let mut glogits: [Vec<f32>; 2] = Default::default();
        let mut last = vec![0f32; 2 * hs];
        for (pos, (&a, &b)) in prompt_ids.iter().zip(&uncond).enumerate() {
            let mut x = vec![0f32; 2 * hs];
            x[..hs].copy_from_slice(&Self::embed_row(&self.embed_prefill, a as usize, hs));
            x[hs..].copy_from_slice(&Self::embed_row(&self.embed_prefill, b as usize, hs));
            if graph {
                if self.graph_step(&mut x, pos, cap, &mut cache, &mut glogits) {
                    last = x;
                    if pos % 64 == 0 {
                        progress(0, want);
                    }
                    continue;
                }
                // Refused on the first token: nothing has been committed,
                // so the host arm starts clean from position 0.
                if pos > 0 {
                    return Err("the token graph refused mid-prefill".into());
                }
                graph = false;
            }
            self.step_blocks(&mut x, 2, pos, &mut cache);
            last = x;
            if pos % 64 == 0 {
                progress(0, want);
            }
        }
        let mut rng = Rng::new(seed);
        let mut out: Vec<f32> = Vec::with_capacity(want * self.codebooks * hs);
        let mut done = 0usize;
        let scale = (self.codebooks as f32).powf(-0.5);
        for frame in 0..want {
            let mut normed = last.clone();
            self.norm.apply(&mut normed, hs);
            // c0 with classifier-free guidance and a top-k mask taken
            // from the CONDITIONED logits, per the reference. The graph
            // folds the final norm and lm_head into its own submit, so
            // when it is driving the logits are already here.
            let vocab = self.lm_head_rows();
            let mut logits = vec![0f32; 2 * vocab];
            if graph {
                logits[..vocab].copy_from_slice(&glogits[0][..vocab]);
                logits[vocab..].copy_from_slice(&glogits[1][..vocab]);
            } else {
                self.lm_head.matmat(&normed, 2, &mut logits, pool);
            }
            let (cond, unc) = logits.split_at(vocab);
            let mut thr: Vec<f32> = cond.to_vec();
            thr.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
            let cut = thr[self.top_k.min(vocab) - 1];
            let guided: Vec<f32> = (0..vocab)
                .map(|i| {
                    if cond[i] < cut {
                        f32::NEG_INFINITY
                    } else {
                        unc[i] + (cond[i] - unc[i]) * self.cfg_scale
                    }
                })
                .collect();
            let code = sample_topk(&guided, self.top_k, &mut rng);
            if code == 0 {
                break; // stop token
            }
            let c0 = code - 1;
            let c0_embed = Self::embed_row(&self.embed_audio, c0, hs);
            // Depth: the remaining seven codebooks, and their hidden
            // states are seven eighths of what the DiT will see.
            let mut seq = self.depth.project(&normed[..hs]);
            seq.extend_from_slice(&self.depth.project(&c0_embed));
            let mut codes = vec![c0];
            let mut frame_hidden = normed[..hs].to_vec();
            for level in 1..self.codebooks {
                let n = seq.len() / hs;
                let h = self.depth.forward_last(&seq, n);
                frame_hidden.extend_from_slice(&h);
                let lg = self.depth.head(level, &h);
                let c = sample_topk(&lg, self.top_k, &mut rng);
                codes.push(c);
                if level < self.codebooks - 1 {
                    let e = Self::embed_row(&self.embed_extra, c + (level - 1) * self.audio_vocab, hs);
                    seq.extend_from_slice(&self.depth.project(&e));
                }
            }
            out.extend_from_slice(&frame_hidden);
            done += 1;
            progress(done, want);
            if done >= want {
                break;
            }
            // Feed the whole frame back: c0's embedding plus the extras,
            // scaled by 1/sqrt(codebooks).
            let mut fb = Self::embed_row(&self.embed_audio, codes[0], hs);
            for (level, &c) in codes.iter().enumerate().skip(1) {
                let e = Self::embed_row(&self.embed_extra, c + (level - 1) * self.audio_vocab, hs);
                for (a, b) in fb.iter_mut().zip(&e) {
                    *a += b;
                }
            }
            for a in fb.iter_mut() {
                *a *= scale;
            }
            let mut x = vec![0f32; 2 * hs];
            x[..hs].copy_from_slice(&fb);
            x[hs..].copy_from_slice(&fb);
            let pos = prompt_ids.len() + frame;
            if !graph || !self.graph_step(&mut x, pos, cap, &mut cache, &mut glogits) {
                if graph {
                    return Err("the token graph refused mid-song".into());
                }
                self.step_blocks(&mut x, 2, pos, &mut cache);
            }
            last = x;
        }
        if done == 0 {
            return Err("MiniMax-Music-3 generated zero audio frames".into());
        }
        Ok((out, done))
    }

    fn lm_head_rows(&self) -> usize {
        match &self.lm_head {
            Proj::F32 { rows, .. } => *rows,
            Proj::Q(q) => q.rows(),
        }
    }
}

/// Split-half rotation over one head, angle `pos·inv_freq[i]`.
fn rope_half(x: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        let (s, c) = (pos as f32 * inv_freq[i]).sin_cos();
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = b * c + a * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The half that is finished, end to end: noise → Euler sampler →
    /// vocoder → PCM. It does not make music (the conditioning would
    /// come from the AR stack), but it proves the three implemented
    /// components compose — the sampler's σ walk, the DiT's timestep
    /// convention and the vocoder's hop all have to agree for the
    /// sample count to land, and the count is fixed by the reference.
    #[test]
    fn music3_sampler_and_vocoder_compose() {
        let Ok(p) = std::env::var("CMF_MUSIC3_DIT") else {
            eprintln!("CMF_MUSIC3_DIT unset — skipping Music-3 chain test");
            return;
        };
        let model = Arc::new(CmfModel::open(&p).expect("open pack"));
        let dit = Music3Dit::from_cmf(&model).expect("load DiT");
        let dav = crate::audiovae::Music3Dav::from_cmf(&model).expect("load DAV");
        let n = 6usize;
        // A fixed pseudo-noise: the test must not depend on an RNG.
        let noise: Vec<f32> = (0..Music3Dit::IN_CH * n)
            .map(|i| (((i * 2654435761) % 1000) as f32 / 500.0) - 1.0)
            .collect();
        let cond = vec![0f32; Music3Dit::COND_CH * n];
        let steps = 3;
        let mut seen = 0usize;
        let latent = dit.sample(&noise, &cond, n, steps, |i, t| {
            assert_eq!(t, steps);
            seen = i;
        });
        assert_eq!(seen, steps, "sampler reported every step");
        assert_eq!(latent.len(), Music3Dit::IN_CH * n);
        assert!(latent.iter().all(|v| v.is_finite()), "latent went non-finite");
        let pcm = dav.decode(&latent, n, None);
        assert_eq!(
            pcm.len(),
            n * crate::audiovae::Music3Dav::HOP * 2,
            "the chain's sample count is frames x 512 x 2"
        );
        assert!(pcm.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
        let secs = (n * crate::audiovae::Music3Dav::HOP) as f32
            / crate::audiovae::Music3Dav::SAMPLE_RATE as f32;
        eprintln!(
            "music3 chain: {n} frames -> {} stereo samples ({secs:.3} s at 44.1 kHz)",
            pcm.len() / 2
        );
    }

    /// The depth decoder must be CAUSAL — that is the only thing its
    /// attention mask does, and losing it lets a codebook level see its
    /// own answer while the model still samples plausible codes.
    /// Appending a position may not change any earlier output.
    #[test]
    fn music3_rvq_depth_decoder_is_causal() {
        let Ok(p) = std::env::var("CMF_MUSIC3_TE") else {
            eprintln!("CMF_MUSIC3_TE unset — skipping RVQ decoder test");
            return;
        };
        let model = Arc::new(CmfModel::open(&p).expect("open packed AR stack"));
        let dec = RvqDepthDecoder::from_cmf(&model).expect("load RVQ decoder");
        assert_eq!(dec.codebooks(), 8, "eight codebooks");
        let hs = dec.hidden;
        let seq: Vec<f32> = (0..3 * hs).map(|i| 0.05 * ((i as f32) * 0.013).sin()).collect();
        let a = dec.forward_last(&seq[..2 * hs], 2);
        let b = dec.forward_last(&seq, 3);
        assert_eq!(a.len(), hs);
        assert!(a.iter().all(|v| v.is_finite()) && b.iter().all(|v| v.is_finite()));
        // Position 1's own output is read by forward_last at n=2; adding
        // position 2 must leave the stack's view of 0..=1 untouched, so
        // re-running with the shorter prefix must agree with itself.
        let a2 = dec.forward_last(&seq[..2 * hs], 2);
        let d = a.iter().zip(&a2).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(d == 0.0, "not deterministic: {d}");
        let logits = dec.head(1, &b);
        assert_eq!(logits.len(), 1024, "audio vocab is 1024 per level");
        assert!(logits.iter().all(|v| v.is_finite()));
        let spread = logits.iter().fold(f32::MIN, |m, v| m.max(*v))
            - logits.iter().fold(f32::MAX, |m, v| m.min(*v));
        assert!(spread > 1e-3, "head is flat, spread {spread}");
        eprintln!("music3 rvq: 8 codebooks, head spread {spread:.3}");
    }

    /// Forward the packed DiT and check what the reference fixes: a
    /// `[128, n]` velocity, finite, and responsive to BOTH inputs.
    /// `CMF_MUSIC3_DIT=<file.cmf>` points at a pack.
    ///
    /// The two response checks are the point. A forward that silently
    /// drops the condition — the easiest way to get the 2304-wide concat
    /// wrong — still returns a plausible velocity, and so does one that
    /// ignores the timestep token.
    #[test]
    fn music3_dit_forward_has_the_reference_geometry() {
        let Ok(p) = std::env::var("CMF_MUSIC3_DIT") else {
            eprintln!("CMF_MUSIC3_DIT unset — skipping Music-3 DiT test");
            return;
        };
        let model = Arc::new(CmfModel::open(&p).expect("open packed DiT"));
        let dit = Music3Dit::from_cmf(&model).expect("load DiT");
        let n = 4usize;
        let x: Vec<f32> = (0..Music3Dit::IN_CH * n)
            .map(|i| 0.3 * ((i as f32) * 0.017).sin())
            .collect();
        let cond: Vec<f32> = (0..Music3Dit::COND_CH * n)
            .map(|i| 0.2 * ((i as f32) * 0.011).cos())
            .collect();
        let v = dit.forward(&x, &cond, n, 0.7);
        assert_eq!(v.len(), Music3Dit::IN_CH * n, "velocity is [128, n]");
        assert!(v.iter().all(|q| q.is_finite()), "non-finite velocity");
        let rms = (v.iter().map(|q| q * q).sum::<f32>() / v.len() as f32).sqrt();
        assert!(rms > 1e-6, "velocity is silent, rms {rms}");

        let zero_cond = vec![0f32; Music3Dit::COND_CH * n];
        let v0 = dit.forward(&x, &zero_cond, n, 0.7);
        let dc = v
            .iter()
            .zip(&v0)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(dc > 1e-5, "condition changed nothing — concat is wrong");

        let vt = dit.forward(&x, &cond, n, 0.2);
        let dt = v
            .iter()
            .zip(&vt)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(dt > 1e-5, "timestep changed nothing — the token is lost");
        eprintln!("music3 dit: rms {rms:.4}, d/dcond {dc:.4}, d/dt {dt:.4}");
    }
}
