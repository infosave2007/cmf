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
        })
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
        for hh in 0..nh {
            for i in 0..n {
                let qi = &q[i * hs + hh * hd..i * hs + hh * hd + hd];
                let mut scores = vec![0f32; n];
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
                let dst = &mut attn[i * hs + hh * hd..i * hs + hh * hd + hd];
                for (j, &sc) in scores.iter().enumerate() {
                    let w = sc * inv;
                    let vj = &v[j * hs + hh * hd..j * hs + hh * hd + hd];
                    for (d, &vv) in dst.iter_mut().zip(vj) {
                        *d += w * vv;
                    }
                }
            }
        }
        let mut proj = vec![0f32; n * hs];
        blk.out.matmat(&attn, n, &mut proj, pool);
        for (a, b) in x.iter_mut().zip(&proj) {
            *a += b;
        }

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
