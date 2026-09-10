//! The LTX-2.5 duration head: how long the shot the prompt describes should
//! be, read off the connector outputs before a single denoising step runs.
//!
//! Both connector streams are projected into a shared 256-wide space, tagged
//! with a learnable per-modality embedding so the pooler can tell them
//! apart, and cross-attended by one learnable query. A two-layer MLP turns
//! the pooled vector into a *log*-duration — the head was trained in log
//! seconds so its loss spreads evenly across orders of magnitude — and the
//! exponential of that is the answer.

use crate::ltxdit::{Lin, gelu_tanh, softmax};
use cortiq_core::CmfModel;
use std::sync::Arc;

fn vecf(model: &Arc<CmfModel>, name: &str) -> Result<Vec<f32>, String> {
    crate::dit::cmf_f32(model, name)
}

pub struct DurationHead {
    v_proj: Lin,
    a_proj: Lin,
    v_emb: Vec<f32>,
    a_emb: Vec<f32>,
    query: Vec<f32>,
    in_proj_w: Vec<f32>,
    in_proj_b: Vec<f32>,
    out_proj: Lin,
    mlp_hidden: Lin,
    mlp_out: Lin,
    dim: usize,
    heads: usize,
}

impl DurationHead {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<DurationHead, String> {
        let p = "dhead.duration_head";
        let query = vecf(model, &format!("{p}.attention_pooler.query_tokens"))?;
        let dim = query.len();
        Ok(DurationHead {
            v_proj: Lin::load(model, &format!("{p}.video_input_proj"), true)?,
            a_proj: Lin::load(model, &format!("{p}.audio_input_proj"), true)?,
            v_emb: vecf(model, &format!("{p}.video_modality_emb"))?,
            a_emb: vecf(model, &format!("{p}.audio_modality_emb"))?,
            query,
            in_proj_w: vecf(
                model,
                &format!("{p}.attention_pooler.cross_attn.in_proj_weight"),
            )?,
            in_proj_b: vecf(
                model,
                &format!("{p}.attention_pooler.cross_attn.in_proj_bias"),
            )?,
            out_proj: Lin::load(
                model,
                &format!("{p}.attention_pooler.cross_attn.out_proj"),
                true,
            )?,
            mlp_hidden: Lin::load(model, &format!("{p}.mlp_hidden"), true)?,
            mlp_out: Lin::load(model, &format!("{p}.mlp_out"), true)?,
            dim,
            heads: 4,
        })
    }

    /// Seconds, from the two connector outputs.
    pub fn seconds(
        &self,
        video: &[f32],
        audio: &[f32],
        ctx_len: usize,
        pool: Option<&crate::pool::Pool>,
    ) -> f32 {
        let d = self.dim;
        let mut tokens = self.v_proj.apply(video, ctx_len, pool);
        for (i, v) in tokens.iter_mut().enumerate() {
            *v += self.v_emb[i % d];
        }
        let mut at = self.a_proj.apply(audio, ctx_len, pool);
        for (i, v) in at.iter_mut().enumerate() {
            *v += self.a_emb[i % d];
        }
        tokens.extend_from_slice(&at);
        let n = 2 * ctx_len;

        // one query, cross-attending every token: q from the learnable
        // token, k and v from the stream. torch packs the three input
        // projections into one matrix, in that order.
        let proj = |x: &[f32], off: usize| -> Vec<f32> {
            (0..d)
                .map(|o| {
                    let row = &self.in_proj_w[(off + o) * d..(off + o) * d + d];
                    self.in_proj_b[off + o] + row.iter().zip(x).map(|(&a, &b)| a * b).sum::<f32>()
                })
                .collect()
        };
        let q = proj(&self.query, 0);
        let hd = d / self.heads;
        let mut ctx = vec![0f32; d];
        let mut k = vec![0f32; d];
        let mut v = vec![0f32; d];
        let mut scores = vec![vec![0f32; n]; self.heads];
        let mut vs = vec![0f32; n * d];
        for t in 0..n {
            let row = &tokens[t * d..(t + 1) * d];
            k.copy_from_slice(&proj(row, d));
            v.copy_from_slice(&proj(row, 2 * d));
            vs[t * d..(t + 1) * d].copy_from_slice(&v);
            for h in 0..self.heads {
                let s: f32 = (0..hd).map(|i| q[h * hd + i] * k[h * hd + i]).sum();
                scores[h][t] = s / (hd as f32).sqrt();
            }
        }
        for h in 0..self.heads {
            softmax(&mut scores[h]);
            for (t, &p) in scores[h].iter().enumerate() {
                for i in 0..hd {
                    ctx[h * hd + i] += p * vs[t * d + h * hd + i];
                }
            }
        }
        let pooled = self.out_proj.apply(&ctx, 1, pool);
        let mut hidden = self.mlp_hidden.apply(&pooled, 1, pool);
        hidden.iter_mut().for_each(|v| *v = gelu_tanh(*v));
        self.mlp_out.apply(&hidden, 1, pool)[0].exp()
    }
}

/// A duration in seconds → a frame count on the VAE's `8k + 1` grid,
/// clamped so a misbehaving prediction cannot ask for a degenerate or
/// enormous generation.
pub fn frames_for(seconds: f32, fps: f64, min_seconds: f64, max_seconds: f64) -> usize {
    let scale = crate::ltxpipe::SCALE_TIME;
    let min_frames = (min_seconds * fps).round().max(1.0) as usize;
    let max_frames = (max_seconds * fps).round() as usize;
    let raw = ((seconds as f64 * fps).round() as usize).clamp(min_frames, max_frames);
    // snap down to the grid, then up if that fell under the floor
    let frames = ((raw.saturating_sub(1)) / scale) * scale + 1;
    if frames < min_frames {
        (min_frames.saturating_sub(1)).div_ceil(scale) * scale + 1
    } else {
        frames
    }
    .min(max_frames)
}
