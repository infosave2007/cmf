//! MiniMax-H3: the packed-token audio-video DiT.
//!
//! One stream of tokens — `[text | audio | video]` — denoised by fifty
//! blocks of full self-attention, 3-axis RoPE and adaLN modulation, with
//! the video and audio latents riding different flow schedules and a
//! separate output head each.
//!
//! Three things make it unlike the image DiT next door:
//!
//! * **The sequence is packed, not batched.** Text, audio rows and video
//!   rows sit in one sequence and attend to each other; there is no
//!   cross-attention. Modulation is what tells a token which modality it
//!   is: every block emits six modulation vectors for each of three
//!   modality tags at each of the (at most two, for t2va) distinct
//!   timesteps, and a segment table says which row each token reads.
//!
//! * **adaLN arrives as a curve, not a matrix.** The released weight is
//!   [96768, 2688] per block — 13 B parameters, 40% of the model — for a
//!   map whose input is one number. `cortiq animate-pack` collapses it
//!   onto a rank-24 basis of the timestep curve that already carries the
//!   Turbo LoRA, so what lands here is [96768, 24] and a [1025, 24]
//!   table to interpolate. Measured against the full matrix: rms 8.7e-5
//!   on a signal of rms 0.46.
//!
//! * **Two clocks.** The sampler hands in the video sigma; the audio
//!   stream's own sigma is a closed-form remap of it (shift 12 → 3), and
//!   the two are integrated separately. `forward` returns both
//!   velocities unscaled, each on its own schedule — the reference
//!   returns the audio one pre-multiplied by d(σ_a)/d(σ_v) so that a
//!   single-schedule sampler is approximately right, which at four steps
//!   it is not.
//!
//! Parity: `tools/mk_mmh3_toy.py` builds a toy checkpoint carrying the
//! release's real tensor names and a golden forward from ComfyUI's own
//! module; `tools/mmh3_toy_gate.sh` diffs this port against it.

use crate::dit::Proj;
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Frames a video latent token spans, cycling with period 5 — the
/// checkpoint's temporal grid, which is NOT uniform.
const FRAME_PER_TOKEN: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
const FRAME_RESCALE: f64 = 5.0 / 3.0;

/// Modality tags, as the adaLN row layout orders them.
const TAG_VIDEO: usize = 0;
const TAG_TEXT: usize = 1;
const TAG_AUDIO: usize = 2;
const MODALITIES: usize = 3;
/// shift/scale/gate for attention, then the same three for the MLP.
const EXPAND: usize = 6;

// ── the flow-schedule remap ─────────────────────────────────────────

/// σ on `to`'s schedule at the same point of the shared base grid.
pub fn time_shift_sigma(sigma: f64, from_shift: f64, to_shift: f64) -> f64 {
    let base = sigma / (from_shift + sigma * (1.0 - from_shift));
    to_shift * base / (1.0 + (to_shift - 1.0) * base)
}

/// d(σ_to)/d(σ_from) at the same base-grid point.
pub fn time_shift_slope(sigma: f64, from_shift: f64, to_shift: f64) -> f64 {
    let base = sigma / (from_shift + sigma * (1.0 - from_shift));
    (to_shift * (1.0 + (from_shift - 1.0) * base).powi(2))
        / (from_shift * (1.0 + (to_shift - 1.0) * base).powi(2))
}

// ── the packed layout ───────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Text,
    Audio,
    Video,
}

/// One contiguous run of rows sharing a modality tag and a timestep.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub start: usize,
    pub stop: usize,
    pub kind: Kind,
}

/// The static structure of one shape signature: where each stream sits
/// in the sequence and what 3-D position every row carries.
pub struct Layout {
    pub seq_len: usize,
    pub segments: Vec<Segment>,
    /// [seq_len, 3] — (t, h, w), f64 because the axes are fractional.
    pub pos: Vec<[f64; 3]>,
    pub text_len: usize,
    pub audio_t: usize,
    pub latent_t: usize,
    pub lat_h: usize,
    pub lat_w: usize,
    /// Rows per latent frame after the 2×2 patch.
    pub frame_rows: usize,
}

/// `linspace((1 − ratio)/2, (1 + ratio)/2, dim/patch, endpoint=False) · 32`
fn axis_from_sqrt_area(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let n = dim / patch;
    (0..n)
        .map(|i| (i as f64 * (ratio / n as f64) + (1.0 - ratio) / 2.0) * 32.0)
        .collect()
}

impl Layout {
    /// t2va: `[text | audio | video]`, the target streams last and in
    /// that order. Keyframe and reference blocks would slot between the
    /// text and the audio; this port does text-to-video only.
    pub fn t2va(text_len: usize, latent_t: usize, lat_h: usize, lat_w: usize, audio_t: usize) -> Self {
        let area = ((lat_h * lat_w) as f64).sqrt();
        let h_axis = axis_from_sqrt_area(lat_h, 2, area);
        let w_axis = axis_from_sqrt_area(lat_w, 2, area);
        let frame_rows = h_axis.len() * w_axis.len();

        let mut pos: Vec<[f64; 3]> = Vec::new();
        let mut segments = Vec::new();

        segments.push(Segment { start: 0, stop: text_len, kind: Kind::Text });
        for i in 0..text_len {
            pos.push([i as f64, 0.0, 0.0]);
        }

        // Both target streams share this origin: the text runs out at
        // `text_len` and audio and video start together from there.
        let cursor = text_len as f64;

        // Audio is channel-major stereo: every latent frame once per
        // channel, the two channels pinned to the frame grid's extreme
        // w coordinates so they are distinguishable under RoPE.
        let (w_low, w_high) = (w_axis[0], w_axis[w_axis.len() - 1]);
        let a_start = pos.len();
        for ch in 0..2 {
            let w = if ch == 0 { w_low } else { w_high };
            for i in 0..audio_t {
                pos.push([cursor + i as f64, 0.0, w]);
            }
        }
        segments.push(Segment { start: a_start, stop: pos.len(), kind: Kind::Audio });

        // Video: the t axis advances by the per-token frame spans, not
        // by one; h/w come from the shared frame grid.
        let v_start = pos.len();
        let mut t_coord = cursor;
        for k in 0..latent_t {
            for &h in &h_axis {
                for &w in &w_axis {
                    pos.push([t_coord, h, w]);
                }
            }
            t_coord += FRAME_RESCALE * FRAME_PER_TOKEN[k % 5];
        }
        segments.push(Segment { start: v_start, stop: pos.len(), kind: Kind::Video });

        Self {
            seq_len: pos.len(),
            segments,
            pos,
            text_len,
            audio_t,
            latent_t,
            lat_h,
            lat_w,
            frame_rows,
        }
    }

    fn segment(&self, kind: Kind) -> Segment {
        *self
            .segments
            .iter()
            .find(|s| s.kind == kind)
            .expect("every layout carries all three streams")
    }
}

// ── weights ─────────────────────────────────────────────────────────

struct Adaln {
    /// [out, rank] — the curve basis, already carrying the LoRA.
    w: Proj,
    b: Vec<f32>,
    /// [grid, rank]
    table: Vec<f32>,
    rank: usize,
    out: usize,
}

impl Adaln {
    fn load(model: &Arc<CmfModel>, prefix: &str) -> Result<Self, String> {
        let w = Proj::from_model(model, &format!("{prefix}.weight"))?;
        let table = crate::dit::cmf_f32(model, &format!("{prefix}.table"))?;
        let b = crate::dit::cmf_f32(model, &format!("{prefix}.bias"))?;
        let out = w.rows();
        let rank = table.len() / CURVE_GRID;
        Ok(Self { w, b, table, rank, out })
    }

    /// The modulation vectors at `ts`: `[ts.len() · MODALITIES, expand ·
    /// hidden]` laid out so row `t·MODALITIES + tag`, chunk `e`, starts
    /// at `(t·MODALITIES + tag)·expand·hidden + e·hidden`.
    fn eval(&self, ts: &[f64], pool: Option<&Pool>) -> Vec<f32> {
        let g = CURVE_GRID;
        let mut coords = vec![0f32; ts.len() * self.rank];
        for (i, &t) in ts.iter().enumerate() {
            // t → fractional grid index; out-of-range clamps to the ends,
            // and the last interval is kept whole so t = 1 does not read
            // past the table.
            let p = (t.clamp(0.0, 1.0) * (g - 1) as f64) as f32;
            let i0 = (p.floor() as usize).min(g - 2);
            let f = p - i0 as f32;
            for k in 0..self.rank {
                let (a, b) = (self.table[i0 * self.rank + k], self.table[(i0 + 1) * self.rank + k]);
                coords[i * self.rank + k] = a + (b - a) * f;
            }
        }
        let mut out = vec![0f32; ts.len() * self.out];
        self.w.matmat(&coords, ts.len(), &mut out, pool);
        for row in out.chunks_exact_mut(self.out) {
            for (v, &bv) in row.iter_mut().zip(&self.b) {
                *v += bv;
            }
        }
        out
    }
}

struct Block {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    qkv: Proj,   // [3·heads·hd, hidden]
    out: Proj,   // [hidden, heads·hd]
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    fc1: Proj,   // [2·ffn, hidden]
    fc2: Proj,   // [hidden, ffn]
    adaln: Option<Adaln>,
}

pub(crate) const CURVE_GRID: usize = 1025;

pub struct MiniMaxH3 {
    video_patch: Proj,
    video_patch_b: Vec<f32>,
    audio_patch: Proj,
    audio_patch_b: Vec<f32>,
    condition: Proj,
    condition_b: Vec<f32>,
    refiner: Vec<Block>,
    refiner_norm: Vec<f32>,
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
    final_adaln: Adaln,
    video_out: Proj,
    video_out_b: Vec<f32>,
    audio_out: Proj,
    audio_out_b: Vec<f32>,
    inv_freq: Vec<f32>,
    pool: Option<Arc<Pool>>,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub latents_dim: usize,
    pub audio_dim: usize,
    pub text_dim: usize,
    pub shift_video: f64,
    pub shift_audio: f64,
    eps: f64,
    qk_eps: f64,
    final_eps: f64,
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

impl MiniMaxH3 {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model.tensor_bytes("dit.config_json").map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("dit.config_json: {e}"))?;
        let u = |k: &str| cfg[k].as_u64().unwrap_or(0) as usize;
        let f = |k: &str, d: f64| cfg[k].as_f64().unwrap_or(d);
        let n_blocks = u("num_layers");
        let n_refiner = u("token_refiner_num_layers");
        let f32v = |n: &str| crate::dit::cmf_f32(model, n);

        let load_block = |prefix: &str, with_adaln: bool| -> Result<Block, String> {
            Ok(Block {
                norm1: f32v(&format!("dit.{prefix}.norm1.weight"))?,
                norm2: f32v(&format!("dit.{prefix}.norm2.weight"))?,
                qkv: Proj::from_model(model, &format!("dit.{prefix}.attn.qkv_proj.weight"))?,
                out: Proj::from_model(model, &format!("dit.{prefix}.attn.out_proj.weight"))?,
                q_norm: f32v(&format!("dit.{prefix}.attn.q_norm.weight"))?,
                k_norm: f32v(&format!("dit.{prefix}.attn.k_norm.weight"))?,
                fc1: Proj::from_model(model, &format!("dit.{prefix}.mlp.fc1.weight"))?,
                fc2: Proj::from_model(model, &format!("dit.{prefix}.mlp.fc2.weight"))?,
                adaln: if with_adaln {
                    Some(Adaln::load(model, &format!("dit.{prefix}.adaln"))?)
                } else {
                    None
                },
            })
        };
        let blocks = (0..n_blocks)
            .map(|i| load_block(&format!("blocks.{i}"), true))
            .collect::<Result<Vec<_>, _>>()?;
        let refiner = (0..n_refiner)
            .map(|i| load_block(&format!("token_refiner.blocks.{i}"), false))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            video_patch: Proj::from_model(model, "dit.video_patch_proj.weight")?,
            video_patch_b: f32v("dit.video_patch_proj.bias")?,
            audio_patch: Proj::from_model(model, "dit.audio_patch_proj.weight")?,
            audio_patch_b: f32v("dit.audio_patch_proj.bias")?,
            condition: Proj::from_model(model, "dit.condition_proj.weight")?,
            condition_b: f32v("dit.condition_proj.bias")?,
            refiner,
            refiner_norm: f32v("dit.token_refiner.final_norm.weight")?,
            blocks,
            final_norm: f32v("dit.final_layer.norm.weight")?,
            final_adaln: Adaln::load(model, "dit.final_layer.adaln")?,
            video_out: Proj::from_model(model, "dit.final_layer.video_out.weight")?,
            video_out_b: f32v("dit.final_layer.video_out.bias")?,
            audio_out: Proj::from_model(model, "dit.final_layer.audio_out.weight")?,
            audio_out_b: f32v("dit.final_layer.audio_out.bias")?,
            inv_freq: f32v("dit.rope_inv_freq")?,
            pool: Pool::from_env(),
            hidden: u("hidden_size"),
            heads: u("num_attention_heads"),
            head_dim: u("attention_head_dim"),
            ffn: u("ffn_hidden_size"),
            latents_dim: u("latents_dim"),
            audio_dim: u("audio_latents_dim"),
            text_dim: u("text_dim"),
            shift_video: f("sigma_shift_video", 12.0),
            shift_audio: f("sigma_shift_audio", 3.0),
            eps: f("norm_eps", 1e-5),
            qk_eps: f("qk_norm_eps", 1e-5),
            final_eps: f("final_norm_eps", 1e-5),
        })
    }

    /// Qwen3-VL states `[n, text_dim]` → refined text embeds
    /// `[n, hidden]`. Prompt-only, so the caller does this once per
    /// generation rather than once per step.
    pub fn refine_text(&self, states: &[f32], n: usize) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let mut h = vec![0f32; n * self.hidden];
        self.condition.matmat(states, n, &mut h, pool);
        for row in h.chunks_exact_mut(self.hidden) {
            for (v, &b) in row.iter_mut().zip(&self.condition_b) {
                *v += b;
            }
        }
        let ids: Vec<[f64; 3]> = Vec::new();
        for blk in &self.refiner {
            self.block_forward(blk, &mut h, n, None, &ids, &[]);
        }
        let mut out = vec![0f32; n * self.hidden];
        for (o, x) in out.chunks_exact_mut(self.hidden).zip(h.chunks_exact(self.hidden)) {
            rms_norm_into(x, &self.refiner_norm, self.final_eps, o);
        }
        out
    }

    /// The rotation angles of every row: `[n, 48]`. Each of the three
    /// position axes contributes `inv_freq.len()` angles, and the pair
    /// (j, j+48) of the first 96 head dims rotates by angle j.
    fn rope_angles(&self, pos: &[[f64; 3]]) -> Vec<f32> {
        let k = self.inv_freq.len();
        let mut out = Vec::with_capacity(pos.len() * 3 * k);
        for p in pos {
            for axis in 0..3 {
                for j in 0..k {
                    out.push((p[axis] * self.inv_freq[j] as f64) as f32);
                }
            }
        }
        out
    }

    /// Per-head RMSNorm then the partial split-half rotation, in place.
    /// `angles` carries 48 angles a row — three axes × 16 frequencies —
    /// and pair (j, j+48) of the head's first 96 dims turns by angle j.
    /// Dims 96..128 are not rotated at all, which is the checkpoint's
    /// own `rope_inv_freq_len` arithmetic and not a truncation.
    /// In place on a STRIDED view of the fused qkv buffer: `stride` is
    /// the row pitch and `off` the plane's start. Normalizing q and k
    /// through a scratch copy cost four full passes over `n·heads·hd`
    /// per block — 10 G element copies over a render — for arithmetic
    /// that touches each element once.
    #[allow(clippy::too_many_arguments)]
    fn norm_rope_w(
        &self,
        v: &mut [f32],
        n: usize,
        heads: usize,
        w: &[f32],
        angles: &[f32],
        stride: usize,
        off: usize,
    ) {
        let hd = self.head_dim;
        let pairs = if angles.is_empty() { 0 } else { angles.len() / n };
        let pool = self.pool.as_deref();
        let ptr = SendPtr(v.as_mut_ptr());
        let work = |lo: usize, hi: usize| {
            for p in lo..hi {
                for h in 0..heads {
                    // SAFETY: workers own disjoint token ranges, and the
                    // heads of one token are disjoint within it.
                    let x = unsafe { ptr.row(p * stride + off + h * hd, hd) };
                    let ss = x.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>() / hd as f64;
                    let inv = 1.0 / (ss + self.qk_eps).sqrt();
                    for (d, &g) in x.iter_mut().zip(w) {
                        *d = (*d as f64 * inv) as f32 * g;
                    }
                    for j in 0..pairs {
                        let a = angles[p * pairs + j];
                        let (s, c) = a.sin_cos();
                        let (lo_v, hi_v) = (x[j], x[j + pairs]);
                        x[j] = lo_v * c - hi_v * s;
                        x[j + pairs] = lo_v * s + hi_v * c;
                    }
                }
            }
        };
        match pool {
            Some(pl) => pl.run_rows(n, &work),
            None => work(0, n),
        }
    }

    /// Full bidirectional attention over the packed sequence.
    fn attention(&self, qkv: &[f32], n: usize, attn: &mut [f32]) {
        let (nh, hd) = (self.heads, self.head_dim);
        let inner = nh * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let pool = self.pool.as_deref();
        let mut qh = vec![0f32; n * hd];
        let mut kh = vec![0f32; n * hd];
        let mut vt = vec![0f32; hd * n];
        let mut scores = vec![0f32; n * n];
        let mut oh = vec![0f32; n * hd];
        for h in 0..nh {
            for p in 0..n {
                let base = p * 3 * inner;
                let qs = &qkv[base + h * hd..base + (h + 1) * hd];
                for (d, &val) in qs.iter().enumerate() {
                    qh[p * hd + d] = val * scale;
                }
                kh[p * hd..(p + 1) * hd]
                    .copy_from_slice(&qkv[base + inner + h * hd..base + inner + (h + 1) * hd]);
                let vs = &qkv[base + 2 * inner + h * hd..base + 2 * inner + (h + 1) * hd];
                for (d, &val) in vs.iter().enumerate() {
                    vt[d * n + p] = val;
                }
            }
            crate::fcd_ops::gemm_nt(&qh, &kh, &mut scores, n, hd, n, pool);
            let sp = SendPtr(scores.as_mut_ptr());
            let soft = |lo: usize, hi: usize| {
                for r in lo..hi {
                    // SAFETY: workers own disjoint score rows.
                    softmax_inplace(unsafe { sp.row(r * n, n) });
                }
            };
            match pool {
                Some(pl) => pl.run_rows(n, &soft),
                None => soft(0, n),
            }
            crate::fcd_ops::gemm_nt(&scores, &vt, &mut oh, n, n, hd, pool);
            for p in 0..n {
                attn[p * inner + h * hd..p * inner + (h + 1) * hd]
                    .copy_from_slice(&oh[p * hd..(p + 1) * hd]);
            }
        }
    }

    /// One block. `mods` is the block's modulation buffer and `rows` the
    /// per-token row index into it; both empty for the refiner, which is
    /// unmodulated and unrotated.
    fn block_forward(
        &self,
        blk: &Block,
        x: &mut [f32],
        n: usize,
        mods: Option<&[f32]>,
        pos: &[[f64; 3]],
        rows: &[u32],
    ) {
        let hs = self.hidden;
        let pool = self.pool.as_deref();
        let inner = self.heads * self.head_dim;
        let angles = if pos.is_empty() {
            Vec::new()
        } else {
            self.rope_angles(pos)
        };

        let mut xn = vec![0f32; n * hs];
        self.norm_rows(&mut xn, x, n, hs, &blk.norm1, self.eps);
        if let (Some(m), false) = (mods, rows.is_empty()) {
            self.modulate(&mut xn, hs, m, rows, 0, 1);
        }
        let mut qkv = vec![0f32; n * 3 * inner];
        blk.qkv.matmat(&xn, n, &mut qkv, pool);
        // q and k are the first two thirds of every row; normalize and
        // rotate them where they lie, leaving v alone.
        for (which, w) in [(0usize, &blk.q_norm), (1usize, &blk.k_norm)] {
            self.norm_rope_w(&mut qkv, n, self.heads, w, &angles, 3 * inner, which * inner);
        }
        let mut attn = vec![0f32; n * inner];
        self.attention(&qkv, n, &mut attn);
        let mut proj = vec![0f32; n * hs];
        blk.out.matmat(&attn, n, &mut proj, pool);
        self.residual(x, hs, &proj, mods, rows, 2);

        self.norm_rows(&mut xn, x, n, hs, &blk.norm2, self.eps);
        if let (Some(m), false) = (mods, rows.is_empty()) {
            self.modulate(&mut xn, hs, m, rows, 3, 4);
        }
        let mut gu = vec![0f32; n * 2 * self.ffn];
        blk.fc1.matmat(&xn, n, &mut gu, pool);
        // SwiGLU: fc1's output is [gate | up] per row.
        let ffn = self.ffn;
        let mut act = vec![0f32; n * ffn];
        let ap = SendPtr(act.as_mut_ptr());
        pool_rows(pool, n, &|lo, hi| {
            for p in lo..hi {
                let row = &gu[p * 2 * ffn..(p + 1) * 2 * ffn];
                let (g, up) = row.split_at(ffn);
                // SAFETY: workers own disjoint token ranges.
                for (o, (&a, &b)) in unsafe { ap.row(p * ffn, ffn) }
                    .iter_mut()
                    .zip(g.iter().zip(up))
                {
                    *o = silu(a) * b;
                }
            }
        });
        blk.fc2.matmat(&act, n, &mut proj, pool);
        self.residual(x, hs, &proj, mods, rows, 5);
    }

    /// RMSNorm every row of `src` into `dst`, across the pool. One
    /// block does this four times over `n·hidden`; on a 1 879-token
    /// pack that is 40 M elements a block, and it was running on one
    /// thread while forty-seven sat idle.
    fn norm_rows(&self, dst: &mut [f32], src: &[f32], n: usize, hs: usize, w: &[f32], eps: f64) {
        let ptr = SendPtr(dst.as_mut_ptr());
        pool_rows(self.pool.as_deref(), n, &|lo, hi| {
            for p in lo..hi {
                // SAFETY: workers own disjoint token ranges.
                rms_norm_into(&src[p * hs..(p + 1) * hs], w, eps, unsafe {
                    ptr.row(p * hs, hs)
                });
            }
        });
    }

    /// `x = x·(1 + scale[row]) + shift[row]`, per token, across the pool.
    fn modulate(
        &self,
        x: &mut [f32],
        hs: usize,
        mods: &[f32],
        rows: &[u32],
        shift_e: usize,
        scale_e: usize,
    ) {
        let stride = EXPAND * hs;
        let ptr = SendPtr(x.as_mut_ptr());
        pool_rows(self.pool.as_deref(), rows.len(), &|lo, hi| {
            for p in lo..hi {
                let base = rows[p] as usize * stride;
                let shift = &mods[base + shift_e * hs..base + (shift_e + 1) * hs];
                let scale = &mods[base + scale_e * hs..base + (scale_e + 1) * hs];
                // SAFETY: workers own disjoint token ranges.
                for ((v, &sc), &sh) in unsafe { ptr.row(p * hs, hs) }
                    .iter_mut()
                    .zip(scale)
                    .zip(shift)
                {
                    *v = *v * (1.0 + sc) + sh;
                }
            }
        });
    }

    /// The gated residual — or a plain one where there is no modulation
    /// (the token refiner).
    fn residual(
        &self,
        x: &mut [f32],
        hs: usize,
        other: &[f32],
        mods: Option<&[f32]>,
        rows: &[u32],
        gate_e: usize,
    ) {
        let n = x.len() / hs;
        let stride = EXPAND * hs;
        let ptr = SendPtr(x.as_mut_ptr());
        let gated = mods.filter(|_| !rows.is_empty());
        pool_rows(self.pool.as_deref(), n, &|lo, hi| {
            for p in lo..hi {
                // SAFETY: workers own disjoint token ranges.
                let row = unsafe { ptr.row(p * hs, hs) };
                let src = &other[p * hs..(p + 1) * hs];
                match gated {
                    Some(m) => {
                        let base = rows[p] as usize * stride;
                        let gate = &m[base + gate_e * hs..base + (gate_e + 1) * hs];
                        for ((v, &g), &o) in row.iter_mut().zip(gate).zip(src) {
                            *v += g * o;
                        }
                    }
                    None => {
                        for (v, &o) in row.iter_mut().zip(src) {
                            *v += o;
                        }
                    }
                }
            }
        });
    }

    /// One denoise evaluation.
    ///
    /// `video` is `[latents_dim, latent_t, lat_h, lat_w]` and `audio` is
    /// `[audio_dim, 2, audio_t]`, both in the reference's channel-major
    /// order. `text` is the refined `[n, hidden]` stream. Returns the
    /// two velocities, EACH ON ITS OWN SCHEDULE and unscaled.
    pub fn forward(
        &self,
        layout: &Layout,
        text: &[f32],
        video: &[f32],
        audio: &[f32],
        sigma_v: f64,
    ) -> (Vec<f32>, Vec<f32>) {
        let hs = self.hidden;
        let pool = self.pool.as_deref();
        let sigma_v = sigma_v.max(1e-6);
        let t_v = 1.0 - sigma_v;
        let t_a = 1.0 - time_shift_sigma(sigma_v, self.shift_video, self.shift_audio);

        // Distinct timesteps, sorted — the adaLN row index is a position
        // in this list, so the order is part of the contract.
        let mut ts = vec![t_v, t_a];
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts.dedup();
        let row_of = |t: f64| ts.iter().position(|&x| x == t).unwrap();
        let (row_v, row_a) = (row_of(t_v), row_of(t_a));

        // Per-token modulation row: t_row · MODALITIES + tag.
        let mut rows = vec![0u32; layout.seq_len];
        for s in &layout.segments {
            let r = match s.kind {
                Kind::Text => row_v * MODALITIES + TAG_TEXT,
                Kind::Video => row_v * MODALITIES + TAG_VIDEO,
                Kind::Audio => row_a * MODALITIES + TAG_AUDIO,
            };
            for v in rows[s.start..s.stop].iter_mut() {
                *v = r as u32;
            }
        }

        // ── embed ──
        let v_rows = patchify_video(video, self.latents_dim, layout.latent_t, layout.lat_h, layout.lat_w);
        let v_n = v_rows.len() / (self.latents_dim * 4);
        let a_rows = pack_audio(audio, self.audio_dim, layout.audio_t);

        let mut h = vec![0f32; layout.seq_len * hs];
        let vseg = layout.segment(Kind::Video);
        let aseg = layout.segment(Kind::Audio);
        let tseg = layout.segment(Kind::Text);
        h[tseg.start * hs..tseg.stop * hs].copy_from_slice(&text[..(tseg.stop - tseg.start) * hs]);
        self.video_patch.matmat(&v_rows, v_n, &mut h[vseg.start * hs..vseg.stop * hs], pool);
        self.audio_patch.matmat(
            &a_rows,
            aseg.stop - aseg.start,
            &mut h[aseg.start * hs..aseg.stop * hs],
            pool,
        );
        for row in h[vseg.start * hs..vseg.stop * hs].chunks_exact_mut(hs) {
            for (v, &b) in row.iter_mut().zip(&self.video_patch_b) {
                *v += b;
            }
        }
        for row in h[aseg.start * hs..aseg.stop * hs].chunks_exact_mut(hs) {
            for (v, &b) in row.iter_mut().zip(&self.audio_patch_b) {
                *v += b;
            }
        }

        // ── blocks ──
        for (i, blk) in self.blocks.iter().enumerate() {
            let mods = blk.adaln.as_ref().unwrap().eval(&ts, pool);
            self.block_forward(blk, &mut h, layout.seq_len, Some(&mods), &layout.pos, &rows);
            if std::env::var_os("CMF_DIT_PROGRESS").is_some() {
                eprint!("\r  block {}/{}", i + 1, self.blocks.len());
            }
        }

        // ── heads ──
        let fm = self.final_adaln.eval(&ts, pool);
        let vd = self.latents_dim * 4;
        let mut video_out = vec![0f32; (vseg.stop - vseg.start) * vd];
        let mut audio_out = vec![0f32; (aseg.stop - aseg.start) * self.audio_dim];
        for (seg, row, w, b, dst, dim) in [
            (vseg, row_v, &self.video_out, &self.video_out_b, &mut video_out, vd),
            (aseg, row_a, &self.audio_out, &self.audio_out_b, &mut audio_out, self.audio_dim),
        ] {
            let n = seg.stop - seg.start;
            let mut hn = vec![0f32; n * hs];
            for (o, src) in hn.chunks_exact_mut(hs).zip(h[seg.start * hs..seg.stop * hs].chunks_exact(hs)) {
                rms_norm_into(src, &self.final_norm, self.final_eps, o);
            }
            // The final layer's adaLN has one modality, so the row IS
            // the timestep index.
            let shift = &fm[row * 2 * hs..row * 2 * hs + hs];
            let scale = &fm[row * 2 * hs + hs..(row + 1) * 2 * hs];
            for r in hn.chunks_exact_mut(hs) {
                for ((v, &sc), &sh) in r.iter_mut().zip(scale).zip(shift) {
                    *v = *v * (1.0 + sc) + sh;
                }
            }
            w.matmat(&hn, n, dst, pool);
            for r in dst.chunks_exact_mut(dim) {
                for (v, &bv) in r.iter_mut().zip(b.iter()) {
                    *v += bv;
                }
            }
        }

        // The reference predicts toward the data and the sampler steps
        // σ down, hence the sign; the audio velocity is returned on its
        // OWN clock rather than pre-scaled by d(σ_a)/d(σ_v).
        let video = unpatchify_video(
            &video_out,
            self.latents_dim,
            layout.latent_t,
            layout.lat_h,
            layout.lat_w,
        );
        let audio = unpack_audio(&audio_out, self.audio_dim, layout.audio_t);
        (
            video.iter().map(|&v| -v).collect(),
            audio.iter().map(|&v| -v).collect(),
        )
    }
}

// ── modulation helpers ──────────────────────────────────────────────

/// Rows of `n` items split across pool workers (serial without a pool).
fn pool_rows(pool: Option<&Pool>, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) => p.run_rows(n, f),
        None => f(0, n),
    }
}

// ── stream (un)packing ──────────────────────────────────────────────

/// `[C, T, H, W]` → `[T·(H/2)·(W/2), C·4]`, the 2×2 spatial patch
/// flattened channel-major-outer as `einsum("nctrhpwq->nthwcrpq")`.
pub fn patchify_video(x: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let (ph, pw) = (h / 2, w / 2);
    let mut out = vec![0f32; t * ph * pw * c * 4];
    let mut i = 0;
    for ti in 0..t {
        for hi in 0..ph {
            for wi in 0..pw {
                for ci in 0..c {
                    for p in 0..2 {
                        for q in 0..2 {
                            out[i] = x[((ci * t + ti) * h + hi * 2 + p) * w + wi * 2 + q];
                            i += 1;
                        }
                    }
                }
            }
        }
    }
    out
}

/// The inverse of `patchify_video`.
pub fn unpatchify_video(rows: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let (ph, pw) = (h / 2, w / 2);
    let mut out = vec![0f32; c * t * h * w];
    let mut i = 0;
    for ti in 0..t {
        for hi in 0..ph {
            for wi in 0..pw {
                for ci in 0..c {
                    for p in 0..2 {
                        for q in 0..2 {
                            out[((ci * t + ti) * h + hi * 2 + p) * w + wi * 2 + q] = rows[i];
                            i += 1;
                        }
                    }
                }
            }
        }
    }
    out
}

/// `[C, 2, T]` → `[2·T, C]`, channel-major: channel 0's frames then
/// channel 1's.
pub fn pack_audio(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0f32; 2 * t * c];
    for ch in 0..2 {
        for ti in 0..t {
            for ci in 0..c {
                out[(ch * t + ti) * c + ci] = x[(ci * 2 + ch) * t + ti];
            }
        }
    }
    out
}

/// The inverse of `pack_audio`.
pub fn unpack_audio(rows: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0f32; c * 2 * t];
    for ch in 0..2 {
        for ti in 0..t {
            for ci in 0..c {
                out[(ci * 2 + ch) * t + ti] = rows[(ch * t + ti) * c + ci];
            }
        }
    }
    out
}

// ── small shared bits ───────────────────────────────────────────────

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

fn softmax_inplace(row: &mut [f32]) {
    let mx = row.iter().cloned().fold(f32::MIN, f32::max);
    let mut den = 0f32;
    for r in row.iter_mut() {
        *r = (*r - mx).exp();
        den += *r;
    }
    if den > 0.0 {
        let inv = 1.0 / den;
        for r in row.iter_mut() {
            *r *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_remap_is_an_involution() {
        for &s in &[1e-3, 0.25, 0.5, 0.8, 0.972_973, 1.0] {
            let a = time_shift_sigma(s, 12.0, 3.0);
            let back = time_shift_sigma(a, 3.0, 12.0);
            assert!((back - s).abs() < 1e-9, "{s} -> {a} -> {back}");
        }
    }

    #[test]
    fn slope_matches_a_finite_difference() {
        let h = 1e-6;
        for &s in &[0.2, 0.5, 0.9] {
            let num = (time_shift_sigma(s + h, 12.0, 3.0) - time_shift_sigma(s - h, 12.0, 3.0))
                / (2.0 * h);
            let got = time_shift_slope(s, 12.0, 3.0);
            assert!((num - got).abs() < 1e-5, "{s}: {num} vs {got}");
        }
    }

    #[test]
    fn patchify_round_trips() {
        let (c, t, h, w) = (3usize, 2usize, 4usize, 6usize);
        let x: Vec<f32> = (0..c * t * h * w).map(|i| i as f32).collect();
        let rows = patchify_video(&x, c, t, h, w);
        assert_eq!(rows.len(), t * (h / 2) * (w / 2) * c * 4);
        assert_eq!(unpatchify_video(&rows, c, t, h, w), x);
    }

    #[test]
    fn audio_pack_round_trips() {
        let (c, t) = (5usize, 7usize);
        let x: Vec<f32> = (0..c * 2 * t).map(|i| i as f32).collect();
        let rows = pack_audio(&x, c, t);
        assert_eq!(unpack_audio(&rows, c, t), x);
    }

    #[test]
    fn layout_places_the_target_streams_last() {
        let l = Layout::t2va(8, 3, 8, 12, 5);
        assert_eq!(l.frame_rows, 4 * 6);
        assert_eq!(l.seq_len, 8 + 2 * 5 + 3 * 24);
        assert_eq!(l.segments[0].kind, Kind::Text);
        assert_eq!(l.segments[1].kind, Kind::Audio);
        assert_eq!(l.segments[2].kind, Kind::Video);
        // The video t axis advances by the 1,4,4,4,4 span pattern.
        let v = l.segment(Kind::Video);
        let t0 = l.pos[v.start][0];
        let t1 = l.pos[v.start + l.frame_rows][0];
        assert!((t1 - t0 - FRAME_RESCALE).abs() < 1e-12);
    }
}
