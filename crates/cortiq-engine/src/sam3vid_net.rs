//! SAM3 video — the tracker network: everything that reads the CMF's
//! `trk.*` section (309 tensors) plus the tracker's own FPN neck
//! (`fpn.sam2_convs`, byte-identical to HF's `tracker_neck`).
//!
//! Reference: transformers 5.20.2 `Sam3TrackerVideoModel` (mask decoder,
//! prompt encoder, memory encoder/attention) and `Sam3VisionNeck`. Shapes
//! and op order follow the reference exactly; batch size is always 1.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use cortiq_core::CmfModel;

use crate::inference::gelu_erf;
use crate::pool::Pool;
use crate::qtensor::QTensor;
use crate::sam3;

use crate::sam3vid::{MASK_SIDE, VidCfg};

/// Mask-decoder feature grid (1008 / 14).
pub(crate) const EMB: usize = 72;
pub(crate) const HID: usize = 256;
/// Memory channel count (`memory_encoder_output_channels`).
pub(crate) const MEM: usize = 64;
/// Neck levels the tracker consumes (288/144/72).
pub(crate) const TRK_LEVELS: [usize; 3] = [MASK_SIDE, 144, EMB];
/// The mask replaced by this logit when the object is predicted absent.
pub(crate) const NO_OBJ_SCORE: f32 = -1024.0;

pub(crate) fn sigm(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn relu_vec(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x.max(0.0)).collect()
}

/// `y = x·Wᵀ + b` with `W` stored `[out, in]`.
pub(crate) fn dense(w: &QTensor, b: &[f32], x: &[f32], rows: usize, pool: Option<&Pool>) -> Vec<f32> {
    let out_dim = w.rows();
    debug_assert_eq!(out_dim, b.len());
    let mut out = vec![0f32; rows * out_dim];
    w.matmat(x, rows, &mut out, pool);
    sam3::add_bias(&mut out, out_dim, b);
    out
}

/// LayerNorm over the channel axis of `[c, h, w]` (SAM channels_first LN).
pub(crate) fn ln_chw(x: &mut [f32], c: usize, hw: usize, w: &[f32], b: &[f32], eps: f32) {
    debug_assert_eq!(w.len(), c);
    debug_assert_eq!(b.len(), c);
    for pos in 0..hw {
        let mut mean = 0f32;
        for ch in 0..c {
            mean += x[ch * hw + pos];
        }
        mean /= c as f32;
        let mut var = 0f32;
        for ch in 0..c {
            let d = x[ch * hw + pos] - mean;
            var += d * d;
        }
        var /= c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for ch in 0..c {
            let i = ch * hw + pos;
            x[i] = (x[i] - mean) * inv * w[ch] + b[ch];
        }
    }
}

/// Bilinear resample of a single-channel map, `F.interpolate(...,
/// align_corners=False)` (no antialias: every resample here is an upscale).
pub(crate) fn resize2d(x: &[f32], h: usize, w: usize, sh: usize, sw: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), h * w, "resize2d source {h}x{w} but {} values", x.len());
    if x.len() != h * w {
        panic!("resize2d: source {h}x{w} != {} values", x.len());
    }
    if h == sh && w == sw {
        return x.to_vec();
    }
    let xs = w as f32 / sw as f32;
    let ys = h as f32 / sh as f32;
    let x_off = |o: f32| (o + 0.5) * xs - 0.5;
    let y_off = |o: f32| (o + 0.5) * ys - 0.5;
    let mut out = vec![0f32; sh * sw];
    for oy in 0..sh {
        let fy = y_off(oy as f32);
        let y0 = fy.floor();
        let y1 = y0 + 1.0;
        let wy = fy - y0;
        let iy0 = y0.clamp(0.0, (h - 1) as f32) as usize;
        let iy1 = y1.clamp(0.0, (h - 1) as f32) as usize;
        for ox in 0..sw {
            let fx = x_off(ox as f32);
            let x0 = fx.floor();
            let x1 = x0 + 1.0;
            let wx = fx - x0;
            let ix0 = x0.clamp(0.0, (w - 1) as f32) as usize;
            let ix1 = x1.clamp(0.0, (w - 1) as f32) as usize;
            let a = x[iy0 * w + ix0] * (1.0 - wx) + x[iy0 * w + ix1] * wx;
            let b = x[iy1 * w + ix0] * (1.0 - wx) + x[iy1 * w + ix1] * wx;
            out[oy * sw + ox] = a * (1.0 - wy) + b * wy;
        }
    }
    out
}

/// Rotate the first `rows` rows of `buf` (row stride `stride`, `head_dim`
/// wide) against `cos`/`sin` of shape `[seq, head_dim]`. Cross-attention
/// repeats the frequencies for every spatial memory frame; trailing object
/// pointers are excluded by the caller.
fn rope_rows(buf: &mut [f32], rows: usize, stride: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    for i in 0..rows {
        let base = (i * head_dim) % cos.len();
        let row_start = i * stride;
        for p in 0..head_dim / 2 {
            let (c, s) = (cos[base + p * 2], sin[base + p * 2]);
            let (a, d) = (buf[row_start + p * 2], buf[row_start + p * 2 + 1]);
            buf[row_start + p * 2] = a * c - d * s;
            buf[row_start + p * 2 + 1] = d * c + a * s;
        }
    }
}

/// One frame's tracker features: neck levels 0/1 after `conv_s0`/`conv_s1`
/// plus the raw level-2 map — all channels-first.
pub(crate) struct TrkFeats {
    pub l0: Vec<f32>,
    pub l1: Vec<f32>,
    pub l2: Vec<f32>,
}

/// Everything reading `trk.*` / `fpn.sam2_convs`.
pub(crate) struct Net<'a> {
    pub m: &'a Arc<CmfModel>,
    pub pool: Option<&'a Pool>,
    pub cfg: &'a VidCfg,
}

impl<'a> Net<'a> {
    pub fn w(&self, name: &str) -> Result<QTensor> {
        sam3::lin(self.m, name)
    }
    pub fn v(&self, name: &str) -> Result<Vec<f32>> {
        sam3::load_vec(self.m, name)
    }
    pub(crate) fn lin(&self, prefix: &str, what: &str, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let w = self.w(&format!("{prefix}{what}.weight"))?;
        let b = self.v(&format!("{prefix}{what}.bias"))?;
        Ok(dense(&w, &b, x, rows, self.pool))
    }
    pub(crate) fn ln_rows(&self, x: &mut [f32], w: &str, b: &str, eps: f32) -> Result<()> {
        let ww = self.v(w)?;
        let bb = self.v(b)?;
        sam3::layer_norm_rows(x, &ww, &bb, eps);
        Ok(())
    }

    /// Multi-head attention with separate q/k/v/out projections
    /// (`{prefix}q_proj.weight` …). `rope = (cos, sin, n_k_rotated)` rotates
    /// every query row and the first `n_k_rotated` key rows (object pointers
    /// stay unrotated — `apply_rotary_pos_emb_2d(num_k_exclude_rope=…)`).
    #[allow(clippy::too_many_arguments)]
    pub fn attn(
        &self,
        prefix: &str,
        xs_q: &[f32],
        xs_k: &[f32],
        xs_v: &[f32],
        nq: usize,
        nk: usize,
        q_in: usize,
        kv_in: usize,
        int_dim: usize,
        rope: Option<(&[f32], &[f32], usize)>,
        out: &mut [f32],
    ) -> Result<()> {
        let wq = self.w(&format!("{prefix}q_proj.weight"))?;
        let bq = self.v(&format!("{prefix}q_proj.bias"))?;
        let wk = self.w(&format!("{prefix}k_proj.weight"))?;
        let bk = self.v(&format!("{prefix}k_proj.bias"))?;
        let wv = self.w(&format!("{prefix}v_proj.weight"))?;
        let bv = self.v(&format!("{prefix}v_proj.bias"))?;
        let wo = self.w(&format!("{prefix}out_proj.weight"))?;
        let ob = self.v(&format!("{prefix}out_proj.bias"))?;
        debug_assert_eq!(wq.cols(), q_in, "{prefix}q_proj in-dim");
        debug_assert_eq!(wk.cols(), kv_in, "{prefix}k_proj in-dim");
        debug_assert_eq!(wv.cols(), kv_in, "{prefix}v_proj in-dim");
        debug_assert_eq!(bq.len(), int_dim);
        debug_assert_eq!(bk.len(), int_dim);
        debug_assert_eq!(bv.len(), int_dim);
        debug_assert_eq!(wo.cols(), int_dim);
        debug_assert_eq!(ob.len(), wo.rows());
        debug_assert_eq!(out.len(), nq * wo.rows());

        let mut q = vec![0f32; nq * int_dim];
        wq.matmat(xs_q, nq, &mut q, self.pool);
        sam3::add_bias(&mut q, int_dim, &bq);
        let mut k = vec![0f32; nk * int_dim];
        wk.matmat(xs_k, nk, &mut k, self.pool);
        sam3::add_bias(&mut k, int_dim, &bk);
        let mut v = vec![0f32; nk * int_dim];
        wv.matmat(xs_v, nk, &mut v, self.pool);
        sam3::add_bias(&mut v, int_dim, &bv);

        let n_heads = self.n_heads(prefix)?;
        let hd = int_dim / n_heads;
        // [DEBUG-cross] dump projections before/after rope for the cross layer
        let dbg_dir = std::env::var("SAM3_ATTN_DUMP").ok().filter(|_| {
            prefix == "trk.transformer.encoder.layers.0.cross_attn_image."
        });
        let dbgw = |name: &str, v: &[f32]| {
            if let Some(d) = &dbg_dir {
                let _ = std::fs::create_dir_all(d);
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{d}/{name}.f32"), b);
            }
        };
        dbgw("q_pre", &q);
        dbgw("k_pre", &k);
        dbgw("v_pre", &v);
        if let Some((cos, sin, nk_rot)) = rope {
            rope_rows(&mut q, nq, int_dim, hd, cos, sin);
            rope_rows(&mut k, nk_rot.min(nk), int_dim, hd, cos, sin);
        }
        dbgw("q_rope", &q);
        dbgw("k_rope", &k);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut res = vec![0f32; nq * int_dim];
        let rb = res.as_mut_ptr() as usize;
        let run = |s: usize, e: usize| {
            let res = unsafe { std::slice::from_raw_parts_mut(rb as *mut f32, nq * int_dim) };
            let mut scores = vec![0f32; nk];
            for i in s..e {
                for t in 0..n_heads {
                    let qr = &q[i * int_dim + t * hd..i * int_dim + (t + 1) * hd];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..nk {
                        let kr = &k[j * int_dim + t * hd..j * int_dim + (t + 1) * hd];
                        let mut acc = 0f32;
                        for (a, b) in qr.iter().zip(kr) {
                            acc += *a * *b;
                        }
                        let val = acc * scale;
                        scores[j] = val;
                        if val > mx {
                            mx = val;
                        }
                    }
                    let mut sum = 0f32;
                    for j in 0..nk {
                        scores[j] = (scores[j] - mx).exp();
                        sum += scores[j];
                    }
                    let inv = 1.0 / sum;
                    let orow = &mut res[i * int_dim + t * hd..i * int_dim + (t + 1) * hd];
                    for j in 0..nk {
                        let wgt = scores[j] * inv;
                        if wgt == 0.0 {
                            continue;
                        }
                        let vr = &v[j * int_dim + t * hd..j * int_dim + (t + 1) * hd];
                        for (o, vv) in orow.iter_mut().zip(vr) {
                            *o += wgt * *vv;
                        }
                    }
                }
            }
        };
        match self.pool {
            Some(p) => p.run_rows(nq, &run),
            None => run(0, nq),
        }
        wo.matmat(&res, nq, out, self.pool);
        sam3::add_bias(out, wo.rows(), &ob);
        Ok(())
    }

    /// Attention head count: 8 for the SAM mask decoder, 1 for the memory
    /// attention (`memory_attention_num_attention_heads`).
    fn n_heads(&self, prefix: &str) -> Result<usize> {
        // memory attention: memory_attention_num_attention_heads = 1
        if prefix.starts_with("trk.transformer.encoder") {
            Ok(1)
        } else {
            Ok(8) // mask decoder: mask_decoder_config.num_attention_heads
        }
    }

    // ── features ───────────────────────────────────────────────────

    /// Tracker neck over the detector's trunk map + `conv_s0`/`conv_s1`.
    pub fn feats(&self, trunk: &[f32]) -> Result<TrkFeats> {
        let levels = sam3::fpn_levels(self.m, "fpn.sam2_convs", trunk, self.pool, 3)?;
        let conv = |name: &str, x: &[f32], side: usize| -> Result<Vec<f32>> {
            let w = self.v(&format!("trk.sam_mask_decoder.{name}.weight"))?;
            let b = self.v(&format!("trk.sam_mask_decoder.{name}.bias"))?;
            Ok(sam3::conv2d(&w, &b, x, HID, side, side, 1, 1, 1, 0, self.pool))
        };
        Ok(TrkFeats {
            l0: conv("conv_s0", &levels[0], TRK_LEVELS[0])?,
            l1: conv("conv_s1", &levels[1], TRK_LEVELS[1])?,
            l2: levels[2].clone(),
        })
    }

    /// Level-2 map as `(HW, C)` rows — the memory encoder's `pix_feat`.
    pub fn pix_rows(f: &TrkFeats) -> Vec<f32> {
        sam3::chw_to_rows(&f.l2, EMB, EMB)
    }

    /// `(HW, C)` sine position rows of the level-2 map — the neck's
    /// `fpn_position_encoding`, which conditions the MEMORY ATTENTION
    /// (`Sam3VisionNeck` → `Sam3SinePositionEmbedding`).
    pub fn neck_pos() -> Vec<f32> {
        sam3::sine_pos(EMB, EMB, HID / 2)
    }

    /// The mask decoder's image-wide positional embedding: NOT the neck's
    /// sine PE but the prompt encoder's 2-D Gaussian encoding
    /// (`get_image_wide_positional_embeddings` → `shared_image_embedding`,
    /// the `pe_layer.positional_encoding_gaussian_matrix` checkpoint).
    /// Coordinates are normalized pixel centres `(x+0.5)/72`,
    /// `(y+0.5)/72`, shifted to [-1,1], projected by the [2,128] matrix,
    /// scaled by 2π and split into sin/cos.
    pub fn image_pos(&self) -> Result<Vec<f32>> {
        let w = self.v("trk.sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix")?;
        anyhow::ensure!(w.len() == 2 * 128, "gaussian PE matrix {}", w.len());
        let n = EMB * EMB;
        let mut out = vec![0f32; n * HID];
        for y in 0..EMB {
            for x in 0..EMB {
                let cx = 2.0 * ((x as f32 + 0.5) / EMB as f32) - 1.0;
                let cy = 2.0 * ((y as f32 + 0.5) / EMB as f32) - 1.0;
                let base = (y * EMB + x) * HID;
                for i in 0..128 {
                    let v = 2.0 * std::f32::consts::PI * (cx * w[i] + cy * w[128 + i]);
                    out[base + i] = v.sin();
                    out[base + 128 + i] = v.cos();
                }
            }
        }
        Ok(out)
    }

    // ── prompt encoder ─────────────────────────────────────────────

    /// Dense prompt from a 288² mask (`sam_prompt_encoder.mask_downscaling`:
    /// conv1 1→4 s2, LN, GELU, conv2 4→16 s2, LN, GELU, conv3 16→256).
    pub fn mask_prompt(&self, mask: &[f32]) -> Result<Vec<f32>> {
        let p = "trk.sam_prompt_encoder.mask_downscaling.";
        anyhow::ensure!(mask.len() == MASK_SIDE * MASK_SIDE, "mask prompt must be {MASK_SIDE}²");
        let mut x = mask.to_vec();
        let mut c_in = 1usize;
        let mut side = MASK_SIDE;
        for (step, ch) in [(0usize, 4usize), (3, 16), (6, 256)] {
            let w = self.v(&format!("{p}{step}.weight"))?;
            let b = self.v(&format!("{p}{step}.bias"))?;
            let kh = if step == 6 { 1 } else { 2 };
            let stride = if step == 6 { 1 } else { 2 };
            x = sam3::conv2d(&w, &b, &x, c_in, side, side, kh, kh, stride, 0, self.pool);
            side = side / stride;
            c_in = ch;
            if step != 6 {
                let lw = self.v(&format!("{p}{}.weight", step + 1))?;
                let lb = self.v(&format!("{p}{}.bias", step + 1))?;
                ln_chw(&mut x, ch, side * side, &lw, &lb, 1e-6);
                x.iter_mut().for_each(|v| *v = gelu_erf(*v));
            }
        }
        Ok(x)
    }

    /// Dense prompt with no mask input: `no_mask_embed` tiled over the grid.
    pub fn no_mask_prompt(&self) -> Result<Vec<f32>> {
        let e = self.v("trk.sam_prompt_encoder.no_mask_embed.weight")?;
        anyhow::ensure!(e.len() == HID, "no_mask_embed");
        Ok(e.repeat(EMB * EMB))
    }

    /// Sparse prompt when there are no points: two padded points with label
    /// -1, both replaced by `not_a_point_embed`.
    pub fn empty_sparse(&self) -> Result<Vec<f32>> {
        let e = self.v("trk.sam_prompt_encoder.not_a_point_embed.weight")?;
        anyhow::ensure!(e.len() == HID, "not_a_point_embed");
        Ok([e.clone(), e].concat())
    }

    // ── mask decoder ───────────────────────────────────────────────

    /// The SAM mask decoder. Returns `(masks[n·288²] (logits), iou[n],
    /// obj_score_logit, tokens[n·256])`.
    ///
    /// `pix` is the (possibly memory-conditioned) level-2 map, `dense` the
    /// dense prompt, `high` the conv_s0/conv_s1 skip features.
    #[allow(clippy::too_many_arguments)]
    pub fn mask_decoder(
        &self,
        pix: &[f32],
        pos_rows: &[f32],
        dense: &[f32],
        sparse: &[f32],
        high: (&[f32], &[f32]),
        multimask: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, f32, Vec<f32>)> {
        let db = "trk.sam_mask_decoder.";
        let n_sparse = sparse.len() / HID;
        let mut tokens: Vec<f32> = Vec::with_capacity((6 + n_sparse) * HID);
        for name in ["obj_score_token", "iou_token"] {
            tokens.extend_from_slice(&self.v(&format!("{db}{name}.weight"))?);
        }
        tokens.extend_from_slice(&self.v(&format!("{db}mask_tokens.weight"))?);
        tokens.extend_from_slice(sparse);
        let n_tok = tokens.len() / HID;
        let pe = tokens.clone();

        let mut img = pix.to_vec();
        for (a, b) in img.iter_mut().zip(dense) {
            *a += *b;
        }
        let img_rows = sam3::chw_to_rows(&img, EMB, EMB);
        let n_img = EMB * EMB;

        let tb = format!("{db}transformer.");
        let mut queries = tokens;
        let mut keys = img_rows;
        for layer in 0..2 {
            let lb = format!("{tb}layers.{layer}.");
            // self attention; layer 0 skips the query PE and has no residual
            if layer == 0 {
                let mut ao = vec![0f32; n_tok * HID];
                self.attn(&format!("{lb}self_attn."), &queries, &queries, &queries, n_tok, n_tok, HID, HID, HID, None, &mut ao)?;
                queries = ao;
            } else {
                let q: Vec<f32> = queries.iter().zip(&pe).map(|(a, b)| a + b).collect();
                let mut ao = vec![0f32; n_tok * HID];
                self.attn(&format!("{lb}self_attn."), &q, &q, &queries, n_tok, n_tok, HID, HID, HID, None, &mut ao)?;
                for (a, c) in queries.iter_mut().zip(&ao) {
                    *a += *c;
                }
            }
            self.ln_rows(&mut queries, &format!("{lb}norm1.weight"), &format!("{lb}norm1.bias"), 1e-6)?;

            // tokens → image
            let q: Vec<f32> = queries.iter().zip(&pe).map(|(a, b)| a + b).collect();
            let k: Vec<f32> = keys.iter().zip(pos_rows).map(|(a, b)| a + b).collect();
            let mut ao = vec![0f32; n_tok * HID];
            self.attn(&format!("{lb}cross_attn_token_to_image."), &q, &k, &keys, n_tok, n_img, HID, HID, 128, None, &mut ao)?;
            for (a, c) in queries.iter_mut().zip(&ao) {
                *a += *c;
            }
            self.ln_rows(&mut queries, &format!("{lb}norm2.weight"), &format!("{lb}norm2.bias"), 1e-6)?;

            // MLP (the two-way block norms AFTER the residual)
            let h = relu_vec(&self.lin(&format!("{lb}mlp."), "lin1", &queries, n_tok)?);
            let h = self.lin(&format!("{lb}mlp."), "lin2", &h, n_tok)?;
            for (a, c) in queries.iter_mut().zip(&h) {
                *a += *c;
            }
            self.ln_rows(&mut queries, &format!("{lb}norm3.weight"), &format!("{lb}norm3.bias"), 1e-6)?;

            // image → tokens
            let q: Vec<f32> = keys.iter().zip(pos_rows).map(|(a, b)| a + b).collect();
            let k: Vec<f32> = queries.iter().zip(&pe).map(|(a, b)| a + b).collect();
            let mut ao = vec![0f32; n_img * HID];
            self.attn(&format!("{lb}cross_attn_image_to_token."), &q, &k, &queries, n_img, n_tok, HID, HID, 128, None, &mut ao)?;
            for (a, c) in keys.iter_mut().zip(&ao) {
                *a += *c;
            }
            self.ln_rows(&mut keys, &format!("{lb}norm4.weight"), &format!("{lb}norm4.bias"), 1e-6)?;
        }

        // final token → image attention
        let q: Vec<f32> = queries.iter().zip(&pe).map(|(a, b)| a + b).collect();
        let k: Vec<f32> = keys.iter().zip(pos_rows).map(|(a, b)| a + b).collect();
        let mut ao = vec![0f32; n_tok * HID];
        self.attn(&format!("{tb}final_attn_token_to_image."), &q, &k, &keys, n_tok, n_img, HID, HID, 128, None, &mut ao)?;
        for (a, c) in queries.iter_mut().zip(&ao) {
            *a += *c;
        }
        self.ln_rows(&mut queries, &format!("{tb}norm_final_attn.weight"), &format!("{tb}norm_final_attn.bias"), 1e-6)?;

        // heads
        let p = format!("{db}pred_obj_score_head.");
        let mut h = relu_vec(&self.lin(&p, "layers.0", &queries[0..HID], 1)?);
        h = relu_vec(&self.lin(&p, "layers.1", &h, 1)?);
        let obj_score = self.lin(&p, "layers.2", &h, 1)?[0];

        let p = format!("{db}iou_prediction_head.");
        let mut h = relu_vec(&self.lin(&p, "layers.0", &queries[HID..2 * HID], 1)?);
        h = relu_vec(&self.lin(&p, "layers.1", &h, 1)?);
        let iou_all: Vec<f32> = self.lin(&p, "layers.2", &h, 1)?.into_iter().map(sigm).collect();

        let mask_tok = &queries[2 * HID..6 * HID];
        let mut sel_from = if multimask { 1usize } else { 0usize };
        let n_masks = if multimask { 3usize } else { 1usize };
        let mut chosen: Vec<Vec<f32>> = Vec::with_capacity(n_masks);
        if multimask {
            for mi in 0..n_masks {
                chosen.push(self.hyper_mask(&mask_tok[(1 + mi) * HID..(2 + mi) * HID], &keys, 1 + mi, high)?);
            }
        } else {
            let m0 = self.hyper_mask(&mask_tok[0..HID], &keys, 0, high)?;
            let (area_i, area_u) = stability_areas(&m0);
            let stability = if area_u > 0.0 { area_i / area_u } else { 1.0 };
            if stability < 0.98 {
                sel_from = 1;
                for mi in 0..3 {
                    chosen.push(self.hyper_mask(&mask_tok[(1 + mi) * HID..(2 + mi) * HID], &keys, 1 + mi, high)?);
                }
            } else {
                chosen.push(m0);
            }
        }
        let n_out = chosen.len();
        let mut masks = Vec::with_capacity(n_out * MASK_SIDE * MASK_SIDE);
        let mut iou = Vec::with_capacity(n_out);
        let mut toks = Vec::with_capacity(n_out * HID);
        for (mi, m) in chosen.into_iter().enumerate() {
            masks.extend_from_slice(&m);
            iou.push(iou_all.get(sel_from + mi).copied().unwrap_or(1.0));
            toks.extend_from_slice(&mask_tok[(sel_from + mi) * HID..(sel_from + mi + 1) * HID]);
        }
        Ok((masks, iou, obj_score, toks))
    }

/// One hypernetwork mask: token → 32-dim code, dotted with the upscaled
    /// pixel embedding (`upscale_conv1 + feat_s1 → LN → GELU →
    /// upscale_conv2 + feat_s0 → GELU`).
    fn hyper_mask(
        &self,
        tok: &[f32],
        keys: &[f32],
        mlp: usize,
        high: (&[f32], &[f32]),
    ) -> Result<Vec<f32>> {
        let db = "trk.sam_mask_decoder.";
        let p = format!("{db}output_hypernetworks_mlps.{mlp}.");
        let mut h = relu_vec(&self.lin(&p, "layers.0", tok, 1)?);
        h = relu_vec(&self.lin(&p, "layers.1", &h, 1)?);
        let code = self.lin(&p, "layers.2", &h, 1)?;
        anyhow::ensure!(code.len() == 32, "hypernetwork code width {}", code.len());

        let keys_chw = sam3::rows_to_chw(keys, EMB, EMB);
        let w0 = self.v(&format!("{db}output_upscaling.0.weight"))?;
        let b0 = self.v(&format!("{db}output_upscaling.0.bias"))?;
        let mut up = sam3::convt2x2(&w0, &b0, &keys_chw, HID, EMB, EMB, self.pool); // (64,144,144)
        anyhow::ensure!(up.len() == 64 * 144 * 144, "upscale stage 1");
        for (a, b) in up.iter_mut().zip(high.1) {
            *a += *b;
        }
        let lw = self.v(&format!("{db}output_upscaling.1.weight"))?;
        let lb = self.v(&format!("{db}output_upscaling.1.bias"))?;
        ln_chw(&mut up, 64, 144 * 144, &lw, &lb, 1e-6);
        up.iter_mut().for_each(|v| *v = gelu_erf(*v));
        let w3 = self.v(&format!("{db}output_upscaling.3.weight"))?;
        let b3 = self.v(&format!("{db}output_upscaling.3.bias"))?;
        let mut up = sam3::convt2x2(&w3, &b3, &up, 64, 144, 144, self.pool); // (32,288,288)
        anyhow::ensure!(up.len() == 32 * MASK_SIDE * MASK_SIDE, "upscale stage 2");
        for (a, b) in up.iter_mut().zip(high.0) {
            *a += *b;
        }
        up.iter_mut().for_each(|v| *v = gelu_erf(*v));

        let hw = MASK_SIDE * MASK_SIDE;
        let mut out = vec![0f32; hw];
        for c in 0..32 {
            let wgt = code[c];
            if wgt == 0.0 {
                continue;
            }
            for (o, v) in out.iter_mut().zip(&up[c * hw..(c + 1) * hw]) {
                *o += wgt * *v;
            }
        }
        Ok(out)
    }
}

/// Sine PE over an `h × w` grid (2·n channels per cell, y then x).
pub(crate) fn sine(h: usize, w: usize, n: usize) -> Vec<f32> {
    sam3::sine_pos(h, w, n)
}

/// 1-D sine PE of normalized values (`get_1d_sine_pe`).
pub(crate) fn sine_1d(vals: &[f32], dim: usize) -> Vec<f32> {
    let pe = dim / 2;
    let mut dim_t = vec![0f32; pe];
    for i in 0..pe {
        dim_t[i] = 10000f32.powf((2 * (i / 2)) as f32 / pe as f32);
    }
    let mut out = vec![0f32; vals.len() * dim];
    for (v, chunk) in vals.iter().zip(out.chunks_exact_mut(dim)) {
        for i in 0..pe {
            let x = *v / dim_t[i];
            chunk[i] = x.sin();
            chunk[pe + i] = x.cos();
        }
    }
    out
}

/// `IoU(area(logit > δ), area(logit > -δ))` — the dynamic-multimask
/// stability score (`_get_stability_scores`, δ = 0.05).
fn stability_areas(mask: &[f32]) -> (f32, f32) {
    let mut ai = 0f32;
    let mut au = 0f32;
    for v in mask {
        if *v > 0.05 {
            ai += 1.0;
        }
        if *v > -0.05 {
            au += 1.0;
        }
    }
    (ai, au)
}