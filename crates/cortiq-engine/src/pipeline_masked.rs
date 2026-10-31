//! A non-causal forward over one packed sequence, for d1 (`crate::d1`):
//! the System One decision models read a state and a set of questions in
//! one pass in which the full-attention layers see both ways — the state
//! reads and is read by every position, a question's tokens read the state
//! and their own question — while the sliding-window layers stay causal
//! within their window and the short convolutions causal over the row
//! (d1's `attention.Sees`). A child module of `pipeline` so it runs on the
//! layer geometry the decode paths use; it touches no cache.

use super::{AttnKind, FfnKind, Pipeline, dense_ffn_batch};
use crate::inference;
use crate::linear_core::short_conv_forward_batch;
use crate::qtensor::QTensor;
use cortiq_core::NormStyle;

/// d1's `set` mask: positions `[0, state)` are the state; every later
/// position belongs to the question `group[t]`.
pub struct SetMask<'a> {
    pub state: usize,
    pub group: &'a [i32],
}

impl SetMask<'_> {
    /// Whether query `q` reads key `k` in a full-attention layer.
    fn sees(&self, q: usize, k: usize) -> bool {
        q < self.state || k < self.state || self.group[q] == self.group[k]
    }
}

fn rms_rows(x: &[f32], w: &[f32], eps: f64, style: NormStyle, out: &mut [f32]) {
    let d = w.len();
    for (src, dst) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        inference::rms_norm_into(src, w, eps, style, dst);
    }
}

impl Pipeline {
    /// The final-normed hidden state of every position of `rows` (prompt
    /// embedding rows `[T, hidden]`) under d1's `mask`. Only plain LFM3
    /// stacks (short convolution, full / sliding GQA, dense FFN) qualify.
    pub(crate) fn forward_masked(&self, rows: &[f32], mask: &SetMask) -> Result<Vec<f32>, String> {
        let hs = self.hidden_size;
        let t = rows.len() / hs;
        if t == 0 || rows.len() != t * hs || mask.group.len() != t || mask.state > t {
            return Err("masked forward: rows, groups and state length disagree".into());
        }
        let pool = self.pool.as_deref();
        let (eps, style) = (self.rms_eps, self.norm_style);
        let mut h = rows.to_vec();
        let mut normed = vec![0f32; t * hs];
        for li in 0..self.num_layers {
            let lw = &self.weights.layers[self.phys_layer(li)];
            rms_rows(&h, &lw.input_norm, eps, style, &mut normed);
            let mut mixed = match &lw.attn {
                AttnKind::ShortConv(w) => {
                    let cfg = self
                        .short_conv_cfg
                        .ok_or("short-conv layer without its geometry")?;
                    let mut ring = vec![0f32; cfg.state_len()];
                    short_conv_forward_batch(&normed, t, w, &cfg, &mut ring, pool)
                }
                AttnKind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    output_gate: false,
                    softplus_gate,
                    bias: None,
                } => {
                    let ao = self.masked_attention(li, &normed, t, wq, wk, wv, q_norm, k_norm, softplus_gate, mask)?;
                    let mut out = vec![0f32; t * hs];
                    wo.matmat(&ao, t, &mut out, pool);
                    out
                }
                _ => return Err(format!("layer {li}: the masked forward runs LFM3 layers only")),
            };
            if let Some(w) = &lw.attn_out_norm {
                let raw = mixed.clone();
                rms_rows(&raw, w, eps, style, &mut mixed);
            }
            for (a, b) in h.iter_mut().zip(&mixed) {
                *a += *b;
            }
            rms_rows(&h, &lw.post_norm, eps, style, &mut normed);
            let FfnKind::Dense(d) = &lw.ffn else {
                return Err(format!("layer {li}: the masked forward runs dense FFNs only"));
            };
            let mut ffn = dense_ffn_batch(d, &normed, t, pool, None);
            if let Some(w) = &lw.ffn_out_norm {
                let raw = ffn.clone();
                rms_rows(&raw, w, eps, style, &mut ffn);
            }
            for (a, b) in h.iter_mut().zip(&ffn) {
                *a += *b;
            }
        }
        rms_rows(&h, &self.weights.final_norm, eps, style, &mut normed);
        Ok(normed)
    }

    /// One full or sliding attention layer over the packed row: Q/K/V, the
    /// per-head norms, RoPE at each row's own position, attention under the
    /// layer's window (causal) or `mask` (both ways), the per-head gate.
    /// Returns `[T, heads·hd]`, ready for O.
    #[allow(clippy::too_many_arguments)]
    fn masked_attention(
        &self,
        li: usize,
        x: &[f32],
        t: usize,
        wq: &QTensor,
        wk: &QTensor,
        wv: &QTensor,
        q_norm: &Option<Vec<f32>>,
        k_norm: &Option<Vec<f32>>,
        gate: &Option<(QTensor, bool)>,
        mask: &SetMask,
    ) -> Result<Vec<f32>, String> {
        let pool = self.pool.as_deref();
        let nh = self.layer_num_heads(li);
        let (nkv, hd, rd) = self.layer_geom(li);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let inv = self.layer_inv_freq(li);
        let rope_scale = self.layer_rope_scale(li);
        let window = self.layer_window(li);
        if rd != 2 * inv.len() || nkv == 0 || nh % nkv != 0 {
            return Err(format!("layer {li}: unsupported attention geometry"));
        }
        let mut q = vec![0f32; t * qd];
        let mut k = vec![0f32; t * kvd];
        let mut v = vec![0f32; t * kvd];
        wq.matmat(x, t, &mut q, pool);
        wk.matmat(x, t, &mut k, pool);
        wv.matmat(x, t, &mut v, pool);
        let mut tmp = vec![0f32; hd];
        for (buf, norm, heads) in [(&mut q, q_norm, nh), (&mut k, k_norm, nkv)] {
            for pos in 0..t {
                for hh in 0..heads {
                    let head = &mut buf[(pos * heads + hh) * hd..(pos * heads + hh + 1) * hd];
                    if let Some(w) = norm {
                        inference::rms_norm_into(head, w, self.rms_eps, self.norm_style, &mut tmp);
                        head.copy_from_slice(&tmp);
                    }
                    crate::attention::rope_rotate_scaled(&mut head[..rd], pos, &inv, rope_scale);
                }
            }
        }
        // One head at a time: scores [T, T] through the host GEMM, the
        // layer's mask, a row softmax, then P·V.
        let group = nh / nkv;
        let scale = self.attn_scale;
        let mut out = vec![0f32; t * qd];
        let mut qh = vec![0f32; t * hd];
        let mut kh = vec![0f32; t * hd];
        let mut vt = vec![0f32; hd * t];
        let mut sc = vec![0f32; t * t];
        let mut oh = vec![0f32; t * hd];
        for g in 0..nkv {
            for j in 0..t {
                kh[j * hd..(j + 1) * hd].copy_from_slice(&k[j * kvd + g * hd..j * kvd + (g + 1) * hd]);
                for d in 0..hd {
                    vt[d * t + j] = v[j * kvd + g * hd + d];
                }
            }
            for hh in g * group..(g + 1) * group {
                for i in 0..t {
                    for d in 0..hd {
                        qh[i * hd + d] = q[i * qd + hh * hd + d] * scale;
                    }
                }
                crate::fcd_ops::gemm_nt_host(&qh, &kh, &mut sc, t, hd, t, pool);
                {
                    let sp = crate::pool::SendMut::new(sc.as_mut_ptr());
                    let f = |lo: usize, hi: usize| {
                        for i in lo..hi {
                            // SAFETY: disjoint score rows per worker.
                            let row = unsafe { std::slice::from_raw_parts_mut(sp.at(i * t), t) };
                            for (j, s) in row.iter_mut().enumerate() {
                                let keep = match window {
                                    Some(w) => j <= i && j + w > i,
                                    None => mask.sees(i, j),
                                };
                                if !keep {
                                    *s = f32::NEG_INFINITY;
                                }
                            }
                            let mx = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
                            let mut sum = 0f32;
                            for s in row.iter_mut() {
                                *s = (*s - mx).exp();
                                sum += *s;
                            }
                            let inv_sum = 1.0 / sum;
                            for s in row.iter_mut() {
                                *s *= inv_sum;
                            }
                        }
                    };
                    match pool {
                        Some(p) => p.run_rows(t, &f),
                        None => f(0, t),
                    }
                }
                crate::fcd_ops::gemm_nt_host(&sc, &vt, &mut oh, t, t, hd, pool);
                for i in 0..t {
                    out[i * qd + hh * hd..i * qd + (hh + 1) * hd].copy_from_slice(&oh[i * hd..(i + 1) * hd]);
                }
            }
        }
        if let Some((g, per_head)) = gate {
            let mut raw = vec![0f32; t * g.rows()];
            g.matmat(x, t, &mut raw, pool);
            for i in 0..t {
                crate::attention::apply_projected_gate(
                    &mut out[i * qd..(i + 1) * qd],
                    &raw[i * g.rows()..(i + 1) * g.rows()],
                    *per_head,
                    hd,
                    self.proj_gate_sigmoid,
                );
            }
        }
        Ok(out)
    }

    /// Log-probabilities over the vocabulary for `n` final-normed hidden rows
    /// `[n, hidden]`: one head GEMM for all of them, `[n, vocab]` out.
    pub(crate) fn log_probs_rows(&self, hidden: &[f32], n: usize) -> Vec<f32> {
        let v = self.vocab_size;
        let mut logits = vec![0f32; n * v];
        if n == 1 {
            self.weights.lm_head.matvec(hidden, &mut logits, self.pool.as_deref());
        } else {
            self.weights.lm_head.matmat(hidden, n, &mut logits, self.pool.as_deref());
        }
        for row in logits.chunks_exact_mut(v) {
            let mx = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
            let lse = mx as f64 + row.iter().map(|&x| ((x - mx) as f64).exp()).sum::<f64>().ln();
            for x in row.iter_mut() {
                *x = (*x as f64 - lse) as f32;
            }
        }
        logits
    }
}
