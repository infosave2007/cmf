//! The voice-activity model that cuts a long recording for GigaAM: a small
//! Conformer (five layers, 128 wide) over the same log-mel features, run on
//! independent 10-second windows, giving a speech probability every 40 ms;
//! and the segmentation on top of it — speech stretches by hysteresis, long
//! ones split at their least speech-like step, neighbours merged, every
//! segment at most 30 s.

use super::{Exec, Lin, Mel, ops, tensor_f32};
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Segmentation settings (the reference defaults).
#[derive(Clone, Debug)]
pub struct Settings {
    /// speech stretches closer than this (seconds) share a segment
    pub merge_gap: f64,
    /// silence kept around a segment
    pub pad: f64,
    /// a segment is at most this long
    pub max_segment: f64,
    /// a step at or above this probability is speech
    pub threshold: f64,
    /// a step below this one ends speech
    pub neg_threshold: f64,
    pub min_speech: f64,
    pub min_silence: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self { merge_gap: 0.7, pad: 0.2, max_segment: 30.0, threshold: 0.5, neg_threshold: 0.35, min_speech: 0.1, min_silence: 0.1 }
    }
}

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn load(m: &CmfModel, p: &str) -> Result<Self, String> {
        Ok(Self { w: tensor_f32(m, &format!("{p}.weight"))?, b: tensor_f32(m, &format!("{p}.bias"))? })
    }

    /// LayerNorm over rows of `dim` (eps 1e-5).
    fn apply(&self, x: &[f32], dim: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        let op = ops::Shared(out.as_mut_ptr());
        ops::par(pool, x.len() / dim, 256, &|a, b| {
            for r in a..b {
                let src = &x[r * dim..(r + 1) * dim];
                let mean = ops::sum(src) / dim as f32;
                let var = ops::sumsq_about(src, mean) / dim as f32;
                let inv = 1.0 / (var + 1e-5).sqrt();
                // SAFETY: row `r` belongs to this range alone.
                let dst = unsafe { op.slice(r * dim, dim) };
                for (((d, &v), &w), &b) in dst.iter_mut().zip(src).zip(&self.w).zip(&self.b) {
                    *d = (v - mean) * inv * w + b;
                }
            }
        });
        out
    }
}

struct VadLayer {
    norm_ff1: Norm,
    ff1: (Lin, Lin),
    norm_att: Norm,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    norm_conv: Norm,
    pw1: Lin,
    dw_w: Vec<f32>,
    dw_b: Vec<f32>,
    conv_norm: Norm,
    pw2: Lin,
    norm_ff2: Norm,
    ff2: (Lin, Lin),
    norm_out: Norm,
}

/// The voice-activity model.
pub struct Vad {
    mel: Mel,
    /// two stride-2 convolutions over time: `[out, in, 3]` weights and biases
    pre: Vec<(Vec<f32>, Vec<f32>, usize, usize)>,
    layers: Vec<VadLayer>,
    head: Lin,
    dim: usize,
    heads: usize,
    kernel: usize,
    /// feature frames in one window
    window: usize,
    /// rotary angles: `[step][head_dim / 2]`
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Vad {
    pub(crate) fn load(m: &Arc<CmfModel>, cfg: &serde_json::Value) -> Result<Self, String> {
        let enc = cfg.get("encoder").ok_or("GigaAM VAD config: no encoder")?;
        let g = |k: &str, d: u64| enc.get(k).and_then(|v| v.as_u64()).unwrap_or(d) as usize;
        let (dim, heads, n_layers, kernel) = (g("d_model", 128), g("n_heads", 4), g("n_layers", 5), g("conv_kernel_size", 15));
        if g("subsampling_factor", 4) != 4 || enc.get("subsampling").and_then(|v| v.as_str()) != Some("conv1d") {
            return Err("GigaAM VAD: only the 4× conv1d subsampling is supported".into());
        }
        if enc.get("conv_norm_type").and_then(|v| v.as_str()) != Some("layer_norm") {
            return Err("GigaAM VAD: only layer_norm convolution modules are supported".into());
        }
        let mut pre = Vec::new();
        for i in [0usize, 2] {
            let name = format!("vad.encoder.pre_encode.conv.{i}.weight");
            let shape = m.tensor(&name).ok_or_else(|| format!("missing GigaAM tensor {name}"))?.shape.clone();
            if shape.len() != 3 || shape[2] != 3 {
                return Err(format!("{name}: shape {shape:?}, expected [out, in, 3]"));
            }
            // patch layout [k][in] so one product does the convolution
            let w = tensor_f32(m, &name)?;
            let (co, ci) = (shape[0], shape[1]);
            let mut wp = vec![0f32; co * 3 * ci];
            for o in 0..co {
                for c in 0..ci {
                    for k in 0..3 {
                        wp[(o * 3 + k) * ci + c] = w[(o * ci + c) * 3 + k];
                    }
                }
            }
            pre.push((wp, tensor_f32(m, &format!("vad.encoder.pre_encode.conv.{i}.bias"))?, co, ci));
        }
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let p = format!("vad.encoder.layers.{l}");
            let lin = |s: &str| Lin::load(m, &format!("{p}.{s}"));
            layers.push(VadLayer {
                norm_ff1: Norm::load(m, &format!("{p}.norm_feed_forward1"))?,
                ff1: (lin("feed_forward1.linear1")?, lin("feed_forward1.linear2")?),
                norm_att: Norm::load(m, &format!("{p}.norm_self_att"))?,
                q: lin("self_attn.linear_q")?,
                k: lin("self_attn.linear_k")?,
                v: lin("self_attn.linear_v")?,
                o: lin("self_attn.linear_out")?,
                norm_conv: Norm::load(m, &format!("{p}.norm_conv"))?,
                pw1: lin("conv.pointwise_conv1")?,
                dw_w: tensor_f32(m, &format!("{p}.conv.depthwise_conv.weight"))?,
                dw_b: tensor_f32(m, &format!("{p}.conv.depthwise_conv.bias"))?,
                conv_norm: Norm::load(m, &format!("{p}.conv.batch_norm"))?,
                pw2: lin("conv.pointwise_conv2")?,
                norm_ff2: Norm::load(m, &format!("{p}.norm_feed_forward2"))?,
                ff2: (lin("feed_forward2.linear1")?, lin("feed_forward2.linear2")?),
                norm_out: Norm::load(m, &format!("{p}.norm_out"))?,
            });
        }
        let window_sec = cfg.pointer("/vad/window_sec").and_then(|v| v.as_f64()).unwrap_or(10.0);
        let steps = (window_sec * super::SR as f64 / (4 * super::HOP) as f64).round() as usize;
        let hd = dim / heads;
        let base = enc.get("rope_base").and_then(|v| v.as_f64()).unwrap_or(g("pos_emb_max_len", 5000) as f64) as f32;
        let (mut cos, mut sin) = (Vec::with_capacity(steps * hd / 2), Vec::with_capacity(steps * hd / 2));
        for p in 0..steps {
            for i in 0..hd / 2 {
                let a = p as f32 * (1.0 / base.powf(2.0 * i as f32 / hd as f32));
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Ok(Self { mel: Mel::load(m, "vad.preprocessor")?, pre, layers, head: Lin::load(m, "vad.head.decoder_layers.0")?, dim, heads, kernel, window: steps * 4, cos, sin })
    }

    /// Seconds between two probabilities.
    pub fn step_sec() -> f64 {
        (4 * super::HOP) as f64 / super::SR as f64
    }

    /// The speech probability of every 40 ms step of `samples`.
    pub fn probabilities(&self, samples: &[f32], pool: Option<&Pool>) -> Vec<f32> {
        if samples.len() < super::N_FFT {
            return Vec::new();
        }
        let ex = Exec { gpu: false, ..Exec::new(pool) };
        let n_mel = self.mel.n_mel();
        let feats = self.mel.features(samples, pool);
        let frames = feats.len() / n_mel;
        let (complete, tail) = (frames / self.window, frames % self.window);
        let mut probs = Vec::with_capacity(frames / 4 + 1);
        // complete windows a batch at a time, then the tail on its own
        const BATCH: usize = 32;
        let mut w0 = 0;
        while w0 < complete {
            let w1 = (w0 + BATCH).min(complete);
            let lens = vec![self.window; w1 - w0];
            probs.extend(self.forward(&feats[w0 * self.window * n_mel..w1 * self.window * n_mel], &lens, &ex));
            w0 = w1;
        }
        if tail > 0 {
            probs.extend(self.forward(&feats[complete * self.window * n_mel..], &[tail], &ex));
        }
        probs
    }

    /// Probabilities of windows of `lens` feature frames laid back to back.
    fn forward(&self, feats: &[f32], lens: &[usize], ex: &Exec) -> Vec<f32> {
        let n_mel = self.mel.n_mel();
        // two stride-2 convolutions (kernel 3, one zero frame each side), ReLU
        let mut x = feats.to_vec();
        let mut lens: Vec<usize> = lens.to_vec();
        let mut width = n_mel;
        for (w, b, co, ci) in &self.pre {
            debug_assert_eq!(*ci, width);
            let out_lens: Vec<usize> = lens.iter().map(|l| (l - 1) / 2 + 1).collect();
            let n_out: usize = out_lens.iter().sum();
            let mut patches = vec![0f32; n_out * 3 * ci];
            let (mut src0, mut dst0) = (0, 0);
            for (&l, &lo) in lens.iter().zip(&out_lens) {
                for t in 0..lo {
                    for k in 0..3 {
                        let s = (2 * t + k) as isize - 1;
                        if s >= 0 && (s as usize) < l {
                            patches[((dst0 + t) * 3 + k) * ci..][..*ci].copy_from_slice(&x[(src0 + s as usize) * ci..][..*ci]);
                        }
                    }
                }
                src0 += l;
                dst0 += lo;
            }
            let mut y = vec![0f32; n_out * co];
            ops::matmul(&ops::Weight::Dense(w), *co, 3 * ci, &patches, n_out, &mut y, ex.pool);
            for row in y.chunks_exact_mut(*co) {
                for (v, &b) in row.iter_mut().zip(b) {
                    *v = (*v + b).max(0.0);
                }
            }
            x = y;
            lens = out_lens;
            width = *co;
        }
        let n: usize = lens.iter().sum();
        let d = self.dim;
        for layer in &self.layers {
            x = layer.forward(x, &lens, self, ex);
        }
        let logits = self.head.apply(&x, n, ex);
        debug_assert_eq!(logits.len(), n);
        let _ = d;
        logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
    }

    /// Speech segments of a recording, in seconds, each at most
    /// `settings.max_segment` long.
    pub fn segments(&self, samples: &[f32], settings: &Settings, pool: Option<&Pool>) -> Vec<(f64, f64)> {
        let probs = self.probabilities(samples, pool);
        boundaries(&probs, Self::step_sec(), samples.len() as f64 / super::SR as f64, settings)
    }
}

impl VadLayer {
    fn forward(&self, x: Vec<f32>, lens: &[usize], vad: &Vad, ex: &Exec) -> Vec<f32> {
        let (d, heads) = (vad.dim, vad.heads);
        let hd = d / heads;
        let n: usize = lens.iter().sum();
        let mut r = x;
        let ff = |norm: &Norm, f: &(Lin, Lin), r: &mut Vec<f32>| {
            let mut h = f.0.apply(&norm.apply(r, d, ex.pool), n, ex);
            super::silu_par(&mut h, ex.pool);
            let y = f.1.apply(&h, n, ex);
            for (a, b) in r.iter_mut().zip(&y) {
                *a += 0.5 * b;
            }
        };
        ff(&self.norm_ff1, &self.ff1, &mut r);
        // attention: queries and keys are the rotated input, values the input itself
        let xn = self.norm_att.apply(&r, d, ex.pool);
        let mut rot = vec![0f32; n * d];
        let mut row0 = 0;
        for &t in lens {
            for i in 0..t {
                let (cos, sin) = (&vad.cos[i * hd / 2..(i + 1) * hd / 2], &vad.sin[i * hd / 2..(i + 1) * hd / 2]);
                for h in 0..heads {
                    let src = &xn[(row0 + i) * d + h * hd..][..hd];
                    let dst = &mut rot[(row0 + i) * d + h * hd..][..hd];
                    for j in 0..hd / 2 {
                        dst[j] = src[j] * cos[j] - src[hd / 2 + j] * sin[j];
                        dst[hd / 2 + j] = src[hd / 2 + j] * cos[j] + src[j] * sin[j];
                    }
                }
            }
            row0 += t;
        }
        let (q, k, v) = (self.q.apply(&rot, n, ex), self.k.apply(&rot, n, ex), self.v.apply(&xn, n, ex));
        let mut ctx = vec![0f32; n * d];
        let scale = (hd as f32).powf(-0.5);
        let cp = ops::Shared(ctx.as_mut_ptr());
        let mut starts = Vec::with_capacity(lens.len());
        let mut at = 0;
        for &t in lens {
            starts.push(at);
            at += t;
        }
        // every (window, head) is a task of its own
        ops::par(ex.pool, lens.len() * heads, 1, &|a, b| {
            let (mut scores, mut out) = (Vec::new(), Vec::new());
            let (mut qh, mut kh, mut vh) = (Vec::new(), Vec::new(), Vec::new());
            for task in a..b {
                let (wi, h) = (task / heads, task % heads);
                let (row0, t) = (starts[wi], lens[wi]);
                let plane = |src: &[f32], p: &mut Vec<f32>| {
                    p.resize(t * hd, 0.0);
                    for i in 0..t {
                        p[i * hd..(i + 1) * hd].copy_from_slice(&src[(row0 + i) * d + h * hd..][..hd]);
                    }
                };
                plane(&q, &mut qh);
                plane(&k, &mut kh);
                plane(&v, &mut vh);
                scores.resize(t * t, 0.0);
                out.resize(t * hd, 0.0);
                ops::gemm_nt(&qh, &kh, &mut scores, t, t, hd, None);
                for row in scores.chunks_exact_mut(t) {
                    ops::softmax_scaled(row, scale);
                }
                ops::gemm_nn(&scores, &vh, &mut out, t, hd, t, None);
                for i in 0..t {
                    // SAFETY: the head's columns of the window's rows belong to this task alone.
                    unsafe { cp.slice((row0 + i) * d + h * hd, hd) }.copy_from_slice(&out[i * hd..(i + 1) * hd]);
                }
            }
        });
        let a = self.o.apply(&ctx, n, ex);
        for (x, y) in r.iter_mut().zip(&a) {
            *x += y;
        }
        // convolution module: 1×1 up, GLU, depthwise, LayerNorm, SiLU, 1×1 down
        let up = self.pw1.apply(&self.norm_conv.apply(&r, d, ex.pool), n, ex);
        let mut glu = vec![0f32; n * d];
        for (dst, src) in glu.chunks_exact_mut(d).zip(up.chunks_exact(2 * d)) {
            dst.copy_from_slice(&src[..d]);
            ops::mul_sigmoid(dst, &src[d..]);
        }
        let mut conv = vec![0f32; n * d];
        let mut row0 = 0;
        for &t in lens {
            ops::depthwise1d(&glu[row0 * d..(row0 + t) * d], t, d, &self.dw_w, &self.dw_b, vad.kernel, &mut conv[row0 * d..(row0 + t) * d], ex.pool);
            row0 += t;
        }
        let mut c = self.conv_norm.apply(&conv, d, ex.pool);
        super::silu_par(&mut c, ex.pool);
        let c = self.pw2.apply(&c, n, ex);
        for (x, y) in r.iter_mut().zip(&c) {
            *x += y;
        }
        ff(&self.norm_ff2, &self.ff2, &mut r);
        self.norm_out.apply(&r, d, ex.pool)
    }
}

/// Speech stretches in seconds: a step at or above `threshold` opens one,
/// `min_silence` of steps below `neg_threshold` close it.
pub fn speech_timestamps(probs: &[f32], step_sec: f64, duration: f64, s: &Settings) -> Vec<(f64, f64)> {
    let min_silence_steps = (s.min_silence / step_sec - 1e-9).ceil() as i64;
    let steps_per_sec = 1.0 / step_sec;
    let mut speeches = Vec::new();
    let (mut triggered, mut start) = (false, 0usize);
    let mut temp_end: Option<usize> = None;
    for (step, &p) in probs.iter().enumerate() {
        let p = p as f64;
        if p >= s.threshold {
            temp_end = None;
            if !triggered {
                (triggered, start) = (true, step);
            }
            continue;
        }
        if p < s.neg_threshold && triggered {
            let te = *temp_end.get_or_insert(step);
            if ((step - te + 1) as i64) < min_silence_steps {
                continue;
            }
            if (te - start) as f64 * step_sec > s.min_speech {
                speeches.push((start as f64 / steps_per_sec, te as f64 / steps_per_sec));
            }
            (temp_end, triggered) = (None, false);
        }
    }
    if triggered && duration - start as f64 / steps_per_sec > s.min_speech {
        speeches.push((start as f64 / steps_per_sec, duration));
    }
    speeches
}

/// Cut points of a speech stretch longer than `limit`: the least
/// speech-like step among those that keep the number of parts minimal,
/// recursively.
fn min_prob_cuts(start: f64, end: f64, probs: &[f32], sps: f64, limit: f64, margin: f64, out: &mut Vec<f64>) {
    if end - start <= limit {
        return;
    }
    let parts = ((end - start) / limit).ceil();
    let (lo, hi) = (((start + margin) * sps - 1e-9).ceil() as i64, ((end - margin) * sps - 1e-9).ceil() as i64);
    let mut best: Option<usize> = None;
    for k in lo.max(0)..hi.min(probs.len() as i64) {
        let at = k as f64 / sps;
        if ((at - start) / limit).ceil() + ((end - at) / limit).ceil() != parts {
            continue;
        }
        let k = k as usize;
        if best.is_none_or(|b| probs[k] < probs[b]) {
            best = Some(k);
        }
    }
    let Some(k) = best else {
        for q in 1..parts as usize {
            out.push(start + (end - start) * q as f64 / parts);
        }
        return;
    };
    let cut = k as f64 / sps;
    min_prob_cuts(start, cut, probs, sps, limit, margin, out);
    out.push(cut);
    min_prob_cuts(cut, end, probs, sps, limit, margin, out);
}

/// Segment bounds in seconds from step probabilities.
pub fn boundaries(probs: &[f32], step_sec: f64, total: f64, s: &Settings) -> Vec<(f64, f64)> {
    let speech = speech_timestamps(probs, step_sec, total, s);
    let inner = s.max_segment - (2.0 * s.pad).max(s.merge_gap) - 0.001;
    let margin = 0.1 * s.max_segment;
    let mut segs: Vec<(f64, f64)> = Vec::new();
    for (start, end) in speech {
        let mut edges = vec![start];
        min_prob_cuts(start, end, probs, 1.0 / step_sec, inner, margin, &mut edges);
        edges.push(end);
        segs.extend(edges.windows(2).map(|e| (e[0], e[1])));
    }
    let mut groups: Vec<(usize, usize)> = Vec::new();
    for q in 0..segs.len() {
        match groups.last_mut() {
            Some(g) if segs[q].0 - segs[q - 1].1 <= s.merge_gap + 1e-9 => g.1 = q,
            _ => groups.push((q, q)),
        }
    }
    let mut bounds = Vec::new();
    for (i, j) in groups {
        let (mut start, end) = ((segs[i].0 - s.pad).max(0.0), (segs[j].1 + s.pad).min(total));
        let mut k = i;
        while end - start > s.max_segment + 1e-9 {
            // the widest pause that still keeps this piece within the limit
            let mut best: Option<(f64, f64, usize)> = None;
            for m in k..j {
                let mid = (segs[m].1 + segs[m + 1].0) / 2.0;
                if mid - start > s.max_segment + 1e-9 {
                    break;
                }
                let cand = (segs[m + 1].0 - segs[m].1, mid, m);
                if best.is_none_or(|b| (cand.0, cand.1) >= (b.0, b.1)) {
                    best = Some(cand);
                }
            }
            let (mid, m) = match best {
                Some((_, mid, m)) => (mid, m),
                // no pause in reach: a hard cut at the limit
                None => (start + s.max_segment, k),
            };
            bounds.push((start, mid));
            start = mid;
            if best.is_none() {
                continue;
            }
            k = m + 1;
        }
        bounds.push((start, end));
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steps(spans: &[(usize, usize)], n: usize) -> Vec<f32> {
        let mut p = vec![0.02f32; n];
        for &(a, b) in spans {
            p[a..b].fill(0.97);
        }
        p
    }

    #[test]
    fn hysteresis_finds_speech_stretches() {
        let s = Settings::default();
        // speech at 1.0–3.0 s and 5.0–5.04 s (one step: too short), then to the end
        let p = steps(&[(25, 75), (125, 126), (200, 250)], 250);
        let got = speech_timestamps(&p, 0.04, 10.0, &s);
        assert_eq!(got.len(), 2);
        assert!((got[0].0 - 1.0).abs() < 1e-9 && (got[0].1 - 3.0).abs() < 1e-9);
        assert!((got[1].0 - 8.0).abs() < 1e-9 && (got[1].1 - 10.0).abs() < 1e-9);
        // a dip between the thresholds does not end speech
        let mut p = steps(&[(25, 75)], 100);
        p[40..45].fill(0.4);
        assert_eq!(speech_timestamps(&p, 0.04, 4.0, &s).len(), 1);
    }

    #[test]
    fn neighbours_merge_and_long_speech_is_cut_at_its_weakest_step() {
        let s = Settings::default();
        // two stretches 0.4 s apart: one segment, padded
        let p = steps(&[(25, 75), (85, 125)], 250);
        let b = boundaries(&p, 0.04, 10.0, &s);
        assert_eq!(b.len(), 1);
        assert!((b[0].0 - 0.8).abs() < 1e-9 && (b[0].1 - 5.2).abs() < 1e-9);
        // 50 s of unbroken speech with one weaker step at 24 s: cut there
        let mut p = steps(&[(0, 1250)], 1250);
        p[600] = 0.6;
        let b = boundaries(&p, 0.04, 50.0, &s);
        assert_eq!(b.len(), 2);
        assert!((b[0].1 - 24.0).abs() < 1e-9 && (b[1].0 - 24.0).abs() < 1e-9);
        assert!(b.iter().all(|(a, e)| e - a <= s.max_segment + 1e-9));
        assert!((b[0].0, b[1].1) == (0.0, 50.0));
    }

    #[test]
    fn a_chain_of_close_stretches_splits_at_the_widest_pause() {
        let s = Settings::default();
        // 6 s of speech, 0.5 s pauses, one 0.68 s pause: 8 stretches = 51.7 s merged
        let mut spans = Vec::new();
        let mut at = 10usize;
        for i in 0..8 {
            spans.push((at, at + 150));
            at += 150 + if i == 3 { 17 } else { 12 };
        }
        let p = steps(&spans, at + 20);
        let total = (at + 20) as f64 * 0.04;
        let b = boundaries(&p, 0.04, total, &s);
        assert_eq!(b.len(), 2);
        assert!(b.iter().all(|(a, e)| e - a <= s.max_segment + 1e-9));
        // the cut is in the middle of the wide pause after the fourth stretch
        let wide = (spans[3].1 as f64 * 0.04 + spans[4].0 as f64 * 0.04) / 2.0;
        assert!((b[0].1 - wide).abs() < 1e-9, "{b:?} vs {wide}");
    }
}
