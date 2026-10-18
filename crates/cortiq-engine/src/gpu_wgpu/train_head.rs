//! The trainer's loss: the frozen vocabulary head on the matrix units and
//! the distillation KL against the teacher's top-k distribution.
//!
//! Per token, with the teacher's top-k ids and probabilities renormalized
//! over the k (`Σ p_T = 1`):
//!
//! ```text
//! L_t = Σ_k p_T,k · (log p_T,k − log p_S,id_k)      log p_S = s − logsumexp(s)
//! ∂L_t/∂s_j = p_S,j − p_T,j                          (p_T,j = 0 off the top-k)
//! ```
//!
//! The loss is averaged over all tokens of the step; the logits gradient
//! is written in place (scaled by the f16 gradient scale), staged and sent
//! back through the head: `dh = dlogits · W`.

use super::train_mm::{self, Op};
use super::train_ops::{self, Disp};
use super::Ctx;

const SRC: &str = r#"
struct P { ntok: u32, v: u32, k: u32, pad: u32, coef: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> LG: array<f32>;    // [ntok][v] logits -> dlogits
@group(0) @binding(2) var<storage, read_write> TID: array<u32>;   // [ntok][k]
@group(0) @binding(3) var<storage, read_write> TP: array<f32>;    // [ntok][k]
@group(0) @binding(4) var<storage, read_write> LOSS: array<f32>;  // [ntok]
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn kl_rows(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row = wg.x + wg.y * 65535u;
    if (row >= p.ntok) { return; }
    let base = row * p.v;
    var mx = -3.0e38;
    for (var j = t; j < p.v; j = j + 256u) { mx = max(mx, LG[base + j]); }
    red[t] = mx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = max(red[t], red[t + s]); }
        workgroupBarrier();
    }
    mx = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var j = t; j < p.v; j = j + 256u) { sum = sum + exp(LG[base + j] - mx); }
    red[t] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        workgroupBarrier();
    }
    let lse = mx + log(red[0]);
    workgroupBarrier();
    // loss over the teacher's support (read the logits before overwriting)
    var lp = 0.0;
    for (var q = t; q < p.k; q = q + 256u) {
        let pt = TP[row * p.k + q];
        if (pt > 0.0) { lp = lp + pt * (log(pt) - (LG[base + TID[row * p.k + q]] - lse)); }
    }
    red[t] = lp;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        workgroupBarrier();
    }
    if (t == 0u) { LOSS[row] = red[0]; }
    workgroupBarrier();
    for (var j = t; j < p.v; j = j + 256u) { LG[base + j] = exp(LG[base + j] - lse) * p.coef; }
    workgroupBarrier();
    for (var q = t; q < p.k; q = q + 256u) {
        let id = TID[row * p.k + q];
        LG[base + id] = LG[base + id] - TP[row * p.k + q] * p.coef;
    }
}

// LG[i] += TP[i] for i < ntok (= the element count): the input gradient of a
// later vocabulary slice added to the first one's.
@compute @workgroup_size(256)
fn acc_add(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 65535u * 256u;
    if (i >= p.ntok) { return; }
    LG[i] = LG[i] + TP[i];
}

// Teacher top-k per row: each thread keeps its best 8 of a strided scan,
// the 2048 candidates are bitonic-sorted in shared memory, the first k
// are written with probabilities renormalized over them.
var<workgroup> cv: array<f32, 2048>;
var<workgroup> ci: array<u32, 2048>;
@compute @workgroup_size(256)
fn topk_rows(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row = wg.x + wg.y * 65535u;
    if (row >= p.ntok) { return; }
    let base = row * p.v;
    var bv: array<f32, 8>;
    var bi: array<u32, 8>;
    for (var q = 0u; q < 8u; q = q + 1u) { bv[q] = -3.0e38; bi[q] = 0u; }
    for (var j = t; j < p.v; j = j + 256u) {
        let x = LG[base + j];
        if (x > bv[7]) {
            var pos = 7u;
            loop {
                if (pos == 0u || bv[pos - 1u] >= x) { break; }
                bv[pos] = bv[pos - 1u];
                bi[pos] = bi[pos - 1u];
                pos = pos - 1u;
            }
            bv[pos] = x;
            bi[pos] = j;
        }
    }
    for (var q = 0u; q < 8u; q = q + 1u) { cv[t * 8u + q] = bv[q]; ci[t * 8u + q] = bi[q]; }
    workgroupBarrier();
    // bitonic sort, descending
    for (var kk = 2u; kk <= 2048u; kk = kk << 1u) {
        for (var jj = kk >> 1u; jj > 0u; jj = jj >> 1u) {
            for (var e = t; e < 2048u; e = e + 256u) {
                let x = e ^ jj;
                if (x > e) {
                    let desc = (e & kk) == 0u;
                    let a = cv[e];
                    let b = cv[x];
                    if ((desc && a < b) || (!desc && a > b)) {
                        cv[e] = b; cv[x] = a;
                        let ti = ci[e]; ci[e] = ci[x]; ci[x] = ti;
                    }
                }
            }
            workgroupBarrier();
        }
    }
    // renormalized probabilities over the top k (softmax restricted to them)
    let mx = cv[0];
    var s = 0.0;
    for (var q = t; q < p.k; q = q + 256u) { s = s + exp(cv[q] - mx); }
    red[t] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
    }
    let z = red[0];
    for (var q = t; q < p.k; q = q + 256u) {
        TID[row * p.k + q] = ci[q];
        TP[row * p.k + q] = exp(cv[q] - mx) / z;
    }
}
"#;

fn module(c: &Ctx) -> Option<&'static train_ops::Mod> {
    train_ops::module(c, "train_head", SRC, &["kl_rows", "topk_rows", "acc_add"])
}

/// Teacher top-k of `ntok` normed hidden rows: (ids, renormalized probs),
/// `[ntok][k]` each, k ≤ 2048.
pub fn topk(c: &Ctx, head: &Head, s: &HeadScratch, x: &wgpu::Buffer, ntok: usize, k: usize) -> Option<(Vec<u32>, Vec<f32>)> {
    let (v, hs) = (head.v, head.hs);
    let m = module(c)?;
    let mut ids = Vec::with_capacity(ntok * k);
    let mut probs = Vec::with_capacity(ntok * k);
    let mut r0 = 0;
    while r0 < ntok {
        let rows = (ntok - r0).min(s.chunk);
        let xin = train_ops::zeros(c, train_mm::pad_rows(rows) * hs);
        let tidc = train_ops::zeros(c, rows * k);
        let tpc = train_ops::zeros(c, rows * k);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("topk_in") });
        enc.copy_buffer_to_buffer(x, (r0 * hs * 4) as u64, &xin, 0, (rows * hs * 4) as u64);
        super::submit(c, enc.finish());
        let ops = vec![
            Op::D(train_mm::cvt(c, &xin, None, &s.act, rows, hs, 1.0)?),
        ];
        let mut ops = ops;
        for (si, &(off, vs)) in head.cuts.iter().enumerate() {
            ops.push(Op::M(train_mm::mm_cols(c, &head.w[si], &s.act, &s.logits, rows, vs, hs, v, off, 1.0)?));
        }
        ops.extend([
            Op::D(Disp {
                pipe: &m.p["topk_rows"],
                uni: vec![rows as u32, v as u32, k as u32, 0, 0, 0, 0, 0],
                bufs: vec![(1, &s.logits), (2, &tidc), (3, &tpc)],
                grid: train_ops::wg_fold(rows),
            }),
        ]);
        train_mm::run(c, &ops);
        let t_raw = train_ops::download(c, &tidc, rows * k)?;
        ids.extend(t_raw.iter().map(|f| f.to_bits()));
        probs.extend(train_ops::download(c, &tpc, rows * k)?);
        r0 += rows;
    }
    Some((ids, probs))
}

/// The KL rows dispatch: loss per token into `loss`, logits replaced by
/// `coef · (p_S − p_T)`.
#[allow(clippy::too_many_arguments)]
pub fn kl_rows<'a>(c: &'a Ctx, logits: &'a wgpu::Buffer, tid: &'a wgpu::Buffer, tp: &'a wgpu::Buffer, loss: &'a wgpu::Buffer, ntok: usize, v: usize, k: usize, coef: f32) -> Option<Disp<'a>> {
    let m = module(c)?;
    Some(Disp {
        pipe: &m.p["kl_rows"],
        uni: vec![ntok as u32, v as u32, k as u32, 0, coef.to_bits(), 0, 0, 0],
        bufs: vec![(1, logits), (2, tid), (3, tp), (4, loss)],
        grid: train_ops::wg_fold(ntok),
    })
}

/// The frozen head in vocabulary slices (`cuts`: column offset, width), each
/// an f16 plane `w[s]` `[width][hs]` and its transpose `wt[s]` `[hs][width]`
/// (empty for a forward-only teacher): a 248k × 5120 head is 2.5 GB, past
/// the 2 GiB a storage binding may span.
pub struct Head {
    pub v: usize,
    pub hs: usize,
    pub cuts: Vec<(usize, usize)>,
    pub w: Vec<wgpu::Buffer>,
    pub wt: Vec<wgpu::Buffer>,
}

impl Head {
    /// Slices of at most `max_rows` vocabulary rows (multiples of 128; None:
    /// the largest that fits a 2 GiB binding) from the f32 head `w` `[v][hs]`.
    pub fn new(c: &Ctx, w: &[f32], v: usize, hs: usize, with_t: bool, max_rows: Option<usize>) -> Self {
        let cap = max_rows.unwrap_or(((1usize << 31) - 4) / (2 * hs)) / 128 * 128;
        let n = v.div_ceil(cap.max(128));
        let per = v.div_ceil(n).div_ceil(128) * 128;
        let mut cuts = Vec::new();
        let mut off = 0;
        while off < v {
            let width = per.min(v - off);
            cuts.push((off, width));
            off += width;
        }
        let plane = |vals: &[f32]| {
            let mut h: Vec<u16> = vals.iter().map(|&f| cortiq_core::quant::f32_to_f16(f)).collect();
            if h.len() % 2 == 1 {
                h.push(0);
            }
            let b = train_mm::f16_buf(c, h.len());
            c.queue.write_buffer(&b, 0, bytemuck::cast_slice(&h));
            b
        };
        let mut wv = Vec::new();
        let mut wtv = Vec::new();
        for &(off, width) in &cuts {
            let sl = &w[off * hs..(off + width) * hs];
            wv.push(plane(sl));
            if with_t {
                let t: Vec<f32> = (0..hs * width).map(|i| sl[(i % width) * hs + i / width]).collect();
                wtv.push(plane(&t));
            }
            c.queue.submit(std::iter::empty());
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        }
        Head { v, hs, cuts, w: wv, wt: wtv }
    }
}

/// Scratch for `chunk` tokens at a time.
pub struct HeadScratch {
    pub chunk: usize,
    act: wgpu::Buffer,
    logits: wgpu::Buffer,
    dact: wgpu::Buffer,
    dpart: wgpu::Buffer,
    /// Per-chunk views of the inputs (fully overwritten before use; the
    /// activation staging zeroes its own padding rows).
    xin: wgpu::Buffer,
    tidc: wgpu::Buffer,
    tpc: wgpu::Buffer,
    lossc: wgpu::Buffer,
    k_max: usize,
}

/// Top-k entries per token the per-chunk scratch holds.
const K_MAX: usize = 256;

impl HeadScratch {
    pub fn new(c: &Ctx, chunk: usize, v: usize, hs: usize) -> Self {
        let pr = train_mm::pad_rows(chunk);
        HeadScratch {
            chunk,
            act: train_mm::f16_buf(c, pr * hs.max(v)),
            logits: train_ops::zeros_u32(c, pr * v),
            dact: train_ops::zeros_u32(c, pr * hs),
            dpart: train_ops::zeros_u32(c, pr * hs),
            xin: train_ops::zeros_u32(c, pr * hs),
            tidc: train_ops::zeros_u32(c, chunk * K_MAX),
            tpc: train_ops::zeros_u32(c, chunk * K_MAX),
            lossc: train_ops::zeros_u32(c, chunk),
            k_max: K_MAX,
        }
    }
}

/// Loss and hidden-state gradient for `ntok` normed hidden rows `x`
/// (`[ntok][hs]`) against the teacher top-k (`tid`, `tp`: `[ntok][k]`).
/// `dx` (`[ntok][hs]`) is overwritten with `∂(mean L)/∂x · gscale`·(1/gscale)
/// = the true gradient; `loss` gets the per-token KL. `total` = tokens the
/// step averages over (for chunked calls).
#[allow(clippy::too_many_arguments)]
pub fn head_kl(
    c: &Ctx,
    head: &Head,
    s: &HeadScratch,
    x: &wgpu::Buffer,
    tid: &wgpu::Buffer,
    tp: &wgpu::Buffer,
    k: usize,
    dx: &wgpu::Buffer,
    loss: &wgpu::Buffer,
    ntok: usize,
    total: usize,
    gscale: f32,
) -> Option<()> {
    let (v, hs) = (head.v, head.hs);
    let ch = s.chunk;
    let mut r0 = 0;
    while r0 < ntok {
        let rows = (ntok - r0).min(ch);
        // views of this chunk: copy rows in/out (small relative to the GEMMs)
        if k > s.k_max {
            return None;
        }
        let (xin, tidc, tpc, lossc) = (&s.xin, &s.tidc, &s.tpc, &s.lossc);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("head_in") });
        let mut mk = super::train_prof::enc_begin(c, &mut enc);
        enc.copy_buffer_to_buffer(x, (r0 * hs * 4) as u64, xin, 0, (rows * hs * 4) as u64);
        enc.copy_buffer_to_buffer(tid, (r0 * k * 4) as u64, tidc, 0, (rows * k * 4) as u64);
        enc.copy_buffer_to_buffer(tp, (r0 * k * 4) as u64, tpc, 0, (rows * k * 4) as u64);
        super::train_prof::enc_mark(c, &mut enc, &mut mk, "copy");
        super::submit(c, enc.finish());
        let m = module(c)?;
        let mut ops = vec![Op::D(train_mm::cvt(c, xin, None, &s.act, rows, hs, 1.0)?)];
        for (si, &(off, vs)) in head.cuts.iter().enumerate() {
            ops.push(Op::M(train_mm::mm_cols(c, &head.w[si], &s.act, &s.logits, rows, vs, hs, v, off, 1.0)?));
        }
        ops.push(Op::D(kl_rows(c, &s.logits, tidc, tpc, lossc, rows, v, k, gscale / total as f32)?));
        // dh = Σ_slices dlogits[:, slice] · W[slice]
        for (si, &(off, vs)) in head.cuts.iter().enumerate() {
            ops.push(Op::D(train_mm::cvt_cols(c, &s.logits, &s.act, rows, vs, v, off, 1.0)?));
            let dst = if si == 0 { &s.dact } else { &s.dpart };
            ops.push(Op::M(train_mm::mm(c, &head.wt[si], &s.act, dst, rows, hs, vs, 1.0 / gscale)?));
            if si > 0 {
                ops.push(Op::D(Disp {
                    pipe: &m.p["acc_add"],
                    uni: vec![(rows * hs) as u32, 0, 0, 0, 0, 0, 0, 0],
                    bufs: vec![(1, &s.dact), (3, &s.dpart)],
                    grid: train_ops::elt_fold(rows * hs),
                }));
            }
        }
        train_mm::run(c, &ops);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("head_out") });
        let mut mk = super::train_prof::enc_begin(c, &mut enc);
        enc.copy_buffer_to_buffer(&s.dact, 0, dx, (r0 * hs * 4) as u64, (rows * hs * 4) as u64);
        enc.copy_buffer_to_buffer(lossc, 0, loss, (r0 * 4) as u64, (rows * 4) as u64);
        super::train_prof::enc_mark(c, &mut enc, &mut mk, "copy");
        super::submit(c, enc.finish());
        r0 += rows;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::train_ops::{download, upload, zeros};

    fn rnd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32 * scale
            })
            .collect()
    }

    #[test]
    fn head_kl_matches_host() {
        let Some(c) = train_ops::dev() else { return };
        if !train_mm::matrix_units() {
            eprintln!("no cooperative matrices on this device — skipping");
            return;
        }
        let (v, hs, ntok, k) = (1024usize, 256usize, 45usize, 16usize);
        let w = rnd(v * hs, 1, 0.1);
        let x = rnd(ntok * hs, 2, 1.0);
        // teacher: top-k ids (distinct) and renormalized probabilities
        let mut tid = vec![0u32; ntok * k];
        let mut tp = vec![0f32; ntok * k];
        for t in 0..ntok {
            let raw = rnd(k, 100 + t as u64, 1.0);
            let z: f32 = raw.iter().map(|r| r.exp()).sum();
            for q in 0..k {
                tid[t * k + q] = ((t * 37 + q * 61) % v) as u32;
                tp[t * k + q] = raw[q].exp() / z;
            }
        }
        let to16 = |src: &[f32]| -> Vec<u16> { src.iter().map(|&f| cortiq_core::quant::f32_to_f16(f)).collect() };
        let wt: Vec<f32> = (0..hs * v).map(|i| w[(i % v) * hs + i / v]).collect();
        let buf16 = |vals: &[u16]| {
            let b = train_mm::f16_buf(c, vals.len());
            let mut padded = vals.to_vec();
            if padded.len() % 2 == 1 {
                padded.push(0);
            }
            c.queue.write_buffer(&b, 0, bytemuck::cast_slice(&padded));
            b
        };
        let _ = (&wt, &buf16, &to16);
        // three vocabulary slices (the multi-slice path a 248k head takes)
        let head = Head::new(c, &w, v, hs, true, Some(384));
        assert_eq!(head.cuts.len(), 3);
        let sc = HeadScratch::new(c, 32, v, hs);
        let ub = |vals: &[u32]| {
            let b = zeros(c, vals.len());
            c.queue.write_buffer(&b, 0, bytemuck::cast_slice(vals));
            b
        };
        let (bx, btid, btp, bdx, bloss) = (upload(c, &x), ub(&tid), upload(c, &tp), zeros(c, ntok * hs), zeros(c, ntok));
        head_kl(c, &head, &sc, &bx, &btid, &btp, k, &bdx, &bloss, ntok, ntok, 256.0).expect("head");
        // reference with the same f16-rounded weights and inputs
        let wr: Vec<f64> = w.iter().map(|&f| cortiq_core::quant::f16_to_f32(cortiq_core::quant::f32_to_f16(f)) as f64).collect();
        let xr: Vec<f64> = x.iter().map(|&f| cortiq_core::quant::f16_to_f32(cortiq_core::quant::f32_to_f16(f)) as f64).collect();
        let mut loss = vec![0f64; ntok];
        let mut dx = vec![0f64; ntok * hs];
        for t in 0..ntok {
            let s: Vec<f64> = (0..v).map(|j| (0..hs).map(|i| xr[t * hs + i] * wr[j * hs + i]).sum()).collect();
            let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = mx + s.iter().map(|z| (z - mx).exp()).sum::<f64>().ln();
            let mut d: Vec<f64> = s.iter().map(|z| (z - lse).exp() / ntok as f64).collect();
            for q in 0..k {
                let (id, pt) = (tid[t * k + q] as usize, tp[t * k + q] as f64);
                loss[t] += pt * (pt.ln() - (s[id] - lse));
                d[id] -= pt / ntok as f64;
            }
            for i in 0..hs {
                dx[t * hs + i] = (0..v).map(|j| d[j] * wr[j * hs + i]).sum();
            }
        }
        let rel = |name: &str, got: &[f32], want: &[f64], tol: f64| {
            let num: f64 = got.iter().zip(want).map(|(g, w)| (*g as f64 - w).powi(2)).sum();
            let den: f64 = want.iter().map(|w| w * w).sum::<f64>().max(1e-30);
            let e = (num / den).sqrt();
            eprintln!("{name}: relative error {e:e}");
            assert!(e < tol, "{name}: relative error {e:e}");
        };
        rel("loss", &download(c, &bloss, ntok).unwrap(), &loss, 1e-4);
        rel("dx", &download(c, &bdx, ntok * hs).unwrap(), &dx, 3e-3);
    }
}
