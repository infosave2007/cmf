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
"#;

fn module(c: &Ctx) -> Option<&'static train_ops::Mod> {
    train_ops::module(c, "train_head", SRC, &["kl_rows"])
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

/// The frozen head: `w` as an f16 plane `[v][hs]` and its transpose `[hs][v]`.
pub struct Head {
    pub v: usize,
    pub hs: usize,
    pub w: wgpu::Buffer,
    pub wt: wgpu::Buffer,
}

/// Scratch for `chunk` tokens at a time.
pub struct HeadScratch {
    pub chunk: usize,
    act: wgpu::Buffer,
    logits: wgpu::Buffer,
    dact: wgpu::Buffer,
}

impl HeadScratch {
    pub fn new(c: &Ctx, chunk: usize, v: usize, hs: usize) -> Self {
        HeadScratch {
            chunk,
            act: train_mm::f16_buf(c, train_mm::pad_rows(chunk) * hs.max(v)),
            logits: train_ops::zeros(c, train_mm::pad_rows(chunk) * v),
            dact: train_ops::zeros(c, train_mm::pad_rows(chunk) * hs),
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
        let xin = train_ops::zeros(c, train_mm::pad_rows(rows) * hs);
        let tidc = train_ops::zeros(c, rows * k);
        let tpc = train_ops::zeros(c, rows * k);
        let lossc = train_ops::zeros(c, rows);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("head_in") });
        enc.copy_buffer_to_buffer(x, (r0 * hs * 4) as u64, &xin, 0, (rows * hs * 4) as u64);
        enc.copy_buffer_to_buffer(tid, (r0 * k * 4) as u64, &tidc, 0, (rows * k * 4) as u64);
        enc.copy_buffer_to_buffer(tp, (r0 * k * 4) as u64, &tpc, 0, (rows * k * 4) as u64);
        super::submit(c, enc.finish());
        let ops = vec![
            Op::D(train_mm::cvt(c, &xin, None, &s.act, rows, hs, 1.0)?),
            Op::M(train_mm::mm(c, &head.w, &s.act, &s.logits, rows, v, hs, 1.0)?),
            Op::D(kl_rows(c, &s.logits, &tidc, &tpc, &lossc, rows, v, k, gscale / total as f32)?),
            Op::D(train_mm::cvt(c, &s.logits, None, &s.act, rows, v, 1.0)?),
            Op::M(train_mm::mm(c, &head.wt, &s.act, &s.dact, rows, hs, v, 1.0 / gscale)?),
        ];
        train_mm::run(c, &ops);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("head_out") });
        enc.copy_buffer_to_buffer(&s.dact, 0, dx, (r0 * hs * 4) as u64, (rows * hs * 4) as u64);
        enc.copy_buffer_to_buffer(&lossc, 0, loss, (r0 * 4) as u64, (rows * 4) as u64);
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
        let head = Head { v, hs, w: buf16(&to16(&w)), wt: buf16(&to16(&wt)) };
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
