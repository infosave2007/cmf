//! Tensor-core GEMMs of the trainer and the binary low-rank (`blr`) layer
//! built on them.
//!
//! The GEMM is `zi_mm` (f16 operands, f32 accumulators): `out[M][N] =
//! act[M][K] · plane[N][K]ᵀ`, N a multiple of 128, K of 32, the activation
//! rows padded to 128. Operands are staged by `cvt` (f32 → f16, scaled per
//! column and by a power of two that keeps gradients out of the f16
//! subnormals; the GEMM's output multiplier divides it back out).
//!
//! A `blr` path `Ŵ = diag(h)·S_U·diag(ℓ)·S_Vᵀ·diag(g)` keeps its signs as
//! bitmaps (1 bit per sign) and expands them per layer into the four f16
//! planes the GEMMs read (S_U, S_Uᵀ, S_V, S_Vᵀ — ±1 is exact in f16):
//!
//! ```text
//! forward   a = (x⊙g)·S_V      c = (a⊙ℓ)·S_Uᵀ      y += h⊙c
//! backward  dh = Σ dy⊙c        db = (dy⊙h)·S_U      dℓ = Σ a⊙db
//!           dx' = (db⊙ℓ)·S_Vᵀ  dg = Σ x⊙dx'         dx += g⊙dx'
//! ```

use super::train_ops::{self, Disp, elt_fold};
use super::zimage::{self as zi, Epi};
use super::{Ctx, submit};

const SRC: &str = r#"
enable f16;
struct P { m: u32, k: u32, mpad: u32, rows: u32, r: u32, words: u32, pad0: u32, pad1: u32, scale: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> COL: array<f32>;
@group(0) @binding(3) var<storage, read_write> DST: array<f16>;
@group(0) @binding(4) var<storage, read_write> BITS: array<u32>;
@group(0) @binding(5) var<storage, read_write> PL: array<f16>;
@group(0) @binding(6) var<storage, read_write> PLT: array<f16>;
@group(0) @binding(7) var<storage, read_write> A: array<f32>;
@group(0) @binding(8) var<storage, read_write> B: array<f32>;
@group(0) @binding(9) var<storage, read_write> OUT: array<f32>;

fn idx(g: vec3<u32>) -> u32 { return g.x + g.y * 65535u * 256u; }

// DST[mpad][k] = f16(SRC[m][k] · COL[k] · scale), zero rows past m.
@compute @workgroup_size(256)
fn cvt_col(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.mpad * p.k) { return; }
    let row = i / p.k;
    let col = i % p.k;
    var v = 0.0;
    if (row < p.m) { v = SRC[i] * COL[col] * p.scale; }
    DST[i] = f16(v);
}

// Same without the column factor.
@compute @workgroup_size(256)
fn cvt(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.mpad * p.k) { return; }
    var v = 0.0;
    if (i / p.k < p.m) { v = SRC[i] * p.scale; }
    DST[i] = f16(v);
}

// Sign bitmap [rows][words] (bit j of word w = rank 32w+j, 1 = +1) →
// plane PL[rows][r] and its transpose PLT[r][rows].
@compute @workgroup_size(256)
fn expand(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.rows * p.r) { return; }
    let row = i / p.r;
    let j = i % p.r;
    let bit = (BITS[row * p.words + j / 32u] >> (j % 32u)) & 1u;
    let v = select(f16(-1.0), f16(1.0), bit == 1u);
    PL[i] = v;
    PLT[j * p.rows + row] = v;
}

// OUT[col] += Σ_row A[row][col]·B[row][col]   (one thread per column)
@compute @workgroup_size(256)
fn colsum_prod(@builtin(global_invocation_id) g: vec3<u32>) {
    let col = g.x;
    if (col >= p.k) { return; }
    var acc = 0.0;
    for (var row = 0u; row < p.m; row = row + 1u) { acc = acc + A[row * p.k + col] * B[row * p.k + col]; }
    OUT[col] = OUT[col] + acc;
}

// Small dense f32 linear: OUT[m][r] (+)= A[m][k] · B[r][k]ᵀ  (rows flag: accumulate)
// one thread per output element; for the few-row projections (GDN a/b).
@compute @workgroup_size(256)
fn dense_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.m * p.r) { return; }
    let row = i / p.r;
    let o = i % p.r;
    var acc = 0.0;
    for (var j = 0u; j < p.k; j = j + 1u) { acc = acc + A[row * p.k + j] * B[o * p.k + j]; }
    if (p.rows == 1u) { OUT[i] = OUT[i] + acc; } else { OUT[i] = acc; }
}

// OUT[m][k] += A[m][r] · B[r][k]   (input gradient of dense_fwd)
@compute @workgroup_size(256)
fn dense_bwd(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.m * p.k) { return; }
    let row = i / p.k;
    let j = i % p.k;
    var acc = 0.0;
    for (var o = 0u; o < p.r; o = o + 1u) { acc = acc + A[row * p.r + o] * B[o * p.k + j]; }
    OUT[i] = OUT[i] + acc;
}

// Per-head [q(hd); gate(hd)] rows of A ([m][heads][2hd]) -> OUT = q, B = gate
// (rows flag 0); rows flag 1: the reverse (OUT, B -> A).  k = heads·hd, r = hd.
@compute @workgroup_size(256)
fn split_qg(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.m * p.k) { return; }
    let row = i / p.k;
    let c = i % p.k;
    let h = c / p.r;
    let d = c % p.r;
    let src = row * 2u * p.k + h * 2u * p.r + d;
    if (p.rows == 0u) {
        OUT[i] = A[src];
        B[i] = A[src + p.r];
    } else {
        A[src] = OUT[i];
        A[src + p.r] = B[i];
    }
}

// Adam (no weight decay) over k entries: SRC = params, A = grads,
// B = first moment, OUT = second moment, COL = mask (rows = 1: masked).
// scale = lr, f1 = beta1, f2 = beta2, f3 = eps, m = step (1-based);
// r = 1: relative steps (× max(|w|, pad1)).
@compute @workgroup_size(256)
fn adam(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.k) { return; }
    if (p.rows == 1u && COL[i] == 0.0) { return; }
    let gr = A[i] * bitcast<f32>(p.pad0);
    let m1 = p.f1 * B[i] + (1.0 - p.f1) * gr;
    let m2 = p.f2 * OUT[i] + (1.0 - p.f2) * gr * gr;
    B[i] = m1;
    OUT[i] = m2;
    let t = f32(p.m);
    let mh = m1 / (1.0 - pow(p.f1, t));
    let vh = m2 / (1.0 - pow(p.f2, t));
    let st = p.scale * mh / (sqrt(vh) + p.f3);
    // r = 1: the step is relative to the value's magnitude, floored at pad1
    if (p.r == 1u) {
        SRC[i] = SRC[i] - st * max(abs(SRC[i]), bitcast<f32>(p.pad1));
    } else {
        SRC[i] = SRC[i] - st;
    }
}

// Partial sums of squares of A[0..k] per workgroup into OUT[wg]
// (non-finite values propagate, which is the point: the host checks).
var<workgroup> ssq: array<f32, 256>;
@compute @workgroup_size(256)
fn sumsq(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    var acc = 0.0;
    for (var i = g.x; i < p.k; i = i + 256u * p.m) { acc = acc + A[i] * A[i]; }
    ssq[t] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { ssq[t] = ssq[t] + ssq[t + s]; }
        workgroupBarrier();
    }
    if (t == 0u) { OUT[wg.x] = ssq[0]; }
}

// OUT[m][k] (+)= A[m][k]·COL[k]   (rows = 1: accumulate, 0: overwrite)
@compute @workgroup_size(256)
fn scale_acc(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.m * p.k) { return; }
    let v = A[i] * COL[i % p.k];
    if (p.rows == 1u) { OUT[i] = OUT[i] + v; } else { OUT[i] = v; }
}
"#;

fn module(c: &Ctx) -> Option<&'static train_ops::Mod> {
    train_ops::module(c, "train_mm", SRC, &["cvt_col", "cvt", "expand", "colsum_prod", "scale_acc", "dense_fwd", "dense_bwd", "split_qg", "adam", "sumsq"])
}

/// One step of a trainer pass: an elementwise/reduction dispatch or a GEMM.
pub enum Op<'a> {
    D(Disp<'a>),
    M(zi::MmCall),
}

/// Record every op into one compute pass and submit.
pub fn run(c: &Ctx, ops: &[Op]) {
    let mut keep = Vec::new();
    let mut groups = Vec::new();
    for op in ops {
        if let Op::D(d) = op {
            let mut words = d.uni.clone();
            while words.len() % 4 != 0 {
                words.push(0);
            }
            let u = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("train_p"),
                size: (words.len() * 4) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            c.queue.write_buffer(&u, 0, bytemuck::cast_slice(&words));
            let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() }];
            for &(b, buf) in &d.bufs {
                entries.push(wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() });
            }
            groups.push(Some(c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("train"),
                layout: &d.pipe.get_bind_group_layout(0),
                entries: &entries,
            })));
            keep.push(u);
        } else {
            groups.push(None);
        }
    }
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("train_mm") });
    {
        let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        for (op, bg) in ops.iter().zip(&groups) {
            match op {
                Op::D(d) => {
                    ps.set_pipeline(d.pipe);
                    ps.set_bind_group(0, bg.as_ref().unwrap(), &[]);
                    ps.dispatch_workgroups(d.grid.0, d.grid.1, d.grid.2);
                }
                Op::M(m) => m.record(&mut ps),
            }
        }
    }
    submit(c, enc.finish());
}

/// Rows the GEMM's activation operand is padded to.
pub const MPAD: usize = 128;

pub fn pad_rows(m: usize) -> usize {
    m.div_ceil(MPAD) * MPAD
}

/// f16 buffer of `n` halves (rounded up to whole words).
pub fn f16_buf(c: &Ctx, n: usize) -> wgpu::Buffer {
    c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("train_f16"),
        size: ((n.div_ceil(2) * 4).max(16)) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// An f16 plane holding `v` (row-major, as given).
pub fn f16_from_f32(c: &Ctx, v: &[f32]) -> wgpu::Buffer {
    let mut h: Vec<u16> = v.iter().map(|&f| cortiq_core::quant::f32_to_f16(f)).collect();
    if h.len() % 2 == 1 {
        h.push(0);
    }
    let b = f16_buf(c, h.len());
    c.queue.write_buffer(&b, 0, bytemuck::cast_slice(&h));
    b
}

/// Stage `src` (`[m][k]` f32) as an f16 activation operand, rows padded,
/// each column multiplied by `col[k]` (if given) and everything by `scale`.
#[allow(clippy::too_many_arguments)]
pub fn cvt<'a>(c: &'a Ctx, src: &'a wgpu::Buffer, col: Option<&'a wgpu::Buffer>, dst: &'a wgpu::Buffer, m: usize, k: usize, scale: f32) -> Option<Disp<'a>> {
    let md = module(c)?;
    let mpad = pad_rows(m);
    let uni = vec![m as u32, k as u32, mpad as u32, 0, 0, 0, 0, 0, scale.to_bits(), 0, 0, 0];
    Some(match col {
        Some(cb) => Disp { pipe: &md.p["cvt_col"], uni, bufs: vec![(1, src), (2, cb), (3, dst)], grid: elt_fold(mpad * k) },
        None => Disp { pipe: &md.p["cvt"], uni, bufs: vec![(1, src), (3, dst)], grid: elt_fold(mpad * k) },
    })
}

/// Expand a sign bitmap `[rows][ceil(r/32)]` into `pl` `[rows][r]` and `plt` `[r][rows]`.
pub fn expand<'a>(c: &'a Ctx, bits: &'a wgpu::Buffer, pl: &'a wgpu::Buffer, plt: &'a wgpu::Buffer, rows: usize, r: usize) -> Option<Disp<'a>> {
    let md = module(c)?;
    Some(Disp {
        pipe: &md.p["expand"],
        uni: vec![0, 0, 0, rows as u32, r as u32, r.div_ceil(32) as u32, 0, 0, 0, 0, 0, 0],
        bufs: vec![(4, bits), (5, pl), (6, plt)],
        grid: elt_fold(rows * r),
    })
}

/// `out[k] += Σ_rows a[row][k]·b[row][k]`.
pub fn colsum_prod<'a>(c: &'a Ctx, a: &'a wgpu::Buffer, b: &'a wgpu::Buffer, out: &'a wgpu::Buffer, m: usize, k: usize) -> Option<Disp<'a>> {
    let md = module(c)?;
    Some(Disp {
        pipe: &md.p["colsum_prod"],
        uni: vec![m as u32, k as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        bufs: vec![(7, a), (8, b), (9, out)],
        grid: (k.div_ceil(256) as u32, 1, 1),
    })
}

/// `out = a ⊙ col` per row (or `out += …` with `acc`).
#[allow(clippy::too_many_arguments)]
pub fn scale_acc<'a>(c: &'a Ctx, a: &'a wgpu::Buffer, col: &'a wgpu::Buffer, out: &'a wgpu::Buffer, m: usize, k: usize, acc: bool) -> Option<Disp<'a>> {
    let md = module(c)?;
    Some(Disp {
        pipe: &md.p["scale_acc"],
        uni: vec![m as u32, k as u32, 0, acc as u32, 0, 0, 0, 0, 0, 0, 0, 0],
        bufs: vec![(2, col), (7, a), (9, out)],
        grid: elt_fold(m * k),
    })
}

/// Small dense linear `out[m][r] (+)= a[m][k]·w[r][k]ᵀ` in f32.
#[allow(clippy::too_many_arguments)]
pub fn dense_fwd<'a>(c: &'a Ctx, a: &'a wgpu::Buffer, w: &'a wgpu::Buffer, out: &'a wgpu::Buffer, m: usize, k: usize, r: usize, acc: bool) -> Option<Disp<'a>> {
    let md = module(c)?;
    Some(Disp {
        pipe: &md.p["dense_fwd"],
        uni: vec![m as u32, k as u32, 0, acc as u32, r as u32, 0, 0, 0, 0, 0, 0, 0],
        bufs: vec![(7, a), (8, w), (9, out)],
        grid: elt_fold(m * r),
    })
}

/// Input gradient of [`dense_fwd`]: `dx[m][k] += dy[m][r]·w[r][k]`.
#[allow(clippy::too_many_arguments)]
pub fn dense_bwd<'a>(c: &'a Ctx, dy: &'a wgpu::Buffer, w: &'a wgpu::Buffer, dx: &'a wgpu::Buffer, m: usize, k: usize, r: usize) -> Option<Disp<'a>> {
    let md = module(c)?;
    Some(Disp {
        pipe: &md.p["dense_bwd"],
        uni: vec![m as u32, k as u32, 0, 0, r as u32, 0, 0, 0, 0, 0, 0, 0],
        bufs: vec![(7, dy), (8, w), (9, dx)],
        grid: elt_fold(m * k),
    })
}

/// Split the per-head `[q; gate]` projection rows into q and gate
/// (`merge` = the reverse, for the gradients).
#[allow(clippy::too_many_arguments)]
pub fn split_qg<'a>(c: &'a Ctx, qg: &'a wgpu::Buffer, q: &'a wgpu::Buffer, gate: &'a wgpu::Buffer, m: usize, heads: usize, hd: usize, merge: bool) -> Option<Disp<'a>> {
    let md = module(c)?;
    Some(Disp {
        pipe: &md.p["split_qg"],
        uni: vec![m as u32, (heads * hd) as u32, 0, merge as u32, hd as u32, 0, 0, 0, 0, 0, 0, 0],
        bufs: vec![(7, qg), (8, gate), (9, q)],
        grid: elt_fold(m * heads * hd),
    })
}

/// One Adam update of `n` parameters (`mask` 0 = frozen entry).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn adam<'a>(c: &'a Ctx, w: &'a wgpu::Buffer, g: &'a wgpu::Buffer, m1: &'a wgpu::Buffer, m2: &'a wgpu::Buffer, mask: Option<&'a wgpu::Buffer>, n: usize, lr: f32, step: u32, b1: f32, b2: f32, eps: f32, gmul: f32, rel: f32) -> Option<Disp<'a>> {
    let md = module(c)?;
    // The mask binding is statically used: without a mask it aliases the
    // parameters (never read: the flag is off).
    let bufs = vec![(1, w), (2, mask.unwrap_or(w)), (7, g), (8, m1), (9, m2)];
    Some(Disp {
        pipe: &md.p["adam"],
        uni: vec![step, n as u32, 0, mask.is_some() as u32, (rel > 0.0) as u32, 0, gmul.to_bits(), rel.to_bits(), lr.to_bits(), b1.to_bits(), b2.to_bits(), eps.to_bits()],
        bufs,
        grid: elt_fold(n),
    })
}

/// Sum of squares of `n` floats of `a` (64 workgroup partials summed on the host).
pub fn sumsq(c: &Ctx, a: &wgpu::Buffer, n: usize) -> Option<f64> {
    let md = module(c)?;
    let out = train_ops::zeros(c, 64);
    let d = Disp { pipe: &md.p["sumsq"], uni: vec![64, n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], bufs: vec![(7, a), (9, &out)], grid: (64, 1, 1) };
    run(c, &[Op::D(d)]);
    Some(train_ops::download(c, &out, 64)?.iter().map(|&x| x as f64).sum())
}

/// `out[m][n] = act[m][k] · plane[n][k]ᵀ · oscale` on the matrix units.
#[allow(clippy::too_many_arguments)]
pub fn mm(c: &Ctx, plane: &wgpu::Buffer, act: &wgpu::Buffer, out: &wgpu::Buffer, m: usize, n: usize, k: usize, oscale: f32) -> Option<zi::MmCall> {
    let g = zi::default_cfg(Epi::F32);
    let a = zi::MmArgs { m: m as u32, n: n as u32, k: k as u32, ldo: n as u32, ocol: 0, arow: 0, oscale, conv: [0; 3] };
    zi::mm_call(c, g, &a, plane, act, out)
}

/// `y[ntok][n] = (x[ntok][k] · scale) · plane[n][k]ᵀ / scale` through the
/// staging buffer `act16`: one dense frozen linear (or its transpose for
/// the input gradient, with a gradient `scale`).
#[allow(clippy::too_many_arguments)]
pub fn plane_linear(c: &Ctx, plane: &wgpu::Buffer, x: &wgpu::Buffer, act16: &wgpu::Buffer, y: &wgpu::Buffer, ntok: usize, n: usize, k: usize, scale: f32) -> Option<()> {
    let ops = vec![Op::D(cvt(c, x, None, act16, ntok, k, scale)?), Op::M(mm(c, plane, act16, y, ntok, n, k, 1.0 / scale)?)];
    run(c, &ops);
    Some(())
}

/// Device state of one binary low-rank path (signs as bitmaps, the metric
/// in f32, its gradients).
pub struct BlrPath {
    pub bits_u: wgpu::Buffer,
    pub bits_v: wgpu::Buffer,
    pub h: wgpu::Buffer,
    pub l: wgpu::Buffer,
    pub g: wgpu::Buffer,
    pub dh: wgpu::Buffer,
    pub dl: wgpu::Buffer,
    pub dg: wgpu::Buffer,
}

/// A binary low-rank linear `y = Σ_p Ŵ_p x` of shape m×n, rank r per path
/// (r a multiple of 128).
pub struct Blr {
    pub m: usize,
    pub n: usize,
    pub r: usize,
    pub paths: Vec<BlrPath>,
}

/// Per-layer scratch of the blr ops for up to `ntok` tokens.
pub struct BlrScratch {
    pub ntok: usize,
    pl_u: wgpu::Buffer,
    plt_u: wgpu::Buffer,
    pl_v: wgpu::Buffer,
    plt_v: wgpu::Buffer,
    xh: wgpu::Buffer,   // f16 act [pad(ntok)][max(m,n,r)]
    a: Vec<wgpu::Buffer>, // f32 [ntok][r] per path
    cbuf: Vec<wgpu::Buffer>, // f32 [ntok][m] per path (pre-h output)
    t1: wgpu::Buffer,   // f32 [ntok][max(m,n)]
    t2: wgpu::Buffer,   // f32 [ntok][r]
}

impl BlrScratch {
    pub fn new(c: &Ctx, ntok: usize, m: usize, n: usize, r: usize, paths: usize) -> Self {
        let mx = m.max(n).max(r);
        let z = |len: usize| train_ops::zeros(c, len);
        BlrScratch {
            ntok,
            pl_u: f16_buf(c, m * r),
            plt_u: f16_buf(c, m * r),
            pl_v: f16_buf(c, n * r),
            plt_v: f16_buf(c, n * r),
            xh: f16_buf(c, pad_rows(ntok) * mx),
            a: (0..paths).map(|_| z(pad_rows(ntok) * r)).collect(),
            cbuf: (0..paths).map(|_| z(pad_rows(ntok) * m)).collect(),
            t1: z(pad_rows(ntok) * m.max(n)),
            t2: z(pad_rows(ntok) * r),
        }
    }
}

/// Forward of `layer` on `x` (`[ntok][n]`) into `y` (`[ntok][m]`, overwritten).
/// Keeps a and c of every path in the scratch for [`blr_backward`].
pub fn blr_forward(c: &Ctx, layer: &Blr, s: &BlrScratch, x: &wgpu::Buffer, y: &wgpu::Buffer, ntok: usize) -> Option<()> {
    let (m, n, r) = (layer.m, layer.n, layer.r);
    for (pi, p) in layer.paths.iter().enumerate() {
        let ops = vec![
            Op::D(expand(c, &p.bits_u, &s.pl_u, &s.plt_u, m, r)?),
            Op::D(expand(c, &p.bits_v, &s.pl_v, &s.plt_v, n, r)?),
            Op::D(cvt(c, x, Some(&p.g), &s.xh, ntok, n, 1.0)?),
            Op::M(mm(c, &s.plt_v, &s.xh, &s.a[pi], ntok, r, n, 1.0)?),
            Op::D(cvt(c, &s.a[pi], Some(&p.l), &s.xh, ntok, r, 1.0)?),
            Op::M(mm(c, &s.pl_u, &s.xh, &s.cbuf[pi], ntok, m, r, 1.0)?),
            Op::D(scale_acc(c, &s.cbuf[pi], &p.h, y, ntok, m, pi > 0)?),
        ];
        run(c, &ops);
    }
    Some(())
}

/// Backward of the last [`blr_forward`] on the same `x`: metric gradients
/// accumulate into each path's dh/dℓ/dg; `dx` (`[ntok][n]`) is overwritten
/// with the input gradient. `gscale` is the power of two the gradients are
/// staged under in f16.
#[allow(clippy::too_many_arguments)]
pub fn blr_backward(c: &Ctx, layer: &Blr, s: &BlrScratch, x: &wgpu::Buffer, dy: &wgpu::Buffer, dx: &wgpu::Buffer, ntok: usize, gscale: f32) -> Option<()> {
    let (m, n, r) = (layer.m, layer.n, layer.r);
    for (pi, p) in layer.paths.iter().enumerate() {
        let ops = vec![
            Op::D(expand(c, &p.bits_u, &s.pl_u, &s.plt_u, m, r)?),
            Op::D(expand(c, &p.bits_v, &s.pl_v, &s.plt_v, n, r)?),
            Op::D(colsum_prod(c, dy, &s.cbuf[pi], &p.dh, ntok, m)?),
            Op::D(cvt(c, dy, Some(&p.h), &s.xh, ntok, m, gscale)?),
            Op::M(mm(c, &s.plt_u, &s.xh, &s.t2, ntok, r, m, 1.0 / gscale)?),
            Op::D(colsum_prod(c, &s.a[pi], &s.t2, &p.dl, ntok, r)?),
            Op::D(cvt(c, &s.t2, Some(&p.l), &s.xh, ntok, r, gscale)?),
            Op::M(mm(c, &s.pl_v, &s.xh, &s.t1, ntok, n, r, 1.0 / gscale)?),
            Op::D(colsum_prod(c, x, &s.t1, &p.dg, ntok, n)?),
            Op::D(scale_acc(c, &s.t1, &p.g, dx, ntok, n, pi > 0)?),
        ];
        run(c, &ops);
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

    /// Two-path blr layer forward and backward against an f64 reference
    /// (tolerance set by the f16 operands: ~1e-3 relative).
    #[test]
    fn blr_layer_matches_host() {
        let Some(c) = train_ops::dev() else { return };
        let (m, n, r, ntok, np) = (256usize, 384usize, 128usize, 70usize, 2usize);
        let mut paths_h = Vec::new();
        let mut paths = Vec::new();
        for pi in 0..np {
            let su: Vec<f32> = rnd(m * r, 10 + pi as u64, 1.0).iter().map(|v| if *v >= 0.0 { 1.0 } else { -1.0 }).collect();
            let sv: Vec<f32> = rnd(n * r, 20 + pi as u64, 1.0).iter().map(|v| if *v >= 0.0 { 1.0 } else { -1.0 }).collect();
            let pack = |s: &[f32], rows: usize| -> Vec<u32> {
                let w = r.div_ceil(32);
                let mut b = vec![0u32; rows * w];
                for row in 0..rows {
                    for j in 0..r {
                        if s[row * r + j] > 0.0 {
                            b[row * w + j / 32] |= 1 << (j % 32);
                        }
                    }
                }
                b
            };
            let h: Vec<f32> = rnd(m, 30 + pi as u64, 0.5).iter().map(|v| 0.02 + v.abs() * 0.05).collect();
            let l: Vec<f32> = rnd(r, 40 + pi as u64, 0.5).iter().map(|v| 0.5 + v.abs()).collect();
            let g: Vec<f32> = rnd(n, 50 + pi as u64, 0.5).iter().map(|v| 0.1 + v.abs() * 0.2).collect();
            let bu = pack(&su, m);
            let bv = pack(&sv, n);
            let ub = |v: &[u32]| {
                let b = train_ops::zeros(c, v.len());
                c.queue.write_buffer(&b, 0, bytemuck::cast_slice(v));
                b
            };
            paths.push(BlrPath { bits_u: ub(&bu), bits_v: ub(&bv), h: upload(c, &h), l: upload(c, &l), g: upload(c, &g), dh: zeros(c, m), dl: zeros(c, r), dg: zeros(c, n) });
            paths_h.push((su, sv, h, l, g));
        }
        let layer = Blr { m, n, r, paths };
        let x = rnd(ntok * n, 60, 2.0);
        let dy = rnd(ntok * m, 61, 1e-3);
        let s = BlrScratch::new(c, ntok, m, n, r, np);
        let (bx, by, bdy, bdx) = (upload(c, &x), zeros(c, pad_rows(ntok) * m), upload(c, &dy), zeros(c, pad_rows(ntok) * n));
        blr_forward(c, &layer, &s, &bx, &by, ntok).expect("forward");
        blr_backward(c, &layer, &s, &bx, &bdy, &bdx, ntok, 1024.0).expect("backward");
        // f64 reference
        let mut y = vec![0f64; ntok * m];
        let mut dx = vec![0f64; ntok * n];
        let mut grads = Vec::new();
        for (su, sv, h, l, g) in &paths_h {
            let mut dh = vec![0f64; m];
            let mut dl = vec![0f64; r];
            let mut dg = vec![0f64; n];
            for t in 0..ntok {
                let a: Vec<f64> = (0..r).map(|k| (0..n).map(|j| x[t * n + j] as f64 * g[j] as f64 * sv[j * r + k] as f64).sum()).collect();
                let cc: Vec<f64> = (0..m).map(|i| (0..r).map(|k| a[k] * l[k] as f64 * su[i * r + k] as f64).sum()).collect();
                for i in 0..m {
                    y[t * m + i] += h[i] as f64 * cc[i];
                    dh[i] += dy[t * m + i] as f64 * cc[i];
                }
                let db: Vec<f64> = (0..r).map(|k| (0..m).map(|i| dy[t * m + i] as f64 * h[i] as f64 * su[i * r + k] as f64).sum()).collect();
                for k in 0..r {
                    dl[k] += a[k] * db[k];
                }
                for j in 0..n {
                    let dxp: f64 = (0..r).map(|k| db[k] * l[k] as f64 * sv[j * r + k] as f64).sum();
                    dg[j] += x[t * n + j] as f64 * dxp;
                    dx[t * n + j] += g[j] as f64 * dxp;
                }
            }
            grads.push((dh, dl, dg));
        }
        let rel = |name: &str, got: &[f32], want: &[f64]| {
            let num: f64 = got.iter().zip(want).map(|(g, w)| (*g as f64 - w).powi(2)).sum();
            let den: f64 = want.iter().map(|w| w * w).sum::<f64>().max(1e-30);
            let e = (num / den).sqrt();
            eprintln!("{name}: relative error {e:e}");
            assert!(e < 3e-3, "{name}: relative error {e:e}");
        };
        rel("y", &download(c, &by, ntok * m).unwrap(), &y);
        rel("dx", &download(c, &bdx, ntok * n).unwrap(), &dx);
        for (pi, (dh, dl, dg)) in grads.iter().enumerate() {
            rel(&format!("dh{pi}"), &download(c, &layer.paths[pi].dh, m).unwrap(), dh);
            rel(&format!("dl{pi}"), &download(c, &layer.paths[pi].dl, r).unwrap(), dl);
            rel(&format!("dg{pi}"), &download(c, &layer.paths[pi].dg, n).unwrap(), dg);
        }
    }
}
