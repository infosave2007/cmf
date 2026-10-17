//! Device linear algebra for the NVG `blr` encoder (docs/BLR_NVG.md §4),
//! f32 on the card.
//!
//! * `gemm` — `C = α·op(A)·op(B) (⊙ E) + β·C` over sub-matrix views
//!   (offset, leading dimension, optional transpose per operand): 128×128
//!   output tiles, 16-deep K steps staged through workgroup memory with a
//!   register prefetch of the next step, 8×8 outputs per thread. Exact f32
//!   FMA (never the matrix units: their operands are f16/tf32 class).
//! * elementwise / reduction kernels the encoder's stages are made of
//!   (row/column scaling, signs, row dots, column sums of products,
//!   Frobenius partials, equilibration, act-order permutation, transpose);
//! * dense SPD solve: equilibrated blocked Cholesky (the GPTQ fold's
//!   kernels, `gptq_dev::chol_lower`) and blocked triangular solves;
//! * `flipu` — the output-side exact single-bit flips, one thread per row;
//! * `gv_block` — the input-side column-sequential sign search of one
//!   128-column block in the latent space (see `cortiq-cli` `nvg_gpu.rs`).
//!
//! Everything works on [`DMat`] (a row-major f32 storage buffer with its
//! shape). Each call records its dispatches and submits; readbacks wait.

use super::train_ops::{self, Disp, Mod, elt_fold};
use super::{Ctx, ctx, submit};

/// A row-major f32 matrix on the device.
pub struct DMat {
    pub buf: wgpu::Buffer,
    pub r: usize,
    pub c: usize,
}

fn dev() -> &'static Ctx {
    ctx().expect("nvg_dev: no wgpu device")
}

/// The device is up and the NVG kernels compile on it.
pub fn available() -> bool {
    ctx().is_some_and(|c| elt(c).is_some() && gemm_mod(c, false, false).is_some())
}

impl DMat {
    pub fn zeros(r: usize, c: usize) -> DMat {
        let cx = dev();
        let buf = cx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nvg_mat"),
            size: ((r * c).max(4) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        DMat { buf, r, c }
    }
    pub fn from_f32(r: usize, c: usize, v: &[f32]) -> DMat {
        assert_eq!(v.len(), r * c);
        let m = DMat::zeros(r, c);
        if !v.is_empty() {
            dev().queue.write_buffer(&m.buf, 0, bytemuck::cast_slice(v));
        }
        m
    }
    pub fn from_f64(r: usize, c: usize, v: &[f64]) -> DMat {
        let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        DMat::from_f32(r, c, &f)
    }
    pub fn write_f64(&self, v: &[f64]) {
        assert!(v.len() <= self.r * self.c);
        let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        dev().queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(&f));
    }
    pub fn len(&self) -> usize {
        self.r * self.c
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn to_f32(&self) -> Vec<f32> {
        train_ops::download(dev(), &self.buf, self.len()).expect("nvg_dev readback")
    }
    pub fn to_f64(&self) -> Vec<f64> {
        self.to_f32().into_iter().map(|x| x as f64).collect()
    }
    /// A device copy.
    pub fn dup(&self) -> DMat {
        let o = DMat::zeros(self.r, self.c);
        copy(self, &o);
        o
    }
}

/// `dst ← src` (same element count).
pub fn copy(src: &DMat, dst: &DMat) {
    assert_eq!(src.len(), dst.len());
    let c = dev();
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nvg_copy") });
    enc.copy_buffer_to_buffer(&src.buf, 0, &dst.buf, 0, (src.len() * 4) as u64);
    submit(c, enc.finish());
}

/// Wait until every submitted dispatch has finished (stage timing).
pub fn sync() {
    let _ = dev().device.poll(wgpu::PollType::wait_indefinitely());
}

// ───────────────────────────── GEMM ─────────────────────────────

fn gemm_src(ta: bool, tb: bool) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    s.push_str(
        r#"
struct GP { m: u32, n: u32, k: u32, lda: u32, ldb: u32, ldc: u32, oa: u32, ob: u32, oc: u32, flags: u32, alpha: f32, beta: f32 }
@group(0) @binding(0) var<uniform> p: GP;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> E: array<f32>;
var<workgroup> sa: array<f32, 2112>;
var<workgroup> sb: array<f32, 2112>;

@compute @workgroup_size(256)
fn gemm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wg.y * 128u;
    let col0 = wg.x * 128u;
    let tx = li % 16u;
    let ty = li / 16u;
"#,
    );
    // per-thread load coordinates (8 elements of each tile)
    for e in 0..8 {
        let _ = writeln!(s, "    let ia{e} = li + {}u;", 256 * e);
    }
    // A element (i, k) of the tile for load e
    let a_ik = |e: usize| -> (String, String) {
        if !ta {
            (format!("(ia{e} / 16u)"), format!("(ia{e} % 16u)"))
        } else {
            (format!("(ia{e} % 128u)"), format!("(ia{e} / 128u)"))
        }
    };
    let b_kj = |e: usize| -> (String, String) {
        if !tb {
            (format!("(ia{e} / 128u)"), format!("(ia{e} % 128u)"))
        } else {
            (format!("(ia{e} % 16u)"), format!("(ia{e} / 16u)"))
        }
    };
    // load functions
    let _ = writeln!(s, "    var pa: array<f32, 8>;\n    var pb: array<f32, 8>;");
    let load = |s: &mut String, k0: &str| {
        for e in 0..8 {
            let (i, k) = a_ik(e);
            let addr = if !ta {
                format!("p.oa + (row0 + {i}) * p.lda + {k0} + {k}")
            } else {
                format!("p.oa + ({k0} + {k}) * p.lda + row0 + {i}")
            };
            let _ = writeln!(
                s,
                "        {{ let gi = row0 + {i}; let gk = {k0} + {k}; var v = 0.0; if (gi < p.m && gk < p.k) {{ v = A[{addr}]; }} pa[{e}] = v; }}"
            );
            let (k, j) = b_kj(e);
            let addr = if !tb {
                format!("p.ob + ({k0} + {k}) * p.ldb + col0 + {j}")
            } else {
                format!("p.ob + (col0 + {j}) * p.ldb + {k0} + {k}")
            };
            let _ = writeln!(
                s,
                "        {{ let gj = col0 + {j}; let gk = {k0} + {k}; var v = 0.0; if (gj < p.n && gk < p.k) {{ v = B[{addr}]; }} pb[{e}] = v; }}"
            );
        }
    };
    let store = |s: &mut String| {
        for e in 0..8 {
            let (i, k) = a_ik(e);
            let _ = writeln!(s, "        sa[{k} * 132u + {i}] = pa[{e}];");
            let (k, j) = b_kj(e);
            let _ = writeln!(s, "        sb[{k} * 132u + {j}] = pb[{e}];");
        }
    };
    for i in 0..16 {
        let _ = writeln!(s, "    var c{i} = vec4<f32>(0.0);");
    }
    s.push_str("    {\n");
    load(&mut s, "0u");
    s.push_str("    }\n");
    s.push_str("    for (var k0 = 0u; k0 < p.k; k0 = k0 + 16u) {\n        workgroupBarrier();\n");
    store(&mut s);
    s.push_str("        workgroupBarrier();\n        let kn = k0 + 16u;\n        if (kn < p.k) {\n");
    load(&mut s, "kn");
    s.push_str("        }\n        for (var kk = 0u; kk < 16u; kk = kk + 1u) {\n");
    s.push_str("            let ba = kk * 132u;\n");
    s.push_str("            let a0 = vec4<f32>(sa[ba + ty * 4u], sa[ba + ty * 4u + 1u], sa[ba + ty * 4u + 2u], sa[ba + ty * 4u + 3u]);\n");
    s.push_str("            let a1 = vec4<f32>(sa[ba + 64u + ty * 4u], sa[ba + 64u + ty * 4u + 1u], sa[ba + 64u + ty * 4u + 2u], sa[ba + 64u + ty * 4u + 3u]);\n");
    s.push_str("            let b0 = vec4<f32>(sb[ba + tx * 4u], sb[ba + tx * 4u + 1u], sb[ba + tx * 4u + 2u], sb[ba + tx * 4u + 3u]);\n");
    s.push_str("            let b1 = vec4<f32>(sb[ba + 64u + tx * 4u], sb[ba + 64u + tx * 4u + 1u], sb[ba + 64u + tx * 4u + 2u], sb[ba + 64u + tx * 4u + 3u]);\n");
    // c{2*i + h}: row i (0..8: a0.x..a0.w, a1.x..a1.w), col half h (b0 / b1)
    let comps = ["x", "y", "z", "w"];
    for i in 0..8 {
        let av = if i < 4 { format!("a0.{}", comps[i]) } else { format!("a1.{}", comps[i - 4]) };
        let _ = writeln!(s, "            c{} = fma(vec4<f32>({av}), b0, c{});", 2 * i, 2 * i);
        let _ = writeln!(s, "            c{} = fma(vec4<f32>({av}), b1, c{});", 2 * i + 1, 2 * i + 1);
    }
    s.push_str("        }\n    }\n");
    // epilogue
    for i in 0..8 {
        let row = if i < 4 { format!("row0 + ty * 4u + {i}u") } else { format!("row0 + 64u + ty * 4u + {}u", i - 4) };
        let _ = writeln!(s, "    {{\n        let gi = {row};\n        if (gi < p.m) {{");
        for h in 0..2 {
            for q in 0..4 {
                let col = if h == 0 { format!("col0 + tx * 4u + {q}u") } else { format!("col0 + 64u + tx * 4u + {q}u") };
                let _ = writeln!(
                    s,
                    "            {{ let gj = {col}; if (gj < p.n) {{ let ix = p.oc + gi * p.ldc + gj; var v = p.alpha * c{}.{}; if ((p.flags & 1u) != 0u) {{ v = v * E[ix]; }} if ((p.flags & 2u) != 0u) {{ v = v + p.beta * C[ix]; }} C[ix] = v; }} }}",
                    2 * i + h,
                    comps[q]
                );
            }
        }
        s.push_str("        }\n    }\n");
    }
    s.push_str("}\n");
    s
}

static GEMM_SRCS: std::sync::OnceLock<[&'static str; 4]> = std::sync::OnceLock::new();

fn gemm_mod(c: &Ctx, ta: bool, tb: bool) -> Option<&'static Mod> {
    let srcs = GEMM_SRCS.get_or_init(|| {
        [
            Box::leak(gemm_src(false, false).into_boxed_str()),
            Box::leak(gemm_src(false, true).into_boxed_str()),
            Box::leak(gemm_src(true, false).into_boxed_str()),
            Box::leak(gemm_src(true, true).into_boxed_str()),
        ]
    });
    let (name, k) = match (ta, tb) {
        (false, false) => ("nvg_gemm_nn", 0),
        (false, true) => ("nvg_gemm_nt", 1),
        (true, false) => ("nvg_gemm_tn", 2),
        (true, true) => ("nvg_gemm_tt", 3),
    };
    train_ops::module(c, name, srcs[k], &["gemm"])
}

/// An operand view: element (i, k) of `op(M)` is `M[off + i·ld + k]`, or
/// `M[off + k·ld + i]` when `t`.
#[derive(Clone, Copy)]
pub struct V<'a> {
    pub m: &'a DMat,
    pub off: usize,
    pub ld: usize,
    pub t: bool,
}

impl<'a> V<'a> {
    /// The whole matrix, as stored.
    pub fn n(m: &'a DMat) -> V<'a> {
        V { m, off: 0, ld: m.c, t: false }
    }
    /// The whole matrix, transposed.
    pub fn t(m: &'a DMat) -> V<'a> {
        V { m, off: 0, ld: m.c, t: true }
    }
}

fn dummy(c: &Ctx) -> &'static wgpu::Buffer {
    static D: std::sync::OnceLock<wgpu::Buffer> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nvg_dummy"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    })
}

/// `C[oc + i·ldc + j] = α·Σ_k op(A)(i,k)·op(B)(k,j) [⊙ E] + β·C` for an
/// `m×n` output (`k` the inner dimension). `E` (same layout as C) when given.
#[allow(clippy::too_many_arguments)]
pub fn gemm_v(m: usize, n: usize, k: usize, alpha: f32, a: V, b: V, beta: f32, c: &DMat, oc: usize, ldc: usize, e: Option<&DMat>) {
    if m == 0 || n == 0 {
        return;
    }
    let cx = dev();
    let md = gemm_mod(cx, a.t, b.t).expect("nvg gemm pipeline");
    let flags = (e.is_some() as u32) | (((beta != 0.0) as u32) << 1);
    let uni = vec![
        m as u32, n as u32, k as u32, a.ld as u32, b.ld as u32, ldc as u32, a.off as u32, b.off as u32, oc as u32, flags,
        alpha.to_bits(), beta.to_bits(),
    ];
    let d = Disp {
        pipe: &md.p["gemm"],
        uni,
        bufs: vec![(1, &a.m.buf), (2, &b.m.buf), (3, &c.buf), (4, e.map(|x| &x.buf).unwrap_or(dummy(cx)))],
        grid: (n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1),
    };
    train_ops::run(cx, &[d]);
}

/// `C = α·op(A)·op(B) + β·C` on whole matrices.
pub fn gemm(alpha: f32, a: &DMat, ta: bool, b: &DMat, tb: bool, beta: f32, c: &DMat) {
    let (m, k) = if ta { (a.c, a.r) } else { (a.r, a.c) };
    let (k2, n) = if tb { (b.c, b.r) } else { (b.r, b.c) };
    assert_eq!(k, k2, "nvg gemm inner dims");
    assert!(c.r == m && c.c == n, "nvg gemm output dims");
    let va = V { m: a, off: 0, ld: a.c, t: ta };
    let vb = V { m: b, off: 0, ld: b.c, t: tb };
    gemm_v(m, n, k, alpha, va, vb, beta, c, 0, n, None);
}

/// `op(A)·op(B)` into a new matrix.
pub fn mm(a: &DMat, ta: bool, b: &DMat, tb: bool) -> DMat {
    let m = if ta { a.c } else { a.r };
    let n = if tb { b.r } else { b.c };
    let c = DMat::zeros(m, n);
    gemm(1.0, a, ta, b, tb, 0.0, &c);
    c
}

// ───────────────────────────── elementwise / reductions ─────────────────────────────

const ELT_SRC: &str = r#"
struct P { u0: u32, u1: u32, u2: u32, u3: u32, u4: u32, u5: u32, u6: u32, u7: u32, u8: u32, u9: u32, u10: u32, u11: u32, f0: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read> R: array<f32>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read_write> C: array<f32>;
@group(0) @binding(6) var<storage, read_write> O: array<f32>;
@group(0) @binding(7) var<storage, read> U: array<u32>;
var<workgroup> red: array<f32, 256>;
var<workgroup> tile: array<f32, 1056>;

fn idx(g: vec3<u32>) -> u32 { return g.x + g.y * 65535u * 256u; }
fn sgn(v: f32) -> f32 { return select(-1.0, 1.0, v >= 0.0); }

// C[oc + i*ldc + j] = alpha * f(X[i,j]) * R[i] * S[j] (+ beta*C), X = A (or C when in place).
// u0 rows, u1 cols, u2 lda, u3 oa, u4 ldc, u5 oc, u6 flags:
// 1 row scale, 2 col scale, 4 accumulate (beta), 8 lower mask (j > i -> 0),
// 16 sign, 32 abs, 64 in place (X = C at the output index).
@compute @workgroup_size(256)
fn scale_rc(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = idx(g);
    if (t >= p.u0 * p.u1) { return; }
    let i = t / p.u1;
    let j = t % p.u1;
    let fl = p.u6;
    let ix = p.u5 + i * p.u4 + j;
    var v = 0.0;
    if ((fl & 64u) != 0u) { v = C[ix]; } else { v = A[p.u3 + i * p.u2 + j]; }
    if ((fl & 16u) != 0u) { v = sgn(v); }
    if ((fl & 32u) != 0u) { v = abs(v); }
    v = v * p.f0;
    if ((fl & 1u) != 0u) { v = v * R[i]; }
    if ((fl & 2u) != 0u) { v = v * S[j]; }
    if ((fl & 8u) != 0u && j > i) { v = 0.0; }
    if ((fl & 4u) != 0u) { v = v + p.f1 * C[ix]; }
    C[ix] = v;
}

fn wg_sum(t: u32, v: f32) -> f32 {
    red[t] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

// O[i] = sum_j A[oa + i*lda + j] * B[ob + i*ldb + j], j < u1; one workgroup per row.
// u0 rows, u1 cols, u2 lda, u3 oa, u4 ldb, u5 ob
@compute @workgroup_size(256)
fn rowdot(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let i = wg.x + wg.y * 65535u;
    if (i >= p.u0) { return; }
    var acc = 0.0;
    for (var j = t; j < p.u1; j = j + 256u) { acc = acc + A[p.u3 + i * p.u2 + j] * B[p.u5 + i * p.u4 + j]; }
    let s = wg_sum(t, acc);
    if (t == 0u) { O[i] = s; }
}

// In place: C[oc + i*ldc + j] /= max(||row i||, tiny); one workgroup per row.
// u0 rows, u1 cols, u4 ldc, u5 oc
@compute @workgroup_size(256)
fn rownorm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let i = wg.x + wg.y * 65535u;
    if (i >= p.u0) { return; }
    var acc = 0.0;
    for (var j = t; j < p.u1; j = j + 256u) { let v = C[p.u5 + i * p.u4 + j]; acc = acc + v * v; }
    let s = wg_sum(t, acc);
    let inv = 1.0 / max(sqrt(s), 1.0e-30);
    for (var j = t; j < p.u1; j = j + 256u) { C[p.u5 + i * p.u4 + j] = C[p.u5 + i * p.u4 + j] * inv; }
}

// Column partial sums: O[ch*u1 + j] = sum over rows of chunk ch of A[oa+i*lda+j]*B[ob+i*ldb+j].
// u0 rows, u1 cols, u2 lda, u3 oa, u4 ldb, u5 ob, u6 rows per chunk; grid (cols/256, chunks)
@compute @workgroup_size(256)
fn colsum_part(@builtin(global_invocation_id) g: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let j = g.x;
    if (j >= p.u1) { return; }
    let ch = wg.y;
    let r0 = ch * p.u6;
    let r1 = min(r0 + p.u6, p.u0);
    var acc = 0.0;
    for (var i = r0; i < r1; i = i + 1u) { acc = acc + A[p.u3 + i * p.u2 + j] * B[p.u5 + i * p.u4 + j]; }
    O[ch * p.u1 + j] = acc;
}

// Frobenius partials: O[wg] = sum over the workgroup's stride of A[i]*B[i], i < u0; u1 workgroups.
@compute @workgroup_size(256)
fn frob_part(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    var acc = 0.0;
    for (var i = g.x; i < p.u0; i = i + 256u * p.u1) { acc = acc + A[i] * B[i]; }
    let s = wg_sum(t, acc);
    if (t == 0u) { O[wg.x] = s; }
}

// O[i] = A[oa + i*lda + i]; u0 n, u2 lda, u3 oa
@compute @workgroup_size(256)
fn diag(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.u0) { O[i] = A[p.u3 + i * p.u2 + i]; }
}

// C[i,j] = (A[a,b] + (a == b) * f0) / (R[a] * R[b]); (a, b) = (i, j), or with
// u6 & 1 the reversed act order a = U[n-1-i], b = U[n-1-j]. u0 n.
@compute @workgroup_size(256)
fn equil(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = idx(g);
    let n = p.u0;
    if (t >= n * n) { return; }
    let i = t / n;
    let j = t % n;
    var a = i;
    var b = j;
    if ((p.u6 & 1u) != 0u) { a = U[n - 1u - i]; b = U[n - 1u - j]; }
    var v = A[a * n + b];
    if (a == b) { v = v + p.f0; }
    C[t] = v / (R[a] * R[b]);
}

// C[j][k] = A[(n-1-j)*n + (n-1-k)] / R[k] for k >= j, else 0. u0 n.
@compute @workgroup_size(256)
fn make_uc(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = idx(g);
    let n = p.u0;
    if (t >= n * n) { return; }
    let j = t / n;
    let k = t % n;
    var v = 0.0;
    if (k >= j) { v = A[(n - 1u - j) * n + (n - 1u - k)] / R[k]; }
    C[t] = v;
}

// Row gather (u6 = 0): C[j,:] = A[U[j],:]; scatter (u6 = 1): C[U[j],:] = A[j,:]. u0 rows, u1 cols.
@compute @workgroup_size(256)
fn permute_rows(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = idx(g);
    if (t >= p.u0 * p.u1) { return; }
    let j = t / p.u1;
    let c = t % p.u1;
    if (p.u6 == 0u) { C[t] = A[U[j] * p.u1 + c]; } else { C[U[j] * p.u1 + c] = A[t]; }
}

// C (cols x rows) = A (rows x cols) transposed; 32x32 tiles. u0 rows, u1 cols.
@compute @workgroup_size(256)
fn transpose(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r0 = wg.y * 32u;
    let c0 = wg.x * 32u;
    for (var e = li; e < 1024u; e = e + 256u) {
        let rr = e / 32u;
        let cc = e % 32u;
        var v = 0.0;
        if (r0 + rr < p.u0 && c0 + cc < p.u1) { v = A[(r0 + rr) * p.u1 + c0 + cc]; }
        tile[rr * 33u + cc] = v;
    }
    workgroupBarrier();
    for (var e = li; e < 1024u; e = e + 256u) {
        let cc = e / 32u;
        let rr = e % 32u;
        if (r0 + rr < p.u0 && c0 + cc < p.u1) { C[(c0 + cc) * p.u0 + r0 + rr] = tile[rr * 33u + cc]; }
    }
}

// Forward block solve of the lower factor (A, ld u0) at k0 = u1: C[k0..k0+bs] = L_kk^-1 C[..]
// (u6 = 0), or the transposed back substitution L_kk^-T (u6 = 1). One workgroup of 64.
@compute @workgroup_size(64)
fn trsv_diag(@builtin(local_invocation_index) t: u32) {
    let n = p.u0;
    let k0 = p.u1;
    let bs = min(64u, n - k0);
    var y = 0.0;
    if (t < bs) { y = C[k0 + t]; }
    if (p.u6 == 0u) {
        for (var c = 0u; c < bs; c = c + 1u) {
            if (t == c) { y = y / A[(k0 + c) * n + k0 + c]; red[0] = y; }
            workgroupBarrier();
            let yc = red[0];
            if (t > c && t < bs) { y = y - A[(k0 + t) * n + k0 + c] * yc; }
            workgroupBarrier();
        }
    } else {
        for (var cc = 0u; cc < bs; cc = cc + 1u) {
            let c = bs - 1u - cc;
            if (t == c) { y = y / A[(k0 + c) * n + k0 + c]; red[0] = y; }
            workgroupBarrier();
            let yc = red[0];
            if (t < c) { y = y - A[(k0 + c) * n + k0 + t] * yc; }
            workgroupBarrier();
        }
    }
    if (t < bs) { C[k0 + t] = y; }
}

// Updates after the block at k0 = u1 (bs = min(64, n-k0)): forward (u6 = 0)
// C[i] -= sum_t A[i][k0+t] C[k0+t] for i >= k0+bs; backward (u6 = 1)
// C[i] -= sum_t A[k0+t][i] C[k0+t] for i < k0.
@compute @workgroup_size(256)
fn trsv_upd(@builtin(global_invocation_id) g: vec3<u32>) {
    let n = p.u0;
    let k0 = p.u1;
    let bs = min(64u, n - k0);
    var i = g.x;
    var acc = 0.0;
    if (p.u6 == 0u) {
        i = k0 + bs + g.x;
        if (i >= n) { return; }
        for (var t = 0u; t < bs; t = t + 1u) { acc = acc + A[i * n + k0 + t] * C[k0 + t]; }
    } else {
        if (i >= k0) { return; }
        for (var t = 0u; t < bs; t = t + 1u) { acc = acc + A[(k0 + t) * n + i] * C[k0 + t]; }
    }
    C[i] = C[i] - acc;
}
"#;

fn elt(c: &Ctx) -> Option<&'static Mod> {
    train_ops::module(
        c,
        "nvg_elt",
        ELT_SRC,
        &[
            "scale_rc", "rowdot", "rownorm", "colsum_part", "frob_part", "diag", "equil", "make_uc", "permute_rows", "transpose",
            "trsv_diag", "trsv_upd",
        ],
    )
}

fn uni(u: &[u32], f: &[f32]) -> Vec<u32> {
    let mut w = vec![0u32; 16];
    w[..u.len()].copy_from_slice(u);
    for (k, v) in f.iter().enumerate() {
        w[12 + k] = v.to_bits();
    }
    w
}

/// Scaling flags of [`scale_rc`].
pub const ROW: u32 = 1;
pub const COL: u32 = 2;
pub const ACC: u32 = 4;
pub const LOWER: u32 = 8;
pub const SIGN: u32 = 16;
pub const ABS: u32 = 32;
pub const INPLACE: u32 = 64;

/// `C[oc + i·ldc + j] = α·f(X(i,j))·R[i]·S[j] (+ β·C)` over `rows × cols`,
/// `X(i,j) = A[oa + i·lda + j]` (or C itself with [`INPLACE`]).
#[allow(clippy::too_many_arguments)]
pub fn scale_rc(
    rows: usize,
    cols: usize,
    a: Option<(&DMat, usize, usize)>,
    c: (&DMat, usize, usize),
    r: Option<&DMat>,
    s: Option<&DMat>,
    flags: u32,
    alpha: f32,
    beta: f32,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    let cx = dev();
    let md = elt(cx).expect("nvg elt pipelines");
    let d = dummy(cx);
    let (ab, lda, oa) = match a {
        Some((m, off, ld)) => (&m.buf, ld, off),
        None => (d, 0, 0),
    };
    let flags = flags | if r.is_some() { ROW } else { 0 } | if s.is_some() { COL } else { 0 };
    let dp = Disp {
        pipe: &md.p["scale_rc"],
        uni: uni(&[rows as u32, cols as u32, lda as u32, oa as u32, c.2 as u32, c.1 as u32, flags], &[alpha, beta]),
        bufs: vec![(1, ab), (3, r.map(|x| &x.buf).unwrap_or(d)), (4, s.map(|x| &x.buf).unwrap_or(d)), (5, &c.0.buf)],
        grid: elt_fold(rows * cols),
    };
    train_ops::run(cx, &[dp]);
}

/// Whole-matrix `C = α·f(A)·diag-scalings`.
pub fn scale_into(a: &DMat, c: &DMat, r: Option<&DMat>, s: Option<&DMat>, flags: u32, alpha: f32) {
    assert_eq!((a.r, a.c), (c.r, c.c));
    scale_rc(a.r, a.c, Some((a, 0, a.c)), (c, 0, c.c), r, s, flags, alpha, 0.0);
}

/// `C ← α·A + β·C` (whole matrices).
pub fn axpby(alpha: f32, a: &DMat, beta: f32, c: &DMat) {
    assert_eq!((a.r, a.c), (c.r, c.c));
    scale_rc(a.r, a.c, Some((a, 0, a.c)), (c, 0, c.c), None, None, ACC, alpha, beta);
}

/// `O[i] = Σ_j A[oa+i·lda+j]·B[ob+i·ldb+j]` for `rows` rows of `cols` columns.
#[allow(clippy::too_many_arguments)]
pub fn rowdot(rows: usize, cols: usize, a: &DMat, oa: usize, lda: usize, b: &DMat, ob: usize, ldb: usize, out: &DMat) {
    let cx = dev();
    let md = elt(cx).expect("nvg elt pipelines");
    let dp = Disp {
        pipe: &md.p["rowdot"],
        uni: uni(&[rows as u32, cols as u32, lda as u32, oa as u32, ldb as u32, ob as u32], &[]),
        bufs: vec![(1, &a.buf), (2, &b.buf), (6, &out.buf)],
        grid: train_ops::wg_fold(rows),
    };
    train_ops::run(cx, &[dp]);
}

/// Row-normalise `rows × cols` of `c` (at `oc`, leading dimension `ldc`) in place.
pub fn rownorm(rows: usize, cols: usize, c: &DMat, oc: usize, ldc: usize) {
    let cx = dev();
    let md = elt(cx).expect("nvg elt pipelines");
    let dp = Disp {
        pipe: &md.p["rownorm"],
        uni: uni(&[rows as u32, cols as u32, 0, 0, ldc as u32, oc as u32], &[]),
        bufs: vec![(5, &c.buf)],
        grid: train_ops::wg_fold(rows),
    };
    train_ops::run(cx, &[dp]);
}

/// `Σ_i A[oa+i·lda+j]·B[ob+i·ldb+j]` per column, summed in f64 on the host.
#[allow(clippy::too_many_arguments)]
pub fn colsum_prod(rows: usize, cols: usize, a: &DMat, oa: usize, lda: usize, b: &DMat, ob: usize, ldb: usize) -> Vec<f64> {
    let cx = dev();
    let md = elt(cx).expect("nvg elt pipelines");
    let per = 64usize.max(rows.div_ceil(256));
    let nch = rows.div_ceil(per).max(1);
    let out = DMat::zeros(nch, cols);
    let dp = Disp {
        pipe: &md.p["colsum_part"],
        uni: uni(&[rows as u32, cols as u32, lda as u32, oa as u32, ldb as u32, ob as u32, per as u32], &[]),
        bufs: vec![(1, &a.buf), (2, &b.buf), (6, &out.buf)],
        grid: (cols.div_ceil(256) as u32, nch as u32, 1),
    };
    train_ops::run(cx, &[dp]);
    let v = out.to_f32();
    let mut s = vec![0f64; cols];
    for ch in 0..nch {
        for j in 0..cols {
            s[j] += v[ch * cols + j] as f64;
        }
    }
    s
}

/// `Σ A ⊙ B` over the first `len` elements, f64 on the host.
pub fn frob(a: &DMat, b: &DMat, len: usize) -> f64 {
    let cx = dev();
    let md = elt(cx).expect("nvg elt pipelines");
    let nwg = 512usize.min(len.div_ceil(256)).max(1);
    let out = DMat::zeros(1, nwg);
    let dp = Disp {
        pipe: &md.p["frob_part"],
        uni: uni(&[len as u32, nwg as u32], &[]),
        bufs: vec![(1, &a.buf), (2, &b.buf), (6, &out.buf)],
        grid: (nwg as u32, 1, 1),
    };
    train_ops::run(cx, &[dp]);
    out.to_f32().iter().map(|&x| x as f64).sum()
}

/// Diagonal of the square `n×n` view at `oa` with leading dimension `lda`.
pub fn diag(a: &DMat, n: usize, oa: usize, lda: usize) -> Vec<f64> {
    let cx = dev();
    let md = elt(cx).expect("nvg elt pipelines");
    let out = DMat::zeros(1, n);
    let dp = Disp {
        pipe: &md.p["diag"],
        uni: uni(&[n as u32, 0, lda as u32, oa as u32], &[]),
        bufs: vec![(1, &a.buf), (6, &out.buf)],
        grid: (n.div_ceil(256) as u32, 1, 1),
    };
    train_ops::run(cx, &[dp]);
    out.to_f64()
}

/// A u32 device vector (permutations).
pub fn upload_u32(v: &[u32]) -> wgpu::Buffer {
    let cx = dev();
    let b = cx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nvg_u32"),
        size: (v.len().max(4) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    cx.queue.write_buffer(&b, 0, bytemuck::cast_slice(v));
    b
}

/// `F = (S + ridge·I) / (d dᵀ)` (`d` = `dv`), optionally in reversed act
/// order (`perm`: F[i][j] uses S[perm[n-1-i]][perm[n-1-j]]).
pub fn equil(s: &DMat, f: &DMat, dv: &DMat, ridge: f32, perm: Option<&wgpu::Buffer>) {
    let n = s.r;
    let cx = dev();
    let md = elt(cx).expect("nvg elt pipelines");
    let d = dummy(cx);
    let dp = Disp {
        pipe: &md.p["equil"],
        uni: uni(&[n as u32, 0, 0, 0, 0, 0, perm.is_some() as u32], &[ridge]),
        bufs: vec![(1, &s.buf), (3, &dv.buf), (5, &f.buf), (7, perm.unwrap_or(d))],
        grid: elt_fold(n * n),
    };
    train_ops::run(cx, &[dp]);
}

/// `Uc[j][k] = X[n-1-j][n-1-k] / dp[k]` (k ≥ j).
pub fn make_uc(x: &DMat, dp_: &DMat, uc: &DMat) {
    let n = x.r;
    let cx = dev();
    let md = elt(cx).expect("nvg elt pipelines");
    let dp = Disp {
        pipe: &md.p["make_uc"],
        uni: uni(&[n as u32], &[]),
        bufs: vec![(1, &x.buf), (3, &dp_.buf), (5, &uc.buf)],
        grid: elt_fold(n * n),
    };
    train_ops::run(cx, &[dp]);
}

/// Row gather `C[j] = A[perm[j]]` or scatter `C[perm[j]] = A[j]`.
pub fn permute_rows(a: &DMat, c: &DMat, perm: &wgpu::Buffer, scatter: bool) {
    assert_eq!((a.r, a.c), (c.r, c.c));
    let cx = dev();
    let md = elt(cx).expect("nvg elt pipelines");
    let dp = Disp {
        pipe: &md.p["permute_rows"],
        uni: uni(&[a.r as u32, a.c as u32, 0, 0, 0, 0, scatter as u32], &[]),
        bufs: vec![(1, &a.buf), (5, &c.buf), (7, perm)],
        grid: elt_fold(a.r * a.c),
    };
    train_ops::run(cx, &[dp]);
}

/// `Aᵀ` into a new matrix.
pub fn transpose(a: &DMat) -> DMat {
    let c = DMat::zeros(a.c, a.r);
    let cx = dev();
    let md = elt(cx).expect("nvg elt pipelines");
    let dp = Disp {
        pipe: &md.p["transpose"],
        uni: uni(&[a.r as u32, a.c as u32], &[]),
        bufs: vec![(1, &a.buf), (5, &c.buf)],
        grid: (a.c.div_ceil(32) as u32, a.r.div_ceil(32) as u32, 1),
    };
    train_ops::run(cx, &[dp]);
    c
}

/// Solve `L Lᵀ x = b` in place (`y` holds b; `l` the dense lower factor).
pub fn chol_solve_dev(l: &DMat, y: &DMat) {
    let n = l.r;
    let cx = dev();
    let md = elt(cx).expect("nvg elt pipelines");
    let nb = n.div_ceil(64);
    let mut ds = Vec::with_capacity(4 * nb);
    for k in 0..nb {
        let k0 = (k * 64) as u32;
        ds.push(Disp { pipe: &md.p["trsv_diag"], uni: uni(&[n as u32, k0, 0, 0, 0, 0, 0], &[]), bufs: vec![(1, &l.buf), (5, &y.buf)], grid: (1, 1, 1) });
        let rest = n.saturating_sub(k * 64 + 64);
        if rest > 0 {
            ds.push(Disp { pipe: &md.p["trsv_upd"], uni: uni(&[n as u32, k0, 0, 0, 0, 0, 0], &[]), bufs: vec![(1, &l.buf), (5, &y.buf)], grid: (rest.div_ceil(256) as u32, 1, 1) });
        }
    }
    for k in (0..nb).rev() {
        let k0 = (k * 64) as u32;
        ds.push(Disp { pipe: &md.p["trsv_diag"], uni: uni(&[n as u32, k0, 0, 0, 0, 0, 1], &[]), bufs: vec![(1, &l.buf), (5, &y.buf)], grid: (1, 1, 1) });
        if k > 0 {
            ds.push(Disp { pipe: &md.p["trsv_upd"], uni: uni(&[n as u32, k0, 0, 0, 0, 0, 1], &[]), bufs: vec![(1, &l.buf), (5, &y.buf)], grid: ((k * 64).div_ceil(256) as u32, 1, 1) });
        }
    }
    train_ops::run(cx, &ds);
}

/// Lower Cholesky of the square `a` in place (`Some(false)`: a pivot failed).
pub fn chol_in_place(a: &DMat) -> Option<bool> {
    super::gptq_dev::chol_lower(&a.buf, a.r)
}

/// `x = L⁻¹` (`l` as [`chol_in_place`] leaves it).
pub fn tri_inv(l: &DMat, x: &DMat) -> Option<()> {
    super::gptq_dev::tri_inv_lower(&l.buf, &x.buf, l.r)
}

/// `(A + rel·tr(A)/n·I) x = b` for the SPD `a` (left untouched): Jacobi
/// equilibration, f32 blocked Cholesky on the device, the ridge ×100 on a
/// failed pivot (the host `solve_spd` rule), triangular solves on the
/// device. Zeros when the ridge runs past 10·tr.
pub fn spd_solve(a: &DMat, b: &[f64], rel: f64) -> Vec<f64> {
    let n = a.r;
    let dg = diag(a, n, 0, n);
    let tr = dg.iter().sum::<f64>() / n.max(1) as f64;
    let mut ridge = rel * tr.max(1e-300);
    let f = DMat::zeros(n, n);
    loop {
        let d: Vec<f64> = dg.iter().map(|&v| (v + ridge).max(1e-30).sqrt()).collect();
        let dv = DMat::from_f64(1, n, &d);
        equil(a, &f, &dv, ridge as f32, None);
        if chol_in_place(&f) == Some(true) {
            let y = DMat::from_f64(1, n, &b.iter().zip(&d).map(|(b, d)| b / d).collect::<Vec<_>>());
            chol_solve_dev(&f, &y);
            return y.to_f64().iter().zip(&d).map(|(y, d)| y / d).collect();
        }
        ridge = (ridge * 100.0).max(1e-300);
        if ridge > tr.abs() * 10.0 + 1e-200 {
            return vec![0.0; n];
        }
    }
}

// ───────────────────────────── discrete knots ─────────────────────────────

const KNOT_SRC: &str = r#"
struct P { u0: u32, u1: u32, u2: u32, u3: u32, u4: u32, u5: u32, u6: u32, u7: u32, u8: u32, u9: u32, u10: u32, u11: u32, f0: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read> R: array<f32>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read_write> C: array<f32>;
@group(0) @binding(6) var<storage, read_write> O: array<f32>;
@group(0) @binding(8) var<storage, read_write> EG: array<f32>;
@group(0) @binding(9) var<storage, read_write> CNT: array<atomic<u32>>;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
var<workgroup> bc: array<f32, 4>;

const NQ: u32 = __NQ__u;

// Output-side knots, one thread per output row i (m = u0, r = u1):
// C = S_U^T (r x m, +-1), O = qs^T (r x m), A = whb^T (r x m), B = Q (r x r),
// R = h (m), S = l (r). Greedy passes (<= 4) of exact single-bit flips.
@compute @workgroup_size(256)
fn flipu(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 65535u * 256u;
    let m = p.u0;
    let r = p.u1;
    if (i >= m) { return; }
    let h = R[i];
    var nf = 0u;
    for (var sweep = 0u; sweep < 4u; sweep = sweep + 1u) {
        var changed = 0u;
        for (var kk = 0u; kk < r; kk = kk + 1u) {
            let s = C[kk * m + i];
            let qd = B[kk * r + kk];
            let bm = A[kk * m + i] * S[kk];
            let d = -4.0 * h * h * s * (O[kk * m + i] - qd * s) + 4.0 * h * s * bm;
            if (d < -p.f0 * (h * h * qd + 1.0e-30)) {
                for (var t = 0u; t < r; t = t + 1u) { O[t * m + i] = O[t * m + i] - 2.0 * s * B[kk * r + t]; }
                C[kk * m + i] = -s;
                changed = changed + 1u;
            }
        }
        nf = nf + changed;
        if (changed == 0u) { break; }
    }
    if (nf > 0u) { atomicAdd(&CNT[0], nf); }
}

fn wsum(t: u32, v: f32) -> f32 {
    rv[t] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { rv[t] = rv[t] + rv[t + s]; }
        workgroupBarrier();
    }
    let v0 = workgroupUniformLoad(&rv[0]);
    workgroupBarrier();
    return v0;
}

// Input-side knots of one 128-column block [u2, u3) in act order, in the
// latent space (n = u0 columns, r = u1, max flips per round u4):
// C = BV (n x r: M^T w_j, trailing feedback applied), O = S (n x r, +-1),
// A = KS0 (n x r: K s_j of the incoming signs), B = K (r x r), R = Uc (n x n),
// EG = [E (128 x r) | G (n)]. Column j: bv_j minus the in-block feedback,
// best-improvement flips (<= 3 rounds), e_j = (bv_j - g_j K s_j) / Uc[j][j].
@compute @workgroup_size(256)
fn gv_block(@builtin(local_invocation_index) t: u32) {
    let n = p.u0;
    let r = p.u1;
    let b0 = p.u2;
    let b1 = p.u3;
    let maxf = p.u4;
    let tol = p.f0;
    let goff = 128u * r;
    var kd: array<f32, NQ>;
    var s: array<f32, NQ>;
    var ks: array<f32, NQ>;
    var bv: array<f32, NQ>;
    for (var q = 0u; q < NQ; q = q + 1u) {
        let x = t + 256u * q;
        kd[q] = 0.0;
        if (x < r) { kd[q] = B[x * r + x]; }
    }
    var flips = 0u;
    for (var j = b0; j < b1; j = j + 1u) {
        for (var q = 0u; q < NQ; q = q + 1u) {
            let x = t + 256u * q;
            s[q] = 0.0; ks[q] = 0.0; bv[q] = 0.0;
            if (x < r) {
                s[q] = O[j * r + x];
                ks[q] = A[j * r + x];
                var b = C[j * r + x];
                for (var jj = b0; jj < j; jj = jj + 1u) { b = b - R[jj * n + j] * EG[(jj - b0) * r + x]; }
                bv[q] = b;
            }
        }
        var a1 = 0.0;
        var a2 = 0.0;
        for (var q = 0u; q < NQ; q = q + 1u) { a1 = a1 + bv[q] * s[q]; a2 = a2 + s[q] * ks[q]; }
        let num = wsum(t, a1);
        let den = wsum(t, a2);
        var gj = num / max(den, 1.0e-30);
        if (gj < 0.0) {
            for (var q = 0u; q < NQ; q = q + 1u) { s[q] = -s[q]; ks[q] = -ks[q]; }
            gj = -gj;
        }
        for (var rd = 0u; rd < 3u; rd = rd + 1u) {
            var nf = 0u;
            loop {
                if (nf >= maxf) { break; }
                var bestv = 3.4e38;
                var besti = 0xffffffffu;
                for (var q = 0u; q < NQ; q = q + 1u) {
                    let x = t + 256u * q;
                    if (x < r) {
                        let d = -4.0 * gj * gj * s[q] * (ks[q] - kd[q] * s[q]) + 4.0 * gj * s[q] * bv[q];
                        if (d < bestv) { bestv = d; besti = x; }
                    }
                }
                rv[t] = bestv;
                ri[t] = besti;
                workgroupBarrier();
                for (var st = 128u; st > 0u; st = st >> 1u) {
                    if (t < st) {
                        let ov = rv[t + st];
                        let oi = ri[t + st];
                        if (ov < rv[t] || (ov == rv[t] && oi < ri[t])) { rv[t] = ov; ri[t] = oi; }
                    }
                    workgroupBarrier();
                }
                let bvm = workgroupUniformLoad(&rv[0]);
                let bi = workgroupUniformLoad(&ri[0]);
                let kdb = B[bi * r + bi];
                if (bvm >= -tol * (gj * gj * kdb + 1.0e-30)) { break; }
                for (var q = 0u; q < NQ; q = q + 1u) {
                    if (t + 256u * q == bi) { bc[0] = s[q]; s[q] = -s[q]; }
                }
                let sb = workgroupUniformLoad(&bc[0]);
                for (var q = 0u; q < NQ; q = q + 1u) {
                    let x = t + 256u * q;
                    if (x < r) { ks[q] = ks[q] - 2.0 * sb * B[bi * r + x]; }
                }
                nf = nf + 1u;
            }
            flips = flips + nf;
            var c1 = 0.0;
            var c2 = 0.0;
            for (var q = 0u; q < NQ; q = q + 1u) { c1 = c1 + bv[q] * s[q]; c2 = c2 + s[q] * ks[q]; }
            let num2 = wsum(t, c1);
            let den2 = wsum(t, c2);
            var g2 = num2 / max(den2, 1.0e-30);
            if (g2 < 0.0) {
                for (var q = 0u; q < NQ; q = q + 1u) { s[q] = -s[q]; ks[q] = -ks[q]; }
                g2 = -g2;
            }
            let done = nf == 0u || abs(g2 - gj) <= 1.0e-6 * abs(gj);
            gj = g2;
            if (done) { break; }
        }
        let ujj = R[j * n + j];
        for (var q = 0u; q < NQ; q = q + 1u) {
            let x = t + 256u * q;
            if (x < r) {
                EG[(j - b0) * r + x] = (bv[q] - gj * ks[q]) / ujj;
                O[j * r + x] = s[q];
            }
        }
        if (t == 0u) { EG[goff + j] = gj; }
    }
    if (t == 0u) { atomicAdd(&CNT[0], flips); }
}
"#;

fn knot(c: &Ctx, nq: usize) -> Option<&'static Mod> {
    static SRCS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    let srcs = SRCS.get_or_init(|| {
        [1usize, 2, 4, 8, 16]
            .iter()
            .map(|q| &*Box::leak(KNOT_SRC.replace("__NQ__", &q.to_string()).into_boxed_str()))
            .collect()
    });
    let (k, name) = match nq {
        0 | 1 => (0, "nvg_knot1"),
        2 => (1, "nvg_knot2"),
        3 | 4 => (2, "nvg_knot4"),
        5..=8 => (3, "nvg_knot8"),
        _ => (4, "nvg_knot16"),
    };
    train_ops::module(c, name, srcs[k], &["flipu", "gv_block"])
}

// The input-side search with subgroup reductions: 256 threads, each holding
// its NQ = ceil(r/256) lanes in registers (the source is generated per NQ
// with every lane loop unrolled — a dynamically indexed array would live in
// local memory), the argmin and the dot products reduced per subgroup and
// across subgroups through one exchange (two barriers per flip instead of
// the tree's ten). In its own module: subgroup builtins must never reach a
// device without the feature (naga 30 takes them without an `enable`).
fn gv_sg_src(nq: usize) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    s.push_str(
        r#"
struct P { u0: u32, u1: u32, u2: u32, u3: u32, u4: u32, u5: u32, u6: u32, u7: u32, u8: u32, u9: u32, u10: u32, u11: u32, f0: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read> R: array<f32>;
@group(0) @binding(5) var<storage, read_write> C: array<f32>;
@group(0) @binding(6) var<storage, read_write> O: array<f32>;
@group(0) @binding(8) var<storage, read_write> EG: array<f32>;
@group(0) @binding(9) var<storage, read_write> CNT: array<atomic<u32>>;
var<workgroup> pv: array<vec4<f32>, 64>;
var<workgroup> fin: vec4<f32>;

fn sum2(t: u32, lane: u32, sg: u32, nsg: u32, x: f32, y: f32) -> vec2<f32> {
    let a = subgroupAdd(x);
    let b = subgroupAdd(y);
    if (lane == 0u) { pv[sg] = vec4<f32>(a, b, 0.0, 0.0); }
    workgroupBarrier();
    if (sg == 0u) {
        var v = vec2<f32>(0.0);
        if (lane < nsg) { v = pv[lane].xy; }
        let tx = subgroupAdd(v.x);
        let ty = subgroupAdd(v.y);
        if (lane == 0u) { fin = vec4<f32>(tx, ty, 0.0, 0.0); }
    }
    return workgroupUniformLoad(&fin).xy;
}

@compute @workgroup_size(256)
fn gv_block_sg(@builtin(local_invocation_index) t: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) ssz: u32) {
    let n = p.u0;
    let r = p.u1;
    let b0 = p.u2;
    let b1 = p.u3;
    let maxf = p.u4;
    let tol = p.f0;
    let goff = 128u * r;
    let sg = t / ssz;
    let nsg = 256u / ssz;
"#,
    );
    let q = |s: &mut String, f: &dyn Fn(usize) -> String| {
        for k in 0..nq {
            s.push_str(&f(k));
        }
    };
    q(&mut s, &|k| format!("    let x{k} = t + {}u;\n    var kd{k} = 0.0;\n    var s{k} = 0.0;\n    var ks{k} = 0.0;\n    var bv{k} = 0.0;\n    if (x{k} < r) {{ kd{k} = B[x{k} * r + x{k}]; }}\n", 256 * k));
    s.push_str("    var flips = 0u;\n    for (var j = b0; j < b1; j = j + 1u) {\n");
    q(&mut s, &|k| format!(
        "        s{k} = 0.0; ks{k} = 0.0; bv{k} = 0.0;\n        if (x{k} < r) {{ s{k} = O[j * r + x{k}]; ks{k} = A[j * r + x{k}]; bv{k} = C[j * r + x{k}]; }}\n"
    ));
    s.push_str("        for (var jj = b0; jj < j; jj = jj + 1u) {\n            let u = R[jj * n + j];\n            let eb = (jj - b0) * r;\n");
    q(&mut s, &|k| format!("            if (x{k} < r) {{ bv{k} = bv{k} - u * EG[eb + x{k}]; }}\n"));
    s.push_str("        }\n");
    let mut dot1 = String::from("0.0");
    let mut dot2 = String::from("0.0");
    for k in 0..nq {
        let _ = write!(dot1, " + bv{k} * s{k}");
        let _ = write!(dot2, " + s{k} * ks{k}");
    }
    let _ = writeln!(s, "        let dd = sum2(t, lane, sg, nsg, {dot1}, {dot2});");
    s.push_str("        var gj = dd.x / max(dd.y, 1.0e-30);\n        if (gj < 0.0) {\n");
    q(&mut s, &|k| format!("            s{k} = -s{k}; ks{k} = -ks{k};\n"));
    s.push_str("            gj = -gj;\n        }\n");
    s.push_str("        for (var rd = 0u; rd < 3u; rd = rd + 1u) {\n            var nf = 0u;\n            loop {\n                if (nf >= maxf) { break; }\n");
    s.push_str("                var bvl = 3.4e38;\n                var bil = 0xffffffffu;\n                var bsl = 0.0;\n                var bkl = 0.0;\n");
    q(&mut s, &|k| format!(
        "                if (x{k} < r) {{ let d = -4.0 * gj * gj * s{k} * (ks{k} - kd{k} * s{k}) + 4.0 * gj * s{k} * bv{k}; if (d < bvl) {{ bvl = d; bil = x{k}; bsl = s{k}; bkl = kd{k}; }} }}\n"
    ));
    s.push_str(
        r#"                let mv = subgroupMin(bvl);
                let mi = subgroupMin(select(0xffffffffu, bil, bvl == mv));
                let win = bil == mi;
                let ms = subgroupMax(select(-2.0, bsl, win));
                let mk = subgroupMax(select(-3.4e38, bkl, win));
                if (lane == 0u) { pv[sg] = vec4<f32>(mv, bitcast<f32>(mi), ms, mk); }
                workgroupBarrier();
                if (sg == 0u) {
                    var o = vec4<f32>(3.4e38, bitcast<f32>(0xffffffffu), 0.0, 0.0);
                    if (lane < nsg) { o = pv[lane]; }
                    let gv = subgroupMin(o.x);
                    let oi = bitcast<u32>(o.y);
                    let gi = subgroupMin(select(0xffffffffu, oi, o.x == gv));
                    let gw = oi == gi;
                    let gs = subgroupMax(select(-2.0, o.z, gw));
                    let gk = subgroupMax(select(-3.4e38, o.w, gw));
                    if (lane == 0u) { fin = vec4<f32>(gv, bitcast<f32>(gi), gs, gk); }
                }
                let f = workgroupUniformLoad(&fin);
                let bi = bitcast<u32>(f.y);
                if (bi >= r) { break; }
                if (f.x >= -tol * (gj * gj * f.w + 1.0e-30)) { break; }
                let sb = f.z;
                let kr = bi * r;
"#,
    );
    q(&mut s, &|k| format!("                if (x{k} == bi) {{ s{k} = -s{k}; }}\n"));
    q(&mut s, &|k| format!("                if (x{k} < r) {{ ks{k} = ks{k} - 2.0 * sb * B[kr + x{k}]; }}\n"));
    s.push_str("                nf = nf + 1u;\n            }\n            flips = flips + nf;\n");
    let _ = writeln!(s, "            let d2 = sum2(t, lane, sg, nsg, {dot1}, {dot2});");
    s.push_str("            var g2 = d2.x / max(d2.y, 1.0e-30);\n            if (g2 < 0.0) {\n");
    q(&mut s, &|k| format!("                s{k} = -s{k}; ks{k} = -ks{k};\n"));
    s.push_str("                g2 = -g2;\n            }\n            let done = nf == 0u || abs(g2 - gj) <= 1.0e-6 * abs(gj);\n            gj = g2;\n            if (done) { break; }\n        }\n");
    s.push_str("        let ujj = R[j * n + j];\n");
    q(&mut s, &|k| format!(
        "        if (x{k} < r) {{ EG[(j - b0) * r + x{k}] = (bv{k} - gj * ks{k}) / ujj; O[j * r + x{k}] = s{k}; }}\n"
    ));
    s.push_str("        if (t == 0u) { EG[goff + j] = gj; }\n    }\n    if (t == 0u) { atomicAdd(&CNT[0], flips); }\n}\n");
    s
}

/// The subgroup search kernel is usable: the device has subgroups of at
/// least 32 lanes (`CMF_NVG_SG=0` keeps the workgroup-tree kernel).
fn sg_ok(c: &Ctx) -> bool {
    std::env::var("CMF_NVG_SG").as_deref() != Ok("0")
        && c.device.features().contains(wgpu::Features::SUBGROUP)
        && c.adapter_info.subgroup_min_size >= 32
        && c.adapter_info.subgroup_max_size <= 256
}

fn gv_sg(c: &Ctx, r: usize) -> Option<&'static Mod> {
    static SRCS: std::sync::Mutex<Vec<(usize, &'static str, &'static str)>> = std::sync::Mutex::new(Vec::new());
    let nq = r.div_ceil(256).max(1);
    if nq > 16 {
        return None;
    }
    let (name, src) = {
        let mut g = SRCS.lock().unwrap();
        match g.iter().find(|e| e.0 == nq) {
            Some(e) => (e.1, e.2),
            None => {
                let name: &'static str = Box::leak(format!("nvg_gv_sg{nq}").into_boxed_str());
                let src: &'static str = Box::leak(gv_sg_src(nq).into_boxed_str());
                g.push((nq, name, src));
                (name, src)
            }
        }
    };
    train_ops::module(c, name, src, &["gv_block_sg"])
}

// Output-side flips, one thread per row, the kk axis in chunks of 8 whose
// sign / qs / whb entries are fetched together (independent loads instead of
// a dependent load chain per bit); a flip updates the row's qs in memory and
// the chunk's later entries in registers. Same greedy order as the host.
fn flipu_src() -> String {
    use std::fmt::Write;
    const CH: usize = 8;
    let mut s = String::from(
        r#"
struct P { u0: u32, u1: u32, u2: u32, u3: u32, u4: u32, u5: u32, u6: u32, u7: u32, u8: u32, u9: u32, u10: u32, u11: u32, f0: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read> R: array<f32>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read_write> C: array<f32>;
@group(0) @binding(6) var<storage, read_write> O: array<f32>;
@group(0) @binding(9) var<storage, read_write> CNT: array<atomic<u32>>;

@compute @workgroup_size(64)
fn flipu8(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 65535u * 64u;
    let m = p.u0;
    let r = p.u1;
    if (i >= m) { return; }
    let h = R[i];
    let thr = p.f0;
    var nf = 0u;
    for (var sweep = 0u; sweep < 4u; sweep = sweep + 1u) {
        var changed = 0u;
        for (var k0 = 0u; k0 < r; k0 = k0 + 8u) {
"#,
    );
    for c in 0..CH {
        let _ = writeln!(
            s,
            "            var s{c} = 0.0; var q{c} = 0.0; var w{c} = 0.0; var d{c} = 0.0;\n            if (k0 + {c}u < r) {{ let kk = k0 + {c}u; s{c} = C[kk * m + i]; q{c} = O[kk * m + i]; w{c} = A[kk * m + i] * S[kk]; d{c} = B[kk * r + kk]; }}"
        );
    }
    for c in 0..CH {
        let _ = writeln!(
            s,
            "            if (k0 + {c}u < r) {{\n                let kk = k0 + {c}u;\n                let d = -4.0 * h * h * s{c} * (q{c} - d{c} * s{c}) + 4.0 * h * s{c} * w{c};\n                if (d < -thr * (h * h * d{c} + 1.0e-30)) {{\n                    let tw = 2.0 * s{c};\n                    let kr = kk * r;\n                    for (var t = 0u; t < r; t = t + 1u) {{ O[t * m + i] = O[t * m + i] - tw * B[kr + t]; }}"
        );
        for c2 in c + 1..CH {
            let _ = writeln!(s, "                    if (k0 + {c2}u < r) {{ q{c2} = q{c2} - tw * B[kr + k0 + {c2}u]; }}");
        }
        let _ = writeln!(s, "                    C[kk * m + i] = -s{c};\n                    changed = changed + 1u;\n                }}\n            }}");
    }
    s.push_str(
        r#"        }
        nf = nf + changed;
        if (changed == 0u) { break; }
    }
    if (nf > 0u) { atomicAdd(&CNT[0], nf); }
}
"#,
    );
    s
}

fn flipu_mod(c: &Ctx) -> Option<&'static Mod> {
    static SRC: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    let src = SRC.get_or_init(|| Box::leak(flipu_src().into_boxed_str()));
    train_ops::module(c, "nvg_flipu8", src, &["flipu8"])
}

// Blocked output-side flips: the kk axis in blocks of 32. One thread per row
// decides the block's 32 bits in order (the block's qs entries in registers,
// updated through the block's 32×32 corner of Q held in workgroup memory)
// and records the flips F (m×32, entries 2·s_old or 0); the rest of every
// row's qs then moves with one GEMM `qs −= F·Q[block, :]`. The decisions see
// exactly what the host's sequential pass sees.
fn flipblk_src() -> String {
    use std::fmt::Write;
    let mut s = String::from(
        r#"
struct P { u0: u32, u1: u32, u2: u32, u3: u32, u4: u32, u5: u32, u6: u32, u7: u32, u8: u32, u9: u32, u10: u32, u11: u32, f0: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read> R: array<f32>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read_write> C: array<f32>;
@group(0) @binding(6) var<storage, read_write> O: array<f32>;
@group(0) @binding(8) var<storage, read_write> F: array<f32>;
@group(0) @binding(9) var<storage, read_write> CNT: array<atomic<u32>>;
var<workgroup> qb: array<f32, 1024>;
var<workgroup> lb: array<f32, 32>;

// m = u0, r = u1, b0 = u2; C = S_U (m x r), O = qs (m x r, read here),
// A = whb (m x r), B = Q (r x r), R = h, S = l, F = flips (m x 32).
@compute @workgroup_size(64)
fn flipblk(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let m = p.u0;
    let r = p.u1;
    let b0 = p.u2;
    let bs = min(32u, r - b0);
    for (var e = li; e < 1024u; e = e + 64u) {
        let a = e / 32u;
        let b = e % 32u;
        var v = 0.0;
        if (a < bs && b < bs) { v = B[(b0 + a) * r + b0 + b]; }
        qb[e] = v;
    }
    if (li < 32u) { var v = 0.0; if (li < bs) { v = S[b0 + li]; } lb[li] = v; }
    workgroupBarrier();
    let i = (wg.x + wg.y * 65535u) * 64u + li;
    if (i >= m) { return; }
    let h = R[i];
    let thr = p.f0;
    let base = i * r + b0;
    var nf = 0u;
"#,
    );
    for c in 0..32 {
        let _ = writeln!(
            s,
            "    var s{c} = 0.0; var q{c} = 0.0; var w{c} = 0.0; var f{c} = 0.0;\n    if ({c}u < bs) {{ s{c} = C[base + {c}u]; q{c} = O[base + {c}u]; w{c} = A[base + {c}u] * lb[{c}u]; }}"
        );
    }
    for c in 0..32 {
        let _ = writeln!(
            s,
            "    if ({c}u < bs) {{\n        let qd = qb[{}u];\n        let d = -4.0 * h * h * s{c} * (q{c} - qd * s{c}) + 4.0 * h * s{c} * w{c};\n        if (d < -thr * (h * h * qd + 1.0e-30)) {{\n            f{c} = 2.0 * s{c};\n            s{c} = -s{c};\n            nf = nf + 1u;",
            c * 33
        );
        for c2 in c + 1..32 {
            let _ = writeln!(s, "            q{c2} = q{c2} - f{c} * qb[{}u];", c * 32 + c2);
        }
        s.push_str("        }\n    }\n");
    }
    for c in 0..32 {
        let _ = writeln!(s, "    if ({c}u < bs) {{ C[base + {c}u] = s{c}; }}\n    F[i * 32u + {c}u] = f{c};");
    }
    s.push_str("    if (nf > 0u) { atomicAdd(&CNT[0], nf); }\n}\n");
    s
}

fn flipblk_mod(c: &Ctx) -> Option<&'static Mod> {
    static SRC: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    let src = SRC.get_or_init(|| Box::leak(flipblk_src().into_boxed_str()));
    train_ops::module(c, "nvg_flipblk", src, &["flipblk"])
}

/// Output-side flips by blocks (see `flipblk`): `su`, `qs`, `whb` row-major
/// m×r (`su` and `qs` updated), `q` = diag(ℓ)·K·diag(ℓ), `h`, `l`. At most
/// four sweeps, stopping after one without a flip. Returns the flip count.
#[allow(clippy::too_many_arguments)]
pub fn flipu_blocked(su: &DMat, qs: &DMat, whb: &DMat, q: &DMat, h: &DMat, l: &DMat, m: usize, r: usize, tol: f32) -> Option<u64> {
    let cx = dev();
    let md = flipblk_mod(cx)?;
    let f = DMat::zeros(m, 32);
    let mut total = 0u64;
    let g = m.div_ceil(64);
    let grid = if g <= 65535 { (g as u32, 1, 1) } else { (65535, g.div_ceil(65535) as u32, 1) };
    for _ in 0..4 {
        let cnt = DMat::zeros(1, 4);
        let mut b0 = 0;
        while b0 < r {
            let bs = 32.min(r - b0);
            let dp = Disp {
                pipe: &md.p["flipblk"],
                uni: uni(&[m as u32, r as u32, b0 as u32], &[tol]),
                bufs: vec![(1, &whb.buf), (2, &q.buf), (3, &h.buf), (4, &l.buf), (5, &su.buf), (6, &qs.buf), (8, &f.buf), (9, &cnt.buf)],
                grid,
            };
            train_ops::run(cx, &[dp]);
            // qs −= F · Q[b0..b0+bs, :]
            gemm_v(m, r, bs, -1.0, V { m: &f, off: 0, ld: 32, t: false }, V { m: q, off: b0 * r, ld: r, t: false }, 1.0, qs, 0, r, None);
            b0 += bs;
        }
        let n = cnt.to_f32()[0].to_bits() as u64;
        total += n;
        if n == 0 {
            break;
        }
    }
    Some(total)
}

/// The largest rank [`gv_block`] takes (16 lanes per thread × 256).
pub const GV_MAX_RANK: usize = 4096;

/// Output-side flips (see the kernel). Returns the flip count.
#[allow(clippy::too_many_arguments)]
pub fn flipu(sut: &DMat, qst: &DMat, whbt: &DMat, q: &DMat, h: &DMat, l: &DMat, m: usize, r: usize, tol: f32) -> u64 {
    let cx = dev();
    let cnt = DMat::zeros(1, 4);
    let dp = match flipu_mod(cx).filter(|_| std::env::var("CMF_NVG_FLIPU8").as_deref() != Ok("0")) {
        Some(md) => Disp {
            pipe: &md.p["flipu8"],
            uni: uni(&[m as u32, r as u32], &[tol]),
            bufs: vec![(1, &whbt.buf), (2, &q.buf), (3, &h.buf), (4, &l.buf), (5, &sut.buf), (6, &qst.buf), (9, &cnt.buf)],
            grid: { let g = m.div_ceil(64); if g <= 65535 { (g as u32, 1, 1) } else { (65535, g.div_ceil(65535) as u32, 1) } },
        },
        None => {
            let md = knot(cx, 1).expect("nvg knot pipelines");
            Disp {
                pipe: &md.p["flipu"],
                uni: uni(&[m as u32, r as u32], &[tol]),
                bufs: vec![(1, &whbt.buf), (2, &q.buf), (3, &h.buf), (4, &l.buf), (5, &sut.buf), (6, &qst.buf), (9, &cnt.buf)],
                grid: elt_fold(m),
            }
        }
    };
    train_ops::run(cx, &[dp]);
    cnt.to_f32()[0].to_bits() as u64
}

/// One 128-column block of the input-side search (see the kernel), then
/// the trailing feedback `BV[b1:] −= Uc[b0:b1, b1:]ᵀ · E` as a GEMM.
/// `cnt` accumulates the flips.
#[allow(clippy::too_many_arguments)]
pub fn gv_block(bv: &DMat, s: &DMat, ks0: &DMat, k: &DMat, uc: &DMat, eg: &DMat, cnt: &DMat, b0: usize, b1: usize, max_flips: usize, tol: f32) {
    let (n, r) = (bv.r, bv.c);
    let cx = dev();
    let md = if sg_ok(cx) { gv_sg(cx, r) } else { None }.or_else(|| knot(cx, r.div_ceil(256))).expect("nvg knot pipelines");
    let dp = Disp {
        pipe: md.p.get("gv_block_sg").unwrap_or_else(|| &md.p["gv_block"]),
        uni: uni(&[n as u32, r as u32, b0 as u32, b1 as u32, max_flips as u32], &[tol]),
        bufs: vec![(1, &ks0.buf), (2, &k.buf), (3, &uc.buf), (5, &bv.buf), (6, &s.buf), (8, &eg.buf), (9, &cnt.buf)],
        grid: (1, 1, 1),
    };
    train_ops::run(cx, &[dp]);
    if b1 < n {
        let a = V { m: uc, off: b0 * n + b1, ld: n, t: true };
        let e = V { m: eg, off: 0, ld: r, t: false };
        gemm_v(n - b1, r, b1 - b0, -1.0, a, e, 1.0, bv, b1 * r, r, None);
    }
}

// ───────────────────────────── symmetric eigensolver ─────────────────────────────
//
// Top-r eigenpairs of a dense symmetric k×k matrix: Householder
// tridiagonalisation on the device (one reflector per step: vector, trailing
// matvec, w, rank-2 update — four dispatches recorded into one pass with
// dynamic uniform offsets), the r largest eigenvalues of the tridiagonal by
// bisection and their vectors by inverse iteration on the host in f64, and
// the back-transformation `Q·Z` on the device in blocks of 32 reflectors
// (compact WY, three GEMMs a block).

const EIG_SRC: &str = r#"
struct P { k: u32, i: u32, b: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> VS: array<f32>;
@group(0) @binding(3) var<storage, read_write> PV: array<f32>;
@group(0) @binding(4) var<storage, read_write> WV: array<f32>;
@group(0) @binding(5) var<storage, read_write> DE: array<f32>;
@group(0) @binding(6) var<storage, read_write> VB: array<f32>;
var<workgroup> red: array<f32, 256>;

fn wsum(t: u32, v: f32) -> f32 {
    red[t] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        workgroupBarrier();
    }
    let r = workgroupUniformLoad(&red[0]);
    workgroupBarrier();
    return r;
}

// Reflector of step i (LAPACK dlarfg on A[i+1.., i]): v (v0 = 1) into VS and
// below the subdiagonal of column i; d_i, beta (= e_i), tau into DE.
@compute @workgroup_size(256)
fn hh_vec(@builtin(local_invocation_index) t: u32) {
    let k = p.k;
    let i = p.i;
    let L = k - i - 1u;
    let x0 = A[(i + 1u) * k + i];
    var ss = 0.0;
    for (var j = 1u + t; j < L; j = j + 256u) { let v = A[(i + 1u + j) * k + i]; ss = ss + v * v; }
    let xn2 = wsum(t, ss);
    var tau = 0.0;
    var beta = x0;
    var scale = 0.0;
    if (xn2 > 0.0) {
        let nrm = sqrt(x0 * x0 + xn2);
        beta = select(nrm, -nrm, x0 >= 0.0);
        tau = (beta - x0) / beta;
        scale = 1.0 / (x0 - beta);
    }
    for (var j = t; j < L; j = j + 256u) {
        var v = 1.0;
        if (j > 0u) {
            v = A[(i + 1u + j) * k + i] * scale;
            A[(i + 1u + j) * k + i] = v;
        }
        VS[j] = v;
    }
    if (t == 0u) {
        DE[i] = A[i * k + i];
        DE[k + i] = beta;
        DE[2u * k + i] = tau;
    }
}

// p = tau * A22 v, one workgroup per row of the trailing block.
@compute @workgroup_size(256)
fn hh_matvec(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let k = p.k;
    let i = p.i;
    let L = k - i - 1u;
    let j = wg.x + wg.y * 65535u;
    if (j >= L) { return; }
    let base = (i + 1u + j) * k + i + 1u;
    var acc = 0.0;
    for (var c = t; c < L; c = c + 256u) { acc = acc + A[base + c] * VS[c]; }
    let s = wsum(t, acc);
    if (t == 0u) { PV[j] = DE[2u * k + i] * s; }
}

// w = p - (tau/2)(p.v) v
@compute @workgroup_size(256)
fn hh_w(@builtin(local_invocation_index) t: u32) {
    let k = p.k;
    let i = p.i;
    let L = k - i - 1u;
    var acc = 0.0;
    for (var j = t; j < L; j = j + 256u) { acc = acc + PV[j] * VS[j]; }
    let kk = 0.5 * DE[2u * k + i] * wsum(t, acc);
    for (var j = t; j < L; j = j + 256u) { WV[j] = PV[j] - kk * VS[j]; }
}

// A22 -= v w^T + w v^T
@compute @workgroup_size(16, 16)
fn hh_rank2(@builtin(global_invocation_id) g: vec3<u32>) {
    let k = p.k;
    let i = p.i;
    let L = k - i - 1u;
    let c = g.x;
    let j = g.y;
    if (j >= L || c >= L) { return; }
    let ix = (i + 1u + j) * k + i + 1u + c;
    A[ix] = A[ix] - (VS[j] * WV[c] + WV[j] * VS[c]);
}

@compute @workgroup_size(1)
fn hh_last() {
    let k = p.k;
    DE[k - 1u] = A[(k - 1u) * k + k - 1u];
}

// Block of b reflectors starting at i0 = p.i as a dense k x b matrix:
// column c is reflector i = i0 + c (1 at row i+1, A[row][i] below, 0 above).
@compute @workgroup_size(256)
fn hh_vblock(@builtin(global_invocation_id) g: vec3<u32>) {
    let k = p.k;
    let b = p.b;
    let e = g.x + g.y * 65535u * 256u;
    if (e >= k * b) { return; }
    let row = e / b;
    let c = e % b;
    let i = p.i + c;
    var v = 0.0;
    if (i + 1u < k) {
        if (row == i + 1u) { v = 1.0; } else if (row > i + 1u) { v = A[row * k + i]; }
    }
    VB[e] = v;
}
"#;

const EIG_ENTRIES: [&str; 6] = ["hh_vec", "hh_matvec", "hh_w", "hh_rank2", "hh_last", "hh_vblock"];
const EIG_PSTRIDE: u64 = 256;

struct EigPipes {
    layout: wgpu::BindGroupLayout,
    p: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
}

fn eig_pipes(c: &Ctx) -> Option<&'static EigPipes> {
    static P: std::sync::OnceLock<Option<EigPipes>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZeroU64::new(16),
            },
            count: None,
        }];
        for b in 1..7 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            });
        }
        let layout = c.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("nvg_eig"), entries: &entries });
        let pl = c.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("nvg_eig"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("nvg_eig"), source: wgpu::ShaderSource::Wgsl(EIG_SRC.into()) });
        if let Some(e) = pollster::block_on(sc.pop()) {
            tracing::warn!("nvg_eig module rejected: {e}");
            return None;
        }
        let mut p = std::collections::HashMap::new();
        for ep in EIG_ENTRIES {
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipe = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: Some(&pl),
                module: &m,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            });
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("nvg_eig pipeline {ep} rejected: {e}");
                return None;
            }
            p.insert(ep, pipe);
        }
        Some(EigPipes { layout, p })
    })
    .as_ref()
}

/// Householder tridiagonalisation of the symmetric `a` (k×k) in place: the
/// reflectors stay below the subdiagonal of `a`; returns `[d | e | tau]`
/// (3k values, f64) — `e[i]` couples i and i+1, `tau[i]` is reflector i's.
pub fn tridiag(a: &DMat) -> Vec<f64> {
    let k = a.r;
    let c = dev();
    let pp = eig_pipes(c).expect("nvg eig pipelines");
    let de = DMat::zeros(1, 3 * k);
    let (vs, pv, wv) = (DMat::zeros(1, k), DMat::zeros(1, k), DMat::zeros(1, k));
    if k < 2 {
        return a.to_f64().into_iter().chain([0.0, 0.0]).collect();
    }
    let steps = k - 1;
    let mut bytes = vec![0u8; (steps + 1) * EIG_PSTRIDE as usize];
    for i in 0..=steps {
        let w = [k as u32, i.min(k - 1) as u32, 0, 0];
        bytes[i * EIG_PSTRIDE as usize..i * EIG_PSTRIDE as usize + 16].copy_from_slice(bytemuck::cast_slice(&w));
    }
    let prm = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nvg_eig_p"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    c.queue.write_buffer(&prm, 0, &bytes);
    let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nvg_eig"),
        layout: &pp.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &prm, offset: 0, size: std::num::NonZeroU64::new(16) }) },
            wgpu::BindGroupEntry { binding: 1, resource: a.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: vs.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pv.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wv.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: de.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: dummy(c).as_entire_binding() },
        ],
    });
    // A pass per chunk of steps keeps each command buffer moderate.
    let chunk = 512usize;
    let mut i0 = 0;
    while i0 < steps {
        let i1 = (i0 + chunk).min(steps);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nvg_tridiag") });
        {
            let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            for i in i0..i1 {
                let l = k - i - 1;
                ps.set_bind_group(0, &bg, &[(i as u64 * EIG_PSTRIDE) as u32]);
                ps.set_pipeline(&pp.p["hh_vec"]);
                ps.dispatch_workgroups(1, 1, 1);
                let g = train_ops::wg_fold(l);
                ps.set_pipeline(&pp.p["hh_matvec"]);
                ps.dispatch_workgroups(g.0, g.1, 1);
                ps.set_pipeline(&pp.p["hh_w"]);
                ps.dispatch_workgroups(1, 1, 1);
                ps.set_pipeline(&pp.p["hh_rank2"]);
                ps.dispatch_workgroups(l.div_ceil(16) as u32, l.div_ceil(16) as u32, 1);
            }
            if i1 == steps {
                ps.set_bind_group(0, &bg, &[(steps as u64 * EIG_PSTRIDE) as u32]);
                ps.set_pipeline(&pp.p["hh_last"]);
                ps.dispatch_workgroups(1, 1, 1);
            }
        }
        submit(c, enc.finish());
        i0 = i1;
    }
    de.to_f64()
}

/// Number of eigenvalues of the tridiagonal (d, e) below `x` (Sturm count).
fn sturm(d: &[f64], e2: &[f64], x: f64, pivmin: f64) -> usize {
    let mut cnt = 0;
    let mut q = d[0] - x;
    if q.abs() < pivmin {
        q = -pivmin;
    }
    if q < 0.0 {
        cnt += 1;
    }
    for i in 1..d.len() {
        q = d[i] - x - e2[i - 1] / q;
        if q.abs() < pivmin {
            q = -pivmin;
        }
        if q < 0.0 {
            cnt += 1;
        }
    }
    cnt
}

/// One eigenvector of the tridiagonal (d, e) for eigenvalue `lam` by
/// inverse iteration (LU with partial pivoting of T − λI), started from a
/// fixed pseudo-random vector; `prev` are vectors it is orthogonalised
/// against (a tight cluster).
fn inv_iter(d: &[f64], e: &[f64], lam: f64, tnorm: f64, seed: u64, prev: &[Vec<f64>]) -> Vec<f64> {
    let k = d.len();
    let eps = f64::EPSILON;
    let pert = eps * tnorm;
    // LU of the tridiagonal T − λI with row interchanges (dgttrf)
    let mut dl: Vec<f64> = e.to_vec();
    let mut dd: Vec<f64> = d.iter().map(|v| v - lam).collect();
    let mut du: Vec<f64> = e.to_vec();
    let mut du2 = vec![0f64; k.saturating_sub(2)];
    let mut piv: Vec<bool> = vec![false; k];
    for i in 0..k.saturating_sub(1) {
        if dd[i].abs() >= dl[i].abs() {
            if dd[i].abs() < pert {
                dd[i] = pert;
            }
            let f = dl[i] / dd[i];
            dl[i] = f;
            dd[i + 1] -= f * du[i];
        } else {
            let f = dd[i] / dl[i];
            dd[i] = dl[i];
            dl[i] = f;
            let t = du[i];
            du[i] = dd[i + 1];
            dd[i + 1] = t - f * dd[i + 1];
            if i + 2 < k {
                du2[i] = du[i + 1];
                du[i + 1] = -f * du[i + 1];
            }
            piv[i] = true;
        }
    }
    if dd[k - 1].abs() < pert {
        dd[k - 1] = pert;
    }
    let mut s = seed | 1;
    let mut x: Vec<f64> = (0..k)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        })
        .collect();
    for _ in 0..4 {
        // solve L U y = P x
        for i in 0..k.saturating_sub(1) {
            if piv[i] {
                x.swap(i, i + 1);
            }
            x[i + 1] -= dl[i] * x[i];
        }
        x[k - 1] /= dd[k - 1];
        if k >= 2 {
            x[k - 2] = (x[k - 2] - du[k - 2] * x[k - 1]) / dd[k - 2];
        }
        for i in (0..k.saturating_sub(2)).rev() {
            x[i] = (x[i] - du[i] * x[i + 1] - du2[i] * x[i + 2]) / dd[i];
        }
        for q in prev {
            let dot: f64 = q.iter().zip(&x).map(|(a, b)| a * b).sum();
            x.iter_mut().zip(q).for_each(|(v, qv)| *v -= dot * qv);
        }
        let nrm = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        x.iter_mut().for_each(|v| *v /= nrm);
    }
    x
}

/// The `r` largest eigenpairs of the symmetric tridiagonal (d, e), f64:
/// eigenvalues descending, eigenvectors as the columns of a k×r row-major
/// matrix.
pub fn tridiag_top(d: &[f64], e: &[f64], r: usize, pool: Option<&crate::pool::Pool>) -> (Vec<f64>, Vec<f64>) {
    let k = d.len();
    let r = r.min(k);
    let e2: Vec<f64> = e.iter().map(|v| v * v).collect();
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for i in 0..k {
        let rad = if i > 0 { e[i - 1].abs() } else { 0.0 } + if i + 1 < k { e[i].abs() } else { 0.0 };
        lo = lo.min(d[i] - rad);
        hi = hi.max(d[i] + rad);
    }
    let tnorm = lo.abs().max(hi.abs()).max(1e-300);
    let pivmin = f64::MIN_POSITIVE.max(tnorm * f64::EPSILON * f64::EPSILON);
    // eigenvalue j (descending) is ascending index k-1-j
    let mut lam = vec![0f64; r];
    {
        let lp = SendPtr(lam.as_mut_ptr());
        let job = |s0: usize, s1: usize| {
            let lp = lp;
            for j in s0..s1 {
                let target = k - 1 - j; // count(x) <= target  <=>  x <= lambda_target
                let (mut a, mut b) = (lo, hi);
                for _ in 0..200 {
                    let mid = 0.5 * (a + b);
                    if mid <= a || mid >= b || (b - a) <= 2.0 * f64::EPSILON * tnorm.max(a.abs().max(b.abs())) * 0.5 {
                        break;
                    }
                    if sturm(d, &e2, mid, pivmin) > target {
                        b = mid;
                    } else {
                        a = mid;
                    }
                }
                // SAFETY: disjoint indices per chunk.
                unsafe { *lp.0.add(j) = 0.5 * (a + b) };
            }
        };
        match pool {
            Some(p) if r > 8 => p.run_rows(r, &job),
            _ => job(0, r),
        }
    }
    // clusters: consecutive eigenvalues closer than 1e-9·‖T‖ share an orthogonalisation
    let ortol = 1e-9 * tnorm;
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut g0 = 0;
    for j in 1..=r {
        if j == r || (lam[j - 1] - lam[j]).abs() > ortol {
            groups.push((g0, j));
            g0 = j;
        }
    }
    let mut vecs: Vec<Vec<f64>> = vec![Vec::new(); r];
    {
        let vp = SendPtr(vecs.as_mut_ptr() as *mut f64);
        let groups = &groups;
        let lam = &lam;
        let job = |s0: usize, s1: usize| {
            let vp = vp;
            let vptr = vp.0 as *mut Vec<f64>;
            for &(a, b) in &groups[s0..s1] {
                let mut done: Vec<Vec<f64>> = Vec::new();
                for j in a..b {
                    // nudge identical eigenvalues apart so the LU differs
                    let shift = (j - a) as f64 * f64::EPSILON * 10.0 * tnorm;
                    let v = inv_iter(d, e, lam[j] + shift, tnorm, 0x9E37_79B9 ^ (j as u64 * 7919), &done);
                    done.push(v);
                }
                for (t, v) in done.into_iter().enumerate() {
                    // SAFETY: disjoint groups per chunk.
                    unsafe { *vptr.add(a + t) = v };
                }
            }
        };
        match pool {
            Some(p) if groups.len() > 8 => p.run_rows(groups.len(), &job),
            _ => job(0, groups.len()),
        }
    }
    let mut z = vec![0f64; k * r];
    for (j, v) in vecs.iter().enumerate() {
        for i in 0..k {
            z[i * r + j] = v[i];
        }
    }
    (lam, z)
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// The `r` largest eigenpairs of the symmetric `a` (k×k, destroyed):
/// eigenvalues descending (f64) and the eigenvectors as a k×r device matrix.
pub fn sym_eig_top(a: &DMat, r: usize, pool: Option<&crate::pool::Pool>) -> (Vec<f64>, DMat) {
    let k = a.r;
    let r = r.min(k);
    let de = tridiag(a);
    let (d, e, tau) = (&de[..k], &de[k..2 * k - 1], &de[2 * k..3 * k]);
    let (lam, z) = tridiag_top(d, e, r, pool);
    let zd = DMat::from_f64(k, r, &z);
    // x = H_0 H_1 ⋯ H_{k-2} z, blocks of reflectors applied last-first.
    let c = dev();
    let pp = eig_pipes(c).expect("nvg eig pipelines");
    let bsz = 32usize;
    let nref = k - 1;
    let vb = DMat::zeros(k, bsz);
    let mut blocks: Vec<usize> = (0..nref).step_by(bsz).collect();
    blocks.reverse();
    let w = DMat::zeros(bsz, r);
    let w2 = DMat::zeros(bsz, r);
    for i0 in blocks {
        let b = bsz.min(nref - i0);
        let prm = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nvg_eig_vb"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        c.queue.write_buffer(&prm, 0, bytemuck::cast_slice(&[k as u32, i0 as u32, bsz as u32, 0]));
        let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nvg_eig_vb"),
            layout: &pp.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &prm, offset: 0, size: std::num::NonZeroU64::new(16) }) },
                wgpu::BindGroupEntry { binding: 1, resource: a.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dummy(c).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dummy(c).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dummy(c).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dummy(c).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: vb.buf.as_entire_binding() },
            ],
        });
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nvg_vblock") });
        {
            let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            ps.set_bind_group(0, &bg, &[0]);
            ps.set_pipeline(&pp.p["hh_vblock"]);
            let g = elt_fold(k * bsz);
            ps.dispatch_workgroups(g.0, g.1, 1);
        }
        submit(c, enc.finish());
        // T (dlarft, forward, columnwise) from VᵀV and tau on the host
        let vtv = mm(&vb, true, &vb, false).to_f64();
        let mut t = vec![0f64; bsz * bsz];
        for i in 0..b {
            let ti = tau[i0 + i];
            t[i * bsz + i] = ti;
            if i > 0 && ti != 0.0 {
                // T[0:i, i] = -tau_i · T[0:i, 0:i] · (Vᵀ v_i)[0:i]
                for row in 0..i {
                    let mut acc = 0.0;
                    for q in row..i {
                        acc += t[row * bsz + q] * vtv[q * bsz + i];
                    }
                    t[row * bsz + i] = -ti * acc;
                }
            }
        }
        let td = DMat::from_f64(bsz, bsz, &t);
        // z -= V (T (Vᵀ z))
        gemm(1.0, &vb, true, &zd, false, 0.0, &w);
        gemm(1.0, &td, false, &w, false, 0.0, &w2);
        gemm(-1.0, &vb, false, &w2, false, 1.0, &zd);
    }
    (lam, zd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    fn host_mm(a: &[f64], ar: usize, ac: usize, ta: bool, b: &[f64], br: usize, bc: usize, tb: bool) -> (Vec<f64>, usize, usize) {
        let (m, k) = if ta { (ac, ar) } else { (ar, ac) };
        let (k2, n) = if tb { (bc, br) } else { (br, bc) };
        assert_eq!(k, k2);
        let mut c = vec![0f64; m * n];
        for i in 0..m {
            for t in 0..k {
                let av = if ta { a[t * ac + i] } else { a[i * ac + t] };
                for j in 0..n {
                    let bv = if tb { b[j * bc + t] } else { b[t * bc + j] };
                    c[i * n + j] += av * bv;
                }
            }
        }
        (c, m, n)
    }

    fn relerr(got: &[f64], want: &[f64]) -> f64 {
        let num: f64 = got.iter().zip(want).map(|(g, w)| (g - w).powi(2)).sum();
        let den: f64 = want.iter().map(|w| w * w).sum::<f64>().max(1e-300);
        (num / den).sqrt()
    }

    /// Every transpose variant at ragged shapes, α/β and the Hadamard
    /// epilogue, against an f64 reference.
    #[test]
    fn gemm_variants_match_host() {
        if !available() {
            eprintln!("no device — skipped");
            return;
        }
        for &(m, n, k) in &[(1usize, 1usize, 1usize), (37, 129, 70), (200, 64, 300), (130, 257, 17)] {
            for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
                let (ar, ac) = if ta { (k, m) } else { (m, k) };
                let (br, bc) = if tb { (n, k) } else { (k, n) };
                let a = rnd(ar * ac, 1 + m as u64);
                let b = rnd(br * bc, 2 + n as u64);
                let c0 = rnd(m * n, 3);
                let e = rnd(m * n, 4);
                let (ab, bb) = (DMat::from_f64(ar, ac, &a), DMat::from_f64(br, bc, &b));
                let cb = DMat::from_f64(m, n, &c0);
                let eb = DMat::from_f64(m, n, &e);
                gemm_v(m, n, k, 0.5, V { m: &ab, off: 0, ld: ac, t: ta }, V { m: &bb, off: 0, ld: bc, t: tb }, -1.5, &cb, 0, n, Some(&eb));
                let (h, _, _) = host_mm(&a, ar, ac, ta, &b, br, bc, tb);
                let want: Vec<f64> = (0..m * n).map(|i| 0.5 * h[i] * e[i] - 1.5 * c0[i]).collect();
                let err = relerr(&cb.to_f64(), &want);
                assert!(err < 1e-5, "gemm {m}x{n}x{k} ta {ta} tb {tb}: {err:e}");
            }
        }
    }

    /// Throughput of the GEMM variants and of the dense Cholesky at encoder
    /// sizes (`--ignored --nocapture`).
    #[test]
    #[ignore]
    fn gemm_and_chol_throughput() {
        if !available() {
            return;
        }
        for &(m, n, k, ta, tb) in &[
            (4096usize, 4096usize, 4096usize, false, false),
            (17408, 1064, 5120, false, false),
            (2152, 2152, 22528, true, false),
            (5120, 5120, 17408, false, true),
            (17408, 1064, 17408, false, false),
        ] {
            let (ar, ac) = if ta { (k, m) } else { (m, k) };
            let (br, bc) = if tb { (n, k) } else { (k, n) };
            let a = DMat::from_f64(ar, ac, &rnd(ar * ac, 1));
            let b = DMat::from_f64(br, bc, &rnd(br * bc, 2));
            let c = DMat::zeros(m, n);
            gemm(1.0, &a, ta, &b, tb, 0.0, &c);
            sync();
            let t = std::time::Instant::now();
            let reps = 5;
            for _ in 0..reps {
                gemm(1.0, &a, ta, &b, tb, 0.0, &c);
            }
            sync();
            let s = t.elapsed().as_secs_f64() / reps as f64;
            eprintln!("gemm {m}x{n}x{k} ta {ta} tb {tb}: {:.2} ms, {:.1} TFLOP/s", s * 1e3, 2.0 * (m * n * k) as f64 / s / 1e12);
        }
        for &n in &[2152usize, 5120, 17408] {
            let x = DMat::from_f64(n, 64, &rnd(n * 64, 3));
            let a = DMat::zeros(n, n);
            gemm(1.0, &x, false, &x, true, 0.0, &a);
            let id: Vec<f64> = (0..n).map(|_| 1.0).collect();
            let dv = DMat::from_f64(1, n, &id);
            let f = DMat::zeros(n, n);
            equil(&a, &f, &dv, 1.0, None);
            sync();
            let t = std::time::Instant::now();
            let ok = chol_in_place(&f);
            let tc = t.elapsed().as_secs_f64();
            let y = DMat::from_f64(1, n, &id);
            let t = std::time::Instant::now();
            chol_solve_dev(&f, &y);
            sync();
            let ts = t.elapsed().as_secs_f64();
            let xi = DMat::zeros(n, n);
            let t = std::time::Instant::now();
            tri_inv(&f, &xi).unwrap();
            sync();
            eprintln!(
                "chol n={n}: {ok:?} {:.1} ms ({:.1} TFLOP/s), solve {:.1} ms, tri_inv {:.1} ms",
                tc * 1e3,
                (n as f64).powi(3) * 2.0 / 3.0 / tc / 1e12,
                ts * 1e3,
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    /// Top-r eigenpairs (device tridiagonalisation + host bisection /
    /// inverse iteration + device back-transform) of a matrix with a known
    /// graded spectrum.
    #[test]
    fn sym_eig_top_matches_known_spectrum() {
        if !available() {
            eprintln!("no device — skipped");
            return;
        }
        let (k, r) = (300usize, 40usize);
        let mu: Vec<f64> = (0..k).map(|i| 1000.0 * 0.96f64.powi(i as i32) + 1e-3 * (i % 3) as f64).collect();
        // Q = product of three reflectors
        let mut q = vec![0f64; k * k];
        for i in 0..k {
            q[i * k + i] = 1.0;
        }
        for s in 0..3u64 {
            let w = rnd(k, 50 + s);
            let ww: f64 = w.iter().map(|x| x * x).sum();
            let mut q2 = q.clone();
            for i in 0..k {
                let dot: f64 = (0..k).map(|j| q[i * k + j] * w[j]).sum();
                for j in 0..k {
                    q2[i * k + j] -= 2.0 * dot * w[j] / ww;
                }
            }
            q = q2;
        }
        let mut a = vec![0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = (0..k).map(|t| q[i * k + t] * mu[t] * q[j * k + t]).sum();
            }
        }
        let ad = DMat::from_f64(k, k, &a);
        let (lam, z) = sym_eig_top(&ad, r, None);
        let z = z.to_f64();
        for j in 0..r {
            assert!((lam[j] - mu[j]).abs() <= 2e-5 * mu[0], "lambda {j}: {} vs {}", lam[j], mu[j]);
            // residual |A z - lam z| / |A|
            let mut res = 0f64;
            for i in 0..k {
                let az: f64 = (0..k).map(|t| a[i * k + t] * z[t * r + j]).sum();
                res += (az - lam[j] * z[i * r + j]).powi(2);
            }
            assert!(res.sqrt() <= 2e-4 * mu[0], "residual {j}: {}", res.sqrt());
            for jj in 0..=j {
                let d: f64 = (0..k).map(|i| z[i * r + j] * z[i * r + jj]).sum();
                assert!((d - if j == jj { 1.0 } else { 0.0 }).abs() < 1e-3, "orthogonality {j},{jj}: {d}");
            }
        }
    }

    /// Dense SPD solve (equilibrated device Cholesky + triangular solves)
    /// and the act-order upper factor of the inverse.
    #[test]
    fn spd_solve_and_uc_match_host() {
        if !available() {
            eprintln!("no device — skipped");
            return;
        }
        let n = 150usize;
        let x = rnd(n * (n + 20), 7);
        let (mut a, _, _) = host_mm(&x, n, n + 20, false, &x, n, n + 20, true);
        for i in 0..n {
            a[i * n + i] += 0.5 + 40.0 * (i % 7 == 0) as u8 as f64; // uneven diagonal
        }
        let b = rnd(n, 9);
        let ad = DMat::from_f64(n, n, &a);
        let xs = spd_solve(&ad, &b, 1e-10);
        let ax: Vec<f64> = (0..n).map(|i| (0..n).map(|j| a[i * n + j] * xs[j]).sum()).collect();
        let err = relerr(&ax, &b);
        assert!(err < 1e-4, "spd residual {err:e}");
        // Uc: upper factor of the permuted inverse, Ucᵀ Uc Hp = I
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by(|&p, &q| a[q * n + q].partial_cmp(&a[p * n + p]).unwrap());
        let pb = upload_u32(&perm.iter().map(|&v| v as u32).collect::<Vec<_>>());
        let d: Vec<f64> = (0..n).map(|i| a[i * n + i].sqrt()).collect();
        let dv = DMat::from_f64(1, n, &d);
        let f = DMat::zeros(n, n);
        equil(&ad, &f, &dv, 0.0, Some(&pb));
        assert_eq!(chol_in_place(&f), Some(true));
        let xi = DMat::zeros(n, n);
        tri_inv(&f, &xi).unwrap();
        let dpv: Vec<f64> = perm.iter().map(|&p| d[p]).collect();
        let uc = DMat::zeros(n, n);
        make_uc(&xi, &DMat::from_f64(1, n, &dpv), &uc);
        let u = uc.to_f64();
        let (utu, _, _) = host_mm(&u, n, n, true, &u, n, n, false);
        let mut worst = 0f64;
        for i in 0..n {
            for j in 0..n {
                let v: f64 = (0..n).map(|t| utu[i * n + t] * a[perm[t] * n + perm[j]]).sum();
                worst = worst.max((v - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        assert!(worst < 1e-3, "Ucᵀ Uc Hp − I: {worst:e}");
    }
}
