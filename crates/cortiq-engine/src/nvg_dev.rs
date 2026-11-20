//! Device linear algebra of the NVG `blr` encoder on the selected backend:
//! `gpu_wgpu::nvg_dev` (the default), or `gpu_cuda::nvg_dev` when the build
//! has the `cuda` feature and `CMF_GPU=cuda` brought the card up. One driver
//! (`cortiq-cli` `nvg_gpu.rs`) serves both: every call here forwards to the
//! backend the process runs on, a matrix made on one backend never meets the
//! other.

#[cfg(feature = "cuda")]
use crate::gpu_cuda::nvg_dev as cu;
use crate::gpu_wgpu::nvg_dev as w;

pub use w::{ABS, ACC, COL, INPLACE, LOWER, ROW, SIGN};

/// The largest rank [`gv_block`] takes (the same on both backends).
pub const GV_MAX_RANK: usize = w::GV_MAX_RANK;

/// The CUDA twin is selected and its kernels compile on the card.
pub fn on_cuda() -> bool {
    #[cfg(feature = "cuda")]
    {
        // `CMF_NVG_ENC=cuda|wgpu` picks the encoder's backend apart from the
        // engine's (a calibration on wgpu with the encoder on CUDA, or the
        // reverse: an A/B on identical Grams)
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| match std::env::var("CMF_NVG_ENC").as_deref() {
            Ok("cuda") => cu::available(),
            Ok("wgpu") => false,
            _ => crate::gpu_cuda::selected() && cu::available(),
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// `cuda` or `wgpu`: where the encoder's device work runs.
pub fn backend_name() -> &'static str {
    if on_cuda() { "cuda" } else { "wgpu" }
}

macro_rules! be {
    ($w:expr, $c:expr) => {{
        #[cfg(feature = "cuda")]
        {
            if on_cuda() { $c } else { $w }
        }
        #[cfg(not(feature = "cuda"))]
        {
            $w
        }
    }};
}

enum Inner {
    W(w::DMat),
    #[cfg(feature = "cuda")]
    C(cu::DMat),
}

/// A row-major f32 matrix on the device.
pub struct DMat {
    pub r: usize,
    pub c: usize,
    i: Inner,
}

impl DMat {
    fn of_w(m: w::DMat) -> DMat {
        DMat { r: m.r, c: m.c, i: Inner::W(m) }
    }
    #[cfg(feature = "cuda")]
    fn of_c(m: cu::DMat) -> DMat {
        DMat { r: m.r, c: m.c, i: Inner::C(m) }
    }
    fn w(&self) -> &w::DMat {
        match &self.i {
            Inner::W(m) => m,
            #[cfg(feature = "cuda")]
            Inner::C(_) => panic!("a CUDA matrix in a wgpu NVG call"),
        }
    }
    #[cfg(feature = "cuda")]
    fn cu(&self) -> &cu::DMat {
        match &self.i {
            Inner::C(m) => m,
            Inner::W(_) => panic!("a wgpu matrix in a CUDA NVG call"),
        }
    }
    pub fn zeros(r: usize, c: usize) -> DMat {
        be!(DMat::of_w(w::DMat::zeros(r, c)), DMat::of_c(cu::DMat::zeros(r, c)))
    }
    pub fn from_f32(r: usize, c: usize, v: &[f32]) -> DMat {
        be!(DMat::of_w(w::DMat::from_f32(r, c, v)), DMat::of_c(cu::DMat::from_f32(r, c, v)))
    }
    pub fn from_f64(r: usize, c: usize, v: &[f64]) -> DMat {
        be!(DMat::of_w(w::DMat::from_f64(r, c, v)), DMat::of_c(cu::DMat::from_f64(r, c, v)))
    }
    pub fn write_f64(&self, v: &[f64]) {
        be!(self.w().write_f64(v), self.cu().write_f64(v))
    }
    pub fn len(&self) -> usize {
        self.r * self.c
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn to_f32(&self) -> Vec<f32> {
        be!(self.w().to_f32(), self.cu().to_f32())
    }
    pub fn to_f64(&self) -> Vec<f64> {
        be!(self.w().to_f64(), self.cu().to_f64())
    }
    /// A device copy.
    pub fn dup(&self) -> DMat {
        be!(DMat::of_w(self.w().dup()), DMat::of_c(self.cu().dup()))
    }
}

/// A u32 device vector (permutations).
pub enum U32Buf {
    W(wgpu::Buffer),
    #[cfg(feature = "cuda")]
    C(cu::U32Buf),
}

impl U32Buf {
    fn w(&self) -> &wgpu::Buffer {
        match self {
            U32Buf::W(b) => b,
            #[cfg(feature = "cuda")]
            U32Buf::C(_) => panic!("a CUDA index vector in a wgpu NVG call"),
        }
    }
    #[cfg(feature = "cuda")]
    fn cu(&self) -> &cu::U32Buf {
        match self {
            U32Buf::C(b) => b,
            U32Buf::W(_) => panic!("a wgpu index vector in a CUDA NVG call"),
        }
    }
}

/// The device is up and the NVG kernels compile on it.
pub fn available() -> bool {
    on_cuda() || w::available()
}

/// Wait until everything queued so far has finished (stage timing).
pub fn sync() {
    be!(w::sync(), cu::sync())
}

/// `dst ← src` (same element count).
pub fn copy(src: &DMat, dst: &DMat) {
    be!(w::copy(src.w(), dst.w()), cu::copy(src.cu(), dst.cu()))
}

/// Copy `len` floats `src[so..] → dst[do_..]`.
pub fn copy_range(src: &DMat, so: usize, dst: &DMat, do_: usize, len: usize) {
    be!(w::copy_range(src.w(), so, dst.w(), do_, len), cu::copy_range(src.cu(), so, dst.cu(), do_, len))
}

/// A u32 device vector.
pub fn upload_u32(v: &[u32]) -> U32Buf {
    be!(U32Buf::W(w::upload_u32(v)), U32Buf::C(cu::upload_u32(v)))
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
    fn w(&self) -> w::V<'a> {
        w::V { m: self.m.w(), off: self.off, ld: self.ld, t: self.t }
    }
    #[cfg(feature = "cuda")]
    fn cu(&self) -> cu::V<'a> {
        cu::V { m: self.m.cu(), off: self.off, ld: self.ld, t: self.t }
    }
}

/// `C[oc + i·ldc + j] = α·Σ_k op(A)(i,k)·op(B)(k,j) [⊙ E] + β·C` for an
/// `m×n` output (`k` the inner dimension). `E` (same layout as C) when given.
#[allow(clippy::too_many_arguments)]
pub fn gemm_v(m: usize, n: usize, k: usize, alpha: f32, a: V, b: V, beta: f32, c: &DMat, oc: usize, ldc: usize, e: Option<&DMat>) {
    be!(
        w::gemm_v(m, n, k, alpha, a.w(), b.w(), beta, c.w(), oc, ldc, e.map(|x| x.w())),
        cu::gemm_v(m, n, k, alpha, a.cu(), b.cu(), beta, c.cu(), oc, ldc, e.map(|x| x.cu()))
    )
}

/// `C = α·op(A)·op(B) + β·C` on whole matrices.
pub fn gemm(alpha: f32, a: &DMat, ta: bool, b: &DMat, tb: bool, beta: f32, c: &DMat) {
    be!(w::gemm(alpha, a.w(), ta, b.w(), tb, beta, c.w()), cu::gemm(alpha, a.cu(), ta, b.cu(), tb, beta, c.cu()))
}

/// `op(A)·op(B)` into a new matrix.
pub fn mm(a: &DMat, ta: bool, b: &DMat, tb: bool) -> DMat {
    be!(DMat::of_w(w::mm(a.w(), ta, b.w(), tb)), DMat::of_c(cu::mm(a.cu(), ta, b.cu(), tb)))
}

/// `C[oc + i·ldc + j] = α·f(X(i,j))·R[i]·S[j] (+ β·C)` over `rows × cols`,
/// `X(i,j) = A[oa + i·lda + j]` (or C itself with [`INPLACE`]).
#[allow(clippy::too_many_arguments)]
pub fn scale_rc(rows: usize, cols: usize, a: Option<(&DMat, usize, usize)>, c: (&DMat, usize, usize), r: Option<&DMat>, s: Option<&DMat>, flags: u32, alpha: f32, beta: f32) {
    be!(
        w::scale_rc(rows, cols, a.map(|(m, o, l)| (m.w(), o, l)), (c.0.w(), c.1, c.2), r.map(|x| x.w()), s.map(|x| x.w()), flags, alpha, beta),
        cu::scale_rc(rows, cols, a.map(|(m, o, l)| (m.cu(), o, l)), (c.0.cu(), c.1, c.2), r.map(|x| x.cu()), s.map(|x| x.cu()), flags, alpha, beta)
    )
}

/// Whole-matrix `C = α·f(A)·diag-scalings`.
pub fn scale_into(a: &DMat, c: &DMat, r: Option<&DMat>, s: Option<&DMat>, flags: u32, alpha: f32) {
    be!(
        w::scale_into(a.w(), c.w(), r.map(|x| x.w()), s.map(|x| x.w()), flags, alpha),
        cu::scale_into(a.cu(), c.cu(), r.map(|x| x.cu()), s.map(|x| x.cu()), flags, alpha)
    )
}

/// `C ← α·A + β·C` (whole matrices).
pub fn axpby(alpha: f32, a: &DMat, beta: f32, c: &DMat) {
    be!(w::axpby(alpha, a.w(), beta, c.w()), cu::axpby(alpha, a.cu(), beta, c.cu()))
}

/// `O[i] = Σ_j A[oa+i·lda+j]·B[ob+i·ldb+j]` for `rows` rows of `cols` columns.
#[allow(clippy::too_many_arguments)]
pub fn rowdot(rows: usize, cols: usize, a: &DMat, oa: usize, lda: usize, b: &DMat, ob: usize, ldb: usize, out: &DMat) {
    be!(w::rowdot(rows, cols, a.w(), oa, lda, b.w(), ob, ldb, out.w()), cu::rowdot(rows, cols, a.cu(), oa, lda, b.cu(), ob, ldb, out.cu()))
}

/// Row-normalise `rows × cols` of `c` (at `oc`, leading dimension `ldc`) in place.
pub fn rownorm(rows: usize, cols: usize, c: &DMat, oc: usize, ldc: usize) {
    be!(w::rownorm(rows, cols, c.w(), oc, ldc), cu::rownorm(rows, cols, c.cu(), oc, ldc))
}

/// `Σ_i A[oa+i·lda+j]·B[ob+i·ldb+j]` per column, summed in f64 on the host.
#[allow(clippy::too_many_arguments)]
pub fn colsum_prod(rows: usize, cols: usize, a: &DMat, oa: usize, lda: usize, b: &DMat, ob: usize, ldb: usize) -> Vec<f64> {
    be!(w::colsum_prod(rows, cols, a.w(), oa, lda, b.w(), ob, ldb), cu::colsum_prod(rows, cols, a.cu(), oa, lda, b.cu(), ob, ldb))
}

/// `out[oo + j] = Σ_i A[oa+i·lda+j]·B[ob+i·ldb+j]` on the device (no readback).
#[allow(clippy::too_many_arguments)]
pub fn colsum_dev(rows: usize, cols: usize, a: &DMat, oa: usize, lda: usize, b: &DMat, ob: usize, ldb: usize, out: &DMat, oo: usize) {
    be!(
        w::colsum_dev(rows, cols, a.w(), oa, lda, b.w(), ob, ldb, out.w(), oo),
        cu::colsum_dev(rows, cols, a.cu(), oa, lda, b.cu(), ob, ldb, out.cu(), oo)
    )
}

/// `Σ A ⊙ B` over the first `len` elements, f64 on the host.
pub fn frob(a: &DMat, b: &DMat, len: usize) -> f64 {
    be!(w::frob(a.w(), b.w(), len), cu::frob(a.cu(), b.cu(), len))
}

/// Several `Σ A⊙B` read back together: `add` queues one (no readback),
/// `read` returns them all in f64 after one wait.
pub enum FrobBatch {
    W(w::FrobBatch),
    #[cfg(feature = "cuda")]
    C(cu::FrobBatch),
}

impl FrobBatch {
    pub fn new() -> Self {
        be!(FrobBatch::W(w::FrobBatch::new()), FrobBatch::C(cu::FrobBatch::new()))
    }
    /// Queue `Σ_{i<len} a[i]·b[i]`; returns its slot.
    pub fn add(&mut self, a: &DMat, b: &DMat, len: usize) -> usize {
        match self {
            FrobBatch::W(f) => f.add(a.w(), b.w(), len),
            #[cfg(feature = "cuda")]
            FrobBatch::C(f) => f.add(a.cu(), b.cu(), len),
        }
    }
    pub fn read(self) -> Vec<f64> {
        match self {
            FrobBatch::W(f) => f.read(),
            #[cfg(feature = "cuda")]
            FrobBatch::C(f) => f.read(),
        }
    }
}

impl Default for FrobBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Diagonal of the square `n×n` view at `oa` with leading dimension `lda`.
pub fn diag(a: &DMat, n: usize, oa: usize, lda: usize) -> Vec<f64> {
    be!(w::diag(a.w(), n, oa, lda), cu::diag(a.cu(), n, oa, lda))
}

/// `F = (S + ridge·I) / (d dᵀ)` (`d` = `dv`), optionally in reversed act
/// order (`perm`: F[i][j] uses S[perm[n-1-i]][perm[n-1-j]]).
pub fn equil(s: &DMat, f: &DMat, dv: &DMat, ridge: f32, perm: Option<&U32Buf>) {
    be!(w::equil(s.w(), f.w(), dv.w(), ridge, perm.map(|p| p.w())), cu::equil(s.cu(), f.cu(), dv.cu(), ridge, perm.map(|p| p.cu())))
}

/// `Uc[j][k] = X[n-1-j][n-1-k] / dp[k]` (k ≥ j).
pub fn make_uc(x: &DMat, dp: &DMat, uc: &DMat) {
    be!(w::make_uc(x.w(), dp.w(), uc.w()), cu::make_uc(x.cu(), dp.cu(), uc.cu()))
}

/// Row gather `C[j] = A[perm[j]]` or scatter `C[perm[j]] = A[j]`.
pub fn permute_rows(a: &DMat, c: &DMat, perm: &U32Buf, scatter: bool) {
    be!(w::permute_rows(a.w(), c.w(), perm.w(), scatter), cu::permute_rows(a.cu(), c.cu(), perm.cu(), scatter))
}

/// `Aᵀ` into a new matrix.
pub fn transpose(a: &DMat) -> DMat {
    be!(DMat::of_w(w::transpose(a.w())), DMat::of_c(cu::transpose(a.cu())))
}

/// [`spd_solve`] with the right-hand side and the solution on the device.
pub fn spd_solve_d(a: &DMat, b: &DMat, rel: f64) -> DMat {
    be!(DMat::of_w(w::spd_solve_d(a.w(), b.w(), rel)), DMat::of_c(cu::spd_solve_d(a.cu(), b.cu(), rel)))
}

/// Per-row h solve on the device; `h1` unused for P = 1.
#[allow(clippy::too_many_arguments)]
pub fn hsolve(m: usize, np: usize, rh0: &DMat, rh1: &DMat, g00: &DMat, g01: &DMat, g11: &DMat, h0: &DMat, h1: &DMat) {
    be!(
        w::hsolve(m, np, rh0.w(), rh1.w(), g00.w(), g01.w(), g11.w(), h0.w(), h1.w()),
        cu::hsolve(m, np, rh0.cu(), rh1.cu(), g00.cu(), g01.cu(), g11.cu(), h0.cu(), h1.cu())
    )
}

/// `dst[perm[j]] = src[so + j]` for `n` entries.
pub fn scatter_vec(src: &DMat, so: usize, dst: &DMat, perm: &U32Buf, n: usize) {
    be!(w::scatter_vec(src.w(), so, dst.w(), perm.w(), n), cu::scatter_vec(src.cu(), so, dst.cu(), perm.cu(), n))
}

/// `out = a / ‖a‖_F` on the device (no readback).
pub fn normalize_frob(a: &DMat, out: &DMat) {
    be!(w::normalize_frob(a.w(), out.w()), cu::normalize_frob(a.cu(), out.cu()))
}

/// Solve `L Lᵀ x = b` in place (`y` holds b; `l` the dense lower factor).
pub fn chol_solve_dev(l: &DMat, y: &DMat) {
    be!(w::chol_solve_dev(l.w(), y.w()), cu::chol_solve_dev(l.cu(), y.cu()))
}

/// Lower Cholesky of the square `a` in place (`Some(false)`: a pivot failed).
pub fn chol_in_place(a: &DMat) -> Option<bool> {
    be!(w::chol_in_place(a.w()), cu::chol_in_place(a.cu()))
}

/// `x = L⁻¹` (`l` as [`chol_in_place`] leaves it).
pub fn tri_inv(l: &DMat, x: &DMat) -> Option<()> {
    be!(w::tri_inv(l.w(), x.w()), cu::tri_inv(l.cu(), x.cu()))
}

/// Output-side flips by blocks of 32 ranks; returns the flip count.
#[allow(clippy::too_many_arguments)]
pub fn flipu_blocked(su: &DMat, qs: &DMat, whb: &DMat, q: &DMat, h: &DMat, l: &DMat, m: usize, r: usize, tol: f32) -> Option<u64> {
    be!(
        w::flipu_blocked(su.w(), qs.w(), whb.w(), q.w(), h.w(), l.w(), m, r, tol),
        cu::flipu_blocked(su.cu(), qs.cu(), whb.cu(), q.cu(), h.cu(), l.cu(), m, r, tol)
    )
}

/// Output-side flips, one thread per row over transposed operands.
#[allow(clippy::too_many_arguments)]
pub fn flipu(sut: &DMat, qst: &DMat, whbt: &DMat, q: &DMat, h: &DMat, l: &DMat, m: usize, r: usize, tol: f32) -> u64 {
    be!(
        w::flipu(sut.w(), qst.w(), whbt.w(), q.w(), h.w(), l.w(), m, r, tol),
        cu::flipu(sut.cu(), qst.cu(), whbt.cu(), q.cu(), h.cu(), l.cu(), m, r, tol)
    )
}

/// One 128-column block of the input-side search, then the trailing
/// feedback as a GEMM. `cnt` accumulates the flips (a u32 in its first word).
#[allow(clippy::too_many_arguments)]
pub fn gv_block(bv: &DMat, s: &DMat, ks0: &DMat, k: &DMat, uc: &DMat, eg: &DMat, cnt: &DMat, b0: usize, b1: usize, max_flips: usize, tol: f32) {
    be!(
        w::gv_block(bv.w(), s.w(), ks0.w(), k.w(), uc.w(), eg.w(), cnt.w(), b0, b1, max_flips, tol),
        cu::gv_block(bv.cu(), s.cu(), ks0.cu(), k.cu(), uc.cu(), eg.cu(), cnt.cu(), b0, b1, max_flips, tol)
    )
}

/// The `r` largest eigenpairs of the symmetric `a` (k×k, destroyed):
/// eigenvalues descending (f64) and the eigenvectors as a k×r device matrix.
pub fn sym_eig_top(a: &DMat, r: usize, pool: Option<&crate::pool::Pool>) -> (Vec<f64>, DMat) {
    be!(
        {
            let (lam, z) = w::sym_eig_top(a.w(), r, pool);
            (lam, DMat::of_w(z))
        },
        {
            let (lam, z) = cu::sym_eig_top(a.cu(), r, pool);
            (lam, DMat::of_c(z))
        }
    )
}
