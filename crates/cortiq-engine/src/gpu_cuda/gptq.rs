//! The host-heavy parts of quantize-moe's GPTQ on the card (`cortiq-cli`
//! `gptq.rs`): the row-square GEMM `A·B` (`B` square — a Hessian; the
//! debias products and the refinement's seed) on cuBLAS in plain f32, and
//! SFq's code refinement (greedy single-code moves on `r·H_f·rᵀ`) a block a
//! row. Both run beside the wgpu engine that does the gram, fold and sweeps:
//! on a pod of 7.65 CPUs they were most of an expert's encode.

use super::*;
use std::sync::Mutex;

const SRC: &str = r#"
typedef unsigned char u8;

// One block a row: the host's `tiled_refine_codes` loop. code/step/res are
// the row's levels, steps (0 = frozen) and residual Ŵ − W; G = R·H_f its
// seed; hd = diag(H_f); live: H_jj > 0. Every move is the best single-code
// step (ties: the lowest column), the arithmetic the host's (f32, no fused
// multiply-add). stats: per row Σ r·g and Σ r² before and after, moves.
extern "C" __global__ void gq_refine(u8* __restrict__ code, const float* __restrict__ step, const float* __restrict__ res,
                                     const float* __restrict__ G, const float* __restrict__ Hf, const float* __restrict__ hd,
                                     const u8* __restrict__ live, int cols, int maxc, int iters,
                                     double* __restrict__ stats, int* __restrict__ moves)
{
    extern __shared__ float sm[];
    float* g = sm;
    float* r = sm + cols;
    float* st = sm + 2 * cols;
    u8* cd = (u8*)(sm + 3 * cols);
    __shared__ float bv[32];
    __shared__ int bj[32];
    __shared__ int bd[32];
    __shared__ double red[2][32];
    __shared__ int sel_j, sel_d;
    const int row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x, lane = tid & 31, warp = tid >> 5, nw = (nt + 31) >> 5;
    const size_t base = (size_t)row * cols;
    for (int j = tid; j < cols; j += nt) {
        g[j] = G[base + j];
        r[j] = res[base + j];
        st[j] = step[base + j];
        cd[j] = code[base + j];
    }
    __syncthreads();
    // the objective before
    {
        double a = 0.0, b = 0.0;
        for (int j = tid; j < cols; j += nt) {
            a += (double)r[j] * (double)g[j];
            b += (double)r[j] * (double)r[j];
        }
        for (int o = 16; o > 0; o >>= 1) {
            a += __shfl_xor_sync(0xffffffffu, a, o);
            b += __shfl_xor_sync(0xffffffffu, b, o);
        }
        if (lane == 0) {
            red[0][warp] = a;
            red[1][warp] = b;
        }
        __syncthreads();
        if (tid == 0) {
            double sa = 0.0, sb = 0.0;
            for (int w = 0; w < nw; w++) {
                sa += red[0][w];
                sb += red[1][w];
            }
            stats[row * 4 + 0] = sa;
            stats[row * 4 + 1] = sb;
        }
        __syncthreads();
    }
    int nmoves = 0;
    for (int it = 0; it < iters; it++) {
        float best = 0.0f;
        int bjj = 0x7fffffff, bdd = 0;
        for (int j = tid; j < cols; j += nt) {
            const float sj = st[j];
            if (sj == 0.0f || !live[j]) continue;
            const float q = __fmul_rn(__fmul_rn(sj, sj), hd[j]);
            const float gj2 = __fmul_rn(__fmul_rn(2.0f, sj), g[j]);
            const float up = __fadd_rn(gj2, q), dn = __fsub_rn(q, gj2);
            if (cd[j] < maxc && up < best) {
                best = up;
                bjj = j;
                bdd = 1;
            }
            if (cd[j] > 0 && dn < best) {
                best = dn;
                bjj = j;
                bdd = -1;
            }
        }
        // the block's best (value, then the lowest column)
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oj = __shfl_xor_sync(0xffffffffu, bjj, o);
            const int od = __shfl_xor_sync(0xffffffffu, bdd, o);
            if (ov < best || (ov == best && oj < bjj)) {
                best = ov;
                bjj = oj;
                bdd = od;
            }
        }
        if (lane == 0) {
            bv[warp] = best;
            bj[warp] = bjj;
            bd[warp] = bdd;
        }
        __syncthreads();
        if (tid == 0) {
            float b = bv[0];
            int j = bj[0], d = bd[0];
            for (int w = 1; w < nw; w++)
                if (bv[w] < b || (bv[w] == b && bj[w] < j)) {
                    b = bv[w];
                    j = bj[w];
                    d = bd[w];
                }
            sel_j = (d != 0 && j < cols) ? j : -1;
            sel_d = d;
        }
        __syncthreads();
        const int j = sel_j;
        if (j < 0) break;
        const float d = __fmul_rn((float)sel_d, st[j]);
        if (tid == 0) {
            cd[j] = (u8)((int)cd[j] + sel_d);
            r[j] = __fadd_rn(r[j], d);
        }
        const float* hr = Hf + (size_t)j * cols;
        for (int k = tid; k < cols; k += nt) g[k] = __fadd_rn(g[k], __fmul_rn(d, hr[k]));
        nmoves++;
        __syncthreads();
    }
    // the objective after
    {
        double a = 0.0, b = 0.0;
        for (int j = tid; j < cols; j += nt) {
            a += (double)r[j] * (double)g[j];
            b += (double)r[j] * (double)r[j];
        }
        for (int o = 16; o > 0; o >>= 1) {
            a += __shfl_xor_sync(0xffffffffu, a, o);
            b += __shfl_xor_sync(0xffffffffu, b, o);
        }
        if (lane == 0) {
            red[0][warp] = a;
            red[1][warp] = b;
        }
        __syncthreads();
        if (tid == 0) {
            double sa = 0.0, sb = 0.0;
            for (int w = 0; w < nw; w++) {
                sa += red[0][w];
                sb += red[1][w];
            }
            stats[row * 4 + 2] = sa;
            stats[row * 4 + 3] = sb;
            moves[row] = nmoves;
        }
    }
    for (int j = tid; j < cols; j += nt) code[base + j] = cd[j];
}
"#;

struct GqState {
    f: Option<Result<cs::CUfunction, String>>,
    scr: std::collections::HashMap<&'static str, Buf>,
}

// SAFETY: used under the context's state lock.
unsafe impl Send for GqState {}

static STATE: Mutex<Option<GqState>> = Mutex::new(None);

impl GqState {
    fn scr(&mut self, name: &'static str, bytes: usize) -> Res<u64> {
        let bytes = bytes.max(64);
        if !self.scr.get(name).is_some_and(|b| b.bytes >= bytes) {
            self.scr.remove(name);
            self.scr.insert(name, Buf::direct(bytes.next_multiple_of(1 << 20))?);
        }
        Ok(self.scr[name].ptr)
    }
}

/// `CMF_GPTQ_CUDA=0`: the host does these parts (read each call: a test
/// compares both in one process).
fn on() -> bool {
    std::env::var("CMF_GPTQ_CUDA").as_deref() != Ok("0")
}

/// `out[m][k] = a[m][k] · b[k][k]` (row-major) on cuBLAS in plain f32.
/// False: the caller's host GEMM.
pub fn gptq_gemm_rows_square(a: &[f32], m: usize, k: usize, b: &[f32], out: &mut [f32]) -> bool {
    if !on() || m == 0 || k == 0 || a.len() < m * k || b.len() < k * k || out.len() < m * k {
        return false;
    }
    let Some(c) = ctx() else { return false };
    if c.current().is_err() {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(|| GqState { f: None, scr: Default::default() });
    let mut run = || -> Res<()> {
        let ap = s.scr("gq_a", m * k * 4)?;
        let bp = s.scr("gq_b", k * k * 4)?;
        let op = s.scr("gq_o", m * k * 4)?;
        unsafe {
            cu::memcpy_htod_async(ap, &a[..m * k], c.stream).map_err(derr)?;
            cu::memcpy_htod_async(bp, &b[..k * k], c.stream).map_err(derr)?;
            // column-major: outᵀ[k][m] = bᵀ·aᵀ, the row-major arrays as they are
            c.gemm(false, false, k, m, k, 1.0, bp, k, ap, k, 0.0, op, k, Gemm::F32)?;
            cu::memcpy_dtoh_async(&mut out[..m * k], op, c.stream).map_err(derr)?;
        }
        c.sync()
    };
    match run() {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("gptq gemm", &e);
            false
        }
    }
}

/// Upper Cholesky `A = UᵀU` in place of the column-major `n×n` matrix at
/// `a` (only its upper triangle is read or written), right-looking in blocks
/// of `FOLD_NB`: a diagonal block on the host in f64 (the host fold's pivot
/// rule: a pivot that is not finite and positive fails), its row panel by
/// `dtrsm`, the trailing matrix by `dsyrk`. Ok(false): not positive definite.
const FOLD_NB: usize = 256;

fn blas_ok(s: bl::cublasStatus_t, what: &str) -> Res<()> {
    if s == bl::cublasStatus_t::CUBLAS_STATUS_SUCCESS { Ok(()) } else { Err(format!("cuBLAS {what}: {s:?}")) }
}

unsafe fn chol_upper_dev(c: &Ctx, a: u64, n: usize, blk: &mut Vec<f64>) -> Res<bool> {
    use bl::{cublasDiagType_t as Dg, cublasFillMode_t as Fm, cublasOperation_t as Op, cublasSideMode_t as Sd};
    let at = |r: usize, col: usize| a + ((r + col * n) * 8) as u64;
    let mut k = 0;
    while k < n {
        let kb = FOLD_NB.min(n - k);
        // the block's trailing updates are done (the stream is in order)
        c.sync()?;
        blk.resize(kb * kb, 0.0);
        unsafe {
            blas_ok(bl::cublasGetMatrix(kb as i32, kb as i32, 8, at(k, k) as *const c_void, n as i32, blk.as_mut_ptr().cast(), kb as i32), "get block")?;
        }
        // U11ᵀU11 = A11, column-major, upper: u(i, j) = blk[i + j kb], i ≤ j
        for j in 0..kb {
            let mut d = blk[j + j * kb];
            for i in 0..j {
                d -= blk[i + j * kb] * blk[i + j * kb];
            }
            if !(d > 0.0) || !d.is_finite() {
                return Ok(false);
            }
            let djj = d.sqrt();
            blk[j + j * kb] = djj;
            for col in j + 1..kb {
                let mut v = blk[j + col * kb];
                for i in 0..j {
                    v -= blk[i + j * kb] * blk[i + col * kb];
                }
                blk[j + col * kb] = v / djj;
            }
        }
        unsafe {
            blas_ok(bl::cublasSetMatrix(kb as i32, kb as i32, 8, blk.as_ptr().cast(), kb as i32, at(k, k) as *mut c_void, n as i32), "set block")?;
        }
        let rest = n - k - kb;
        if rest > 0 {
            let (one, m_one) = (1.0f64, -1.0f64);
            unsafe {
                // A12 ← U11⁻ᵀ A12
                blas_ok(
                    bl::cublasDtrsm_v2(c.blas, Sd::CUBLAS_SIDE_LEFT, Fm::CUBLAS_FILL_MODE_UPPER, Op::CUBLAS_OP_T, Dg::CUBLAS_DIAG_NON_UNIT,
                        kb as i32, rest as i32, &one, at(k, k) as *const f64, n as i32, at(k, k + kb) as *mut f64, n as i32),
                    "dtrsm panel",
                )?;
                // A22 ← A22 − A12ᵀ A12
                blas_ok(
                    bl::cublasDsyrk_v2(c.blas, Fm::CUBLAS_FILL_MODE_UPPER, Op::CUBLAS_OP_T, rest as i32, kb as i32, &m_one,
                        at(k, k + kb) as *const f64, n as i32, &one, at(k + kb, k + kb) as *mut f64, n as i32),
                    "dsyrk trailing",
                )?;
            }
        }
        k += kb;
    }
    c.sync()?;
    Ok(true)
}

/// The dense algebra of GPTQ's fold (`cortiq-cli` `gptq_fold_operator_par`)
/// on cuBLAS in f64: `a` is the damped Hessian `H + λ·mean(diag)·I` (n×n,
/// symmetric, so its row-major array is also its column-major one). On the
/// card: `H = UᵀU`, `V = U⁻¹` (`dtrsm` against I), `H⁻¹ = V·Vᵀ` (`dsyrk`),
/// and `H⁻¹ = WᵀW` again; returned is `W`'s array, which read row-major is
/// the host's lower `M` with `H⁻¹ = M·Mᵀ` (its upper half undefined).
/// The host's math in another summation order — the codes it leads to can
/// differ at rounding boundaries, not in kind.
/// `Some(Err(1))`: H is not positive definite at this λ (the caller damps
/// more), `Some(Err(2))`: H⁻¹ is not (the caller's identity fold);
/// None: no card, or `CMF_GPTQ_CUDA=0` (the host's path).
pub fn gptq_fold_chol_inv(a: &[f64], n: usize) -> Option<Result<Vec<f64>, u8>> {
    if !on() || n == 0 || a.len() < n * n {
        return None;
    }
    let c = ctx()?;
    if c.current().is_err() {
        return None;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(|| GqState { f: None, scr: Default::default() });
    let mut run = || -> Res<Result<Vec<f64>, u8>> {
        use bl::{cublasDiagType_t as Dg, cublasFillMode_t as Fm, cublasOperation_t as Op, cublasSideMode_t as Sd};
        let bytes = n * n * 8;
        let ap = s.scr("gq_fold_a", bytes)?;
        let vp = s.scr("gq_fold_v", bytes)?;
        let onep = s.scr("gq_fold_one", 8)?;
        let mut blk = Vec::new();
        // the host sources outlive the copies (a pageable copy may still be
        // reading when it returns): both are alive past the sync
        let one_h = [1.0f64];
        // `CMF_CUDA_PROF`: the fold's stages, one line a fold
        let t0 = std::time::Instant::now();
        let mut marks = [0f64; 4];
        let mut mark = |i: usize| -> Res<()> {
            if prof_on() {
                c.sync()?;
                marks[i] = t0.elapsed().as_secs_f64() * 1e3;
            }
            Ok(())
        };
        unsafe {
            cu::memcpy_htod_async(ap, &a[..n * n], c.stream).map_err(derr)?;
            cu::memcpy_htod_async(onep, &one_h[..], c.stream).map_err(derr)?;
            c.sync()?;
            mark(0)?;
            if !chol_upper_dev(c, ap, n, &mut blk)? {
                return Ok(Err(1));
            }
            mark(1)?;
            // V = I, then V ← U⁻¹ V
            cu::memset_d8_async(vp, 0, bytes, c.stream).map_err(derr)?;
            blas_ok(bl::cublasDcopy_v2(c.blas, n as i32, onep as *const f64, 0, vp as *mut f64, n as i32 + 1), "dcopy eye")?;
            let (one, zero) = (1.0f64, 0.0f64);
            blas_ok(
                bl::cublasDtrsm_v2(c.blas, Sd::CUBLAS_SIDE_LEFT, Fm::CUBLAS_FILL_MODE_UPPER, Op::CUBLAS_OP_N, Dg::CUBLAS_DIAG_NON_UNIT,
                    n as i32, n as i32, &one, ap as *const f64, n as i32, vp as *mut f64, n as i32),
                "dtrsm inverse",
            )?;
            // H⁻¹ = V Vᵀ into A's upper triangle
            blas_ok(
                bl::cublasDsyrk_v2(c.blas, Fm::CUBLAS_FILL_MODE_UPPER, Op::CUBLAS_OP_N, n as i32, n as i32, &one,
                    vp as *const f64, n as i32, &zero, ap as *mut f64, n as i32),
                "dsyrk inverse",
            )?;
            mark(2)?;
            if !chol_upper_dev(c, ap, n, &mut blk)? {
                return Ok(Err(2));
            }
            mark(3)?;
            let mut out = vec![0f64; n * n];
            cu::memcpy_dtoh_async(&mut out[..], ap, c.stream).map_err(derr)?;
            c.sync()?;
            if prof_on() {
                eprintln!(
                    "cuda gptq fold {n}: upload {:.0} ms, cholesky {:.0}, inverse {:.0}, cholesky {:.0}, readback {:.0} ms",
                    marks[0],
                    marks[1] - marks[0],
                    marks[2] - marks[1],
                    marks[3] - marks[2],
                    t0.elapsed().as_secs_f64() * 1e3 - marks[3]
                );
            }
            let _ = &one_h;
            Ok(Ok(out))
        }
    };
    match run() {
        Ok(r) => Some(r),
        Err(e) => {
            let _ = c.sync();
            decline("gptq fold", &e);
            None
        }
    }
}

/// SFq's code refinement of `rows` rows on the card: `code` (levels, a byte
/// a weight, rewritten), `step` (a weight's step, 0 frozen), `res` (Ŵ − W),
/// `hf` (`H + ridge·mean(diag H)·I`, f32), `live` (H_jj > 0). Returns the
/// moves and Σ r·g / Σ r² before and after, or None (the host's loop).
#[allow(clippy::too_many_arguments)]
pub fn gptq_refine_rows(
    code: &mut [u8],
    step: &[f32],
    res: &[f32],
    hf: &[f32],
    live: &[bool],
    rows: usize,
    cols: usize,
    maxc: u8,
    iters: usize,
) -> Option<(usize, [f64; 4])> {
    // three f32 rows and the levels in shared memory (48 KB without opting in)
    if !on() || rows == 0 || cols == 0 || 13 * cols > 48 << 10 || code.len() < rows * cols || step.len() < rows * cols || res.len() < rows * cols || hf.len() < cols * cols || live.len() < cols {
        return None;
    }
    let c = ctx()?;
    if c.current().is_err() {
        return None;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(|| GqState { f: None, scr: Default::default() });
    let f = match s.f.get_or_insert_with(|| compile(SRC, vec![], c.arch).and_then(|m| func(m, "gq_refine"))) {
        Ok(f) => *f,
        Err(e) => {
            decline("gptq refine", e);
            return None;
        }
    };
    let n = rows * cols;
    let hd: Vec<f32> = (0..cols).map(|j| hf[j * cols + j]).collect();
    let lv: Vec<u8> = live[..cols].iter().map(|&b| b as u8).collect();
    let mut run = || -> Res<(usize, [f64; 4])> {
        let rp = s.scr("gq_r", n * 4)?;
        let hp = s.scr("gq_h", cols * cols * 4)?;
        let gp = s.scr("gq_g", n * 4)?;
        let sp = s.scr("gq_s", n * 4)?;
        let cp = s.scr("gq_c", n)?;
        let dp = s.scr("gq_d", cols * 4)?;
        let lp = s.scr("gq_l", cols)?;
        let stp = s.scr("gq_st", rows * 4 * 8)?;
        let mp = s.scr("gq_m", rows * 4)?;
        let mut stats = vec![0f64; rows * 4];
        let mut moves = vec![0i32; rows];
        unsafe {
            cu::memcpy_htod_async(rp, &res[..n], c.stream).map_err(derr)?;
            cu::memcpy_htod_async(hp, &hf[..cols * cols], c.stream).map_err(derr)?;
            cu::memcpy_htod_async(sp, &step[..n], c.stream).map_err(derr)?;
            cu::memcpy_htod_async(cp, &code[..n], c.stream).map_err(derr)?;
            cu::memcpy_htod_async(dp, hd.as_slice(), c.stream).map_err(derr)?;
            cu::memcpy_htod_async(lp, lv.as_slice(), c.stream).map_err(derr)?;
            // the seed: G = R·H_f (H_f symmetric), plain f32
            c.gemm(true, false, cols, rows, cols, 1.0, hp, cols, rp, cols, 0.0, gp, cols, Gemm::F32)?;
            let (cols_i, maxc_i, iters_i) = (cols as i32, maxc as i32, iters as i32);
            c.launch(f, (rows as u32, 1), 256, 13 * cols, kargs!(cp, sp, rp, gp, hp, dp, lp, cols_i, maxc_i, iters_i, stp, mp))?;
            cu::memcpy_dtoh_async(&mut code[..n], cp, c.stream).map_err(derr)?;
            cu::memcpy_dtoh_async(stats.as_mut_slice(), stp, c.stream).map_err(derr)?;
            cu::memcpy_dtoh_async(moves.as_mut_slice(), mp, c.stream).map_err(derr)?;
        }
        c.sync()?;
        let mut tot = [0f64; 4];
        for r in 0..rows {
            for k in 0..4 {
                tot[k] += stats[r * 4 + k];
            }
        }
        Ok((moves.iter().map(|&m| m.max(0) as usize).sum(), tot))
    };
    match run() {
        Ok(v) => Some(v),
        Err(e) => {
            let _ = c.sync();
            decline("gptq refine", &e);
            None
        }
    }
}
