//! Binary low-rank (`blr`, dtype 19) runtime: factored ±1 matvec / matmat
//! straight from the mmap (docs/BLR_NVG.md §3). `W` is never materialised.
//!
//! Per token and path `p`:
//! ```text
//! z_p = x ⊙ g_p
//! t_p[k] = ℓ_p[k] · Σ_j V_p(k,j) · z_p[j]          stage V: P·r binary dots of length d_in
//! y[i]  = Σ_p h_p[i] · Σ_k U_p(i,k) · t_p[k]        stage U: d_out rows, P binary dots of length r
//! ```
//! A binary dot reads its signs four at a time through a 16-entry signed
//! partial-sum table per group of four inputs. Batches of `BLR_GEMM_MIN` or
//! more tokens unpack the signs to ±1 f32 and run two GEMMs.

use crate::pool::Pool;
use cortiq_core::CmfModel;
use cortiq_core::blr::{BlrHeader, blr_f16s, blr_path_offsets, validate_blr};
use std::sync::Arc;

/// Below this many tokens a batch loops the table matvec.
pub const BLR_GEMM_MIN: usize = 4;

/// Whether the blr kernels may take the GPU when the device is up
/// (`CMF_BLR_GPU=0` keeps them on the CPU).
pub fn gpu_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_BLR_GPU").as_deref() != Ok("0"))
}

/// A mapped `blr` tensor: the model keeps the bytes, `core` the parsed scales.
pub struct BlrTensor {
    pub model: Arc<CmfModel>,
    pub idx: usize,
    pub rows: usize,
    pub cols: usize,
    pub core: BlrCore,
}

impl BlrTensor {
    pub fn from_model(model: &Arc<CmfModel>, idx: usize) -> Result<Self, String> {
        let e = &model.tensors[idx];
        let core = BlrCore::parse(&e.shape, model.entry_bytes(e))
            .map_err(|m| format!("{}: {m}", e.name))?;
        Ok(BlrTensor {
            model: model.clone(),
            idx,
            rows: core.rows,
            cols: core.cols,
            core,
        })
    }

    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.model.entry_bytes(&self.model.tensors[self.idx])
    }

    pub fn name(&self) -> &str {
        &self.model.tensors[self.idx].name
    }

    pub fn matvec(&self, x: &[f32], out: &mut [f32], pool: Option<&Pool>) {
        self.core.matvec(self.bytes(), x, out, pool)
    }

    pub fn matmat(&self, xs: &[f32], b: usize, out: &mut [f32], pool: Option<&Pool>) {
        self.core.matmat(self.bytes(), xs, b, out, pool)
    }

    pub fn row_f32(&self, i: usize, dst: &mut [f32]) {
        self.core.row_f32(self.bytes(), i, dst)
    }

    pub fn add_col_scaled(&self, c: usize, w: f32, out: &mut [f32]) {
        self.core.add_col_scaled(self.bytes(), c, w, out)
    }

    /// Dense f32 copy (tests, parity).
    pub fn dequant(&self) -> Vec<f32> {
        let mut w = vec![0f32; self.rows * self.cols];
        cortiq_core::blr::dequant_blr(self.bytes(), self.rows, self.cols, &mut w)
            .expect("validated at load");
        w
    }
}

/// Parsed header + f32 scales of a `blr` payload; kernels take the bytes.
pub struct BlrCore {
    pub rows: usize,
    pub cols: usize,
    pub paths: usize,
    pub rank: usize,
    /// Per path: h (rows), g (cols), ℓ (rank), f32.
    pub h: Vec<Vec<f32>>,
    pub g: Vec<Vec<f32>>,
    pub l: Vec<Vec<f32>>,
    /// Per path: byte offsets of U (rows × r/8) and V (r × cols/8) in the payload.
    pub u_off: Vec<usize>,
    pub v_off: Vec<usize>,
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl BlrCore {
    pub fn parse(shape: &[usize], bytes: &[u8]) -> Result<Self, String> {
        let hdr: BlrHeader = validate_blr(shape, bytes)?;
        let mut t = BlrCore {
            rows: hdr.d_out,
            cols: hdr.d_in,
            paths: hdr.paths,
            rank: hdr.rank,
            h: Vec::new(),
            g: Vec::new(),
            l: Vec::new(),
            u_off: Vec::new(),
            v_off: Vec::new(),
        };
        for p in 0..hdr.paths {
            let o = blr_path_offsets(&hdr, p);
            t.h.push(blr_f16s(bytes, o.h, hdr.d_out));
            t.g.push(blr_f16s(bytes, o.g, hdr.d_in));
            t.l.push(blr_f16s(bytes, o.l, hdr.rank));
            t.u_off.push(o.u);
            t.v_off.push(o.v);
        }
        Ok(t)
    }

    /// Sign bits of path `p`: (U, V) byte slices.
    pub fn planes<'a>(&self, b: &'a [u8], p: usize) -> (&'a [u8], &'a [u8]) {
        let ub = self.rows * self.rank / 8;
        let vb = self.rank * self.cols / 8;
        (
            &b[self.u_off[p]..self.u_off[p] + ub],
            &b[self.v_off[p]..self.v_off[p] + vb],
        )
    }

    /// Single-vector product `out[..rows] = W x`.
    pub fn matvec(&self, bytes: &[u8], x: &[f32], out: &mut [f32], pool: Option<&Pool>) {
        let (rows, cols, r, np) = (self.rows, self.cols, self.rank, self.paths);
        assert!(
            x.len() >= cols && out.len() >= rows,
            "blr matvec {rows}x{cols}: x {} out {}",
            x.len(),
            out.len()
        );
        // stage V
        let mut tv = vec![0f32; np * cols * 4];
        let mut z = vec![0f32; cols];
        for p in 0..np {
            for j in 0..cols {
                z[j] = x[j] * self.g[p][j];
            }
            build_table(&z, &mut tv[p * cols * 4..(p + 1) * cols * 4]);
        }
        let mut t = vec![0f32; np * r];
        {
            let tp = SendPtr(t.as_mut_ptr());
            let tv = &tv;
            let run = |s: usize, e: usize| {
                let tp = tp;
                let rb = cols / 8;
                for row in s..e {
                    let (p, k) = (row / r, row % r);
                    let (_, vb) = self.planes(bytes, p);
                    let d = table_dot(
                        &vb[k * rb..(k + 1) * rb],
                        &tv[p * cols * 4..(p + 1) * cols * 4],
                    );
                    unsafe { *tp.0.add(row) = d * self.l[p][k] };
                }
            };
            dispatch(pool, np * r, np * r * cols, &run);
        }
        // stage U
        let mut tu = vec![0f32; np * r * 4];
        for p in 0..np {
            build_table(&t[p * r..(p + 1) * r], &mut tu[p * r * 4..(p + 1) * r * 4]);
        }
        let op = SendPtr(out.as_mut_ptr());
        let tu = &tu;
        let run = |s: usize, e: usize| {
            let op = op;
            let rb = r / 8;
            for i in s..e {
                let mut y = 0f32;
                for p in 0..np {
                    let (ub, _) = self.planes(bytes, p);
                    y += self.h[p][i]
                        * table_dot(&ub[i * rb..(i + 1) * rb], &tu[p * r * 4..(p + 1) * r * 4]);
                }
                unsafe { *op.0.add(i) = y };
            }
        };
        dispatch(pool, rows, rows * r * np, &run);
    }

    /// Batched product: `out[b·rows] = xs[b·cols] · Wᵀ`.
    pub fn matmat(&self, bytes: &[u8], xs: &[f32], b: usize, out: &mut [f32], pool: Option<&Pool>) {
        let (rows, cols, r, np) = (self.rows, self.cols, self.rank, self.paths);
        assert!(xs.len() >= b * cols && out.len() >= b * rows);
        if b < BLR_GEMM_MIN || std::env::var("CMF_BLR_GEMM").as_deref() == Ok("0") {
            for t in 0..b {
                self.matvec(
                    bytes,
                    &xs[t * cols..(t + 1) * cols],
                    &mut out[t * rows..(t + 1) * rows],
                    pool,
                );
            }
            return;
        }
        out[..b * rows].iter_mut().for_each(|v| *v = 0.0);
        let mut vf = vec![0f32; r * cols];
        let mut uf = vec![0f32; rows * r];
        let mut zp = vec![0f32; b * cols];
        let mut tp = vec![0f32; b * r];
        let mut yp = vec![0f32; b * rows];
        for p in 0..np {
            let (ub, vb) = self.planes(bytes, p);
            unpack_signs(vb, &mut vf, pool);
            unpack_signs(ub, &mut uf, pool);
            for t in 0..b {
                for j in 0..cols {
                    zp[t * cols + j] = xs[t * cols + j] * self.g[p][j];
                }
            }
            // T = Z · Vᵀ  (b × r), V stored [r, cols]
            par_sgemm_bt(b, r, cols, &zp, cols, &vf, cols, &mut tp, r, pool);
            for t in 0..b {
                for k in 0..r {
                    tp[t * r + k] *= self.l[p][k];
                }
            }
            // Y_p = T · Uᵀ (b × rows), U stored [rows, r]
            par_sgemm_bt(b, rows, r, &tp, r, &uf, r, &mut yp, rows, pool);
            for t in 0..b {
                let o = &mut out[t * rows..(t + 1) * rows];
                let y = &yp[t * rows..(t + 1) * rows];
                for i in 0..rows {
                    o[i] += y[i] * self.h[p][i];
                }
            }
        }
    }

    /// One row of `W` (embedding-style lookup; rare path).
    pub fn row_f32(&self, bytes: &[u8], i: usize, dst: &mut [f32]) {
        let (cols, r) = (self.cols, self.rank);
        dst[..cols].iter_mut().for_each(|v| *v = 0.0);
        for p in 0..self.paths {
            let (ub, vb) = self.planes(bytes, p);
            for k in 0..r {
                let su = (ub[i * (r / 8) + k / 8] >> (k % 8)) & 1 == 1;
                let c = self.h[p][i] * if su { self.l[p][k] } else { -self.l[p][k] };
                let row = &vb[k * (cols / 8)..(k + 1) * (cols / 8)];
                for j in 0..cols {
                    let s = (row[j / 8] >> (j % 8)) & 1 == 1;
                    let v = self.g[p][j] * c;
                    dst[j] += if s { v } else { -v };
                }
            }
        }
    }

    /// One column of `W`: `out[i] += w · W[i, c]`.
    pub fn add_col_scaled(&self, bytes: &[u8], c: usize, w: f32, out: &mut [f32]) {
        let (cols, r) = (self.cols, self.rank);
        for p in 0..self.paths {
            let (ub, vb) = self.planes(bytes, p);
            // t[k] = ℓ_k · V(k,c) · g_c · w
            let t: Vec<f32> = (0..r)
                .map(|k| {
                    let s = (vb[k * (cols / 8) + c / 8] >> (c % 8)) & 1 == 1;
                    let v = self.l[p][k] * self.g[p][c] * w;
                    if s { v } else { -v }
                })
                .collect();
            for i in 0..self.rows {
                let mut y = 0f32;
                for k in 0..r {
                    let s = (ub[i * (r / 8) + k / 8] >> (k % 8)) & 1 == 1;
                    y += if s { t[k] } else { -t[k] };
                }
                out[i] += self.h[p][i] * y;
            }
        }
    }
}

fn dispatch(pool: Option<&Pool>, rows: usize, work: usize, run: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) if rows >= 64 && work >= 65_536 => p.run_rows(rows, run),
        _ => run(0, rows),
    }
}

/// 16 signed partial sums per group of four inputs:
/// `T[16q + c] = Σ_{t<4} (bit t of c ? +1 : −1) · z[4q + t]`.
#[inline]
pub fn build_table(z: &[f32], t: &mut [f32]) {
    debug_assert_eq!(z.len() % 4, 0);
    debug_assert_eq!(t.len(), z.len() * 4);
    for q in 0..z.len() / 4 {
        let zz = [z[4 * q], z[4 * q + 1], z[4 * q + 2], z[4 * q + 3]];
        let tt = &mut t[16 * q..16 * q + 16];
        tt[0] = -(zz[0] + zz[1] + zz[2] + zz[3]);
        for c in 1..16usize {
            tt[c] = tt[c & (c - 1)] + 2.0 * zz[c.trailing_zeros() as usize];
        }
    }
}

/// Exact ±1 dot of one packed row (n bits) with the vector behind `t`.
#[inline]
pub fn table_dot(row: &[u8], t: &[f32]) -> f32 {
    // Four independent accumulators: the adds are a dependency chain otherwise.
    let mut acc = [0f32; 4];
    let n2 = row.len() / 2;
    for i in 0..n2 {
        let b0 = row[2 * i] as usize;
        let b1 = row[2 * i + 1] as usize;
        let base = 64 * i;
        acc[0] += t[base + (b0 & 15)];
        acc[1] += t[base + 16 + (b0 >> 4)];
        acc[2] += t[base + 32 + (b1 & 15)];
        acc[3] += t[base + 48 + (b1 >> 4)];
    }
    if row.len() % 2 == 1 {
        let i = row.len() - 1;
        let b0 = row[i] as usize;
        acc[0] += t[32 * i + (b0 & 15)];
        acc[1] += t[32 * i + 16 + (b0 >> 4)];
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// Packed sign rows → ±1.0 f32 (same row-major order, 8 values per byte).
fn unpack_signs(bits: &[u8], dst: &mut [f32], pool: Option<&Pool>) {
    debug_assert_eq!(dst.len(), bits.len() * 8);
    let n = bits.len();
    let dp = SendPtr(dst.as_mut_ptr());
    let run = |s: usize, e: usize| {
        let dp = dp;
        for i in s..e {
            let b = bits[i];
            for t in 0..8 {
                unsafe { *dp.0.add(8 * i + t) = if (b >> t) & 1 == 1 { 1.0 } else { -1.0 } };
            }
        }
    };
    dispatch(pool, n, n * 8, &run);
}

/// `C[m, n] = A[m, k] · B[n, k]ᵀ` (row-major, B stored by rows of length k),
/// split over the larger output dimension across the pool.
#[allow(clippy::too_many_arguments)]
pub fn par_sgemm_bt(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    bm: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    pool: Option<&Pool>,
) {
    assert!(a.len() >= (m - 1) * lda + k && bm.len() >= (n - 1) * ldb + k);
    assert!(c.len() >= (m - 1) * ldc + n);
    let cp = SendPtr(c.as_mut_ptr());
    let workers = pool.map(|p| p.n_workers() + 1).unwrap_or(1);
    let big = m * n * k >= 1 << 20;
    // Chunks of at least 32 so every block is a real GEMM (each re-packs
    // its operand panel).
    let split_m = m >= n;
    let d = if split_m { m } else { n };
    let cs = d.div_ceil(workers).max(32);
    let nch = d.div_ceil(cs);
    let run = |s: usize, e: usize| {
        let cp = cp;
        for ch in s..e {
            let (lo, hi) = (ch * cs, ((ch + 1) * cs).min(d));
            if hi <= lo {
                continue;
            }
            unsafe {
                if split_m {
                    gemm_block(
                        hi - lo,
                        n,
                        k,
                        a.as_ptr().add(lo * lda),
                        lda,
                        bm.as_ptr(),
                        ldb,
                        cp.0.add(lo * ldc),
                        ldc,
                    );
                } else {
                    gemm_block(
                        m,
                        hi - lo,
                        k,
                        a.as_ptr(),
                        lda,
                        bm.as_ptr().add(lo * ldb),
                        ldb,
                        cp.0.add(lo),
                        ldc,
                    );
                }
            }
        }
    };
    match pool {
        Some(p) if big && nch > 1 => p.run_rows(nch, &run),
        _ => run(0, nch),
    }
}

/// One GEMM block: `C[m, nb] = A[m, k] · B[nb, k]ᵀ`.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_block(
    m: usize,
    nb: usize,
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            nb,
            1.0,
            a,
            lda as isize,
            1,
            b,
            1,
            ldb as isize,
            0.0,
            c,
            ldc as isize,
            1,
        );
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cortiq_core::blr::{BlrPathIn, dequant_blr, encode_blr};

    pub(crate) struct Rng(pub u64);
    impl Rng {
        pub fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        pub fn f(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }
    }

    #[test]
    fn table_dot_is_exact_sign_dot() {
        let n = 40usize;
        let z: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.25)
            .collect();
        let bits: Vec<u8> = (0..n / 8).map(|i| (i * 89 + 13) as u8).collect();
        let mut t = vec![0f32; 4 * n];
        build_table(&z, &mut t);
        let want: f32 = (0..n)
            .map(|j| {
                if (bits[j / 8] >> (j % 8)) & 1 == 1 {
                    z[j]
                } else {
                    -z[j]
                }
            })
            .sum();
        assert!((table_dot(&bits, &t) - want).abs() < 1e-5);
    }

    /// A random `blr` payload (P paths of rank r) for kernel tests.
    pub(crate) fn make(rng: &mut Rng, d_out: usize, d_in: usize, r: usize, np: usize) -> Vec<u8> {
        let ps: Vec<_> = (0..np)
            .map(|_| {
                let h: Vec<f32> = (0..d_out).map(|_| 1.0 + rng.f()).collect();
                let g: Vec<f32> = (0..d_in).map(|_| 1.0 + rng.f()).collect();
                let l: Vec<f32> = (0..r).map(|_| 0.05 * (1.0 + rng.f())).collect();
                let su: Vec<bool> = (0..d_out * r).map(|_| rng.next() & 1 == 1).collect();
                let sv: Vec<bool> = (0..d_in * r).map(|_| rng.next() & 1 == 1).collect();
                (h, g, l, su, sv)
            })
            .collect();
        let ins: Vec<BlrPathIn> = ps
            .iter()
            .map(|(h, g, l, su, sv)| BlrPathIn { h, g, l, su, sv })
            .collect();
        encode_blr(d_out, d_in, r, &ins)
    }

    /// matvec and matmat (b = 1, 3, 4, 17) against dequant + dense, P = 1 and
    /// P = 2, r = 8 edge, with and without the pool.
    #[test]
    fn kernel_parity_with_dense() {
        let mut rng = Rng(11);
        let pool = Pool::new(3);
        for &(d_out, d_in, r, np) in &[
            (64usize, 96usize, 8usize, 1usize),
            (96, 64, 24, 2),
            (256, 512, 40, 2),
            (512, 256, 64, 1),
        ] {
            let bytes = make(&mut rng, d_out, d_in, r, np);
            let core = BlrCore::parse(&[d_out, d_in], &bytes).unwrap();
            let mut w = vec![0f32; d_out * d_in];
            dequant_blr(&bytes, d_out, d_in, &mut w).unwrap();
            for &b in &[1usize, 3, 4, 17] {
                let xs: Vec<f32> = (0..b * d_in).map(|_| rng.f()).collect();
                let mut want = vec![0f64; b * d_out];
                for t in 0..b {
                    for i in 0..d_out {
                        let mut s = 0f64;
                        for j in 0..d_in {
                            s += w[i * d_in + j] as f64 * xs[t * d_in + j] as f64;
                        }
                        want[t * d_out + i] = s;
                    }
                }
                let scale = want.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
                for pl in [None, Some(&pool)] {
                    let mut got = vec![0f32; b * d_out];
                    core.matmat(&bytes, &xs, b, &mut got, pl);
                    let err = got
                        .iter()
                        .zip(&want)
                        .map(|(g, w)| (*g as f64 - w).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    assert!(
                        err / scale < 1e-5,
                        "{d_out}x{d_in} r{r} P{np} b{b}: rel {}",
                        err / scale
                    );
                    if b == 1 {
                        let mut mv = vec![0f32; d_out];
                        core.matvec(&bytes, &xs, &mut mv, pl);
                        let e = mv
                            .iter()
                            .zip(&want)
                            .map(|(g, w)| (*g as f64 - w).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        assert!(e / scale < 1e-5);
                    }
                }
            }
            // rare paths: row and column reconstruction
            let mut row = vec![0f32; d_in];
            core.row_f32(&bytes, d_out / 2, &mut row);
            for j in 0..d_in {
                assert!((row[j] - w[(d_out / 2) * d_in + j]).abs() < 1e-4);
            }
            let mut col = vec![0f32; d_out];
            core.add_col_scaled(&bytes, d_in / 3, 2.0, &mut col);
            for i in 0..d_out {
                assert!((col[i] - 2.0 * w[i * d_in + d_in / 3]).abs() < 2e-4);
            }
        }
    }
}
