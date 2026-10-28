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
use super::train_sign;
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

// cvt_col into up to four outputs from one read of SRC (paths sharing an
// input): DST·COL, PL·A, PLT·B, DST4·OUT (rows = number of outputs); each
// value as cvt_col computes it.
@group(0) @binding(11) var<storage, read_write> DST4: array<f16>;
@compute @workgroup_size(256)
fn cvt_multi(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.mpad * p.k) { return; }
    let row = i / p.k;
    let col = i % p.k;
    var x = 0.0;
    if (row < p.m) { x = SRC[i]; }
    var v0 = 0.0;
    var v1 = 0.0;
    var v2 = 0.0;
    var v3 = 0.0;
    if (row < p.m) {
        v0 = x * COL[col] * p.scale;
        v1 = x * A[col] * p.scale;
        v2 = x * B[col] * p.scale;
        v3 = x * OUT[col] * p.scale;
    }
    DST[i] = f16(v0);
    if (p.rows > 1u) { PL[i] = f16(v1); }
    if (p.rows > 2u) { PLT[i] = f16(v2); }
    if (p.rows > 3u) { DST4[i] = f16(v3); }
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

// DST[mpad][k] = f16(SRC[m][words] columns r..r+k · scale), zero rows past m
// (a column block of a wider f32 matrix, e.g. one vocabulary slice).
@compute @workgroup_size(256)
fn cvt_cols(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.mpad * p.k) { return; }
    let row = i / p.k;
    var v = 0.0;
    if (row < p.m) { v = SRC[row * p.words + p.r + i % p.k] * p.scale; }
    DST[i] = f16(v);
}

// Transposed staging: DST[rows][mpad] = f16(SRC[t][row] · COL[row] (r = 1) ·
// scale) for t < m tokens, row < k; zero for padded tokens and rows.
// SRC is [m][k] f32; rows = pad_rows(k), mpad = the padded token count.
var<workgroup> tile: array<f32, 1056>;
@compute @workgroup_size(32, 8)
fn cvt_t(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let t0 = wg.x * 32u;
    let r0 = wg.y * 32u;
    for (var j = 0u; j < 32u; j = j + 8u) {
        let t = t0 + l.y + j;
        let row = r0 + l.x;
        var v = 0.0;
        if (t < p.m && row < p.k) {
            if (p.r == 1u) { v = SRC[t * p.k + row] * COL[row] * p.scale; } else { v = SRC[t * p.k + row] * p.scale; }
        }
        tile[(l.y + j) * 33u + l.x] = v;
    }
    workgroupBarrier();
    for (var j = 0u; j < 32u; j = j + 8u) {
        let row = r0 + l.y + j;
        let t = t0 + l.x;
        DST[row * p.mpad + t] = f16(tile[l.x * 33u + l.y + j]);
    }
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

// Only one orientation of the expansion (each GEMM phase reads one of each
// matrix): PL[rows][r] ...
@compute @workgroup_size(256)
fn expand_n(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.rows * p.r) { return; }
    let row = i / p.r;
    let j = i % p.r;
    let bit = (BITS[row * p.words + j / 32u] >> (j % 32u)) & 1u;
    PL[i] = select(f16(-1.0), f16(1.0), bit == 1u);
}

// ... or PLT[r][rows], consecutive threads along rows (coalesced stores).
@compute @workgroup_size(256)
fn expand_t(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.rows * p.r) { return; }
    let j = i / p.rows;
    let row = i % p.rows;
    let bit = (BITS[row * p.words + j / 32u] >> (j % 32u)) & 1u;
    PLT[i] = select(f16(-1.0), f16(1.0), bit == 1u);
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

// Two-pass column reduction, pass 1: SRC[blk][k] = Σ over the rows of block
// blk (r rows per block) of A·B (rows = 0), A·B·COL[row] (1) or A (2).
@compute @workgroup_size(256)
fn colsum_part(@builtin(global_invocation_id) g: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let col = g.x;
    if (col >= p.k) { return; }
    let r0 = wg.y * p.r;
    let r1 = min(r0 + p.r, p.m);
    var acc = 0.0;
    if (p.rows == 0u) {
        for (var row = r0; row < r1; row = row + 1u) { acc = acc + A[row * p.k + col] * B[row * p.k + col]; }
    } else if (p.rows == 1u) {
        for (var row = r0; row < r1; row = row + 1u) { acc = acc + A[row * p.k + col] * B[row * p.k + col] * COL[row]; }
    } else {
        for (var row = r0; row < r1; row = row + 1u) { acc = acc + A[row * p.k + col]; }
    }
    SRC[wg.y * p.k + col] = acc;
}

// colsum_part (A·B) fused with the elementwise op on one of its operands, so
// that operand is read once: rows = 0: DST = f16(A·COL[col]·scale) (cvt_col of
// A), 1: DST = f16(B·COL[col]·scale) (cvt_col of B), 2: OUT (+)= B·COL[col]
// (scale_acc of B, pad1 = 1: accumulate). Same partials, same element ops.
@compute @workgroup_size(256)
fn colsum_fused(@builtin(global_invocation_id) g: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let col = g.x;
    if (col >= p.k) { return; }
    let r0 = wg.y * p.r;
    let r1 = min(r0 + p.r, p.m);
    let cf = COL[col];
    var acc = 0.0;
    if (p.rows == 0u) {
        for (var row = r0; row < r1; row = row + 1u) {
            let i = row * p.k + col;
            let a = A[i];
            acc = acc + a * B[i];
            DST[i] = f16(a * cf * p.scale);
        }
    } else if (p.rows == 1u) {
        for (var row = r0; row < r1; row = row + 1u) {
            let i = row * p.k + col;
            let b = B[i];
            acc = acc + A[i] * b;
            DST[i] = f16(b * cf * p.scale);
        }
    } else {
        for (var row = r0; row < r1; row = row + 1u) {
            let i = row * p.k + col;
            let b = B[i];
            acc = acc + A[i] * b;
            let v = b * cf;
            if (p.pad1 == 1u) { OUT[i] = OUT[i] + v; } else { OUT[i] = v; }
        }
    }
    SRC[wg.y * p.k + col] = acc;
}

// colsum_fused that also writes the transposed f16 operand of the sign
// gradient GEMM (DSTT[col][kpad], kpad = mpad): rows = 0: the DST value
// (dy·h·gs), 1: the DST value (t2·l·gs), 2: f16(A·COL·f1) (x·g) — the values
// cvt_t would stage, from the operand this pass reads anyway. The 256×64
// transposed block goes through workgroup memory and leaves as coalesced
// 128-byte rows. Row blocks of exactly 64 rows.
@group(0) @binding(10) var<storage, read_write> DSTT: array<vec4<f16>>;
var<workgroup> ttile: array<f16, 16896>;   // [256 cols][66] (64 rows + pad)
@compute @workgroup_size(256)
fn colsum_fused_t(@builtin(global_invocation_id) g: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let col = g.x;
    let live = col < p.k;
    let r0 = wg.y * 64u;
    let r1 = min(r0 + 64u, p.m);
    var cf = 0.0;
    if (live) { cf = COL[col]; }
    var acc = 0.0;
    for (var row = r0; row < r1; row = row + 1u) {
        var tv = f16(0.0);
        if (live) {
            let i = row * p.k + col;
            if (p.rows == 0u) {
                let a = A[i]; acc = acc + a * B[i]; let hv = f16(a * cf * p.scale); DST[i] = hv; tv = hv;
            } else if (p.rows == 1u) {
                let b = B[i]; acc = acc + A[i] * b; let hv = f16(b * cf * p.scale); DST[i] = hv; tv = hv;
            } else {
                let a = A[i]; let b = B[i]; acc = acc + a * b; let v = b * cf;
                if (p.pad1 == 1u) { OUT[i] = OUT[i] + v; } else { OUT[i] = v; }
                tv = f16(a * cf * p.f1);
            }
        }
        ttile[t * 66u + (row - r0)] = tv;
    }
    if (live) { SRC[wg.y * p.k + col] = acc; }
    workgroupBarrier();
    // 256 columns × 16 vec4 (64 rows): 16 lanes per column row of DSTT
    let c0 = wg.x * 256u;
    for (var e = 0u; e < 16u; e = e + 1u) {
        let v = t + 256u * e;
        let cl = v / 16u;
        let q = v % 16u;
        if (c0 + cl < p.k && r0 + q * 4u < r1) {
            let b = cl * 66u + q * 4u;
            DSTT[((c0 + cl) * p.mpad + r0) / 4u + q] = vec4<f16>(ttile[b], ttile[b + 1u], ttile[b + 2u], ttile[b + 3u]);
        }
    }
}

// Pass 2: OUT[col] (+)= Σ_blk SRC[blk][col]  (words = blocks, pad0 = 1: accumulate).
@compute @workgroup_size(256)
fn colsum_fin(@builtin(global_invocation_id) g: vec3<u32>) {
    let col = g.x;
    if (col >= p.k) { return; }
    var acc = 0.0;
    for (var b = 0u; b < p.words; b = b + 1u) { acc = acc + SRC[b * p.k + col]; }
    if (p.pad0 == 1u) { OUT[col] = OUT[col] + acc; } else { OUT[col] = acc; }
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

// dense_fwd for r ≤ 48 outputs, tiled: a workgroup = 64 rows × all outputs,
// thread (row group ty: 4 rows, lane tx) owns outputs tx, tx+16, tx+32 of
// its 4 rows; k staged through workgroup memory in chunks of 64 (vec4
// loads). Each output still sums j = 0..k-1 in order with one fma per term,
// as dense_fwd's `acc + a·b` does.
var<workgroup> dta: array<f32, 4160>;   // [64][65]
var<workgroup> dtb: array<f32, 3120>;   // [48][65]
@compute @workgroup_size(256)
fn dense_fwd_t(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let r0 = wg.x * 64u;
    let ty = tid / 16u;
    let tx = tid % 16u;
    var a00 = 0.0;
    var a01 = 0.0;
    var a02 = 0.0;
    var a10 = 0.0;
    var a11 = 0.0;
    var a12 = 0.0;
    var a20 = 0.0;
    var a21 = 0.0;
    var a22 = 0.0;
    var a30 = 0.0;
    var a31 = 0.0;
    var a32 = 0.0;
    for (var k0 = 0u; k0 < p.k; k0 = k0 + 64u) {
        for (var e = tid; e < 1024u; e = e + 256u) {
            let rr = e / 16u;
            let kq = (e % 16u) * 4u;
            var v = vec4<f32>(0.0);
            if (r0 + rr < p.m && k0 + kq < p.k) {
                let b = (r0 + rr) * p.k + k0 + kq;
                v = vec4<f32>(A[b], A[b + 1u], A[b + 2u], A[b + 3u]);
            }
            let d = rr * 65u + kq;
            dta[d] = v.x; dta[d + 1u] = v.y; dta[d + 2u] = v.z; dta[d + 3u] = v.w;
        }
        for (var e = tid; e < 768u; e = e + 256u) {
            let o = e / 16u;
            let kq = (e % 16u) * 4u;
            var v = vec4<f32>(0.0);
            if (o < p.r && k0 + kq < p.k) {
                let b = o * p.k + k0 + kq;
                v = vec4<f32>(B[b], B[b + 1u], B[b + 2u], B[b + 3u]);
            }
            let d = o * 65u + kq;
            dtb[d] = v.x; dtb[d + 1u] = v.y; dtb[d + 2u] = v.z; dtb[d + 3u] = v.w;
        }
        workgroupBarrier();
        let kn = min(64u, p.k - k0);
        for (var kk = 0u; kk < kn; kk = kk + 1u) {
            let x0 = dta[(ty * 4u + 0u) * 65u + kk];
            let x1 = dta[(ty * 4u + 1u) * 65u + kk];
            let x2 = dta[(ty * 4u + 2u) * 65u + kk];
            let x3 = dta[(ty * 4u + 3u) * 65u + kk];
            let w0 = dtb[(tx + 0u) * 65u + kk];
            let w1 = dtb[(tx + 16u) * 65u + kk];
            let w2 = dtb[(tx + 32u) * 65u + kk];
            a00 = fma(x0, w0, a00);
            a01 = fma(x0, w1, a01);
            a02 = fma(x0, w2, a02);
            a10 = fma(x1, w0, a10);
            a11 = fma(x1, w1, a11);
            a12 = fma(x1, w2, a12);
            a20 = fma(x2, w0, a20);
            a21 = fma(x2, w1, a21);
            a22 = fma(x2, w2, a22);
            a30 = fma(x3, w0, a30);
            a31 = fma(x3, w1, a31);
            a32 = fma(x3, w2, a32);
        }
        workgroupBarrier();
    }
    if (r0 + ty * 4u + 0u < p.m && tx + 0u < p.r) {
        let i = (r0 + ty * 4u + 0u) * p.r + tx + 0u;
        if (p.rows == 1u) { OUT[i] = OUT[i] + a00; } else { OUT[i] = a00; }
    }
    if (r0 + ty * 4u + 0u < p.m && tx + 16u < p.r) {
        let i = (r0 + ty * 4u + 0u) * p.r + tx + 16u;
        if (p.rows == 1u) { OUT[i] = OUT[i] + a01; } else { OUT[i] = a01; }
    }
    if (r0 + ty * 4u + 0u < p.m && tx + 32u < p.r) {
        let i = (r0 + ty * 4u + 0u) * p.r + tx + 32u;
        if (p.rows == 1u) { OUT[i] = OUT[i] + a02; } else { OUT[i] = a02; }
    }
    if (r0 + ty * 4u + 1u < p.m && tx + 0u < p.r) {
        let i = (r0 + ty * 4u + 1u) * p.r + tx + 0u;
        if (p.rows == 1u) { OUT[i] = OUT[i] + a10; } else { OUT[i] = a10; }
    }
    if (r0 + ty * 4u + 1u < p.m && tx + 16u < p.r) {
        let i = (r0 + ty * 4u + 1u) * p.r + tx + 16u;
        if (p.rows == 1u) { OUT[i] = OUT[i] + a11; } else { OUT[i] = a11; }
    }
    if (r0 + ty * 4u + 1u < p.m && tx + 32u < p.r) {
        let i = (r0 + ty * 4u + 1u) * p.r + tx + 32u;
        if (p.rows == 1u) { OUT[i] = OUT[i] + a12; } else { OUT[i] = a12; }
    }
    if (r0 + ty * 4u + 2u < p.m && tx + 0u < p.r) {
        let i = (r0 + ty * 4u + 2u) * p.r + tx + 0u;
        if (p.rows == 1u) { OUT[i] = OUT[i] + a20; } else { OUT[i] = a20; }
    }
    if (r0 + ty * 4u + 2u < p.m && tx + 16u < p.r) {
        let i = (r0 + ty * 4u + 2u) * p.r + tx + 16u;
        if (p.rows == 1u) { OUT[i] = OUT[i] + a21; } else { OUT[i] = a21; }
    }
    if (r0 + ty * 4u + 2u < p.m && tx + 32u < p.r) {
        let i = (r0 + ty * 4u + 2u) * p.r + tx + 32u;
        if (p.rows == 1u) { OUT[i] = OUT[i] + a22; } else { OUT[i] = a22; }
    }
    if (r0 + ty * 4u + 3u < p.m && tx + 0u < p.r) {
        let i = (r0 + ty * 4u + 3u) * p.r + tx + 0u;
        if (p.rows == 1u) { OUT[i] = OUT[i] + a30; } else { OUT[i] = a30; }
    }
    if (r0 + ty * 4u + 3u < p.m && tx + 16u < p.r) {
        let i = (r0 + ty * 4u + 3u) * p.r + tx + 16u;
        if (p.rows == 1u) { OUT[i] = OUT[i] + a31; } else { OUT[i] = a31; }
    }
    if (r0 + ty * 4u + 3u < p.m && tx + 32u < p.r) {
        let i = (r0 + ty * 4u + 3u) * p.r + tx + 32u;
        if (p.rows == 1u) { OUT[i] = OUT[i] + a32; } else { OUT[i] = a32; }
    }
}

// dense_bwd with 4 rows per thread (each weight load serves 4 outputs);
// every output sums o = 0..r-1 in order, as dense_bwd does.
@compute @workgroup_size(256)
fn dense_bwd_t(@builtin(global_invocation_id) g: vec3<u32>) {
    let j = g.x;
    let row0 = g.y * 4u;
    if (j >= p.k) { return; }
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    let ok1 = row0 + 1u < p.m;
    let ok2 = row0 + 2u < p.m;
    let ok3 = row0 + 3u < p.m;
    let i1 = select(row0, row0 + 1u, ok1);
    let i2 = select(row0, row0 + 2u, ok2);
    let i3 = select(row0, row0 + 3u, ok3);
    for (var o = 0u; o < p.r; o = o + 1u) {
        let b = B[o * p.k + j];
        a0 = a0 + A[row0 * p.r + o] * b;
        a1 = a1 + A[i1 * p.r + o] * b;
        a2 = a2 + A[i2 * p.r + o] * b;
        a3 = a3 + A[i3 * p.r + o] * b;
    }
    OUT[row0 * p.k + j] = OUT[row0 * p.k + j] + a0;
    if (ok1) { OUT[i1 * p.k + j] = OUT[i1 * p.k + j] + a1; }
    if (ok2) { OUT[i2 * p.k + j] = OUT[i2 * p.k + j] + a2; }
    if (ok3) { OUT[i3 * p.k + j] = OUT[i3 * p.k + j] + a3; }
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

// adam over a concatenation of parameters (one dispatch for all of them):
// COL = per-element mask (0: frozen), BITS = per-element relative floor
// (f32 bits; 0: plain steps); the arithmetic of `adam` element for element.
@compute @workgroup_size(256)
fn adam_cat(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.k) { return; }
    if (COL[i] == 0.0) { return; }
    let gr = A[i] * bitcast<f32>(p.pad0);
    let m1 = p.f1 * B[i] + (1.0 - p.f1) * gr;
    let m2 = p.f2 * OUT[i] + (1.0 - p.f2) * gr * gr;
    B[i] = m1;
    OUT[i] = m2;
    let t = f32(p.m);
    let mh = m1 / (1.0 - pow(p.f1, t));
    let vh = m2 / (1.0 - pow(p.f2, t));
    let st = p.scale * mh / (sqrt(vh) + p.f3);
    let rf = bitcast<f32>(BITS[i]);
    if (rf > 0.0) {
        SRC[i] = SRC[i] - st * max(abs(SRC[i]), rf);
    } else {
        SRC[i] = SRC[i] - st;
    }
}

// `sumsq` for every parameter of a concatenation: workgroup (x < 64, y =
// parameter), BITS[2y], BITS[2y+1] = its offset and length; OUT[64y + x].
@compute @workgroup_size(256)
fn sumsq_cat(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let off = BITS[2u * wg.y];
    let n = BITS[2u * wg.y + 1u];
    var acc = 0.0;
    for (var i = wg.x * 256u + t; i < n; i = i + 256u * p.m) { acc = acc + A[off + i] * A[off + i]; }
    ssq[t] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { ssq[t] = ssq[t] + ssq[t + s]; }
        workgroupBarrier();
    }
    if (t == 0u) { OUT[wg.y * p.m + wg.x] = ssq[0]; }
}

// OUT = A·COL[k] + B·SRC[k]: the two paths of a blr linear in one pass,
// rounded as scale_acc twice (overwrite, then accumulate): fl(fl(a0c0) +
// fl(a1c1)). Each product passes through an xor with a uniform zero (words),
// so the driver cannot fuse either into the sum (left to itself it does,
// and which one depends on the surrounding code).
@compute @workgroup_size(256)
fn scale_acc2(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.m * p.k) { return; }
    let c = i % p.k;
    let v0 = bitcast<f32>(bitcast<u32>(A[i] * COL[c]) ^ p.words);
    let v1 = bitcast<f32>(bitcast<u32>(B[i] * SRC[c]) ^ p.words);
    OUT[i] = v0 + v1;
}

// OUT[i] = f32(DST[i]) for i < k (the embedding rows, uploaded as f16).
@compute @workgroup_size(256)
fn f16_to_f32(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.k) { return; }
    OUT[i] = f32(DST[i]);
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
    train_ops::module(c, "train_mm", SRC, &["cvt_col", "cvt", "cvt_cols", "cvt_t", "cvt_multi", "expand", "expand_n", "expand_t", "colsum_prod", "colsum_part", "colsum_fin", "colsum_fused", "colsum_fused_t", "scale_acc", "dense_fwd", "dense_bwd", "dense_fwd_t", "dense_bwd_t", "split_qg", "adam", "sumsq", "adam_cat", "sumsq_cat", "f16_to_f32", "scale_acc2"])
}

/// One step of a trainer pass: an elementwise/reduction dispatch or a GEMM.
pub enum Op<'a> {
    D(Disp<'a>),
    M(zi::MmCall),
}

/// Record every op into one compute pass and submit.
pub fn run(c: &Ctx, ops: &[Op]) {
    run_tagged(c, ops, usize::MAX, "");
}

/// [`run`], with the ops from index `from` on attributed to `tag` by the
/// profiler (`train_prof`); identical recording.
pub fn run_tagged(c: &Ctx, ops: &[Op], from: usize, tag: &'static str) {
    let prof = super::train_prof::on();
    let mut keep = Vec::new();
    let mut groups = Vec::new();
    for op in ops {
        if let Op::D(d) = op {
            let mut words = d.uni.clone();
            while words.len() % 4 != 0 {
                words.push(0);
            }
            if prof {
                use super::train_prof::{Cnt, count};
                count(Cnt::Buf, 1);
                count(Cnt::Write, 1);
                count(Cnt::WriteBytes, (words.len() * 4) as u64);
                count(Cnt::BindGroup, 1);
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
        let mut mk = super::train_prof::pass_begin(c, &mut ps);
        for (i, (op, bg)) in ops.iter().zip(&groups).enumerate() {
            match op {
                Op::D(d) => {
                    ps.set_pipeline(d.pipe);
                    ps.set_bind_group(0, bg.as_ref().unwrap(), &[]);
                    ps.dispatch_workgroups(d.grid.0, d.grid.1, d.grid.2);
                }
                Op::M(m) => m.record(&mut ps),
            }
            if prof {
                let (name, mnk) = match op {
                    Op::D(d) => (super::train_prof::pipe_name(d.pipe), [0; 3]),
                    Op::M(m) => ("mm", m.mnk),
                };
                let name = if i >= from { super::train_prof::tagged(tag, name) } else { name };
                super::train_prof::pass_mark(c, &mut ps, &mut mk, name, mnk);
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
    super::train_prof::count(super::train_prof::Cnt::Buf, 1);
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

/// [`cvt`] with column factors into up to four staging buffers from one read
/// of `src` (`cols[i]` → `dsts[i]`, 1 ≤ n ≤ 4).
#[allow(clippy::too_many_arguments)]
pub fn cvt_multi<'a>(c: &'a Ctx, src: &'a wgpu::Buffer, cols: &[&'a wgpu::Buffer], dsts: &[&'a wgpu::Buffer], m: usize, k: usize, scale: f32) -> Option<Disp<'a>> {
    let md = module(c)?;
    let n = cols.len();
    if n == 0 || n > 4 || dsts.len() != n {
        return None;
    }
    let mpad = pad_rows(m);
    let col = |i: usize| cols[i.min(n - 1)];
    let dst = |i: usize| dsts[i.min(n - 1)];
    let uni = vec![m as u32, k as u32, mpad as u32, n as u32, 0, 0, 0, 0, scale.to_bits(), 0, 0, 0];
    Some(Disp {
        pipe: &md.p["cvt_multi"],
        uni,
        bufs: vec![(1, src), (2, col(0)), (7, col(1)), (8, col(2)), (9, col(3)), (3, dst(0)), (5, dst(1)), (6, dst(2)), (11, dst(3))],
        grid: elt_fold(mpad * k),
    })
}

/// `dst` (f16 `[pad_rows(m)][k]`) = columns `col0..col0+k` of `src`
/// (f32 `[m][stride]`) · `scale`.
#[allow(clippy::too_many_arguments)]
pub fn cvt_cols<'a>(c: &'a Ctx, src: &'a wgpu::Buffer, dst: &'a wgpu::Buffer, m: usize, k: usize, stride: usize, col0: usize, scale: f32) -> Option<Disp<'a>> {
    let md = module(c)?;
    let mpad = pad_rows(m);
    let uni = vec![m as u32, k as u32, mpad as u32, 0, col0 as u32, stride as u32, 0, 0, scale.to_bits(), 0, 0, 0];
    Some(Disp { pipe: &md.p["cvt_cols"], uni, bufs: vec![(1, src), (3, dst)], grid: elt_fold(mpad * k) })
}

/// Transposed staging: `dst` (f16 `[pad_rows(k)][kpad]`, kpad = `ntok`
/// padded to 32) = `src` (f32 `[ntok][k]`) transposed, each source column
/// multiplied by `col[k]` (if given) and everything by `scale`; padded
/// tokens and rows are zero. Element for element the values [`cvt`] stages.
#[allow(clippy::too_many_arguments)]
pub fn cvt_t<'a>(c: &'a Ctx, src: &'a wgpu::Buffer, col: Option<&'a wgpu::Buffer>, dst: &'a wgpu::Buffer, ntok: usize, k: usize, scale: f32) -> Option<Disp<'a>> {
    let md = module(c)?;
    let kpad = ntok.div_ceil(32) * 32;
    let rows = pad_rows(k);
    // without a column factor COL aliases SRC (statically bound, never read)
    let uni = vec![ntok as u32, k as u32, kpad as u32, rows as u32, col.is_some() as u32, 0, 0, 0, scale.to_bits(), 0, 0, 0];
    Some(Disp { pipe: &md.p["cvt_t"], uni, bufs: vec![(1, src), (2, col.unwrap_or(src)), (3, dst)], grid: ((kpad / 32) as u32, (rows / 32) as u32, 1) })
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

/// Words of the shared partial-sum buffer of the column reductions.
const PART_WORDS: usize = 256 * 32768;

/// The column reductions' partial sums (one buffer: every op is ordered).
fn part_buf(c: &Ctx) -> &'static wgpu::Buffer {
    static PART: std::sync::OnceLock<wgpu::Buffer> = std::sync::OnceLock::new();
    PART.get_or_init(|| train_ops::zeros(c, PART_WORDS))
}

/// Term of a column reduction.
#[derive(Clone, Copy)]
pub enum ColTerm<'a> {
    /// a·b
    Prod(&'a wgpu::Buffer),
    /// a·b·f[row]
    ProdRow(&'a wgpu::Buffer, &'a wgpu::Buffer),
    /// a
    Plain,
}

/// Column reduction `out[k] (+)= Σ_rows term(a[row][k], …)` over `m` rows in
/// two passes (row blocks of ≥ 64 rows, at most 256 blocks, then the block
/// partials in order): latency-free where a thread-per-column loop over
/// thousands of rows ran on a handful of workgroups. Deterministic.
#[allow(clippy::too_many_arguments)]
pub fn colsum<'a>(c: &'a Ctx, a: &'a wgpu::Buffer, term: ColTerm<'a>, out: &'a wgpu::Buffer, m: usize, k: usize, acc: bool) -> Option<[Disp<'a>; 2]> {
    let md = module(c)?;
    let rpb = 64usize.max(m.div_ceil(256));
    let nblk = m.div_ceil(rpb).max(1);
    if nblk * k > PART_WORDS {
        return None;
    }
    let part = part_buf(c);
    let (mode, b, f) = match term {
        ColTerm::Prod(b) => (0u32, b, a),
        ColTerm::ProdRow(b, f) => (1, b, f),
        ColTerm::Plain => (2, a, a),
    };
    let p1 = Disp {
        pipe: &md.p["colsum_part"],
        uni: vec![m as u32, k as u32, 0, mode, rpb as u32, 0, 0, 0, 0, 0, 0, 0],
        bufs: vec![(1, part), (2, f), (7, a), (8, b)],
        grid: (k.div_ceil(256) as u32, nblk as u32, 1),
    };
    let p2 = Disp {
        pipe: &md.p["colsum_fin"],
        uni: vec![m as u32, k as u32, 0, 0, 0, nblk as u32, acc as u32, 0, 0, 0, 0, 0],
        bufs: vec![(1, part), (9, out)],
        grid: (k.div_ceil(256) as u32, 1, 1),
    };
    Some([p1, p2])
}

/// One orientation of [`expand`]: `pl` `[rows][r]` (`trans` false) or
/// `[r][rows]` (`trans` true).
pub fn expand1<'a>(c: &'a Ctx, bits: &'a wgpu::Buffer, pl: &'a wgpu::Buffer, rows: usize, r: usize, trans: bool) -> Option<Disp<'a>> {
    let md = module(c)?;
    let (ep, b) = if trans { ("expand_t", 6) } else { ("expand_n", 5) };
    Some(Disp {
        pipe: &md.p[ep],
        uni: vec![0, 0, 0, rows as u32, r as u32, r.div_ceil(32) as u32, 0, 0, 0, 0, 0, 0],
        bufs: vec![(4, bits), (b, pl)],
        grid: elt_fold(rows * r),
    })
}

/// The elementwise op fused into a column reduction's first pass.
#[derive(Clone, Copy)]
pub enum Fuse<'a> {
    /// `dst = f16(a·col·scale)` (the [`cvt`] of `a` with a column factor)
    CvtA(&'a wgpu::Buffer, &'a wgpu::Buffer, f32),
    /// `dst = f16(b·col·scale)`
    CvtB(&'a wgpu::Buffer, &'a wgpu::Buffer, f32),
    /// `out (+)= b·col` (the [`scale_acc`] of `b`)
    ScaleB(&'a wgpu::Buffer, &'a wgpu::Buffer, bool),
}

/// `sum[k] += Σ_rows a·b` fused with one elementwise op on `a` or `b`
/// (each operand read once). Same partials and element arithmetic as
/// [`colsum_prod`] followed by the separate op; `None` where the separate
/// ops must run (the f16 staging needs `m` = its padded row count).
#[allow(clippy::too_many_arguments)]
pub fn colsum_fused<'a>(c: &'a Ctx, a: &'a wgpu::Buffer, b: &'a wgpu::Buffer, sum: &'a wgpu::Buffer, m: usize, k: usize, op: Fuse<'a>) -> Option<[Disp<'a>; 2]> {
    colsum_fused_tr(c, a, b, sum, m, k, op, None)
}

/// [`colsum_fused`] that also writes the transposed f16 operand of the sign
/// gradient GEMM into `tr` = `[k][m]` (`m` tokens, a multiple of 32): the
/// staged value of `Fuse::CvtA`/`CvtB`, or `f16(a·col)` for `Fuse::ScaleB` —
/// what [`cvt_t`] would write.
#[allow(clippy::too_many_arguments)]
pub fn colsum_fused_tr<'a>(c: &'a Ctx, a: &'a wgpu::Buffer, b: &'a wgpu::Buffer, sum: &'a wgpu::Buffer, m: usize, k: usize, op: Fuse<'a>, tr: Option<&'a wgpu::Buffer>) -> Option<[Disp<'a>; 2]> {
    let md = module(c)?;
    if pad_rows(m) != m || (tr.is_some() && (m % 32 != 0 || 64usize.max(m.div_ceil(256)) != 64)) {
        return None;
    }
    let rpb = 64usize.max(m.div_ceil(256));
    let nblk = m.div_ceil(rpb).max(1);
    if nblk * k > PART_WORDS {
        return None;
    }
    let part = part_buf(c);
    // DST (f16) and OUT (f32) are both statically bound; the one the mode
    // does not write aliases the partials buffer
    let (mode, col, scale, accf, dst, out): (u32, &wgpu::Buffer, f32, u32, &wgpu::Buffer, &wgpu::Buffer) = match op {
        Fuse::CvtA(colv, dst, sc) => (0, colv, sc, 0, dst, part),
        Fuse::CvtB(colv, dst, sc) => (1, colv, sc, 0, dst, part),
        Fuse::ScaleB(colv, out, acc) => (2, colv, 1.0, acc as u32, part, out),
    };
    let p1 = match tr {
        None => Disp {
            pipe: &md.p["colsum_fused"],
            uni: vec![m as u32, k as u32, 0, mode, rpb as u32, 0, 0, accf, scale.to_bits(), 0, 0, 0],
            bufs: vec![(1, part), (2, col), (3, dst), (7, a), (8, b), (9, out)],
            grid: (k.div_ceil(256) as u32, nblk as u32, 1),
        },
        Some(t) => Disp {
            pipe: &md.p["colsum_fused_t"],
            uni: vec![m as u32, k as u32, m as u32, mode, rpb as u32, 0, 0, accf, scale.to_bits(), 1f32.to_bits(), 0, 0],
            bufs: vec![(1, part), (2, col), (3, dst), (7, a), (8, b), (9, out), (10, t)],
            grid: (k.div_ceil(256) as u32, nblk as u32, 1),
        },
    };
    let p2 = Disp {
        pipe: &md.p["colsum_fin"],
        uni: vec![m as u32, k as u32, 0, 0, 0, nblk as u32, 1, 0, 0, 0, 0, 0],
        bufs: vec![(1, part), (9, sum)],
        grid: (k.div_ceil(256) as u32, 1, 1),
    };
    Some([p1, p2])
}

/// `out[k] += Σ_rows a[row][k]·b[row][k]` (two dispatches, see [`colsum`]).
pub fn colsum_prod<'a>(c: &'a Ctx, a: &'a wgpu::Buffer, b: &'a wgpu::Buffer, out: &'a wgpu::Buffer, m: usize, k: usize) -> Option<[Disp<'a>; 2]> {
    colsum(c, a, ColTerm::Prod(b), out, m, k, true)
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

/// `out[i] = f32(src[i])` for `n` halves of `src`.
pub fn f16_to_f32<'a>(c: &'a Ctx, src: &'a wgpu::Buffer, out: &'a wgpu::Buffer, n: usize) -> Option<Disp<'a>> {
    let md = module(c)?;
    Some(Disp { pipe: &md.p["f16_to_f32"], uni: vec![0, n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], bufs: vec![(3, src), (9, out)], grid: elt_fold(n) })
}

/// `out = a0 ⊙ col0 + a1 ⊙ col1` per row: [`scale_acc`] of path 0 then path 1
/// (accumulating) in one pass, the same rounding.
#[allow(clippy::too_many_arguments)]
pub fn scale_acc2<'a>(c: &'a Ctx, a0: &'a wgpu::Buffer, col0: &'a wgpu::Buffer, a1: &'a wgpu::Buffer, col1: &'a wgpu::Buffer, out: &'a wgpu::Buffer, m: usize, k: usize) -> Option<Disp<'a>> {
    let md = module(c)?;
    Some(Disp {
        pipe: &md.p["scale_acc2"],
        uni: vec![m as u32, k as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        bufs: vec![(1, col1), (2, col0), (7, a0), (8, a1), (9, out)],
        grid: elt_fold(m * k),
    })
}

/// Small dense linear `out[m][r] (+)= a[m][k]·w[r][k]ᵀ` in f32.
#[allow(clippy::too_many_arguments)]
pub fn dense_fwd<'a>(c: &'a Ctx, a: &'a wgpu::Buffer, w: &'a wgpu::Buffer, out: &'a wgpu::Buffer, m: usize, k: usize, r: usize, acc: bool) -> Option<Disp<'a>> {
    let md = module(c)?;
    if r <= 48 && k % 4 == 0 && m.div_ceil(64) <= 65535 {
        return Some(Disp {
            pipe: &md.p["dense_fwd_t"],
            uni: vec![m as u32, k as u32, 0, acc as u32, r as u32, 0, 0, 0, 0, 0, 0, 0],
            bufs: vec![(7, a), (8, w), (9, out)],
            grid: (m.div_ceil(64) as u32, 1, 1),
        });
    }
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
    if m.div_ceil(4) <= 65535 {
        return Some(Disp {
            pipe: &md.p["dense_bwd_t"],
            uni: vec![m as u32, k as u32, 0, 0, r as u32, 0, 0, 0, 0, 0, 0, 0],
            bufs: vec![(7, dy), (8, w), (9, dx)],
            grid: (k.div_ceil(256) as u32, m.div_ceil(4) as u32, 1),
        });
    }
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

/// [`adam`] over a concatenation of `n` parameter entries (`mask`: 0 =
/// frozen; `relf`: per-entry relative floor as f32 bits, 0 = plain steps).
#[allow(clippy::too_many_arguments)]
pub fn adam_cat<'a>(c: &'a Ctx, w: &'a wgpu::Buffer, g: &'a wgpu::Buffer, m1: &'a wgpu::Buffer, m2: &'a wgpu::Buffer, mask: &'a wgpu::Buffer, relf: &'a wgpu::Buffer, n: usize, lr: f32, step: u32, b1: f32, b2: f32, eps: f32, gmul: f32) -> Option<Disp<'a>> {
    let md = module(c)?;
    Some(Disp {
        pipe: &md.p["adam_cat"],
        uni: vec![step, n as u32, 0, 0, 0, 0, gmul.to_bits(), 0, lr.to_bits(), b1.to_bits(), b2.to_bits(), eps.to_bits()],
        bufs: vec![(1, w), (2, mask), (4, relf), (7, g), (8, m1), (9, m2)],
        grid: elt_fold(n),
    })
}

/// [`sumsq`] partials of `np` parameters of a concatenation `a` (`tab`:
/// offset, length per parameter) into `out[64·np]`.
pub fn sumsq_cat<'a>(c: &'a Ctx, a: &'a wgpu::Buffer, tab: &'a wgpu::Buffer, out: &'a wgpu::Buffer, np: usize) -> Option<Disp<'a>> {
    let md = module(c)?;
    Some(Disp { pipe: &md.p["sumsq_cat"], uni: vec![64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], bufs: vec![(4, tab), (7, a), (9, out)], grid: (64, np as u32, 1) })
}

/// Record `ops` as one compute pass into `enc` (no submit).
pub fn record(c: &Ctx, enc: &mut wgpu::CommandEncoder, ops: &[Op]) {
    let prof = super::train_prof::on();
    let mut groups = Vec::new();
    for op in ops {
        if let Op::D(d) = op {
            let mut words = d.uni.clone();
            while words.len() % 4 != 0 {
                words.push(0);
            }
            if prof {
                use super::train_prof::{Cnt, count};
                count(Cnt::Buf, 1);
                count(Cnt::Write, 1);
                count(Cnt::WriteBytes, (words.len() * 4) as u64);
                count(Cnt::BindGroup, 1);
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
        } else {
            groups.push(None);
        }
    }
    let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
    let mut mk = super::train_prof::pass_begin(c, &mut ps);
    for (op, bg) in ops.iter().zip(&groups) {
        match op {
            Op::D(d) => {
                ps.set_pipeline(d.pipe);
                ps.set_bind_group(0, bg.as_ref().unwrap(), &[]);
                ps.dispatch_workgroups(d.grid.0, d.grid.1, d.grid.2);
            }
            Op::M(m) => m.record(&mut ps),
        }
        if prof {
            let (name, mnk) = match op {
                Op::D(d) => (super::train_prof::pipe_name(d.pipe), [0; 3]),
                Op::M(m) => ("mm", m.mnk),
            };
            super::train_prof::pass_mark(c, &mut ps, &mut mk, name, mnk);
        }
    }
}

/// Sum of squares of `n` floats of `a` (64 workgroup partials summed on the host).
pub fn sumsq(c: &Ctx, a: &wgpu::Buffer, n: usize) -> Option<f64> {
    let md = module(c)?;
    let out = train_ops::zeros(c, 64);
    let d = Disp { pipe: &md.p["sumsq"], uni: vec![64, n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], bufs: vec![(7, a), (9, &out)], grid: (64, 1, 1) };
    run(c, &[Op::D(d)]);
    Some(train_ops::download(c, &out, 64)?.iter().map(|&x| x as f64).sum())
}

/// Partial sums of squares of `a[..n]` into `out[0..64]` (one per workgroup).
pub fn sumsq_part<'a>(c: &'a Ctx, a: &'a wgpu::Buffer, n: usize, out: &'a wgpu::Buffer) -> Option<Disp<'a>> {
    let md = module(c)?;
    Some(Disp { pipe: &md.p["sumsq"], uni: vec![64, n as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], bufs: vec![(7, a), (9, out)], grid: (64, 1, 1) })
}

/// `out[m][n] = act[m][k] · plane[n][k]ᵀ · oscale` on the matrix units.
#[allow(clippy::too_many_arguments)]
pub fn mm(c: &Ctx, plane: &wgpu::Buffer, act: &wgpu::Buffer, out: &wgpu::Buffer, m: usize, n: usize, k: usize, oscale: f32) -> Option<zi::MmCall> {
    let g = zi::default_cfg(Epi::F32);
    let a = zi::MmArgs { m: m as u32, n: n as u32, k: k as u32, ldo: n as u32, ocol: 0, arow: 0, oscale, conv: [0; 3] };
    zi::mm_call(c, g, &a, plane, act, out)
}

/// Whether this device runs the trainer's matrix-unit GEMMs (cooperative
/// matrices, f16, 32-wide subgroups); without them every `mm` declines.
pub fn matrix_units() -> bool {
    zi::zctx().is_some()
}

/// [`mm`] with the plane given as a sign bitmap of `words` u32 per row,
/// expanded to ±1 inside the GEMM's staging (no f16 plane, no expand pass):
/// `trans` false = bits `[n][words]` (the plane's own rows), true = bits
/// `[k][words]` (the plane is the bitmap's transpose). Same products as
/// [`mm`] on the expanded plane. `None` where unavailable (`CMF_MM_BITS=0`).
#[allow(clippy::too_many_arguments)]
pub fn mm_bits(c: &Ctx, bits: &wgpu::Buffer, words: usize, trans: bool, act: &wgpu::Buffer, out: &wgpu::Buffer, m: usize, n: usize, k: usize, oscale: f32) -> Option<zi::MmCall> {
    if std::env::var("CMF_MM_BITS").as_deref() == Ok("0") {
        return None;
    }
    let g = zi::MmCfg { bits: if trans { 2 } else { 1 }, ..zi::default_cfg(Epi::F32) };
    let a = zi::MmArgs { m: m as u32, n: n as u32, k: k as u32, ldo: n as u32, ocol: 0, arow: 0, oscale, conv: [words as u32, 0, 0] };
    zi::mm_call(c, g, &a, bits, act, out)
}

/// [`mm`] into columns `ocol..ocol+n` of an output with row stride `ldo`.
#[allow(clippy::too_many_arguments)]
pub fn mm_cols(c: &Ctx, plane: &wgpu::Buffer, act: &wgpu::Buffer, out: &wgpu::Buffer, m: usize, n: usize, k: usize, ldo: usize, ocol: usize, oscale: f32) -> Option<zi::MmCall> {
    let g = zi::default_cfg(Epi::F32);
    let a = zi::MmArgs { m: m as u32, n: n as u32, k: k as u32, ldo: ldo as u32, ocol: ocol as u32, arow: 0, oscale, conv: [0; 3] };
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

/// `dw[m][k] = dy[ntok][m]ᵀ · x[ntok][k]` (overwritten): both operands
/// staged transposed in f16 (`xt`, `dyt`: `[rows][ceil32(ntok)]`; dy at the
/// gradient scale), the token axis as the GEMM's K. `k` a multiple of 128.
#[allow(clippy::too_many_arguments)]
pub fn weight_grad(c: &Ctx, x: &wgpu::Buffer, dy: &wgpu::Buffer, xt: &wgpu::Buffer, dyt: &wgpu::Buffer, dw: &wgpu::Buffer, ntok: usize, m: usize, k: usize, gscale: f32) -> Option<()> {
    let kp = ntok.div_ceil(32) * 32;
    let ops = vec![
        Op::D(cvt_t(c, x, None, xt, ntok, k, 1.0)?),
        Op::D(cvt_t(c, dy, None, dyt, ntok, m, gscale)?),
        Op::M(mm(c, xt, dyt, dw, m, k, kp, 1.0 / gscale)?),
    ];
    run(c, &ops);
    Some(())
}

/// `acc[k][k] += x[..ntok][k]ᵀ · x` (the input Gram of a linear): `x`
/// staged transposed in f16 (`xt`, rows past `ntok` zero), the product into
/// `tmp`, then added. `k` a multiple of 128.
#[allow(clippy::too_many_arguments)]
pub fn gram_acc(c: &Ctx, x: &wgpu::Buffer, xt: &wgpu::Buffer, acc: &wgpu::Buffer, tmp: &wgpu::Buffer, ntok: usize, k: usize) -> Option<()> {
    let kp = ntok.div_ceil(32) * 32;
    let ops = vec![
        Op::D(cvt_t(c, x, None, xt, ntok, k, 1.0)?),
        Op::M(mm(c, xt, xt, tmp, k, k, kp, 1.0)?),
        Op::D(train_ops::elt(c, "add", k * k, acc, tmp, Some(acc), None, None, None)?),
    ];
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
    /// Sign latents and optimizer state (sign training only).
    pub sign: Option<train_sign::SignPath>,
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
    pub(super) a: Vec<wgpu::Buffer>, // f32 [ntok][r] per path
    cbuf: Vec<wgpu::Buffer>, // f32 [ntok][m] per path (pre-h output)
    t1: wgpu::Buffer,   // f32 [ntok][max(m,n)]
    pub(super) t2: wgpu::Buffer, // f32 [ntok][r]
}

/// The transient buffers of the blr ops — each lives only inside one path's
/// op list — shared by the scratch of every role (one layer runs at a time).
/// A path's forward reads S_U and S_Vᵀ, its backward S_Uᵀ and S_V, so one
/// plane buffer per factor serves both orientations.
pub struct BlrShared {
    pu: wgpu::Buffer,
    pv: wgpu::Buffer,
    xh: wgpu::Buffer,
    t1: wgpu::Buffer,
    t2: wgpu::Buffer,
}

impl BlrShared {
    /// For roles of at most `mr` = m·r and `nr` = n·r plane entries, `mx` =
    /// max(m, n, r), `mn` = max(m, n) and rank `r`.
    pub fn new(c: &Ctx, ntok: usize, mr: usize, nr: usize, mx: usize, mn: usize, r: usize) -> Self {
        let pr = pad_rows(ntok);
        BlrShared {
            pu: f16_buf(c, mr),
            pv: f16_buf(c, nr),
            xh: f16_buf(c, pr * mx),
            t1: train_ops::zeros_u32(c, pr * mn),
            t2: train_ops::zeros_u32(c, pr * r),
        }
    }
}

impl BlrScratch {
    pub fn new(c: &Ctx, ntok: usize, m: usize, n: usize, r: usize, paths: usize) -> Self {
        let sh = BlrShared::new(c, ntok, m * r, n * r, m.max(n).max(r), m.max(n), r);
        Self::with_shared(c, ntok, m, r, paths, &sh)
    }

    /// A role's scratch: its own a and c per path, the transients from `sh`.
    pub fn with_shared(c: &Ctx, ntok: usize, m: usize, r: usize, paths: usize, sh: &BlrShared) -> Self {
        let z = |len: usize| train_ops::zeros_u32(c, len);
        BlrScratch {
            ntok,
            pl_u: sh.pu.clone(),
            plt_u: sh.pu.clone(),
            pl_v: sh.pv.clone(),
            plt_v: sh.pv.clone(),
            xh: sh.xh.clone(),
            a: (0..paths).map(|_| z(pad_rows(ntok) * r)).collect(),
            cbuf: (0..paths).map(|_| z(pad_rows(ntok) * m)).collect(),
            t1: sh.t1.clone(),
            t2: sh.t2.clone(),
        }
    }
}

/// Forward of `layer` on `x` (`[ntok][n]`) into `y` (`[ntok][m]`, overwritten).
/// Keeps a and c of every path in the scratch for [`blr_backward`].
pub fn blr_forward(c: &Ctx, layer: &Blr, s: &BlrScratch, x: &wgpu::Buffer, y: &wgpu::Buffer, ntok: usize) -> Option<()> {
    blr_forward_staged(c, layer, s, x, y, ntok, None)
}

/// [`blr_forward`] with each path's input staging `f16(x⊙g_p)` already done
/// (`xin[p]`, e.g. by one [`cvt_multi`] for every path reading `x`).
pub fn blr_forward_staged(c: &Ctx, layer: &Blr, s: &BlrScratch, x: &wgpu::Buffer, y: &wgpu::Buffer, ntok: usize, xin: Option<&[wgpu::Buffer]>) -> Option<()> {
    let (m, n, r) = (layer.m, layer.n, layer.r);
    let np = layer.paths.len();
    for (pi, p) in layer.paths.iter().enumerate() {
        let w = r / 32;
        let xs = xin.map(|v| &v[pi]);
        let mut ops = match (mm_bits(c, &p.bits_v, w, true, xs.unwrap_or(&s.xh), &s.a[pi], ntok, r, n, 1.0), mm_bits(c, &p.bits_u, w, false, &s.xh, &s.cbuf[pi], ntok, m, r, 1.0)) {
            // the GEMMs read the bitmaps themselves
            (Some(g1), Some(g2)) => {
                let mut v = Vec::with_capacity(5);
                if xs.is_none() {
                    v.push(Op::D(cvt(c, x, Some(&p.g), &s.xh, ntok, n, 1.0)?));
                }
                v.extend([Op::M(g1), Op::D(cvt(c, &s.a[pi], Some(&p.l), &s.xh, ntok, r, 1.0)?), Op::M(g2)]);
                v
            }
            // expanded planes (stages its own input into xh)
            _ => vec![
                Op::D(expand1(c, &p.bits_u, &s.pl_u, m, r, false)?),
                Op::D(expand1(c, &p.bits_v, &s.plt_v, n, r, true)?),
                Op::D(cvt(c, x, Some(&p.g), &s.xh, ntok, n, 1.0)?),
                Op::M(mm(c, &s.plt_v, &s.xh, &s.a[pi], ntok, r, n, 1.0)?),
                Op::D(cvt(c, &s.a[pi], Some(&p.l), &s.xh, ntok, r, 1.0)?),
                Op::M(mm(c, &s.pl_u, &s.xh, &s.cbuf[pi], ntok, m, r, 1.0)?),
            ],
        };
        // y = Σ_p h_p ⊙ c_p: a two-path linear in one pass after its second path
        if np != 2 {
            ops.push(Op::D(scale_acc(c, &s.cbuf[pi], &p.h, y, ntok, m, pi > 0)?));
        } else if pi == 1 {
            ops.push(Op::D(scale_acc2(c, &s.cbuf[0], &layer.paths[0].h, &s.cbuf[1], &p.h, y, ntok, m)?));
        }
        run(c, &ops);
    }
    Some(())
}

/// Backward of the last [`blr_forward`] on the same `x`: metric gradients
/// accumulate into each path's dh/dℓ/dg; `dx` (`[ntok][n]`) is overwritten
/// with the input gradient. `gscale` is the power of two the gradients are
/// staged under in f16.
///
/// With `sign` (and sign state on the path), the sign gradients follow in
/// the same pass, then each matrix's latent/bitmap update (see
/// `train_sign`): G_U = diag(h)·dyᵀ·a·diag(ℓ) from the transposed operands
/// `dy⊙h·gs` and `a⊙ℓ`, then G_V = diag(g)·xᵀ·b·diag(ℓ) from `x⊙g` and
/// `b⊙ℓ·gs` (b = t2, overwritten by the next path: G_V stays inside this
/// path's op list). The input gradient was already formed from the old bits.
#[allow(clippy::too_many_arguments)]
pub fn blr_backward(c: &Ctx, layer: &Blr, s: &BlrScratch, x: &wgpu::Buffer, dy: &wgpu::Buffer, dx: &wgpu::Buffer, ntok: usize, gscale: f32, sign: Option<&train_sign::SignCtx>) -> Option<()> {
    let (m, n, r) = (layer.m, layer.n, layer.r);
    for (pi, p) in layer.paths.iter().enumerate() {
        let w = r / 32;
        let bits_mm = (mm_bits(c, &p.bits_u, w, true, &s.xh, &s.t2, ntok, r, m, 1.0 / gscale), mm_bits(c, &p.bits_v, w, false, &s.xh, &s.t1, ntok, n, r, 1.0 / gscale));
        let (g_t2, g_t1) = match bits_mm {
            (Some(a), Some(b)) => (a, b),
            _ => (mm(c, &s.plt_u, &s.xh, &s.t2, ntok, r, m, 1.0 / gscale)?, mm(c, &s.pl_v, &s.xh, &s.t1, ntok, n, r, 1.0 / gscale)?),
        };
        let mut ops = Vec::new();
        if !g_t2.reads_bits() {
            ops.push(Op::D(expand1(c, &p.bits_u, &s.plt_u, m, r, true)?));
            ops.push(Op::D(expand1(c, &p.bits_v, &s.pl_v, n, r, false)?));
        }
        // each column reduction fused with the elementwise op that reads the
        // same operand (dy, t2, t1 read once); with sign training the same
        // passes also write the transposed f16 operands of the sign
        // gradients (dy·h·gs, t2·l·gs, x·g); separate ops where they cannot
        let sgn = match (sign, p.sign.as_ref()) {
            (Some(ctx), Some(_)) => Some(ctx.sc),
            _ => None,
        };
        let (mut tr_dy, mut tr_t2, mut tr_x) = (false, false, false);
        match colsum_fused_tr(c, dy, &s.cbuf[pi], &p.dh, ntok, m, Fuse::CvtA(&p.h, &s.xh, gscale), sgn.map(|sc| &sc.t_big)) {
            Some([a, b]) => {
                tr_dy = sgn.is_some();
                ops.extend([Op::D(a), Op::D(b)]);
            }
            None => match colsum_fused(c, dy, &s.cbuf[pi], &p.dh, ntok, m, Fuse::CvtA(&p.h, &s.xh, gscale)) {
                Some([a, b]) => ops.extend([Op::D(a), Op::D(b)]),
                None => {
                    let [a, b] = colsum_prod(c, dy, &s.cbuf[pi], &p.dh, ntok, m)?;
                    ops.extend([Op::D(a), Op::D(b), Op::D(cvt(c, dy, Some(&p.h), &s.xh, ntok, m, gscale)?)]);
                }
            },
        }
        ops.push(Op::M(g_t2));
        match colsum_fused_tr(c, &s.a[pi], &s.t2, &p.dl, ntok, r, Fuse::CvtB(&p.l, &s.xh, gscale), sgn.map(|sc| &sc.t_small2)) {
            Some([a, b]) => {
                tr_t2 = sgn.is_some();
                ops.extend([Op::D(a), Op::D(b)]);
            }
            None => match colsum_fused(c, &s.a[pi], &s.t2, &p.dl, ntok, r, Fuse::CvtB(&p.l, &s.xh, gscale)) {
                Some([a, b]) => ops.extend([Op::D(a), Op::D(b)]),
                None => {
                    let [a, b] = colsum_prod(c, &s.a[pi], &s.t2, &p.dl, ntok, r)?;
                    ops.extend([Op::D(a), Op::D(b), Op::D(cvt(c, &s.t2, Some(&p.l), &s.xh, ntok, r, gscale)?)]);
                }
            },
        }
        ops.push(Op::M(g_t1));
        match colsum_fused_tr(c, x, &s.t1, &p.dg, ntok, n, Fuse::ScaleB(&p.g, dx, pi > 0), sgn.map(|sc| &sc.t_big2)) {
            Some([a, b]) => {
                tr_x = sgn.is_some();
                ops.extend([Op::D(a), Op::D(b)]);
            }
            None => match colsum_fused(c, x, &s.t1, &p.dg, ntok, n, Fuse::ScaleB(&p.g, dx, pi > 0)) {
                Some([a, b]) => ops.extend([Op::D(a), Op::D(b)]),
                None => {
                    let [a, b] = colsum_prod(c, x, &s.t1, &p.dg, ntok, n)?;
                    ops.extend([Op::D(a), Op::D(b), Op::D(scale_acc(c, &s.t1, &p.g, dx, ntok, n, pi > 0)?)]);
                }
            },
        }
        let sign_from = ops.len();
        if let (Some(ctx), Some(sp)) = (sign, p.sign.as_ref()) {
            let sc = ctx.sc;
            let kpad = sc.kpad;
            if !tr_dy {
                ops.push(Op::D(cvt_t(c, dy, Some(&p.h), &sc.t_big, ntok, m, gscale)?));
            }
            ops.push(Op::D(cvt_t(c, &s.a[pi], Some(&p.l), &sc.t_small, ntok, r, 1.0)?));
            ops.push(Op::M(mm(c, &sc.t_small, &sc.t_big, &sc.g, m, r, kpad, 1.0 / gscale)?));
            ops.extend(train_sign::grad_consumer_ops(c, ctx, &sp.u, &p.bits_u, 0)?);
            if !tr_x {
                ops.push(Op::D(cvt_t(c, x, Some(&p.g), &sc.t_big2, ntok, n, 1.0)?));
            }
            if !tr_t2 {
                ops.push(Op::D(cvt_t(c, &s.t2, Some(&p.l), &sc.t_small2, ntok, r, gscale)?));
            }
            ops.push(Op::M(mm(c, &sc.t_small2, &sc.t_big2, &sc.g, n, r, kpad, 1.0 / gscale)?));
            ops.extend(train_sign::grad_consumer_ops(c, ctx, &sp.v, &p.bits_v, 1)?);
        }
        run_tagged(c, &ops, sign_from, "sign");
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

    /// The one-pass two-path scale gives the bits of scale_acc twice.
    #[test]
    fn scale_acc2_matches_two_passes() {
        let Some(c) = train_ops::dev() else { return };
        let (m, k) = (64usize, 1024usize);
        let a0 = rnd(m * k, 1, 1.0);
        let a1 = rnd(m * k, 2, 1.0);
        let c0 = rnd(k, 3, 1.0);
        let c1 = rnd(k, 4, 1.0);
        let (ba0, ba1, bc0, bc1) = (upload(c, &a0), upload(c, &a1), upload(c, &c0), upload(c, &c1));
        let y = zeros(c, m * k);
        run(c, &[Op::D(scale_acc(c, &ba0, &bc0, &y, m, k, false).unwrap()), Op::D(scale_acc(c, &ba1, &bc1, &y, m, k, true).unwrap())]);
        let want = download(c, &y, m * k).unwrap();
        let out = zeros(c, m * k);
        run(c, &[Op::D(scale_acc2(c, &ba0, &bc0, &ba1, &bc1, &out, m, k).unwrap())]);
        let got = download(c, &out, m * k).unwrap();
        let diff = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(diff, 0, "{diff} of {} differ", m * k);
    }

    /// Two-path blr layer forward and backward against an f64 reference
    /// (tolerance set by the f16 operands: ~1e-3 relative).
    #[test]
    fn blr_layer_matches_host() {
        let Some(c) = train_ops::dev() else { return };
        if !matrix_units() {
            eprintln!("no cooperative matrices on this device — skipping");
            return;
        }
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
            paths.push(BlrPath { bits_u: ub(&bu), bits_v: ub(&bv), h: upload(c, &h), l: upload(c, &l), g: upload(c, &g), dh: zeros(c, m), dl: zeros(c, r), dg: zeros(c, n), sign: None });
            paths_h.push((su, sv, h, l, g));
        }
        let layer = Blr { m, n, r, paths };
        let x = rnd(ntok * n, 60, 2.0);
        let dy = rnd(ntok * m, 61, 1e-3);
        let s = BlrScratch::new(c, ntok, m, n, r, np);
        let (bx, by, bdy, bdx) = (upload(c, &x), zeros(c, pad_rows(ntok) * m), upload(c, &dy), zeros(c, pad_rows(ntok) * n));
        blr_forward(c, &layer, &s, &bx, &by, ntok).expect("forward");
        blr_backward(c, &layer, &s, &bx, &bdy, &bdx, ntok, 1024.0, None).expect("backward");
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

    fn pack_bits(s: &[f32], rows: usize, r: usize) -> Vec<u32> {
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
    }

    fn ubuf(c: &Ctx, v: &[u32]) -> wgpu::Buffer {
        let b = train_ops::zeros_u32(c, v.len());
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(v));
        b
    }

    fn f16_bits(c: &Ctx, b: &wgpu::Buffer, n: usize) -> Vec<u16> {
        let w = train_ops::download_u32(c, b, n.div_ceil(2)).unwrap();
        let mut out: Vec<u16> = w.iter().flat_map(|&x| [x as u16, (x >> 16) as u16]).collect();
        out.truncate(n);
        out
    }

    /// `cvt_t` is the transpose of `cvt`, bit for bit in f16, padded rows
    /// and tokens zero (ntok 70 → 96 padded tokens, k 200 → 256 rows).
    #[test]
    fn cvt_t_matches_cvt() {
        let Some(c) = train_ops::dev() else { return };
        let (ntok, k) = (70usize, 200usize);
        let kpad = 96usize;
        let src = upload(c, &rnd(ntok * k, 3, 3.0));
        let col = upload(c, &rnd(k, 4, 2.0));
        for (cb, scale) in [(Some(&col), 1024.0f32), (None, 0.5)] {
            let a = f16_buf(c, pad_rows(ntok) * k);
            let t = f16_buf(c, pad_rows(k) * kpad);
            // poison the transposed buffer so unwritten pads would show
            c.queue.write_buffer(&t, 0, bytemuck::cast_slice(&vec![0x3c00_3c00u32; (pad_rows(k) * kpad).div_ceil(2)]));
            run(c, &[Op::D(cvt(c, &src, cb, &a, ntok, k, scale).unwrap()), Op::D(cvt_t(c, &src, cb, &t, ntok, k, scale).unwrap())]);
            let av = f16_bits(c, &a, pad_rows(ntok) * k);
            let tv = f16_bits(c, &t, pad_rows(k) * kpad);
            let mut bad = 0;
            for row in 0..pad_rows(k) {
                for tok in 0..kpad {
                    let want = if row < k && tok < ntok { av[tok * k + row] } else { 0 };
                    if tv[row * kpad + tok] != want {
                        bad += 1;
                    }
                }
            }
            eprintln!("cvt_t (col {}): {bad} mismatches of {}", cb.is_some(), pad_rows(k) * kpad);
            assert_eq!(bad, 0);
        }
    }

    struct SignLayer {
        layer: Blr,
        host: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>,
    }

    /// A blr layer with sign state on every path (deferred: the gradients are kept).
    #[allow(clippy::too_many_arguments)]
    fn sign_layer(c: &Ctx, m: usize, n: usize, r: usize, np: usize, exact: bool) -> SignLayer {
        let sgn = |v: &f32| if *v >= 0.0 { 1.0 } else { -1.0 };
        let mut host = Vec::new();
        let mut paths = Vec::new();
        let mut id = 0u32;
        for pi in 0..np {
            let su: Vec<f32> = rnd(m * r, 10 + pi as u64, 1.0).iter().map(sgn).collect();
            let sv: Vec<f32> = rnd(n * r, 20 + pi as u64, 1.0).iter().map(sgn).collect();
            let h: Vec<f32> = rnd(m, 30 + pi as u64, 0.5).iter().map(|v| 0.02 + v.abs() * 0.05).collect();
            // exact: ℓ and g powers of two, so every f16 operand of the forward is exact
            let l: Vec<f32> = if exact {
                rnd(r, 40 + pi as u64, 1.0).iter().map(|v| [0.5f32, 1.0, 2.0][((v.abs() * 2.999) as usize).min(2)]).collect()
            } else {
                rnd(r, 40 + pi as u64, 0.5).iter().map(|v| 0.5 + v.abs()).collect()
            };
            let g: Vec<f32> = if exact {
                rnd(n, 50 + pi as u64, 1.0).iter().map(|v| if *v >= 0.0 { 1.0 } else { 0.5 }).collect()
            } else {
                rnd(n, 50 + pi as u64, 0.5).iter().map(|v| 0.1 + v.abs() * 0.2).collect()
            };
            let bu = ubuf(c, &pack_bits(&su, m, r));
            let bv = ubuf(c, &pack_bits(&sv, n, r));
            let init = train_sign::SignInit { opt: train_sign::SignOpt::Fact, tau: 0.25, seed: 1, keep_bits0: false, defer: true };
            let u = train_sign::init_mat(c, &bu, m, r, r, &init, id).unwrap();
            let v = train_sign::init_mat(c, &bv, n, r, r, &init, id + 1).unwrap();
            id += 2;
            paths.push(BlrPath {
                bits_u: bu,
                bits_v: bv,
                h: upload(c, &h),
                l: upload(c, &l),
                g: upload(c, &g),
                dh: zeros(c, m),
                dl: zeros(c, r),
                dg: zeros(c, n),
                sign: Some(train_sign::SignPath { u, v }),
            });
            host.push((su, sv, h, l, g));
        }
        SignLayer { layer: Blr { m, n, r, paths }, host }
    }

    fn defer_cfg() -> train_sign::SignStepCfg {
        train_sign::SignStepCfg { opt: train_sign::SignOpt::Fact, eta: 0.0, step: 1, b1: 0.9, b2: 0.999, eps: 1e-3, clamp: 4.0, hyst: 0.0, k_ss: 8.0, seed: 1, defer: true }
    }

    /// f64 sign gradients of a layer: per path (G_U [m][r], G_V [n][r], a [ntok][r]).
    #[allow(clippy::type_complexity)]
    fn host_sign_grads(sl: &SignLayer, x: &[f32], dy: &[f32], ntok: usize) -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let Blr { m, n, r, .. } = sl.layer;
        sl.host
            .iter()
            .map(|(su, sv, h, l, g)| {
                let mut a = vec![0f64; ntok * r];
                let mut b = vec![0f64; ntok * r];
                for t in 0..ntok {
                    for k in 0..r {
                        a[t * r + k] = (0..n).map(|j| x[t * n + j] as f64 * g[j] as f64 * sv[j * r + k] as f64).sum();
                        b[t * r + k] = (0..m).map(|i| dy[t * m + i] as f64 * h[i] as f64 * su[i * r + k] as f64).sum();
                    }
                }
                let mut gu = vec![0f64; m * r];
                let mut gv = vec![0f64; n * r];
                for k in 0..r {
                    for i in 0..m {
                        gu[i * r + k] = h[i] as f64 * l[k] as f64 * (0..ntok).map(|t| dy[t * m + i] as f64 * a[t * r + k]).sum::<f64>();
                    }
                    for j in 0..n {
                        gv[j * r + k] = g[j] as f64 * l[k] as f64 * (0..ntok).map(|t| x[t * n + j] as f64 * b[t * r + k]).sum::<f64>();
                    }
                }
                (gu, gv, a)
            })
            .collect()
    }

    /// The device sign gradients G_U, G_V of a two-path layer (formed inside
    /// `blr_backward`, kept by the deferred mode) against f64.
    #[test]
    fn blr_sign_grad_matches_host() {
        let Some(c) = train_ops::dev() else { return };
        if !matrix_units() {
            eprintln!("no cooperative matrices on this device — skipping");
            return;
        }
        let (m, n, r, ntok, np) = (256usize, 384usize, 128usize, 70usize, 2usize);
        let sl = sign_layer(c, m, n, r, np, false);
        let x = rnd(ntok * n, 60, 2.0);
        let dy = rnd(ntok * m, 61, 1e-3);
        let s = BlrScratch::new(c, ntok, m, n, r, np);
        let sc = train_sign::SignScratch::new(c, ntok, m.max(n), r);
        let cfg = defer_cfg();
        let ctx = train_sign::SignCtx { sc: &sc, cfg: &cfg, role: 0 };
        let (bx, by, bdy, bdx) = (upload(c, &x), zeros(c, pad_rows(ntok) * m), upload(c, &dy), zeros(c, pad_rows(ntok) * n));
        blr_forward(c, &sl.layer, &s, &bx, &by, ntok).unwrap();
        blr_backward(c, &sl.layer, &s, &bx, &bdy, &bdx, ntok, 1024.0, Some(&ctx)).unwrap();
        let want = host_sign_grads(&sl, &x, &dy, ntok);
        for (pi, (gu, gv, _)) in want.iter().enumerate() {
            let sp = sl.layer.paths[pi].sign.as_ref().unwrap();
            for (name, mat, w) in [("G_U", &sp.u, gu), ("G_V", &sp.v, gv)] {
                let got = download(c, mat.gsave.as_ref().unwrap(), mat.rows * r).unwrap();
                let num: f64 = got.iter().zip(w).map(|(g, w)| (*g as f64 - w).powi(2)).sum();
                let den: f64 = w.iter().map(|w| w * w).sum::<f64>().max(1e-30);
                let e = (num / den).sqrt();
                eprintln!("{name}{pi}: relative error {e:e}");
                assert!(e < 3e-3, "{name}{pi}: relative error {e:e}");
            }
        }
    }

    /// Discrete flips on the device: with L = ½‖y − y*‖², L is quadratic in
    /// each sign, so flipping U[i,k] changes it by exactly
    /// −2s·G_U + 2h_i²ℓ_k²·Σ_t a_tk², and V[j,k] by
    /// −2s·G_V + 2g_j²ℓ_k²·(Σ_t x_tj²)(Σ_i h_i²). Inputs on a grid that keeps
    /// every f16 operand of the forward exact; each flip rewrites the bitmap
    /// and reruns the device forward (this also checks the bit mapping).
    #[test]
    fn blr_sign_flip_exact() {
        let Some(c) = train_ops::dev() else { return };
        if !matrix_units() {
            eprintln!("no cooperative matrices on this device — skipping");
            return;
        }
        let (m, n, r, ntok, np) = (256usize, 384usize, 128usize, 70usize, 2usize);
        let sl = sign_layer(c, m, n, r, np, true);
        let x: Vec<f32> = rnd(ntok * n, 70, 1.0).iter().map(|v| ((v * 4.0).round()) / 8.0).collect();
        let s = BlrScratch::new(c, ntok, m, n, r, np);
        let sc = train_sign::SignScratch::new(c, ntok, m.max(n), r);
        let (bx, by, bdx) = (upload(c, &x), zeros(c, pad_rows(ntok) * m), zeros(c, pad_rows(ntok) * n));
        blr_forward(c, &sl.layer, &s, &bx, &by, ntok).unwrap();
        let y0 = download(c, &by, ntok * m).unwrap();
        let rms = (y0.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / y0.len() as f64).sqrt() as f32;
        let ystar: Vec<f32> = y0.iter().zip(rnd(ntok * m, 71, 1.0)).map(|(y, z)| y + 0.7 * rms * z).collect();
        let dy: Vec<f32> = y0.iter().zip(&ystar).map(|(y, s)| y - s).collect();
        let loss = |y: &[f32]| -> Vec<f64> { y.iter().zip(&ystar).map(|(a, b)| 0.5 * (*a as f64 - *b as f64).powi(2)).collect() };
        let l0 = loss(&y0);
        let bdy = upload(c, &dy);
        let cfg = defer_cfg();
        let ctx = train_sign::SignCtx { sc: &sc, cfg: &cfg, role: 0 };
        blr_backward(c, &sl.layer, &s, &bx, &bdy, &bdx, ntok, 1024.0, Some(&ctx)).unwrap();
        let want = host_sign_grads(&sl, &x, &dy, ntok);
        let (mut worst_dev, mut worst_host) = (0f64, 0f64);
        let picks = [(0usize, 0usize), (1, 31), (2, 32), (255, 127), (17, 63), (100, 64), (200, 95), (5, 96)];
        for pi in 0..np {
            let (su, sv, h, l, g) = &sl.host[pi];
            let (gu_h, gv_h, a) = &want[pi];
            let sp = sl.layer.paths[pi].sign.as_ref().unwrap();
            let gu_d = download(c, sp.u.gsave.as_ref().unwrap(), m * r).unwrap();
            let gv_d = download(c, sp.v.gsave.as_ref().unwrap(), n * r).unwrap();
            let sum_h2: f64 = h.iter().map(|v| (*v as f64).powi(2)).sum();
            for side in 0..2 {
                let (rows, bits_buf, signs) = if side == 0 { (m, &sl.layer.paths[pi].bits_u, su) } else { (n, &sl.layer.paths[pi].bits_v, sv) };
                let base = pack_bits(signs, rows, r);
                for (q, &(i0, k0)) in picks.iter().enumerate() {
                    let i = (i0 + 37 * q * side) % rows;
                    let k = (k0 + 11 * pi) % r;
                    let s0 = signs[i * r + k] as f64;
                    let (gd, gh, quad) = if side == 0 {
                        let sa: f64 = (0..ntok).map(|t| a[t * r + k].powi(2)).sum();
                        (gu_d[i * r + k] as f64, gu_h[i * r + k], 2.0 * (h[i] as f64).powi(2) * (l[k] as f64).powi(2) * sa)
                    } else {
                        let sx: f64 = (0..ntok).map(|t| (x[t * n + i] as f64).powi(2)).sum();
                        (gv_d[i * r + k] as f64, gv_h[i * r + k], 2.0 * (g[i] as f64).powi(2) * (l[k] as f64).powi(2) * sx * sum_h2)
                    };
                    let mut fl = base.clone();
                    fl[i * (r / 32) + k / 32] ^= 1 << (k % 32);
                    c.queue.write_buffer(bits_buf, 0, bytemuck::cast_slice(&fl));
                    blr_forward(c, &sl.layer, &s, &bx, &by, ntok).unwrap();
                    let l1 = loss(&download(c, &by, ntok * m).unwrap());
                    c.queue.write_buffer(bits_buf, 0, bytemuck::cast_slice(&base));
                    let meas: f64 = l1.iter().zip(&l0).map(|(a, b)| a - b).sum();
                    let scale = 2.0 * gh.abs() + quad;
                    let ed = (meas - (-2.0 * s0 * gd + quad)).abs() / scale;
                    let eh = (meas - (-2.0 * s0 * gh + quad)).abs() / scale;
                    worst_dev = worst_dev.max(ed);
                    worst_host = worst_host.max(eh);
                    if q < 2 {
                        eprintln!("path {pi} {} [{i},{k}]: ΔL {meas:.6e}  pred(dev G) {:.6e}  pred(f64 G) {:.6e}", ["U", "V"][side], -2.0 * s0 * gd + quad, -2.0 * s0 * gh + quad);
                    }
                }
            }
        }
        eprintln!("flip ΔL vs prediction: worst {worst_dev:e} (device G), {worst_host:e} (f64 G), relative to 2|G| + quad");
        // f32 rounding of the two-path sum over 17920 outputs leaves ~1e-4 of
        // the term scale; a wrong bit index or formula is off by O(1)
        assert!(worst_host < 5e-4, "identity/bit mapping: {worst_host:e}");
        assert!(worst_dev < 2e-3, "device G: {worst_dev:e}");
    }
}
