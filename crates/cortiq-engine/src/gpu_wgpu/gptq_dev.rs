//! GPTQ fold operator and q4tp row sweep on the device.
//!
//! The host fold (`cortiq-cli` `gptq_fold_operator_par` + `gptq_quantize_q4tp`)
//! is O(n³) f64 per Hessian group plus O(rows·n²) per tensor: on Qwen3.8-27B
//! (64 `down_proj` groups at n = 17408) the 27 host cores needed ~19 minutes
//! for each round of 11 concurrent groups. Here, per group, in f32:
//!
//! 1. `H_d = H + λ·mean(diag)·I` (dead channels: row/column zero, diagonal 1,
//!    as on the host), equilibrated `H̃ = D⁻¹ H_d D⁻¹` with `D = √diag(H_d)` —
//!    unit diagonal, so f32 sees correlations instead of raw activation
//!    energies (Qwen's outlier channels are ~10⁴× the median);
//! 2. `Ã = P H̃ P` (P reverses the channel order), blocked Cholesky `Ã = L Lᵀ`;
//! 3. `X = L⁻¹` (blocked, right-looking on the identity);
//! 4. `U = P X P · D⁻¹` — the upper Cholesky factor of `H_d⁻¹`, i.e. the
//!    host's fold operator (unique, so equal up to rounding), at 2/3·n³
//!    instead of the host's 4/3·n³ (it forms `H⁻¹` and factors it again);
//! 5. the q4tp sweep: per 32-column group one thread per row picks the rung
//!    (the host's rule: nominal from the corrected absmax, best of
//!    nominal−2..+1 by `H_ii`-weighted error), rounds with in-group error
//!    feedback, and the group's residuals reach the rest of every row through
//!    one GEMM (`W[:, c0+32..] −= E·U[c0..c0+32, c0+32..]`).
//!
//! A pivot that is not positive (or NaN) raises λ ×10 and refactors, as the
//! host does.

use super::{Ctx, submit, trainer_ctx as ctx};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const SRC: &str = r#"
struct P { n: u32, k0: u32, m: u32, aux: u32, damp: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> H: array<f32>;
@group(0) @binding(4) var<storage, read_write> D: array<f32>;
@group(0) @binding(5) var<storage, read_write> FLAG: array<u32>;
@group(0) @binding(6) var<storage, read_write> W: array<f32>;
@group(0) @binding(7) var<storage, read_write> TAB: array<f32>;
@group(0) @binding(8) var<storage, read_write> LOST: array<f32>;
@group(0) @binding(9) var<storage, read_write> HD: array<f32>;
@group(0) @binding(10) var<storage, read_write> QN: array<u32>;
@group(0) @binding(11) var<storage, read_write> CODES: array<u32>;
@group(0) @binding(12) var<storage, read_write> E: array<f32>;
@group(0) @binding(13) var<storage, read_write> MI: array<f32>;
var<workgroup> sa: array<f32, 4096>;
var<workgroup> sb: array<f32, 4096>;

fn poff(r: u32, n: u32) -> u32 { return (r * (2u * n - r + 1u)) / 2u; }

// Packed-triangle diagonal into X[0..n].
@compute @workgroup_size(256)
fn diag_get(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) { X[i] = H[poff(i, p.n)]; }
}

// A = reversed, damped, equilibrated H (full matrix). D[k] = d_k > 0, or
// -d_k for a dead channel (row/column zero, diagonal 1 before damping).
@compute @workgroup_size(16, 16)
fn build_a(@builtin(global_invocation_id) g: vec3<u32>) {
    let n = p.n;
    let i = g.y;
    let j = g.x;
    if (i >= n || j >= n) { return; }
    let ii = n - 1u - i;
    let jj = n - 1u - j;
    let di = D[ii];
    let dj = D[jj];
    var v = 0.0;
    if (di < 0.0 || dj < 0.0) {
        if (ii == jj) { v = 1.0; }
    } else {
        let a = min(ii, jj);
        let b = max(ii, jj);
        var h = H[poff(a, n) + (b - a)];
        if (ii == jj) { h = h + p.damp; }
        v = h / (di * dj);
    }
    A[i * n + j] = v;
}

// Unblocked Cholesky of the 64x64 diagonal block at k0 (lower, in place).
@compute @workgroup_size(256)
fn chol_diag(@builtin(local_invocation_index) t: u32) {
    let n = p.n;
    let k0 = p.k0;
    for (var e = t; e < 4096u; e = e + 256u) {
        let r = e / 64u;
        let c = e % 64u;
        var v = 0.0;
        if (k0 + r < n && k0 + c < n) { v = A[(k0 + r) * n + k0 + c]; }
        else if (r == c) { v = 1.0; }
        sa[e] = v;
    }
    workgroupBarrier();
    for (var c = 0u; c < 64u; c = c + 1u) {
        if (t == 0u) {
            var piv = sa[c * 64u + c];
            if (!(piv > 0.0) || piv != piv) { FLAG[0] = 1u; piv = 1.0; }
            sa[c * 64u + c] = sqrt(piv);
        }
        workgroupBarrier();
        let dc = sa[c * 64u + c];
        for (var r = c + 1u + t; r < 64u; r = r + 256u) {
            sa[r * 64u + c] = sa[r * 64u + c] / dc;
        }
        workgroupBarrier();
        for (var e = t; e < 4096u; e = e + 256u) {
            let r = e / 64u;
            let s = e % 64u;
            if (r > c && s > c && s <= r) {
                sa[e] = sa[e] - sa[r * 64u + c] * sa[s * 64u + c];
            }
        }
        workgroupBarrier();
    }
    for (var e = t; e < 4096u; e = e + 256u) {
        let r = e / 64u;
        let c = e % 64u;
        if (c <= r && k0 + r < n && k0 + c < n) { A[(k0 + r) * n + k0 + c] = sa[e]; }
    }
}

// Panel: rows below the block, x Lkkᵀ = a (forward substitution per row).
@compute @workgroup_size(256)
fn chol_panel(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let n = p.n;
    let k0 = p.k0;
    for (var e = t; e < 4096u; e = e + 256u) {
        let r = e / 64u;
        let c = e % 64u;
        var v = 0.0;
        if (k0 + r < n && k0 + c < n) { v = A[(k0 + r) * n + k0 + c]; }
        else if (r == c) { v = 1.0; }
        sa[e] = v;
    }
    workgroupBarrier();
    let row = k0 + 64u + wg.x * 256u + t;
    if (row >= n) { return; }
    var x: array<f32, 64>;
    for (var c = 0u; c < 64u; c = c + 1u) {
        var s = 0.0;
        if (k0 + c < n) { s = A[row * n + k0 + c]; }
        for (var q = 0u; q < c; q = q + 1u) { s = s - x[q] * sa[c * 64u + q]; }
        x[c] = s / sa[c * 64u + c];
    }
    for (var c = 0u; c < 64u; c = c + 1u) {
        if (k0 + c < n) { A[row * n + k0 + c] = x[c]; }
    }
}

// Trailing update A[i][j] -= sum_t P[i][t] P[j][t] on lower 64x64 tiles.
@compute @workgroup_size(16, 16)
fn chol_update(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = p.n;
    let k0 = p.k0;
    let s0 = k0 + 64u;
    if (wg.x > wg.y) { return; }
    let i0 = s0 + wg.y * 64u;
    let j0 = s0 + wg.x * 64u;
    for (var e = li; e < 4096u; e = e + 256u) {
        let r = e / 64u;
        let q = e % 64u;
        var va = 0.0;
        var vb = 0.0;
        if (k0 + q < n) {
            if (i0 + r < n) { va = A[(i0 + r) * n + k0 + q]; }
            if (j0 + r < n) { vb = A[(j0 + r) * n + k0 + q]; }
        }
        sa[q * 64u + r] = va;
        sb[q * 64u + r] = vb;
    }
    workgroupBarrier();
    var acc: array<vec4<f32>, 4>;
    for (var r = 0u; r < 4u; r = r + 1u) { acc[r] = vec4<f32>(0.0); }
    for (var q = 0u; q < 64u; q = q + 1u) {
        let ra = q * 64u + lid.y * 4u;
        let rb = q * 64u + lid.x * 4u;
        let a = vec4<f32>(sa[ra], sa[ra + 1u], sa[ra + 2u], sa[ra + 3u]);
        let b = vec4<f32>(sb[rb], sb[rb + 1u], sb[rb + 2u], sb[rb + 3u]);
        acc[0] = fma(vec4<f32>(a.x), b, acc[0]);
        acc[1] = fma(vec4<f32>(a.y), b, acc[1]);
        acc[2] = fma(vec4<f32>(a.z), b, acc[2]);
        acc[3] = fma(vec4<f32>(a.w), b, acc[3]);
    }
    for (var r = 0u; r < 4u; r = r + 1u) {
        let i = i0 + lid.y * 4u + r;
        if (i < n) {
            for (var c = 0u; c < 4u; c = c + 1u) {
                let j = j0 + lid.x * 4u + c;
                if (j < n) { A[i * n + j] = A[i * n + j] - acc[r][c]; }
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn eye(@builtin(global_invocation_id) g: vec3<u32>) {
    let n = p.n;
    if (g.x < n && g.y < n) { X[g.y * n + g.x] = select(0.0, 1.0, g.x == g.y); }
}

// MI = inverse of the 64x64 lower block of L at k0 (one column per thread).
@compute @workgroup_size(64)
fn inv_diag(@builtin(local_invocation_index) t: u32) {
    let n = p.n;
    let k0 = p.k0;
    for (var e = t; e < 4096u; e = e + 64u) {
        let r = e / 64u;
        let c = e % 64u;
        var v = 0.0;
        if (k0 + r < n && k0 + c < n) { v = A[(k0 + r) * n + k0 + c]; }
        else if (r == c) { v = 1.0; }
        sa[e] = v;
        sb[e] = 0.0;
    }
    workgroupBarrier();
    let c = t;
    sb[c * 64u + c] = 1.0 / sa[c * 64u + c];
    for (var r = c + 1u; r < 64u; r = r + 1u) {
        var s = 0.0;
        for (var q = c; q < r; q = q + 1u) { s = s + sa[r * 64u + q] * sb[q * 64u + c]; }
        sb[r * 64u + c] = -s / sa[r * 64u + r];
    }
    workgroupBarrier();
    for (var e = t; e < 4096u; e = e + 64u) { MI[e] = sb[e]; }
}

// X rows k0..k0+64, column tile wg.x (columns < k0+64): X_k = MI · X_k.
@compute @workgroup_size(16, 16)
fn inv_rows(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = p.n;
    let k0 = p.k0;
    let c0 = wg.x * 64u;
    for (var e = li; e < 4096u; e = e + 256u) {
        let r = e / 64u;
        let q = e % 64u;
        sa[e] = MI[e];
        var v = 0.0;
        if (k0 + r < n && c0 + q < n) { v = X[(k0 + r) * n + c0 + q]; }
        sb[e] = v;
    }
    workgroupBarrier();
    var acc: array<vec4<f32>, 4>;
    for (var r = 0u; r < 4u; r = r + 1u) { acc[r] = vec4<f32>(0.0); }
    for (var q = 0u; q < 64u; q = q + 1u) {
        let b = vec4<f32>(sb[q * 64u + lid.x * 4u], sb[q * 64u + lid.x * 4u + 1u],
                          sb[q * 64u + lid.x * 4u + 2u], sb[q * 64u + lid.x * 4u + 3u]);
        for (var r = 0u; r < 4u; r = r + 1u) {
            acc[r] = fma(vec4<f32>(sa[(lid.y * 4u + r) * 64u + q]), b, acc[r]);
        }
    }
    for (var r = 0u; r < 4u; r = r + 1u) {
        let i = k0 + lid.y * 4u + r;
        if (i < n) {
            for (var c = 0u; c < 4u; c = c + 1u) {
                let j = c0 + lid.x * 4u + c;
                if (j < n) { X[i * n + j] = acc[r][c]; }
            }
        }
    }
}

// X[i][c] -= sum_t L[i][k0+t] X[k0+t][c] for i >= k0+64, c < k0+64.
@compute @workgroup_size(16, 16)
fn inv_update(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = p.n;
    let k0 = p.k0;
    let i0 = k0 + 64u + wg.y * 64u;
    let c0 = wg.x * 64u;
    for (var e = li; e < 4096u; e = e + 256u) {
        let r = e / 64u;
        let q = e % 64u;
        var va = 0.0;
        if (i0 + r < n && k0 + q < n) { va = A[(i0 + r) * n + k0 + q]; }
        sa[q * 64u + r] = va;
        var vb = 0.0;
        if (k0 + r < n && c0 + q < n) { vb = X[(k0 + r) * n + c0 + q]; }
        sb[r * 64u + q] = vb;
    }
    workgroupBarrier();
    var acc: array<vec4<f32>, 4>;
    for (var r = 0u; r < 4u; r = r + 1u) { acc[r] = vec4<f32>(0.0); }
    for (var q = 0u; q < 64u; q = q + 1u) {
        let ra = q * 64u + lid.y * 4u;
        let a = vec4<f32>(sa[ra], sa[ra + 1u], sa[ra + 2u], sa[ra + 3u]);
        let rb = q * 64u + lid.x * 4u;
        let b = vec4<f32>(sb[rb], sb[rb + 1u], sb[rb + 2u], sb[rb + 3u]);
        acc[0] = fma(vec4<f32>(a.x), b, acc[0]);
        acc[1] = fma(vec4<f32>(a.y), b, acc[1]);
        acc[2] = fma(vec4<f32>(a.z), b, acc[2]);
        acc[3] = fma(vec4<f32>(a.w), b, acc[3]);
    }
    for (var r = 0u; r < 4u; r = r + 1u) {
        let i = i0 + lid.y * 4u + r;
        if (i < n) {
            for (var c = 0u; c < 4u; c = c + 1u) {
                let j = c0 + lid.x * 4u + c;
                if (j < n) { X[i * n + j] = X[i * n + j] - acc[r][c]; }
            }
        }
    }
}

// U[j][k] = X[n-1-j][n-1-k] / |D[k]| (upper; zero below), into A.
@compute @workgroup_size(16, 16)
fn make_u(@builtin(global_invocation_id) g: vec3<u32>) {
    let n = p.n;
    let j = g.y;
    let k = g.x;
    if (j >= n || k >= n) { return; }
    var v = 0.0;
    if (k >= j) { v = X[(n - 1u - j) * n + (n - 1u - k)] / abs(D[k]); }
    A[j * n + k] = v;
}

fn round_away(x: f32) -> f32 { return sign(x) * floor(abs(x) + 0.5); }

// One q4tp group per row: rung choice, rounding with in-group feedback,
// nibbles into QN (the tile bytes), residuals into E.
@compute @workgroup_size(128)
fn q4tp_group(@builtin(global_invocation_id) g: vec3<u32>) {
    let n = p.n;          // cols
    let rows = p.m;
    let grp = p.k0;
    let row = g.x;
    if (row >= rows) { return; }
    let c0 = grp * 32u;
    var w: array<f32, 32>;
    var amax = 0.0;
    for (var t = 0u; t < 32u; t = t + 1u) {
        w[t] = W[row * n + c0 + t];
        amax = max(amax, abs(w[t]));
    }
    let lo = LOST[row * 2u];
    let st = LOST[row * 2u + 1u];
    var nominal = 0u;
    if (amax > 0.0 && st > 0.0) {
        nominal = u32(clamp(round_away((log2(amax / 7.0) - lo) / st), 0.0, 31.0));
    }
    var best_c = nominal;
    if (amax > 0.0) {
        var best_e = 3.4e38;
        var lo_c = 0u;
        if (nominal >= 2u) { lo_c = nominal - 2u; }
        let hi_c = min(nominal + 1u, 31u);
        for (var cand = lo_c; cand <= hi_c; cand = cand + 1u) {
            let s_c = TAB[row * 32u + cand];
            if (s_c <= 0.0) { continue; }
            let iv = 1.0 / s_c;
            var err = 0.0;
            for (var t = 0u; t < 32u; t = t + 1u) {
                let q = clamp(round(w[t] * iv), -8.0, 7.0);
                let d = w[t] - q * s_c;
                err = err + HD[c0 + t] * d * d;
            }
            if (err < best_e) { best_e = err; best_c = cand; }
        }
    }
    let s = TAB[row * 32u + best_c];
    CODES[row * (n / 32u) + grp] = best_c;
    var words: array<u32, 4>;
    for (var t = 0u; t < 4u; t = t + 1u) { words[t] = 0u; }
    for (var j = 0u; j < 32u; j = j + 1u) {
        var q = 0.0;
        if (s > 0.0) { q = clamp(round(w[j] / s), -8.0, 7.0); }
        let nb = u32(i32(q) + 8) & 15u;
        words[j / 8u] = words[j / 8u] | (nb << (4u * (j % 8u)));
        let dj = A[(c0 + j) * n + c0 + j];
        var e = 0.0;
        if (dj > 0.0) { e = (w[j] - q * s) / dj; }
        E[row * 32u + j] = e;
        if (e != 0.0) {
            for (var t = j + 1u; t < 32u; t = t + 1u) {
                w[t] = w[t] - e * A[(c0 + j) * n + c0 + t];
            }
        }
    }
    let base = (row * (n / 32u) + grp) * 4u;
    for (var t = 0u; t < 4u; t = t + 1u) { QN[base + t] = words[t]; }
}

fn q2_level(x: f32) -> f32 { return clamp(round(clamp(x, -1.0, 4.0)), 0.0, 3.0); }

// One q2tp group per row, the "placed" rule: the best of rungs 1..31 by
// H_ii-weighted rounding error (rung 0 = an all-zero group), rounding onto
// ±0.5/±1.5 with in-group feedback; 2-bit codes into QN (8 bytes a group).
@compute @workgroup_size(128)
fn q2tp_group(@builtin(global_invocation_id) g: vec3<u32>) {
    let n = p.n;
    let rows = p.m;
    let grp = p.k0;
    let row = g.x;
    if (row >= rows) { return; }
    let c0 = grp * 32u;
    var w: array<f32, 32>;
    var amax = 0.0;
    for (var t = 0u; t < 32u; t = t + 1u) {
        w[t] = W[row * n + c0 + t];
        amax = max(amax, abs(w[t]));
    }
    var best_c = 0u;
    if (amax > 0.0) {
        var best_e = 3.4e38;
        best_c = 1u;
        for (var cand = 1u; cand < 32u; cand = cand + 1u) {
            let s_c = TAB[row * 32u + cand];
            if (!(s_c > 0.0)) { continue; }
            let iv = 1.0 / s_c;
            var err = 0.0;
            for (var t = 0u; t < 32u; t = t + 1u) {
                let d = w[t] - (q2_level(w[t] * iv + 1.5) - 1.5) * s_c;
                err = err + HD[c0 + t] * d * d;
            }
            if (err < best_e) { best_e = err; best_c = cand; }
        }
    }
    let s = TAB[row * 32u + best_c];
    CODES[row * (n / 32u) + grp] = best_c;
    var words: array<u32, 2>;
    words[0] = 0u;
    words[1] = 0u;
    for (var j = 0u; j < 32u; j = j + 1u) {
        var q = 0.0;
        if (s > 0.0) { q = q2_level(w[j] / s + 1.5); }
        words[j / 16u] = words[j / 16u] | (u32(q) << (2u * (j % 16u)));
        let dj = A[(c0 + j) * n + c0 + j];
        var e = 0.0;
        if (dj > 0.0) { e = (w[j] - (q - 1.5) * s) / dj; }
        E[row * 32u + j] = e;
        if (e != 0.0) {
            for (var t = j + 1u; t < 32u; t = t + 1u) {
                w[t] = w[t] - e * A[(c0 + j) * n + c0 + t];
            }
        }
    }
    let base = (row * (n / 32u) + grp) * 2u;
    QN[base] = words[0];
    QN[base + 1u] = words[1];
}

// W[:, c0+32..] -= E · U[c0..c0+32, c0+32..] (64x64 tiles, K = 32).
@compute @workgroup_size(16, 16)
fn q4tp_trail(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = p.n;
    let rows = p.m;
    let c0 = p.k0 * 32u;
    let r0 = wg.y * 64u;
    let j0 = c0 + 32u + wg.x * 64u;
    for (var e = li; e < 2048u; e = e + 256u) {
        let r = e / 32u;
        let q = e % 32u;
        var va = 0.0;
        if (r0 + r < rows) { va = E[(r0 + r) * 32u + q]; }
        sa[q * 64u + r] = va;
        let qq = e / 64u;
        let cc = e % 64u;
        var vb = 0.0;
        if (j0 + cc < n) { vb = A[(c0 + qq) * n + j0 + cc]; }
        sb[qq * 64u + cc] = vb;
    }
    workgroupBarrier();
    var acc: array<vec4<f32>, 4>;
    for (var r = 0u; r < 4u; r = r + 1u) { acc[r] = vec4<f32>(0.0); }
    for (var q = 0u; q < 32u; q = q + 1u) {
        let ra = q * 64u + lid.y * 4u;
        let a = vec4<f32>(sa[ra], sa[ra + 1u], sa[ra + 2u], sa[ra + 3u]);
        let rb = q * 64u + lid.x * 4u;
        let b = vec4<f32>(sb[rb], sb[rb + 1u], sb[rb + 2u], sb[rb + 3u]);
        acc[0] = fma(vec4<f32>(a.x), b, acc[0]);
        acc[1] = fma(vec4<f32>(a.y), b, acc[1]);
        acc[2] = fma(vec4<f32>(a.z), b, acc[2]);
        acc[3] = fma(vec4<f32>(a.w), b, acc[3]);
    }
    for (var r = 0u; r < 4u; r = r + 1u) {
        let i = r0 + lid.y * 4u + r;
        if (i < rows) {
            for (var c = 0u; c < 4u; c = c + 1u) {
                let j = j0 + lid.x * 4u + c;
                if (j < n) { W[i * n + j] = W[i * n + j] - acc[r][c]; }
            }
        }
    }
}
"#;

const ENTRIES: [&str; 13] = [
    "diag_get", "build_a", "chol_diag", "chol_panel", "chol_update", "eye", "inv_diag", "inv_rows",
    "inv_update", "make_u", "q4tp_group", "q2tp_group", "q4tp_trail",
];
const NBIND: u32 = 14;
/// Per-dispatch parameter stride (the uniform offset alignment).
const PSTRIDE: u64 = 256;

struct Pipes {
    layout: wgpu::BindGroupLayout,
    p: HashMap<&'static str, wgpu::ComputePipeline>,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let lim = c.device.limits();
            if lim.max_compute_workgroup_storage_size < 32768 {
                return None;
            }
            let mut entries = vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(32),
                },
                count: None,
            }];
            for b in 1..NBIND {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }
            let layout = c.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gptq_dev"),
                entries: &entries,
            });
            let pl = c.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gptq_dev"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gptq_dev"),
                source: wgpu::ShaderSource::Wgsl(SRC.into()),
            });
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("gptq_dev module rejected: {e}");
                return None;
            }
            let mut p = HashMap::new();
            for ep in ENTRIES {
                let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
                let pipe = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(ep),
                    layout: Some(&pl),
                    module: &m,
                    entry_point: Some(ep),
                    compilation_options: crate::gpu_wgpu::pco(),
                    cache: None,
                });
                if let Some(e) = pollster::block_on(sc.pop()) {
                    tracing::warn!("gptq_dev pipeline {ep} rejected: {e}");
                    return None;
                }
                p.insert(ep, pipe);
            }
            Some(Pipes { layout, p })
        })
        .as_ref()
}

/// A folded group: `U` (n×n f32, upper) on the device, plus what the
/// sweep and the host need about the channels.
pub struct Fold {
    u: wgpu::Buffer,
    pub n: usize,
    pub dead: Vec<bool>,
    pub hdiag: Vec<f32>,
    pub lambda: f64,
}

static FOLDS: Mutex<Option<(u64, HashMap<u64, Fold>)>> = Mutex::new(None);

fn buf(c: &Ctx, label: &str, size: u64, extra: wgpu::BufferUsages) -> Option<wgpu::Buffer> {
    let lim = c.device.limits();
    let size = size.max(16);
    if size > lim.max_buffer_size || size > lim.max_storage_buffer_binding_size as u64 {
        return None;
    }
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let b = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | extra,
        mapped_at_creation: false,
    });
    if pollster::block_on(sc.pop()).is_some() {
        return None;
    }
    Some(b)
}

fn params(c: &Ctx, list: &[[u32; 8]]) -> wgpu::Buffer {
    let mut bytes = vec![0u8; list.len().max(1) * PSTRIDE as usize];
    for (k, v) in list.iter().enumerate() {
        bytes[k * PSTRIDE as usize..k * PSTRIDE as usize + 32].copy_from_slice(bytemuck::cast_slice(v));
    }
    let b = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gptq_dev_p"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    c.queue.write_buffer(&b, 0, &bytes);
    b
}

fn bind(c: &Ctx, pp: &Pipes, prm: &wgpu::Buffer, bufs: &[&wgpu::Buffer; 13]) -> wgpu::BindGroup {
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: prm,
            offset: 0,
            size: std::num::NonZeroU64::new(32),
        }),
    }];
    for (k, b) in bufs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry { binding: k as u32 + 1, resource: b.as_entire_binding() });
    }
    c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gptq_dev"), layout: &pp.layout, entries: &entries })
}

fn read_f32(c: &Ctx, src: &wgpu::Buffer, len: usize) -> Option<Vec<f32>> {
    let size = (len * 4) as u64;
    let st = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gptq_dev_rb"),
        size: size.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gptq_dev_rb") });
    enc.copy_buffer_to_buffer(src, 0, &st, 0, size);
    submit(c, enc.finish());
    let slice = st.slice(..size);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let out = {
        let data = slice.get_mapped_range().ok()?;
        let mut v = vec![0f32; len];
        super::par_copy(&mut v, bytemuck::cast_slice(&data[..size as usize]));
        v
    };
    st.unmap();
    Some(out)
}

/// Fold the device Hessian `h_id` (a `gram` accumulator, freed on success,
/// left in place on failure so the caller can read it back): the operator
/// stays on the device under the returned id.
pub fn fold(h_id: u64, n: usize, lambda: f64) -> Option<u64> {
    let id = fold_inner(h_id, n, lambda)?;
    super::gram::free(h_id);
    Some(id)
}

fn fold_inner(h_id: u64, n: usize, lambda: f64) -> Option<u64> {
    let c = ctx()?;
    let pp = pipes(c)?;
    let (h, hn) = super::gram::buffer(h_id)?;
    if hn != n || n == 0 {
        return None;
    }
    let nn = (n as u64) * (n as u64) * 4;
    let a = buf(c, "gptq_a", nn, wgpu::BufferUsages::empty())?;
    let x = buf(c, "gptq_x", nn, wgpu::BufferUsages::empty())?;
    let d_buf = buf(c, "gptq_d", (n * 4) as u64, wgpu::BufferUsages::empty())?;
    let flag = buf(c, "gptq_flag", 16, wgpu::BufferUsages::empty())?;
    let mi = buf(c, "gptq_mi", 4096 * 4, wgpu::BufferUsages::empty())?;
    let dummy = buf(c, "gptq_dummy", 16, wgpu::BufferUsages::empty())?;
    let all = |prm: &wgpu::Buffer| {
        bind(c, pp, prm, &[&a, &x, &h, &d_buf, &flag, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &mi])
    };
    // Diagonal → host: dead channels, mean, the equilibration.
    let p0 = params(c, &[[n as u32, 0, 0, 0, 0, 0, 0, 0]]);
    let bg0 = all(&p0);
    {
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gptq_diag") });
        {
            let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            ps.set_pipeline(&pp.p["diag_get"]);
            ps.set_bind_group(0, &bg0, &[0]);
            ps.dispatch_workgroups(n.div_ceil(256) as u32, 1, 1);
        }
        submit(c, enc.finish());
    }
    let hdiag = read_f32(c, &x, n)?;
    let dead: Vec<bool> = hdiag.iter().map(|&v| !(v > 0.0)).collect();
    let mean = hdiag.iter().zip(&dead).map(|(&v, &dd)| if dd { 1.0 } else { v as f64 }).sum::<f64>() / n as f64;
    let base = mean.max(1e-12);
    let nb = n.div_ceil(64);
    let nt = n.div_ceil(16) as u32;
    let mut lam = lambda.max(0.0);
    loop {
        let damp = lam * base;
        let dv: Vec<f32> = (0..n)
            .map(|i| {
                if dead[i] {
                    -((1.0 + damp).sqrt() as f32)
                } else {
                    ((hdiag[i] as f64 + damp).sqrt()) as f32
                }
            })
            .collect();
        c.queue.write_buffer(&d_buf, 0, bytemuck::cast_slice(&dv));
        c.queue.write_buffer(&flag, 0, bytemuck::cast_slice(&[0u32; 4]));
        // Step k: params entry k (k0 = 64·k); the last entry drives the
        // whole-matrix kernels.
        let mut list: Vec<[u32; 8]> = (0..nb)
            .map(|k| [n as u32, (k * 64) as u32, 0, 0, (damp as f32).to_bits(), 0, 0, 0])
            .collect();
        list.push([n as u32, 0, 0, 0, (damp as f32).to_bits(), 0, 0, 0]);
        let prm = params(c, &list);
        let bg = all(&prm);
        let last = (nb as u64) * PSTRIDE;
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gptq_fold") });
        {
            let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            ps.set_pipeline(&pp.p["build_a"]);
            ps.set_bind_group(0, &bg, &[last as u32]);
            ps.dispatch_workgroups(nt, nt, 1);
            for k in 0..nb {
                let off = (k as u64 * PSTRIDE) as u32;
                let rest = n.saturating_sub(k * 64 + 64);
                ps.set_bind_group(0, &bg, &[off]);
                ps.set_pipeline(&pp.p["chol_diag"]);
                ps.dispatch_workgroups(1, 1, 1);
                if rest > 0 {
                    ps.set_pipeline(&pp.p["chol_panel"]);
                    ps.dispatch_workgroups(rest.div_ceil(256) as u32, 1, 1);
                    let t = rest.div_ceil(64) as u32;
                    ps.set_pipeline(&pp.p["chol_update"]);
                    ps.dispatch_workgroups(t, t, 1);
                }
            }
        }
        submit(c, enc.finish());
        let f = read_u32(c, &flag)?;
        if f == 0 {
            break;
        }
        lam = if lam == 0.0 { 1e-4 } else { lam * 10.0 };
        if lam > 10.0 {
            return None;
        }
    }
    // X = L⁻¹, then U = P X P D⁻¹ into A.
    let list: Vec<[u32; 8]> = (0..=nb).map(|k| [n as u32, (k * 64).min(n) as u32, 0, 0, 0, 0, 0, 0]).collect();
    let prm = params(c, &list);
    let bg = all(&prm);
    let last = (nb as u64 * PSTRIDE) as u32;
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gptq_inv") });
    {
        let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        ps.set_bind_group(0, &bg, &[last]);
        ps.set_pipeline(&pp.p["eye"]);
        ps.dispatch_workgroups(nt, nt, 1);
        for k in 0..nb {
            let off = (k as u64 * PSTRIDE) as u32;
            let k0 = k * 64;
            ps.set_bind_group(0, &bg, &[off]);
            ps.set_pipeline(&pp.p["inv_diag"]);
            ps.dispatch_workgroups(1, 1, 1);
            ps.set_pipeline(&pp.p["inv_rows"]);
            ps.dispatch_workgroups((k0 + 64).min(n).div_ceil(64) as u32, 1, 1);
            let rest = n.saturating_sub(k0 + 64);
            if rest > 0 {
                ps.set_pipeline(&pp.p["inv_update"]);
                ps.dispatch_workgroups((k0 + 64).min(n).div_ceil(64) as u32, rest.div_ceil(64) as u32, 1);
            }
        }
        ps.set_bind_group(0, &bg, &[last]);
        ps.set_pipeline(&pp.p["make_u"]);
        ps.dispatch_workgroups(nt, nt, 1);
    }
    submit(c, enc.finish());
    c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    drop((x, h, d_buf, flag, mi, dummy));
    let mut g = FOLDS.lock().unwrap();
    let (next, map) = g.get_or_insert_with(|| (1, HashMap::new()));
    let id = *next;
    *next += 1;
    map.insert(id, Fold { u: a, n, dead, hdiag, lambda: lam });
    Some(id)
}

/// Blocked lower Cholesky of the dense `n×n` f32 matrix in `a`, in place
/// (the fold's `chol_diag`/`chol_panel`/`chol_update` steps on a caller's
/// buffer; the strict upper triangle outside the diagonal tiles is left
/// alone, inside them it is overwritten). `Some(false)` when a pivot is
/// not positive (or NaN).
pub(crate) fn chol_lower(a: &wgpu::Buffer, n: usize) -> Option<bool> {
    let c = ctx()?;
    let pp = pipes(c)?;
    let flag = buf(c, "chol_flag", 16, wgpu::BufferUsages::empty())?;
    let dummy = buf(c, "chol_dummy", 16, wgpu::BufferUsages::empty())?;
    c.queue.write_buffer(&flag, 0, bytemuck::cast_slice(&[0u32; 4]));
    let nb = n.div_ceil(64);
    let list: Vec<[u32; 8]> = (0..nb).map(|k| [n as u32, (k * 64) as u32, 0, 0, 0, 0, 0, 0]).collect();
    let prm = params(c, &list);
    let bg = bind(c, pp, &prm, &[a, &dummy, &dummy, &dummy, &flag, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy]);
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("chol_lower") });
    {
        let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        for k in 0..nb {
            let rest = n.saturating_sub(k * 64 + 64);
            ps.set_bind_group(0, &bg, &[(k as u64 * PSTRIDE) as u32]);
            ps.set_pipeline(&pp.p["chol_diag"]);
            ps.dispatch_workgroups(1, 1, 1);
            if rest > 0 {
                ps.set_pipeline(&pp.p["chol_panel"]);
                ps.dispatch_workgroups(rest.div_ceil(256) as u32, 1, 1);
                let t = rest.div_ceil(64) as u32;
                ps.set_pipeline(&pp.p["chol_update"]);
                ps.dispatch_workgroups(t, t, 1);
            }
        }
    }
    submit(c, enc.finish());
    Some(read_u32(c, &flag)? == 0)
}

/// `x = L⁻¹` for the lower factor `L` held in `a` (dense `n×n`, as
/// [`chol_lower`] leaves it): the fold's blocked right-looking inverse.
pub(crate) fn tri_inv_lower(a: &wgpu::Buffer, x: &wgpu::Buffer, n: usize) -> Option<()> {
    let c = ctx()?;
    let pp = pipes(c)?;
    let dummy = buf(c, "inv_dummy", 16, wgpu::BufferUsages::empty())?;
    let mi = buf(c, "inv_mi", 4096 * 4, wgpu::BufferUsages::empty())?;
    let nb = n.div_ceil(64);
    let nt = n.div_ceil(16) as u32;
    let list: Vec<[u32; 8]> = (0..=nb).map(|k| [n as u32, (k * 64).min(n) as u32, 0, 0, 0, 0, 0, 0]).collect();
    let prm = params(c, &list);
    let bg = bind(c, pp, &prm, &[a, x, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &mi]);
    let last = (nb as u64 * PSTRIDE) as u32;
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tri_inv_lower") });
    {
        let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        ps.set_bind_group(0, &bg, &[last]);
        ps.set_pipeline(&pp.p["eye"]);
        ps.dispatch_workgroups(nt, nt, 1);
        for k in 0..nb {
            let k0 = k * 64;
            ps.set_bind_group(0, &bg, &[(k as u64 * PSTRIDE) as u32]);
            ps.set_pipeline(&pp.p["inv_diag"]);
            ps.dispatch_workgroups(1, 1, 1);
            ps.set_pipeline(&pp.p["inv_rows"]);
            ps.dispatch_workgroups((k0 + 64).min(n).div_ceil(64) as u32, 1, 1);
            let rest = n.saturating_sub(k0 + 64);
            if rest > 0 {
                ps.set_pipeline(&pp.p["inv_update"]);
                ps.dispatch_workgroups((k0 + 64).min(n).div_ceil(64) as u32, rest.div_ceil(64) as u32, 1);
            }
        }
    }
    submit(c, enc.finish());
    Some(())
}

fn read_u32(c: &Ctx, src: &wgpu::Buffer) -> Option<u32> {
    let v = read_f32(c, src, 1)?;
    Some(v[0].to_bits())
}

/// Dead channels, raw `H_ii` and the damping actually used.
pub fn info(id: u64) -> Option<(Vec<bool>, Vec<f32>, f64)> {
    let g = FOLDS.lock().unwrap();
    let f = g.as_ref()?.1.get(&id)?;
    Some((f.dead.clone(), f.hdiag.clone(), f.lambda))
}

/// `U` read back (n×n, row-major, upper).
pub fn read_u(id: u64) -> Option<Vec<f32>> {
    let c = ctx()?;
    let g = FOLDS.lock().unwrap();
    let f = g.as_ref()?.1.get(&id)?;
    read_f32(c, &f.u, f.n * f.n)
}

pub fn free(id: u64) {
    if let Some((_, m)) = FOLDS.lock().unwrap().as_mut() {
        m.remove(&id);
    }
}

/// The q4tp GPTQ sweep of `w` (`rows × n`, dead columns already zeroed)
/// against fold `id`: `tab` = each row's 32-rung ladder, `lost` = its
/// (lo, step); writes the nibble plane (`rows·n/2` bytes, (row, group)
/// tiles) and one rung code per (row, group).
pub fn sweep_q4tp(
    id: u64,
    w: &[f32],
    rows: usize,
    tab: &[f32],
    lost: &[f32],
    nib: &mut [u8],
    codes: &mut [u32],
) -> bool {
    sweep(id, w, rows, tab, lost, None, nib, codes)
}

/// The q2tp GPTQ sweep (the "placed" rule) of `w` against fold `id`:
/// `tab` = each row's ladder with rung 0 = 0, `hw` = the clipped `H_ii`;
/// writes the 2-bit chunk plane (`rows·n/4` bytes) and the rung codes.
pub fn sweep_q2tp(id: u64, w: &[f32], rows: usize, tab: &[f32], hw: &[f32], chunks: &mut [u8], codes: &mut [u32]) -> bool {
    let lost = vec![0f32; rows * 2];
    sweep(id, w, rows, tab, &lost, Some(hw), chunks, codes)
}

#[allow(clippy::too_many_arguments)]
fn sweep(
    id: u64,
    w: &[f32],
    rows: usize,
    tab: &[f32],
    lost: &[f32],
    q2_hw: Option<&[f32]>,
    nib: &mut [u8],
    codes: &mut [u32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let Some(pp) = pipes(c) else { return false };
    let g = FOLDS.lock().unwrap();
    let Some(f) = g.as_ref().and_then(|g| g.1.get(&id)) else { return false };
    let n = f.n;
    let gpr = n / 32;
    if n % 32 != 0 || w.len() != rows * n || tab.len() != rows * 32 || lost.len() != rows * 2 {
        return false;
    }
    let plane = if q2_hw.is_some() { rows * n / 4 } else { rows * n / 2 };
    if nib.len() != plane || codes.len() != rows * gpr {
        return false;
    }
    let mk = |label: &str, bytes: u64| buf(c, label, bytes, wgpu::BufferUsages::empty());
    let (Some(wb), Some(tb), Some(lb), Some(hb), Some(qb), Some(cb), Some(eb), Some(dummy)) = (
        mk("gptq_w", (w.len() * 4) as u64),
        mk("gptq_tab", (tab.len() * 4) as u64),
        mk("gptq_lost", (lost.len() * 4) as u64),
        mk("gptq_hd", (n * 4) as u64),
        mk("gptq_qn", (nib.len()) as u64),
        mk("gptq_codes", (codes.len() * 4) as u64),
        mk("gptq_e", (rows * 32 * 4) as u64),
        mk("gptq_dummy", 16),
    ) else {
        return false;
    };
    c.queue.write_buffer(&wb, 0, bytemuck::cast_slice(w));
    c.queue.write_buffer(&tb, 0, bytemuck::cast_slice(tab));
    c.queue.write_buffer(&lb, 0, bytemuck::cast_slice(lost));
    c.queue.write_buffer(&hb, 0, bytemuck::cast_slice(q2_hw.unwrap_or(&f.hdiag)));
    let list: Vec<[u32; 8]> = (0..gpr).map(|k| [n as u32, k as u32, rows as u32, 0, 0, 0, 0, 0]).collect();
    let prm = params(c, &list);
    let bg = bind(c, pp, &prm, &[&f.u, &dummy, &dummy, &dummy, &dummy, &wb, &tb, &lb, &hb, &qb, &cb, &eb, &dummy]);
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gptq_sweep") });
    {
        let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        for k in 0..gpr {
            ps.set_bind_group(0, &bg, &[(k as u64 * PSTRIDE) as u32]);
            ps.set_pipeline(&pp.p[if q2_hw.is_some() { "q2tp_group" } else { "q4tp_group" }]);
            ps.dispatch_workgroups(rows.div_ceil(128) as u32, 1, 1);
            let rest = n - (k + 1) * 32;
            if rest > 0 {
                ps.set_pipeline(&pp.p["q4tp_trail"]);
                ps.dispatch_workgroups(rest.div_ceil(64) as u32, rows.div_ceil(64) as u32, 1);
            }
        }
    }
    submit(c, enc.finish());
    drop(g);
    let Some(q) = read_f32(c, &qb, nib.len() / 4) else { return false };
    nib.copy_from_slice(bytemuck::cast_slice(&q));
    let Some(cv) = read_f32(c, &cb, codes.len()) else { return false };
    for (d, s) in codes.iter_mut().zip(&cv) {
        *d = s.to_bits();
    }
    true
}
