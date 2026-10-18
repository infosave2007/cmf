//! Sign training of the binary low-rank (`blr`) paths: one straight-through
//! latent per sign, updated during the trainer's reverse sweep right after
//! the path's own backward. Nothing later in the step reads that layer again
//! (lower layers recompute from checkpoints written before any update, and
//! the gradient they receive was computed with the old bits), so the
//! in-place update equals updating every matrix after the sweep.
//!
//! Sign gradients of a path `y = h ⊙ (((x⊙g)·S_V ⊙ ℓ)·S_Uᵀ)` are built in
//! `train_mm::blr_backward` from operands it already stages:
//!
//! ```text
//! G_U = diag(h)·dyᵀ·a·diag(ℓ)     (m × rp)     a = (x⊙g)·S_V
//! G_V = diag(g)·xᵀ·b·diag(ℓ)      (n × rp)     b = (dy⊙h)·S_U
//! ```
//!
//! Latents Λ ∈ [−1, 1] are int16 `q = round(32767·Λ)`, two per u32, in the
//! bitmap's row layout (`[rows][rp]`); the sign is +1 exactly when q ≥ 0
//! (the file convention: bit 1 = +1). Optimizers:
//!
//! * `fact` (2 B per sign): factored second moment (row and column EMAs of
//!   G², Adafactor-style), no momentum, step clamped to ±c, stochastic
//!   rounding of the int16 latent;
//! * `fact-mom` (4 B): the same plus a bf16 first moment;
//! * `ref` (12 B): f32 latent, per-element Adam over the SmoothSign
//!   gradient `G·κ·(1 − tanh²(κΛ))` — the small-model reference recipe.

use super::Ctx;
use super::train_mm::Op;
use super::train_ops::{self, Disp, elt_fold};

const SRC: &str = r#"
struct P {
    rows: u32, r: u32, rp: u32, wpr: u32,
    id: u32, seed: u32, step: u32, slot: u32,
    eta: f32, b1: f32, b2: f32, eps: f32,
    clampv: f32, hyst: f32, kss: f32, bc2: f32,
    bc1: f32, chunks: u32, tau: f32, pad0: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> LAT: array<u32>;
@group(0) @binding(3) var<storage, read_write> MOM: array<u32>;
@group(0) @binding(4) var<storage, read_write> VV: array<f32>;
@group(0) @binding(5) var<storage, read_write> VROW: array<f32>;
@group(0) @binding(6) var<storage, read_write> VCOL: array<f32>;
@group(0) @binding(7) var<storage, read_write> BITS: array<u32>;
@group(0) @binding(8) var<storage, read_write> PART: array<f32>;
@group(0) @binding(9) var<storage, read_write> SCAL: array<f32>;
@group(0) @binding(10) var<storage, read_write> CNT: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> BITS0: array<u32>;
@group(0) @binding(12) var<storage, read_write> GOUT: array<f32>;

fn idx(g: vec3<u32>) -> u32 { return g.x + g.y * 65535u * 256u; }
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn unif24(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }
fn fin(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
// Standard normal for entry e of the matrix whose stream is `base` (Box–Muller).
fn gauss(base: u32, e: u32) -> f32 {
    let u1 = unif24(pcg(base ^ pcg(2u * e))) + (1.0 / 16777216.0);
    let u2 = unif24(pcg(base ^ pcg(2u * e + 1u)));
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
fn init_q(base: u32, row: u32, k: u32, pos: bool) -> i32 {
    var mag = 1;
    if (k < p.r) {
        let z = gauss(base, row * p.rp + k);
        mag = clamp(i32(round(32767.0 * p.tau * abs(z))), 1, 32767);
    }
    return select(-mag, mag, pos);
}
// Live ranks of bitmap word w (bit j = rank 32w + j < r).
fn live_mask(w: u32) -> u32 {
    let k0 = w * 32u;
    if (k0 >= p.r) { return 0u; }
    if (k0 + 32u <= p.r) { return 0xffffffffu; }
    return (1u << (p.r - k0)) - 1u;
}
fn newbit(q: i32, old: u32) -> u32 {
    if (p.hyst <= 0.0) { return select(0u, 1u, q >= 0); }
    let th = i32(32767.0 * p.hyst);
    if (q > th) { return 1u; }
    if (q < -th) { return 0u; }
    return old;
}
// int16 latent after a step of eta·u, stochastically rounded with uniform h:
// (new q, 1 if the ±32767 clamp was hit).
fn qstep(q: i32, u: f32, h: u32) -> vec2<i32> {
    let delta = -32767.0 * p.eta * u;
    let fl = floor(delta);
    let fr = delta - fl;
    var qn = q + i32(fl) + select(0, 1, unif24(h) < fr);
    var sat = 0;
    if (qn > 32767 || qn < -32767) { qn = clamp(qn, -32767, 32767); sat = 1; }
    return vec2<i32>(qn, sat);
}
fn bf16(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    return (b + 0x7fffu + ((b >> 16u) & 1u)) >> 16u;
}
fn half_of(w: u32, h: u32) -> i32 {
    if (h == 0u) { return bitcast<i32>(w << 16u) >> 16u; }
    return bitcast<i32>(w) >> 16u;
}

// int16 latents from the bitmap: q = ±clamp(round(32767·τ·|z|), 1, 32767),
// sign from the bit; pad ranks get ±1.  One thread per bitmap word.
@compute @workgroup_size(256)
fn lat_init(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.rows * p.wpr) { return; }
    let row = i / p.wpr;
    let w = i % p.wpr;
    let bw = BITS[i];
    let base = pcg(p.seed ^ pcg(p.id ^ 0x5bd1e995u));
    let half = p.rp / 2u;
    for (var j = 0u; j < 32u; j = j + 2u) {
        var pair = 0u;
        for (var h = 0u; h < 2u; h = h + 1u) {
            let q = init_q(base, row, w * 32u + j + h, ((bw >> (j + h)) & 1u) == 1u);
            pair = pair | ((bitcast<u32>(q) & 0xffffu) << (16u * h));
        }
        LAT[row * half + w * 16u + j / 2u] = pair;
    }
}

// The same values as f32 latents [rows][rp] (the reference optimizer).
@compute @workgroup_size(256)
fn lat_init_f32(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.rows * p.wpr) { return; }
    let row = i / p.wpr;
    let w = i % p.wpr;
    let bw = BITS[i];
    let base = pcg(p.seed ^ pcg(p.id ^ 0x5bd1e995u));
    for (var j = 0u; j < 32u; j = j + 1u) {
        let q = init_q(base, row, w * 32u + j, ((bw >> j) & 1u) == 1u);
        LAT[row * p.rp + w * 32u + j] = bitcast<u32>(f32(q) / 32767.0);
    }
}

var<workgroup> red: array<f32, 256>;
var<workgroup> redn: array<u32, 256>;

// Row statistics: R_i <- b2·R_i + (1-b2)·mean_{k<r} G_ik² (non-finite = 0,
// counted).  One workgroup per row.
@compute @workgroup_size(256)
fn g_rowsq(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row = wg.x;
    var acc = 0.0;
    var bad = 0u;
    for (var k = t; k < p.r; k = k + 256u) {
        let v = G[row * p.rp + k];
        if (fin(v)) { acc = acc + v * v; } else { bad = bad + 1u; }
    }
    red[t] = acc;
    redn[t] = bad;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            red[t] = red[t] + red[t + s];
            redn[t] = redn[t] + redn[t + s];
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        VROW[row] = p.b2 * VROW[row] + (1.0 - p.b2) * (red[0] / f32(p.r));
        if (redn[0] > 0u) { atomicAdd(&CNT[40], redn[0]); }
    }
}

// Column partial sums of G² over 512-row chunks: PART[chunk][col].
@compute @workgroup_size(256)
fn g_colsq(@builtin(global_invocation_id) g: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let col = g.x;
    if (col >= p.rp) { return; }
    let r0 = wg.y * 512u;
    let r1 = min(r0 + 512u, p.rows);
    var acc = 0.0;
    for (var row = r0; row < r1; row = row + 1u) {
        let v = G[row * p.rp + col];
        if (fin(v)) { acc = acc + v * v; }
    }
    PART[wg.y * p.rp + col] = acc;
}

// C_k <- b2·C_k + (1-b2)·mean_i G_ik² for k < r.
@compute @workgroup_size(256)
fn g_colfin(@builtin(global_invocation_id) g: vec3<u32>) {
    let col = g.x;
    if (col >= p.r) { return; }
    var s = 0.0;
    for (var c = 0u; c < p.chunks; c = c + 1u) { s = s + PART[c * p.rp + col]; }
    VCOL[col] = p.b2 * VCOL[col] + (1.0 - p.b2) * (s / f32(p.rows));
}

// SCAL[0] = mean_{k<r} C_k.  One workgroup.
@compute @workgroup_size(256)
fn g_cmean(@builtin(local_invocation_index) t: u32) {
    var acc = 0.0;
    for (var k = t; k < p.r; k = k + 256u) { acc = acc + VCOL[k]; }
    red[t] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        workgroupBarrier();
    }
    if (t == 0u) { SCAL[0] = red[0] / f32(p.r); }
}

var<workgroup> wflip: atomic<u32>;
var<workgroup> wsat: atomic<u32>;
var<workgroup> wclp: atomic<u32>;

fn wg_flush(t: u32) {
    workgroupBarrier();
    if (t == 0u) {
        let f = atomicLoad(&wflip);
        if (f > 0u) { atomicAdd(&CNT[p.slot], f); }
        let s = atomicLoad(&wsat);
        if (s > 0u) { atomicAdd(&CNT[41], s); }
        let c = atomicLoad(&wclp);
        if (c > 0u) { atomicAdd(&CNT[42], c); }
    }
}
fn wg_reset(t: u32) {
    if (t == 0u) {
        atomicStore(&wflip, 0u);
        atomicStore(&wsat, 0u);
        atomicStore(&wclp, 0u);
    }
    workgroupBarrier();
}

// Factored-second-moment step (no momentum).  One thread per (row, word).
@compute @workgroup_size(256)
fn sign_upd(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    wg_reset(t);
    let i = idx(g);
    if (i < p.rows * p.wpr) {
        let row = i / p.wpr;
        let w = i % p.wpr;
        let live = live_mask(w);
        if (live != 0u) {
            let mm = SCAL[0];
            let inv = 1.0 / (mm * p.bc2);
            let epsd = p.eps * sqrt(mm / p.bc2);
            let rv = VROW[row];
            let base = pcg(p.seed ^ pcg(p.step ^ pcg(p.id ^ 0x2545f491u)));
            let bw = BITS[i];
            var nb = bw;
            var sat = 0u;
            var clp = 0u;
            let half = p.rp / 2u;
            for (var j = 0u; j < 32u; j = j + 2u) {
                let li = row * half + w * 16u + j / 2u;
                let pw = LAT[li];
                var outw = 0u;
                for (var h = 0u; h < 2u; h = h + 1u) {
                    var q = half_of(pw, h);
                    let jb = j + h;
                    if (((live >> jb) & 1u) == 1u) {
                        let k = w * 32u + jb;
                        let e = row * p.rp + k;
                        var gv = G[e];
                        if (!fin(gv)) { gv = 0.0; }
                        let den = sqrt(max(rv * VCOL[k] * inv, 0.0)) + epsd;
                        var u = 0.0;
                        if (den > 0.0) { u = gv / den; }
                        if (!fin(u)) { u = 0.0; }
                        if (abs(u) > p.clampv) { u = clamp(u, -p.clampv, p.clampv); clp = clp + 1u; }
                        let st = qstep(q, u, pcg(base ^ pcg(e)));
                        q = st.x;
                        sat = sat + u32(st.y);
                        let b = newbit(q, (bw >> jb) & 1u);
                        nb = (nb & ~(1u << jb)) | (b << jb);
                    }
                    outw = outw | ((bitcast<u32>(q) & 0xffffu) << (16u * h));
                }
                LAT[li] = outw;
            }
            BITS[i] = nb;
            let f = countOneBits(bw ^ nb);
            if (f > 0u) { atomicAdd(&wflip, f); }
            if (sat > 0u) { atomicAdd(&wsat, sat); }
            if (clp > 0u) { atomicAdd(&wclp, clp); }
        }
    }
    wg_flush(t);
}

// The same with a bf16 first moment (MOM, two per u32, the latent layout).
@compute @workgroup_size(256)
fn sign_upd_mom(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    wg_reset(t);
    let i = idx(g);
    if (i < p.rows * p.wpr) {
        let row = i / p.wpr;
        let w = i % p.wpr;
        let live = live_mask(w);
        if (live != 0u) {
            let mm = SCAL[0];
            let inv = 1.0 / (mm * p.bc2);
            let epsd = p.eps * sqrt(mm / p.bc2);
            let rv = VROW[row];
            let base = pcg(p.seed ^ pcg(p.step ^ pcg(p.id ^ 0x2545f491u)));
            let bw = BITS[i];
            var nb = bw;
            var sat = 0u;
            var clp = 0u;
            let half = p.rp / 2u;
            for (var j = 0u; j < 32u; j = j + 2u) {
                let li = row * half + w * 16u + j / 2u;
                let pw = LAT[li];
                let mw = MOM[li];
                var outw = 0u;
                var outm = 0u;
                for (var h = 0u; h < 2u; h = h + 1u) {
                    var q = half_of(pw, h);
                    let jb = j + h;
                    var mb = (mw >> (16u * h)) & 0xffffu;
                    if (((live >> jb) & 1u) == 1u) {
                        let k = w * 32u + jb;
                        let e = row * p.rp + k;
                        var gv = G[e];
                        if (!fin(gv)) { gv = 0.0; }
                        let mn = p.b1 * bitcast<f32>(mb << 16u) + (1.0 - p.b1) * gv;
                        mb = bf16(mn);
                        let den = sqrt(max(rv * VCOL[k] * inv, 0.0)) + epsd;
                        var u = 0.0;
                        if (den > 0.0) { u = (mn / p.bc1) / den; }
                        if (!fin(u)) { u = 0.0; }
                        if (abs(u) > p.clampv) { u = clamp(u, -p.clampv, p.clampv); clp = clp + 1u; }
                        let st = qstep(q, u, pcg(base ^ pcg(e)));
                        q = st.x;
                        sat = sat + u32(st.y);
                        let b = newbit(q, (bw >> jb) & 1u);
                        nb = (nb & ~(1u << jb)) | (b << jb);
                    }
                    outw = outw | ((bitcast<u32>(q) & 0xffffu) << (16u * h));
                    outm = outm | (mb << (16u * h));
                }
                LAT[li] = outw;
                MOM[li] = outm;
            }
            BITS[i] = nb;
            let f = countOneBits(bw ^ nb);
            if (f > 0u) { atomicAdd(&wflip, f); }
            if (sat > 0u) { atomicAdd(&wsat, sat); }
            if (clp > 0u) { atomicAdd(&wclp, clp); }
        }
    }
    wg_flush(t);
}

// Reference: f32 latent, Adam over the SmoothSign gradient G·κ·(1 − tanh²(κΛ)).
@compute @workgroup_size(256)
fn sign_upd_ref(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    wg_reset(t);
    let i = idx(g);
    if (i < p.rows * p.wpr) {
        let row = i / p.wpr;
        let w = i % p.wpr;
        let live = live_mask(w);
        if (live != 0u) {
            let bw = BITS[i];
            var nb = bw;
            for (var jb = 0u; jb < 32u; jb = jb + 1u) {
                if (((live >> jb) & 1u) == 1u) {
                    let e = row * p.rp + w * 32u + jb;
                    var lam = bitcast<f32>(LAT[e]);
                    var gv = G[e];
                    if (!fin(gv)) { gv = 0.0; }
                    let th = tanh(p.kss * lam);
                    let gs = gv * p.kss * (1.0 - th * th);
                    let m = p.b1 * bitcast<f32>(MOM[e]) + (1.0 - p.b1) * gs;
                    let v = p.b2 * VV[e] + (1.0 - p.b2) * gs * gs;
                    MOM[e] = bitcast<u32>(m);
                    VV[e] = v;
                    let den = sqrt(v / p.bc2) + 1e-30;
                    lam = lam - p.eta * (m / p.bc1) / den;
                    LAT[e] = bitcast<u32>(lam);
                    var b = select(0u, 1u, lam >= 0.0);
                    if (p.hyst > 0.0) {
                        b = (bw >> jb) & 1u;
                        if (lam > p.hyst) { b = 1u; }
                        if (lam < -p.hyst) { b = 0u; }
                    }
                    nb = (nb & ~(1u << jb)) | (b << jb);
                }
            }
            BITS[i] = nb;
            let f = countOneBits(bw ^ nb);
            if (f > 0u) { atomicAdd(&wflip, f); }
        }
    }
    wg_flush(t);
}

// CNT[slot] += popcount((BITS ^ BITS0) & live)  (distinct flips of one matrix).
@compute @workgroup_size(256)
fn bits_diff(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    if (t == 0u) { atomicStore(&wflip, 0u); }
    workgroupBarrier();
    let i = idx(g);
    if (i < p.rows * p.wpr) {
        let f = countOneBits((BITS[i] ^ BITS0[i]) & live_mask(i % p.wpr));
        if (f > 0u) { atomicAdd(&wflip, f); }
    }
    workgroupBarrier();
    if (t == 0u) {
        let f = atomicLoad(&wflip);
        if (f > 0u) { atomicAdd(&CNT[p.slot], f); }
    }
}

// GOUT[0..rows·rp] = G (deferred mode: keep one matrix's gradient).
@compute @workgroup_size(256)
fn g_copy(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.rows * p.rp) { return; }
    GOUT[i] = G[i];
}
"#;

fn module(c: &Ctx) -> Option<&'static train_ops::Mod> {
    train_ops::module(
        c,
        "train_sign",
        SRC,
        &["lat_init", "lat_init_f32", "g_rowsq", "g_colsq", "g_colfin", "g_cmean", "sign_upd", "sign_upd_mom", "sign_upd_ref", "bits_diff", "g_copy"],
    )
}

// `sign_upd_mom` with one 32-lane subgroup per bitmap word and one lane per
// sign: the gradient, latent and moment reads are coalesced (the original's
// thread walks its word's 32 entries, lanes 128 B apart), the word comes back
// with a ballot, the int16 pairs with one shuffle. Per entry the arithmetic
// and the rounding stream are those of `sign_upd_mom`.
const MOM2_SRC: &str = r#"
@compute @workgroup_size(256)
fn sign_upd_mom2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32, @builtin(subgroup_invocation_id) lane: u32) {
    wg_reset(t);
    let i = (wg.x + wg.y * 65535u) * 8u + t / 32u;
    let valid = i < p.rows * p.wpr;
    let ii = select(0u, i, valid);
    let row = ii / p.wpr;
    let w = ii % p.wpr;
    let live = select(0u, live_mask(w), valid);
    let k = w * 32u + lane;
    let e = row * p.rp + k;
    let li = e / 2u;
    let h = lane & 1u;
    let pw = LAT[li];
    let mw = MOM[li];
    var q = half_of(pw, h);
    var mb = (mw >> (16u * h)) & 0xffffu;
    let bw = BITS[ii];
    let ob = (bw >> lane) & 1u;
    var nbit = ob;
    var sat = 0u;
    var clp = 0u;
    if (((live >> lane) & 1u) == 1u) {
        let mm = SCAL[0];
        let inv = 1.0 / (mm * p.bc2);
        let epsd = p.eps * sqrt(mm / p.bc2);
        let rv = VROW[row];
        let base = pcg(p.seed ^ pcg(p.step ^ pcg(p.id ^ 0x2545f491u)));
        var gv = G[e];
        if (!fin(gv)) { gv = 0.0; }
        let mn = p.b1 * bitcast<f32>(mb << 16u) + (1.0 - p.b1) * gv;
        mb = bf16(mn);
        let den = sqrt(max(rv * VCOL[k] * inv, 0.0)) + epsd;
        var u = 0.0;
        if (den > 0.0) { u = (mn / p.bc1) / den; }
        if (!fin(u)) { u = 0.0; }
        if (abs(u) > p.clampv) { u = clamp(u, -p.clampv, p.clampv); clp = 1u; }
        let st = qstep(q, u, pcg(base ^ pcg(e)));
        q = st.x;
        sat = u32(st.y);
        nbit = newbit(q, ob);
    }
    let qb = bitcast<u32>(q) & 0xffffu;
    let qo = subgroupShuffleXor(qb, 1u);
    let mo = subgroupShuffleXor(mb, 1u);
    let nw = subgroupBallot(nbit == 1u).x;
    let ns = subgroupAdd(sat);
    let nc = subgroupAdd(clp);
    if (live != 0u) {
        if (h == 0u) {
            LAT[li] = qb | (qo << 16u);
            MOM[li] = mb | (mo << 16u);
        }
        if (lane == 0u) {
            BITS[ii] = nw;
            let f = countOneBits(bw ^ nw);
            if (f > 0u) { atomicAdd(&wflip, f); }
            if (ns > 0u) { atomicAdd(&wsat, ns); }
            if (nc > 0u) { atomicAdd(&wclp, nc); }
        }
    }
    wg_flush(t);
}
"#;

/// The subgroup module (`sign_upd_mom2`), where 32-wide subgroups exist.
fn module_sg(c: &Ctx) -> Option<&'static train_ops::Mod> {
    if !c.device.features().contains(wgpu::Features::SUBGROUP)
        || c.adapter_info.subgroup_min_size != 32
        || c.adapter_info.subgroup_max_size != 32
        || std::env::var("CMF_SIGN_SG").as_deref() == Ok("0")
    {
        return None;
    }
    static SRC2: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    let src = SRC2.get_or_init(|| {
        let head = &SRC[..SRC.find("// int16 latents from the bitmap").expect("train_sign layout")];
        let a = SRC.find("var<workgroup> wflip").expect("train_sign layout");
        let b = SRC.find("// Factored-second-moment step").expect("train_sign layout");
        Box::leak(format!("{head}{}{MOM2_SRC}", &SRC[a..b]).into_boxed_str())
    });
    train_ops::module(c, "train_sign_sg", src, &["sign_upd_mom2"])
}

/// Roles of the binary low-rank linears (flip counters are kept per role and side).
pub const ROLES: [&str; 10] = ["gdn.qkv", "gdn.z", "gdn.out", "attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down"];

/// Counter slots: 2·role + side (0 = U, 1 = V) for the step's flips, then
/// these.
pub const CNT_NONFINITE: usize = 40;
pub const CNT_SATURATED: usize = 41;
pub const CNT_CLAMPED: usize = 42;
pub const CNT_LEN: usize = 64;

pub fn role_id(role: &str) -> u32 {
    ROLES.iter().position(|r| *r == role).unwrap_or(0) as u32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignOpt {
    Fact,
    FactMom,
    Ref,
}

impl SignOpt {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "fact" => Some(SignOpt::Fact),
            "fact-mom" => Some(SignOpt::FactMom),
            "ref" => Some(SignOpt::Ref),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            SignOpt::Fact => "fact",
            SignOpt::FactMom => "fact-mom",
            SignOpt::Ref => "ref",
        }
    }
    /// Optimizer state per sign, bytes (latent + moments).
    pub fn bytes_per_sign(self) -> usize {
        match self {
            SignOpt::Fact => 2,
            SignOpt::FactMom => 4,
            SignOpt::Ref => 12,
        }
    }
}

/// How the latents start.
#[derive(Clone, Copy, Debug)]
pub struct SignInit {
    pub opt: SignOpt,
    /// RMS of |Λ| at the start (magnitudes τ·|z|, z ~ N(0,1), sign from the bit).
    pub tau: f32,
    pub seed: u32,
    /// Keep a copy of the starting bits (distinct-flip statistics).
    pub keep_bits0: bool,
    /// Allocate a per-matrix gradient buffer (deferred updates; tests).
    pub defer: bool,
}

/// Optimizer state of one sign matrix (`rows × rp`, live ranks `r_true`).
pub struct SignMat {
    pub rows: usize,
    pub r_true: usize,
    pub rp: usize,
    pub id: u32,
    pub opt: SignOpt,
    /// int16 pairs `[rows][rp/2]` (fact, fact-mom) or f32 `[rows][rp]` (ref).
    pub lat: wgpu::Buffer,
    /// bf16 pairs (fact-mom) or f32 (ref) first moment.
    pub mom: Option<wgpu::Buffer>,
    /// f32 per-element second moment (ref).
    pub v: Option<wgpu::Buffer>,
    /// Factored second moment: row EMA `[rows]`, column EMA `[rp]`.
    pub vrow: wgpu::Buffer,
    pub vcol: wgpu::Buffer,
    pub bits0: Option<wgpu::Buffer>,
    pub gsave: Option<wgpu::Buffer>,
}

impl SignMat {
    pub fn wpr(&self) -> usize {
        self.rp / 32
    }
    pub fn live(&self) -> u64 {
        (self.rows * self.r_true) as u64
    }
    /// Words of the latent buffer.
    pub fn lat_words(&self) -> usize {
        match self.opt {
            SignOpt::Ref => self.rows * self.rp,
            _ => self.rows * self.rp / 2,
        }
    }
    pub fn mom_words(&self) -> usize {
        self.lat_words()
    }
}

/// Sign state of one path: U (`m` rows) and V (`n` rows).
pub struct SignPath {
    pub u: SignMat,
    pub v: SignMat,
}

/// Workspace shared by every matrix (one matrix is updated at a time).
pub struct SignScratch {
    /// f32 gradient `[pad_rows(max_rows)][rp_max]`.
    pub g: wgpu::Buffer,
    /// f16 transposed activation operand `[pad_rows(max_rows)][kpad]`.
    pub t_big: wgpu::Buffer,
    /// f16 transposed plane operand `[rp_max][kpad]`.
    pub t_small: wgpu::Buffer,
    pub part: wgpu::Buffer,
    pub scal: wgpu::Buffer,
    /// u32[64] counters (see the CNT_* slots).
    pub cnt: wgpu::Buffer,
    pub kpad: usize,
    pub max_rows: usize,
    pub rp_max: usize,
}

fn storage(c: &Ctx, words: usize) -> wgpu::Buffer {
    train_ops::zeros_u32(c, words)
}

impl SignScratch {
    pub fn new(c: &Ctx, ntok: usize, max_rows: usize, rp_max: usize) -> Self {
        let kpad = ntok.div_ceil(32) * 32;
        let pr = super::train_mm::pad_rows(max_rows);
        SignScratch {
            g: storage(c, pr * rp_max),
            t_big: super::train_mm::f16_buf(c, pr * kpad),
            t_small: super::train_mm::f16_buf(c, rp_max * kpad),
            part: storage(c, max_rows.div_ceil(512) * rp_max),
            scal: storage(c, 4),
            cnt: storage(c, CNT_LEN),
            kpad,
            max_rows,
            rp_max,
        }
    }
    /// Device bytes of this scratch.
    pub fn bytes(&self) -> u64 {
        self.g.size() + self.t_big.size() + self.t_small.size() + self.part.size() + self.scal.size() + self.cnt.size()
    }
}

/// One sign step's settings (shared by every matrix of the step).
#[derive(Clone, Copy, Debug)]
pub struct SignStepCfg {
    pub opt: SignOpt,
    pub eta: f32,
    /// 1-based sign step (bias corrections, rounding stream).
    pub step: u32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub clamp: f32,
    pub hyst: f32,
    /// SmoothSign sharpness κ of the reference optimizer.
    pub k_ss: f32,
    pub seed: u32,
    /// Keep each matrix's gradient and apply the updates after the sweep (tests).
    pub defer: bool,
}

/// What `blr_backward` needs to append a path's sign ops.
pub struct SignCtx<'a> {
    pub sc: &'a SignScratch,
    pub cfg: &'a SignStepCfg,
    pub role: u32,
}

fn uni(mat: &SignMat, cfg: Option<&SignStepCfg>, slot: u32, tau: f32, seed: u32) -> Vec<u32> {
    let (eta, step, b1, b2, eps, clampv, hyst, kss, bc1, bc2) = match cfg {
        Some(c) => {
            let bc1 = 1.0 - (c.b1 as f64).powi(c.step as i32);
            let bc2 = 1.0 - (c.b2 as f64).powi(c.step as i32);
            (c.eta, c.step, c.b1, c.b2, c.eps, c.clamp, c.hyst, c.k_ss, bc1 as f32, bc2 as f32)
        }
        None => (0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    };
    let seed = cfg.map(|c| c.seed).unwrap_or(seed);
    vec![
        mat.rows as u32,
        mat.r_true as u32,
        mat.rp as u32,
        mat.wpr() as u32,
        mat.id,
        seed,
        step,
        slot,
        eta.to_bits(),
        b1.to_bits(),
        b2.to_bits(),
        eps.to_bits(),
        clampv.to_bits(),
        hyst.to_bits(),
        kss.to_bits(),
        bc2.to_bits(),
        bc1.to_bits(),
        mat.rows.div_ceil(512) as u32,
        tau.to_bits(),
        0,
    ]
}

/// Sign state for the bitmap `bits` (`[rows][rp/32]`), latents drawn from it.
#[allow(clippy::too_many_arguments)]
pub fn init_mat(c: &Ctx, bits: &wgpu::Buffer, rows: usize, r_true: usize, rp: usize, init: &SignInit, id: u32) -> Option<SignMat> {
    let md = module(c)?;
    let opt = init.opt;
    let lat_words = if opt == SignOpt::Ref { rows * rp } else { rows * rp / 2 };
    let mat = SignMat {
        rows,
        r_true,
        rp,
        id,
        opt,
        lat: storage(c, lat_words),
        mom: (opt != SignOpt::Fact).then(|| storage(c, lat_words)),
        v: (opt == SignOpt::Ref).then(|| storage(c, rows * rp)),
        vrow: storage(c, rows),
        vcol: storage(c, rp),
        bits0: init.keep_bits0.then(|| {
            let b = storage(c, rows * rp / 32);
            let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bits0") });
            enc.copy_buffer_to_buffer(bits, 0, &b, 0, (rows * rp / 32 * 4) as u64);
            super::submit(c, enc.finish());
            b
        }),
        gsave: init.defer.then(|| storage(c, rows * rp)),
    };
    let ep = if opt == SignOpt::Ref { "lat_init_f32" } else { "lat_init" };
    let d = Disp { pipe: &md.p[ep], uni: uni(&mat, None, 0, init.tau, init.seed), bufs: vec![(2, &mat.lat), (7, bits)], grid: elt_fold(rows * rp / 32) };
    train_ops::run(c, &[d]);
    Some(mat)
}

/// The ops of one sign update of `mat` from the gradient `g` (`[rows][rp]`
/// f32): statistics, then the latent and bitmap step; flips counted in
/// counter `slot`.
pub fn update_ops<'a>(c: &'a Ctx, cfg: &SignStepCfg, sc: &'a SignScratch, mat: &'a SignMat, g: &'a wgpu::Buffer, bits: &'a wgpu::Buffer, slot: u32) -> Option<Vec<Op<'a>>> {
    let md = module(c)?;
    let u = uni(mat, Some(cfg), slot, 0.0, 0);
    let (rows, rp) = (mat.rows, mat.rp);
    let mut ops = Vec::with_capacity(5);
    let words = rows * rp / 32;
    if mat.opt == SignOpt::Ref {
        let (mom, v) = (mat.mom.as_ref()?, mat.v.as_ref()?);
        ops.push(Op::D(Disp {
            pipe: &md.p["sign_upd_ref"],
            uni: u,
            bufs: vec![(1, g), (2, &mat.lat), (3, mom), (4, v), (7, bits), (10, &sc.cnt)],
            grid: elt_fold(words),
        }));
        return Some(ops);
    }
    ops.push(Op::D(Disp { pipe: &md.p["g_rowsq"], uni: u.clone(), bufs: vec![(1, g), (5, &mat.vrow), (10, &sc.cnt)], grid: (rows as u32, 1, 1) }));
    ops.push(Op::D(Disp {
        pipe: &md.p["g_colsq"],
        uni: u.clone(),
        bufs: vec![(1, g), (8, &sc.part)],
        grid: (rp.div_ceil(256) as u32, rows.div_ceil(512) as u32, 1),
    }));
    ops.push(Op::D(Disp { pipe: &md.p["g_colfin"], uni: u.clone(), bufs: vec![(6, &mat.vcol), (8, &sc.part)], grid: (rp.div_ceil(256) as u32, 1, 1) }));
    ops.push(Op::D(Disp { pipe: &md.p["g_cmean"], uni: u.clone(), bufs: vec![(6, &mat.vcol), (9, &sc.scal)], grid: (1, 1, 1) }));
    let mut bufs = vec![(1, g), (2, &mat.lat), (5, &mat.vrow), (6, &mat.vcol), (7, bits), (9, &sc.scal), (10, &sc.cnt)];
    if mat.opt == SignOpt::FactMom {
        bufs.push((3, mat.mom.as_ref()?));
        if let Some(m2) = module_sg(c) {
            ops.push(Op::D(Disp { pipe: &m2.p["sign_upd_mom2"], uni: u, bufs, grid: train_ops::wg_fold(words.div_ceil(8)) }));
            return Some(ops);
        }
    }
    let ep = if mat.opt == SignOpt::FactMom { "sign_upd_mom" } else { "sign_upd" };
    ops.push(Op::D(Disp { pipe: &md.p[ep], uni: u, bufs, grid: elt_fold(words) }));
    Some(ops)
}

/// Deferred mode: copy the scratch gradient into the matrix's own buffer.
pub fn save_grad_op<'a>(c: &'a Ctx, sc: &'a SignScratch, mat: &'a SignMat) -> Option<Op<'a>> {
    let md = module(c)?;
    Some(Op::D(Disp {
        pipe: &md.p["g_copy"],
        uni: uni(mat, None, 0, 0.0, 0),
        bufs: vec![(1, &sc.g), (12, mat.gsave.as_ref()?)],
        grid: elt_fold(mat.rows * mat.rp),
    }))
}

/// Ops of the sign update of one matrix (in place: from the scratch
/// gradient; deferred: save it instead).
pub fn grad_consumer_ops<'a>(c: &'a Ctx, ctx: &SignCtx<'a>, mat: &'a SignMat, bits: &'a wgpu::Buffer, side: u32) -> Option<Vec<Op<'a>>> {
    if ctx.cfg.defer {
        return Some(vec![save_grad_op(c, ctx.sc, mat)?]);
    }
    update_ops(c, ctx.cfg, ctx.sc, mat, &ctx.sc.g, bits, 2 * ctx.role + side)
}

/// Deferred mode: apply one matrix's update from its saved gradient.
pub fn apply_deferred(c: &Ctx, cfg: &SignStepCfg, sc: &SignScratch, mat: &SignMat, bits: &wgpu::Buffer, slot: u32) -> Option<()> {
    let g = mat.gsave.as_ref()?;
    let ops = update_ops(c, cfg, sc, mat, g, bits, slot)?;
    super::train_mm::run(c, &ops);
    Some(())
}

/// `diff[slot] += distinct flips of `bits` against the matrix's starting bits.
pub fn bits_diff_op<'a>(c: &'a Ctx, mat: &'a SignMat, bits: &'a wgpu::Buffer, diff: &'a wgpu::Buffer, slot: u32) -> Option<Disp<'a>> {
    let md = module(c)?;
    Some(Disp {
        pipe: &md.p["bits_diff"],
        uni: uni(mat, None, slot, 0.0, 0),
        bufs: vec![(7, bits), (10, diff), (11, mat.bits0.as_ref()?)],
        grid: elt_fold(mat.rows * mat.rp / 32),
    })
}

/// Read the step counters and zero them.
pub fn read_reset_counters(c: &Ctx, sc: &SignScratch) -> Option<[u32; CNT_LEN]> {
    let v = train_ops::download_u32(c, &sc.cnt, CNT_LEN)?;
    reset_counters(c, sc);
    let mut out = [0u32; CNT_LEN];
    out.copy_from_slice(&v);
    Some(out)
}

pub fn reset_counters(c: &Ctx, sc: &SignScratch) {
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sign_cnt") });
    let mut mk = super::train_prof::enc_begin(c, &mut enc);
    enc.clear_buffer(&sc.cnt, 0, None);
    super::train_prof::enc_mark(c, &mut enc, &mut mk, "clear");
    super::submit(c, enc.finish());
}

// ───────────────────────────── host mirrors ─────────────────────────────

/// The shader's PCG hash.
pub fn pcg(v: u32) -> u32 {
    let s = v.wrapping_mul(747796405).wrapping_add(2891336453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277803737);
    (w >> 22) ^ w
}

/// The shader's 24-bit uniform in [0, 1).
pub fn unif24(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16777216.0)
}

/// Rounding stream base of one matrix at one step (the shader's `base`).
pub fn round_base(seed: u32, step: u32, id: u32) -> u32 {
    pcg(seed ^ pcg(step ^ pcg(id ^ 0x2545_f491)))
}

/// bf16 bits of `v`, round to nearest even (the shader's `bf16`).
pub fn bf16_rne(v: f32) -> u32 {
    let b = v.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16
}

/// Host replica of the factored statistics of one update: the row and
/// column EMAs of G² (non-finite entries 0) and their column mean M.
/// Returns (M, non-finite count).
pub fn host_stats(cfg: &SignStepCfg, rows: usize, r: usize, rp: usize, g: &[f32], vrow: &mut [f32], vcol: &mut [f32]) -> (f32, u64) {
    let mut nonfin = 0u64;
    for i in 0..rows {
        let mut s = 0f64;
        for k in 0..r {
            let v = g[i * rp + k];
            if v.is_finite() {
                s += (v as f64) * (v as f64);
            } else {
                nonfin += 1;
            }
        }
        vrow[i] = cfg.b2 * vrow[i] + (1.0 - cfg.b2) * (s / r as f64) as f32;
    }
    for k in 0..r {
        let mut s = 0f64;
        for i in 0..rows {
            let v = g[i * rp + k];
            if v.is_finite() {
                s += (v as f64) * (v as f64);
            }
        }
        vcol[k] = cfg.b2 * vcol[k] + (1.0 - cfg.b2) * (s / rows as f64) as f32;
    }
    ((vcol[..r].iter().map(|&x| x as f64).sum::<f64>() / r as f64) as f32, nonfin)
}

/// Host replica of the `fact`/`fact-mom` latent and bitmap step from given
/// statistics (f32 math as the shader does it); `lat`/`mom` hold one int16
/// / bf16 value per entry `[rows][rp]`. Returns (flips, saturated, clamped).
#[allow(clippy::too_many_arguments)]
pub fn host_step_fact(
    cfg: &SignStepCfg,
    id: u32,
    rows: usize,
    r: usize,
    rp: usize,
    g: &[f32],
    lat: &mut [i32],
    mut mom: Option<&mut [u32]>,
    vrow: &[f32],
    vcol: &[f32],
    mm: f32,
    bits: &mut [u32],
) -> (u64, u64, u64) {
    let bc1 = (1.0 - (cfg.b1 as f64).powi(cfg.step as i32)) as f32;
    let bc2 = (1.0 - (cfg.b2 as f64).powi(cfg.step as i32)) as f32;
    let inv = 1.0 / (mm * bc2);
    let epsd = cfg.eps * (mm / bc2).sqrt();
    let base = round_base(cfg.seed, cfg.step, id);
    let wpr = rp / 32;
    let (mut flips, mut sat, mut clp) = (0u64, 0u64, 0u64);
    for i in 0..rows {
        for w in 0..wpr {
            let old = bits[i * wpr + w];
            let mut nb = old;
            for jb in 0..32 {
                let k = w * 32 + jb;
                if k >= r {
                    continue;
                }
                let e = i * rp + k;
                let gv = if g[e].is_finite() { g[e] } else { 0.0 };
                let den = (vrow[i] * vcol[k] * inv).max(0.0).sqrt() + epsd;
                let num = match mom.as_deref_mut() {
                    Some(m) => {
                        let mn = cfg.b1 * f32::from_bits(m[e] << 16) + (1.0 - cfg.b1) * gv;
                        m[e] = bf16_rne(mn);
                        mn / bc1
                    }
                    None => gv,
                };
                let mut u = if den > 0.0 { num / den } else { 0.0 };
                if !u.is_finite() {
                    u = 0.0;
                }
                if u.abs() > cfg.clamp {
                    u = u.clamp(-cfg.clamp, cfg.clamp);
                    clp += 1;
                }
                let delta = -32767.0 * cfg.eta * u;
                let fl = delta.floor();
                let fr = delta - fl;
                let mut qn = lat[e] + fl as i32 + if unif24(pcg(base ^ pcg(e as u32))) < fr { 1 } else { 0 };
                if !(-32767..=32767).contains(&qn) {
                    qn = qn.clamp(-32767, 32767);
                    sat += 1;
                }
                lat[e] = qn;
                let ob = (old >> jb) & 1;
                let b = if cfg.hyst <= 0.0 {
                    (qn >= 0) as u32
                } else {
                    let th = (32767.0 * cfg.hyst) as i32;
                    if qn > th {
                        1
                    } else if qn < -th {
                        0
                    } else {
                        ob
                    }
                };
                nb = (nb & !(1 << jb)) | (b << jb);
            }
            bits[i * wpr + w] = nb;
            flips += (old ^ nb).count_ones() as u64;
        }
    }
    (flips, sat, clp)
}

/// Unpack int16 latent pairs `[rows][rp/2]` into one value per entry.
pub fn unpack_i16(words: &[u32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(words.len() * 2);
    for &w in words {
        out.push((w as u16) as i16 as i32);
        out.push(((w >> 16) as u16) as i16 as i32);
    }
    out
}

/// Pack one int16 value per entry into pairs.
pub fn pack_i16(v: &[i32]) -> Vec<u32> {
    v.chunks(2).map(|p| ((p[0] as i16 as u16) as u32) | (((p[1] as i16 as u16) as u32) << 16)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::train_ops::{download, download_u32, upload, zeros_u32};

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

    fn ub(c: &Ctx, v: &[u32]) -> wgpu::Buffer {
        let b = zeros_u32(c, v.len());
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(v));
        b
    }

    /// Host f64 check of the sign-gradient formulas themselves: the central
    /// difference of L = ½‖y − y*‖² with each sign treated as a real number
    /// (L is quadratic in each entry, so the difference is exact up to
    /// rounding) against G_U = diag(h)·dyᵀ·a·diag(ℓ), G_V = diag(g)·xᵀ·b·diag(ℓ).
    #[test]
    fn sign_grad_reference_fd() {
        let (m, n, r, ntok) = (12usize, 10usize, 6usize, 7usize);
        let x: Vec<f64> = rnd(ntok * n, 1, 1.0).iter().map(|&v| v as f64).collect();
        let ys: Vec<f64> = rnd(ntok * m, 2, 1.0).iter().map(|&v| v as f64).collect();
        let h: Vec<f64> = rnd(m, 3, 0.5).iter().map(|&v| 0.5 + v.abs() as f64).collect();
        let l: Vec<f64> = rnd(r, 4, 0.5).iter().map(|&v| 0.5 + v.abs() as f64).collect();
        let g: Vec<f64> = rnd(n, 5, 0.5).iter().map(|&v| 0.5 + v.abs() as f64).collect();
        let sgn = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
        let su: Vec<f64> = rnd(m * r, 6, 1.0).iter().map(|&v| sgn(v)).collect();
        let sv: Vec<f64> = rnd(n * r, 7, 1.0).iter().map(|&v| sgn(v)).collect();
        let fwd = |su: &[f64], sv: &[f64]| -> (f64, Vec<f64>, Vec<f64>) {
            let mut a = vec![0f64; ntok * r];
            let mut y = vec![0f64; ntok * m];
            for t in 0..ntok {
                for k in 0..r {
                    a[t * r + k] = (0..n).map(|j| x[t * n + j] * g[j] * sv[j * r + k]).sum();
                }
                for i in 0..m {
                    y[t * m + i] = h[i] * (0..r).map(|k| a[t * r + k] * l[k] * su[i * r + k]).sum::<f64>();
                }
            }
            let loss = 0.5 * y.iter().zip(&ys).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
            (loss, a, y)
        };
        let (_, a, y) = fwd(&su, &sv);
        let dy: Vec<f64> = y.iter().zip(&ys).map(|(a, b)| a - b).collect();
        let mut b = vec![0f64; ntok * r];
        for t in 0..ntok {
            for k in 0..r {
                b[t * r + k] = (0..m).map(|i| dy[t * m + i] * h[i] * su[i * r + k]).sum();
            }
        }
        let gu = |i: usize, k: usize| h[i] * l[k] * (0..ntok).map(|t| dy[t * m + i] * a[t * r + k]).sum::<f64>();
        let gv = |j: usize, k: usize| g[j] * l[k] * (0..ntok).map(|t| x[t * n + j] * b[t * r + k]).sum::<f64>();
        let eps = 1e-3;
        let mut worst = 0f64;
        for i in 0..m {
            for k in 0..r {
                let mut p = su.clone();
                p[i * r + k] += eps;
                let lp = fwd(&p, &sv).0;
                p[i * r + k] -= 2.0 * eps;
                let lm = fwd(&p, &sv).0;
                let fd = (lp - lm) / (2.0 * eps);
                let an = gu(i, k);
                worst = worst.max((fd - an).abs() / an.abs().max(1e-3));
            }
        }
        for j in 0..n {
            for k in 0..r {
                let mut p = sv.clone();
                p[j * r + k] += eps;
                let lp = fwd(&su, &p).0;
                p[j * r + k] -= 2.0 * eps;
                let lm = fwd(&su, &p).0;
                let fd = (lp - lm) / (2.0 * eps);
                let an = gv(j, k);
                worst = worst.max((fd - an).abs() / an.abs().max(1e-3));
            }
        }
        eprintln!("sign gradient vs central difference: worst relative error {worst:e}");
        assert!(worst < 1e-8, "worst {worst:e}");
    }

    /// Latent init: sgn(Λ) = bit everywhere, RMS |Λ| = τ within 1 %, no
    /// zeros; pad ranks ±1 with the bit's sign.
    #[test]
    fn lat_init_consistent() {
        let Some(c) = train_ops::dev() else { return };
        let (rows, r, rp) = (4100usize, 250usize, 256usize);
        let wpr = rp / 32;
        let bits: Vec<u32> = rnd(rows * wpr, 9, 1.0).iter().map(|v| v.to_bits() ^ 0x5a5a_5a5a).collect();
        let bb = ub(c, &bits);
        for opt in [SignOpt::Fact, SignOpt::Ref] {
            let init = SignInit { opt, tau: 0.25, seed: 42, keep_bits0: false, defer: false };
            let mat = init_mat(c, &bb, rows, r, rp, &init, 7).expect("init");
            let q: Vec<f64> = match opt {
                SignOpt::Ref => download(c, &mat.lat, rows * rp).unwrap().iter().map(|&v| v as f64).collect(),
                _ => unpack_i16(&download_u32(c, &mat.lat, rows * rp / 2).unwrap()).iter().map(|&v| v as f64 / 32767.0).collect(),
            };
            let (mut ss, mut nl, mut zeros, mut bad) = (0f64, 0usize, 0usize, 0usize);
            for i in 0..rows {
                for k in 0..rp {
                    let v = q[i * rp + k];
                    let bit = (bits[i * wpr + k / 32] >> (k % 32)) & 1;
                    if v == 0.0 {
                        zeros += 1;
                    }
                    if (v >= 0.0) != (bit == 1) {
                        bad += 1;
                    }
                    if k < r {
                        ss += v * v;
                        nl += 1;
                    } else if (v.abs() * 32767.0 - 1.0).abs() > 1e-3 {
                        bad += 1;
                    }
                }
            }
            let rms = (ss / nl as f64).sqrt();
            eprintln!("{}: rms |Λ| {rms:.5} (τ 0.25), zeros {zeros}, sign mismatches {bad}", opt.name());
            assert_eq!(zeros, 0);
            assert_eq!(bad, 0);
            assert!((rms / 0.25 - 1.0).abs() < 0.01, "rms {rms}");
        }
    }

    /// One update of every optimizer on the device against the host replica:
    /// random G with injected NaN/inf, preset latents and moments.
    #[test]
    fn sign_update_matches_host() {
        let Some(c) = train_ops::dev() else { return };
        let (rows, r, rp) = (300usize, 200usize, 256usize);
        let wpr = rp / 32;
        let mut g = rnd(rows * rp, 11, 1e-3);
        for i in 0..rows {
            for k in r..rp {
                g[i * rp + k] = 0.0;
            }
        }
        // a heavy row and column so the factored scale matters, plus non-finite entries
        for k in 0..r {
            g[5 * rp + k] *= 20.0;
        }
        for i in 0..rows {
            g[i * rp + 17] *= 0.05;
        }
        g[3 * rp + 4] = f32::NAN;
        g[100 * rp + 150] = f32::INFINITY;
        g[200 * rp + 9] = f32::NEG_INFINITY;
        let lat0: Vec<i32> = rnd(rows * rp, 12, 1.0)
            .iter()
            .map(|&v| {
                let q = (v * 600.0) as i32;
                if q == 0 { 1 } else { q }
            })
            .collect();
        let mut lat0 = lat0;
        // pushed past ±32767 (clamped steps, saturation counted)
        lat0[7 * rp + 3] = 32767;
        lat0[8 * rp + 3] = -32767;
        g[7 * rp + 3] = -1.0;
        g[8 * rp + 3] = 1.0;
        let mut bits0 = vec![0u32; rows * wpr];
        for i in 0..rows {
            for k in 0..rp {
                if lat0[i * rp + k] >= 0 {
                    bits0[i * wpr + k / 32] |= 1 << (k % 32);
                }
            }
        }
        let vrow0: Vec<f32> = rnd(rows, 13, 1e-6).iter().map(|v| v.abs() + 1e-7).collect();
        let mut vcol0: Vec<f32> = rnd(rp, 14, 1e-6).iter().map(|v| v.abs() + 1e-7).collect();
        for v in vcol0.iter_mut().skip(r) {
            *v = 0.0;
        }
        let sc = SignScratch::new(c, 64, rows, rp);
        for opt in [SignOpt::Fact, SignOpt::FactMom, SignOpt::Ref] {
            let cfg = SignStepCfg { opt, eta: 1e-3, step: 7, b1: 0.9, b2: 0.999, eps: 1e-3, clamp: 4.0, hyst: 0.0, k_ss: 8.0, seed: 99, defer: false };
            let bb = ub(c, &bits0);
            let init = SignInit { opt, tau: 0.25, seed: 1, keep_bits0: true, defer: false };
            let mat = init_mat(c, &bb, rows, r, rp, &init, 5).expect("init");
            // preset the state
            let mom0: Vec<u32> = rnd(rows * rp, 15, 1e-4).iter().map(|&v| bf16_rne(v)).collect();
            let momf: Vec<f32> = rnd(rows * rp, 15, 1e-4);
            let vv0: Vec<f32> = rnd(rows * rp, 16, 1e-6).iter().map(|v| v * v).collect();
            let latf: Vec<f32> = lat0.iter().map(|&q| q as f32 / 32767.0).collect();
            match opt {
                SignOpt::Ref => {
                    c.queue.write_buffer(&mat.lat, 0, bytemuck::cast_slice(&latf));
                    c.queue.write_buffer(mat.mom.as_ref().unwrap(), 0, bytemuck::cast_slice(&momf));
                    c.queue.write_buffer(mat.v.as_ref().unwrap(), 0, bytemuck::cast_slice(&vv0));
                }
                _ => {
                    c.queue.write_buffer(&mat.lat, 0, bytemuck::cast_slice(&pack_i16(&lat0)));
                    if opt == SignOpt::FactMom {
                        let packed: Vec<u32> = mom0.chunks(2).map(|p| p[0] | (p[1] << 16)).collect();
                        c.queue.write_buffer(mat.mom.as_ref().unwrap(), 0, bytemuck::cast_slice(&packed));
                    }
                }
            }
            c.queue.write_buffer(&mat.vrow, 0, bytemuck::cast_slice(&vrow0));
            c.queue.write_buffer(&mat.vcol, 0, bytemuck::cast_slice(&vcol0));
            let gb = upload(c, &g);
            reset_counters(c, &sc);
            let ops = update_ops(c, &cfg, &sc, &mat, &gb, &bb, 3).expect("ops");
            super::super::train_mm::run(c, &ops);
            let cnt = read_reset_counters(c, &sc).unwrap();
            let bits_d = download_u32(c, &bb, rows * wpr).unwrap();
            let mut bits_h = bits0.clone();
            match opt {
                SignOpt::Ref => {
                    // host replica of the reference step
                    let mut lat = latf.clone();
                    let mut m1 = momf.clone();
                    let mut v2 = vv0.clone();
                    let bc1 = (1.0 - 0.9f64.powi(7)) as f32;
                    let bc2 = (1.0 - 0.999f64.powi(7)) as f32;
                    let mut flips = 0u64;
                    for i in 0..rows {
                        for w in 0..wpr {
                            let old = bits_h[i * wpr + w];
                            let mut nb = old;
                            for jb in 0..32 {
                                let k = w * 32 + jb;
                                if k >= r {
                                    continue;
                                }
                                let e = i * rp + k;
                                let gv = if g[e].is_finite() { g[e] } else { 0.0 };
                                let th = (cfg.k_ss * lat[e]).tanh();
                                let gs = gv * cfg.k_ss * (1.0 - th * th);
                                m1[e] = cfg.b1 * m1[e] + (1.0 - cfg.b1) * gs;
                                v2[e] = cfg.b2 * v2[e] + (1.0 - cfg.b2) * gs * gs;
                                lat[e] -= cfg.eta * (m1[e] / bc1) / ((v2[e] / bc2).sqrt() + 1e-30);
                                let b = (lat[e] >= 0.0) as u32;
                                nb = (nb & !(1 << jb)) | (b << jb);
                            }
                            bits_h[i * wpr + w] = nb;
                            flips += (old ^ nb).count_ones() as u64;
                        }
                    }
                    let lat_d = download(c, &mat.lat, rows * rp).unwrap();
                    let mut worst = 0f32;
                    let mut bit_bad = 0;
                    // GPU tanh/sqrt are a few ulp off: measure against the step taken,
                    // at least η (where the preset second moment is tiny the Adam step is
                    // hundreds of η). Metal's fast tanh loses relative precision of
                    // 1 − tanh² near saturation: 6e-5 on an M4, 1.6e-2 on the CI runner's
                    // GPU; the bits and the flip count must still match exactly.
                    let tol = if c.adapter_info.backend == wgpu::Backend::Metal { 5e-2 } else { 1e-2 };
                    for e in 0..rows * rp {
                        let step = (lat[e] - latf[e]).abs().max(cfg.eta);
                        worst = worst.max((lat_d[e] - lat[e]).abs() / step);
                    }
                    for (a, b) in bits_d.iter().zip(&bits_h) {
                        bit_bad += (a ^ b).count_ones();
                    }
                    eprintln!("ref: latent worst |Δ|/max(η, |step|) {worst:e}, bit mismatches {bit_bad}, flips dev {} host {flips}", cnt[3]);
                    assert!(worst < tol, "ref latent error {worst:e} ≥ {tol:e}");
                    assert_eq!(bit_bad, 0);
                    assert_eq!(cnt[3] as u64, flips);
                }
                _ => {
                    let mut lat = lat0.clone();
                    let mut vrow = vrow0.clone();
                    let mut vcol = vcol0.clone();
                    let mut mom = mom0.clone();
                    let (mm, nonfin) = host_stats(&cfg, rows, r, rp, &g, &mut vrow, &mut vcol);
                    let lat_d = unpack_i16(&download_u32(c, &mat.lat, rows * rp / 2).unwrap());
                    let vrow_d = download(c, &mat.vrow, rows).unwrap();
                    let vcol_d = download(c, &mat.vcol, rp).unwrap();
                    let mm_d = download(c, &sc.scal, 1).unwrap()[0];
                    // the step from the device's own statistics (the sums differ in the last bits)
                    let (flips, sat, clp) = host_step_fact(&cfg, 5, rows, r, rp, &g, &mut lat, (opt == SignOpt::FactMom).then_some(&mut mom[..]), &vrow_d, &vcol_d, mm_d, &mut bits_h);
                    let (mut neq, mut far) = (0usize, 0usize);
                    for e in 0..rows * rp {
                        if lat_d[e] != lat[e] {
                            neq += 1;
                            if (lat_d[e] - lat[e]).abs() > 1 {
                                far += 1;
                            }
                        }
                        let k = e % rp;
                        if k >= r {
                            assert_eq!(lat_d[e], lat0[e], "pad latent moved");
                        }
                    }
                    let relv = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(1e-30)).fold(0f32, f32::max);
                    let mut bit_bad = 0;
                    for (i, (a, b)) in bits_d.iter().zip(&bits_h).enumerate() {
                        bit_bad += (a ^ b).count_ones();
                        // bits equal Λ >= 0 on the device side
                        for jb in 0..32 {
                            let e = (i / wpr) * rp + (i % wpr) * 32 + jb;
                            assert_eq!((a >> jb) & 1, (lat_d[e] >= 0) as u32, "bit != (q >= 0) at {e}");
                        }
                    }
                    let fl_d: u64 = bits_d.iter().zip(&bits0).map(|(a, b)| (a ^ b).count_ones() as u64).sum();
                    eprintln!(
                        "{}: latents differing {neq} (|Δ|>1: {far}) of {}, R rel {:e}, C rel {:e}, M dev {mm_d:e} host {mm:e}, flips dev {} host {flips} popcount {fl_d}, sat {}/{sat} clamp {}/{clp} nonfinite {}/{nonfin}, bit mismatches {bit_bad}",
                        opt.name(),
                        rows * rp,
                        relv(&vrow_d, &vrow),
                        relv(&vcol_d[..r], &vcol[..r]),
                        cnt[3],
                        cnt[CNT_SATURATED],
                        cnt[CNT_CLAMPED],
                        cnt[CNT_NONFINITE]
                    );
                    assert_eq!(far, 0);
                    assert!(neq <= 4, "{neq} latents differ");
                    assert!(relv(&vrow_d, &vrow) < 1e-5 && relv(&vcol_d[..r], &vcol[..r]) < 1e-5);
                    assert!(((mm_d - mm) / mm).abs() < 1e-5);
                    assert_eq!(cnt[3] as u64, fl_d);
                    assert!(bit_bad as usize <= neq);
                    assert_eq!(cnt[CNT_NONFINITE] as u64, nonfin);
                    assert_eq!(cnt[CNT_SATURATED] as u64, sat);
                    assert_eq!(cnt[CNT_CLAMPED] as u64, clp);
                    // the pushed entries saturate; only the momentum-free step reaches the clamp here
                    assert!(sat >= 1 && flips >= 1);
                    assert!(opt != SignOpt::Fact || clp >= 1);
                    if opt == SignOpt::FactMom {
                        let md = download_u32(c, mat.mom.as_ref().unwrap(), rows * rp / 2).unwrap();
                        // f32 contraction (FMA) on the device moves the last bit of
                        // β1·m + (1−β1)·g now and then: a bf16 neighbour, never more
                        let (mut mbad, mut mfar) = (0, 0);
                        for e in 0..rows * rp {
                            let dv = (md[e / 2] >> (16 * (e % 2))) & 0xffff;
                            if dv != mom[e] {
                                mbad += 1;
                                if (dv as i32 - mom[e] as i32).abs() > 1 {
                                    mfar += 1;
                                }
                            }
                        }
                        eprintln!("fact-mom: momentum mismatches {mbad} (more than one bf16 step: {mfar})");
                        assert_eq!(mfar, 0);
                        assert!(mbad <= rows * r / 1000);
                    }
                }
            }
            // distinct flips against the starting bits
            let diff = zeros_u32(c, 8);
            train_ops::run(c, &[bits_diff_op(c, &mat, &bb, &diff, 2).unwrap()]);
            let dv = download_u32(c, &diff, 8).unwrap();
            let fl_d: u64 = bits_d.iter().zip(&bits0).map(|(a, b)| (a ^ b).count_ones() as u64).sum();
            assert_eq!(dv[2] as u64, fl_d);
        }
    }
}
