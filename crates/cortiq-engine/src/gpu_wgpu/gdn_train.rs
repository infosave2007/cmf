//! GatedDeltaNet forward and full backward (BPTT) on the device, for
//! training — the same operator and conventions as `fcd_ops::gdn_seq_*`
//! (depthwise causal conv + SiLU over the raw qkv projection, l2-normalized
//! q/k with `1/√dk` on q, `g = exp(−e^{A_log}·softplus(a + dt_bias))`,
//! `β = σ(b)`, delta rule `S ← g·S; S += k̂ ⊗ β(v − Sᵀk̂); o = Sᵀq̂`, gated
//! RMSNorm `ô·w·silu(z)` with the norm before the gate).
//!
//! Shapes: B sequences of T tokens, each a fresh state (training windows).
//! `dk = dv = 128` (Qwen3.5/3.8: every linear-attention head).
//!
//! The recurrence is serial in time, parallel over (sequence, v-head): one
//! workgroup of 256 threads per pair holds the 128×128 state in registers,
//! an 8×8 block per thread; the reductions over rows (Sᵀk̂, Sᵀq̂, dSᵀk̂) and
//! over columns (S·do, dS·u, S·dkv) go through one 16×128 shared tile.
//! The forward checkpoints S every `C` steps; the backward walks the
//! segments in reverse, replays one segment from its checkpoint into a
//! global scratch, and runs the exact reverse sweep of `gdn_group_bwd` over
//! it with dS carried in registers across segments. q/k are shared by the
//! `rep = nv/nk` v-heads of a k-head: each v-head writes its dq̂/dk̂ rows, a
//! separate pass sums them and applies the l2-norm backward.

use super::{Ctx, ctx, submit};
use std::sync::OnceLock;

const HEAD: &str = r#"
struct P {
    nb: u32, t: u32, nv: u32, nk: u32,
    c_dim: u32, vd: u32, kd: u32, rep: u32,
    kk: u32, cs: u32, nseg: u32, off_dt: u32,
    off_norm: u32, off_conv: u32, seg: u32, pad: u32,
    eps: f32, f1: f32, f2: f32, f3: f32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> RAW: array<f32>;   // [B,T,c_dim] raw qkv projection
@group(0) @binding(2) var<storage, read_write> CQ: array<f32>;    // [B,T,c_dim] conv+silu output
@group(0) @binding(3) var<storage, read_write> Z: array<f32>;     // [B,T,vd]
@group(0) @binding(4) var<storage, read_write> AB: array<f32>;    // [2][B,T,nv]  a, b
@group(0) @binding(5) var<storage, read_write> PRM: array<f32>;   // a_log[nv] dt_bias[nv] norm[dv] conv[c_dim*kk]
@group(0) @binding(6) var<storage, read_write> QKN: array<f32>;   // [2][B,T,nk,128] q̂, k̂
@group(0) @binding(7) var<storage, read_write> NQ: array<f32>;    // [2][B,T,nk] Σq²+eps, Σk²+eps
@group(0) @binding(8) var<storage, read_write> GB: array<f32>;    // [3][B,T,nv] g, β, a+dt_bias
@group(0) @binding(9) var<storage, read_write> OUT: array<f32>;   // [B,T,vd]
@group(0) @binding(10) var<storage, read_write> OS: array<f32>;   // [B,T,vd] o before the norm
@group(0) @binding(11) var<storage, read_write> INV: array<f32>;  // [B,T,nv]
@group(0) @binding(12) var<storage, read_write> CK: array<f32>;   // [B,nv,nseg+1,16384]
@group(0) @binding(13) var<storage, read_write> SCR: array<f32>;  // [B,nv,cs,16384]
@group(0) @binding(14) var<storage, read_write> KVS: array<f32>;  // [B,nv,cs,128]
@group(0) @binding(15) var<storage, read_write> DOUT: array<f32>; // [B,T,vd]
@group(0) @binding(16) var<storage, read_write> DCQ: array<f32>;  // [B,T,c_dim]
@group(0) @binding(17) var<storage, read_write> DZ: array<f32>;   // [B,T,vd]
@group(0) @binding(18) var<storage, read_write> DAB: array<f32>;  // [2][B,T,nv]
@group(0) @binding(19) var<storage, read_write> DQK: array<f32>;  // [2][B,T,nv,128] per-v-head dq̂, dk̂
@group(0) @binding(20) var<storage, read_write> DRAW: array<f32>; // [B,T,c_dim]

fn sigm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
fn silu_bwd(x: f32) -> f32 { let s = sigm(x); return s * (1.0 + x * (1.0 - s)); }
fn softplus(x: f32) -> f32 { if (x > 20.0) { return x; } return log(1.0 + exp(x)); }

"#;

const SRC: &str = r#"
var<workgroup> red: array<f32, 2048>;
var<workgroup> sq: array<f32, 128>;
var<workgroup> sk: array<f32, 128>;
var<workgroup> sv: array<f32, 128>;
var<workgroup> va: array<f32, 128>;
var<workgroup> vb: array<f32, 128>;
var<workgroup> vc: array<f32, 128>;
var<workgroup> scal: array<f32, 8>;

// ---- conv + silu forward: one thread per (b, t, c) ----
@compute @workgroup_size(256)
fn conv_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 65535u * 256u;
    let total = p.nb * p.t * p.c_dim;
    if (i >= total) { return; }
    let c = i % p.c_dim;
    let bt = i / p.c_dim;
    let ti = bt % p.t;
    let base = bt - ti;
    var acc = 0.0;
    for (var j = 0u; j < p.kk; j = j + 1u) {
        let pos = i32(ti) - i32(p.kk - 1u) + i32(j);
        if (pos >= 0) {
            acc = acc + RAW[(base + u32(pos)) * p.c_dim + c] * PRM[p.off_conv + c * p.kk + j];
        }
    }
    CQ[i] = silu(acc);
}

// ---- conv + silu backward (through-grad into the raw projection) ----
@compute @workgroup_size(256)
fn conv_bwd(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 65535u * 256u;
    let total = p.nb * p.t * p.c_dim;
    if (i >= total) { return; }
    let c = i % p.c_dim;
    let bt = i / p.c_dim;
    let ti = bt % p.t;
    let base = bt - ti;
    // raw position ti feeds conv outputs at ti + (kk-1) - j through tap j.
    var acc = 0.0;
    for (var j = 0u; j < p.kk; j = j + 1u) {
        let to = ti + (p.kk - 1u) - j;
        if (to < p.t) {
            var pre = 0.0;
            for (var q = 0u; q < p.kk; q = q + 1u) {
                let pos = i32(to) - i32(p.kk - 1u) + i32(q);
                if (pos >= 0) {
                    pre = pre + RAW[(base + u32(pos)) * p.c_dim + c] * PRM[p.off_conv + c * p.kk + q];
                }
            }
            acc = acc + DCQ[(base + to) * p.c_dim + c] * silu_bwd(pre) * PRM[p.off_conv + c * p.kk + j];
        }
    }
    DRAW[i] = acc;
}

// ---- q/k l2 norms: one workgroup (128 threads) per (b, t, k-head) ----
@compute @workgroup_size(128)
fn qk_prep(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let idx = wg.x + wg.y * 65535u;   // (b*T + ti)*nk + ko
    if (idx >= p.nb * p.t * p.nk) { return; }
    let ko = idx % p.nk;
    let bt = idx / p.nk;
    let q = CQ[bt * p.c_dim + ko * 128u + t];
    let k = CQ[bt * p.c_dim + p.kd + ko * 128u + t];
    red[t] = q * q;
    red[128u + t] = k * k;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (t < s) {
            red[t] = red[t] + red[t + s];
            red[128u + t] = red[128u + t] + red[128u + t + s];
        }
        workgroupBarrier();
    }
    let nq2 = red[0] + 1e-6;
    let nk2 = red[128] + 1e-6;
    let n = p.nb * p.t * p.nk;
    QKN[idx * 128u + t] = q / (sqrt(nq2) * sqrt(128.0));
    QKN[(n + idx) * 128u + t] = k / sqrt(nk2);
    if (t == 0u) { NQ[idx] = nq2; NQ[n + idx] = nk2; }
}

// ---- gates: one thread per (b, t, v-head) ----
@compute @workgroup_size(256)
fn gate_prep(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    let n = p.nb * p.t * p.nv;
    if (i >= n) { return; }
    let h = i % p.nv;
    let arg = AB[i] + PRM[p.off_dt + h];
    let ea = exp(PRM[h]);
    GB[i] = exp(-ea * softplus(arg));
    GB[n + i] = sigm(AB[n + i]);
    GB[2u * n + i] = arg;
}

// Column reduction (over rows di): each thread's colp[8] partials for its
// 8 columns → out[128]. Row reduction: rowp[8] for its 8 rows → out.
fn col_reduce_store(ty: u32, tx: u32, j: u32, v: f32) { red[ty * 128u + tx * 8u + j] = v; }
fn row_reduce_store(ty: u32, tx: u32, i: u32, v: f32) { red[tx * 128u + ty * 8u + i] = v; }
fn reduce16(lin: u32) -> f32 {
    var s = 0.0;
    for (var r = 0u; r < 16u; r = r + 1u) { s = s + red[r * 128u + lin]; }
    return s;
}

// ---- forward recurrence: one workgroup per (b, v-head) ----
@compute @workgroup_size(256)
fn rec_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let bidx = wg.x / p.nv;
    let h = wg.x % p.nv;
    let ko = h / p.rep;
    let ty = tid / 16u;
    let tx = tid % 16u;
    let n_bt = p.nb * p.t;
    var s: array<f32, 64>;
    for (var e = 0u; e < 64u; e = e + 1u) { s[e] = 0.0; }
    let ckb = (bidx * p.nv + h) * (p.nseg + 1u) * 16384u;
    for (var e = 0u; e < 64u; e = e + 1u) { CK[ckb + tid * 64u + e] = 0.0; }
    for (var ti = 0u; ti < p.t; ti = ti + 1u) {
        let bt = bidx * p.t + ti;
        if (tid < 128u) {
            sq[tid] = QKN[(bt * p.nk + ko) * 128u + tid];
            sk[tid] = QKN[((n_bt + bt) * p.nk + ko) * 128u + tid];
            sv[tid] = CQ[bt * p.c_dim + 2u * p.kd + h * 128u + tid];
        }
        workgroupBarrier();
        let g = GB[bt * p.nv + h];
        let beta = GB[(n_bt + bt) * p.nv + h];
        // S_pre = g·S ; kv = S_preᵀ k̂
        for (var j = 0u; j < 8u; j = j + 1u) {
            var acc = 0.0;
            for (var i = 0u; i < 8u; i = i + 1u) {
                let e = i * 8u + j;
                s[e] = s[e] * g;
                acc = acc + s[e] * sk[ty * 8u + i];
            }
            col_reduce_store(ty, tx, j, acc);
        }
        workgroupBarrier();
        if (tid < 128u) { va[tid] = beta * (sv[tid] - reduce16(tid)); }
        workgroupBarrier();
        // S += k̂ ⊗ u ; o = Sᵀ q̂
        for (var j = 0u; j < 8u; j = j + 1u) {
            var acc = 0.0;
            let u = va[tx * 8u + j];
            for (var i = 0u; i < 8u; i = i + 1u) {
                let e = i * 8u + j;
                s[e] = s[e] + sk[ty * 8u + i] * u;
                acc = acc + s[e] * sq[ty * 8u + i];
            }
            col_reduce_store(ty, tx, j, acc);
        }
        workgroupBarrier();
        if (tid < 128u) {
            let o = reduce16(tid);
            vb[tid] = o;
            OS[bt * p.vd + h * 128u + tid] = o;
        }
        workgroupBarrier();
        if (tid < 128u) { red[tid] = vb[tid] * vb[tid]; }
        workgroupBarrier();
        for (var st = 64u; st > 0u; st = st >> 1u) {
            if (tid < st) { red[tid] = red[tid] + red[tid + st]; }
            workgroupBarrier();
        }
        let inv = 1.0 / sqrt(red[0] / 128.0 + p.eps);
        if (tid < 128u) {
            let zz = Z[bt * p.vd + h * 128u + tid];
            OUT[bt * p.vd + h * 128u + tid] = vb[tid] * inv * PRM[p.off_norm + tid] * silu(zz);
        }
        if (tid == 0u) { INV[bt * p.nv + h] = inv; }
        if ((ti + 1u) % p.cs == 0u) {
            let ci = (ti + 1u) / p.cs;
            for (var e = 0u; e < 64u; e = e + 1u) { CK[ckb + ci * 16384u + tid * 64u + e] = s[e]; }
        }
        workgroupBarrier();
    }
}

// ---- backward recurrence: one workgroup per (b, v-head) ----
@compute @workgroup_size(256)
fn rec_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let bidx = wg.x / p.nv;
    let h = wg.x % p.nv;
    let ko = h / p.rep;
    let ty = tid / 16u;
    let tx = tid % 16u;
    let n_bt = p.nb * p.t;
    let ea = exp(PRM[h]);
    let ckb = (bidx * p.nv + h) * (p.nseg + 1u) * 16384u;
    let scb = (bidx * p.nv + h) * p.cs * 16384u;
    let kvb = (bidx * p.nv + h) * p.cs * 128u;
    var ds: array<f32, 64>;
    var st: array<f32, 64>;   // S_t of the current reverse step
    var sp: array<f32, 64>;   // S_{t-1}
    for (var e = 0u; e < 64u; e = e + 1u) { ds[e] = 0.0; }
    for (var sgi = 0u; sgi < p.nseg; sgi = sgi + 1u) {
        let seg = p.nseg - 1u - sgi;
        let t0 = seg * p.cs;
        let t1 = min(t0 + p.cs, p.t);
        // replay the segment from its checkpoint into the scratch
        for (var e = 0u; e < 64u; e = e + 1u) { sp[e] = CK[ckb + seg * 16384u + tid * 64u + e]; }
        for (var ti = t0; ti < t1; ti = ti + 1u) {
            let bt = bidx * p.t + ti;
            if (tid < 128u) {
                sk[tid] = QKN[((n_bt + bt) * p.nk + ko) * 128u + tid];
                sv[tid] = CQ[bt * p.c_dim + 2u * p.kd + h * 128u + tid];
            }
            workgroupBarrier();
            let g = GB[bt * p.nv + h];
            let beta = GB[(n_bt + bt) * p.nv + h];
            for (var j = 0u; j < 8u; j = j + 1u) {
                var acc = 0.0;
                for (var i = 0u; i < 8u; i = i + 1u) {
                    let e = i * 8u + j;
                    sp[e] = sp[e] * g;
                    acc = acc + sp[e] * sk[ty * 8u + i];
                }
                col_reduce_store(ty, tx, j, acc);
            }
            workgroupBarrier();
            if (tid < 128u) {
                let kv = reduce16(tid);
                KVS[kvb + (ti - t0) * 128u + tid] = kv;
                va[tid] = beta * (sv[tid] - kv);
            }
            workgroupBarrier();
            for (var j = 0u; j < 8u; j = j + 1u) {
                let u = va[tx * 8u + j];
                for (var i = 0u; i < 8u; i = i + 1u) {
                    let e = i * 8u + j;
                    sp[e] = sp[e] + sk[ty * 8u + i] * u;
                }
            }
            for (var e = 0u; e < 64u; e = e + 1u) { SCR[scb + (ti - t0) * 16384u + tid * 64u + e] = sp[e]; }
            workgroupBarrier();
        }
        // reverse sweep over the segment
        for (var e = 0u; e < 64u; e = e + 1u) { st[e] = SCR[scb + (t1 - 1u - t0) * 16384u + tid * 64u + e]; }
        for (var rr = 0u; rr < t1 - t0; rr = rr + 1u) {
            let ti = t1 - 1u - rr;
            let bt = bidx * p.t + ti;
            if (ti > t0) {
                for (var e = 0u; e < 64u; e = e + 1u) { sp[e] = SCR[scb + (ti - 1u - t0) * 16384u + tid * 64u + e]; }
            } else {
                for (var e = 0u; e < 64u; e = e + 1u) { sp[e] = CK[ckb + seg * 16384u + tid * 64u + e]; }
            }
            let g = GB[bt * p.nv + h];
            let beta = GB[(n_bt + bt) * p.nv + h];
            let arg = GB[(2u * n_bt + bt) * p.nv + h];
            let inv = INV[bt * p.nv + h];
            if (tid < 128u) {
                sq[tid] = QKN[(bt * p.nk + ko) * 128u + tid];
                sk[tid] = QKN[((n_bt + bt) * p.nk + ko) * 128u + tid];
                sv[tid] = CQ[bt * p.c_dim + 2u * p.kd + h * 128u + tid];
                let kv = KVS[kvb + (ti - t0) * 128u + tid];
                vc[tid] = sv[tid] - kv;              // v - kv
                // gated rmsnorm backward, part 1
                let o = OS[bt * p.vd + h * 128u + tid];
                let zz = Z[bt * p.vd + h * 128u + tid];
                let w = PRM[p.off_norm + tid];
                let d = DOUT[bt * p.vd + h * 128u + tid];
                red[tid] = d * w * silu(zz) * o;
                DZ[bt * p.vd + h * 128u + tid] = d * o * inv * w * silu_bwd(zz);
            }
            workgroupBarrier();
            for (var s2 = 64u; s2 > 0u; s2 = s2 >> 1u) {
                if (tid < s2) { red[tid] = red[tid] + red[tid + s2]; }
                workgroupBarrier();
            }
            let coef = red[0] / 128.0 * inv * inv * inv;
            workgroupBarrier();
            if (tid < 128u) {
                let o = OS[bt * p.vd + h * 128u + tid];
                let zz = Z[bt * p.vd + h * 128u + tid];
                let w = PRM[p.off_norm + tid];
                let d = DOUT[bt * p.vd + h * 128u + tid];
                va[tid] = inv * w * silu(zz) * d - o * coef;   // do
                vb[tid] = beta * vc[tid];                       // u
            }
            workgroupBarrier();
            // dS += q̂ ⊗ do ; dq̂ = S_t · do (row reduction)
            for (var i = 0u; i < 8u; i = i + 1u) {
                var acc = 0.0;
                let qf = sq[ty * 8u + i];
                for (var j = 0u; j < 8u; j = j + 1u) {
                    let e = i * 8u + j;
                    let dj = va[tx * 8u + j];
                    ds[e] = ds[e] + qf * dj;
                    acc = acc + st[e] * dj;
                }
                row_reduce_store(ty, tx, i, acc);
            }
            workgroupBarrier();
            if (tid < 128u) { DQK[(bt * p.nv + h) * 128u + tid] = reduce16(tid); }
            workgroupBarrier();
            // du = dSᵀ k̂ (column reduction)
            for (var j = 0u; j < 8u; j = j + 1u) {
                var acc = 0.0;
                for (var i = 0u; i < 8u; i = i + 1u) { acc = acc + ds[i * 8u + j] * sk[ty * 8u + i]; }
                col_reduce_store(ty, tx, j, acc);
            }
            workgroupBarrier();
            if (tid < 128u) { va[tid] = reduce16(tid); }   // du
            workgroupBarrier();
            // dk̂ = dS · u (row reduction) -> vc reused after its last use below
            for (var i = 0u; i < 8u; i = i + 1u) {
                var acc = 0.0;
                for (var j = 0u; j < 8u; j = j + 1u) { acc = acc + ds[i * 8u + j] * vb[tx * 8u + j]; }
                row_reduce_store(ty, tx, i, acc);
            }
            workgroupBarrier();
            var dkh_part = 0.0;
            if (tid < 128u) {
                dkh_part = reduce16(tid);
                let du = va[tid];
                DCQ[bt * p.c_dim + 2u * p.kd + h * 128u + tid] = beta * du;   // dv (exclusive)
                scal[0] = 0.0;
            }
            workgroupBarrier();
            if (tid < 128u) {
                red[tid] = va[tid] * vc[tid];      // dβ partials
                vb[tid] = -(beta * va[tid]);       // dkv
                sv[tid] = dkh_part;                // stash dk̂ partial
            }
            workgroupBarrier();
            for (var s2 = 64u; s2 > 0u; s2 = s2 >> 1u) {
                if (tid < s2) { red[tid] = red[tid] + red[tid + s2]; }
                workgroupBarrier();
            }
            let dbeta = red[0];
            workgroupBarrier();
            // dS_pre = dS + k̂⊗dkv ; dk̂ += (g·S_{t-1})·dkv ; dg = <dS_pre, S_{t-1}> ; dS = g·dS_pre
            var dgp = 0.0;
            for (var i = 0u; i < 8u; i = i + 1u) {
                var acc = 0.0;
                let kf = sk[ty * 8u + i];
                for (var j = 0u; j < 8u; j = j + 1u) {
                    let e = i * 8u + j;
                    let dkv = vb[tx * 8u + j];
                    let dspre = ds[e] + kf * dkv;
                    acc = acc + (sp[e] * g) * dkv;
                    dgp = dgp + dspre * sp[e];
                    ds[e] = g * dspre;
                }
                row_reduce_store(ty, tx, i, acc);
            }
            workgroupBarrier();
            if (tid < 128u) {
                DQK[((n_bt + bt) * p.nv + h) * 128u + tid] = sv[tid] + reduce16(tid);
            }
            workgroupBarrier();
            red[tid] = dgp;
            workgroupBarrier();
            for (var s2 = 128u; s2 > 0u; s2 = s2 >> 1u) {
                if (tid < s2) { red[tid] = red[tid] + red[tid + s2]; }
                workgroupBarrier();
            }
            if (tid == 0u) {
                let dg = red[0];
                DAB[bt * p.nv + h] = dg * g * (-ea) * sigm(arg);
                DAB[(n_bt + bt) * p.nv + h] = dbeta * beta * (1.0 - beta);
            }
            for (var e = 0u; e < 64u; e = e + 1u) { st[e] = sp[e]; }
            workgroupBarrier();
        }
    }
}

// ---- q/k backward: sum the rep v-heads, l2-norm backward ----
@compute @workgroup_size(128)
fn qk_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let idx = wg.x + wg.y * 65535u;   // (b*T + ti)*nk + ko
    if (idx >= p.nb * p.t * p.nk) { return; }
    let ko = idx % p.nk;
    let bt = idx / p.nk;
    let n_bt = p.nb * p.t;
    var dq = 0.0;
    var dk = 0.0;
    for (var hh = 0u; hh < p.rep; hh = hh + 1u) {
        let h = ko * p.rep + hh;
        dq = dq + DQK[(bt * p.nv + h) * 128u + t];
        dk = dk + DQK[((n_bt + bt) * p.nv + h) * 128u + t];
    }
    let q = CQ[bt * p.c_dim + ko * 128u + t];
    let k = CQ[bt * p.c_dim + p.kd + ko * 128u + t];
    red[t] = dq * q;
    red[128u + t] = dk * k;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (t < s) {
            red[t] = red[t] + red[t + s];
            red[128u + t] = red[128u + t] + red[128u + t + s];
        }
        workgroupBarrier();
    }
    let n = p.nb * p.t * p.nk;
    let nq2 = NQ[idx];
    let nk2 = NQ[n + idx];
    let invq = 1.0 / (sqrt(nq2) * sqrt(128.0));
    let invk = 1.0 / sqrt(nk2);
    DCQ[bt * p.c_dim + ko * 128u + t] = invq * dq - q * red[0] * invq / nq2;
    DCQ[bt * p.c_dim + p.kd + ko * 128u + t] = invk * dk - k * red[128] * invk / nk2;
}
"#;

// ---- column-split recurrence on 32-wide subgroups ----
// Workgroup = (window, v-head, block of 32 state columns) — four per head;
// warp w = columns 4w..4w+3 of the block; lane = (c = lane>>3: column in the
// warp, rg = lane&7: rows 16·rg..16·rg+15). Columns of S evolve
// independently in the forward, so the forward needs no workgroup memory;
// reductions over rows are three shuffles. The forward keeps kv per step
// (KVS) and S every `cs` steps (CK, [B,nv,nseg+1][col][row]); the backward
// replays a segment elementwise from them (u = β(v − kv) bit for bit) into
// the scratch (SCR, [B·nv·4][cs][32 cols][128 rows]). Row reductions over
// columns (dq̂, dk̂) and the scalar ones (dβ, dg) leave the recurrence as
// per-block partials (DQK, [B,T,nv][4][258]) summed by qk_bwd2.
const SRC2: &str = r#"
const NWG: u32 = 4u;      // workgroups (column blocks) per (window, head)
const PW: u32 = 260u;     // words per warp in the partial buffer

fn rsum8(v: f32) -> f32 {
    var x = v + subgroupShuffleXor(v, 1u);
    x = x + subgroupShuffleXor(x, 2u);
    return x + subgroupShuffleXor(x, 4u);
}
fn csum4(v: f32) -> f32 {
    let x = v + subgroupShuffleXor(v, 8u);
    return x + subgroupShuffleXor(x, 16u);
}

// ---- gated RMSNorm over each (token, head) row of o: one warp per row ----
@compute @workgroup_size(256)
fn norm_fwd2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    let row = (wg.x + wg.y * 65535u) * 8u + lid / 32u;   // bt*nv + h
    let live = row < p.nb * p.t * p.nv;
    let rw = select(0u, row, live);
    let bt = rw / p.nv;
    let h = rw % p.nv;
    let base = bt * p.vd + h * 128u + lane * 4u;
    var o: array<f32, 4>;
    var ss = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) { o[j] = OS[base + j]; ss = fma(o[j], o[j], ss); }
    ss = csum4(rsum8(ss));
    let inv = 1.0 / sqrt(ss / 128.0 + p.eps);
    if (live) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            OUT[base + j] = o[j] * inv * PRM[p.off_norm + lane * 4u + j] * silu(Z[base + j]);
        }
        if (lane == 0u) { INV[rw] = inv; }
    }
}

// ---- its backward: DZ, and do written over o (OS) for rec_bwd2 ----
@compute @workgroup_size(256)
fn norm_bwd2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    let row = (wg.x + wg.y * 65535u) * 8u + lid / 32u;
    let live = row < p.nb * p.t * p.nv;
    let rw = select(0u, row, live);
    let bt = rw / p.nv;
    let h = rw % p.nv;
    let base = bt * p.vd + h * 128u + lane * 4u;
    let inv = INV[rw];
    var o: array<f32, 4>;
    var a: array<f32, 4>;   // w·silu(z)·d
    var red = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
        o[j] = OS[base + j];
        let zz = Z[base + j];
        let w = PRM[p.off_norm + lane * 4u + j];
        let d = DOUT[base + j];
        a[j] = w * silu(zz) * d;
        red = fma(a[j], o[j], red);
        if (live) { DZ[base + j] = d * o[j] * inv * w * silu_bwd(zz); }
    }
    red = csum4(rsum8(red));
    let coef = red / 128.0 * inv * inv * inv;
    if (live) {
        for (var j = 0u; j < 4u; j = j + 1u) { OS[base + j] = inv * a[j] - o[j] * coef; }
    }
}

var<workgroup> wpart: array<f32, 4160>;   // [2][8 warps][PW]
@group(0) @binding(21) var<storage, read_write> QKN4: array<vec4<f32>>;
@group(0) @binding(22) var<storage, read_write> CK4: array<vec4<f32>>;
@group(0) @binding(23) var<storage, read_write> SCR4: array<vec4<f32>>;

// ---- q/k backward over the column-block partials, and dβ/dg -> a, b ----
@compute @workgroup_size(128)
fn qk_bwd2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let idx = wg.x + wg.y * 65535u;   // (b*T + ti)*nk + ko
    if (idx >= p.nb * p.t * p.nk) { return; }
    let ko = idx % p.nk;
    let bt = idx / p.nk;
    let n_bt = p.nb * p.t;
    var dq = 0.0;
    var dk = 0.0;
    for (var hh = 0u; hh < p.rep; hh = hh + 1u) {
        let h = ko * p.rep + hh;
        for (var w = 0u; w < NWG; w = w + 1u) {
            let pb = ((bt * p.nv + h) * NWG + w) * 258u;
            dq = dq + DQK[pb + t];
            dk = dk + DQK[pb + 128u + t];
        }
    }
    if (t < p.rep) {
        let h = ko * p.rep + t;
        var dg = 0.0;
        var db = 0.0;
        for (var w = 0u; w < NWG; w = w + 1u) {
            let pb = ((bt * p.nv + h) * NWG + w) * 258u;
            db = db + DQK[pb + 256u];
            dg = dg + DQK[pb + 257u];
        }
        let g = GB[bt * p.nv + h];
        let beta = GB[(n_bt + bt) * p.nv + h];
        let arg = GB[(2u * n_bt + bt) * p.nv + h];
        DAB[bt * p.nv + h] = dg * g * (-exp(PRM[h])) * sigm(arg);
        DAB[(n_bt + bt) * p.nv + h] = db * beta * (1.0 - beta);
    }
    let q = CQ[bt * p.c_dim + ko * 128u + t];
    let k = CQ[bt * p.c_dim + p.kd + ko * 128u + t];
    red2[t] = dq * q;
    red2[128u + t] = dk * k;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (t < s) {
            red2[t] = red2[t] + red2[t + s];
            red2[128u + t] = red2[128u + t] + red2[128u + t + s];
        }
        workgroupBarrier();
    }
    let n = p.nb * p.t * p.nk;
    let nq2 = NQ[idx];
    let nk2 = NQ[n + idx];
    let invq = 1.0 / (sqrt(nq2) * sqrt(128.0));
    let invk = 1.0 / sqrt(nk2);
    DCQ[bt * p.c_dim + ko * 128u + t] = invq * dq - q * red2[0] * invq / nq2;
    DCQ[bt * p.c_dim + p.kd + ko * 128u + t] = invk * dk - k * red2[128] * invk / nk2;
}
var<workgroup> red2: array<f32, 256>;
"#;

/// The two recurrence kernels of `SRC2`, generated with the state block in
/// named registers (`s0..s15`, …) rather than arrays a driver may spill.
/// Every per-step operand (q̂, k̂, g, β, v, kv, do) is staged through
/// workgroup memory a block of steps at a time (one coalesced load and two
/// barriers per block instead of a global-latency stall every step), and
/// the backward prefetches the next step's S_{t-1} from the scratch while
/// it computes the current one. Bindings 21-23 are vec4 views of QKN, CK, SCR.
fn rec_src(cs: usize) -> String {
    const TS: usize = 16; // forward staging block (steps)
    let v16 = |f: &dyn Fn(usize) -> String| (0..16).map(f).collect::<Vec<_>>().join("\n");
    // 16 floats from 4 consecutive vec4 of `arr` at `base`: `{dst}{r}` scalars
    let ld4 = |ind: &str, dst: &str, arr: &str, base: &str| {
        (0..4)
            .map(|q| format!("{ind}let {dst}v{q} = {arr}[{base} + {q}u]; let {dst}{a} = {dst}v{q}.x; let {dst}{b} = {dst}v{q}.y; let {dst}{c} = {dst}v{q}.z; let {dst}{d} = {dst}v{q}.w;", a = 4 * q, b = 4 * q + 1, c = 4 * q + 2, d = 4 * q + 3))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let st4 = |arr: &str, base: &str, src: &str| (0..4).map(|q| format!("{arr}[{base} + {q}u] = vec4<f32>({src}{}, {src}{}, {src}{}, {src}{});", 4 * q, 4 * q + 1, 4 * q + 2, 4 * q + 3)).collect::<Vec<_>>().join(" ");
    let chain = |ind: &str, acc: &str, a: &str, b: &str| {
        let mut o = format!("{ind}var {acc}0 = {a}0 * {b}0;\n{ind}var {acc}1 = {a}1 * {b}1;\n");
        o += &(2..16).map(|r| format!("{ind}{acc}{} = fma({a}{r}, {b}{r}, {acc}{});", r % 2, r % 2)).collect::<Vec<_>>().join("\n");
        o
    };
    let mut o = String::new();
    // ---------------- forward ----------------
    o += &format!(
        "var<workgroup> fq: array<vec4<f32>, {n4}>;
var<workgroup> fk: array<vec4<f32>, {n4}>;
var<workgroup> fv: array<f32, {n32}>;
var<workgroup> fgb: array<f32, {n2}>;
@compute @workgroup_size(256)
fn rec_fwd2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) lane: u32) {{
    let bh = wg.x / NWG;
    let wgc = wg.x % NWG;
    let bidx = bh / p.nv;
    let h = bh % p.nv;
    let ko = h / p.rep;
    let cw = (lid / 32u) * 4u + (lane >> 3u);
    let col = wgc * 32u + cw;
    let r4 = (lane & 7u) * 4u;
    let first = (lane & 7u) == 0u;
    let n_bt = p.nb * p.t;
    let keep = p.seg == 1u;
    let ckb4 = (bh * (p.nseg + 1u) * 16384u + col * 128u) / 4u + r4;
",
        n4 = TS * 32,
        n32 = TS * 32,
        n2 = (TS * 2).max(4)
    );
    o += &v16(&|r| format!("    var s{r} = 0.0;"));
    o += &format!(
        "
    for (var tb = 0u; tb < p.t; tb = tb + {TS}u) {{
        let nst = min({TS}u, p.t - tb);
        workgroupBarrier();
        for (var e = lid; e < {n4}u; e = e + 256u) {{
            let j = e / 32u;
            if (j < nst) {{
                let bt = bidx * p.t + tb + j;
                fq[e] = QKN4[(bt * p.nk + ko) * 32u + e % 32u];
                fk[e] = QKN4[((n_bt + bt) * p.nk + ko) * 32u + e % 32u];
                fv[e] = CQ[bt * p.c_dim + 2u * p.kd + h * 128u + wgc * 32u + e % 32u];
            }}
        }}
        if (lid < {n2}u) {{
            let j = lid / 2u;
            if (j < nst) {{ fgb[lid] = GB[((lid % 2u) * n_bt + bidx * p.t + tb + j) * p.nv + h]; }}
        }}
        workgroupBarrier();
        for (var j = 0u; j < nst; j = j + 1u) {{
            let ti = tb + j;
            let bt = bidx * p.t + ti;
            let g = fgb[2u * j];
            let beta = fgb[2u * j + 1u];
            let v = fv[j * 32u + cw];
            let vi = bt * p.vd + h * 128u + col;
",
        n4 = TS * 32,
        n2 = TS * 2
    );
    o += &ld4("            ", "k", "fk", "j * 32u + r4");
    o += "\n";
    o += &v16(&|r| format!("            s{r} = s{r} * g;"));
    o += "\n";
    o += &chain("            ", "p", "s", "k");
    o += "
            let kv = rsum8(p0 + p1);
            let u = beta * (v - kv);
";
    o += &v16(&|r| format!("            s{r} = fma(k{r}, u, s{r});"));
    o += "\n";
    o += &ld4("            ", "q", "fq", "j * 32u + r4");
    o += "\n";
    o += &chain("            ", "o", "s", "q");
    o += &format!(
        "
            let o = rsum8(o0 + o1);
            if (first) {{
                OS[vi] = o;
                if (keep) {{ KVS[vi] = kv; }}
            }}
            if (keep && (ti + 1u) % p.cs == 0u) {{
                let cb = ckb4 + ((ti + 1u) / p.cs) * 4096u;
                {}
            }}
        }}
    }}
}}
",
        st4("CK4", "cb", "s")
    );
    // ---------------- backward ----------------
    let n4 = cs * 32;
    o += &format!(
        "var<workgroup> bq: array<vec4<f32>, {n4}>;
var<workgroup> bk: array<vec4<f32>, {n4}>;
var<workgroup> bv: array<f32, {n32}>;
var<workgroup> bkv: array<f32, {n32}>;
var<workgroup> bdo: array<f32, {n32}>;
var<workgroup> bgb: array<f32, {n2}>;
@compute @workgroup_size(256)
fn rec_bwd2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) lane: u32) {{
    let bh = wg.x / NWG;
    let wgc = wg.x % NWG;
    let bidx = bh / p.nv;
    let h = bh % p.nv;
    let ko = h / p.rep;
    let warp = lid / 32u;
    let c = lane >> 3u;
    let rg = lane & 7u;
    let cw = warp * 4u + c;
    let col = wgc * 32u + cw;
    let r0 = rg * 16u;
    let r4 = rg * 4u;
    let first = rg == 0u;
    let hi0 = (c & 1u) == 1u;
    let hi1 = c >= 2u;
    let rowq = r0 + (c & 1u) * 8u + (c >> 1u) * 4u;
    let n_bt = p.nb * p.t;
    let ckb4 = (bh * (p.nseg + 1u) * 16384u + col * 128u) / 4u + r4;
    let scb4 = (wg.x * p.cs * 4096u + cw * 128u) / 4u + r4;
",
        n32 = cs * 32,
        n2 = (cs * 2).max(4)
    );
    o += &v16(&|r| format!("    var ds{r} = 0.0;\n    var s{r} = 0.0;"));
    o += "\n    var sp0 = vec4<f32>(0.0); var sp1 = vec4<f32>(0.0); var sp2 = vec4<f32>(0.0); var sp3 = vec4<f32>(0.0);\n";
    o += "    var pn0 = vec4<f32>(0.0); var pn1 = vec4<f32>(0.0); var pn2 = vec4<f32>(0.0); var pn3 = vec4<f32>(0.0);\n";
    o += &format!(
        "    var par = 0u;
    for (var sgi = 0u; sgi < p.nseg; sgi = sgi + 1u) {{
        let seg = p.nseg - 1u - sgi;
        let t0 = seg * p.cs;
        let nst = min(p.cs, p.t - t0);
        workgroupBarrier();
        for (var e = lid; e < {n4}u; e = e + 256u) {{
            let j = e / 32u;
            if (j < nst) {{
                let bt = bidx * p.t + t0 + j;
                bq[e] = QKN4[(bt * p.nk + ko) * 32u + e % 32u];
                bk[e] = QKN4[((n_bt + bt) * p.nk + ko) * 32u + e % 32u];
                let ci = bt * p.vd + h * 128u + wgc * 32u + e % 32u;
                bv[e] = CQ[bt * p.c_dim + 2u * p.kd + h * 128u + wgc * 32u + e % 32u];
                bkv[e] = KVS[ci];
                bdo[e] = OS[ci];
            }}
        }}
        if (lid < {n2}u) {{
            let j = lid / 2u;
            if (j < nst) {{ bgb[lid] = GB[((lid % 2u) * n_bt + bidx * p.t + t0 + j) * p.nv + h]; }}
        }}
        workgroupBarrier();
        // replay the segment from its checkpoint (S_{{t0-1}}) into the scratch
        if (seg > 0u) {{
",
        n2 = cs * 2
    );
    o += &ld4("            ", "c", "CK4", "ckb4 + seg * 4096u");
    o += "\n";
    o += &v16(&|r| format!("            s{r} = c{r};"));
    o += "\n        } else {\n";
    o += &v16(&|r| format!("            s{r} = 0.0;"));
    o += "
        }
        for (var j = 0u; j < nst; j = j + 1u) {
            let g = bgb[2u * j];
            let u = bgb[2u * j + 1u] * (bv[j * 32u + cw] - bkv[j * 32u + cw]);
";
    o += &ld4("            ", "k", "bk", "j * 32u + r4");
    o += "\n";
    o += &v16(&|r| format!("            s{r} = s{r} * g;\n            s{r} = fma(k{r}, u, s{r});"));
    o += &format!(
        "
            if (j + 1u < nst) {{
                let sb = scb4 + j * 1024u;
                {}
            }}
        }}
        // S_{{t-1}} of the last step of the segment
        if (nst > 1u) {{
            let sb = scb4 + (nst - 2u) * 1024u;
            sp0 = SCR4[sb]; sp1 = SCR4[sb + 1u]; sp2 = SCR4[sb + 2u]; sp3 = SCR4[sb + 3u];
        }} else if (nst == 1u && seg > 0u) {{
            let sb = ckb4 + seg * 4096u;
            sp0 = CK4[sb]; sp1 = CK4[sb + 1u]; sp2 = CK4[sb + 2u]; sp3 = CK4[sb + 3u];
        }} else {{
            sp0 = vec4<f32>(0.0); sp1 = vec4<f32>(0.0); sp2 = vec4<f32>(0.0); sp3 = vec4<f32>(0.0);
        }}
        for (var rr = 0u; rr < nst; rr = rr + 1u) {{
            let j = nst - 1u - rr;
            let ti = t0 + j;
            let bt = bidx * p.t + ti;
            // prefetch S_{{t-2}} for the next step
            if (j >= 2u) {{
                let sb = scb4 + (j - 2u) * 1024u;
                pn0 = SCR4[sb]; pn1 = SCR4[sb + 1u]; pn2 = SCR4[sb + 2u]; pn3 = SCR4[sb + 3u];
            }} else if (j == 1u && seg > 0u) {{
                let sb = ckb4 + seg * 4096u;
                pn0 = CK4[sb]; pn1 = CK4[sb + 1u]; pn2 = CK4[sb + 2u]; pn3 = CK4[sb + 3u];
            }} else {{
                pn0 = vec4<f32>(0.0); pn1 = vec4<f32>(0.0); pn2 = vec4<f32>(0.0); pn3 = vec4<f32>(0.0);
            }}
",
        st4("SCR4", "sb", "s")
    );
    o += "            let sp0_ = sp0; let sp1_ = sp1; let sp2_ = sp2; let sp3_ = sp3;\n";
    o += &(0..16).map(|r| format!("            let sp{r}s = sp{}_.{};", r / 4, ["x", "y", "z", "w"][r % 4])).collect::<Vec<_>>().join("\n");
    o += "
            let g = bgb[2u * j];
            let beta = bgb[2u * j + 1u];
            let vc = bv[j * 32u + cw] - bkv[j * 32u + cw];
            let u = beta * vc;
            let dd = bdo[j * 32u + cw];
";
    o += &ld4("            ", "k", "bk", "j * 32u + r4");
    o += "\n";
    o += &ld4("            ", "q", "bq", "j * 32u + r4");
    o += "\n";
    o += &v16(&|r| format!("            ds{r} = fma(q{r}, dd, ds{r});\n            let dq{r} = s{r} * dd;"));
    o += "\n";
    o += &chain("            ", "d", "ds", "k");
    o += "
            let du = rsum8(d0 + d1);
            let dkv = -(beta * du);
";
    o += &v16(&|r| format!("            let dk{r} = fma(ds{r}, u, (sp{r}s * g) * dkv);\n            let e{r} = fma(k{r}, dkv, ds{r});"));
    o += "\n";
    o += "            var g0 = e0 * sp0s;\n            var g1 = e1 * sp1s;\n";
    o += &(2..16).map(|r| format!("            g{} = fma(e{r}, sp{r}s, g{});", r % 2, r % 2)).collect::<Vec<_>>().join("\n");
    o += "\n";
    o += &v16(&|r| format!("            ds{r} = g * e{r};"));
    o += "
            let dgw = csum4(rsum8(g0 + g1));
            let dbw = csum4(du * vc);
            if (first) { DCQ[bt * p.c_dim + 2u * p.kd + h * 128u + col] = beta * du; }
            let wb = par * 2080u + warp * PW;
";
    for (nm, off) in [("dq", "0u"), ("dk", "128u")] {
        for i in 0..4 {
            o += &format!(
                "            {{
                let a0 = select({nm}{i}, {nm}{j8}, hi0) + subgroupShuffleXor(select({nm}{j8}, {nm}{i}, hi0), 8u);
                let a1 = select({nm}{j4}, {nm}{j12}, hi0) + subgroupShuffleXor(select({nm}{j12}, {nm}{j4}, hi0), 8u);
                wpart[wb + {off} + rowq + {i}u] = select(a0, a1, hi1) + subgroupShuffleXor(select(a1, a0, hi1), 16u);
            }}
",
                j8 = i + 8,
                j4 = i + 4,
                j12 = i + 12
            );
        }
    }
    o += "            if (lane == 0u) {
                wpart[wb + 256u] = dbw;
                wpart[wb + 257u] = dgw;
            }
            workgroupBarrier();
            let ob = ((bt * p.nv + h) * NWG + wgc) * 258u;
            let rb = par * 2080u;
            var acc = 0.0;
            for (var w = 0u; w < 8u; w = w + 1u) { acc = acc + wpart[rb + w * PW + lid]; }
            DQK[ob + lid] = acc;
            if (lid < 2u) {
                var a2 = 0.0;
                for (var w = 0u; w < 8u; w = w + 1u) { a2 = a2 + wpart[rb + w * PW + 256u + lid]; }
                DQK[ob + 256u + lid] = a2;
            }
            par = par ^ 1u;
";
    o += &(0..16).map(|r| format!("            s{r} = sp{r}s;")).collect::<Vec<_>>().join("\n");
    o += "
            sp0 = pn0; sp1 = pn1; sp2 = pn2; sp3 = pn3;
        }
    }
}
";
    o
}

const ENTRIES: [&str; 7] = ["conv_fwd", "conv_bwd", "qk_prep", "gate_prep", "rec_fwd", "rec_bwd", "qk_bwd"];
const ENTRIES2: [&str; 5] = ["rec_fwd2", "norm_fwd2", "norm_bwd2", "rec_bwd2", "qk_bwd2"];
const NBIND: u32 = 24;
/// Column blocks (workgroups) per (window, head) of the subgroup kernels.
const NWG: usize = 4;

struct Pipes {
    layout: wgpu::BindGroupLayout,
    p: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
    /// The column-split subgroup kernels (`SRC2`) are built.
    sg: bool,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

fn build(c: &Ctx, pl: &wgpu::PipelineLayout, label: &str, src: &str, entries: &[&'static str], p: &mut std::collections::HashMap<&'static str, wgpu::ComputePipeline>) -> bool {
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let m = super::train_ops::trainer_module(c, label, src);
    if let Some(e) = pollster::block_on(sc.pop()) {
        tracing::warn!("{label} module rejected: {e}");
        return false;
    }
    for &ep in entries {
        let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipe = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: Some(pl),
            module: &m,
            entry_point: Some(ep),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(sc.pop()) {
            tracing::warn!("{label} pipeline {ep} rejected: {e}");
            return false;
        }
        p.insert(ep, pipe);
    }
    true
}

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let lim = c.device.limits();
            if lim.max_storage_buffers_per_shader_stage < NBIND - 1 || lim.max_compute_workgroup_storage_size < 17408 {
                return None;
            }
            let mut entries = vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(80),
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
                label: Some("gdn_train"),
                entries: &entries,
            });
            let pl = c.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gdn_train"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let mut p = std::collections::HashMap::new();
            if !build(c, &pl, "gdn_train", &format!("{HEAD}{SRC}"), &ENTRIES, &mut p) {
                return None;
            }
            // The subgroup kernels assume 32-wide subgroups (lane bits = rows
            // and columns of the state block); elsewhere the per-head kernels run.
            let sg = c.device.features().contains(wgpu::Features::SUBGROUP)
                && c.adapter_info.subgroup_min_size == 32
                && c.adapter_info.subgroup_max_size == 32
                && std::env::var("CMF_GDN_SG").as_deref() != Ok("0")
                && build(c, &pl, "gdn_train_sg", &format!("{HEAD}{SRC2}{}", rec_src(ckpt_interval())), &ENTRIES2, &mut p);
            for (ep, pipe) in &p {
                super::train_prof::name_pipe(pipe, ep);
            }
            Some(Pipes { layout, p, sg })
        })
        .as_ref()
}

/// Geometry and frozen parameters of one GDN layer.
pub struct GdnTrainCfg<'a> {
    pub nv: usize,
    pub nk: usize,
    pub kk: usize,
    pub rms_eps: f64,
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
    pub conv: &'a [f32],
}

/// Device buffers of one forward, kept for its backward, with their bind
/// groups (`bg[1]` records the tape — checkpoints and kv — for a backward,
/// `bg[0]` is a forward whose backward never runs).
pub struct GdnTape {
    nb: usize,
    t: usize,
    bufs: Vec<wgpu::Buffer>,
    bg: [wgpu::BindGroup; 2],
}

fn sbuf(c: &Ctx, n: usize) -> wgpu::Buffer {
    c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gdn_train"),
        size: ((n.max(4)) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn read(c: &Ctx, b: &wgpu::Buffer, n: usize) -> Option<Vec<f32>> {
    let size = (n * 4) as u64;
    let st = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gdn_train_rb"),
        size: size.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(b, 0, &st, 0, size);
    submit(c, enc.finish());
    let slice = st.slice(..size);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let out = {
        let data = slice.get_mapped_range().ok()?;
        let mut v = vec![0f32; n];
        super::par_copy(&mut v, bytemuck::cast_slice(&data[..size as usize]));
        v
    };
    st.unmap();
    Some(out)
}

/// Checkpoint interval of the state (steps per replay segment); the
/// subgroup backward's scratch is `nb·nv·cs` states (`CMF_GDN_CS`).
fn ckpt_interval() -> usize {
    std::env::var("CMF_GDN_CS").ok().and_then(|v| v.parse().ok()).filter(|&v: &usize| v >= 2).unwrap_or(8)
}

/// Device-resident tape of one GDN layer for `nb` windows of `t` tokens:
/// its buffers are allocated once and reused by every step.
pub fn gdn_tape(cfg: &GdnTrainCfg, nb: usize, t: usize) -> Option<GdnTape> {
    let c = ctx()?;
    let pp = pipes(c)?;
    let (nv, nk, kk) = (cfg.nv, cfg.nk, cfg.kk);
    let kd = nk * 128;
    let vd = nv * 128;
    let c_dim = 2 * kd + vd;
    let n_bt = nb * t;
    let cs = ckpt_interval();
    let nseg = t.div_ceil(cs);
    let mut prm = Vec::with_capacity(2 * nv + 128 + c_dim * kk);
    prm.extend_from_slice(cfg.a_log);
    prm.extend_from_slice(cfg.dt_bias);
    prm.extend_from_slice(cfg.norm);
    prm.extend_from_slice(cfg.conv);
    let sizes = [
        n_bt * c_dim,                                  // 1 RAW
        n_bt * c_dim,                                  // 2 CQ
        n_bt * vd,                                     // 3 Z
        2 * n_bt * nv,                                 // 4 AB
        prm.len(),                                     // 5 PRM
        2 * n_bt * nk * 128,                           // 6 QKN
        2 * n_bt * nk,                                 // 7 NQ
        3 * n_bt * nv,                                 // 8 GB
        n_bt * vd,                                     // 9 OUT
        n_bt * vd,                                     // 10 OS
        n_bt * nv,                                     // 11 INV
        nb * nv * (nseg + 1) * 16384,                  // 12 CK
        nb * nv * cs * 16384,                          // 13 SCR
        (n_bt * vd).max(nb * nv * cs * 128),           // 14 KVS (kv per step | per segment)
        n_bt * vd,                                     // 15 DOUT
        n_bt * c_dim,                                  // 16 DCQ
        n_bt * vd,                                     // 17 DZ
        2 * n_bt * nv,                                 // 18 DAB
        (n_bt * nv * NWG * 258).max(2 * n_bt * nv * 128), // 19 DQK (block partials | per head)
        n_bt * c_dim,                                  // 20 DRAW
    ];
    let bufs: Vec<wgpu::Buffer> = sizes.iter().map(|&n| sbuf(c, n)).collect();
    c.queue.write_buffer(&bufs[4], 0, bytemuck::cast_slice(&prm));
    let mk = |keep: u32| {
        let u: [u32; 20] = [
            nb as u32, t as u32, nv as u32, nk as u32, c_dim as u32, vd as u32, kd as u32, (nv / nk) as u32,
            kk as u32, cs as u32, nseg as u32, nv as u32, (2 * nv) as u32, (2 * nv + 128) as u32, keep, 0,
            (cfg.rms_eps as f32).to_bits(), 0, 0, 0,
        ];
        let uni = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gdn_train_p"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        c.queue.write_buffer(&uni, 0, bytemuck::cast_slice(&u));
        bind(c, pp, &uni, &bufs)
    };
    let bg = [mk(0), mk(1)];
    Some(GdnTape { nb, t, bufs, bg })
}

/// Forward of one GDN layer over `nb` windows of `t` tokens from the raw
/// projections (qkv `[nb·t, c_dim]`, z `[nb·t, nv·128]`, a/b `[nb·t, nv]`).
/// Returns the output `[nb·t, nv·128]` and the tape for [`gdn_backward`].
#[allow(clippy::too_many_arguments)]
pub fn gdn_forward(
    cfg: &GdnTrainCfg,
    nb: usize,
    t: usize,
    qkv: &[f32],
    z: &[f32],
    a: &[f32],
    b: &[f32],
) -> Option<(Vec<f32>, GdnTape)> {
    let c = ctx()?;
    let (nv, nk) = (cfg.nv, cfg.nk);
    let vd = nv * 128;
    let c_dim = 2 * nk * 128 + vd;
    let n_bt = nb * t;
    if nv % nk != 0 || qkv.len() != n_bt * c_dim || z.len() != n_bt * vd || a.len() != n_bt * nv || b.len() != n_bt * nv {
        return None;
    }
    let tape = gdn_tape(cfg, nb, t)?;
    c.queue.write_buffer(&tape.bufs[0], 0, bytemuck::cast_slice(qkv));
    c.queue.write_buffer(&tape.bufs[2], 0, bytemuck::cast_slice(z));
    c.queue.write_buffer(&tape.bufs[3], 0, bytemuck::cast_slice(a));
    c.queue.write_buffer(&tape.bufs[3], (n_bt * nv * 4) as u64, bytemuck::cast_slice(b));
    gdn_forward_dev(&tape, nk, nv, c_dim, true)?;
    let out = read(c, &tape.bufs[8], n_bt * vd)?;
    Some((out, tape))
}

/// Backward through the layer recorded in `tape` for the output gradient
/// `dout`: (dqkv, dz, da, db) — through-grads into the four projection
/// streams; the layer's own parameters are frozen.
#[allow(clippy::type_complexity)]
pub fn gdn_backward(cfg: &GdnTrainCfg, tape: &GdnTape, dout: &[f32]) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let c = ctx()?;
    let (nv, nk) = (cfg.nv, cfg.nk);
    let n_bt = tape.nb * tape.t;
    let vd = nv * 128;
    let c_dim = 2 * nk * 128 + vd;
    if dout.len() != n_bt * vd {
        return None;
    }
    c.queue.write_buffer(&tape.bufs[14], 0, bytemuck::cast_slice(dout));
    gdn_backward_dev(tape, nk, nv, c_dim)?;
    let draw = read(c, &tape.bufs[19], n_bt * c_dim)?;
    let dz = read(c, &tape.bufs[16], n_bt * vd)?;
    let dab = read(c, &tape.bufs[17], 2 * n_bt * nv)?;
    let (da, db) = dab.split_at(n_bt * nv);
    Some((draw, dz, da.to_vec(), db.to_vec()))
}

impl GdnTape {
    /// Input buffers: raw qkv `[n,c_dim]`, z `[n,vd]`, a|b `[2][n,nv]`.
    pub fn raw(&self) -> &wgpu::Buffer { &self.bufs[0] }
    /// Frozen parameters: a_log[nv] dt_bias[nv] norm[128] conv[c_dim·kk].
    pub fn prm(&self) -> &wgpu::Buffer { &self.bufs[4] }
    pub fn z(&self) -> &wgpu::Buffer { &self.bufs[2] }
    pub fn ab(&self) -> &wgpu::Buffer { &self.bufs[3] }
    /// Output `[n, vd]` of the last forward.
    pub fn out(&self) -> &wgpu::Buffer { &self.bufs[8] }
    /// Upstream gradient slot `[n, vd]` for the backward.
    pub fn dout(&self) -> &wgpu::Buffer { &self.bufs[14] }
    /// Gradients of the last backward: raw qkv, z, a|b.
    pub fn draw(&self) -> &wgpu::Buffer { &self.bufs[19] }
    pub fn dz(&self) -> &wgpu::Buffer { &self.bufs[16] }
    pub fn dab(&self) -> &wgpu::Buffer { &self.bufs[17] }
}

/// Forward on the tape's input buffers (filled by the caller); the output
/// stays in `tape.out()`. `keep`: record what the backward reads
/// (checkpoints, kv); a forward that is recomputed before its backward
/// (the first pass of a step) skips it.
pub fn gdn_forward_dev(tape: &GdnTape, nk: usize, nv: usize, c_dim: usize, keep: bool) -> Option<()> {
    let c = ctx()?;
    let pp = pipes(c)?;
    let n_bt = tape.nb * tape.t;
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gdn_fwd") });
    {
        let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        let mut mk = super::train_prof::pass_begin(c, &mut ps);
        ps.set_bind_group(0, &tape.bg[keep as usize], &[]);
        let (gx, gy) = grid(n_bt * c_dim, 256);
        ps.set_pipeline(&pp.p["conv_fwd"]);
        ps.dispatch_workgroups(gx, gy, 1);
        super::train_prof::pass_mark(c, &mut ps, &mut mk, "gdn.conv_fwd", [0; 3]);
        let (wx, wy) = wg_grid(n_bt * nk);
        ps.set_pipeline(&pp.p["qk_prep"]);
        ps.dispatch_workgroups(wx, wy, 1);
        super::train_prof::pass_mark(c, &mut ps, &mut mk, "gdn.qk_prep", [0; 3]);
        ps.set_pipeline(&pp.p["gate_prep"]);
        ps.dispatch_workgroups((n_bt * nv).div_ceil(256) as u32, 1, 1);
        super::train_prof::pass_mark(c, &mut ps, &mut mk, "gdn.gate_prep", [0; 3]);
        if pp.sg {
            ps.set_pipeline(&pp.p["rec_fwd2"]);
            ps.dispatch_workgroups((tape.nb * nv * NWG) as u32, 1, 1);
            super::train_prof::pass_mark(c, &mut ps, &mut mk, "gdn.rec_fwd", [0; 3]);
            let (rx, ry) = wg_grid((n_bt * nv).div_ceil(8));
            ps.set_pipeline(&pp.p["norm_fwd2"]);
            ps.dispatch_workgroups(rx, ry, 1);
            super::train_prof::pass_mark(c, &mut ps, &mut mk, "gdn.norm_fwd", [0; 3]);
        } else {
            ps.set_pipeline(&pp.p["rec_fwd"]);
            ps.dispatch_workgroups((tape.nb * nv) as u32, 1, 1);
            super::train_prof::pass_mark(c, &mut ps, &mut mk, "gdn.rec_fwd", [0; 3]);
        }
    }
    submit(c, enc.finish());
    Some(())
}

/// Backward for the gradient the caller wrote into `tape.dout()` (the
/// tape of a `keep` forward).
pub fn gdn_backward_dev(tape: &GdnTape, nk: usize, nv: usize, c_dim: usize) -> Option<()> {
    let c = ctx()?;
    let pp = pipes(c)?;
    let n_bt = tape.nb * tape.t;
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gdn_bwd") });
    {
        let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        let mut mk = super::train_prof::pass_begin(c, &mut ps);
        ps.set_bind_group(0, &tape.bg[1], &[]);
        let (wx, wy) = wg_grid(n_bt * nk);
        if pp.sg {
            let (rx, ry) = wg_grid((n_bt * nv).div_ceil(8));
            ps.set_pipeline(&pp.p["norm_bwd2"]);
            ps.dispatch_workgroups(rx, ry, 1);
            super::train_prof::pass_mark(c, &mut ps, &mut mk, "gdn.norm_bwd", [0; 3]);
            ps.set_pipeline(&pp.p["rec_bwd2"]);
            ps.dispatch_workgroups((tape.nb * nv * NWG) as u32, 1, 1);
            super::train_prof::pass_mark(c, &mut ps, &mut mk, "gdn.rec_bwd", [0; 3]);
            ps.set_pipeline(&pp.p["qk_bwd2"]);
            ps.dispatch_workgroups(wx, wy, 1);
            super::train_prof::pass_mark(c, &mut ps, &mut mk, "gdn.qk_bwd", [0; 3]);
        } else {
            ps.set_pipeline(&pp.p["rec_bwd"]);
            ps.dispatch_workgroups((tape.nb * nv) as u32, 1, 1);
            super::train_prof::pass_mark(c, &mut ps, &mut mk, "gdn.rec_bwd", [0; 3]);
            ps.set_pipeline(&pp.p["qk_bwd"]);
            ps.dispatch_workgroups(wx, wy, 1);
            super::train_prof::pass_mark(c, &mut ps, &mut mk, "gdn.qk_bwd", [0; 3]);
        }
        let (gx, gy) = grid(n_bt * c_dim, 256);
        ps.set_pipeline(&pp.p["conv_bwd"]);
        ps.dispatch_workgroups(gx, gy, 1);
        super::train_prof::pass_mark(c, &mut ps, &mut mk, "gdn.conv_bwd", [0; 3]);
    }
    submit(c, enc.finish());
    Some(())
}

fn bind(c: &Ctx, pp: &Pipes, uni: &wgpu::Buffer, bufs: &[wgpu::Buffer]) -> wgpu::BindGroup {
    super::train_prof::count(super::train_prof::Cnt::BindGroup, 1);
    let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: uni.as_entire_binding() }];
    for (k, b) in bufs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry { binding: k as u32 + 1, resource: b.as_entire_binding() });
    }
    // vec4 views of QKN, CK and SCR for the subgroup kernels
    for (k, i) in [(21u32, 5usize), (22, 11), (23, 12)] {
        entries.push(wgpu::BindGroupEntry { binding: k, resource: bufs[i].as_entire_binding() });
    }
    c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gdn_train"), layout: &pp.layout, entries: &entries })
}

/// 1-D element grid folded into (x, y) under the 65 535 per-dimension cap.
fn grid(n: usize, wg: usize) -> (u32, u32) {
    let groups = n.div_ceil(wg);
    if groups <= 65535 { (groups as u32, 1) } else { (65535, groups.div_ceil(65535) as u32) }
}

/// One workgroup per item, folded the same way.
fn wg_grid(n: usize) -> (u32, u32) {
    if n <= 65535 { (n as u32, 1) } else { (65535, n.div_ceil(65535) as u32) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fcd_ops::{GdnSeqCfg, gdn_seq_bwd, gdn_seq_fwd};

    fn rnd(n: usize, seed: u64, scale: f64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32 * scale as f32
            })
            .collect()
    }

    /// Timing of the device forward (with the tape) and backward at the
    /// 27B student's shape (8 windows of 512 tokens, 48 v-heads, 16 k-heads):
    /// `cargo test --release -p cortiq-engine --features gpu --lib gdn_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gdn_bench() {
        let nb: usize = std::env::var("GDN_BENCH_NB").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let (nv, nk, kk, t) = (48usize, 16usize, 4usize, 512usize);
        let (kd, vd) = (nk * 128, nv * 128);
        let c_dim = 2 * kd + vd;
        let conv = rnd(c_dim * kk, 1, 0.5);
        let a_log: Vec<f32> = (0..nv).map(|h| -0.5 + 0.03 * h as f32).collect();
        let dt_bias = rnd(nv, 2, 0.5);
        let norm: Vec<f32> = rnd(128, 3, 0.3).iter().map(|v| 1.0 + v).collect();
        let cfg = GdnTrainCfg { nv, nk, kk, rms_eps: 1e-6, a_log: &a_log, dt_bias: &dt_bias, norm: &norm, conv: &conv };
        let Some(c) = ctx() else { return };
        let Some(tape) = gdn_tape(&cfg, nb, t) else { return };
        let n = nb * t;
        c.queue.write_buffer(&tape.bufs[0], 0, bytemuck::cast_slice(&rnd(n * c_dim, 4, 1.0)));
        c.queue.write_buffer(&tape.bufs[2], 0, bytemuck::cast_slice(&rnd(n * vd, 5, 1.0)));
        c.queue.write_buffer(&tape.bufs[3], 0, bytemuck::cast_slice(&rnd(2 * n * nv, 6, 1.0)));
        c.queue.write_buffer(&tape.bufs[14], 0, bytemuck::cast_slice(&rnd(n * vd, 8, 1.0)));
        let sync = || {
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        };
        sync();
        let reps = 5;
        let (mut tf, mut tf0, mut tb) = (0f64, 0f64, 0f64);
        for i in 0..=reps {
            let t0 = std::time::Instant::now();
            gdn_forward_dev(&tape, nk, nv, c_dim, false).unwrap();
            sync();
            let t1 = std::time::Instant::now();
            gdn_forward_dev(&tape, nk, nv, c_dim, true).unwrap();
            sync();
            let t2 = std::time::Instant::now();
            gdn_backward_dev(&tape, nk, nv, c_dim).unwrap();
            sync();
            let t3 = std::time::Instant::now();
            if i > 0 {
                tf0 += (t1 - t0).as_secs_f64();
                tf += (t2 - t1).as_secs_f64();
                tb += (t3 - t2).as_secs_f64();
            }
        }
        let ms = |x: f64| 1e3 * x / reps as f64;
        eprintln!("gdn_bench nb {nb} cs {}: forward {:.3} ms (with tape {:.3} ms), backward {:.3} ms", ckpt_interval(), ms(tf0), ms(tf), ms(tb));
    }

    /// Device forward and BPTT against the f64 host reference: two windows,
    /// T not a multiple of the checkpoint interval, three v-heads per k-head.
    #[test]
    fn gdn_device_matches_host_reference() {
        let (nv, nk, kk, nb, t) = (6usize, 2usize, 4usize, 2usize, 45usize);
        let (kd, vd) = (nk * 128, nv * 128);
        let c_dim = 2 * kd + vd;
        let conv = rnd(c_dim * kk, 1, 0.5);
        let a_log: Vec<f32> = (0..nv).map(|h| -0.5 + 0.3 * h as f32).collect();
        let dt_bias = rnd(nv, 2, 0.5);
        let norm: Vec<f32> = rnd(128, 3, 0.3).iter().map(|v| 1.0 + v).collect();
        let cfg = GdnTrainCfg { nv, nk, kk, rms_eps: 1e-6, a_log: &a_log, dt_bias: &dt_bias, norm: &norm, conv: &conv };
        let qkv = rnd(nb * t * c_dim, 4, 1.0);
        let z = rnd(nb * t * vd, 5, 1.0);
        let a = rnd(nb * t * nv, 6, 1.0);
        let b = rnd(nb * t * nv, 7, 1.0);
        let dout = rnd(nb * t * vd, 8, 1.0);
        let Some((out, tape)) = gdn_forward(&cfg, nb, t, &qkv, &z, &a, &b) else {
            eprintln!("no device — skipped");
            return;
        };
        let (draw, dz, da, db) = gdn_backward(&cfg, &tape, &dout).expect("backward");
        let hc = GdnSeqCfg { nv, nk, dk: 128, dv: 128, kk, rms_eps: 1e-6, conv: &conv, a_log: &a_log, dt_bias: &dt_bias, norm: &norm };
        let f = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let rel = |name: &str, got: &[f32], want: &[f64]| {
            let num: f64 = got.iter().zip(want).map(|(g, w)| (*g as f64 - w).powi(2)).sum();
            let den: f64 = want.iter().map(|w| w * w).sum::<f64>().max(1e-30);
            let r = (num / den).sqrt();
            eprintln!("{name}: relative error {r:e}");
            assert!(r < 2e-4, "{name}: relative error {r:e}");
        };
        for s in 0..nb {
            let sl = |v: &[f32], w: usize| f(&v[s * t * w..(s + 1) * t * w]);
            let (q, zz, aa, bb, dd) = (sl(&qkv, c_dim), sl(&z, vd), sl(&a, nv), sl(&b, nv), sl(&dout, vd));
            let mut want = vec![0f64; t * vd];
            gdn_seq_fwd(&q, &zz, &aa, &bb, t, &hc, &mut want);
            rel("out", &out[s * t * vd..(s + 1) * t * vd], &want);
            let (mut dq, mut dzz, mut dda, mut ddb) = (vec![0f64; t * c_dim], vec![0f64; t * vd], vec![0f64; t * nv], vec![0f64; t * nv]);
            gdn_seq_bwd(&q, &zz, &aa, &bb, t, &hc, &dd, &mut dq, &mut dzz, &mut dda, &mut ddb);
            rel("dqkv", &draw[s * t * c_dim..(s + 1) * t * c_dim], &dq);
            rel("dz", &dz[s * t * vd..(s + 1) * t * vd], &dzz);
            rel("da", &da[s * t * nv..(s + 1) * t * nv], &dda);
            rel("db", &db[s * t * nv..(s + 1) * t * nv], &ddb);
        }
    }
}
