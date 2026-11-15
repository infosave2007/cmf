// Kimi-Linear wgpu kernels: the WGSL source (no crate dependencies, so a
// standalone naga check can include it; `kimi.rs` includes it).

/// Key width of every KDA head the kernels take (Kimi-Linear-48B).
pub(crate) const KDA_DK: usize = 128;


fn scan_src() -> String {
    // The state column a lane owns in 32 named registers (s0..s31): rows
    // w·32 .. w·32 + 31 of head h, column j. Two partial chains per dot
    // product, as `hy_kda_scan4`.
    let mut s = String::new();
    s.push_str(
        "@compute @workgroup_size(128)\nfn kda_scan(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {\n\
         \x20   let lane = lid & 31u;\n    let w = lid >> 5u;\n    let dv = p.dv;\n    let wpr = dv >> 5u;\n\
         \x20   let h = wid.x / wpr;\n    let j = (wid.x - h * wpr) * 32u + lane;\n\
         \x20   let kd = p.kd;\n    let cdim = p.cdim;\n    let nh = p.nh;\n\
         \x20   let base = (h * 128u + w * 32u) * dv + j;\n",
    );
    for i in 0..32 {
        s.push_str(&format!("    var s{i} = kS[base + {i}u * dv];\n"));
    }
    s.push_str(
        "    for (var t0 = 0u; t0 < p.kb; t0 = t0 + 16u) {\n\
         \x20       let nn = min(p.kb - t0, 16u);\n\
         \x20       // the previous tile's reads are done\n\
         \x20       workgroupBarrier();\n\
         \x20       for (var t = 0u; t < nn; t = t + 1u) {\n\
         \x20           let row = (t0 + t) * cdim + h * 128u + lid;\n\
         \x20           qs[t * 128u + lid] = kcq[row];\n\
         \x20           ks[t * 128u + lid] = kcq[row + kd];\n\
         \x20           gs[t * 128u + lid] = kgd[(t0 + t) * kd + h * 128u + lid];\n\
         \x20       }\n\
         \x20       if (lid < nn) { bs[lid] = kbeta[(t0 + lid) * nh + h]; }\n\
         \x20       workgroupBarrier();\n\
         \x20       for (var t = 0u; t < nn; t = t + 1u) {\n\
         \x20           let vt = kcq[(t0 + t) * cdim + 2u * kd + h * dv + j];\n\
         \x20           let r = t * 128u + w * 32u;\n\
         \x20           var a0 = 0.0;\n            var a1 = 0.0;\n",
    );
    for i in (0..32).step_by(2) {
        s.push_str(&format!(
            "            a0 = fma(s{i}, ks[r + {i}u] * gs[r + {i}u], a0);\n            a1 = fma(s{n}, ks[r + {n}u] * gs[r + {n}u], a1);\n",
            n = i + 1
        ));
    }
    s.push_str(
        "            part[w * 32u + lane] = a0 + a1;\n\
         \x20           workgroupBarrier();\n\
         \x20           let kv = ((part[lane] + part[32u + lane]) + part[64u + lane]) + part[96u + lane];\n\
         \x20           let dlt = (vt - kv) * bs[t];\n\
         \x20           var b0 = 0.0;\n            var b1 = 0.0;\n",
    );
    for i in (0..32).step_by(2) {
        s.push_str(&format!(
            "            s{i} = fma(ks[r + {i}u], dlt, gs[r + {i}u] * s{i});\n            s{n} = fma(ks[r + {n}u], dlt, gs[r + {n}u] * s{n});\n            b0 = fma(qs[r + {i}u], s{i}, b0);\n            b1 = fma(qs[r + {n}u], s{n}, b1);\n",
            n = i + 1
        ));
    }
    s.push_str(
        "            part[128u + w * 32u + lane] = b0 + b1;\n\
         \x20           workgroupBarrier();\n\
         \x20           if (w == 0u) {\n\
         \x20               ko[(t0 + t) * nh * dv + h * dv + j] =\n\
         \x20                   ((part[128u + lane] + part[160u + lane]) + part[192u + lane]) + part[224u + lane];\n\
         \x20           }\n\
         \x20       }\n\
         \x20   }\n",
    );
    for i in 0..32 {
        s.push_str(&format!("    kS[base + {i}u * dv] = s{i};\n"));
    }
    s.push_str("}\n");
    s
}

const COMMON_SRC: &str = r#"
struct KP {
    kb: u32, nh: u32, dk: u32, dv: u32,
    kk: u32, cdim: u32, kd: u32, vd: u32,
    a_len: u32, lb_on: u32, lb: f32, eps: f32,
    _p0: u32, _p1: u32, _p2: u32, _p3: u32,
};
@group(0) @binding(0) var<uniform> p: KP;
@group(0) @binding(1) var<storage, read> kxq: array<f32>;
@group(0) @binding(2) var<storage, read> kxk: array<f32>;
@group(0) @binding(3) var<storage, read> kxv: array<f32>;
@group(0) @binding(4) var<storage, read> ktaps: array<f32>;
@group(0) @binding(5) var<storage, read_write> kring: array<f32>;
@group(0) @binding(6) var<storage, read_write> kcq: array<f32>;
@group(0) @binding(8) var<storage, read> kbraw: array<f32>;
@group(0) @binding(9) var<storage, read> kalog: array<f32>;
@group(0) @binding(10) var<storage, read> kdtb: array<f32>;
@group(0) @binding(11) var<storage, read_write> kgd: array<f32>;
@group(0) @binding(12) var<storage, read_write> kbeta: array<f32>;
@group(0) @binding(13) var<storage, read_write> kS: array<f32>;
@group(0) @binding(14) var<storage, read_write> ko: array<f32>;
@group(0) @binding(15) var<storage, read> kgate: array<f32>;
@group(0) @binding(16) var<storage, read> konorm: array<f32>;

var<workgroup> fr: array<f32, 128>;
var<workgroup> qs: array<f32, 2048>;
var<workgroup> ks: array<f32, 2048>;
var<workgroup> gs: array<f32, 2048>;
var<workgroup> bs: array<f32, 16>;
var<workgroup> part: array<f32, 256>;

// The raw (pre-conv) value of channel c at row t: q | k | v.
fn kraw(t: u32, c: u32) -> f32 {
    if (c < p.kd) { return kxq[t * p.kd + c]; }
    if (c < 2u * p.kd) { return kxk[t * p.kd + c - p.kd]; }
    return kxv[t * p.vd + c - 2u * p.kd];
}

// Row t + j + 1 - kk of the conv window of row t: a row of the call, or the
// ring row before it (ring row 0 the oldest).
fn kwin(t: u32, j: u32, c: u32) -> f32 {
    let kk = p.kk;
    if (t + j + 1u >= kk) { return kraw(t + j + 1u - kk, c); }
    return kring[(t + j) * p.cdim + c];
}

@compute @workgroup_size(128)
fn kda_front(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let u = wid.x;
    let t = wid.y;
    let c = u * 128u + lid;
    let kk = p.kk;
    let tb = c * kk;
    var acc = kraw(t, c) * ktaps[tb + kk - 1u];
    for (var j = 0u; j + 1u < kk; j = j + 1u) {
        acc = fma(kwin(t, j, c), ktaps[tb + j], acc);
    }
    var v = acc / (1.0 + exp(-acc));
    fr[lid] = v * v;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (lid < s) { fr[lid] = fr[lid] + fr[lid + s]; }
        workgroupBarrier();
    }
    if (u < 2u * p.nh) {
        var inv = 1.0 / sqrt(fr[0] + 1e-6);
        if (u < p.nh) { inv = inv / sqrt(f32(p.dk)); }
        v = v * inv;
    }
    kcq[t * p.cdim + c] = v;
}

fn ksoftplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}

@compute @workgroup_size(128)
fn kda_norm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let t = wid.y;
    let idx = (t * p.nh + h) * p.dv + lid;
    let x = ko[idx];
    fr[lid] = x * x;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (lid < s) { fr[lid] = fr[lid] + fr[lid + s]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(fr[0] / f32(p.dv) + p.eps);
    let z = kgate[idx];
    ko[idx] = x * inv * konorm[lid] * (1.0 / (1.0 + exp(-z)));
    // the ring after the call's rows, by the first row's workgroups (the
    // front read the old ring two dispatches back; each lane its channels)
    if (wid.y == 0u && p.kk > 1u) {
        let kk = p.kk;
        for (var c = h * 128u + lid; c < p.cdim; c = c + p.nh * 128u) {
            var nr: array<f32, 8>;
            for (var j = 0u; j + 1u < kk && j < 8u; j = j + 1u) { nr[j] = kwin(p.kb, j, c); }
            for (var j = 0u; j + 1u < kk && j < 8u; j = j + 1u) { kring[j * p.cdim + c] = nr[j]; }
        }
    }
}

// The low-rank decay's second stage with the gates folded in, and the
// output gate's: rows [0, kd) gd = exp(log-decay(f_b · fl + bias)) (and
// beta of the head on its first channel), rows [kd, kd + vd) the gate
// g_b · gl. 64 lanes a row, as `f32_matvec_b`.
@group(0) @binding(27) var<storage, read> kfbw: array<f32>;
@group(0) @binding(28) var<storage, read> kgbw: array<f32>;
@group(0) @binding(29) var<storage, read> kfl: array<f32>;
@group(0) @binding(30) var<storage, read> kgl: array<f32>;
@group(0) @binding(31) var<storage, read_write> kgate_w: array<f32>;
var<workgroup> lrp: array<f32, 64>;
@compute @workgroup_size(64)
fn kda_lr_out(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let r = wid.x;
    let t = wid.y;
    let kd = p.kd;
    let isf = r < kd;
    let rank = select(p._p1, p._p0, isf);
    var acc = 0.0;
    if (isf) {
        for (var i = lid; i < rank; i = i + 64u) { acc = fma(kfbw[r * rank + i], kfl[t * rank + i], acc); }
    } else {
        let rr = r - kd;
        for (var i = lid; i < rank; i = i + 64u) { acc = fma(kgbw[rr * rank + i], kgl[t * rank + i], acc); }
    }
    lrp[lid] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (lid < s) { lrp[lid] = lrp[lid] + lrp[lid + s]; }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let v = lrp[0];
        if (isf) {
            let h = r / p.dk;
            let d = r - h * p.dk;
            var a = 0.0;
            if (p.a_len == p.nh) {
                a = kalog[h];
            } else if (p.a_len == p.dk) {
                a = kalog[d];
            } else {
                a = kalog[r];
            }
            let x = v + kdtb[r];
            var ld = 0.0;
            if (p.lb_on != 0u) {
                ld = p.lb / (1.0 + exp(-(exp(a) * x)));
            } else {
                ld = -(exp(a) * ksoftplus(x));
            }
            kgd[t * kd + r] = exp(ld);
            if (d == 0u) {
                kbeta[t * p.nh + h] = 1.0 / (1.0 + exp(-kbraw[t * p.nh + h]));
            }
        } else {
            kgate_w[t * p.vd + r - kd] = v;
        }
    }
}

struct MP {
    kb: u32, nh: u32, rope: u32, nope: u32,
    vd: u32, lora: u32, kstride: u32, eps: f32,
};
@group(0) @binding(20) var<uniform> mp: MP;
@group(0) @binding(21) var<storage, read> mkva: array<f32>;
@group(0) @binding(22) var<storage, read> mlw: array<f32>;
@group(0) @binding(23) var<storage, read_write> mlat: array<f32>;
@group(0) @binding(24) var<storage, read> mkvb: array<f32>;
@group(0) @binding(25) var<storage, read_write> mk: array<f32>;
@group(0) @binding(26) var<storage, read_write> mv: array<f32>;
var<workgroup> mred: array<f32, 256>;

// The latent of row r (its first `lora` values) RMS-normed with mlw.
@compute @workgroup_size(256)
fn mla_lat(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let r = wid.x;
    let lora = mp.lora;
    let x0 = r * mp.kstride;
    var acc = 0.0;
    for (var i = lid; i < lora; i = i + 256u) {
        let x = mkva[x0 + i];
        acc = fma(x, x, acc);
    }
    mred[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { mred[lid] = mred[lid] + mred[lid + s]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(mred[0] / f32(lora) + mp.eps);
    for (var i = lid; i < lora; i = i + 256u) {
        mlat[r * lora + i] = mkva[x0 + i] * inv * mlw[i];
    }
}

// k [rows][nh][rope + nope] = [k_rope (the row's shared rope key) | k_nope],
// v [rows][nh][vd]; kva rows kstride long (the latent, then the rope key),
// kvb [rows][nh][nope + vd].
@compute @workgroup_size(256)
fn mla_kv(@builtin(global_invocation_id) g: vec3<u32>) {
    let hd = mp.rope + mp.nope;
    let i = g.x;
    let t = g.y;
    if (i >= mp.nh * hd) { return; }
    let h = i / hd;
    let d = i - h * hd;
    let b = (t * mp.nh + h) * (mp.nope + mp.vd);
    if (d < mp.rope) {
        mk[t * mp.nh * hd + i] = mkva[t * mp.kstride + mp.lora + d];
    } else {
        mk[t * mp.nh * hd + i] = mkvb[b + d - mp.rope];
    }
    if (d < mp.vd) {
        mv[(t * mp.nh + h) * mp.vd + d] = mkvb[b + mp.nope + d];
    }
}
"#;

/// The module's WGSL.
pub(crate) fn src() -> String {
    let mut s = String::from(COMMON_SRC);
    s.push_str(&scan_src());
    s
}

